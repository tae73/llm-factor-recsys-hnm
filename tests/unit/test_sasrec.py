"""Unit tests for SASRec (Self-Attentive Sequential Recommendation) model.

Tests model initialization, forward pass, causal masking, padding masking,
position embedding, gradient flow, JIT, and training step.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from src.config import SASRecConfig
from src.losses import binary_cross_entropy
from src.models.sasrec import SASRec, SASRecInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return SASRecConfig(
        d_embed=8,
        n_heads=2,
        n_blocks=1,
        max_seq_len=5,
        dropout_rate=0.0,
    )


@pytest.fixture
def model(small_config):
    rngs = nnx.Rngs(params=42, dropout=43)
    return SASRec(n_items=20, config=small_config, rngs=rngs)


@pytest.fixture
def sample_input():
    """Small batch of 4 sequences."""
    rng = np.random.default_rng(42)
    B, T = 4, 5
    # Left-padded: zeros at the beginning
    history = np.zeros((B, T), dtype=np.int32)
    history[0, 2:] = rng.integers(1, 20, size=3)  # 3 items
    history[1, :] = rng.integers(1, 20, size=5)    # 5 items (full)
    history[2, 4:] = rng.integers(1, 20, size=1)   # 1 item
    history[3, 1:] = rng.integers(1, 20, size=4)   # 4 items
    return SASRecInput(
        history=jnp.array(history, dtype=jnp.int32),
        hist_len=jnp.array([3, 5, 1, 4], dtype=jnp.int32),
    )


@pytest.fixture
def sample_targets():
    rng = np.random.default_rng(42)
    return jnp.array(rng.integers(1, 20, size=4), dtype=jnp.int32)


@pytest.fixture
def sample_labels():
    return jnp.array([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# SASRec Model Tests
# ---------------------------------------------------------------------------


class TestSASRecModel:
    def test_init(self, model):
        """Model should initialize without errors."""
        assert model is not None

    def test_forward_shape(self, model, sample_input, sample_targets):
        """Forward pass with targets should return (B,) logits."""
        logits = model(sample_input, sample_targets)
        assert logits.shape == (4,)

    def test_forward_without_target_returns_embeddings(self, model, sample_input):
        """Forward without target should return user embeddings (B, d)."""
        user_embed = model(sample_input)
        assert user_embed.shape == (4, 8)  # d_embed=8

    def test_output_is_finite(self, model, sample_input, sample_targets):
        """No NaN/Inf in output."""
        logits = model(sample_input, sample_targets)
        assert jnp.all(jnp.isfinite(logits))

    def test_causal_mask(self, model, sample_input):
        """Causal mask should be lower triangular."""
        mask = model._build_mask(sample_input.history)
        # shape: (B, 1, T, T)
        B, _, T, _ = mask.shape
        assert mask.shape == (4, 1, 5, 5)

        # For a fully filled sequence (user 1), check lower triangular
        # (after combining with key padding, non-pad positions should be causal)
        for t_q in range(T):
            for t_k in range(T):
                if t_k > t_q:
                    # Future positions should be masked for causal attention
                    assert not mask[1, 0, t_q, t_k], f"Position ({t_q},{t_k}) should be masked"

    def test_padding_mask(self, model, sample_input):
        """Padded positions should not be attended to."""
        mask = model._build_mask(sample_input.history)
        # User 2 has only 1 item at position 4, positions 0-3 are PAD
        # Key positions 0-3 should be masked for all queries
        for t_q in range(5):
            for t_k in range(4):
                assert not mask[2, 0, t_q, t_k], (
                    f"Padded key {t_k} should be masked for query {t_q}"
                )

    def test_position_embedding_shape(self, model):
        """Position embedding should have (max_seq_len, d_embed) shape."""
        pos_emb = model.position_embedding.embedding
        assert pos_emb.shape == (5, 8)  # max_seq_len=5, d_embed=8

    def test_predict_proba_range(self, model, sample_input, sample_targets):
        """Probabilities should be in [0, 1]."""
        model.eval()
        probs = model.predict_proba(sample_input, sample_targets)
        model.train()
        assert jnp.all(probs >= 0.0)
        assert jnp.all(probs <= 1.0)

    def test_gradient_flow(self, model, sample_input, sample_targets, sample_labels):
        """All parameters should receive non-zero gradients."""

        def loss_fn(m):
            logits = m(sample_input, sample_targets)
            return binary_cross_entropy(logits, sample_labels)

        _, grads = nnx.value_and_grad(loss_fn)(model)
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, "shape"))
        assert has_nonzero, "All gradients are zero"

    def test_no_nan_gradients(self, model, sample_input, sample_targets, sample_labels):
        """Gradients should not contain NaN."""

        def loss_fn(m):
            return binary_cross_entropy(m(sample_input, sample_targets), sample_labels)

        _, grads = nnx.value_and_grad(loss_fn)(model)
        for g in jax.tree.leaves(grads):
            if hasattr(g, "shape"):
                assert jnp.all(jnp.isfinite(g)), "NaN in gradients"

    def test_jit_compatible(self, model, sample_input, sample_targets, sample_labels):
        """Model should work under nnx.jit."""
        opt = nnx.Optimizer(model, optax.adam(learning_rate=1e-3), wrt=nnx.Param)

        @nnx.jit
        def step(m, o, inp, targets, labels):
            def loss_fn(m):
                return binary_cross_entropy(m(inp, targets), labels)

            loss, grads = nnx.value_and_grad(loss_fn)(m)
            o.update(m, grads)
            return loss

        loss = step(model, opt, sample_input, sample_targets, sample_labels)
        assert jnp.isfinite(loss)

    def test_deterministic_with_seed(self, small_config, sample_input, sample_targets):
        """Same seed should produce identical outputs."""
        rngs1 = nnx.Rngs(params=42, dropout=43)
        m1 = SASRec(n_items=20, config=small_config, rngs=rngs1)
        m1.eval()
        out1 = m1(sample_input, sample_targets)

        rngs2 = nnx.Rngs(params=42, dropout=43)
        m2 = SASRec(n_items=20, config=small_config, rngs=rngs2)
        m2.eval()
        out2 = m2(sample_input, sample_targets)

        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-6)

    def test_different_sequences_different_outputs(self, model, sample_targets):
        """Different input sequences should produce different scores."""
        inp1 = SASRecInput(
            history=jnp.array([[0, 0, 1, 2, 3]], dtype=jnp.int32),
            hist_len=jnp.array([3], dtype=jnp.int32),
        )
        inp2 = SASRecInput(
            history=jnp.array([[0, 0, 5, 10, 15]], dtype=jnp.int32),
            hist_len=jnp.array([3], dtype=jnp.int32),
        )
        model.eval()
        s1 = float(model(inp1, sample_targets[:1])[0])
        s2 = float(model(inp2, sample_targets[:1])[0])
        model.train()
        assert s1 != s2

    def test_full_catalog_scoring_shape(self, model, sample_input):
        """score_all_items should return (B, n_items+1) scores."""
        model.eval()
        scores = model.score_all_items(sample_input)
        model.train()
        # n_items=20, +1 for PAD → embedding table has 21 entries
        assert scores.shape == (4, 21)

    def test_bce_loss_computation(self, model, sample_input, sample_targets, sample_labels):
        """BCE loss should be a finite scalar."""
        logits = model(sample_input, sample_targets)
        loss = binary_cross_entropy(logits, sample_labels)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_loss_decreases(self, small_config):
        """Loss should decrease over multiple training steps."""
        rngs = nnx.Rngs(params=42, dropout=43)
        m = SASRec(n_items=20, config=small_config, rngs=rngs)
        opt = nnx.Optimizer(m, optax.adam(learning_rate=1e-2), wrt=nnx.Param)

        rng = np.random.default_rng(42)
        B, T = 32, 5
        history = np.zeros((B, T), dtype=np.int32)
        for i in range(B):
            n = rng.integers(1, T + 1)
            history[i, T - n :] = rng.integers(1, 20, size=n)
        hist_len = jnp.array([int(np.sum(h > 0)) for h in history], dtype=jnp.int32)

        inp = SASRecInput(
            history=jnp.array(history, dtype=jnp.int32),
            hist_len=hist_len,
        )
        targets = jnp.array(rng.integers(1, 20, size=B), dtype=jnp.int32)
        labels = jnp.array(rng.choice([0.0, 1.0], size=B), dtype=jnp.float32)

        losses = []
        for _ in range(20):

            def loss_fn(m):
                return binary_cross_entropy(m(inp, targets), labels)

            loss, grads = nnx.value_and_grad(loss_fn)(m)
            opt.update(m, grads)
            losses.append(float(loss))

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
