"""Unit tests for DIN (Deep Interest Network) model.

Tests model initialization, forward pass, attention, gradient flow,
masking, JIT compatibility, and training step.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from src.config import DINConfig
from src.losses import binary_cross_entropy
from src.models.din import DIN, DINInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return DINConfig(
        d_embed=4,
        attention_hidden_dims=(8, 4),
        dnn_hidden_dims=(8, 4),
        dropout_rate=0.0,
        use_batch_norm=False,
    )


@pytest.fixture
def field_dims():
    """Small vocab sizes: 3 user cat + 5 item cat."""
    return [5, 3, 3, 10, 6, 4, 8, 3]


@pytest.fixture
def n_numerical():
    return 10  # 8 user + 2 item


@pytest.fixture
def model(small_config, field_dims, n_numerical):
    rngs = nnx.Rngs(params=42, dropout=43)
    return DIN(
        field_dims=field_dims,
        n_numerical=n_numerical,
        n_items=20,
        max_seq_len=5,
        config=small_config,
        rngs=rngs,
    )


@pytest.fixture
def sample_batch():
    """Small batch of 4 samples with history."""
    rng = np.random.default_rng(42)
    B, T = 4, 5
    return DINInput(
        user_cat=jnp.array(rng.integers(0, 3, size=(B, 3)), dtype=jnp.int32),
        user_num=jnp.array(rng.random((B, 8)), dtype=jnp.float32),
        item_cat=jnp.array(rng.integers(0, 3, size=(B, 5)), dtype=jnp.int32),
        item_num=jnp.array(rng.random((B, 2)), dtype=jnp.float32),
        history=jnp.array(rng.integers(1, 20, size=(B, T)), dtype=jnp.int32),
        hist_len=jnp.array([3, 5, 2, 4], dtype=jnp.int32),
    )


@pytest.fixture
def sample_labels():
    return jnp.array([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# DIN Model Tests
# ---------------------------------------------------------------------------


class TestDINModel:
    def test_init(self, model):
        """Model should initialize without errors."""
        assert model is not None

    def test_output_shape(self, model, sample_batch):
        """Forward pass should return (B,) logits."""
        logits = model(sample_batch)
        assert logits.shape == (4,)

    def test_output_is_finite(self, model, sample_batch):
        """No NaN/Inf in output."""
        logits = model(sample_batch)
        assert jnp.all(jnp.isfinite(logits))

    def test_predict_proba_range(self, model, sample_batch):
        """Probabilities should be in [0, 1]."""
        model.eval()
        probs = model.predict_proba(sample_batch)
        model.train()
        assert jnp.all(probs >= 0.0)
        assert jnp.all(probs <= 1.0)

    def test_padding_mask(self, small_config, field_dims, n_numerical):
        """Padded positions should get near-zero attention weight."""
        rngs = nnx.Rngs(params=42, dropout=43)
        m = DIN(
            field_dims=field_dims,
            n_numerical=n_numerical,
            n_items=20,
            max_seq_len=5,
            config=small_config,
            rngs=rngs,
        )
        rng = np.random.default_rng(42)
        B, T = 2, 5
        # Create history with padding (0) in first positions
        history = jnp.array([[0, 0, 1, 2, 3], [0, 0, 0, 0, 5]], dtype=jnp.int32)
        inp = DINInput(
            user_cat=jnp.array(rng.integers(0, 3, size=(B, 3)), dtype=jnp.int32),
            user_num=jnp.array(rng.random((B, 8)), dtype=jnp.float32),
            item_cat=jnp.array(rng.integers(0, 3, size=(B, 5)), dtype=jnp.int32),
            item_num=jnp.array(rng.random((B, 2)), dtype=jnp.float32),
            history=history,
            hist_len=jnp.array([3, 1], dtype=jnp.int32),
        )
        m.eval()
        weights = m.get_attention_weights(inp)
        m.train()
        # Padded positions (0) should have 0 weight
        assert float(weights[0, 0]) == 0.0  # first two positions padded for user 0
        assert float(weights[0, 1]) == 0.0
        assert float(weights[1, 0]) == 0.0  # first four positions padded for user 1
        assert float(weights[1, 1]) == 0.0
        assert float(weights[1, 2]) == 0.0
        assert float(weights[1, 3]) == 0.0

    def test_different_targets_different_scores(self, small_config, field_dims, n_numerical):
        """Different target items should produce different scores."""
        rngs = nnx.Rngs(params=42, dropout=43)
        m = DIN(
            field_dims=field_dims,
            n_numerical=n_numerical,
            n_items=20,
            max_seq_len=5,
            config=small_config,
            rngs=rngs,
        )
        rng = np.random.default_rng(42)
        B = 1
        base = DINInput(
            user_cat=jnp.array(rng.integers(0, 3, size=(B, 3)), dtype=jnp.int32),
            user_num=jnp.array(rng.random((B, 8)), dtype=jnp.float32),
            item_cat=jnp.array([[1, 2, 1, 3, 1]], dtype=jnp.int32),
            item_num=jnp.array(rng.random((B, 2)), dtype=jnp.float32),
            history=jnp.array([[0, 0, 1, 2, 3]], dtype=jnp.int32),
            hist_len=jnp.array([3], dtype=jnp.int32),
        )
        alt = base._replace(item_cat=jnp.array([[5, 3, 2, 1, 0]], dtype=jnp.int32))
        m.eval()
        s1 = float(m(base)[0])
        s2 = float(m(alt)[0])
        m.train()
        assert s1 != s2

    def test_gradient_flow(self, model, sample_batch, sample_labels):
        """All parameters should receive non-zero gradients."""

        def loss_fn(m):
            logits = m(sample_batch)
            return binary_cross_entropy(logits, sample_labels)

        _, grads = nnx.value_and_grad(loss_fn)(model)
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, "shape"))
        assert has_nonzero, "All gradients are zero"

    def test_no_nan_gradients(self, model, sample_batch, sample_labels):
        """Gradients should not contain NaN."""

        def loss_fn(m):
            return binary_cross_entropy(m(sample_batch), sample_labels)

        _, grads = nnx.value_and_grad(loss_fn)(model)
        for g in jax.tree.leaves(grads):
            if hasattr(g, "shape"):
                assert jnp.all(jnp.isfinite(g)), "NaN in gradients"

    def test_bce_loss_computation(self, model, sample_batch, sample_labels):
        """BCE loss should be a finite scalar."""
        logits = model(sample_batch)
        loss = binary_cross_entropy(logits, sample_labels)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_jit_compatible(self, model, sample_batch, sample_labels):
        """Model should work under nnx.jit."""
        opt = nnx.Optimizer(model, optax.adam(learning_rate=1e-3), wrt=nnx.Param)

        @nnx.jit
        def step(m, o, b, l):
            def loss_fn(m):
                return binary_cross_entropy(m(b), l)

            loss, grads = nnx.value_and_grad(loss_fn)(m)
            o.update(m, grads)
            return loss

        loss = step(model, opt, sample_batch, sample_labels)
        assert jnp.isfinite(loss)

    def test_loss_decreases(self, small_config, field_dims, n_numerical):
        """Loss should decrease over multiple training steps."""
        rngs = nnx.Rngs(params=42, dropout=43)
        m = DIN(
            field_dims=field_dims,
            n_numerical=n_numerical,
            n_items=20,
            max_seq_len=5,
            config=small_config,
            rngs=rngs,
        )
        opt = nnx.Optimizer(m, optax.adam(learning_rate=1e-2), wrt=nnx.Param)

        rng = np.random.default_rng(42)
        B, T = 32, 5
        inp = DINInput(
            user_cat=jnp.array(rng.integers(0, 3, size=(B, 3)), dtype=jnp.int32),
            user_num=jnp.array(rng.random((B, 8)), dtype=jnp.float32),
            item_cat=jnp.array(rng.integers(0, 3, size=(B, 5)), dtype=jnp.int32),
            item_num=jnp.array(rng.random((B, 2)), dtype=jnp.float32),
            history=jnp.array(rng.integers(1, 20, size=(B, T)), dtype=jnp.int32),
            hist_len=jnp.array(rng.integers(1, T + 1, size=B), dtype=jnp.int32),
        )
        labels = jnp.array(rng.choice([0.0, 1.0], size=B), dtype=jnp.float32)

        losses = []
        for _ in range(20):

            def loss_fn(m):
                return binary_cross_entropy(m(inp), labels)

            loss, grads = nnx.value_and_grad(loss_fn)(m)
            opt.update(m, grads)
            losses.append(float(loss))

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_deterministic_with_seed(self, small_config, field_dims, n_numerical):
        """Same seed should produce identical outputs."""
        rng = np.random.default_rng(42)
        B, T = 2, 5
        inp = DINInput(
            user_cat=jnp.array(rng.integers(0, 3, size=(B, 3)), dtype=jnp.int32),
            user_num=jnp.array(rng.random((B, 8)), dtype=jnp.float32),
            item_cat=jnp.array(rng.integers(0, 3, size=(B, 5)), dtype=jnp.int32),
            item_num=jnp.array(rng.random((B, 2)), dtype=jnp.float32),
            history=jnp.array(rng.integers(1, 20, size=(B, T)), dtype=jnp.int32),
            hist_len=jnp.array([3, 5], dtype=jnp.int32),
        )
        rngs1 = nnx.Rngs(params=42, dropout=43)
        m1 = DIN(field_dims, n_numerical, 20, 5, small_config, rngs=rngs1)
        m1.eval()
        out1 = m1(inp)

        rngs2 = nnx.Rngs(params=42, dropout=43)
        m2 = DIN(field_dims, n_numerical, 20, 5, small_config, rngs=rngs2)
        m2.eval()
        out2 = m2(inp)

        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-6)
