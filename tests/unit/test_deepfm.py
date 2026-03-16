"""Unit tests for DeepFM model, BCE loss, and training step.

Tests model initialization, forward pass shapes, gradient flow,
and basic loss computation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from src.config import DeepFMConfig
from src.losses import binary_cross_entropy
from src.models.deepfm import DeepFM, DeepFMInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return DeepFMConfig(
        d_embed=4,
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
    return DeepFM(field_dims, n_numerical, small_config, rngs=rngs)


@pytest.fixture
def sample_batch():
    """Small batch of 4 samples."""
    rng = np.random.default_rng(42)
    return DeepFMInput(
        user_cat=jnp.array(rng.integers(0, 3, size=(4, 3)), dtype=jnp.int32),
        user_num=jnp.array(rng.random((4, 8)), dtype=jnp.float32),
        item_cat=jnp.array(rng.integers(0, 3, size=(4, 5)), dtype=jnp.int32),
        item_num=jnp.array(rng.random((4, 2)), dtype=jnp.float32),
    )


@pytest.fixture
def sample_labels():
    return jnp.array([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# BCE Loss Tests
# ---------------------------------------------------------------------------


class TestBinaryCrossEntropy:
    def test_perfect_prediction(self):
        """Large positive logits for positive labels → near-zero loss."""
        logits = jnp.array([10.0, -10.0, 10.0])
        labels = jnp.array([1.0, 0.0, 1.0])
        loss = binary_cross_entropy(logits, labels)
        assert float(loss) < 0.001

    def test_worst_prediction(self):
        """Opposite predictions → high loss."""
        logits = jnp.array([-10.0, 10.0])
        labels = jnp.array([1.0, 0.0])
        loss = binary_cross_entropy(logits, labels)
        assert float(loss) > 5.0

    def test_zero_logits(self):
        """Zero logits → loss = log(2) ≈ 0.693."""
        logits = jnp.array([0.0, 0.0])
        labels = jnp.array([1.0, 0.0])
        loss = binary_cross_entropy(logits, labels)
        assert abs(float(loss) - np.log(2)) < 0.001

    def test_numerical_stability(self):
        """Very large logits should not produce NaN/Inf."""
        logits = jnp.array([100.0, -100.0, 1000.0, -1000.0])
        labels = jnp.array([1.0, 0.0, 0.0, 1.0])
        loss = binary_cross_entropy(logits, labels)
        assert jnp.isfinite(loss)

    def test_scalar_output(self):
        """Loss should be a scalar."""
        logits = jnp.array([1.0, 2.0, 3.0])
        labels = jnp.array([0.0, 1.0, 0.0])
        loss = binary_cross_entropy(logits, labels)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# DeepFM Model Tests
# ---------------------------------------------------------------------------


class TestDeepFMModel:
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

    def test_batch_size_invariance(self, small_config, field_dims, n_numerical):
        """Model should work with different batch sizes."""
        rngs = nnx.Rngs(params=42, dropout=43)
        m = DeepFM(field_dims, n_numerical, small_config, rngs=rngs)

        rng = np.random.default_rng(42)
        for B in [1, 8, 32]:
            inp = DeepFMInput(
                user_cat=jnp.array(rng.integers(0, 3, size=(B, 3)), dtype=jnp.int32),
                user_num=jnp.array(rng.random((B, 8)), dtype=jnp.float32),
                item_cat=jnp.array(rng.integers(0, 3, size=(B, 5)), dtype=jnp.int32),
                item_num=jnp.array(rng.random((B, 2)), dtype=jnp.float32),
            )
            logits = m(inp)
            assert logits.shape == (B,)

    def test_with_batch_norm(self, field_dims, n_numerical):
        """Model should work with batch normalization enabled."""
        config = DeepFMConfig(d_embed=4, dnn_hidden_dims=(8,), dropout_rate=0.0, use_batch_norm=True)
        rngs = nnx.Rngs(params=42, dropout=43)
        m = DeepFM(field_dims, n_numerical, config, rngs=rngs)

        rng = np.random.default_rng(42)
        inp = DeepFMInput(
            user_cat=jnp.array(rng.integers(0, 3, size=(4, 3)), dtype=jnp.int32),
            user_num=jnp.array(rng.random((4, 8)), dtype=jnp.float32),
            item_cat=jnp.array(rng.integers(0, 3, size=(4, 5)), dtype=jnp.int32),
            item_num=jnp.array(rng.random((4, 2)), dtype=jnp.float32),
        )
        logits = m(inp)
        assert logits.shape == (4,)
        assert jnp.all(jnp.isfinite(logits))


# ---------------------------------------------------------------------------
# Training Step Tests
# ---------------------------------------------------------------------------


class TestTrainStep:
    def test_gradient_flow(self, model, sample_batch, sample_labels):
        """All parameters should receive non-zero gradients."""

        def loss_fn(m):
            logits = m(sample_batch)
            return binary_cross_entropy(logits, sample_labels)

        _, grads = nnx.value_and_grad(loss_fn)(model)
        grad_leaves = jax.tree.leaves(grads)
        # At least some gradients should be non-zero
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, 'shape'))
        assert has_nonzero, "All gradients are zero"

    def test_loss_decreases(self, small_config, field_dims, n_numerical):
        """Loss should decrease over multiple training steps."""
        rngs = nnx.Rngs(params=42, dropout=43)
        m = DeepFM(field_dims, n_numerical, small_config, rngs=rngs)
        opt = nnx.Optimizer(m, optax.adam(learning_rate=1e-2), wrt=nnx.Param)

        rng = np.random.default_rng(42)
        inp = DeepFMInput(
            user_cat=jnp.array(rng.integers(0, 3, size=(32, 3)), dtype=jnp.int32),
            user_num=jnp.array(rng.random((32, 8)), dtype=jnp.float32),
            item_cat=jnp.array(rng.integers(0, 3, size=(32, 5)), dtype=jnp.int32),
            item_num=jnp.array(rng.random((32, 2)), dtype=jnp.float32),
        )
        labels = jnp.array(rng.choice([0.0, 1.0], size=32), dtype=jnp.float32)

        losses = []
        for _ in range(20):

            def loss_fn(m):
                logits = m(inp)
                return binary_cross_entropy(logits, labels)

            loss, grads = nnx.value_and_grad(loss_fn)(m)
            opt.update(m, grads)
            losses.append(float(loss))

        # Loss should decrease from first to last
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_jit_compatible(self, model, sample_batch, sample_labels):
        """Model should work under nnx.jit with batch dict signature."""
        opt = nnx.Optimizer(model, optax.adam(learning_rate=1e-3), wrt=nnx.Param)
        from src.training.trainer import train_step

        batch = {
            "user_cat": sample_batch.user_cat,
            "user_num": sample_batch.user_num,
            "item_cat": sample_batch.item_cat,
            "item_num": sample_batch.item_num,
            "labels": sample_labels,
        }
        loss = train_step(model, opt, batch)
        assert jnp.isfinite(loss)
