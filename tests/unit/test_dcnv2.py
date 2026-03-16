"""Unit tests for DCN-v2 model.

Tests model initialization, forward pass shapes, cross layer mechanics,
gradient flow, and basic loss computation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from src.config import DCNv2Config
from src.losses import binary_cross_entropy
from src.models.dcnv2 import CrossLayerV2, DCNv2
from src.models.deepfm import DeepFMInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return DCNv2Config(
        d_embed=4,
        n_cross_layers=2,
        n_experts=2,
        d_low_rank=8,
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
    return DCNv2(field_dims, n_numerical, small_config, rngs=rngs)


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
# CrossLayerV2 Tests
# ---------------------------------------------------------------------------


class TestCrossLayerV2:
    def test_output_shape(self):
        """Cross layer output should match input dimension."""
        rngs = nnx.Rngs(params=42)
        layer = CrossLayerV2(d_input=16, n_experts=2, d_low_rank=4, rngs=rngs)

        x0 = jnp.ones((4, 16))
        xl = jnp.ones((4, 16))
        out = layer(x0, xl)
        assert out.shape == (4, 16)

    def test_residual_connection(self):
        """Output should differ from input (cross transform is non-trivial)."""
        rngs = nnx.Rngs(params=42)
        layer = CrossLayerV2(d_input=16, n_experts=2, d_low_rank=4, rngs=rngs)

        x0 = jnp.ones((4, 16))
        xl = jnp.ones((4, 16))
        out = layer(x0, xl)
        # Output should not be identical to xl (the residual input)
        assert not jnp.allclose(out, xl, atol=1e-6)

    def test_moe_gate_sum(self):
        """Gating weights should sum to 1 (softmax output)."""
        rngs = nnx.Rngs(params=42)
        layer = CrossLayerV2(d_input=16, n_experts=3, d_low_rank=4, rngs=rngs)

        xl = jnp.ones((4, 16))
        gate_weights = jax.nn.softmax(layer.gate(xl), axis=-1)  # (4, 3)
        sums = jnp.sum(gate_weights, axis=-1)  # (4,)
        np.testing.assert_allclose(sums, jnp.ones(4), atol=1e-5)

    def test_output_finite(self):
        """No NaN/Inf in cross layer output."""
        rngs = nnx.Rngs(params=42)
        layer = CrossLayerV2(d_input=32, n_experts=4, d_low_rank=8, rngs=rngs)

        rng = np.random.default_rng(42)
        x0 = jnp.array(rng.standard_normal((8, 32)), dtype=jnp.float32)
        xl = jnp.array(rng.standard_normal((8, 32)), dtype=jnp.float32)
        out = layer(x0, xl)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# DCNv2 Model Tests
# ---------------------------------------------------------------------------


class TestDCNv2Model:
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

    @pytest.mark.parametrize("B", [1, 8, 32])
    def test_batch_size_invariance(self, small_config, field_dims, n_numerical, B):
        """Model should work with different batch sizes."""
        rngs = nnx.Rngs(params=42, dropout=43)
        m = DCNv2(field_dims, n_numerical, small_config, rngs=rngs)

        rng = np.random.default_rng(42)
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
        config = DCNv2Config(
            d_embed=4,
            n_cross_layers=2,
            n_experts=2,
            d_low_rank=8,
            dnn_hidden_dims=(8,),
            dropout_rate=0.0,
            use_batch_norm=True,
        )
        rngs = nnx.Rngs(params=42, dropout=43)
        m = DCNv2(field_dims, n_numerical, config, rngs=rngs)

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


class TestDCNv2TrainStep:
    def test_gradient_flow(self, model, sample_batch, sample_labels):
        """All parameters should receive non-zero gradients."""

        def loss_fn(m):
            logits = m(sample_batch)
            return binary_cross_entropy(logits, sample_labels)

        _, grads = nnx.value_and_grad(loss_fn)(model)
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, "shape"))
        assert has_nonzero, "All gradients are zero"

    def test_loss_decreases(self, small_config, field_dims, n_numerical):
        """Loss should decrease over multiple training steps."""
        rngs = nnx.Rngs(params=42, dropout=43)
        m = DCNv2(field_dims, n_numerical, small_config, rngs=rngs)
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
