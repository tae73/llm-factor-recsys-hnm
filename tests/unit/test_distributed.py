"""Unit tests for distributed training infrastructure (Mesh + NamedSharding).

Tests run on CPU (macOS, n_devices=1) to verify sharding code path
works correctly as a no-op on single device.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from src.config import DeepFMConfig
from src.models.deepfm import DeepFM
from src.training.trainer import train_step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mesh():
    """Single-device mesh (CPU)."""
    devices = jax.devices()
    return Mesh(np.array(devices).reshape(-1), ("data",))


@pytest.fixture
def data_sharding(mesh):
    return NamedSharding(mesh, PartitionSpec("data"))


@pytest.fixture
def small_model():
    config = DeepFMConfig(d_embed=4, dnn_hidden_dims=(8, 4), dropout_rate=0.0, use_batch_norm=False)
    rngs = nnx.Rngs(params=42, dropout=43)
    return DeepFM(
        field_dims=[5, 3, 3, 10, 6, 4, 8, 3],
        n_numerical=10,
        config=config,
        rngs=rngs,
    )


@pytest.fixture
def sample_batch():
    rng = np.random.default_rng(42)
    return {
        "user_cat": jnp.array(rng.integers(0, 3, size=(4, 3)), dtype=jnp.int32),
        "user_num": jnp.array(rng.random((4, 8)), dtype=jnp.float32),
        "item_cat": jnp.array(rng.integers(0, 3, size=(4, 5)), dtype=jnp.int32),
        "item_num": jnp.array(rng.random((4, 2)), dtype=jnp.float32),
        "labels": jnp.array([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32),
    }


# ---------------------------------------------------------------------------
# Mesh Creation
# ---------------------------------------------------------------------------


class TestMeshCreation:
    def test_single_device_mesh(self, mesh):
        """Valid 1-device mesh should be created."""
        assert mesh is not None
        assert len(mesh.devices.flat) >= 1

    def test_mesh_axis_names(self, mesh):
        """Mesh should have 'data' axis."""
        assert mesh.axis_names == ("data",)


# ---------------------------------------------------------------------------
# Data Sharding
# ---------------------------------------------------------------------------


class TestDataSharding:
    def test_device_put_preserves_shape(self, data_sharding, sample_batch):
        """Sharding should preserve array shapes."""
        sharded = jax.device_put(sample_batch, data_sharding)
        for key in sample_batch:
            assert sharded[key].shape == sample_batch[key].shape

    def test_device_put_single_device_noop(self, data_sharding, sample_batch):
        """On single device, values should be preserved exactly."""
        sharded = jax.device_put(sample_batch, data_sharding)
        for key in sample_batch:
            np.testing.assert_array_almost_equal(
                np.array(sharded[key]), np.array(sample_batch[key])
            )


# ---------------------------------------------------------------------------
# Sharded Train Step
# ---------------------------------------------------------------------------


class TestShardedTrainStep:
    def test_train_step_with_sharded_batch(self, small_model, data_sharding, sample_batch):
        """train_step should work with sharded batch inputs."""
        opt = nnx.Optimizer(small_model, optax.adam(learning_rate=1e-3), wrt=nnx.Param)
        sharded_batch = jax.device_put(sample_batch, data_sharding)
        loss = train_step(small_model, opt, sharded_batch)
        assert loss.shape == ()

    def test_loss_finite_with_sharding(self, small_model, data_sharding, sample_batch):
        """Loss should be finite after sharded train step."""
        opt = nnx.Optimizer(small_model, optax.adam(learning_rate=1e-3), wrt=nnx.Param)
        sharded_batch = jax.device_put(sample_batch, data_sharding)
        loss = train_step(small_model, opt, sharded_batch)
        assert jnp.isfinite(loss)
