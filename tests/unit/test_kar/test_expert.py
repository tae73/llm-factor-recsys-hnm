"""Tests for KAR Expert MLP module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.config import ExpertConfig
from src.kar.expert import Expert


@pytest.fixture
def default_config():
    return ExpertConfig(d_enc=768, d_hidden=256, d_rec=64, n_layers=2, dropout_rate=0.1)


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


def test_expert_output_shape(default_config, rngs):
    """Expert output shape: (B, d_rec)."""
    expert = Expert(default_config, rngs=rngs)
    h = jnp.ones((4, 768))
    out = expert(h)
    assert out.shape == (4, 64)


def test_expert_single_sample(default_config, rngs):
    """Expert works with batch size 1."""
    expert = Expert(default_config, rngs=rngs)
    h = jnp.ones((1, 768))
    out = expert(h)
    assert out.shape == (1, 64)


def test_expert_different_d_rec(rngs):
    """Expert respects custom d_rec."""
    config = ExpertConfig(d_enc=768, d_hidden=128, d_rec=32, n_layers=2)
    expert = Expert(config, rngs=rngs)
    h = jnp.ones((4, 768))
    out = expert(h)
    assert out.shape == (4, 32)


def test_expert_gradient_flows(default_config, rngs):
    """Gradients flow through expert."""
    expert = Expert(default_config, rngs=rngs)
    h = jnp.ones((4, 768))

    def loss_fn(model):
        return jnp.mean(model(h) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(expert)
    grad_leaves = jax.tree.leaves(grads)
    assert any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, 'shape'))


def test_expert_jit_compatible(default_config, rngs):
    """Expert can be JIT-compiled."""
    expert = Expert(default_config, rngs=rngs)

    @nnx.jit
    def forward(model, h):
        return model(h)

    h = jnp.ones((4, 768))
    out = forward(expert, h)
    assert out.shape == (4, 64)


def test_expert_eval_mode_no_dropout(default_config, rngs):
    """Expert in eval mode produces deterministic output."""
    expert = Expert(default_config, rngs=rngs)
    expert.eval()
    h = jnp.ones((4, 768))
    out1 = expert(h)
    out2 = expert(h)
    assert jnp.allclose(out1, out2)
    expert.train()


def test_expert_different_input_dim(rngs):
    """Expert works with non-768 input dim."""
    config = ExpertConfig(d_enc=384, d_hidden=128, d_rec=32, n_layers=2)
    expert = Expert(config, rngs=rngs)
    h = jnp.ones((4, 384))
    out = expert(h)
    assert out.shape == (4, 32)


def test_expert_zero_input(default_config, rngs):
    """Expert handles zero input without NaN."""
    expert = Expert(default_config, rngs=rngs)
    expert.eval()
    h = jnp.zeros((4, 768))
    out = expert(h)
    assert not jnp.any(jnp.isnan(out))


def test_two_experts_independent_params(default_config, rngs):
    """Two Expert instances have independent parameters."""
    expert1 = Expert(default_config, rngs=rngs)
    expert2 = Expert(default_config, rngs=nnx.Rngs(params=42, dropout=43))
    h = jnp.ones((4, 768))
    expert1.eval()
    expert2.eval()
    out1 = expert1(h)
    out2 = expert2(h)
    assert not jnp.allclose(out1, out2)


def test_expert_3_layers(rngs):
    """Expert with 3 layers."""
    config = ExpertConfig(d_enc=768, d_hidden=256, d_rec=64, n_layers=3)
    expert = Expert(config, rngs=rngs)
    h = jnp.ones((4, 768))
    out = expert(h)
    assert out.shape == (4, 64)
