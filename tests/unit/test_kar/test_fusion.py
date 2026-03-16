"""Tests for KAR Fusion modules (F1-F4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.config import FusionConfig
from src.kar.fusion import (
    F1ConcatFusion,
    F2AdditionFusion,
    F3GatedFusion,
    F4CrossAttentionFusion,
    create_fusion,
)


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


@pytest.fixture
def inputs():
    """Fake backbone embed (B=4, d_backbone=288) and aug (B=4, d_rec=64)."""
    x_backbone = jnp.ones((4, 288)) * 0.5
    e_aug = jnp.ones((4, 64)) * 0.3
    return x_backbone, e_aug


# --- F1: Concat ---

def test_f1_output_shape(rngs, inputs):
    fusion = F1ConcatFusion(d_backbone=288, d_rec=64, rngs=rngs)
    out = fusion(*inputs)
    assert out.shape == (4, 576)  # d_backbone * 2


def test_f1_gradient_flows(rngs, inputs):
    fusion = F1ConcatFusion(d_backbone=288, d_rec=64, rngs=rngs)

    def loss_fn(model):
        return jnp.mean(model(*inputs) ** 2)

    _, grads = nnx.value_and_grad(loss_fn)(fusion)
    grad_leaves = jax.tree.leaves(grads)
    assert any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, 'shape'))


# --- F2: Addition ---

def test_f2_output_shape(rngs, inputs):
    fusion = F2AdditionFusion(d_backbone=288, d_rec=64, rngs=rngs)
    out = fusion(*inputs)
    assert out.shape == (4, 288)  # same as backbone


def test_f2_preserves_dim(rngs, inputs):
    fusion = F2AdditionFusion(d_backbone=288, d_rec=64, rngs=rngs)
    out = fusion(*inputs)
    assert out.shape == inputs[0].shape


def test_f2_alpha_effect(rngs, inputs):
    """Non-zero alpha produces different output from backbone."""
    fusion = F2AdditionFusion(d_backbone=288, d_rec=64, alpha_init=0.5, rngs=rngs)
    out = fusion(*inputs)
    assert not jnp.allclose(out, inputs[0])


def test_f2_gradient_flows(rngs, inputs):
    fusion = F2AdditionFusion(d_backbone=288, d_rec=64, rngs=rngs)

    def loss_fn(model):
        return jnp.mean(model(*inputs) ** 2)

    _, grads = nnx.value_and_grad(loss_fn)(fusion)
    grad_leaves = jax.tree.leaves(grads)
    assert any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, 'shape'))


# --- F3: Gated ---

def test_f3_output_shape(rngs, inputs):
    fusion = F3GatedFusion(d_backbone=288, d_rec=64, rngs=rngs)
    out = fusion(*inputs)
    assert out.shape == (4, 288)


def test_f3_gradient_flows(rngs, inputs):
    fusion = F3GatedFusion(d_backbone=288, d_rec=64, rngs=rngs)

    def loss_fn(model):
        return jnp.mean(model(*inputs) ** 2)

    _, grads = nnx.value_and_grad(loss_fn)(fusion)
    grad_leaves = jax.tree.leaves(grads)
    assert any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, 'shape'))


# --- F4: Cross-Attention ---

def test_f4_output_shape(rngs, inputs):
    fusion = F4CrossAttentionFusion(d_backbone=288, d_rec=64, n_heads=4, rngs=rngs)
    out = fusion(*inputs)
    assert out.shape == (4, 288)


def test_f4_gradient_flows(rngs, inputs):
    fusion = F4CrossAttentionFusion(d_backbone=288, d_rec=64, n_heads=4, rngs=rngs)

    def loss_fn(model):
        return jnp.mean(model(*inputs) ** 2)

    _, grads = nnx.value_and_grad(loss_fn)(fusion)
    grad_leaves = jax.tree.leaves(grads)
    assert any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, 'shape'))


# --- Factory ---

def test_create_fusion_f1(rngs):
    f = create_fusion(FusionConfig(variant="f1"), 288, 64, rngs=rngs)
    assert isinstance(f, F1ConcatFusion)


def test_create_fusion_f2(rngs):
    f = create_fusion(FusionConfig(variant="f2"), 288, 64, rngs=rngs)
    assert isinstance(f, F2AdditionFusion)


def test_create_fusion_f3(rngs):
    f = create_fusion(FusionConfig(variant="f3"), 288, 64, rngs=rngs)
    assert isinstance(f, F3GatedFusion)


def test_create_fusion_f4(rngs):
    f = create_fusion(FusionConfig(variant="f4"), 288, 64, rngs=rngs)
    assert isinstance(f, F4CrossAttentionFusion)


def test_create_fusion_invalid(rngs):
    with pytest.raises(ValueError, match="Unknown fusion"):
        create_fusion(FusionConfig(variant="f99"), 288, 64, rngs=rngs)
