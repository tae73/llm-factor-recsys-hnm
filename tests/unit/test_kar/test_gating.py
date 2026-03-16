"""Tests for KAR Gating modules (G1-G4)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from flax import nnx

from src.config import GatingConfig
from src.kar.gating import (
    G1FixedGating,
    G2ExpertGating,
    G3ContextGating,
    G4CrossGating,
    create_gating,
)


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


@pytest.fixture
def expert_outputs():
    """Fake expert outputs (B=4, d_rec=64)."""
    e_fact = jnp.ones((4, 64)) * 0.5
    e_reason = jnp.ones((4, 64)) * 0.3
    return e_fact, e_reason


# --- G1: Fixed Gating ---

def test_g1_output_shape(rngs, expert_outputs):
    gate = G1FixedGating(d_rec=64, rngs=rngs)
    g_f, g_r = gate(*expert_outputs)
    assert g_f.shape == (4, 1)
    assert g_r.shape == (4, 1)


def test_g1_weights_sum_to_one(rngs, expert_outputs):
    gate = G1FixedGating(d_rec=64, rngs=rngs)
    g_f, g_r = gate(*expert_outputs)
    sums = g_f + g_r
    assert jnp.allclose(sums, 1.0, atol=1e-6)


def test_g1_same_for_all_samples(rngs, expert_outputs):
    gate = G1FixedGating(d_rec=64, rngs=rngs)
    g_f, g_r = gate(*expert_outputs)
    assert jnp.allclose(g_f[0], g_f[1])


# --- G2: Expert-Conditioned Gating ---

def test_g2_output_shape(rngs, expert_outputs):
    gate = G2ExpertGating(d_rec=64, rngs=rngs)
    g_f, g_r = gate(*expert_outputs)
    assert g_f.shape == (4, 1)
    assert g_r.shape == (4, 1)


def test_g2_weights_sum_to_one(rngs, expert_outputs):
    gate = G2ExpertGating(d_rec=64, rngs=rngs)
    g_f, g_r = gate(*expert_outputs)
    sums = g_f + g_r
    assert jnp.allclose(sums, 1.0, atol=1e-6)


def test_g2_input_dependent(rngs):
    """G2 produces different weights for different inputs."""
    gate = G2ExpertGating(d_rec=64, rngs=rngs)
    e1 = jnp.ones((1, 64))
    e2 = jnp.ones((1, 64)) * 2.0
    g_f1, _ = gate(e1, e1)
    g_f2, _ = gate(e2, e1)
    assert not jnp.allclose(g_f1, g_f2)


# --- G3: Context Gating ---

def test_g3_output_shape(rngs, expert_outputs):
    gate = G3ContextGating(d_rec=64, d_context=8, rngs=rngs)
    context = jnp.ones((4, 8))
    g_f, g_r = gate(*expert_outputs, context=context)
    assert g_f.shape == (4, 1)
    assert g_r.shape == (4, 1)


def test_g3_weights_sum_to_one(rngs, expert_outputs):
    gate = G3ContextGating(d_rec=64, d_context=8, rngs=rngs)
    context = jnp.ones((4, 8))
    g_f, g_r = gate(*expert_outputs, context=context)
    assert jnp.allclose(g_f + g_r, 1.0, atol=1e-6)


def test_g3_requires_context(rngs, expert_outputs):
    gate = G3ContextGating(d_rec=64, d_context=8, rngs=rngs)
    with pytest.raises(ValueError, match="requires context"):
        gate(*expert_outputs, context=None)


# --- G4: Cross Gating ---

def test_g4_output_shape(rngs, expert_outputs):
    gate = G4CrossGating(d_rec=64, rngs=rngs)
    g_f, g_r = gate(*expert_outputs)
    assert g_f.shape == (4, 1)
    assert g_r.shape == (4, 1)


def test_g4_weights_sum_to_one(rngs, expert_outputs):
    gate = G4CrossGating(d_rec=64, rngs=rngs)
    g_f, g_r = gate(*expert_outputs)
    assert jnp.allclose(g_f + g_r, 1.0, atol=1e-6)


# --- Factory ---

def test_create_gating_g1(rngs):
    gate = create_gating(GatingConfig(variant="g1"), d_rec=64, rngs=rngs)
    assert isinstance(gate, G1FixedGating)


def test_create_gating_g2(rngs):
    gate = create_gating(GatingConfig(variant="g2"), d_rec=64, rngs=rngs)
    assert isinstance(gate, G2ExpertGating)


def test_create_gating_g3(rngs):
    gate = create_gating(GatingConfig(variant="g3", d_context=8), d_rec=64, rngs=rngs)
    assert isinstance(gate, G3ContextGating)


def test_create_gating_g4(rngs):
    gate = create_gating(GatingConfig(variant="g4"), d_rec=64, rngs=rngs)
    assert isinstance(gate, G4CrossGating)


def test_create_gating_invalid(rngs):
    with pytest.raises(ValueError, match="Unknown gating"):
        create_gating(GatingConfig(variant="g99"), d_rec=64, rngs=rngs)


def test_create_gating_g3_requires_d_context(rngs):
    with pytest.raises(ValueError, match="d_context > 0"):
        create_gating(GatingConfig(variant="g3", d_context=0), d_rec=64, rngs=rngs)
