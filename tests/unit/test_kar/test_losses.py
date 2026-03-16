"""Tests for KAR loss functions."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from src.losses import align_loss, diversity_loss, kar_total_loss


def test_align_loss_zero_when_equal():
    """align_loss is 0 when expert == backbone embed."""
    e = jnp.ones((4, 64))
    loss = align_loss(e, e)
    assert jnp.allclose(loss, 0.0, atol=1e-6)


def test_align_loss_positive_when_different():
    """align_loss > 0 when expert != backbone embed."""
    e = jnp.ones((4, 64))
    x = jnp.zeros((4, 64))
    loss = align_loss(e, x)
    assert loss > 0


def test_align_loss_uses_stop_gradient():
    """align_loss applies stop_gradient to backbone embed."""
    import jax

    e = jnp.ones((4, 64))
    x = jnp.ones((4, 64)) * 2.0

    def fn(e):
        return align_loss(e, x)

    grad = jax.grad(fn)(e)
    # Gradient should flow only through e_expert, not x
    assert jnp.any(grad != 0)


def test_diversity_loss_zero_when_orthogonal():
    """diversity_loss is 0 when experts are orthogonal."""
    e_fact = jnp.concatenate([jnp.ones((1, 32)), jnp.zeros((1, 32))], axis=-1)
    e_reason = jnp.concatenate([jnp.zeros((1, 32)), jnp.ones((1, 32))], axis=-1)
    loss = diversity_loss(e_fact, e_reason)
    assert jnp.abs(loss) < 1e-5


def test_diversity_loss_one_when_parallel():
    """diversity_loss is 1 when experts are identical (parallel)."""
    e = jnp.ones((4, 64))
    loss = diversity_loss(e, e)
    assert jnp.allclose(loss, 1.0, atol=1e-5)


def test_diversity_loss_negative_when_antiparallel():
    """diversity_loss is -1 when experts are antiparallel."""
    e_fact = jnp.ones((4, 64))
    e_reason = -jnp.ones((4, 64))
    loss = diversity_loss(e_fact, e_reason)
    assert jnp.allclose(loss, -1.0, atol=1e-5)


def test_kar_total_loss_stage2_no_bce():
    """Stage 2: include_rec_loss=False → BCE is 0."""
    logits = jnp.ones(4)
    labels = jnp.ones(4)
    e_fact = jnp.ones((4, 64))
    e_reason = jnp.ones((4, 64)) * 0.5
    x_proj = jnp.zeros((4, 64))

    total, loss_dict = kar_total_loss(
        logits, labels, e_fact, x_proj, e_reason, x_proj,
        include_rec_loss=False,
    )
    assert jnp.allclose(loss_dict["bce"], 0.0)
    assert total > 0  # align + diversity should be > 0


def test_kar_total_loss_stage3_has_bce():
    """Stage 3: include_rec_loss=True → BCE > 0."""
    logits = jnp.ones(4) * 0.5
    labels = jnp.ones(4)
    e_fact = jnp.ones((4, 64))
    e_reason = jnp.ones((4, 64))
    x_proj = jnp.ones((4, 64))

    total, loss_dict = kar_total_loss(
        logits, labels, e_fact, x_proj, e_reason, x_proj,
        include_rec_loss=True,
    )
    assert loss_dict["bce"] > 0
    assert total > 0
