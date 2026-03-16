"""Gating networks for KAR.

Dynamic weighting of factual and reasoning expert outputs.
4 variants (G1-G4) with a common interface:
    (e_fact, e_reason, context=None) → (g_fact, g_reason)

Each gate returns (B, 1) weights that sum to 1 per sample.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.config import GatingConfig


class G1FixedGating(nnx.Module):
    """G1: Learnable scalar gate, input-independent.

    A single learnable logit determines the fixed split between
    factual and reasoning experts (same for all samples).
    """

    def __init__(self, d_rec: int, *, rngs: nnx.Rngs):
        del d_rec  # unused, interface consistency
        # Logit for factual weight; reasoning = 1 - factual
        self.logit = nnx.Param(jnp.zeros(()))

    def __call__(
        self,
        e_fact: jax.Array,
        e_reason: jax.Array,
        context: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Returns (g_fact, g_reason) each (B, 1)."""
        g_fact = jax.nn.sigmoid(self.logit[...])  # scalar
        B = e_fact.shape[0]
        g_fact_batch = jnp.full((B, 1), g_fact)
        g_reason_batch = 1.0 - g_fact_batch
        return g_fact_batch, g_reason_batch


class G2ExpertGating(nnx.Module):
    """G2: Expert-conditioned gating (DEFAULT).

    Softmax(W · [e_fact; e_reason]) → per-sample weights.
    """

    def __init__(self, d_rec: int, *, rngs: nnx.Rngs):
        self.gate_linear = nnx.Linear(d_rec * 2, 2, rngs=rngs)

    def __call__(
        self,
        e_fact: jax.Array,
        e_reason: jax.Array,
        context: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Returns (g_fact, g_reason) each (B, 1)."""
        combined = jnp.concatenate([e_fact, e_reason], axis=-1)  # (B, 2*d_rec)
        logits = self.gate_linear(combined)  # (B, 2)
        weights = jax.nn.softmax(logits, axis=-1)  # (B, 2)
        return weights[:, 0:1], weights[:, 1:2]


class G3ContextGating(nnx.Module):
    """G3: Context-conditioned gating.

    Incorporates user demographic context alongside expert outputs.
    Softmax(W · [e_fact; e_reason; context]) → per-sample weights.
    """

    def __init__(self, d_rec: int, d_context: int, *, rngs: nnx.Rngs):
        self.gate_linear = nnx.Linear(d_rec * 2 + d_context, 2, rngs=rngs)

    def __call__(
        self,
        e_fact: jax.Array,
        e_reason: jax.Array,
        context: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Returns (g_fact, g_reason) each (B, 1).

        Args:
            context: (B, d_context) user demographic features. Required for G3.
        """
        if context is None:
            raise ValueError("G3ContextGating requires context input")
        combined = jnp.concatenate([e_fact, e_reason, context], axis=-1)
        logits = self.gate_linear(combined)
        weights = jax.nn.softmax(logits, axis=-1)
        return weights[:, 0:1], weights[:, 1:2]


class G4CrossGating(nnx.Module):
    """G4: Cross-interaction gating.

    Uses element-wise product of expert outputs for gating.
    Linear(e_fact ⊙ e_reason) → softmax → per-sample weights.
    """

    def __init__(self, d_rec: int, *, rngs: nnx.Rngs):
        self.gate_linear = nnx.Linear(d_rec, 2, rngs=rngs)

    def __call__(
        self,
        e_fact: jax.Array,
        e_reason: jax.Array,
        context: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Returns (g_fact, g_reason) each (B, 1)."""
        cross = e_fact * e_reason  # (B, d_rec) element-wise
        logits = self.gate_linear(cross)  # (B, 2)
        weights = jax.nn.softmax(logits, axis=-1)
        return weights[:, 0:1], weights[:, 1:2]


def create_gating(
    config: GatingConfig, d_rec: int, *, rngs: nnx.Rngs
) -> nnx.Module:
    """Factory: create gating network by variant name.

    Args:
        config: GatingConfig with variant and optional d_context.
        d_rec: Expert output dimension.
        rngs: NNX random number generators.

    Returns:
        Gating module instance.
    """
    if config.variant == "g1":
        return G1FixedGating(d_rec, rngs=rngs)
    if config.variant == "g2":
        return G2ExpertGating(d_rec, rngs=rngs)
    if config.variant == "g3":
        if config.d_context <= 0:
            raise ValueError("G3 requires d_context > 0")
        return G3ContextGating(d_rec, config.d_context, rngs=rngs)
    if config.variant == "g4":
        return G4CrossGating(d_rec, rngs=rngs)
    raise ValueError(f"Unknown gating variant '{config.variant}'. Choose from: g1, g2, g3, g4")
