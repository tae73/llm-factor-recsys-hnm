"""Fusion strategies for KAR.

Merge augmented expert vector (d_rec) into backbone embedding (d_backbone).
Each fusion module contains an internal Linear projection from d_rec → d_backbone
to bridge the dimension gap.

4 variants (F1-F4) with a common interface:
    (x_backbone, e_aug) → x_augmented

Where x_backbone shape depends on backbone type:
- DeepFM/DCNv2: (B, n_fields, d_embed) — stacked field embeddings
- LightGCN: (B, d) — user or item embedding
- DIN: (B, dnn_input_dim) — concatenated DNN input
- SASRec: (B, d) — user embedding

For stacked embeddings (DeepFM/DCNv2), fusion operates on the flattened vector.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.config import FusionConfig


class F1ConcatFusion(nnx.Module):
    """F1: Concatenation fusion.

    x_augmented = concat(x_backbone, proj(e_aug))
    Output dim = d_backbone + d_backbone (projected aug has same dim).

    Note: This changes the downstream input dimension, so the backbone's
    predict_from_embedding must accept the expanded input.
    """

    def __init__(self, d_backbone: int, d_rec: int, *, rngs: nnx.Rngs):
        self.projection = nnx.Linear(d_rec, d_backbone, rngs=rngs)
        self._d_backbone = d_backbone

    def __call__(self, x_backbone: jax.Array, e_aug: jax.Array) -> jax.Array:
        """Concat fusion.

        Args:
            x_backbone: (B, d_backbone) backbone embedding.
            e_aug: (B, d_rec) gated expert output.

        Returns:
            (B, d_backbone * 2) concatenated embedding.
        """
        projected = self.projection(e_aug)  # (B, d_backbone)
        return jnp.concatenate([x_backbone, projected], axis=-1)


class F2AdditionFusion(nnx.Module):
    """F2: Addition fusion (DEFAULT).

    x_augmented = x_backbone + α · proj(e_aug)

    Dimension-preserving. α is a learnable scalar initialized to alpha_init.
    """

    def __init__(
        self, d_backbone: int, d_rec: int, alpha_init: float = 0.1, *, rngs: nnx.Rngs
    ):
        self.projection = nnx.Linear(d_rec, d_backbone, rngs=rngs)
        self.alpha = nnx.Param(jnp.array(alpha_init))

    def __call__(self, x_backbone: jax.Array, e_aug: jax.Array) -> jax.Array:
        """Addition fusion.

        Args:
            x_backbone: (B, d_backbone) backbone embedding.
            e_aug: (B, d_rec) gated expert output.

        Returns:
            (B, d_backbone) augmented embedding.
        """
        projected = self.projection(e_aug)
        return x_backbone + self.alpha[...] * projected


class F3GatedFusion(nnx.Module):
    """F3: Gated fusion.

    gate = sigmoid(W_g · [x_backbone; proj(e_aug)])
    x_augmented = gate ⊙ x_backbone + (1 - gate) ⊙ proj(e_aug)

    Per-element gating for fine-grained control.
    """

    def __init__(self, d_backbone: int, d_rec: int, *, rngs: nnx.Rngs):
        self.projection = nnx.Linear(d_rec, d_backbone, rngs=rngs)
        self.gate_linear = nnx.Linear(d_backbone * 2, d_backbone, rngs=rngs)

    def __call__(self, x_backbone: jax.Array, e_aug: jax.Array) -> jax.Array:
        """Gated fusion.

        Args:
            x_backbone: (B, d_backbone) backbone embedding.
            e_aug: (B, d_rec) gated expert output.

        Returns:
            (B, d_backbone) gated augmented embedding.
        """
        projected = self.projection(e_aug)  # (B, d_backbone)
        combined = jnp.concatenate([x_backbone, projected], axis=-1)
        gate = jax.nn.sigmoid(self.gate_linear(combined))  # (B, d_backbone)
        return gate * x_backbone + (1.0 - gate) * projected


class F4CrossAttentionFusion(nnx.Module):
    """F4: Cross-attention fusion.

    Uses multi-head attention where x_backbone attends to projected e_aug.
    Residual connection preserves original backbone embedding.

    x_augmented = x_backbone + MHA(Q=x_backbone, KV=proj(e_aug))
    """

    def __init__(
        self, d_backbone: int, d_rec: int, n_heads: int = 4, *, rngs: nnx.Rngs
    ):
        self.projection = nnx.Linear(d_rec, d_backbone, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_backbone,
            decode=False,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(d_backbone, rngs=rngs)

    def __call__(self, x_backbone: jax.Array, e_aug: jax.Array) -> jax.Array:
        """Cross-attention fusion.

        Args:
            x_backbone: (B, d_backbone) backbone embedding.
            e_aug: (B, d_rec) gated expert output.

        Returns:
            (B, d_backbone) attention-augmented embedding.
        """
        projected = self.projection(e_aug)  # (B, d_backbone)

        # MHA expects (B, T, d) — add sequence dim
        q = x_backbone[:, None, :]  # (B, 1, d_backbone)
        kv = projected[:, None, :]  # (B, 1, d_backbone)

        attn_out = self.attention(q, kv)  # (B, 1, d_backbone)
        attn_out = attn_out.squeeze(1)  # (B, d_backbone)

        return self.norm(x_backbone + attn_out)


def create_fusion(
    config: FusionConfig, d_backbone: int, d_rec: int, *, rngs: nnx.Rngs
) -> nnx.Module:
    """Factory: create fusion module by variant name.

    Args:
        config: FusionConfig with variant, alpha_init, n_heads.
        d_backbone: Backbone embedding dimension (flattened).
        d_rec: Expert output dimension.
        rngs: NNX random number generators.

    Returns:
        Fusion module instance.
    """
    if config.variant == "f1":
        return F1ConcatFusion(d_backbone, d_rec, rngs=rngs)
    if config.variant == "f2":
        return F2AdditionFusion(d_backbone, d_rec, config.alpha_init, rngs=rngs)
    if config.variant == "f3":
        return F3GatedFusion(d_backbone, d_rec, rngs=rngs)
    if config.variant == "f4":
        return F4CrossAttentionFusion(d_backbone, d_rec, config.n_heads, rngs=rngs)
    raise ValueError(f"Unknown fusion variant '{config.variant}'. Choose from: f1, f2, f3, f4")
