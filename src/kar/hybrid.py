"""KARModel: Composition wrapper unifying backbone + Expert + Gating + Fusion.

Architecture flow:
    1. Expert forward: h_fact (768) → e_fact (d_rec), h_reason (768) → e_reason (d_rec)
    2. Gating: (e_fact, e_reason) → (g_fact, g_reason) each (B, 1)
    3. e_aug = g_fact * e_fact + g_reason * e_reason  (B, d_rec)
    4. backbone.embed(base_input) → x_backbone
    5. fusion(flatten(x_backbone), e_aug) → x_augmented
    6. backbone.predict_from_embedding(reshape(x_augmented)) → logits (B,)

KARModel owns the backbone — callers interact only with KARModel.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from src.config import KARConfig
from src.kar.expert import Expert
from src.kar.fusion import create_fusion
from src.kar.gating import create_gating


class KARInput(NamedTuple):
    """Input to KARModel."""

    base_input: Any  # DeepFMInput, LightGCNInput, DINInput, SASRecInput
    h_fact: jax.Array  # (B, 768) item BGE embedding
    h_reason: jax.Array  # (B, 768) user BGE embedding
    context: jax.Array | None = None  # (B, d_context) for G3 gating
    target_item_idx: jax.Array | None = None  # SASRec target item


class KARModel(nnx.Module):
    """Knowledge-Augmented Recommendation model.

    Composition: backbone + factual_expert + reasoning_expert + gating + fusion.

    Args:
        backbone: Pre-initialized backbone model (DeepFM, DCNv2, etc.).
        backbone_name: Name for dispatch ("deepfm", "dcnv2", "lightgcn", "din", "sasrec").
        kar_config: KAR module configuration.
        d_backbone: Backbone flattened embedding dimension.
        rngs: NNX random number generators.
    """

    def __init__(
        self,
        backbone: nnx.Module,
        backbone_name: str,
        kar_config: KARConfig,
        d_backbone: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.backbone = backbone
        self.backbone_name = backbone_name
        self._d_backbone = d_backbone

        expert_config = kar_config.expert
        self.factual_expert = Expert(expert_config, rngs=rngs)
        self.reasoning_expert = Expert(expert_config, rngs=rngs)

        d_rec = expert_config.d_rec
        self.gating = create_gating(kar_config.gating, d_rec, rngs=rngs)
        self.fusion = create_fusion(kar_config.fusion, d_backbone, d_rec, rngs=rngs)

        # Projection from d_backbone → d_rec for align_loss computation
        self.align_proj = nnx.Linear(d_backbone, d_rec, rngs=rngs)

    def __call__(self, x: KARInput) -> jax.Array:
        """Forward pass → logits (B,)."""
        logits, _ = self.forward_with_intermediates(x)
        return logits

    def predict_proba(self, x: KARInput) -> jax.Array:
        """Inference: returns sigmoid probabilities (B,)."""
        return jax.nn.sigmoid(self.__call__(x))

    def forward_with_intermediates(
        self, x: KARInput
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Forward pass returning logits + intermediate tensors for loss computation.

        Returns:
            (logits, intermediates) where intermediates has keys:
            - e_fact: (B, d_rec) factual expert output
            - e_reason: (B, d_rec) reasoning expert output
            - g_fact: (B, 1) factual gating weight
            - g_reason: (B, 1) reasoning gating weight
            - x_backbone_flat: (B, d_backbone) flattened backbone embedding
        """
        # 1. Expert forward
        e_fact = self.factual_expert(x.h_fact)  # (B, d_rec)
        e_reason = self.reasoning_expert(x.h_reason)  # (B, d_rec)

        # 2. Gating
        g_fact, g_reason = self.gating(e_fact, e_reason, x.context)

        # 3. Augmented vector
        e_aug = g_fact * e_fact + g_reason * e_reason  # (B, d_rec)

        # 4. Backbone embed
        x_backbone_flat = self._backbone_embed(x)  # (B, d_backbone)

        # 5. Fusion
        x_augmented = self.fusion(x_backbone_flat, e_aug)  # (B, d_backbone) or (B, d_backbone*2)

        # 6. Predict from augmented embedding
        logits = self._backbone_predict(x_augmented, x)

        intermediates = {
            "e_fact": e_fact,
            "e_reason": e_reason,
            "g_fact": g_fact,
            "g_reason": g_reason,
            "x_backbone_flat": x_backbone_flat,
        }
        return logits, intermediates

    def get_expert_outputs(
        self, h_fact: jax.Array, h_reason: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Pre-store: compute Expert outputs only.

        Args:
            h_fact: (B, 768) item BGE embeddings.
            h_reason: (B, 768) user BGE embeddings.

        Returns:
            (e_fact, e_reason): each (B, d_rec).
        """
        return self.factual_expert(h_fact), self.reasoning_expert(h_reason)

    def get_align_targets(
        self, x_backbone_flat: jax.Array
    ) -> jax.Array:
        """Project backbone embedding to d_rec for align_loss.

        Args:
            x_backbone_flat: (B, d_backbone) flattened backbone embedding.

        Returns:
            (B, d_rec) projected embedding.
        """
        return self.align_proj(x_backbone_flat)

    # ------------------------------------------------------------------
    # Private: backbone-specific embed/predict dispatch
    # ------------------------------------------------------------------

    def _backbone_embed(self, x: KARInput) -> jax.Array:
        """Extract flattened backbone embedding. Shape (B, d_backbone)."""
        name = self.backbone_name

        if name in ("deepfm",):
            stacked, _ = self.backbone.embed(x.base_input)
            return stacked.reshape(stacked.shape[0], -1)

        if name in ("dcnv2",):
            stacked = self.backbone.embed(x.base_input)
            return stacked.reshape(stacked.shape[0], -1)

        if name == "lightgcn":
            u, i = self.backbone.embed(x.base_input)
            return jnp.concatenate([u, i], axis=-1)

        if name == "din":
            user_interest, target_query, static_flat = self.backbone.embed(x.base_input)
            return jnp.concatenate([user_interest, target_query, static_flat], axis=-1)

        if name == "sasrec":
            user_embed, target_embed = self.backbone.embed(x.base_input, x.target_item_idx)
            if target_embed is not None:
                return jnp.concatenate([user_embed, target_embed], axis=-1)
            return user_embed

        raise ValueError(f"Unknown backbone: {name}")

    def _backbone_predict(
        self, x_augmented: jax.Array, x: KARInput
    ) -> jax.Array:
        """Predict from augmented embedding. Dispatch by backbone type."""
        name = self.backbone_name

        if name in ("deepfm",):
            # Reshape back to (B, n_fields, d_embed) + reconstruct first_order
            _, first_order = self.backbone.embed(x.base_input)
            n_fields = self.backbone._n_fields + self.backbone._n_numerical
            d_embed = self.backbone._d_embed
            stacked = x_augmented.reshape(x_augmented.shape[0], n_fields, d_embed)
            return self.backbone.predict_from_embedding(stacked, first_order)

        if name in ("dcnv2",):
            n_fields = self.backbone._n_fields + self.backbone._n_numerical
            d_embed = self.backbone._d_embed
            stacked = x_augmented.reshape(x_augmented.shape[0], n_fields, d_embed)
            return self.backbone.predict_from_embedding(stacked)

        if name == "lightgcn":
            d = self.backbone._d_embed
            u_embed = x_augmented[:, :d]
            i_embed = x_augmented[:, d:]
            return self.backbone.predict_from_embedding(u_embed, i_embed)

        if name == "din":
            d = self.backbone._d_embed
            user_interest = x_augmented[:, :d]
            target_query = x_augmented[:, d : 2 * d]
            static_flat = x_augmented[:, 2 * d :]
            return self.backbone.predict_from_embedding(user_interest, target_query, static_flat)

        if name == "sasrec":
            d = self.backbone._d_embed
            user_embed = x_augmented[:, :d]
            target_embed = x_augmented[:, d:]
            return self.backbone.predict_from_embedding(user_embed, target_embed)

        raise ValueError(f"Unknown backbone: {name}")


def compute_d_backbone(backbone_name: str, backbone: nnx.Module) -> int:
    """Compute flattened backbone embedding dimension.

    Args:
        backbone_name: Backbone name.
        backbone: Initialized backbone model.

    Returns:
        Integer dimension of flattened embedding.
    """
    if backbone_name in ("deepfm", "dcnv2"):
        n_fields = backbone._n_fields + backbone._n_numerical
        return n_fields * backbone._d_embed
    if backbone_name == "lightgcn":
        return backbone._d_embed * 2  # concat(user, item)
    if backbone_name == "din":
        d = backbone._d_embed
        static_dim = (backbone._n_fields + backbone._n_numerical) * d
        return d + d + static_dim  # interest + target + static
    if backbone_name == "sasrec":
        return backbone._d_embed * 2  # concat(user, target)
    raise ValueError(f"Unknown backbone: {backbone_name}")
