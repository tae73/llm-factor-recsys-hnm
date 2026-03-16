"""Loss functions for recommendation models.

BCE loss for feature-based models, BPR loss for graph-based models.
Align and diversity losses added in Phase 4.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def binary_cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Numerically stable binary cross-entropy from logits.

    Formula: max(x, 0) - x*y + log(1 + exp(-|x|))

    Args:
        logits: Raw model output (B,) — no sigmoid applied.
        labels: Binary labels (B,) — 0.0 or 1.0.

    Returns:
        Scalar mean loss.
    """
    return jnp.mean(
        jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )


def bpr_loss(pos_scores: jax.Array, neg_scores: jax.Array) -> jax.Array:
    """Bayesian Personalized Ranking loss.

    BPR: -mean(log(sigmoid(pos - neg)))

    Args:
        pos_scores: Scores for positive (user, item) pairs (B,).
        neg_scores: Scores for negative (user, item) pairs (B,).

    Returns:
        Scalar mean BPR loss.
    """
    return -jnp.mean(jax.nn.log_sigmoid(pos_scores - neg_scores))


def embedding_l2_reg(
    user_embeds: jax.Array, item_embeds: jax.Array, weight: float
) -> jax.Array:
    """L2 regularization on initial embeddings.

    Penalizes large embedding norms: weight * (||e_u||^2 + ||e_i||^2) / (2B)

    Args:
        user_embeds: User embedding vectors (B, d).
        item_embeds: Item embedding vectors (B, d).
        weight: Regularization strength (lambda).

    Returns:
        Scalar regularization term.
    """
    batch_size = user_embeds.shape[0]
    l2 = jnp.sum(user_embeds**2) + jnp.sum(item_embeds**2)
    return weight * l2 / (2.0 * batch_size)


# ---------------------------------------------------------------------------
# KAR Losses (Phase 4)
# ---------------------------------------------------------------------------


def align_loss(e_expert: jax.Array, x_backbone_sg: jax.Array) -> jax.Array:
    """Alignment loss: MSE between expert output and stop-gradient backbone embed.

    Encourages expert to produce embeddings close to the backbone's representation
    space. Computed in d_rec space (caller must project backbone embed to d_rec).

    Args:
        e_expert: Expert output (B, d_rec).
        x_backbone_sg: Stop-gradient backbone embedding projected to d_rec (B, d_rec).

    Returns:
        Scalar mean squared error.
    """
    return jnp.mean((e_expert - jax.lax.stop_gradient(x_backbone_sg)) ** 2)


def diversity_loss(e_fact: jax.Array, e_reason: jax.Array) -> jax.Array:
    """Diversity loss: mean cosine similarity between expert outputs.

    Minimizing this encourages factual and reasoning experts to produce
    complementary (non-redundant) representations.

    Args:
        e_fact: Factual expert output (B, d_rec).
        e_reason: Reasoning expert output (B, d_rec).

    Returns:
        Scalar mean cosine similarity (to minimize).
    """
    e_fact_norm = e_fact / (jnp.linalg.norm(e_fact, axis=-1, keepdims=True) + 1e-8)
    e_reason_norm = e_reason / (jnp.linalg.norm(e_reason, axis=-1, keepdims=True) + 1e-8)
    cos_sim = jnp.sum(e_fact_norm * e_reason_norm, axis=-1)  # (B,)
    return jnp.mean(cos_sim)


def kar_total_loss(
    logits: jax.Array,
    labels: jax.Array,
    e_fact: jax.Array,
    x_item_sg: jax.Array,
    e_reason: jax.Array,
    x_user_sg: jax.Array,
    align_weight: float = 0.1,
    diversity_weight: float = 0.01,
    include_rec_loss: bool = True,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Combined KAR loss: BCE + align + diversity.

    Stage 2: include_rec_loss=False (expert adaptor, align+div only).
    Stage 3: include_rec_loss=True (end-to-end, BCE+align+div).

    Args:
        logits: Model logits (B,).
        labels: Binary labels (B,).
        e_fact: Factual expert output (B, d_rec).
        x_item_sg: Backbone item embed projected to d_rec (B, d_rec).
        e_reason: Reasoning expert output (B, d_rec).
        x_user_sg: Backbone user embed projected to d_rec (B, d_rec).
        align_weight: Weight for alignment losses.
        diversity_weight: Weight for diversity loss.
        include_rec_loss: Whether to include BCE loss.

    Returns:
        (total_loss, loss_dict) where loss_dict has 'bce', 'align', 'diversity' keys.
    """
    bce = binary_cross_entropy(logits, labels) if include_rec_loss else jnp.float32(0.0)

    l_align_fact = align_loss(e_fact, x_item_sg)
    l_align_reason = align_loss(e_reason, x_user_sg)
    l_align = l_align_fact + l_align_reason

    l_div = diversity_loss(e_fact, e_reason)

    total = bce + align_weight * l_align + diversity_weight * l_div

    loss_dict = {
        "bce": bce,
        "align": l_align,
        "diversity": l_div,
        "total": total,
    }
    return total, loss_dict
