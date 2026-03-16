"""BGE embedding index aligned with feature store indices.

Maps pre-computed BGE embeddings (keyed by article_id/customer_id) to
the integer indices used by the feature store, so that batch[item_idx]
directly retrieves the correct 768-dim embedding vector.

48 items present in the feature store but missing from BGE embeddings
(0.05% of 105K) are zero-padded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.embeddings import load_embeddings
from src.features.store import load_id_maps


def build_aligned_embeddings(
    features_dir: Path,
    embeddings_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature-index-aligned embedding matrices.

    Args:
        features_dir: Path to feature .npz files (id_maps.npz).
        embeddings_dir: Path containing item_bge_embeddings.npz and
            user_bge_embeddings.npz.

    Returns:
        (item_embeddings, user_embeddings):
        - item_embeddings: (n_items, 768) float32, indexed by item feature idx.
        - user_embeddings: (n_users, 768) float32, indexed by user feature idx.
    """
    user_to_idx, _, item_to_idx, _ = load_id_maps(features_dir)

    item_emb = _align_item_embeddings(
        embeddings_dir / "item_bge_embeddings.npz", item_to_idx
    )
    user_emb = _align_user_embeddings(
        embeddings_dir / "user_bge_embeddings.npz", user_to_idx
    )
    return item_emb, user_emb


def _align_item_embeddings(
    emb_path: Path, item_to_idx: dict[str, int]
) -> np.ndarray:
    """Align item BGE embeddings to feature store indices.

    Missing items (not in BGE) get zero vectors.
    """
    embeddings, article_ids = load_embeddings(emb_path)
    d_enc = embeddings.shape[1]
    n_items = len(item_to_idx)

    aligned = np.zeros((n_items, d_enc), dtype=np.float32)

    # Build article_id → embedding row mapping
    id_to_emb_row = {str(aid): i for i, aid in enumerate(article_ids)}

    for aid_str, feat_idx in item_to_idx.items():
        emb_row = id_to_emb_row.get(str(aid_str))
        if emb_row is not None:
            aligned[feat_idx] = embeddings[emb_row]

    return aligned


def _align_user_embeddings(
    emb_path: Path, user_to_idx: dict[str, int]
) -> np.ndarray:
    """Align user BGE embeddings to feature store indices.

    Missing users (not in BGE) get zero vectors.
    """
    embeddings, customer_ids = load_embeddings(emb_path)
    d_enc = embeddings.shape[1]
    n_users = len(user_to_idx)

    aligned = np.zeros((n_users, d_enc), dtype=np.float32)

    id_to_emb_row = {str(cid): i for i, cid in enumerate(customer_ids)}

    for cid_str, feat_idx in user_to_idx.items():
        emb_row = id_to_emb_row.get(str(cid_str))
        if emb_row is not None:
            aligned[feat_idx] = embeddings[emb_row]

    return aligned
