"""Vectorized numpy batch iterator for recommendation model training.

Replaces Grain DataLoader with NumpyBatchIterator:
- Vectorized numpy fancy indexing (arr[batch_indices]) instead of per-sample __getitem__
- No multiprocess workers — numpy fancy indexing is fast enough (~<1ms/batch)
- Deterministic shuffling (same seed → same batch order)
- Eliminates Grain fork() + JAX GPU context deadlock

Data flow (all backbones):
    NumpyBatchIterator.__iter__():
        perm = rng.permutation(N)
        for batch_indices in chunks(perm, batch_size):
            u, i = user_idx[batch_indices], item_idx[batch_indices]
            yield batch_fn(u, i, labels[batch_indices])
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from src.features.store import load_item_features, load_train_pairs, load_user_features


# ---------------------------------------------------------------------------
# NumpyBatchIterator
# ---------------------------------------------------------------------------


class NumpyBatchIterator:
    """Vectorized batch iterator using numpy fancy indexing.

    Each iteration yields a dict[str, np.ndarray] with the same structure
    as the former Grain pipeline, so trainer.py requires no changes.
    """

    def __init__(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        labels: np.ndarray,
        batch_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, np.ndarray]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_remainder: bool = True,
    ) -> None:
        self._user_idx = user_idx
        self._item_idx = item_idx
        self._labels = labels
        self._batch_fn = batch_fn
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._drop_remainder = drop_remainder
        self._n = len(labels)

    def __len__(self) -> int:
        if self._drop_remainder:
            return self._n // self._batch_size
        return (self._n + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        rng = np.random.RandomState(self._seed)
        indices = rng.permutation(self._n) if self._shuffle else np.arange(self._n)

        for start in range(0, self._n, self._batch_size):
            end = start + self._batch_size
            if end > self._n and self._drop_remainder:
                break
            idx = indices[start:end]
            u = self._user_idx[idx]
            i = self._item_idx[idx]
            lab = self._labels[idx]
            yield self._batch_fn(u, i, lab)


# ---------------------------------------------------------------------------
# Batch function builders (replace Grain transforms)
# ---------------------------------------------------------------------------


def _make_feature_batch_fn(
    user_cat: np.ndarray,
    user_num: np.ndarray,
    item_cat: np.ndarray,
    item_num: np.ndarray,
    item_emb: np.ndarray | None = None,
    user_emb: np.ndarray | None = None,
) -> Callable[..., dict[str, Any]]:
    """Feature-based backbones (DeepFM, DCNv2), optionally with KAR."""

    def batch_fn(
        u: np.ndarray, i: np.ndarray, labels: np.ndarray
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {
            "user_cat": user_cat[u],
            "user_num": user_num[u],
            "item_cat": item_cat[i],
            "item_num": item_num[i],
            "labels": labels,
        }
        if item_emb is not None:
            result["h_fact"] = item_emb[i]
            result["h_reason"] = user_emb[u]  # type: ignore[index]
        return result

    return batch_fn


def _make_index_batch_fn(
    item_emb: np.ndarray | None = None,
    user_emb: np.ndarray | None = None,
) -> Callable[..., dict[str, Any]]:
    """Graph-based backbone (LightGCN), optionally with KAR."""

    def batch_fn(
        u: np.ndarray, i: np.ndarray, labels: np.ndarray
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {
            "user_idx": u.astype(np.int32),
            "item_idx": i.astype(np.int32),
            "labels": labels,
        }
        if item_emb is not None:
            result["h_fact"] = item_emb[i]
            result["h_reason"] = user_emb[u]  # type: ignore[index]
        return result

    return batch_fn


def _make_din_batch_fn(
    user_cat: np.ndarray,
    user_num: np.ndarray,
    item_cat: np.ndarray,
    item_num: np.ndarray,
    sequences: np.ndarray,
    seq_lengths: np.ndarray,
    item_emb: np.ndarray | None = None,
    user_emb: np.ndarray | None = None,
) -> Callable[..., dict[str, Any]]:
    """DIN backbone, optionally with KAR."""

    def batch_fn(
        u: np.ndarray, i: np.ndarray, labels: np.ndarray
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {
            "user_cat": user_cat[u],
            "user_num": user_num[u],
            "item_cat": item_cat[i],
            "item_num": item_num[i],
            "history": sequences[u],
            "hist_len": seq_lengths[u],
            "labels": labels,
        }
        if item_emb is not None:
            result["h_fact"] = item_emb[i]
            result["h_reason"] = user_emb[u]  # type: ignore[index]
        return result

    return batch_fn


def _make_sasrec_batch_fn(
    sequences: np.ndarray,
    seq_lengths: np.ndarray,
    item_emb: np.ndarray | None = None,
    user_emb: np.ndarray | None = None,
) -> Callable[..., dict[str, Any]]:
    """SASRec backbone, optionally with KAR."""

    def batch_fn(
        u: np.ndarray, i: np.ndarray, labels: np.ndarray
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {
            "history": sequences[u],
            "hist_len": seq_lengths[u],
            "target_item_seq_idx": (i + 1).astype(np.int32),
            "labels": labels,
        }
        if item_emb is not None:
            result["h_fact"] = item_emb[i]
            result["h_reason"] = user_emb[u]  # type: ignore[index]
        return result

    return batch_fn


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------


def create_train_loader(
    features_dir: Path,
    batch_size: int,
    seed: int,
    worker_count: int = 0,
    prefetch_buffer_size: int = 2,
    shuffle: bool = True,
    backbone_name: str = "deepfm",
    use_kar: bool = False,
    item_embeddings: np.ndarray | None = None,
    user_embeddings: np.ndarray | None = None,
) -> NumpyBatchIterator:
    """Create a NumpyBatchIterator for training.

    Each epoch should create a new loader with seed=base_seed+epoch
    for different shuffle order per epoch.

    Args:
        features_dir: Path to feature .npz files.
        batch_size: Samples per batch.
        seed: Random seed for deterministic shuffling.
        worker_count: Ignored (kept for backward compatibility).
        prefetch_buffer_size: Ignored (kept for backward compatibility).
        shuffle: Whether to shuffle indices.
        backbone_name: Backbone name to determine batch structure.
        use_kar: If True, include h_fact/h_reason BGE embeddings in batch.
        item_embeddings: (n_items, 768) aligned item BGE embeddings (required if use_kar).
        user_embeddings: (n_users, 768) aligned user BGE embeddings (required if use_kar).

    Returns:
        NumpyBatchIterator yielding batched dicts.
    """
    pairs = load_train_pairs(features_dir)
    user_idx = pairs["user_idx"]
    item_idx = pairs["item_idx"]
    labels = pairs["labels"]

    if use_kar:
        assert item_embeddings is not None and user_embeddings is not None, (
            "item_embeddings and user_embeddings required when use_kar=True"
        )

    kar_kwargs: dict[str, np.ndarray | None] = (
        {"item_emb": item_embeddings, "user_emb": user_embeddings} if use_kar else {}
    )

    from src.models import get_backbone

    spec = get_backbone(backbone_name)

    if spec.needs_graph:
        batch_fn = _make_index_batch_fn(**kar_kwargs)
    elif spec.needs_sequence:
        from src.features.sequences import load_sequences

        seq_data = load_sequences(features_dir)
        sequences = seq_data["sequences"]
        seq_lengths = seq_data["seq_lengths"]

        if backbone_name == "din":
            user_features = load_user_features(features_dir)
            item_features = load_item_features(features_dir)
            batch_fn = _make_din_batch_fn(
                user_features["categorical"],
                user_features["numerical"],
                item_features["categorical"],
                item_features["numerical"],
                sequences,
                seq_lengths,
                **kar_kwargs,
            )
        else:
            batch_fn = _make_sasrec_batch_fn(sequences, seq_lengths, **kar_kwargs)
    else:
        user_features = load_user_features(features_dir)
        item_features = load_item_features(features_dir)
        batch_fn = _make_feature_batch_fn(
            user_features["categorical"],
            user_features["numerical"],
            item_features["categorical"],
            item_features["numerical"],
            **kar_kwargs,
        )

    return NumpyBatchIterator(
        user_idx=user_idx,
        item_idx=item_idx,
        labels=labels,
        batch_fn=batch_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_remainder=True,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def steps_per_epoch(features_dir: Path, batch_size: int) -> int:
    """Compute number of steps per epoch (accounting for drop_remainder)."""
    pairs = load_train_pairs(features_dir)
    n_samples = len(pairs["labels"])
    return n_samples // batch_size
