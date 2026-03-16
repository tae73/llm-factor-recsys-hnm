"""Grain-based data loader for recommendation model training.

Replaces numpy BatchIterator with grain.python DataLoader for:
- Multiprocess prefetch (worker_count > 0)
- Deterministic shuffling (same seed → same batch order)
- Multi-device sharding (ShardByJaxProcess, no-op on single device)

Data flow (feature-based: DeepFM, DCNv2):
    TrainPairsSource[idx] → {user_idx, item_idx, label}
        → FeatureLookupTransform → {user_cat, user_num, item_cat, item_num, labels}
            → grain.Batch → batched dict[str, np.ndarray]

Data flow (graph-based: LightGCN):
    TrainPairsSource[idx] → {user_idx, item_idx, label}
        → IndexOnlyTransform → {user_idx, item_idx, labels}
            → grain.Batch → batched dict[str, np.ndarray]

Data flow (DIN):
    TrainPairsSource[idx] → {user_idx, item_idx, label}
        → DINLookupTransform → {user_cat, user_num, item_cat, item_num, history, hist_len, labels}
            → grain.Batch → batched dict[str, np.ndarray]

Data flow (SASRec):
    TrainPairsSource[idx] → {user_idx, item_idx, label}
        → SASRecTransform → {history, hist_len, target_item_seq_idx, labels}
            → grain.Batch → batched dict[str, np.ndarray]
"""

from __future__ import annotations

from pathlib import Path

import grain.python as grain
import numpy as np

from src.features.store import load_item_features, load_train_pairs, load_user_features


# ---------------------------------------------------------------------------
# Data Source
# ---------------------------------------------------------------------------


class TrainPairsSource:
    """RandomAccessDataSource of (user_idx, item_idx, label) triples.

    Stores only indices and labels — feature arrays are referenced
    by FeatureLookupTransform. Minimizes pickling cost when grain
    workers fork.
    """

    def __init__(self, features_dir: Path) -> None:
        pairs = load_train_pairs(features_dir)
        self._user_idx: np.ndarray = pairs["user_idx"]  # (N,) int32
        self._item_idx: np.ndarray = pairs["item_idx"]  # (N,) int32
        self._labels: np.ndarray = pairs["labels"]  # (N,) float32
        self._len = len(self._labels)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, int | float]:
        return {
            "user_idx": int(self._user_idx[idx]),
            "item_idx": int(self._item_idx[idx]),
            "label": float(self._labels[idx]),
        }


# ---------------------------------------------------------------------------
# Feature Lookup Transform (DeepFM, DCNv2)
# ---------------------------------------------------------------------------


class FeatureLookupTransform(grain.MapTransform):
    """Index → feature numpy array lookup.

    Runs in worker processes. macOS/Linux fork() CoW shares
    the underlying feature arrays without duplication.
    """

    def __init__(
        self,
        user_features: dict[str, np.ndarray],
        item_features: dict[str, np.ndarray],
    ) -> None:
        self._user_num = user_features["numerical"]  # (n_users, 8)
        self._user_cat = user_features["categorical"]  # (n_users, 3)
        self._item_num = item_features["numerical"]  # (n_items, 2)
        self._item_cat = item_features["categorical"]  # (n_items, 5)

    def map(self, element: dict) -> dict[str, np.ndarray]:
        u = element["user_idx"]
        i = element["item_idx"]
        return {
            "user_cat": self._user_cat[u],
            "user_num": self._user_num[u],
            "item_cat": self._item_cat[i],
            "item_num": self._item_num[i],
            "labels": np.float32(element["label"]),
        }


# ---------------------------------------------------------------------------
# Index-Only Transform (LightGCN)
# ---------------------------------------------------------------------------


class IndexOnlyTransform(grain.MapTransform):
    """LightGCN: no feature lookup, just pass indices + label."""

    def map(self, element: dict) -> dict[str, np.ndarray]:
        return {
            "user_idx": np.int32(element["user_idx"]),
            "item_idx": np.int32(element["item_idx"]),
            "labels": np.float32(element["label"]),
        }


# ---------------------------------------------------------------------------
# Sequential Transforms (DIN, SASRec)
# ---------------------------------------------------------------------------


class DINLookupTransform(grain.MapTransform):
    """Feature lookup + sequence for DIN.

    Combines static user/item features with purchase history sequence.
    """

    def __init__(
        self,
        user_features: dict[str, np.ndarray],
        item_features: dict[str, np.ndarray],
        sequences: np.ndarray,
        seq_lengths: np.ndarray,
    ) -> None:
        self._user_num = user_features["numerical"]
        self._user_cat = user_features["categorical"]
        self._item_num = item_features["numerical"]
        self._item_cat = item_features["categorical"]
        self._sequences = sequences    # (n_users, max_seq_len) int32
        self._seq_lengths = seq_lengths  # (n_users,) int32

    def map(self, element: dict) -> dict[str, np.ndarray]:
        u = element["user_idx"]
        i = element["item_idx"]
        return {
            "user_cat": self._user_cat[u],
            "user_num": self._user_num[u],
            "item_cat": self._item_cat[i],
            "item_num": self._item_num[i],
            "history": self._sequences[u],
            "hist_len": self._seq_lengths[u],
            "labels": np.float32(element["label"]),
        }


class SASRecTransform(grain.MapTransform):
    """Sequence-only lookup for SASRec.

    Returns history, hist_len, target item sequence index, and label.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        seq_lengths: np.ndarray,
    ) -> None:
        self._sequences = sequences
        self._seq_lengths = seq_lengths

    def map(self, element: dict) -> dict[str, np.ndarray]:
        u = element["user_idx"]
        i = element["item_idx"]
        return {
            "history": self._sequences[u],
            "hist_len": self._seq_lengths[u],
            "target_item_seq_idx": np.int32(i + 1),  # +1 for PAD offset
            "labels": np.float32(element["label"]),
        }


# ---------------------------------------------------------------------------
# KAR Transforms (adds BGE embeddings to existing transforms)
# ---------------------------------------------------------------------------


class KARFeatureLookupTransform(grain.MapTransform):
    """Feature lookup + BGE embedding lookup for KAR with feature-based backbones.

    Extends FeatureLookupTransform with h_fact (item BGE) and h_reason (user BGE).
    """

    def __init__(
        self,
        user_features: dict[str, np.ndarray],
        item_features: dict[str, np.ndarray],
        item_embeddings: np.ndarray,
        user_embeddings: np.ndarray,
    ) -> None:
        self._user_num = user_features["numerical"]
        self._user_cat = user_features["categorical"]
        self._item_num = item_features["numerical"]
        self._item_cat = item_features["categorical"]
        self._item_emb = item_embeddings  # (n_items, 768) float32
        self._user_emb = user_embeddings  # (n_users, 768) float32

    def map(self, element: dict) -> dict[str, np.ndarray]:
        u = element["user_idx"]
        i = element["item_idx"]
        return {
            "user_cat": self._user_cat[u],
            "user_num": self._user_num[u],
            "item_cat": self._item_cat[i],
            "item_num": self._item_num[i],
            "h_fact": self._item_emb[i],
            "h_reason": self._user_emb[u],
            "labels": np.float32(element["label"]),
        }


class KARIndexTransform(grain.MapTransform):
    """Index + BGE embedding lookup for KAR with LightGCN."""

    def __init__(
        self,
        item_embeddings: np.ndarray,
        user_embeddings: np.ndarray,
    ) -> None:
        self._item_emb = item_embeddings
        self._user_emb = user_embeddings

    def map(self, element: dict) -> dict[str, np.ndarray]:
        u = element["user_idx"]
        i = element["item_idx"]
        return {
            "user_idx": np.int32(u),
            "item_idx": np.int32(i),
            "h_fact": self._item_emb[i],
            "h_reason": self._user_emb[u],
            "labels": np.float32(element["label"]),
        }


class KARDINLookupTransform(grain.MapTransform):
    """DIN features + sequences + BGE embeddings for KAR."""

    def __init__(
        self,
        user_features: dict[str, np.ndarray],
        item_features: dict[str, np.ndarray],
        sequences: np.ndarray,
        seq_lengths: np.ndarray,
        item_embeddings: np.ndarray,
        user_embeddings: np.ndarray,
    ) -> None:
        self._user_num = user_features["numerical"]
        self._user_cat = user_features["categorical"]
        self._item_num = item_features["numerical"]
        self._item_cat = item_features["categorical"]
        self._sequences = sequences
        self._seq_lengths = seq_lengths
        self._item_emb = item_embeddings
        self._user_emb = user_embeddings

    def map(self, element: dict) -> dict[str, np.ndarray]:
        u = element["user_idx"]
        i = element["item_idx"]
        return {
            "user_cat": self._user_cat[u],
            "user_num": self._user_num[u],
            "item_cat": self._item_cat[i],
            "item_num": self._item_num[i],
            "history": self._sequences[u],
            "hist_len": self._seq_lengths[u],
            "h_fact": self._item_emb[i],
            "h_reason": self._user_emb[u],
            "labels": np.float32(element["label"]),
        }


class KARSASRecTransform(grain.MapTransform):
    """SASRec sequences + BGE embeddings for KAR."""

    def __init__(
        self,
        sequences: np.ndarray,
        seq_lengths: np.ndarray,
        item_embeddings: np.ndarray,
        user_embeddings: np.ndarray,
    ) -> None:
        self._sequences = sequences
        self._seq_lengths = seq_lengths
        self._item_emb = item_embeddings
        self._user_emb = user_embeddings

    def map(self, element: dict) -> dict[str, np.ndarray]:
        u = element["user_idx"]
        i = element["item_idx"]
        return {
            "history": self._sequences[u],
            "hist_len": self._seq_lengths[u],
            "target_item_seq_idx": np.int32(i + 1),
            "h_fact": self._item_emb[i],
            "h_reason": self._user_emb[u],
            "labels": np.float32(element["label"]),
        }


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------


def create_train_loader(
    features_dir: Path,
    batch_size: int,
    seed: int,
    worker_count: int = 4,
    prefetch_buffer_size: int = 2,
    shuffle: bool = True,
    backbone_name: str = "deepfm",
    use_kar: bool = False,
    item_embeddings: np.ndarray | None = None,
    user_embeddings: np.ndarray | None = None,
) -> grain.DataLoader:
    """Create a Grain DataLoader for training.

    Each epoch should create a new loader with seed=base_seed+epoch
    for different shuffle order per epoch.

    Args:
        features_dir: Path to feature .npz files.
        batch_size: Samples per batch.
        seed: Random seed for deterministic shuffling.
        worker_count: Multiprocess workers (0 = same process).
        prefetch_buffer_size: Batches to prefetch per worker.
        shuffle: Whether to shuffle indices.
        backbone_name: Backbone name to determine transform type.
        use_kar: If True, include h_fact/h_reason BGE embeddings in batch.
        item_embeddings: (n_items, 768) aligned item BGE embeddings (required if use_kar).
        user_embeddings: (n_users, 768) aligned user BGE embeddings (required if use_kar).

    Returns:
        grain.DataLoader yielding batched dicts.
    """
    source = TrainPairsSource(features_dir)

    if use_kar:
        assert item_embeddings is not None and user_embeddings is not None, (
            "item_embeddings and user_embeddings required when use_kar=True"
        )

    # Determine transform based on backbone
    from src.models import get_backbone

    spec = get_backbone(backbone_name)

    if spec.needs_graph:
        if use_kar:
            transform: grain.MapTransform = KARIndexTransform(
                item_embeddings, user_embeddings
            )
        else:
            transform = IndexOnlyTransform()
    elif spec.needs_sequence:
        from src.features.sequences import load_sequences

        seq_data = load_sequences(features_dir)
        sequences = seq_data["sequences"]
        seq_lengths = seq_data["seq_lengths"]

        if backbone_name == "din":
            user_features = load_user_features(features_dir)
            item_features = load_item_features(features_dir)
            if use_kar:
                transform = KARDINLookupTransform(
                    user_features, item_features, sequences, seq_lengths,
                    item_embeddings, user_embeddings,
                )
            else:
                transform = DINLookupTransform(
                    user_features, item_features, sequences, seq_lengths
                )
        else:
            # sasrec
            if use_kar:
                transform = KARSASRecTransform(
                    sequences, seq_lengths, item_embeddings, user_embeddings
                )
            else:
                transform = SASRecTransform(sequences, seq_lengths)
    else:
        user_features = load_user_features(features_dir)
        item_features = load_item_features(features_dir)
        if use_kar:
            transform = KARFeatureLookupTransform(
                user_features, item_features, item_embeddings, user_embeddings
            )
        else:
            transform = FeatureLookupTransform(user_features, item_features)

    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=1,  # caller manages epoch loop
        shard_options=grain.ShardByJaxProcess(),  # multi-device auto-sharding
        shuffle=shuffle,
        seed=seed,
    )

    operations: list[grain.MapTransform | grain.Batch] = [
        transform,
        grain.Batch(batch_size=batch_size, drop_remainder=True),
    ]

    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count,
        worker_buffer_size=prefetch_buffer_size,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def steps_per_epoch(features_dir: Path, batch_size: int) -> int:
    """Compute number of steps per epoch (accounting for drop_remainder).

    grain.DataLoader does not support __len__, so we pre-compute.
    """
    pairs = load_train_pairs(features_dir)
    n_samples = len(pairs["labels"])
    return n_samples // batch_size
