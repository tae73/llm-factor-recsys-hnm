"""Unit tests for NumpyBatchIterator-based data loader.

Tests NumpyBatchIterator, batch function builders, create_train_loader,
and steps_per_epoch with small fixture data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.training.data_loader import (
    NumpyBatchIterator,
    _make_feature_batch_fn,
    _make_index_batch_fn,
    create_train_loader,
    steps_per_epoch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_USERS = 20
N_ITEMS = 10
N_PAIRS = 100
N_USER_NUM = 8
N_USER_CAT = 3
N_ITEM_NUM = 2
N_ITEM_CAT = 5


@pytest.fixture
def features_dir(tmp_path: Path) -> Path:
    """Create minimal .npz + .json files mimicking data/features/."""
    rng = np.random.default_rng(42)

    # Train pairs
    user_idx = rng.integers(0, N_USERS, size=N_PAIRS).astype(np.int32)
    item_idx = rng.integers(0, N_ITEMS, size=N_PAIRS).astype(np.int32)
    labels = rng.choice([0.0, 1.0], size=N_PAIRS).astype(np.float32)
    np.savez_compressed(
        tmp_path / "train_pairs.npz",
        user_idx=user_idx,
        item_idx=item_idx,
        labels=labels,
    )

    # User features
    np.savez_compressed(
        tmp_path / "user_features.npz",
        numerical=rng.random((N_USERS, N_USER_NUM)).astype(np.float32),
        categorical=rng.integers(0, 5, size=(N_USERS, N_USER_CAT)).astype(np.int32),
    )

    # Item features
    np.savez_compressed(
        tmp_path / "item_features.npz",
        numerical=rng.random((N_ITEMS, N_ITEM_NUM)).astype(np.float32),
        categorical=rng.integers(0, 8, size=(N_ITEMS, N_ITEM_CAT)).astype(np.int32),
    )

    # Feature meta (needed by some paths)
    meta = {
        "n_users": N_USERS,
        "n_items": N_ITEMS,
        "n_train_pairs": N_PAIRS,
        "n_user_numerical": N_USER_NUM,
        "n_user_categorical": N_USER_CAT,
        "n_item_numerical": N_ITEM_NUM,
        "n_item_categorical": N_ITEM_CAT,
        "user_cat_names": ["age_group", "club_member_status", "fashion_news_frequency"],
        "item_cat_names": ["product_type", "colour_group", "garment_group", "section", "index"],
        "user_cat_vocab_sizes": {"age_group": 7, "club_member_status": 4, "fashion_news_frequency": 4},
        "item_cat_vocab_sizes": {
            "product_type": 10,
            "colour_group": 8,
            "garment_group": 6,
            "section": 5,
            "index": 4,
        },
    }
    (tmp_path / "feature_meta.json").write_text(json.dumps(meta))

    return tmp_path


@pytest.fixture
def user_features() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "numerical": rng.random((N_USERS, N_USER_NUM)).astype(np.float32),
        "categorical": rng.integers(0, 5, size=(N_USERS, N_USER_CAT)).astype(np.int32),
    }


@pytest.fixture
def item_features() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "numerical": rng.random((N_ITEMS, N_ITEM_NUM)).astype(np.float32),
        "categorical": rng.integers(0, 8, size=(N_ITEMS, N_ITEM_CAT)).astype(np.int32),
    }


# ---------------------------------------------------------------------------
# NumpyBatchIterator
# ---------------------------------------------------------------------------


class TestNumpyBatchIterator:
    def test_len_with_drop_remainder(self):
        n = 100
        batch_fn = lambda u, i, l: {"labels": l}
        it = NumpyBatchIterator(
            np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.float32),
            batch_fn, batch_size=30, shuffle=False,
        )
        assert len(it) == 3  # 100 // 30

    def test_len_without_drop_remainder(self):
        n = 100
        batch_fn = lambda u, i, l: {"labels": l}
        it = NumpyBatchIterator(
            np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.float32),
            batch_fn, batch_size=30, shuffle=False, drop_remainder=False,
        )
        assert len(it) == 4  # ceil(100 / 30)

    def test_all_batches_correct_size(self):
        n = 100
        batch_fn = lambda u, i, l: {"labels": l}
        it = NumpyBatchIterator(
            np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.float32),
            batch_fn, batch_size=30, shuffle=False,
        )
        batches = list(it)
        assert len(batches) == 3
        for b in batches:
            assert b["labels"].shape[0] == 30

    def test_deterministic_same_seed(self):
        rng = np.random.default_rng(0)
        u = rng.integers(0, 20, 100).astype(np.int32)
        i = rng.integers(0, 10, 100).astype(np.int32)
        l = rng.random(100).astype(np.float32)
        batch_fn = lambda u, i, l: {"u": u, "i": i, "labels": l}

        it1 = NumpyBatchIterator(u, i, l, batch_fn, 10, shuffle=True, seed=42)
        it2 = NumpyBatchIterator(u, i, l, batch_fn, 10, shuffle=True, seed=42)
        b1, b2 = next(iter(it1)), next(iter(it2))
        np.testing.assert_array_equal(b1["labels"], b2["labels"])

    def test_different_seed_different_order(self):
        rng = np.random.default_rng(0)
        u = rng.integers(0, 20, 100).astype(np.int32)
        i = rng.integers(0, 10, 100).astype(np.int32)
        l = rng.random(100).astype(np.float32)
        batch_fn = lambda u, i, l: {"labels": l}

        it1 = NumpyBatchIterator(u, i, l, batch_fn, 10, shuffle=True, seed=42)
        it2 = NumpyBatchIterator(u, i, l, batch_fn, 10, shuffle=True, seed=99)
        b1, b2 = next(iter(it1)), next(iter(it2))
        assert not np.array_equal(b1["labels"], b2["labels"])

    def test_no_shuffle_preserves_order(self):
        u = np.arange(20, dtype=np.int32)
        i = np.arange(20, dtype=np.int32)
        l = np.arange(20, dtype=np.float32)
        batch_fn = lambda u, i, l: {"labels": l}

        it = NumpyBatchIterator(u, i, l, batch_fn, 5, shuffle=False)
        first_batch = next(iter(it))
        np.testing.assert_array_equal(first_batch["labels"], np.arange(5, dtype=np.float32))


# ---------------------------------------------------------------------------
# Batch function builders
# ---------------------------------------------------------------------------


class TestBatchFnBuilders:
    def test_feature_batch_fn_keys(self, user_features, item_features):
        fn = _make_feature_batch_fn(
            user_features["categorical"], user_features["numerical"],
            item_features["categorical"], item_features["numerical"],
        )
        u = np.array([0, 1, 2], dtype=np.int32)
        i = np.array([3, 4, 5], dtype=np.int32)
        l = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        result = fn(u, i, l)
        assert set(result.keys()) == {"user_cat", "user_num", "item_cat", "item_num", "labels"}

    def test_feature_batch_fn_shapes(self, user_features, item_features):
        fn = _make_feature_batch_fn(
            user_features["categorical"], user_features["numerical"],
            item_features["categorical"], item_features["numerical"],
        )
        u = np.array([0, 1, 2], dtype=np.int32)
        i = np.array([3, 4, 5], dtype=np.int32)
        l = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        result = fn(u, i, l)
        assert result["user_cat"].shape == (3, N_USER_CAT)
        assert result["user_num"].shape == (3, N_USER_NUM)
        assert result["item_cat"].shape == (3, N_ITEM_CAT)
        assert result["item_num"].shape == (3, N_ITEM_NUM)
        assert result["labels"].shape == (3,)

    def test_feature_batch_fn_correct_lookup(self, user_features, item_features):
        fn = _make_feature_batch_fn(
            user_features["categorical"], user_features["numerical"],
            item_features["categorical"], item_features["numerical"],
        )
        u = np.array([3], dtype=np.int32)
        i = np.array([7], dtype=np.int32)
        l = np.array([1.0], dtype=np.float32)
        result = fn(u, i, l)
        np.testing.assert_array_equal(result["user_num"][0], user_features["numerical"][3])
        np.testing.assert_array_equal(result["item_cat"][0], item_features["categorical"][7])

    def test_feature_batch_fn_kar(self, user_features, item_features):
        rng = np.random.default_rng(0)
        item_emb = rng.random((N_ITEMS, 768)).astype(np.float32)
        user_emb = rng.random((N_USERS, 768)).astype(np.float32)
        fn = _make_feature_batch_fn(
            user_features["categorical"], user_features["numerical"],
            item_features["categorical"], item_features["numerical"],
            item_emb, user_emb,
        )
        u = np.array([0, 1], dtype=np.int32)
        i = np.array([2, 3], dtype=np.int32)
        l = np.array([1.0, 0.0], dtype=np.float32)
        result = fn(u, i, l)
        assert "h_fact" in result and "h_reason" in result
        assert result["h_fact"].shape == (2, 768)
        np.testing.assert_array_equal(result["h_fact"][0], item_emb[2])
        np.testing.assert_array_equal(result["h_reason"][1], user_emb[1])

    def test_index_batch_fn_keys(self):
        fn = _make_index_batch_fn()
        u = np.array([0, 1], dtype=np.int32)
        i = np.array([2, 3], dtype=np.int32)
        l = np.array([1.0, 0.0], dtype=np.float32)
        result = fn(u, i, l)
        assert set(result.keys()) == {"user_idx", "item_idx", "labels"}

    def test_index_batch_fn_kar(self):
        rng = np.random.default_rng(0)
        item_emb = rng.random((10, 768)).astype(np.float32)
        user_emb = rng.random((20, 768)).astype(np.float32)
        fn = _make_index_batch_fn(item_emb, user_emb)
        u = np.array([0, 1], dtype=np.int32)
        i = np.array([2, 3], dtype=np.int32)
        l = np.array([1.0, 0.0], dtype=np.float32)
        result = fn(u, i, l)
        assert "h_fact" in result
        assert result["h_fact"].shape == (2, 768)


# ---------------------------------------------------------------------------
# create_train_loader
# ---------------------------------------------------------------------------


class TestCreateTrainLoader:
    def test_basic_iteration(self, features_dir: Path):
        loader = create_train_loader(
            features_dir, batch_size=10, seed=42, shuffle=False
        )
        batches = list(loader)
        assert len(batches) > 0

    def test_batch_shapes(self, features_dir: Path):
        batch_size = 10
        loader = create_train_loader(
            features_dir, batch_size=batch_size, seed=42, shuffle=False
        )
        batch = next(iter(loader))
        assert batch["user_cat"].shape == (batch_size, N_USER_CAT)
        assert batch["user_num"].shape == (batch_size, N_USER_NUM)
        assert batch["item_cat"].shape == (batch_size, N_ITEM_CAT)
        assert batch["item_num"].shape == (batch_size, N_ITEM_NUM)
        assert batch["labels"].shape == (batch_size,)

    def test_deterministic_same_seed(self, features_dir: Path):
        loader1 = create_train_loader(
            features_dir, batch_size=10, seed=42, shuffle=True
        )
        loader2 = create_train_loader(
            features_dir, batch_size=10, seed=42, shuffle=True
        )
        b1 = next(iter(loader1))
        b2 = next(iter(loader2))
        np.testing.assert_array_equal(b1["user_cat"], b2["user_cat"])
        np.testing.assert_array_equal(b1["labels"], b2["labels"])

    def test_different_seed_different_order(self, features_dir: Path):
        loader1 = create_train_loader(
            features_dir, batch_size=10, seed=42, shuffle=True
        )
        loader2 = create_train_loader(
            features_dir, batch_size=10, seed=99, shuffle=True
        )
        b1 = next(iter(loader1))
        b2 = next(iter(loader2))
        assert not np.array_equal(b1["labels"], b2["labels"])

    def test_drop_remainder(self, features_dir: Path):
        batch_size = 10
        loader = create_train_loader(
            features_dir, batch_size=batch_size, seed=42, shuffle=False
        )
        for batch in loader:
            assert batch["labels"].shape[0] == batch_size

    def test_returns_numpy_batch_iterator(self, features_dir: Path):
        loader = create_train_loader(
            features_dir, batch_size=10, seed=42, shuffle=False
        )
        assert isinstance(loader, NumpyBatchIterator)

    def test_worker_count_ignored(self, features_dir: Path):
        """worker_count is kept for backward compat but ignored."""
        loader = create_train_loader(
            features_dir, batch_size=10, seed=42, worker_count=4, shuffle=False
        )
        assert isinstance(loader, NumpyBatchIterator)
        batch = next(iter(loader))
        assert batch["user_cat"].shape[0] == 10


# ---------------------------------------------------------------------------
# steps_per_epoch
# ---------------------------------------------------------------------------


class TestStepsPerEpoch:
    def test_correct_calculation(self, features_dir: Path):
        result = steps_per_epoch(features_dir, batch_size=10)
        assert result == N_PAIRS // 10  # 100 // 10 = 10

    def test_with_remainder(self, features_dir: Path):
        result = steps_per_epoch(features_dir, batch_size=30)
        assert result == N_PAIRS // 30  # 100 // 30 = 3
