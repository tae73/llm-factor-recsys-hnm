"""Unit tests for Grain-based data loader.

Tests TrainPairsSource, FeatureLookupTransform, create_train_loader,
and steps_per_epoch with small fixture data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.training.data_loader import (
    FeatureLookupTransform,
    TrainPairsSource,
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
# TrainPairsSource
# ---------------------------------------------------------------------------


class TestTrainPairsSource:
    def test_len_matches_labels(self, features_dir: Path):
        source = TrainPairsSource(features_dir)
        assert len(source) == N_PAIRS

    def test_getitem_returns_correct_keys(self, features_dir: Path):
        source = TrainPairsSource(features_dir)
        element = source[0]
        assert set(element.keys()) == {"user_idx", "item_idx", "label"}

    def test_getitem_types(self, features_dir: Path):
        source = TrainPairsSource(features_dir)
        element = source[0]
        assert isinstance(element["user_idx"], int)
        assert isinstance(element["item_idx"], int)
        assert isinstance(element["label"], float)


# ---------------------------------------------------------------------------
# FeatureLookupTransform
# ---------------------------------------------------------------------------


class TestFeatureLookupTransform:
    def test_map_output_keys(self, user_features, item_features):
        transform = FeatureLookupTransform(user_features, item_features)
        element = {"user_idx": 0, "item_idx": 0, "label": 1.0}
        result = transform.map(element)
        assert set(result.keys()) == {"user_cat", "user_num", "item_cat", "item_num", "labels"}

    def test_map_output_shapes(self, user_features, item_features):
        transform = FeatureLookupTransform(user_features, item_features)
        element = {"user_idx": 5, "item_idx": 3, "label": 0.0}
        result = transform.map(element)
        assert result["user_num"].shape == (N_USER_NUM,)
        assert result["user_cat"].shape == (N_USER_CAT,)
        assert result["item_num"].shape == (N_ITEM_NUM,)
        assert result["item_cat"].shape == (N_ITEM_CAT,)

    def test_map_correct_lookup(self, user_features, item_features):
        transform = FeatureLookupTransform(user_features, item_features)
        u_idx, i_idx = 3, 7
        element = {"user_idx": u_idx, "item_idx": i_idx, "label": 1.0}
        result = transform.map(element)
        np.testing.assert_array_equal(result["user_num"], user_features["numerical"][u_idx])
        np.testing.assert_array_equal(result["item_cat"], item_features["categorical"][i_idx])


# ---------------------------------------------------------------------------
# create_train_loader
# ---------------------------------------------------------------------------


class TestCreateTrainLoader:
    def test_basic_iteration(self, features_dir: Path):
        loader = create_train_loader(
            features_dir, batch_size=10, seed=42, worker_count=0, shuffle=False
        )
        batches = list(loader)
        assert len(batches) > 0

    def test_batch_shapes(self, features_dir: Path):
        batch_size = 10
        loader = create_train_loader(
            features_dir, batch_size=batch_size, seed=42, worker_count=0, shuffle=False
        )
        batch = next(iter(loader))
        assert batch["user_cat"].shape == (batch_size, N_USER_CAT)
        assert batch["user_num"].shape == (batch_size, N_USER_NUM)
        assert batch["item_cat"].shape == (batch_size, N_ITEM_CAT)
        assert batch["item_num"].shape == (batch_size, N_ITEM_NUM)
        assert batch["labels"].shape == (batch_size,)

    def test_deterministic_same_seed(self, features_dir: Path):
        loader1 = create_train_loader(
            features_dir, batch_size=10, seed=42, worker_count=0, shuffle=True
        )
        loader2 = create_train_loader(
            features_dir, batch_size=10, seed=42, worker_count=0, shuffle=True
        )
        b1 = next(iter(loader1))
        b2 = next(iter(loader2))
        np.testing.assert_array_equal(b1["user_cat"], b2["user_cat"])
        np.testing.assert_array_equal(b1["labels"], b2["labels"])

    def test_different_seed_different_order(self, features_dir: Path):
        loader1 = create_train_loader(
            features_dir, batch_size=10, seed=42, worker_count=0, shuffle=True
        )
        loader2 = create_train_loader(
            features_dir, batch_size=99, seed=99, worker_count=0, shuffle=True
        )
        b1 = next(iter(loader1))
        b2 = next(iter(loader2))
        # Very unlikely to be identical with different seeds
        assert not np.array_equal(b1["labels"], b2["labels"])

    def test_drop_remainder(self, features_dir: Path):
        batch_size = 10
        loader = create_train_loader(
            features_dir, batch_size=batch_size, seed=42, worker_count=0, shuffle=False
        )
        for batch in loader:
            assert batch["labels"].shape[0] == batch_size

    def test_worker_count_zero(self, features_dir: Path):
        """Single-process mode should work."""
        loader = create_train_loader(
            features_dir, batch_size=10, seed=42, worker_count=0, shuffle=False
        )
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
