"""Unit tests for feature store save/load round-trip consistency."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.features.store import (
    load_cat_vocab,
    load_feature_meta,
    load_id_maps,
    load_item_features,
    load_train_pairs,
    load_user_features,
    save_features,
)


@pytest.fixture
def sample_data():
    """Create sample data for save/load testing."""
    rng = np.random.default_rng(42)
    n_users = 10
    n_items = 20
    n_pairs = 50

    train_pairs = {
        "user_idx": rng.integers(0, n_users, size=n_pairs).astype(np.int32),
        "item_idx": rng.integers(0, n_items, size=n_pairs).astype(np.int32),
        "labels": rng.choice([0.0, 1.0], size=n_pairs).astype(np.float32),
    }

    user_features_npz = {
        "numerical": rng.random((n_users, 8)).astype(np.float32),
        "categorical": rng.integers(0, 5, size=(n_users, 3)).astype(np.int32),
    }

    item_features_npz = {
        "numerical": rng.random((n_items, 2)).astype(np.float32),
        "categorical": rng.integers(0, 10, size=(n_items, 5)).astype(np.int32),
    }

    feature_meta = {
        "n_users": n_users,
        "n_items": n_items,
        "n_train_pairs": n_pairs,
        "user_num_names": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
        "user_cat_names": ["c1", "c2", "c3"],
        "item_num_names": ["n1", "n2"],
        "item_cat_names": ["ic1", "ic2", "ic3", "ic4", "ic5"],
        "user_cat_vocab_sizes": {"c1": 5, "c2": 3, "c3": 4},
        "item_cat_vocab_sizes": {"ic1": 10, "ic2": 8, "ic3": 6, "ic4": 4, "ic5": 3},
        "n_user_numerical": 8,
        "n_item_numerical": 2,
    }

    id_maps = {
        "user_to_idx": {f"U{i:03d}": i for i in range(n_users)},
        "idx_to_user": {str(i): f"U{i:03d}" for i in range(n_users)},
        "item_to_idx": {f"A{i:03d}": i for i in range(n_items)},
        "idx_to_item": {str(i): f"A{i:03d}" for i in range(n_items)},
    }

    cat_vocab = {
        "user": {"c1": {"UNKNOWN": 0, "A": 1, "B": 2}},
        "item": {"ic1": {"UNKNOWN": 0, "X": 1, "Y": 2}},
    }

    return train_pairs, user_features_npz, item_features_npz, feature_meta, id_maps, cat_vocab


class TestSaveLoadRoundTrip:
    def test_train_pairs_roundtrip(self, sample_data):
        train_pairs, user_feat, item_feat, meta, id_maps, vocab = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            save_features(tmpdir, train_pairs, user_feat, item_feat, meta, id_maps, vocab)

            loaded = load_train_pairs(tmpdir)
            np.testing.assert_array_equal(loaded["user_idx"], train_pairs["user_idx"])
            np.testing.assert_array_equal(loaded["item_idx"], train_pairs["item_idx"])
            np.testing.assert_array_equal(loaded["labels"], train_pairs["labels"])

    def test_user_features_roundtrip(self, sample_data):
        train_pairs, user_feat, item_feat, meta, id_maps, vocab = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            save_features(tmpdir, train_pairs, user_feat, item_feat, meta, id_maps, vocab)

            loaded = load_user_features(tmpdir)
            np.testing.assert_array_almost_equal(loaded["numerical"], user_feat["numerical"])
            np.testing.assert_array_equal(loaded["categorical"], user_feat["categorical"])

    def test_item_features_roundtrip(self, sample_data):
        train_pairs, user_feat, item_feat, meta, id_maps, vocab = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            save_features(tmpdir, train_pairs, user_feat, item_feat, meta, id_maps, vocab)

            loaded = load_item_features(tmpdir)
            np.testing.assert_array_almost_equal(loaded["numerical"], item_feat["numerical"])
            np.testing.assert_array_equal(loaded["categorical"], item_feat["categorical"])

    def test_feature_meta_roundtrip(self, sample_data):
        train_pairs, user_feat, item_feat, meta, id_maps, vocab = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            save_features(tmpdir, train_pairs, user_feat, item_feat, meta, id_maps, vocab)

            loaded = load_feature_meta(tmpdir)
            assert loaded["n_users"] == meta["n_users"]
            assert loaded["n_items"] == meta["n_items"]
            assert loaded["user_cat_vocab_sizes"] == meta["user_cat_vocab_sizes"]

    def test_id_maps_roundtrip(self, sample_data):
        train_pairs, user_feat, item_feat, meta, id_maps, vocab = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            save_features(tmpdir, train_pairs, user_feat, item_feat, meta, id_maps, vocab)

            u2i, i2u, it2i, i2it = load_id_maps(tmpdir)

            assert u2i == id_maps["user_to_idx"]
            assert it2i == id_maps["item_to_idx"]
            # idx_to_user/item have int keys after loading
            for k, v in i2u.items():
                assert id_maps["idx_to_user"][str(k)] == v

    def test_cat_vocab_roundtrip(self, sample_data):
        train_pairs, user_feat, item_feat, meta, id_maps, vocab = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            save_features(tmpdir, train_pairs, user_feat, item_feat, meta, id_maps, vocab)

            loaded = load_cat_vocab(tmpdir)
            assert loaded == vocab
