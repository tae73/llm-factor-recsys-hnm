"""Unit tests for sequential feature pipeline.

Tests sequence building, padding, truncation, dtypes, and determinism.
"""

import json
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.config import SequenceConfig
from src.features.sequences import build_sequences, load_sequences


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temp data_dir and features_dir with minimal test data."""
    data_dir = tmp_path / "processed"
    features_dir = tmp_path / "features"
    data_dir.mkdir()
    features_dir.mkdir()

    # Create 5 users, 10 items
    n_users, n_items = 5, 10
    user_ids = [f"user_{i}" for i in range(n_users)]
    item_ids = [f"item_{i}" for i in range(n_items)]

    # ID maps (0-based indices)
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    idx_to_user = {str(i): uid for i, uid in enumerate(user_ids)}
    idx_to_item = {str(i): iid for i, iid in enumerate(item_ids)}
    id_maps = {
        "user_to_idx": user_to_idx,
        "idx_to_user": idx_to_user,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
    }
    (features_dir / "id_maps.json").write_text(json.dumps(id_maps))

    # Create train transactions parquet
    rng = np.random.default_rng(42)
    rows = []
    # user_0: 3 purchases
    for d in range(3):
        rows.append({"customer_id": "user_0", "article_id": f"item_{d}", "t_dat": f"2020-06-{10+d:02d}"})
    # user_1: 8 purchases (will be truncated if max_seq_len < 8)
    for d in range(8):
        rows.append({"customer_id": "user_1", "article_id": f"item_{d}", "t_dat": f"2020-06-{10+d:02d}"})
    # user_2: 0 purchases (no rows)
    # user_3: 1 purchase
    rows.append({"customer_id": "user_3", "article_id": "item_5", "t_dat": "2020-06-15"})
    # user_4: 2 purchases
    rows.append({"customer_id": "user_4", "article_id": "item_7", "t_dat": "2020-06-10"})
    rows.append({"customer_id": "user_4", "article_id": "item_9", "t_dat": "2020-06-20"})

    table = pa.table({
        "customer_id": [r["customer_id"] for r in rows],
        "article_id": [r["article_id"] for r in rows],
        "t_dat": [r["t_dat"] for r in rows],
    })
    pq.write_table(table, data_dir / "train_transactions.parquet")

    return data_dir, features_dir, n_users, n_items


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildSequences:
    def test_output_shape(self, tmp_dirs):
        """Sequences should have shape (n_users, max_seq_len)."""
        data_dir, features_dir, n_users, _ = tmp_dirs
        config = SequenceConfig(max_seq_len=5)
        build_sequences(data_dir, features_dir, config)

        data = load_sequences(features_dir)
        assert data["sequences"].shape == (n_users, 5)
        assert data["seq_lengths"].shape == (n_users,)

    def test_padding_with_zeros(self, tmp_dirs):
        """Short sequences should be left-padded with 0."""
        data_dir, features_dir, _, _ = tmp_dirs
        config = SequenceConfig(max_seq_len=10)
        build_sequences(data_dir, features_dir, config)

        data = load_sequences(features_dir)
        # user_0 has 3 items → 7 padding zeros at the start
        seq_0 = data["sequences"][0]
        assert np.all(seq_0[:7] == 0)
        assert np.all(seq_0[7:] > 0)

    def test_right_truncation(self, tmp_dirs):
        """Long sequences should be truncated to most recent items."""
        data_dir, features_dir, _, _ = tmp_dirs
        config = SequenceConfig(max_seq_len=4)
        build_sequences(data_dir, features_dir, config)

        data = load_sequences(features_dir)
        # user_1 has 8 items → truncated to last 4
        assert data["seq_lengths"][1] == 4
        assert np.all(data["sequences"][1] > 0)  # all 4 positions filled

    def test_seq_lengths_match(self, tmp_dirs):
        """seq_lengths should match actual non-zero counts."""
        data_dir, features_dir, _, _ = tmp_dirs
        config = SequenceConfig(max_seq_len=50)
        build_sequences(data_dir, features_dir, config)

        data = load_sequences(features_dir)
        assert data["seq_lengths"][0] == 3   # user_0: 3 purchases
        assert data["seq_lengths"][1] == 8   # user_1: 8 purchases
        assert data["seq_lengths"][2] == 0   # user_2: no purchases
        assert data["seq_lengths"][3] == 1   # user_3: 1 purchase
        assert data["seq_lengths"][4] == 2   # user_4: 2 purchases

    def test_dtype_int32(self, tmp_dirs):
        """Both sequences and seq_lengths should be int32."""
        data_dir, features_dir, _, _ = tmp_dirs
        config = SequenceConfig(max_seq_len=10)
        build_sequences(data_dir, features_dir, config)

        data = load_sequences(features_dir)
        assert data["sequences"].dtype == np.int32
        assert data["seq_lengths"].dtype == np.int32

    def test_deterministic(self, tmp_dirs):
        """Same config should produce identical output."""
        data_dir, features_dir, _, _ = tmp_dirs
        config = SequenceConfig(max_seq_len=10)

        build_sequences(data_dir, features_dir, config)
        data1 = load_sequences(features_dir)

        build_sequences(data_dir, features_dir, config)
        data2 = load_sequences(features_dir)

        np.testing.assert_array_equal(data1["sequences"], data2["sequences"])
        np.testing.assert_array_equal(data1["seq_lengths"], data2["seq_lengths"])
