"""Unit tests for feature engineering module.

Tests DuckDB aggregation accuracy, null handling, negative sampling,
and feature pipeline consistency.
"""

import json
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.config import FeatureConfig
from src.features.engineering import (
    build_id_maps,
    compute_item_features,
    compute_user_features,
    generate_train_pairs,
    run_feature_engineering,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir():
    """Create minimal test Parquet files mimicking preprocessed data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Articles
        articles = pd.DataFrame(
            {
                "article_id": ["A001", "A002", "A003", "A004", "A005"],
                "product_type_name": ["Trousers", "T-shirt", "Dress", "Sweater", "Trousers"],
                "colour_group_name": ["Black", "White", "Red", "Blue", "Black"],
                "garment_group_name": ["Garment Lower", "Garment Upper", "Garment Full", "Garment Upper", "Garment Lower"],
                "section_name": ["Ladieswear", "Menswear", "Ladieswear", "Menswear", "Ladieswear"],
                "index_name": ["Ladieswear", "Menswear", "Ladieswear", "Menswear", "Ladieswear"],
                "detail_desc": ["trousers", "t-shirt", "dress", "sweater", "trousers"],
            }
        )
        articles.to_parquet(tmpdir / "articles.parquet", index=False)

        # Customers
        customers = pd.DataFrame(
            {
                "customer_id": ["U001", "U002", "U003"],
                "age": [25.0, None, 45.0],
                "club_member_status": ["ACTIVE", None, "PRE-CREATE"],
                "fashion_news_frequency": ["Regularly", "NONE", None],
            }
        )
        customers.to_parquet(tmpdir / "customers.parquet", index=False)

        # Train transactions
        transactions = pd.DataFrame(
            {
                "t_dat": pd.to_datetime(
                    ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01",
                     "2020-01-15", "2020-03-15", "2020-06-01"]
                ),
                "customer_id": ["U001", "U001", "U001", "U002", "U002", "U003", "U003", "U001"],
                "article_id": ["A001", "A002", "A003", "A001", "A004", "A002", "A005", "A005"],
                "price": [0.05, 0.03, 0.08, 0.05, 0.04, 0.03, 0.05, 0.05],
                "sales_channel_id": [1, 2, 1, 2, 2, 1, 1, 2],
            }
        )
        transactions.to_parquet(tmpdir / "train_transactions.parquet", index=False)

        # Ground truth (for training pair validation)
        ground_truth = {
            "U001": ["A001", "A002"],
            "U002": ["A003"],
            "U003": ["A004"],
        }
        (tmpdir / "val_ground_truth.json").write_text(json.dumps(ground_truth))

        yield tmpdir


@pytest.fixture
def config():
    return FeatureConfig(neg_sample_ratio=2, random_seed=42)


# ---------------------------------------------------------------------------
# User Features
# ---------------------------------------------------------------------------


class TestComputeUserFeatures:
    def test_returns_correct_user_count(self, tmp_data_dir, config):
        con = duckdb.connect()
        result = compute_user_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            tmp_data_dir / "customers.parquet",
            config,
        )
        con.close()
        # 3 unique users in transactions
        assert len(result.user_ids) == 3

    def test_numerical_shape(self, tmp_data_dir, config):
        con = duckdb.connect()
        result = compute_user_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            tmp_data_dir / "customers.parquet",
            config,
        )
        con.close()
        assert result.numerical.shape == (3, 8)
        assert result.numerical.dtype == np.float32

    def test_categorical_shape(self, tmp_data_dir, config):
        con = duckdb.connect()
        result = compute_user_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            tmp_data_dir / "customers.parquet",
            config,
        )
        con.close()
        assert result.categorical.shape == (3, 3)
        assert result.categorical.dtype == np.int32

    def test_no_nan_or_inf(self, tmp_data_dir, config):
        con = duckdb.connect()
        result = compute_user_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            tmp_data_dir / "customers.parquet",
            config,
        )
        con.close()
        assert not np.any(np.isnan(result.numerical))
        assert not np.any(np.isinf(result.numerical))

    def test_null_age_handled(self, tmp_data_dir, config):
        """U002 has null age → should map to UNKNOWN (idx 0)."""
        con = duckdb.connect()
        result = compute_user_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            tmp_data_dir / "customers.parquet",
            config,
        )
        con.close()
        u002_idx = result.user_ids.index("U002")
        assert result.categorical[u002_idx, 0] == 0  # UNKNOWN


# ---------------------------------------------------------------------------
# Item Features
# ---------------------------------------------------------------------------


class TestComputeItemFeatures:
    def test_returns_all_articles(self, tmp_data_dir, config):
        con = duckdb.connect()
        result = compute_item_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            config,
        )
        con.close()
        # All 5 articles (full catalog)
        assert len(result.item_ids) == 5

    def test_numerical_shape(self, tmp_data_dir, config):
        con = duckdb.connect()
        result = compute_item_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            config,
        )
        con.close()
        assert result.numerical.shape == (5, 2)
        assert result.numerical.dtype == np.float32

    def test_categorical_shape(self, tmp_data_dir, config):
        con = duckdb.connect()
        result = compute_item_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            config,
        )
        con.close()
        assert result.categorical.shape == (5, 5)
        assert result.categorical.dtype == np.int32

    def test_vocab_has_unknown(self, tmp_data_dir, config):
        con = duckdb.connect()
        result = compute_item_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            config,
        )
        con.close()
        for vocab in result.cat_vocabs.values():
            assert "UNKNOWN" in vocab
            assert vocab["UNKNOWN"] == 0


# ---------------------------------------------------------------------------
# ID Maps
# ---------------------------------------------------------------------------


class TestBuildIdMaps:
    def test_bidirectional_consistency(self, tmp_data_dir, config):
        con = duckdb.connect()
        user_feats = compute_user_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            tmp_data_dir / "customers.parquet",
            config,
        )
        item_feats = compute_item_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            config,
        )
        con.close()

        u2i, i2u, it2i, i2it = build_id_maps(user_feats, item_feats)

        for uid, idx in u2i.items():
            assert i2u[idx] == uid
        for iid, idx in it2i.items():
            assert i2it[idx] == iid


# ---------------------------------------------------------------------------
# Negative Sampling
# ---------------------------------------------------------------------------


class TestGenerateTrainPairs:
    def test_output_shapes(self, tmp_data_dir, config):
        con = duckdb.connect()
        user_feats = compute_user_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            tmp_data_dir / "customers.parquet",
            config,
        )
        item_feats = compute_item_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            config,
        )
        u2i, _, it2i, _ = build_id_maps(user_feats, item_feats)

        pairs = generate_train_pairs(
            con,
            tmp_data_dir / "train_transactions.parquet",
            u2i,
            it2i,
            config,
        )
        con.close()

        assert pairs["user_idx"].shape == pairs["item_idx"].shape == pairs["labels"].shape

    def test_label_ratio(self, tmp_data_dir, config):
        """Check negative:positive ratio matches config."""
        con = duckdb.connect()
        user_feats = compute_user_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            tmp_data_dir / "customers.parquet",
            config,
        )
        item_feats = compute_item_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            config,
        )
        u2i, _, it2i, _ = build_id_maps(user_feats, item_feats)

        pairs = generate_train_pairs(
            con,
            tmp_data_dir / "train_transactions.parquet",
            u2i,
            it2i,
            config,
        )
        con.close()

        n_pos = int(np.sum(pairs["labels"] == 1))
        n_neg = int(np.sum(pairs["labels"] == 0))
        assert n_neg == n_pos * config.neg_sample_ratio

    def test_no_positive_in_negatives(self, tmp_data_dir, config):
        """Negative items should not overlap with user's positive items."""
        con = duckdb.connect()
        user_feats = compute_user_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            tmp_data_dir / "customers.parquet",
            config,
        )
        item_feats = compute_item_features(
            con,
            tmp_data_dir / "train_transactions.parquet",
            tmp_data_dir / "articles.parquet",
            config,
        )
        u2i, _, it2i, _ = build_id_maps(user_feats, item_feats)

        # Load positive pairs
        pos_df = con.execute(
            f"""
            SELECT DISTINCT customer_id, article_id
            FROM read_parquet('{tmp_data_dir / "train_transactions.parquet"}')
            """
        ).fetchdf()
        user_pos_items: dict[int, set[int]] = {}
        for _, row in pos_df.iterrows():
            uid = row["customer_id"]
            iid = row["article_id"]
            u_idx = u2i.get(uid)
            i_idx = it2i.get(iid)
            if u_idx is not None and i_idx is not None:
                user_pos_items.setdefault(u_idx, set()).add(i_idx)

        pairs = generate_train_pairs(
            con,
            tmp_data_dir / "train_transactions.parquet",
            u2i,
            it2i,
            config,
        )
        con.close()

        # Check each negative pair
        neg_mask = pairs["labels"] == 0
        neg_users = pairs["user_idx"][neg_mask]
        neg_items = pairs["item_idx"][neg_mask]
        for u, i in zip(neg_users, neg_items):
            assert i not in user_pos_items.get(int(u), set()), (
                f"Negative item {i} is in user {u}'s positive set"
            )


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------


class TestRunFeatureEngineering:
    def test_output_files_created(self, tmp_data_dir, config):
        with tempfile.TemporaryDirectory() as out_dir:
            out_dir = Path(out_dir)
            run_feature_engineering(tmp_data_dir, out_dir, config)

            assert (out_dir / "train_pairs.npz").exists()
            assert (out_dir / "user_features.npz").exists()
            assert (out_dir / "item_features.npz").exists()
            assert (out_dir / "feature_meta.json").exists()
            assert (out_dir / "id_maps.json").exists()
            assert (out_dir / "cat_vocab.json").exists()

    def test_result_fields(self, tmp_data_dir, config):
        with tempfile.TemporaryDirectory() as out_dir:
            out_dir = Path(out_dir)
            result = run_feature_engineering(tmp_data_dir, out_dir, config)

            assert result.n_users == 3
            assert result.n_items == 5
            assert result.n_train_pairs > 0
            assert result.n_user_num_features == 8
            assert result.n_user_cat_features == 3
            assert result.n_item_num_features == 2
            assert result.n_item_cat_features == 5
