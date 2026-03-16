"""Unit tests for src/baselines module."""
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pytest

from src.baselines.popularity import (
    compute_global_popularity,
    compute_recent_popularity,
    predict_popularity,
)
from src.baselines.utils import build_interaction_matrix


@pytest.fixture
def sample_train_parquet(tmp_path):
    """Create a small training transactions parquet."""
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT * FROM (VALUES
                ('2020-01-15'::DATE, 'u1', 'i1', 0.05, 2),
                ('2020-02-10'::DATE, 'u1', 'i2', 0.03, 1),
                ('2020-03-20'::DATE, 'u1', 'i1', 0.05, 2),
                ('2020-04-15'::DATE, 'u2', 'i1', 0.05, 1),
                ('2020-05-10'::DATE, 'u2', 'i3', 0.06, 2),
                ('2020-06-01'::DATE, 'u3', 'i2', 0.03, 1),
                ('2020-06-15'::DATE, 'u3', 'i3', 0.04, 2),
                ('2020-06-20'::DATE, 'u3', 'i1', 0.05, 1),
                ('2020-06-25'::DATE, 'u1', 'i3', 0.06, 2),
                ('2020-06-28'::DATE, 'u2', 'i2', 0.03, 1)
            ) AS t(t_dat, customer_id, article_id, price, sales_channel_id)
        ) TO '{tmp_path}/train_transactions.parquet' (FORMAT PARQUET)
    """)
    con.close()
    return tmp_path / "train_transactions.parquet"


def test_global_popularity_returns_k_items(sample_train_parquet):
    con = duckdb.connect()
    popular = compute_global_popularity(con, sample_train_parquet, k=2)
    con.close()
    assert len(popular) == 2
    # i1 has most purchases (4 times)
    assert popular[0] == "i1"


def test_recent_popularity(sample_train_parquet):
    con = duckdb.connect()
    popular = compute_recent_popularity(con, sample_train_parquet, k=3, window_days=10)
    con.close()
    assert len(popular) > 0
    assert all(isinstance(item, str) for item in popular)


def test_predict_popularity_all_users_same(sample_train_parquet):
    popular = ["i1", "i2"]
    users = ["u1", "u2", "u3"]
    preds = predict_popularity(popular, users)
    assert len(preds) == 3
    for uid in users:
        assert preds[uid] == ["i1", "i2"]


def test_interaction_matrix_shape(sample_train_parquet):
    con = duckdb.connect()
    idata = build_interaction_matrix(con, sample_train_parquet)
    con.close()

    # 3 users, 3 items
    assert idata.matrix.shape == (3, 3)
    assert len(idata.user_to_idx) == 3
    assert len(idata.item_to_idx) == 3
    assert idata.matrix.nnz > 0


def test_interaction_matrix_index_mapping(sample_train_parquet):
    con = duckdb.connect()
    idata = build_interaction_matrix(con, sample_train_parquet)
    con.close()

    # Check round-trip: user_to_idx -> idx_to_user
    for uid, idx in idata.user_to_idx.items():
        assert idata.idx_to_user[idx] == uid
    for iid, idx in idata.item_to_idx.items():
        assert idata.idx_to_item[idx] == iid
