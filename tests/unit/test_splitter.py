"""Unit tests for src/data/splitter module."""

import json

import duckdb
import pytest

from src.config import FilterConfig, SplitConfig
from src.data.splitter import (
    build_ground_truth,
    build_immediate_eval,
    compute_split_statistics,
    filter_customers_by_activity,
    split_transactions_temporal,
)


@pytest.fixture
def sample_transactions(tmp_path):
    """Create a small transactions parquet for testing."""
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT * FROM (VALUES
                ('2020-01-15'::DATE, 'c001', '0108775015', 0.05, 2),
                ('2020-02-10'::DATE, 'c001', '0108775044', 0.03, 1),
                ('2020-03-20'::DATE, 'c001', '0108775015', 0.05, 2),
                ('2020-04-15'::DATE, 'c001', '0108775099', 0.04, 1),
                ('2020-05-10'::DATE, 'c001', '0108775100', 0.06, 2),
                ('2020-06-01'::DATE, 'c002', '0108775015', 0.05, 1),
                ('2020-06-15'::DATE, 'c002', '0108775044', 0.03, 2),
                ('2020-07-05'::DATE, 'c001', '0108775015', 0.05, 2),
                ('2020-07-20'::DATE, 'c002', '0108775099', 0.04, 1),
                ('2020-07-25'::DATE, 'c003', '0108775200', 0.07, 2),
                ('2020-09-02'::DATE, 'c001', '0108775044', 0.03, 1),
                ('2020-09-03'::DATE, 'c003', '0108775015', 0.05, 2)
            ) AS t(t_dat, customer_id, article_id, price, sales_channel_id)
        ) TO '{tmp_path}/transactions.parquet' (FORMAT PARQUET)
    """)
    con.close()
    return tmp_path


def test_temporal_split_no_leakage(sample_transactions):
    con = duckdb.connect()
    config = SplitConfig()
    txn_path = sample_transactions / "transactions.parquet"

    train_path, val_path, test_path = split_transactions_temporal(
        con, txn_path, sample_transactions, config
    )

    # Train should have no dates after 2020-06-30
    max_train = con.execute(f"SELECT MAX(t_dat) FROM read_parquet('{train_path}')").fetchone()[0]
    assert str(max_train) <= "2020-06-30"

    # Val should have dates in [2020-07-01, 2020-08-31]
    val_dates = con.execute(
        f"SELECT MIN(t_dat), MAX(t_dat) FROM read_parquet('{val_path}')"
    ).fetchone()
    assert str(val_dates[0]) >= "2020-07-01"
    assert str(val_dates[1]) <= "2020-08-31"

    # Test should have dates in [2020-09-01, 2020-09-07]
    test_dates = con.execute(
        f"SELECT MIN(t_dat), MAX(t_dat) FROM read_parquet('{test_path}')"
    ).fetchone()
    assert str(test_dates[0]) >= "2020-09-01"
    assert str(test_dates[1]) <= "2020-09-07"

    con.close()


def test_filter_active_sparse(sample_transactions):
    con = duckdb.connect()
    config = SplitConfig()
    txn_path = sample_transactions / "transactions.parquet"

    train_path, _, _ = split_transactions_temporal(con, txn_path, sample_transactions, config)

    filter_cfg = FilterConfig(active_min=5, sparse_min=1)
    active_ids, sparse_ids = filter_customers_by_activity(con, train_path, filter_cfg)

    # c001 has 5 train transactions -> active
    assert "c001" in active_ids
    # c002 has 2 train transactions -> sparse
    assert "c002" in sparse_ids
    # No overlap
    assert not set(active_ids) & set(sparse_ids)
    con.close()


def test_build_ground_truth_dedup(sample_transactions):
    """Ground truth should deduplicate same article purchases per user."""
    con = duckdb.connect()
    config = SplitConfig()
    txn_path = sample_transactions / "transactions.parquet"

    _, val_path, _ = split_transactions_temporal(con, txn_path, sample_transactions, config)

    gt = build_ground_truth(con, val_path)

    # c001 bought 0108775015 in val — should appear only once
    if "c001" in gt:
        assert len(gt["c001"]) == len(set(gt["c001"]))

    con.close()


def test_build_immediate_eval_window_only(sample_transactions):
    """immediate GT covers (train_end, train_end+horizon] only — not the full val."""
    con = duckdb.connect()
    config = SplitConfig()
    txn_path = sample_transactions / "transactions.parquet"
    # Build the val split so val_transactions.parquet (the immediate source) exists.
    split_transactions_temporal(con, txn_path, sample_transactions, config)
    con.close()

    gt_path = build_immediate_eval(
        processed_dir=sample_transactions,
        output_dir=sample_transactions,
        train_end="2020-06-30",
        horizon_days=7,
    )
    assert gt_path.name == "immediate_ground_truth.json"
    gt = json.loads(gt_path.read_text())

    # Window (2020-06-30, 2020-07-07]: only c001's 2020-07-05 purchase qualifies.
    assert gt == {"c001": ["0108775015"]}
    # c002 (2020-07-20) and c003 (2020-07-25) are outside the 7-day horizon.
    assert "c002" not in gt
    assert "c003" not in gt


def test_build_immediate_eval_horizon_expands(sample_transactions):
    """A longer horizon captures purchases the 7-day window excludes."""
    con = duckdb.connect()
    config = SplitConfig()
    txn_path = sample_transactions / "transactions.parquet"
    split_transactions_temporal(con, txn_path, sample_transactions, config)
    con.close()

    gt_path = build_immediate_eval(
        processed_dir=sample_transactions,
        output_dir=sample_transactions,
        train_end="2020-06-30",
        horizon_days=31,  # (2020-06-30, 2020-07-31]
    )
    gt = json.loads(gt_path.read_text())
    # Now c002 (2020-07-20) and c003 (2020-07-25) fall inside the window.
    assert gt["c001"] == ["0108775015"]
    assert gt["c002"] == ["0108775099"]
    assert gt["c003"] == ["0108775200"]


def test_cold_start_detection(sample_transactions):
    con = duckdb.connect()
    config = SplitConfig()
    txn_path = sample_transactions / "transactions.parquet"

    train_path, val_path, test_path = split_transactions_temporal(
        con, txn_path, sample_transactions, config
    )

    stats = compute_split_statistics(con, train_path, val_path, test_path)

    # c003 appears in val but not in train -> cold-start user
    assert stats["n_cold_start_users_val"] >= 1

    # 0108775200 appears in val but not in train -> cold-start item
    assert stats["n_cold_start_items_val"] >= 1

    con.close()
