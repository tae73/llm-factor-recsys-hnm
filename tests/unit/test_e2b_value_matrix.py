"""Unit tests for E2-3 strengthening building blocks (merch signals, weekly/continuous
lead-lag, buyer-population, probe helpers). Pure functions only; the full probe is an
integration concern. No network, no real corpus."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.features.audience_signals import (
    build_buyer_population,
    grand_std,
    segment_divergence_weighted,
)
from src.features.enrichment_matrix import compute_merch_signals
from src.features.lead_lag import monthly_attribute_share


# ------------------------------------------------------------- merch signals
def test_compute_merch_signals(tmp_path):
    # item A: prices 1.0→0.5 (markdown 0.5), 3 txns days 0/1/10 (2 in first week), 2 online/1 store
    tx = pd.DataFrame(
        {
            "article_id": ["A", "A", "A", "B"],
            "customer_id": ["u1", "u2", "u3", "u4"],
            "t_dat": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-15", "2020-01-01"]),
            "price": [1.0, 0.8, 0.5, 0.2],
            "sales_channel_id": [2, 2, 1, 2],
        }
    )
    p = tmp_path / "train.parquet"
    tx.to_parquet(p, index=False)
    con = duckdb.connect()
    df = compute_merch_signals(con, p).set_index("article_id")
    con.close()
    assert df.loc["A", "markdown_depth"] == pytest.approx(0.5)  # (1.0-0.5)/1.0
    assert df.loc["A", "first_week_sell_through"] == pytest.approx(2 / 3)  # days 0,1 in first 7d
    assert df.loc["A", "online_ratio"] == pytest.approx(2 / 3)
    assert df.loc["B", "markdown_depth"] == pytest.approx(0.0)  # single price


# ------------------------------------------------------ lead_lag weekly/continuous
def test_lead_lag_weekly_and_continuous(tmp_path):
    tx = pd.DataFrame(
        {
            "article_id": ["A", "B", "A", "B"],
            "t_dat": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-13", "2020-01-14"]),
        }
    )
    art = pd.DataFrame({"article_id": ["A", "B"], "index_name": ["L", "L"]})
    axis = pd.DataFrame({"article_id": ["A", "B"], "mom": [2.0, -1.0]})  # A positive momentum
    tp, ap = tmp_path / "tx.parquet", tmp_path / "art.parquet"
    tx.to_parquet(tp, index=False)
    art.to_parquet(ap, index=False)
    con = duckdb.connect()
    # weekly granularity → 2 distinct weeks
    out = monthly_attribute_share(con, tp, ap, axis, "mom", granularity="week", weight_col="mom")
    con.close()
    assert out["mo"].nunique() == 2  # weekly buckets
    # continuous: val_sales = sum(max(mom,0)); week1 has A(2.0)+B(0 since -1 clipped)=2.0 over 2 sales
    wk1 = out.sort_values("mo").iloc[0]
    assert wk1["val_sales"] == pytest.approx(2.0)
    assert wk1["share"] == pytest.approx(1.0)  # 2.0/2 sales


# --------------------------------------------------------- buyer population
def test_build_buyer_population(tmp_path):
    tx = pd.DataFrame(
        {
            "article_id": ["A", "A", "B"],
            "customer_id": ["u1", "u2", "u1"],
            "t_dat": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "price": [0.1, 0.1, 0.2],
            "sales_channel_id": [2, 1, 2],
        }
    )
    cust = pd.DataFrame({"customer_id": ["u1", "u2"], "age": [20, 40], "Active": [1, 0]})
    tp, cp = tmp_path / "tx.parquet", tmp_path / "cust.parquet"
    tx.to_parquet(tp, index=False)
    cust.to_parquet(cp, index=False)
    con = duckdb.connect()
    agg = build_buyer_population(con, tp, cp).set_index("article_id")
    con.close()
    assert agg.loc["A", "n_txn"] == 2 and agg.loc["A", "n_age"] == 2
    assert agg.loc["A", "sum_age"] == 60  # 20+40
    assert agg.loc["A", "n_online"] == 1 and agg.loc["A", "n_active"] == 1


def test_divergence_and_grand_std():
    # two segments with mean age 20 vs 40, equal n → divergence = 10
    labels = np.array(["a", "b"])
    sum_age = np.array([20.0, 40.0])
    n_age = np.array([1.0, 1.0])
    assert segment_divergence_weighted(labels, sum_age, n_age) == pytest.approx(10.0)
    # single segment → 0 divergence
    assert segment_divergence_weighted(np.array(["a", "a"]), sum_age, n_age) == pytest.approx(0.0)
    # grand std of {20,40} = 10
    assert grand_std(np.array([60.0]), np.array([2000.0]), np.array([2.0])) == pytest.approx(10.0)


def test_mode_det():
    from witnesses.probe_E2b_value_matrix import _mode_det

    assert _mode_det(["x", "x", "y"]) == "x"
    assert _mode_det(["b", "a"]) == "a"  # tie → lexical


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
