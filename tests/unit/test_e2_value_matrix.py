"""Unit tests for the E2 value-matrix building blocks (enrichment_matrix, lead_lag, probe helpers).

The full probe run is an integration concern (minutes, real data); here we test the pure
functions: gap computation, sell-through SQL, lead-lag correlation, and the probe's
segment-divergence helper. No network, no real corpus.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.features.enrichment_matrix import (
    compute_sell_through,
    compute_trend_gap,
    compute_value_gap,
)
from src.features.lead_lag import lead_lag_corr, lead_lag_vs_baseline, monthly_attribute_share


# --------------------------------------------------------------------- gaps
def test_value_and_trend_gap():
    merged = pd.DataFrame(
        {
            "e2_price_look": [5, 1, 3],
            "e2_price_tier_actual": ["T1", "T5", "T3"],  # ranks 1,5,3
            "e2_trend_look": ["Emerging", "Dated", "Classic"],  # ranks 5,1,2
            "e2_trend_phase_actual": ["Declining", "Emerging", "Insufficient"],  # ranks 1,5,None
        }
    )
    vg = compute_value_gap(merged)
    assert list(vg) == [4, -4, 0]  # 5-1, 1-5, 3-3
    tg = compute_trend_gap(merged)
    assert tg[0] == 4 and tg[1] == -4  # 5-1, 1-5
    assert pd.isna(tg[2])  # Insufficient phase → NA


# ------------------------------------------------------------ sell-through
def test_sell_through(tmp_path):
    # item A: 4 purchases over 3 days (day 0..3), 2 buyers; item B: 1 purchase
    tx = pd.DataFrame(
        {
            "article_id": ["A", "A", "A", "A", "B"],
            "customer_id": ["u1", "u1", "u2", "u2", "u3"],
            "t_dat": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-01"]
            ),
        }
    )
    p = tmp_path / "train.parquet"
    tx.to_parquet(p, index=False)
    con = duckdb.connect()
    df = compute_sell_through(con, p).set_index("article_id")
    con.close()
    assert df.loc["A", "total_purchases"] == 4
    assert df.loc["A", "n_buyers"] == 2
    assert df.loc["A", "lifespan_days"] == 3  # day 1 → day 4
    assert df.loc["A", "velocity"] == pytest.approx(4 / 3)
    assert df.loc["A", "buyer_concentration"] == pytest.approx(2.0)  # 4 purchases / 2 buyers
    assert df.loc["B", "velocity"] == pytest.approx(1.0)  # lifespan 0 → max(.,1)=1


# ---------------------------------------------------------------- lead-lag
def test_lead_lag_corr_perfect_lead():
    # one category; share(t) perfectly predicts sales(t+1)
    mo = pd.date_range("2020-01-01", periods=8, freq="MS")
    share = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    sales = np.r_[[0], (share[:-1] * 100).astype(int)]  # sales(t) = 100*share(t-1)
    df = pd.DataFrame(
        {"cat": "X", "mo": mo, "share": share, "cat_sales": sales, "val_sales": sales}
    )
    corr = lead_lag_corr(df, lags=(1, 2))
    assert corr[1] > 0.9  # share(t) leads sales(t+1) strongly


def test_lead_lag_vs_baseline_keys():
    mo = pd.date_range("2020-01-01", periods=8, freq="MS")

    def mk(seed):
        rng = np.random.default_rng(seed)
        rows = []
        for cat in ["A", "B", "C", "D"]:
            s = rng.random(8)
            rows += [
                {"cat": cat, "mo": m, "share": s[i], "cat_sales": 100 + i, "val_sales": 1}
                for i, m in enumerate(mo)
            ]
        return pd.DataFrame(rows)

    res = lead_lag_vs_baseline(mk(1), mk(2), lags=(1, 2), n_boot=50, seed=42)
    assert set(res) >= {"best_lag", "r_attr", "r_meta", "delta", "ci_lo", "ci_hi", "n_categories"}


def test_monthly_attribute_share(tmp_path):
    tx = pd.DataFrame(
        {
            "article_id": ["A", "B", "A", "C"],
            "t_dat": pd.to_datetime(["2020-01-05", "2020-01-06", "2020-02-01", "2020-02-02"]),
        }
    )
    art = pd.DataFrame(
        {"article_id": ["A", "B", "C"], "index_name": ["Ladies", "Ladies", "Ladies"]}
    )
    axis = pd.DataFrame(
        {"article_id": ["A", "B", "C"], "phase": ["Emerging", "Declining", "Emerging"]}
    )
    tp, ap = tmp_path / "tx.parquet", tmp_path / "art.parquet"
    tx.to_parquet(tp, index=False)
    art.to_parquet(ap, index=False)
    con = duckdb.connect()
    out = monthly_attribute_share(con, tp, ap, axis, "phase", ["Emerging"], "index_name")
    con.close()
    jan = out[out["mo"] == pd.Timestamp("2020-01-01")].iloc[0]
    assert jan["cat_sales"] == 2 and jan["val_sales"] == 1  # A(Emerging)+B(Declining)
    assert jan["share"] == pytest.approx(0.5)


# ------------------------------------------------------------ probe helper
def test_segment_divergence():
    from witnesses.probe_E2_value_matrix import _segment_divergence

    labels = np.array(["a", "a", "b", "b"])
    assert _segment_divergence(labels, np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert _segment_divergence(labels, np.array([0.0, 0.0, 1.0, 1.0])) > 0.4  # max separation


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
