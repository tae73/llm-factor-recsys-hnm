"""Unit tests for the E2d gap-axis FUTURE decision-lift probe + its src builder.

Pure-fn logic (incremental readout, decision rule, gap permutations) is tested with synthetic
signals; ``build_article_future_outcomes`` is tested with tiny tmp_path parquet fixtures
asserting train-frozen references, the left-join NaN/0 contract, and determinism. No API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------
def _synth_sub(n: int, signal: bool, rng: np.random.Generator):
    """A value_gap sub-frame whose constituents are RANDOM; only the gap (optionally) encodes y.

    So a positive incremental delta means the gap adds OVER its constituents — exactly the
    quantity the probe measures.
    """
    y = rng.integers(0, 3, n)
    pl = rng.integers(1, 6, n)  # price_look 1..5, independent of y
    pt = rng.choice(["T1", "T2", "T3", "T4", "T5"], n)  # tier, independent of y
    pg = rng.choice(["A", "B", "C"], n)
    if signal:  # gap direction+magnitude tracks the class
        gap = (y - 1).astype(float) * 2.0 + rng.normal(0, 0.25, n)
    else:
        gap = rng.integers(-3, 4, n).astype(float)
    sub = pd.DataFrame(
        {
            "article_id": [str(i) for i in range(n)],
            "e2_value_gap": gap,
            "e2_price_look": pl,
            "e2_price_tier_actual": pt,
            "product_group_name": pg,
        }
    )
    return sub, y


def test_incremental_detects_gap_signal_over_constituents():
    from witnesses.probe_D5_value_matrix import _paired_sig
    from witnesses.probe_E2d_gap_decision import _build_matrices, _nested_cv

    rng = np.random.default_rng(0)
    sub, y = _synth_sub(900, signal=True, rng=rng)
    mats = _build_matrices(sub, "e2_value_gap")
    assert mats["full"].shape[1] == mats["base"].shape[1] + 2  # sign + |gap|
    cv = _nested_cv(mats["base"], mats["full"], y)
    sig = _paired_sig(cv["f1_full"], cv["f1_base"], min_delta=0.01)
    assert sig["mean_delta"] > 0.05 and sig["significant"]


def test_incremental_null_on_noise_gap():
    from witnesses.probe_D5_value_matrix import _paired_sig
    from witnesses.probe_E2d_gap_decision import _build_matrices, _nested_cv

    rng = np.random.default_rng(1)
    sub, y = _synth_sub(900, signal=False, rng=rng)
    mats = _build_matrices(sub, "e2_value_gap")
    cv = _nested_cv(mats["base"], mats["full"], y)
    sig = _paired_sig(cv["f1_full"], cv["f1_base"], min_delta=0.01)
    assert not sig["significant"]  # a random gap adds nothing over constituents


def test_saturated_upper_bounds_full_when_gap_consistent():
    """When gap = c1 − rank(c2) (the real-data condition), the c1×c2 interaction SPANS the gap,
    so f1_sat ≥ f1_full up to CV noise. (The property does NOT hold if the gap is decoupled
    from its constituents — that decoupling is impossible in real data.)"""
    from witnesses.probe_E2d_gap_decision import _build_matrices, _f1_only, _nested_cv

    rng = np.random.default_rng(2)
    n = 1200
    pl = rng.integers(1, 6, n)
    pt_rank = rng.integers(1, 6, n)
    pt = np.array(["T1", "T2", "T3", "T4", "T5"])[pt_rank - 1]
    gap = (pl - pt_rank).astype(float)  # consistent gap
    y = (np.sign(gap) + 1).astype(int)  # signal lives in the diagonal the gap encodes
    sub = pd.DataFrame(
        {
            "article_id": [str(i) for i in range(n)],
            "e2_value_gap": gap,
            "e2_price_look": pl,
            "e2_price_tier_actual": pt,
            "product_group_name": rng.choice(["A", "B"], n),
        }
    )
    mats = _build_matrices(sub, "e2_value_gap")
    cv = _nested_cv(mats["base"], mats["full"], y)
    assert _f1_only(mats["sat"], y) + 0.05 >= np.mean(cv["f1_full"])


def test_perm_within_preserves_invariants():
    from witnesses.probe_E2d_gap_decision import _perm_within

    rng = np.random.default_rng(3)
    gap = rng.integers(-3, 4, 200).astype(float)
    buckets = rng.choice(["x", "y"], 200)
    shuffled = _perm_within(gap, buckets, rng, mode="shuffle")
    signrand = _perm_within(gap, buckets, rng, mode="signrand")
    # shuffle preserves the within-bucket multiset
    for b in np.unique(buckets):
        m = buckets == b
        assert sorted(shuffled[m]) == sorted(gap[m])
    # sign-randomization preserves |gap| exactly
    assert np.allclose(np.abs(signrand), np.abs(gap))


def test_decision_rule_flags_signal():
    from witnesses.probe_E2d_gap_decision import _decision_rule

    rng = np.random.default_rng(4)
    n = 1500
    gap = rng.integers(-3, 4, n).astype(float)
    realized = (gap >= 2).astype(float)  # the extreme-gap flag is exactly the realized decision
    base_feats = rng.normal(size=(n, 4))  # baseline can't see gap
    r = _decision_rule(gap, realized, base_feats, flag_sign=1, n_boot=200, rng=rng)
    assert r["lift"] > 1.5 and r["prec_gap"] > r["prec_model"]


# ---------------------------------------------------------------------------
# build_article_future_outcomes — tmp_path parquet fixtures
# ---------------------------------------------------------------------------
def _write_fixture(tmp_path):
    train = pd.DataFrame(
        {
            "article_id": ["1", "1", "1", "2", "2"],
            "customer_id": ["a", "b", "c", "a", "b"],
            "t_dat": pd.to_datetime(
                ["2020-06-01", "2020-06-02", "2020-06-03", "2020-04-01", "2020-04-02"]
            ),
            "price": [10.0, 10.0, 10.0, 5.0, 5.0],
            "sales_channel_id": [1, 1, 2, 1, 1],
        }
    )
    val = pd.DataFrame(
        {
            "article_id": ["1", "1", "1", "1"],
            "customer_id": ["a", "b", "c", "d"],
            "t_dat": pd.to_datetime(["2020-07-01", "2020-07-02", "2020-07-20", "2020-07-21"]),
            "price": [8.0, 8.0, 6.0, 6.0],
            "sales_channel_id": [1, 1, 1, 2],
        }
    )
    train.to_parquet(tmp_path / "train_transactions.parquet")
    val.to_parquet(tmp_path / "val_transactions.parquet")


def test_future_outcomes_values_and_train_frozen(tmp_path):
    from src.features.enrichment_matrix import build_article_future_outcomes

    _write_fixture(tmp_path)
    out = build_article_future_outcomes(["1", "2", "999"], tmp_path).set_index("article_id")

    # art 1: present in val → drop uses TRAIN mean price (10), not contaminated by val
    assert out.loc["1", "has_val_sale"] == 1
    assert out.loc["1", "fut_val_n"] == 4
    assert out.loc["1", "fut_price_drop"] == pytest.approx(np.log(7.0) - np.log(10.0), rel=1e-6)
    assert out.loc["1", "fut_markdown_depth"] == pytest.approx((8.0 - 6.0) / 8.0, rel=1e-6)
    # first 7 days since first val sale (07-01): 07-01,07-02 in window; 07-20,07-21 out → 2/4
    assert out.loc["1", "fut_first_week_st"] == pytest.approx(0.5, rel=1e-6)

    # art 2: no val sale → survival 0, future outcomes NaN, count 0
    assert out.loc["2", "has_val_sale"] == 0
    assert out.loc["2", "fut_val_n"] == 0
    assert np.isnan(out.loc["2", "fut_price_drop"])

    # art 999: absent everywhere → left-join NaN/0 contract honored
    assert out.loc["999", "has_val_sale"] == 0
    assert np.isnan(out.loc["999", "fut_velocity"])


def test_future_outcomes_deterministic(tmp_path):
    from src.features.enrichment_matrix import build_article_future_outcomes

    _write_fixture(tmp_path)
    a = build_article_future_outcomes(["1", "2"], tmp_path)
    b = build_article_future_outcomes(["1", "2"], tmp_path)
    assert a.equals(b)  # PRAGMA threads=1 → byte-identical AVG summation


def test_future_outcomes_order_preserved(tmp_path):
    from src.features.enrichment_matrix import build_article_future_outcomes

    _write_fixture(tmp_path)
    ids = ["2", "999", "1"]
    out = build_article_future_outcomes(ids, tmp_path)
    assert list(out["article_id"]) == ids


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
