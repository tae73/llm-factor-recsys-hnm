"""Unit tests for the E2-4 user-side value pieces (predictive cell logic + data assembly).

The predictive-cell logic is tested with a synthetic signal; the data assembly is a light
real-file smoke (fast). No API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_predictive_cell_detects_real_signal():
    """rep that encodes the label → (demo+rep) beats demo → PASS; noise rep → NO."""
    from witnesses.probe_E2c_user_value import _predictive_cell

    rng = np.random.default_rng(0)
    n = 900
    y = rng.integers(0, 3, n)
    demo = rng.normal(size=(n, 6))  # noise baseline
    rep_signal = np.eye(3)[y] + rng.normal(scale=0.3, size=(n, 3))  # encodes label
    rep_noise = rng.normal(size=(n, 5))
    fut = pd.DataFrame({"o": y.astype(str)})

    c_sig = _predictive_cell("r", rep_signal, demo, fut, ["o"], "audience", "b", "t")
    assert c_sig["lift_verdict"] == "PASS" and c_sig["lift_value"] >= 0.01
    c_noise = _predictive_cell("r", rep_noise, demo, fut, ["o"], "audience", "b", "t")
    assert c_noise["lift_verdict"] in ("NO", "MARGINAL")


def test_predictive_cell_skips_thin_outcome():
    from witnesses.probe_E2c_user_value import _predictive_cell

    fut = pd.DataFrame({"o": ["a"] * 100})  # <500 and single class
    c = _predictive_cell(
        "r", np.zeros((100, 3)), np.zeros((100, 3)), fut, ["o"], "audience", "b", "t"
    )
    assert c["lift_verdict"] == "NO" and c["per_outcome"] == {}


@pytest.mark.parametrize("fn", ["select_cohort", "build_future_outcomes"])
def test_user_axes_smoke(fn):
    """Light real-file smoke: cohort is active-LLM users; future outcomes are val-derived."""
    from src.features import user_axes

    ids = user_axes.select_cohort(n_sample=300, seed=42)
    assert 100 < len(ids) <= 300 and all(isinstance(c, str) for c in ids)
    if fn == "build_future_outcomes":
        fut = user_axes.build_future_outcomes(ids)
        assert len(fut) == len(ids)
        for col in ["fut_price_tier", "fut_online", "fut_repurchase"]:
            assert col in fut.columns
        assert fut["fut_price_tier"].between(0, 5).all()  # digitized tiers


def test_demographic_and_reps_aligned():
    from src.features import user_axes

    ids = user_axes.select_cohort(n_sample=300, seed=42)
    demo = user_axes.build_demographic(ids)
    reps = user_axes.build_user_representations(ids)
    assert demo.shape[0] == len(ids)
    for r, X in reps.items():
        assert X.shape[0] == len(ids), r
    assert reps["reasoning_bge"].shape[1] == 50  # PCA-50


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
