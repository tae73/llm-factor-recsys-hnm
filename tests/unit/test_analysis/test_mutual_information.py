"""Unit tests for src/analysis/mutual_information.py."""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.mutual_information import (
    MIResult,
    PMIResult,
    _compute_mi_for_column,
    _parse_multi_value,
    mi_results_to_dataframe,
)


# ---------------------------------------------------------------------------
# _parse_multi_value
# ---------------------------------------------------------------------------


class TestParseMultiValue:
    def test_json_array(self):
        assert _parse_multi_value('["Casual", "Minimalist"]') == ["Casual", "Minimalist"]

    def test_plain_string(self):
        assert _parse_multi_value("Casual") == ["Casual"]

    def test_none(self):
        assert _parse_multi_value(None) == []

    def test_nan(self):
        assert _parse_multi_value(float("nan")) == []

    def test_empty_string(self):
        assert _parse_multi_value("") == []

    def test_nan_string(self):
        assert _parse_multi_value("nan") == []

    def test_malformed_json(self):
        assert _parse_multi_value("[invalid") == ["[invalid"]


# ---------------------------------------------------------------------------
# _compute_mi_for_column
# ---------------------------------------------------------------------------


class TestComputeMI:
    def test_perfect_correlation(self):
        """When attribute perfectly predicts label, MI should be high."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        values = np.array(["A", "A", "A", "B", "B", "B"])
        mi, nmi, n_vals = _compute_mi_for_column(labels, values)
        assert mi > 0
        assert nmi > 0.99  # Perfect correlation → NMI ≈ 1.0
        assert n_vals == 2

    def test_no_correlation(self):
        """When attribute is independent of label, MI should be ~0."""
        rng = np.random.default_rng(42)
        n = 10_000
        labels = rng.integers(0, 2, size=n).astype(float)
        values = rng.choice(["A", "B", "C", "D"], size=n)
        mi, nmi, n_vals = _compute_mi_for_column(labels, values)
        assert mi < 0.01  # Near zero
        assert n_vals == 4

    def test_single_value(self):
        """Single-value attribute → MI = 0."""
        labels = np.array([0, 1, 0, 1])
        values = np.array(["A", "A", "A", "A"])
        mi, nmi, n_vals = _compute_mi_for_column(labels, values)
        assert mi == 0.0
        assert nmi == 0.0
        assert n_vals == 1

    def test_partial_correlation(self):
        """Partial correlation → 0 < MI < max."""
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # A correlates with label but not perfectly
        values = np.array(["A", "A", "B", "B", "A", "B", "B", "B"])
        mi, nmi, n_vals = _compute_mi_for_column(labels, values)
        assert 0 < mi
        assert 0 < nmi < 1.0

    def test_mi_non_negative(self):
        """MI is always non-negative."""
        rng = np.random.default_rng(123)
        for _ in range(10):
            n = 100
            labels = rng.integers(0, 2, size=n).astype(float)
            values = rng.choice(["X", "Y", "Z"], size=n)
            mi, nmi, _ = _compute_mi_for_column(labels, values)
            assert mi >= -1e-10  # Allow tiny float errors


# ---------------------------------------------------------------------------
# mi_results_to_dataframe
# ---------------------------------------------------------------------------


class TestMIResultsToDataFrame:
    def test_conversion(self):
        results = [
            MIResult(attribute="style_mood", layer="l2", mi=0.05, nmi=0.1, n_values=17),
            MIResult(attribute="product_type", layer="metadata", mi=0.1, nmi=0.2, n_values=50),
        ]
        df = mi_results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(df.columns) == {"attribute", "layer", "mi", "nmi", "n_values"}
        assert df.iloc[0]["attribute"] == "style_mood"

    def test_empty_results(self):
        df = mi_results_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
