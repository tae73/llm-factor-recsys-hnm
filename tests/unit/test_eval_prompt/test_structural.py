"""Tests for structural evaluation functions."""

import numpy as np
import pandas as pd

from src.eval_prompt.structural import (
    CompletenessResult,
    CoverageResult,
    DiscriminabilityResult,
    DistributionResult,
    SchemaCheckResult,
    TokenBudgetResult,
    check_completeness,
    check_discriminability,
    check_token_budget,
    compute_coverage,
    compute_distributions,
    REASONING_FIELDS,
)


# ---------------------------------------------------------------------------
# Coverage Tests
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    """Test compute_coverage()."""

    def test_full_coverage(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = compute_coverage(df, ["a", "b"])
        assert result.overall_coverage == 1.0
        assert result.n_items == 3
        assert result.field_coverage["a"] == 1.0
        assert result.field_coverage["b"] == 1.0

    def test_partial_coverage(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "", "z"]})
        result = compute_coverage(df, ["a", "b"])
        # a: 2/3 non-null (numeric), b: 2/3 non-empty (string)
        assert abs(result.field_coverage["a"] - 2 / 3) < 0.01
        assert abs(result.field_coverage["b"] - 2 / 3) < 0.01

    def test_missing_column(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = compute_coverage(df, ["a", "missing"])
        assert result.field_coverage["missing"] == 0.0
        assert result.field_coverage["a"] == 1.0

    def test_empty_df(self):
        df = pd.DataFrame()
        result = compute_coverage(df, ["a"])
        assert result.n_items == 0
        assert result.overall_coverage == 0.0

    def test_result_type(self):
        df = pd.DataFrame({"a": [1]})
        result = compute_coverage(df, ["a"])
        assert isinstance(result, CoverageResult)


# ---------------------------------------------------------------------------
# Token Budget Tests
# ---------------------------------------------------------------------------


class TestCheckTokenBudget:
    """Test check_token_budget()."""

    def test_within_budget(self):
        texts = ["hello world " * 10] * 5  # ~20 words × 1.3 ≈ 26 tokens
        result = check_token_budget(texts, budget_limit=512)
        assert result.n_over_budget == 0
        assert result.pct_over_budget == 0.0
        assert result.budget_limit == 512

    def test_over_budget(self):
        texts = ["word " * 500]  # ~500 × 1.3 = 650 > 512
        result = check_token_budget(texts, budget_limit=512)
        assert result.n_over_budget == 1
        assert result.pct_over_budget == 1.0

    def test_empty_texts(self):
        result = check_token_budget([], budget_limit=512)
        assert result.mean_tokens == 0.0
        assert result.max_tokens == 0

    def test_statistics(self):
        texts = ["word " * 100, "word " * 200, "word " * 300]
        result = check_token_budget(texts, budget_limit=512)
        assert result.mean_tokens > 0
        assert result.median_tokens > 0
        assert result.max_tokens >= result.mean_tokens

    def test_result_type(self):
        result = check_token_budget(["hello"], budget_limit=100)
        assert isinstance(result, TokenBudgetResult)


# ---------------------------------------------------------------------------
# Distribution Tests
# ---------------------------------------------------------------------------


class TestComputeDistributions:
    """Test compute_distributions()."""

    def test_simple_distribution(self):
        df = pd.DataFrame({"l2_trendiness": ["Classic", "Classic", "Current", "Emerging"]})
        result = compute_distributions(df, ["l2_trendiness"])
        assert result.value_counts["l2_trendiness"]["Classic"] == 2
        assert result.n_unique["l2_trendiness"] == 3
        assert result.entropy["l2_trendiness"] > 0

    def test_json_array_field(self):
        df = pd.DataFrame({
            "l2_style_mood": ['["Casual", "Minimalist"]', '["Casual"]', '["Sporty"]']
        })
        result = compute_distributions(df, ["l2_style_mood"])
        assert result.value_counts["l2_style_mood"]["Casual"] == 2
        assert result.value_counts["l2_style_mood"]["Minimalist"] == 1

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = compute_distributions(df, ["l2_trendiness"])
        assert result.value_counts["l2_trendiness"] == {}
        assert result.entropy["l2_trendiness"] == 0.0

    def test_single_value_entropy(self):
        df = pd.DataFrame({"l2_trendiness": ["Classic"] * 10})
        result = compute_distributions(df, ["l2_trendiness"])
        assert result.entropy["l2_trendiness"] == 0.0  # No diversity

    def test_result_type(self):
        df = pd.DataFrame({"l2_trendiness": ["A", "B"]})
        result = compute_distributions(df, ["l2_trendiness"])
        assert isinstance(result, DistributionResult)


# ---------------------------------------------------------------------------
# Completeness Tests
# ---------------------------------------------------------------------------


class TestCheckCompleteness:
    """Test check_completeness()."""

    def test_full_completeness(self):
        records = [
            {f: f"Value for {f}" for f in REASONING_FIELDS},
            {f: f"Another {f}" for f in REASONING_FIELDS},
        ]
        import json
        df = pd.DataFrame({"reasoning_json": [json.dumps(r) for r in records]})
        result = check_completeness(df, REASONING_FIELDS)
        assert result.overall_completeness == 1.0
        assert result.n_generic == 0

    def test_generic_detection(self):
        records = [{"style_mood_preference": "Unknown", "occasion_preference": "N/A"}]
        import json
        df = pd.DataFrame({"reasoning_json": [json.dumps(records[0])]})
        result = check_completeness(df, ["style_mood_preference", "occasion_preference"])
        assert result.overall_completeness == 0.0
        assert result.n_generic == 1

    def test_empty_df(self):
        df = pd.DataFrame()
        result = check_completeness(df, REASONING_FIELDS)
        assert result.overall_completeness == 0.0
        assert result.n_generic == 0

    def test_result_type(self):
        df = pd.DataFrame({"reasoning_json": ['{"style_mood_preference": "Casual"}']})
        result = check_completeness(df, ["style_mood_preference"])
        assert isinstance(result, CompletenessResult)


# ---------------------------------------------------------------------------
# Discriminability Tests
# ---------------------------------------------------------------------------


class TestCheckDiscriminability:
    """Test check_discriminability()."""

    def test_identical_texts(self):
        texts = ["The same text repeated"] * 10
        result = check_discriminability(texts)
        assert result.mean_pairwise_sim > 0.9  # Very similar

    def test_diverse_texts(self):
        texts = [
            "This user prefers casual minimalist fashion with neutral tones",
            "A sporty active customer who loves bright colors and athletic wear",
            "Classic elegant shopper focused on formal occasions and dark palettes",
            "Bohemian free spirit with eclectic mix of vintage and modern pieces",
        ]
        result = check_discriminability(texts)
        assert result.mean_pairwise_sim < 0.9  # More diverse

    def test_empty_texts(self):
        result = check_discriminability([])
        assert result.mean_pairwise_sim == 0.0
        assert result.mean_trigrams == 0.0

    def test_trigram_count(self):
        texts = ["one two three four five six seven"]
        result = check_discriminability(texts)
        assert result.mean_trigrams > 0

    def test_result_type(self):
        result = check_discriminability(["hello world test"])
        assert isinstance(result, DiscriminabilityResult)
