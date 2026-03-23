"""Unit tests for src/analysis/preference_diversity.py."""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.preference_diversity import (
    DiversityResult,
    _build_user_attr_distributions,
    _compute_pairwise_jsd,
    _compute_user_entropy,
    _parse_multi_value,
    diversity_results_to_dataframe,
)


# ---------------------------------------------------------------------------
# _compute_user_entropy
# ---------------------------------------------------------------------------


class TestUserEntropy:
    def test_uniform_distribution(self):
        """Uniform distribution → max entropy."""
        dists = np.array([[0.25, 0.25, 0.25, 0.25]])
        entropies = _compute_user_entropy(dists)
        assert entropies[0] == pytest.approx(2.0, abs=0.01)  # log2(4) = 2

    def test_peaked_distribution(self):
        """Peaked distribution → low entropy."""
        dists = np.array([[0.97, 0.01, 0.01, 0.01]])
        entropies = _compute_user_entropy(dists)
        assert entropies[0] < 0.3

    def test_degenerate_distribution(self):
        """All mass on one value → entropy = 0."""
        dists = np.array([[1.0, 0.0, 0.0]])
        entropies = _compute_user_entropy(dists)
        assert entropies[0] == pytest.approx(0.0, abs=1e-6)

    def test_multiple_users(self):
        """Multiple users computed correctly."""
        dists = np.array([
            [0.5, 0.5],
            [1.0, 0.0],
        ])
        entropies = _compute_user_entropy(dists)
        assert len(entropies) == 2
        assert entropies[0] == pytest.approx(1.0, abs=0.01)
        assert entropies[1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _compute_pairwise_jsd
# ---------------------------------------------------------------------------


class TestPairwiseJSD:
    def test_identical_users(self):
        """Identical distributions → JSD ≈ 0."""
        dists = np.array([
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2],
        ])
        jsd = _compute_pairwise_jsd(dists, n_pairs=100)
        assert jsd < 0.01

    def test_different_users(self):
        """Very different distributions → high JSD."""
        dists = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        jsd = _compute_pairwise_jsd(dists, n_pairs=100)
        assert jsd > 0.5

    def test_single_user(self):
        """Single user → JSD = 0."""
        dists = np.array([[0.5, 0.5]])
        jsd = _compute_pairwise_jsd(dists, n_pairs=10)
        assert jsd == 0.0


# ---------------------------------------------------------------------------
# _build_user_attr_distributions
# ---------------------------------------------------------------------------


class TestBuildDistributions:
    def test_basic_aggregation(self):
        """Simple aggregation without time decay."""
        df = pd.DataFrame({
            "customer_id": ["u1", "u1", "u1", "u2", "u2"],
            "attr": ["A", "A", "B", "B", "C"],
        })
        dists, vocab = _build_user_attr_distributions(df, "attr", is_multi=False)
        assert set(vocab) == {"A", "B", "C"}
        assert dists.shape[0] == 2  # 2 users
        # u1: A=2, B=1 → [2/3, 1/3, 0]
        a_idx = vocab.index("A")
        b_idx = vocab.index("B")
        assert dists[0, a_idx] == pytest.approx(2 / 3, abs=0.01)
        assert dists[0, b_idx] == pytest.approx(1 / 3, abs=0.01)

    def test_multi_value(self):
        """Multi-value attributes are exploded."""
        df = pd.DataFrame({
            "customer_id": ["u1", "u1"],
            "attr": ['["Casual", "Minimalist"]', '["Casual", "Sporty"]'],
        })
        dists, vocab = _build_user_attr_distributions(df, "attr", is_multi=True)
        assert "Casual" in vocab
        assert "Minimalist" in vocab
        assert "Sporty" in vocab
        casual_idx = vocab.index("Casual")
        # Casual appears in both rows → 2 out of 4 total
        assert dists[0, casual_idx] == pytest.approx(2 / 4, abs=0.01)

    def test_with_time_decay(self):
        """Recent purchases get higher weight."""
        df = pd.DataFrame({
            "customer_id": ["u1", "u1"],
            "attr": ["A", "B"],
            "t_dat": pd.to_datetime(["2020-06-30", "2020-01-01"]),
        })
        dists, vocab = _build_user_attr_distributions(
            df, "attr", is_multi=False, decay_halflife_days=90,
        )
        a_idx = vocab.index("A")
        b_idx = vocab.index("B")
        # A is more recent → should have higher weight
        assert dists[0, a_idx] > dists[0, b_idx]


# ---------------------------------------------------------------------------
# diversity_results_to_dataframe
# ---------------------------------------------------------------------------


class TestDiversityToDF:
    def test_conversion(self):
        results = [
            DiversityResult("style_mood", "l2", 2.5, 0.3, 0.4, 0.85, 0.16),
        ]
        df = diversity_results_to_dataframe(results)
        assert len(df) == 1
        assert "recommendation_value_index" in df.columns
