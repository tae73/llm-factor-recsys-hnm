"""Unit tests for src/analysis/cold_start.py."""

import numpy as np
import pytest

from src.analysis.cold_start import (
    ACTIVITY_BRACKETS,
    BracketResult,
    _compute_hr_ndcg_mrr,
    bracket_results_to_dataframe,
    compute_contentbased_retrieval,
)


# ---------------------------------------------------------------------------
# _compute_hr_ndcg_mrr
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_hit(self):
        """First item is a hit."""
        item_ids = np.array(["a", "b", "c", "d", "e"])
        ranked = np.array([0, 1, 2])  # items a, b, c
        gt = {"a"}
        hr, ndcg, mrr = _compute_hr_ndcg_mrr(ranked, gt, item_ids, k=3)
        assert hr == 1.0
        assert mrr == 1.0
        assert ndcg > 0

    def test_no_hit(self):
        """No items in ground truth."""
        item_ids = np.array(["a", "b", "c", "d", "e"])
        ranked = np.array([0, 1, 2])
        gt = {"d", "e"}
        hr, ndcg, mrr = _compute_hr_ndcg_mrr(ranked, gt, item_ids, k=3)
        assert hr == 0.0
        assert mrr == 0.0
        assert ndcg == 0.0

    def test_hit_at_position_2(self):
        """Hit at position 2 (0-indexed)."""
        item_ids = np.array(["a", "b", "c"])
        ranked = np.array([0, 1, 2])
        gt = {"c"}
        hr, ndcg, mrr = _compute_hr_ndcg_mrr(ranked, gt, item_ids, k=3)
        assert hr == 1.0
        assert mrr == pytest.approx(1.0 / 3)

    def test_multiple_hits(self):
        """Multiple hits → higher NDCG."""
        item_ids = np.array(["a", "b", "c"])
        ranked = np.array([0, 1, 2])
        gt = {"a", "b", "c"}
        hr, ndcg, mrr = _compute_hr_ndcg_mrr(ranked, gt, item_ids, k=3)
        assert hr == 1.0
        assert ndcg == pytest.approx(1.0)
        assert mrr == 1.0


# ---------------------------------------------------------------------------
# compute_contentbased_retrieval
# ---------------------------------------------------------------------------


class TestContentbasedRetrieval:
    def test_basic_retrieval(self):
        """Test with small synthetic data."""
        rng = np.random.default_rng(42)
        n_items = 50
        d = 16

        # Create embeddings with clear cluster structure
        emb = rng.standard_normal((n_items, d)).astype(np.float32)
        # Normalize
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        item_ids = np.array([f"item_{i:03d}" for i in range(n_items)])

        user_history = {
            "u1": ["item_000", "item_001", "item_002"],  # 3 purchases
            "u2": ["item_010"],  # 1 purchase
        }
        val_gt = {
            "u1": {"item_003", "item_004"},
            "u2": {"item_011"},
        }

        results = compute_contentbased_retrieval(
            embeddings=emb,
            item_ids=item_ids,
            user_history=user_history,
            val_ground_truth=val_gt,
            layer_combo="test",
            k=12,
            sample_users=None,
        )

        assert len(results) == len(ACTIVITY_BRACKETS)
        assert all(isinstance(r, BracketResult) for r in results)
        # u2 has 1 purchase → bracket "1"
        bracket_1 = [r for r in results if r.bracket == "1"]
        assert len(bracket_1) == 1
        assert bracket_1[0].n_users == 1

    def test_empty_users(self):
        """No valid users → empty results."""
        emb = np.eye(5, dtype=np.float32)
        item_ids = np.array(["a", "b", "c", "d", "e"])
        results = compute_contentbased_retrieval(
            emb, item_ids, {}, {}, "test", k=3,
        )
        assert results == []


# ---------------------------------------------------------------------------
# bracket_results_to_dataframe
# ---------------------------------------------------------------------------


class TestBracketResultsToDF:
    def test_conversion(self):
        results = [
            BracketResult("1", "L1", 0.1, 0.05, 0.08, 100),
            BracketResult("5-9", "L1", 0.2, 0.1, 0.15, 200),
        ]
        df = bracket_results_to_dataframe(results)
        assert len(df) == 2
        assert "bracket" in df.columns
        assert "layer_combo" in df.columns
