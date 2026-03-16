"""Unit tests for src/evaluation/metrics module."""
import pytest

from src.config import EvalConfig
from src.evaluation.metrics import (
    compute_ap_at_k,
    compute_hr_at_k,
    compute_map_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    evaluate,
)


class TestAPAtK:
    def test_perfect_prediction(self):
        """All actual items predicted in correct order."""
        predicted = ["a", "b", "c"]
        actual = ["a", "b", "c"]
        # P(1)=1/1, P(2)=2/2, P(3)=3/3 -> AP = (1+1+1)/3 = 1.0
        assert compute_ap_at_k(predicted, actual, k=12) == pytest.approx(1.0)

    def test_no_hit(self):
        """No overlap between predicted and actual."""
        predicted = ["x", "y", "z"]
        actual = ["a", "b", "c"]
        assert compute_ap_at_k(predicted, actual, k=12) == 0.0

    def test_partial_hit(self):
        """Some items match."""
        predicted = ["a", "x", "b", "y"]
        actual = ["a", "b"]
        # P(1)=1/1*1, P(2)=1/2*0, P(3)=2/3*1, P(4)=... -> AP = (1 + 2/3) / 2
        expected = (1.0 + 2.0 / 3.0) / 2.0
        assert compute_ap_at_k(predicted, actual, k=12) == pytest.approx(expected)

    def test_empty_actual(self):
        assert compute_ap_at_k(["a", "b"], [], k=12) == 0.0

    def test_k_cutoff(self):
        """Only top-K predictions should be considered."""
        predicted = ["x", "y", "a"]
        actual = ["a"]
        # At k=2, "a" is not in top-2
        assert compute_ap_at_k(predicted, actual, k=2) == 0.0
        # At k=3, "a" is at position 3
        assert compute_ap_at_k(predicted, actual, k=3) == pytest.approx(1.0 / 3.0)


class TestMAPAtK:
    def test_single_user_perfect(self):
        preds = {"u1": ["a", "b"]}
        gt = {"u1": ["a", "b"]}
        assert compute_map_at_k(preds, gt, k=12) == pytest.approx(1.0)

    def test_empty_ground_truth(self):
        assert compute_map_at_k({}, {}, k=12) == 0.0


class TestHRAtK:
    def test_all_hits(self):
        preds = {"u1": ["a"], "u2": ["b"]}
        gt = {"u1": ["a"], "u2": ["b"]}
        assert compute_hr_at_k(preds, gt, k=12) == pytest.approx(1.0)

    def test_no_hits(self):
        preds = {"u1": ["x"]}
        gt = {"u1": ["a"]}
        assert compute_hr_at_k(preds, gt, k=12) == 0.0

    def test_missing_user(self):
        """User in GT but not in predictions -> 0 HR."""
        preds = {}
        gt = {"u1": ["a"]}
        assert compute_hr_at_k(preds, gt, k=12) == 0.0


class TestNDCGAtK:
    def test_perfect_single(self):
        preds = {"u1": ["a"]}
        gt = {"u1": ["a"]}
        assert compute_ndcg_at_k(preds, gt, k=12) == pytest.approx(1.0)

    def test_no_hit(self):
        preds = {"u1": ["x"]}
        gt = {"u1": ["a"]}
        assert compute_ndcg_at_k(preds, gt, k=12) == 0.0


class TestMRR:
    def test_first_position(self):
        preds = {"u1": ["a", "b"]}
        gt = {"u1": ["a"]}
        assert compute_mrr(preds, gt, k=12) == pytest.approx(1.0)

    def test_second_position(self):
        preds = {"u1": ["x", "a"]}
        gt = {"u1": ["a"]}
        assert compute_mrr(preds, gt, k=12) == pytest.approx(0.5)


class TestEvaluate:
    def test_returns_eval_result(self):
        preds = {"u1": ["a", "b"], "u2": ["c"]}
        gt = {"u1": ["a"], "u2": ["c"]}
        result = evaluate(preds, gt, EvalConfig(k=12))
        assert hasattr(result, "map_at_k")
        assert hasattr(result, "hr_at_k")
        assert hasattr(result, "ndcg_at_k")
        assert hasattr(result, "mrr")
        assert result.map_at_k > 0
        assert result.hr_at_k > 0
