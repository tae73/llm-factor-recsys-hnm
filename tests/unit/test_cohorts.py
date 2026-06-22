"""Unit tests for src/evaluation/cohorts module (cohort + discovery eval)."""

import pandas as pd
import pytest

from src.evaluation.cohorts import (
    activity_cohorts,
    discovery_map,
    evaluate_cohorts,
    repurchase_vs_new_decomposition,
)

# ---------------------------------------------------------------------------
# Activity cohort assignment
# ---------------------------------------------------------------------------


def test_activity_cohorts_brackets():
    """Users are bucketed by train purchase count into the right brackets."""
    txn = pd.DataFrame(
        {
            "customer_id": (
                ["u1"]  # 1 purchase  -> "1"
                + ["u2", "u2", "u2"]  # 3 purchases -> "2-4"
                + ["u3"] * 7  # 7 purchases -> "5-9"
                + ["u4"] * 15  # 15 purchases -> "10-19"
                + ["u5"] * 25  # 25 purchases -> "20+"
            ),
            "article_id": ["a"] * (1 + 3 + 7 + 15 + 25),
        }
    )
    cohorts = activity_cohorts(txn)
    assert cohorts["u1"] == "1"
    assert cohorts["u2"] == "2-4"
    assert cohorts["u3"] == "5-9"
    assert cohorts["u4"] == "10-19"
    assert cohorts["u5"] == "20+"


def test_evaluate_cohorts_assigns_new_users(repurchase_history):
    """A user absent from train history is scored in the 'new' bracket."""
    ground_truth = {"u1": ["a1"], "ghost": ["zz"]}
    predictions = {"u1": ["a1"], "ghost": ["zz"]}
    results = evaluate_cohorts(predictions, ground_truth, repurchase_history, k=12)
    # u1 has 2 train items -> "2-4"; ghost has none -> "new".
    assert "new" in results
    assert "2-4" in results
    # Each bracket scored only its own member (both perfect hits here).
    assert results["new"].hr_at_k == pytest.approx(1.0)
    assert results["2-4"].hr_at_k == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Discovery-only evaluation (key new metric)
# ---------------------------------------------------------------------------


@pytest.fixture
def repurchase_history():
    """Train history sets: u1 bought {a1, a2}, u2 bought {a3}."""
    return {"u1": {"a1", "a2"}, "u2": {"a3"}}


def test_discovery_map_excludes_repurchased_gt(repurchase_history):
    """GT items already in the user's train history are dropped before scoring."""
    # u1 GT = [a1 (repurchase), n1 (new)]. Predicting a1 (owned) must NOT count;
    # only n1 is in the discovery GT, and it is not predicted -> 0 discovery MAP.
    ground_truth = {"u1": ["a1", "n1"]}
    predictions = {"u1": ["a1"]}  # repurchase-only predictor
    res = discovery_map(predictions, ground_truth, repurchase_history, k=12)
    assert res.map_at_k == pytest.approx(0.0)

    # Now predict the new item -> discovery MAP becomes positive.
    predictions2 = {"u1": ["n1"]}
    res2 = discovery_map(predictions2, ground_truth, repurchase_history, k=12)
    assert res2.map_at_k == pytest.approx(1.0)


def test_discovery_map_skips_users_with_no_new_gt(repurchase_history):
    """A user whose entire GT is repurchases contributes nothing to discovery."""
    # u2 GT = [a3] which is a repurchase -> empty discovery GT -> skipped.
    # u1 GT = [n1] (new), predicted -> perfect. Mean over scored users = 1.0.
    ground_truth = {"u1": ["n1"], "u2": ["a3"]}
    predictions = {"u1": ["n1"], "u2": ["a3"]}
    res = discovery_map(predictions, ground_truth, repurchase_history, k=12)
    # Only u1 is scored (u2 skipped), so MAP = u1's AP = 1.0 (not averaged with u2's 0).
    assert res.map_at_k == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Repurchase-vs-new decomposition
# ---------------------------------------------------------------------------


def test_decomposition_fractions(repurchase_history):
    """Fractions count (user, GT item) pairs as repurchase vs new correctly."""
    # u1 GT: a1 (rep), a2 (rep), n1 (new) -> 2 rep, 1 new
    # u2 GT: a3 (rep), n2 (new)           -> 1 rep, 1 new
    ground_truth = {"u1": ["a1", "a2", "n1"], "u2": ["a3", "n2"]}
    dec = repurchase_vs_new_decomposition(ground_truth, repurchase_history)
    assert dec.n_gt_items == 5
    assert dec.repurchase_frac == pytest.approx(3 / 5)
    assert dec.new_frac == pytest.approx(2 / 5)
    assert dec.repurchase_frac + dec.new_frac == pytest.approx(1.0)


def test_decomposition_empty():
    dec = repurchase_vs_new_decomposition({}, {})
    assert dec.n_gt_items == 0
    assert dec.repurchase_frac == 0.0
    assert dec.new_frac == 0.0
