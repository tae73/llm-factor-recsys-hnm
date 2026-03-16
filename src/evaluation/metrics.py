"""Evaluation metrics for recommendation quality.

Implements MAP@K (Kaggle official), HR@K, NDCG@K, MRR with parallel computation.
All predictions and ground truth use dict[str, list[str]] format:
    {customer_id: [article_id ranked list]}
"""

from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from typing import NamedTuple

import numpy as np

from src.config import EvalConfig, EvalResult


# ---------------------------------------------------------------------------
# Per-user metric functions (vectorized where possible)
# ---------------------------------------------------------------------------


def compute_ap_at_k(predicted: list[str], actual: list[str], k: int = 12) -> float:
    """Average Precision at K — Kaggle official formula.

    AP@K = (1 / min(m, K)) * sum_{i=1}^{K} P(i) * rel(i)
    where m = |actual|, P(i) = precision at cutoff i, rel(i) = 1 if predicted[i] in actual.
    """
    if not actual:
        return 0.0

    predicted_k = predicted[:k]
    actual_set = set(actual)
    m = min(len(actual), k)

    hits = 0
    score = 0.0
    for i, pred in enumerate(predicted_k):
        if pred in actual_set:
            hits += 1
            score += hits / (i + 1)

    return score / m


def _compute_hr_at_k_single(predicted: list[str], actual: list[str], k: int) -> float:
    """Hit Rate for a single user: 1 if any hit in top-K, else 0."""
    if not actual:
        return 0.0
    return 1.0 if set(predicted[:k]) & set(actual) else 0.0


def _compute_ndcg_at_k_single(predicted: list[str], actual: list[str], k: int) -> float:
    """NDCG@K for a single user."""
    if not actual:
        return 0.0

    predicted_k = predicted[:k]
    actual_set = set(actual)

    # DCG
    dcg = sum(
        1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed, log2(rank+1) where rank starts at 1
        for i, pred in enumerate(predicted_k)
        if pred in actual_set
    )

    # Ideal DCG
    n_relevant = min(len(actual), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def _compute_rr_single(predicted: list[str], actual: list[str], k: int) -> float:
    """Reciprocal Rank for a single user (within top-K)."""
    if not actual:
        return 0.0

    actual_set = set(actual)
    for i, pred in enumerate(predicted[:k]):
        if pred in actual_set:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Aggregate metric functions
# ---------------------------------------------------------------------------


def compute_map_at_k(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    k: int = 12,
) -> float:
    """Mean Average Precision at K across all users in ground_truth."""
    users = list(ground_truth.keys())
    if not users:
        return 0.0

    scores = np.array(
        list(
            map(
                lambda u: compute_ap_at_k(predictions.get(u, []), ground_truth[u], k),
                users,
            )
        )
    )
    return float(np.mean(scores))


def compute_hr_at_k(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    k: int = 12,
) -> float:
    """Hit Rate at K across all users."""
    users = list(ground_truth.keys())
    if not users:
        return 0.0

    scores = np.array(
        list(
            map(
                lambda u: _compute_hr_at_k_single(predictions.get(u, []), ground_truth[u], k),
                users,
            )
        )
    )
    return float(np.mean(scores))


def compute_ndcg_at_k(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    k: int = 12,
) -> float:
    """NDCG at K across all users."""
    users = list(ground_truth.keys())
    if not users:
        return 0.0

    scores = np.array(
        list(
            map(
                lambda u: _compute_ndcg_at_k_single(predictions.get(u, []), ground_truth[u], k),
                users,
            )
        )
    )
    return float(np.mean(scores))


def compute_mrr(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    k: int = 12,
) -> float:
    """Mean Reciprocal Rank across all users."""
    users = list(ground_truth.keys())
    if not users:
        return 0.0

    scores = np.array(
        list(
            map(
                lambda u: _compute_rr_single(predictions.get(u, []), ground_truth[u], k),
                users,
            )
        )
    )
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


def evaluate(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    config: EvalConfig = EvalConfig(),
) -> EvalResult:
    """Compute all metrics in parallel using ThreadPoolExecutor."""
    metric_fns = {
        "map": lambda: compute_map_at_k(predictions, ground_truth, config.k),
        "hr": lambda: compute_hr_at_k(predictions, ground_truth, config.k),
        "ndcg": lambda: compute_ndcg_at_k(predictions, ground_truth, config.k),
        "mrr": lambda: compute_mrr(predictions, ground_truth, config.k),
    }

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {name: pool.submit(fn) for name, fn in metric_fns.items()}
        results = {name: future.result() for name, future in futures.items()}

    return EvalResult(
        map_at_k=results["map"],
        hr_at_k=results["hr"],
        ndcg_at_k=results["ndcg"],
        mrr=results["mrr"],
    )


def evaluate_by_cohort(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    cohorts: dict[str, set[str]],
    config: EvalConfig = EvalConfig(),
) -> dict[str, EvalResult]:
    """Evaluate metrics separately for each user cohort (e.g., active, sparse, cold)."""
    results = {}
    for cohort_name, user_set in cohorts.items():
        cohort_gt = {u: gt for u, gt in ground_truth.items() if u in user_set}
        cohort_pred = {u: predictions.get(u, []) for u in cohort_gt}
        results[cohort_name] = evaluate(cohort_pred, cohort_gt, config)
    return results
