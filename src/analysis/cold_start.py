"""Component D: Cold-Start Knowledge Utility analysis.

Measures how much attribute knowledge compensates for sparse CF signals
by computing content-based retrieval accuracy stratified by user
purchase-count bracket and item popularity bracket.

For each ablation variant (7 layer combos), computes HR@12 and NDCG@12
using a pure embedding-similarity retrieval: user purchase history
centroid → cosine to all items → top-12.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

ACTIVITY_BRACKETS: dict[str, tuple[int, int]] = {
    "1": (1, 1),
    "2-4": (2, 4),
    "5-9": (5, 9),
    "10-19": (10, 19),
    "20-49": (20, 49),
    "50+": (50, 999_999),
}


class BracketResult(NamedTuple):
    """Content-based retrieval result for one bracket × layer combo."""

    bracket: str
    layer_combo: str
    hr_at_12: float
    ndcg_at_12: float
    mrr: float
    n_users: int


class ItemPopularityResult(NamedTuple):
    """Retrieval result stratified by item popularity."""

    popularity_bracket: str
    layer_combo: str
    hr_at_12: float
    ndcg_at_12: float
    n_items_in_bracket: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_hr_ndcg_mrr(
    ranked_items: np.ndarray,
    ground_truth: set[str],
    item_ids: np.ndarray,
    k: int = 12,
) -> tuple[float, float, float]:
    """Compute HR@k, NDCG@k, MRR for a single user.

    Args:
        ranked_items: Top-k item indices sorted by score (descending).
        ground_truth: Set of article_ids the user purchased in val.
        item_ids: Array mapping index → article_id.
        k: Cutoff.

    Returns:
        (hr, ndcg, mrr)
    """
    hits = np.array([item_ids[idx] in ground_truth for idx in ranked_items[:k]])
    hr = float(hits.any())

    # NDCG
    dcg = np.sum(hits / np.log2(np.arange(2, k + 2)))
    ideal_hits = min(len(ground_truth), k)
    idcg = np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2))) if ideal_hits > 0 else 0.0
    ndcg = float(dcg / idcg) if idcg > 0 else 0.0

    # MRR
    hit_positions = np.where(hits)[0]
    mrr = float(1.0 / (hit_positions[0] + 1)) if len(hit_positions) > 0 else 0.0

    return hr, ndcg, mrr


def _build_user_purchase_history(
    train_txn_path: Path,
) -> dict[str, list[str]]:
    """Build user → [article_id, ...] from train transactions."""
    df = pd.read_parquet(train_txn_path, columns=["customer_id", "article_id"])
    return (
        df.groupby("customer_id")["article_id"]
        .apply(list)
        .to_dict()
    )


def _load_val_ground_truth(val_gt_path: Path) -> dict[str, set[str]]:
    """Load val ground truth: user_id → set of article_ids."""
    raw = json.loads(val_gt_path.read_text())
    return {uid: set(items) for uid, items in raw.items()}


# ---------------------------------------------------------------------------
# Content-based retrieval
# ---------------------------------------------------------------------------


def compute_contentbased_retrieval(
    embeddings: np.ndarray,
    item_ids: np.ndarray,
    user_history: dict[str, list[str]],
    val_ground_truth: dict[str, set[str]],
    layer_combo: str,
    k: int = 12,
    sample_users: int | None = 50_000,
    random_seed: int = 42,
) -> list[BracketResult]:
    """Compute content-based retrieval per activity bracket.

    For each user: centroid of purchased item embeddings → cosine to all items → HR@12.

    Args:
        embeddings: (n_items, d) item embeddings (L2-normalized).
        item_ids: (n_items,) article_id strings.
        user_history: user_id → list of purchased article_ids (train).
        val_ground_truth: user_id → set of article_ids (val).
        layer_combo: Name of the layer combo (e.g., "L1+L2+L3").
        k: Cutoff for metrics.
        sample_users: Max users to evaluate (None = all).
        random_seed: Random seed.

    Returns:
        List of BracketResult, one per bracket.
    """
    # Build article_id → embedding index
    id_to_idx = {str(aid): i for i, aid in enumerate(item_ids)}

    # Filter to users in both history and ground truth
    eval_users = [u for u in val_ground_truth if u in user_history and len(user_history[u]) > 0]
    if sample_users is not None and len(eval_users) > sample_users:
        rng = np.random.default_rng(random_seed)
        eval_users = list(rng.choice(eval_users, sample_users, replace=False))

    logger.info(
        "Evaluating %d users for combo %s (k=%d)", len(eval_users), layer_combo, k,
    )

    # Pre-compute user centroids and bracket assignment
    user_centroids: list[np.ndarray] = []
    user_gt: list[set[str]] = []
    user_brackets: list[str] = []
    valid_users: list[str] = []

    for uid in eval_users:
        history = user_history[uid]
        hist_indices = [id_to_idx[aid] for aid in history if aid in id_to_idx]
        if not hist_indices:
            continue

        centroid = embeddings[hist_indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        n_purchases = len(history)
        bracket = None
        for bname, (lo, hi) in ACTIVITY_BRACKETS.items():
            if lo <= n_purchases <= hi:
                bracket = bname
                break
        if bracket is None:
            bracket = "50+"

        user_centroids.append(centroid)
        user_gt.append(val_ground_truth[uid])
        user_brackets.append(bracket)
        valid_users.append(uid)

    if not user_centroids:
        return []

    # Batch compute scores: (n_users, n_items) via matrix multiply
    centroids_mat = np.stack(user_centroids)  # (n_users, d)
    scores = centroids_mat @ embeddings.T  # (n_users, n_items)

    # Per-user top-k and metrics
    bracket_metrics: dict[str, list[tuple[float, float, float]]] = {
        b: [] for b in ACTIVITY_BRACKETS
    }

    for i in range(len(valid_users)):
        top_k_idx = np.argpartition(-scores[i], k)[:k]
        top_k_idx = top_k_idx[np.argsort(-scores[i, top_k_idx])]
        hr, ndcg, mrr = _compute_hr_ndcg_mrr(top_k_idx, user_gt[i], item_ids, k)
        bracket_metrics[user_brackets[i]].append((hr, ndcg, mrr))

    results = []
    for bname in ACTIVITY_BRACKETS:
        metrics = bracket_metrics[bname]
        if not metrics:
            results.append(BracketResult(
                bracket=bname, layer_combo=layer_combo,
                hr_at_12=0.0, ndcg_at_12=0.0, mrr=0.0, n_users=0,
            ))
            continue
        hrs, ndcgs, mrrs = zip(*metrics)
        results.append(BracketResult(
            bracket=bname,
            layer_combo=layer_combo,
            hr_at_12=float(np.mean(hrs)),
            ndcg_at_12=float(np.mean(ndcgs)),
            mrr=float(np.mean(mrrs)),
            n_users=len(metrics),
        ))
        logger.info(
            "  %s [%s]: HR@%d=%.4f, NDCG@%d=%.4f, MRR=%.4f (n=%d)",
            layer_combo, bname, k, results[-1].hr_at_12,
            k, results[-1].ndcg_at_12, results[-1].mrr, results[-1].n_users,
        )

    return results


def run_all_combos(
    embeddings_by_combo: dict[str, tuple[np.ndarray, np.ndarray]],
    train_txn_path: Path,
    val_gt_path: Path,
    k: int = 12,
    sample_users: int | None = 50_000,
    random_seed: int = 42,
) -> list[BracketResult]:
    """Run content-based retrieval for all ablation combos.

    Args:
        embeddings_by_combo: combo_name → (embeddings, item_ids) arrays.
        train_txn_path: Path to train_transactions.parquet.
        val_gt_path: Path to val_ground_truth.json.
        k: Cutoff.
        sample_users: Max users to evaluate.
        random_seed: Random seed.

    Returns:
        List of BracketResult across all combos and brackets.
    """
    user_history = _build_user_purchase_history(train_txn_path)
    val_gt = _load_val_ground_truth(val_gt_path)

    all_results: list[BracketResult] = []
    for combo, (emb, ids) in embeddings_by_combo.items():
        results = compute_contentbased_retrieval(
            emb, ids, user_history, val_gt, combo, k, sample_users, random_seed,
        )
        all_results.extend(results)

    return all_results


def compute_item_popularity_retrieval(
    embeddings: np.ndarray,
    item_ids: np.ndarray,
    train_txn_path: Path,
    val_gt_path: Path,
    layer_combo: str,
    k: int = 12,
    sample_users: int | None = 50_000,
    random_seed: int = 42,
) -> list[ItemPopularityResult]:
    """Compute retrieval accuracy stratified by item popularity.

    Groups val ground truth items into popularity brackets based on
    train purchase count, then measures HR@12 for items in each bracket.
    """
    user_history = _build_user_purchase_history(train_txn_path)
    val_gt = _load_val_ground_truth(val_gt_path)

    # Item popularity from train
    txn = pd.read_parquet(train_txn_path, columns=["article_id"])
    item_counts = txn["article_id"].value_counts()

    # Popularity brackets
    pop_brackets = {
        "cold (0)": (0, 0),
        "tail (1-10)": (1, 10),
        "low (11-100)": (11, 100),
        "mid (101-1K)": (101, 1000),
        "head (1K+)": (1001, 999_999),
    }

    id_to_idx = {str(aid): i for i, aid in enumerate(item_ids)}
    eval_users = [u for u in val_gt if u in user_history and len(user_history[u]) > 0]

    if sample_users is not None and len(eval_users) > sample_users:
        rng = np.random.default_rng(random_seed)
        eval_users = list(rng.choice(eval_users, sample_users, replace=False))

    # Compute centroids and scores
    centroids = []
    valid_users = []
    for uid in eval_users:
        history = user_history[uid]
        hist_indices = [id_to_idx[aid] for aid in history if aid in id_to_idx]
        if not hist_indices:
            continue
        centroid = embeddings[hist_indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        centroids.append(centroid)
        valid_users.append(uid)

    if not centroids:
        return []

    centroids_mat = np.stack(centroids)
    scores = centroids_mat @ embeddings.T

    # For each val ground truth item, check if it appears in top-k
    bracket_hits: dict[str, list[float]] = {b: [] for b in pop_brackets}
    bracket_item_counts: dict[str, set[str]] = {b: set() for b in pop_brackets}

    for i, uid in enumerate(valid_users):
        top_k_idx = np.argpartition(-scores[i], k)[:k]
        top_k_set = {item_ids[j] for j in top_k_idx}

        for gt_item in val_gt[uid]:
            count = int(item_counts.get(gt_item, 0))
            for bname, (lo, hi) in pop_brackets.items():
                if lo <= count <= hi:
                    bracket_hits[bname].append(1.0 if gt_item in top_k_set else 0.0)
                    bracket_item_counts[bname].add(gt_item)
                    break

    results = []
    for bname in pop_brackets:
        hits = bracket_hits[bname]
        results.append(ItemPopularityResult(
            popularity_bracket=bname,
            layer_combo=layer_combo,
            hr_at_12=float(np.mean(hits)) if hits else 0.0,
            ndcg_at_12=0.0,  # simplified: HR only for popularity analysis
            n_items_in_bracket=len(bracket_item_counts[bname]),
        ))

    return results


def bracket_results_to_dataframe(results: list[BracketResult]) -> pd.DataFrame:
    """Convert bracket results to a DataFrame for plotting."""
    return pd.DataFrame([r._asdict() for r in results])
