"""Component B: Layer Incremental Information analysis.

Measures how much each attribute layer (L1/L2/L3) adds unique information
at the BGE embedding level — the actual input to KAR.

Three analyses:
1. CKA (Centered Kernel Alignment) between 7 ablation embedding variants
2. Purchase Coherence: intra-user purchase cosine similarity per variant
3. Purchase Separation AUC: pos vs neg pair scoring per variant
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class CKAResult(NamedTuple):
    """CKA similarity between two ablation variants."""

    combo_a: str
    combo_b: str
    cka: float


class CoherenceResult(NamedTuple):
    """Intra-user purchase coherence for one bracket × variant."""

    layer_combo: str
    activity_bracket: str
    mean_coherence: float
    std_coherence: float
    n_users: int


class SeparationResult(NamedTuple):
    """Purchase separation AUC for one variant."""

    layer_combo: str
    auc: float
    mean_pos_sim: float
    mean_neg_sim: float


# ---------------------------------------------------------------------------
# CKA computation
# ---------------------------------------------------------------------------


def _linear_kernel(X: np.ndarray) -> np.ndarray:
    """Compute linear kernel matrix K = X @ X.T."""
    return X @ X.T


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix: K_c = H @ K @ H where H = I - 1/n."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Compute HSIC (Hilbert-Schmidt Independence Criterion)."""
    n = K.shape[0]
    Kc = _center_kernel(K)
    Lc = _center_kernel(L)
    return float(np.trace(Kc @ Lc) / ((n - 1) ** 2))


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: (n_samples, d1) first representation.
        Y: (n_samples, d2) second representation.

    Returns:
        CKA value in [0, 1]. Higher = more similar representations.
    """
    Kx = _linear_kernel(X)
    Ky = _linear_kernel(Y)
    hsic_xy = _hsic(Kx, Ky)
    hsic_xx = _hsic(Kx, Kx)
    hsic_yy = _hsic(Ky, Ky)
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


def compute_cka_matrix(
    embeddings_by_combo: dict[str, np.ndarray],
    sample_size: int = 5_000,
    random_seed: int = 42,
) -> list[CKAResult]:
    """Compute pairwise CKA between all ablation embedding variants.

    Args:
        embeddings_by_combo: combo_name → (n_items, d) embeddings.
            All must have same n_items and same item ordering.
        sample_size: Subsample items for speed (CKA is O(n^2)).
        random_seed: Random seed.

    Returns:
        List of CKAResult for all pairs.
    """
    combos = sorted(embeddings_by_combo.keys())
    n_items = next(iter(embeddings_by_combo.values())).shape[0]

    # Subsample for computation
    rng = np.random.default_rng(random_seed)
    if n_items > sample_size:
        idx = rng.choice(n_items, sample_size, replace=False)
    else:
        idx = np.arange(n_items)

    sampled = {c: embeddings_by_combo[c][idx] for c in combos}

    results = []
    for i, ca in enumerate(combos):
        for j, cb in enumerate(combos):
            if j < i:
                continue
            cka = compute_linear_cka(sampled[ca], sampled[cb])
            results.append(CKAResult(combo_a=ca, combo_b=cb, cka=cka))
            if ca != cb:
                results.append(CKAResult(combo_a=cb, combo_b=ca, cka=cka))
            logger.info("  CKA(%s, %s) = %.4f", ca, cb, cka)

    return results


def cka_results_to_matrix(results: list[CKAResult]) -> pd.DataFrame:
    """Convert CKA results to a square DataFrame for heatmap plotting."""
    combos = sorted({r.combo_a for r in results})
    mat = pd.DataFrame(0.0, index=combos, columns=combos)
    for r in results:
        mat.loc[r.combo_a, r.combo_b] = r.cka
    return mat


# ---------------------------------------------------------------------------
# Purchase Coherence
# ---------------------------------------------------------------------------


ACTIVITY_BRACKETS: dict[str, tuple[int, int]] = {
    "sparse (1-4)": (1, 4),
    "light (5-9)": (5, 9),
    "moderate (10-49)": (10, 49),
    "heavy (50+)": (50, 999_999),
}


def compute_purchase_coherence(
    embeddings: np.ndarray,
    item_ids: np.ndarray,
    user_history: dict[str, list[str]],
    layer_combo: str,
    sample_users: int | None = 50_000,
    random_seed: int = 42,
) -> list[CoherenceResult]:
    """Compute mean intra-user purchase cosine similarity.

    For each user, compute the average pairwise cosine between their
    purchased items' embeddings. Higher coherence = the embedding space
    better captures the user's preference consistency.

    Args:
        embeddings: (n_items, d) L2-normalized item embeddings.
        item_ids: (n_items,) article_id strings.
        user_history: user_id → list of purchased article_ids.
        layer_combo: Name of the ablation variant.
        sample_users: Max users to evaluate.
        random_seed: Random seed.

    Returns:
        List of CoherenceResult, one per activity bracket.
    """
    id_to_idx = {str(aid): i for i, aid in enumerate(item_ids)}

    # Filter users with >=2 purchases (need pairs for coherence)
    eval_users = [u for u, h in user_history.items() if len(h) >= 2]
    if sample_users is not None and len(eval_users) > sample_users:
        rng = np.random.default_rng(random_seed)
        eval_users = list(rng.choice(eval_users, sample_users, replace=False))

    bracket_coherences: dict[str, list[float]] = {b: [] for b in ACTIVITY_BRACKETS}

    for uid in eval_users:
        history = user_history[uid]
        indices = [id_to_idx[aid] for aid in history if aid in id_to_idx]
        if len(indices) < 2:
            continue

        emb = embeddings[indices]  # (m, d)
        # Pairwise cosine = emb @ emb.T (already L2-normalized)
        sim_matrix = emb @ emb.T
        # Mean of upper triangle (excluding diagonal)
        m = len(indices)
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        coherence = float(sim_matrix[mask].mean())

        n_purchases = len(history)
        for bname, (lo, hi) in ACTIVITY_BRACKETS.items():
            if lo <= n_purchases <= hi:
                bracket_coherences[bname].append(coherence)
                break

    results = []
    for bname in ACTIVITY_BRACKETS:
        vals = bracket_coherences[bname]
        results.append(CoherenceResult(
            layer_combo=layer_combo,
            activity_bracket=bname,
            mean_coherence=float(np.mean(vals)) if vals else 0.0,
            std_coherence=float(np.std(vals)) if vals else 0.0,
            n_users=len(vals),
        ))

    return results


# ---------------------------------------------------------------------------
# Purchase Separation AUC
# ---------------------------------------------------------------------------


def compute_purchase_separation_auc(
    embeddings: np.ndarray,
    item_ids: np.ndarray,
    user_history: dict[str, list[str]],
    val_ground_truth: dict[str, set[str]],
    layer_combo: str,
    n_neg_per_user: int = 50,
    sample_users: int | None = 10_000,
    random_seed: int = 42,
) -> SeparationResult:
    """Compute AUC of separating purchased vs non-purchased items.

    For each user: centroid of train purchases → cosine to val-purchased
    items (positive) and random non-purchased items (negative).

    Args:
        embeddings: (n_items, d) L2-normalized.
        item_ids: (n_items,) article_ids.
        user_history: user_id → train article_ids.
        val_ground_truth: user_id → val article_ids.
        layer_combo: Ablation variant name.
        n_neg_per_user: Negative samples per user.
        sample_users: Max users.
        random_seed: Random seed.

    Returns:
        SeparationResult with AUC.
    """
    id_to_idx = {str(aid): i for i, aid in enumerate(item_ids)}
    n_items = len(item_ids)
    rng = np.random.default_rng(random_seed)

    eval_users = [
        u for u in val_ground_truth
        if u in user_history and len(user_history[u]) > 0
    ]
    if sample_users is not None and len(eval_users) > sample_users:
        eval_users = list(rng.choice(eval_users, sample_users, replace=False))

    pos_sims: list[float] = []
    neg_sims: list[float] = []

    for uid in eval_users:
        # Centroid from train history
        hist_indices = [id_to_idx[aid] for aid in user_history[uid] if aid in id_to_idx]
        if not hist_indices:
            continue
        centroid = embeddings[hist_indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        # Positive: val ground truth items
        for gt_item in val_ground_truth[uid]:
            if gt_item in id_to_idx:
                sim = float(centroid @ embeddings[id_to_idx[gt_item]])
                pos_sims.append(sim)

        # Negative: random items not in history or ground truth
        excluded = set(hist_indices) | {id_to_idx.get(a) for a in val_ground_truth[uid] if a in id_to_idx}
        neg_candidates = [i for i in range(n_items) if i not in excluded]
        if len(neg_candidates) > n_neg_per_user:
            neg_indices = rng.choice(neg_candidates, n_neg_per_user, replace=False)
        else:
            neg_indices = np.array(neg_candidates)
        for ni in neg_indices:
            sim = float(centroid @ embeddings[ni])
            neg_sims.append(sim)

    # Compute AUC via rank-based method
    all_sims = np.array(pos_sims + neg_sims)
    all_labels = np.array([1] * len(pos_sims) + [0] * len(neg_sims))

    if len(pos_sims) == 0 or len(neg_sims) == 0:
        return SeparationResult(layer_combo=layer_combo, auc=0.5, mean_pos_sim=0.0, mean_neg_sim=0.0)

    # Sort by score descending, count concordant pairs
    order = np.argsort(-all_sims)
    sorted_labels = all_labels[order]
    n_pos = len(pos_sims)
    n_neg = len(neg_sims)
    # AUC = P(score_pos > score_neg) = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    ranks = np.empty(len(all_sims))
    ranks[order] = np.arange(1, len(all_sims) + 1)
    pos_rank_sum = ranks[all_labels == 1].sum()
    auc = float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    # The rank formula gives P(neg_rank > pos_rank), so flip
    auc = 1.0 - auc

    result = SeparationResult(
        layer_combo=layer_combo,
        auc=auc,
        mean_pos_sim=float(np.mean(pos_sims)),
        mean_neg_sim=float(np.mean(neg_sims)),
    )
    logger.info(
        "  %s: AUC=%.4f, mean_pos=%.4f, mean_neg=%.4f",
        layer_combo, result.auc, result.mean_pos_sim, result.mean_neg_sim,
    )
    return result
