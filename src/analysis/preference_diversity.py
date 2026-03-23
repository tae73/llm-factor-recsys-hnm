"""Component C: Attribute Preference Diversity analysis.

Replaces user-level clustering with direct measurement of how diverse
user preferences are *within* each attribute dimension.

For each attribute:
- User-level entropy: how sharp is each user's preference? (low = focused)
- Pairwise JSD: how different are users from each other? (high = differentiated)
- Temporal stability: are preferences consistent over time? (high = real, not noise)
- Recommendation Value Index (RVI) = JSD / entropy — high = high recommendation value
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class DiversityResult(NamedTuple):
    """Diversity metrics for a single attribute."""

    attribute: str
    layer: str
    mean_user_entropy: float
    std_user_entropy: float
    mean_pairwise_jsd: float
    temporal_stability: float  # mean cosine between first/second half
    recommendation_value_index: float  # JSD / entropy


# ---------------------------------------------------------------------------
# Attribute definitions (same as mutual_information.py)
# ---------------------------------------------------------------------------

# column_name, label, is_multi_value, layer
_ANALYSIS_ATTRS: list[tuple[str, str, bool, str]] = [
    # Metadata
    ("product_type_name", "product_type", False, "metadata"),
    ("colour_group_name", "colour_group", False, "metadata"),
    ("section_name", "section", False, "metadata"),
    # L1
    ("l1_material", "material", False, "l1"),
    ("l1_closure", "closure", False, "l1"),
    # L2
    ("l2_style_mood", "style_mood", True, "l2"),
    ("l2_occasion", "occasion", True, "l2"),
    ("l2_perceived_quality", "perceived_quality", False, "l2"),
    ("l2_trendiness", "trendiness", False, "l2"),
    ("l2_season_fit", "season_fit", False, "l2"),
    ("l2_versatility", "versatility", False, "l2"),
    # L3
    ("l3_color_harmony", "color_harmony", False, "l3"),
    ("l3_tone_season", "tone_season", False, "l3"),
    ("l3_coordination_role", "coordination_role", False, "l3"),
    ("l3_visual_weight", "visual_weight", False, "l3"),
    ("l3_style_lineage", "style_lineage", True, "l3"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_multi_value(val: object) -> list[str]:
    """Parse JSON array string into list of values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s == "nan":
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if v]
        except json.JSONDecodeError:
            pass
    return [s]


def _build_user_attr_distributions(
    txn_with_attrs: pd.DataFrame,
    col: str,
    is_multi: bool,
    decay_halflife_days: int = 90,
) -> tuple[np.ndarray, list[str]]:
    """Build user attribute distributions with exponential time decay.

    Args:
        txn_with_attrs: DataFrame with customer_id, t_dat, and attribute column.
        col: Attribute column name.
        is_multi: Whether the column contains JSON arrays.
        decay_halflife_days: Half-life for exponential decay weighting.

    Returns:
        (distributions, vocab) where distributions is (n_users, |vocab|) and
        vocab is the list of attribute values.
    """
    # Compute time weights
    if "t_dat" in txn_with_attrs.columns:
        max_date = txn_with_attrs["t_dat"].max()
        days_ago = (max_date - txn_with_attrs["t_dat"]).dt.days.values.astype(np.float64)
        weights = np.exp(-np.log(2) * days_ago / decay_halflife_days)
    else:
        weights = np.ones(len(txn_with_attrs))

    # Build value list per row
    if is_multi:
        row_values = txn_with_attrs[col].apply(_parse_multi_value)
    else:
        row_values = txn_with_attrs[col].astype(str).replace("nan", "_NONE_").replace("<NA>", "_NONE_").apply(lambda x: [x])

    # Collect vocabulary
    all_values: set[str] = set()
    for vals in row_values:
        all_values.update(vals)
    vocab = sorted(all_values)
    val_to_idx = {v: i for i, v in enumerate(vocab)}

    # Aggregate per user
    user_ids = txn_with_attrs["customer_id"].values
    unique_users = np.unique(user_ids)
    user_to_uidx = {u: i for i, u in enumerate(unique_users)}
    n_users = len(unique_users)
    n_vocab = len(vocab)

    distributions = np.zeros((n_users, n_vocab), dtype=np.float64)

    for row_idx in range(len(txn_with_attrs)):
        uid = user_to_uidx[user_ids[row_idx]]
        w = weights[row_idx]
        for v in row_values.iloc[row_idx]:
            if v in val_to_idx:
                distributions[uid, val_to_idx[v]] += w

    # Normalize to probability distributions
    row_sums = distributions.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    distributions /= row_sums

    return distributions, vocab


def _compute_user_entropy(distributions: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy for each user's attribute distribution."""
    # H = -sum(p * log2(p)) for each row
    safe = np.maximum(distributions, 1e-12)
    return -np.sum(distributions * np.log2(safe), axis=1) * (distributions > 0).astype(float).max(axis=1)


def _compute_pairwise_jsd(
    distributions: np.ndarray,
    n_pairs: int = 100_000,
    random_seed: int = 42,
) -> float:
    """Compute mean pairwise JSD between random user pairs."""
    n_users = distributions.shape[0]
    if n_users < 2:
        return 0.0

    rng = np.random.default_rng(random_seed)
    max_pairs = min(n_pairs, n_users * (n_users - 1) // 2)

    # Sample random pairs
    idx_a = rng.integers(0, n_users, size=max_pairs)
    idx_b = rng.integers(0, n_users, size=max_pairs)
    # Ensure a != b
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % n_users

    jsds = np.array([
        jensenshannon(distributions[a], distributions[b])
        for a, b in zip(idx_a, idx_b)
    ])

    return float(np.nanmean(jsds))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_preference_diversity(
    train_txn_path: Path,
    fk_path: Path,
    articles_path: Path,
    decay_halflife_days: int = 90,
    n_jsd_pairs: int = 100_000,
    sample_users: int | None = 100_000,
    random_seed: int = 42,
) -> list[DiversityResult]:
    """Compute preference diversity metrics for all attributes.

    Args:
        train_txn_path: Path to train_transactions.parquet.
        fk_path: Path to factual_knowledge.parquet.
        articles_path: Path to articles.parquet.
        decay_halflife_days: Half-life for exponential decay.
        n_jsd_pairs: Number of random user pairs for JSD.
        sample_users: Max users to analyze (None = all).
        random_seed: Random seed.

    Returns:
        List of DiversityResult, one per attribute.
    """
    import duckdb

    logger.info("Loading transactions and attributes...")
    txn = pd.read_parquet(train_txn_path, columns=["customer_id", "article_id", "t_dat"])
    fk = pd.read_parquet(fk_path)
    articles = pd.read_parquet(articles_path, columns=[
        "article_id", "product_type_name", "colour_group_name", "section_name",
    ])
    fk["article_id"] = fk["article_id"].astype(str)
    articles["article_id"] = articles["article_id"].astype(str)
    txn["article_id"] = txn["article_id"].astype(str)

    # Subsample users
    if sample_users is not None:
        rng = np.random.default_rng(random_seed)
        unique_users = txn["customer_id"].unique()
        if len(unique_users) > sample_users:
            keep_users = set(rng.choice(unique_users, sample_users, replace=False))
            txn = txn[txn["customer_id"].isin(keep_users)].reset_index(drop=True)

    # Join transactions with attributes
    con = duckdb.connect()
    con.register("txn", txn)
    con.register("fk", fk)
    con.register("articles", articles)

    attr_cols = [c for c, _, _, _ in _ANALYSIS_ATTRS]
    meta_cols = ["product_type_name", "colour_group_name", "section_name"]
    fk_cols = [c for c in attr_cols if c not in meta_cols and c in fk.columns]

    fk_select = ", ".join(f"fk.{c}" for c in fk_cols)
    meta_select = ", ".join(f"articles.{c}" for c in meta_cols)

    query = f"""
        SELECT txn.customer_id, txn.t_dat, {meta_select}, {fk_select}
        FROM txn
        LEFT JOIN articles ON txn.article_id = articles.article_id
        LEFT JOIN fk ON txn.article_id = fk.article_id
    """
    merged = con.execute(query).fetchdf()
    con.close()

    logger.info("Merged %d transaction-attribute rows", len(merged))

    results: list[DiversityResult] = []

    for col, label, is_multi, layer in _ANALYSIS_ATTRS:
        if col not in merged.columns:
            logger.warning("Column %s not found, skipping", col)
            continue

        logger.info("Computing diversity for %s.%s...", layer, label)

        # Build distributions
        distributions, vocab = _build_user_attr_distributions(
            merged, col, is_multi, decay_halflife_days,
        )

        if distributions.shape[0] < 2 or len(vocab) < 2:
            results.append(DiversityResult(
                attribute=label, layer=layer,
                mean_user_entropy=0.0, std_user_entropy=0.0,
                mean_pairwise_jsd=0.0, temporal_stability=0.0,
                recommendation_value_index=0.0,
            ))
            continue

        # User entropy
        entropies = _compute_user_entropy(distributions)
        mean_entropy = float(np.mean(entropies))
        std_entropy = float(np.std(entropies))

        # Pairwise JSD
        mean_jsd = _compute_pairwise_jsd(distributions, n_jsd_pairs, random_seed)

        # Temporal stability: split by time, compare distributions
        temporal_stability = _compute_temporal_stability(
            merged, col, is_multi, decay_halflife_days,
        )

        # RVI
        rvi = mean_jsd / mean_entropy if mean_entropy > 0 else 0.0

        results.append(DiversityResult(
            attribute=label,
            layer=layer,
            mean_user_entropy=mean_entropy,
            std_user_entropy=std_entropy,
            mean_pairwise_jsd=mean_jsd,
            temporal_stability=temporal_stability,
            recommendation_value_index=rvi,
        ))
        logger.info(
            "  %s.%s: entropy=%.3f, JSD=%.3f, stability=%.3f, RVI=%.3f",
            layer, label, mean_entropy, mean_jsd, temporal_stability, rvi,
        )

    return results


def _compute_temporal_stability(
    merged: pd.DataFrame,
    col: str,
    is_multi: bool,
    decay_halflife_days: int,
    sample_users: int = 10_000,
) -> float:
    """Compute temporal stability: cosine between first/second half distributions."""
    if "t_dat" not in merged.columns:
        return 0.0

    # Split each user's transactions at their median date
    user_medians = merged.groupby("customer_id")["t_dat"].transform("median")
    first_half = merged[merged["t_dat"] <= user_medians].copy()
    second_half = merged[merged["t_dat"] > user_medians].copy()

    if first_half.empty or second_half.empty:
        return 0.0

    # Subsample users present in both halves
    users_both = set(first_half["customer_id"].unique()) & set(second_half["customer_id"].unique())
    if len(users_both) < 10:
        return 0.0

    users_both_list = sorted(users_both)
    if len(users_both_list) > sample_users:
        rng = np.random.default_rng(42)
        users_both_list = list(rng.choice(users_both_list, sample_users, replace=False))

    first_half = first_half[first_half["customer_id"].isin(users_both_list)]
    second_half = second_half[second_half["customer_id"].isin(users_both_list)]

    dist_first, vocab_first = _build_user_attr_distributions(
        first_half, col, is_multi, decay_halflife_days,
    )
    dist_second, vocab_second = _build_user_attr_distributions(
        second_half, col, is_multi, decay_halflife_days,
    )

    # Align vocabularies
    vocab_union = sorted(set(vocab_first) | set(vocab_second))
    idx_first = {v: i for i, v in enumerate(vocab_first)}
    idx_second = {v: i for i, v in enumerate(vocab_second)}

    n_users = dist_first.shape[0]
    d = len(vocab_union)
    aligned_first = np.zeros((n_users, d))
    aligned_second = np.zeros((n_users, d))

    for j, v in enumerate(vocab_union):
        if v in idx_first:
            aligned_first[:, j] = dist_first[:, idx_first[v]]
        if v in idx_second:
            aligned_second[:, j] = dist_second[:, idx_second[v]]

    # Cosine similarity per user
    norms_f = np.linalg.norm(aligned_first, axis=1)
    norms_s = np.linalg.norm(aligned_second, axis=1)
    valid = (norms_f > 0) & (norms_s > 0)
    if not valid.any():
        return 0.0

    cosines = np.sum(aligned_first[valid] * aligned_second[valid], axis=1) / (
        norms_f[valid] * norms_s[valid]
    )
    return float(np.mean(cosines))


def diversity_results_to_dataframe(results: list[DiversityResult]) -> pd.DataFrame:
    """Convert diversity results to a DataFrame for plotting."""
    return pd.DataFrame([r._asdict() for r in results])
