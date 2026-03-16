"""Segment profiling, cross-layer validation, and statistics.

Provides analysis functions for interpreting segmentation results:
- Segment profiles (top attributes, auto-labels)
- Discriminative profiling (over/under-represented attributes)
- Cross-layer ARI matrix (5x5)
- Per-segment statistics (purchases, price, diversity)
- Effective k (entropy-based)
- L3 dimension heatmap
- Cross-category excess similarity
- Topic sensitivity analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

logger = logging.getLogger(__name__)


class SegmentProfile(NamedTuple):
    """Profile for a single segment."""

    segment_id: int
    size: int
    fraction: float
    top_attributes: dict[str, list[str]]  # field_name → top values
    label: str  # auto-generated descriptive label


def profile_segments(
    segments_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    level: str = "l1",
    top_n: int = 5,
) -> list[SegmentProfile]:
    """Profile each segment by its dominant attributes.

    Args:
        segments_df: DataFrame with customer_id + segment columns.
        profiles_df: User profiles with top_*_json columns.
        level: Which segment level to profile ("l1", "l2", "l3", "semantic", "topic").
        top_n: Number of top attributes per field.

    Returns:
        List of SegmentProfile, one per segment.
    """
    col = f"{level}_segment"
    if col not in segments_df.columns:
        raise ValueError(f"Column {col} not found in segments_df")

    merged = segments_df[["customer_id", col]].merge(profiles_df, on="customer_id", how="inner")

    profiles = []
    total = len(merged)

    for seg_id, group in merged.groupby(col):
        size = len(group)
        fraction = size / total if total > 0 else 0.0
        top_attrs: dict[str, list[str]] = {}

        # Parse and aggregate JSON columns
        for json_col in ["top_categories_json", "top_colors_json", "top_materials_json"]:
            if json_col in group.columns:
                top_attrs[json_col] = _aggregate_json_top(group[json_col], top_n)

        # Auto-label from top category + color
        label_parts = []
        cats = top_attrs.get("top_categories_json", [])
        colors = top_attrs.get("top_colors_json", [])
        if cats:
            label_parts.append(cats[0])
        if colors:
            label_parts.append(colors[0])
        label = f"Seg-{seg_id}: {' / '.join(label_parts)}" if label_parts else f"Seg-{seg_id}"

        profiles.append(SegmentProfile(
            segment_id=int(seg_id),
            size=size,
            fraction=round(fraction, 4),
            top_attributes=top_attrs,
            label=label,
        ))

    return profiles


def cross_layer_ari(segments_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 5×5 ARI matrix across segmentation levels.

    Returns DataFrame with levels as both index and columns.
    """
    levels = ["l1_segment", "l2_segment", "l3_segment", "semantic_segment", "topic_segment"]
    available = [l for l in levels if l in segments_df.columns]

    # Drop rows with any -1 (unassigned)
    mask = (segments_df[available] >= 0).all(axis=1)
    df = segments_df.loc[mask, available]

    n = len(available)
    ari_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                ari_matrix[i, j] = 1.0
            elif i < j:
                score = adjusted_rand_score(df[available[i]], df[available[j]])
                ari_matrix[i, j] = score
                ari_matrix[j, i] = score

    labels = [l.replace("_segment", "") for l in available]
    return pd.DataFrame(ari_matrix, index=labels, columns=labels)


def compute_segment_statistics(
    segments_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    level: str = "l1",
) -> pd.DataFrame:
    """Compute per-segment statistics from user profiles.

    Returns DataFrame with segment_id as index and columns:
    - n_users, mean_purchases, mean_diversity, mean_online_ratio, mean_price_quintile
    """
    col = f"{level}_segment"
    merged = segments_df[["customer_id", col]].merge(profiles_df, on="customer_id", how="inner")

    stats = merged.groupby(col).agg(
        n_users=("customer_id", "count"),
        mean_diversity=("category_diversity", "mean"),
        mean_online_ratio=("online_ratio", "mean"),
        mean_price_quintile=("avg_price_quintile", "mean"),
    )

    # Add n_purchases if available
    if "n_purchases" in merged.columns:
        stats["mean_purchases"] = merged.groupby(col)["n_purchases"].mean()

    stats.index.name = "segment_id"
    return stats.round(4)


def save_segment_profiles(
    profiles: list[SegmentProfile],
    output_path: Path,
) -> None:
    """Save segment profiles to JSON."""
    data = [
        {
            "segment_id": p.segment_id,
            "size": p.size,
            "fraction": p.fraction,
            "top_attributes": p.top_attributes,
            "label": p.label,
        }
        for p in profiles
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def _aggregate_json_top(series: pd.Series, top_n: int) -> list[str]:
    """Aggregate JSON distribution columns across users, return top-N keys."""
    from collections import Counter

    counter: Counter[str] = Counter()
    for val in series:
        if val is None:
            continue
        if isinstance(val, str):
            try:
                dist = json.loads(val)
                if isinstance(dist, dict):
                    counter.update(dist.keys())
            except (json.JSONDecodeError, TypeError):
                pass
    return [k for k, _ in counter.most_common(top_n)]


# ---------------------------------------------------------------------------
# Discriminative Profiling
# ---------------------------------------------------------------------------


class DiscriminativeProfile(NamedTuple):
    """Discriminative profile for a single segment."""

    segment_id: int
    size: int
    fraction: float
    over_represented: dict[str, list[tuple[str, float]]]  # field → [(attr, ratio)]
    under_represented: dict[str, list[tuple[str, float]]]
    label: str


def _compute_weighted_freq(series: pd.Series) -> dict[str, float]:
    """Compute weighted frequency from JSON distribution columns.

    Each row is a JSON dict like {"T-shirt": 0.5, "Vest": 0.3}.
    Returns aggregated weighted frequencies normalized to sum=1.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    for val in series:
        if val is None:
            continue
        if isinstance(val, str):
            try:
                dist = json.loads(val)
                if isinstance(dist, dict):
                    for k, v in dist.items():
                        counter[k] += float(v)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def profile_segments_discriminative(
    segments_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    level: str = "l1",
    top_n: int = 3,
    min_population_freq: float = 0.01,
) -> list[DiscriminativeProfile]:
    """Profile segments by over/under-represented attributes vs population.

    Computes ratio = segment_weighted_freq / population_weighted_freq for each
    attribute. Ratio > 1 means over-represented, < 1 means under-represented.

    Args:
        segments_df: DataFrame with customer_id + segment columns.
        profiles_df: User profiles with top_*_json columns.
        level: Which segment level to profile.
        top_n: Number of top over/under-represented attributes per field.
        min_population_freq: Minimum population frequency to consider.

    Returns:
        List of DiscriminativeProfile, one per segment.
    """
    col = f"{level}_segment"
    if col not in segments_df.columns:
        raise ValueError(f"Column {col} not found in segments_df")

    merged = segments_df[["customer_id", col]].merge(profiles_df, on="customer_id", how="inner")
    total = len(merged)

    json_cols = [c for c in ["top_categories_json", "top_colors_json", "top_materials_json"] if c in merged.columns]

    # Population-level weighted frequencies
    pop_freqs = {jc: _compute_weighted_freq(merged[jc]) for jc in json_cols}

    profiles: list[DiscriminativeProfile] = []

    for seg_id, group in merged.groupby(col):
        size = len(group)
        fraction = size / total if total > 0 else 0.0
        over: dict[str, list[tuple[str, float]]] = {}
        under: dict[str, list[tuple[str, float]]] = {}

        for jc in json_cols:
            seg_freq = _compute_weighted_freq(group[jc])
            pop_freq = pop_freqs[jc]

            ratios = {}
            for attr in set(list(seg_freq.keys()) + list(pop_freq.keys())):
                pf = pop_freq.get(attr, 0.0)
                sf = seg_freq.get(attr, 0.0)
                if pf < min_population_freq:
                    continue
                ratios[attr] = sf / pf if pf > 0 else 0.0

            sorted_over = sorted(
                [(a, r) for a, r in ratios.items() if r > 1.0],
                key=lambda x: -x[1],
            )[:top_n]
            sorted_under = sorted(
                [(a, r) for a, r in ratios.items() if r < 1.0],
                key=lambda x: x[1],
            )[:top_n]

            field_name = jc.replace("_json", "").replace("top_", "")
            over[field_name] = sorted_over
            under[field_name] = sorted_under

        # Auto-label from top over-represented category
        label_parts = []
        cats_over = over.get("categories", [])
        colors_over = over.get("colors", [])
        if cats_over:
            label_parts.append(f"{cats_over[0][0]}({cats_over[0][1]:.1f}x)")
        if colors_over:
            label_parts.append(f"{colors_over[0][0]}({colors_over[0][1]:.1f}x)")
        label = f"Seg-{seg_id}: {' / '.join(label_parts)}" if label_parts else f"Seg-{seg_id}"

        profiles.append(DiscriminativeProfile(
            segment_id=int(seg_id),
            size=size,
            fraction=round(fraction, 4),
            over_represented=over,
            under_represented=under,
            label=label,
        ))

    return profiles


# ---------------------------------------------------------------------------
# Effective k (entropy-based)
# ---------------------------------------------------------------------------


class EffectiveKResult(NamedTuple):
    """Effective cluster count based on Shannon entropy."""

    level: str
    nominal_k: int
    effective_k: float  # exp(H)
    entropy: float  # -sum(p_i * ln(p_i))
    evenness: float  # H / ln(k)


def compute_effective_k(
    segments_df: pd.DataFrame,
    levels: list[str] | None = None,
) -> list[EffectiveKResult]:
    """Compute effective k (entropy-based) for each segmentation level.

    Effective k = exp(H) where H is the Shannon entropy of segment sizes.
    Evenness = H / ln(k), measuring how uniform the distribution is.

    Args:
        segments_df: DataFrame with segment columns.
        levels: Levels to compute (default: all available).

    Returns:
        List of EffectiveKResult.
    """
    all_levels = ["l1", "l2", "l3", "semantic", "topic"]
    if levels is None:
        levels = [l for l in all_levels if f"{l}_segment" in segments_df.columns]

    results: list[EffectiveKResult] = []
    for level in levels:
        col = f"{level}_segment"
        if col not in segments_df.columns:
            continue

        labels = segments_df[col]
        valid = labels[labels >= 0]
        counts = valid.value_counts()
        nominal_k = len(counts)

        if nominal_k <= 1:
            results.append(EffectiveKResult(
                level=level, nominal_k=nominal_k,
                effective_k=1.0, entropy=0.0, evenness=0.0,
            ))
            continue

        probs = counts.values / counts.sum()
        entropy = -float(np.sum(probs * np.log(probs)))
        effective_k = float(np.exp(entropy))
        evenness = entropy / np.log(nominal_k)

        results.append(EffectiveKResult(
            level=level,
            nominal_k=nominal_k,
            effective_k=round(effective_k, 2),
            entropy=round(entropy, 4),
            evenness=round(evenness, 4),
        ))

    return results


# ---------------------------------------------------------------------------
# L3 Dimension Heatmap Data
# ---------------------------------------------------------------------------


def compute_l3_segment_heatmap_data(
    l3_vectors: np.ndarray,
    segment_labels: np.ndarray,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute L3 per-segment mean values for all 37 dimensions.

    Returns:
        DataFrame with feature_names as index and segment IDs as columns.
        Values are per-segment means.
    """
    unique_labels = sorted(set(segment_labels[segment_labels >= 0]))
    n_features = l3_vectors.shape[1]

    if feature_names is None:
        feature_names = [f"dim_{i}" for i in range(n_features)]

    data = {}
    for seg_id in unique_labels:
        mask = segment_labels == seg_id
        data[f"seg_{seg_id}"] = l3_vectors[mask].mean(axis=0)

    return pd.DataFrame(data, index=feature_names[:n_features])


# ---------------------------------------------------------------------------
# Cross-category Excess Similarity
# ---------------------------------------------------------------------------


class ExcessSimilarityResult(NamedTuple):
    """Excess similarity above baseline for cross-category pairs."""

    threshold: float
    n_pairs: int
    mean_similarity: float
    mean_excess: float
    max_excess: float


def compute_cross_category_excess_similarity(
    cross_pairs: pd.DataFrame,
    baseline_mean: float,
    thresholds: tuple[float, ...] = (0.85, 0.90, 0.95),
) -> list[ExcessSimilarityResult]:
    """Compute cross-category similarity excess above baseline.

    Args:
        cross_pairs: DataFrame with 'similarity' column.
        baseline_mean: Mean pairwise cosine similarity across all items.
        thresholds: Similarity thresholds to evaluate at.

    Returns:
        List of ExcessSimilarityResult for each threshold.
    """
    if len(cross_pairs) == 0 or "similarity" not in cross_pairs.columns:
        return [
            ExcessSimilarityResult(t, 0, 0.0, 0.0, 0.0)
            for t in thresholds
        ]

    results: list[ExcessSimilarityResult] = []
    for thresh in thresholds:
        above = cross_pairs[cross_pairs["similarity"] >= thresh]
        n = len(above)
        if n == 0:
            results.append(ExcessSimilarityResult(thresh, 0, 0.0, 0.0, 0.0))
            continue

        sims = above["similarity"].values
        mean_sim = float(sims.mean())
        excess = sims - baseline_mean
        results.append(ExcessSimilarityResult(
            threshold=thresh,
            n_pairs=n,
            mean_similarity=round(mean_sim, 4),
            mean_excess=round(float(excess.mean()), 4),
            max_excess=round(float(excess.max()), 4),
        ))

    return results


# ---------------------------------------------------------------------------
# Topic Sensitivity Analysis
# ---------------------------------------------------------------------------


class TopicSensitivityResult(NamedTuple):
    """Result of topic sensitivity analysis for a single min_cluster_size."""

    min_cluster_size: int
    n_topics: int
    largest_topic_pct: float
    effective_k: float


def run_topic_sensitivity(
    embeddings: np.ndarray,
    texts: list[str],
    min_cluster_sizes: tuple[int, ...] = (500, 2000, 5000, 10000),
    base_config: SegmentationConfig | None = None,
) -> list[TopicSensitivityResult]:
    """Run topic modeling with varying min_cluster_size and report sensitivity.

    Args:
        embeddings: BGE embeddings (N, 768).
        texts: Corresponding texts.
        min_cluster_sizes: Values of min_cluster_size to test.
        base_config: Base segmentation config (other params held constant).

    Returns:
        List of TopicSensitivityResult.
    """
    from src.config import SegmentationConfig
    from src.segmentation.topics import fit_topics

    if base_config is None:
        base_config = SegmentationConfig()

    results: list[TopicSensitivityResult] = []
    for mcs in min_cluster_sizes:
        config = base_config._replace(hdbscan_min_cluster_size=mcs)
        topic_result = fit_topics(embeddings, texts, config=config)

        sizes = np.array(list(topic_result.topic_sizes.values()))
        total = sizes.sum()
        largest_pct = float(sizes.max() / total * 100) if total > 0 else 0.0

        n_topics = topic_result.n_topics
        if n_topics > 1:
            probs = sizes / total
            entropy = -float(np.sum(probs * np.log(probs)))
            eff_k = float(np.exp(entropy))
        else:
            eff_k = 1.0

        results.append(TopicSensitivityResult(
            min_cluster_size=mcs,
            n_topics=n_topics,
            largest_topic_pct=round(largest_pct, 1),
            effective_k=round(eff_k, 2),
        ))
        logger.info(
            "Topic sensitivity: min_cluster_size=%d → %d topics, largest=%.1f%%, eff_k=%.2f",
            mcs, n_topics, largest_pct, eff_k,
        )

    return results
