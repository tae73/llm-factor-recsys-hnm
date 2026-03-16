"""Structured attribute vectorizer for L1/L2/L3 customer profiles.

Converts user purchase history into multi-hot / numeric vectors by
aggregating item attributes across transactions.

L1 vector (~89D): top categories, colors, materials + price/channel/diversity
L2 vector (~49D): style_mood, occasion, trendiness, season_fit, target_impression + quality/versatility
L3 vector (~36D): color_harmony, tone_season, coordination_role, style_lineage + visual_weight
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import NamedTuple

import duckdb
import numpy as np
import pandas as pd

from src.knowledge.factual.prompts import (
    COLOR_HARMONY_VALUES,
    COORDINATION_ROLE_VALUES,
    OCCASION_VALUES,
    SEASON_FIT_VALUES,
    STYLE_LINEAGE_VALUES,
    STYLE_MOOD_VALUES,
    TONE_SEASON_VALUES,
    TRENDINESS_VALUES,
)

logger = logging.getLogger(__name__)


class VectorizerResult(NamedTuple):
    """Result of vectorization for one layer."""

    vectors: np.ndarray  # (n_users, d)
    customer_ids: np.ndarray
    dim: int
    feature_names: list[str]


# ---------------------------------------------------------------------------
# L1 Vector (~89D) — from user_profiles.parquet directly
# ---------------------------------------------------------------------------

# Top-N categories/colors/materials for multi-hot encoding
_L1_TOP_CATEGORIES = 50
_L1_TOP_COLORS = 20
_L1_TOP_MATERIALS = 12
_L1_PRICE_BINS = 5


def vectorize_l1(rk_path: Path) -> VectorizerResult:
    """Build L1 structured vectors from user_profiles.parquet.

    Dimensions:
      - top 50 product_type multi-hot (50D)
      - top 20 colour_group multi-hot (20D)
      - top 12 material multi-hot (12D)
      - avg_price_quintile one-hot (5D)
      - online_ratio (1D)
      - category_diversity (1D)
    Total: ~89D (exact depends on observed vocab)
    """
    df = pd.read_parquet(
        rk_path,
        columns=[
            "customer_id",
            "top_categories_json",
            "top_colors_json",
            "top_materials_json",
            "avg_price_quintile",
            "online_ratio",
            "category_diversity",
        ],
    )

    # Collect global top-N items from all users
    cat_vocab = _build_vocab(df["top_categories_json"], _L1_TOP_CATEGORIES)
    color_vocab = _build_vocab(df["top_colors_json"], _L1_TOP_COLORS)
    mat_vocab = _build_vocab(df["top_materials_json"], _L1_TOP_MATERIALS)

    n = len(df)
    dim = len(cat_vocab) + len(color_vocab) + len(mat_vocab) + _L1_PRICE_BINS + 2
    vectors = np.zeros((n, dim), dtype=np.float32)

    feature_names = (
        [f"cat_{c}" for c in cat_vocab]
        + [f"color_{c}" for c in color_vocab]
        + [f"mat_{m}" for m in mat_vocab]
        + [f"price_q{q}" for q in range(1, _L1_PRICE_BINS + 1)]
        + ["online_ratio", "category_diversity"]
    )

    offset_cat = 0
    offset_color = len(cat_vocab)
    offset_mat = offset_color + len(color_vocab)
    offset_price = offset_mat + len(mat_vocab)
    offset_extra = offset_price + _L1_PRICE_BINS

    for i, row in enumerate(df.itertuples(index=False)):
        # Multi-hot for categories
        _fill_multihot_from_json(vectors, i, offset_cat, row.top_categories_json, cat_vocab)
        _fill_multihot_from_json(vectors, i, offset_color, row.top_colors_json, color_vocab)
        _fill_multihot_from_json(vectors, i, offset_mat, row.top_materials_json, mat_vocab)

        # Price quintile one-hot
        q = int(row.avg_price_quintile) if not _is_nan(row.avg_price_quintile) else 3
        q = max(1, min(q, _L1_PRICE_BINS))
        vectors[i, offset_price + q - 1] = 1.0

        # Scalar features
        vectors[i, offset_extra] = float(row.online_ratio) if not _is_nan(row.online_ratio) else 0.5
        vectors[i, offset_extra + 1] = (
            float(row.category_diversity) if not _is_nan(row.category_diversity) else 0.0
        )

    logger.info("L1 vectors: shape=%s, %d features", vectors.shape, dim)
    return VectorizerResult(
        vectors=vectors,
        customer_ids=df["customer_id"].values,
        dim=dim,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# L2 Vector (~49D) — DuckDB aggregation from transactions x factual_knowledge
# ---------------------------------------------------------------------------

_L2_TOP_IMPRESSIONS = 10


def vectorize_l2(
    txn_path: Path,
    fk_path: Path,
    customer_ids: np.ndarray | None = None,
) -> VectorizerResult:
    """Build L2 structured vectors via DuckDB aggregation.

    Dimensions:
      - style_mood distribution (24D, STYLE_MOOD_VALUES)
      - occasion distribution (14D, OCCASION_VALUES)
      - perceived_quality mean (1D)
      - trendiness distribution (4D)
      - season_fit distribution (5D)
      - versatility mean (1D)
    Total: 49D
    """
    con = duckdb.connect()
    con.execute(f"CREATE VIEW txn AS SELECT * FROM parquet_scan('{txn_path}')")
    con.execute(f"CREATE VIEW fk AS SELECT * FROM parquet_scan('{fk_path}')")

    query = """
    SELECT
        t.customer_id,
        fk.l2_style_mood,
        fk.l2_occasion,
        fk.l2_perceived_quality,
        fk.l2_trendiness,
        fk.l2_season_fit,
        fk.l2_versatility
    FROM txn t
    LEFT JOIN fk ON t.article_id = fk.article_id
    """
    df = con.execute(query).fetchdf()
    con.close()

    # Build per-user aggregations
    style_mood_idx = {v: i for i, v in enumerate(STYLE_MOOD_VALUES)}
    occasion_idx = {v: i for i, v in enumerate(OCCASION_VALUES)}
    trendiness_idx = {v: i for i, v in enumerate(TRENDINESS_VALUES)}
    season_idx = {v: i for i, v in enumerate(SEASON_FIT_VALUES)}

    d_style = len(STYLE_MOOD_VALUES)
    d_occ = len(OCCASION_VALUES)
    d_trend = len(TRENDINESS_VALUES)
    d_season = len(SEASON_FIT_VALUES)
    dim = d_style + d_occ + 1 + d_trend + d_season + 1  # 49

    feature_names = (
        [f"l2_mood_{v}" for v in STYLE_MOOD_VALUES]
        + [f"l2_occ_{v}" for v in OCCASION_VALUES]
        + ["l2_quality_mean"]
        + [f"l2_trend_{v}" for v in TRENDINESS_VALUES]
        + [f"l2_season_{v}" for v in SEASON_FIT_VALUES]
        + ["l2_versatility_mean"]
    )

    grouped = df.groupby("customer_id")
    cids = sorted(grouped.groups.keys())
    n = len(cids)
    cid_to_idx = {c: i for i, c in enumerate(cids)}
    vectors = np.zeros((n, dim), dtype=np.float32)

    for cid, group in grouped:
        idx = cid_to_idx[cid]
        off = 0

        # Style mood distribution
        _aggregate_array_field(vectors, idx, off, group["l2_style_mood"], style_mood_idx)
        off += d_style

        # Occasion distribution
        _aggregate_array_field(vectors, idx, off, group["l2_occasion"], occasion_idx)
        off += d_occ

        # Quality mean
        q_vals = pd.to_numeric(group["l2_perceived_quality"], errors="coerce").dropna()
        vectors[idx, off] = q_vals.mean() / 5.0 if len(q_vals) > 0 else 0.5
        off += 1

        # Trendiness distribution
        _aggregate_scalar_field(vectors, idx, off, group["l2_trendiness"], trendiness_idx)
        off += d_trend

        # Season distribution
        _aggregate_scalar_field(vectors, idx, off, group["l2_season_fit"], season_idx)
        off += d_season

        # Versatility mean
        v_vals = pd.to_numeric(group["l2_versatility"], errors="coerce").dropna()
        vectors[idx, off] = v_vals.mean() / 5.0 if len(v_vals) > 0 else 0.5

    # Normalize distribution sections to sum to 1
    _normalize_section(vectors, 0, d_style)
    _normalize_section(vectors, d_style, d_occ)
    _normalize_section(vectors, d_style + d_occ + 1, d_trend)
    _normalize_section(vectors, d_style + d_occ + 1 + d_trend, d_season)

    logger.info("L2 vectors: shape=%s, %d features", vectors.shape, dim)
    return VectorizerResult(
        vectors=vectors,
        customer_ids=np.array(cids),
        dim=dim,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# L3 Vector (~36D) — DuckDB aggregation
# ---------------------------------------------------------------------------


def vectorize_l3(
    txn_path: Path,
    fk_path: Path,
    customer_ids: np.ndarray | None = None,
) -> VectorizerResult:
    """Build L3 structured vectors via DuckDB aggregation.

    Dimensions:
      - color_harmony distribution (9D)
      - tone_season distribution (6D)
      - coordination_role distribution (6D)
      - visual_weight mean (1D)
      - style_lineage top-15 (15D)
    Total: 37D
    """
    con = duckdb.connect()
    con.execute(f"CREATE VIEW txn AS SELECT * FROM parquet_scan('{txn_path}')")
    con.execute(f"CREATE VIEW fk AS SELECT * FROM parquet_scan('{fk_path}')")

    query = """
    SELECT
        t.customer_id,
        fk.l3_color_harmony,
        fk.l3_tone_season,
        fk.l3_coordination_role,
        fk.l3_visual_weight,
        fk.l3_style_lineage
    FROM txn t
    LEFT JOIN fk ON t.article_id = fk.article_id
    """
    df = con.execute(query).fetchdf()
    con.close()

    harmony_idx = {v: i for i, v in enumerate(COLOR_HARMONY_VALUES)}
    tone_idx = {v: i for i, v in enumerate(TONE_SEASON_VALUES)}
    coord_idx = {v: i for i, v in enumerate(COORDINATION_ROLE_VALUES)}
    # Top 15 style lineages for multi-hot
    top_lineages = STYLE_LINEAGE_VALUES[:15]
    lineage_idx = {v: i for i, v in enumerate(top_lineages)}

    d_harm = len(COLOR_HARMONY_VALUES)
    d_tone = len(TONE_SEASON_VALUES)
    d_coord = len(COORDINATION_ROLE_VALUES)
    d_lin = len(top_lineages)
    dim = d_harm + d_tone + d_coord + 1 + d_lin  # 37

    feature_names = (
        [f"l3_harm_{v}" for v in COLOR_HARMONY_VALUES]
        + [f"l3_tone_{v}" for v in TONE_SEASON_VALUES]
        + [f"l3_coord_{v}" for v in COORDINATION_ROLE_VALUES]
        + ["l3_visual_weight_mean"]
        + [f"l3_lineage_{v}" for v in top_lineages]
    )

    grouped = df.groupby("customer_id")
    cids = sorted(grouped.groups.keys())
    n = len(cids)
    cid_to_idx = {c: i for i, c in enumerate(cids)}
    vectors = np.zeros((n, dim), dtype=np.float32)

    for cid, group in grouped:
        idx = cid_to_idx[cid]
        off = 0

        _aggregate_scalar_field(vectors, idx, off, group["l3_color_harmony"], harmony_idx)
        off += d_harm

        _aggregate_scalar_field(vectors, idx, off, group["l3_tone_season"], tone_idx)
        off += d_tone

        _aggregate_scalar_field(vectors, idx, off, group["l3_coordination_role"], coord_idx)
        off += d_coord

        # Visual weight mean
        w_vals = pd.to_numeric(group["l3_visual_weight"], errors="coerce").dropna()
        vectors[idx, off] = w_vals.mean() / 5.0 if len(w_vals) > 0 else 0.5
        off += 1

        # Style lineage multi-hot
        _aggregate_array_field(vectors, idx, off, group["l3_style_lineage"], lineage_idx)

    _normalize_section(vectors, 0, d_harm)
    _normalize_section(vectors, d_harm, d_tone)
    _normalize_section(vectors, d_harm + d_tone, d_coord)

    logger.info("L3 vectors: shape=%s, %d features", vectors.shape, dim)
    return VectorizerResult(
        vectors=vectors,
        customer_ids=np.array(cids),
        dim=dim,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_vocab(json_col: pd.Series, top_n: int) -> list[str]:
    """Build ordered vocabulary from JSON distribution columns."""
    from collections import Counter

    counter: Counter[str] = Counter()
    for val in json_col:
        dist = _safe_parse_json(val)
        counter.update(dist.keys())
    return [k for k, _ in counter.most_common(top_n)]


def _fill_multihot_from_json(
    vectors: np.ndarray,
    row_idx: int,
    offset: int,
    json_val: str | None,
    vocab: list[str] | dict[str, int],
) -> None:
    """Fill multi-hot from a JSON distribution string."""
    dist = _safe_parse_json(json_val)
    idx_map = vocab if isinstance(vocab, dict) else {v: i for i, v in enumerate(vocab)}
    for key, weight in dist.items():
        if key in idx_map:
            vectors[row_idx, offset + idx_map[key]] = float(weight)


def _aggregate_array_field(
    vectors: np.ndarray,
    row_idx: int,
    offset: int,
    series: pd.Series,
    idx_map: dict[str, int],
) -> None:
    """Aggregate an array-type column into distribution counts."""
    for val in series:
        items = _parse_list(val)
        for item in items:
            if item in idx_map:
                vectors[row_idx, offset + idx_map[item]] += 1.0


def _aggregate_scalar_field(
    vectors: np.ndarray,
    row_idx: int,
    offset: int,
    series: pd.Series,
    idx_map: dict[str, int],
) -> None:
    """Aggregate a scalar-type column into distribution counts."""
    for val in series:
        if val is not None and not _is_nan(val):
            key = str(val)
            if key in idx_map:
                vectors[row_idx, offset + idx_map[key]] += 1.0


def _normalize_section(vectors: np.ndarray, start: int, length: int) -> None:
    """Normalize a section of the vector to sum to 1 per row."""
    section = vectors[:, start : start + length]
    row_sums = section.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    vectors[:, start : start + length] = section / row_sums


def _safe_parse_json(val) -> dict:
    """Safely parse a JSON string or return empty dict."""
    if val is None or _is_nan(val):
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            result = json.loads(val)
            return result if isinstance(result, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _parse_list(val) -> list[str]:
    """Parse a list field (may be JSON string, list, or scalar)."""
    if val is None or _is_nan(val):
        return []
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except (json.JSONDecodeError, TypeError):
            return [val]
    return [str(val)]


def _is_nan(val) -> bool:
    """Check if a value is NaN (works for float, str, None)."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.lower() in ("nan", "none", ""):
        return True
    return False
