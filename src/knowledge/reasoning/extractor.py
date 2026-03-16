"""L1 direct aggregation + sparse user fallback + profile assembly.

Stage A: DuckDB bulk aggregation of train transactions + factual knowledge
         → per-user L1 statistics (category/color/material distributions, price, channel, diversity)
         → recent N items with L2 attributes
Stage C: Template-based reasoning text for sparse users (1-4 purchases)
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import ReasoningConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L1 Direct Aggregation (DuckDB bulk)
# ---------------------------------------------------------------------------


def aggregate_l1_profiles(
    txn_path: Path,
    articles_path: Path,
    fk_path: Path,
    config: ReasoningConfig = ReasoningConfig(),
) -> pd.DataFrame:
    """Bulk-aggregate L1 statistics for all users via DuckDB.

    Computes per-user:
      - n_purchases, n_unique_articles, n_unique_types
      - Top categories/colors/materials with proportions (JSON)
      - avg_price_quintile, online_ratio
      - category_diversity (normalized Shannon entropy)

    Uses exponential decay weighting: exp(-ln(2) * days_since / halflife).

    Args:
        txn_path: Path to train_transactions.parquet.
        articles_path: Path to articles.parquet.
        fk_path: Path to factual_knowledge.parquet.
        config: Reasoning configuration.

    Returns:
        DataFrame with one row per customer_id.
    """
    halflife = config.l1_time_weight_halflife_days
    con = duckdb.connect()

    # Register parquet files
    con.execute(f"CREATE VIEW txn AS SELECT * FROM parquet_scan('{txn_path}')")
    con.execute(f"CREATE VIEW articles AS SELECT * FROM parquet_scan('{articles_path}')")
    con.execute(f"CREATE VIEW fk AS SELECT * FROM parquet_scan('{fk_path}')")

    # Get max date for decay calculation
    max_date = con.execute("SELECT MAX(t_dat) FROM txn").fetchone()[0]

    # Bulk aggregation query
    query = f"""
    WITH base AS (
        SELECT
            t.customer_id,
            t.article_id,
            t.t_dat,
            t.price,
            t.sales_channel_id,
            a.product_type_name,
            a.colour_group_name,
            fk.l1_material,
            -- Exponential decay weight
            exp(-ln(2) * ('{max_date}'::DATE - t.t_dat) / {halflife}) AS time_weight
        FROM txn t
        LEFT JOIN articles a ON t.article_id = a.article_id
        LEFT JOIN fk ON t.article_id = fk.article_id
    )
    SELECT
        customer_id,
        COUNT(*) AS n_purchases,
        COUNT(DISTINCT article_id) AS n_unique_articles,
        COUNT(DISTINCT product_type_name) AS n_unique_types,
        -- Weighted average price
        SUM(price * time_weight) / NULLIF(SUM(time_weight), 0) AS avg_price,
        -- Online ratio
        SUM(CASE WHEN sales_channel_id = 2 THEN 1.0 ELSE 0.0 END) / COUNT(*) AS online_ratio,
        -- Top categories (weighted counts as JSON)
        LIST(product_type_name) AS all_types,
        LIST(colour_group_name) AS all_colors,
        LIST(l1_material) AS all_materials,
        -- Time weights for diversity calculation
        LIST(time_weight) AS all_weights,
        LIST(product_type_name || '::' || CAST(time_weight AS VARCHAR)) AS types_weighted
    FROM base
    GROUP BY customer_id
    """
    df = con.execute(query).fetchdf()
    con.close()

    # Post-process: compute distributions and diversity
    df = df.assign(
        top_categories_json=lambda d: d.apply(
            lambda r: _weighted_distribution(r["all_types"], r["all_weights"], top_n=10), axis=1
        ),
        top_colors_json=lambda d: d.apply(
            lambda r: _weighted_distribution(r["all_colors"], r["all_weights"], top_n=10), axis=1
        ),
        top_materials_json=lambda d: d.apply(
            lambda r: _weighted_distribution(r["all_materials"], r["all_weights"], top_n=10), axis=1
        ),
        category_diversity=lambda d: d.apply(
            lambda r: _compute_diversity_score(r["all_types"], r["all_weights"]), axis=1
        ),
        avg_price_quintile=lambda d: _price_to_quintile(d["avg_price"]),
    ).drop(columns=["all_types", "all_colors", "all_materials", "all_weights", "types_weighted", "avg_price"])

    logger.info("Aggregated L1 profiles for %d users", len(df))
    return df


def _weighted_distribution(
    values: list | np.ndarray,
    weights: list | np.ndarray,
    top_n: int = 10,
) -> str:
    """Compute weighted distribution of categorical values, return as JSON string."""
    if values is None or (hasattr(values, '__len__') and len(values) == 0):
        return "{}"
    counter: dict[str, float] = {}
    for v, w in zip(values, weights):
        if v is not None and str(v) != "None":
            counter[str(v)] = counter.get(str(v), 0.0) + float(w)
    total = sum(counter.values())
    if total == 0:
        return "{}"
    # Normalize and take top N
    dist = {k: round(v / total, 4) for k, v in sorted(counter.items(), key=lambda x: -x[1])[:top_n]}
    return json.dumps(dist)


def _compute_diversity_score(values: list | np.ndarray, weights: list | np.ndarray) -> float:
    """Compute normalized Shannon entropy of weighted categorical distribution."""
    if values is None or (hasattr(values, '__len__') and len(values) == 0):
        return 0.0
    counter: dict[str, float] = {}
    for v, w in zip(values, weights):
        if v is not None and str(v) != "None":
            counter[str(v)] = counter.get(str(v), 0.0) + float(w)
    total = sum(counter.values())
    if total == 0 or len(counter) <= 1:
        return 0.0
    probs = [v / total for v in counter.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(counter))
    return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0


def _price_to_quintile(prices: pd.Series) -> pd.Series:
    """Convert raw prices to 1-5 quintile positions."""
    # H&M prices are Kaggle-normalized (0-1 range), map to 1-5 quintiles
    quintiles = pd.qcut(prices.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    return quintiles.astype(float)


# ---------------------------------------------------------------------------
# Recent Items with L2 Attributes
# ---------------------------------------------------------------------------


def get_recent_items_batch(
    txn_path: Path,
    fk_path: Path,
    customer_ids: list[str],
    limit: int = 20,
) -> dict[str, list[dict]]:
    """Get recent N items with L2 attributes for multiple users at once.

    Args:
        txn_path: Path to train_transactions.parquet.
        fk_path: Path to factual_knowledge.parquet.
        customer_ids: List of customer IDs.
        limit: Max recent items per user.

    Returns:
        {customer_id: [{"article_id": ..., "product_type_name": ..., L2 fields...}, ...]}
    """
    con = duckdb.connect()
    con.execute(f"CREATE VIEW txn AS SELECT * FROM parquet_scan('{txn_path}')")
    con.execute(f"CREATE VIEW fk AS SELECT * FROM parquet_scan('{fk_path}')")

    # Create temp table for customer filter
    con.execute("CREATE TEMP TABLE target_customers (customer_id VARCHAR)")
    con.executemany(
        "INSERT INTO target_customers VALUES (?)",
        [(cid,) for cid in customer_ids],
    )

    query = f"""
    WITH ranked AS (
        SELECT
            t.customer_id,
            t.article_id,
            t.t_dat,
            fk.l2_style_mood,
            fk.l2_occasion,
            fk.l2_perceived_quality,
            fk.l2_trendiness,
            fk.l2_season_fit,
            fk.l2_target_impression,
            fk.l2_versatility,
            fk.super_category,
            ROW_NUMBER() OVER (PARTITION BY t.customer_id ORDER BY t.t_dat DESC) AS rn
        FROM txn t
        INNER JOIN target_customers tc ON t.customer_id = tc.customer_id
        LEFT JOIN fk ON t.article_id = fk.article_id
    )
    SELECT * FROM ranked WHERE rn <= {limit}
    ORDER BY customer_id, rn
    """
    df = con.execute(query).fetchdf()
    con.close()

    # Group by customer_id
    result: dict[str, list[dict]] = {}
    for cid, group in df.groupby("customer_id"):
        items = []
        for _, row in group.iterrows():
            item = {
                "article_id": row["article_id"],
                "l2_style_mood": _parse_list_field(row.get("l2_style_mood")),
                "l2_occasion": _parse_list_field(row.get("l2_occasion")),
                "l2_perceived_quality": row.get("l2_perceived_quality"),
                "l2_trendiness": row.get("l2_trendiness"),
                "l2_season_fit": row.get("l2_season_fit"),
                "l2_target_impression": row.get("l2_target_impression"),
                "l2_versatility": row.get("l2_versatility"),
                "super_category": row.get("super_category"),
            }
            items.append(item)
        result[str(cid)] = items

    return result


def _parse_list_field(val) -> list[str]:
    """Parse a list field from Parquet (may be string JSON or actual list)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except json.JSONDecodeError:
            return [val]
    return [str(val)]


# ---------------------------------------------------------------------------
# L3 Distribution Computation
# ---------------------------------------------------------------------------


def compute_l3_distributions_batch(
    txn_path: Path,
    fk_path: Path,
    customer_ids: list[str],
) -> dict[str, dict]:
    """Compute L3 attribute distributions for multiple users.

    Returns:
        {customer_id: {
            "shared": {"color_harmony": {"Monochromatic": 0.45, ...}, ...},
            "by_category": {"Apparel": {"l3_slot6": {"I-line": 0.55, ...}, ...}, ...}
        }}
    """
    con = duckdb.connect()
    con.execute(f"CREATE VIEW txn AS SELECT * FROM parquet_scan('{txn_path}')")
    con.execute(f"CREATE VIEW fk AS SELECT * FROM parquet_scan('{fk_path}')")

    con.execute("CREATE TEMP TABLE target_customers (customer_id VARCHAR)")
    con.executemany(
        "INSERT INTO target_customers VALUES (?)",
        [(cid,) for cid in customer_ids],
    )

    query = """
    SELECT
        t.customer_id,
        fk.super_category,
        fk.l3_color_harmony,
        fk.l3_tone_season,
        fk.l3_coordination_role,
        fk.l3_visual_weight,
        fk.l3_style_lineage,
        fk.l3_slot6,
        fk.l3_slot7
    FROM txn t
    INNER JOIN target_customers tc ON t.customer_id = tc.customer_id
    LEFT JOIN fk ON t.article_id = fk.article_id
    """
    df = con.execute(query).fetchdf()
    con.close()

    # Shared L3 fields
    shared_fields = ["l3_color_harmony", "l3_tone_season", "l3_coordination_role", "l3_visual_weight"]
    # Category-specific L3 slots
    cat_fields = ["l3_slot6", "l3_slot7"]

    result: dict[str, dict] = {}
    for cid, group in df.groupby("customer_id"):
        shared = {}
        for field in shared_fields:
            vals = group[field].dropna().tolist()
            if field == "l3_style_lineage":
                # style_lineage is an array field — flatten
                flat = []
                for v in group["l3_style_lineage"].dropna():
                    flat.extend(_parse_list_field(v))
                vals = flat
            elif field == "l3_visual_weight":
                # Numeric — compute mean + std
                nums = [x for x in vals if isinstance(x, (int, float))]
                shared[field] = {
                    "mean": round(np.mean(nums), 2) if nums else None,
                    "std": round(np.std(nums), 2) if nums else None,
                }
                continue
            shared[field] = _count_distribution(vals)

        # Add style_lineage
        flat_lineage = []
        for v in group["l3_style_lineage"].dropna():
            flat_lineage.extend(_parse_list_field(v))
        shared["l3_style_lineage"] = _count_distribution(flat_lineage)

        # Category-specific distributions
        by_category: dict[str, dict] = {}
        for cat, cat_group in group.groupby("super_category"):
            cat_dist: dict[str, dict] = {"n": len(cat_group)}
            for field in cat_fields:
                vals = cat_group[field].dropna().tolist()
                cat_dist[field] = _count_distribution(vals)
            by_category[str(cat)] = cat_dist

        result[str(cid)] = {"shared": shared, "by_category": by_category}

    return result


def _count_distribution(values: list, top_n: int = 10) -> dict[str, float]:
    """Count values and return normalized distribution dict."""
    if not values:
        return {}
    counter = Counter(str(v) for v in values if v is not None and str(v) != "None")
    total = sum(counter.values())
    if total == 0:
        return {}
    return {
        k: round(v / total, 3)
        for k, v in counter.most_common(top_n)
    }


# ---------------------------------------------------------------------------
# Sparse User Fallback (1-4 purchases, no LLM)
# ---------------------------------------------------------------------------


def build_sparse_user_profiles(
    txn_path: Path,
    fk_path: Path,
    sparse_customer_ids: list[str],
) -> pd.DataFrame:
    """Build template-based profiles for sparse users (1-4 purchases).

    No LLM call needed. Aggregates purchased item attributes and generates
    a templated reasoning_text.

    Returns:
        DataFrame with columns: customer_id, n_purchases, reasoning_text, reasoning_json, profile_source
    """
    if not sparse_customer_ids:
        return pd.DataFrame(columns=[
            "customer_id", "n_purchases", "reasoning_text", "reasoning_json", "profile_source",
        ])

    con = duckdb.connect()
    con.execute(f"CREATE VIEW txn AS SELECT * FROM parquet_scan('{txn_path}')")
    con.execute(f"CREATE VIEW fk AS SELECT * FROM parquet_scan('{fk_path}')")

    con.execute("CREATE TEMP TABLE sparse_users (customer_id VARCHAR)")
    con.executemany(
        "INSERT INTO sparse_users VALUES (?)",
        [(cid,) for cid in sparse_customer_ids],
    )

    query = """
    SELECT
        t.customer_id,
        t.article_id,
        t.price,
        t.sales_channel_id,
        fk.l2_style_mood,
        fk.l2_occasion,
        fk.l2_perceived_quality,
        fk.l2_trendiness,
        fk.l2_season_fit,
        fk.l3_color_harmony,
        fk.l3_tone_season,
        fk.l3_coordination_role,
        fk.l3_visual_weight,
        fk.super_category,
        fk.l3_slot6,
        fk.l3_slot7
    FROM txn t
    INNER JOIN sparse_users su ON t.customer_id = su.customer_id
    LEFT JOIN fk ON t.article_id = fk.article_id
    ORDER BY t.customer_id, t.t_dat DESC
    """
    df = con.execute(query).fetchdf()
    con.close()

    rows = []
    for cid, group in df.groupby("customer_id"):
        profile = _build_single_sparse_profile(str(cid), group)
        rows.append(profile)

    # Handle users with no transactions found
    found_ids = {r["customer_id"] for r in rows}
    for cid in sparse_customer_ids:
        if cid not in found_ids:
            rows.append({
                "customer_id": cid,
                "n_purchases": 0,
                "reasoning_text": "(a) Style mood: Unknown. (b) Occasion: Unknown. "
                                  "(c) Quality-price: Unknown. (d) Trend: Unknown. "
                                  "(e) Season: Unknown. (f) Form: Unknown. "
                                  "(g) Color: Unknown. (h) Coordination: Unknown. "
                                  "(i) Identity: New user with no purchase history.",
                "reasoning_json": "{}",
                "profile_source": "template",
            })

    return pd.DataFrame(rows)


def _build_single_sparse_profile(customer_id: str, group: pd.DataFrame) -> dict:
    """Build a single sparse user profile from purchased item attributes."""
    n = len(group)

    # Aggregate L2 attributes
    style_moods = _flatten_list_column(group, "l2_style_mood")
    occasions = _flatten_list_column(group, "l2_occasion")
    qualities = group["l2_perceived_quality"].dropna().tolist()
    trendiness = group["l2_trendiness"].dropna().tolist()
    seasons = group["l2_season_fit"].dropna().tolist()

    # Aggregate L3 attributes
    harmonies = group["l3_color_harmony"].dropna().tolist()
    tones = group["l3_tone_season"].dropna().tolist()
    coord_roles = group["l3_coordination_role"].dropna().tolist()
    weights = group["l3_visual_weight"].dropna().tolist()

    # Build reasoning JSON from aggregated attributes
    reasoning_json = {
        "style_mood_preference": _top_values(style_moods, 3),
        "occasion_preference": _top_values(occasions, 3),
        "quality_price_tendency": _describe_quality(qualities),
        "trend_sensitivity": _top_values(trendiness, 1),
        "seasonal_pattern": _top_values(seasons, 2),
        "form_preference": _describe_form(group),
        "color_tendency": _describe_color(harmonies, tones),
        "coordination_tendency": _top_values(coord_roles, 2),
        "identity_summary": f"User with {n} purchase(s), limited history for deep analysis.",
    }

    reasoning_text = compose_sparse_reasoning_text(reasoning_json)

    return {
        "customer_id": customer_id,
        "n_purchases": n,
        "reasoning_text": reasoning_text,
        "reasoning_json": json.dumps(reasoning_json),
        "profile_source": "template",
    }


def compose_sparse_reasoning_text(reasoning_json: dict) -> str:
    """Convert sparse profile JSON to structured reasoning text.

    Format matches the LLM output format: (a)~(i) fields.
    """
    parts = [
        f"(a) Style mood: {reasoning_json.get('style_mood_preference', 'Unknown')}.",
        f"(b) Occasion: {reasoning_json.get('occasion_preference', 'Unknown')}.",
        f"(c) Quality-price: {reasoning_json.get('quality_price_tendency', 'Unknown')}.",
        f"(d) Trend: {reasoning_json.get('trend_sensitivity', 'Unknown')}.",
        f"(e) Season: {reasoning_json.get('seasonal_pattern', 'Unknown')}.",
        f"(f) Form: {reasoning_json.get('form_preference', 'Unknown')}.",
        f"(g) Color: {reasoning_json.get('color_tendency', 'Unknown')}.",
        f"(h) Coordination: {reasoning_json.get('coordination_tendency', 'Unknown')}.",
        f"(i) Identity: {reasoning_json.get('identity_summary', 'Unknown')}.",
    ]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Sparse profile helper functions
# ---------------------------------------------------------------------------


def _flatten_list_column(df: pd.DataFrame, col: str) -> list[str]:
    """Flatten a column containing list values into a single flat list."""
    result: list[str] = []
    for val in df[col].dropna():
        result.extend(_parse_list_field(val))
    return result


def _top_values(values: list, top_n: int) -> str:
    """Get top N most common values as comma-separated string."""
    if not values:
        return "Unknown"
    counter = Counter(str(v) for v in values if v is not None)
    if not counter:
        return "Unknown"
    return ", ".join(v for v, _ in counter.most_common(top_n))


def _describe_quality(qualities: list) -> str:
    """Describe quality-price tendency from perceived_quality values."""
    if not qualities:
        return "Unknown"
    nums = [float(q) for q in qualities if q is not None]
    if not nums:
        return "Unknown"
    avg = sum(nums) / len(nums)
    if avg <= 2:
        return f"Budget-conscious (avg {avg:.1f}/5)"
    elif avg <= 3.5:
        return f"Mid-range (avg {avg:.1f}/5)"
    else:
        return f"Quality-oriented (avg {avg:.1f}/5)"


def _describe_form(group: pd.DataFrame) -> str:
    """Describe form preference from category-specific L3 slots."""
    slot6_vals = group["l3_slot6"].dropna().tolist()
    if not slot6_vals:
        return "Insufficient data"
    return _top_values(slot6_vals, 2)


def _describe_color(harmonies: list, tones: list) -> str:
    """Describe color tendency from harmony and tone distributions."""
    parts = []
    if harmonies:
        parts.append(_top_values(harmonies, 2))
    if tones:
        parts.append(_top_values(tones, 2))
    return "; ".join(parts) if parts else "Unknown"
