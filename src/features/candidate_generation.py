"""Multi-source candidate generation for 2-stage recommendation.

Generates candidate item pools from multiple sources:
- Repurchase: items the user previously bought (+ same product_code SKU variants)
- Age-group popularity: top items for user's age group
- Recency: recently popular items (time-windowed)

Each source returns the same format as extract_stage1_candidates():
    {"user_indices": (N,), "candidate_indices": (N, K), "candidate_scores": (N, K)}

blend_candidates() merges multiple sources into a unified top-K pool.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np


def extract_repurchase_candidates(
    con: duckdb.DuckDBPyConnection,
    train_path: Path,
    articles_path: Path,
    target_user_ids: list[str],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    top_k: int = 50,
) -> dict[str, np.ndarray]:
    """Previously purchased items + same product_code SKU variants, recency-weighted."""
    # Get user's purchase history with product_code for SKU expansion
    df = con.execute(f"""
        WITH user_purchases AS (
            SELECT
                t.customer_id,
                t.article_id,
                a.product_code,
                MAX(t.t_dat) AS last_date,
                COUNT(*) AS purchase_count
            FROM read_parquet('{train_path}') t
            JOIN read_parquet('{articles_path}') a ON t.article_id = a.article_id
            WHERE t.customer_id IN (SELECT UNNEST(?::VARCHAR[]))
            GROUP BY t.customer_id, t.article_id, a.product_code
        ),
        sku_expansion AS (
            -- Add other SKUs from the same product_code
            SELECT
                up.customer_id,
                a2.article_id,
                up.last_date,
                up.purchase_count,
                CASE WHEN up.article_id = a2.article_id THEN 1.0 ELSE 0.5 END AS score_mult
            FROM user_purchases up
            JOIN read_parquet('{articles_path}') a2 ON up.product_code = a2.product_code
        )
        SELECT
            customer_id,
            article_id,
            MAX(last_date) AS last_date,
            MAX(purchase_count * score_mult) AS score
        FROM sku_expansion
        GROUP BY customer_id, article_id
        ORDER BY customer_id, score DESC, last_date DESC
    """, [target_user_ids]).fetchdf()

    return _df_to_candidate_arrays(df, target_user_ids, user_to_idx, item_to_idx, top_k)


def extract_age_popularity_candidates(
    con: duckdb.DuckDBPyConnection,
    train_path: Path,
    customers_path: Path,
    target_user_ids: list[str],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    top_k: int = 50,
) -> dict[str, np.ndarray]:
    """Top items purchased by user's age group."""
    df = con.execute(f"""
        WITH user_age AS (
            SELECT customer_id,
                CASE
                    WHEN age IS NULL THEN 'unknown'
                    WHEN age < 20 THEN 'teen'
                    WHEN age < 30 THEN '20s'
                    WHEN age < 40 THEN '30s'
                    WHEN age < 50 THEN '40s'
                    WHEN age < 60 THEN '50s'
                    ELSE '60plus'
                END AS age_group
            FROM read_parquet('{customers_path}')
        ),
        age_popularity AS (
            SELECT
                ua.age_group,
                t.article_id,
                COUNT(*) AS cnt
            FROM read_parquet('{train_path}') t
            JOIN user_age ua ON t.customer_id = ua.customer_id
            GROUP BY ua.age_group, t.article_id
        ),
        ranked AS (
            SELECT age_group, article_id, cnt,
                ROW_NUMBER() OVER (PARTITION BY age_group ORDER BY cnt DESC) AS rn
            FROM age_popularity
        )
        SELECT
            ua2.customer_id,
            r.article_id,
            r.cnt AS score
        FROM user_age ua2
        JOIN ranked r ON ua2.age_group = r.age_group AND r.rn <= {top_k}
        WHERE ua2.customer_id IN (SELECT UNNEST(?::VARCHAR[]))
        ORDER BY ua2.customer_id, r.cnt DESC
    """, [target_user_ids]).fetchdf()

    return _df_to_candidate_arrays(df, target_user_ids, user_to_idx, item_to_idx, top_k)


def extract_recency_candidates(
    con: duckdb.DuckDBPyConnection,
    train_path: Path,
    target_user_ids: list[str],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    window_days: int = 14,
    top_k: int = 50,
) -> dict[str, np.ndarray]:
    """Recently popular items (same for all users)."""
    df_items = con.execute(f"""
        WITH max_date AS (SELECT MAX(t_dat) AS md FROM read_parquet('{train_path}'))
        SELECT article_id, COUNT(*) AS score
        FROM read_parquet('{train_path}') t, max_date m
        WHERE t.t_dat >= m.md - INTERVAL '{window_days}' DAY
        GROUP BY article_id
        ORDER BY score DESC
        LIMIT {top_k}
    """).fetchdf()

    if df_items.empty:
        return _empty_candidates(target_user_ids, user_to_idx, top_k)

    # Same candidates for all users
    item_ids = df_items["article_id"].astype(str).tolist()
    scores = df_items["score"].values.astype(np.float32)

    candidate_idx = np.array(
        [item_to_idx.get(aid, 0) for aid in item_ids], dtype=np.int32
    )
    n_cands = len(candidate_idx)

    user_indices = []
    all_cand_indices = []
    all_cand_scores = []

    for uid in target_user_ids:
        if uid not in user_to_idx:
            continue
        user_indices.append(user_to_idx[uid])
        padded_idx = np.zeros(top_k, dtype=np.int32)
        padded_scores = np.zeros(top_k, dtype=np.float32)
        n = min(n_cands, top_k)
        padded_idx[:n] = candidate_idx[:n]
        padded_scores[:n] = scores[:n]
        all_cand_indices.append(padded_idx)
        all_cand_scores.append(padded_scores)

    return {
        "user_indices": np.array(user_indices, dtype=np.int32),
        "candidate_indices": np.stack(all_cand_indices),
        "candidate_scores": np.stack(all_cand_scores),
    }


def blend_candidates(
    sources: list[dict[str, np.ndarray]],
    top_k: int = 100,
) -> dict[str, np.ndarray]:
    """Merge multiple candidate sources into unified top-K pool.

    For each user, union all candidate items across sources,
    keep max score per item, select top-K by score.
    """
    if not sources:
        raise ValueError("At least one candidate source required")
    if len(sources) == 1:
        return sources[0]

    # Build user_idx → source mapping
    all_user_idxs = set()
    for src in sources:
        all_user_idxs.update(src["user_indices"].tolist())
    all_user_idxs = sorted(all_user_idxs)

    # For each user, merge candidates
    result_user_indices = []
    result_cand_indices = []
    result_cand_scores = []

    # Build lookup: user_idx → row in each source
    src_lookups = []
    for src in sources:
        lookup = {}
        for row_idx, uid in enumerate(src["user_indices"]):
            lookup[int(uid)] = row_idx
        src_lookups.append(lookup)

    for user_idx in all_user_idxs:
        item_scores: dict[int, float] = {}

        for src, lookup in zip(sources, src_lookups):
            if user_idx not in lookup:
                continue
            row = lookup[user_idx]
            cand_idx = src["candidate_indices"][row]
            cand_scores = src["candidate_scores"][row]
            for item_idx, score in zip(cand_idx.tolist(), cand_scores.tolist()):
                if item_idx == 0 and score == 0.0:
                    continue  # skip padding
                if item_idx not in item_scores or score > item_scores[item_idx]:
                    item_scores[item_idx] = score

        if not item_scores:
            continue

        # Sort by score descending, take top-K
        sorted_items = sorted(item_scores.items(), key=lambda x: -x[1])[:top_k]
        padded_idx = np.zeros(top_k, dtype=np.int32)
        padded_scores = np.zeros(top_k, dtype=np.float32)
        for i, (idx, sc) in enumerate(sorted_items):
            padded_idx[i] = idx
            padded_scores[i] = sc

        result_user_indices.append(user_idx)
        result_cand_indices.append(padded_idx)
        result_cand_scores.append(padded_scores)

    return {
        "user_indices": np.array(result_user_indices, dtype=np.int32),
        "candidate_indices": np.stack(result_cand_indices),
        "candidate_scores": np.stack(result_cand_scores),
    }


def build_interaction_data(
    con: duckdb.DuckDBPyConnection,
    train_path: Path,
    articles_path: Path,
    ref_date: str = "2020-06-30",
) -> dict[str, dict]:
    """Build user-item interaction lookup for ReRanker features.

    Returns:
        {customer_id: {
            "items": {article_id: {"count": int, "last_days": int}},
            "categories": {product_type_name: int},
        }}
    """
    # Per-user item purchases
    item_df = con.execute(f"""
        SELECT
            customer_id,
            article_id,
            COUNT(*) AS cnt,
            DATE_DIFF('day', MAX(t_dat), '{ref_date}'::DATE) AS days_since
        FROM read_parquet('{train_path}')
        GROUP BY customer_id, article_id
    """).fetchdf()

    # Per-user category purchases
    cat_df = con.execute(f"""
        SELECT
            t.customer_id,
            a.product_type_name,
            COUNT(*) AS cnt
        FROM read_parquet('{train_path}') t
        JOIN read_parquet('{articles_path}') a ON t.article_id = a.article_id
        GROUP BY t.customer_id, a.product_type_name
    """).fetchdf()

    interactions: dict[str, dict] = {}

    for _, row in item_df.iterrows():
        uid = str(row["customer_id"])
        aid = str(row["article_id"])
        if uid not in interactions:
            interactions[uid] = {"items": {}, "categories": {}}
        interactions[uid]["items"][aid] = {
            "count": int(row["cnt"]),
            "last_days": int(row["days_since"]),
        }

    for _, row in cat_df.iterrows():
        uid = str(row["customer_id"])
        cat = str(row["product_type_name"])
        if uid not in interactions:
            interactions[uid] = {"items": {}, "categories": {}}
        interactions[uid]["categories"][cat] = int(row["cnt"])

    return interactions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _df_to_candidate_arrays(
    df,
    target_user_ids: list[str],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    top_k: int,
) -> dict[str, np.ndarray]:
    """Convert DuckDB result DataFrame to candidate arrays."""
    if df.empty:
        return _empty_candidates(target_user_ids, user_to_idx, top_k)

    user_indices = []
    cand_indices = []
    cand_scores = []

    for uid in target_user_ids:
        if uid not in user_to_idx:
            continue
        user_rows = df[df["customer_id"].astype(str) == uid]
        if user_rows.empty:
            user_indices.append(user_to_idx[uid])
            cand_indices.append(np.zeros(top_k, dtype=np.int32))
            cand_scores.append(np.zeros(top_k, dtype=np.float32))
            continue

        items = user_rows["article_id"].astype(str).tolist()[:top_k]
        scores = user_rows["score"].values[:top_k].astype(np.float32)

        padded_idx = np.zeros(top_k, dtype=np.int32)
        padded_scores = np.zeros(top_k, dtype=np.float32)
        for i, (aid, sc) in enumerate(zip(items, scores)):
            padded_idx[i] = item_to_idx.get(aid, 0)
            padded_scores[i] = sc

        user_indices.append(user_to_idx[uid])
        cand_indices.append(padded_idx)
        cand_scores.append(padded_scores)

    return {
        "user_indices": np.array(user_indices, dtype=np.int32),
        "candidate_indices": np.stack(cand_indices),
        "candidate_scores": np.stack(cand_scores),
    }


def _empty_candidates(
    target_user_ids: list[str],
    user_to_idx: dict[str, int],
    top_k: int,
) -> dict[str, np.ndarray]:
    """Return empty candidate arrays."""
    valid = [user_to_idx[uid] for uid in target_user_ids if uid in user_to_idx]
    n = len(valid)
    return {
        "user_indices": np.array(valid, dtype=np.int32),
        "candidate_indices": np.zeros((n, top_k), dtype=np.int32),
        "candidate_scores": np.zeros((n, top_k), dtype=np.float32),
    }
