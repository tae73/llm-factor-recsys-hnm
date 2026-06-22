"""Enrichment v2 — matrix-ready table for the 4-use value matrix (E2-2).

Persists, per ``article_id``, the four DE1-v2-passing decision-axes plus a per-item
sell-through proxy, so the value-matrix probe can read them directly:

  * ``e2_trend_phase_actual`` / ``e2_outfit_role`` — behavior-derived (full catalog).
  * ``e2_value_gap`` / ``e2_trend_gap`` — perception×behavior gaps. These were only
    computed transiently inside ``witnesses/probe_DE1_v2_new_attributes.py``; we promote
    that logic here (reusing the rank maps from ``enrichment_v2.schema``) so they are a
    durable per-article column. Gaps exist only on the ~5.3K pilot subset and exclude
    items whose ``e2_trend_phase_actual == 'Insufficient'`` (no momentum rank).
  * sell-through proxy — ``velocity`` (= total_purchases / lifespan_days),
    ``buyer_concentration``, ``n_buyers`` — derived from transactions (③ merch cell).

No API, CPU/DuckDB only. Output: ``data/knowledge/enrichment_v2/matrix_axes.parquet``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.knowledge.enrichment_v2.schema import (
    PRICE_TIER_RANK,
    TREND_LOOK_RANK,
    TREND_PHASE_RANK,
)

logger = logging.getLogger(__name__)

OUTPUT_FILENAME = "matrix_axes.parquet"
PASSING_AXES = [
    "e2_trend_phase_actual",
    "e2_outfit_role",
    "e2_value_gap",
    "e2_trend_gap",
]


# ---------------------------------------------------------------------------
# Gap axes (promoted from probe_DE1_v2_new_attributes.py:_assemble)
# ---------------------------------------------------------------------------
def compute_value_gap(merged: pd.DataFrame) -> pd.Series:
    """value_gap = e2_price_look (LLM 1-5) − e2_price_tier_actual rank (T1..T5 → 1..5)."""
    price_rank = merged["e2_price_tier_actual"].map(PRICE_TIER_RANK)
    return (merged["e2_price_look"] - price_rank).astype("Int64")


def compute_trend_gap(merged: pd.DataFrame) -> pd.Series:
    """trend_gap = trend_look rank − trend_phase rank (Insufficient phase → NA)."""
    look_rank = merged["e2_trend_look"].map(TREND_LOOK_RANK)
    phase_rank = merged["e2_trend_phase_actual"].map(TREND_PHASE_RANK)
    return (look_rank - phase_rank).astype("Int64")


# ---------------------------------------------------------------------------
# Per-item sell-through proxy (③ merchandising)
# ---------------------------------------------------------------------------
def compute_sell_through(con: duckdb.DuckDBPyConnection, train_path: Path) -> pd.DataFrame:
    """Per-article sell-through proxy from train transactions.

    velocity = total_purchases / max(lifespan_days, 1); buyer_concentration =
    total_purchases / n_buyers (>=1 = repeat-buying). No inventory data exists, so
    velocity is the available demand-met proxy.
    """
    df = con.execute(f"""
        SELECT CAST(article_id AS VARCHAR)          AS article_id,
               COUNT(*)                             AS total_purchases,
               COUNT(DISTINCT customer_id)          AS n_buyers,
               date_diff('day', MIN(t_dat), MAX(t_dat)) AS lifespan_days
        FROM read_parquet('{train_path}')
        GROUP BY article_id
        """).fetchdf()
    df["velocity"] = df["total_purchases"] / np.maximum(df["lifespan_days"], 1)
    df["buyer_concentration"] = df["total_purchases"] / df["n_buyers"].clip(lower=1)
    logger.info("sell_through: %d articles", len(df))
    return df


def compute_merch_signals(con: duckdb.DuckDBPyConnection, train_path: Path) -> pd.DataFrame:
    """Richer per-article merchandising signals (E2-3 ③):

    - ``markdown_depth`` = (max_price − min_price) / max_price — clearance/liquidation proxy.
    - ``first_week_sell_through`` = sales in the first 7 days since the item's first sale / total.
    - ``online_ratio`` = fraction of purchases via online (sales_channel_id=2).
    """
    df = con.execute(f"""
        WITH base AS (
            SELECT CAST(article_id AS VARCHAR)                  AS article_id,
                   MIN(t_dat)                                   AS first_dt,
                   MAX(price)                                   AS max_price,
                   MIN(price)                                   AS min_price,
                   COUNT(*)                                     AS n,
                   SUM(CASE WHEN sales_channel_id = 2 THEN 1 ELSE 0 END) AS n_online
            FROM read_parquet('{train_path}')
            GROUP BY article_id
        ),
        fw AS (
            SELECT CAST(t.article_id AS VARCHAR) AS article_id,
                   SUM(CASE WHEN date_diff('day', b.first_dt, t.t_dat) <= 6 THEN 1 ELSE 0 END) AS n_first_week
            FROM read_parquet('{train_path}') t
            JOIN base b ON CAST(t.article_id AS VARCHAR) = b.article_id
            GROUP BY t.article_id
        )
        SELECT b.article_id,
               (b.max_price - b.min_price) / NULLIF(b.max_price, 0) AS markdown_depth,
               CAST(fw.n_first_week AS DOUBLE) / b.n               AS first_week_sell_through,
               CAST(b.n_online AS DOUBLE) / b.n                    AS online_ratio
        FROM base b
        JOIN fw USING (article_id)
        """).fetchdf()
    logger.info("merch_signals: %d articles", len(df))
    return df


# ---------------------------------------------------------------------------
# Per-article held-out FUTURE outcomes (E2-5 gap-axis decision-lift)
# ---------------------------------------------------------------------------
# Val window = 2020-07-01..2020-08-31 (≈2 months). For a comparable momentum-change
# baseline we use the LAST 2 months of train (2020-05-01..2020-06-30) so both spans are
# equal-length → the change is not driven by window-length mismatch. Train-frozen: every
# reference (mean price, trailing volume) is computed on TRAIN only; nothing on val
# normalizes itself (mirrors user_axes.build_future_outcomes honesty rule).
_TRAIN_TRAILING_START = "2020-05-01"


def build_article_future_outcomes(article_ids: Sequence[str], data_dir: Path) -> pd.DataFrame:
    """Per-article held-out FUTURE (val 2020-07+) decision outcomes, train-frozen refs.

    Item-side analogue of ``user_axes.build_future_outcomes``. Reuses
    :func:`compute_sell_through` / :func:`compute_merch_signals` on the VAL window for
    future velocity / markdown, and computes a train-frozen reference price + a comparable
    2-month trailing volume so the outcomes never normalize themselves on val.

    Columns (left-joined onto ``article_ids`` → canonical order; absent items → NaN/0):
      * ``fut_price_drop``      = log(mean val price) − log(mean train price)  (mispricing)
      * ``fut_markdown_depth``  = (max−min)/max of val price                  (clearance)
      * ``fut_velocity``        = val purchases / val lifespan-days            (sell-through)
      * ``fut_first_week_st``   = val first-7-day share                       (launch demand)
      * ``fut_momentum_change`` = log1p(val 2-mo vol) − log1p(train last-2-mo vol)
      * ``fut_val_n``           = val purchase count                          (cohort support)
      * ``has_val_sale``        = 1 if any val purchase else 0                (survival)

    Args:
        article_ids: articles to score (order preserved; duplicates kept).
        data_dir: directory holding ``train_transactions.parquet`` + ``val_transactions.parquet``.
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train_transactions.parquet"
    val_path = data_dir / "val_transactions.parquet"
    ids = [str(a) for a in article_ids]

    con = duckdb.connect()
    try:
        # Single-thread aggregation → deterministic AVG summation order (parallel reduction
        # drifts at ~1e-15, which can flip a rank-bucket tie at the margin). REPRO guard.
        con.execute("PRAGMA threads=1")
        val_sell = compute_sell_through(con, val_path)
        val_merch = compute_merch_signals(con, val_path)
        train_ref = con.execute(f"""
            SELECT CAST(article_id AS VARCHAR)                                          AS article_id,
                   AVG(price)                                                           AS train_mean_price,
                   SUM(CASE WHEN t_dat >= DATE '{_TRAIN_TRAILING_START}' THEN 1 ELSE 0 END) AS train_trail_vol
            FROM read_parquet('{train_path}')
            GROUP BY article_id
            """).fetchdf()
        val_ref = con.execute(f"""
            SELECT CAST(article_id AS VARCHAR) AS article_id,
                   AVG(price)                  AS val_mean_price,
                   COUNT(*)                    AS val_vol
            FROM read_parquet('{val_path}')
            GROUP BY article_id
            """).fetchdf()
    finally:
        con.close()

    out = (
        pd.DataFrame({"article_id": ids})
        .merge(
            val_sell[["article_id", "velocity"]].rename(columns={"velocity": "fut_velocity"}),
            on="article_id",
            how="left",
        )
        .merge(
            val_merch[["article_id", "markdown_depth", "first_week_sell_through"]].rename(
                columns={
                    "markdown_depth": "fut_markdown_depth",
                    "first_week_sell_through": "fut_first_week_st",
                }
            ),
            on="article_id",
            how="left",
        )
        .merge(train_ref, on="article_id", how="left")
        .merge(val_ref, on="article_id", how="left")
    )
    out["fut_price_drop"] = np.log(out["val_mean_price"]) - np.log(out["train_mean_price"])
    out["fut_momentum_change"] = np.log1p(out["val_vol"].fillna(0.0)) - np.log1p(
        out["train_trail_vol"].fillna(0.0)
    )
    out["fut_val_n"] = out["val_vol"].fillna(0).astype(int)
    out["has_val_sale"] = (out["fut_val_n"] > 0).astype(int)
    cols = [
        "article_id",
        "fut_price_drop",
        "fut_markdown_depth",
        "fut_velocity",
        "fut_first_week_st",
        "fut_momentum_change",
        "fut_val_n",
        "has_val_sale",
    ]
    out = out[cols]
    logger.info(
        "article_future_outcomes: %d ids (%d with val sale)",
        len(out),
        int(out["has_val_sale"].sum()),
    )
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_matrix_table(data_dir: Path, e2_dir: Path) -> Path:
    """Assemble the matrix-ready table → ``matrix_axes.parquet`` (keyed by article_id).

    Args:
        data_dir: directory with ``train_transactions.parquet``.
        e2_dir: directory with ``behavioral_axes.parquet`` + ``enrichment_v2_llm.parquet``.
    """
    train_path = data_dir / "train_transactions.parquet"
    bh = pd.read_parquet(e2_dir / "behavioral_axes.parquet")
    bh["article_id"] = bh["article_id"].astype(str)
    llm = pd.read_parquet(e2_dir / "enrichment_v2_llm.parquet")
    llm["article_id"] = llm["article_id"].astype(str)

    # Base = full catalog behavioral axes; left-join the pilot LLM perception fields.
    merged = bh.merge(
        llm[["article_id", "e2_price_look", "e2_trend_look"]], on="article_id", how="left"
    )
    merged["e2_value_gap"] = compute_value_gap(merged)
    merged["e2_trend_gap"] = compute_trend_gap(merged)

    con = duckdb.connect()
    try:
        sell = compute_sell_through(con, train_path)
        merch = compute_merch_signals(con, train_path)
    finally:
        con.close()

    out = merged.merge(sell, on="article_id", how="left").merge(merch, on="article_id", how="left")
    keep = [
        "article_id",
        "e2_trend_phase_actual",
        "e2_outfit_role",
        "e2_value_gap",
        "e2_trend_gap",
        "e2_trend_momentum",
        "e2_price_tier_actual",
        "velocity",
        "total_purchases",
        "lifespan_days",
        "n_buyers",
        "buyer_concentration",
        "markdown_depth",
        "first_week_sell_through",
        "online_ratio",
    ]
    out = out[[c for c in keep if c in out.columns]]

    e2_dir.mkdir(parents=True, exist_ok=True)
    output_path = e2_dir / OUTPUT_FILENAME
    out.to_parquet(output_path, index=False)
    n_gap = int(out["e2_value_gap"].notna().sum())
    logger.info("matrix_axes: %d articles (%d with gap axes) → %s", len(out), n_gap, output_path)
    return output_path
