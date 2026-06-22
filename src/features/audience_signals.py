"""Buyer-population signals for the marketing-audience cell (E2-3 ④).

E2-2's ④ measured per-item *repurchase* divergence — the wrong target. "Marketing
audience" means *who buys*: segment items by an axis and ask whether the BUYER
populations (age, channel) differ across segments more than a metadata partition does.

This module computes **per-item buyer aggregates** from the transaction↔customer join
(age sums, online sums, active sums, counts). Aggregates are per-item so the probe can
(a) re-aggregate to any segmentation (axis or metadata k-means) and (b) run a permutation
null by shuffling per-item segment labels — both cheaply, without holding 28M rows.

DuckDB/CPU only, no API.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_buyer_population(con, train_path: Path, customers_path: Path) -> pd.DataFrame:
    """Per-article buyer aggregates from train txn ⋈ customers.

    Returns DataFrame[article_id, n_txn, sum_age, sum_age_sq, n_age, n_online, n_active].
    age-known counts (``n_age``) are tracked separately so age means/std are over the
    subset with non-null age (98.8% of customers).
    """
    df = con.execute(f"""
        SELECT CAST(t.article_id AS VARCHAR)                          AS article_id,
               COUNT(*)                                              AS n_txn,
               SUM(CASE WHEN c.age IS NOT NULL THEN c.age ELSE 0 END)        AS sum_age,
               SUM(CASE WHEN c.age IS NOT NULL THEN c.age*c.age ELSE 0 END)  AS sum_age_sq,
               SUM(CASE WHEN c.age IS NOT NULL THEN 1 ELSE 0 END)            AS n_age,
               SUM(CASE WHEN t.sales_channel_id = 2 THEN 1 ELSE 0 END)       AS n_online,
               SUM(CASE WHEN c."Active" = 1 THEN 1 ELSE 0 END)               AS n_active
        FROM read_parquet('{train_path}') t
        JOIN read_parquet('{customers_path}') c ON t.customer_id = c.customer_id
        GROUP BY t.article_id
        """).fetchdf()
    df["article_id"] = df["article_id"].astype(str)
    logger.info("buyer_population: %d articles", len(df))
    return df


def grand_std(sum_y: np.ndarray, sum_y_sq: np.ndarray, n_y: np.ndarray) -> float:
    """Transaction-grain std of a buyer KPI from per-item sums (for the practical margin)."""
    n = float(n_y.sum())
    if n <= 1:
        return 0.0
    mean = sum_y.sum() / n
    var = max(0.0, sum_y_sq.sum() / n - mean * mean)
    return float(np.sqrt(var))


def segment_divergence_weighted(labels: np.ndarray, sum_y: np.ndarray, n_y: np.ndarray) -> float:
    """Size(n_y)-weighted std of per-segment means, where segment_mean = Σsum_y / Σn_y.

    Mirrors ``_segment_divergence`` but on per-item aggregates (so it weights by buyer
    exposure, not by item count). Used for both the axis partition and the metadata
    k-means baseline on the SAME buyer KPI.
    """
    df = pd.DataFrame({"lab": labels, "sum_y": sum_y, "n_y": n_y})
    g = df.groupby("lab", sort=False).agg(sum_y=("sum_y", "sum"), n_y=("n_y", "sum"))
    g = g[g["n_y"] > 0]
    if len(g) < 2:
        return 0.0
    means = (g["sum_y"] / g["n_y"]).to_numpy()
    w = (g["n_y"] / g["n_y"].sum()).to_numpy()
    grand = float((w * means).sum())
    return float(np.sqrt((w * (means - grand) ** 2).sum()))
