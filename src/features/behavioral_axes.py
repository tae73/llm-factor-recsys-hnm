"""Enrichment v2 — behavior-derived decision-axes (no LLM, no images).

Three of the six v2 axes are computed from transactions, because the DE1 lesson is
that the LLM cannot OBSERVE these — it can only guess, and a guess recodes metadata
(the failed LLM ``l3_coordination_role`` Foundation/Accent was 67% concentrated). The
data-grounded versions are discriminable by construction:

  * ``e2_price_tier_actual``  — within-product_group quintile of actual mean price.
  * ``e2_trend_phase_actual`` — sales-momentum life-cycle phase (recent vs prior volume),
                                assigned by terciles so no bucket dominates.
  * ``e2_outfit_role``        — co-purchase graph role from same-basket cross-group
                                pairs (intensity × partner-diversity), RESIDUALIZED vs
                                product_group so it captures within-category structure
                                metadata cannot.

All three are computed over the FULL catalog (CPU/DuckDB) so the DE1 re-screen has
them at STRONG statistical power even when only ~500 codes get LLM extraction. Output:
``behavioral_axes.parquet`` keyed by ``article_id`` (str).
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_FILENAME = "behavioral_axes.parquet"

# Global train window months used for trend momentum (filled at runtime from data).
_RECENT_WINDOW = 4  # last N calendar months = "recent"
_PRIOR_WINDOW = 4  # the N months before that = "prior"
_MIN_PURCHASES_TREND = 10  # below => Insufficient (matches the pilot sampling floor)
_MIN_COPURCHASE_EVENTS = 5  # below => Standalone (rarely co-bought cross-category)


# ---------------------------------------------------------------------------
# Axis 2 — price-tier (within-product_group quintile of actual mean price)
# ---------------------------------------------------------------------------
def compute_price_tier(
    con: duckdb.DuckDBPyConnection, train_path: Path, articles_path: Path
) -> pd.DataFrame:
    """Per-article mean price → quintile WITHIN ``product_group_name`` (T1..T5).

    Within-group residualizes the cross-category price signal that metadata
    (product_type/section) already encodes, maximizing NON-REDUNDANCY.
    """
    df = con.execute(f"""
        WITH p AS (
            SELECT t.article_id AS article_id,
                   AVG(t.price)  AS avg_price,
                   a.product_group_name AS pg
            FROM read_parquet('{train_path}') t
            JOIN read_parquet('{articles_path}') a USING (article_id)
            GROUP BY t.article_id, a.product_group_name
        )
        SELECT CAST(article_id AS VARCHAR) AS article_id,
               avg_price AS e2_avg_price,
               'T' || CAST(NTILE(5) OVER (PARTITION BY pg ORDER BY avg_price) AS VARCHAR)
                   AS e2_price_tier_actual
        FROM p
        """).fetchdf()
    logger.info("price_tier: %d articles tiered", len(df))
    return df


# ---------------------------------------------------------------------------
# Axis 1 — trend-phase (sales-momentum life-cycle, tercile-balanced)
# ---------------------------------------------------------------------------
def compute_trend_phase(con: duckdb.DuckDBPyConnection, train_path: Path) -> pd.DataFrame:
    """Per-article life-cycle phase from monthly sales trajectory.

    Momentum = log((recent_vol+1)/(prior_vol+1)) over the last ``_RECENT_WINDOW`` vs the
    preceding ``_PRIOR_WINDOW`` calendar months of the train window. Phases are assigned
    by momentum terciles (+ recency / peak position) so the top bucket stays ≤ ~0.40.
    """
    monthly = con.execute(f"""
        SELECT CAST(article_id AS VARCHAR) AS article_id,
               date_trunc('month', t_dat)  AS mo,
               COUNT(*)                     AS v
        FROM read_parquet('{train_path}')
        GROUP BY 1, 2
        """).fetchdf()
    monthly["mo"] = pd.to_datetime(monthly["mo"])

    months = np.sort(monthly["mo"].unique())
    recent_set = set(months[-_RECENT_WINDOW:])
    prior_set = set(months[-(_RECENT_WINDOW + _PRIOR_WINDOW) : -_RECENT_WINDOW])

    monthly["is_recent"] = monthly["mo"].isin(recent_set)
    monthly["is_prior"] = monthly["mo"].isin(prior_set)
    recent_floor = months[-6] if len(months) >= 6 else months[0]  # "new" = first sale in last 6 mo

    g = monthly.groupby("article_id")
    stats = pd.DataFrame(
        {
            "total": g["v"].sum(),
            "first_month": g["mo"].min(),
            "peak_vol": g["v"].max(),
            "recent_vol": g.apply(lambda d: d.loc[d["is_recent"], "v"].sum(), include_groups=False),
            "prior_vol": g.apply(lambda d: d.loc[d["is_prior"], "v"].sum(), include_groups=False),
        }
    ).reset_index()

    stats["momentum"] = np.log1p(stats["recent_vol"]) - np.log1p(stats["prior_vol"])
    stats["is_new"] = stats["first_month"] >= recent_floor
    sufficient = stats["total"] >= _MIN_PURCHASES_TREND

    # Tercile thresholds on momentum among sufficient items (balanced buckets).
    mom = stats.loc[sufficient, "momentum"]
    lo, hi = mom.quantile([1 / 3, 2 / 3]).tolist() if len(mom) else (0.0, 0.0)

    def _phase(r: pd.Series) -> str:
        if r["total"] < _MIN_PURCHASES_TREND:
            return "Insufficient"
        if r["is_new"] and r["momentum"] > 0:
            return "Emerging"
        if r["momentum"] >= hi:
            return "Rising"
        if r["momentum"] <= lo:
            return "Declining"
        # middle tercile: at-peak vs past-peak
        return "Peak" if r["recent_vol"] >= 0.7 * r["peak_vol"] else "Mature"

    stats["e2_trend_phase_actual"] = stats.apply(_phase, axis=1)
    out = stats[["article_id", "e2_trend_phase_actual", "momentum", "total"]].rename(
        columns={"momentum": "e2_trend_momentum", "total": "e2_train_purchases"}
    )
    logger.info(
        "trend_phase: %d articles; dist=%s",
        len(out),
        out["e2_trend_phase_actual"].value_counts().to_dict(),
    )
    return out


# ---------------------------------------------------------------------------
# Axis 4 — outfit-role (co-purchase graph: intensity × diversity, residualized)
# ---------------------------------------------------------------------------
def compute_outfit_role(
    con: duckdb.DuckDBPyConnection, train_path: Path, articles_path: Path
) -> pd.DataFrame:
    """Per-article outfit role from same-basket CROSS-product_group co-purchase.

    Baskets = (customer_id, t_dat) with 2..6 items (probe_13 idiom). For each item we
    count cross-group co-purchase events, distinct cross-group partners, and distinct
    partner product_groups. Intensity (log events) and diversity (partner groups) are
    RESIDUALIZED against product_group, then thresholded into 5 roles so the role
    captures within-category hub-vs-spoke variation metadata cannot.
    """
    stats = con.execute(f"""
        WITH b AS (
            SELECT t.customer_id AS cust, t.t_dat AS dt,
                   CAST(t.article_id AS VARCHAR) AS aid,
                   a.product_group_name AS pg,
                   COUNT(*) OVER (PARTITION BY t.customer_id, t.t_dat) AS bsize
            FROM read_parquet('{train_path}') t
            JOIN read_parquet('{articles_path}') a USING (article_id)
        ),
        bf AS (SELECT * FROM b WHERE bsize BETWEEN 2 AND 6),
        pairs AS (
            SELECT x.aid AS aid, y.aid AS partner, y.pg AS partner_pg
            FROM bf x
            JOIN bf y ON x.cust = y.cust AND x.dt = y.dt
                     AND x.aid <> y.aid AND x.pg <> y.pg
        )
        SELECT aid AS article_id,
               COUNT(*)                  AS copurchase_events,
               COUNT(DISTINCT partner)   AS n_cross_partners,
               COUNT(DISTINCT partner_pg) AS n_partner_groups
        FROM pairs
        GROUP BY aid
        """).fetchdf()

    art = con.execute(
        f"SELECT CAST(article_id AS VARCHAR) AS article_id, "
        f"product_group_name AS pg FROM read_parquet('{articles_path}')"
    ).fetchdf()
    df = art.merge(stats, on="article_id", how="left")
    df["copurchase_events"] = df["copurchase_events"].fillna(0)
    df["n_partner_groups"] = df["n_partner_groups"].fillna(0)
    df["n_cross_partners"] = df["n_cross_partners"].fillna(0)

    df["log_intensity"] = np.log1p(df["copurchase_events"])
    # residualize intensity & diversity vs product_group (subtract group mean)
    df["intensity_resid"] = df["log_intensity"] - df.groupby("pg")["log_intensity"].transform(
        "mean"
    )
    df["diversity_resid"] = df["n_partner_groups"] - df.groupby("pg")["n_partner_groups"].transform(
        "mean"
    )

    active = df["copurchase_events"] >= _MIN_COPURCHASE_EVENTS
    int_med = df.loc[active, "intensity_resid"].median()
    div_med = df.loc[active, "diversity_resid"].median()

    def _role(r: pd.Series) -> str:
        if r["copurchase_events"] < _MIN_COPURCHASE_EVENTS:
            return "Standalone"
        hi_int = r["intensity_resid"] >= int_med
        hi_div = r["diversity_resid"] >= div_med
        if hi_int and hi_div:
            return "Anchor-hub"
        if hi_int and not hi_div:
            return "Complement-addon"
        if (not hi_int) and hi_div:
            return "Versatile-connector"
        return "Niche-pair"

    df["e2_outfit_role"] = df.apply(_role, axis=1)
    out = df[["article_id", "e2_outfit_role", "copurchase_events", "n_partner_groups"]].rename(
        columns={
            "copurchase_events": "e2_copurchase_events",
            "n_partner_groups": "e2_copurchase_degree",
        }
    )
    logger.info(
        "outfit_role: %d articles; dist=%s",
        len(out),
        out["e2_outfit_role"].value_counts().to_dict(),
    )
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_behavioral_axes(data_dir: Path, output_dir: Path) -> Path:
    """Compute all three behavior-derived axes over the full catalog → Parquet.

    Args:
        data_dir: directory with ``train_transactions.parquet`` + ``articles.parquet``.
        output_dir: output directory for ``behavioral_axes.parquet``.

    Returns:
        Path to the written parquet (keyed by ``article_id``).
    """
    train_path = data_dir / "train_transactions.parquet"
    articles_path = data_dir / "articles.parquet"
    for p in (train_path, articles_path):
        if not p.exists():
            raise FileNotFoundError(p)

    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        price = compute_price_tier(con, train_path, articles_path)
        trend = compute_trend_phase(con, train_path)
        role = compute_outfit_role(con, train_path, articles_path)
    finally:
        con.close()

    # Base on the full articles catalog so every article_id gets a row.
    base = pd.read_parquet(articles_path, columns=["article_id"])
    base["article_id"] = base["article_id"].astype(str)
    out = (
        base.merge(price, on="article_id", how="left")
        .merge(trend, on="article_id", how="left")
        .merge(role, on="article_id", how="left")
    )
    # Items with no behavior get explicit "no-signal" sentinels (excluded from screen).
    out["e2_trend_phase_actual"] = out["e2_trend_phase_actual"].fillna("Insufficient")
    out["e2_outfit_role"] = out["e2_outfit_role"].fillna("Standalone")

    output_path = output_dir / OUTPUT_FILENAME
    out.to_parquet(output_path, index=False)
    logger.info("behavioral_axes: %d articles → %s", len(out), output_path)
    return output_path
