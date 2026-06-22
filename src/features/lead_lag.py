"""Lead-lag analysis — does an attribute's composition LEAD category sales? (E2-2 ②).

For the trend-lead-time value-matrix cell: build a per-category monthly time series of
*attribute share* (fraction of a category's sales carried by items in a target attribute
value-set, e.g. trend_phase ∈ {Emerging, Rising}) and test whether share(t) correlates
with category sales(t+k) for lag k≥1 — i.e. the attribute mix *precedes* demand. The
honest baseline is the SAME lead-lag computed on a metadata-proxy share (product_type),
and significance is a block-bootstrap over categories (only ~22 train months ⇒ thin
power, so a wide CI → MARGINAL is the expected honest verdict).

DuckDB monthly granularity (``date_trunc('month')``), CPU only, no API.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def monthly_attribute_share(
    con: duckdb.DuckDBPyConnection,
    train_path: Path,
    articles_path: Path,
    axis_df: pd.DataFrame,
    axis_col: str,
    values: list | None = None,
    category_col: str = "index_name",
    granularity: str = "month",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Per (category, period): total sales + share carried by a target attribute (E2-2/E2-3 ②).

    Two modes (defaults reproduce E2-2 exactly):
    - **binary** (``weight_col=None``): ``share`` = fraction of period sales whose ``axis_col ∈ values``.
    - **continuous** (``weight_col`` set): ``share`` = mean of ``max(weight, 0)`` over period sales
      (momentum-weighted; uses the full ordinal signal, e.g. ``e2_trend_momentum``).
    ``granularity`` ∈ {"month","week",...} (E2-3 uses "week" → ~94 vs 22 points → tighter CI).

    Returns DataFrame[cat, mo, cat_sales, val_sales, share].
    """
    use_weight = weight_col is not None
    src_col = weight_col if use_weight else axis_col
    tbl = axis_df[["article_id", src_col]].copy()
    tbl["article_id"] = tbl["article_id"].astype(str)
    tbl = tbl.rename(columns={src_col: "av"})
    if not use_weight:
        tbl["av"] = tbl["av"].astype(str)
    con.register("axis_tbl", tbl)
    if use_weight:
        val_expr = "SUM(GREATEST(CAST(av AS DOUBLE), 0))"
    else:
        val_list = ", ".join("'" + str(v).replace("'", "''") + "'" for v in (values or []))
        val_expr = f"SUM(CASE WHEN av IN ({val_list}) THEN 1 ELSE 0 END)"
    df = con.execute(f"""
        WITH tx AS (
            SELECT date_trunc('{granularity}', t.t_dat) AS mo,
                   a.{category_col} AS cat,
                   x.av AS av
            FROM read_parquet('{train_path}') t
            JOIN read_parquet('{articles_path}') a
                 ON CAST(t.article_id AS VARCHAR) = CAST(a.article_id AS VARCHAR)
            JOIN axis_tbl x
                 ON CAST(t.article_id AS VARCHAR) = x.article_id
        )
        SELECT cat, mo,
               COUNT(*) AS cat_sales,
               {val_expr} AS val_sales
        FROM tx
        GROUP BY cat, mo
        ORDER BY cat, mo
        """).fetchdf()
    con.unregister("axis_tbl")
    df["mo"] = pd.to_datetime(df["mo"])
    df["share"] = df["val_sales"] / df["cat_sales"].clip(lower=1)
    return df


def _per_category_lag_corr(share_df: pd.DataFrame, lag: int) -> dict[str, float]:
    """Per-category Pearson corr of share(t) vs cat_sales(t+lag). {category: r}."""
    out: dict[str, float] = {}
    for cat, g in share_df.groupby("cat"):
        g = g.sort_values("mo")
        s = g["share"].to_numpy(dtype=float)
        sales = g["cat_sales"].to_numpy(dtype=float)
        if len(g) <= lag + 2:
            continue
        x, y = s[:-lag], sales[lag:]
        if x.std() == 0 or y.std() == 0:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if np.isfinite(r):
            out[cat] = float(r)
    return out


def lead_lag_corr(share_df: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 4)) -> dict[int, float]:
    """Mean across categories of corr(share(t), sales(t+k)) for each lag k."""
    out: dict[int, float] = {}
    for k in lags:
        per_cat = _per_category_lag_corr(share_df, k)
        out[k] = float(np.mean(list(per_cat.values()))) if per_cat else float("nan")
    return out


def lead_lag_vs_baseline(
    axis_share: pd.DataFrame,
    meta_share: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 4),
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Best-lag attr corr vs metadata-proxy corr, with block-bootstrap over categories.

    Picks the lag maximizing the mean attribute corr (k≥1 = genuine lead), then bootstraps
    the per-category (attr_r − meta_r) at that lag. Returns best_lag, r_attr, r_meta, delta,
    ci_lo, ci_hi, n_categories.
    """
    attr_by_lag = lead_lag_corr(axis_share, lags)
    finite = {k: v for k, v in attr_by_lag.items() if np.isfinite(v)}
    if not finite:
        return {
            "best_lag": None,
            "r_attr": float("nan"),
            "r_meta": float("nan"),
            "delta": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_categories": 0,
            "attr_by_lag": attr_by_lag,
        }
    best_lag = max(finite, key=finite.get)

    attr_cat = _per_category_lag_corr(axis_share, best_lag)
    meta_cat = _per_category_lag_corr(meta_share, best_lag)
    cats = sorted(set(attr_cat) & set(meta_cat))
    deltas = np.array([attr_cat[c] - meta_cat[c] for c in cats], dtype=float)

    rng = np.random.default_rng(seed)
    if len(deltas) >= 2:
        idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
        boot = deltas[idx].mean(axis=1)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    else:
        ci_lo = ci_hi = float("nan")

    return {
        "best_lag": int(best_lag),
        "r_attr": round(float(np.mean([attr_cat[c] for c in cats])), 4) if cats else float("nan"),
        "r_meta": round(float(np.mean([meta_cat[c] for c in cats])), 4) if cats else float("nan"),
        "delta": round(float(deltas.mean()), 4) if len(deltas) else float("nan"),
        "ci_lo": round(float(ci_lo), 4),
        "ci_hi": round(float(ci_hi), 4),
        "n_categories": len(cats),
        "attr_by_lag": {
            int(k): (round(v, 4) if np.isfinite(v) else None) for k, v in attr_by_lag.items()
        },
    }
