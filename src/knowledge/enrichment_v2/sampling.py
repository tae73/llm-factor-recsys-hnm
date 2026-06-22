"""Pilot sampling for the enrichment-v2 de-risk run (~500 product_codes).

Dual objective: DISCRIMINATION realism (breadth across garment groups, so the DE1
top1-share / entropy are measured on a realistic distribution) AND behavioral POWER
(bias toward product_codes with ≥``floor`` train purchases so the preliminary
behavioral read has signal). Strategy: eligible = codes whose SKUs sum to ≥floor
purchases; stratify across ``garment_group_name`` (floor ≥``per_group_floor`` each for
breadth); within each stratum take the top-k by purchase count (power bias). Frozen to
a CSV + manifest (seed 42) for deterministic, resumable extraction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

SAMPLE_FILENAME = "pilot_sample.csv"
MANIFEST_FILENAME = "pilot_sample_manifest.json"


def build_pilot_sample(
    data_dir: Path,
    n_codes: int = 500,
    floor: int = 10,
    per_group_floor: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Build the stratified pilot sample of product_codes.

    Returns:
        (sample_df, manifest_dict). sample_df columns: product_code, garment_group_name,
        product_group_name, product_type_name, pc_purchases, stratum.
    """
    train_path = data_dir / "train_transactions.parquet"
    articles_path = data_dir / "articles.parquet"

    con = duckdb.connect()
    pc = con.execute(f"""
        WITH item_pop AS (
            SELECT CAST(article_id AS VARCHAR) AS article_id, COUNT(*) AS cnt
            FROM read_parquet('{train_path}') GROUP BY article_id
        )
        SELECT CAST(a.product_code AS VARCHAR)        AS product_code,
               ANY_VALUE(a.garment_group_name)        AS garment_group_name,
               ANY_VALUE(a.product_group_name)        AS product_group_name,
               ANY_VALUE(a.product_type_name)         AS product_type_name,
               COALESCE(SUM(ip.cnt), 0)               AS pc_purchases
        FROM read_parquet('{articles_path}') a
        LEFT JOIN item_pop ip ON CAST(a.article_id AS VARCHAR) = ip.article_id
        GROUP BY a.product_code
        """).fetchdf()
    con.close()

    eligible = pc[pc["pc_purchases"] >= floor].copy()
    eligible = eligible.sort_values(
        ["pc_purchases", "product_code"], ascending=[False, True]
    ).reset_index(drop=True)
    logger.info(
        "Pilot pool: %d product_codes total, %d eligible (>=%d purchases)",
        len(pc),
        len(eligible),
        floor,
    )

    groups = sorted(eligible["garment_group_name"].dropna().unique().tolist())
    counts = eligible["garment_group_name"].value_counts().to_dict()

    # Floor allocation, then proportional remainder by eligible group size.
    alloc = {g: min(per_group_floor, counts.get(g, 0)) for g in groups}
    used = sum(alloc.values())
    remaining = max(0, n_codes - used)
    total_eligible = sum(counts.get(g, 0) for g in groups)
    if remaining > 0 and total_eligible > 0:
        for g in groups:
            extra = int(round(remaining * counts.get(g, 0) / total_eligible))
            alloc[g] = min(counts.get(g, 0), alloc[g] + extra)

    # Take top-k by purchases within each group (deterministic).
    picks: list[pd.DataFrame] = []
    for g in groups:
        k = alloc[g]
        if k <= 0:
            continue
        picks.append(eligible[eligible["garment_group_name"] == g].head(k))
    sample = pd.concat(picks, ignore_index=True) if picks else eligible.head(0).copy()

    # Trim/pad to ~n_codes (rounding can drift a few): take the highest-purchase extras.
    if len(sample) > n_codes:
        sample = sample.sort_values(["pc_purchases", "product_code"], ascending=[False, True]).head(
            n_codes
        )
    elif len(sample) < n_codes:
        chosen = set(sample["product_code"])
        extras = eligible[~eligible["product_code"].isin(chosen)].head(n_codes - len(sample))
        sample = pd.concat([sample, extras], ignore_index=True)

    sample = (
        sample.assign(stratum=sample["garment_group_name"])
        .sort_values(["garment_group_name", "pc_purchases"], ascending=[True, False])
        .reset_index(drop=True)
    )

    manifest = {
        "n_codes_target": n_codes,
        "n_codes_selected": int(len(sample)),
        "floor_purchases": floor,
        "per_group_floor": per_group_floor,
        "seed": seed,
        "n_eligible": int(len(eligible)),
        "n_strata": int(sample["garment_group_name"].nunique()),
        "pc_purchases": {
            "min": int(sample["pc_purchases"].min()),
            "median": float(sample["pc_purchases"].median()),
            "max": int(sample["pc_purchases"].max()),
        },
        "stratum_counts": sample["garment_group_name"].value_counts().to_dict(),
    }
    return sample, manifest


def freeze_pilot_sample(data_dir: Path, output_dir: Path, **kwargs) -> tuple[Path, Path]:
    """Build and persist the pilot sample CSV + manifest. Returns (csv_path, manifest_path)."""
    sample, manifest = build_pilot_sample(data_dir, **kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / SAMPLE_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    sample.to_csv(csv_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Froze pilot sample: %d codes → %s", len(sample), csv_path)
    return csv_path, manifest_path


def load_pilot_articles(data_dir: Path, sample_csv: Path) -> pd.DataFrame:
    """Load all article SKUs for the sampled product_codes (the extractor's input)."""
    sample = pd.read_csv(sample_csv, dtype={"product_code": str})
    codes = set(sample["product_code"].astype(str))
    articles = pd.read_parquet(data_dir / "articles.parquet")
    articles["product_code"] = articles["product_code"].astype(str)
    articles["article_id"] = articles["article_id"].astype(str)
    subset = articles[articles["product_code"].isin(codes)].reset_index(drop=True)
    logger.info("Loaded %d article SKUs for %d sampled product_codes", len(subset), len(codes))
    return subset
