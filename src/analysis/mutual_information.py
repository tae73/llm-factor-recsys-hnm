"""Component A: Attribute-Purchase Mutual Information analysis.

Measures how much each attribute (metadata / L1 / L2 / L3) reduces uncertainty
about purchase decisions.  Uses train_pairs (pos+neg) to compute NMI per
attribute, PMI per attribute-value, and conditional MI for layer incremental
value.

All heavy joins are done via DuckDB; multi-value attributes (style_mood,
occasion, style_lineage — stored as JSON arrays) are exploded into binary
indicators.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class MIResult(NamedTuple):
    """MI result for a single attribute."""

    attribute: str
    layer: str  # "metadata", "l1", "l2", "l3"
    mi: float
    nmi: float
    n_values: int


class PMIResult(NamedTuple):
    """PMI for a single attribute value."""

    attribute: str
    value: str
    pmi: float
    p_given_pos: float  # P(value | purchased)
    p_catalog: float  # P(value | catalog)
    n_pos: int


# ---------------------------------------------------------------------------
# Attribute definitions
# ---------------------------------------------------------------------------

# Layer → list of (column_name, human_label, is_multi_value)
METADATA_ATTRS: list[tuple[str, str, bool]] = [
    ("product_type_name", "product_type", False),
    ("colour_group_name", "colour_group", False),
    ("section_name", "section", False),
    ("index_name", "index", False),
    ("garment_group_name", "garment_group", False),
]

L1_ATTRS: list[tuple[str, str, bool]] = [
    ("l1_material", "material", False),
    ("l1_closure", "closure", False),
    # Skip l1_design_details/material_detail (free text)
    # Category-specific slots (mixed meaning but still enum-valued)
    ("l1_slot4", "l1_slot4", False),
    ("l1_slot5", "l1_slot5", False),
    ("l1_slot6", "l1_slot6", False),
    ("l1_slot7", "l1_slot7", False),
]

L2_ATTRS: list[tuple[str, str, bool]] = [
    ("l2_style_mood", "style_mood", True),
    ("l2_occasion", "occasion", True),
    ("l2_perceived_quality", "perceived_quality", False),
    ("l2_trendiness", "trendiness", False),
    ("l2_season_fit", "season_fit", False),
    ("l2_versatility", "versatility", False),
]

L3_ATTRS: list[tuple[str, str, bool]] = [
    ("l3_color_harmony", "color_harmony", False),
    ("l3_tone_season", "tone_season", False),
    ("l3_coordination_role", "coordination_role", False),
    ("l3_visual_weight", "visual_weight", False),
    ("l3_style_lineage", "style_lineage", True),
    ("l3_slot6", "l3_slot6", False),
    ("l3_slot7", "l3_slot7", False),
]

ALL_ATTRS: dict[str, list[tuple[str, str, bool]]] = {
    "metadata": METADATA_ATTRS,
    "l1": L1_ATTRS,
    "l2": L2_ATTRS,
    "l3": L3_ATTRS,
}


# ---------------------------------------------------------------------------
# Core MI computation
# ---------------------------------------------------------------------------


def _parse_multi_value(val: object) -> list[str]:
    """Parse a JSON array string or plain string into a list of values."""
    if val is None:
        return []
    try:
        if isinstance(val, float) and np.isnan(val):
            return []
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s in ("nan", "<NA>", "None"):
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if v]
        except json.JSONDecodeError:
            pass
    return [s]


def _join_multi(val: object) -> str:
    """Join multi-value attribute into a pipe-separated string."""
    parsed = _parse_multi_value(val)
    return "|".join(sorted(parsed)) if parsed else "_NONE_"


def _compute_mi_for_column(
    labels: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, int]:
    """Compute MI and NMI between binary labels and categorical values.

    Args:
        labels: Binary array (0/1) of shape (N,).
        values: Categorical string array of shape (N,).

    Returns:
        (mi, nmi, n_unique_values)
    """
    # Build contingency table
    unique_vals, val_codes = np.unique(values, return_inverse=True)
    n_values = len(unique_vals)
    if n_values <= 1:
        return 0.0, 0.0, n_values

    n = len(labels)
    labels_int = labels.astype(np.int32)

    # Joint counts: (2, n_values)
    joint = np.zeros((2, n_values), dtype=np.float64)
    np.add.at(joint[0], val_codes[labels_int == 0], 1)
    np.add.at(joint[1], val_codes[labels_int == 1], 1)

    # Marginals
    p_joint = joint / n
    p_label = p_joint.sum(axis=1, keepdims=True)  # (2, 1)
    p_value = p_joint.sum(axis=0, keepdims=True)  # (1, n_values)

    # MI = sum p(y,v) * log(p(y,v) / (p(y)*p(v)))
    denom = p_label * p_value
    mask = (p_joint > 0) & (denom > 0)
    mi = np.sum(p_joint[mask] * np.log2(p_joint[mask] / denom[mask]))

    # H(A) for normalization
    p_val_flat = p_value.flatten()
    h_a = -np.sum(p_val_flat[p_val_flat > 0] * np.log2(p_val_flat[p_val_flat > 0]))

    nmi = mi / h_a if h_a > 0 else 0.0
    return float(mi), float(nmi), n_values


def _prepare_pairs_with_attributes(
    features_dir: Path,
    fk_path: Path,
    articles_path: Path,
    sample_size: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load train pairs and join with factual knowledge attributes.

    Returns DataFrame with columns: label + all attribute columns.
    """
    import duckdb

    from src.features.store import load_id_maps

    logger.info("Loading ID maps and train pairs...")
    _, _, _, idx_to_item = load_id_maps(features_dir)
    pairs = np.load(features_dir / "train_pairs.npz")
    item_idx = pairs["item_idx"]
    labels = pairs["labels"]

    # Map item_idx → article_id
    article_ids = np.array([idx_to_item[int(i)] for i in item_idx])

    df_pairs = pd.DataFrame({"article_id": article_ids, "label": labels})

    if sample_size is not None and len(df_pairs) > sample_size:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(df_pairs), sample_size, replace=False)
        df_pairs = df_pairs.iloc[idx].reset_index(drop=True)
        logger.info("Sampled %d pairs from %d", sample_size, len(pairs["labels"]))

    # Load factual knowledge + article metadata
    fk = pd.read_parquet(fk_path)
    articles = pd.read_parquet(articles_path, columns=[
        "article_id", "product_type_name", "colour_group_name",
        "section_name", "index_name", "garment_group_name",
    ])
    # String article_id for join
    fk["article_id"] = fk["article_id"].astype(str)
    articles["article_id"] = articles["article_id"].astype(str)
    df_pairs["article_id"] = df_pairs["article_id"].astype(str)

    con = duckdb.connect()
    con.register("pairs", df_pairs)
    con.register("fk", fk)
    con.register("articles", articles)

    # Collect needed columns
    fk_cols = []
    for layer_attrs in ALL_ATTRS.values():
        for col, _, _ in layer_attrs:
            fk_cols.append(col)

    # Separate metadata cols from fk cols
    meta_cols = [c for c, _, _ in METADATA_ATTRS]
    fk_only_cols = [c for c in fk_cols if c not in meta_cols]

    fk_select = ", ".join(f"fk.{c}" for c in fk_only_cols if c in fk.columns)
    meta_select = ", ".join(f"articles.{c}" for c in meta_cols)

    query = f"""
        SELECT pairs.label, {meta_select}, {fk_select}
        FROM pairs
        LEFT JOIN articles ON pairs.article_id = articles.article_id
        LEFT JOIN fk ON pairs.article_id = fk.article_id
    """
    result = con.execute(query).fetchdf()
    con.close()

    logger.info("Prepared %d pairs with %d attribute columns", len(result), len(result.columns) - 1)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_attribute_mi(
    features_dir: Path,
    fk_path: Path,
    articles_path: Path,
    sample_size: int | None = 10_000_000,
    random_seed: int = 42,
) -> list[MIResult]:
    """Compute MI and NMI for each attribute across all layers.

    Args:
        features_dir: Path to data/features/ (train_pairs.npz, id_maps.json).
        fk_path: Path to factual_knowledge.parquet.
        articles_path: Path to articles.parquet.
        sample_size: Subsample train pairs for speed (None = use all 121M).
        random_seed: Random seed for subsampling.

    Returns:
        List of MIResult sorted by NMI descending.
    """
    df = _prepare_pairs_with_attributes(
        features_dir, fk_path, articles_path, sample_size, random_seed,
    )
    labels = df["label"].values

    results: list[MIResult] = []

    for layer, attrs in ALL_ATTRS.items():
        for col, label, is_multi in attrs:
            if col not in df.columns:
                logger.warning("Column %s not found, skipping", col)
                continue

            if is_multi:
                # Explode multi-value into binary indicators, compute MI per value
                # then aggregate as joint MI across all values
                raw = df[col].values
                all_values: list[str] = []
                all_labels: list[float] = []
                for i, val in enumerate(raw):
                    parsed = _parse_multi_value(val)
                    if parsed:
                        for v in parsed:
                            all_values.append(v)
                            all_labels.append(labels[i])
                    else:
                        all_values.append("_NONE_")
                        all_labels.append(labels[i])

                vals_arr = np.array(all_values)
                labs_arr = np.array(all_labels)
            else:
                vals_arr = df[col].astype(str).replace("nan", "_NONE_").replace("<NA>", "_NONE_").values
                labs_arr = labels

            mi, nmi, n_vals = _compute_mi_for_column(labs_arr, vals_arr)
            results.append(MIResult(
                attribute=label, layer=layer, mi=mi, nmi=nmi, n_values=n_vals,
            ))
            logger.info("  %s.%s: MI=%.6f, NMI=%.6f, |V|=%d", layer, label, mi, nmi, n_vals)

    results.sort(key=lambda r: r.nmi, reverse=True)
    return results


def compute_pmi_by_value(
    features_dir: Path,
    fk_path: Path,
    articles_path: Path,
    attribute: str,
    layer: str,
    top_n: int = 15,
    sample_size: int | None = 10_000_000,
    random_seed: int = 42,
) -> list[PMIResult]:
    """Compute PMI for each value of a given attribute.

    Returns top_n values by absolute PMI.
    """
    df = _prepare_pairs_with_attributes(
        features_dir, fk_path, articles_path, sample_size, random_seed,
    )
    labels = df["label"].values

    # Find column name
    attrs = ALL_ATTRS[layer]
    col = None
    is_multi = False
    for c, lbl, multi in attrs:
        if lbl == attribute:
            col = c
            is_multi = multi
            break
    if col is None or col not in df.columns:
        raise ValueError(f"Attribute {attribute} not found in layer {layer}")

    # Build value-label pairs
    if is_multi:
        raw = df[col].values
        values_list: list[str] = []
        labels_list: list[float] = []
        for i, val in enumerate(raw):
            parsed = _parse_multi_value(val)
            for v in (parsed or ["_NONE_"]):
                values_list.append(v)
                labels_list.append(labels[i])
        vals_arr = np.array(values_list)
        labs_arr = np.array(labels_list)
    else:
        vals_arr = df[col].astype(str).replace("nan", "_NONE_").replace("<NA>", "_NONE_").values
        labs_arr = labels

    # Compute per-value PMI
    unique_vals = np.unique(vals_arr)
    pos_mask = labs_arr == 1.0
    p_pos = pos_mask.mean()

    results: list[PMIResult] = []
    for v in unique_vals:
        v_mask = vals_arr == v
        n_pos = int((v_mask & pos_mask).sum())
        n_total = int(v_mask.sum())
        if n_total == 0:
            continue
        p_v_given_pos = (v_mask & pos_mask).sum() / pos_mask.sum() if pos_mask.sum() > 0 else 0.0
        p_v = n_total / len(vals_arr)
        if p_v > 0 and p_v_given_pos > 0:
            pmi = float(np.log2(p_v_given_pos / p_v))
        else:
            pmi = 0.0
        results.append(PMIResult(
            attribute=attribute, value=str(v), pmi=pmi,
            p_given_pos=float(p_v_given_pos), p_catalog=float(p_v), n_pos=n_pos,
        ))

    results.sort(key=lambda r: abs(r.pmi), reverse=True)
    return results[:top_n]


def compute_conditional_mi(
    features_dir: Path,
    fk_path: Path,
    articles_path: Path,
    condition_layer: str,
    target_layer: str,
    n_bins: int = 10,
    sample_size: int | None = 10_000_000,
    random_seed: int = 42,
) -> float:
    """Compute MI(purchase; target_layer | condition_layer).

    Approximated by binning the condition layer's concatenated values
    and computing MI within each bin, then averaging weighted by bin size.

    Returns:
        Conditional MI value.
    """
    df = _prepare_pairs_with_attributes(
        features_dir, fk_path, articles_path, sample_size, random_seed,
    )
    labels = df["label"].values

    # Concatenate condition layer attributes into a single bin key
    cond_attrs = ALL_ATTRS[condition_layer]
    cond_parts = []
    for col, _, is_multi in cond_attrs:
        if col not in df.columns:
            continue
        if is_multi:
            cond_parts.append(df[col].apply(
                lambda x: _join_multi(x)
            ))
        else:
            cond_parts.append(df[col].astype(str).replace("nan", "_NONE_").replace("<NA>", "_NONE_"))
    cond_key = pd.concat(cond_parts, axis=1).apply(lambda row: "||".join(row), axis=1)

    # Hash → bin (too many unique combos to use directly)
    bin_codes = pd.util.hash_pandas_object(cond_key, index=False).values % n_bins

    # Concatenate target layer attributes
    target_attrs = ALL_ATTRS[target_layer]
    target_parts = []
    for col, _, is_multi in target_attrs:
        if col not in df.columns:
            continue
        if is_multi:
            target_parts.append(df[col].apply(
                lambda x: _join_multi(x)
            ))
        else:
            target_parts.append(df[col].astype(str).replace("nan", "_NONE_").replace("<NA>", "_NONE_"))
    target_key = pd.concat(target_parts, axis=1).apply(lambda row: "||".join(row), axis=1)
    target_arr = target_key.values

    # Compute conditional MI = weighted average of MI within each bin
    total_mi = 0.0
    total_weight = 0.0
    for b in range(n_bins):
        mask = bin_codes == b
        n_b = mask.sum()
        if n_b < 100:
            continue
        mi_b, _, _ = _compute_mi_for_column(labels[mask], target_arr[mask])
        total_mi += mi_b * n_b
        total_weight += n_b

    conditional_mi = total_mi / total_weight if total_weight > 0 else 0.0
    logger.info(
        "Conditional MI(purchase; %s | %s) = %.6f",
        target_layer, condition_layer, conditional_mi,
    )
    return conditional_mi


def mi_results_to_dataframe(results: list[MIResult]) -> pd.DataFrame:
    """Convert MI results to a DataFrame for plotting."""
    return pd.DataFrame([r._asdict() for r in results])
