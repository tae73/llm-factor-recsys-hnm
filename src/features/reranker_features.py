"""Build tabular features for the GBDT Re-Ranker from Stage 1 candidates.

Two modes:
  - Base (~21D): score + rank + user/item features only
  - Full (~127D): Base + L1/L2/L3 attributes + cross features + BGE similarity

Feature groups:
  1. Stage 1 score + rank (3D)
  2. User features (8 num + 3 cat = 11D)
  3. Item features (2 num + 5 cat = 7D)
  4. L1 attributes (~7D): material, closure, design_details_count, category-specific slots
  5. L2 attributes (~43D): style_mood multi-hot, occasion multi-hot, ordinal, categorical
  6. L3 attributes (~52D): style_lineage multi-hot, categorical, ordinal, category-specific slots
  7. Cross features (3D)
  8. BGE similarity (1D)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# ---------------------------------------------------------------------------
# Column definitions (aligned with text_composer.py)
# ---------------------------------------------------------------------------

# L1 shared categorical columns
L1_CAT_COLS = ["l1_material", "l1_closure"]

# L1 category-specific columns (unified as slot4-7 across super_category)
L1_SLOT_COLS_MAP: dict[str, list[str]] = {
    "Apparel": ["l1_neckline", "l1_sleeve_type", "l1_fit", "l1_length"],
    "Footwear": ["l1_toe_shape", "l1_shaft_height", "l1_heel_type", "l1_sole_type"],
    "Accessories": ["l1_form_factor", "l1_size_scale", "l1_wearing_method", "l1_primary_function"],
}
L1_SLOT_NAMES = ["l1_slot4", "l1_slot5", "l1_slot6", "l1_slot7"]

# L1 count feature
L1_COUNT_COL = "l1_design_details"

# L2 columns
L2_CAT_COLS = ["l2_trendiness", "l2_season_fit"]
L2_ORDINAL_COLS = ["l2_perceived_quality", "l2_versatility"]
L2_MULTIHOT_COLS = ["l2_style_mood", "l2_occasion"]

# L3 shared columns
L3_CAT_COLS = ["l3_color_harmony", "l3_coordination_role", "l3_tone_season"]
L3_ORDINAL_COLS = ["l3_visual_weight"]
L3_MULTIHOT_COLS = ["l3_style_lineage"]

# L3 category-specific
L3_SLOT_COLS_MAP: dict[str, list[str]] = {
    "Apparel": ["l3_silhouette", "l3_proportion_effect"],
    "Footwear": ["l3_foot_silhouette", "l3_height_effect"],
    "Accessories": ["l3_visual_form", "l3_styling_effect"],
}
L3_SLOT_NAMES = ["l3_slot6", "l3_slot7"]


# ---------------------------------------------------------------------------
# Attribute Encoding
# ---------------------------------------------------------------------------


def _unify_slots(
    fk: pd.DataFrame, slot_cols_map: dict[str, list[str]], slot_names: list[str]
) -> pd.DataFrame:
    """Unify category-specific columns into generic slot columns."""
    df = fk.copy()
    for i, slot_name in enumerate(slot_names):
        df[slot_name] = "UNKNOWN"
        for super_cat, cols in slot_cols_map.items():
            if i < len(cols) and cols[i] in df.columns:
                mask = df["super_category"] == super_cat
                df.loc[mask, slot_name] = df.loc[mask, cols[i]].fillna("UNKNOWN").astype(str)
    return df


def _parse_list_col(series: pd.Series) -> list[list[str]]:
    """Parse list column — handles both actual lists and JSON strings."""
    result = []
    for val in series:
        if isinstance(val, list):
            result.append([str(v) for v in val])
        elif isinstance(val, str):
            try:
                parsed = json.loads(val)
                result.append([str(v) for v in parsed] if isinstance(parsed, list) else [])
            except (json.JSONDecodeError, TypeError):
                result.append([])
        else:
            result.append([])
    return result


def build_attribute_encoders(fk: pd.DataFrame) -> dict[str, Any]:
    """Fit LabelEncoders and MultiLabelBinarizers on factual_knowledge.

    Args:
        fk: factual_knowledge DataFrame with L1/L2/L3 columns.

    Returns:
        dict of fitted encoders keyed by column name.
    """
    fk = _unify_slots(fk, L1_SLOT_COLS_MAP, L1_SLOT_NAMES)
    fk = _unify_slots(fk, L3_SLOT_COLS_MAP, L3_SLOT_NAMES)

    encoders: dict[str, Any] = {}

    # Categorical → LabelEncoder
    all_cat_cols = L1_CAT_COLS + L1_SLOT_NAMES + L2_CAT_COLS + L3_CAT_COLS + L3_SLOT_NAMES
    for col in all_cat_cols:
        le = LabelEncoder()
        values = fk[col].fillna("UNKNOWN").astype(str) if col in fk.columns else pd.Series(["UNKNOWN"])
        le.fit(np.append(values.unique(), "UNKNOWN"))
        encoders[col] = le

    # Multi-hot → MultiLabelBinarizer
    for col in L2_MULTIHOT_COLS + L3_MULTIHOT_COLS:
        mlb = MultiLabelBinarizer()
        if col in fk.columns:
            parsed = _parse_list_col(fk[col])
            mlb.fit(parsed)
        else:
            mlb.fit([[]])
        encoders[col] = mlb

    return encoders


def save_encoders(encoders: dict[str, Any], path: Path) -> None:
    """Save fitted encoders to pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(encoders, f)


def load_encoders(path: Path) -> dict[str, Any]:
    """Load fitted encoders from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def encode_item_attributes(
    fk: pd.DataFrame,
    encoders: dict[str, Any],
    idx_to_item: dict[int, str],
) -> tuple[np.ndarray, list[str]]:
    """Encode all items' L1+L2+L3 attributes to numerical features.

    Args:
        fk: factual_knowledge DataFrame.
        encoders: Fitted encoders from build_attribute_encoders().
        idx_to_item: Feature index → article_id mapping.

    Returns:
        (features, feature_names):
        - features: (n_items, D_attr) float32
        - feature_names: list of D_attr names
    """
    n_items = len(idx_to_item)

    # Build article_id → fk row index mapping
    fk = _unify_slots(fk, L1_SLOT_COLS_MAP, L1_SLOT_NAMES)
    fk = _unify_slots(fk, L3_SLOT_COLS_MAP, L3_SLOT_NAMES)
    fk_indexed = fk.set_index("article_id")

    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    # --- L1 categorical ---
    for col in L1_CAT_COLS + L1_SLOT_NAMES:
        le: LabelEncoder = encoders[col]
        vals = np.full(n_items, "UNKNOWN", dtype=object)
        for idx in range(n_items):
            aid = idx_to_item.get(idx)
            if aid and aid in fk_indexed.index:
                v = fk_indexed.at[aid, col]
                vals[idx] = str(v) if pd.notna(v) else "UNKNOWN"
        # Transform with unknown handling
        encoded = np.zeros(n_items, dtype=np.float32)
        for i, v in enumerate(vals):
            if v in le.classes_:
                encoded[i] = float(le.transform([v])[0])
        feature_blocks.append(encoded.reshape(-1, 1))
        feature_names.append(col)

    # --- L1 design_details count ---
    dd_count = np.zeros(n_items, dtype=np.float32)
    if L1_COUNT_COL in fk_indexed.columns:
        for idx in range(n_items):
            aid = idx_to_item.get(idx)
            if aid and aid in fk_indexed.index:
                val = fk_indexed.at[aid, L1_COUNT_COL]
                if isinstance(val, list):
                    dd_count[idx] = float(len(val))
                elif isinstance(val, str):
                    try:
                        dd_count[idx] = float(len(json.loads(val)))
                    except (json.JSONDecodeError, TypeError):
                        pass
    feature_blocks.append(dd_count.reshape(-1, 1))
    feature_names.append("l1_design_details_count")

    # --- L2 categorical ---
    for col in L2_CAT_COLS:
        le = encoders[col]
        vals = np.full(n_items, "UNKNOWN", dtype=object)
        for idx in range(n_items):
            aid = idx_to_item.get(idx)
            if aid and aid in fk_indexed.index:
                v = fk_indexed.at[aid, col]
                vals[idx] = str(v) if pd.notna(v) else "UNKNOWN"
        encoded = np.zeros(n_items, dtype=np.float32)
        for i, v in enumerate(vals):
            if v in le.classes_:
                encoded[i] = float(le.transform([v])[0])
        feature_blocks.append(encoded.reshape(-1, 1))
        feature_names.append(col)

    # --- L2 ordinal ---
    for col in L2_ORDINAL_COLS:
        arr = np.zeros(n_items, dtype=np.float32)
        if col in fk_indexed.columns:
            for idx in range(n_items):
                aid = idx_to_item.get(idx)
                if aid and aid in fk_indexed.index:
                    v = fk_indexed.at[aid, col]
                    if pd.notna(v):
                        arr[idx] = float(v)
        feature_blocks.append(arr.reshape(-1, 1))
        feature_names.append(col)

    # --- L2 multi-hot ---
    for col in L2_MULTIHOT_COLS:
        mlb: MultiLabelBinarizer = encoders[col]
        n_classes = len(mlb.classes_)
        arr = np.zeros((n_items, n_classes), dtype=np.float32)
        if col in fk_indexed.columns:
            for idx in range(n_items):
                aid = idx_to_item.get(idx)
                if aid and aid in fk_indexed.index:
                    val = fk_indexed.at[aid, col]
                    parsed = _parse_list_col(pd.Series([val]))[0]
                    # Only encode known classes
                    known = [v for v in parsed if v in mlb.classes_]
                    if known:
                        arr[idx] = mlb.transform([known])[0]
        feature_blocks.append(arr)
        feature_names.extend([f"{col}_{c}" for c in mlb.classes_])

    # --- L3 categorical ---
    for col in L3_CAT_COLS + L3_SLOT_NAMES:
        le = encoders[col]
        vals = np.full(n_items, "UNKNOWN", dtype=object)
        for idx in range(n_items):
            aid = idx_to_item.get(idx)
            if aid and aid in fk_indexed.index:
                v = fk_indexed.at[aid, col]
                vals[idx] = str(v) if pd.notna(v) else "UNKNOWN"
        encoded = np.zeros(n_items, dtype=np.float32)
        for i, v in enumerate(vals):
            if v in le.classes_:
                encoded[i] = float(le.transform([v])[0])
        feature_blocks.append(encoded.reshape(-1, 1))
        feature_names.append(col)

    # --- L3 ordinal ---
    for col in L3_ORDINAL_COLS:
        arr = np.zeros(n_items, dtype=np.float32)
        if col in fk_indexed.columns:
            for idx in range(n_items):
                aid = idx_to_item.get(idx)
                if aid and aid in fk_indexed.index:
                    v = fk_indexed.at[aid, col]
                    if pd.notna(v):
                        arr[idx] = float(v)
        feature_blocks.append(arr.reshape(-1, 1))
        feature_names.append(col)

    # --- L3 multi-hot ---
    for col in L3_MULTIHOT_COLS:
        mlb = encoders[col]
        n_classes = len(mlb.classes_)
        arr = np.zeros((n_items, n_classes), dtype=np.float32)
        if col in fk_indexed.columns:
            for idx in range(n_items):
                aid = idx_to_item.get(idx)
                if aid and aid in fk_indexed.index:
                    val = fk_indexed.at[aid, col]
                    parsed = _parse_list_col(pd.Series([val]))[0]
                    known = [v for v in parsed if v in mlb.classes_]
                    if known:
                        arr[idx] = mlb.transform([known])[0]
        feature_blocks.append(arr)
        feature_names.extend([f"{col}_{c}" for c in mlb.classes_])

    features = np.concatenate(feature_blocks, axis=1)
    return features, feature_names


# ---------------------------------------------------------------------------
# Re-Ranker Feature Construction
# ---------------------------------------------------------------------------


def build_reranker_features(
    user_indices: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    user_features: dict[str, np.ndarray],
    item_features: dict[str, np.ndarray],
    item_attributes: np.ndarray | None = None,
    attribute_names: list[str] | None = None,
    user_bge: np.ndarray | None = None,
    item_bge: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix for all user×candidate pairs.

    Args:
        user_indices: (N,) user feature indices.
        candidate_indices: (N, K) item feature indices per user.
        candidate_scores: (N, K) Stage 1 scores (descending).
        user_features: {categorical: (n_users, 3), numerical: (n_users, 8)}.
        item_features: {categorical: (n_items, 5), numerical: (n_items, 2)}.
        item_attributes: (n_items, D_attr) encoded attributes or None for Base mode.
        attribute_names: Feature names for attributes or None.
        user_bge: (n_users, 768) or None.
        item_bge: (n_items, 768) or None.

    Returns:
        (X, feature_names): (N*K, D_total), list of D_total names.
    """
    n_users_batch = user_indices.shape[0]
    top_k = candidate_indices.shape[1]
    n_samples = n_users_batch * top_k

    blocks: list[np.ndarray] = []
    names: list[str] = []

    # --- 1. Score features (3D) ---
    flat_scores = candidate_scores.reshape(n_samples).astype(np.float32)
    # Rank position (0-indexed)
    rank_pos = np.tile(np.arange(top_k, dtype=np.float32), n_users_batch)
    # Score gap to rank-1
    rank1_scores = np.repeat(candidate_scores[:, 0], top_k)
    gap_to_rank1 = (rank1_scores - flat_scores).astype(np.float32)

    blocks.append(np.column_stack([flat_scores, rank_pos, gap_to_rank1]))
    names.extend(["stage1_score", "rank_position", "gap_to_rank1"])

    # --- 2. User features (11D) ---
    u_num = user_features["numerical"][user_indices]  # (N, 8)
    u_cat = user_features["categorical"][user_indices]  # (N, 3)
    u_num_rep = np.repeat(u_num, top_k, axis=0).astype(np.float32)
    u_cat_rep = np.repeat(u_cat, top_k, axis=0).astype(np.float32)
    blocks.append(u_num_rep)
    names.extend([
        "u_n_purchases", "u_avg_price", "u_price_std", "u_n_unique_categories",
        "u_n_unique_colors", "u_days_since_first", "u_days_since_last", "u_online_ratio",
    ])
    blocks.append(u_cat_rep)
    names.extend(["u_age_group", "u_club_status", "u_fashion_news"])

    # --- 3. Item features (7D) ---
    flat_items = candidate_indices.reshape(n_samples)
    i_num = item_features["numerical"][flat_items].astype(np.float32)  # (N*K, 2)
    i_cat = item_features["categorical"][flat_items].astype(np.float32)  # (N*K, 5)
    blocks.append(i_num)
    names.extend(["i_total_purchases", "i_avg_price"])
    blocks.append(i_cat)
    names.extend(["i_product_type", "i_colour", "i_garment_group", "i_section", "i_index"])

    # --- 4-6. Attribute features (Full mode) ---
    if item_attributes is not None and attribute_names is not None:
        attr_feats = item_attributes[flat_items]  # (N*K, D_attr)
        blocks.append(attr_feats.astype(np.float32))
        names.extend(attribute_names)

    # --- 7. Cross features (3D) ---
    u_age = u_cat_rep[:, 0]  # age_group
    i_section = i_cat[:, 3]  # section_name
    cross_age_section = (u_age * 100 + i_section).astype(np.float32)

    u_avg_price = u_num_rep[:, 1]  # avg_price
    i_avg_price = i_num[:, 1]      # avg_price
    price_ratio = np.where(
        i_avg_price > 0, u_avg_price / (i_avg_price + 1e-8), 0.0
    ).astype(np.float32)

    u_recency = u_num_rep[:, 6]     # days_since_last_purchase
    i_popularity = i_num[:, 0]       # total_purchases
    recency_x_pop = (u_recency * i_popularity).astype(np.float32)

    blocks.append(np.column_stack([cross_age_section, price_ratio, recency_x_pop]))
    names.extend(["cross_age_section", "cross_price_ratio", "cross_recency_popularity"])

    # --- 8. BGE similarity (1D) ---
    if user_bge is not None and item_bge is not None:
        u_emb = user_bge[user_indices]  # (N, 768)
        u_emb_rep = np.repeat(u_emb, top_k, axis=0)  # (N*K, 768)
        i_emb = item_bge[flat_items]  # (N*K, 768)
        # Cosine similarity
        dot = np.sum(u_emb_rep * i_emb, axis=1)
        u_norm = np.linalg.norm(u_emb_rep, axis=1) + 1e-8
        i_norm = np.linalg.norm(i_emb, axis=1) + 1e-8
        cosine_sim = (dot / (u_norm * i_norm)).astype(np.float32)
        blocks.append(cosine_sim.reshape(-1, 1))
        names.append("bge_cosine_similarity")

    X = np.concatenate(blocks, axis=1)
    return X, names


def build_reranker_labels(
    user_indices: np.ndarray,
    candidate_indices: np.ndarray,
    ground_truth: dict[str, list[str]],
    idx_to_user: dict[int, str],
    idx_to_item: dict[int, str],
) -> np.ndarray:
    """Build binary labels: 1 if candidate item is in ground truth.

    Args:
        user_indices: (N,) user feature indices.
        candidate_indices: (N, K) item feature indices.
        ground_truth: {user_id: [article_id, ...]}.
        idx_to_user: Feature index → user_id.
        idx_to_item: Feature index → article_id.

    Returns:
        (N*K,) float32 binary labels.
    """
    n_users = user_indices.shape[0]
    top_k = candidate_indices.shape[1]
    labels = np.zeros(n_users * top_k, dtype=np.float32)

    for i in range(n_users):
        uid = idx_to_user.get(int(user_indices[i]), "")
        gt_items = set(ground_truth.get(uid, []))
        if not gt_items:
            continue
        for j in range(top_k):
            item_id = idx_to_item.get(int(candidate_indices[i, j]), "")
            if item_id in gt_items:
                labels[i * top_k + j] = 1.0

    return labels
