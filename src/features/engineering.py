"""DuckDB-based feature engineering for recommendation models.

Computes user/item features from train split only (no data leakage),
generates negative samples, and produces numpy arrays for training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import duckdb
import numpy as np
import pandas as pd

from src.config import FeatureConfig, FeatureResult


# ---------------------------------------------------------------------------
# Feature Result Types
# ---------------------------------------------------------------------------


class UserFeatures(NamedTuple):
    """User feature arrays with metadata."""

    user_ids: list[str]
    numerical: np.ndarray  # (n_users, 8) float32
    categorical: np.ndarray  # (n_users, 3) int32
    num_names: list[str]
    cat_names: list[str]
    cat_vocabs: dict[str, dict[str, int]]


class ItemFeatures(NamedTuple):
    """Item feature arrays with metadata."""

    item_ids: list[str]
    numerical: np.ndarray  # (n_items, 2) float32
    categorical: np.ndarray  # (n_items, 5) int32
    num_names: list[str]
    cat_names: list[str]
    cat_vocabs: dict[str, dict[str, int]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_NUM_NAMES = [
    "n_purchases",
    "avg_price",
    "price_std",
    "n_unique_categories",
    "n_unique_colors",
    "days_since_first_purchase",
    "days_since_last_purchase",
    "online_purchase_ratio",
]

USER_CAT_NAMES = ["age_group", "club_member_status", "fashion_news_frequency"]

ITEM_NUM_NAMES = ["total_purchases", "avg_price"]

ITEM_CAT_NAMES = [
    "product_type_name",
    "colour_group_name",
    "garment_group_name",
    "section_name",
    "index_name",
]

# Club member status vocabulary (including UNKNOWN for nulls)
CLUB_MEMBER_VALUES = ["UNKNOWN", "ACTIVE", "PRE-CREATE", "LEFT CLUB", "UNKNOWN_OTHER"]

# Fashion news frequency vocabulary
FASHION_NEWS_VALUES = ["UNKNOWN", "NONE", "Regularly", "Monthly", "UNKNOWN_OTHER"]


# ---------------------------------------------------------------------------
# User Features
# ---------------------------------------------------------------------------


def compute_user_features(
    con: duckdb.DuckDBPyConnection,
    train_path: Path,
    articles_path: Path,
    customers_path: Path,
    config: FeatureConfig,
) -> UserFeatures:
    """Compute user features from train transactions only.

    Numerical (8): n_purchases, avg_price, price_std, n_unique_categories,
                   n_unique_colors, days_since_first, days_since_last, online_ratio
    Categorical (3): age_group, club_member_status, fashion_news_frequency
    """
    ref_date = config.reference_date

    # --- Numerical features from transactions ---
    txn_stats = con.execute(
        f"""
        SELECT
            t.customer_id,
            COUNT(*) AS n_purchases,
            AVG(t.price) AS avg_price,
            COALESCE(STDDEV_SAMP(t.price), 0) AS price_std,
            COUNT(DISTINCT a.product_type_name) AS n_unique_categories,
            COUNT(DISTINCT a.colour_group_name) AS n_unique_colors,
            DATE_DIFF('day', MIN(t.t_dat), '{ref_date}'::DATE) AS days_since_first,
            DATE_DIFF('day', MAX(t.t_dat), '{ref_date}'::DATE) AS days_since_last,
            AVG(CASE WHEN t.sales_channel_id = 2 THEN 1.0 ELSE 0.0 END) AS online_ratio
        FROM read_parquet('{train_path}') t
        JOIN read_parquet('{articles_path}') a ON t.article_id = a.article_id
        GROUP BY t.customer_id
        ORDER BY t.customer_id
        """
    ).fetchdf()

    user_ids = txn_stats["customer_id"].tolist()
    user_num = txn_stats[
        [
            "n_purchases",
            "avg_price",
            "price_std",
            "n_unique_categories",
            "n_unique_colors",
            "days_since_first",
            "days_since_last",
            "online_ratio",
        ]
    ].values.astype(np.float32)

    # Replace NaN/Inf with 0
    user_num = np.nan_to_num(user_num, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Categorical features from customers table ---
    cust_df = con.execute(
        f"""
        SELECT customer_id, age, club_member_status, fashion_news_frequency
        FROM read_parquet('{customers_path}')
        ORDER BY customer_id
        """
    ).fetchdf()

    # Build customer lookup (customer_id → row)
    cust_lookup = dict(zip(cust_df["customer_id"].tolist(), range(len(cust_df))))

    # Age binning
    age_bins = list(config.age_bins)
    age_labels = list(config.age_labels)

    # Build age_group vocab: idx 0 = UNKNOWN
    age_vocab = {"UNKNOWN": 0}
    for i, label in enumerate(age_labels):
        age_vocab[label] = i + 1 if label != "unknown" else 0

    # Club member vocab
    club_vocab = {v: i for i, v in enumerate(CLUB_MEMBER_VALUES)}
    fn_vocab = {v: i for i, v in enumerate(FASHION_NEWS_VALUES)}

    # Compute categorical arrays aligned to user_ids (from txn_stats)
    n_users = len(user_ids)
    user_cat = np.zeros((n_users, 3), dtype=np.int32)

    for i, uid in enumerate(user_ids):
        cust_idx = cust_lookup.get(uid)
        if cust_idx is None:
            continue  # all zeros (UNKNOWN)

        row = cust_df.iloc[cust_idx]

        # Age group
        age = row["age"]
        if pd.isna(age):
            user_cat[i, 0] = 0  # UNKNOWN
        else:
            age_int = int(age)
            assigned = False
            for j in range(len(age_bins) - 1):
                if age_bins[j] <= age_int < age_bins[j + 1]:
                    user_cat[i, 0] = age_vocab.get(age_labels[j], 0)
                    assigned = True
                    break
            if not assigned:
                user_cat[i, 0] = 0

        # Club member status
        cms = str(row["club_member_status"]) if not pd.isna(row["club_member_status"]) else "UNKNOWN"
        user_cat[i, 1] = club_vocab.get(cms, club_vocab.get("UNKNOWN_OTHER", 0))

        # Fashion news frequency
        fnf = (
            str(row["fashion_news_frequency"])
            if not pd.isna(row["fashion_news_frequency"])
            else "UNKNOWN"
        )
        user_cat[i, 2] = fn_vocab.get(fnf, fn_vocab.get("UNKNOWN_OTHER", 0))

    cat_vocabs = {
        "age_group": age_vocab,
        "club_member_status": club_vocab,
        "fashion_news_frequency": fn_vocab,
    }

    return UserFeatures(
        user_ids=user_ids,
        numerical=user_num,
        categorical=user_cat,
        num_names=USER_NUM_NAMES,
        cat_names=USER_CAT_NAMES,
        cat_vocabs=cat_vocabs,
    )


# ---------------------------------------------------------------------------
# Item Features
# ---------------------------------------------------------------------------


def compute_item_features(
    con: duckdb.DuckDBPyConnection,
    train_path: Path,
    articles_path: Path,
    config: FeatureConfig,
) -> ItemFeatures:
    """Compute item features. All articles included (full catalog scoring).

    Numerical (2): total_purchases, avg_price (from train split)
    Categorical (5): product_type_name, colour_group_name, garment_group_name,
                     section_name, index_name (from articles metadata)
    """
    # --- Numerical: transaction stats per item ---
    item_stats = con.execute(
        f"""
        SELECT
            article_id,
            COUNT(*) AS total_purchases,
            AVG(price) AS avg_price
        FROM read_parquet('{train_path}')
        GROUP BY article_id
        """
    ).fetchdf()

    item_stats_lookup = dict(
        zip(
            item_stats["article_id"].tolist(),
            zip(item_stats["total_purchases"].tolist(), item_stats["avg_price"].tolist()),
        )
    )

    # --- All articles (full catalog) ---
    articles_df = con.execute(
        f"""
        SELECT
            article_id,
            COALESCE(product_type_name, 'UNKNOWN') AS product_type_name,
            COALESCE(colour_group_name, 'UNKNOWN') AS colour_group_name,
            COALESCE(garment_group_name, 'UNKNOWN') AS garment_group_name,
            COALESCE(section_name, 'UNKNOWN') AS section_name,
            COALESCE(index_name, 'UNKNOWN') AS index_name
        FROM read_parquet('{articles_path}')
        ORDER BY article_id
        """
    ).fetchdf()

    item_ids = articles_df["article_id"].tolist()
    n_items = len(item_ids)

    # Numerical: fill items without train transactions with 0
    item_num = np.zeros((n_items, 2), dtype=np.float32)
    for i, aid in enumerate(item_ids):
        stats = item_stats_lookup.get(aid)
        if stats is not None:
            item_num[i, 0] = stats[0]  # total_purchases
            item_num[i, 1] = stats[1]  # avg_price

    item_num = np.nan_to_num(item_num, nan=0.0, posinf=0.0, neginf=0.0)

    # Categorical: build vocabularies (idx 0 = UNKNOWN)
    cat_vocabs: dict[str, dict[str, int]] = {}
    item_cat = np.zeros((n_items, 5), dtype=np.int32)

    for col_idx, col_name in enumerate(ITEM_CAT_NAMES):
        unique_vals = sorted(articles_df[col_name].unique().tolist())
        vocab = {"UNKNOWN": 0}
        for val in unique_vals:
            if val != "UNKNOWN" and val not in vocab:
                vocab[val] = len(vocab)
        cat_vocabs[col_name] = vocab

        item_cat[:, col_idx] = np.array(
            list(map(lambda v: vocab.get(v, 0), articles_df[col_name].tolist())),
            dtype=np.int32,
        )

    return ItemFeatures(
        item_ids=item_ids,
        numerical=item_num,
        categorical=item_cat,
        num_names=ITEM_NUM_NAMES,
        cat_names=ITEM_CAT_NAMES,
        cat_vocabs=cat_vocabs,
    )


# ---------------------------------------------------------------------------
# ID Maps
# ---------------------------------------------------------------------------


def build_id_maps(
    user_features: UserFeatures,
    item_features: ItemFeatures,
) -> tuple[dict[str, int], dict[int, str], dict[str, int], dict[int, str]]:
    """Build bidirectional user/item ID ↔ index mappings."""
    user_to_idx = {uid: i for i, uid in enumerate(user_features.user_ids)}
    idx_to_user = {i: uid for i, uid in enumerate(user_features.user_ids)}
    item_to_idx = {iid: i for i, iid in enumerate(item_features.item_ids)}
    idx_to_item = {i: iid for i, iid in enumerate(item_features.item_ids)}
    return user_to_idx, idx_to_user, item_to_idx, idx_to_item


# ---------------------------------------------------------------------------
# Negative Sampling + Train Pairs
# ---------------------------------------------------------------------------


def generate_train_pairs(
    con: duckdb.DuckDBPyConnection,
    train_path: Path,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    config: FeatureConfig,
) -> dict[str, np.ndarray]:
    """Generate (user_idx, item_idx, label) training pairs with negative sampling.

    Positive pairs from train transactions, negatives sampled from non-purchased items.
    Returns dict with keys: user_idx, item_idx, labels.
    """
    rng = np.random.default_rng(config.random_seed)
    n_items = len(item_to_idx)

    # Load positive pairs (deduplicated)
    pos_df = con.execute(
        f"""
        SELECT DISTINCT customer_id, article_id
        FROM read_parquet('{train_path}')
        """
    ).fetchdf()

    # Build user → purchased items set for efficient negative check
    user_pos_items: dict[int, set[int]] = {}
    pos_user_idxs = []
    pos_item_idxs = []

    for _, row in pos_df.iterrows():
        uid = row["customer_id"]
        iid = row["article_id"]
        u_idx = user_to_idx.get(uid)
        i_idx = item_to_idx.get(iid)
        if u_idx is not None and i_idx is not None:
            pos_user_idxs.append(u_idx)
            pos_item_idxs.append(i_idx)
            user_pos_items.setdefault(u_idx, set()).add(i_idx)

    n_pos = len(pos_user_idxs)
    print(f"[features] Positive pairs: {n_pos:,}")

    # Generate negative samples
    neg_ratio = config.neg_sample_ratio
    neg_user_idxs = []
    neg_item_idxs = []

    for i in range(n_pos):
        u_idx = pos_user_idxs[i]
        pos_set = user_pos_items[u_idx]
        count = 0
        while count < neg_ratio:
            neg_item = rng.integers(0, n_items)
            if neg_item not in pos_set:
                neg_user_idxs.append(u_idx)
                neg_item_idxs.append(neg_item)
                count += 1

    n_neg = len(neg_user_idxs)
    print(f"[features] Negative pairs: {n_neg:,} (ratio={neg_ratio})")

    # Combine and shuffle
    all_user_idx = np.array(pos_user_idxs + neg_user_idxs, dtype=np.int32)
    all_item_idx = np.array(pos_item_idxs + neg_item_idxs, dtype=np.int32)
    all_labels = np.concatenate(
        [np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)]
    )

    perm = rng.permutation(len(all_user_idx))
    return {
        "user_idx": all_user_idx[perm],
        "item_idx": all_item_idx[perm],
        "labels": all_labels[perm],
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def run_feature_engineering(
    data_dir: Path,
    output_dir: Path,
    config: FeatureConfig = FeatureConfig(),
) -> FeatureResult:
    """Run complete feature engineering pipeline.

    1. Compute user features (train transactions only)
    2. Compute item features (full catalog)
    3. Build ID maps
    4. Generate training pairs with negative sampling
    5. Save all outputs
    """
    from src.features.store import save_features

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_transactions.parquet"
    articles_path = data_dir / "articles.parquet"
    customers_path = data_dir / "customers.parquet"

    con = duckdb.connect()

    try:
        print("[features] Computing user features...")
        user_features = compute_user_features(con, train_path, articles_path, customers_path, config)
        print(f"  Users: {len(user_features.user_ids):,}")
        print(f"  Numerical: {user_features.numerical.shape}")
        print(f"  Categorical: {user_features.categorical.shape}")

        print("[features] Computing item features...")
        item_features = compute_item_features(con, train_path, articles_path, config)
        print(f"  Items: {len(item_features.item_ids):,}")
        print(f"  Numerical: {item_features.numerical.shape}")
        print(f"  Categorical: {item_features.categorical.shape}")

        print("[features] Building ID maps...")
        user_to_idx, idx_to_user, item_to_idx, idx_to_item = build_id_maps(
            user_features, item_features
        )

        print("[features] Generating training pairs...")
        train_pairs = generate_train_pairs(con, train_path, user_to_idx, item_to_idx, config)
        print(f"  Total pairs: {len(train_pairs['labels']):,}")

    finally:
        con.close()

    # Build feature metadata
    feature_meta = {
        "n_users": len(user_features.user_ids),
        "n_items": len(item_features.item_ids),
        "n_train_pairs": len(train_pairs["labels"]),
        "user_num_names": user_features.num_names,
        "user_cat_names": user_features.cat_names,
        "item_num_names": item_features.num_names,
        "item_cat_names": item_features.cat_names,
        "user_cat_vocab_sizes": {k: len(v) for k, v in user_features.cat_vocabs.items()},
        "item_cat_vocab_sizes": {k: len(v) for k, v in item_features.cat_vocabs.items()},
        "n_user_numerical": user_features.numerical.shape[1],
        "n_item_numerical": item_features.numerical.shape[1],
        "neg_sample_ratio": config.neg_sample_ratio,
        "reference_date": config.reference_date,
        "random_seed": config.random_seed,
    }

    # Prepare .npz dicts
    user_features_npz = {
        "numerical": user_features.numerical,
        "categorical": user_features.categorical,
    }
    item_features_npz = {
        "numerical": item_features.numerical,
        "categorical": item_features.categorical,
    }

    id_maps = {
        "user_to_idx": user_to_idx,
        "idx_to_user": {str(k): v for k, v in idx_to_user.items()},
        "item_to_idx": item_to_idx,
        "idx_to_item": {str(k): v for k, v in idx_to_item.items()},
    }

    cat_vocab = {
        "user": user_features.cat_vocabs,
        "item": item_features.cat_vocabs,
    }

    print("[features] Saving outputs...")
    save_features(output_dir, train_pairs, user_features_npz, item_features_npz, feature_meta, id_maps, cat_vocab)

    result = FeatureResult(
        output_dir=output_dir,
        n_users=len(user_features.user_ids),
        n_items=len(item_features.item_ids),
        n_train_pairs=len(train_pairs["labels"]),
        n_user_num_features=user_features.numerical.shape[1],
        n_user_cat_features=user_features.categorical.shape[1],
        n_item_num_features=item_features.numerical.shape[1],
        n_item_cat_features=item_features.categorical.shape[1],
        user_cat_vocab_sizes={k: len(v) for k, v in user_features.cat_vocabs.items()},
        item_cat_vocab_sizes={k: len(v) for k, v in item_features.cat_vocabs.items()},
    )

    print(f"[features] Done. Output: {output_dir}")
    return result
