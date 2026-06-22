"""User-side enrichment representations + FUTURE-behavior outcomes (E2-4, KAR reasoning leg).

The value matrix's ① control and ④ audience cells are USER decisions; the item-side axes
can't reach them. This module assembles the **user/reasoning** leg of KAR so those cells
can be tested with user enrichment.

THREE train-derived user representations (aligned on a common customer_id order):
  - ``reasoning_bge``    — PCA-50 of the BGE(reasoning_text) embedding (the ReasoningExpert input).
  - ``reasoning_fields`` — TF-IDF→SVD of the 9 LLM preference prose fields + L1-aggregate numerics.
  - ``demographic``      — the 11 metadata user features (the BASELINE reasoning must beat).

THE HONESTY FIX: ``reasoning_*`` is built from the user's TRAIN purchases, so testing it on
train behavior is tautological. Every outcome here is **held-out FUTURE behavior** (val
2020-07+), and the demographic baseline is equally train-derived → reasoning-vs-demographics
is apples-to-apples on PREDICTING THE FUTURE. CPU/DuckDB only, seed 42, no API.
"""

from __future__ import annotations

import json
import logging

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42
PROFILES = "data/knowledge/reasoning/user_profiles.parquet"
USER_BGE = "data/embeddings/user_bge_embeddings.npz"
USER_FEAT = "data/features/user_features.npz"
FEAT_META = "data/features/feature_meta.json"
ID_MAPS = "data/features/id_maps.json"
VAL_TXN = "data/processed/val_transactions.parquet"
TRAIN_TXN = "data/processed/train_transactions.parquet"
ARTICLES = "data/processed/articles.parquet"

REASONING_FIELDS = [
    "style_mood_preference",
    "occasion_preference",
    "quality_price_tendency",
    "trend_sensitivity",
    "seasonal_pattern",
    "form_preference",
    "color_tendency",
    "coordination_tendency",
    "identity_summary",
]


# ---------------------------------------------------------------------------
# Cohort: active-LLM users who purchased in the FUTURE (val) window
# ---------------------------------------------------------------------------
def select_cohort(n_sample: int = 40000, seed: int = SEED) -> list[str]:
    """Active-LLM users who appear in val_transactions (so future outcomes are defined)."""
    con = duckdb.connect()
    df = con.execute(f"""
        WITH val_users AS (SELECT DISTINCT customer_id FROM read_parquet('{VAL_TXN}'))
        SELECT p.customer_id
        FROM read_parquet('{PROFILES}') p
        JOIN val_users v ON p.customer_id = v.customer_id
        WHERE p.is_active = TRUE AND p.profile_source = 'llm'
        ORDER BY p.customer_id
        """).fetchdf()
    con.close()
    ids = df["customer_id"].astype(str).tolist()
    if n_sample < len(ids):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.permutation(len(ids))[:n_sample])
        ids = [ids[i] for i in idx]
    logger.info("cohort: %d active-LLM users with val purchases", len(ids))
    return ids


# ---------------------------------------------------------------------------
# Representations (all aligned to `customer_ids` order)
# ---------------------------------------------------------------------------
def _pca(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    k = min(k, X.shape[1], X.shape[0])
    return PCA(n_components=k, random_state=SEED).fit_transform(X.astype(np.float64))


def build_reasoning_bge(customer_ids: list[str]) -> np.ndarray:
    """PCA-50 of isotropy-corrected BGE(reasoning_text) for the cohort."""
    d = np.load(USER_BGE, allow_pickle=True)
    pos = {str(c): i for i, c in enumerate(d["customer_ids"].astype(str))}
    sel = np.array([pos[c] for c in customer_ids])
    emb = d["embeddings"][sel].astype(np.float32)
    emb = emb - emb.mean(axis=0, keepdims=True)  # isotropy correction (matches segmentation)
    return _pca(emb, 50)


def build_reasoning_fields(customer_ids: list[str]) -> np.ndarray:
    """TF-IDF→SVD-50 of the 9 prose fields + L1-aggregate numerics (cohort-aligned)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    p = pd.read_parquet(
        PROFILES,
        columns=[
            "customer_id",
            "reasoning_json",
            "avg_price_quintile",
            "online_ratio",
            "category_diversity",
            "n_purchases",
        ],
    )
    p["customer_id"] = p["customer_id"].astype(str)
    p = p.set_index("customer_id").reindex(customer_ids).reset_index()
    prose = [
        (
            " ".join(str(json.loads(j).get(f, "")) for f in REASONING_FIELDS)
            if isinstance(j, str)
            else ""
        )
        for j in p["reasoning_json"]
    ]
    tfidf = TfidfVectorizer(max_features=4000, stop_words="english", min_df=5)
    Xt = tfidf.fit_transform(prose)
    svd = TruncatedSVD(n_components=50, random_state=SEED).fit_transform(Xt)
    agg = StandardScaler().fit_transform(
        p[["avg_price_quintile", "online_ratio", "category_diversity", "n_purchases"]]
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    return np.concatenate([svd, agg], axis=1)


def build_demographic(customer_ids: list[str]) -> np.ndarray:
    """The 11 metadata user features → 8 standardized numeric + one-hot categoricals (BASELINE)."""
    from sklearn.preprocessing import StandardScaler

    f = np.load(USER_FEAT, allow_pickle=True)
    meta = json.load(open(FEAT_META))
    u2i = json.load(open(ID_MAPS))["user_to_idx"]
    idx = np.array([u2i[c] for c in customer_ids])
    num = StandardScaler().fit_transform(f["numerical"][idx].astype(np.float64))
    cat = f["categorical"][idx]  # (n, 3) int codes
    sizes = [meta["user_cat_vocab_sizes"][c] for c in meta["user_cat_names"]]
    onehots = [
        np.eye(sz, dtype=np.float64)[np.clip(cat[:, j], 0, sz - 1)] for j, sz in enumerate(sizes)
    ]
    return np.concatenate([num, *onehots], axis=1)


def build_user_representations(customer_ids: list[str]) -> dict[str, np.ndarray]:
    reps = {
        "reasoning_bge": build_reasoning_bge(customer_ids),
        "reasoning_fields": build_reasoning_fields(customer_ids),
        "demographic": build_demographic(customer_ids),
    }
    for k, v in reps.items():
        logger.info("rep %s: %s", k, v.shape)
    return reps


# ---------------------------------------------------------------------------
# FUTURE behavior outcomes (val window; train-frozen tiers)
# ---------------------------------------------------------------------------
def build_future_outcomes(customer_ids: list[str]) -> pd.DataFrame:
    """Per-user FUTURE (val) behavior labels — outcome touches ONLY val/train-history, never train txn for the predictor."""
    con = duckdb.connect()
    # train-frozen price-tier edges (per-user train avg price quintiles)
    edges = con.execute(f"""
        WITH up AS (SELECT customer_id, AVG(price) ap FROM read_parquet('{TRAIN_TXN}') GROUP BY customer_id)
        SELECT quantile_cont(ap, [0.2,0.4,0.6,0.8]) FROM up
        """).fetchone()[0]
    edges = list(edges)
    # per-user val aggregates (deterministic fut_top_group via tie-broken ROW_NUMBER, NOT MODE)
    val = con.execute(f"""
        WITH j AS (
            SELECT CAST(t.customer_id AS VARCHAR) AS cid, t.price AS price,
                   t.sales_channel_id AS ch, a.product_type_name AS ptype, a.product_group_name AS pg
            FROM read_parquet('{VAL_TXN}') t
            JOIN read_parquet('{ARTICLES}') a ON CAST(t.article_id AS VARCHAR) = CAST(a.article_id AS VARCHAR)
        ),
        agg AS (
            SELECT cid, AVG(price) AS val_avg_price,
                   AVG(CASE WHEN ch = 2 THEN 1.0 ELSE 0.0 END) AS val_online,
                   COUNT(DISTINCT ptype) AS val_n_types, COUNT(*) AS n_val
            FROM j GROUP BY cid
        ),
        grp AS (SELECT cid, pg, COUNT(*) c FROM j GROUP BY cid, pg),
        topg AS (
            SELECT cid, pg, ROW_NUMBER() OVER (PARTITION BY cid ORDER BY c DESC, pg) rn FROM grp
        )
        SELECT a.cid AS customer_id, a.val_avg_price, a.val_online, a.val_n_types, a.n_val,
               tg.pg AS fut_top_group
        FROM agg a JOIN topg tg ON a.cid = tg.cid AND tg.rn = 1
        """).fetchdf()
    val["customer_id"] = val["customer_id"].astype(str)
    # future repurchase: val article ∈ user's train history
    rep = con.execute(f"""
        WITH th AS (SELECT CAST(customer_id AS VARCHAR) c, CAST(article_id AS VARCHAR) a FROM read_parquet('{TRAIN_TXN}')),
             vh AS (SELECT CAST(customer_id AS VARCHAR) c, CAST(article_id AS VARCHAR) a FROM read_parquet('{VAL_TXN}'))
        SELECT vh.c AS customer_id, MAX(CASE WHEN th.a IS NOT NULL THEN 1 ELSE 0 END) AS fut_repurchase
        FROM vh LEFT JOIN th ON vh.c = th.c AND vh.a = th.a GROUP BY vh.c
        """).fetchdf()
    rep["customer_id"] = rep["customer_id"].astype(str)
    con.close()

    df = (
        pd.DataFrame({"customer_id": customer_ids})
        .merge(val, on="customer_id", how="left")
        .merge(rep, on="customer_id", how="left")
    )
    df["fut_price_tier"] = np.digitize(df["val_avg_price"].fillna(-1).to_numpy(), edges).astype(int)
    df["fut_online"] = (df["val_online"] > 0.5).astype(int)
    df["fut_n_types_tertile"] = pd.qcut(
        df["val_n_types"].rank(method="first"), 3, labels=False
    ).astype("Int64")
    df["fut_repurchase"] = df["fut_repurchase"].fillna(0).astype(int)
    logger.info("future outcomes for %d users", len(df))
    return df
