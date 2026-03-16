"""Product clustering using BGE embeddings.

Clusters items by LLM-extracted semantic attributes, compares with
H&M native categories (ARI), and detects cross-category similar items.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.config import SegmentationConfig
from src.embeddings import load_embeddings
from src.segmentation.clustering import (
    ClusterResult,
    fit_clusters,
    reduce_pca,
    select_k,
)

logger = logging.getLogger(__name__)


class ProductClusterResult(NamedTuple):
    """Result of product clustering."""

    cluster: ClusterResult
    ari_vs_native: float  # ARI between LLM clusters and H&M product_type
    cross_category_pairs: pd.DataFrame  # similar items across different product_types
    clusters_df: pd.DataFrame  # article_id → cluster_id + metadata


def run_product_clustering(
    item_emb_path: Path,
    articles_path: Path,
    output_dir: Path,
    config: SegmentationConfig = SegmentationConfig(),
) -> ProductClusterResult:
    """Run product clustering pipeline.

    Args:
        item_emb_path: Path to item_bge_embeddings.npz.
        articles_path: Path to articles.parquet.
        output_dir: Output directory.
        config: Segmentation config.

    Returns:
        ProductClusterResult.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings and metadata
    embeddings, article_ids = load_embeddings(item_emb_path)
    articles_df = pd.read_parquet(
        articles_path,
        columns=["article_id", "product_type_name", "product_group_name", "garment_group_name"],
    )

    # BGE isotropy correction: subtract mean embedding to reduce anisotropy
    emb_mean = embeddings.mean(axis=0, keepdims=True)
    embeddings_centered = embeddings - emb_mean
    logger.info("BGE isotropy correction: mean norm=%.6f", np.linalg.norm(emb_mean))

    # PCA + k-selection + clustering
    logger.info("=== Product Clustering ===")
    pca_result = reduce_pca(embeddings_centered, max_components=50, standardize=False)
    ksel = select_k(
        pca_result.X_reduced,
        config.product_k_range,
        subsample_size=config.subsample_size,
        random_seed=config.random_seed,
    )
    cluster = fit_clusters(pca_result.X_reduced, ksel.best_k, random_seed=config.random_seed)

    # Build output DataFrame
    clusters_df = pd.DataFrame({
        "article_id": article_ids,
        "cluster_id": cluster.labels,
    })
    clusters_df = clusters_df.merge(articles_df, on="article_id", how="left")

    # ARI vs native categories
    merged = clusters_df.dropna(subset=["product_type_name"])
    if len(merged) > 0:
        # Encode product_type_name as integers
        pt_codes = pd.Categorical(merged["product_type_name"]).codes
        ari = adjusted_rand_score(pt_codes, merged["cluster_id"].values)
    else:
        ari = 0.0
    logger.info("ARI(LLM clusters, product_type): %.4f", ari)

    # Cross-category similar items (FAISS cosine)
    cross_pairs = _find_cross_category_pairs(
        embeddings, article_ids, articles_df, threshold=0.85, max_pairs=500
    )
    logger.info("Cross-category pairs: %d (cos > 0.85)", len(cross_pairs))

    # Save
    clusters_df.to_parquet(output_dir / "product_clusters.parquet", index=False)
    cross_pairs.to_parquet(output_dir / "cross_category_pairs.parquet", index=False)

    return ProductClusterResult(
        cluster=cluster,
        ari_vs_native=ari,
        cross_category_pairs=cross_pairs,
        clusters_df=clusters_df,
    )


def _find_cross_category_pairs(
    embeddings: np.ndarray,
    article_ids: np.ndarray,
    articles_df: pd.DataFrame,
    threshold: float = 0.85,
    max_pairs: int = 500,
    k_neighbors: int = 20,
) -> pd.DataFrame:
    """Find similar item pairs across different product_types using FAISS.

    Returns DataFrame with columns: article_id_1, article_id_2, similarity,
    product_type_1, product_type_2.
    """
    try:
        import faiss
    except ImportError:
        logger.warning("faiss-cpu not installed, skipping cross-category detection")
        return pd.DataFrame(columns=["article_id_1", "article_id_2", "similarity", "product_type_1", "product_type_2"])

    # Build article_id → product_type mapping
    id_to_type = dict(zip(articles_df["article_id"], articles_df["product_type_name"]))

    # Normalize for cosine similarity
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    emb_norm = emb_norm.astype(np.float32)

    # FAISS inner product index
    d = emb_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb_norm)

    # Search k nearest neighbors
    D, I = index.search(emb_norm, k_neighbors + 1)  # +1 for self

    pairs = []
    seen = set()
    for i in range(len(article_ids)):
        aid_i = article_ids[i]
        pt_i = id_to_type.get(aid_i)
        if pt_i is None:
            continue

        for j_pos in range(1, k_neighbors + 1):  # skip self at position 0
            j = I[i, j_pos]
            sim = float(D[i, j_pos])
            if sim < threshold:
                break

            aid_j = article_ids[j]
            pt_j = id_to_type.get(aid_j)
            if pt_j is None or pt_i == pt_j:
                continue

            pair_key = tuple(sorted([str(aid_i), str(aid_j)]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            pairs.append({
                "article_id_1": aid_i,
                "article_id_2": aid_j,
                "similarity": sim,
                "product_type_1": pt_i,
                "product_type_2": pt_j,
            })

            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    return pd.DataFrame(pairs)
