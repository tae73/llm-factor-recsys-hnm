"""Customer segmentation orchestration: L1/L2/L3/Semantic/Topic.

Runs 5-level segmentation pipeline and saves consolidated results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.config import SegmentationConfig
from src.embeddings import load_embeddings
from src.segmentation.clustering import (
    ClusterResult,
    compute_umap_2d,
    fit_clusters,
    reduce_pca,
    select_k,
)
from src.segmentation.topics import TopicResult, fit_topics
from src.segmentation.vectorizer import vectorize_l1, vectorize_l2, vectorize_l3

logger = logging.getLogger(__name__)


class CustomerSegmentResult(NamedTuple):
    """Result of full customer segmentation."""

    l1_cluster: ClusterResult
    l2_cluster: ClusterResult
    l3_cluster: ClusterResult
    semantic_cluster: ClusterResult
    topic: TopicResult
    customer_ids: np.ndarray
    segments_df: pd.DataFrame  # customer_id + 5 segment columns


def run_customer_segmentation(
    rk_path: Path,
    txn_path: Path,
    fk_path: Path,
    user_emb_path: Path,
    output_dir: Path,
    config: SegmentationConfig = SegmentationConfig(),
) -> CustomerSegmentResult:
    """Run 5-level customer segmentation pipeline.

    Levels:
      1. L1: Structured product/color/material vectors → PCA → K-Means
      2. L2: Structured perceptual vectors → PCA → K-Means
      3. L3: Structured theory vectors → PCA → K-Means
      4. Semantic: BGE user embeddings → PCA → K-Means
      5. Topic: BGE user embeddings → UMAP → HDBSCAN → c-TF-IDF

    Args:
        rk_path: Path to user_profiles.parquet.
        txn_path: Path to transactions.parquet.
        fk_path: Path to factual_knowledge.parquet.
        user_emb_path: Path to user_bge_embeddings.npz.
        output_dir: Output directory.
        config: Segmentation config.

    Returns:
        CustomerSegmentResult with all 5 level results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Level 1: L1 ---
    logger.info("=== L1 Segmentation ===")
    l1_result = vectorize_l1(rk_path)
    np.savez_compressed(output_dir / "customer_l1_vectors.npz", vectors=l1_result.vectors, customer_ids=l1_result.customer_ids)
    l1_pca = reduce_pca(l1_result.vectors, variance_threshold=config.pca_variance_threshold)
    l1_ksel = select_k(l1_pca.X_reduced, config.customer_k_range, subsample_size=config.subsample_size, random_seed=config.random_seed)
    l1_cluster = fit_clusters(l1_pca.X_reduced, l1_ksel.best_k, random_seed=config.random_seed)

    # --- Level 2: L2 ---
    logger.info("=== L2 Segmentation ===")
    l2_result = vectorize_l2(txn_path, fk_path)
    np.savez_compressed(output_dir / "customer_l2_vectors.npz", vectors=l2_result.vectors, customer_ids=l2_result.customer_ids)
    l2_pca = reduce_pca(l2_result.vectors, variance_threshold=config.pca_variance_threshold)
    l2_ksel = select_k(l2_pca.X_reduced, config.customer_k_range, subsample_size=config.subsample_size, random_seed=config.random_seed)
    l2_cluster = fit_clusters(l2_pca.X_reduced, l2_ksel.best_k, random_seed=config.random_seed)

    # --- Level 3: L3 ---
    logger.info("=== L3 Segmentation ===")
    l3_result = vectorize_l3(txn_path, fk_path)
    np.savez_compressed(output_dir / "customer_l3_vectors.npz", vectors=l3_result.vectors, customer_ids=l3_result.customer_ids)
    l3_pca = reduce_pca(l3_result.vectors, variance_threshold=config.pca_variance_threshold)
    l3_ksel = select_k(l3_pca.X_reduced, config.customer_k_range, subsample_size=config.subsample_size, random_seed=config.random_seed)
    l3_cluster = fit_clusters(l3_pca.X_reduced, l3_ksel.best_k, random_seed=config.random_seed)

    # --- Level 4: Semantic ---
    logger.info("=== Semantic Segmentation ===")
    user_emb, user_ids = load_embeddings(user_emb_path)

    # BGE isotropy correction: subtract mean embedding to reduce anisotropy
    emb_mean = user_emb.mean(axis=0, keepdims=True)
    user_emb_centered = user_emb - emb_mean
    logger.info("BGE isotropy correction: mean norm=%.6f", np.linalg.norm(emb_mean))

    sem_pca = reduce_pca(user_emb_centered, max_components=50, standardize=False)
    sem_ksel = select_k(sem_pca.X_reduced, config.customer_k_range, subsample_size=config.subsample_size, random_seed=config.random_seed)
    sem_cluster = fit_clusters(sem_pca.X_reduced, sem_ksel.best_k, random_seed=config.random_seed)

    # --- Level 5: Topic ---
    logger.info("=== Topic Segmentation ===")
    profiles_df = pd.read_parquet(rk_path, columns=["customer_id", "reasoning_text"])
    texts = profiles_df["reasoning_text"].fillna("").tolist()
    topic_result = fit_topics(user_emb_centered, texts, config=config)

    # --- Build consolidated DataFrame ---
    # Use L1 customer_ids as the base (from profiles)
    base_ids = l1_result.customer_ids

    # Map each level's labels to the base customer IDs
    segments_df = pd.DataFrame({"customer_id": base_ids})

    # L1 labels — same order as profiles
    segments_df["l1_segment"] = l1_cluster.labels

    # L2/L3 — may have different customer order (sorted by DuckDB)
    l2_map = dict(zip(l2_result.customer_ids, l2_cluster.labels))
    l3_map = dict(zip(l3_result.customer_ids, l3_cluster.labels))
    sem_map = dict(zip(user_ids, sem_cluster.labels))
    topic_map = dict(zip(user_ids, topic_result.labels))

    segments_df["l2_segment"] = segments_df["customer_id"].map(l2_map).fillna(-1).astype(int)
    segments_df["l3_segment"] = segments_df["customer_id"].map(l3_map).fillna(-1).astype(int)
    segments_df["semantic_segment"] = segments_df["customer_id"].map(sem_map).fillna(-1).astype(int)
    segments_df["topic_segment"] = segments_df["customer_id"].map(topic_map).fillna(-1).astype(int)

    # Save
    segments_df.to_parquet(output_dir / "customer_segments.parquet", index=False)
    logger.info("Saved customer_segments.parquet: %d rows × %d cols", len(segments_df), len(segments_df.columns))

    # Save clustering metadata
    meta = {
        "l1": {"k": l1_cluster.k, "silhouette": l1_cluster.silhouette, "k_scores": l1_ksel.scores},
        "l2": {"k": l2_cluster.k, "silhouette": l2_cluster.silhouette, "k_scores": l2_ksel.scores},
        "l3": {"k": l3_cluster.k, "silhouette": l3_cluster.silhouette, "k_scores": l3_ksel.scores},
        "semantic": {"k": sem_cluster.k, "silhouette": sem_cluster.silhouette, "k_scores": sem_ksel.scores},
        "topic": {
            "n_topics": topic_result.n_topics,
            "outlier_count": topic_result.outlier_count,
            "topic_sizes": topic_result.topic_sizes,
        },
    }
    with open(output_dir / "clustering_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return CustomerSegmentResult(
        l1_cluster=l1_cluster,
        l2_cluster=l2_cluster,
        l3_cluster=l3_cluster,
        semantic_cluster=sem_cluster,
        topic=topic_result,
        customer_ids=base_ids,
        segments_df=segments_df,
    )
