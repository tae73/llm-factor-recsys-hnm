"""BERTopic-style topic modeling: UMAP + HDBSCAN + c-TF-IDF.

Discovers data-driven customer segments from BGE embeddings, without
predefined k. Extracts interpretable keywords per topic via
class-based TF-IDF.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.config import SegmentationConfig

logger = logging.getLogger(__name__)


class TopicResult(NamedTuple):
    """Result of topic modeling."""

    labels: np.ndarray  # (N,) topic assignments (-1 = outlier before reassign)
    n_topics: int
    topic_keywords: dict[int, list[tuple[str, float]]]  # topic_id → [(word, score)]
    topic_sizes: dict[int, int]  # topic_id → count
    outlier_count: int  # original outlier count before reassignment
    umap_2d: np.ndarray  # (M, 2) for visualization


def fit_topics(
    embeddings: np.ndarray,
    texts: list[str],
    config: SegmentationConfig = SegmentationConfig(),
    reassign_outliers: bool = True,
) -> TopicResult:
    """Run BERTopic-style pipeline: UMAP → HDBSCAN → c-TF-IDF.

    Args:
        embeddings: BGE embeddings (N, 768).
        texts: Corresponding texts for c-TF-IDF keyword extraction.
        config: Segmentation config.
        reassign_outliers: Whether to reassign HDBSCAN outliers (-1) to nearest topic.

    Returns:
        TopicResult with labels, keywords, and visualization coordinates.
    """
    import hdbscan
    import umap

    n = len(embeddings)
    logger.info("Topic modeling: %d documents", n)

    # Subsample for UMAP + HDBSCAN if needed
    ss = config.subsample_size
    if n > ss:
        rng = np.random.RandomState(config.random_seed)
        sub_idx = rng.choice(n, ss, replace=False)
        emb_sub = embeddings[sub_idx]
        texts_sub = [texts[i] for i in sub_idx]
    else:
        sub_idx = np.arange(n)
        emb_sub = embeddings
        texts_sub = texts

    # Step 1: UMAP reduction for clustering (5D)
    logger.info("  UMAP reduction: %dD → %dD", emb_sub.shape[1], config.umap_cluster_n_components)
    reducer_cluster = umap.UMAP(
        n_components=config.umap_cluster_n_components,
        n_neighbors=config.umap_n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=config.random_seed,
    )
    X_cluster = reducer_cluster.fit_transform(emb_sub)

    # Step 2: HDBSCAN density-based clustering
    logger.info(
        "  HDBSCAN: min_cluster_size=%d, min_samples=%d",
        config.hdbscan_min_cluster_size,
        config.hdbscan_min_samples,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels_sub = clusterer.fit_predict(X_cluster)

    outlier_count = int(np.sum(labels_sub == -1))
    n_topics_raw = len(set(labels_sub)) - (1 if -1 in labels_sub else 0)
    logger.info("  Found %d topics, %d outliers (%.1f%%)", n_topics_raw, outlier_count, outlier_count / len(labels_sub) * 100)

    # Step 3: Reassign outliers to nearest topic centroid
    if reassign_outliers and outlier_count > 0 and n_topics_raw > 0:
        labels_sub = _reassign_outliers(X_cluster, labels_sub)

    # Step 4: c-TF-IDF for keyword extraction
    topic_keywords = _compute_ctfidf_keywords(texts_sub, labels_sub, top_n=10)

    # Step 5: UMAP 2D for visualization
    logger.info("  UMAP 2D for visualization")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric="cosine",
        random_state=config.random_seed,
    )
    umap_2d = reducer_2d.fit_transform(emb_sub)

    # If subsampled, propagate labels to full dataset via nearest centroid
    if n > ss:
        labels_full = _propagate_labels(embeddings, emb_sub, labels_sub)
    else:
        labels_full = labels_sub

    unique_labels = sorted(set(labels_full))
    topic_sizes = {int(lbl): int(np.sum(labels_full == lbl)) for lbl in unique_labels}
    n_topics = len(unique_labels)

    logger.info("Final: %d topics, sizes=%s", n_topics, dict(list(topic_sizes.items())[:5]))
    return TopicResult(
        labels=labels_full,
        n_topics=n_topics,
        topic_keywords=topic_keywords,
        topic_sizes=topic_sizes,
        outlier_count=outlier_count,
        umap_2d=umap_2d,
    )


def _reassign_outliers(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reassign outlier points (-1) to nearest topic centroid."""
    labels = labels.copy()
    valid_mask = labels >= 0
    unique_topics = np.unique(labels[valid_mask])

    centroids = np.array([X[labels == t].mean(axis=0) for t in unique_topics])

    outlier_mask = labels == -1
    outlier_points = X[outlier_mask]

    if len(outlier_points) > 0:
        # Compute distances to centroids
        dists = np.linalg.norm(outlier_points[:, None] - centroids[None, :], axis=2)
        nearest = unique_topics[np.argmin(dists, axis=1)]
        labels[outlier_mask] = nearest

    return labels


def _compute_ctfidf_keywords(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    """Compute class-based TF-IDF keywords per topic.

    Concatenates all documents per topic, then applies TF-IDF
    with class-level IDF weighting.
    """
    unique_labels = sorted(set(labels))
    # Concatenate texts per topic
    topic_docs = {}
    for lbl in unique_labels:
        mask = labels == lbl
        topic_docs[lbl] = " ".join(t for t, m in zip(texts, mask) if m)

    if not topic_docs:
        return {}

    # Fit CountVectorizer + TF-IDF
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
    )
    ordered_labels = sorted(topic_docs.keys())
    docs = [topic_docs[lbl] for lbl in ordered_labels]

    try:
        count_matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return {lbl: [] for lbl in unique_labels}

    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
    tfidf_matrix = tfidf.fit_transform(count_matrix)

    feature_names = vectorizer.get_feature_names_out()
    result: dict[int, list[tuple[str, float]]] = {}

    for i, lbl in enumerate(ordered_labels):
        scores = tfidf_matrix[i].toarray().flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        keywords = [(str(feature_names[idx]), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        result[lbl] = keywords

    return result


def _propagate_labels(
    X_full: np.ndarray,
    X_sub: np.ndarray,
    labels_sub: np.ndarray,
) -> np.ndarray:
    """Propagate cluster labels from subsample to full dataset via nearest neighbor."""
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(X_sub, labels_sub)
    return knn.predict(X_full)
