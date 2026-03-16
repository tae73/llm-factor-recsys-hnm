"""Clustering utilities: PCA, UMAP, K-Means, silhouette-based k selection.

Pure functions operating on numpy arrays, returning NamedTuples.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import SegmentationConfig

logger = logging.getLogger(__name__)


class ClusterResult(NamedTuple):
    """Result of clustering."""

    labels: np.ndarray  # (N,) int cluster assignments
    k: int
    centroids: np.ndarray  # (k, d)
    silhouette: float
    inertia: float


class KSelectionResult(NamedTuple):
    """Result of k selection search."""

    best_k: int
    scores: dict[int, float]  # k -> silhouette score
    inertias: dict[int, float]  # k -> inertia


class DimReduceResult(NamedTuple):
    """Result of dimensionality reduction."""

    X_reduced: np.ndarray
    n_components: int
    explained_variance_ratio: np.ndarray | None  # PCA only


def select_k(
    X: np.ndarray,
    k_range: tuple[int, ...] | list[int],
    method: str = "kmeans",
    subsample_size: int = 50_000,
    random_seed: int = 42,
) -> KSelectionResult:
    """Find optimal k via silhouette score on subsampled data.

    Args:
        X: Input features (N, d).
        k_range: Candidate k values.
        method: "kmeans" (only supported method).
        subsample_size: Max samples for silhouette computation.
        random_seed: Random seed.

    Returns:
        KSelectionResult with best_k and per-k scores.
    """
    rng = np.random.RandomState(random_seed)
    if len(X) > subsample_size:
        indices = rng.choice(len(X), subsample_size, replace=False)
        X_sub = X[indices]
    else:
        X_sub = X

    scores: dict[int, float] = {}
    inertias: dict[int, float] = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_seed, n_init=10, max_iter=300)
        labels = km.fit_predict(X_sub)
        sil = silhouette_score(X_sub, labels, sample_size=min(10_000, len(X_sub)))
        scores[k] = float(sil)
        inertias[k] = float(km.inertia_)
        logger.info("  k=%d  silhouette=%.4f  inertia=%.1f", k, sil, km.inertia_)

    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    logger.info("Best k=%d (silhouette=%.4f)", best_k, scores[best_k])
    return KSelectionResult(best_k=best_k, scores=scores, inertias=inertias)


def fit_clusters(
    X: np.ndarray,
    k: int,
    random_seed: int = 42,
) -> ClusterResult:
    """Fit K-Means clustering.

    Args:
        X: Input features (N, d).
        k: Number of clusters.
        random_seed: Random seed.

    Returns:
        ClusterResult with labels, centroids, silhouette.
    """
    km = KMeans(n_clusters=k, random_state=random_seed, n_init=10, max_iter=300)
    labels = km.fit_predict(X)

    # Silhouette on subsample for large datasets
    sample_size = min(50_000, len(X))
    sil = float(silhouette_score(X, labels, sample_size=sample_size))

    logger.info("Clustered %d samples into %d clusters, silhouette=%.4f", len(X), k, sil)
    return ClusterResult(
        labels=labels,
        k=k,
        centroids=km.cluster_centers_,
        silhouette=sil,
        inertia=float(km.inertia_),
    )


def reduce_pca(
    X: np.ndarray,
    variance_threshold: float = 0.95,
    max_components: int | None = None,
    standardize: bool = True,
    whiten: bool = True,
) -> DimReduceResult:
    """PCA dimensionality reduction preserving target variance.

    Args:
        X: Input features (N, d).
        variance_threshold: Cumulative variance to preserve.
        max_components: Maximum components (overrides variance threshold).
        standardize: Apply StandardScaler before PCA (recommended for mixed-scale features).
        whiten: Apply PCA whitening to normalize principal component variances.

    Returns:
        DimReduceResult with reduced features.
    """
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        logger.info("StandardScaler applied: %d features", X.shape[1])

    if max_components is not None:
        n_comp = min(max_components, X.shape[1], X.shape[0])
    else:
        n_comp = min(X.shape[1], X.shape[0])

    pca = PCA(n_components=n_comp, whiten=whiten)
    X_pca = pca.fit_transform(X)

    if max_components is None:
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_keep = max(2, min(n_keep, n_comp))
        X_pca = X_pca[:, :n_keep]
        logger.info(
            "PCA: %d → %d dims (%.1f%% variance, whiten=%s)",
            X.shape[1],
            n_keep,
            cumvar[n_keep - 1] * 100,
            whiten,
        )
    else:
        n_keep = n_comp
        logger.info("PCA: %d → %d dims (whiten=%s)", X.shape[1], n_keep, whiten)

    return DimReduceResult(
        X_reduced=X_pca,
        n_components=n_keep,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def compute_umap_2d(
    X: np.ndarray,
    config: SegmentationConfig = SegmentationConfig(),
    subsample_size: int | None = None,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Compute 2D UMAP embedding for visualization.

    For large datasets, subsamples with stratified sampling if labels provided.

    Args:
        X: Input features (N, d).
        config: Segmentation config.
        subsample_size: Max samples (None = use config.subsample_size).
        labels: Optional cluster labels for stratified sampling.

    Returns:
        UMAP 2D coordinates (M, 2) where M <= subsample_size.
    """
    import umap

    ss = subsample_size or config.subsample_size
    if len(X) > ss:
        indices = _stratified_subsample(len(X), ss, labels, config.random_seed)
        X_sub = X[indices]
    else:
        X_sub = X

    reducer = umap.UMAP(
        n_components=config.umap_n_components,
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        random_state=config.random_seed,
        metric="cosine",
    )
    X_2d = reducer.fit_transform(X_sub)
    logger.info("UMAP 2D: %d samples → shape %s", len(X_sub), X_2d.shape)
    return X_2d


def _stratified_subsample(
    n: int,
    target: int,
    labels: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    """Subsample with optional stratification by cluster labels."""
    rng = np.random.RandomState(seed)
    if labels is None:
        return rng.choice(n, target, replace=False)

    unique_labels = np.unique(labels)
    per_label = max(1, target // len(unique_labels))
    indices = []
    for lbl in unique_labels:
        mask = np.where(labels == lbl)[0]
        take = min(per_label, len(mask))
        indices.extend(rng.choice(mask, take, replace=False).tolist())

    # Fill remaining
    remaining = target - len(indices)
    if remaining > 0:
        all_idx = set(range(n)) - set(indices)
        indices.extend(rng.choice(list(all_idx), min(remaining, len(all_idx)), replace=False).tolist())

    return np.array(indices[:target])
