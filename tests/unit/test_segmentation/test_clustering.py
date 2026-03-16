"""Unit tests for src/segmentation/clustering.py."""

import numpy as np
import pytest

from src.config import SegmentationConfig
from src.segmentation.clustering import (
    ClusterResult,
    DimReduceResult,
    KSelectionResult,
    _stratified_subsample,
    compute_umap_2d,
    fit_clusters,
    reduce_pca,
    select_k,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data() -> np.ndarray:
    """3 well-separated clusters in 10D."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(50, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    c2 = rng.randn(50, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    c3 = rng.randn(50, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
    return np.vstack([c1, c2, c3]).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: select_k
# ---------------------------------------------------------------------------


def test_select_k_finds_correct_k(synthetic_data):
    result = select_k(synthetic_data, k_range=(2, 3, 4, 5), random_seed=42)
    assert isinstance(result, KSelectionResult)
    assert result.best_k == 3  # 3 true clusters


def test_select_k_returns_all_scores(synthetic_data):
    result = select_k(synthetic_data, k_range=(2, 3, 4), random_seed=42)
    assert len(result.scores) == 3
    assert len(result.inertias) == 3


def test_select_k_subsample():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5).astype(np.float32)
    result = select_k(X, k_range=(2, 3), subsample_size=50, random_seed=42)
    assert result.best_k in (2, 3)


# ---------------------------------------------------------------------------
# Tests: fit_clusters
# ---------------------------------------------------------------------------


def test_fit_clusters_labels_count(synthetic_data):
    result = fit_clusters(synthetic_data, k=3, random_seed=42)
    assert isinstance(result, ClusterResult)
    assert len(np.unique(result.labels)) == 3
    assert result.k == 3


def test_fit_clusters_centroids_shape(synthetic_data):
    result = fit_clusters(synthetic_data, k=3, random_seed=42)
    assert result.centroids.shape == (3, 10)


def test_fit_clusters_silhouette_positive(synthetic_data):
    result = fit_clusters(synthetic_data, k=3, random_seed=42)
    assert result.silhouette > 0.3  # Well-separated clusters


def test_fit_clusters_deterministic(synthetic_data):
    r1 = fit_clusters(synthetic_data, k=3, random_seed=42)
    r2 = fit_clusters(synthetic_data, k=3, random_seed=42)
    np.testing.assert_array_equal(r1.labels, r2.labels)


# ---------------------------------------------------------------------------
# Tests: reduce_pca
# ---------------------------------------------------------------------------


def test_reduce_pca_variance_threshold(synthetic_data):
    result = reduce_pca(synthetic_data, variance_threshold=0.95)
    assert isinstance(result, DimReduceResult)
    assert result.n_components <= synthetic_data.shape[1]
    assert result.X_reduced.shape[0] == len(synthetic_data)
    assert result.X_reduced.shape[1] == result.n_components


def test_reduce_pca_max_components(synthetic_data):
    result = reduce_pca(synthetic_data, max_components=3)
    assert result.n_components == 3
    assert result.X_reduced.shape == (150, 3)


def test_reduce_pca_variance_ratio(synthetic_data):
    result = reduce_pca(synthetic_data, variance_threshold=0.95)
    assert result.explained_variance_ratio is not None
    assert len(result.explained_variance_ratio) >= result.n_components


def test_reduce_pca_with_standardize(synthetic_data):
    result = reduce_pca(synthetic_data, variance_threshold=0.95, standardize=True, whiten=False)
    assert result.X_reduced.shape[0] == len(synthetic_data)
    assert result.n_components >= 2


def test_reduce_pca_with_whiten(synthetic_data):
    result = reduce_pca(synthetic_data, variance_threshold=0.95, standardize=True, whiten=True)
    # Whitened PCA: each component should have ~unit variance
    variances = np.var(result.X_reduced, axis=0)
    np.testing.assert_allclose(variances, 1.0, atol=0.15)


def test_reduce_pca_no_standardize_no_whiten(synthetic_data):
    """Explicit standardize=False, whiten=False reproduces legacy behavior."""
    result = reduce_pca(synthetic_data, variance_threshold=0.95, standardize=False, whiten=False)
    assert result.X_reduced.shape[0] == len(synthetic_data)
    assert result.n_components >= 2


def test_reduce_pca_backward_compat(synthetic_data):
    """Default params (standardize=True, whiten=True) should not raise."""
    result = reduce_pca(synthetic_data, variance_threshold=0.95)
    assert result.n_components >= 2


# ---------------------------------------------------------------------------
# Tests: _stratified_subsample
# ---------------------------------------------------------------------------


def test_stratified_subsample_without_labels():
    indices = _stratified_subsample(100, 30, None, seed=42)
    assert len(indices) == 30
    assert len(set(indices)) == 30


def test_stratified_subsample_with_labels():
    labels = np.array([0] * 50 + [1] * 50)
    indices = _stratified_subsample(100, 20, labels, seed=42)
    assert len(indices) == 20
