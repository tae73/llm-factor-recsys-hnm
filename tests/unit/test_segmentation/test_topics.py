"""Unit tests for src/segmentation/topics.py."""

import numpy as np
import pytest

from src.config import SegmentationConfig
from src.segmentation.topics import (
    TopicResult,
    _compute_ctfidf_keywords,
    _reassign_outliers,
    fit_topics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_embeddings_and_texts():
    """3 synthetic topics with distinct vocabulary."""
    rng = np.random.RandomState(42)
    n_per = 100

    # 3 clusters in 768D (simulating BGE embeddings)
    c1 = rng.randn(n_per, 768).astype(np.float32) + 2.0
    c2 = rng.randn(n_per, 768).astype(np.float32) - 2.0
    c3 = rng.randn(n_per, 768).astype(np.float32) * 0.1

    embeddings = np.vstack([c1, c2, c3])

    texts = (
        ["casual sporty everyday comfort active wear"] * n_per
        + ["formal elegant luxury classic tailored suit"] * n_per
        + ["bohemian vintage retro eclectic free spirit"] * n_per
    )

    return embeddings, texts


# ---------------------------------------------------------------------------
# Tests: _reassign_outliers
# ---------------------------------------------------------------------------


def test_reassign_outliers_removes_negatives():
    X = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [10, 11]], dtype=np.float32)
    labels = np.array([0, 0, -1, 1, 1])
    result = _reassign_outliers(X, labels)
    assert -1 not in result
    assert result[2] == 0  # (0,1) closer to cluster 0 center (0.5, 0)


def test_reassign_outliers_preserves_valid():
    X = np.array([[0, 0], [1, 0], [10, 10]], dtype=np.float32)
    labels = np.array([0, 0, 1])
    result = _reassign_outliers(X, labels)
    np.testing.assert_array_equal(result, labels)


# ---------------------------------------------------------------------------
# Tests: _compute_ctfidf_keywords
# ---------------------------------------------------------------------------


def test_ctfidf_keywords_returns_dict():
    texts = ["cat dog fish"] * 10 + ["car plane train"] * 10
    labels = np.array([0] * 10 + [1] * 10)
    keywords = _compute_ctfidf_keywords(texts, labels, top_n=3)
    assert isinstance(keywords, dict)
    assert 0 in keywords
    assert 1 in keywords


def test_ctfidf_keywords_has_tuples():
    texts = ["red blue green yellow"] * 20 + ["alpha beta gamma"] * 20
    labels = np.array([0] * 20 + [1] * 20)
    keywords = _compute_ctfidf_keywords(texts, labels, top_n=3)
    for lbl, kw_list in keywords.items():
        for word, score in kw_list:
            assert isinstance(word, str)
            assert isinstance(score, float)
            assert score >= 0


def test_ctfidf_keywords_empty_texts():
    keywords = _compute_ctfidf_keywords([], np.array([]), top_n=5)
    assert keywords == {}


# ---------------------------------------------------------------------------
# Tests: fit_topics (integration)
# ---------------------------------------------------------------------------


def test_fit_topics_returns_topic_result(synthetic_embeddings_and_texts):
    embeddings, texts = synthetic_embeddings_and_texts

    config = SegmentationConfig(
        hdbscan_min_cluster_size=30,
        hdbscan_min_samples=5,
        subsample_size=300,
        umap_cluster_n_components=5,
    )
    result = fit_topics(embeddings, texts, config=config)

    assert isinstance(result, TopicResult)
    assert len(result.labels) == len(embeddings)
    assert result.n_topics > 0


def test_fit_topics_umap_2d_shape(synthetic_embeddings_and_texts):
    embeddings, texts = synthetic_embeddings_and_texts

    config = SegmentationConfig(
        hdbscan_min_cluster_size=30,
        hdbscan_min_samples=5,
        subsample_size=300,
        umap_cluster_n_components=5,
    )
    result = fit_topics(embeddings, texts, config=config)
    assert result.umap_2d.shape[1] == 2


def test_fit_topics_topic_sizes_sum(synthetic_embeddings_and_texts):
    embeddings, texts = synthetic_embeddings_and_texts

    config = SegmentationConfig(
        hdbscan_min_cluster_size=30,
        hdbscan_min_samples=5,
        subsample_size=300,
        umap_cluster_n_components=5,
    )
    result = fit_topics(embeddings, texts, config=config)
    total = sum(result.topic_sizes.values())
    assert total == len(embeddings)
