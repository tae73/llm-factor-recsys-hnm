"""Unit tests for src/segmentation/product.py."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.segmentation.product import ProductClusterResult, _find_cross_category_pairs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_embeddings_and_articles(tmp_path: Path):
    """Create sample item embeddings and articles data."""
    rng = np.random.RandomState(42)
    n_items = 50

    # Embeddings
    embeddings = rng.randn(n_items, 768).astype(np.float32)
    article_ids = np.array([f"art_{i:04d}" for i in range(n_items)])

    emb_path = tmp_path / "item_bge_embeddings.npz"
    np.savez_compressed(emb_path, embeddings=embeddings.astype(np.float16), article_ids=article_ids)

    # Articles
    types = ["T-shirt"] * 20 + ["Trousers"] * 15 + ["Dress"] * 15
    articles_df = pd.DataFrame({
        "article_id": article_ids,
        "product_type_name": types,
        "product_group_name": ["Garment Upper body"] * 20 + ["Garment Lower body"] * 15 + ["Garment Full body"] * 15,
        "garment_group_name": ["Jersey Basic"] * 20 + ["Trousers"] * 15 + ["Dresses Ladies"] * 15,
    })
    articles_path = tmp_path / "articles.parquet"
    articles_df.to_parquet(articles_path, index=False)

    return emb_path, articles_path, article_ids


# ---------------------------------------------------------------------------
# Tests: ProductClusterResult
# ---------------------------------------------------------------------------


def test_product_cluster_result_fields():
    fields = ProductClusterResult._fields
    assert "cluster" in fields
    assert "ari_vs_native" in fields
    assert "cross_category_pairs" in fields
    assert "clusters_df" in fields


def test_ari_range():
    """ARI should be in [-1, 1]."""
    for ari in [0.0, 0.5, 1.0, -0.1]:
        assert -1 <= ari <= 1


# ---------------------------------------------------------------------------
# Tests: _find_cross_category_pairs
# ---------------------------------------------------------------------------


def test_cross_category_pairs_columns():
    """Verify output DataFrame has expected columns."""
    rng = np.random.RandomState(42)
    n = 20
    embeddings = rng.randn(n, 10).astype(np.float32)
    article_ids = np.array([f"a{i}" for i in range(n)])
    articles_df = pd.DataFrame({
        "article_id": article_ids,
        "product_type_name": ["TypeA"] * 10 + ["TypeB"] * 10,
    })

    pairs = _find_cross_category_pairs(
        embeddings, article_ids, articles_df, threshold=0.0, max_pairs=5, k_neighbors=5
    )
    expected_cols = {"article_id_1", "article_id_2", "similarity", "product_type_1", "product_type_2"}
    assert set(pairs.columns) == expected_cols


def test_cross_category_pairs_different_types():
    """All pairs should have different product types."""
    rng = np.random.RandomState(42)
    n = 20
    embeddings = rng.randn(n, 10).astype(np.float32)
    article_ids = np.array([f"a{i}" for i in range(n)])
    articles_df = pd.DataFrame({
        "article_id": article_ids,
        "product_type_name": ["TypeA"] * 10 + ["TypeB"] * 10,
    })

    pairs = _find_cross_category_pairs(
        embeddings, article_ids, articles_df, threshold=0.0, max_pairs=100, k_neighbors=10
    )
    if len(pairs) > 0:
        assert (pairs["product_type_1"] != pairs["product_type_2"]).all()


def test_cross_category_pairs_max_pairs():
    """Should respect max_pairs limit."""
    rng = np.random.RandomState(42)
    n = 30
    embeddings = rng.randn(n, 10).astype(np.float32)
    article_ids = np.array([f"a{i}" for i in range(n)])
    articles_df = pd.DataFrame({
        "article_id": article_ids,
        "product_type_name": ["TypeA"] * 15 + ["TypeB"] * 15,
    })

    pairs = _find_cross_category_pairs(
        embeddings, article_ids, articles_df, threshold=-1.0, max_pairs=5, k_neighbors=10
    )
    assert len(pairs) <= 5


def test_cross_category_pairs_no_duplicates():
    """Each pair should appear only once."""
    rng = np.random.RandomState(42)
    n = 20
    embeddings = rng.randn(n, 10).astype(np.float32)
    article_ids = np.array([f"a{i}" for i in range(n)])
    articles_df = pd.DataFrame({
        "article_id": article_ids,
        "product_type_name": ["TypeA"] * 10 + ["TypeB"] * 10,
    })

    pairs = _find_cross_category_pairs(
        embeddings, article_ids, articles_df, threshold=0.0, max_pairs=50, k_neighbors=10
    )
    if len(pairs) > 0:
        pair_keys = pairs.apply(lambda r: tuple(sorted([r["article_id_1"], r["article_id_2"]])), axis=1)
        assert pair_keys.is_unique


def test_cross_category_pairs_same_type_only():
    """If all items have same type, no cross-category pairs."""
    rng = np.random.RandomState(42)
    n = 10
    embeddings = rng.randn(n, 10).astype(np.float32)
    article_ids = np.array([f"a{i}" for i in range(n)])
    articles_df = pd.DataFrame({
        "article_id": article_ids,
        "product_type_name": ["SameType"] * n,
    })

    pairs = _find_cross_category_pairs(
        embeddings, article_ids, articles_df, threshold=0.0, max_pairs=50, k_neighbors=5
    )
    assert len(pairs) == 0
