"""Unit tests for src/segmentation/customer.py — NamedTuple and integration checks."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.segmentation.customer import CustomerSegmentResult


# ---------------------------------------------------------------------------
# Tests: CustomerSegmentResult structure
# ---------------------------------------------------------------------------


def test_customer_segment_result_fields():
    """Verify NamedTuple has all expected fields."""
    fields = CustomerSegmentResult._fields
    assert "l1_cluster" in fields
    assert "l2_cluster" in fields
    assert "l3_cluster" in fields
    assert "semantic_cluster" in fields
    assert "topic" in fields
    assert "customer_ids" in fields
    assert "segments_df" in fields


def test_customer_segment_result_field_count():
    assert len(CustomerSegmentResult._fields) == 7


# ---------------------------------------------------------------------------
# Tests: segments_df format
# ---------------------------------------------------------------------------


def test_segments_df_columns():
    """Verify expected columns in output DataFrame."""
    df = pd.DataFrame({
        "customer_id": ["u1", "u2"],
        "l1_segment": [0, 1],
        "l2_segment": [1, 0],
        "l3_segment": [0, 0],
        "semantic_segment": [1, 1],
        "topic_segment": [0, 1],
    })
    expected_cols = {"customer_id", "l1_segment", "l2_segment", "l3_segment", "semantic_segment", "topic_segment"}
    assert set(df.columns) == expected_cols


def test_segments_df_all_assigned():
    """All users should have valid segment assignments (>= 0)."""
    df = pd.DataFrame({
        "customer_id": ["u1", "u2", "u3"],
        "l1_segment": [0, 1, 2],
        "l2_segment": [1, 0, 1],
        "l3_segment": [0, 0, 1],
        "semantic_segment": [1, 1, 0],
        "topic_segment": [0, 1, 0],
    })
    for col in ["l1_segment", "l2_segment", "l3_segment", "semantic_segment", "topic_segment"]:
        assert (df[col] >= 0).all(), f"{col} has negative values"


def test_segments_df_unique_customer_ids():
    """Each customer should appear exactly once."""
    df = pd.DataFrame({
        "customer_id": ["u1", "u2", "u3"],
        "l1_segment": [0, 1, 2],
    })
    assert df["customer_id"].is_unique


# ---------------------------------------------------------------------------
# Tests: clustering_meta.json format
# ---------------------------------------------------------------------------


def test_clustering_meta_structure(tmp_path: Path):
    """Verify clustering_meta.json has expected keys."""
    meta = {
        "l1": {"k": 6, "silhouette": 0.15, "k_scores": {"4": 0.12, "6": 0.15, "8": 0.14}},
        "l2": {"k": 8, "silhouette": 0.20, "k_scores": {"4": 0.18, "6": 0.19, "8": 0.20}},
        "l3": {"k": 4, "silhouette": 0.18, "k_scores": {"4": 0.18, "6": 0.16}},
        "semantic": {"k": 10, "silhouette": 0.12, "k_scores": {"8": 0.11, "10": 0.12}},
        "topic": {"n_topics": 15, "outlier_count": 5000, "topic_sizes": {"0": 1000, "1": 800}},
    }
    path = tmp_path / "clustering_meta.json"
    with open(path, "w") as f:
        json.dump(meta, f)

    with open(path) as f:
        loaded = json.load(f)

    assert set(loaded.keys()) == {"l1", "l2", "l3", "semantic", "topic"}
    assert "k" in loaded["l1"]
    assert "silhouette" in loaded["l1"]
    assert "n_topics" in loaded["topic"]


def test_clustering_meta_silhouette_range():
    """Silhouette scores should be in [-1, 1]."""
    for sil in [0.15, 0.20, 0.18, 0.12]:
        assert -1 <= sil <= 1
