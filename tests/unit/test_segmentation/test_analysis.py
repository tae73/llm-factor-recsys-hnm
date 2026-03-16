"""Unit tests for src/segmentation/analysis.py."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.segmentation.analysis import (
    DiscriminativeProfile,
    EffectiveKResult,
    ExcessSimilarityResult,
    SegmentProfile,
    _aggregate_json_top,
    _compute_weighted_freq,
    compute_cross_category_excess_similarity,
    compute_effective_k,
    compute_l3_segment_heatmap_data,
    compute_segment_statistics,
    cross_layer_ari,
    profile_segments,
    profile_segments_discriminative,
    save_segment_profiles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_segments_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
        "l1_segment": [0, 0, 1, 1, 2, 2],
        "l2_segment": [0, 1, 0, 1, 0, 1],
        "l3_segment": [0, 0, 0, 1, 1, 1],
        "semantic_segment": [0, 0, 1, 1, 0, 1],
        "topic_segment": [0, 0, 0, 1, 1, 1],
    })


@pytest.fixture
def sample_profiles_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
        "top_categories_json": [
            json.dumps({"T-shirt": 0.5, "Vest": 0.5}),
            json.dumps({"T-shirt": 0.8}),
            json.dumps({"Dress": 0.7}),
            json.dumps({"Trousers": 0.9}),
            json.dumps({"Shorts": 0.6}),
            json.dumps({"Sweater": 1.0}),
        ],
        "top_colors_json": [
            json.dumps({"Black": 0.6}),
            json.dumps({"Black": 0.8}),
            json.dumps({"Red": 0.7}),
            json.dumps({"Blue": 0.5}),
            json.dumps({"White": 0.9}),
            json.dumps({"Green": 0.4}),
        ],
        "top_materials_json": [
            json.dumps({"Cotton": 0.8}),
            json.dumps({"Cotton": 0.6}),
            json.dumps({"Silk": 0.5}),
            json.dumps({"Wool": 0.7}),
            json.dumps({"Polyester": 0.3}),
            json.dumps({"Cotton": 0.9}),
        ],
        "category_diversity": [0.8, 0.6, 0.4, 0.3, 0.5, 0.2],
        "online_ratio": [0.9, 0.7, 0.5, 0.3, 0.8, 0.1],
        "avg_price_quintile": [2.0, 3.0, 4.0, 1.0, 5.0, 2.0],
        "n_purchases": [10, 20, 5, 15, 8, 30],
    })


# ---------------------------------------------------------------------------
# Tests: profile_segments
# ---------------------------------------------------------------------------


def test_profile_segments_count(sample_segments_df, sample_profiles_df):
    profiles = profile_segments(sample_segments_df, sample_profiles_df, level="l1")
    assert len(profiles) == 3  # 3 L1 segments


def test_profile_segments_type(sample_segments_df, sample_profiles_df):
    profiles = profile_segments(sample_segments_df, sample_profiles_df, level="l1")
    assert all(isinstance(p, SegmentProfile) for p in profiles)


def test_profile_segments_sizes_sum(sample_segments_df, sample_profiles_df):
    profiles = profile_segments(sample_segments_df, sample_profiles_df, level="l1")
    total = sum(p.size for p in profiles)
    assert total == 6


def test_profile_segments_invalid_level(sample_segments_df, sample_profiles_df):
    with pytest.raises(ValueError):
        profile_segments(sample_segments_df, sample_profiles_df, level="invalid")


# ---------------------------------------------------------------------------
# Tests: cross_layer_ari
# ---------------------------------------------------------------------------


def test_cross_layer_ari_shape(sample_segments_df):
    ari_df = cross_layer_ari(sample_segments_df)
    assert ari_df.shape == (5, 5)


def test_cross_layer_ari_diagonal_ones(sample_segments_df):
    ari_df = cross_layer_ari(sample_segments_df)
    np.testing.assert_array_almost_equal(np.diag(ari_df.values), 1.0)


def test_cross_layer_ari_symmetric(sample_segments_df):
    ari_df = cross_layer_ari(sample_segments_df)
    np.testing.assert_array_almost_equal(ari_df.values, ari_df.values.T)


# ---------------------------------------------------------------------------
# Tests: compute_segment_statistics
# ---------------------------------------------------------------------------


def test_segment_statistics_columns(sample_segments_df, sample_profiles_df):
    stats = compute_segment_statistics(sample_segments_df, sample_profiles_df, level="l1")
    assert "n_users" in stats.columns
    assert "mean_diversity" in stats.columns
    assert "mean_online_ratio" in stats.columns


def test_segment_statistics_rows(sample_segments_df, sample_profiles_df):
    stats = compute_segment_statistics(sample_segments_df, sample_profiles_df, level="l1")
    assert len(stats) == 3  # 3 segments


# ---------------------------------------------------------------------------
# Tests: save_segment_profiles
# ---------------------------------------------------------------------------


def test_save_segment_profiles(tmp_path: Path, sample_segments_df, sample_profiles_df):
    profiles = profile_segments(sample_segments_df, sample_profiles_df, level="l1")
    out_path = tmp_path / "profiles.json"
    save_segment_profiles(profiles, out_path)

    with open(out_path) as f:
        data = json.load(f)
    assert len(data) == 3
    assert "segment_id" in data[0]
    assert "label" in data[0]


# ---------------------------------------------------------------------------
# Tests: _aggregate_json_top
# ---------------------------------------------------------------------------


def test_aggregate_json_top_returns_list():
    series = pd.Series([json.dumps({"a": 0.5, "b": 0.3}), json.dumps({"a": 0.8})])
    result = _aggregate_json_top(series, top_n=2)
    assert isinstance(result, list)
    assert result[0] == "a"  # most common


# ---------------------------------------------------------------------------
# Tests: _compute_weighted_freq
# ---------------------------------------------------------------------------


def test_compute_weighted_freq_basic():
    series = pd.Series([json.dumps({"a": 0.6, "b": 0.4}), json.dumps({"a": 0.2, "c": 0.8})])
    freq = _compute_weighted_freq(series)
    assert "a" in freq
    assert "b" in freq
    assert "c" in freq
    assert abs(sum(freq.values()) - 1.0) < 1e-6


def test_compute_weighted_freq_empty():
    series = pd.Series([None, None])
    freq = _compute_weighted_freq(series)
    assert freq == {}


# ---------------------------------------------------------------------------
# Tests: profile_segments_discriminative
# ---------------------------------------------------------------------------


def test_discriminative_profile_count(sample_segments_df, sample_profiles_df):
    profiles = profile_segments_discriminative(sample_segments_df, sample_profiles_df, level="l1")
    assert len(profiles) == 3  # 3 L1 segments


def test_discriminative_profile_type(sample_segments_df, sample_profiles_df):
    profiles = profile_segments_discriminative(sample_segments_df, sample_profiles_df, level="l1")
    assert all(isinstance(p, DiscriminativeProfile) for p in profiles)


def test_discriminative_profile_has_over_under(sample_segments_df, sample_profiles_df):
    profiles = profile_segments_discriminative(sample_segments_df, sample_profiles_df, level="l1")
    for p in profiles:
        assert isinstance(p.over_represented, dict)
        assert isinstance(p.under_represented, dict)


def test_discriminative_profile_ratios(sample_segments_df, sample_profiles_df):
    profiles = profile_segments_discriminative(
        sample_segments_df, sample_profiles_df, level="l1", min_population_freq=0.0
    )
    for p in profiles:
        for field, items in p.over_represented.items():
            for attr, ratio in items:
                assert ratio >= 1.0, f"Over-represented ratio should be >= 1.0, got {ratio}"
        for field, items in p.under_represented.items():
            for attr, ratio in items:
                assert ratio <= 1.0, f"Under-represented ratio should be <= 1.0, got {ratio}"


# ---------------------------------------------------------------------------
# Tests: compute_effective_k
# ---------------------------------------------------------------------------


def test_effective_k_count(sample_segments_df):
    results = compute_effective_k(sample_segments_df)
    assert len(results) == 5  # 5 levels


def test_effective_k_type(sample_segments_df):
    results = compute_effective_k(sample_segments_df)
    assert all(isinstance(r, EffectiveKResult) for r in results)


def test_effective_k_range(sample_segments_df):
    results = compute_effective_k(sample_segments_df)
    for r in results:
        assert 1.0 <= r.effective_k <= r.nominal_k + 0.01
        assert 0.0 <= r.evenness <= 1.01


def test_effective_k_specific_levels(sample_segments_df):
    results = compute_effective_k(sample_segments_df, levels=["l1", "l2"])
    assert len(results) == 2
    assert results[0].level == "l1"
    assert results[1].level == "l2"


# ---------------------------------------------------------------------------
# Tests: compute_l3_segment_heatmap_data
# ---------------------------------------------------------------------------


def test_l3_heatmap_shape():
    rng = np.random.RandomState(42)
    l3_vectors = rng.randn(100, 37).astype(np.float32)
    labels = np.array([0] * 40 + [1] * 30 + [2] * 30)
    result = compute_l3_segment_heatmap_data(l3_vectors, labels)
    assert result.shape == (37, 3)  # 37 dims × 3 segments


def test_l3_heatmap_feature_names():
    rng = np.random.RandomState(42)
    l3_vectors = rng.randn(50, 5).astype(np.float32)
    labels = np.array([0] * 25 + [1] * 25)
    names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
    result = compute_l3_segment_heatmap_data(l3_vectors, labels, feature_names=names)
    assert list(result.index) == names


# ---------------------------------------------------------------------------
# Tests: compute_cross_category_excess_similarity
# ---------------------------------------------------------------------------


def test_excess_similarity_basic():
    cross_pairs = pd.DataFrame({"similarity": [0.92, 0.88, 0.95, 0.86, 0.91]})
    results = compute_cross_category_excess_similarity(cross_pairs, baseline_mean=0.80)
    assert len(results) == 3  # 3 default thresholds
    assert all(isinstance(r, ExcessSimilarityResult) for r in results)


def test_excess_similarity_decreasing_pairs():
    cross_pairs = pd.DataFrame({"similarity": [0.92, 0.88, 0.95, 0.86, 0.91]})
    results = compute_cross_category_excess_similarity(cross_pairs, baseline_mean=0.80)
    # Higher threshold → fewer pairs
    assert results[0].n_pairs >= results[1].n_pairs >= results[2].n_pairs


def test_excess_similarity_empty():
    cross_pairs = pd.DataFrame({"similarity": []})
    results = compute_cross_category_excess_similarity(cross_pairs, baseline_mean=0.80)
    assert all(r.n_pairs == 0 for r in results)
