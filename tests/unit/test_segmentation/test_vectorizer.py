"""Unit tests for src/segmentation/vectorizer.py."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.segmentation.vectorizer import (
    VectorizerResult,
    _aggregate_array_field,
    _aggregate_scalar_field,
    _build_vocab,
    _fill_multihot_from_json,
    _is_nan,
    _normalize_section,
    _parse_list,
    _safe_parse_json,
    vectorize_l1,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_profiles(tmp_path: Path) -> Path:
    """Create a minimal user_profiles.parquet for testing."""
    df = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3", "u4", "u5"],
            "top_categories_json": [
                json.dumps({"T-shirt": 0.5, "Trousers": 0.3, "Vest": 0.2}),
                json.dumps({"Dress": 0.8, "Vest": 0.2}),
                json.dumps({"T-shirt": 0.4, "Shorts": 0.6}),
                json.dumps({"Sweater": 1.0}),
                json.dumps({}),
            ],
            "top_colors_json": [
                json.dumps({"Black": 0.6, "White": 0.4}),
                json.dumps({"Red": 0.7, "Blue": 0.3}),
                json.dumps({"Black": 1.0}),
                json.dumps({"Green": 0.5, "Black": 0.5}),
                json.dumps({}),
            ],
            "top_materials_json": [
                json.dumps({"Cotton": 0.8, "Polyester": 0.2}),
                json.dumps({"Silk": 1.0}),
                json.dumps({"Cotton": 0.5, "Denim": 0.5}),
                json.dumps({"Wool": 1.0}),
                json.dumps({}),
            ],
            "avg_price_quintile": [1.0, 3.0, 5.0, 2.0, float("nan")],
            "online_ratio": [0.8, 0.5, 1.0, 0.0, float("nan")],
            "category_diversity": [0.9, 0.3, 0.6, 0.0, float("nan")],
        }
    )
    path = tmp_path / "user_profiles.parquet"
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests: _safe_parse_json
# ---------------------------------------------------------------------------


def test_safe_parse_json_valid():
    assert _safe_parse_json('{"a": 1}') == {"a": 1}


def test_safe_parse_json_none():
    assert _safe_parse_json(None) == {}


def test_safe_parse_json_nan():
    assert _safe_parse_json(float("nan")) == {}


def test_safe_parse_json_invalid():
    assert _safe_parse_json("not json") == {}


def test_safe_parse_json_dict():
    assert _safe_parse_json({"x": 1}) == {"x": 1}


# ---------------------------------------------------------------------------
# Tests: _parse_list
# ---------------------------------------------------------------------------


def test_parse_list_json_array():
    assert _parse_list('["a", "b"]') == ["a", "b"]


def test_parse_list_python_list():
    assert _parse_list(["x", "y"]) == ["x", "y"]


def test_parse_list_none():
    assert _parse_list(None) == []


def test_parse_list_nan():
    assert _parse_list(float("nan")) == []


def test_parse_list_scalar():
    assert _parse_list("hello") == ["hello"]


# ---------------------------------------------------------------------------
# Tests: _is_nan
# ---------------------------------------------------------------------------


def test_is_nan_float():
    assert _is_nan(float("nan")) is True


def test_is_nan_none():
    assert _is_nan(None) is True


def test_is_nan_string_nan():
    assert _is_nan("nan") is True


def test_is_nan_valid_float():
    assert _is_nan(3.14) is False


def test_is_nan_valid_string():
    assert _is_nan("hello") is False


# ---------------------------------------------------------------------------
# Tests: _build_vocab
# ---------------------------------------------------------------------------


def test_build_vocab_top_n():
    series = pd.Series([
        json.dumps({"a": 0.5, "b": 0.3, "c": 0.2}),
        json.dumps({"a": 0.8, "d": 0.2}),
    ])
    vocab = _build_vocab(series, top_n=2)
    assert len(vocab) == 2
    assert vocab[0] == "a"  # most common


# ---------------------------------------------------------------------------
# Tests: _normalize_section
# ---------------------------------------------------------------------------


def test_normalize_section_sums_to_one():
    v = np.array([[3.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 5.0, 5.0, 0.0]], dtype=np.float32)
    _normalize_section(v, 0, 5)
    np.testing.assert_allclose(v.sum(axis=1), [1.0, 1.0], atol=1e-6)


def test_normalize_section_zero_row():
    v = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    _normalize_section(v, 0, 3)
    np.testing.assert_array_equal(v, [[0.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# Tests: vectorize_l1
# ---------------------------------------------------------------------------


def test_vectorize_l1_shape(sample_profiles: Path):
    result = vectorize_l1(sample_profiles)
    assert isinstance(result, VectorizerResult)
    assert result.vectors.shape[0] == 5
    assert result.vectors.shape[1] == result.dim
    assert len(result.customer_ids) == 5


def test_vectorize_l1_no_nans(sample_profiles: Path):
    result = vectorize_l1(sample_profiles)
    assert not np.any(np.isnan(result.vectors))


def test_vectorize_l1_feature_names_match(sample_profiles: Path):
    result = vectorize_l1(sample_profiles)
    assert len(result.feature_names) == result.dim
