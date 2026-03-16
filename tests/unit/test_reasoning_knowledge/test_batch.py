"""Tests for reasoning batch JSONL preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import ReasoningConfig
from src.knowledge.reasoning.batch import prepare_reasoning_batch_jsonl_chunked


def _make_user_data(customer_id: str) -> dict:
    """Create minimal user data dict for testing."""
    return {
        "customer_id": customer_id,
        "l1_summary": {
            "n_purchases": 20,
            "n_unique_types": 5,
            "category_diversity": 0.7,
            "top_categories_json": '{"T-shirt": 0.4}',
            "avg_price_quintile": 3.0,
            "online_ratio": 0.3,
        },
        "recent_items_l2": [
            {
                "article_id": "art_001",
                "l2_style_mood": ["Casual"],
                "l2_occasion": ["Everyday"],
                "l2_perceived_quality": 3,
                "l2_trendiness": "Classic",
                "l2_season_fit": "All-season",
                "l2_target_impression": "effortless",
                "l2_versatility": 5,
                "super_category": "Apparel",
            },
        ],
        "l3_distributions": {
            "shared": {
                "l3_color_harmony": {"Monochromatic": 0.5},
                "l3_visual_weight": {"mean": 2.5, "std": 0.8},
            },
            "by_category": {},
        },
    }


def _count_lines(path: Path) -> int:
    """Count lines in a file."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


@pytest.fixture
def config() -> ReasoningConfig:
    return ReasoningConfig(model="gpt-4.1-nano")


class TestPrepareReasoningBatchJsonlChunked:
    def test_single_user_single_chunk(self, tmp_path: Path, config: ReasoningConfig):
        users = [_make_user_data("cust_001")]
        paths = prepare_reasoning_batch_jsonl_chunked(
            users, config, tmp_path / "batch", max_bytes=10_000_000
        )
        assert len(paths) == 1
        assert _count_lines(paths[0]) == 1

    def test_splits_by_request_count(self, tmp_path: Path, config: ReasoningConfig):
        users = [_make_user_data(f"cust_{i:04d}") for i in range(12)]
        paths = prepare_reasoning_batch_jsonl_chunked(
            users, config, tmp_path / "batch",
            max_bytes=999_999_999,
            max_requests=5,
        )
        assert len(paths) == 3
        line_counts = [_count_lines(p) for p in paths]
        assert line_counts == [5, 5, 2]

    def test_splits_by_byte_size(self, tmp_path: Path, config: ReasoningConfig):
        users = [_make_user_data(f"cust_{i:04d}") for i in range(10)]
        paths = prepare_reasoning_batch_jsonl_chunked(
            users, config, tmp_path / "batch",
            max_bytes=500,  # Very small to force splitting
        )
        assert len(paths) > 1
        total_lines = sum(_count_lines(p) for p in paths)
        assert total_lines == 10

    def test_valid_jsonl_content(self, tmp_path: Path, config: ReasoningConfig):
        users = [_make_user_data("cust_001")]
        paths = prepare_reasoning_batch_jsonl_chunked(
            users, config, tmp_path / "batch"
        )
        with open(paths[0]) as f:
            for line in f:
                entry = json.loads(line.strip())
                assert entry["custom_id"] == "cust_001"
                assert entry["method"] == "POST"
                assert entry["url"] == "/v1/responses"
                assert "user_reasoning_profile" in entry["body"]["text"]["format"]["name"]

    def test_uses_config_defaults(self, tmp_path: Path):
        custom_config = ReasoningConfig(batch_max_requests=3)
        users = [_make_user_data(f"cust_{i:04d}") for i in range(7)]
        paths = prepare_reasoning_batch_jsonl_chunked(
            users, custom_config, tmp_path / "batch"
        )
        assert len(paths) == 3  # 3+3+1
        line_counts = [_count_lines(p) for p in paths]
        assert line_counts == [3, 3, 1]

    def test_file_naming_convention(self, tmp_path: Path, config: ReasoningConfig):
        users = [_make_user_data(f"cust_{i:04d}") for i in range(3)]
        paths = prepare_reasoning_batch_jsonl_chunked(
            users, config, tmp_path / "batch",
            max_requests=2,
        )
        assert all(p.name.startswith("input_") and p.name.endswith(".jsonl") for p in paths)
        assert paths[0].name == "input_000.jsonl"
        assert paths[1].name == "input_001.jsonl"

    def test_creates_output_dir(self, tmp_path: Path, config: ReasoningConfig):
        nested = tmp_path / "a" / "b" / "batch"
        users = [_make_user_data("cust_001")]
        paths = prepare_reasoning_batch_jsonl_chunked(users, config, nested)
        assert nested.exists()
        assert len(paths) == 1
