"""Tests for extractor.py — L1 aggregation + sparse fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import ReasoningConfig
from src.knowledge.reasoning.extractor import (
    _compute_diversity_score,
    _count_distribution,
    _describe_color,
    _describe_quality,
    _parse_list_field,
    _top_values,
    _weighted_distribution,
    aggregate_l1_profiles,
    build_sparse_user_profiles,
    compose_sparse_reasoning_text,
    compute_l3_distributions_batch,
    get_recent_items_batch,
)

# ---------------------------------------------------------------------------
# Fixtures — create minimal Parquet files for DuckDB queries
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data(tmp_path: Path) -> dict[str, Path]:
    """Create minimal transaction, article, and factual knowledge parquets."""
    import datetime

    # Transactions
    txn_df = pd.DataFrame({
        "t_dat": [
            datetime.date(2020, 6, 1),
            datetime.date(2020, 6, 15),
            datetime.date(2020, 5, 1),
            datetime.date(2020, 6, 20),
            datetime.date(2020, 6, 25),
            datetime.date(2020, 3, 1),
            # Sparse user — only 2 purchases
            datetime.date(2020, 6, 10),
            datetime.date(2020, 6, 12),
        ],
        "customer_id": [
            "user_A", "user_A", "user_A", "user_A", "user_A", "user_A",
            "user_B", "user_B",
        ],
        "article_id": [
            "art_001", "art_002", "art_003", "art_004", "art_005", "art_001",
            "art_001", "art_002",
        ],
        "price": [0.05, 0.10, 0.15, 0.03, 0.08, 0.05, 0.05, 0.10],
        "sales_channel_id": [1, 2, 1, 2, 2, 1, 2, 2],
    })
    txn_path = tmp_path / "train_transactions.parquet"
    txn_df.to_parquet(txn_path, index=False)

    # Articles
    articles_df = pd.DataFrame({
        "article_id": ["art_001", "art_002", "art_003", "art_004", "art_005"],
        "product_code": ["PC001", "PC002", "PC003", "PC004", "PC005"],
        "product_type_name": ["T-shirt", "Trousers", "Sweater", "T-shirt", "Dress"],
        "colour_group_name": ["Black", "Blue", "Grey", "White", "Red"],
        "product_group_name": ["Garment Upper body"] * 5,
        "garment_group_name": ["Jersey Basic"] * 5,
    })
    articles_path = tmp_path / "articles.parquet"
    articles_df.to_parquet(articles_path, index=False)

    # Factual knowledge
    fk_df = pd.DataFrame({
        "article_id": ["art_001", "art_002", "art_003", "art_004", "art_005"],
        "l1_material": ["Cotton", "Polyester", "Wool", "Cotton", "Silk"],
        "l2_style_mood": [
            json.dumps(["Casual", "Minimalist"]),
            json.dumps(["Classic"]),
            json.dumps(["Cozy", "Casual"]),
            json.dumps(["Casual"]),
            json.dumps(["Romantic", "Feminine"]),
        ],
        "l2_occasion": [
            json.dumps(["Everyday"]),
            json.dumps(["Work"]),
            json.dumps(["Everyday", "Lounge"]),
            json.dumps(["Everyday"]),
            json.dumps(["Party", "Date"]),
        ],
        "l2_perceived_quality": [3, 4, 3, 2, 4],
        "l2_trendiness": ["Classic", "Classic", "Current", "Classic", "Current"],
        "l2_season_fit": ["All-season", "All-season", "Winter", "Summer", "Spring"],
        "l2_target_impression": ["effortless", "polished", "cozy weekend", "basic", "elegant evening"],
        "l2_versatility": [5, 4, 3, 4, 2],
        "l3_color_harmony": ["Monochromatic", "Analogous", "Neutral", "Monochromatic", "Complementary"],
        "l3_tone_season": ["Cool-Winter", "Cool-Summer", "Neutral-Cool", "Neutral-Cool", "Warm-Autumn"],
        "l3_coordination_role": ["Basic", "Basic", "Layering", "Basic", "Statement"],
        "l3_visual_weight": [2, 3, 3, 2, 3],
        "l3_style_lineage": [
            json.dumps(["Scandinavian Minimalism"]),
            json.dumps(["Classic Formal"]),
            json.dumps(["Knit Heritage"]),
            json.dumps(["Scandinavian Minimalism"]),
            json.dumps(["Romantic Victorian"]),
        ],
        "super_category": ["Apparel", "Apparel", "Apparel", "Apparel", "Apparel"],
        "l3_slot6": ["I-line", "H-line", "O-line", "I-line", "A-line"],
        "l3_slot7": ["Streamlining", "Balanced", "Volume-adding", "Streamlining", "Broadening"],
    })
    fk_path = tmp_path / "factual_knowledge.parquet"
    fk_df.to_parquet(fk_path, index=False)

    return {
        "txn_path": txn_path,
        "articles_path": articles_path,
        "fk_path": fk_path,
    }


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------


class TestParseListField:
    def test_none(self):
        assert _parse_list_field(None) == []

    def test_nan(self):
        assert _parse_list_field(float("nan")) == []

    def test_list(self):
        assert _parse_list_field(["a", "b"]) == ["a", "b"]

    def test_json_string(self):
        assert _parse_list_field('["Casual", "Minimalist"]') == ["Casual", "Minimalist"]

    def test_plain_string(self):
        assert _parse_list_field("Casual") == ["Casual"]


class TestWeightedDistribution:
    def test_basic(self):
        result = json.loads(_weighted_distribution(
            ["a", "a", "b"], [1.0, 1.0, 1.0]
        ))
        assert result["a"] == pytest.approx(0.6667, abs=0.001)
        assert result["b"] == pytest.approx(0.3333, abs=0.001)

    def test_empty(self):
        assert _weighted_distribution([], []) == "{}"

    def test_none_values_filtered(self):
        result = json.loads(_weighted_distribution(
            ["a", None, "b"], [1.0, 1.0, 1.0]
        ))
        assert len(result) == 2


class TestComputeDiversityScore:
    def test_single_value(self):
        assert _compute_diversity_score(["a", "a", "a"], [1.0, 1.0, 1.0]) == 0.0

    def test_uniform_two(self):
        score = _compute_diversity_score(["a", "b"], [1.0, 1.0])
        assert score == pytest.approx(1.0, abs=0.01)

    def test_empty(self):
        assert _compute_diversity_score([], []) == 0.0


class TestCountDistribution:
    def test_basic(self):
        result = _count_distribution(["a", "a", "b", "c"])
        assert result["a"] == 0.5
        assert result["b"] == 0.25

    def test_empty(self):
        assert _count_distribution([]) == {}


class TestTopValues:
    def test_basic(self):
        assert _top_values(["a", "a", "b", "c"], 2) == "a, b"

    def test_empty(self):
        assert _top_values([], 3) == "Unknown"


class TestDescribeQuality:
    def test_budget(self):
        assert "Budget" in _describe_quality([1, 2, 1])

    def test_midrange(self):
        assert "Mid-range" in _describe_quality([3, 3, 3])

    def test_quality(self):
        assert "Quality" in _describe_quality([4, 5, 4])

    def test_empty(self):
        assert _describe_quality([]) == "Unknown"


class TestDescribeColor:
    def test_basic(self):
        result = _describe_color(["Monochromatic", "Neutral"], ["Cool-Winter"])
        assert "Monochromatic" in result
        assert "Cool-Winter" in result

    def test_empty(self):
        assert _describe_color([], []) == "Unknown"


class TestComposeSparseReasoningText:
    def test_has_all_fields(self):
        reasoning_json = {
            "style_mood_preference": "Casual, Minimalist",
            "occasion_preference": "Everyday",
            "quality_price_tendency": "Mid-range (avg 3.0/5)",
            "trend_sensitivity": "Classic",
            "seasonal_pattern": "All-season",
            "form_preference": "I-line",
            "color_tendency": "Monochromatic; Cool-Winter",
            "coordination_tendency": "Basic",
            "identity_summary": "User with 2 purchase(s).",
        }
        text = compose_sparse_reasoning_text(reasoning_json)
        assert "(a)" in text
        assert "(i)" in text
        assert "Casual" in text
        assert "Identity:" in text


# ---------------------------------------------------------------------------
# Integration tests — DuckDB queries
# ---------------------------------------------------------------------------


class TestAggregateL1Profiles:
    def test_basic_aggregation(self, sample_data):
        config = ReasoningConfig()
        df = aggregate_l1_profiles(
            sample_data["txn_path"],
            sample_data["articles_path"],
            sample_data["fk_path"],
            config,
        )
        assert len(df) == 2  # user_A and user_B
        user_a = df[df["customer_id"] == "user_A"].iloc[0]
        assert user_a["n_purchases"] == 6
        assert user_a["n_unique_articles"] == 5
        assert user_a["online_ratio"] == pytest.approx(0.5, abs=0.01)
        assert 0.0 <= user_a["category_diversity"] <= 1.0

        # Check JSON distributions are valid
        cats = json.loads(user_a["top_categories_json"])
        assert "T-shirt" in cats
        assert isinstance(cats["T-shirt"], float)

    def test_user_b_has_two_purchases(self, sample_data):
        config = ReasoningConfig()
        df = aggregate_l1_profiles(
            sample_data["txn_path"],
            sample_data["articles_path"],
            sample_data["fk_path"],
            config,
        )
        user_b = df[df["customer_id"] == "user_B"].iloc[0]
        assert user_b["n_purchases"] == 2


class TestGetRecentItemsBatch:
    def test_returns_items_with_l2(self, sample_data):
        result = get_recent_items_batch(
            sample_data["txn_path"],
            sample_data["fk_path"],
            ["user_A"],
            limit=3,
        )
        assert "user_A" in result
        items = result["user_A"]
        assert len(items) <= 3
        assert "l2_style_mood" in items[0]
        assert isinstance(items[0]["l2_style_mood"], list)

    def test_limit_respected(self, sample_data):
        result = get_recent_items_batch(
            sample_data["txn_path"],
            sample_data["fk_path"],
            ["user_A"],
            limit=2,
        )
        assert len(result["user_A"]) <= 2


class TestComputeL3DistributionsBatch:
    def test_shared_distributions(self, sample_data):
        result = compute_l3_distributions_batch(
            sample_data["txn_path"],
            sample_data["fk_path"],
            ["user_A"],
        )
        assert "user_A" in result
        shared = result["user_A"]["shared"]
        assert "l3_color_harmony" in shared
        assert isinstance(shared["l3_color_harmony"], dict)
        assert "l3_visual_weight" in shared
        assert "mean" in shared["l3_visual_weight"]

    def test_by_category(self, sample_data):
        result = compute_l3_distributions_batch(
            sample_data["txn_path"],
            sample_data["fk_path"],
            ["user_A"],
        )
        by_cat = result["user_A"]["by_category"]
        assert "Apparel" in by_cat
        assert "l3_slot6" in by_cat["Apparel"]


class TestBuildSparseUserProfiles:
    def test_builds_template_profiles(self, sample_data):
        df = build_sparse_user_profiles(
            sample_data["txn_path"],
            sample_data["fk_path"],
            ["user_B"],
        )
        assert len(df) == 1
        row = df.iloc[0]
        assert row["customer_id"] == "user_B"
        assert row["profile_source"] == "template"
        assert "(a)" in row["reasoning_text"]
        assert "(i)" in row["reasoning_text"]
        assert row["n_purchases"] == 2

    def test_handles_missing_user(self, sample_data):
        df = build_sparse_user_profiles(
            sample_data["txn_path"],
            sample_data["fk_path"],
            ["nonexistent_user"],
        )
        assert len(df) == 1
        assert df.iloc[0]["n_purchases"] == 0
        assert "Unknown" in df.iloc[0]["reasoning_text"]

    def test_empty_list(self, sample_data):
        df = build_sparse_user_profiles(
            sample_data["txn_path"],
            sample_data["fk_path"],
            [],
        )
        assert len(df) == 0
