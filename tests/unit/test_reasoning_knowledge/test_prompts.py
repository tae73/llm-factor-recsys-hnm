"""Tests for prompts.py — LLM prompting, schema, text composition."""

from __future__ import annotations

import json

import pytest

from src.config import ReasoningConfig
from src.knowledge.reasoning.prompts import (
    REASONING_SCHEMA,
    SYSTEM_PROMPT,
    _parse_json_field,
    build_reasoning_request_line,
    build_reasoning_user_message,
    compose_reasoning_text,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_l1_summary() -> dict:
    return {
        "n_purchases": 45,
        "n_unique_types": 12,
        "category_diversity": 0.72,
        "top_categories_json": json.dumps({"T-shirt": 0.22, "Trousers": 0.15, "Sweater": 0.12}),
        "top_colors_json": json.dumps({"Black": 0.30, "Blue": 0.20}),
        "top_materials_json": json.dumps({"Cotton": 0.40}),
        "avg_price_quintile": 2.3,
        "online_ratio": 0.35,
    }


@pytest.fixture
def sample_recent_items() -> list[dict]:
    return [
        {
            "article_id": "art_001",
            "l2_style_mood": ["Casual", "Minimalist"],
            "l2_occasion": ["Everyday"],
            "l2_perceived_quality": 3,
            "l2_trendiness": "Classic",
            "l2_season_fit": "All-season",
            "l2_target_impression": "effortless everyday essential",
            "l2_versatility": 5,
            "super_category": "Apparel",
        },
        {
            "article_id": "art_002",
            "l2_style_mood": ["Classic"],
            "l2_occasion": ["Work"],
            "l2_perceived_quality": 4,
            "l2_trendiness": "Classic",
            "l2_season_fit": "All-season",
            "l2_target_impression": "polished professional",
            "l2_versatility": 4,
            "super_category": "Apparel",
        },
    ]


@pytest.fixture
def sample_l3_distributions() -> dict:
    return {
        "shared": {
            "l3_color_harmony": {"Monochromatic": 0.45, "Neutral": 0.30, "Analogous": 0.15},
            "l3_tone_season": {"Cool-Winter": 0.40, "Neutral-Cool": 0.25},
            "l3_coordination_role": {"Basic": 0.60, "Layering": 0.20},
            "l3_visual_weight": {"mean": 2.5, "std": 0.8},
            "l3_style_lineage": {"Scandinavian Minimalism": 0.35, "Classic Formal": 0.20},
        },
        "by_category": {
            "Apparel": {
                "n": 38,
                "l3_slot6": {"I-line": 0.55, "H-line": 0.25},
                "l3_slot7": {"Balanced": 0.40, "Streamlining": 0.30},
            },
            "Footwear": {
                "n": 5,
                "l3_slot6": {"Streamlined": 0.80},
                "l3_slot7": {"Neutral": 0.60},
            },
        },
    }


@pytest.fixture
def sample_reasoning_json() -> dict:
    return {
        "style_mood_preference": "Casual minimalist with occasional formal touches",
        "occasion_preference": "Everyday basics with weekend casual",
        "quality_price_tendency": "Mid-range with selective premium purchases",
        "trend_sensitivity": "Classic core with occasional current pieces",
        "seasonal_pattern": "Year-round basics buyer, heavier winter purchases",
        "form_preference": "Prefers streamlined I-line silhouettes, slim fits",
        "color_tendency": "Monochromatic neutral palette, Cool-Winter tones",
        "coordination_tendency": "Builds from basics, adds occasional statement pieces",
        "identity_summary": "A practical minimalist who values quality basics and neutral palettes",
    }


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestReasoningSchema:
    def test_has_all_9_fields(self):
        assert len(REASONING_SCHEMA["required"]) == 9

    def test_required_fields(self):
        expected = [
            "style_mood_preference",
            "occasion_preference",
            "quality_price_tendency",
            "trend_sensitivity",
            "seasonal_pattern",
            "form_preference",
            "color_tendency",
            "coordination_tendency",
            "identity_summary",
        ]
        assert REASONING_SCHEMA["required"] == expected

    def test_all_fields_are_string_type(self):
        for field, schema in REASONING_SCHEMA["properties"].items():
            assert schema["type"] == "string", f"{field} should be string"

    def test_no_additional_properties(self):
        assert REASONING_SCHEMA["additionalProperties"] is False


class TestSystemPrompt:
    def test_not_empty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_mentions_9_dimensional(self):
        assert "9-dimensional" in SYSTEM_PROMPT

    def test_mentions_fashion(self):
        assert "fashion" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# User message construction
# ---------------------------------------------------------------------------


class TestBuildReasoningUserMessage:
    def test_has_three_sections(
        self, sample_l1_summary, sample_recent_items, sample_l3_distributions
    ):
        msg = build_reasoning_user_message(
            sample_l1_summary, sample_recent_items, sample_l3_distributions
        )
        assert "--- Customer Overview ---" in msg
        assert "--- Recent Items (L2 Attributes) ---" in msg
        assert "--- Attribute Patterns (L3 Theory-Based) ---" in msg

    def test_includes_purchase_count(
        self, sample_l1_summary, sample_recent_items, sample_l3_distributions
    ):
        msg = build_reasoning_user_message(
            sample_l1_summary, sample_recent_items, sample_l3_distributions
        )
        assert "45 items" in msg

    def test_includes_categories(
        self, sample_l1_summary, sample_recent_items, sample_l3_distributions
    ):
        msg = build_reasoning_user_message(
            sample_l1_summary, sample_recent_items, sample_l3_distributions
        )
        assert "T-shirt" in msg

    def test_includes_recent_items(
        self, sample_l1_summary, sample_recent_items, sample_l3_distributions
    ):
        msg = build_reasoning_user_message(
            sample_l1_summary, sample_recent_items, sample_l3_distributions
        )
        assert "Casual, Minimalist" in msg
        assert "polished professional" in msg

    def test_includes_l3_distributions(
        self, sample_l1_summary, sample_recent_items, sample_l3_distributions
    ):
        msg = build_reasoning_user_message(
            sample_l1_summary, sample_recent_items, sample_l3_distributions
        )
        assert "Color Harmony" in msg
        assert "Monochromatic" in msg
        assert "I-line" in msg

    def test_handles_empty_items(self, sample_l1_summary, sample_l3_distributions):
        msg = build_reasoning_user_message(sample_l1_summary, [], sample_l3_distributions)
        assert "--- Recent Items" in msg

    def test_handles_empty_l3(self, sample_l1_summary, sample_recent_items):
        msg = build_reasoning_user_message(
            sample_l1_summary, sample_recent_items, {"shared": {}, "by_category": {}}
        )
        assert "--- Attribute Patterns" in msg


# ---------------------------------------------------------------------------
# Reasoning text composition
# ---------------------------------------------------------------------------


class TestComposeReasoningText:
    def test_has_all_nine_labels(self, sample_reasoning_json):
        text = compose_reasoning_text(sample_reasoning_json)
        for letter in "abcdefghi":
            assert f"({letter})" in text

    def test_includes_values(self, sample_reasoning_json):
        text = compose_reasoning_text(sample_reasoning_json)
        assert "Casual minimalist" in text
        assert "practical minimalist" in text

    def test_missing_fields_show_unknown(self):
        text = compose_reasoning_text({"style_mood_preference": "Casual"})
        assert "Unknown" in text

    def test_token_length_reasonable(self, sample_reasoning_json):
        text = compose_reasoning_text(sample_reasoning_json)
        # Should be well under BGE-base 512 token limit (~200 tokens expected)
        words = text.split()
        assert len(words) < 300


# ---------------------------------------------------------------------------
# Batch API request line
# ---------------------------------------------------------------------------


class TestBuildReasoningRequestLine:
    def test_returns_valid_jsonl(
        self, sample_l1_summary, sample_recent_items, sample_l3_distributions
    ):
        line = build_reasoning_request_line(
            "cust_001",
            sample_l1_summary,
            sample_recent_items,
            sample_l3_distributions,
        )
        assert isinstance(line, bytes)
        assert line.endswith(b"\n")

        # Parse as JSON
        parsed = json.loads(line.decode("utf-8"))
        assert parsed["custom_id"] == "cust_001"
        assert parsed["method"] == "POST"
        assert parsed["url"] == "/v1/responses"

    def test_contains_system_prompt(
        self, sample_l1_summary, sample_recent_items, sample_l3_distributions
    ):
        line = build_reasoning_request_line(
            "cust_001",
            sample_l1_summary,
            sample_recent_items,
            sample_l3_distributions,
        )
        parsed = json.loads(line.decode("utf-8"))
        messages = parsed["body"]["input"]
        assert messages[0]["role"] == "system"
        assert "fashion consumer analyst" in messages[0]["content"]

    def test_contains_json_schema(
        self, sample_l1_summary, sample_recent_items, sample_l3_distributions
    ):
        line = build_reasoning_request_line(
            "cust_001",
            sample_l1_summary,
            sample_recent_items,
            sample_l3_distributions,
        )
        parsed = json.loads(line.decode("utf-8"))
        text_format = parsed["body"]["text"]["format"]
        assert text_format["type"] == "json_schema"
        assert text_format["name"] == "user_reasoning_profile"
        assert text_format["strict"] is True

    def test_uses_config_model(
        self, sample_l1_summary, sample_recent_items, sample_l3_distributions
    ):
        config = ReasoningConfig(model="gpt-4.1-mini")
        line = build_reasoning_request_line(
            "cust_001",
            sample_l1_summary,
            sample_recent_items,
            sample_l3_distributions,
            config=config,
        )
        parsed = json.loads(line.decode("utf-8"))
        assert parsed["body"]["model"] == "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestParseJsonField:
    def test_dict_passthrough(self):
        assert _parse_json_field({"a": 1}) == {"a": 1}

    def test_json_string(self):
        assert _parse_json_field('{"a": 1}') == {"a": 1}

    def test_invalid_string(self):
        assert _parse_json_field("not json") == {}

    def test_none(self):
        assert _parse_json_field(None) == {}
