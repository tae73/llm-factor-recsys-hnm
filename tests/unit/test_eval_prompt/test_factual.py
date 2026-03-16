"""Tests for factual knowledge evaluation orchestration."""

from src.eval_prompt.factual import (
    FACTUAL_DIMENSIONS,
    FactualEvalConfig,
    FactualEvalReport,
    build_factual_judge_message,
)
from src.eval_prompt.judge import JudgeConfig


class TestFactualDimensions:
    """Test factual dimension definitions."""

    def test_has_five_dimensions(self):
        assert len(FACTUAL_DIMENSIONS) == 5

    def test_dimension_names(self):
        names = [d.name for d in FACTUAL_DIMENSIONS]
        expected = ["accuracy", "specificity", "coherence", "source_alignment", "informativeness"]
        assert names == expected

    def test_descriptions_not_empty(self):
        for d in FACTUAL_DIMENSIONS:
            assert len(d.description) > 10, f"Description too short for {d.name}"


class TestFactualEvalConfig:
    """Test FactualEvalConfig defaults."""

    def test_defaults(self):
        config = FactualEvalConfig()
        assert config.token_budget_limit == 512
        assert config.run_judge is True
        assert isinstance(config.judge_config, JudgeConfig)

    def test_skip_judge(self):
        config = FactualEvalConfig(run_judge=False)
        assert config.run_judge is False


class TestBuildFactualJudgeMessage:
    """Test factual judge message construction."""

    def test_text_only(self):
        meta = {
            "product_type_name": "T-shirt",
            "colour_group_name": "Black",
            "detail_desc": "A basic cotton t-shirt",
            "garment_group_name": "Jersey Basic",
        }
        knowledge = {
            "l1_material": "Cotton",
            "l2_style_mood": ["Casual"],
            "l3_visual_weight": 2,
        }
        result = build_factual_judge_message(meta, knowledge)
        assert isinstance(result, str)
        assert "T-shirt" in result
        assert "Cotton" in result
        assert "Evaluate" in result

    def test_with_image(self):
        meta = {"product_type_name": "Dress", "colour_group_name": "Red",
                "detail_desc": "Red dress", "garment_group_name": "Dresses"}
        knowledge = {"l1_material": "Silk"}
        result = build_factual_judge_message(meta, knowledge, image_b64="abc123")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "input_image"
        assert result[1]["type"] == "input_text"

    def test_missing_metadata_fields(self):
        meta = {}
        knowledge = {"l1_material": "Cotton"}
        result = build_factual_judge_message(meta, knowledge)
        assert "N/A" in result  # Default for missing fields

    def test_only_includes_attribute_fields(self):
        knowledge = {
            "l1_material": "Cotton",
            "l2_style_mood": ["Casual"],
            "article_id": "123456",  # Should NOT appear as attribute
            "non_attribute": "value",  # Should NOT appear
        }
        result = build_factual_judge_message({}, knowledge)
        assert "l1_material" in result
        assert "l2_style_mood" in result
        assert "article_id" not in result or "article_id: 123456" not in result


class TestFactualEvalReport:
    """Test FactualEvalReport construction."""

    def test_report_without_judge(self):
        from src.eval_prompt.structural import (
            CoverageResult,
            DomainCheckResult,
            DistributionResult,
            SchemaCheckResult,
            TokenBudgetResult,
        )

        report = FactualEvalReport(
            coverage=CoverageResult({"a": 1.0}, 1.0, 10),
            schema=SchemaCheckResult(10, 0, {}, {}),
            domain=DomainCheckResult(0, 0, 0, {}),
            distributions=DistributionResult({}, {}, {}),
            token_budget=TokenBudgetResult(100, 95, 200, 250, 300, 0, 0.0, 512),
            judge=None,
            timestamp="2024-01-01T00:00:00Z",
        )
        assert report.judge is None
        assert report.coverage.overall_coverage == 1.0
        assert report.schema.n_valid == 10
