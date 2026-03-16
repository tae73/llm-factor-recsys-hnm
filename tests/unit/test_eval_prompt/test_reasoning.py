"""Tests for user reasoning knowledge evaluation orchestration."""

from src.eval_prompt.reasoning import (
    REASONING_DIMENSIONS,
    ReasoningEvalConfig,
    ReasoningEvalReport,
    build_reasoning_judge_message,
)


class TestReasoningDimensions:
    """Test reasoning dimension definitions."""

    def test_has_five_dimensions(self):
        assert len(REASONING_DIMENSIONS) == 5

    def test_dimension_names(self):
        names = [d.name for d in REASONING_DIMENSIONS]
        expected = ["accuracy", "specificity", "coherence", "source_alignment", "informativeness"]
        assert names == expected

    def test_descriptions_not_empty(self):
        for d in REASONING_DIMENSIONS:
            assert len(d.description) > 10, f"Description too short for {d.name}"

    def test_descriptions_differ_from_factual(self):
        from src.eval_prompt.factual import FACTUAL_DIMENSIONS

        for rd, fd in zip(REASONING_DIMENSIONS, FACTUAL_DIMENSIONS):
            assert rd.description != fd.description, (
                f"Reasoning and factual descriptions identical for {rd.name}"
            )


class TestReasoningEvalConfig:
    """Test ReasoningEvalConfig defaults."""

    def test_defaults(self):
        config = ReasoningEvalConfig()
        assert config.token_budget_limit == 512
        assert config.run_judge is True
        assert isinstance(config.generic_markers, tuple)
        assert "Unknown" in config.generic_markers

    def test_skip_judge(self):
        config = ReasoningEvalConfig(run_judge=False)
        assert config.run_judge is False


class TestBuildReasoningJudgeMessage:
    """Test reasoning judge message construction."""

    def test_basic_message(self):
        l1_summary = {
            "n_purchases": 42,
            "n_unique_types": 10,
            "category_diversity": 0.75,
            "top_categories_json": '{"Trousers": 0.4, "T-shirts": 0.3}',
            "avg_price_quintile": 3.0,
            "online_ratio": 0.6,
        }
        recent_items_l2 = [
            {
                "super_category": "Apparel",
                "l2_style_mood": ["Casual"],
                "l2_occasion": ["Everyday"],
                "l2_perceived_quality": 3,
                "l2_trendiness": "Classic",
                "l2_season_fit": "All-season",
                "l2_target_impression": "clean casual",
                "l2_versatility": 4,
            }
        ]
        l3_distributions = {
            "shared": {
                "l3_color_harmony": {"Monochromatic": 0.5, "Neutral": 0.3},
                "l3_visual_weight": {"mean": 2.5, "std": 0.8},
            },
            "by_category": {},
        }
        reasoning_json = {
            "style_mood_preference": "Casual minimalist",
            "occasion_preference": "Everyday basics",
            "quality_price_tendency": "Mid-range quintile 3",
            "trend_sensitivity": "Classic core",
            "seasonal_pattern": "Year-round",
            "form_preference": "Slim streamlined",
            "color_tendency": "Neutral palette",
            "coordination_tendency": "Basic foundation",
            "identity_summary": "A practical minimalist",
        }

        result = build_reasoning_judge_message(
            l1_summary, recent_items_l2, l3_distributions, reasoning_json
        )
        assert isinstance(result, str)
        assert "Source Data" in result
        assert "Generated Reasoning Profile" in result
        assert "style_mood_preference" in result
        assert "Casual minimalist" in result

    def test_empty_reasoning(self):
        result = build_reasoning_judge_message(
            {"n_purchases": 0}, [], {}, {}
        )
        assert "Generated Reasoning Profile" in result


class TestReasoningEvalReport:
    """Test ReasoningEvalReport construction."""

    def test_report_without_judge(self):
        from src.eval_prompt.structural import (
            CompletenessResult,
            CoverageResult,
            DiscriminabilityResult,
            TokenBudgetResult,
        )

        report = ReasoningEvalReport(
            completeness=CompletenessResult({"a": 0.9}, 0.9, 1, 0),
            discriminability=DiscriminabilityResult(0.3, 0.25, {}, 50.0),
            coverage=CoverageResult({"reasoning_text": 0.95}, 0.95, 100),
            token_budget=TokenBudgetResult(80, 75, 150, 180, 200, 0, 0.0, 512),
            judge=None,
            timestamp="2024-01-01T00:00:00Z",
        )
        assert report.judge is None
        assert report.completeness.overall_completeness == 0.9
        assert report.discriminability.mean_pairwise_sim == 0.3
