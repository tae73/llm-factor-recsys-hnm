"""Tests for LLM-as-Judge common protocol."""

from src.eval_prompt.judge import (
    DIMENSION_NAMES,
    JudgeConfig,
    JudgeDimension,
    JudgeReport,
    JudgeResult,
    build_judge_schema,
    build_judge_system_prompt,
)


class TestDimensionNames:
    """Test dimension name constants."""

    def test_has_five_dimensions(self):
        assert len(DIMENSION_NAMES) == 5

    def test_contains_all_dimensions(self):
        expected = {"accuracy", "specificity", "coherence", "source_alignment", "informativeness"}
        assert set(DIMENSION_NAMES) == expected


class TestJudgeConfig:
    """Test JudgeConfig defaults."""

    def test_defaults(self):
        config = JudgeConfig()
        assert config.model == "gpt-4.1-mini"
        assert config.sample_size == 50
        assert config.max_concurrent == 10
        assert config.temperature == 0.0
        assert config.pass_threshold == 3.5

    def test_custom_values(self):
        config = JudgeConfig(model="gpt-4.1", sample_size=100, pass_threshold=4.0)
        assert config.model == "gpt-4.1"
        assert config.sample_size == 100
        assert config.pass_threshold == 4.0


class TestJudgeResult:
    """Test JudgeResult construction."""

    def test_overall_score(self):
        result = JudgeResult(
            item_id="test_001",
            scores={"accuracy": 4, "specificity": 3, "coherence": 5,
                    "source_alignment": 4, "informativeness": 4},
            justifications={"accuracy": "Good", "specificity": "OK", "coherence": "Great",
                           "source_alignment": "Good", "informativeness": "Good"},
            overall_score=4.0,
        )
        assert result.item_id == "test_001"
        assert result.overall_score == 4.0
        assert result.scores["coherence"] == 5


class TestJudgeReport:
    """Test JudgeReport construction."""

    def test_empty_report(self):
        report = JudgeReport(
            results=[],
            per_dimension_mean={"accuracy": 0.0},
            overall_mean=0.0,
            n_evaluated=0,
            n_passed=0,
            pass_rate=0.0,
        )
        assert report.n_evaluated == 0
        assert report.pass_rate == 0.0

    def test_report_with_results(self):
        r1 = JudgeResult("a", {"accuracy": 4}, {"accuracy": "Good"}, 4.0)
        r2 = JudgeResult("b", {"accuracy": 3}, {"accuracy": "OK"}, 3.0)
        report = JudgeReport(
            results=[r1, r2],
            per_dimension_mean={"accuracy": 3.5},
            overall_mean=3.5,
            n_evaluated=2,
            n_passed=1,
            pass_rate=0.5,
        )
        assert report.n_evaluated == 2
        assert report.n_passed == 1


class TestBuildJudgeSchema:
    """Test JSON schema builder."""

    def test_schema_structure(self):
        dims = [
            JudgeDimension("accuracy", "Test accuracy"),
            JudgeDimension("specificity", "Test specificity"),
        ]
        schema = build_judge_schema(dims)
        assert schema["type"] == "object"
        assert "accuracy_score" in schema["properties"]
        assert "accuracy_justification" in schema["properties"]
        assert "specificity_score" in schema["properties"]
        assert "specificity_justification" in schema["properties"]
        assert len(schema["required"]) == 4

    def test_all_five_dimensions(self):
        dims = [JudgeDimension(name, f"Test {name}") for name in DIMENSION_NAMES]
        schema = build_judge_schema(dims)
        assert len(schema["properties"]) == 10  # 5 scores + 5 justifications
        assert len(schema["required"]) == 10

    def test_score_is_integer(self):
        dims = [JudgeDimension("accuracy", "Test")]
        schema = build_judge_schema(dims)
        assert schema["properties"]["accuracy_score"]["type"] == "integer"

    def test_justification_is_string(self):
        dims = [JudgeDimension("accuracy", "Test")]
        schema = build_judge_schema(dims)
        assert schema["properties"]["accuracy_justification"]["type"] == "string"


class TestBuildJudgeSystemPrompt:
    """Test system prompt builder."""

    def test_contains_domain(self):
        dims = [JudgeDimension("accuracy", "Test accuracy")]
        prompt = build_judge_system_prompt("factual_knowledge", dims)
        assert "factual_knowledge" in prompt

    def test_contains_dimensions(self):
        dims = [
            JudgeDimension("accuracy", "Accuracy description"),
            JudgeDimension("specificity", "Specificity description"),
        ]
        prompt = build_judge_system_prompt("user_profile", dims)
        assert "accuracy" in prompt
        assert "specificity" in prompt
        assert "Accuracy description" in prompt

    def test_contains_scoring_scale(self):
        dims = [JudgeDimension("accuracy", "Test")]
        prompt = build_judge_system_prompt("test", dims)
        assert "1-5" in prompt
        assert "1 = Very poor" in prompt
        assert "5 = Excellent" in prompt
