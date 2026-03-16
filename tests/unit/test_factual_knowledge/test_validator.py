"""Tests for L1+L2+L3 knowledge validation and domain consistency rules."""

import numpy as np

from src.knowledge.factual.validator import (
    DomainViolation,
    ValidationResult,
    _L1_SHARED,
    _L3_POSTPROCESSED,
    _L3_SHARED,
    validate_domain_consistency,
    validate_final_knowledge,
    validate_knowledge,
)


def _make_valid_apparel() -> dict:
    """Create a valid Apparel knowledge dict (21 fields, no tone_season)."""
    return {
        "l1_material": "Cotton",
        "l1_closure": "Pullover",
        "l1_design_details": ["ribbed neckline"],
        "l1_material_detail": "100% cotton, jersey knit",
        "l1_neckline": "Crew",
        "l1_sleeve_type": "Short",
        "l1_fit": "Slim",
        "l1_length": "Hip",
        "l2_style_mood": ["Casual", "Minimalist"],
        "l2_occasion": ["Everyday"],
        "l2_perceived_quality": 3,
        "l2_trendiness": "Classic",
        "l2_season_fit": "All-season",
        "l2_target_impression": "effortless everyday essential",
        "l2_versatility": 5,
        "l3_color_harmony": "Monochromatic",
        "l3_coordination_role": "Basic",
        "l3_visual_weight": 2,
        "l3_style_lineage": ["Scandinavian Minimalism"],
        "l3_silhouette": "I-line",
        "l3_proportion_effect": "Streamlining",
    }


def _make_valid_footwear() -> dict:
    """Create a valid Footwear knowledge dict (21 fields, no tone_season)."""
    return {
        "l1_material": "Canvas",
        "l1_closure": "Lace-up",
        "l1_design_details": ["rubber toe cap"],
        "l1_material_detail": "cotton canvas, vulcanized rubber sole",
        "l1_toe_shape": "Round",
        "l1_shaft_height": "Low-top",
        "l1_heel_type": "Flat",
        "l1_sole_type": "Rubber",
        "l2_style_mood": ["Casual"],
        "l2_occasion": ["Everyday"],
        "l2_perceived_quality": 3,
        "l2_trendiness": "Classic",
        "l2_season_fit": "All-season",
        "l2_target_impression": "clean casual",
        "l2_versatility": 5,
        "l3_color_harmony": "Monochromatic",
        "l3_coordination_role": "Basic",
        "l3_visual_weight": 2,
        "l3_style_lineage": ["Americana Prep"],
        "l3_foot_silhouette": "Streamlined",
        "l3_height_effect": "Grounding",
    }


class TestFieldLists:
    """Test internal field list constants."""

    def test_l1_shared_includes_material_detail(self):
        assert "l1_material_detail" in _L1_SHARED

    def test_l3_shared_excludes_tone_season(self):
        assert "l3_tone_season" not in _L3_SHARED

    def test_l3_postprocessed_has_tone_season(self):
        assert "l3_tone_season" in _L3_POSTPROCESSED


class TestValidKnowledge:
    """Test validation passes for correct inputs."""

    def test_valid_apparel(self):
        result = validate_knowledge(_make_valid_apparel(), "Apparel")
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_valid_footwear(self):
        result = validate_knowledge(_make_valid_footwear(), "Footwear")
        assert result.is_valid is True
        assert len(result.errors) == 0


class TestMissingFields:
    """Test validation catches missing required fields."""

    def test_missing_l1_field(self):
        knowledge = _make_valid_apparel()
        del knowledge["l1_material"]
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is False
        assert any("l1_material" in e for e in result.errors)

    def test_missing_l2_field(self):
        knowledge = _make_valid_apparel()
        del knowledge["l2_style_mood"]
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is False
        assert any("l2_style_mood" in e for e in result.errors)

    def test_missing_l3_field(self):
        knowledge = _make_valid_apparel()
        del knowledge["l3_silhouette"]
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is False

    def test_missing_material_detail(self):
        knowledge = _make_valid_apparel()
        del knowledge["l1_material_detail"]
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is False
        assert any("l1_material_detail" in e for e in result.errors)


class TestTypeErrors:
    """Test validation catches type mismatches."""

    def test_integer_as_string(self):
        knowledge = _make_valid_apparel()
        knowledge["l2_perceived_quality"] = "three"
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is False
        assert any("l2_perceived_quality" in e for e in result.errors)

    def test_integer_out_of_range(self):
        knowledge = _make_valid_apparel()
        knowledge["l2_perceived_quality"] = 6
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is False

    def test_string_as_integer(self):
        knowledge = _make_valid_apparel()
        knowledge["l1_material"] = 123
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is False

    def test_array_as_string(self):
        knowledge = _make_valid_apparel()
        knowledge["l2_style_mood"] = "Casual"
        result = validate_knowledge(knowledge, "Apparel")
        # Should fail since it expects array
        assert result.is_valid is False


class TestNdarrayCompat:
    """Test validation accepts np.ndarray for array fields (Parquet roundtrip)."""

    def test_ndarray_style_mood_valid(self):
        knowledge = _make_valid_apparel()
        knowledge["l2_style_mood"] = np.array(["Casual", "Minimalist"])
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_ndarray_design_details_valid(self):
        knowledge = _make_valid_apparel()
        knowledge["l1_design_details"] = np.array(["ribbed neckline"])
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_ndarray_style_lineage_valid(self):
        knowledge = _make_valid_apparel()
        knowledge["l3_style_lineage"] = np.array(["Scandinavian Minimalism"])
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_ndarray_occasion_valid(self):
        knowledge = _make_valid_apparel()
        knowledge["l2_occasion"] = np.array(["Everyday"])
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_ndarray_with_invalid_enum_warns(self):
        knowledge = _make_valid_apparel()
        knowledge["l2_style_mood"] = np.array(["Casual", "InvalidMood"])
        result = validate_knowledge(knowledge, "Apparel")
        assert result.is_valid is True  # Enum mismatch is warning, not error
        assert len(result.warnings) > 0


class TestEnumWarnings:
    """Test validation warns on out-of-enum values."""

    def test_invalid_enum_string(self):
        knowledge = _make_valid_apparel()
        knowledge["l2_trendiness"] = "SuperTrendy"
        result = validate_knowledge(knowledge, "Apparel")
        # Enum mismatch is a warning, not error
        assert len(result.warnings) > 0

    def test_invalid_enum_in_array(self):
        knowledge = _make_valid_apparel()
        knowledge["l2_style_mood"] = ["Casual", "InvalidMood"]
        result = validate_knowledge(knowledge, "Apparel")
        assert len(result.warnings) > 0


class TestCategoryMismatch:
    """Test validation with wrong category fields."""

    def test_apparel_validated_as_footwear(self):
        # Apparel knowledge missing Footwear-specific fields
        knowledge = _make_valid_apparel()
        result = validate_knowledge(knowledge, "Footwear")
        assert result.is_valid is False
        assert any("l1_toe_shape" in e for e in result.errors)


class TestInvalidCategory:
    """Test validation with unknown category."""

    def test_unknown_category(self):
        result = validate_knowledge({}, "InvalidCategory")
        assert result.is_valid is False
        assert any("Unknown super_category" in e for e in result.errors)


class TestNullValues:
    """Test validation handles null values."""

    def test_null_field_warns(self):
        knowledge = _make_valid_apparel()
        knowledge["l1_material"] = None
        result = validate_knowledge(knowledge, "Apparel")
        # Null is a warning, not an error (field is present but null)
        assert len(result.warnings) > 0


class TestValidationResultNamedTuple:
    """Test ValidationResult structure."""

    def test_namedtuple_fields(self):
        result = ValidationResult(is_valid=True, errors=[], warnings=["minor"])
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == ["minor"]


class TestValidateFinalKnowledge:
    """Test validate_final_knowledge (22 fields with tone_season)."""

    def test_valid_final_apparel(self):
        knowledge = _make_valid_apparel()
        knowledge["l3_tone_season"] = "Cool-Winter"
        result = validate_final_knowledge(knowledge, "Apparel")
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_final_missing_tone_season(self):
        knowledge = _make_valid_apparel()
        # No tone_season → missing field error
        result = validate_final_knowledge(knowledge, "Apparel")
        assert result.is_valid is False
        assert any("l3_tone_season" in e for e in result.errors)

    def test_final_has_22_expected_fields(self):
        """validate_final_knowledge expects 22 fields = 21 LLM + 1 post-processed."""
        knowledge = _make_valid_apparel()
        knowledge["l3_tone_season"] = "Cool-Winter"
        result = validate_final_knowledge(knowledge, "Apparel")
        assert result.is_valid is True

    def test_final_invalid_category(self):
        result = validate_final_knowledge({}, "InvalidCategory")
        assert result.is_valid is False


class TestDomainViolationNamedTuple:
    """Test DomainViolation structure."""

    def test_namedtuple_fields(self):
        v = DomainViolation("test_rule", "Error", "test description")
        assert v.rule_name == "test_rule"
        assert v.severity == "Error"
        assert v.description == "test description"


class TestDomainConsistencyCoordWeight:
    """Test coordination_role × visual_weight rules."""

    def test_basic_high_weight_error(self):
        k = _make_valid_apparel()
        k["l3_coordination_role"] = "Basic"
        k["l3_visual_weight"] = 4
        violations = validate_domain_consistency(k, "Apparel")
        errors = [v for v in violations if v.rule_name == "coordination_x_visual_weight"]
        assert len(errors) >= 1
        assert errors[0].severity == "Error"

    def test_basic_low_weight_ok(self):
        k = _make_valid_apparel()
        k["l3_coordination_role"] = "Basic"
        k["l3_visual_weight"] = 2
        violations = validate_domain_consistency(k, "Apparel")
        coord_errors = [v for v in violations if v.rule_name == "coordination_x_visual_weight"]
        assert len(coord_errors) == 0

    def test_statement_low_weight_error(self):
        k = _make_valid_apparel()
        k["l3_coordination_role"] = "Statement"
        k["l3_visual_weight"] = 1
        k["l1_fit"] = "Oversized"  # Avoid fit conflict
        k["l3_silhouette"] = "O-line"  # Avoid silhouette conflict
        violations = validate_domain_consistency(k, "Apparel")
        errors = [v for v in violations if v.rule_name == "coordination_x_visual_weight"]
        assert len(errors) >= 1


class TestDomainConsistencyApparel:
    """Test Apparel-specific domain rules."""

    def test_iline_high_weight_error(self):
        k = _make_valid_apparel()
        k["l3_silhouette"] = "I-line"
        k["l3_visual_weight"] = 4
        violations = validate_domain_consistency(k, "Apparel")
        errors = [v for v in violations if v.rule_name == "silhouette_x_visual_weight"]
        assert len(errors) >= 1

    def test_oline_low_weight_error(self):
        k = _make_valid_apparel()
        k["l3_silhouette"] = "O-line"
        k["l3_visual_weight"] = 1
        k["l1_fit"] = "Oversized"
        violations = validate_domain_consistency(k, "Apparel")
        errors = [v for v in violations if v.rule_name == "silhouette_x_visual_weight"]
        assert len(errors) >= 1

    def test_slim_high_weight_error(self):
        k = _make_valid_apparel()
        k["l1_fit"] = "Slim"
        k["l3_visual_weight"] = 4
        violations = validate_domain_consistency(k, "Apparel")
        errors = [v for v in violations if v.rule_name == "fit_x_visual_weight"]
        assert len(errors) >= 1

    def test_oversized_low_weight_error(self):
        k = _make_valid_apparel()
        k["l1_fit"] = "Oversized"
        k["l3_visual_weight"] = 1
        violations = validate_domain_consistency(k, "Apparel")
        errors = [v for v in violations if v.rule_name == "fit_x_visual_weight"]
        assert len(errors) >= 1

    def test_sleeveless_winter_error(self):
        k = _make_valid_apparel()
        k["l1_sleeve_type"] = "Sleeveless"
        k["l2_season_fit"] = "Winter"
        violations = validate_domain_consistency(k, "Apparel")
        errors = [v for v in violations if v.rule_name == "sleeve_x_season"]
        assert len(errors) == 1

    def test_strapless_long_sleeve_error(self):
        k = _make_valid_apparel()
        k["l1_neckline"] = "Strapless"
        k["l1_sleeve_type"] = "Long"
        violations = validate_domain_consistency(k, "Apparel")
        errors = [v for v in violations if v.rule_name == "neckline_x_sleeve"]
        assert len(errors) == 1

    def test_strapless_sleeveless_ok(self):
        k = _make_valid_apparel()
        k["l1_neckline"] = "Strapless"
        k["l1_sleeve_type"] = "Sleeveless"
        violations = validate_domain_consistency(k, "Apparel")
        errors = [v for v in violations if v.rule_name == "neckline_x_sleeve"]
        assert len(errors) == 0


class TestDomainConsistencyFootwear:
    """Test Footwear-specific domain rules."""

    def test_foam_winter_error(self):
        k = _make_valid_footwear()
        k["l1_sole_type"] = "Foam"
        k["l2_season_fit"] = "Winter"
        violations = validate_domain_consistency(k, "Footwear")
        errors = [v for v in violations if v.rule_name == "sole_x_season"]
        assert len(errors) == 1

    def test_stiletto_outdoor_warning(self):
        k = _make_valid_footwear()
        k["l1_heel_type"] = "Stiletto"
        k["l2_occasion"] = ["Outdoor"]
        violations = validate_domain_consistency(k, "Footwear")
        warns = [v for v in violations if v.rule_name == "heel_x_occasion"]
        assert len(warns) == 1
        assert warns[0].severity == "Warning"


class TestDomainConsistencyAccessories:
    """Test Accessories-specific domain rules."""

    def _make_valid_accessories(self) -> dict:
        return {
            "l1_material": "Leather",
            "l1_closure": "Magnetic",
            "l1_design_details": ["gold hardware"],
            "l1_material_detail": "full-grain cowhide",
            "l1_form_factor": "Small",
            "l1_size_scale": "Small",
            "l1_wearing_method": "Crossbody",
            "l1_primary_function": "Storage",
            "l2_style_mood": ["Classic"],
            "l2_occasion": ["Everyday"],
            "l2_perceived_quality": 4,
            "l2_trendiness": "Classic",
            "l2_season_fit": "All-season",
            "l2_target_impression": "polished companion",
            "l2_versatility": 4,
            "l3_color_harmony": "Monochromatic",
            "l3_coordination_role": "Finishing",
            "l3_visual_weight": 2,
            "l3_style_lineage": ["French Chic"],
            "l3_visual_form": "Structured",
            "l3_styling_effect": "Cohesion",
        }

    def test_storage_mini_error(self):
        k = self._make_valid_accessories()
        k["l1_primary_function"] = "Storage"
        k["l1_form_factor"] = "Mini"
        violations = validate_domain_consistency(k, "Accessories")
        errors = [v for v in violations if v.rule_name == "function_x_form_factor"]
        assert len(errors) == 1

    def test_wrist_large_error(self):
        k = self._make_valid_accessories()
        k["l1_wearing_method"] = "Wrist"
        k["l1_size_scale"] = "Large"
        violations = validate_domain_consistency(k, "Accessories")
        errors = [v for v in violations if v.rule_name == "wearing_x_size_scale"]
        assert len(errors) == 1

    def test_finger_small_ok(self):
        k = self._make_valid_accessories()
        k["l1_wearing_method"] = "Finger"
        k["l1_size_scale"] = "Small"
        violations = validate_domain_consistency(k, "Accessories")
        errors = [v for v in violations if v.rule_name == "wearing_x_size_scale"]
        assert len(errors) == 0


class TestDomainConsistencyWarnings:
    """Test warning-severity domain rules."""

    def test_bohemian_formal_warning(self):
        k = _make_valid_apparel()
        k["l2_style_mood"] = ["Bohemian"]
        k["l2_occasion"] = ["Formal"]
        violations = validate_domain_consistency(k, "Apparel")
        warns = [v for v in violations if v.rule_name == "mood_x_occasion"]
        assert len(warns) == 1
        assert warns[0].severity == "Warning"

    def test_basic_complementary_warning(self):
        k = _make_valid_apparel()
        k["l3_coordination_role"] = "Basic"
        k["l3_color_harmony"] = "Complementary"
        violations = validate_domain_consistency(k, "Apparel")
        warns = [v for v in violations if v.rule_name == "coordination_x_harmony"]
        assert len(warns) >= 1

    def test_accent_monochromatic_warning(self):
        k = _make_valid_apparel()
        k["l3_coordination_role"] = "Accent"
        k["l3_color_harmony"] = "Monochromatic"
        violations = validate_domain_consistency(k, "Apparel")
        warns = [v for v in violations if v.rule_name == "coordination_x_harmony"]
        assert len(warns) >= 1

    def test_punk_classic_lineage_mood_warning(self):
        k = _make_valid_apparel()
        k["l3_style_lineage"] = ["Punk"]
        k["l2_style_mood"] = ["Classic"]
        violations = validate_domain_consistency(k, "Apparel")
        warns = [v for v in violations if v.rule_name == "lineage_x_mood"]
        assert len(warns) == 1

    def test_no_contradiction_ok(self):
        k = _make_valid_apparel()
        k["l3_style_lineage"] = ["Scandinavian Minimalism"]
        k["l2_style_mood"] = ["Minimalist"]
        violations = validate_domain_consistency(k, "Apparel")
        warns = [v for v in violations if v.rule_name == "lineage_x_mood"]
        assert len(warns) == 0
