"""Tests for Super-Category routing, JSON schemas, and message building."""

import pytest

from src.knowledge.factual.prompts import (
    ACCESSORIES_SCHEMA,
    APPAREL_SCHEMA,
    CLOSURE_VALUES_ACCESSORIES,
    CLOSURE_VALUES_APPAREL,
    CLOSURE_VALUES_FOOTWEAR,
    FOOTWEAR_SCHEMA,
    MATERIAL_VALUES_ACCESSORIES,
    MATERIAL_VALUES_APPAREL,
    MATERIAL_VALUES_FOOTWEAR,
    SCHEMA_MAP,
    STYLE_LINEAGE_VALUES,
    STYLE_MOOD_VALUES,
    OCCASION_VALUES,
    SUPER_CATEGORY_MAP,
    _PRODUCT_GROUP_SUPER_CATEGORY,
    build_user_message,
    get_prompt_and_schema,
    map_to_canonical_slots,
    resolve_super_category,
)


class TestSuperCategoryRouting:
    """Test garment_group_name → Super-Category mapping."""

    def test_apparel_groups(self):
        apparel_groups = [
            "Jersey Fancy",
            "Jersey Basic",
            "Knitwear",
            "Under-, Nightwear",
            "Trousers",
            "Blouses",
            "Dresses Ladies",
            "Outdoor",
            "Trousers Denim",
            "Swimwear",
            "Shirts",
            "Woven/Jersey/Knitted mix Baby",
            "Shorts",
            "Dresses/Skirts girls",
            "Skirts",
            "Dressed",
        ]
        for group in apparel_groups:
            assert SUPER_CATEGORY_MAP[group] == "Apparel", f"{group} should be Apparel"

    def test_footwear_groups(self):
        assert SUPER_CATEGORY_MAP["Shoes"] == "Footwear"
        assert SUPER_CATEGORY_MAP["Socks and Tights"] == "Footwear"

    def test_accessories_groups(self):
        assert SUPER_CATEGORY_MAP["Accessories"] == "Accessories"

    def test_mixed_groups(self):
        assert SUPER_CATEGORY_MAP["Unknown"] == "Mixed"
        assert SUPER_CATEGORY_MAP["Special Offers"] == "Mixed"

    def test_all_21_groups_mapped(self):
        assert len(SUPER_CATEGORY_MAP) == 21


class TestResolveSuperCategory:
    """Test two-level resolve_super_category() routing."""

    def test_direct_apparel(self):
        assert resolve_super_category("Jersey Fancy") == "Apparel"
        assert resolve_super_category("Trousers") == "Apparel"

    def test_direct_footwear(self):
        assert resolve_super_category("Shoes") == "Footwear"
        assert resolve_super_category("Socks and Tights") == "Footwear"

    def test_direct_accessories(self):
        assert resolve_super_category("Accessories") == "Accessories"

    def test_mixed_falls_back_to_product_group(self):
        assert resolve_super_category("Unknown", "Garment Upper body") == "Apparel"
        assert resolve_super_category("Special Offers", "Shoes") == "Footwear"
        assert resolve_super_category("Unknown", "Bags") == "Accessories"

    def test_mixed_without_product_group_defaults_to_apparel(self):
        assert resolve_super_category("Unknown") == "Apparel"
        assert resolve_super_category("Unknown", None) == "Apparel"

    def test_unknown_garment_group_defaults_to_apparel(self):
        assert resolve_super_category("SomeNewCategory") == "Apparel"

    def test_product_group_fallback_coverage(self):
        """All _PRODUCT_GROUP_SUPER_CATEGORY values are valid categories."""
        valid = {"Apparel", "Footwear", "Accessories"}
        for pg, cat in _PRODUCT_GROUP_SUPER_CATEGORY.items():
            assert cat in valid, f"{pg} maps to invalid category: {cat}"


class TestJsonSchemas:
    """Test JSON schema structure validity."""

    @pytest.mark.parametrize("schema", [APPAREL_SCHEMA, FOOTWEAR_SCHEMA, ACCESSORIES_SCHEMA])
    def test_schema_has_21_properties(self, schema):
        assert len(schema["properties"]) == 21

    @pytest.mark.parametrize("schema", [APPAREL_SCHEMA, FOOTWEAR_SCHEMA, ACCESSORIES_SCHEMA])
    def test_schema_required_matches_properties(self, schema):
        assert set(schema["required"]) == set(schema["properties"].keys())

    @pytest.mark.parametrize("schema", [APPAREL_SCHEMA, FOOTWEAR_SCHEMA, ACCESSORIES_SCHEMA])
    def test_schema_no_additional_properties(self, schema):
        assert schema["additionalProperties"] is False

    def test_shared_l1_fields_in_all(self):
        shared = ["l1_material", "l1_closure", "l1_design_details", "l1_material_detail"]
        for field in shared:
            for name, schema in SCHEMA_MAP.items():
                assert field in schema["properties"], f"{field} missing from {name}"

    def test_l1_material_detail_in_all(self):
        """l1_material_detail is a shared L1 field in all schemas."""
        for name, schema in SCHEMA_MAP.items():
            assert "l1_material_detail" in schema["properties"], f"missing from {name}"
            prop = schema["properties"]["l1_material_detail"]
            assert prop["type"] == "string"

    def test_l3_tone_season_not_in_schema(self):
        """l3_tone_season is NOT in LLM schema (post-processed only)."""
        for name, schema in SCHEMA_MAP.items():
            assert "l3_tone_season" not in schema["properties"], (
                f"l3_tone_season should not be in {name} schema"
            )

    def test_l2_fields_universal(self):
        l2_fields = [
            "l2_style_mood",
            "l2_occasion",
            "l2_perceived_quality",
            "l2_trendiness",
            "l2_season_fit",
            "l2_target_impression",
            "l2_versatility",
        ]
        for field in l2_fields:
            for name, schema in SCHEMA_MAP.items():
                assert field in schema["properties"], f"{field} missing from {name}"

    def test_apparel_specific_l1(self):
        for field in ["l1_neckline", "l1_sleeve_type", "l1_fit", "l1_length"]:
            assert field in APPAREL_SCHEMA["properties"]
            assert field not in FOOTWEAR_SCHEMA["properties"]

    def test_footwear_specific_l1(self):
        for field in ["l1_toe_shape", "l1_shaft_height", "l1_heel_type", "l1_sole_type"]:
            assert field in FOOTWEAR_SCHEMA["properties"]
            assert field not in APPAREL_SCHEMA["properties"]

    def test_accessories_specific_l1(self):
        for field in ["l1_form_factor", "l1_size_scale", "l1_wearing_method", "l1_primary_function"]:
            assert field in ACCESSORIES_SCHEMA["properties"]
            assert field not in APPAREL_SCHEMA["properties"]

    def test_integer_fields_have_bounds(self):
        int_fields = ["l2_perceived_quality", "l2_versatility", "l3_visual_weight"]
        for field in int_fields:
            for schema in SCHEMA_MAP.values():
                prop = schema["properties"][field]
                assert prop["type"] == "integer"
                assert prop["minimum"] == 1
                assert prop["maximum"] == 5


class TestNewEnums:
    """Test new enum constants."""

    def test_material_enum_apparel(self):
        assert "Cotton" in MATERIAL_VALUES_APPAREL
        assert "Other" in MATERIAL_VALUES_APPAREL
        assert len(MATERIAL_VALUES_APPAREL) == 22

    def test_material_enum_footwear(self):
        assert "Leather" in MATERIAL_VALUES_FOOTWEAR
        assert "Other" in MATERIAL_VALUES_FOOTWEAR
        assert len(MATERIAL_VALUES_FOOTWEAR) == 12

    def test_material_enum_accessories(self):
        assert "Metal" in MATERIAL_VALUES_ACCESSORIES
        assert "Other" in MATERIAL_VALUES_ACCESSORIES
        assert len(MATERIAL_VALUES_ACCESSORIES) == 13

    def test_closure_enum_apparel(self):
        assert "Pullover" in CLOSURE_VALUES_APPAREL
        assert "N/A" in CLOSURE_VALUES_APPAREL
        assert len(CLOSURE_VALUES_APPAREL) == 11

    def test_closure_enum_footwear(self):
        assert "Lace-up" in CLOSURE_VALUES_FOOTWEAR
        assert "Slip-on" in CLOSURE_VALUES_FOOTWEAR
        assert len(CLOSURE_VALUES_FOOTWEAR) == 9

    def test_closure_enum_accessories(self):
        assert "Clasp" in CLOSURE_VALUES_ACCESSORIES
        assert "N/A" in CLOSURE_VALUES_ACCESSORIES
        assert len(CLOSURE_VALUES_ACCESSORIES) == 11

    def test_style_lineage_enum(self):
        assert "Scandinavian Minimalism" in STYLE_LINEAGE_VALUES
        assert "Y2K" in STYLE_LINEAGE_VALUES
        assert "Quiet Luxury" in STYLE_LINEAGE_VALUES
        assert len(STYLE_LINEAGE_VALUES) == 45

    def test_material_enum_in_schema(self):
        """l1_material has enum in all schemas."""
        assert APPAREL_SCHEMA["properties"]["l1_material"]["enum"] == MATERIAL_VALUES_APPAREL
        assert FOOTWEAR_SCHEMA["properties"]["l1_material"]["enum"] == MATERIAL_VALUES_FOOTWEAR
        assert ACCESSORIES_SCHEMA["properties"]["l1_material"]["enum"] == MATERIAL_VALUES_ACCESSORIES

    def test_closure_enum_in_schema(self):
        """l1_closure has enum in all schemas."""
        assert APPAREL_SCHEMA["properties"]["l1_closure"]["enum"] == CLOSURE_VALUES_APPAREL
        assert FOOTWEAR_SCHEMA["properties"]["l1_closure"]["enum"] == CLOSURE_VALUES_FOOTWEAR
        assert ACCESSORIES_SCHEMA["properties"]["l1_closure"]["enum"] == CLOSURE_VALUES_ACCESSORIES

    def test_style_lineage_enum_in_schema(self):
        """l3_style_lineage items have enum in all schemas."""
        for schema in SCHEMA_MAP.values():
            items = schema["properties"]["l3_style_lineage"]["items"]
            assert "enum" in items
            assert items["enum"] == STYLE_LINEAGE_VALUES

    def test_style_mood_expanded(self):
        """style_mood includes new values."""
        assert "Cottagecore" in STYLE_MOOD_VALUES
        assert "Dark-academic" in STYLE_MOOD_VALUES
        assert "Y2K" in STYLE_MOOD_VALUES
        assert "Quiet-luxury" in STYLE_MOOD_VALUES

    def test_occasion_expanded(self):
        """occasion includes new values."""
        assert "School" in OCCASION_VALUES
        assert "Ceremony" in OCCASION_VALUES


class TestMessageBuilding:
    """Test build_user_message output format."""

    def test_text_only(self):
        article = {"product_type_name": "T-shirt", "colour_group_name": "Black"}
        content = build_user_message(article, "A black cotton t-shirt", None)
        assert len(content) == 1
        assert content[0]["type"] == "input_text"
        assert "T-shirt" in content[0]["text"]
        assert "black cotton t-shirt" in content[0]["text"]

    def test_with_image(self):
        article = {"product_type_name": "Sneaker"}
        content = build_user_message(article, "White canvas shoe", "abc123base64")
        assert len(content) == 2
        assert content[0]["type"] == "input_text"
        assert content[1]["type"] == "input_image"
        assert "abc123base64" in content[1]["image_url"]
        assert content[1]["detail"] == "low"

    def test_no_description(self):
        article = {"product_type_name": "Hat"}
        content = build_user_message(article, "", None)
        assert "No text description available" in content[0]["text"]

    def test_nan_description(self):
        article = {"product_type_name": "Hat"}
        content = build_user_message(article, "nan", None)
        assert "No text description available" in content[0]["text"]

    def test_dynamic_suffix_present(self):
        """build_user_message appends per-item specificity suffix."""
        article = {"product_type_name": "T-shirt"}
        content = build_user_message(article, "A cotton tee", None)
        text = content[0]["text"]
        assert "Analyze THIS specific T-shirt" in text
        assert "Do not assign generic attributes" in text

    def test_dynamic_suffix_uses_product_type(self):
        """Suffix uses the product_type_name from article metadata."""
        article = {"product_type_name": "Sneaker"}
        content = build_user_message(article, "A shoe", None)
        assert "Analyze THIS specific Sneaker" in content[0]["text"]


class TestGetPromptAndSchema:
    """Test prompt/schema retrieval."""

    @pytest.mark.parametrize("category", ["Apparel", "Footwear", "Accessories"])
    def test_valid_categories(self, category):
        prompt, schema = get_prompt_and_schema(category)
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Unknown super_category"):
            get_prompt_and_schema("InvalidCategory")

    @pytest.mark.parametrize("category", ["Apparel", "Footwear", "Accessories"])
    def test_cross_attribute_rules_in_prompt(self, category):
        """System prompts contain cross-attribute consistency rules."""
        prompt, _ = get_prompt_and_schema(category)
        assert "Cross-Attribute Consistency Rules" in prompt
        assert "visual_weight = FORM and VOLUME" in prompt
        assert "Per-Item Specificity" in prompt


class TestCanonicalSlotMapping:
    """Test category-specific → canonical slot mapping."""

    def test_apparel_mapping(self):
        knowledge = {
            "l1_material": "Cotton",
            "l1_neckline": "Crew",
            "l1_sleeve_type": "Short",
            "l1_fit": "Regular",
            "l1_length": "Hip",
            "l3_silhouette": "I-line",
            "l3_proportion_effect": "Streamlining",
        }
        result = map_to_canonical_slots(knowledge, "Apparel")
        assert result["l1_slot4"] == "Crew"
        assert result["l1_slot5"] == "Short"
        assert result["l1_slot6"] == "Regular"
        assert result["l1_slot7"] == "Hip"
        assert result["l3_slot6"] == "I-line"
        assert result["l3_slot7"] == "Streamlining"
        assert "l1_neckline" not in result

    def test_footwear_mapping(self):
        knowledge = {
            "l1_toe_shape": "Round",
            "l1_shaft_height": "Low-top",
            "l1_heel_type": "Flat",
            "l1_sole_type": "Rubber",
            "l3_foot_silhouette": "Streamlined",
            "l3_height_effect": "Grounding",
        }
        result = map_to_canonical_slots(knowledge, "Footwear")
        assert result["l1_slot4"] == "Round"
        assert result["l3_slot6"] == "Streamlined"

    def test_material_detail_not_slotted(self):
        """l1_material_detail is a shared L1 field and not mapped to slots."""
        knowledge = {
            "l1_material": "Cotton",
            "l1_material_detail": "100% cotton, jersey knit",
            "l1_neckline": "Crew",
            "l1_sleeve_type": "Short",
            "l1_fit": "Regular",
            "l1_length": "Hip",
            "l3_silhouette": "I-line",
            "l3_proportion_effect": "Streamlining",
        }
        result = map_to_canonical_slots(knowledge, "Apparel")
        # material_detail stays as-is (not slotted)
        assert result.get("l1_material_detail") == "100% cotton, jersey knit"
