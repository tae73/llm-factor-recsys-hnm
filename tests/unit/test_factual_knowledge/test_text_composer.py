"""Tests for factual text composition and ablation variants."""

from src.knowledge.factual.text_composer import (
    LAYER_COMBOS,
    _L1_SHARED,
    _L3_SHARED_LABELS,
    build_all_ablation_texts,
    construct_factual_text,
)


def _make_article_meta() -> dict:
    return {
        "product_type_name": "T-shirt",
        "product_group_name": "Garment Upper body",
        "colour_group_name": "Black",
        "graphical_appearance_name": "Solid",
        "section_name": "Menswear",
    }


def _make_full_knowledge() -> dict:
    return {
        # L1
        "l1_material": "Cotton",
        "l1_closure": "Pullover",
        "l1_design_details": ["ribbed neckline"],
        "l1_material_detail": "100% cotton, jersey knit",
        "l1_neckline": "Crew",
        "l1_sleeve_type": "Short",
        "l1_fit": "Slim",
        "l1_length": "Hip",
        # L2
        "l2_style_mood": ["Casual", "Minimalist"],
        "l2_occasion": ["Everyday"],
        "l2_perceived_quality": 3,
        "l2_trendiness": "Classic",
        "l2_season_fit": "All-season",
        "l2_target_impression": "effortless everyday essential",
        "l2_versatility": 5,
        # L3 (includes post-processed tone_season)
        "l3_color_harmony": "Monochromatic",
        "l3_tone_season": "Cool-Winter",
        "l3_coordination_role": "Basic",
        "l3_visual_weight": 2,
        "l3_style_lineage": ["Scandinavian Minimalism"],
        "l3_silhouette": "I-line",
        "l3_proportion_effect": "Streamlining",
    }


class TestFieldLists:
    """Test text_composer field list constants."""

    def test_l1_shared_includes_material_detail(self):
        assert "l1_material_detail" in _L1_SHARED

    def test_l3_shared_labels_includes_tone_season(self):
        """tone_season is in L3 shared labels (reads from final dict)."""
        assert "l3_tone_season" in _L3_SHARED_LABELS


class TestLayerCombos:
    """Test LAYER_COMBOS constant."""

    def test_seven_combos(self):
        assert len(LAYER_COMBOS) == 7

    def test_all_expected_combos(self):
        expected = {"L1", "L2", "L3", "L1+L2", "L1+L3", "L2+L3", "L1+L2+L3"}
        assert set(LAYER_COMBOS) == expected


class TestConstructFactualText:
    """Test individual text construction."""

    def test_metadata_only(self):
        text = construct_factual_text(_make_article_meta(), None, None, None, "Apparel")
        assert "T-shirt" in text
        assert "Black" in text
        assert "[Product]" not in text
        assert "[Perceptual]" not in text
        assert "[Theory]" not in text

    def test_l1_only(self):
        knowledge = _make_full_knowledge()
        l1 = {k: v for k, v in knowledge.items() if k.startswith("l1_")}
        text = construct_factual_text(_make_article_meta(), l1, None, None, "Apparel")
        assert "[Product]" in text
        assert "Cotton" in text
        assert "[Perceptual]" not in text

    def test_l1_includes_material_detail(self):
        knowledge = _make_full_knowledge()
        l1 = {k: v for k, v in knowledge.items() if k.startswith("l1_")}
        text = construct_factual_text(_make_article_meta(), l1, None, None, "Apparel")
        assert "100% cotton, jersey knit" in text

    def test_l2_only(self):
        knowledge = _make_full_knowledge()
        l2 = {k: v for k, v in knowledge.items() if k.startswith("l2_")}
        text = construct_factual_text(_make_article_meta(), None, l2, None, "Apparel")
        assert "[Perceptual]" in text
        assert "Casual" in text
        assert "[Product]" not in text

    def test_l3_only(self):
        knowledge = _make_full_knowledge()
        l3 = {k: v for k, v in knowledge.items() if k.startswith("l3_")}
        text = construct_factual_text(_make_article_meta(), None, None, l3, "Apparel")
        assert "[Theory]" in text
        assert "Monochromatic" in text
        assert "[Product]" not in text

    def test_l3_includes_tone_season(self):
        """tone_season from final dict is included in L3 text."""
        knowledge = _make_full_knowledge()
        l3 = {k: v for k, v in knowledge.items() if k.startswith("l3_")}
        text = construct_factual_text(_make_article_meta(), None, None, l3, "Apparel")
        assert "Cool-Winter" in text

    def test_full_l1_l2_l3(self):
        knowledge = _make_full_knowledge()
        l1 = {k: v for k, v in knowledge.items() if k.startswith("l1_")}
        l2 = {k: v for k, v in knowledge.items() if k.startswith("l2_")}
        l3 = {k: v for k, v in knowledge.items() if k.startswith("l3_")}
        text = construct_factual_text(_make_article_meta(), l1, l2, l3, "Apparel")
        assert "[Product]" in text
        assert "[Perceptual]" in text
        assert "[Theory]" in text

    def test_na_values_excluded(self):
        l1 = {"l1_material": "Cotton", "l1_neckline": "N/A", "l1_fit": "Slim"}
        text = construct_factual_text(_make_article_meta(), l1, None, None, "Apparel")
        assert "N/A" not in text
        assert "Cotton" in text

    def test_list_values_joined(self):
        knowledge = _make_full_knowledge()
        l2 = {k: v for k, v in knowledge.items() if k.startswith("l2_")}
        text = construct_factual_text(_make_article_meta(), None, l2, None, "Apparel")
        assert "Casual, Minimalist" in text


class TestBuildAllAblationTexts:
    """Test all 7 ablation text variants."""

    def test_returns_7_variants(self):
        texts = build_all_ablation_texts(
            _make_article_meta(), _make_full_knowledge(), "Apparel"
        )
        assert len(texts) == 7
        assert set(texts.keys()) == set(LAYER_COMBOS)

    def test_variants_differ(self):
        texts = build_all_ablation_texts(
            _make_article_meta(), _make_full_knowledge(), "Apparel"
        )
        # All texts should be unique
        values = list(texts.values())
        assert len(set(values)) == 7

    def test_l1_variant_has_product_only(self):
        texts = build_all_ablation_texts(
            _make_article_meta(), _make_full_knowledge(), "Apparel"
        )
        l1_text = texts["L1"]
        assert "[Product]" in l1_text
        assert "[Perceptual]" not in l1_text
        assert "[Theory]" not in l1_text

    def test_full_variant_has_all_layers(self):
        texts = build_all_ablation_texts(
            _make_article_meta(), _make_full_knowledge(), "Apparel"
        )
        full_text = texts["L1+L2+L3"]
        assert "[Product]" in full_text
        assert "[Perceptual]" in full_text
        assert "[Theory]" in full_text

    def test_metadata_in_all_variants(self):
        texts = build_all_ablation_texts(
            _make_article_meta(), _make_full_knowledge(), "Apparel"
        )
        for combo, text in texts.items():
            assert "T-shirt" in text, f"Metadata missing in {combo}"
