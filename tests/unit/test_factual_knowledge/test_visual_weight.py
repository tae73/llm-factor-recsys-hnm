"""Unit tests for visual_weight rule-based correction (correct_visual_weight)."""

from __future__ import annotations

import pytest

from src.knowledge.factual.extractor import correct_visual_weight


def _make_apparel(
    silhouette: str = "H-line",
    fit: str = "Regular",
    coordination: str = "Layering",
    weight: int = 3,
) -> dict:
    """Build a minimal Apparel knowledge dict for testing."""
    return {
        "l3_silhouette": silhouette,
        "l1_fit": fit,
        "l3_coordination_role": coordination,
        "l3_visual_weight": weight,
        "l1_material": "Cotton",
    }


class TestCorrectVisualWeight:
    """Tests for correct_visual_weight() clamping logic."""

    def test_iline_clamp_down(self) -> None:
        """I-line + weight=4 → clamp to 3."""
        k = _make_apparel(silhouette="I-line", weight=4)
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == 3

    def test_oline_clamp_up(self) -> None:
        """O-line + weight=2 → clamp to 3."""
        k = _make_apparel(silhouette="O-line", weight=2)
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == 3

    def test_slim_fit_clamp_down(self) -> None:
        """Slim fit + weight=4 → clamp to 3."""
        k = _make_apparel(fit="Slim", weight=4)
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == 3

    def test_oversized_fit_clamp_up(self) -> None:
        """Oversized fit + weight=1 → clamp to 3."""
        k = _make_apparel(fit="Oversized", weight=1)
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == 3

    def test_basic_coord_clamp_down(self) -> None:
        """Basic coordination + weight=4 → clamp to 3."""
        k = _make_apparel(coordination="Basic", weight=4)
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == 3

    def test_statement_coord_clamp_up(self) -> None:
        """Statement coordination + weight=2 → clamp to 3."""
        k = _make_apparel(coordination="Statement", weight=2)
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == 3

    def test_intersection_iline_slim_basic(self) -> None:
        """I-line(1-3) + Slim(1-3) + Basic(1-3) → intersection (1-3), weight=5 → 3."""
        k = _make_apparel(silhouette="I-line", fit="Slim", coordination="Basic", weight=5)
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == 3

    def test_intersection_oline_oversized_statement(self) -> None:
        """O-line(3-5) + Oversized(3-5) + Statement(3-5) → intersection (3-5), weight=1 → 3."""
        k = _make_apparel(
            silhouette="O-line", fit="Oversized", coordination="Statement", weight=1
        )
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == 3

    def test_empty_intersection_silhouette_priority(self) -> None:
        """I-line(1-3) + Oversized(3-5) → empty intersection → silhouette (1-3) wins."""
        k = _make_apparel(silhouette="I-line", fit="Oversized", weight=5)
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == 3

    def test_footwear_passthrough(self) -> None:
        """Footwear → no correction applied."""
        k = {"l3_visual_weight": 5, "l3_coordination_role": "Basic"}
        result = correct_visual_weight(k, "Footwear")
        assert result["l3_visual_weight"] == 5

    def test_accessories_passthrough(self) -> None:
        """Accessories → no correction applied."""
        k = {"l3_visual_weight": 5, "l3_coordination_role": "Basic"}
        result = correct_visual_weight(k, "Accessories")
        assert result["l3_visual_weight"] == 5

    def test_already_in_range_no_change(self) -> None:
        """Weight already within range → dict returned unchanged (same object)."""
        k = _make_apparel(silhouette="I-line", fit="Slim", coordination="Basic", weight=2)
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == 2
        assert result is k  # No copy made

    def test_non_int_weight_passthrough(self) -> None:
        """Non-integer weight → no correction."""
        k = _make_apparel(weight=3)
        k["l3_visual_weight"] = "high"  # type: ignore[assignment]
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == "high"

    def test_missing_weight_passthrough(self) -> None:
        """Missing weight key → no correction."""
        k = {"l3_silhouette": "I-line", "l1_fit": "Slim"}
        result = correct_visual_weight(k, "Apparel")
        assert "l3_visual_weight" not in result

    def test_correction_returns_new_dict(self) -> None:
        """When correction happens, original dict is not mutated."""
        k = _make_apparel(silhouette="I-line", weight=5)
        result = correct_visual_weight(k, "Apparel")
        assert result["l3_visual_weight"] == 3
        assert k["l3_visual_weight"] == 5  # Original unchanged
        assert result is not k
