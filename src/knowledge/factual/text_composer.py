"""Compose factual text from metadata + L1/L2/L3 knowledge for KAR encoding.

Generates 7 ablation variants by selectively including/excluding layers.
The resulting text is fed to BGE-base-en-v1.5 for dense vector encoding.
"""

import json
from typing import Optional

from src.knowledge.factual.prompts import (
    CATEGORY_SPECIFIC_L1_FIELDS,
    CATEGORY_SPECIFIC_L3_FIELDS,
)

LAYER_COMBOS: list[str] = [
    "L1",
    "L2",
    "L3",
    "L1+L2",
    "L1+L3",
    "L2+L3",
    "L1+L2+L3",
]

# L1 shared fields
_L1_SHARED = ["l1_material", "l1_closure", "l1_design_details", "l1_material_detail"]

# L2 fields with human-readable labels
_L2_FIELD_LABELS: dict[str, str] = {
    "l2_style_mood": "Style",
    "l2_occasion": "Occasion",
    "l2_perceived_quality": "Quality",
    "l2_trendiness": "Trendiness",
    "l2_season_fit": "Season",
    "l2_target_impression": "Impression",
    "l2_versatility": "Versatility",
}

# L3 shared fields with labels
_L3_SHARED_LABELS: dict[str, str] = {
    "l3_color_harmony": "Color Harmony",
    "l3_tone_season": "Tone Season",
    "l3_coordination_role": "Coordination Role",
    "l3_visual_weight": "Visual Weight",
    "l3_style_lineage": "Style Lineage",
}

# Category-specific L1 labels
_L1_SPECIFIC_LABELS: dict[str, dict[str, str]] = {
    "Apparel": {
        "l1_neckline": "Neckline",
        "l1_sleeve_type": "Sleeve",
        "l1_fit": "Fit",
        "l1_length": "Length",
    },
    "Footwear": {
        "l1_toe_shape": "Toe Shape",
        "l1_shaft_height": "Shaft Height",
        "l1_heel_type": "Heel",
        "l1_sole_type": "Sole",
    },
    "Accessories": {
        "l1_form_factor": "Form Factor",
        "l1_size_scale": "Size",
        "l1_wearing_method": "Wearing Method",
        "l1_primary_function": "Function",
    },
}

# Category-specific L3 labels
_L3_SPECIFIC_LABELS: dict[str, dict[str, str]] = {
    "Apparel": {
        "l3_silhouette": "Silhouette",
        "l3_proportion_effect": "Proportion Effect",
    },
    "Footwear": {
        "l3_foot_silhouette": "Foot Silhouette",
        "l3_height_effect": "Height Effect",
    },
    "Accessories": {
        "l3_visual_form": "Visual Form",
        "l3_styling_effect": "Styling Effect",
    },
}


def _format_value(value: object) -> str:
    """Format a single attribute value for text composition."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return ", ".join(str(v) for v in parsed)
        except json.JSONDecodeError:
            pass
    return str(value)


def _compose_metadata_text(article_meta: dict) -> str:
    """Build base text from article metadata."""
    parts: list[str] = []
    for field, label in [
        ("product_type_name", "Type"),
        ("product_group_name", "Group"),
        ("colour_group_name", "Color"),
        ("graphical_appearance_name", "Pattern"),
        ("section_name", "Section"),
    ]:
        val = article_meta.get(field)
        if val and str(val).strip() and str(val) != "nan":
            parts.append(f"{label}: {val}")
    return "; ".join(parts) if parts else ""


def _compose_l1_text(knowledge: dict, super_category: str) -> str:
    """Compose L1 attribute text."""
    parts: list[str] = []

    # Shared L1
    for field in _L1_SHARED:
        val = _format_value(knowledge.get(field))
        if val:
            label = field.replace("l1_", "").replace("_", " ").title()
            parts.append(f"{label}: {val}")

    # Category-specific L1
    specific_labels = _L1_SPECIFIC_LABELS.get(super_category, {})
    specific_fields = CATEGORY_SPECIFIC_L1_FIELDS.get(super_category, [])
    for field in specific_fields:
        val = _format_value(knowledge.get(field))
        if val and val != "N/A":
            label = specific_labels.get(field, field)
            parts.append(f"{label}: {val}")

    return "; ".join(parts) if parts else ""


def _compose_l2_text(knowledge: dict) -> str:
    """Compose L2 attribute text."""
    parts: list[str] = []
    for field, label in _L2_FIELD_LABELS.items():
        val = _format_value(knowledge.get(field))
        if val:
            parts.append(f"{label}: {val}")
    return "; ".join(parts) if parts else ""


def _compose_l3_text(knowledge: dict, super_category: str) -> str:
    """Compose L3 attribute text."""
    parts: list[str] = []

    # Shared L3
    for field, label in _L3_SHARED_LABELS.items():
        val = _format_value(knowledge.get(field))
        if val:
            parts.append(f"{label}: {val}")

    # Category-specific L3
    specific_labels = _L3_SPECIFIC_LABELS.get(super_category, {})
    specific_fields = CATEGORY_SPECIFIC_L3_FIELDS.get(super_category, [])
    for field in specific_fields:
        val = _format_value(knowledge.get(field))
        if val:
            label = specific_labels.get(field, field)
            parts.append(f"{label}: {val}")

    return "; ".join(parts) if parts else ""


def construct_factual_text(
    article_meta: dict,
    l1_knowledge: Optional[dict],
    l2_knowledge: Optional[dict],
    l3_knowledge: Optional[dict],
    super_category: str,
) -> str:
    """Compose factual text from metadata + selected layer knowledge.

    Args:
        article_meta: Article metadata (product_type_name, colour_group_name, etc.).
        l1_knowledge: L1 attribute dict (or None to exclude).
        l2_knowledge: L2 attribute dict (or None to exclude).
        l3_knowledge: L3 attribute dict (or None to exclude).
        super_category: 'Apparel', 'Footwear', or 'Accessories'.

    Returns:
        Composed text for BGE encoding.
    """
    sections: list[str] = []

    # Always include metadata as base
    meta_text = _compose_metadata_text(article_meta)
    if meta_text:
        sections.append(meta_text)

    # L1
    if l1_knowledge:
        l1_text = _compose_l1_text(l1_knowledge, super_category)
        if l1_text:
            sections.append(f"[Product] {l1_text}")

    # L2
    if l2_knowledge:
        l2_text = _compose_l2_text(l2_knowledge)
        if l2_text:
            sections.append(f"[Perceptual] {l2_text}")

    # L3
    if l3_knowledge:
        l3_text = _compose_l3_text(l3_knowledge, super_category)
        if l3_text:
            sections.append(f"[Theory] {l3_text}")

    return ". ".join(sections)


def build_all_ablation_texts(
    article_meta: dict,
    knowledge: dict,
    super_category: str,
) -> dict[str, str]:
    """Generate all 7 layer ablation text variants.

    Args:
        article_meta: Article metadata dict.
        knowledge: Full L1+L2+L3 knowledge dict.
        super_category: 'Apparel', 'Footwear', or 'Accessories'.

    Returns:
        Dict mapping combo name (e.g., "L1+L2+L3") to composed text.
    """
    # Split knowledge into layer-specific dicts
    l1 = {k: v for k, v in knowledge.items() if k.startswith("l1_")}
    l2 = {k: v for k, v in knowledge.items() if k.startswith("l2_")}
    l3 = {k: v for k, v in knowledge.items() if k.startswith("l3_")}

    combo_map: dict[str, tuple[Optional[dict], Optional[dict], Optional[dict]]] = {
        "L1": (l1, None, None),
        "L2": (None, l2, None),
        "L3": (None, None, l3),
        "L1+L2": (l1, l2, None),
        "L1+L3": (l1, None, l3),
        "L2+L3": (None, l2, l3),
        "L1+L2+L3": (l1, l2, l3),
    }

    return {
        combo: construct_factual_text(article_meta, l1_k, l2_k, l3_k, super_category)
        for combo, (l1_k, l2_k, l3_k) in combo_map.items()
    }
