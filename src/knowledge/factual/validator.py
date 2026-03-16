"""Schema validation for L1+L2+L3 extracted knowledge.

Validates both structural correctness (types, required fields, enum ranges)
and semantic plausibility (e.g., sleeveless items shouldn't have sleeve_type=Long).
"""

import json
from typing import NamedTuple

import numpy as np

from src.knowledge.factual.prompts import (
    CATEGORY_SPECIFIC_L1_FIELDS,
    CATEGORY_SPECIFIC_L3_FIELDS,
    SCHEMA_MAP,
)


class ValidationResult(NamedTuple):
    """Result of knowledge validation."""

    is_valid: bool
    errors: list[str]  # Blocking issues
    warnings: list[str]  # Non-blocking quality concerns


class DomainViolation(NamedTuple):
    """Cross-attribute domain consistency violation."""

    rule_name: str
    severity: str  # "Error" | "Warning"
    description: str


# L1 shared fields (present in all categories)
_L1_SHARED = ["l1_material", "l1_closure", "l1_design_details", "l1_material_detail"]

# L2 fields (universal)
_L2_FIELDS = [
    "l2_style_mood",
    "l2_occasion",
    "l2_perceived_quality",
    "l2_trendiness",
    "l2_season_fit",
    "l2_target_impression",
    "l2_versatility",
]

# L3 shared fields (LLM-extracted — does NOT include tone_season)
_L3_SHARED = [
    "l3_color_harmony",
    "l3_coordination_role",
    "l3_visual_weight",
    "l3_style_lineage",
]

# L3 post-processed fields (added after LLM extraction via rule-based mapping)
_L3_POSTPROCESSED = ["l3_tone_season"]


def _get_all_fields(super_category: str) -> list[str]:
    """Get all expected field names for a super-category."""
    l1_specific = CATEGORY_SPECIFIC_L1_FIELDS[super_category]
    l3_specific = CATEGORY_SPECIFIC_L3_FIELDS[super_category]
    return _L1_SHARED + l1_specific + _L2_FIELDS + _L3_SHARED + l3_specific


def _validate_field_against_schema(
    field_name: str,
    value: object,
    schema_props: dict,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a single field value against its JSON schema definition."""
    if field_name not in schema_props:
        return

    spec = schema_props[field_name]
    field_type = spec.get("type")

    if field_type == "string":
        if not isinstance(value, str):
            errors.append(f"{field_name}: expected string, got {type(value).__name__}")
            return
        if "enum" in spec and value not in spec["enum"]:
            warnings.append(
                f"{field_name}: value '{value}' not in allowed enum {spec['enum'][:5]}..."
            )

    elif field_type == "integer":
        if not isinstance(value, int):
            errors.append(f"{field_name}: expected int, got {type(value).__name__}")
            return
        low = spec.get("minimum", float("-inf"))
        high = spec.get("maximum", float("inf"))
        if not (low <= value <= high):
            errors.append(f"{field_name}: value {value} out of range [{low}, {high}]")

    elif field_type == "array":
        if not isinstance(value, (list, np.ndarray)):
            # Accept JSON string representation of array
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        value = parsed
                    else:
                        errors.append(f"{field_name}: expected array, got string")
                        return
                except json.JSONDecodeError:
                    errors.append(f"{field_name}: expected array, got non-JSON string")
                    return
            else:
                errors.append(f"{field_name}: expected array, got {type(value).__name__}")
                return
        # Convert ndarray to list for uniform downstream handling
        if isinstance(value, np.ndarray):
            value = value.tolist()
        max_items = spec.get("maxItems", float("inf"))
        if len(value) > max_items:
            warnings.append(f"{field_name}: {len(value)} items exceeds maxItems={max_items}")
        items_spec = spec.get("items", {})
        if "enum" in items_spec:
            for item in value:
                if item not in items_spec["enum"]:
                    warnings.append(
                        f"{field_name}: item '{item}' not in allowed enum"
                    )


def validate_knowledge(knowledge: dict, super_category: str) -> ValidationResult:
    """Validate LLM-extracted L1+L2+L3 knowledge against schema (21 fields).

    This validates the raw LLM output BEFORE post-processing (no tone_season).

    Args:
        knowledge: Extracted knowledge dict with l1_*, l2_*, l3_* fields.
        super_category: One of 'Apparel', 'Footwear', 'Accessories'.

    Returns:
        ValidationResult with is_valid, errors, and warnings.
    """
    if super_category not in SCHEMA_MAP:
        return ValidationResult(
            is_valid=False,
            errors=[f"Unknown super_category: {super_category}"],
            warnings=[],
        )

    errors: list[str] = []
    warnings: list[str] = []

    schema = SCHEMA_MAP[super_category]
    schema_props = schema["properties"]
    expected_fields = _get_all_fields(super_category)

    # Check required fields
    for field in expected_fields:
        if field not in knowledge:
            errors.append(f"Missing required field: {field}")
        elif knowledge[field] is None:
            warnings.append(f"Null value for field: {field}")
        else:
            _validate_field_against_schema(
                field, knowledge[field], schema_props, errors, warnings
            )

    # Check for unexpected fields (not an error, just informational)
    known_fields = set(expected_fields)
    extra_fields = [k for k in knowledge if k not in known_fields]
    if extra_fields:
        warnings.append(f"Unexpected fields: {extra_fields}")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def _get_all_final_fields(super_category: str) -> list[str]:
    """Get all expected field names for final output (LLM + post-processed)."""
    return _get_all_fields(super_category) + _L3_POSTPROCESSED


def validate_final_knowledge(knowledge: dict, super_category: str) -> ValidationResult:
    """Validate final knowledge after post-processing (22 fields, includes tone_season).

    Args:
        knowledge: Final knowledge dict including post-processed fields.
        super_category: One of 'Apparel', 'Footwear', 'Accessories'.

    Returns:
        ValidationResult with is_valid, errors, and warnings.
    """
    if super_category not in SCHEMA_MAP:
        return ValidationResult(
            is_valid=False,
            errors=[f"Unknown super_category: {super_category}"],
            warnings=[],
        )

    errors: list[str] = []
    warnings: list[str] = []

    schema = SCHEMA_MAP[super_category]
    schema_props = schema["properties"]
    expected_fields = _get_all_final_fields(super_category)

    # Import tone_season enum for validation
    from src.knowledge.factual.prompts import TONE_SEASON_VALUES

    # Extended schema props including post-processed fields
    extended_props = dict(schema_props)
    extended_props["l3_tone_season"] = {
        "type": "string",
        "enum": TONE_SEASON_VALUES,
    }

    for field in expected_fields:
        if field not in knowledge:
            errors.append(f"Missing required field: {field}")
        elif knowledge[field] is None:
            warnings.append(f"Null value for field: {field}")
        else:
            _validate_field_against_schema(
                field, knowledge[field], extended_props, errors, warnings
            )

    known_fields = set(expected_fields)
    extra_fields = [k for k in knowledge if k not in known_fields]
    if extra_fields:
        warnings.append(f"Unexpected fields: {extra_fields}")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Domain Consistency Rules
# ---------------------------------------------------------------------------

# Contradictory lineage-mood pairs
_LINEAGE_MOOD_CONTRADICTIONS: set[tuple[str, str]] = {
    ("Punk", "Classic"),
    ("Grunge", "Luxury"),
    ("Streetwear", "Classic"),
    ("Punk", "Romantic"),
    ("Grunge", "Glam"),
}


def validate_domain_consistency(
    knowledge: dict, super_category: str
) -> list[DomainViolation]:
    """Validate cross-attribute domain consistency rules.

    Args:
        knowledge: Knowledge dict with l1_*, l2_*, l3_* fields.
        super_category: One of 'Apparel', 'Footwear', 'Accessories'.

    Returns:
        List of DomainViolation for any rules violated.
    """
    violations: list[DomainViolation] = []

    coord = knowledge.get("l3_coordination_role", "")
    weight = knowledge.get("l3_visual_weight")
    harmony = knowledge.get("l3_color_harmony", "")

    # --- Error-severity rules ---

    # 1. coordination_role × visual_weight
    if coord == "Basic" and isinstance(weight, int) and weight > 3:
        violations.append(DomainViolation(
            "coordination_x_visual_weight",
            "Error",
            f"Basic coordination_role but visual_weight={weight} (should be ≤3)",
        ))
    if coord == "Statement" and isinstance(weight, int) and weight < 3:
        violations.append(DomainViolation(
            "coordination_x_visual_weight",
            "Error",
            f"Statement coordination_role but visual_weight={weight} (should be ≥3)",
        ))

    if super_category == "Apparel":
        silhouette = knowledge.get("l3_silhouette", "")
        fit = knowledge.get("l1_fit", "")
        sleeve = knowledge.get("l1_sleeve_type", "")
        neckline = knowledge.get("l1_neckline", "")
        season = knowledge.get("l2_season_fit", "")

        # 2. silhouette × visual_weight
        if silhouette == "I-line" and isinstance(weight, int) and weight > 3:
            violations.append(DomainViolation(
                "silhouette_x_visual_weight",
                "Error",
                f"I-line silhouette but visual_weight={weight} (should be ≤3)",
            ))
        if silhouette == "O-line" and isinstance(weight, int) and weight < 3:
            violations.append(DomainViolation(
                "silhouette_x_visual_weight",
                "Error",
                f"O-line silhouette but visual_weight={weight} (should be ≥3)",
            ))

        # 3. fit × visual_weight
        if fit in ("Slim", "Skinny") and isinstance(weight, int) and weight > 3:
            violations.append(DomainViolation(
                "fit_x_visual_weight",
                "Error",
                f"{fit} fit but visual_weight={weight} (should be ≤3)",
            ))
        if fit in ("Oversized", "Loose", "Boxy") and isinstance(weight, int) and weight < 3:
            violations.append(DomainViolation(
                "fit_x_visual_weight",
                "Error",
                f"{fit} fit but visual_weight={weight} (should be ≥3)",
            ))

        # 4. sleeve × season
        if sleeve == "Sleeveless" and season == "Winter":
            violations.append(DomainViolation(
                "sleeve_x_season",
                "Error",
                "Sleeveless but season_fit=Winter",
            ))

        # 5. neckline × sleeve
        if neckline == "Strapless" and sleeve not in ("Sleeveless", "N/A"):
            violations.append(DomainViolation(
                "neckline_x_sleeve",
                "Error",
                f"Strapless neckline but sleeve_type={sleeve} (should be Sleeveless or N/A)",
            ))

    elif super_category == "Footwear":
        sole = knowledge.get("l1_sole_type", "")
        season = knowledge.get("l2_season_fit", "")

        # 6. sole × season
        if sole == "Foam" and season == "Winter":
            violations.append(DomainViolation(
                "sole_x_season",
                "Error",
                "Foam sole but season_fit=Winter",
            ))

    elif super_category == "Accessories":
        function = knowledge.get("l1_primary_function", "")
        form = knowledge.get("l1_form_factor", "")
        wearing = knowledge.get("l1_wearing_method", "")
        size = knowledge.get("l1_size_scale", "")

        # 7. function × form_factor
        if function == "Storage" and form in ("Mini", "Compact"):
            violations.append(DomainViolation(
                "function_x_form_factor",
                "Error",
                f"Storage function but form_factor={form} (too small for storage)",
            ))

        # 8. wearing_method × size_scale
        if wearing in ("Wrist", "Finger") and size not in ("Petite", "Small"):
            violations.append(DomainViolation(
                "wearing_x_size_scale",
                "Error",
                f"wearing_method={wearing} but size_scale={size} (should be Petite or Small)",
            ))

    # --- Warning-severity rules (all categories) ---

    moods = knowledge.get("l2_style_mood", [])
    if isinstance(moods, str):
        moods = [moods]
    occasions = knowledge.get("l2_occasion", [])
    if isinstance(occasions, str):
        occasions = [occasions]

    # 9. mood × occasion
    if "Bohemian" in moods and any(o in ("Work", "Formal") for o in occasions):
        violations.append(DomainViolation(
            "mood_x_occasion",
            "Warning",
            "Bohemian mood with Work/Formal occasion is unusual",
        ))

    # 10. heel × occasion (Footwear)
    if super_category == "Footwear":
        heel = knowledge.get("l1_heel_type", "")
        if heel == "Stiletto" and "Outdoor" in occasions:
            violations.append(DomainViolation(
                "heel_x_occasion",
                "Warning",
                "Stiletto heel with Outdoor occasion is unusual",
            ))

    # 11. coordination × harmony
    if coord == "Basic" and harmony not in ("Monochromatic", "Neutral", "Earth-tone", ""):
        violations.append(DomainViolation(
            "coordination_x_harmony",
            "Warning",
            f"Basic coordination_role but color_harmony={harmony} "
            "(expected Monochromatic, Neutral, or Earth-tone)",
        ))
    if coord == "Accent" and harmony == "Monochromatic":
        violations.append(DomainViolation(
            "coordination_x_harmony",
            "Warning",
            "Accent coordination_role but color_harmony=Monochromatic",
        ))

    # 12. lineage × mood contradictions
    lineages = knowledge.get("l3_style_lineage", [])
    if isinstance(lineages, str):
        lineages = [lineages]
    for lineage in lineages:
        for mood in moods:
            if (lineage, mood) in _LINEAGE_MOOD_CONTRADICTIONS:
                violations.append(DomainViolation(
                    "lineage_x_mood",
                    "Warning",
                    f"Contradictory lineage '{lineage}' with mood '{mood}'",
                ))

    return violations
