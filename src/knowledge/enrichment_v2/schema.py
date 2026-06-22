"""Enrichment v2 — 6 decision-axes attribute schema (single source of truth).

This is the catalog-enrichment redesign that follows the DE1 screen, which proved
the v1 L2/L3 attributes were mostly REDUNDANT recodings of metadata (the v1 prompt
SHOWED the LLM the item's metadata, so it just re-encoded it) or CONCENTRATED
(``l2_occasion`` was 81% "Everyday"). DE1 thresholds (pre-registered):
DISCRIMINATION ``top1_share <= 0.65`` & ``entropy >= 0.55``; NON-REDUNDANCY metadata/L1
lift ``< 0.55``; BEHAVIORAL ``signal >= 0.02``. The v2 axes are built to PASS those.

Two grounding families:

* ``E2_LLM_AXES`` — extracted multimodally (image + bare product name + a
  metadata-STRIPPED detail_desc), NEVER shown the categorical metadata fields the
  axis could recode. Each axis is designed orthogonal to a specific v1 field:
  occasion (no metadata column at all), fit-intent (vs ``l1_fit``), care-burden
  (vs ``l1_material``). Plus two PERCEIVED axes (price_look, trend_look) whose value
  is the GAP vs the behavioral truth.
* ``E2_BEHAVIORAL_AXES`` — computed from transactions only (no LLM, no images), in
  ``src/features/behavioral_axes.py``: actual price-tier (within-group quintile),
  trend-phase (sales momentum), outfit-role (co-purchase graph). Discriminable by
  construction; the data-grounded replacement for the failed LLM proxies.

Only ``E2_LLM_AXES`` enter the OpenAI structured-output schema. The behavioral axes
and the GAP columns (``e2_value_gap`` = price_look − price_tier; ``e2_trend_gap``)
are joined/derived downstream.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Output column names (stable; referenced by extractor, behavioral_axes, DE1 v2)
# ---------------------------------------------------------------------------

# Enum value sets ------------------------------------------------------------

# fine-occasion — "Everyday" deliberately DELETED (v1's 81% attractor). Forced to
# the single most specific situation; discrimination is a SCHEMA-design problem.
OCCASION_VALUES = [
    "Workday-office",
    "Smart-casual",
    "Weekend-errands",
    "Going-out-evening",
    "Date-romantic",
    "Active-gym",
    "Outdoor-activity",
    "Lounge-home",
    "Beach-resort",
    "Formal-event",
    "Party-festive",
    "Travel-commute",
    "School-campus",
]

# body-fit / size-intent — the INTENDED garment↔body relationship (styling reason),
# orthogonal to ``l1_fit`` which names the cut. Two "relaxed"-cut garments differ here.
FIT_INTENT_VALUES = [
    "Body-skimming",
    "True-to-form",
    "Easy-comfort",
    "Volume-statement",
    "Structured-tailored",
    "Petite-proportioned",
    "Elongating-draped",
]

# care / practicality — maintenance concerns from CONSTRUCTION/EMBELLISHMENT, which
# vary WITHIN a fabric (so not a deterministic recode of ``l1_material``).
CARE_FLAG_VALUES = [
    "Wrinkle-prone",
    "Shows-stains-easily",
    "Delicate-trims",
    "Shape-retention-needed",
    "Pilling-risk",
    "Color-bleed-risk",
    "Structured-pressing",
    "Low-maintenance",
]

# perceived trend (ordinal freshness) — its value is the GAP vs behavioral momentum.
TREND_LOOK_VALUES = ["Dated", "Classic", "Current", "Emerging"]

ORDINAL_1_5 = [1, 2, 3, 4, 5]


# LLM-extracted axes (enter the structured-output schema) --------------------
# kind: "enum" (single str) | "enum_list" (list[str]) | "int_enum" (single int)
E2_LLM_AXES: dict[str, dict[str, Any]] = {
    "e2_occasion_primary": {
        "kind": "enum",
        "is_list": False,
        "enum": OCCASION_VALUES,
        "description": (
            "The single MOST specific situation a typical buyer wears this in. "
            "Pick the most distinctive context — do NOT default to a generic "
            "catch-all, and do NOT infer it from the garment type."
        ),
    },
    "e2_occasion_secondary": {
        "kind": "enum_list",
        "is_list": True,
        "enum": OCCASION_VALUES,
        "min_items": 0,
        "max_items": 2,
        "description": "Up to 2 OTHER contexts it also suits; must differ from the primary.",
    },
    "e2_occasion_formality": {
        "kind": "int_enum",
        "is_list": False,
        "enum": ORDINAL_1_5,
        "description": (
            "1=athletic/loungewear, 2=casual, 3=smart-casual, 4=business/dressy, "
            "5=formal/black-tie. Use the FULL range."
        ),
    },
    "e2_fit_intent": {
        "kind": "enum",
        "is_list": False,
        "enum": FIT_INTENT_VALUES,
        "description": (
            "The INTENDED relationship between garment and body (the styling reason), "
            "NOT the cut name. Judge from how it sits in the image."
        ),
    },
    "e2_body_ease": {
        "kind": "int_enum",
        "is_list": False,
        "enum": ORDINAL_1_5,
        "description": (
            "Room between garment and body. 1=second-skin/compression, 3=standard ease, "
            "5=maximum drape/oversize-on-purpose. Judge from the image, not the size label."
        ),
    },
    "e2_care_burden": {
        "kind": "int_enum",
        "is_list": False,
        "enum": ORDINAL_1_5,
        "description": (
            "Real-world upkeep effort, INDEPENDENT of base fabric. 1=wash-and-forget, "
            "3=normal, 5=high-maintenance. Driven by CONSTRUCTION and EMBELLISHMENT "
            "(pleats, sequins, delicate trims, light colors, structure), NOT fabric type."
        ),
    },
    "e2_care_flags": {
        "kind": "enum_list",
        "is_list": True,
        "enum": CARE_FLAG_VALUES,
        "min_items": 1,
        "max_items": 3,
        "description": (
            "Practical maintenance concerns visible from construction/finish/"
            "embellishment. 'Low-maintenance' is mutually exclusive with the others."
        ),
    },
    "e2_price_look": {
        "kind": "int_enum",
        "is_list": False,
        "enum": ORDINAL_1_5,
        "description": (
            "Does it LOOK budget→luxury? 1=looks cheap/basic, 5=looks high-end. "
            "Perception only — you do NOT know the real price."
        ),
    },
    "e2_trend_look": {
        "kind": "enum",
        "is_list": False,
        "enum": TREND_LOOK_VALUES,
        "description": (
            "Does the STYLE look Dated, Classic (timeless), Current, or Emerging "
            "(fashion-forward)? Perception only — you do NOT know the sales trajectory."
        ),
    },
}

# Behavior-derived axes (computed in src/features/behavioral_axes.py; NOT in schema)
E2_BEHAVIORAL_AXES: dict[str, dict[str, Any]] = {
    "e2_price_tier_actual": {
        "is_list": False,
        "values": ["T1", "T2", "T3", "T4", "T5"],
        "description": "Within-product_group quintile of actual mean price (T1=cheapest).",
    },
    "e2_trend_phase_actual": {
        "is_list": False,
        "values": ["Emerging", "Rising", "Peak", "Mature", "Declining", "Insufficient"],
        "description": "Sales-momentum life-cycle phase from monthly volume trajectory.",
    },
    "e2_outfit_role": {
        "is_list": False,
        "values": [
            "Anchor-hub",
            "Versatile-connector",
            "Complement-addon",
            "Niche-pair",
            "Standalone",
        ],
        "description": "Co-purchase graph role (degree/direction/diversity, residualized vs product_group).",
    },
}

# Derived GAP columns (computed post-join; DE1-safe by construction) ----------
GAP_AXES = ["e2_value_gap", "e2_trend_gap"]

# Rank maps for the gap (both projected onto a shared 1..5 freshness/tier scale).
PRICE_TIER_RANK = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5}
TREND_LOOK_RANK = {"Dated": 1, "Classic": 2, "Current": 4, "Emerging": 5}
TREND_PHASE_RANK = {
    "Declining": 1,
    "Mature": 2,
    "Peak": 3,
    "Rising": 4,
    "Emerging": 5,
    "Insufficient": None,  # excluded from gap
}

# Multi-label (list) columns — consumed by the DE1 v2 re-screen LIST_COLS.
E2_LIST_COLS = [c for c, m in E2_LLM_AXES.items() if m["is_list"]]

# All e2 columns the final enrichment_v2.parquet carries (besides keys).
E2_ALL_COLUMNS = list(E2_LLM_AXES) + list(E2_BEHAVIORAL_AXES) + GAP_AXES


# ---------------------------------------------------------------------------
# OpenAI strict structured-output schema (LLM axes only)
# ---------------------------------------------------------------------------
def _property_schema(meta: dict[str, Any]) -> dict[str, Any]:
    """Render one axis' JSON-schema property from its E2_LLM_AXES metadata."""
    kind = meta["kind"]
    desc = meta["description"]
    if kind == "enum":
        return {"type": "string", "enum": list(meta["enum"]), "description": desc}
    if kind == "int_enum":
        return {"type": "integer", "enum": list(meta["enum"]), "description": desc}
    if kind == "enum_list":
        return {
            "type": "array",
            "items": {"type": "string", "enum": list(meta["enum"])},
            "minItems": meta["min_items"],
            "maxItems": meta["max_items"],
            "description": desc,
        }
    raise ValueError(f"Unknown axis kind: {kind}")


def build_e2_json_schema() -> dict[str, Any]:
    """Strict JSON Schema (all-required, additionalProperties=false) for the LLM axes."""
    properties = {name: _property_schema(meta) for name, meta in E2_LLM_AXES.items()}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(E2_LLM_AXES),
    }


# OpenAI structured-output `text.format` payload (Responses API).
E2_RESPONSE_FORMAT: dict[str, Any] = {
    "format": {
        "type": "json_schema",
        "name": "enrichment_v2_axes",
        "schema": build_e2_json_schema(),
        "strict": True,
    }
}
