"""LLM Factorization Prompting for user reasoning knowledge extraction.

System prompt + JSON schema + user message construction + reasoning text composition.
Each active user (5+ purchases) gets a single LLM call that produces a 9-field
structured JSON, which is then composed into natural language reasoning_text
for the KAR Reasoning Expert.
"""

from __future__ import annotations

import json
import logging

from src.config import ReasoningConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category-Specific L3 Slot Display Names
# ---------------------------------------------------------------------------
# Maps canonical Parquet slot names (l3_slot6, l3_slot7) to human-readable
# labels per super-category.  Used in build_reasoning_user_message() so the LLM
# sees "Silhouette: Y-line 60%" instead of "Slot6: Y-line 60%".

_L3_SLOT_DISPLAY: dict[str, dict[str, str]] = {
    "Apparel": {"l3_slot6": "Silhouette", "l3_slot7": "Proportion Effect"},
    "Footwear": {"l3_slot6": "Foot Silhouette", "l3_slot7": "Height Effect"},
    "Accessories": {"l3_slot6": "Form", "l3_slot7": "Accent Function"},
}


# ---------------------------------------------------------------------------
# JSON Schema — 9-field Reasoning Profile
# ---------------------------------------------------------------------------

REASONING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "style_mood_preference": {
            "type": "string",
            "description": (
                "Dominant style/mood tendencies (e.g., 'Casual minimalist with occasional formal touches'). "
                "Synthesize from recurring L2 style_mood and target_impression patterns."
            ),
        },
        "occasion_preference": {
            "type": "string",
            "description": (
                "Primary occasions this customer shops for (e.g., 'Everyday basics with weekend casual'). "
                "Infer from L2 occasion patterns and purchase frequency."
            ),
        },
        "quality_price_tendency": {
            "type": "string",
            "description": (
                "Price positioning and value orientation. "
                "IMPORTANT: Base this primarily on the Customer Overview's price quintile "
                "(1=lowest, 5=highest actual spending), NOT on perceived_quality scores from individual items. "
                "Then note how perceived quality compares to actual spending "
                "(e.g., 'Premium spender (quintile 5) who buys mid-quality basics — values variety over per-item quality')."
            ),
        },
        "trend_sensitivity": {
            "type": "string",
            "description": (
                "How trend-aware the customer is (e.g., 'Classic core with occasional current pieces'). "
                "Infer from L2 trendiness distribution."
            ),
        },
        "seasonal_pattern": {
            "type": "string",
            "description": (
                "Seasonal shopping behavior (e.g., 'Year-round basics buyer, heavier winter purchases'). "
                "Combine L2 season_fit distribution with purchase timing."
            ),
        },
        "form_preference": {
            "type": "string",
            "description": (
                "Cross-category form preferences (e.g., 'Prefers streamlined I-line silhouettes, slim fits'). "
                "Synthesize from L3 silhouette/foot_silhouette/visual_form patterns across categories."
            ),
        },
        "color_tendency": {
            "type": "string",
            "description": (
                "Color preference pattern (e.g., 'Monochromatic neutral palette, Cool-Winter tones'). "
                "Combine L3 color_harmony and tone_season distributions."
            ),
        },
        "coordination_tendency": {
            "type": "string",
            "description": (
                "Outfit coordination style (e.g., 'Builds from basics, adds occasional statement pieces'). "
                "Infer from L3 coordination_role and visual_weight distributions."
            ),
        },
        "identity_summary": {
            "type": "string",
            "description": (
                "One-sentence identity summary capturing this customer's fashion identity "
                "(e.g., 'A practical minimalist who values quality basics and neutral palettes')."
            ),
        },
    },
    "required": [
        "style_mood_preference",
        "occasion_preference",
        "quality_price_tendency",
        "trend_sensitivity",
        "seasonal_pattern",
        "form_preference",
        "color_tendency",
        "coordination_tendency",
        "identity_summary",
    ],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """You are a fashion consumer analyst. Given a customer's purchase summary and attribute patterns, produce a structured 9-dimensional preference profile.

**Input format:**
1. Customer Overview — purchase count, top categories, price position, online ratio, diversity score
2. Recent Items (L2 Attributes) — last 20 items with style mood, occasion, quality, trendiness, season, impression, versatility
3. Attribute Patterns (L3 Theory-Based) — distributions of color harmony, tone season, coordination role, visual weight, style lineage, plus category-specific silhouette/form patterns

**Rules:**
- Synthesize patterns across ALL purchased items, not just one or two.
- For each field, identify the DOMINANT tendency and note significant secondary patterns.
- Be specific to THIS customer — avoid generic descriptions.
- Consider cross-attribute relationships (e.g., high versatility + basic coordination → wardrobe builder).
- For quality_price_tendency, distinguish ACTUAL spending level (price quintile from Customer Overview) from PERCEIVED quality (from individual items). Price quintile reflects real behavior; perceived quality is subjective.
- The identity_summary should be a single compelling sentence that captures the customer's unique fashion identity.
- Keep each field concise (1-2 sentences max).
- If data is sparse for a dimension, note the limitation rather than speculating."""


# ---------------------------------------------------------------------------
# User Message Construction
# ---------------------------------------------------------------------------


def build_reasoning_user_message(
    l1_summary: dict,
    recent_items_l2: list[dict],
    l3_distributions: dict,
) -> str:
    """Build the user message for LLM profiling.

    Args:
        l1_summary: L1 aggregated stats (n_purchases, top_categories_json, etc.)
        recent_items_l2: List of recent item dicts with L2 attributes.
        l3_distributions: {"shared": {...}, "by_category": {...}}

    Returns:
        Formatted user message string.
    """
    # --- Section 1: Customer Overview ---
    parts = ["--- Customer Overview ---"]
    n_purchases = l1_summary.get("n_purchases", 0)
    n_unique_types = l1_summary.get("n_unique_types", 0)
    diversity = l1_summary.get("category_diversity", 0.0)
    parts.append(
        f"Purchases: {n_purchases} items, {n_unique_types} unique types, "
        f"diversity {diversity:.2f}"
    )

    # Top categories
    top_cats = _parse_json_field(l1_summary.get("top_categories_json", "{}"))
    if top_cats:
        cat_str = ", ".join(f"{k} {v*100:.0f}%" for k, v in list(top_cats.items())[:5])
        parts.append(f"Top categories: {cat_str}")

    # Price and channel
    price_q = l1_summary.get("avg_price_quintile", 3.0)
    online_ratio = l1_summary.get("online_ratio", 0.0)
    parts.append(f"Price position: avg {price_q:.1f}/5, Online ratio: {online_ratio*100:.0f}%")

    # --- Section 2: Recent Items (L2 Attributes) ---
    parts.append("\n--- Recent Items (L2 Attributes) ---")
    for i, item in enumerate(recent_items_l2[:20], 1):
        item_parts = []
        cat = item.get("super_category", "")
        if cat:
            item_parts.append(cat)

        moods = item.get("l2_style_mood", [])
        if moods:
            item_parts.append(f"Style: {', '.join(moods[:3])}")

        occasions = item.get("l2_occasion", [])
        if occasions:
            item_parts.append(f"Occasion: {', '.join(occasions[:3])}")

        quality = item.get("l2_perceived_quality")
        if quality is not None:
            item_parts.append(f"Quality: {quality}/5")

        trend = item.get("l2_trendiness")
        if trend:
            item_parts.append(f"Trend: {trend}")

        season = item.get("l2_season_fit")
        if season:
            item_parts.append(f"Season: {season}")

        impression = item.get("l2_target_impression")
        if impression:
            item_parts.append(f"Impression: {impression}")

        versatility = item.get("l2_versatility")
        if versatility is not None:
            item_parts.append(f"Versatility: {versatility}/5")

        parts.append(f"{i}. {' | '.join(item_parts)}")

    # --- Section 3: Attribute Patterns (L3 Theory-Based) ---
    parts.append("\n--- Attribute Patterns (L3 Theory-Based) ---")
    shared = l3_distributions.get("shared", {})

    for field_name, display_name in [
        ("l3_color_harmony", "Color Harmony"),
        ("l3_tone_season", "Tone Season"),
        ("l3_coordination_role", "Coordination Role"),
        ("l3_style_lineage", "Style Lineage"),
    ]:
        dist = shared.get(field_name, {})
        if dist:
            dist_str = ", ".join(f"{k} {v*100:.0f}%" for k, v in list(dist.items())[:5])
            parts.append(f"{display_name}: {dist_str}")

    vw = shared.get("l3_visual_weight", {})
    if vw and vw.get("mean") is not None:
        parts.append(f"Visual Weight: mean {vw['mean']:.1f}, std {vw.get('std', 0):.1f}")

    # Category-specific patterns (with semantic labels instead of Slot6/Slot7)
    by_category = l3_distributions.get("by_category", {})
    if by_category:
        parts.append("Category-Specific:")
        for cat, cat_dist in by_category.items():
            n_items = cat_dist.get("n", 0)
            slot6 = cat_dist.get("l3_slot6", {})
            slot7 = cat_dist.get("l3_slot7", {})
            display_names = _L3_SLOT_DISPLAY.get(cat, {})
            cat_parts = [f"  {cat} (n={n_items}):"]
            if slot6:
                s6_str = ", ".join(f"{k} {v*100:.0f}%" for k, v in list(slot6.items())[:3])
                s6_label = display_names.get("l3_slot6", "Slot6")
                cat_parts.append(f"{s6_label}: {s6_str}")
            if slot7:
                s7_str = ", ".join(f"{k} {v*100:.0f}%" for k, v in list(slot7.items())[:3])
                s7_label = display_names.get("l3_slot7", "Slot7")
                cat_parts.append(f"{s7_label}: {s7_str}")
            parts.append(" | ".join(cat_parts))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Reasoning Text Composition
# ---------------------------------------------------------------------------


def compose_reasoning_text(reasoning_json: dict) -> str:
    """Convert 9-field reasoning JSON to natural language reasoning_text.

    Format: (a) Style mood: ... (b) Occasion: ... ... (i) Identity: ...
    This format is consistent with sparse user template output.
    """
    field_map = [
        ("a", "style_mood_preference", "Style mood"),
        ("b", "occasion_preference", "Occasion"),
        ("c", "quality_price_tendency", "Quality-price"),
        ("d", "trend_sensitivity", "Trend"),
        ("e", "seasonal_pattern", "Season"),
        ("f", "form_preference", "Form"),
        ("g", "color_tendency", "Color"),
        ("h", "coordination_tendency", "Coordination"),
        ("i", "identity_summary", "Identity"),
    ]
    parts = []
    for letter, key, label in field_map:
        value = reasoning_json.get(key, "Unknown")
        parts.append(f"({letter}) {label}: {value}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Batch API JSONL Line Builder
# ---------------------------------------------------------------------------


def build_reasoning_request_line(
    customer_id: str,
    l1_summary: dict,
    recent_items_l2: list[dict],
    l3_distributions: dict,
    config: ReasoningConfig = ReasoningConfig(),
) -> bytes:
    """Build a single JSONL request line for Batch API.

    Returns:
        UTF-8 bytes (including trailing newline).
    """
    user_message = build_reasoning_user_message(l1_summary, recent_items_l2, l3_distributions)

    request = {
        "custom_id": customer_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": config.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "user_reasoning_profile",
                    "schema": REASONING_SCHEMA,
                    "strict": True,
                }
            },
        },
    }
    return (json.dumps(request) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_field(val) -> dict:
    """Parse a JSON string field to dict."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return {}
    return {}
