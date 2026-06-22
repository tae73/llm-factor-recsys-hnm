"""Combined external-knowledge prompt: prose + structured complements in ONE call.

Realizes KAR's open-world premise (probe_16 / probe_19). Given ONE product a
customer bought, the LLM (acting as a fashion stylist) produces, in a SINGLE
structured-output call:

1. ``prose``       — a free-text complement description (probe_16 genre), and
2. ``complements`` — structured complement attributes (probe_19 genre):
                     product_type / colour / material / style.

The single combined json_schema call halves API volume vs. extracting prose and
structured complements separately. ``render_external_text`` turns the parsed
result into two attribute texts: ``prose_text`` (raw prose) and
``structured_text`` (structured complements rendered in L1's attribute-list
template, so BGE puts it in the same sub-space as L1 — the format-unify fix).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# System prompt (merges probe_16 PROSE intent + probe_19 STRUCTURED intent)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a professional fashion stylist with broad knowledge of outfit "
    "coordination, current trends, and styling rules. Given ONE product a "
    "customer bought, describe the COMPLEMENTARY items (in OTHER categories) "
    "that complete an outfit with it. Use external styling knowledge — do NOT "
    "just re-describe the given product. "
    "Return TWO things in the required JSON object: "
    "(1) `prose`: 2-3 concrete sentences naming the complementary product "
    "types, colors, materials, and the styling rationale; "
    "(2) `complements`: 1-3 complement items as STRUCTURED attributes "
    "(product_type, colour, material, style), each in a DIFFERENT category "
    "from the bought product."
)


# ---------------------------------------------------------------------------
# Combined JSON schema (strict structured output)
# ---------------------------------------------------------------------------

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "prose": {
            "type": "string",
            "description": (
                "2-3 sentence prose description of complementary items and the "
                "styling rationale (external knowledge, not a re-description)."
            ),
        },
        "complements": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "product_type": {"type": "string"},
                    "colour": {"type": "string"},
                    "material": {"type": "string"},
                    "style": {"type": "string"},
                },
                "required": ["product_type", "colour", "material", "style"],
            },
        },
    },
    "required": ["prose", "complements"],
}

# OpenAI structured-output `text.format` payload (Responses API).
RESPONSE_FORMAT: dict[str, Any] = {
    "format": {
        "type": "json_schema",
        "name": "external_complement",
        "schema": JSON_SCHEMA,
        "strict": True,
    }
}


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def build_item_description(meta: dict[str, Any], detail_desc_chars: int = 200) -> str:
    """Render representative product metadata into the user-message description.

    Mirrors probe_16/probe_19 ``desc`` construction.
    """
    detail = str(meta.get("detail_desc") or "")[:detail_desc_chars]
    return (
        f"Product type: {meta.get('product_type_name')}; "
        f"Color: {meta.get('colour_group_name')}; "
        f"Section: {meta.get('section_name')}; "
        f"Group: {meta.get('product_group_name')}. "
        f"{detail}"
    )


def build_messages(meta: dict[str, Any], detail_desc_chars: int = 200) -> list[dict[str, str]]:
    """Build the system+user message list for one product's combined extraction."""
    desc = build_item_description(meta, detail_desc_chars)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"The customer bought: {desc}"},
    ]


# ---------------------------------------------------------------------------
# Rendering: parsed result → attribute texts
# ---------------------------------------------------------------------------


def render_structured_text(complements: list[dict[str, Any]]) -> str:
    """Render structured complements in L1's attribute-list genre (probe_19._render).

    Produces a format-unified string so BGE places it in the same sub-space as
    L1 product attributes.
    """
    blocks = [
        (
            f"Type: {c.get('product_type', '')}; "
            f"Color: {c.get('colour', '')}; "
            f"[Product] Material: {c.get('material', '')}; "
            f"Style: {c.get('style', '')}"
        )
        for c in complements
    ]
    return " . ".join(blocks)


def render_external_text(result: dict[str, Any]) -> dict[str, str]:
    """Convert one parsed combined result into ``{prose_text, structured_text}``.

    Args:
        result: Parsed JSON ``{"prose": str, "complements": [{...}, ...]}``.

    Returns:
        ``{"prose_text": <prose>, "structured_text": <rendered attribute list>}``.
    """
    prose_text = str(result.get("prose", "") or "").strip()
    complements = result.get("complements") or []
    structured_text = render_structured_text(complements) if complements else ""
    return {"prose_text": prose_text, "structured_text": structured_text}
