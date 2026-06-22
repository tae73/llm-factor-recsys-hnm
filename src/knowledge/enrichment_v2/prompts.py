"""Enrichment v2 prompt — the ANTI-RECODING multimodal message builder.

The v1 factual prompt (``factual/prompts.py::build_user_message``) injected the
item's metadata block (product_type_name, colour_group_name, graphical_appearance_name,
section_name, department_name, ...) into the user message. DE1 proved this caused
deterministic recoding (``l3_color_harmony``/``l3_tone_season`` metadata-lift = 1.00).

This builder inverts that. The model sees ONLY:
  * the product IMAGE (primary signal), and
  * ``prod_name`` (a short marketing name), and
  * a metadata-STRIPPED ``detail_desc`` (fabric-composition + care sentences removed,
    so the care/fit axes cannot read off the material and recode it).

It is NEVER passed product_type_name, colour_group_name, graphical_appearance_name,
section_name, department_name, garment_group_name, index_name. The system prompt
forbids inferring occasion from garment type and care from fabric type. ``render_e2_row``
turns the parsed structured output into the ``e2_*`` columns.
"""

from __future__ import annotations

import re
from typing import Any

from src.knowledge.enrichment_v2.schema import E2_LLM_AXES, E2_RESPONSE_FORMAT  # noqa: F401

# ---------------------------------------------------------------------------
# System prompt — no-metadata, image-first, with explicit anti-recoding blocks
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = (
    "You are a professional fashion stylist. You are given ONLY a product IMAGE and a "
    "short product name (sometimes a brief description). Judge the requested decision "
    "axes from what you SEE in the image and basic styling knowledge.\n"
    "HARD RULES (read carefully):\n"
    "1. Do NOT guess from category words. A pair of trousers is not automatically "
    "'Workday-office'; a dress is not automatically 'Formal-event'. Choose the most "
    "SPECIFIC occasion the actual garment suggests, and use the full range of occasions.\n"
    "2. Do NOT infer care from fabric type. 'Silk → dry-clean' is forbidden reasoning. "
    "Judge care BURDEN from visible construction, embellishment, trims, structure, and "
    "color — concerns that vary even within the same fabric.\n"
    "3. Fit-intent is the INTENDED garment-to-body relationship (the styling reason), "
    "NOT a cut label. Judge ease/drape from how the garment sits in the image.\n"
    "4. price_look and trend_look are PERCEPTIONS from the image only — you do NOT know "
    "the real price or sales trajectory; report how it LOOKS.\n"
    "If a product name contains a fabric or fit word, ignore it for these judgments.\n"
    "Return exactly the required JSON object."
)

# ---------------------------------------------------------------------------
# detail_desc metadata stripper (kills the material/care recoding channel)
# ---------------------------------------------------------------------------
# Phrases that leak fabric composition or explicit care → removed before the model
# sees detail_desc, so care_burden / fit_intent cannot trivially recode material.
_FABRIC_WORDS = (
    r"cotton|polyester|viscose|elastane|wool|silk|linen|nylon|acrylic|cashmere|"
    r"leather|denim|jersey|modal|lyocell|spandex|rayon|polyamide|satin|velvet|"
    r"corduroy|fleece|chiffon|organza|tulle"
)
_STRIP_PATTERNS = [
    re.compile(r"[^.]*\b\d{1,3}\s?%[^.]*\.", re.IGNORECASE),  # "... 95% cotton, 5% elastane."
    re.compile(
        rf"[^.]*\b(?:{_FABRIC_WORDS})\b[^.]*\.", re.IGNORECASE
    ),  # fabric-mentioning sentences
    re.compile(
        r"[^.]*\b(machine wash|hand wash|dry clean|do not bleach|tumble dry|iron|"
        r"wash inside out|wash separately)\b[^.]*\.",
        re.IGNORECASE,
    ),
]


_FABRIC_WORD_RE = re.compile(rf"\b(?:{_FABRIC_WORDS})\b", re.IGNORECASE)


def strip_metadata(detail_desc: str) -> str:
    """Remove fabric-composition and care sentences from detail_desc (anti-recoding)."""
    text = str(detail_desc or "")
    for pat in _STRIP_PATTERNS:
        text = pat.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_fabric_words(text: str) -> str:
    """Remove bare fabric words from a short string (e.g. prod_name 'Slim cotton tee').

    The marketing name is the most deterministic care/material recoding channel, so we
    drop fabric tokens from it while keeping shape/fit cues the stylist legitimately sees.
    """
    return re.sub(r"\s+", " ", _FABRIC_WORD_RE.sub(" ", str(text or ""))).strip()


# ---------------------------------------------------------------------------
# Message builder (Responses API multimodal input)
# ---------------------------------------------------------------------------
def build_e2_messages(
    rep: dict[str, Any], image_base64: str | None, detail_desc_chars: int = 180
) -> list[dict[str, Any]]:
    """Build the system+user message list for one product's enrichment-v2 extraction.

    Args:
        rep: representative row dict; uses ONLY ``prod_name`` and ``detail_desc``.
            (Categorical metadata fields are intentionally never read.)
        image_base64: base64 JPEG of the product image, or None (text-only fallback).
        detail_desc_chars: truncation of the stripped detail_desc.

    Returns:
        Responses API ``input`` list: a system message + a user message whose content
        is a parts list (text + optional image). Contains NO categorical metadata.
    """
    prod_name = strip_fabric_words(rep.get("prod_name", ""))
    stripped = strip_metadata(rep.get("detail_desc", ""))[:detail_desc_chars]

    text = f"Product name: {prod_name}."
    if stripped:
        text += f" Note: {stripped}"
    text += (
        " Judge the decision axes from the image (and the name) per the rules. "
        "Return the required JSON."
    )

    user_content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    if image_base64:
        user_content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_base64}",
                "detail": "low",
            }
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Parsed result → e2_* row
# ---------------------------------------------------------------------------
def render_e2_row(parsed: dict[str, Any]) -> dict[str, Any]:
    """Convert one parsed structured result into the ``e2_*`` LLM columns.

    Missing/invalid fields are coerced to ``None`` (the validator flags them).
    List fields stay as Python lists (the cache JSON-encodes them for Parquet).
    """
    row: dict[str, Any] = {}
    for name, meta in E2_LLM_AXES.items():
        val = parsed.get(name)
        if meta["is_list"]:
            row[name] = list(val) if isinstance(val, (list, tuple)) else None
        else:
            row[name] = val
    return row
