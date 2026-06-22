"""Enrichment v2 — interpretable multi-purpose catalog enrichment (6 decision-axes).

Public surface:
    schema:    E2_LLM_AXES, E2_BEHAVIORAL_AXES, E2_LIST_COLS, E2_ALL_COLUMNS,
               build_e2_json_schema, E2_RESPONSE_FORMAT, GAP_AXES
    prompts:   SYSTEM_PROMPT, build_e2_messages, render_e2_row, strip_metadata
    extractor: estimate_e2_cost, extract_e2_pilot, group_e2_representatives
    validator: validate_e2, ValidationResult
"""

from src.knowledge.enrichment_v2.extractor import (
    estimate_e2_cost,
    extract_e2_pilot,
    group_e2_representatives,
)
from src.knowledge.enrichment_v2.prompts import (
    SYSTEM_PROMPT,
    build_e2_messages,
    render_e2_row,
    strip_metadata,
)
from src.knowledge.enrichment_v2.schema import (
    E2_ALL_COLUMNS,
    E2_BEHAVIORAL_AXES,
    E2_LIST_COLS,
    E2_LLM_AXES,
    E2_RESPONSE_FORMAT,
    GAP_AXES,
    build_e2_json_schema,
)
from src.knowledge.enrichment_v2.validator import ValidationResult, validate_e2

__all__ = [
    "E2_ALL_COLUMNS",
    "E2_BEHAVIORAL_AXES",
    "E2_LIST_COLS",
    "E2_LLM_AXES",
    "E2_RESPONSE_FORMAT",
    "GAP_AXES",
    "SYSTEM_PROMPT",
    "ValidationResult",
    "build_e2_json_schema",
    "build_e2_messages",
    "estimate_e2_cost",
    "extract_e2_pilot",
    "group_e2_representatives",
    "render_e2_row",
    "strip_metadata",
    "validate_e2",
]
