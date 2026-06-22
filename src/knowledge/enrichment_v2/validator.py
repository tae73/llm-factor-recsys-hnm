"""Enrichment v2 schema validation (enum range, type, list-size, required fields).

Mirrors the lenient stance of ``factual/validator.py``: structural problems are
ERRORS; out-of-enum values are WARNINGS (the model occasionally drifts and we keep
the row rather than drop it). Used post-parse in the extractor and in unit tests.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from src.knowledge.enrichment_v2.schema import E2_LLM_AXES


class ValidationResult(NamedTuple):
    """Result of validating one parsed enrichment-v2 row."""

    is_valid: bool  # False only on structural errors (missing field / wrong type)
    errors: list[str]
    warnings: list[str]


def validate_e2(row: dict[str, Any]) -> ValidationResult:
    """Validate one parsed row against the LLM-axes schema.

    Errors (is_valid=False): a required axis is missing, or has the wrong container
    type (list vs scalar). Warnings (is_valid stays True): out-of-enum value, list
    size outside [min_items, max_items], or 'Low-maintenance' mixed with other flags.
    """
    errors: list[str] = []
    warnings: list[str] = []

    for name, meta in E2_LLM_AXES.items():
        if name not in row or row[name] is None:
            errors.append(f"missing required axis: {name}")
            continue
        val = row[name]
        enum = set(meta["enum"])

        if meta["is_list"]:
            if not isinstance(val, (list, tuple)):
                errors.append(f"{name}: expected list, got {type(val).__name__}")
                continue
            lo, hi = meta["min_items"], meta["max_items"]
            if not (lo <= len(val) <= hi):
                warnings.append(f"{name}: list size {len(val)} outside [{lo},{hi}]")
            bad = [v for v in val if v not in enum]
            if bad:
                warnings.append(f"{name}: out-of-enum {bad}")
        else:
            if isinstance(val, (list, tuple)):
                errors.append(f"{name}: expected scalar, got list")
                continue
            if val not in enum:
                warnings.append(f"{name}: out-of-enum value {val!r}")

    # Domain consistency: 'Low-maintenance' is mutually exclusive with real concerns.
    flags = row.get("e2_care_flags")
    if isinstance(flags, (list, tuple)) and "Low-maintenance" in flags and len(flags) > 1:
        warnings.append("e2_care_flags: 'Low-maintenance' mixed with other concerns")

    # Domain consistency: secondary occasion should differ from primary.
    prim = row.get("e2_occasion_primary")
    sec = row.get("e2_occasion_secondary")
    if prim is not None and isinstance(sec, (list, tuple)) and prim in sec:
        warnings.append("e2_occasion_secondary: contains the primary occasion")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
