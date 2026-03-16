"""Structural (programmatic) evaluation for prompt outputs.

Provides deterministic, mathematical, and counting-based checks that LLMs
cannot reliably perform. Shared functions (coverage, token budget) are used
by both factual knowledge and user profile evaluation domains.

Domain-specific checks:
  - Factual: schema validation, domain consistency rules, enum distributions
  - Profile: field completeness, generic detection, discriminability
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.knowledge.factual.prompts import (
    CATEGORY_SPECIFIC_L1_FIELDS,
    CATEGORY_SPECIFIC_L3_FIELDS,
)
from src.knowledge.factual.validator import (
    validate_domain_consistency,
    validate_final_knowledge,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common Result Types
# ---------------------------------------------------------------------------


class CoverageResult(NamedTuple):
    """Non-null coverage statistics."""

    field_coverage: dict[str, float]  # field → non-null fraction
    overall_coverage: float
    n_items: int


class TokenBudgetResult(NamedTuple):
    """Token count statistics against a budget limit."""

    mean_tokens: float
    median_tokens: float
    p95_tokens: float
    p99_tokens: float
    max_tokens: int
    n_over_budget: int
    pct_over_budget: float
    budget_limit: int


# ---------------------------------------------------------------------------
# Factual-Specific Result Types
# ---------------------------------------------------------------------------


class SchemaCheckResult(NamedTuple):
    """Schema validation aggregate statistics."""

    n_valid: int
    n_invalid: int
    error_counts: dict[str, int]  # error message → count
    warning_counts: dict[str, int]


class DomainCheckResult(NamedTuple):
    """Domain consistency rule aggregate statistics."""

    n_items_with_violations: int
    n_error_violations: int
    n_warning_violations: int
    rule_counts: dict[str, int]  # rule_name → violation count


class DistributionResult(NamedTuple):
    """Enum value distribution statistics."""

    value_counts: dict[str, dict[str, int]]  # field → {value: count}
    entropy: dict[str, float]  # field → Shannon entropy
    n_unique: dict[str, int]  # field → count of unique values


# ---------------------------------------------------------------------------
# Profile-Specific Result Types
# ---------------------------------------------------------------------------


class CompletenessResult(NamedTuple):
    """Profile field completeness statistics."""

    field_completeness: dict[str, float]  # field → non-null/non-generic fraction
    overall_completeness: float
    n_generic: int  # profiles with generic marker content
    n_short: int  # profiles with very short reasoning text


class DiscriminabilityResult(NamedTuple):
    """Profile discriminability statistics (how distinct profiles are)."""

    mean_pairwise_sim: float
    median_pairwise_sim: float
    per_field_unique_ratio: dict[str, float]  # field → unique/total ratio
    mean_trigrams: float  # mean unique trigram count per text


# ---------------------------------------------------------------------------
# Common Functions
# ---------------------------------------------------------------------------


def compute_coverage(df: pd.DataFrame, fields: list[str]) -> CoverageResult:
    """Compute non-null coverage for each field.

    Args:
        df: DataFrame with columns to check.
        fields: List of column names to evaluate.

    Returns:
        CoverageResult with per-field and overall coverage.
    """
    field_cov: dict[str, float] = {}
    n = len(df)
    if n == 0:
        return CoverageResult(field_coverage={}, overall_coverage=0.0, n_items=0)

    for f in fields:
        if f not in df.columns:
            field_cov[f] = 0.0
        else:
            non_null = df[f].notna().sum()
            # Also exclude empty strings (safe for columns that may hold lists/arrays)
            if df[f].dtype == object:
                def _is_filled(v: object) -> bool:
                    if v is None:
                        return False
                    if isinstance(v, float) and pd.isna(v):
                        return False
                    if isinstance(v, str):
                        return v != ""
                    # lists, np.ndarray, etc. — treat as filled
                    return True
                non_null = df[f].apply(_is_filled).sum()
            field_cov[f] = float(non_null / n)

    overall = sum(field_cov.values()) / len(field_cov) if field_cov else 0.0
    return CoverageResult(field_coverage=field_cov, overall_coverage=overall, n_items=n)


def check_token_budget(
    texts: list[str],
    budget_limit: int = 512,
    model_name: str = "bge-base-en-v1.5",
) -> TokenBudgetResult:
    """Check token counts against a budget limit using whitespace tokenization.

    Uses a simple whitespace-split approximation (×1.3 factor for subword overhead)
    suitable for BGE/transformer models. For exact counts, use the model's tokenizer.

    Args:
        texts: List of text strings to count.
        budget_limit: Maximum allowed token count (default 512 for BGE-base).
        model_name: Model name (informational, uses whitespace approx regardless).

    Returns:
        TokenBudgetResult with distribution statistics.
    """
    if not texts:
        return TokenBudgetResult(
            mean_tokens=0.0,
            median_tokens=0.0,
            p95_tokens=0.0,
            p99_tokens=0.0,
            max_tokens=0,
            n_over_budget=0,
            pct_over_budget=0.0,
            budget_limit=budget_limit,
        )

    # Whitespace-split × 1.3 factor for subword overhead
    counts = np.array([int(len(t.split()) * 1.3) for t in texts])
    n_over = int((counts > budget_limit).sum())

    return TokenBudgetResult(
        mean_tokens=float(np.mean(counts)),
        median_tokens=float(np.median(counts)),
        p95_tokens=float(np.percentile(counts, 95)),
        p99_tokens=float(np.percentile(counts, 99)),
        max_tokens=int(np.max(counts)),
        n_over_budget=n_over,
        pct_over_budget=n_over / len(counts),
        budget_limit=budget_limit,
    )


# ---------------------------------------------------------------------------
# Factual-Specific Functions
# ---------------------------------------------------------------------------

# All knowledge fields (L1+L2+L3 shared + category-specific + tone_season)
_L1_SHARED = ["l1_material", "l1_closure", "l1_design_details", "l1_material_detail"]
_L2_FIELDS = [
    "l2_style_mood", "l2_occasion", "l2_perceived_quality", "l2_trendiness",
    "l2_season_fit", "l2_target_impression", "l2_versatility",
]
_L3_SHARED = [
    "l3_color_harmony", "l3_tone_season", "l3_coordination_role",
    "l3_visual_weight", "l3_style_lineage",
]

# Enum fields suitable for distribution analysis
FACTUAL_ENUM_FIELDS: list[str] = [
    "l2_style_mood", "l2_occasion", "l2_trendiness", "l2_season_fit",
    "l3_color_harmony", "l3_tone_season", "l3_coordination_role", "l3_style_lineage",
]


def _get_factual_fields(super_category: str) -> list[str]:
    """Get all expected fields for a super-category (including tone_season)."""
    l1_specific = CATEGORY_SPECIFIC_L1_FIELDS.get(super_category, [])
    l3_specific = CATEGORY_SPECIFIC_L3_FIELDS.get(super_category, [])
    return _L1_SHARED + l1_specific + _L2_FIELDS + _L3_SHARED + l3_specific


def _reverse_canonical_slots(knowledge: dict, super_category: str) -> dict:
    """Reverse-map Parquet canonical slots back to semantic field names.

    Parquet stores category-specific fields as l1_slot4-7 / l3_slot6-7, but
    validators expect semantic names (l1_neckline, l3_silhouette, etc.).
    """
    from src.knowledge.factual.prompts import L1_SLOT_NAMES, L3_SLOT_NAMES

    result = dict(knowledge)
    l1_fields = CATEGORY_SPECIFIC_L1_FIELDS.get(super_category, [])
    for slot_name, field_name in zip(L1_SLOT_NAMES, l1_fields):
        if slot_name in result:
            result[field_name] = result.pop(slot_name)
    l3_fields = CATEGORY_SPECIFIC_L3_FIELDS.get(super_category, [])
    for slot_name, field_name in zip(L3_SLOT_NAMES, l3_fields):
        if slot_name in result:
            result[field_name] = result.pop(slot_name)
    return result


def _build_id_to_cat(articles_df: pd.DataFrame) -> dict[str, str]:
    """Build article_id → super_category mapping vectorized."""
    from src.knowledge.factual.prompts import resolve_super_category

    id_to_cat: dict[str, str] = {}
    for _, row in articles_df.iterrows():
        aid = str(row.get("article_id", ""))
        gg = str(row.get("garment_group_name", "Unknown"))
        pg = str(row.get("product_group_name", "")) if "product_group_name" in row.index else None
        id_to_cat[aid] = resolve_super_category(gg, pg)
    return id_to_cat


def run_schema_checks(
    knowledge_df: pd.DataFrame,
    articles_df: pd.DataFrame,
) -> SchemaCheckResult:
    """Run schema validation on all knowledge rows.

    Args:
        knowledge_df: Factual knowledge DataFrame with article_id + attribute columns.
        articles_df: Articles metadata DataFrame (needs garment_group_name for super_category).

    Returns:
        SchemaCheckResult with aggregate error/warning counts.
    """
    id_to_cat = _build_id_to_cat(articles_df)

    n_valid = 0
    n_invalid = 0
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()

    for _, row in knowledge_df.iterrows():
        aid = str(row.get("article_id", ""))
        cat = id_to_cat.get(aid, "Apparel")
        knowledge = _reverse_canonical_slots(row.to_dict(), cat)

        result = validate_final_knowledge(knowledge, cat)
        if result.is_valid:
            n_valid += 1
        else:
            n_invalid += 1
        for e in result.errors:
            error_counts[e] += 1
        for w in result.warnings:
            warning_counts[w] += 1

    return SchemaCheckResult(
        n_valid=n_valid,
        n_invalid=n_invalid,
        error_counts=dict(error_counts),
        warning_counts=dict(warning_counts),
    )


def run_domain_checks(
    knowledge_df: pd.DataFrame,
    articles_df: pd.DataFrame,
) -> DomainCheckResult:
    """Run domain consistency rules on all knowledge rows.

    Args:
        knowledge_df: Factual knowledge DataFrame.
        articles_df: Articles metadata DataFrame.

    Returns:
        DomainCheckResult with aggregate violation counts.
    """
    id_to_cat = _build_id_to_cat(articles_df)

    n_items_with_violations = 0
    n_errors = 0
    n_warnings = 0
    rule_counts: Counter[str] = Counter()

    for _, row in knowledge_df.iterrows():
        aid = str(row.get("article_id", ""))
        cat = id_to_cat.get(aid, "Apparel")
        knowledge = _reverse_canonical_slots(row.to_dict(), cat)

        violations = validate_domain_consistency(knowledge, cat)
        if violations:
            n_items_with_violations += 1
        for v in violations:
            rule_counts[v.rule_name] += 1
            if v.severity == "Error":
                n_errors += 1
            else:
                n_warnings += 1

    return DomainCheckResult(
        n_items_with_violations=n_items_with_violations,
        n_error_violations=n_errors,
        n_warning_violations=n_warnings,
        rule_counts=dict(rule_counts),
    )


def compute_distributions(
    df: pd.DataFrame,
    enum_fields: list[str] | None = None,
) -> DistributionResult:
    """Compute value distributions and entropy for enum fields.

    Handles both scalar string fields and JSON array string fields.

    Args:
        df: DataFrame with enum attribute columns.
        enum_fields: Fields to analyze (defaults to FACTUAL_ENUM_FIELDS).

    Returns:
        DistributionResult with value counts, entropy, and unique counts.
    """
    if enum_fields is None:
        enum_fields = FACTUAL_ENUM_FIELDS

    value_counts: dict[str, dict[str, int]] = {}
    entropy: dict[str, float] = {}
    n_unique: dict[str, int] = {}

    for field in enum_fields:
        if field not in df.columns:
            value_counts[field] = {}
            entropy[field] = 0.0
            n_unique[field] = 0
            continue

        # Collect all values (explode arrays)
        all_values: list[str] = []
        for val in df[field].dropna():
            if isinstance(val, list):
                all_values.extend(str(v) for v in val)
            elif isinstance(val, str):
                # Try JSON array parse
                if val.startswith("["):
                    try:
                        parsed = json.loads(val)
                        all_values.extend(str(v) for v in parsed)
                        continue
                    except json.JSONDecodeError:
                        pass
                all_values.append(val)
            else:
                all_values.append(str(val))

        counts = Counter(all_values)
        value_counts[field] = dict(counts)
        n_unique[field] = len(counts)

        # Shannon entropy
        total = sum(counts.values())
        if total > 0:
            probs = [c / total for c in counts.values()]
            entropy[field] = -sum(p * math.log2(p) for p in probs if p > 0)
        else:
            entropy[field] = 0.0

    return DistributionResult(
        value_counts=value_counts,
        entropy=entropy,
        n_unique=n_unique,
    )


# ---------------------------------------------------------------------------
# Profile-Specific Functions
# ---------------------------------------------------------------------------

REASONING_FIELDS: list[str] = [
    "style_mood_preference",
    "occasion_preference",
    "quality_price_tendency",
    "trend_sensitivity",
    "seasonal_pattern",
    "form_preference",
    "color_tendency",
    "coordination_tendency",
    "identity_summary",
]


def check_completeness(
    df: pd.DataFrame,
    fields: list[str] | None = None,
    generic_markers: tuple[str, ...] = (
        "Unknown", "N/A", "Not available", "No data", "Insufficient",
    ),
) -> CompletenessResult:
    """Check profile field completeness and detect generic content.

    Args:
        df: Profile DataFrame with reasoning_json column or individual fields.
        fields: Profile fields to check (defaults to PROFILE_FIELDS).
        generic_markers: Strings indicating generic/placeholder content.

    Returns:
        CompletenessResult with per-field completeness and generic detection.
    """
    if fields is None:
        fields = REASONING_FIELDS

    n = len(df)
    if n == 0:
        return CompletenessResult(
            field_completeness={}, overall_completeness=0.0, n_generic=0, n_short=0
        )

    # Parse reasoning_json if individual fields not present
    parsed_records: list[dict] = []
    if "reasoning_json" in df.columns and fields[0] not in df.columns:
        for val in df["reasoning_json"]:
            if isinstance(val, dict):
                parsed_records.append(val)
            elif isinstance(val, str):
                try:
                    parsed_records.append(json.loads(val))
                except json.JSONDecodeError:
                    parsed_records.append({})
            else:
                parsed_records.append({})
    else:
        parsed_records = [row.to_dict() for _, row in df.iterrows()]

    field_completeness: dict[str, float] = {}
    n_generic = 0
    n_short = 0

    generic_lower = tuple(g.lower() for g in generic_markers)

    for f in fields:
        non_generic_count = 0
        for record in parsed_records:
            val = record.get(f, "")
            if val and isinstance(val, str):
                val_lower = val.strip().lower()
                if val_lower and not any(val_lower.startswith(g) for g in generic_lower):
                    non_generic_count += 1
        field_completeness[f] = non_generic_count / n

    overall = sum(field_completeness.values()) / len(field_completeness) if field_completeness else 0.0

    # Count profiles with any generic field
    for record in parsed_records:
        has_generic = False
        total_len = 0
        for f in fields:
            val = record.get(f, "")
            if isinstance(val, str):
                total_len += len(val)
                val_lower = val.strip().lower()
                if any(val_lower.startswith(g) for g in generic_lower):
                    has_generic = True
        if has_generic:
            n_generic += 1
        if total_len < 100:
            n_short += 1

    return CompletenessResult(
        field_completeness=field_completeness,
        overall_completeness=overall,
        n_generic=n_generic,
        n_short=n_short,
    )


def check_discriminability(
    texts: list[str],
    fields_df: pd.DataFrame | None = None,
    max_sample: int = 500,
) -> DiscriminabilityResult:
    """Check how discriminable (distinct) profile texts are from each other.

    Uses TF-IDF cosine similarity for pairwise comparison and unique trigram
    counts for information richness.

    Args:
        texts: List of reasoning_text strings.
        fields_df: Optional DataFrame with individual profile fields for per-field uniqueness.
        max_sample: Max profiles to sample for pairwise similarity (performance).

    Returns:
        DiscriminabilityResult with similarity and uniqueness metrics.
    """
    if not texts:
        return DiscriminabilityResult(
            mean_pairwise_sim=0.0,
            median_pairwise_sim=0.0,
            per_field_unique_ratio={},
            mean_trigrams=0.0,
        )

    # Sample if too many texts
    if len(texts) > max_sample:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(texts), size=max_sample, replace=False)
        sampled_texts = [texts[i] for i in indices]
    else:
        sampled_texts = texts

    # TF-IDF pairwise similarity
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sampled_texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        # Extract upper triangle (excluding diagonal)
        n = sim_matrix.shape[0]
        upper_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_indices]
        mean_sim = float(np.mean(pairwise_sims))
        median_sim = float(np.median(pairwise_sims))
    except ValueError:
        mean_sim = 0.0
        median_sim = 0.0

    # Unique trigram count per text
    trigram_counts = []
    for t in sampled_texts:
        words = t.lower().split()
        trigrams = set()
        for i in range(len(words) - 2):
            trigrams.add((words[i], words[i + 1], words[i + 2]))
        trigram_counts.append(len(trigrams))
    mean_trigrams = float(np.mean(trigram_counts)) if trigram_counts else 0.0

    # Per-field unique ratio
    per_field_unique: dict[str, float] = {}
    if fields_df is not None and len(fields_df) > 0:
        for f in REASONING_FIELDS:
            if f in fields_df.columns:
                n_total = fields_df[f].notna().sum()
                n_unique = fields_df[f].nunique()
                per_field_unique[f] = n_unique / n_total if n_total > 0 else 0.0

    return DiscriminabilityResult(
        mean_pairwise_sim=mean_sim,
        median_pairwise_sim=median_sim,
        per_field_unique_ratio=per_field_unique,
        mean_trigrams=mean_trigrams,
    )
