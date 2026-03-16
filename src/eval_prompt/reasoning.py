"""User reasoning knowledge evaluation orchestration.

Combines structural checks (coverage, completeness, discriminability, token budget)
with LLM-as-Judge (5 dimensions) for comprehensive user reasoning evaluation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from src.eval_prompt.judge import (
    JudgeConfig,
    JudgeDimension,
    JudgeReport,
    build_judge_system_prompt,
    evaluate_batch,
)
from src.eval_prompt.structural import (
    REASONING_FIELDS,
    CompletenessResult,
    CoverageResult,
    DiscriminabilityResult,
    TokenBudgetResult,
    check_completeness,
    check_discriminability,
    check_token_budget,
    compute_coverage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reasoning Dimensions (domain-specific descriptions)
# ---------------------------------------------------------------------------

REASONING_DIMENSIONS: list[JudgeDimension] = [
    JudgeDimension(
        "accuracy",
        "Does the reasoning accurately reflect the user's purchase data? "
        "Check if style mood, occasion, quality-price, trend, and color preferences "
        "are supported by the L1/L2/L3 patterns provided in the source data.",
    ),
    JudgeDimension(
        "specificity",
        "Is the reasoning specific to this user, not generic boilerplate? "
        "Penalize phrases like 'values quality and comfort' without specific details. "
        "Reward references to concrete patterns (e.g., '70% casual purchases, "
        "primarily I-line silhouettes').",
    ),
    JudgeDimension(
        "coherence",
        "Do the 9 profile fields form a coherent fashion identity without contradictions? "
        "For example, a 'minimalist' style shouldn't pair with 'maximalist statement pieces' "
        "in form_preference, or 'budget' quality-price shouldn't pair with 'luxury' trend.",
    ),
    JudgeDimension(
        "source_alignment",
        "Does the reasoning align with the source data (price quintile, L1 category stats, "
        "L2 distributions, L3 patterns)? Specifically check that quality_price_tendency "
        "references the actual price quintile, not just perceived quality scores.",
    ),
    JudgeDimension(
        "informativeness",
        "Does the reasoning provide meaningful fashion preference insights useful for "
        "recommendation? Does the identity_summary capture a distinctive fashion personality? "
        "Are the preferences specific enough to guide item matching?",
    ),
]

# ---------------------------------------------------------------------------
# Configuration & Report Types
# ---------------------------------------------------------------------------


class ReasoningEvalConfig(NamedTuple):
    """User reasoning evaluation configuration."""

    judge_config: JudgeConfig = JudgeConfig()
    token_budget_limit: int = 512
    generic_markers: tuple[str, ...] = (
        "Unknown", "N/A", "Not available", "No data", "Insufficient",
    )
    run_judge: bool = True


class ReasoningEvalReport(NamedTuple):
    """Complete user reasoning evaluation report."""

    completeness: CompletenessResult
    discriminability: DiscriminabilityResult
    coverage: CoverageResult
    token_budget: TokenBudgetResult
    judge: JudgeReport | None
    timestamp: str


# ---------------------------------------------------------------------------
# Judge Message Builder
# ---------------------------------------------------------------------------


def build_reasoning_judge_message(
    l1_summary: dict[str, Any],
    recent_items_l2: list[dict],
    l3_distributions: dict,
    reasoning_json: dict[str, str],
) -> str:
    """Build the user message for reasoning judge evaluation.

    Args:
        l1_summary: L1 aggregated stats (n_purchases, top_categories, price quintile, etc.)
        recent_items_l2: Recent items with L2 attributes.
        l3_distributions: L3 distribution data.
        reasoning_json: The 9-field reasoning profile to evaluate.

    Returns:
        Formatted text message for the judge.
    """
    from src.knowledge.reasoning.prompts import build_reasoning_user_message

    # Reconstruct source data (same format as LLM input)
    source_text = build_reasoning_user_message(l1_summary, recent_items_l2, l3_distributions)

    # Format the reasoning output
    reasoning_lines = []
    if reasoning_json is None:
        reasoning_json = {}
    for key, val in reasoning_json.items():
        reasoning_lines.append(f"  {key}: {val}")
    reasoning_str = "\n".join(reasoning_lines)

    return (
        f"--- Source Data (LLM Input) ---\n{source_text}\n\n"
        f"--- Generated Reasoning Profile ---\n{reasoning_str}\n\n"
        "Evaluate the quality of this reasoning profile based on the source data above."
    )


# ---------------------------------------------------------------------------
# Judge Runner
# ---------------------------------------------------------------------------


async def run_reasoning_judge(
    profiles_df: pd.DataFrame,
    txn_path: Path,
    fk_path: Path,
    config: JudgeConfig = JudgeConfig(),
) -> JudgeReport:
    """Run LLM-as-Judge on a sample of user reasoning profiles.

    Args:
        profiles_df: User profiles DataFrame (customer_id, reasoning_json, reasoning_text).
        txn_path: Path to transactions Parquet file.
        fk_path: Path to factual knowledge Parquet file.
        config: Judge configuration.

    Returns:
        JudgeReport with per-dimension and overall scores.
    """
    from src.knowledge.reasoning.extractor import (
        compute_l3_distributions_batch,
        get_recent_items_batch,
    )

    # Sample profiles — only those with reasoning_json (sparse users have None)
    has_reasoning = profiles_df[profiles_df["reasoning_json"].notna()]
    logger.info(
        "Profiles with reasoning_json: %d / %d", len(has_reasoning), len(profiles_df)
    )
    if len(has_reasoning) > config.sample_size:
        sampled = has_reasoning.sample(n=config.sample_size, random_state=42)
    else:
        sampled = has_reasoning

    customer_ids = sampled["customer_id"].astype(str).tolist()

    # Fetch source data for sampled profiles
    logger.info("Fetching source data for %d sampled profiles...", len(customer_ids))
    recent_items = get_recent_items_batch(txn_path, fk_path, customer_ids)
    l3_dists = compute_l3_distributions_batch(txn_path, fk_path, customer_ids)

    # Build items list
    items: list[dict[str, Any]] = []
    for _, row in sampled.iterrows():
        cid = str(row["customer_id"])

        # Parse reasoning_json
        rj = row.get("reasoning_json")
        if rj is None or (isinstance(rj, float) and pd.isna(rj)):
            rj = {}
        elif isinstance(rj, str):
            try:
                rj = json.loads(rj)
            except json.JSONDecodeError:
                rj = {}

        # L1 summary from profile columns
        l1_summary = {
            "n_purchases": row.get("n_purchases", 0),
            "n_unique_types": row.get("n_unique_types", 0),
            "category_diversity": row.get("category_diversity", 0.0),
            "top_categories_json": row.get("top_categories_json", "{}"),
            "avg_price_quintile": row.get("avg_price_quintile", 3.0),
            "online_ratio": row.get("online_ratio", 0.0),
        }

        items.append({
            "item_id": cid,
            "l1_summary": l1_summary,
            "recent_items_l2": recent_items.get(cid, []),
            "l3_distributions": l3_dists.get(cid, {}),
            "reasoning_json": rj,
        })

    system_prompt = build_judge_system_prompt("user_profile", REASONING_DIMENSIONS)

    def _build_msg(item: dict[str, Any]) -> str:
        return build_reasoning_judge_message(
            item["l1_summary"],
            item["recent_items_l2"],
            item["l3_distributions"],
            item["reasoning_json"],
        )

    return await evaluate_batch(
        items=items,
        system_prompt=system_prompt,
        build_user_msg_fn=_build_msg,
        dimensions=REASONING_DIMENSIONS,
        config=config,
    )


# ---------------------------------------------------------------------------
# Full Evaluation Orchestration
# ---------------------------------------------------------------------------


async def run_reasoning_eval(
    profiles_df: pd.DataFrame,
    txn_path: Path,
    fk_path: Path,
    config: ReasoningEvalConfig = ReasoningEvalConfig(),
) -> ReasoningEvalReport:
    """Run complete user reasoning evaluation (structural + judge).

    Args:
        profiles_df: User profiles DataFrame.
        txn_path: Path to transactions Parquet file.
        fk_path: Path to factual knowledge Parquet file.
        config: Evaluation configuration.

    Returns:
        ReasoningEvalReport with all check results.
    """
    logger.info("Running reasoning evaluation on %d profiles...", len(profiles_df))

    # 1. Coverage
    coverage_fields = ["reasoning_text", "reasoning_json", "n_purchases"]
    coverage = compute_coverage(profiles_df, coverage_fields)
    logger.info("Coverage: %.1f%% overall", coverage.overall_coverage * 100)

    # 2. Completeness (9-field)
    completeness = check_completeness(
        profiles_df, REASONING_FIELDS, config.generic_markers
    )
    logger.info(
        "Completeness: %.1f%% overall, %d generic, %d short",
        completeness.overall_completeness * 100,
        completeness.n_generic,
        completeness.n_short,
    )

    # 3. Discriminability
    texts = profiles_df["reasoning_text"].dropna().astype(str).tolist()
    discriminability = check_discriminability(texts)
    logger.info(
        "Discriminability: mean_sim=%.3f, mean_trigrams=%.0f",
        discriminability.mean_pairwise_sim,
        discriminability.mean_trigrams,
    )

    # 4. Token budget
    token_budget = check_token_budget(texts, budget_limit=config.token_budget_limit)
    logger.info(
        "Token budget: mean=%.0f, p95=%.0f, %d over limit",
        token_budget.mean_tokens, token_budget.p95_tokens, token_budget.n_over_budget,
    )

    # 5. LLM-as-Judge (optional)
    judge_report = None
    if config.run_judge:
        logger.info("Running LLM-as-Judge on %d samples...", config.judge_config.sample_size)
        judge_report = await run_reasoning_judge(
            profiles_df, txn_path, fk_path, config.judge_config
        )
        logger.info(
            "Judge: overall=%.2f, pass_rate=%.1f%%",
            judge_report.overall_mean, judge_report.pass_rate * 100,
        )

    return ReasoningEvalReport(
        completeness=completeness,
        discriminability=discriminability,
        coverage=coverage,
        token_budget=token_budget,
        judge=judge_report,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_reasoning_eval_sync(
    profiles_df: pd.DataFrame,
    txn_path: Path,
    fk_path: Path,
    config: ReasoningEvalConfig = ReasoningEvalConfig(),
) -> ReasoningEvalReport:
    """Synchronous wrapper for run_reasoning_eval."""
    return asyncio.run(run_reasoning_eval(profiles_df, txn_path, fk_path, config))
