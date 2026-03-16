"""Factual knowledge evaluation orchestration.

Combines structural checks (coverage, schema, domain, distribution, token budget)
with LLM-as-Judge (5 dimensions) for comprehensive factual knowledge evaluation.
"""

from __future__ import annotations

import asyncio
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
    CoverageResult,
    DistributionResult,
    DomainCheckResult,
    SchemaCheckResult,
    TokenBudgetResult,
    check_token_budget,
    compute_coverage,
    compute_distributions,
    run_domain_checks,
    run_schema_checks,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factual Dimensions (domain-specific descriptions)
# ---------------------------------------------------------------------------

FACTUAL_DIMENSIONS: list[JudgeDimension] = [
    JudgeDimension(
        "accuracy",
        "Are the extracted L1/L2/L3 attributes factually correct for this product? "
        "Check material, closure, style mood, occasion, color harmony, silhouette, etc. "
        "against the product image and metadata.",
    ),
    JudgeDimension(
        "specificity",
        "Are the attributes specific to this particular item, not generic defaults? "
        "Penalize if design_details is vague, style_lineage is overly broad, or "
        "attributes could apply to any similar product.",
    ),
    JudgeDimension(
        "coherence",
        "Do the L1/L2/L3 attributes form a coherent, non-contradictory picture? "
        "For example, a sporty sneaker shouldn't have formal occasion, or a slim fit "
        "shouldn't have high visual weight.",
    ),
    JudgeDimension(
        "source_alignment",
        "Do the attributes match the source image and metadata (detail_desc, "
        "colour_group_name, product_type_name)? Check that material, color, and "
        "structural details are consistent with what is visible/described.",
    ),
    JudgeDimension(
        "informativeness",
        "Do the attributes provide sufficient useful information for recommendation? "
        "Are design_details rich and specific? Does the overall attribute set give "
        "enough signal to distinguish this item's appeal and match it to user preferences?",
    ),
]

# ---------------------------------------------------------------------------
# Configuration & Report Types
# ---------------------------------------------------------------------------


class FactualEvalConfig(NamedTuple):
    """Factual knowledge evaluation configuration."""

    judge_config: JudgeConfig = JudgeConfig()
    token_budget_limit: int = 512
    run_judge: bool = True


class FactualEvalReport(NamedTuple):
    """Complete factual knowledge evaluation report."""

    coverage: CoverageResult
    schema: SchemaCheckResult
    domain: DomainCheckResult
    distributions: DistributionResult
    token_budget: TokenBudgetResult
    judge: JudgeReport | None
    timestamp: str


# ---------------------------------------------------------------------------
# Factual Text Fields (for coverage + token budget)
# ---------------------------------------------------------------------------

_ALL_FACTUAL_FIELDS: list[str] = [
    "l1_material", "l1_closure", "l1_design_details", "l1_material_detail",
    "l2_style_mood", "l2_occasion", "l2_perceived_quality", "l2_trendiness",
    "l2_season_fit", "l2_target_impression", "l2_versatility",
    "l3_color_harmony", "l3_tone_season", "l3_coordination_role",
    "l3_visual_weight", "l3_style_lineage",
]

# ---------------------------------------------------------------------------
# Judge Message Builder
# ---------------------------------------------------------------------------


def build_factual_judge_message(
    article_meta: dict[str, Any],
    knowledge: dict[str, Any],
    image_b64: str | None = None,
) -> str | list[dict]:
    """Build the user message for factual knowledge judge evaluation.

    Args:
        article_meta: Article metadata (product_type_name, colour_group_name, detail_desc, etc.)
        knowledge: Extracted L1+L2+L3 attributes dict.
        image_b64: Optional base64-encoded product image.

    Returns:
        String message or multimodal content blocks (if image provided).
    """
    # Format knowledge attributes
    attr_lines = []
    for key, val in sorted(knowledge.items()):
        if key.startswith(("l1_", "l2_", "l3_")):
            attr_lines.append(f"  {key}: {val}")
    attrs_str = "\n".join(attr_lines)

    # Format metadata
    meta_lines = [
        f"Product Type: {article_meta.get('product_type_name', 'N/A')}",
        f"Colour Group: {article_meta.get('colour_group_name', 'N/A')}",
        f"Detail Description: {article_meta.get('detail_desc', 'N/A')}",
        f"Garment Group: {article_meta.get('garment_group_name', 'N/A')}",
    ]
    meta_str = "\n".join(meta_lines)

    text_content = (
        f"--- Source Metadata ---\n{meta_str}\n\n"
        f"--- Extracted Attributes ---\n{attrs_str}\n\n"
        "Evaluate the quality of these extracted attributes based on the source metadata"
        " and image (if provided)."
    )

    if image_b64:
        return [
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_b64}",
            },
            {"type": "input_text", "text": text_content},
        ]
    return text_content


# ---------------------------------------------------------------------------
# Judge Runner
# ---------------------------------------------------------------------------


async def run_factual_judge(
    knowledge_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    images_dir: Path | None = None,
    config: JudgeConfig = JudgeConfig(),
) -> JudgeReport:
    """Run LLM-as-Judge on a stratified sample of factual knowledge.

    Args:
        knowledge_df: Full factual knowledge DataFrame.
        articles_df: Articles metadata DataFrame.
        images_dir: Optional path to product images directory.
        config: Judge configuration.

    Returns:
        JudgeReport with per-dimension and overall scores.
    """
    from src.knowledge.factual.image_utils import get_image_for_article
    from src.knowledge.factual.prompts import resolve_super_category

    # Build article_id → metadata lookup
    meta_lookup: dict[str, dict] = {}
    for _, row in articles_df.iterrows():
        aid = str(row.get("article_id", ""))
        meta_lookup[aid] = row.to_dict()

    # Stratified sampling by super_category
    knowledge_with_cat = knowledge_df.copy()
    knowledge_with_cat["_super_cat"] = knowledge_with_cat["article_id"].apply(
        lambda aid: resolve_super_category(
            meta_lookup.get(str(aid), {}).get("garment_group_name", "Unknown"),
            meta_lookup.get(str(aid), {}).get("product_group_name"),
        )
    )

    sampled = (
        knowledge_with_cat
        .groupby("_super_cat", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), max(1, config.sample_size // 3)), random_state=42))
    )
    if len(sampled) > config.sample_size:
        sampled = sampled.sample(n=config.sample_size, random_state=42)

    # Build items list
    items: list[dict[str, Any]] = []
    for _, row in sampled.iterrows():
        aid = str(row.get("article_id", ""))
        meta = meta_lookup.get(aid, {})
        knowledge = row.to_dict()

        image_b64 = None
        if images_dir is not None:
            image_b64 = get_image_for_article(images_dir, aid)

        items.append({
            "item_id": aid,
            "article_meta": meta,
            "knowledge": knowledge,
            "image_b64": image_b64,
        })

    system_prompt = build_judge_system_prompt("factual_knowledge", FACTUAL_DIMENSIONS)

    def _build_msg(item: dict[str, Any]) -> str | list[dict]:
        return build_factual_judge_message(
            item["article_meta"], item["knowledge"], item.get("image_b64")
        )

    return await evaluate_batch(
        items=items,
        system_prompt=system_prompt,
        build_user_msg_fn=_build_msg,
        dimensions=FACTUAL_DIMENSIONS,
        config=config,
    )


# ---------------------------------------------------------------------------
# Full Evaluation Orchestration
# ---------------------------------------------------------------------------


async def run_factual_eval(
    knowledge_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    images_dir: Path | None = None,
    factual_text_col: str = "factual_text",
    config: FactualEvalConfig = FactualEvalConfig(),
) -> FactualEvalReport:
    """Run complete factual knowledge evaluation (structural + judge).

    Args:
        knowledge_df: Factual knowledge DataFrame.
        articles_df: Articles metadata DataFrame.
        images_dir: Optional product images directory.
        factual_text_col: Column containing composed factual text for token budget.
        config: Evaluation configuration.

    Returns:
        FactualEvalReport with all check results.
    """
    logger.info("Running factual knowledge evaluation on %d items...", len(knowledge_df))

    # 1. Coverage
    coverage = compute_coverage(knowledge_df, _ALL_FACTUAL_FIELDS)
    logger.info("Coverage: %.1f%% overall", coverage.overall_coverage * 100)

    # 2. Schema validation
    schema = run_schema_checks(knowledge_df, articles_df)
    logger.info("Schema: %d valid, %d invalid", schema.n_valid, schema.n_invalid)

    # 3. Domain consistency
    domain = run_domain_checks(knowledge_df, articles_df)
    logger.info(
        "Domain: %d items with violations (%d errors, %d warnings)",
        domain.n_items_with_violations, domain.n_error_violations, domain.n_warning_violations,
    )

    # 4. Distributions
    distributions = compute_distributions(knowledge_df)

    # 5. Token budget
    texts: list[str] = []
    if factual_text_col in knowledge_df.columns:
        texts = knowledge_df[factual_text_col].dropna().astype(str).tolist()
    token_budget = check_token_budget(texts, budget_limit=config.token_budget_limit)
    logger.info(
        "Token budget: mean=%.0f, p95=%.0f, %d over limit",
        token_budget.mean_tokens, token_budget.p95_tokens, token_budget.n_over_budget,
    )

    # 6. LLM-as-Judge (optional)
    judge_report = None
    if config.run_judge:
        logger.info("Running LLM-as-Judge on %d samples...", config.judge_config.sample_size)
        judge_report = await run_factual_judge(
            knowledge_df, articles_df, images_dir, config.judge_config
        )
        logger.info(
            "Judge: overall=%.2f, pass_rate=%.1f%%",
            judge_report.overall_mean, judge_report.pass_rate * 100,
        )

    return FactualEvalReport(
        coverage=coverage,
        schema=schema,
        domain=domain,
        distributions=distributions,
        token_budget=token_budget,
        judge=judge_report,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_factual_eval_sync(
    knowledge_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    images_dir: Path | None = None,
    factual_text_col: str = "factual_text",
    config: FactualEvalConfig = FactualEvalConfig(),
) -> FactualEvalReport:
    """Synchronous wrapper for run_factual_eval."""
    return asyncio.run(
        run_factual_eval(knowledge_df, articles_df, images_dir, factual_text_col, config)
    )
