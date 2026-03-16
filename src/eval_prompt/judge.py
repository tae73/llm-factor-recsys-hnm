"""LLM-as-Judge common protocol for prompt output evaluation.

Provides a unified 5-dimension scoring framework (1-5 scale) shared by
both factual knowledge and user profile evaluation domains.

Dimensions:
    accuracy, specificity, coherence, source_alignment, informativeness
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, NamedTuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimension Names (shared across domains)
# ---------------------------------------------------------------------------

DIMENSION_NAMES: list[str] = [
    "accuracy",
    "specificity",
    "coherence",
    "source_alignment",
    "informativeness",
]

# ---------------------------------------------------------------------------
# NamedTuples
# ---------------------------------------------------------------------------


class JudgeConfig(NamedTuple):
    """LLM-as-Judge configuration."""

    model: str = "gpt-4.1-mini"
    sample_size: int = 50
    max_concurrent: int = 10
    temperature: float = 0.0
    pass_threshold: float = 3.5


class JudgeDimension(NamedTuple):
    """Single evaluation dimension with domain-specific description."""

    name: str  # One of DIMENSION_NAMES
    description: str  # Domain-specific rubric


class JudgeResult(NamedTuple):
    """Evaluation result for a single item."""

    item_id: str
    scores: dict[str, int]  # dimension → 1-5
    justifications: dict[str, str]  # dimension → 1-sentence rationale
    overall_score: float


class JudgeReport(NamedTuple):
    """Aggregated judge evaluation report."""

    results: list[JudgeResult]
    per_dimension_mean: dict[str, float]
    overall_mean: float
    n_evaluated: int
    n_passed: int
    pass_rate: float


# ---------------------------------------------------------------------------
# JSON Schema Builder
# ---------------------------------------------------------------------------


def build_judge_schema(dimensions: list[JudgeDimension]) -> dict:
    """Build JSON schema for structured judge output.

    Each dimension produces a score (1-5) and a justification (1 sentence).

    Args:
        dimensions: List of JudgeDimension to evaluate.

    Returns:
        JSON schema dict for OpenAI structured output.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for dim in dimensions:
        score_key = f"{dim.name}_score"
        justification_key = f"{dim.name}_justification"

        properties[score_key] = {
            "type": "integer",
            "description": f"Score for {dim.name} (1=very poor, 5=excellent). {dim.description}",
        }
        properties[justification_key] = {
            "type": "string",
            "description": f"One-sentence justification for the {dim.name} score.",
        }
        required.extend([score_key, justification_key])

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Single Item Evaluation
# ---------------------------------------------------------------------------


async def _evaluate_single(
    client: AsyncOpenAI,
    item_id: str,
    system_prompt: str,
    user_message: str | list[dict],
    dimensions: list[JudgeDimension],
    schema: dict,
    config: JudgeConfig,
) -> JudgeResult | None:
    """Evaluate a single item using LLM-as-Judge.

    Args:
        client: AsyncOpenAI client.
        item_id: Identifier for the evaluated item.
        system_prompt: Judge system prompt with rubric.
        user_message: User message (str or multimodal content blocks).
        dimensions: List of dimensions to evaluate.
        schema: JSON schema for structured output.
        config: Judge configuration.

    Returns:
        JudgeResult or None on failure.
    """
    input_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    if isinstance(user_message, str):
        input_messages.append({"role": "user", "content": user_message})
    else:
        input_messages.append({"role": "user", "content": user_message})

    try:
        response = await client.responses.create(
            model=config.model,
            input=input_messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "judge_evaluation",
                    "schema": schema,
                    "strict": True,
                }
            },
            temperature=config.temperature,
        )

        raw = json.loads(response.output_text)

        scores: dict[str, int] = {}
        justifications: dict[str, str] = {}
        for dim in dimensions:
            score = raw.get(f"{dim.name}_score", 0)
            scores[dim.name] = max(1, min(5, int(score)))
            justifications[dim.name] = raw.get(f"{dim.name}_justification", "")

        overall = sum(scores.values()) / len(scores) if scores else 0.0

        return JudgeResult(
            item_id=item_id,
            scores=scores,
            justifications=justifications,
            overall_score=overall,
        )

    except Exception:
        logger.exception("Judge evaluation failed for item %s", item_id)
        return None


# ---------------------------------------------------------------------------
# Batch Evaluation
# ---------------------------------------------------------------------------


async def evaluate_batch(
    items: list[dict[str, Any]],
    system_prompt: str,
    build_user_msg_fn: Callable[[dict[str, Any]], str | list[dict]],
    dimensions: list[JudgeDimension],
    config: JudgeConfig = JudgeConfig(),
    id_field: str = "item_id",
) -> JudgeReport:
    """Evaluate a batch of items using LLM-as-Judge with concurrency control.

    Args:
        items: List of item dicts to evaluate.
        system_prompt: Judge system prompt with rubric.
        build_user_msg_fn: Function mapping item dict → user message.
        dimensions: List of JudgeDimension to evaluate.
        config: Judge configuration.
        id_field: Key in item dict for the item identifier.

    Returns:
        JudgeReport with aggregated results.
    """
    client = AsyncOpenAI()
    schema = build_judge_schema(dimensions)
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def _eval_with_semaphore(item: dict[str, Any]) -> JudgeResult | None:
        async with semaphore:
            item_id = str(item.get(id_field, "unknown"))
            user_msg = build_user_msg_fn(item)
            return await _evaluate_single(
                client, item_id, system_prompt, user_msg, dimensions, schema, config
            )

    tasks = [_eval_with_semaphore(item) for item in items]
    raw_results = await asyncio.gather(*tasks)

    results = [r for r in raw_results if r is not None]
    n_evaluated = len(results)

    if n_evaluated == 0:
        return JudgeReport(
            results=[],
            per_dimension_mean={d.name: 0.0 for d in dimensions},
            overall_mean=0.0,
            n_evaluated=0,
            n_passed=0,
            pass_rate=0.0,
        )

    per_dim_mean: dict[str, float] = {}
    for dim in dimensions:
        dim_scores = [r.scores[dim.name] for r in results]
        per_dim_mean[dim.name] = sum(dim_scores) / len(dim_scores)

    overall_mean = sum(r.overall_score for r in results) / n_evaluated
    n_passed = sum(1 for r in results if r.overall_score >= config.pass_threshold)

    return JudgeReport(
        results=results,
        per_dimension_mean=per_dim_mean,
        overall_mean=overall_mean,
        n_evaluated=n_evaluated,
        n_passed=n_passed,
        pass_rate=n_passed / n_evaluated,
    )


# ---------------------------------------------------------------------------
# Judge System Prompt Builder
# ---------------------------------------------------------------------------


def build_judge_system_prompt(domain: str, dimensions: list[JudgeDimension]) -> str:
    """Build a system prompt for the LLM judge.

    Args:
        domain: "factual_knowledge" or "user_profile".
        dimensions: List of dimensions with descriptions.

    Returns:
        System prompt string.
    """
    dim_rubric = "\n".join(
        f"- **{d.name}** (1-5): {d.description}" for d in dimensions
    )

    return f"""You are an expert evaluator for {domain} quality in a fashion recommendation system.

Score the provided output on each dimension using a 1-5 scale:
  1 = Very poor: fundamentally incorrect or useless
  2 = Poor: significant issues, limited value
  3 = Acceptable: mostly correct but with notable gaps
  4 = Good: minor issues, generally useful
  5 = Excellent: accurate, specific, and highly useful

**Dimensions:**
{dim_rubric}

**Rules:**
- Score each dimension independently.
- Provide a concise one-sentence justification for each score.
- Be critical but fair — reward specificity and penalize generic/boilerplate content.
- Consider the source data provided when evaluating accuracy and alignment."""
