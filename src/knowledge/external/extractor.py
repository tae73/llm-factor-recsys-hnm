"""External-knowledge extraction engine: product_code grouping → combined API call.

For each unique PRODUCT_CODE (not article), a single structured-output call to
GPT-4.1-nano extracts BOTH a prose complement description and structured
complement attributes (see ``prompts.py``). Results are propagated to every
article SKU sharing the product_code and persisted to a resume-safe Parquet.

Default has NO API side effects on import. The async path only runs when
``extract_external_knowledge`` is called; ``estimate_external_cost`` is pure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import ExternalExtractionConfig, ExternalExtractionResult
from src.knowledge.external.prompts import (
    RESPONSE_FORMAT,
    build_messages,
    render_external_text,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token costs (USD per 1M tokens) — GPT-4.1-nano real-time pricing
# ---------------------------------------------------------------------------
_INPUT_COST_PER_1M = 0.10  # $0.10 / 1M input tokens
_OUTPUT_COST_PER_1M = 0.40  # $0.40 / 1M output tokens

# Combined-call token heuristics (system+user ≈ 240 in, prose+structured ≈ 220 out).
_EST_INPUT_TOKENS_PER_ITEM = 240
_EST_OUTPUT_TOKENS_PER_ITEM = 220

# Representative metadata columns pulled from articles.parquet.
META_COLUMNS = [
    "article_id",
    "product_code",
    "product_type_name",
    "colour_group_name",
    "section_name",
    "product_group_name",
    "detail_desc",
]

OUTPUT_FILENAME = "external_knowledge_full.parquet"
CHECKPOINT_FILENAME = "external_checkpoint.parquet"
OUTPUT_COLUMNS = ["article_id", "product_code", "prose_text", "structured_text"]


# ---------------------------------------------------------------------------
# Env / cost helpers (pure)
# ---------------------------------------------------------------------------


def load_env_key() -> None:
    """Load OPENAI_API_KEY from .env into os.environ if not already set (probe_16 parity)."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = (
                    line.split("=", 1)[1].strip().strip('"').strip("'")
                )
                return


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token counts (real-time pricing, no batch discount)."""
    return (
        input_tokens / 1_000_000 * _INPUT_COST_PER_1M
        + output_tokens / 1_000_000 * _OUTPUT_COST_PER_1M
    )


def estimate_external_cost(n_product_codes: int) -> float:
    """A-priori USD cost estimate for combined extraction over ``n_product_codes``."""
    return _estimate_cost(
        n_product_codes * _EST_INPUT_TOKENS_PER_ITEM,
        n_product_codes * _EST_OUTPUT_TOKENS_PER_ITEM,
    )


# ---------------------------------------------------------------------------
# Product-code grouping (representative = first article per product_code)
# ---------------------------------------------------------------------------


def group_representatives(articles: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Group articles by product_code, picking the FIRST article as representative.

    Returns:
        {product_code: {"meta": <representative row dict>,
                        "article_ids": [<all SKU article_ids as str>]}}.
    """
    cols = [c for c in META_COLUMNS if c in articles.columns]
    df = articles[cols].copy()
    df["article_id"] = df["article_id"].astype(str)
    df["product_code"] = df["product_code"].astype(str)

    groups: dict[str, dict[str, Any]] = {}
    for product_code, group_df in df.groupby("product_code", sort=False):
        representative = group_df.iloc[0].to_dict()
        groups[str(product_code)] = {
            "meta": representative,
            "article_ids": group_df["article_id"].tolist(),
        }

    logger.info(
        "Grouped %d articles → %d product_codes", len(df), len(groups)
    )
    return groups


# ---------------------------------------------------------------------------
# Resume-safe checkpoint (product_code-level)
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path: Path) -> dict[str, dict[str, str]]:
    """Load product_code → {prose_text, structured_text} from a Parquet checkpoint."""
    if not checkpoint_path.exists():
        return {}
    df = pd.read_parquet(checkpoint_path)
    out: dict[str, dict[str, str]] = {}
    for row in df.itertuples(index=False):
        out[str(row.product_code)] = {
            "prose_text": "" if pd.isna(row.prose_text) else str(row.prose_text),
            "structured_text": (
                "" if pd.isna(row.structured_text) else str(row.structured_text)
            ),
        }
    logger.info("Loaded checkpoint: %d product_codes from %s", len(out), checkpoint_path)
    return out


def save_checkpoint(
    checkpoint_path: Path, results: dict[str, dict[str, str]]
) -> None:
    """Persist product_code-level results to a Parquet checkpoint."""
    if not results:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "product_code": pc,
                "prose_text": r.get("prose_text", ""),
                "structured_text": r.get("structured_text", ""),
            }
            for pc, r in results.items()
        ]
    )
    df.to_parquet(checkpoint_path, index=False)
    logger.info("Saved checkpoint: %d product_codes → %s", len(df), checkpoint_path)


# ---------------------------------------------------------------------------
# Propagation product_code → article rows
# ---------------------------------------------------------------------------


def propagate_to_articles(
    groups: dict[str, dict[str, Any]],
    results: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Expand product_code results to every article SKU (resume-safe output frame)."""
    rows: list[dict[str, str]] = []
    for product_code, info in groups.items():
        res = results.get(product_code)
        if res is None:
            continue
        prose_text = res.get("prose_text", "")
        structured_text = res.get("structured_text", "")
        for article_id in info["article_ids"]:
            rows.append(
                {
                    "article_id": article_id,
                    "product_code": product_code,
                    "prose_text": prose_text,
                    "structured_text": structured_text,
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Async extraction (combined prose + structured, single call per product_code)
# ---------------------------------------------------------------------------


async def _extract_one(
    product_code: str,
    meta: dict[str, Any],
    client: "Any",  # openai.AsyncOpenAI
    config: ExternalExtractionConfig,
    semaphore: "asyncio.Semaphore",
) -> tuple[str, dict[str, str], int, int]:
    """Extract combined external knowledge for one product_code.

    Returns:
        (product_code, {prose_text, structured_text}, input_tokens, output_tokens).
        On total failure, returns empty texts and 0 tokens.
    """
    import openai

    messages = build_messages(meta, config.detail_desc_chars)
    async with semaphore:
        for attempt in range(config.max_retries):
            try:
                response = await client.responses.create(
                    model=config.model,
                    input=messages,
                    text=RESPONSE_FORMAT,
                    timeout=config.timeout_seconds,
                )
                parsed = json.loads(response.output_text)
                texts = render_external_text(parsed)
                usage = getattr(response, "usage", None)
                in_tok = int(getattr(usage, "input_tokens", 0) or 0)
                out_tok = int(getattr(usage, "output_tokens", 0) or 0)
                return product_code, texts, in_tok, out_tok
            except (openai.RateLimitError, openai.APITimeoutError) as e:
                if attempt == config.max_retries - 1:
                    logger.warning("Giving up on %s after retries: %s", product_code, e)
                    break
                await asyncio.sleep(2**attempt)
            except Exception as e:  # noqa: BLE001
                if attempt == config.max_retries - 1:
                    logger.warning("Failed on %s: %s", product_code, e)
                    break
                await asyncio.sleep(2**attempt)
    return product_code, {"prose_text": "", "structured_text": ""}, 0, 0


async def extract_external_knowledge(
    articles: pd.DataFrame,
    output_dir: Path,
    config: ExternalExtractionConfig,
    pilot: Optional[int] = None,
    limit: Optional[int] = None,
) -> ExternalExtractionResult:
    """Run combined external-knowledge extraction over all product_codes.

    Args:
        articles: Full articles DataFrame (must contain ``META_COLUMNS``).
        output_dir: Directory for output parquet + checkpoint.
        config: External extraction settings (model, concurrency, cost guard).
        pilot: If set, only extract the FIRST N product_codes (after cache).
        limit: Alias for an upper bound on product_codes to process this run.

    Returns:
        ExternalExtractionResult summary.

    Raises:
        RuntimeError: If estimated cost exceeds ``config.max_cost_usd``.

    NOTE: Calling this WILL spend money. It is never invoked on import.
    """
    import openai

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    output_path = output_dir / OUTPUT_FILENAME

    groups = group_representatives(articles)

    # Resume: skip already-extracted product_codes.
    results = load_checkpoint(checkpoint_path)
    n_cache_hits = len(results)
    pending = [(pc, info) for pc, info in groups.items() if pc not in results]

    # pilot / limit truncation (apply to the pending set).
    cap = pilot if pilot is not None else limit
    if cap is not None:
        pending = pending[:cap]

    n_to_extract = len(pending)
    logger.info(
        "External extraction: %d product_codes pending (%d cached, %d total)",
        n_to_extract,
        n_cache_hits,
        len(groups),
    )

    # Cost guard BEFORE any API call.
    est_cost = estimate_external_cost(n_to_extract)
    logger.info(
        "Estimated cost for %d product_codes: $%.4f (guard: $%.2f)",
        n_to_extract,
        est_cost,
        config.max_cost_usd,
    )
    if est_cost > config.max_cost_usd:
        raise RuntimeError(
            f"Estimated cost ${est_cost:.4f} exceeds max_cost_usd "
            f"${config.max_cost_usd:.2f}. Raise --max-cost or use --pilot/--limit."
        )

    if not pending:
        logger.info("Nothing to extract; writing propagated output from cache.")
        out_df = propagate_to_articles(groups, results)
        out_df.to_parquet(output_path, index=False)
        return _build_result(
            output_path, groups, results, out_df, 0, n_cache_hits, 0, 0
        )

    load_env_key()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found (.env or environment).")

    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(config.concurrency)

    total_input_tokens = 0
    total_output_tokens = 0
    n_api_calls = 0

    tasks = [
        _extract_one(pc, info["meta"], client, config, semaphore)
        for pc, info in pending
    ]
    start = time.monotonic()
    for coro in asyncio.as_completed(tasks):
        product_code, texts, in_tok, out_tok = await coro
        results[product_code] = texts
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        n_api_calls += 1
        if n_api_calls % config.checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, results)
            logger.info(
                "Checkpoint: %d/%d extracted (cost so far $%.4f, %.1fs)",
                n_api_calls,
                n_to_extract,
                _estimate_cost(total_input_tokens, total_output_tokens),
                time.monotonic() - start,
            )

    save_checkpoint(checkpoint_path, results)

    # Propagate to article level and persist.
    out_df = propagate_to_articles(groups, results)
    out_df.to_parquet(output_path, index=False)

    result = _build_result(
        output_path,
        groups,
        results,
        out_df,
        n_api_calls,
        n_cache_hits,
        total_input_tokens,
        total_output_tokens,
    )

    # Quality report.
    quality_report = {
        "n_product_codes": result.n_product_codes,
        "n_articles": result.n_articles,
        "n_api_calls": result.n_api_calls,
        "n_cache_hits": result.n_cache_hits,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "total_cost_usd": result.total_cost_usd,
        "coverage_prose": result.coverage_prose,
        "coverage_structured": result.coverage_structured,
    }
    with open(output_dir / "quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)

    logger.info(
        "External extraction complete: %d product_codes, %d articles, "
        "%d API calls, $%.4f → %s",
        result.n_product_codes,
        result.n_articles,
        result.n_api_calls,
        result.total_cost_usd,
        output_path,
    )
    return result


def _build_result(
    output_path: Path,
    groups: dict[str, dict[str, Any]],
    results: dict[str, dict[str, str]],
    out_df: pd.DataFrame,
    n_api_calls: int,
    n_cache_hits: int,
    total_input_tokens: int,
    total_output_tokens: int,
) -> ExternalExtractionResult:
    """Assemble the ExternalExtractionResult + coverage stats."""
    n_codes = len(results)
    coverage_prose = (
        float(out_df["prose_text"].str.len().gt(0).mean()) if len(out_df) else 0.0
    )
    coverage_structured = (
        float(out_df["structured_text"].str.len().gt(0).mean()) if len(out_df) else 0.0
    )
    return ExternalExtractionResult(
        output_path=output_path,
        n_product_codes=n_codes,
        n_articles=len(out_df),
        n_api_calls=n_api_calls,
        n_cache_hits=n_cache_hits,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=_estimate_cost(total_input_tokens, total_output_tokens),
        coverage_prose=coverage_prose,
        coverage_structured=coverage_structured,
    )
