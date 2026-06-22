"""Enrichment v2 extraction engine: product_code grouping → multimodal API call.

For each unique PRODUCT_CODE a single structured-output call to GPT-4.1-nano extracts
the LLM decision-axes (occasion / fit-intent / care / perceived price+trend) from the
product IMAGE + bare name + a metadata-stripped detail_desc — NEVER the categorical
metadata (the DE1 anti-recoding fix). Results propagate to every SKU and persist to a
resume-safe Parquet. The behavior-derived axes are computed elsewhere (no API).

No API side effects on import. ``estimate_e2_cost`` is pure; the async path only runs
when ``extract_e2_pilot`` is called.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import EnrichmentV2Config, EnrichmentV2Result
from src.knowledge.enrichment_v2.prompts import build_e2_messages, render_e2_row
from src.knowledge.enrichment_v2.schema import E2_LLM_AXES, E2_RESPONSE_FORMAT
from src.knowledge.enrichment_v2.validator import validate_e2
from src.knowledge.external.extractor import load_env_key  # reuse .env loader
from src.knowledge.factual.cache import ProductCodeCache
from src.knowledge.factual.image_utils import get_image_for_article

logger = logging.getLogger(__name__)

# Token costs (USD per 1M tokens) — GPT-4.1-nano real-time pricing.
_INPUT_COST_PER_1M = 0.10
_OUTPUT_COST_PER_1M = 0.40

# Heuristics: system+user text ≈ 320 in, + low-detail image ≈ 85, structured out ≈ 170.
_EST_INPUT_TOKENS_PER_ITEM = 400
_EST_OUTPUT_TOKENS_PER_ITEM = 170

# Columns read for representative selection (NO categorical metadata used downstream).
META_COLUMNS = ["article_id", "product_code", "prod_name", "detail_desc"]

OUTPUT_FILENAME = "enrichment_v2_llm.parquet"
CHECKPOINT_DIRNAME = "checkpoint_llm"
OUTPUT_COLUMNS = ["article_id", "product_code", *E2_LLM_AXES.keys()]


# ---------------------------------------------------------------------------
# Cost helpers (pure)
# ---------------------------------------------------------------------------
def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * _INPUT_COST_PER_1M
        + output_tokens / 1_000_000 * _OUTPUT_COST_PER_1M
    )


def estimate_e2_cost(n_product_codes: int) -> float:
    """A-priori USD cost estimate for multimodal extraction over ``n_product_codes``."""
    return _estimate_cost(
        n_product_codes * _EST_INPUT_TOKENS_PER_ITEM,
        n_product_codes * _EST_OUTPUT_TOKENS_PER_ITEM,
    )


# ---------------------------------------------------------------------------
# Product-code grouping (representative prefers an SKU that HAS an image)
# ---------------------------------------------------------------------------
def group_e2_representatives(articles: pd.DataFrame, images_dir: Path) -> dict[str, dict[str, Any]]:
    """Group articles by product_code; representative = image-having SKU with the
    longest detail_desc (falls back to longest detail_desc if no image on disk).

    Returns:
        {product_code: {"meta": <rep row dict>, "article_ids": [...],
                        "image_article_id": <str|None>}}.
    """
    cols = [c for c in META_COLUMNS if c in articles.columns]
    df = articles[cols].copy()
    df["article_id"] = df["article_id"].astype(str)
    df["product_code"] = df["product_code"].astype(str)
    df["_desc_len"] = df["detail_desc"].fillna("").astype(str).str.len()

    groups: dict[str, dict[str, Any]] = {}
    for product_code, g in df.groupby("product_code", sort=False):
        g_sorted = g.sort_values("_desc_len", ascending=False)
        image_aid = None
        rep_row = g_sorted.iloc[0]
        for aid in g_sorted["article_id"]:
            if get_image_for_article(images_dir, aid) is not None:
                image_aid = aid
                rep_row = g_sorted[g_sorted["article_id"] == aid].iloc[0]
                break
        groups[str(product_code)] = {
            "meta": rep_row.drop(labels="_desc_len").to_dict(),
            "article_ids": g["article_id"].tolist(),
            "image_article_id": image_aid,
        }
    logger.info("Grouped %d articles → %d product_codes", len(df), len(groups))
    return groups


# ---------------------------------------------------------------------------
# Propagation product_code → article rows
# ---------------------------------------------------------------------------
def propagate_e2_to_articles(
    groups: dict[str, dict[str, Any]], cache: ProductCodeCache
) -> pd.DataFrame:
    """Expand cached product_code rows to every article SKU."""
    rows: list[dict[str, Any]] = []
    for product_code, info in groups.items():
        row = cache.get(product_code)
        if row is None:
            continue
        for article_id in info["article_ids"]:
            rows.append({"article_id": article_id, "product_code": product_code, **row})
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Async extraction (single multimodal call per product_code)
# ---------------------------------------------------------------------------
async def _extract_one_e2(
    product_code: str,
    meta: dict[str, Any],
    image_b64: str | None,
    client: Any,  # openai.AsyncOpenAI
    config: EnrichmentV2Config,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, Any] | None, int, int, bool]:
    """Extract enrichment-v2 LLM axes for one product_code.

    Returns:
        (product_code, e2_row|None, input_tokens, output_tokens, had_image).
        On total failure returns (product_code, None, 0, 0, had_image).
    """
    import openai

    messages = build_e2_messages(meta, image_b64, config.detail_desc_chars)
    had_image = image_b64 is not None
    async with semaphore:
        for attempt in range(config.max_retries):
            try:
                response = await client.responses.create(
                    model=config.model,
                    input=messages,
                    text=E2_RESPONSE_FORMAT,
                    timeout=config.timeout_seconds,
                )
                parsed = json.loads(response.output_text)
                row = render_e2_row(parsed)
                vr = validate_e2(row)
                if vr.warnings:
                    logger.debug("%s warnings: %s", product_code, vr.warnings)
                if not vr.is_valid:
                    logger.warning("%s invalid: %s", product_code, vr.errors)
                usage = getattr(response, "usage", None)
                in_tok = int(getattr(usage, "input_tokens", 0) or 0)
                out_tok = int(getattr(usage, "output_tokens", 0) or 0)
                return product_code, row, in_tok, out_tok, had_image
            except (openai.RateLimitError, openai.APITimeoutError) as e:
                if attempt == config.max_retries - 1:
                    logger.warning("Giving up on %s: %s", product_code, e)
                    break
                await asyncio.sleep(2**attempt)
            except Exception as e:  # noqa: BLE001
                if attempt == config.max_retries - 1:
                    logger.warning("Failed on %s: %s", product_code, e)
                    break
                await asyncio.sleep(2**attempt)
    return product_code, None, 0, 0, had_image


def _load_images(
    pending: list[tuple[str, dict[str, Any]]],
    groups: dict[str, dict[str, Any]],
    images_dir: Path,
    image_max_size: int,
) -> dict[str, str | None]:
    """Pre-load base64 images for pending product_codes (None when absent on disk)."""
    out: dict[str, str | None] = {}
    for pc, _ in pending:
        aid = groups[pc].get("image_article_id")
        out[pc] = (
            get_image_for_article(images_dir, aid, max_size=image_max_size)
            if aid is not None
            else None
        )
    return out


async def extract_e2_pilot(
    articles: pd.DataFrame,
    images_dir: Path,
    output_dir: Path,
    config: EnrichmentV2Config,
) -> EnrichmentV2Result:
    """Run enrichment-v2 multimodal extraction over the (already sampled) articles.

    Args:
        articles: the pilot subset of articles (all SKUs of the sampled product_codes).
        images_dir: directory of H&M product images (images/{padded[:3]}/{padded}.jpg).
        output_dir: output directory for the parquet + checkpoint.
        config: extraction settings (model, concurrency, cost guard).

    Returns:
        EnrichmentV2Result summary.

    Raises:
        RuntimeError: if estimated cost exceeds ``config.max_cost_usd`` or no API key.

    NOTE: Calling this WILL spend money. It is never invoked on import.
    """
    import openai

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    cache = ProductCodeCache(checkpoint_dir=output_dir / CHECKPOINT_DIRNAME)
    n_cache_hits = cache.load_checkpoint()

    groups = group_e2_representatives(articles, images_dir)
    pending = [(pc, info["meta"]) for pc, info in groups.items() if cache.get(pc) is None]
    n_to_extract = len(pending)
    logger.info(
        "Enrichment v2: %d product_codes pending (%d cached, %d total)",
        n_to_extract,
        n_cache_hits,
        len(groups),
    )

    est_cost = estimate_e2_cost(n_to_extract)
    logger.info(
        "Estimated cost for %d product_codes: $%.4f (guard $%.2f)",
        n_to_extract,
        est_cost,
        config.max_cost_usd,
    )
    if est_cost > config.max_cost_usd:
        raise RuntimeError(
            f"Estimated cost ${est_cost:.4f} exceeds max_cost_usd "
            f"${config.max_cost_usd:.2f}. Raise --max-cost or shrink the sample."
        )

    if not pending:
        out_df = propagate_e2_to_articles(groups, cache)
        out_df.to_parquet(output_path, index=False)
        return _build_result(output_path, groups, out_df, 0, n_cache_hits, 0, 0, 0)

    load_env_key()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found (.env or environment).")

    images = _load_images(pending, groups, images_dir, config.image_max_size)
    n_imgs = sum(1 for v in images.values() if v is not None)
    logger.info("Pre-loaded images: %d/%d pending have an image on disk", n_imgs, n_to_extract)

    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(config.concurrency)
    total_in = total_out = n_calls = n_with_image = 0

    tasks = [
        _extract_one_e2(pc, meta, images[pc], client, config, semaphore) for pc, meta in pending
    ]
    start = time.monotonic()
    for coro in asyncio.as_completed(tasks):
        pc, row, in_tok, out_tok, had_image = await coro
        if row is not None:
            cache.put(pc, row)
        total_in += in_tok
        total_out += out_tok
        n_with_image += int(had_image)
        n_calls += 1
        if n_calls % config.checkpoint_interval == 0:
            cache.save_checkpoint()
            logger.info(
                "Checkpoint: %d/%d (cost $%.4f, %.1fs)",
                n_calls,
                n_to_extract,
                _estimate_cost(total_in, total_out),
                time.monotonic() - start,
            )

    cache.save_checkpoint()
    out_df = propagate_e2_to_articles(groups, cache)
    out_df.to_parquet(output_path, index=False)

    result = _build_result(
        output_path, groups, out_df, n_calls, n_cache_hits, n_with_image, total_in, total_out
    )
    with open(output_dir / "quality_report.json", "w") as f:
        json.dump(result._asdict(), f, indent=2, default=str)
    logger.info(
        "Enrichment v2 complete: %d codes, %d articles, %d calls (%d w/ image), $%.4f → %s",
        result.n_product_codes,
        result.n_articles,
        result.n_api_calls,
        result.n_with_image,
        result.total_cost_usd,
        output_path,
    )
    return result


def _build_result(
    output_path: Path,
    groups: dict[str, dict[str, Any]],
    out_df: pd.DataFrame,
    n_api_calls: int,
    n_cache_hits: int,
    n_with_image: int,
    total_in: int,
    total_out: int,
) -> EnrichmentV2Result:
    """Assemble the EnrichmentV2Result + coverage."""
    first_axis = next(iter(E2_LLM_AXES))
    coverage = float(out_df[first_axis].notna().mean()) if len(out_df) else 0.0
    return EnrichmentV2Result(
        output_path=output_path,
        n_product_codes=int(out_df["product_code"].nunique()) if len(out_df) else 0,
        n_articles=len(out_df),
        n_api_calls=n_api_calls,
        n_cache_hits=n_cache_hits,
        n_with_image=n_with_image,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cost_usd=_estimate_cost(total_in, total_out),
        coverage=coverage,
    )
