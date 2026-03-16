"""Core extraction engine: product_code grouping → API calls → result propagation.

Supports both real-time API (pilot, asyncio concurrent) and Batch API (full extraction).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from collections import deque
from pathlib import Path

import openai
import pandas as pd

from src.config import ExtractionConfig, ExtractionResult
from src.knowledge.factual.cache import ProductCodeCache
from src.knowledge.factual.image_utils import get_image_for_article
from src.knowledge.factual.prompts import (
    build_user_message,
    get_prompt_and_schema,
    map_to_canonical_slots,
    resolve_super_category,
)
from src.knowledge.factual.text_composer import construct_factual_text
from src.knowledge.factual.validator import validate_domain_consistency, validate_knowledge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token costs (USD per 1M tokens) — GPT-4.1-nano pricing
# ---------------------------------------------------------------------------
_INPUT_COST_PER_1M = 0.10  # $0.10/1M input tokens
_OUTPUT_COST_PER_1M = 0.40  # $0.40/1M output tokens
_BATCH_DISCOUNT = 0.5  # 50% discount for Batch API


def _estimate_cost(
    input_tokens: int, output_tokens: int, is_batch: bool = False
) -> float:
    """Estimate USD cost from token counts."""
    discount = _BATCH_DISCOUNT if is_batch else 1.0
    return (
        input_tokens / 1_000_000 * _INPUT_COST_PER_1M * discount
        + output_tokens / 1_000_000 * _OUTPUT_COST_PER_1M * discount
    )


# ---------------------------------------------------------------------------
# Token-Aware Rate Limiter
# ---------------------------------------------------------------------------


class TokenRateLimiter:
    """Sliding-window TPM rate limiter (pure asyncio, no external deps).

    Tracks token consumption over a rolling window and proactively delays
    requests when the budget would be exceeded.  Prevents thundering-herd
    retries by allowing callers to set a global backpressure pause.
    """

    def __init__(self, tpm_limit: int, window_seconds: float = 60.0):
        self._budget = tpm_limit
        self._window = window_seconds
        self._log: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._total_tokens = 0
        self._total_requests = 0
        self._backpressure_until = 0.0
        self._lock = asyncio.Lock()

    @property
    def _avg_tokens(self) -> int:
        """Running average tokens per request (conservative 2500 default)."""
        if self._total_requests == 0:
            return 2500
        return self._total_tokens // self._total_requests

    def _prune(self, now: float) -> int:
        """Remove expired entries and return current window usage."""
        cutoff = now - self._window
        while self._log and self._log[0][0] < cutoff:
            self._log.popleft()
        return sum(t for _, t in self._log)

    async def acquire(self) -> None:
        """Wait until there is enough token budget for one request."""
        async with self._lock:
            now = time.monotonic()
            # Honour global backpressure (e.g. from a 429)
            if now < self._backpressure_until:
                await asyncio.sleep(self._backpressure_until - now)
                now = time.monotonic()
            used = self._prune(now)
            estimated = self._avg_tokens
            if used + estimated > self._budget:
                # Sleep until enough tokens expire from the window
                needed = used + estimated - self._budget
                cumul = 0
                sleep_until = now + self._window
                for ts, tok in self._log:
                    cumul += tok
                    if cumul >= needed:
                        sleep_until = ts + self._window
                        break
                await asyncio.sleep(max(0.0, sleep_until - now) + 0.1)

    def record(self, tokens: int) -> None:
        """Record actual token consumption (lock-free, asyncio single-thread)."""
        self._log.append((time.monotonic(), tokens))
        self._total_tokens += tokens
        self._total_requests += 1

    def backpressure(self, seconds: float) -> None:
        """Set a global pause — all pending acquire() calls will wait."""
        self._backpressure_until = max(
            self._backpressure_until, time.monotonic() + seconds
        )


# ---------------------------------------------------------------------------
# Product Code Grouping
# ---------------------------------------------------------------------------


def group_by_product_code(
    articles: pd.DataFrame,
    images_dir: Path,
) -> dict[str, dict]:
    """Group articles by product_code and select representative SKU.

    Representative = longest detail_desc among SKUs with an image available.
    Falls back to longest detail_desc if no image found for any SKU.

    Returns:
        {product_code: {"representative": row_dict, "variants": DataFrame, "super_category": str}}
    """
    # Add super_category via two-level routing
    articles = articles.assign(
        super_category=lambda df: df.apply(
            lambda row: resolve_super_category(
                row["garment_group_name"],
                row.get("product_group_name"),
            ),
            axis=1,
        )
    )

    groups: dict[str, dict] = {}
    for product_code, group_df in articles.groupby("product_code"):
        super_category = group_df["super_category"].iloc[0]

        # Score each SKU: prefer longer desc + image exists
        scored = group_df.assign(
            _desc_len=lambda df: df["detail_desc"].fillna("").str.len(),
            _has_image=lambda df: df["article_id"].apply(
                lambda aid: (images_dir / str(aid).zfill(10)[:3] / f"{str(aid).zfill(10)}.jpg").exists()
            ),
            _score=lambda df: df["_desc_len"] + df["_has_image"].astype(int) * 10000,
        ).sort_values("_score", ascending=False)

        representative = scored.iloc[0].to_dict()
        groups[str(product_code)] = {
            "representative": representative,
            "variants": group_df,
            "super_category": super_category,
        }

    logger.info(
        "Grouped %d articles → %d product_codes", len(articles), len(groups)
    )
    return groups


# ---------------------------------------------------------------------------
# Color-Dependent Attribute Update
# ---------------------------------------------------------------------------

# colour_group_name → tone_season mapping (all 50 H&M colour_group_names → 6 tones)
COLOR_TO_TONE: dict[str, str] = {
    "Black": "Cool-Winter",
    "White": "Neutral-Cool",
    "Off White": "Warm-Autumn",
    "Light Beige": "Warm-Spring",
    "Beige": "Warm-Autumn",
    "Dark Beige": "Warm-Autumn",
    "Greyish Beige": "Neutral-Warm",
    "Grey": "Neutral-Cool",
    "Dark Grey": "Cool-Winter",
    "Light Grey": "Cool-Summer",
    "Blue": "Cool-Summer",
    "Dark Blue": "Cool-Winter",
    "Light Blue": "Cool-Summer",
    "Other Blue": "Cool-Summer",
    "Red": "Warm-Autumn",
    "Dark Red": "Warm-Autumn",
    "Light Red": "Warm-Spring",
    "Other Red": "Warm-Autumn",
    "Pink": "Cool-Summer",
    "Light Pink": "Warm-Spring",
    "Dark Pink": "Warm-Autumn",
    "Other Pink": "Cool-Summer",
    "Green": "Warm-Autumn",
    "Dark Green": "Cool-Winter",
    "Light Green": "Warm-Spring",
    "Other Green": "Warm-Autumn",
    "Yellowish Green": "Warm-Spring",
    "Greenish Khaki": "Neutral-Warm",
    "Yellow": "Warm-Spring",
    "Dark Yellow": "Warm-Autumn",
    "Light Yellow": "Warm-Spring",
    "Other Yellow": "Warm-Spring",
    "Orange": "Warm-Autumn",
    "Dark Orange": "Warm-Autumn",
    "Light Orange": "Warm-Spring",
    "Other Orange": "Warm-Autumn",
    "Brown": "Warm-Autumn",
    "Yellowish Brown": "Warm-Autumn",
    "Khaki": "Neutral-Warm",
    "Mole": "Neutral-Warm",
    "Turquoise": "Cool-Summer",
    "Light Turquoise": "Cool-Summer",
    "Dark Turquoise": "Cool-Winter",
    "Other Turquoise": "Cool-Summer",
    "Purple": "Cool-Summer",
    "Light Purple": "Cool-Summer",
    "Dark Purple": "Cool-Winter",
    "Other Purple": "Cool-Summer",
    "Lilac Purple": "Cool-Summer",
    "Bronze/Copper": "Warm-Autumn",
    "Silver": "Cool-Winter",
    "Gold": "Warm-Autumn",
    "Transparent": "Neutral-Cool",
    "Undefined": "Neutral-Cool",
    "Other": "Neutral-Cool",
    "Unknown": "Neutral-Cool",
}

# colour_group_name → color_harmony mapping
COLOR_TO_HARMONY: dict[str, str] = {
    "Black": "Monochromatic",
    "White": "Monochromatic",
    "Off White": "Neutral",
    "Light Beige": "Neutral",
    "Beige": "Neutral",
    "Dark Beige": "Neutral",
    "Greyish Beige": "Neutral",
    "Grey": "Neutral",
    "Dark Grey": "Monochromatic",
    "Light Grey": "Neutral",
    "Blue": "Analogous",
    "Dark Blue": "Analogous",
    "Light Blue": "Analogous",
    "Other Blue": "Analogous",
    "Red": "Complementary",
    "Dark Red": "Complementary",
    "Light Red": "Complementary",
    "Other Red": "Complementary",
    "Pink": "Analogous",
    "Light Pink": "Pastel",
    "Dark Pink": "Analogous",
    "Other Pink": "Analogous",
    "Green": "Analogous",
    "Dark Green": "Analogous",
    "Light Green": "Pastel",
    "Other Green": "Analogous",
    "Yellowish Green": "Analogous",
    "Greenish Khaki": "Earth-tone",
    "Yellow": "Complementary",
    "Dark Yellow": "Earth-tone",
    "Light Yellow": "Complementary",
    "Other Yellow": "Complementary",
    "Orange": "Complementary",
    "Dark Orange": "Complementary",
    "Light Orange": "Complementary",
    "Other Orange": "Complementary",
    "Brown": "Earth-tone",
    "Yellowish Brown": "Earth-tone",
    "Khaki": "Earth-tone",
    "Mole": "Earth-tone",
    "Turquoise": "Triadic",
    "Light Turquoise": "Analogous",
    "Dark Turquoise": "Analogous",
    "Other Turquoise": "Analogous",
    "Purple": "Triadic",
    "Light Purple": "Triadic",
    "Dark Purple": "Triadic",
    "Other Purple": "Triadic",
    "Lilac Purple": "Triadic",
    "Bronze/Copper": "Jewel-tone",
    "Silver": "Neutral",
    "Gold": "Jewel-tone",
    "Transparent": "Neutral",
    "Undefined": "Neutral",
    "Other": "Neutral",
    "Unknown": "Neutral",
}


# ---------------------------------------------------------------------------
# Visual Weight Rule-Based Correction
# ---------------------------------------------------------------------------

# silhouette → (min_weight, max_weight)
SILHOUETTE_WEIGHT_RANGE: dict[str, tuple[int, int]] = {
    "I-line": (1, 3),
    "H-line": (2, 4),
    "X-line": (2, 4),
    "A-line": (2, 4),
    "V-line": (2, 4),
    "O-line": (3, 5),
    "Y-line": (2, 4),
    "Trapeze": (3, 5),
    "Cocoon": (3, 5),
    "Empire": (2, 4),
}

# fit → (min_weight, max_weight)
FIT_WEIGHT_RANGE: dict[str, tuple[int, int]] = {
    "Slim": (1, 3),
    "Skinny": (1, 3),
    "Bodycon": (1, 3),
    "Tailored": (2, 3),
    "Regular": (2, 4),
    "Relaxed": (2, 4),
    "Wide": (3, 5),
    "Loose": (3, 5),
    "Boxy": (3, 5),
    "Oversized": (3, 5),
}

# coordination_role → (min_weight, max_weight)
COORDINATION_WEIGHT_RANGE: dict[str, tuple[int, int]] = {
    "Basic": (1, 3),
    "Foundation": (1, 3),
    "Finishing": (1, 3),
    "Layering": (2, 4),
    "Accent": (2, 4),
    "Statement": (3, 5),
}


def correct_visual_weight(knowledge: dict, super_category: str) -> dict:
    """Rule-based visual_weight correction. Clamp to intersection of silhouette/fit/coordination ranges.

    Applied BEFORE map_to_canonical_slots(), so uses original field names.
    Only applies to Apparel (Footwear/Accessories lack silhouette/fit fields).
    """
    if super_category != "Apparel":
        return knowledge

    weight = knowledge.get("l3_visual_weight")
    if not isinstance(weight, int):
        return knowledge

    ranges: list[tuple[int, int]] = []

    sil = knowledge.get("l3_silhouette")
    if sil and sil in SILHOUETTE_WEIGHT_RANGE:
        ranges.append(SILHOUETTE_WEIGHT_RANGE[sil])

    fit = knowledge.get("l1_fit")
    if fit and fit in FIT_WEIGHT_RANGE:
        ranges.append(FIT_WEIGHT_RANGE[fit])

    coord = knowledge.get("l3_coordination_role")
    if coord and coord in COORDINATION_WEIGHT_RANGE:
        ranges.append(COORDINATION_WEIGHT_RANGE[coord])

    if not ranges:
        return knowledge

    lo = max(r[0] for r in ranges)
    hi = min(r[1] for r in ranges)
    if lo > hi:
        # Intersection empty → use silhouette range (first priority)
        lo, hi = ranges[0]

    clamped = max(lo, min(hi, weight))
    if clamped != weight:
        result = dict(knowledge)
        result["l3_visual_weight"] = clamped
        return result
    return knowledge


def update_color_knowledge(
    base_knowledge: dict,
    colour_group: str,
) -> dict:
    """Update color-dependent L3 fields based on variant's colour_group_name."""
    result = dict(base_knowledge)
    if colour_group and str(colour_group) != "nan":
        result["l3_color_harmony"] = COLOR_TO_HARMONY.get(colour_group, "Neutral")
        result["l3_tone_season"] = COLOR_TO_TONE.get(colour_group, "Neutral-Cool")
    return result


def propagate_to_variants(
    product_knowledge: dict,
    variant_articles: pd.DataFrame,
    representative_article_id: str,
) -> list[dict]:
    """Propagate product_code knowledge to all SKU variants.

    Copies all attributes from representative, then updates color-dependent
    L3 fields (color_harmony, tone_season) based on each variant's colour_group_name.
    """
    rows: list[dict] = []
    for _, variant in variant_articles.iterrows():
        article_id = str(variant["article_id"])
        knowledge = update_color_knowledge(
            product_knowledge, variant.get("colour_group_name", "")
        )
        knowledge["article_id"] = article_id
        knowledge["is_representative"] = article_id == representative_article_id
        rows.append(knowledge)
    return rows


# ---------------------------------------------------------------------------
# Real-time API Extraction (Pilot)
# ---------------------------------------------------------------------------


async def _extract_single_product(
    representative: dict,
    super_category: str,
    image_base64: str | None,
    client: openai.AsyncOpenAI,
    config: ExtractionConfig,
    log_file: Path | None = None,
    rate_limiter: TokenRateLimiter | None = None,
) -> tuple[dict, int, int]:
    """Extract L1+L2+L3 for a single product via real-time API.

    Returns:
        (knowledge_dict, input_tokens, output_tokens)
    """
    system_prompt, json_schema = get_prompt_and_schema(super_category)
    user_content = build_user_message(representative, representative.get("detail_desc", ""), image_base64)

    start = time.monotonic()
    for attempt in range(config.max_retries):
        try:
            if rate_limiter is not None:
                await rate_limiter.acquire()
            response = await client.responses.create(
                model=config.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": f"knowledge_{super_category.lower()}",
                        "schema": json_schema,
                        "strict": True,
                    }
                },
            )

            knowledge = json.loads(response.output_text)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            elapsed = time.monotonic() - start

            # Log
            if log_file:
                log_entry = {
                    "product_code": representative.get("product_code", ""),
                    "article_id": representative.get("article_id", ""),
                    "super_category": super_category,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": _estimate_cost(input_tokens, output_tokens),
                    "latency_s": round(elapsed, 2),
                    "model": config.model,
                    "has_image": image_base64 is not None,
                    "attempt": attempt + 1,
                }
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            if rate_limiter is not None:
                rate_limiter.record(input_tokens + output_tokens)

            return knowledge, input_tokens, output_tokens

        except (openai.RateLimitError, openai.APITimeoutError) as e:
            base_wait = 2 ** (attempt + 1)
            # Parse retry-after hint from 429 message (e.g., "Please try again in 229ms")
            retry_after = 0.0
            match = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", str(e))
            if match:
                value, unit = float(match.group(1)), match.group(2)
                retry_after = value / 1000 if unit == "ms" else value
            wait = max(base_wait, retry_after) + random.uniform(0, base_wait * 0.5)
            if rate_limiter is not None:
                rate_limiter.backpressure(wait)
            logger.warning(
                "API error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, config.max_retries, e, wait,
            )
            await asyncio.sleep(wait)
        except openai.APIError as e:
            logger.error("API error for product %s: %s", representative.get("product_code"), e)
            raise

    logger.error(
        "Failed after %d retries for product %s — skipping",
        config.max_retries, representative.get("product_code"),
    )
    return None, 0, 0


async def extract_pilot(
    articles: pd.DataFrame,
    images_dir: Path,
    output_dir: Path,
    config: ExtractionConfig,
) -> ExtractionResult:
    """Extract knowledge for pilot sample (real-time API, asyncio concurrent).

    Args:
        articles: Full articles DataFrame.
        images_dir: Path to H&M product images.
        output_dir: Output directory for results.
        config: Extraction configuration.

    Returns:
        ExtractionResult summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "extraction_log.jsonl"

    # Group by product_code
    groups = group_by_product_code(articles, images_dir)

    # Load cache
    cache = ProductCodeCache(checkpoint_dir=output_dir / "checkpoint")
    n_cache_hits = cache.load_checkpoint()

    # Select pilot sample (skip cached)
    uncached = [
        (pc, info) for pc, info in groups.items() if cache.get(pc) is None
    ]
    pilot_products = uncached[: config.pilot_size]
    logger.info(
        "Pilot: %d products to extract (%d cached, %d total)",
        len(pilot_products), n_cache_hits, len(groups),
    )

    # Semaphore for concurrency control + token-aware rate limiter
    semaphore = asyncio.Semaphore(config.max_concurrent)
    rate_limiter = TokenRateLimiter(tpm_limit=config.tpm_limit)
    client = openai.AsyncOpenAI()

    total_input_tokens = 0
    total_output_tokens = 0
    n_api_calls = 0
    failed_products: list[str] = []

    async def process_product(product_code: str, info: dict) -> None:
        nonlocal total_input_tokens, total_output_tokens, n_api_calls

        async with semaphore:
            rep = info["representative"]
            super_cat = info["super_category"]
            article_id = str(rep["article_id"])

            image_b64 = get_image_for_article(images_dir, article_id, config.image_max_size)

            knowledge, in_tok, out_tok = await _extract_single_product(
                rep, super_cat, image_b64, client, config, log_file, rate_limiter
            )

            if knowledge is None:
                failed_products.append(product_code)
                return

            # Validate schema
            vr = validate_knowledge(knowledge, super_cat)
            if not vr.is_valid:
                logger.warning(
                    "Validation errors for %s: %s", product_code, vr.errors
                )
            if vr.warnings:
                logger.debug(
                    "Validation warnings for %s: %s", product_code, vr.warnings
                )

            # Validate domain consistency
            domain_violations = validate_domain_consistency(knowledge, super_cat)
            if domain_violations:
                errors = [v for v in domain_violations if v.severity == "Error"]
                warns = [v for v in domain_violations if v.severity == "Warning"]
                if errors:
                    logger.warning(
                        "Domain errors for %s: %s",
                        product_code,
                        [f"{v.rule_name}: {v.description}" for v in errors],
                    )
                if warns:
                    logger.debug(
                        "Domain warnings for %s: %s",
                        product_code,
                        [f"{v.rule_name}: {v.description}" for v in warns],
                    )

            # Store in cache
            knowledge["product_code"] = product_code
            knowledge["super_category"] = super_cat
            cache.put(product_code, knowledge)

            total_input_tokens += in_tok
            total_output_tokens += out_tok
            n_api_calls += 1

            # Periodic checkpoint
            if n_api_calls % config.checkpoint_interval == 0:
                cache.save_checkpoint()
                logger.info(
                    "Checkpoint: %d/%d products extracted (cost: $%.4f)",
                    n_api_calls,
                    len(pilot_products),
                    _estimate_cost(total_input_tokens, total_output_tokens),
                )

    # Run extraction
    tasks = [
        process_product(pc, info) for pc, info in pilot_products
    ]
    await asyncio.gather(*tasks)

    # Final checkpoint
    cache.save_checkpoint()

    if failed_products:
        logger.warning(
            "%d products failed (will retry on next run): %s",
            len(failed_products), failed_products[:20],
        )

    # Build full article-level output
    all_rows = _build_article_rows(groups, cache)

    # Save to Parquet
    output_path = output_dir / "factual_knowledge.parquet"
    df = pd.DataFrame(all_rows)
    df.to_parquet(output_path, index=False)

    # Compute coverage
    coverage = _compute_coverage(df)

    # Save quality report
    quality_report = {
        "n_products": cache.size,
        "n_articles": len(df),
        "n_api_calls": n_api_calls,
        "n_cache_hits": n_cache_hits,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": _estimate_cost(total_input_tokens, total_output_tokens),
        "coverage": coverage,
    }
    with open(output_dir / "quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)

    return ExtractionResult(
        output_path=output_path,
        n_products=cache.size,
        n_articles=len(df),
        n_api_calls=n_api_calls,
        n_cache_hits=n_cache_hits,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=_estimate_cost(total_input_tokens, total_output_tokens),
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Build Article Rows (product_code → article propagation)
# ---------------------------------------------------------------------------


def _build_article_rows(
    groups: dict[str, dict],
    cache: ProductCodeCache,
) -> list[dict]:
    """Build article-level rows by propagating product_code knowledge to variants."""
    all_rows: list[dict] = []
    for product_code, info in groups.items():
        knowledge = cache.get(product_code)
        if knowledge is None:
            continue

        super_cat = info["super_category"]
        rep_article_id = str(info["representative"]["article_id"])
        variant_df = info["variants"]

        # Rule-based visual_weight correction (before canonical slot mapping)
        knowledge = correct_visual_weight(knowledge, super_cat)

        # Map to canonical slots
        slotted = map_to_canonical_slots(knowledge, super_cat)

        # Propagate to variants
        rows = propagate_to_variants(slotted, variant_df, rep_article_id)

        # Build factual text for each variant
        for row in rows:
            article_meta = variant_df[
                variant_df["article_id"].astype(str) == row["article_id"]
            ].iloc[0].to_dict()

            l1 = {k: v for k, v in row.items() if k.startswith("l1_")}
            l2 = {k: v for k, v in row.items() if k.startswith("l2_")}
            l3 = {k: v for k, v in row.items() if k.startswith("l3_")}

            row["factual_text_full"] = construct_factual_text(
                article_meta, l1, l2, l3, super_cat
            )

        all_rows.extend(rows)

    return all_rows


def _compute_coverage(df: pd.DataFrame) -> dict[str, float]:
    """Compute non-null ratio for each attribute column."""
    attr_cols = [c for c in df.columns if c.startswith(("l1_", "l2_", "l3_"))]
    return {col: float(df[col].notna().mean()) for col in attr_cols}
