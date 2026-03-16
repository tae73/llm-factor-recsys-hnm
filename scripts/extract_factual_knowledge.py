"""CLI entry point for L1+L2+L3 factual knowledge extraction.

Extracts structured attributes from product descriptions and images using GPT-4.1-nano.
Supports both real-time API (pilot) and Batch API (full extraction).

Usage:
    # Pilot (500 products, real-time API)
    python scripts/extract_factual_knowledge.py \
        --data-dir data/processed \
        --images-dir data/h-and-m-personalized-fashion-recommendations/images \
        --output-dir data/knowledge/factual \
        --pilot

    # Full batch (47K products, Batch API, 50% discount)
    # Sequential pipeline: splits into ~500-request chunks, submits one at a time
    # to stay within org-level enqueued token limit (2M tokens for gpt-4.1-nano)
    python scripts/extract_factual_knowledge.py \
        --data-dir data/processed \
        --images-dir data/h-and-m-personalized-fashion-recommendations/images \
        --output-dir data/knowledge/factual \
        --batch-api \
        --max-cost 15.0

    # Resume (auto-detects batch_ids.json, skips completed chunks)
    python scripts/extract_factual_knowledge.py \
        --data-dir data/processed \
        --images-dir data/h-and-m-personalized-fashion-recommendations/images \
        --output-dir data/knowledge/factual \
        --batch-api

    # Poll single legacy batch ID
    python scripts/extract_factual_knowledge.py \
        --data-dir data/processed \
        --images-dir data/h-and-m-personalized-fashion-recommendations/images \
        --output-dir data/knowledge/factual \
        --batch-api \
        --batch-id batch_abc123
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env

from src.config import ExtractionConfig
from src.knowledge.factual.batch import (
    load_batch_manifest,
    parse_batch_results,
    poll_batch,
    prepare_batch_jsonl_chunked,
    run_batch_pipeline,
)
from src.knowledge.factual.cache import ProductCodeCache
from src.knowledge.factual.extractor import (
    _build_article_rows,
    _compute_coverage,
    extract_pilot,
    group_by_product_code,
)
from src.knowledge.factual.validator import validate_knowledge

app = typer.Typer(help="Extract L1+L2+L3 factual knowledge via GPT-4.1-nano")


@app.command()
def main(
    data_dir: Path = typer.Option(
        "data/processed",
        help="Directory containing articles.parquet",
    ),
    images_dir: Path = typer.Option(
        "data/h-and-m-personalized-fashion-recommendations/images",
        help="Product image directory",
    ),
    output_dir: Path = typer.Option(
        "data/knowledge/factual",
        help="Output directory for extracted knowledge",
    ),
    model: str = typer.Option("gpt-4.1-nano", help="OpenAI model name"),
    batch_api: bool = typer.Option(False, help="Use Batch API (50%% discount, 24h)"),
    max_concurrent: int = typer.Option(5, help="Real-time API concurrent requests"),
    max_cost: float = typer.Option(15.0, help="Cost limit in USD"),
    tpm_limit: int = typer.Option(200_000, help="Tokens-per-minute limit for real-time API"),
    pilot: bool = typer.Option(False, help="Extract pilot sample only (500 products)"),
    resume: bool = typer.Option(False, help="Resume from checkpoint"),
    batch_id: str = typer.Option("", help="Poll existing batch ID instead of submitting new"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
) -> None:
    """Run factual knowledge extraction pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)

    config = ExtractionConfig(
        model=model,
        use_batch_api=batch_api,
        max_concurrent=max_concurrent,
        max_cost_usd=max_cost,
        tpm_limit=tpm_limit,
    )

    # Load articles
    articles_path = data_dir / "articles.parquet"
    if not articles_path.exists():
        logger.error("articles.parquet not found at %s", articles_path)
        raise typer.Exit(1)

    articles = pd.read_parquet(articles_path)
    logger.info("Loaded %d articles from %s", len(articles), articles_path)

    if pilot:
        # Real-time API pilot extraction
        logger.info("Running pilot extraction (max %d products)...", config.pilot_size)
        result = asyncio.run(extract_pilot(articles, images_dir, output_dir, config))
        n_new = result.n_products - result.n_cache_hits
        logger.info(
            "Pilot complete: %d new + %d cached = %d total products, "
            "%d articles, %d API calls, $%.4f",
            n_new,
            result.n_cache_hits,
            result.n_products,
            result.n_articles,
            result.n_api_calls,
            result.total_cost_usd,
        )
        logger.info("Output: %s", result.output_path)

    elif batch_api:
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_dir = output_dir / "batch"
        batch_dir.mkdir(exist_ok=True)

        if batch_id:
            # Legacy: poll single existing batch
            logger.info("Polling existing batch: %s", batch_id)
            results_path = poll_batch(batch_id)
            results_path = results_path.rename(batch_dir / "output.jsonl")
            _process_batch_results(
                [results_path], articles, images_dir, output_dir, resume, logger
            )

        else:
            # Sequential pipeline: prepare → submit→poll one chunk at a time
            groups = group_by_product_code(articles, images_dir)

            # Load cache for resume
            cache = ProductCodeCache(checkpoint_dir=output_dir / "checkpoint")
            if resume:
                n_loaded = cache.load_checkpoint()
                logger.info("Resumed: %d products from checkpoint", n_loaded)

            # Filter uncached
            uncached_products = [
                {
                    "product_code": pc,
                    "representative": info["representative"],
                    "super_category": info["super_category"],
                }
                for pc, info in groups.items()
                if cache.get(pc) is None
            ]
            logger.info(
                "Batch: %d products to extract (%d cached)",
                len(uncached_products),
                cache.size,
            )

            if not uncached_products:
                logger.info("All products already cached. Nothing to do.")
                raise typer.Exit(0)

            # Check for existing chunks (resume) vs fresh prepare
            manifest = load_batch_manifest(batch_dir)
            existing_chunks = sorted(batch_dir.glob("input_*.jsonl"))

            if existing_chunks and manifest is not None and manifest.get("n_chunks") == len(existing_chunks):
                # Reuse existing chunks (resume interrupted pipeline)
                chunk_paths = existing_chunks
                logger.info(
                    "Reusing %d existing chunks from %s (resume)",
                    len(chunk_paths),
                    batch_dir,
                )
            else:
                # Fresh prepare: split by max_requests to stay within enqueued token limit
                chunk_paths = prepare_batch_jsonl_chunked(
                    uncached_products,
                    images_dir,
                    config,
                    output_dir=batch_dir,
                    max_bytes=config.batch_max_bytes,
                    max_requests=config.batch_max_requests,
                )
                logger.info(
                    "Prepared %d chunks (max %d requests, max %.0f MB each)",
                    len(chunk_paths),
                    config.batch_max_requests,
                    config.batch_max_bytes / 1_000_000,
                )

            # Run sequential pipeline: submit→poll→download one chunk at a time
            result_paths = run_batch_pipeline(
                chunk_paths, batch_dir=batch_dir, poll_interval=60,
            )

            if result_paths:
                _process_batch_results(
                    result_paths, articles, images_dir, output_dir, resume, logger
                )
            else:
                logger.warning("No batch results downloaded. Check logs for errors.")

    else:
        # Real-time API full extraction (with concurrency)
        logger.info("Running full real-time extraction...")
        # Override pilot_size to extract all
        full_config = config._replace(pilot_size=999_999)
        result = asyncio.run(extract_pilot(articles, images_dir, output_dir, full_config))
        n_new = result.n_products - result.n_cache_hits
        logger.info(
            "Complete: %d new + %d cached = %d total products, "
            "%d articles, %d API calls, $%.4f",
            n_new,
            result.n_cache_hits,
            result.n_products,
            result.n_articles,
            result.n_api_calls,
            result.total_cost_usd,
        )


def _process_batch_results(
    result_paths: list[Path],
    articles: pd.DataFrame,
    images_dir: Path,
    output_dir: Path,
    resume: bool,
    logger: logging.Logger,
) -> None:
    """Parse, validate, cache, and save batch results to Parquet.

    Shared logic for single-batch and multi-batch result processing.
    """
    parsed = parse_batch_results(result_paths)
    groups = group_by_product_code(articles, images_dir)

    cache = ProductCodeCache(checkpoint_dir=output_dir / "checkpoint")
    if resume:
        cache.load_checkpoint()

    # Validate and cache results
    n_valid = 0
    for product_code, knowledge in parsed.items():
        if product_code in groups:
            super_cat = groups[product_code]["super_category"]
            vr = validate_knowledge(knowledge, super_cat)
            if not vr.is_valid:
                logger.warning("Validation errors for %s: %s", product_code, vr.errors)
            knowledge["product_code"] = product_code
            knowledge["super_category"] = super_cat
            cache.put(product_code, knowledge)
            n_valid += 1

    cache.save_checkpoint()

    # Build article-level output
    all_rows = _build_article_rows(groups, cache)
    output_path = output_dir / "factual_knowledge.parquet"
    df = pd.DataFrame(all_rows)
    df.to_parquet(output_path, index=False)

    coverage = _compute_coverage(df)
    quality_report = {
        "n_products": cache.size,
        "n_articles": len(df),
        "n_batch_results": len(parsed),
        "n_valid": n_valid,
        "coverage": coverage,
    }
    with open(output_dir / "quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)

    logger.info(
        "Batch complete: %d new + %d cached = %d total products, "
        "%d articles → %s",
        n_valid,
        cache.size - n_valid,
        cache.size,
        len(df),
        output_path,
    )


if __name__ == "__main__":
    app()
