"""CLI entry point for combined EXTERNAL (open-world styling) knowledge extraction.

Per PRODUCT_CODE, a single GPT-4.1-nano structured-output call extracts BOTH a
prose complement description AND structured complement attributes; results are
propagated to every article SKU and written to a resume-safe Parquet.

Running this WILL spend money (it calls the OpenAI API). There is NO API call on
import. Use --pilot N or --limit N to cap a run; the cost guard (--max-cost)
aborts before submission if the estimate exceeds the budget.

Usage:
    # Pilot — extract only 3 product_codes (smoke test, tiny spend)
    python scripts/extract_external_knowledge.py \
        --data-dir data/processed \
        --output-dir data/knowledge/external \
        --pilot 3

    # FULL extraction (~47K product_codes) with cost guard
    python scripts/extract_external_knowledge.py \
        --data-dir data/processed \
        --output-dir data/knowledge/external \
        --max-cost 12.0
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env (no API call here)

from src.config import ExternalExtractionConfig
from src.knowledge.external.extractor import (
    estimate_external_cost,
    extract_external_knowledge,
    group_representatives,
)

app = typer.Typer(
    help="Extract combined external (prose + structured) complement knowledge via GPT-4.1-nano"
)


@app.command()
def main(
    data_dir: Path = typer.Option(
        "data/processed",
        help="Directory containing articles.parquet",
    ),
    output_dir: Path = typer.Option(
        "data/knowledge/external",
        help="Output directory for external_knowledge_full.parquet",
    ),
    model: str = typer.Option("gpt-4.1-nano", help="OpenAI model name"),
    concurrency: int = typer.Option(32, help="Async concurrent requests"),
    max_retries: int = typer.Option(4, help="Retry/backoff attempts per item"),
    max_cost: float = typer.Option(12.0, help="Cost guard in USD (aborts if estimate exceeds)"),
    pilot: Optional[int] = typer.Option(
        None, help="Extract only the first N product_codes (smoke test)"
    ),
    limit: Optional[int] = typer.Option(
        None, help="Upper bound on product_codes processed this run"
    ),
    dry_run: bool = typer.Option(
        False,
        help="Estimate cost + group only; NO API call (validates pipeline offline)",
    ),
    verbose: bool = typer.Option(False, help="Verbose logging"),
) -> None:
    """Run combined external-knowledge extraction (prose + structured complements)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)

    config = ExternalExtractionConfig(
        model=model,
        concurrency=concurrency,
        max_retries=max_retries,
        max_cost_usd=max_cost,
    )

    articles_path = data_dir / "articles.parquet"
    if not articles_path.exists():
        logger.error("articles.parquet not found at %s", articles_path)
        raise typer.Exit(1)

    articles = pd.read_parquet(articles_path)
    logger.info("Loaded %d articles from %s", len(articles), articles_path)

    if dry_run:
        # Offline validation: group + cost estimate, no API spend.
        groups = group_representatives(articles)
        cap = pilot if pilot is not None else limit
        n = len(groups) if cap is None else min(cap, len(groups))
        est = estimate_external_cost(n)
        logger.info(
            "[dry-run] %d product_codes (%d articles); would extract %d → est. $%.4f "
            "(guard $%.2f). NO API call made.",
            len(groups),
            len(articles),
            n,
            est,
            config.max_cost_usd,
        )
        raise typer.Exit(0)

    # Live extraction (SPENDS MONEY).
    result = asyncio.run(
        extract_external_knowledge(
            articles, output_dir, config, pilot=pilot, limit=limit
        )
    )
    logger.info(
        "Done: %d product_codes, %d articles, %d API calls (%d cached), $%.4f",
        result.n_product_codes,
        result.n_articles,
        result.n_api_calls,
        result.n_cache_hits,
        result.total_cost_usd,
    )
    logger.info(
        "Coverage: prose=%.1f%%, structured=%.1f%% → %s",
        result.coverage_prose * 100,
        result.coverage_structured * 100,
        result.output_path,
    )


if __name__ == "__main__":
    app()
