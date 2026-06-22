"""CLI entry point for enrichment-v2 multimodal LLM-axes extraction (~500 pilot).

Per PRODUCT_CODE, a single GPT-4.1-nano structured-output call extracts the LLM
decision-axes (occasion / fit-intent / care / perceived price+trend) from the product
IMAGE + bare name + a metadata-STRIPPED detail_desc — NEVER the categorical metadata
(the DE1 anti-recoding fix). Behavior-derived axes come from build_behavioral_axes.py.

Running the extraction WILL spend money (OpenAI API). There is NO API call on import.
``--build-sample`` (no spend) freezes the pilot sample; ``--dry-run`` (no spend)
validates the pipeline offline; the cost guard ``--max-cost`` aborts before submission.

Usage:
    # 1) Freeze the ~500-code stratified pilot sample (no API)
    python scripts/extract_enrichment_v2.py build-sample \
        --data-dir data/processed --output-dir data/knowledge/enrichment_v2

    # 2) Dry-run the extraction pipeline on the frozen sample (no API)
    python scripts/extract_enrichment_v2.py extract \
        --sample-file data/knowledge/enrichment_v2/pilot_sample.csv --dry-run

    # 3) Live extraction (SPENDS ~$0.1; cost guard at --max-cost)
    python scripts/extract_enrichment_v2.py extract \
        --sample-file data/knowledge/enrichment_v2/pilot_sample.csv \
        --images-dir data/h-and-m-personalized-fashion-recommendations/images \
        --max-cost 0.5
"""

import asyncio
import logging
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env (no API call here)

from src.config import EnrichmentV2Config
from src.knowledge.enrichment_v2.extractor import (
    estimate_e2_cost,
    extract_e2_pilot,
    group_e2_representatives,
)
from src.knowledge.enrichment_v2.prompts import build_e2_messages
from src.knowledge.enrichment_v2.sampling import freeze_pilot_sample, load_pilot_articles

app = typer.Typer(help="Enrichment-v2 multimodal extraction (no-metadata decision-axes)")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.command("build-sample")
def build_sample(
    data_dir: Path = typer.Option("data/processed", help="Directory with processed parquets"),
    output_dir: Path = typer.Option(
        "data/knowledge/enrichment_v2", help="Output dir for pilot_sample.csv + manifest"
    ),
    n_codes: int = typer.Option(500, help="Target number of product_codes"),
    floor: int = typer.Option(10, help="Min train purchases per product_code (behavioral floor)"),
    per_group_floor: int = typer.Option(3, help="Min codes per garment_group (breadth)"),
    seed: int = typer.Option(42, help="Random seed (tie-breaking)"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
) -> None:
    """Freeze the stratified ~500-code pilot sample (no API spend)."""
    _setup_logging(verbose)
    csv_path, manifest_path = freeze_pilot_sample(
        data_dir,
        output_dir,
        n_codes=n_codes,
        floor=floor,
        per_group_floor=per_group_floor,
        seed=seed,
    )
    typer.echo(f"Froze pilot sample → {csv_path}\nManifest → {manifest_path}")


@app.command("extract")
def extract(
    data_dir: Path = typer.Option("data/processed", help="Directory with articles.parquet"),
    sample_file: Path = typer.Option(
        "data/knowledge/enrichment_v2/pilot_sample.csv",
        help="Frozen pilot sample CSV (from build-sample)",
    ),
    images_dir: Path = typer.Option(
        "data/h-and-m-personalized-fashion-recommendations/images",
        help="H&M product images directory",
    ),
    output_dir: Path = typer.Option(
        "data/knowledge/enrichment_v2", help="Output dir for enrichment_v2_llm.parquet"
    ),
    model: str = typer.Option("gpt-4.1-nano", help="OpenAI model name"),
    concurrency: int = typer.Option(16, help="Async concurrent requests"),
    max_cost: float = typer.Option(0.5, help="Cost guard in USD (aborts if estimate exceeds)"),
    dry_run: bool = typer.Option(
        False, help="Group + cost-estimate + schema/image checks only; NO API call"
    ),
    verbose: bool = typer.Option(False, help="Verbose logging"),
) -> None:
    """Run (or dry-run) the multimodal extraction over the frozen pilot sample."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    if not sample_file.exists():
        logger.error("Sample file not found: %s (run build-sample first)", sample_file)
        raise typer.Exit(1)

    articles = load_pilot_articles(data_dir, sample_file)
    config = EnrichmentV2Config(model=model, concurrency=concurrency, max_cost_usd=max_cost)

    if dry_run:
        groups = group_e2_representatives(articles, images_dir)
        n = len(groups)
        n_img = sum(1 for info in groups.values() if info["image_article_id"] is not None)
        est = estimate_e2_cost(n)
        # Schema/anti-recoding self-check on a representative message.
        rep = next(iter(groups.values()))["meta"]
        msgs = build_e2_messages(rep, image_base64=None)
        blob = str(msgs).lower()
        leaks = [
            c
            for c in (
                "product_type_name",
                "colour_group_name",
                "garment_group_name",
                "section_name",
            )
            if c in blob
        ]
        logger.info(
            "[dry-run] %d product_codes; %d (%.0f%%) have an image on disk; est $%.4f (guard $%.2f).",
            n,
            n_img,
            100.0 * n_img / max(n, 1),
            est,
            config.max_cost_usd,
        )
        logger.info("[dry-run] metadata-leak check: %s", "LEAK " + str(leaks) if leaks else "clean")
        if n_img == 0:
            logger.warning(
                "[dry-run] 0%% image hit-rate — GATE 0 (image acquisition) not satisfied. "
                "Live extraction would run TEXT-ONLY (degraded body-fit/care grounding)."
            )
        raise typer.Exit(0)

    result = asyncio.run(extract_e2_pilot(articles, images_dir, output_dir, config))
    logger.info(
        "Done: %d codes, %d articles, %d calls (%d w/ image, %d cached), coverage=%.1f%%, $%.4f → %s",
        result.n_product_codes,
        result.n_articles,
        result.n_api_calls,
        result.n_with_image,
        result.n_cache_hits,
        result.coverage * 100,
        result.total_cost_usd,
        result.output_path,
    )


if __name__ == "__main__":
    app()
