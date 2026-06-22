"""CLI entry point for enrichment-v2 BEHAVIOR-DERIVED axes (no API, no images).

Computes the three data-grounded decision-axes (price-tier, trend-phase, outfit-role)
over the full catalog from transactions, writing ``behavioral_axes.parquet`` keyed by
article_id. CPU/DuckDB only — no API spend.

Usage:
    python scripts/build_behavioral_axes.py \
        --data-dir data/processed \
        --output-dir data/knowledge/enrichment_v2
"""

import logging
import sys
from pathlib import Path

import typer

from src.features.behavioral_axes import build_behavioral_axes

app = typer.Typer(help="Compute enrichment-v2 behavior-derived axes (price/trend/outfit-role)")


@app.command()
def main(
    data_dir: Path = typer.Option(
        "data/processed", help="Directory with train_transactions.parquet + articles.parquet"
    ),
    output_dir: Path = typer.Option(
        "data/knowledge/enrichment_v2", help="Output directory for behavioral_axes.parquet"
    ),
    verbose: bool = typer.Option(False, help="Verbose logging"),
) -> None:
    """Compute and persist the three behavior-derived enrichment-v2 axes."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    output_path = build_behavioral_axes(data_dir, output_dir)
    typer.echo(f"Wrote behavioral axes → {output_path}")


if __name__ == "__main__":
    app()
