"""CLI entry point for the enrichment-v2 matrix-ready table (E2-2, no API).

Persists the 4 DE1-v2-passing decision-axes + per-item sell-through proxy to
``matrix_axes.parquet`` (the input the value-matrix probe reads). Promotes the
transient gap-axis logic from the DE1-v2 probe into a durable column. CPU/DuckDB only.

Usage:
    python scripts/build_enrichment_matrix.py \
        --data-dir data/processed \
        --e2-dir data/knowledge/enrichment_v2
"""

import logging
import sys
from pathlib import Path

import typer

from src.features.enrichment_matrix import build_matrix_table

app = typer.Typer(help="Build enrichment-v2 matrix-ready table (gaps + sell-through)")


@app.command()
def main(
    data_dir: Path = typer.Option(
        "data/processed", help="Directory with train_transactions.parquet"
    ),
    e2_dir: Path = typer.Option(
        "data/knowledge/enrichment_v2",
        help="Directory with behavioral_axes.parquet + enrichment_v2_llm.parquet",
    ),
    verbose: bool = typer.Option(False, help="Verbose logging"),
) -> None:
    """Assemble and persist the matrix-ready table → matrix_axes.parquet."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    output_path = build_matrix_table(data_dir, e2_dir)
    typer.echo(f"Wrote matrix axes → {output_path}")


if __name__ == "__main__":
    app()
