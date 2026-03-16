"""CLI entry point for data preprocessing and temporal splitting.

Converts raw H&M CSV files to Parquet format and creates train/val/test splits.

Usage:
    python scripts/preprocess.py \
        --raw-dir data/h-and-m-personalized-fashion-recommendations \
        --output-dir data/processed
"""

from pathlib import Path

import typer

from src.config import DataPaths, FilterConfig, SplitConfig
from src.data.preprocessing import run_preprocessing
from src.data.splitter import run_split

app = typer.Typer(help="Preprocess H&M data: CSV → Parquet + temporal split")


@app.command()
def main(
    raw_dir: Path = typer.Option(
        ..., help="Raw CSV directory (e.g. data/h-and-m-personalized-fashion-recommendations)"
    ),
    output_dir: Path = typer.Option(..., help="Output directory for Parquet files"),
    active_min: int = typer.Option(5, help="Minimum purchases for active user classification"),
    train_end: str = typer.Option("2020-06-30", help="Train period end date (inclusive)"),
    val_start: str = typer.Option("2020-07-01", help="Validation period start date"),
    val_end: str = typer.Option("2020-08-31", help="Validation period end date"),
    test_start: str = typer.Option("2020-09-01", help="Test period start date"),
    test_end: str = typer.Option("2020-09-07", help="Test period end date"),
    verbose: bool = typer.Option(False, help="Print detailed statistics"),
) -> None:
    """Run preprocessing pipeline: raw CSV → Parquet → temporal split."""
    paths = DataPaths(raw_dir=raw_dir, processed_dir=output_dir)
    split_config = SplitConfig(
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )
    filter_config = FilterConfig(active_min=active_min)

    # Step 1: CSV → Parquet
    preprocess_result = run_preprocessing(paths)

    # Step 2: Temporal split + customer filtering
    split_result = run_split(output_dir, output_dir, split_config, filter_config)

    print("\n=== Split Summary ===")
    print(f"  Train:  {split_result.n_train:>12,} transactions")
    print(f"  Val:    {split_result.n_val:>12,} transactions")
    print(f"  Test:   {split_result.n_test:>12,} transactions")
    print(f"  Active users:       {split_result.n_active_users:>10,}")
    print(f"  Sparse users:       {split_result.n_sparse_users:>10,}")
    print(f"  Cold-start users (val): {split_result.n_cold_start_users_val:>7,}")
    print(f"  Cold-start items (val): {split_result.n_cold_start_items_val:>7,}")


if __name__ == "__main__":
    app()
