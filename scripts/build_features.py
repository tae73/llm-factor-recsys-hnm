"""CLI entry point for feature engineering.

Computes user/item features from preprocessed data and generates training pairs
with negative sampling.

Usage:
    python scripts/build_features.py \
        --data-dir data/processed \
        --output-dir data/features
"""

from pathlib import Path

import typer

from src.config import FeatureConfig
from src.features.engineering import run_feature_engineering

app = typer.Typer(help="Build recommendation features from preprocessed data")


@app.command()
def main(
    data_dir: Path = typer.Option("data/processed", help="Directory with preprocessed Parquet files"),
    output_dir: Path = typer.Option("data/features", help="Output directory for feature matrices"),
    neg_sample_ratio: int = typer.Option(4, help="Negative samples per positive"),
    random_seed: int = typer.Option(42, help="Random seed for negative sampling"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
    build_sequences: bool = typer.Option(False, help="Build sequential features for DIN/SASRec"),
    max_seq_len: int = typer.Option(50, help="Max sequence length (requires --build-sequences)"),
) -> None:
    """Build features: user/item stats + negative sampling → .npz files."""
    config = FeatureConfig(
        neg_sample_ratio=neg_sample_ratio,
        random_seed=random_seed,
    )

    print(f"[build_features] Data dir: {data_dir}")
    print(f"[build_features] Output dir: {output_dir}")
    print(f"[build_features] Neg sample ratio: {neg_sample_ratio}")
    print(f"[build_features] Random seed: {random_seed}")

    result = run_feature_engineering(data_dir, output_dir, config)

    print(f"\n=== Feature Engineering Summary ===")
    print(f"  Users: {result.n_users:,}")
    print(f"  Items: {result.n_items:,}")
    print(f"  Training pairs: {result.n_train_pairs:,}")
    print(f"  User features: {result.n_user_num_features} num + {result.n_user_cat_features} cat")
    print(f"  Item features: {result.n_item_num_features} num + {result.n_item_cat_features} cat")
    print(f"  User cat vocab sizes: {result.user_cat_vocab_sizes}")
    print(f"  Item cat vocab sizes: {result.item_cat_vocab_sizes}")

    if build_sequences:
        from src.config import SequenceConfig
        from src.features.sequences import build_sequences as _build_sequences

        print(f"\n[build_features] Building sequential features (max_seq_len={max_seq_len})...")
        seq_config = SequenceConfig(max_seq_len=max_seq_len, random_seed=random_seed)
        seq_meta = _build_sequences(data_dir, output_dir, seq_config)
        print(f"  Sequential features: {seq_meta}")


if __name__ == "__main__":
    app()
