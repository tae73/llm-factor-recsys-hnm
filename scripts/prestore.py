"""CLI entry point for pre-store computation.

Pre-computes Expert outputs for all items and users, saving as .npz
for fast inference without re-running Expert MLPs at serving time.

Usage:
    python scripts/prestore.py \
        --model-dir results/models \
        --features-dir data/features \
        --embeddings-dir data/embeddings \
        --output-dir data/prestore \
        --backbone deepfm \
        --batch-size 4096
"""

from pathlib import Path

import typer

app = typer.Typer(help="Pre-compute KAR expert outputs for serving")


@app.command()
def main(
    model_dir: Path = typer.Option(..., help="Directory with trained KAR model"),
    features_dir: Path = typer.Option(..., help="Feature directory"),
    embeddings_dir: Path = typer.Option(..., help="BGE embeddings directory"),
    output_dir: Path = typer.Option(..., help="Output directory for prestore .npz"),
    backbone: str = typer.Option("deepfm", help="Backbone model name"),
    batch_size: int = typer.Option(4096, help="Batch size for expert forward"),
) -> None:
    """Pre-compute Expert outputs for all items and users."""
    import numpy as np

    from src.kar.embedding_index import build_aligned_embeddings
    from src.serving.prestore import compute_prestore
    from src.training.trainer import _load_model_state

    print("[prestore] Loading aligned embeddings...")
    item_emb, user_emb = build_aligned_embeddings(features_dir, embeddings_dir)
    print(f"  Items: {item_emb.shape}, Users: {user_emb.shape}")

    print("[prestore] Loading KAR model...")
    # Note: model must be initialized first, then state loaded
    # For now, this script assumes the model state was saved by run_kar_training
    from src.config import (
        DeepFMConfig,
        ExpertConfig,
        FusionConfig,
        GatingConfig,
        KARConfig,
        TrainConfig,
    )
    from src.features.store import load_feature_meta
    from src.training.trainer import create_kar_train_state

    feature_meta = load_feature_meta(features_dir)
    kar_config = KARConfig()  # defaults
    train_config = TrainConfig(use_wandb=False)
    model_config = DeepFMConfig()  # will be overridden by loaded state

    kar_model, _ = create_kar_train_state(
        backbone, model_config, kar_config, train_config, feature_meta, features_dir
    )
    _load_model_state(kar_model, model_dir / f"kar_{backbone}_best")
    print("  Model loaded.")

    print("[prestore] Computing expert outputs...")
    item_path, user_path = compute_prestore(
        kar_model, item_emb, user_emb, output_dir, batch_size
    )
    print(f"\n[prestore] Done.")
    print(f"  Item expert: {item_path}")
    print(f"  User expert: {user_path}")


if __name__ == "__main__":
    app()
