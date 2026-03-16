"""CLI entry point for model training.

Trains baseline models (popularity, userknn, bprmf) or neural backbones
(deepfm, dcnv2, lightgcn, din, sasrec) and saves predictions.

Usage:
    # Baseline
    python scripts/train.py \
        --data-dir data/processed \
        --model-dir results/models \
        --backbone userknn

    # DeepFM (Level 1: metadata baseline)
    python scripts/train.py \
        --data-dir data/processed \
        --features-dir data/features \
        --model-dir results/models \
        --predictions-dir results/predictions \
        --backbone deepfm

    # DCN-v2
    python scripts/train.py \
        --data-dir data/processed \
        --features-dir data/features \
        --model-dir results/models \
        --predictions-dir results/predictions \
        --backbone dcnv2

    # LightGCN
    python scripts/train.py \
        --data-dir data/processed \
        --features-dir data/features \
        --model-dir results/models \
        --predictions-dir results/predictions \
        --backbone lightgcn

    # DIN (requires --build-sequences first)
    python scripts/train.py \
        --data-dir data/processed \
        --features-dir data/features \
        --model-dir results/models \
        --predictions-dir results/predictions \
        --backbone din

    # SASRec (requires --build-sequences first)
    python scripts/train.py \
        --data-dir data/processed \
        --features-dir data/features \
        --model-dir results/models \
        --predictions-dir results/predictions \
        --backbone sasrec
"""

import json
from pathlib import Path
from typing import Optional

import duckdb
import typer

from src.config import BaselineConfig

app = typer.Typer(help="Train recommendation models")

VALID_BACKBONES = (
    "popularity_global",
    "popularity_recent",
    "userknn",
    "bprmf",
    "deepfm",
    "dcnv2",
    "lightgcn",
    "din",
    "sasrec",
)

NEURAL_BACKBONES = ("deepfm", "dcnv2", "lightgcn", "din", "sasrec")


@app.command()
def main(
    data_dir: Path = typer.Option(..., help="Directory with preprocessed Parquet files"),
    model_dir: Path = typer.Option(..., help="Directory to save model artifacts"),
    predictions_dir: Path = typer.Option(
        "results/predictions", help="Directory to save prediction JSON files"
    ),
    backbone: str = typer.Option(
        ..., help=f"Model type: {' | '.join(VALID_BACKBONES)}"
    ),
    k: int = typer.Option(12, help="Number of recommendations per user"),
    split: str = typer.Option("val", help="Split to predict on: val | test"),
    # --- Neural backbone common options ---
    features_dir: Optional[Path] = typer.Option(
        None, help="Feature directory (required for neural backbones)"
    ),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
    batch_size: int = typer.Option(2048, help="Batch size"),
    max_epochs: int = typer.Option(50, help="Max training epochs"),
    patience: int = typer.Option(3, help="Early stopping patience"),
    d_embed: int = typer.Option(16, help="Embedding dimension"),
    dropout_rate: float = typer.Option(0.1, help="Dropout rate"),
    no_wandb: bool = typer.Option(False, help="Disable W&B logging"),
    random_seed: int = typer.Option(42, help="Random seed"),
    num_workers: int = typer.Option(4, help="Grain data loader workers"),
    prefetch_buffer_size: int = typer.Option(2, help="Batches to prefetch per worker"),
    # --- DCNv2-specific options ---
    n_cross_layers: int = typer.Option(3, help="Number of cross layers (dcnv2)"),
    n_experts: int = typer.Option(4, help="Number of MoE experts per cross layer (dcnv2)"),
    d_low_rank: int = typer.Option(64, help="Low-rank dimension per expert (dcnv2)"),
    # --- LightGCN-specific options ---
    n_gcn_layers: int = typer.Option(3, help="Number of GCN propagation layers (lightgcn)"),
    l2_reg: float = typer.Option(1e-4, help="L2 regularization on embeddings (lightgcn)"),
    # --- DIN-specific options ---
    attention_hidden_dims: str = typer.Option("64,32", help="Attention MLP hidden dims, comma-separated (din)"),
    # --- SASRec-specific options ---
    n_heads: int = typer.Option(2, help="Number of attention heads (sasrec)"),
    n_blocks: int = typer.Option(2, help="Number of transformer blocks (sasrec)"),
    max_seq_len: int = typer.Option(50, help="Max sequence length (din, sasrec)"),
    # --- KAR options ---
    use_kar: bool = typer.Option(False, help="Enable KAR knowledge-augmented recommendation"),
    embeddings_dir: Optional[Path] = typer.Option(
        None, help="BGE embeddings directory (required if --use-kar)"
    ),
    gating: str = typer.Option("g2", help="KAR gating variant: g1|g2|g3|g4"),
    fusion: str = typer.Option("f2", help="KAR fusion variant: f1|f2|f3|f4"),
    layer_combo: str = typer.Option("L1+L2+L3", help="Attribute layer combination"),
    d_rec: int = typer.Option(64, help="Expert output dimension"),
    align_weight: float = typer.Option(0.1, help="Alignment loss weight"),
    diversity_weight: float = typer.Option(0.01, help="Diversity loss weight"),
    stage1_epochs: int = typer.Option(2, help="Stage 1 backbone pre-train epochs"),
    stage2_epochs: int = typer.Option(5, help="Stage 2 expert adaptor epochs"),
    stage3_epochs: int = typer.Option(3, help="Stage 3 end-to-end epochs"),
    stage3_lr_factor: float = typer.Option(0.1, help="LR multiplier for stage 3"),
) -> None:
    """Train a model and generate predictions."""
    if backbone not in VALID_BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {VALID_BACKBONES}")

    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Backbone: {backbone}")

    # --- Neural backbone branches ---
    if backbone in NEURAL_BACKBONES:
        if features_dir is None:
            raise ValueError(f"--features-dir is required for {backbone} backbone")

        from src.config import (
            DCNv2Config,
            DeepFMConfig,
            DINConfig,
            ExpertConfig,
            FusionConfig,
            GatingConfig,
            KARConfig,
            LightGCNConfig,
            SASRecConfig,
            TrainConfig,
        )
        from src.training.trainer import run_kar_training, run_training

        if use_kar and embeddings_dir is None:
            raise ValueError("--embeddings-dir is required when --use-kar is set")

        train_config = TrainConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_seed=random_seed,
            use_wandb=not no_wandb,
            num_workers=num_workers,
            prefetch_buffer_size=prefetch_buffer_size,
        )

        if backbone == "deepfm":
            model_config = DeepFMConfig(
                d_embed=d_embed,
                dropout_rate=dropout_rate,
            )
        elif backbone == "dcnv2":
            model_config = DCNv2Config(
                d_embed=d_embed,
                n_cross_layers=n_cross_layers,
                n_experts=n_experts,
                d_low_rank=d_low_rank,
                dropout_rate=dropout_rate,
            )
        elif backbone == "lightgcn":
            model_config = LightGCNConfig(
                d_embed=d_embed if d_embed != 16 else 64,  # default 64 for LightGCN
                n_layers=n_gcn_layers,
                l2_reg=l2_reg,
            )
        elif backbone == "din":
            att_dims = tuple(int(x) for x in attention_hidden_dims.split(","))
            model_config = DINConfig(
                d_embed=d_embed,
                attention_hidden_dims=att_dims,
                dropout_rate=dropout_rate,
            )
        elif backbone == "sasrec":
            model_config = SASRecConfig(
                d_embed=d_embed if d_embed != 16 else 64,  # default 64 for SASRec
                n_heads=n_heads,
                n_blocks=n_blocks,
                max_seq_len=max_seq_len,
                dropout_rate=dropout_rate,
            )

        if use_kar:
            kar_config = KARConfig(
                expert=ExpertConfig(d_rec=d_rec),
                gating=GatingConfig(variant=gating),
                fusion=FusionConfig(variant=fusion),
                layer_combo=layer_combo,
                align_weight=align_weight,
                diversity_weight=diversity_weight,
                stage1_epochs=stage1_epochs,
                stage2_epochs=stage2_epochs,
                stage3_epochs=stage3_epochs,
                stage3_lr_factor=stage3_lr_factor,
            )
            result = run_kar_training(
                backbone_name=backbone,
                model_config=model_config,
                kar_config=kar_config,
                train_config=train_config,
                features_dir=features_dir,
                embeddings_dir=embeddings_dir,
                data_dir=data_dir,
                model_dir=model_dir,
                predictions_dir=predictions_dir,
                split=split,
            )
        else:
            result = run_training(
                model_config=model_config,
                train_config=train_config,
                features_dir=features_dir,
                data_dir=data_dir,
                model_dir=model_dir,
                predictions_dir=predictions_dir,
                split=split,
                backbone_name=backbone,
            )

        print(f"\n[train] Best MAP@12: {result.best_val_map_at_12:.6f} (epoch {result.best_epoch})")
        print(f"[train] Devices used: {result.n_devices}")
        print(f"[train] Total time: {result.total_train_time_seconds:.1f}s")
        return

    # --- Baseline branches ---
    from src.baselines.popularity import (
        compute_global_popularity,
        compute_recent_popularity,
        predict_popularity,
    )
    from src.baselines.utils import build_interaction_matrix, predict_from_implicit_model

    train_path = data_dir / "train_transactions.parquet"
    gt_path = data_dir / f"{split}_ground_truth.json"

    # Load ground truth to get target user IDs
    ground_truth = json.loads(gt_path.read_text())
    target_users = list(ground_truth.keys())
    print(f"[train] Target users: {len(target_users):,}")

    con = duckdb.connect()

    if backbone == "popularity_global":
        popular_items = compute_global_popularity(con, train_path, k=k)
        predictions = predict_popularity(popular_items, target_users)

    elif backbone == "popularity_recent":
        config = BaselineConfig()
        popular_items = compute_recent_popularity(
            con, train_path, k=k, window_days=config.popularity_window_days
        )
        predictions = predict_popularity(popular_items, target_users)

    elif backbone == "userknn":
        from src.baselines.userknn import train_als

        print("[train] Building interaction matrix...")
        interaction_data = build_interaction_matrix(con, train_path)
        print(
            f"  Matrix shape: {interaction_data.matrix.shape}, "
            f"nnz: {interaction_data.matrix.nnz:,}"
        )

        print("[train] Training ALS model...")
        config = BaselineConfig()
        model = train_als(interaction_data, config)

        print("[train] Generating predictions...")
        predictions = predict_from_implicit_model(model, interaction_data, target_users, k=k)

    elif backbone == "bprmf":
        from src.baselines.bprmf import train_bpr

        print("[train] Building interaction matrix...")
        interaction_data = build_interaction_matrix(con, train_path)
        print(
            f"  Matrix shape: {interaction_data.matrix.shape}, "
            f"nnz: {interaction_data.matrix.nnz:,}"
        )

        print("[train] Training BPR model...")
        config = BaselineConfig()
        model = train_bpr(interaction_data, config)

        print("[train] Generating predictions...")
        predictions = predict_from_implicit_model(model, interaction_data, target_users, k=k)

    con.close()

    # Save predictions
    output_path = predictions_dir / f"{backbone}_{split}.json"
    output_path.write_text(json.dumps(predictions, ensure_ascii=False))
    print(f"[train] Predictions saved to {output_path}")
    print(f"[train] Users with predictions: {sum(1 for v in predictions.values() if v):,}")


if __name__ == "__main__":
    app()
