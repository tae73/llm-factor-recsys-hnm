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
    "recent_popularity",
    "repurchase",
    "userknn",
    "bprmf",
    "deepfm",
    "dcnv2",
    "lightgcn",
    "din",
    "sasrec",
)

NEURAL_BACKBONES = ("deepfm", "dcnv2", "lightgcn", "din", "sasrec")

# Ground-truth filename per eval split. "immediate" = Kaggle-comparable
# next-period window built by src.data.splitter.build_immediate_eval.
EVAL_SPLIT_GT = {
    "val": "val_ground_truth.json",
    "test": "test_ground_truth.json",
    "immediate": "immediate_ground_truth.json",
}


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
    eval_split: Optional[str] = typer.Option(
        None,
        help="Eval ground-truth split: val | test | immediate. Overrides --split for "
        "baseline GT selection (immediate = Kaggle-comparable next-period window).",
    ),
    train_end: str = typer.Option(
        "2020-06-30", help="Train cut-off date (repurchase/recent_popularity recency window)"
    ),
    recent_days: int = typer.Option(
        14, help="Recent-popularity window in days (repurchase/recent_popularity)"
    ),
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
    use_id_embed: bool = typer.Option(
        False, help="Add per-user/per-item id embeddings (deepfm/dcnv2 CF capacity)"
    ),
    bce_pos_weight: float = typer.Option(
        1.0, help="Positive-class weight in BCE (>1 up-weights positives; deepfm/dcnv2)"
    ),
    no_wandb: bool = typer.Option(False, help="Disable W&B logging"),
    random_seed: int = typer.Option(42, help="Random seed"),
    num_workers: int = typer.Option(4, help="Grain data loader workers"),
    prefetch_buffer_size: int = typer.Option(2, help="Batches to prefetch per worker"),
    # --- Validation scoring (batched full-catalog, Track B) ---
    val_sample_users: int = typer.Option(50000, help="Epoch-end validation users (drives early stop)"),
    midval_sample_users: int = typer.Option(5000, help="Mid-epoch validation users"),
    pred_chunk_users: int = typer.Option(32, help="Users per batched-scoring chunk (lower if OOM)"),
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
            val_sample_users=val_sample_users,
            midval_sample_users=midval_sample_users,
            pred_chunk_users=pred_chunk_users,
            bce_pos_weight=bce_pos_weight,
        )

        if backbone == "deepfm":
            model_config = DeepFMConfig(
                d_embed=d_embed,
                dropout_rate=dropout_rate,
                use_id_embed=use_id_embed,
            )
        elif backbone == "dcnv2":
            model_config = DCNv2Config(
                d_embed=d_embed,
                n_cross_layers=n_cross_layers,
                n_experts=n_experts,
                d_low_rank=d_low_rank,
                dropout_rate=dropout_rate,
                use_id_embed=use_id_embed,
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

    # Resolve which ground-truth split to evaluate on. --eval-split takes
    # precedence over --split (so "immediate" can be requested explicitly).
    resolved_split = eval_split or split
    if resolved_split not in EVAL_SPLIT_GT:
        raise ValueError(
            f"Unknown eval split '{resolved_split}'. Choose from: {tuple(EVAL_SPLIT_GT)}"
        )
    gt_path = data_dir / EVAL_SPLIT_GT[resolved_split]

    # Load ground truth to get target user IDs
    ground_truth = json.loads(gt_path.read_text())
    target_users = list(ground_truth.keys())
    print(f"[train] Eval split: {resolved_split} ({gt_path.name})")
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

    elif backbone in ("repurchase", "recent_popularity"):
        import pandas as pd

        from src.baselines.repurchase import (
            recent_popularity,
            repurchase_predict,
        )

        print(f"[train] Loading train transactions for {backbone}...")
        train_txn = pd.read_parquet(
            train_path, columns=["customer_id", "article_id", "t_dat"]
        ).assign(
            customer_id=lambda d: d["customer_id"].astype(str),
            article_id=lambda d: d["article_id"].astype(str),
        )
        recent_items = recent_popularity(
            train_txn, train_end=train_end, days=recent_days, k=k
        )
        if backbone == "recent_popularity":
            predictions = predict_popularity(recent_items, target_users)
        else:  # repurchase (hybrid: repurchase + recent-pop fill)
            predictions = repurchase_predict(
                train_txn, target_users, k=k, fill_recent=recent_items
            )

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
    output_path = predictions_dir / f"{backbone}_{resolved_split}.json"
    output_path.write_text(json.dumps(predictions, ensure_ascii=False))
    print(f"[train] Predictions saved to {output_path}")
    print(f"[train] Users with predictions: {sum(1 for v in predictions.values() if v):,}")

    # --- Fair-comparison eval: same cohort split + headline as the neural path ---
    # When --features-dir is provided, evaluate the baseline restricted to the
    # IDENTICAL feature_capable user set (users in user_to_idx) so the headline
    # number matches what DeepFM/DCNv2 report. cold_start is shown separately.
    if features_dir is not None:
        from src.config import EvalConfig
        from src.evaluation.metrics import evaluate, evaluate_by_cohort
        from src.features.store import load_id_maps
        from src.training.trainer import split_eval_cohorts

        user_to_idx, _, _, _ = load_id_maps(features_dir)
        cohorts = split_eval_cohorts(target_users, user_to_idx)
        n_feat = len(cohorts["feature_capable"])
        n_cold = len(cohorts["cold_start"])
        config = EvalConfig(k=k)

        cohort_results = evaluate_by_cohort(predictions, ground_truth, cohorts, config)
        all_result = evaluate(predictions, ground_truth, config)

        def _m(r):
            return {
                "map_at_12": round(r.map_at_k, 6),
                "hr_at_12": round(r.hr_at_k, 6),
                "ndcg_at_12": round(r.ndcg_at_k, 6),
                "mrr": round(r.mrr, 6),
            }

        cohort_metrics = {name: _m(r) for name, r in cohort_results.items()}
        all_metrics = _m(all_result)
        headline = cohort_metrics.get("feature_capable", all_metrics)

        print(
            f"[train] Baseline eval — cohorts: feature_capable={n_feat:,} "
            f"cold_start={n_cold:,} (headline = feature_capable)"
        )
        print(
            f"  [headline: feature_capable, n={n_feat:,}] "
            f"MAP@12={headline['map_at_12']:.6f} HR@12={headline['hr_at_12']:.6f} "
            f"NDCG@12={headline['ndcg_at_12']:.6f} MRR={headline['mrr']:.6f}"
        )
        if n_cold:
            cm = cohort_metrics["cold_start"]
            print(
                f"  [cohort: cold_start, n={n_cold:,}] "
                f"MAP@12={cm['map_at_12']:.6f} HR@12={cm['hr_at_12']:.6f}"
            )

        metrics_out = {
            "headline_cohort": "feature_capable",
            "headline": headline,
            "cohorts": cohort_metrics,
            "cohort_sizes": {"feature_capable": n_feat, "cold_start": n_cold},
            "all_users": all_metrics,
            **headline,
        }
        metrics_path = model_dir / f"{backbone}_metrics.json"
        metrics_path.write_text(json.dumps(metrics_out, indent=2))
        print(f"[train] Baseline metrics saved: {metrics_path}")


if __name__ == "__main__":
    app()
