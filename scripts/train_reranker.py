"""CLI entry point for GBDT Re-Ranker training and evaluation.

2-stage pipeline:
  Stage 1: Trained backbone (DeepFM) → top-K candidates with scores
  Stage 2: LightGBM re-ranker → re-sort top-K → top-12 recommendations

Two modes:
  - Base: score + rank + user/item features only (~21D)
  - Full: Base + L1/L2/L3 attributes + cross features + BGE similarity (~127D)

Usage:
    # ReRank-Base
    python scripts/train_reranker.py \\
        --stage1-model-dir results/models \\
        --stage1-backbone deepfm \\
        --data-dir data/processed \\
        --features-dir data/features \\
        --output-dir results/reranker \\
        --mode base --no-wandb

    # ReRank-Full
    python scripts/train_reranker.py \\
        --stage1-model-dir results/models \\
        --stage1-backbone deepfm \\
        --data-dir data/processed \\
        --features-dir data/features \\
        --fk-dir data/knowledge/factual \\
        --embeddings-dir data/embeddings \\
        --output-dir results/reranker \\
        --mode full --no-wandb
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="Train GBDT Re-Ranker (Stage 2 baseline)")


@app.command()
def main(
    # --- Stage 1 model ---
    stage1_model_dir: Path = typer.Option(..., help="Directory with trained Stage 1 model"),
    stage1_backbone: str = typer.Option("deepfm", help="Stage 1 backbone name"),
    # --- Data ---
    data_dir: Path = typer.Option(..., help="Processed data dir (ground truth)"),
    features_dir: Path = typer.Option(..., help="Feature store directory"),
    fk_dir: Optional[Path] = typer.Option(None, help="Factual knowledge dir (Full mode)"),
    embeddings_dir: Optional[Path] = typer.Option(None, help="BGE embeddings dir (Full mode)"),
    output_dir: Path = typer.Option(..., help="Output dir for reranker"),
    # --- ReRanker mode ---
    mode: str = typer.Option("full", help="'base' or 'full'"),
    top_k: int = typer.Option(100, help="Stage 1 candidate pool size"),
    k: int = typer.Option(12, help="Final recommendation list size"),
    split: str = typer.Option("val", help="Evaluation split"),
    # --- LightGBM hyperparameters ---
    n_estimators: int = typer.Option(500, help="Number of boosting rounds"),
    max_depth: int = typer.Option(6, help="Max tree depth"),
    learning_rate_lgbm: float = typer.Option(0.05, help="LightGBM learning rate"),
    num_leaves: int = typer.Option(31, help="Max leaves per tree"),
    min_child_samples: int = typer.Option(20, help="Min samples per leaf"),
    subsample: float = typer.Option(0.8, help="Row subsampling ratio"),
    colsample_bytree: float = typer.Option(0.8, help="Column subsampling ratio"),
    # --- Misc ---
    random_seed: int = typer.Option(42, help="Random seed"),
    no_wandb: bool = typer.Option(False, help="Disable W&B logging"),
    cache_candidates: bool = typer.Option(True, help="Cache Stage 1 candidates to disk"),
    val_sample_users: int = typer.Option(0, help="Sample N users for quick test (0=all)"),
) -> None:
    """Train GBDT Re-Ranker on Stage 1 candidates."""
    from src.config import EvalConfig, ReRankerConfig, ReRankerResult
    from src.evaluation.metrics import evaluate
    from src.features.reranker_features import (
        build_attribute_encoders,
        build_reranker_features,
        build_reranker_labels,
        encode_item_attributes,
        load_encoders,
        save_encoders,
    )
    from src.features.store import (
        load_feature_meta,
        load_id_maps,
        load_item_features,
        load_user_features,
    )
    from src.models.reranker import ReRanker
    from src.training.trainer import (
        _load_model_state,
        create_train_state,
        extract_stage1_candidates,
    )

    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Config ---
    reranker_config = ReRankerConfig(
        top_k=top_k,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate_lgbm,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_seed=random_seed,
    )

    print(f"=== GBDT Re-Ranker ({mode.upper()}) ===")
    print(f"  Stage 1: {stage1_backbone} from {stage1_model_dir}")
    print(f"  Top-K: {top_k}, Final K: {k}")

    # --- Load features ---
    print("\n[1/6] Loading features...")
    feature_meta = load_feature_meta(features_dir)
    user_features = load_user_features(features_dir)
    item_features = load_item_features(features_dir)
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = load_id_maps(features_dir)

    # --- Load ground truth ---
    gt_path = data_dir / f"{split}_ground_truth.json"
    ground_truth = json.loads(gt_path.read_text())
    target_user_ids = list(ground_truth.keys())

    if val_sample_users > 0:
        rng = np.random.default_rng(random_seed)
        valid_users = [u for u in target_user_ids if u in user_to_idx]
        target_user_ids = rng.choice(
            valid_users, size=min(val_sample_users, len(valid_users)), replace=False
        ).tolist()
        print(f"  Sampled {len(target_user_ids)} users for quick test")

    print(f"  Target users: {len(target_user_ids):,}")

    # --- Stage 1: Extract candidates ---
    print("\n[2/6] Stage 1 candidate extraction...")
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    cache_path = candidates_dir / f"{stage1_backbone}_{split}.npz"

    if cache_candidates and cache_path.exists():
        print(f"  Loading cached candidates from {cache_path}")
        cached = np.load(cache_path)
        candidates = {
            "user_indices": cached["user_indices"],
            "candidate_indices": cached["candidate_indices"],
            "candidate_scores": cached["candidate_scores"],
        }
    else:
        # Load Stage 1 model
        from src.config import DeepFMConfig, DCNv2Config

        if stage1_backbone == "deepfm":
            from src.config import DeepFMConfig
            model_config = DeepFMConfig()
        elif stage1_backbone == "dcnv2":
            from src.config import DCNv2Config
            model_config = DCNv2Config()
        else:
            raise ValueError(f"Unsupported stage1_backbone: {stage1_backbone}")

        from src.config import TrainConfig
        train_config = TrainConfig(random_seed=random_seed)
        model, _ = create_train_state(
            stage1_backbone, model_config, train_config, feature_meta, features_dir
        )
        model_checkpoint = stage1_model_dir / f"{stage1_backbone}_best"
        _load_model_state(model, model_checkpoint)
        print(f"  Loaded Stage 1 model from {model_checkpoint}")

        candidates = extract_stage1_candidates(
            model, target_user_ids, user_features, item_features,
            user_to_idx, top_k=top_k, batch_size=64,
        )

        if cache_candidates:
            np.savez_compressed(cache_path, **candidates)
            print(f"  Cached candidates to {cache_path}")

    n_users_extracted = candidates["user_indices"].shape[0]
    print(f"  Extracted: {n_users_extracted:,} users × {top_k} candidates")

    # --- Build features ---
    print(f"\n[3/6] Building {mode} features...")
    item_attributes = None
    attribute_names = None
    user_bge = None
    item_bge = None

    if mode == "full":
        if fk_dir is None:
            raise typer.BadParameter("--fk-dir required for full mode")

        # Attribute encoding
        encoder_path = output_dir / "encoders.pkl"
        fk = pd.read_parquet(fk_dir / "factual_knowledge.parquet")

        if encoder_path.exists():
            encoders = load_encoders(encoder_path)
            print("  Loaded cached encoders")
        else:
            encoders = build_attribute_encoders(fk)
            save_encoders(encoders, encoder_path)
            print("  Built and cached attribute encoders")

        item_attributes, attribute_names = encode_item_attributes(fk, encoders, idx_to_item)
        print(f"  Item attributes: {item_attributes.shape[1]} features")

        # BGE embeddings
        if embeddings_dir is not None:
            item_emb_path = embeddings_dir / "item_bge_embeddings.npz"
            user_emb_path = embeddings_dir / "user_bge_embeddings.npz"
            if item_emb_path.exists() and user_emb_path.exists():
                from src.kar.embedding_index import build_aligned_embeddings
                item_bge, user_bge = build_aligned_embeddings(features_dir, embeddings_dir)
                print(f"  BGE embeddings: items {item_bge.shape}, users {user_bge.shape}")

    X, feature_names = build_reranker_features(
        candidates["user_indices"],
        candidates["candidate_indices"],
        candidates["candidate_scores"],
        user_features, item_features,
        item_attributes, attribute_names,
        user_bge, item_bge,
    )
    y = build_reranker_labels(
        candidates["user_indices"],
        candidates["candidate_indices"],
        ground_truth, idx_to_user, idx_to_item,
    )

    print(f"  Feature matrix: {X.shape}, positive rate: {y.mean():.4f}")

    # --- Train/Val split (80/20 by user) ---
    print("\n[4/6] Training LightGBM...")
    n_users_total = candidates["user_indices"].shape[0]
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n_users_total)
    split_idx = int(n_users_total * 0.8)
    train_user_mask = np.zeros(n_users_total, dtype=bool)
    train_user_mask[perm[:split_idx]] = True

    # Expand user mask to sample mask (each user has top_k samples)
    train_mask = np.repeat(train_user_mask, top_k)
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"  Train: {X_train.shape[0]:,} samples, Val: {X_val.shape[0]:,} samples")

    reranker = ReRanker(reranker_config)
    train_metrics = reranker.train(X_train, y_train, X_val, y_val, feature_names)
    print(f"  Best iteration: {train_metrics['best_iteration']}, Val AUC: {train_metrics['val_auc']:.4f}")

    # --- Re-rank and evaluate ---
    print(f"\n[5/6] Re-ranking → top-{k}...")
    all_scores = reranker.predict(X)
    all_scores_2d = all_scores.reshape(n_users_extracted, top_k)

    predictions: dict[str, list[str]] = {}
    for i in range(n_users_extracted):
        uid = idx_to_user.get(int(candidates["user_indices"][i]), "")
        if not uid:
            continue
        reranked_order = np.argsort(all_scores_2d[i])[::-1][:k]
        item_indices = candidates["candidate_indices"][i][reranked_order]
        predictions[uid] = [idx_to_item.get(int(idx), "") for idx in item_indices]

    # Evaluate using existing pipeline
    eval_gt = {u: ground_truth[u] for u in predictions if u in ground_truth}
    eval_preds = {u: predictions[u] for u in eval_gt}
    eval_result = evaluate(eval_preds, eval_gt, EvalConfig(k=k))

    print(f"\n[6/6] Results ({mode.upper()}):")
    print(f"  MAP@{k}:  {eval_result.map_at_k:.6f}")
    print(f"  HR@{k}:   {eval_result.hr_at_k:.6f}")
    print(f"  NDCG@{k}: {eval_result.ndcg_at_k:.6f}")
    print(f"  MRR:      {eval_result.mrr:.6f}")

    # --- Save ---
    model_path = output_dir / f"reranker_{mode}"
    reranker.save(model_path)

    # Feature importance (top-20)
    importance = reranker.feature_importance("gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    metrics = {
        "mode": mode,
        "stage1_backbone": stage1_backbone,
        "top_k": top_k,
        "n_features": X.shape[1],
        "n_train_samples": int(X_train.shape[0]),
        "n_val_samples": int(X_val.shape[0]),
        "best_iteration": train_metrics["best_iteration"],
        "val_auc": train_metrics["val_auc"],
        "map_at_12": eval_result.map_at_k,
        "hr_at_12": eval_result.hr_at_k,
        "ndcg_at_12": eval_result.ndcg_at_k,
        "mrr": eval_result.mrr,
        "top_features": sorted_imp,
        "total_time_seconds": time.time() - t0,
    }
    metrics_path = output_dir / f"reranker_{mode}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

    # Save predictions
    pred_path = output_dir / f"reranker_{mode}_{split}.json"
    pred_path.write_text(json.dumps(predictions))

    print(f"\n  Model saved: {model_path}")
    print(f"  Metrics saved: {metrics_path}")
    print(f"  Predictions saved: {pred_path}")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    app()
