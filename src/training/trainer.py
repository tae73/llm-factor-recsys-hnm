"""Training loop for recommendation backbones (DeepFM, DCN-v2, LightGCN, DIN, SASRec).

Core training logic separated from scripts/train.py CLI wrapper for:
- HPO tool reuse (W&B Sweeps can call run_training directly)
- Unit testing (test train_step, score_full_catalog independently)
- Notebook experiments

Design:
- Device-agnostic: auto-detects available JAX devices
- Mesh + NamedSharding for single/multi-device (same code path)
- Grain DataLoader for multiprocess prefetch + deterministic shuffling
- Early stopping on val MAP@12
- Optional W&B logging
- Multi-backbone dispatch via backbone_name parameter
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from src.config import (
    DCNv2Config,
    DeepFMConfig,
    DINConfig,
    EvalConfig,
    KARConfig,
    LightGCNConfig,
    SASRecConfig,
    TrainConfig,
    TrainResult,
)
from src.evaluation.metrics import evaluate
from src.features.store import (
    load_feature_meta,
    load_id_maps,
    load_item_features,
    load_train_pairs,
    load_user_features,
)
from src.losses import align_loss, binary_cross_entropy, diversity_loss, embedding_l2_reg, kar_total_loss
from src.models import get_backbone
from src.models.deepfm import DeepFMInput
from src.models.din import DINInput
from src.models.lightgcn import LightGCNInput, build_normalized_adj
from src.models.sasrec import SASRecInput
from src.training.data_loader import create_train_loader, steps_per_epoch


# ---------------------------------------------------------------------------
# Model Initialization (multi-backbone)
# ---------------------------------------------------------------------------


def create_train_state(
    backbone_name: str,
    model_config: DeepFMConfig | DCNv2Config | LightGCNConfig | DINConfig | SASRecConfig,
    train_config: TrainConfig,
    feature_meta: dict[str, Any],
    features_dir: Path | None = None,
) -> tuple[nnx.Module, nnx.Optimizer]:
    """Initialize model + Optax optimizer for any backbone. Device-agnostic.

    Args:
        backbone_name: One of "deepfm", "dcnv2", "lightgcn", "din", "sasrec".
        model_config: Backbone-specific config NamedTuple.
        train_config: Training loop settings.
        feature_meta: Feature metadata dict from load_feature_meta().
        features_dir: Path to feature .npz files (required for lightgcn, din, sasrec).

    Returns:
        (model, optimizer) tuple.
    """
    spec = get_backbone(backbone_name)
    rngs = nnx.Rngs(params=train_config.random_seed, dropout=train_config.random_seed + 1)

    if spec.needs_graph:
        # Graph-based: LightGCN
        assert features_dir is not None, "features_dir required for graph backbone"
        pairs = load_train_pairs(features_dir)
        adj = build_normalized_adj(
            user_idx=pairs["user_idx"],
            item_idx=pairs["item_idx"],
            n_users=feature_meta["n_users"],
            n_items=feature_meta["n_items"],
        )
        model = spec.model_cls(
            n_users=feature_meta["n_users"],
            n_items=feature_meta["n_items"],
            adj_matrix=adj,
            config=model_config,
            rngs=rngs,
        )
    elif backbone_name == "din":
        # DIN: static features + sequence
        user_cat_vocab_sizes = feature_meta["user_cat_vocab_sizes"]
        item_cat_vocab_sizes = feature_meta["item_cat_vocab_sizes"]
        field_dims = [user_cat_vocab_sizes[name] for name in feature_meta["user_cat_names"]]
        field_dims += [item_cat_vocab_sizes[name] for name in feature_meta["item_cat_names"]]
        n_numerical = feature_meta["n_user_numerical"] + feature_meta["n_item_numerical"]

        model = spec.model_cls(
            field_dims=field_dims,
            n_numerical=n_numerical,
            n_items=feature_meta["n_items"],
            max_seq_len=model_config.d_embed,  # placeholder — actual max_seq_len from sequences
            config=model_config,
            rngs=rngs,
        )
        # Fix max_seq_len: load from sequence metadata if available
        if features_dir is not None:
            seq_path = features_dir / "train_sequences.npz"
            if seq_path.exists():
                seq_data = np.load(seq_path)
                actual_max_len = seq_data["sequences"].shape[1]
                # Re-create model with correct max_seq_len
                model = spec.model_cls(
                    field_dims=field_dims,
                    n_numerical=n_numerical,
                    n_items=feature_meta["n_items"],
                    max_seq_len=actual_max_len,
                    config=model_config,
                    rngs=rngs,
                )
    elif backbone_name == "sasrec":
        # SASRec: sequence only
        model = spec.model_cls(
            n_items=feature_meta["n_items"],
            config=model_config,
            rngs=rngs,
        )
    else:
        # Feature-based: DeepFM, DCNv2
        user_cat_vocab_sizes = feature_meta["user_cat_vocab_sizes"]
        item_cat_vocab_sizes = feature_meta["item_cat_vocab_sizes"]
        field_dims = [user_cat_vocab_sizes[name] for name in feature_meta["user_cat_names"]]
        field_dims += [item_cat_vocab_sizes[name] for name in feature_meta["item_cat_names"]]
        n_numerical = feature_meta["n_user_numerical"] + feature_meta["n_item_numerical"]

        model = spec.model_cls(
            field_dims=field_dims,
            n_numerical=n_numerical,
            config=model_config,
            rngs=rngs,
        )

    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=train_config.learning_rate,
                weight_decay=1e-5,
            ),
        ),
        wrt=nnx.Param,
    )
    return model, optimizer


# ---------------------------------------------------------------------------
# Train Step Factories (JIT-compiled)
# ---------------------------------------------------------------------------


def make_feature_train_step() -> Any:
    """Return JIT-compiled train step for feature-based models (DeepFM, DCNv2)."""

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
    ) -> jax.Array:
        def loss_fn(model: nnx.Module) -> jax.Array:
            logits = model(
                DeepFMInput(
                    user_cat=batch["user_cat"],
                    user_num=batch["user_num"],
                    item_cat=batch["item_cat"],
                    item_num=batch["item_num"],
                )
            )
            return binary_cross_entropy(logits, batch["labels"])

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def make_lightgcn_train_step(l2_reg: float) -> Any:
    """Return JIT-compiled train step for LightGCN (BCE + L2 reg on initial embeddings)."""

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
    ) -> jax.Array:
        def loss_fn(model: nnx.Module) -> jax.Array:
            inp = LightGCNInput(
                user_idx=batch["user_idx"],
                item_idx=batch["item_idx"],
            )
            logits = model(inp)
            bce = binary_cross_entropy(logits, batch["labels"])

            # L2 regularization on initial embeddings
            u_e0, i_e0 = model.get_initial_embeddings(inp)
            reg = embedding_l2_reg(u_e0, i_e0, l2_reg)

            return bce + reg

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def make_din_train_step() -> Any:
    """Return JIT-compiled train step for DIN."""

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
    ) -> jax.Array:
        def loss_fn(model: nnx.Module) -> jax.Array:
            logits = model(
                DINInput(
                    user_cat=batch["user_cat"],
                    user_num=batch["user_num"],
                    item_cat=batch["item_cat"],
                    item_num=batch["item_num"],
                    history=batch["history"],
                    hist_len=batch["hist_len"],
                )
            )
            return binary_cross_entropy(logits, batch["labels"])

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def make_sasrec_train_step() -> Any:
    """Return JIT-compiled train step for SASRec."""

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
    ) -> jax.Array:
        def loss_fn(model: nnx.Module) -> jax.Array:
            logits = model(
                SASRecInput(
                    history=batch["history"],
                    hist_len=batch["hist_len"],
                ),
                target_item_idx=batch["target_item_seq_idx"],
            )
            return binary_cross_entropy(logits, batch["labels"])

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def make_train_step(backbone_name: str, model_config: Any) -> Any:
    """Factory: return the right JIT-compiled train step for a backbone.

    Args:
        backbone_name: One of "deepfm", "dcnv2", "lightgcn", "din", "sasrec".
        model_config: Backbone config (needed for LightGCN l2_reg).

    Returns:
        A callable train_step(model, optimizer, batch) → loss.
    """
    spec = get_backbone(backbone_name)
    if spec.needs_graph:
        return make_lightgcn_train_step(model_config.l2_reg)
    if backbone_name == "din":
        return make_din_train_step()
    if backbone_name == "sasrec":
        return make_sasrec_train_step()
    return make_feature_train_step()


# Backward-compatible alias: feature-based train_step used by existing tests
_feature_train_step = make_feature_train_step()


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
) -> jax.Array:
    """Single training step for feature-based models. JIT-compiled, device-agnostic.

    Backward-compatible: accepts a batch dict with keys:
    user_cat, user_num, item_cat, item_num, labels.
    """

    def loss_fn(model: nnx.Module) -> jax.Array:
        logits = model(
            DeepFMInput(
                user_cat=batch["user_cat"],
                user_num=batch["user_num"],
                item_cat=batch["item_cat"],
                item_num=batch["item_num"],
            )
        )
        return binary_cross_entropy(logits, batch["labels"])

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


# ---------------------------------------------------------------------------
# Full Catalog Scoring (multi-backbone)
# ---------------------------------------------------------------------------


def score_full_catalog(
    model: nnx.Module,
    user_idx: int,
    user_features: dict[str, np.ndarray],
    item_features: dict[str, np.ndarray],
    k: int = 12,
    backbone_name: str = "deepfm",
    sequences: np.ndarray | None = None,
    seq_lengths: np.ndarray | None = None,
) -> list[int]:
    """Score single user × full item catalog → top-K item indices.

    Dispatches to feature-based, graph-based, or sequential scoring.
    """
    spec = get_backbone(backbone_name)

    if spec.needs_graph:
        return _score_full_catalog_graph(model, user_idx, k)
    if backbone_name == "din":
        return _score_full_catalog_din(
            model, user_idx, user_features, item_features, sequences, seq_lengths, k
        )
    if backbone_name == "sasrec":
        return _score_full_catalog_sasrec(model, user_idx, sequences, seq_lengths, k)
    return _score_full_catalog_features(model, user_idx, user_features, item_features, k)


def _score_full_catalog_features(
    model: nnx.Module,
    user_idx: int,
    user_features: dict[str, np.ndarray],
    item_features: dict[str, np.ndarray],
    k: int = 12,
) -> list[int]:
    """Feature-based scoring: broadcasts user features against all items."""
    n_items = item_features["numerical"].shape[0]

    u_cat = np.tile(user_features["categorical"][user_idx], (n_items, 1))
    u_num = np.tile(user_features["numerical"][user_idx], (n_items, 1))
    i_cat = item_features["categorical"]
    i_num = item_features["numerical"]

    inp = DeepFMInput(
        user_cat=jnp.array(u_cat, dtype=jnp.int32),
        user_num=jnp.array(u_num, dtype=jnp.float32),
        item_cat=jnp.array(i_cat, dtype=jnp.int32),
        item_num=jnp.array(i_num, dtype=jnp.float32),
    )

    model.eval()
    scores = model.predict_proba(inp)
    model.train()

    top_k_indices = jnp.argsort(scores)[::-1][:k]
    return top_k_indices.tolist()


def _score_full_catalog_graph(
    model: nnx.Module,
    user_idx: int,
    k: int = 12,
) -> list[int]:
    """Graph-based scoring: propagation → dot product with all items."""
    model.eval()
    user_embeds, item_embeds = model.get_all_embeddings()
    model.train()

    u_embed = user_embeds[user_idx]  # (d,)
    scores = item_embeds @ u_embed  # (n_items,) dot product

    top_k_indices = jnp.argsort(scores)[::-1][:k]
    return top_k_indices.tolist()


def _score_full_catalog_din(
    model: nnx.Module,
    user_idx: int,
    user_features: dict[str, np.ndarray],
    item_features: dict[str, np.ndarray],
    sequences: np.ndarray | None,
    seq_lengths: np.ndarray | None,
    k: int = 12,
) -> list[int]:
    """DIN scoring: iterate items as targets with fixed user history.

    Batches all items, passes user history once per batch.
    """
    assert sequences is not None and seq_lengths is not None
    n_items = item_features["numerical"].shape[0]

    u_cat = np.tile(user_features["categorical"][user_idx], (n_items, 1))
    u_num = np.tile(user_features["numerical"][user_idx], (n_items, 1))
    i_cat = item_features["categorical"]
    i_num = item_features["numerical"]
    hist = np.tile(sequences[user_idx], (n_items, 1))
    h_len = np.full(n_items, seq_lengths[user_idx], dtype=np.int32)

    inp = DINInput(
        user_cat=jnp.array(u_cat, dtype=jnp.int32),
        user_num=jnp.array(u_num, dtype=jnp.float32),
        item_cat=jnp.array(i_cat, dtype=jnp.int32),
        item_num=jnp.array(i_num, dtype=jnp.float32),
        history=jnp.array(hist, dtype=jnp.int32),
        hist_len=jnp.array(h_len, dtype=jnp.int32),
    )

    model.eval()
    scores = model.predict_proba(inp)
    model.train()

    top_k_indices = jnp.argsort(scores)[::-1][:k]
    return top_k_indices.tolist()


def _score_full_catalog_sasrec(
    model: nnx.Module,
    user_idx: int,
    sequences: np.ndarray | None,
    seq_lengths: np.ndarray | None,
    k: int = 12,
) -> list[int]:
    """SASRec scoring: user_embed from history → dot product with all items."""
    assert sequences is not None and seq_lengths is not None

    # Single user: (1, T)
    hist = jnp.array(sequences[user_idx:user_idx + 1], dtype=jnp.int32)
    h_len = jnp.array(seq_lengths[user_idx:user_idx + 1], dtype=jnp.int32)

    model.eval()
    scores = model.score_all_items(SASRecInput(history=hist, hist_len=h_len))  # (1, n_items+1)
    model.train()

    # Remove PAD index (0) from scores, keep items 1..n_items → map back to 0..n_items-1
    item_scores = scores[0, 1:]  # (n_items,)
    top_k_indices = jnp.argsort(item_scores)[::-1][:k]
    return top_k_indices.tolist()


def generate_predictions(
    model: nnx.Module,
    target_user_ids: list[str],
    user_features: dict[str, np.ndarray],
    item_features: dict[str, np.ndarray],
    user_to_idx: dict[str, int],
    idx_to_item: dict[int, str],
    k: int = 12,
    backbone_name: str = "deepfm",
    sequences: np.ndarray | None = None,
    seq_lengths: np.ndarray | None = None,
    batch_size: int = 256,
) -> dict[str, list[str]]:
    """Generate top-K predictions for target users (batched for speed)."""
    spec = get_backbone(backbone_name)

    # Use batched scoring for feature-based models (final full predictions)
    if not spec.needs_graph and not spec.needs_sequence and batch_size > 1:
        return _generate_predictions_batched(
            model, target_user_ids, user_features, item_features,
            user_to_idx, idx_to_item, k, batch_size,
        )

    # Per-user scoring (mid-epoch validation, graph/sequential models)
    predictions: dict[str, list[str]] = {}
    for uid in target_user_ids:
        u_idx = user_to_idx.get(uid)
        if u_idx is None:
            predictions[uid] = []
            continue
        top_item_indices = score_full_catalog(
            model, u_idx, user_features, item_features, k=k, backbone_name=backbone_name,
            sequences=sequences, seq_lengths=seq_lengths,
        )
        predictions[uid] = [idx_to_item[idx] for idx in top_item_indices]
    return predictions


def _generate_predictions_batched(
    model: nnx.Module,
    target_user_ids: list[str],
    user_features: dict[str, np.ndarray],
    item_features: dict[str, np.ndarray],
    user_to_idx: dict[str, int],
    idx_to_item: dict[int, str],
    k: int = 12,
    batch_size: int = 256,
) -> dict[str, list[str]]:
    """Batched full-catalog scoring for feature-based models (DeepFM, DCNv2).

    For each batch of users, broadcasts user features against ALL items
    and scores in a single forward pass: (batch_users * n_items, ...).
    """
    n_items = item_features["categorical"].shape[0]
    item_cat = item_features["categorical"]  # (n_items, 5)
    item_num = item_features["numerical"]    # (n_items, 2)

    # Pre-convert item features to JAX (reused across all user batches)
    item_cat_jax = jnp.array(item_cat, dtype=jnp.int32)
    item_num_jax = jnp.array(item_num, dtype=jnp.float32)

    # Filter valid users
    valid_pairs = [(uid, user_to_idx[uid]) for uid in target_user_ids if uid in user_to_idx]
    invalid_users = {uid for uid in target_user_ids if uid not in user_to_idx}

    model.eval()
    predictions: dict[str, list[str]] = {uid: [] for uid in invalid_users}

    for batch_start in range(0, len(valid_pairs), batch_size):
        batch_pairs = valid_pairs[batch_start:batch_start + batch_size]
        batch_uids = [p[0] for p in batch_pairs]
        batch_idxs = np.array([p[1] for p in batch_pairs])
        n_batch = len(batch_idxs)

        # User features for this batch: (n_batch, d) → repeat for each item
        u_cat = user_features["categorical"][batch_idxs]  # (n_batch, 3)
        u_num = user_features["numerical"][batch_idxs]    # (n_batch, 8)

        # Tile: each user paired with all items → (n_batch * n_items, ...)
        u_cat_tiled = jnp.repeat(jnp.array(u_cat, dtype=jnp.int32), n_items, axis=0)
        u_num_tiled = jnp.repeat(jnp.array(u_num, dtype=jnp.float32), n_items, axis=0)
        i_cat_tiled = jnp.tile(item_cat_jax, (n_batch, 1))
        i_num_tiled = jnp.tile(item_num_jax, (n_batch, 1))

        inp = DeepFMInput(
            user_cat=u_cat_tiled,
            user_num=u_num_tiled,
            item_cat=i_cat_tiled,
            item_num=i_num_tiled,
        )
        scores = model.predict_proba(inp)  # (n_batch * n_items,)
        scores = scores.reshape(n_batch, n_items)  # (n_batch, n_items)

        top_k = jnp.argsort(scores, axis=-1)[:, ::-1][:, :k]  # (n_batch, k)
        top_k_np = np.array(top_k)

        for i, uid in enumerate(batch_uids):
            predictions[uid] = [idx_to_item[int(idx)] for idx in top_k_np[i]]

        if (batch_start // batch_size) % 10 == 0:
            done = batch_start + n_batch
            print(f"  Predictions: {done:,}/{len(valid_pairs):,} users")

    model.train()
    return predictions


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_sample(
    model: nnx.Module,
    data_dir: Path,
    features_dir: Path,
    split: str,
    n_sample_users: int,
    user_features: dict[str, np.ndarray],
    item_features: dict[str, np.ndarray],
    user_to_idx: dict[str, int],
    idx_to_item: dict[int, str],
    k: int = 12,
    seed: int = 42,
    backbone_name: str = "deepfm",
    sequences: np.ndarray | None = None,
    seq_lengths: np.ndarray | None = None,
) -> dict[str, float]:
    """Quick validation on sampled users. Returns metric dict."""
    gt_path = data_dir / f"{split}_ground_truth.json"
    ground_truth = json.loads(gt_path.read_text())

    rng = np.random.default_rng(seed)
    all_gt_users = list(ground_truth.keys())
    valid_users = [u for u in all_gt_users if u in user_to_idx]
    n_sample = min(n_sample_users, len(valid_users))
    sample_users = rng.choice(valid_users, size=n_sample, replace=False).tolist()

    predictions = generate_predictions(
        model, sample_users, user_features, item_features, user_to_idx, idx_to_item, k,
        backbone_name=backbone_name, sequences=sequences, seq_lengths=seq_lengths,
        batch_size=1,  # per-user scoring to avoid OOM during validation
    )

    sample_gt = {u: ground_truth[u] for u in sample_users}
    config = EvalConfig(k=k)
    result = evaluate(predictions, sample_gt, config)

    return {
        "map_at_12": result.map_at_k,
        "hr_at_12": result.hr_at_k,
        "ndcg_at_12": result.ndcg_at_k,
        "mrr": result.mrr,
    }


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------


def run_training(
    model_config: DeepFMConfig | DCNv2Config | LightGCNConfig | DINConfig | SASRecConfig,
    train_config: TrainConfig,
    features_dir: Path,
    data_dir: Path,
    model_dir: Path,
    predictions_dir: Path,
    split: str = "val",
    backbone_name: str = "deepfm",
) -> TrainResult:
    """Full training pipeline.

    1. Load features + metadata
    2. Initialize model/optimizer
    3. Training loop (BCE, early stopping on val MAP@12)
    4. Final evaluation → predictions JSON
    5. Optional W&B logging

    Args:
        model_config: Backbone-specific hyperparameters.
        train_config: Training loop settings.
        features_dir: Path to feature .npz files.
        data_dir: Path to preprocessed data (ground truth).
        model_dir: Path to save model checkpoints.
        predictions_dir: Path to save prediction JSON.
        split: Validation split name ("val" or "test").
        backbone_name: Model backbone ("deepfm", "dcnv2", "lightgcn", "din", "sasrec").

    Returns:
        TrainResult with best metrics and timing.
    """
    spec = get_backbone(backbone_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # --- Device setup + Mesh ---
    devices = jax.devices()
    n_devices = len(devices)
    print(f"[train] JAX devices: {n_devices} ({devices[0].platform})")
    print(f"[train] Backbone: {backbone_name}")

    mesh = Mesh(np.array(devices).reshape(-1), ("data",))
    data_sharding = NamedSharding(mesh, PartitionSpec("data"))

    # --- Load data ---
    print("[train] Loading features...")
    feature_meta = load_feature_meta(features_dir)
    user_features = load_user_features(features_dir)
    item_features = load_item_features(features_dir)
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = load_id_maps(features_dir)

    # Load sequences for sequential models
    sequences: np.ndarray | None = None
    seq_lengths: np.ndarray | None = None
    if spec.needs_sequence:
        from src.features.sequences import load_sequences

        seq_data = load_sequences(features_dir)
        sequences = seq_data["sequences"]
        seq_lengths = seq_data["seq_lengths"]
        print(f"  Sequences: max_len={sequences.shape[1]}, "
              f"users_with_seq={int(np.sum(seq_lengths > 0)):,}")

    # --- Normalize numerical features (z-score, in-memory) ---
    for feat_dict, name in [(user_features, "user"), (item_features, "item")]:
        num = feat_dict["numerical"]
        mu = num.mean(axis=0, keepdims=True)
        sigma = num.std(axis=0, keepdims=True) + 1e-8
        feat_dict["numerical"] = ((num - mu) / sigma).astype(np.float32)
        print(f"  {name} numerical normalized: mean≈{feat_dict['numerical'].mean():.4f}, "
              f"std≈{feat_dict['numerical'].std():.4f}")

    print(f"  Users: {feature_meta['n_users']:,}")
    print(f"  Items: {feature_meta['n_items']:,}")
    print(f"  Training pairs: {feature_meta['n_train_pairs']:,}")

    # --- Initialize model ---
    print("[train] Initializing model...")
    model, optimizer = create_train_state(
        backbone_name, model_config, train_config, feature_meta, features_dir
    )

    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
    print(f"  Parameters: {n_params:,}")

    # --- W&B init ---
    wandb_run = None
    if train_config.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=train_config.wandb_project,
                config={
                    "model": model_config._asdict(),
                    "train": train_config._asdict(),
                    "n_params": n_params,
                    "n_devices": n_devices,
                    "backbone": backbone_name,
                },
                tags=[backbone_name, "baseline", f"devices-{n_devices}"],
            )
            print(f"  W&B run: {wandb_run.name}")
        except Exception as e:
            print(f"  W&B init failed: {e}. Continuing without logging.")
            wandb_run = None

    # --- Data loader ---
    print("[train] Building data loader...")
    n_steps = steps_per_epoch(features_dir, train_config.batch_size)
    print(f"  Steps per epoch: {n_steps:,}")

    # Build correct train step for this backbone
    step_fn = make_train_step(backbone_name, model_config)

    # Determine batch dtype conversion based on backbone
    is_graph = spec.needs_graph
    is_sequential = spec.needs_sequence

    # --- Training loop ---
    best_val_metrics: dict[str, float] = {"map_at_12": 0.0, "hr_at_12": 0.0, "ndcg_at_12": 0.0, "mrr": 0.0}
    best_epoch = 0
    patience_counter = 0
    global_step = 0
    start_time = time.time()

    print(f"[train] Starting training (max {train_config.max_epochs} epochs, patience {train_config.patience})...")

    for epoch in range(train_config.max_epochs):
        epoch_losses = []
        model.train()

        loader = create_train_loader(
            features_dir=features_dir,
            batch_size=train_config.batch_size,
            seed=train_config.random_seed + epoch,
            worker_count=train_config.num_workers,
            prefetch_buffer_size=train_config.prefetch_buffer_size,
            backbone_name=backbone_name,
        )

        for batch in loader:
            # numpy → jax.Array + data sharding
            if is_graph:
                jax_batch = jax.device_put(
                    {
                        "user_idx": jnp.array(batch["user_idx"], dtype=jnp.int32),
                        "item_idx": jnp.array(batch["item_idx"], dtype=jnp.int32),
                        "labels": jnp.array(batch["labels"], dtype=jnp.float32),
                    },
                    data_sharding,
                )
            elif is_sequential:
                # Sequential models: int32 for indices/sequences, float32 for num/labels
                def _seq_dtype(k: str) -> jnp.dtype:
                    if k in ("labels",):
                        return jnp.float32
                    if k in ("user_num", "item_num"):
                        return jnp.float32
                    return jnp.int32  # cat, history, hist_len, target_item_seq_idx

                jax_batch = jax.device_put(
                    {k: jnp.array(v, dtype=_seq_dtype(k)) for k, v in batch.items()},
                    data_sharding,
                )
            else:
                jax_batch = jax.device_put(
                    {
                        k: jnp.array(v, dtype=jnp.int32 if "cat" in k else jnp.float32)
                        for k, v in batch.items()
                    },
                    data_sharding,
                )

            loss = step_fn(model, optimizer, jax_batch)
            loss_val = float(loss)
            epoch_losses.append(loss_val)
            global_step += 1

            # Logging
            if global_step % train_config.log_every_n_steps == 0:
                avg_loss = np.mean(epoch_losses[-train_config.log_every_n_steps :])
                print(f"  [step {global_step:,}] loss={avg_loss:.6f}")
                if wandb_run is not None:
                    import wandb

                    wandb.log({"train/loss": avg_loss, "step": global_step})

            # Mid-epoch validation
            if global_step % train_config.val_every_n_steps == 0:
                print(f"  [step {global_step:,}] Running mid-epoch validation...")
                val_metrics = validate_sample(
                    model,
                    data_dir,
                    features_dir,
                    split,
                    train_config.val_sample_users,
                    user_features,
                    item_features,
                    user_to_idx,
                    idx_to_item,
                    seed=train_config.random_seed,
                    backbone_name=backbone_name,
                    sequences=sequences,
                    seq_lengths=seq_lengths,
                )
                print(
                    f"    MAP@12={val_metrics['map_at_12']:.6f} "
                    f"HR@12={val_metrics['hr_at_12']:.6f}"
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()} | {"step": global_step})

        # End of epoch
        avg_epoch_loss = float(np.mean(epoch_losses))
        print(f"\n[epoch {epoch + 1}/{train_config.max_epochs}] avg_loss={avg_epoch_loss:.6f}")

        # Epoch-end validation
        print(f"  Running epoch-end validation ({train_config.val_sample_users} users)...")
        val_metrics = validate_sample(
            model,
            data_dir,
            features_dir,
            split,
            train_config.val_sample_users,
            user_features,
            item_features,
            user_to_idx,
            idx_to_item,
            seed=train_config.random_seed + epoch,
            backbone_name=backbone_name,
            sequences=sequences,
            seq_lengths=seq_lengths,
        )
        print(
            f"  MAP@12={val_metrics['map_at_12']:.6f} "
            f"HR@12={val_metrics['hr_at_12']:.6f} "
            f"NDCG@12={val_metrics['ndcg_at_12']:.6f} "
            f"MRR={val_metrics['mrr']:.6f}"
        )

        if wandb_run is not None:
            import wandb

            wandb.log(
                {f"val/{k}": v for k, v in val_metrics.items()}
                | {"train/epoch_loss": avg_epoch_loss, "epoch": epoch + 1}
            )

        # Early stopping check
        if val_metrics["map_at_12"] > best_val_metrics["map_at_12"]:
            best_val_metrics = val_metrics
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"  *** New best MAP@12: {best_val_metrics['map_at_12']:.6f} ***")

            _save_model_state(model, model_dir / f"{backbone_name}_best")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{train_config.patience})")
            if patience_counter >= train_config.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    total_time = time.time() - start_time
    print(f"\n[train] Training complete in {total_time:.1f}s")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best MAP@12: {best_val_metrics['map_at_12']:.6f}")

    # --- Final predictions ---
    print("[train] Generating final predictions...")

    _load_model_state(model, model_dir / f"{backbone_name}_best")

    gt_path = data_dir / f"{split}_ground_truth.json"
    ground_truth = json.loads(gt_path.read_text())
    target_users = list(ground_truth.keys())

    predictions = generate_predictions(
        model, target_users, user_features, item_features, user_to_idx, idx_to_item,
        backbone_name=backbone_name, sequences=sequences, seq_lengths=seq_lengths,
        batch_size=4,  # 4 users × 105K items per batch (memory-safe)
    )

    pred_path = predictions_dir / f"{backbone_name}_{split}.json"
    pred_path.write_text(json.dumps(predictions, ensure_ascii=False))
    print(f"  Predictions saved: {pred_path}")
    print(f"  Users with predictions: {sum(1 for v in predictions.values() if v):,}")

    # Final eval on all target users
    print("[train] Final evaluation (all target users)...")
    config = EvalConfig(k=12)
    final_result = evaluate(predictions, ground_truth, config)
    final_metrics = {
        "map_at_12": final_result.map_at_k,
        "hr_at_12": final_result.hr_at_k,
        "ndcg_at_12": final_result.ndcg_at_k,
        "mrr": final_result.mrr,
    }
    print(
        f"  MAP@12={final_metrics['map_at_12']:.6f} "
        f"HR@12={final_metrics['hr_at_12']:.6f} "
        f"NDCG@12={final_metrics['ndcg_at_12']:.6f} "
        f"MRR={final_metrics['mrr']:.6f}"
    )

    metrics_path = model_dir / f"{backbone_name}_metrics.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2))

    if wandb_run is not None:
        import wandb

        wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
        wandb.finish()

    return TrainResult(
        model_dir=model_dir,
        best_epoch=best_epoch,
        best_val_map_at_12=final_metrics["map_at_12"],
        best_val_hr_at_12=final_metrics["hr_at_12"],
        best_val_ndcg_at_12=final_metrics["ndcg_at_12"],
        best_val_mrr=final_metrics["mrr"],
        total_train_steps=global_step,
        total_train_time_seconds=total_time,
        n_devices=n_devices,
    )


# ---------------------------------------------------------------------------
# KAR Training Pipeline (Phase 4)
# ---------------------------------------------------------------------------


def create_kar_train_state(
    backbone_name: str,
    model_config: DeepFMConfig | DCNv2Config | LightGCNConfig | DINConfig | SASRecConfig,
    kar_config: KARConfig,
    train_config: TrainConfig,
    feature_meta: dict[str, Any],
    features_dir: Path,
) -> tuple[Any, nnx.Optimizer]:
    """Initialize KARModel + optimizer.

    Creates backbone first, then wraps in KARModel.
    Returns (KARModel, Optimizer).
    """
    from src.kar.hybrid import KARModel, compute_d_backbone

    backbone, _ = create_train_state(
        backbone_name, model_config, train_config, feature_meta, features_dir
    )

    rngs = nnx.Rngs(
        params=train_config.random_seed + 100,
        dropout=train_config.random_seed + 101,
    )

    d_backbone = compute_d_backbone(backbone_name, backbone)
    kar_model = KARModel(
        backbone=backbone,
        backbone_name=backbone_name,
        kar_config=kar_config,
        d_backbone=d_backbone,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(
        kar_model, optax.adam(learning_rate=train_config.learning_rate), wrt=nnx.Param
    )
    return kar_model, optimizer


def _build_kar_input(batch: dict[str, jax.Array], backbone_name: str) -> Any:
    """Construct KARInput from a batched dict."""
    from src.kar.hybrid import KARInput

    spec = get_backbone(backbone_name)

    if spec.needs_graph:
        base_input = LightGCNInput(
            user_idx=batch["user_idx"],
            item_idx=batch["item_idx"],
        )
        return KARInput(
            base_input=base_input,
            h_fact=batch["h_fact"],
            h_reason=batch["h_reason"],
        )

    if backbone_name == "din":
        base_input = DINInput(
            user_cat=batch["user_cat"],
            user_num=batch["user_num"],
            item_cat=batch["item_cat"],
            item_num=batch["item_num"],
            history=batch["history"],
            hist_len=batch["hist_len"],
        )
        return KARInput(
            base_input=base_input,
            h_fact=batch["h_fact"],
            h_reason=batch["h_reason"],
        )

    if backbone_name == "sasrec":
        base_input = SASRecInput(
            history=batch["history"],
            hist_len=batch["hist_len"],
        )
        return KARInput(
            base_input=base_input,
            h_fact=batch["h_fact"],
            h_reason=batch["h_reason"],
            target_item_idx=batch["target_item_seq_idx"],
        )

    # Feature-based: deepfm, dcnv2
    base_input = DeepFMInput(
        user_cat=batch["user_cat"],
        user_num=batch["user_num"],
        item_cat=batch["item_cat"],
        item_num=batch["item_num"],
    )
    return KARInput(
        base_input=base_input,
        h_fact=batch["h_fact"],
        h_reason=batch["h_reason"],
    )


def make_kar_train_step_stage1(backbone_name: str) -> Any:
    """Stage 1: Backbone pre-train with BCE only (experts bypassed)."""

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
    ) -> jax.Array:
        def loss_fn(model: nnx.Module) -> jax.Array:
            kar_input = _build_kar_input(batch, backbone_name)
            logits = model(kar_input)
            return binary_cross_entropy(logits, batch["labels"])

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def make_kar_train_step_stage2(
    backbone_name: str, align_weight: float, diversity_weight: float
) -> Any:
    """Stage 2: Expert adaptor (backbone frozen via stop_gradient).

    Loss = align_loss + diversity_loss (no BCE).
    """

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
    ) -> jax.Array:
        def loss_fn(model: nnx.Module) -> jax.Array:
            kar_input = _build_kar_input(batch, backbone_name)
            logits, intermediates = model.forward_with_intermediates(kar_input)

            x_item_proj = model.get_align_targets(
                jax.lax.stop_gradient(intermediates["x_backbone_flat"])
            )
            # For user align: use the user portion of backbone flat
            # Simplified: use same backbone flat projection for both
            x_user_proj = x_item_proj  # both align to backbone space

            total, _ = kar_total_loss(
                logits, batch["labels"],
                intermediates["e_fact"], jax.lax.stop_gradient(x_item_proj),
                intermediates["e_reason"], jax.lax.stop_gradient(x_user_proj),
                align_weight=align_weight,
                diversity_weight=diversity_weight,
                include_rec_loss=False,
            )
            return total

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def make_kar_train_step_stage3(
    backbone_name: str, align_weight: float, diversity_weight: float
) -> Any:
    """Stage 3: End-to-end (BCE + align + diversity, all unfrozen)."""

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
    ) -> jax.Array:
        def loss_fn(model: nnx.Module) -> jax.Array:
            kar_input = _build_kar_input(batch, backbone_name)
            logits, intermediates = model.forward_with_intermediates(kar_input)

            x_proj = model.get_align_targets(intermediates["x_backbone_flat"])

            total, _ = kar_total_loss(
                logits, batch["labels"],
                intermediates["e_fact"], x_proj,
                intermediates["e_reason"], x_proj,
                align_weight=align_weight,
                diversity_weight=diversity_weight,
                include_rec_loss=True,
            )
            return total

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def score_full_catalog_kar(
    model: Any,
    user_idx: int,
    user_features: dict[str, np.ndarray],
    item_features: dict[str, np.ndarray],
    item_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
    k: int = 12,
    backbone_name: str = "deepfm",
    sequences: np.ndarray | None = None,
    seq_lengths: np.ndarray | None = None,
) -> list[int]:
    """Score single user × full catalog using KARModel.

    For feature-based backbones, broadcasts user against all items.
    """
    from src.kar.hybrid import KARInput

    n_items = item_embeddings.shape[0]
    h_fact_all = jnp.array(item_embeddings, dtype=jnp.float32)
    h_reason_user = jnp.tile(
        jnp.array(user_embeddings[user_idx], dtype=jnp.float32), (n_items, 1)
    )

    spec = get_backbone(backbone_name)

    if spec.needs_graph:
        base_input = LightGCNInput(
            user_idx=jnp.full((n_items,), user_idx, dtype=jnp.int32),
            item_idx=jnp.arange(n_items, dtype=jnp.int32),
        )
        kar_input = KARInput(base_input=base_input, h_fact=h_fact_all, h_reason=h_reason_user)
    elif backbone_name == "sasrec":
        assert sequences is not None and seq_lengths is not None
        hist = jnp.tile(jnp.array(sequences[user_idx], dtype=jnp.int32), (n_items, 1))
        h_len = jnp.full((n_items,), seq_lengths[user_idx], dtype=jnp.int32)
        base_input = SASRecInput(history=hist, hist_len=h_len)
        target_idxs = jnp.arange(1, n_items + 1, dtype=jnp.int32)
        kar_input = KARInput(
            base_input=base_input, h_fact=h_fact_all, h_reason=h_reason_user,
            target_item_idx=target_idxs,
        )
    elif backbone_name == "din":
        assert sequences is not None and seq_lengths is not None
        u_cat = np.tile(user_features["categorical"][user_idx], (n_items, 1))
        u_num = np.tile(user_features["numerical"][user_idx], (n_items, 1))
        hist = np.tile(sequences[user_idx], (n_items, 1))
        h_len = np.full(n_items, seq_lengths[user_idx], dtype=np.int32)
        base_input = DINInput(
            user_cat=jnp.array(u_cat, dtype=jnp.int32),
            user_num=jnp.array(u_num, dtype=jnp.float32),
            item_cat=jnp.array(item_features["categorical"], dtype=jnp.int32),
            item_num=jnp.array(item_features["numerical"], dtype=jnp.float32),
            history=jnp.array(hist, dtype=jnp.int32),
            hist_len=jnp.array(h_len, dtype=jnp.int32),
        )
        kar_input = KARInput(base_input=base_input, h_fact=h_fact_all, h_reason=h_reason_user)
    else:
        # feature-based: deepfm, dcnv2
        u_cat = np.tile(user_features["categorical"][user_idx], (n_items, 1))
        u_num = np.tile(user_features["numerical"][user_idx], (n_items, 1))
        base_input = DeepFMInput(
            user_cat=jnp.array(u_cat, dtype=jnp.int32),
            user_num=jnp.array(u_num, dtype=jnp.float32),
            item_cat=jnp.array(item_features["categorical"], dtype=jnp.int32),
            item_num=jnp.array(item_features["numerical"], dtype=jnp.float32),
        )
        kar_input = KARInput(base_input=base_input, h_fact=h_fact_all, h_reason=h_reason_user)

    model.eval()
    logits = model(kar_input)
    scores = jax.nn.sigmoid(logits)
    model.train()

    top_k = jnp.argsort(scores)[::-1][:k]
    return top_k.tolist()


def run_kar_training(
    backbone_name: str,
    model_config: DeepFMConfig | DCNv2Config | LightGCNConfig | DINConfig | SASRecConfig,
    kar_config: KARConfig,
    train_config: TrainConfig,
    features_dir: Path,
    embeddings_dir: Path,
    data_dir: Path,
    model_dir: Path,
    predictions_dir: Path,
    split: str = "val",
) -> TrainResult:
    """Full KAR 3-stage training pipeline.

    Stage 1: backbone pre-train (BCE only, stage1_epochs)
    Stage 2: expert adaptor (align+div, backbone frozen, stage2_epochs)
    Stage 3: end-to-end (BCE+align+div, LR×stage3_lr_factor, stage3_epochs)
    """
    from src.kar.embedding_index import build_aligned_embeddings

    spec = get_backbone(backbone_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    devices = jax.devices()
    n_devices = len(devices)
    mesh = Mesh(np.array(devices).reshape(-1), ("data",))
    data_sharding = NamedSharding(mesh, PartitionSpec("data"))
    print(f"[kar-train] JAX devices: {n_devices} ({devices[0].platform})")
    print(f"[kar-train] Backbone: {backbone_name}, Gating: {kar_config.gating.variant}, "
          f"Fusion: {kar_config.fusion.variant}")

    # Load features
    feature_meta = load_feature_meta(features_dir)
    user_features = load_user_features(features_dir)
    item_features = load_item_features(features_dir)
    user_to_idx, _, _, idx_to_item = load_id_maps(features_dir)

    sequences: np.ndarray | None = None
    seq_lengths: np.ndarray | None = None
    if spec.needs_sequence:
        from src.features.sequences import load_sequences
        seq_data = load_sequences(features_dir)
        sequences = seq_data["sequences"]
        seq_lengths = seq_data["seq_lengths"]

    # Load aligned BGE embeddings
    print("[kar-train] Loading aligned BGE embeddings...")
    item_emb, user_emb = build_aligned_embeddings(features_dir, embeddings_dir)
    print(f"  Items: {item_emb.shape}, Users: {user_emb.shape}")

    # Initialize KARModel
    print("[kar-train] Initializing KARModel...")
    kar_model, optimizer = create_kar_train_state(
        backbone_name, model_config, kar_config, train_config, feature_meta, features_dir
    )
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(kar_model)))
    print(f"  Total parameters: {n_params:,}")

    # Batch dtype routing
    is_graph = spec.needs_graph
    is_sequential = spec.needs_sequence

    def _to_jax_batch(batch: dict) -> dict[str, jax.Array]:
        """Convert numpy batch to JAX arrays with correct dtypes."""
        result = {}
        for k, v in batch.items():
            if k in ("labels", "user_num", "item_num", "h_fact", "h_reason"):
                result[k] = jnp.array(v, dtype=jnp.float32)
            else:
                result[k] = jnp.array(v, dtype=jnp.int32)
        return result

    best_val_metrics = {"map_at_12": 0.0, "hr_at_12": 0.0, "ndcg_at_12": 0.0, "mrr": 0.0}
    best_epoch = 0
    global_step = 0
    start_time = time.time()

    # W&B init
    wandb_run = None
    if train_config.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=train_config.wandb_project,
                config={
                    "backbone": backbone_name,
                    "model": model_config._asdict(),
                    "kar": kar_config._asdict(),
                    "train": train_config._asdict(),
                },
                tags=[backbone_name, "kar", kar_config.gating.variant, kar_config.fusion.variant],
            )
        except Exception:
            wandb_run = None

    # ===== Stage 1: Backbone Pre-train =====
    print(f"\n[kar-train] === Stage 1: Backbone Pre-train ({kar_config.stage1_epochs} epochs) ===")
    step_fn = make_kar_train_step_stage1(backbone_name)

    for epoch in range(kar_config.stage1_epochs):
        kar_model.train()
        loader = create_train_loader(
            features_dir, train_config.batch_size,
            seed=train_config.random_seed + epoch,
            worker_count=train_config.num_workers,
            prefetch_buffer_size=train_config.prefetch_buffer_size,
            backbone_name=backbone_name,
            use_kar=True,
            item_embeddings=item_emb,
            user_embeddings=user_emb,
        )
        epoch_losses = []
        for batch in loader:
            jax_batch = jax.device_put(_to_jax_batch(batch), data_sharding)
            loss = step_fn(kar_model, optimizer, jax_batch)
            epoch_losses.append(float(loss))
            global_step += 1
            if global_step % train_config.log_every_n_steps == 0:
                print(f"  [S1 step {global_step:,}] loss={np.mean(epoch_losses[-500:]):.6f}")

        print(f"  [S1 epoch {epoch+1}] avg_loss={np.mean(epoch_losses):.6f}")

    # ===== Stage 2: Expert Adaptor (backbone frozen) =====
    print(f"\n[kar-train] === Stage 2: Expert Adaptor ({kar_config.stage2_epochs} epochs) ===")
    # Recreate optimizer for expert params only
    optimizer = nnx.Optimizer(
        kar_model, optax.adam(learning_rate=train_config.learning_rate), wrt=nnx.Param
    )
    step_fn = make_kar_train_step_stage2(
        backbone_name, kar_config.align_weight, kar_config.diversity_weight
    )

    for epoch in range(kar_config.stage2_epochs):
        kar_model.train()
        loader = create_train_loader(
            features_dir, train_config.batch_size,
            seed=train_config.random_seed + kar_config.stage1_epochs + epoch,
            worker_count=train_config.num_workers,
            prefetch_buffer_size=train_config.prefetch_buffer_size,
            backbone_name=backbone_name,
            use_kar=True,
            item_embeddings=item_emb,
            user_embeddings=user_emb,
        )
        epoch_losses = []
        for batch in loader:
            jax_batch = jax.device_put(_to_jax_batch(batch), data_sharding)
            loss = step_fn(kar_model, optimizer, jax_batch)
            epoch_losses.append(float(loss))
            global_step += 1
            if global_step % train_config.log_every_n_steps == 0:
                print(f"  [S2 step {global_step:,}] loss={np.mean(epoch_losses[-500:]):.6f}")

        print(f"  [S2 epoch {epoch+1}] avg_loss={np.mean(epoch_losses):.6f}")

    # ===== Stage 3: End-to-End =====
    stage3_lr = train_config.learning_rate * kar_config.stage3_lr_factor
    print(f"\n[kar-train] === Stage 3: End-to-End ({kar_config.stage3_epochs} epochs, "
          f"LR={stage3_lr:.1e}) ===")
    optimizer = nnx.Optimizer(
        kar_model, optax.adam(learning_rate=stage3_lr), wrt=nnx.Param
    )
    step_fn = make_kar_train_step_stage3(
        backbone_name, kar_config.align_weight, kar_config.diversity_weight
    )

    patience_counter = 0
    total_s3_epochs = kar_config.stage1_epochs + kar_config.stage2_epochs

    for epoch in range(kar_config.stage3_epochs):
        kar_model.train()
        loader = create_train_loader(
            features_dir, train_config.batch_size,
            seed=train_config.random_seed + total_s3_epochs + epoch,
            worker_count=train_config.num_workers,
            prefetch_buffer_size=train_config.prefetch_buffer_size,
            backbone_name=backbone_name,
            use_kar=True,
            item_embeddings=item_emb,
            user_embeddings=user_emb,
        )
        epoch_losses = []
        for batch in loader:
            jax_batch = jax.device_put(_to_jax_batch(batch), data_sharding)
            loss = step_fn(kar_model, optimizer, jax_batch)
            epoch_losses.append(float(loss))
            global_step += 1
            if global_step % train_config.log_every_n_steps == 0:
                print(f"  [S3 step {global_step:,}] loss={np.mean(epoch_losses[-500:]):.6f}")

        avg_loss = float(np.mean(epoch_losses))
        abs_epoch = total_s3_epochs + epoch + 1
        print(f"  [S3 epoch {epoch+1}] avg_loss={avg_loss:.6f}")

        # Validation
        gt_path = data_dir / f"{split}_ground_truth.json"
        ground_truth = json.loads(gt_path.read_text())
        rng = np.random.default_rng(train_config.random_seed + epoch)
        valid_users = [u for u in ground_truth if u in user_to_idx]
        sample_users = rng.choice(
            valid_users, size=min(train_config.val_sample_users, len(valid_users)), replace=False
        ).tolist()

        predictions: dict[str, list[str]] = {}
        for uid in sample_users:
            u_idx = user_to_idx[uid]
            top_items = score_full_catalog_kar(
                kar_model, u_idx, user_features, item_features,
                item_emb, user_emb, k=12, backbone_name=backbone_name,
                sequences=sequences, seq_lengths=seq_lengths,
            )
            predictions[uid] = [idx_to_item[i] for i in top_items]

        sample_gt = {u: ground_truth[u] for u in sample_users}
        val_result = evaluate(predictions, sample_gt, EvalConfig(k=12))
        val_metrics = {
            "map_at_12": val_result.map_at_k,
            "hr_at_12": val_result.hr_at_k,
            "ndcg_at_12": val_result.ndcg_at_k,
            "mrr": val_result.mrr,
        }
        print(f"  MAP@12={val_metrics['map_at_12']:.6f} HR@12={val_metrics['hr_at_12']:.6f}")

        if wandb_run:
            import wandb
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()} | {
                "train/loss": avg_loss, "epoch": abs_epoch, "stage": 3,
            })

        if val_metrics["map_at_12"] > best_val_metrics["map_at_12"]:
            best_val_metrics = val_metrics
            best_epoch = abs_epoch
            patience_counter = 0
            _save_model_state(kar_model, model_dir / f"kar_{backbone_name}_best")
            print(f"  *** New best MAP@12: {best_val_metrics['map_at_12']:.6f} ***")
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                print(f"  Early stopping at stage 3 epoch {epoch+1}")
                break

    total_time = time.time() - start_time
    print(f"\n[kar-train] Training complete in {total_time:.1f}s")
    print(f"  Best epoch: {best_epoch}, Best MAP@12: {best_val_metrics['map_at_12']:.6f}")

    if wandb_run:
        import wandb
        wandb.finish()

    return TrainResult(
        model_dir=model_dir,
        best_epoch=best_epoch,
        best_val_map_at_12=best_val_metrics["map_at_12"],
        best_val_hr_at_12=best_val_metrics["hr_at_12"],
        best_val_ndcg_at_12=best_val_metrics["ndcg_at_12"],
        best_val_mrr=best_val_metrics["mrr"],
        total_train_steps=global_step,
        total_train_time_seconds=total_time,
        n_devices=n_devices,
    )


# ---------------------------------------------------------------------------
# Model Save/Load (simple numpy-based)
# ---------------------------------------------------------------------------


def _save_model_state(model: nnx.Module, path: Path) -> None:
    """Save model parameters as .npz."""
    path.mkdir(parents=True, exist_ok=True)
    state = nnx.state(model)
    flat_state = jax.tree.leaves_with_path(state)
    save_dict = {}
    for key_path, leaf in flat_state:
        key_str = "/".join(str(k) for k in key_path)
        # PRNGKey arrays cannot be converted directly; extract underlying data
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            save_dict[key_str] = np.array(jax.random.key_data(leaf))
        else:
            save_dict[key_str] = np.array(leaf)
    np.savez(path / "params.npz", **save_dict)


def _load_model_state(model: nnx.Module, path: Path) -> None:
    """Load model parameters from .npz."""
    npz = np.load(path / "params.npz")
    state = nnx.state(model)

    def _restore_leaf(key_path: tuple, leaf: Any) -> Any:
        key_str = "/".join(str(k) for k in key_path)
        if key_str not in npz:
            return leaf
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            return jax.random.wrap_key_data(npz[key_str])
        return jnp.array(npz[key_str])

    restored = jax.tree.map_with_path(_restore_leaf, state)
    nnx.update(model, restored)
