"""Pre-store: offline computation of augmented expert vectors.

Pre-computes Expert outputs for all items and users, saving as .npz
for fast inference without re-running Expert MLPs at serving time.

Output:
    item_expert.npz: (n_items, d_rec) float32 — factual expert outputs
    user_expert.npz: (n_users, d_rec) float32 — reasoning expert outputs
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def compute_prestore(
    model: nnx.Module,
    item_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
    output_dir: Path,
    batch_size: int = 4096,
) -> tuple[Path, Path]:
    """Pre-compute Expert outputs for all items and users.

    Args:
        model: Trained KARModel with factual_expert and reasoning_expert.
        item_embeddings: (n_items, 768) aligned item BGE embeddings.
        user_embeddings: (n_users, 768) aligned user BGE embeddings.
        output_dir: Directory to save .npz files.
        batch_size: Processing batch size.

    Returns:
        (item_expert_path, user_expert_path) tuple.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Pre-compute item expert outputs
    item_expert_path = output_dir / "item_expert.npz"
    item_outputs = _batch_expert_forward(
        model.factual_expert, item_embeddings, batch_size
    )
    np.savez_compressed(item_expert_path, expert_outputs=item_outputs)
    print(f"[prestore] Saved item expert: {item_outputs.shape} → {item_expert_path}")

    # Pre-compute user expert outputs
    user_expert_path = output_dir / "user_expert.npz"
    user_outputs = _batch_expert_forward(
        model.reasoning_expert, user_embeddings, batch_size
    )
    np.savez_compressed(user_expert_path, expert_outputs=user_outputs)
    print(f"[prestore] Saved user expert: {user_outputs.shape} → {user_expert_path}")

    model.train()
    return item_expert_path, user_expert_path


def _batch_expert_forward(
    expert: nnx.Module,
    embeddings: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Run expert forward in batches.

    Args:
        expert: Expert MLP module.
        embeddings: (N, 768) input embeddings.
        batch_size: Processing batch size.

    Returns:
        (N, d_rec) expert outputs as numpy array.
    """
    n_total = embeddings.shape[0]
    outputs = []

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch = jnp.array(embeddings[start:end], dtype=jnp.float32)
        out = expert(batch)
        outputs.append(np.array(out))

    return np.concatenate(outputs, axis=0)


def load_prestore(prestore_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-computed expert outputs.

    Returns:
        (item_expert, user_expert): each (N, d_rec) float32.
    """
    item_data = np.load(prestore_dir / "item_expert.npz")
    user_data = np.load(prestore_dir / "user_expert.npz")
    return item_data["expert_outputs"], user_data["expert_outputs"]
