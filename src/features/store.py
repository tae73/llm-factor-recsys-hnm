"""Feature store: save/load feature arrays and metadata.

Output directory layout:
    data/features/
    ├── train_pairs.npz       # user_idx, item_idx, labels
    ├── user_features.npz     # numerical, categorical
    ├── item_features.npz     # numerical, categorical
    ├── feature_meta.json     # feature names, vocab sizes, stats
    ├── id_maps.json          # user↔idx, item↔idx bidirectional
    └── cat_vocab.json        # categorical vocabulary dictionaries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_features(
    output_dir: Path,
    train_pairs: dict[str, np.ndarray],
    user_features_npz: dict[str, np.ndarray],
    item_features_npz: dict[str, np.ndarray],
    feature_meta: dict[str, Any],
    id_maps: dict[str, Any],
    cat_vocab: dict[str, Any],
) -> None:
    """Save all feature engineering outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_dir / "train_pairs.npz", **train_pairs)
    np.savez_compressed(output_dir / "user_features.npz", **user_features_npz)
    np.savez_compressed(output_dir / "item_features.npz", **item_features_npz)

    (output_dir / "feature_meta.json").write_text(
        json.dumps(feature_meta, indent=2, ensure_ascii=False)
    )
    (output_dir / "id_maps.json").write_text(
        json.dumps(id_maps, ensure_ascii=False)
    )
    (output_dir / "cat_vocab.json").write_text(
        json.dumps(cat_vocab, indent=2, ensure_ascii=False)
    )

    print(f"  train_pairs.npz: {len(train_pairs['labels']):,} pairs")
    print(f"  user_features.npz: {user_features_npz['numerical'].shape}")
    print(f"  item_features.npz: {item_features_npz['numerical'].shape}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_train_pairs(data_dir: Path) -> dict[str, np.ndarray]:
    """Load training pairs: user_idx, item_idx, labels."""
    npz = np.load(data_dir / "train_pairs.npz")
    return {
        "user_idx": npz["user_idx"],
        "item_idx": npz["item_idx"],
        "labels": npz["labels"],
    }


def load_user_features(data_dir: Path) -> dict[str, np.ndarray]:
    """Load user features: numerical (n_users, 8), categorical (n_users, 3)."""
    npz = np.load(data_dir / "user_features.npz")
    return {
        "numerical": npz["numerical"],
        "categorical": npz["categorical"],
    }


def load_item_features(data_dir: Path) -> dict[str, np.ndarray]:
    """Load item features: numerical (n_items, 2), categorical (n_items, 5)."""
    npz = np.load(data_dir / "item_features.npz")
    return {
        "numerical": npz["numerical"],
        "categorical": npz["categorical"],
    }


def load_feature_meta(data_dir: Path) -> dict[str, Any]:
    """Load feature metadata (names, vocab sizes, stats)."""
    return json.loads((data_dir / "feature_meta.json").read_text())


def load_id_maps(
    data_dir: Path,
) -> tuple[dict[str, int], dict[int, str], dict[str, int], dict[int, str]]:
    """Load bidirectional ID maps: user↔idx, item↔idx.

    Returns (user_to_idx, idx_to_user, item_to_idx, idx_to_item).
    """
    raw = json.loads((data_dir / "id_maps.json").read_text())
    user_to_idx = raw["user_to_idx"]
    idx_to_user = {int(k): v for k, v in raw["idx_to_user"].items()}
    item_to_idx = raw["item_to_idx"]
    idx_to_item = {int(k): v for k, v in raw["idx_to_item"].items()}
    return user_to_idx, idx_to_user, item_to_idx, idx_to_item


def load_cat_vocab(data_dir: Path) -> dict[str, Any]:
    """Load categorical vocabulary dictionaries."""
    return json.loads((data_dir / "cat_vocab.json").read_text())
