"""Sequential feature pipeline: build time-ordered item sequences per user.

DuckDB query extracts user purchase histories from train transactions,
truncates to most recent max_seq_len items, and pads shorter sequences with 0.

Output:
    train_sequences.npz:
        sequences: (n_users, max_seq_len) int32 — right-aligned, 0-padded
        seq_lengths: (n_users,) int32 — actual sequence length per user
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from src.config import SequenceConfig


def build_sequences(
    data_dir: Path,
    features_dir: Path,
    config: SequenceConfig = SequenceConfig(),
) -> dict[str, int]:
    """Build time-ordered item sequences per user from train transactions.

    Uses item_to_idx mapping from features_dir/id_maps.json and
    train transactions from data_dir/train_transactions.parquet.

    Item index 0 is reserved as PAD token.

    Args:
        data_dir: Directory with preprocessed Parquet files.
        features_dir: Directory with id_maps.json and output destination.
        config: SequenceConfig with max_seq_len.

    Returns:
        Metadata dict with n_users, n_items, max_seq_len, avg_seq_len.
    """
    # Load ID maps
    id_maps = json.loads((features_dir / "id_maps.json").read_text())
    user_to_idx: dict[str, int] = id_maps["user_to_idx"]
    item_to_idx: dict[str, int] = id_maps["item_to_idx"]
    n_users = len(user_to_idx)

    train_path = str(data_dir / "train_transactions.parquet")

    # Query: per-user time-ordered item sequences
    con = duckdb.connect()
    rows = con.execute(
        """
        SELECT customer_id, LIST(article_id ORDER BY t_dat ASC) AS items
        FROM read_parquet(?)
        GROUP BY customer_id
        """,
        [train_path],
    ).fetchall()
    con.close()

    max_len = config.max_seq_len
    sequences = np.zeros((n_users, max_len), dtype=np.int32)
    seq_lengths = np.zeros(n_users, dtype=np.int32)

    for customer_id, items in rows:
        u_idx = user_to_idx.get(customer_id)
        if u_idx is None:
            continue

        # Map article_ids to indices (1-based; 0 = PAD)
        # item_to_idx already uses 0-based, but we need to shift by +1 for PAD
        item_indices = []
        for aid in items:
            idx = item_to_idx.get(aid)
            if idx is not None:
                item_indices.append(idx + 1)  # +1 so 0 stays PAD

        if not item_indices:
            continue

        # Right-truncate to most recent max_len items
        if len(item_indices) > max_len:
            item_indices = item_indices[-max_len:]

        actual_len = len(item_indices)
        seq_lengths[u_idx] = actual_len
        # Left-pad: items at the end of the array
        sequences[u_idx, max_len - actual_len :] = item_indices

    np.savez_compressed(
        features_dir / "train_sequences.npz",
        sequences=sequences,
        seq_lengths=seq_lengths,
    )

    avg_len = float(np.mean(seq_lengths[seq_lengths > 0])) if np.any(seq_lengths > 0) else 0.0

    print(f"  train_sequences.npz: {n_users:,} users, max_seq_len={max_len}")
    print(f"  Avg sequence length: {avg_len:.1f}")
    print(f"  Users with sequences: {int(np.sum(seq_lengths > 0)):,}")

    return {
        "n_users": n_users,
        "n_items": len(item_to_idx),
        "max_seq_len": max_len,
        "avg_seq_len": round(avg_len, 2),
        "n_users_with_sequences": int(np.sum(seq_lengths > 0)),
    }


def load_sequences(features_dir: Path) -> dict[str, np.ndarray]:
    """Load pre-built sequences.

    Returns:
        dict with 'sequences' (n_users, max_seq_len) and 'seq_lengths' (n_users,).
    """
    npz = np.load(features_dir / "train_sequences.npz")
    return {
        "sequences": npz["sequences"],
        "seq_lengths": npz["seq_lengths"],
    }
