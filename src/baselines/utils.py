"""Shared utilities for baseline models.

Provides interaction matrix construction and implicit model prediction helpers.
"""

from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from src.config import InteractionData


def build_interaction_matrix(con: duckdb.DuckDBPyConnection, train_path: Path) -> InteractionData:
    """Build a sparse user-item interaction matrix from train transactions.

    Uses DuckDB aggregation to count interactions, then constructs a scipy CSR matrix.
    """
    rows = con.execute(
        f"""
        SELECT customer_id, article_id, COUNT(*) as cnt
        FROM read_parquet('{train_path}')
        GROUP BY customer_id, article_id
        """
    ).fetchall()

    # Build index mappings
    users = sorted(set(r[0] for r in rows))
    items = sorted(set(r[1] for r in rows))

    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    item_to_idx = {it: i for i, it in enumerate(items)}
    idx_to_item = {i: it for it, i in item_to_idx.items()}

    # Build COO matrix
    row_indices = np.array([user_to_idx[r[0]] for r in rows], dtype=np.int32)
    col_indices = np.array([item_to_idx[r[1]] for r in rows], dtype=np.int32)
    values = np.array([r[2] for r in rows], dtype=np.float32)

    matrix = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(users), len(items)),
    ).tocsr()

    return InteractionData(
        matrix=matrix,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
    )


def predict_from_implicit_model(
    model,
    interaction_data: InteractionData,
    user_ids: list[str],
    k: int = 12,
) -> dict[str, list[str]]:
    """Generate top-K predictions using an implicit library model.

    Uses implicit's batch recommend API. Filters out already-purchased items.
    """
    predictions: dict[str, list[str]] = {}
    user_items = interaction_data.matrix

    # Collect valid user indices for batch recommendation
    valid_uids = [uid for uid in user_ids if uid in interaction_data.user_to_idx]
    valid_indices = [interaction_data.user_to_idx[uid] for uid in valid_uids]

    if valid_indices:
        # implicit 0.7+ recommend() accepts array of user ids + full user-item matrix
        all_item_indices, all_scores = model.recommend(
            valid_indices,
            user_items[valid_indices],
            N=k,
            filter_already_liked_items=True,
        )
        for uid, item_indices in zip(valid_uids, all_item_indices):
            predictions[uid] = [
                interaction_data.idx_to_item[int(idx)] for idx in item_indices
            ]

    # Users not in training data get empty predictions
    for uid in user_ids:
        if uid not in predictions:
            predictions[uid] = []

    return predictions
