"""Popularity-based baseline recommenders.

Provides global and recent (time-windowed) popularity baselines.
"""

from pathlib import Path

import duckdb


def compute_global_popularity(
    con: duckdb.DuckDBPyConnection,
    train_path: Path,
    k: int = 12,
) -> list[str]:
    """Compute top-K most purchased articles across entire training period."""
    rows = con.execute(
        f"""
        SELECT article_id, COUNT(*) as cnt
        FROM read_parquet('{train_path}')
        GROUP BY article_id
        ORDER BY cnt DESC
        LIMIT {k}
        """
    ).fetchall()
    return [r[0] for r in rows]


def compute_recent_popularity(
    con: duckdb.DuckDBPyConnection,
    train_path: Path,
    k: int = 12,
    window_days: int = 7,
) -> list[str]:
    """Compute top-K most purchased articles in the last N days of training."""
    rows = con.execute(
        f"""
        WITH max_date AS (
            SELECT MAX(t_dat) as md FROM read_parquet('{train_path}')
        )
        SELECT t.article_id, COUNT(*) as cnt
        FROM read_parquet('{train_path}') t, max_date m
        WHERE t.t_dat >= m.md - INTERVAL '{window_days}' DAY
        GROUP BY t.article_id
        ORDER BY cnt DESC
        LIMIT {k}
        """
    ).fetchall()
    return [r[0] for r in rows]


def predict_popularity(
    popular_items: list[str],
    user_ids: list[str],
) -> dict[str, list[str]]:
    """Assign the same popular items to all users."""
    return {uid: list(popular_items) for uid in user_ids}
