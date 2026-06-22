"""Repurchase / recent-popularity / hybrid baselines.

The H&M dominant signal for the *immediate* next period is repurchase + recency:
on the week immediately following ``train_end`` a trivial per-user repurchase
list (own recent items, padded with recent-popular items) scores a
Kaggle-competitive MAP@12 ≈ 0.024 (witnesses/probe_05_*). This module is the
single source of truth for that logic — it reproduces probe_05 exactly so the
hybrid backbone's number is reliable and the discovery gap (the ~96% of
next-week purchases that are NEW items) can be measured against it.
"""

from typing import Iterable, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Recent popularity (fill list)
# ---------------------------------------------------------------------------


def recent_popularity(
    train_txn: pd.DataFrame,
    train_end: str,
    days: int = 14,
    k: int = 12,
) -> list[str]:
    """Top-``k`` articles by purchase count in the last ``days`` of training.

    Matches probe_05's ``recent14`` list: a count-based ranking over the window
    ``(train_end - days, train_end]``. ``train_end`` is the inclusive training
    cut-off (the maximum train date in probe_05).

    Args:
        train_txn: Train transactions with ``article_id`` and ``t_dat`` columns.
        train_end: Training cut-off date (inclusive), e.g. ``"2020-06-30"``.
        days: Window length in days.
        k: Number of articles to return.

    Returns:
        List of ``k`` article_id strings, most popular first.
    """
    tmax = pd.Timestamp(train_end)
    # Normalize t_dat: parquet DATE columns deserialize as datetime.date (object
    # dtype) which cannot be compared to a Timestamp; coerce to datetime64.
    t_dat = pd.to_datetime(train_txn["t_dat"])
    window = train_txn[t_dat > tmax - pd.Timedelta(days=days)]
    return window["article_id"].astype(str).value_counts().head(k).index.tolist()


# ---------------------------------------------------------------------------
# Per-user recent purchased items (repurchase candidates)
# ---------------------------------------------------------------------------


def _last_items(train_txn: pd.DataFrame, k: int = 12) -> "pd.Series":
    """Per-user distinct purchased articles in reverse-recency order (top-``k``).

    Reproduces probe_05's ``last_items``: sort by ``t_dat`` ascending, group per
    customer, reverse the article sequence, drop duplicates keeping the first
    (most-recent) occurrence, truncate to ``k``.
    """
    return (
        train_txn.sort_values("t_dat")
        .assign(
            customer_id=lambda d: d["customer_id"].astype(str),
            article_id=lambda d: d["article_id"].astype(str),
        )
        .groupby("customer_id")["article_id"]
        .apply(lambda s: list(dict.fromkeys(s.tolist()[::-1]))[:k])
    )


# ---------------------------------------------------------------------------
# Repurchase / hybrid prediction
# ---------------------------------------------------------------------------


def repurchase_predict(
    train_txn: pd.DataFrame,
    users: Iterable[str],
    k: int = 12,
    fill_recent: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """Per-user repurchase recommendations, padded with recent-popular items.

    For each user, return their distinct purchased articles in reverse-recency
    order, then pad with ``fill_recent`` (de-duplicated) up to length ``k``.
    Reproduces probe_05's ``repurchase`` function exactly (which scored 0.0243
    on the immediate-next-week window).

    Args:
        train_txn: Train transactions with ``customer_id``, ``article_id``,
            ``t_dat`` columns.
        users: Iterable of customer_id strings to predict for.
        k: Number of recommendations per user.
        fill_recent: Recent-popularity fill list (e.g. ``recent_popularity(...)``).
            ``None`` is treated as an empty list (no padding).

    Returns:
        Mapping ``{customer_id: [article_id, ...]}`` with up to ``k`` items each.
    """
    last_items = _last_items(train_txn, k=k)
    fill = list(fill_recent) if fill_recent else []

    def _predict_one(u: str) -> list[str]:
        out = list(last_items.get(u, []))
        for it in fill:
            if len(out) >= k:
                break
            if it not in out:
                out.append(it)
        return out[:k]

    return {str(u): _predict_one(str(u)) for u in users}


def hybrid_predict(
    train_txn: pd.DataFrame,
    users: Iterable[str],
    train_end: str,
    k: int = 12,
    recent_days: int = 14,
) -> dict[str, list[str]]:
    """Hybrid backbone: repurchase + recent-popularity fill.

    Convenience wrapper that builds the recent-popularity fill list and delegates
    to :func:`repurchase_predict` — the single source of truth for the
    competitive immediate-next-period number. ``hybrid_predict`` and
    ``repurchase_predict`` are intentionally the same predictor (repurchase with
    recent-pop fill); this entry point just computes the fill for the caller.

    Args:
        train_txn: Train transactions with ``customer_id``, ``article_id``,
            ``t_dat`` columns.
        users: Iterable of customer_id strings to predict for.
        train_end: Training cut-off date (inclusive).
        k: Number of recommendations per user.
        recent_days: Recent-popularity window length in days.

    Returns:
        Mapping ``{customer_id: [article_id, ...]}``.
    """
    fill = recent_popularity(train_txn, train_end, days=recent_days, k=k)
    return repurchase_predict(train_txn, users, k=k, fill_recent=fill)
