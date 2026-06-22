"""Activity-cohort and discovery-portion evaluation.

These metrics isolate *where* a recommender's accuracy comes from, the central
question of the hybrid reframe (redesign_2026-06.md §8-9):

- ``activity_cohorts`` / ``evaluate_cohorts`` stratify users by train-history
  size (including brand-new users), exposing how repurchase vs content help
  different activity brackets.
- ``discovery_map`` — the **key new metric** — restricts each user's ground
  truth to NEW items (not in their train history) and scores only that. This
  is the portion repurchase *cannot* solve by construction and where LLM /
  content must work; ~96% of next-week H&M purchases are new items.
- ``repurchase_vs_new_decomposition`` quantifies that repurchase/new split
  (promotes probe_05's logic).

All per-user accuracy is delegated to :mod:`src.evaluation.metrics` (the
existing ``evaluate`` / per-user AP@k); no metric logic is duplicated here.
"""

from typing import NamedTuple

from src.config import EvalConfig, EvalResult
from src.evaluation.metrics import evaluate

# ---------------------------------------------------------------------------
# Activity cohorts
# ---------------------------------------------------------------------------

# Bracket boundaries by train purchase count; "new" = user absent from train.
# Ordered to preserve reporting order. Each value is an inclusive (lo, hi) range.
_ACTIVITY_BRACKETS: tuple[tuple[str, int, int], ...] = (
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-19", 10, 19),
    ("20+", 20, 10**9),
)


def activity_cohorts(train_txn) -> dict[str, str]:
    """Assign each train user to an activity bracket by purchase count.

    Brackets are {new(0), 1, 2-4, 5-9, 10-19, 20+}. Users with zero train
    purchases are labelled ``"new"`` — note that users absent from ``train_txn``
    are simply not present in the returned mapping; callers resolve missing
    users to ``"new"`` (see :func:`evaluate_cohorts`).

    Args:
        train_txn: Train transactions with a ``customer_id`` column
            (pandas DataFrame).

    Returns:
        Mapping ``{customer_id: bracket}`` for every user in ``train_txn``.
    """
    counts = (
        train_txn.assign(customer_id=lambda d: d["customer_id"].astype(str))
        .groupby("customer_id")
        .size()
    )

    def _bracket(n: int) -> str:
        if n <= 0:
            return "new"
        for name, lo, hi in _ACTIVITY_BRACKETS:
            if lo <= n <= hi:
                return name
        return "20+"

    return {u: _bracket(int(n)) for u, n in counts.items()}


def _bracket_for(user: str, train_history: dict[str, set]) -> str:
    """Resolve a user's activity bracket from their train history set."""
    n = len(train_history.get(user, ()))
    if n <= 0:
        return "new"
    for name, lo, hi in _ACTIVITY_BRACKETS:
        if lo <= n <= hi:
            return name
    return "20+"


# ---------------------------------------------------------------------------
# Cohort evaluation
# ---------------------------------------------------------------------------


def evaluate_cohorts(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    train_history: dict[str, set],
    k: int = 12,
) -> dict[str, EvalResult]:
    """Evaluate MAP/HR/NDCG/MRR per activity bracket.

    Each user in ``ground_truth`` is bucketed into {new, 1, 2-4, 5-9, 10-19,
    20+} by the size of their ``train_history`` set, and metrics are computed
    per bucket via :func:`src.evaluation.metrics.evaluate`. Empty buckets are
    omitted.

    Args:
        predictions: ``{customer_id: [article_id ranked, ...]}``.
        ground_truth: ``{customer_id: [article_id, ...]}``.
        train_history: ``{customer_id: set(article_id purchased in train)}``.
        k: Cutoff.

    Returns:
        ``{bracket: EvalResult}`` for every non-empty bracket.
    """
    config = EvalConfig(k=k)
    buckets: dict[str, list[str]] = {}
    for user in ground_truth:
        buckets.setdefault(_bracket_for(user, train_history), []).append(user)

    # Stable reporting order: new first, then ascending activity.
    order = ["new", *(name for name, _, _ in _ACTIVITY_BRACKETS)]
    results: dict[str, EvalResult] = {}
    for bracket in order:
        users = buckets.get(bracket)
        if not users:
            continue
        gt_b = {u: ground_truth[u] for u in users}
        pred_b = {u: predictions.get(u, []) for u in users}
        results[bracket] = evaluate(pred_b, gt_b, config)
    return results


# ---------------------------------------------------------------------------
# Discovery-only evaluation (KEY new metric)
# ---------------------------------------------------------------------------


def discovery_map(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    train_history: dict[str, set],
    k: int = 12,
) -> EvalResult:
    """Evaluate accuracy on NEW (never-before-purchased) ground-truth items only.

    For each user, ground truth is restricted to article_ids NOT in that user's
    ``train_history``. Users whose new-item GT is empty are skipped entirely
    (no GT to score). Predictions are passed through unchanged — a repurchase
    backbone that only proposes already-owned items will score near zero here
    *by construction*, isolating the discovery quality where LLM / content is
    the only lever.

    Args:
        predictions: ``{customer_id: [article_id ranked, ...]}``.
        ground_truth: ``{customer_id: [article_id, ...]}``.
        train_history: ``{customer_id: set(article_id purchased in train)}``.
        k: Cutoff.

    Returns:
        ``EvalResult`` over the discovery-only ground truth.
    """
    config = EvalConfig(k=k)
    new_gt: dict[str, list[str]] = {}
    new_pred: dict[str, list[str]] = {}
    for user, items in ground_truth.items():
        hist = train_history.get(user, set())
        new_items = [it for it in items if it not in hist]
        if not new_items:
            continue
        new_gt[user] = new_items
        new_pred[user] = predictions.get(user, [])

    return evaluate(new_pred, new_gt, config)


# ---------------------------------------------------------------------------
# Repurchase-vs-new decomposition
# ---------------------------------------------------------------------------


class DecompositionResult(NamedTuple):
    """Item-level repurchase vs new-item fractions of the ground truth."""

    repurchase_frac: float
    new_frac: float
    n_gt_items: int


def repurchase_vs_new_decomposition(
    ground_truth: dict[str, list[str]],
    train_history: dict[str, set],
) -> DecompositionResult:
    """Fraction of ground-truth items that are repurchases vs new items.

    Promotes probe_05's decomposition: across all (user, GT item) pairs, count
    how many items the user had already purchased in train (repurchase) versus
    not (new). ~96% of next-week H&M purchases are new items.

    Args:
        ground_truth: ``{customer_id: [article_id, ...]}``.
        train_history: ``{customer_id: set(article_id purchased in train)}``.

    Returns:
        ``DecompositionResult`` with ``repurchase_frac``, ``new_frac`` (summing
        to 1.0 when any GT items exist) and total ``n_gt_items``.
    """
    n_rep = 0
    n_new = 0
    for user, items in ground_truth.items():
        hist = train_history.get(user, set())
        for it in set(items):
            if it in hist:
                n_rep += 1
            else:
                n_new += 1
    total = n_rep + n_new
    if total == 0:
        return DecompositionResult(repurchase_frac=0.0, new_frac=0.0, n_gt_items=0)
    return DecompositionResult(
        repurchase_frac=n_rep / total,
        new_frac=n_new / total,
        n_gt_items=total,
    )
