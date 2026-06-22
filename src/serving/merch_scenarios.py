"""Merchandising decision-support scenarios — productizing the 3 PASS value-matrix cells.

This is the *build* (product-design) leg of the enrichment-v2 research: the value matrix
(``witnesses/probe_E2*.py``) confirmed exactly THREE behavior-validated decision-lift cells
out of sixteen, and this module turns those three into batch merchandising briefs a buyer /
launch-planner / cross-sell merchandiser can act on. The three PASS cells (all on
*behavior-derived* axes — NOT the LLM perception axes, which were 9/9 redundant):

  A. ``e2_trend_phase_actual`` → **lead-time** : a category's share of hot (Emerging/Rising)
     items LEADS its sales by ~3 months (canonical r=0.472 vs permutation null 0.062).
  B. ``e2_trend_phase_actual`` → **merch** : trend-phase predicts launch ``first_week_sell_through``
     (canonical η=0.673 resid product_group vs metadata 0.223).
  C. ``e2_outfit_role`` → **merch** : co-purchase outfit-role predicts ``velocity``
     (canonical η=0.631 resid product_group vs metadata 0.534).

HONESTY PRINCIPLE (drift guard). The *operational* outputs (ranked briefs) are computed
fresh from ``matrix_axes.parquet`` + transactions, reusing the SAME feature functions the
probes used (so e.g. the lead-lag reproduces r=0.4723 deterministically). The *confidence
numbers* (r / η / CI / verdict) are LOADED from the canonical probe JSONs — never recomputed
into a new headline — so the value matrix stays the single source of truth and the briefs
cannot silently diverge from the published evidence. Cells that did NOT pass (① automatic
faceted-control lift, ④ audience, the gap axes) and the separate recommendation-accuracy
negative are surfaced as explicit *non*-productized context, not hidden.

CPU/DuckDB only, seed 42, no API/$. Output tables → ``results/tables/merch_scenarios/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

import duckdb
import numpy as np
import pandas as pd

from src.features.lead_lag import lead_lag_corr, monthly_attribute_share

logger = logging.getLogger(__name__)

# Phases that constitute "hot / on-the-rise" demand (the lead-time + launch cohort).
HOT_PHASES: tuple[str, ...] = ("Emerging", "Rising")
# Roles worth flagging as cross-sell bundle anchors (high co-purchase structure).
ANCHOR_ROLES: tuple[str, ...] = ("Anchor-hub", "Versatile-connector")

# Co-purchase role → merchandising interpretation (decision label, not a score).
BUNDLE_ROLE_LABELS: dict[str, str] = {
    "Anchor-hub": "Cross-sell anchor — high co-purchase intensity & diversity; stock-priority",
    "Versatile-connector": "Category bridge — pairs across many groups; cross-merch glue",
    "Complement-addon": "Attach item — intense but narrow partners; bundle onto anchors",
    "Niche-pair": "Narrow pairing — limited co-purchase; targeted bundles only",
    "Standalone": "No co-purchase signal — merchandise solo",
}

# scenario name → (axis, use) coordinates of its PASS cell in the canonical matrix.
PASS_CELLS: dict[str, tuple[str, str]] = {
    "trend_leadtime": ("e2_trend_phase_actual", "leadlag"),
    "launch_signal": ("e2_trend_phase_actual", "merch"),
    "copurchase_velocity": ("e2_outfit_role", "merch"),
}

# The recommendation-accuracy result lives in a *different* probe family and is a confirmed
# negative; we state it as explicit non-productized context (never silently dropped).
RECSYS_NEGATIVE_NOTE = (
    "추천 정확도(별도 probe): LLM L2/L3/외부지식은 L1 content 대비 H&M 추천 정확도를 "
    "개선하지 못함 — multiply-confirmed negative (full-scale −12%, probe_21/22). "
    "이 value matrix 밖의 결과이며, 제품화하지 않고 맥락화만 한다."
)


# ---------------------------------------------------------------------------
# Config & result objects
# ---------------------------------------------------------------------------
class ScenarioConfig(NamedTuple):
    """Immutable paths + seed for the merchandising scenario engine."""

    matrix_path: Path = Path("data/knowledge/enrichment_v2/matrix_axes.parquet")
    train_path: Path = Path("data/processed/train_transactions.parquet")
    articles_path: Path = Path("data/processed/articles.parquet")
    canonical_e2: Path = Path("witnesses/probe_E2_result.json")
    canonical_e2b: Path = Path("witnesses/probe_E2b_result.json")
    seed: int = 42


class ConfidenceCard(NamedTuple):
    """Canonical PASS-cell evidence (LOADED from probe JSON — not recomputed)."""

    cell: str  # "e2_trend_phase_actual→leadlag"
    metric: str  # human-readable metric description
    value: float  # r_attr (lead-lag) or eta_attr (merch)
    baseline: float  # metadata / permutation-null baseline
    ci_lo: float
    ci_hi: float
    verdict: str  # "PASS"
    source: str  # canonical JSON filename
    best_lag: int | None = None  # lead-lag horizon (months); None for merch cells


class ScenarioBrief(NamedTuple):
    """One batch merchandising brief: a ranked operational table + its evidence."""

    name: str  # "trend_leadtime" | "launch_signal" | "copurchase_velocity"
    title: str
    table: pd.DataFrame  # ranked operational output
    confidence: ConfidenceCard  # the PASS cell it is grounded in
    caveat: str  # honest scope boundary
    extra: dict[str, Any]  # figure-supporting data (time series / grouped stats)


# ---------------------------------------------------------------------------
# Canonical confidence loading (single source of truth)
# ---------------------------------------------------------------------------
def _find_cell(value_matrix: list[dict], axis: str, use: str) -> dict:
    """Locate the (axis, use) cell in a canonical ``value_matrix`` list."""
    for c in value_matrix:
        if c.get("axis") == axis and c.get("use") == use:
            return c
    raise KeyError(f"cell {axis}→{use} not found in canonical value_matrix")


def load_confidence_cards(cfg: ScenarioConfig = ScenarioConfig()) -> dict[str, ConfidenceCard]:
    """Read the 3 PASS-cell confidence numbers from the canonical E2b probe JSON.

    The baseline is derived as ``value − lift_value`` (lift is the excess over the
    metadata / permutation-null baseline), so it stays consistent with the published
    delta without re-parsing free-text baseline strings.
    """
    d = json.loads(cfg.canonical_e2b.read_text())
    vm = d["value_matrix"]
    cards: dict[str, ConfidenceCard] = {}
    for name, (axis, use) in PASS_CELLS.items():
        c = _find_cell(vm, axis, use)
        lift = float(c["lift_value"])
        ci = c.get("lift_ci") or [float("nan"), float("nan")]
        if use == "leadlag":
            value = float(c["r_attr"])
            metric = "lead-lag r: hot-share(t) → category sales(t+best_lag)"
            best_lag = c.get("best_lag")
        else:
            value = float(c["eta_attr"])
            outcome = c.get("outcome", "outcome")
            metric = f"η({outcome} | resid product_group) vs metadata"
            best_lag = None
        cards[name] = ConfidenceCard(
            cell=f"{axis}→{use}",
            metric=metric,
            value=round(value, 4),
            baseline=round(value - lift, 4),
            ci_lo=round(float(ci[0]), 4),
            ci_hi=round(float(ci[1]), 4),
            verdict=str(c["lift_verdict"]),
            source=cfg.canonical_e2b.name,
            best_lag=int(best_lag) if best_lag is not None else None,
        )
    return cards


def _load_matrix(cfg: ScenarioConfig) -> pd.DataFrame:
    """Load ``matrix_axes.parquet`` with ``article_id`` as str."""
    if not cfg.matrix_path.exists():
        raise FileNotFoundError(cfg.matrix_path)
    df = pd.read_parquet(cfg.matrix_path)
    df["article_id"] = df["article_id"].astype(str)
    return df


def _tier_by_rank(ordered_keys: list[str]) -> dict[str, str]:
    """Map keys (best→worst order) to High/Medium/Low thirds."""
    n = len(ordered_keys)
    out: dict[str, str] = {}
    for i, k in enumerate(ordered_keys):
        frac = i / max(n - 1, 1)
        out[k] = "High" if frac <= 1 / 3 else ("Medium" if frac <= 2 / 3 else "Low")
    return out


# ---------------------------------------------------------------------------
# Scenario A — trend lead-time (3-month category early-warning)
# ---------------------------------------------------------------------------
def trend_leadtime_brief(
    cfg: ScenarioConfig = ScenarioConfig(),
    cards: dict[str, ConfidenceCard] | None = None,
    top_k: int | None = None,
) -> ScenarioBrief:
    """Per-category early-warning: hot-item share LEADS category sales by ~3 months.

    Reuses ``monthly_attribute_share`` + ``lead_lag_corr`` (so lag-3 reproduces the canonical
    r=0.4723). The operational table ranks categories by how *elevated/rising* their current
    hot (Emerging+Rising) share is — high share now ⇒ higher sales in ~3 months.
    """
    cards = cards or load_confidence_cards(cfg)
    card = cards["trend_leadtime"]
    mx = _load_matrix(cfg)[["article_id", "e2_trend_phase_actual"]]

    con = duckdb.connect()
    try:
        share = monthly_attribute_share(
            con,
            cfg.train_path,
            cfg.articles_path,
            mx,
            "e2_trend_phase_actual",
            values=list(HOT_PHASES),
            category_col="index_name",
            granularity="month",
        )
    finally:
        con.close()
    by_lag = lead_lag_corr(share, lags=(1, 2, 3, 4))

    rows: list[dict[str, Any]] = []
    for cat, g in share.groupby("cat"):
        g = g.sort_values("mo")
        s = g["share"].to_numpy(dtype=float)
        sales = g["cat_sales"].to_numpy(dtype=float)
        mean, std = float(s.mean()), float(s.std())
        latest = float(s[-1])
        z = (latest - mean) / std if std > 0 else 0.0
        trend3 = latest - float(s[-4]) if len(s) >= 4 else latest - float(s[0])
        rows.append(
            {
                "category": cat,
                "latest_hot_share": round(latest, 4),
                "hist_mean_share": round(mean, 4),
                "share_zscore": round(z, 3),
                "share_trend_3mo": round(trend3, 4),
                "latest_month_sales": int(sales[-1]),
            }
        )
    df = pd.DataFrame(rows)
    df["early_warning"] = pd.cut(
        df["share_zscore"],
        bins=[-np.inf, 0.5, 1.0, np.inf],
        labels=["watch", "elevated", "high"],
    ).astype(str)
    df = df.sort_values("share_zscore", ascending=False).reset_index(drop=True)
    if top_k:
        df = df.head(top_k).reset_index(drop=True)

    lag = card.best_lag or 3
    caveat = (
        f"행동-파생 trend_phase 축(LLM 아님). 검증된 lead-lag r={card.value} "
        f"(permutation null {card.baseline}, lag={lag}mo, CI[{card.ci_lo},{card.ci_hi}]) — "
        "22개월·10 카테고리의 modest 상관(CI 넓음). 카테고리 수준 조기경보용이며 "
        "per-item 예측·추천 정확도(별도 negative)와는 다르다."
    )
    return ScenarioBrief(
        name="trend_leadtime",
        title=f"Trend lead-time — {lag}-month category demand early-warning",
        table=df,
        confidence=card,
        caveat=caveat,
        extra={"share_timeseries": share, "lead_lag_by_lag": by_lag, "best_lag": lag},
    )


# ---------------------------------------------------------------------------
# Scenario B — launch first-week sell-through signal
# ---------------------------------------------------------------------------
def launch_signal_brief(
    cfg: ScenarioConfig = ScenarioConfig(),
    cards: dict[str, ConfidenceCard] | None = None,
    top_k: int | None = 50,
    min_purchases: int = 10,
) -> ScenarioBrief:
    """New-arrival launch scorecard: trend-phase predicts first-week sell-through.

    Phase → expected first-week sell-through tier (the signal model), then the hot launch
    cohort (Emerging+Rising) ranked by measured ``first_week_sell_through`` for stock
    prioritization. Grounded in η=0.673 (resid product_group) vs metadata 0.223. The
    phase-level signal table uses the full population (matches the canonical η); only the
    displayed item cohort applies ``min_purchases`` to suppress single-week noise.
    """
    cards = cards or load_confidence_cards(cfg)
    card = cards["launch_signal"]
    mx = _load_matrix(cfg)
    valid = mx.dropna(subset=["first_week_sell_through"])

    phase_stats = (
        valid.groupby("e2_trend_phase_actual")
        .agg(
            n_items=("article_id", "size"),
            mean_first_week_sell_through=("first_week_sell_through", "mean"),
            median_first_week_sell_through=("first_week_sell_through", "median"),
        )
        .reset_index()
        .sort_values("mean_first_week_sell_through", ascending=False)
        .reset_index(drop=True)
    )
    phase_tier = _tier_by_rank(phase_stats["e2_trend_phase_actual"].tolist())
    phase_stats["sell_through_tier"] = phase_stats["e2_trend_phase_actual"].map(phase_tier)

    cohort = valid[
        valid["e2_trend_phase_actual"].isin(HOT_PHASES)
        & (valid["total_purchases"] >= min_purchases)
    ].copy()
    cohort["expected_sell_through_tier"] = cohort["e2_trend_phase_actual"].map(phase_tier)
    cols = [
        "article_id",
        "e2_trend_phase_actual",
        "e2_trend_momentum",
        "first_week_sell_through",
        "expected_sell_through_tier",
        "total_purchases",
        "n_buyers",
        "velocity",
    ]
    table = cohort.sort_values("first_week_sell_through", ascending=False)[cols].reset_index(
        drop=True
    )
    if top_k:
        table = table.head(top_k).reset_index(drop=True)

    caveat = (
        f"행동-파생 trend_phase 축(LLM 아님). η={card.value} (resid product_group, "
        f"metadata {card.baseline}, CI[{card.ci_lo},{card.ci_hi}]). velocity 대신 "
        "first_week_sell_through 사용(momentum=sales-rate tautology 회피). 신규 런칭 "
        "재고배분·우선순위용이며 추천 정확도(별도 negative)와 무관."
    )
    return ScenarioBrief(
        name="launch_signal",
        title="Launch signal — first-week sell-through by trend-phase",
        table=table,
        confidence=card,
        caveat=caveat,
        extra={"phase_sellthrough": phase_stats},
    )


# ---------------------------------------------------------------------------
# Scenario C — co-purchase velocity / bundle role
# ---------------------------------------------------------------------------
def copurchase_velocity_brief(
    cfg: ScenarioConfig = ScenarioConfig(),
    cards: dict[str, ConfidenceCard] | None = None,
    top_k: int | None = 50,
) -> ScenarioBrief:
    """Bundle-anchor scorecard: co-purchase outfit-role predicts sales velocity.

    Role → velocity tier + a merchandising bundle label, then anchor-role items
    (Anchor-hub + Versatile-connector) ranked by ``velocity`` as cross-sell stock-priority
    candidates. Grounded in η=0.631 (resid product_group) vs metadata 0.534.
    """
    cards = cards or load_confidence_cards(cfg)
    card = cards["copurchase_velocity"]
    mx = _load_matrix(cfg)
    valid = mx.dropna(subset=["velocity"])

    role_stats = (
        valid.groupby("e2_outfit_role")
        .agg(
            n_items=("article_id", "size"),
            mean_velocity=("velocity", "mean"),
            median_velocity=("velocity", "median"),
        )
        .reset_index()
        .sort_values("mean_velocity", ascending=False)
        .reset_index(drop=True)
    )
    role_tier = _tier_by_rank(role_stats["e2_outfit_role"].tolist())
    role_stats["velocity_tier"] = role_stats["e2_outfit_role"].map(role_tier)
    role_stats["bundle_role"] = role_stats["e2_outfit_role"].map(BUNDLE_ROLE_LABELS)

    anchors = valid[valid["e2_outfit_role"].isin(ANCHOR_ROLES)].copy()
    anchors["bundle_role"] = anchors["e2_outfit_role"].map(BUNDLE_ROLE_LABELS)
    anchors["velocity_tier"] = anchors["e2_outfit_role"].map(role_tier)
    cols = [
        "article_id",
        "e2_outfit_role",
        "bundle_role",
        "velocity_tier",
        "velocity",
        "total_purchases",
        "n_buyers",
    ]
    table = anchors.sort_values("velocity", ascending=False)[cols].reset_index(drop=True)
    if top_k:
        table = table.head(top_k).reset_index(drop=True)

    caveat = (
        f"행동-파생 outfit_role 축(LLM 아님). η={card.value} (resid product_group, "
        f"metadata {card.baseline}, CI[{card.ci_lo},{card.ci_hi}]). co-purchase 그래프 "
        "역할 → cross-sell 앵커·번들 우선순위용. 추천 정확도(별도 negative)와 무관."
    )
    return ScenarioBrief(
        name="copurchase_velocity",
        title="Co-purchase velocity — bundle-anchor prioritization by outfit-role",
        table=table,
        confidence=card,
        caveat=caveat,
        extra={"role_velocity": role_stats},
    )


# ---------------------------------------------------------------------------
# Orchestration & honest value-matrix posture
# ---------------------------------------------------------------------------
_BRIEF_FNS = {
    "trend_leadtime": trend_leadtime_brief,
    "launch_signal": launch_signal_brief,
    "copurchase_velocity": copurchase_velocity_brief,
}


def build_brief(
    name: str, cfg: ScenarioConfig = ScenarioConfig(), top_k: int | None = None
) -> ScenarioBrief:
    """Build a single brief by scenario ``name``."""
    if name not in _BRIEF_FNS:
        raise ValueError(f"unknown scenario {name!r}; choose from {list(_BRIEF_FNS)}")
    cards = load_confidence_cards(cfg)
    fn = _BRIEF_FNS[name]
    # top_k=None → use each scenario's own default; pass explicitly only when given.
    return fn(cfg, cards, top_k) if top_k is not None else fn(cfg, cards)


def build_all_briefs(
    cfg: ScenarioConfig = ScenarioConfig(), item_top_k: int | None = 50
) -> list[ScenarioBrief]:
    """Build all three PASS-cell briefs (loads canonical cards once).

    ``item_top_k`` caps the two item-level briefs (launch / co-purchase); the category-level
    lead-time brief always lists all ~10 categories.
    """
    cards = load_confidence_cards(cfg)
    return [
        trend_leadtime_brief(cfg, cards),
        launch_signal_brief(cfg, cards, top_k=item_top_k),
        copurchase_velocity_brief(cfg, cards, top_k=item_top_k),
    ]


def value_matrix_posture(cfg: ScenarioConfig = ScenarioConfig()) -> pd.DataFrame:
    """Full 16-cell honest posture from the canonical E2b matrix.

    Marks which cells are PRODUCTIZED (the 3 PASS) vs surfaced as non-productized context
    (MARGINAL/NO/N-A). ``df.attrs`` carries the headline counts + recsys-negative note.
    """
    d = json.loads(cfg.canonical_e2b.read_text())
    vm = d["value_matrix"]
    pass_set = set(PASS_CELLS.values())
    rows = [
        {
            "axis": c["axis"],
            "use": c["use"],
            "capability": c.get("capability", "YES"),
            "lift_verdict": c["lift_verdict"],
            "lift_value": c.get("lift_value"),
            "productized": (c["axis"], c["use"]) in pass_set,
        }
        for c in vm
    ]
    df = pd.DataFrame(rows)
    df.attrs["capability_yes"] = d.get("capability_yes")
    df.attrs["n_cells"] = len(vm)
    df.attrs["lift_pass"] = d.get("e2b_lift_pass")
    df.attrs["pass_cells"] = [f"{a}→{u}" for a, u in pass_set]
    df.attrs["recsys_negative"] = RECSYS_NEGATIVE_NOTE
    return df
