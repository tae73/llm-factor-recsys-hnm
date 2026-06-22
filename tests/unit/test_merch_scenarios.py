"""Unit tests for src/serving/merch_scenarios (the 3 PASS-cell merchandising briefs).

Two layers:
  * fast canonical-consistency guards (read only the small probe JSONs) — these are the
    drift guard: the engine's confidence numbers MUST equal the published value matrix, so
    if a canonical PASS cell ever changes value/verdict these fail loudly.
  * data-dependent brief tests (need matrix_axes.parquet + train transactions) — skipped
    when the data artifacts are absent (e.g. data-less CI). They check the operational
    tables are well-formed and that the lead-lag deterministically reproduces r≈0.4723.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.serving.merch_scenarios import (
    ANCHOR_ROLES,
    HOT_PHASES,
    PASS_CELLS,
    ScenarioConfig,
    build_all_briefs,
    copurchase_velocity_brief,
    launch_signal_brief,
    load_confidence_cards,
    trend_leadtime_brief,
    value_matrix_posture,
)

CFG = ScenarioConfig()
_DATA_OK = CFG.matrix_path.exists() and CFG.train_path.exists() and CFG.articles_path.exists()
_CANON_OK = CFG.canonical_e2.exists() and CFG.canonical_e2b.exists()

requires_canon = pytest.mark.skipif(not _CANON_OK, reason="canonical probe JSON missing")
requires_data = pytest.mark.skipif(
    not (_DATA_OK and _CANON_OK), reason="matrix_axes / transactions missing"
)

# Published canonical PASS-cell numbers (probe_E2/E2b) — the single source of truth.
EXPECTED = {
    "trend_leadtime": {"value": 0.4723, "baseline": 0.0615, "best_lag": 3},
    "launch_signal": {"value": 0.6727, "baseline": 0.223, "best_lag": None},
    "copurchase_velocity": {"value": 0.631, "baseline": 0.534, "best_lag": None},
}


def _correlation_ratio(groups: np.ndarray, y: np.ndarray) -> float:
    """Eta (sqrt of between-group variance fraction), 0..1 — local copy for the guard."""
    ybar = y.mean()
    ss_tot = ((y - ybar) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    ss_between = sum(
        (y[groups == g].mean() - ybar) ** 2 * (groups == g).sum() for g in np.unique(groups)
    )
    return float(np.sqrt(ss_between / ss_tot))


# ---------------------------------------------------------------------------
# Canonical-consistency guards (fast)
# ---------------------------------------------------------------------------
@requires_canon
def test_confidence_cards_match_canonical():
    """Each PASS-cell card equals the published value/baseline/verdict (drift guard)."""
    cards = load_confidence_cards(CFG)
    assert set(cards) == set(PASS_CELLS)
    for name, card in cards.items():
        exp = EXPECTED[name]
        assert card.verdict == "PASS", f"{name} verdict regressed: {card.verdict}"
        assert card.value == pytest.approx(exp["value"], abs=1e-3), name
        assert card.baseline == pytest.approx(exp["baseline"], abs=2e-3), name
        assert card.best_lag == exp["best_lag"], name
        # excess (value - baseline) is positive — the cell genuinely beats the baseline.
        assert card.value > card.baseline


@requires_canon
def test_value_matrix_posture_counts_and_productized():
    """Posture exposes the full 16-cell honest matrix; exactly the 3 PASS are productized."""
    df = value_matrix_posture(CFG)
    assert len(df) == 16
    assert df.attrs["capability_yes"] == 14
    assert df.attrs["lift_pass"] == 3
    productized = {(r.axis, r.use) for r in df[df["productized"]].itertuples()}
    assert productized == set(PASS_CELLS.values())
    # every productized cell is a PASS; nothing non-PASS slips in.
    assert (df.loc[df["productized"], "lift_verdict"] == "PASS").all()
    assert "추천 정확도" in df.attrs["recsys_negative"]  # recsys-negative surfaced as context


@requires_canon
def test_e2_and_e2b_leadlag_agree():
    """probe_E2 and probe_E2b carry the SAME lead-time r (cell A unchanged across rounds)."""
    import json

    e2 = json.loads(CFG.canonical_e2.read_text())["value_matrix"]
    e2b = json.loads(CFG.canonical_e2b.read_text())["value_matrix"]
    r_e2 = next(c for c in e2 if c["axis"] == "e2_trend_phase_actual" and c["use"] == "leadlag")
    r_e2b = next(c for c in e2b if c["axis"] == "e2_trend_phase_actual" and c["use"] == "leadlag")
    assert r_e2["r_attr"] == pytest.approx(r_e2b["r_attr"], abs=1e-6)
    assert r_e2["r_attr"] == pytest.approx(0.4723, abs=1e-3)


# ---------------------------------------------------------------------------
# Brief well-formedness + deterministic reproduction (data-dependent)
# ---------------------------------------------------------------------------
@requires_data
def test_trend_leadtime_brief_reproduces_canonical_r():
    """Lead-time brief lists all categories and re-derives r≈0.4723 at lag 3."""
    brief = trend_leadtime_brief(CFG)
    assert brief.name == "trend_leadtime"
    assert len(brief.table) == 10  # 10 index_name categories
    assert {"category", "latest_hot_share", "share_zscore", "early_warning"} <= set(
        brief.table.columns
    )
    # deterministic reproduction of the canonical lead-lag (drift / data-corruption guard).
    by_lag = brief.extra["lead_lag_by_lag"]
    assert by_lag[brief.extra["best_lag"]] == pytest.approx(0.4723, abs=0.05)
    # ranked by how elevated the current hot-share is (descending z-score).
    assert brief.table["share_zscore"].is_monotonic_decreasing


@requires_data
def test_launch_signal_brief_cohort_and_grounding():
    """Launch scorecard ranks only the hot cohort and is grounded in the merch PASS cell."""
    brief = launch_signal_brief(CFG, top_k=30)
    assert 0 < len(brief.table) <= 30
    assert set(brief.table["e2_trend_phase_actual"]).issubset(set(HOT_PHASES))
    assert brief.table["first_week_sell_through"].is_monotonic_decreasing
    assert brief.confidence.cell == "e2_trend_phase_actual→merch"
    # phase-level signal table (full population) backs the η evidence figure.
    assert "phase_sellthrough" in brief.extra and len(brief.extra["phase_sellthrough"]) >= 1


@requires_data
def test_copurchase_velocity_brief_anchors_only():
    """Velocity brief surfaces anchor roles only, ranked by velocity, with bundle labels."""
    brief = copurchase_velocity_brief(CFG, top_k=30)
    assert 0 < len(brief.table) <= 30
    assert set(brief.table["e2_outfit_role"]).issubset(set(ANCHOR_ROLES))
    assert brief.table["velocity"].is_monotonic_decreasing
    assert brief.table["bundle_role"].notna().all()


@requires_data
def test_outfit_role_carries_velocity_signal():
    """Sanity: outfit_role groups log-velocity (eta high, Anchor-hub > Standalone)."""
    mx = pd.read_parquet(CFG.matrix_path).dropna(subset=["velocity"])
    eta = _correlation_ratio(
        mx["e2_outfit_role"].to_numpy(), np.log1p(mx["velocity"].to_numpy(dtype=float))
    )
    assert eta > 0.3  # role meaningfully structures velocity (canonical resid η=0.631)
    means = mx.groupby("e2_outfit_role")["velocity"].mean()
    assert means["Anchor-hub"] > means["Standalone"]


@requires_data
def test_build_all_briefs_returns_three():
    """The orchestrator returns the three PASS-cell briefs, all non-empty."""
    briefs = build_all_briefs(CFG, item_top_k=20)
    assert [b.name for b in briefs] == list(PASS_CELLS)
    assert all(len(b.table) > 0 for b in briefs)
