"""PROBE E2c — USER-side enrichment value (KAR reasoning leg, completes the 2-source matrix).

E2/E2b filled the value matrix with ITEM-side axes → lift only at ②lead-time + ③merch
(item decisions). ①control and ④audience failed because they are USER decisions with no
user enrichment supplied. This probe brings the KAR **reasoning(user)** leg — LLM user
profiles — to exactly ①④, producing a 2-source × 4-use decomposition:
    item-enrichment → ②③   |   user-enrichment → ①④ (tested here).

ROWS = user reps {reasoning_bge, reasoning_fields}; BASELINE = 11 demographic features.
HONESTY: reasoning is TRAIN-derived, so every outcome is held-out FUTURE behavior (val
2020-07+); demographics are equally train-derived → reasoning-vs-demographics is
apples-to-apples on PREDICTING THE FUTURE. The established negative (LLM does not improve
ranking ACCURACY) is respected — both cells are PREDICTION of a future user property,
not ranking. PRIMARY metric = does (demo+reasoning) beat demo? (i.e. does reasoning ADD
over the cheap baseline). Operationalization:
  ① control  = predict the user's FUTURE item-FACET (outfit_role / trend_phase) — the
               steerable target; if reasoning can't predict the intended facet better
               than demographics, steering to it can't help.
  ④ audience = predict the user's FUTURE behavior KPIs (price-tier, category-mix, channel,
               repurchase) — who the user is.
②③ are item decisions (no user-rep semantics) → N/A. Never touches probe_E2/E2b JSON.
CPU, seed 42, no API/$.

Usage:  PYTHONPATH=. python -u witnesses/probe_E2c_user_value.py [--quick]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.features.audience_signals import segment_divergence_weighted  # noqa: E402
from src.features.user_axes import (  # noqa: E402
    build_demographic,
    build_future_outcomes,
    build_user_representations,
    select_cohort,
)
from witnesses.probe_D5_value_matrix import _cv_predict, _effective_k, _paired_sig  # noqa: E402
from witnesses.probe_E2_value_matrix import MATRIX_PATH, SEED, _na  # noqa: E402

warnings.filterwarnings("ignore")

RESULT_PATH = ROOT / "witnesses/probe_E2c_user_value.json"
FIG_PATH = ROOT / "results/figures/E2c_user_value.png"
E2_JSON = ROOT / "witnesses/probe_E2_result.json"
E2B_JSON = ROOT / "witnesses/probe_E2b_result.json"
VAL_TXN = ROOT / "data/processed/val_transactions.parquet"

USER_REPS = ["reasoning_bge", "reasoning_fields"]  # rows (vs demographic baseline)
USES = ["control", "leadtime", "merch", "audience"]
# outcome sets per cell (all FUTURE, held-out)
AUDIENCE_OUTCOMES = ["fut_price_tier", "fut_top_group", "fut_online", "fut_repurchase"]
CONTROL_OUTCOMES = ["fut_outfit_role", "fut_trend_phase"]  # the steerable item-facet
PRED_MARGIN = 0.01  # incremental macro-F1 over demographics (D5 PRACTICAL_MARGIN parity)


# ===========================================================================
# Shared: predictive cell — does (demo+reasoning) beat demo on FUTURE outcomes?
# ===========================================================================
def _predictive_cell(
    rep_name: str,
    rep: np.ndarray,
    demo: np.ndarray,
    fut: pd.DataFrame,
    outcomes: list[str],
    use: str,
    cap_basis: str,
    note_tail: str,
) -> dict:
    per_outcome = {}
    best_delta, best_outcome, any_sig = -1.0, None, False
    for outcome in outcomes:
        if outcome not in fut.columns:
            continue
        y = fut[outcome].fillna("__na__").astype(str).to_numpy()
        mask = y != "__na__"
        if mask.sum() < 500 or len(np.unique(y[mask])) < 2:
            continue
        yv = y[mask]
        rm = _cv_predict(demo[mask], yv)  # demographic baseline
        rb = _cv_predict(np.concatenate([demo[mask], rep[mask]], axis=1), yv)  # demo + reasoning
        ra = _cv_predict(rep[mask], yv)  # reasoning alone (harsher)
        sig = _paired_sig(rb.macro_f1_folds, rm.macro_f1_folds)  # incremental over demo
        per_outcome[outcome] = {
            "demo_f1": round(rm.macro_f1_mean, 4),
            "both_f1": round(rb.macro_f1_mean, 4),
            "reasoning_alone_f1": round(ra.macro_f1_mean, 4),
            "incremental_delta": sig["mean_delta"],
            "p": sig["p_value"],
            "significant": sig["significant"],
            "n": int(mask.sum()),
        }
        if sig["significant"] and sig["mean_delta"] > best_delta:
            best_delta, best_outcome = sig["mean_delta"], outcome
        any_sig = any_sig or sig["significant"]
    verdict = (
        "PASS"
        if (any_sig and best_delta >= PRED_MARGIN)
        else ("MARGINAL" if best_outcome else "NO")
    )
    return {
        "axis": rep_name,
        "use": use,
        "capability": "YES",
        "capability_basis": cap_basis,
        "lift_metric": "incremental_macroF1 (demo+reasoning vs demo) on FUTURE outcome",
        "metadata_baseline": "11 demographic user features (same train→future setup)",
        "lift_value": round(float(best_delta), 4) if best_outcome else None,
        "best_outcome": best_outcome,
        "per_outcome": per_outcome,
        "significant": bool(any_sig and best_delta >= PRED_MARGIN),
        "lift_verdict": verdict,
        "power": "STRONG",
        "note": f"best={best_outcome} Δ={best_delta:+.4f}; {note_tail}",
    }


def cell_audience(rep_name, rep, demo, fut, n_perm) -> dict:
    c = _predictive_cell(
        rep_name,
        rep,
        demo,
        fut,
        AUDIENCE_OUTCOMES,
        "audience",
        "user-reasoning is a richer profile than 11 demographics",
        "channel/repurchase=habit→demo wins; category/price=taste→reasoning may add",
    )
    c["divergence"] = _audience_divergence(rep, demo, fut, n_perm)
    return c


def cell_control(rep_name, rep, demo, fut) -> dict:
    return _predictive_cell(
        rep_name,
        rep,
        demo,
        fut,
        CONTROL_OUTCOMES,
        "control",
        "reasoning may predict the user's intended item-facet (steer target)",
        "control prerequisite = predicting the steerable facet better than demo",
    )


def _audience_divergence(rep: np.ndarray, demo: np.ndarray, fut: pd.DataFrame, n_perm: int) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    y = fut["fut_price_tier"].fillna(-1).astype(float).to_numpy()
    mask = y >= 0
    if mask.sum() < 500:
        return {"div_rep": None}
    yv, n = y[mask], np.ones(int(mask.sum()))
    lab_r = KMeans(6, random_state=SEED, n_init=4).fit_predict(
        StandardScaler().fit_transform(rep[mask])
    )
    lab_d = KMeans(6, random_state=SEED, n_init=4).fit_predict(
        StandardScaler().fit_transform(demo[mask])
    )
    div_r = segment_divergence_weighted(lab_r.astype(str), yv, n)
    div_d = segment_divergence_weighted(lab_d.astype(str), yv, n)
    rng = np.random.default_rng(SEED)
    perm = np.array(
        [
            segment_divergence_weighted(rng.permutation(lab_r).astype(str), yv, n)
            for _ in range(n_perm)
        ]
    )
    return {
        "div_rep": round(float(div_r), 4),
        "div_demo": round(float(div_d), 4),
        "ratio": round(float(div_r / max(div_d, 1e-9)), 3),
        "perm_p": round(float((perm >= div_r).mean()), 4),
        "eff_k_rep": round(_effective_k(lab_r), 2),
    }


# ===========================================================================
# Future item-facet outcomes (modal outfit_role / trend_phase over val purchases)
# ===========================================================================
def add_future_facets(fut: pd.DataFrame, customer_ids: list[str]) -> pd.DataFrame:
    df_axis = pd.read_parquet(
        MATRIX_PATH, columns=["article_id", "e2_outfit_role", "e2_trend_phase_actual"]
    )
    df_axis["article_id"] = df_axis["article_id"].astype(str)
    con = duckdb.connect()
    val = con.execute(
        f"SELECT CAST(customer_id AS VARCHAR) c, CAST(article_id AS VARCHAR) a FROM read_parquet('{VAL_TXN}')"
    ).fetchdf()
    con.close()
    role = dict(zip(df_axis["article_id"], df_axis["e2_outfit_role"].astype(str)))
    phase = dict(zip(df_axis["article_id"], df_axis["e2_trend_phase_actual"].astype(str)))
    val["role"] = val["a"].map(role)
    val["phase"] = val["a"].map(phase)
    mr = val.dropna(subset=["role"]).groupby("c")["role"].agg(lambda s: s.mode().iloc[0])
    mp = val.dropna(subset=["phase"]).groupby("c")["phase"].agg(lambda s: s.mode().iloc[0])
    fut = fut.merge(
        mr.rename("fut_outfit_role"), left_on="customer_id", right_index=True, how="left"
    )
    fut = fut.merge(
        mp.rename("fut_trend_phase"), left_on="customer_id", right_index=True, how="left"
    )
    return fut


# ===========================================================================
# Figure
# ===========================================================================
def make_figure(cells: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:  # noqa: BLE001
        print(f"[fig] skipped ({e})", flush=True)
        return
    sns.set_theme(style="white", context="notebook")
    by = {(c["axis"], c["use"]): c for c in cells}
    lift = np.full((len(USER_REPS), len(USES)), np.nan)
    annot = np.empty((len(USER_REPS), len(USES)), dtype=object)
    gl = {"PASS": "✓", "MARGINAL": "~", "NO": "✗", "N/A-SEMANTICS": "—", "N/A-DATA": "—"}
    for i, rp in enumerate(USER_REPS):
        for j, us in enumerate(USES):
            c = by.get((rp, us), {})
            v, lv = c.get("lift_value"), c.get("lift_verdict", "")
            if isinstance(v, (int, float)) and v == v:
                lift[i, j] = v
                annot[i, j] = f"{gl.get(lv,'')}\n{v:+.3f}"
            else:
                annot[i, j] = gl.get(lv, "·")
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    sns.heatmap(
        np.nan_to_num(lift),
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        center=0,
        xticklabels=["① Control\n(facet)", "② Lead-time", "③ Merch", "④ Audience\n(KPI)"],
        yticklabels=[r.replace("_", "\n") for r in USER_REPS],
        linewidths=1.2,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 10},
        cbar_kws={"label": "incr. macro-F1 vs demographic (future)"},
    )
    ax.set_title(
        "KAR user/reasoning leg — does user-enrichment ADD over 11 demographics at ①④?\n"
        "(item-enrichment owns ②③ in E2/E2b; FUTURE-holdout; ✓pass ~marginal ✗no —n/a)",
        fontsize=9.5,
    )
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_PATH}", flush=True)


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    quick = "--quick" in sys.argv
    n_cohort = 12000 if quick else 40000
    n_perm = 300 if quick else 1000
    e2_mtime, e2b_mtime = E2_JSON.stat().st_mtime, E2B_JSON.stat().st_mtime

    print("=" * 80, flush=True)
    print(
        f"PROBE E2c — user-side value (KAR reasoning leg) {'[QUICK]' if quick else ''}", flush=True
    )
    print("=" * 80, flush=True)

    ids = select_cohort(n_sample=n_cohort, seed=SEED)
    reps = build_user_representations(ids)
    demo = build_demographic(ids)
    fut = add_future_facets(build_future_outcomes(ids), ids)
    print(
        f"cohort={len(ids)} | future outcomes: "
        + ", ".join(
            f"{o}({fut[o].notna().mean():.2f})" for o in AUDIENCE_OUTCOMES + CONTROL_OUTCOMES
        ),
        flush=True,
    )

    cells: list[dict] = []
    for rep_name in USER_REPS:
        print(f"\n[rep] {rep_name}", flush=True)
        c_ctl = cell_control(rep_name, reps[rep_name], demo, fut)
        c_lead = _na(rep_name, "leadtime", "N/A-SEMANTICS", note="user rep has no item-momentum")
        c_merch = _na(
            rep_name,
            "merch",
            "N/A-SEMANTICS",
            note="per-item outcome; user rep not the grouping var",
        )
        c_aud = cell_audience(rep_name, reps[rep_name], demo, fut, n_perm)
        cells += [c_ctl, c_lead, c_merch, c_aud]
        for c in [c_ctl, c_lead, c_merch, c_aud]:
            print(f"   {c['use']:9s} lift={c.get('lift_value')} -> {c['lift_verdict']}", flush=True)

    user_pass = [f"{c['axis']}→{c['use']}" for c in cells if c["lift_verdict"] == "PASS"]
    if user_pass:
        decision = (
            f"E2c KAR-SYMMETRY CONFIRMED — user-reasoning ADDS future-predictive value over "
            f"demographics at {user_pass} (USER cells item-enrichment couldn't reach). Item→②③, user→①④."
        )
    else:
        decision = (
            "E2c SHARPER NEGATIVE — even user-side LLM reasoning does not beat 11 demographic "
            "features for control/audience on FUTURE behavior; LLM value localized to item merch/trend."
        )
    print("\n" + "=" * 80 + f"\n  {decision}\n" + "=" * 80, flush=True)

    result = {
        "probe": "E2c_user_value",
        "seed": SEED,
        "quick": quick,
        "n_cohort": len(ids),
        "user_reps": USER_REPS,
        "uses": USES,
        "baseline": "11 demographic user features",
        "audience_outcomes": AUDIENCE_OUTCOMES,
        "control_outcomes": CONTROL_OUTCOMES,
        "value_matrix": cells,
        "user_pass_cells": user_pass,
        "decision": decision,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    make_figure(cells)
    assert (
        E2_JSON.stat().st_mtime == e2_mtime and E2B_JSON.stat().st_mtime == e2b_mtime
    ), "E2/E2b modified!"
    print(f"Confirmation: JSON -> {RESULT_PATH}; E2/E2b canonical untouched", flush=True)


if __name__ == "__main__":
    main()
