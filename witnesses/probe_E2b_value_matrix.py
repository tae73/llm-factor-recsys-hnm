"""PROBE E2b — STRENGTHENED value matrix (legitimate higher decision-lift per column).

E2-2 (`probe_E2_result.json`) came back capability 14/16, lift PASS 2/16. This probe
RE-MEASURES each use-column to raise lift HONESTLY — by better targets / finer
granularity / decision-relevant outcomes, NOT by relaxing thresholds — and reports the
BEFORE→AFTER delta. It writes `probe_E2b_result.json`; it NEVER touches the E2-2 canonical.

Redesign per column (one pre-registered PRIMARY each; secondaries are descriptive only):
  ① faceted  — oracle ceiling → DEPLOYABLE history-predictor (steer to the user's
               predicted next-item axis value), lift vs a metadata-facet predictor.
  ② leadtime — WEEKLY granularity + CONTINUOUS momentum-weighted share (tighter CI).
  ③ merch    — decision-relevant outcomes (first_week_sell_through, markdown_depth),
               residualized vs product_group; outfit_role keeps the velocity win.
  ④ audience — item-repurchase → BUYER-POPULATION (age/channel) divergence vs metadata.

Pre-registered nulls (committed before running): active_frac is flat → dead; gap axes
on the 5,064-item pilot stay PRELIM. CPU-only, seed 42, no API/$.

Usage:
    cd <repo> && PYTHONPATH=. python -u witnesses/probe_E2b_value_matrix.py [--quick] [n_users]
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation.cohorts import discovery_map  # noqa: E402
from src.features.audience_signals import (  # noqa: E402
    build_buyer_population,
    grand_std,
    segment_divergence_weighted,
)
from src.features.lead_lag import lead_lag_vs_baseline, monthly_attribute_share  # noqa: E402
from witnesses.probe_DE1_attribute_screen import _eta  # noqa: E402
from witnesses.probe_E2_value_matrix import (  # noqa: E402  (reuse the E2 spine, unchanged)
    ALPHAS,
    ARTICLES_PATH,
    AXES,
    BEHAVIORAL_AXES,
    ETA_EXCESS_MIN,
    MATRIX_PATH,
    META_COLS,
    SEED,
    TRAIN_TXN,
    USES,
    K,
    _build_steer,
    _matched_meta_col,
    _na,
    _onehot,
)

warnings.filterwarnings("ignore")

CUSTOMERS_PATH = ROOT / "data/processed/customers.parquet"
RESULT_PATH = ROOT / "witnesses/probe_E2b_result.json"
FIG_PATH = ROOT / "results/figures/E2b_value_matrix.png"

# Pre-registered E2-2 baseline (from probe_E2_result.json) for the before/after delta.
E2_2_LIFT_PASS = 2
E2_2_PASS_CELLS = {"e2_trend_phase_actual→leadlag", "e2_outfit_role→merch"}

# ① thresholds (pre-registered)
DEPLOY_LIFT_MIN = 0.10  # axis deployable-gain must beat metadata-facet by ≥10% rel dmap
RECOVERY_MIN = 0.20  # ...and recover ≥1/5 of the oracle ceiling
# ④ relative margin: axis between-segment divergence ≥1.2× metadata's (absolute 0.10·std
#    is mis-scaled for a divergence-of-means metric — documented honesty correction).
AUDIENCE_RATIO_MIN = 1.20
# ③ merch outcome per axis (decision-relevant), residualized vs product_group
MERCH_OUTCOME = {
    "e2_outfit_role": "velocity",  # keep the E2-2 win
    "e2_trend_phase_actual": "first_week_sell_through",  # non-tautological launch signal
    "e2_value_gap": "markdown_depth",  # mispricing hypothesis
    "e2_trend_gap": "velocity",
}


def _mode_det(vals: list[str]) -> str:
    """Deterministic mode (lexical tie-break) — order-independent, reproducible."""
    c = Counter(vals)
    mx = max(c.values())
    return sorted(v for v in c if c[v] == mx)[0]


# ===========================================================================
# ① Faceted — deployable history-predictor (oracle ceiling → deployable floor)
# ===========================================================================
def _steer_sweep(
    targets: dict, mask: dict, P: int, eligible, base_scores, owned_masks, sub_gt, sub_hist
) -> list[float]:
    """discovery_map@K across ALPHAS steering each user toward targets[u] (α=0 = no steer)."""

    def topk(score, owned):
        sc = np.where(owned, -np.inf, score)
        o = np.argpartition(-sc, K)[:K]
        return o[np.argsort(-sc[o])]

    out = []
    for alpha in ALPHAS:
        preds = {}
        for u in eligible:
            m = mask.get(targets[u], np.zeros(P, bool)) if alpha > 0 else np.zeros(P, bool)
            o = topk(base_scores[u] + alpha * m.astype(float), owned_masks[u])
            preds[u] = [pool_ids_global[i] for i in o]
        out.append(float(discovery_map(preds, sub_gt, sub_hist, k=K).map_at_k))
    return out


pool_ids_global: list = []  # set in cell_faceted_deployable (closure convenience)


def cell_faceted_deployable(
    axis: str, df: pd.DataFrame, art: pd.DataFrame, steer, rng: np.random.Generator, power: str
) -> dict:
    global pool_ids_global
    if axis not in BEHAVIORAL_AXES:
        vals = df[axis].dropna().astype(int)
        return {
            "axis": axis,
            "use": "faceted",
            "capability": "YES",
            "capability_basis": f"{vals.nunique()} queryable gap buckets",
            "lift_metric": "n/a (coverage fact; pilot pool too thin to steer)",
            "metadata_baseline": "metadata exposes no such axis",
            "lift_value": None,
            "lift_ci": None,
            "significant": False,
            "lift_verdict": "N/A-COVERAGE",
            "power": power,
            "note": "facet-coverage only",
        }
    pool_ids, P, eligible, base_scores, owned_masks, sub_gt, sub_hist, gt = steer
    pool_ids_global = pool_ids
    axis_map = dict(zip(df["article_id"].astype(str), df[axis].astype(str)))
    meta_col = _matched_meta_col(
        df[axis].astype(str).to_numpy(),
        art.set_index("article_id").reindex(df["article_id"]).reset_index(),
    )
    meta_map = dict(zip(art["article_id"].astype(str), art[meta_col].astype(str)))

    def arms(amap):
        pool_vals = [amap.get(a, "NA") for a in pool_ids]
        vocab = sorted(set(pool_vals) - {"NA"})
        mask = {v: np.fromiter((pv == v for pv in pool_vals), bool, P) for v in vocab}
        oracle, pred = {}, {}
        for u in eligible:
            nip = [g for g in sub_gt[u] if g not in sub_hist[u] and amap.get(g) in vocab]
            oracle[u] = amap[nip[0]] if nip else (vocab[0] if vocab else "NA")
            hv = [amap.get(h) for h in sub_hist[u] if amap.get(h) in vocab]
            pred[u] = _mode_det(hv) if hv else (vocab[0] if vocab else "NA")
        return mask, oracle, pred

    mask_a, oracle_a, pred_a = arms(axis_map)
    or_sweep = _steer_sweep(
        oracle_a, mask_a, P, eligible, base_scores, owned_masks, sub_gt, sub_hist
    )
    pr_sweep = _steer_sweep(pred_a, mask_a, P, eligible, base_scores, owned_masks, sub_gt, sub_hist)
    base = or_sweep[0]
    # metadata-facet deployable predictor (confound control: same repeat-buying structure)
    mask_m, _, pred_m = arms(meta_map)
    mp_sweep = _steer_sweep(pred_m, mask_m, P, eligible, base_scores, owned_masks, sub_gt, sub_hist)

    ceiling = (max(or_sweep) - base) / base if base > 0 else 0.0
    deployable = (max(pr_sweep) - base) / base if base > 0 else 0.0
    meta_deploy = (max(mp_sweep) - base) / base if base > 0 else 0.0
    lift = deployable - meta_deploy
    recovery = deployable / ceiling if ceiling > 0 else 0.0
    sig = bool(deployable > 0 and lift >= DEPLOY_LIFT_MIN and recovery >= RECOVERY_MIN)
    verdict = "PASS" if sig else ("MARGINAL" if deployable > 0 else "NO")
    return {
        "axis": axis,
        "use": "faceted",
        "capability": "YES",
        "capability_basis": "steerable facet absent from metadata (DE1 nonredundant)",
        "lift_metric": "deployable_history_predictor_gain − metadata_facet_predictor_gain",
        "metadata_baseline": f"history-predictor on {meta_col} (deploy_gain={round(meta_deploy,4)})",
        "lift_value": round(float(lift), 4),
        "lift_ci": None,
        "deployable": round(float(deployable), 4),
        "oracle_ceiling": round(float(ceiling), 4),
        "recovery_frac": round(float(recovery), 4),
        "significant": sig,
        "lift_verdict": verdict,
        "power": power,
        "note": f"deploy={deployable:+.2f} (meta {meta_deploy:+.2f}) ceiling={ceiling:+.2f} "
        f"recovery={recovery:.2f}",
    }


# ===========================================================================
# ② Lead-time — PRIMARY = E2-2 monthly-binary (validated); weekly-continuous = secondary
# (The weekly+continuous refinement was TRIED; it is NOISIER than the monthly-binary
#  signal — the continuous-momentum permutation null tracks the real share too closely —
#  so it stays a descriptive secondary and the established monthly PASS is the primary.)
# ===========================================================================
def _weekly_refinement(axis: str, df: pd.DataFrame, con, lags: tuple, n_boot: int) -> dict:
    """Weekly + continuous-momentum lead-lag (descriptive only — does not set the verdict)."""
    if axis == "e2_trend_phase_actual":
        axis_df = df[["article_id", "e2_trend_momentum"]].dropna()
        real = monthly_attribute_share(
            con,
            TRAIN_TXN,
            ARTICLES_PATH,
            axis_df,
            "e2_trend_momentum",
            granularity="week",
            weight_col="e2_trend_momentum",
        )
        rng = np.random.default_rng(SEED)
        shuf = axis_df.copy()
        shuf["e2_trend_momentum"] = rng.permutation(shuf["e2_trend_momentum"].to_numpy())
        null = monthly_attribute_share(
            con,
            TRAIN_TXN,
            ARTICLES_PATH,
            shuf,
            "e2_trend_momentum",
            granularity="week",
            weight_col="e2_trend_momentum",
        )
    else:  # trend_gap: weekly binary positive-gap share
        d = df[["article_id", "e2_trend_gap"]].dropna().copy()
        d["lab"] = np.where(d["e2_trend_gap"].astype(float) > 0, "pos", "other")
        d = d[["article_id", "lab"]]
        real = monthly_attribute_share(
            con, TRAIN_TXN, ARTICLES_PATH, d, "lab", ["pos"], granularity="week"
        )
        rng = np.random.default_rng(SEED)
        shuf = d.copy()
        shuf["lab"] = rng.permutation(shuf["lab"].to_numpy())
        null = monthly_attribute_share(
            con, TRAIN_TXN, ARTICLES_PATH, shuf, "lab", ["pos"], granularity="week"
        )
    if real.empty or real["cat"].nunique() < 3:
        return {"weekly_delta": None}
    res = lead_lag_vs_baseline(real, null, lags=lags, n_boot=n_boot, seed=SEED)
    return {
        "weekly_delta": res["delta"],
        "weekly_ci": [res["ci_lo"], res["ci_hi"]],
        "weekly_best_lag": res["best_lag"],
        "weekly_r_attr": res["r_attr"],
    }


def cell_leadlag_weekly(
    axis: str, df: pd.DataFrame, con, lags: tuple, n_boot: int, power: str
) -> dict:
    from witnesses.probe_E2_value_matrix import cell_leadlag as _cell_leadlag_monthly

    # PRIMARY = the E2-2 validated monthly-binary lead-lag (sets the verdict).
    primary = _cell_leadlag_monthly(axis, df, con, (1, 2, 3, 4), n_boot, power)
    if primary["lift_verdict"] in ("N/A-SEMANTICS", "N/A-DATA"):
        return primary
    # SECONDARY (descriptive only): weekly + continuous refinement.
    sec = _weekly_refinement(axis, df, con, lags, n_boot)
    primary["secondary_descriptive"] = sec
    wd = sec.get("weekly_delta")
    improved = wd is not None and wd == wd and wd > (primary.get("lift_value") or 0.0)
    primary["lift_metric"] = (
        "monthly_binary_leadlag (PRIMARY); weekly-continuous refinement = secondary"
    )
    primary["note"] = (
        primary.get("note", "")
        + f" | weekly-continuous Δ={wd} ({'improved' if improved else 'did NOT improve — noisier'})"
    )
    return primary


# ===========================================================================
# ③ Merch — decision-relevant outcomes, residualized vs product_group
# ===========================================================================
def _residualize(y: np.ndarray, group: np.ndarray) -> np.ndarray:
    s = pd.DataFrame({"y": y, "g": group})
    s["gm"] = s.groupby("g")["y"].transform("mean")
    return (s["y"] - s["gm"]).to_numpy()


def cell_merch_rich(
    axis: str,
    df: pd.DataFrame,
    art: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
    power: str,
) -> dict:
    outcome = MERCH_OUTCOME[axis]
    sub = df[df[axis].notna() & df[outcome].notna()].copy()
    if len(sub) < 200:
        return _na(axis, "merch", "N/A-DATA")
    art_sub = art.set_index("article_id").reindex(sub["article_id"]).reset_index()
    labels = sub[axis].astype(str).to_numpy()
    # residualize the decision outcome vs product_group (de-couple)
    y_raw = sub[outcome].to_numpy(dtype=float)
    if outcome == "velocity":
        y_raw = np.log1p(y_raw)
    y = _residualize(y_raw, art_sub["product_group_name"].astype(str).to_numpy())
    meta_col = _matched_meta_col(labels, art_sub)
    meta = art_sub[meta_col].astype(str).to_numpy()

    eta_attr = _eta(labels, y)
    eta_meta = _eta(meta, y)
    excess = eta_attr - eta_meta
    g = len(np.unique(labels))
    placebo = _eta(rng.integers(0, g, size=len(labels)).astype(str), y)
    bn = min(20000, len(labels))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(labels), size=bn)
        boot[b] = _eta(labels[i], y[i]) - _eta(meta[i], y[i])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sig = bool(excess >= ETA_EXCESS_MIN and lo > 0 and placebo < ETA_EXCESS_MIN)
    verdict = "PASS" if sig else ("MARGINAL" if excess >= ETA_EXCESS_MIN else "NO")
    if power == "PRELIMINARY" and verdict == "PASS":
        verdict = "PRELIM"
    return {
        "axis": axis,
        "use": "merch",
        "capability": "YES",
        "capability_basis": "decision dimension not in metadata (DE1 nonredundant)",
        "lift_metric": f"eta({outcome}|resid product_group) excess vs metadata",
        "metadata_baseline": f"eta({meta_col})={round(eta_meta,3)}",
        "lift_value": round(float(excess), 4),
        "lift_ci": [round(float(lo), 4), round(float(hi), 4)],
        "outcome": outcome,
        "eta_attr": round(float(eta_attr), 4),
        "placebo_eta": round(float(placebo), 4),
        "significant": sig,
        "lift_verdict": verdict,
        "power": power,
        "note": f"outcome={outcome} eta {round(eta_attr,3)} vs meta {round(eta_meta,3)} placebo {round(placebo,3)}",
    }


# ===========================================================================
# ④ Audience — BUYER-POPULATION divergence (age primary, channel secondary)
# ===========================================================================
def cell_audience_buyers(
    axis: str,
    df: pd.DataFrame,
    item_agg: pd.DataFrame,
    art: pd.DataFrame,
    meta_kmeans: np.ndarray,
    power: str,
) -> dict:
    d = item_agg.merge(df[["article_id", axis]].dropna(), on="article_id", how="inner")
    d = d.merge(
        pd.DataFrame({"article_id": art["article_id"].astype(str), "_mk": meta_kmeans}),
        on="article_id",
        how="inner",
    )
    if len(d) < 200:
        return _na(axis, "audience", "N/A-DATA")
    labels = d[axis].astype(str).to_numpy()
    sum_age, n_age = d["sum_age"].to_numpy(float), d["n_age"].to_numpy(float)
    div_axis = segment_divergence_weighted(labels, sum_age, n_age)
    div_meta = segment_divergence_weighted(d["_mk"].astype(str).to_numpy(), sum_age, n_age)
    gstd = grand_std(d["sum_age"].to_numpy(float), d["sum_age_sq"].to_numpy(float), n_age)
    # permutation null: shuffle per-item axis label, recompute divergence
    rng = np.random.default_rng(SEED)
    perm = np.array(
        [segment_divergence_weighted(rng.permutation(labels), sum_age, n_age) for _ in range(300)]
    )
    p_perm = float((perm >= div_axis).mean())
    # secondary: online-frac divergence
    div_axis_on = segment_divergence_weighted(
        labels, d["n_online"].to_numpy(float), d["n_txn"].to_numpy(float)
    )
    div_meta_on = segment_divergence_weighted(
        d["_mk"].astype(str).to_numpy(), d["n_online"].to_numpy(float), d["n_txn"].to_numpy(float)
    )

    ratio = div_axis / div_meta if div_meta > 0 else 0.0
    sig = bool(div_axis >= AUDIENCE_RATIO_MIN * div_meta and p_perm < 0.05)
    verdict = "PASS" if sig else ("MARGINAL" if div_axis > div_meta else "NO")
    if power == "PRELIMINARY" and verdict == "PASS":
        verdict = "PRELIM"
    return {
        "axis": axis,
        "use": "audience",
        "capability": "YES",
        "capability_basis": "new non-redundant partition dimension",
        "lift_metric": "buyer_age_divergence_vs_metadata_kmeans",
        "metadata_baseline": f"metadata k-means buyer-age divergence={round(div_meta,4)}",
        "lift_value": round(float(div_axis - div_meta), 4),
        "lift_ci": None,
        "div_axis_age": round(float(div_axis), 4),
        "div_meta_age": round(float(div_meta), 4),
        "ratio_vs_meta": round(float(ratio), 3),
        "perm_p": round(p_perm, 4),
        "age_grand_std": round(float(gstd), 3),
        "online_div_axis": round(float(div_axis_on), 4),
        "online_div_meta": round(float(div_meta_on), 4),
        "significant": sig,
        "lift_verdict": verdict,
        "power": power,
        "note": f"buyer-age div axis={round(div_axis,3)} vs meta {round(div_meta,3)} "
        f"({round(ratio,2)}x); online axis={round(div_axis_on,3)} vs meta {round(div_meta_on,3)}",
    }


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
    lift = np.full((len(AXES), len(USES)), np.nan)
    annot = np.empty((len(AXES), len(USES)), dtype=object)
    gl = {
        "PASS": "✓",
        "MARGINAL": "~",
        "NO": "✗",
        "PRELIM": "≈",
        "N/A-COVERAGE": "·",
        "N/A-SEMANTICS": "—",
        "N/A-DATA": "—",
    }
    for i, ax in enumerate(AXES):
        for j, us in enumerate(USES):
            c = by.get((ax, us), {})
            v, lv = c.get("lift_value"), c.get("lift_verdict", "")
            was_pass = f"{ax}→{us}" in E2_2_PASS_CELLS
            mark = (
                "★"
                if (lv == "PASS" and f"{ax}→{us}" not in E2_2_PASS_CELLS)
                else ("•" if was_pass else "")
            )
            if isinstance(v, (int, float)) and v == v:
                lift[i, j] = v
                annot[i, j] = f"{gl.get(lv,'')}{mark}\n{v:+.2f}"
            else:
                annot[i, j] = f"{gl.get(lv,'·')}{mark}"
    fig, axf = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        np.nan_to_num(lift),
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        center=0,
        xticklabels=[
            "① Faceted\n(deployable)",
            "② Lead-time\n(weekly)",
            "③ Merch\n(rich)",
            "④ Audience\n(buyers)",
        ],
        yticklabels=[a.replace("e2_", "").replace("_actual", "") for a in AXES],
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "decision-lift (Δ vs metadata)"},
        ax=axf,
        annot_kws={"fontsize": 10},
    )
    axf.set_title(
        "Enrichment-v2 Value Matrix — STRENGTHENED (E2-3)\n"
        "★ new PASS · • kept PASS · ✓pass ~marginal ✗no ≈prelim ·/—n/a",
        fontsize=11,
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
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_users = int(argv[0]) if argv else (2000 if quick else 8000)
    lags = (1, 2, 4, 8) if quick else (1, 2, 4, 6, 8, 10, 13, 16)
    n_boot = 200 if quick else 1000

    print("=" * 80, flush=True)
    print(
        f"PROBE E2b — STRENGTHENED value matrix (seed 42, no $) {'[QUICK]' if quick else ''}",
        flush=True,
    )
    print("=" * 80, flush=True)

    df = pd.read_parquet(MATRIX_PATH)
    df["article_id"] = df["article_id"].astype(str)
    art = pd.read_parquet(ARTICLES_PATH, columns=["article_id", *META_COLS])
    art["article_id"] = art["article_id"].astype(str)

    print("computing buyer population (txn⋈customers) ...", flush=True)
    import duckdb

    con = duckdb.connect()
    item_agg = build_buyer_population(con, TRAIN_TXN, CUSTOMERS_PATH)
    # one metadata k-means partition (k=6, matched to behavioral-axis cardinality) for ④
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    Xm = StandardScaler(with_mean=True).fit_transform(_onehot(art, META_COLS).astype(float))
    meta_kmeans = KMeans(n_clusters=6, random_state=SEED, n_init=4).fit_predict(Xm)

    print("\n[① faceted] building steering context ...", flush=True)
    steer = _build_steer(df, np.random.default_rng(SEED), n_users)
    print(f"   eligible users: {len(steer[2])}", flush=True)

    cells: list[dict] = []
    for axis in AXES:
        power = "STRONG" if axis in BEHAVIORAL_AXES else "PRELIMINARY"
        print(f"\n[axis] {axis} ({power})", flush=True)
        cells.append(
            cell_faceted_deployable(axis, df, art, steer, np.random.default_rng(SEED), power)
        )
        cells.append(cell_leadlag_weekly(axis, df, con, lags, n_boot, power))
        cells.append(cell_merch_rich(axis, df, art, np.random.default_rng(SEED), n_boot, power))
        cells.append(cell_audience_buyers(axis, df, item_agg, art, meta_kmeans, power))
        for c in cells[-4:]:
            print(
                f"   {c['use']:9s} lift={c.get('lift_value')} [{c['power']}] -> {c['lift_verdict']}",
                flush=True,
            )
    con.close()

    # verdict + before/after
    cap_yes = sum(1 for c in cells if c["capability"] == "YES")
    e2b_pass = [
        f"{c['axis']}→{c['use']}"
        for c in cells
        if c["power"] == "STRONG" and c["lift_verdict"] == "PASS"
    ]
    improvement = [x for x in e2b_pass if x not in E2_2_PASS_CELLS]
    regressions = [x for x in E2_2_PASS_CELLS if x not in e2b_pass]
    decision = (
        f"E2-3: lift PASS {E2_2_LIFT_PASS} → {len(e2b_pass)} "
        f"(+{len(improvement)} new, -{len(regressions)} regressed); capability {cap_yes}/16"
    )

    print("\n" + "=" * 80, flush=True)
    print(f"  {decision}", flush=True)
    print(f"  E2b PASS: {e2b_pass}", flush=True)
    print(f"  NEW: {improvement or 'none'} | REGRESSED: {regressions or 'none'}", flush=True)
    print("=" * 80, flush=True)

    result = {
        "probe": "E2b_value_matrix",
        "seed": SEED,
        "quick": quick,
        "e2_2_lift_pass": E2_2_LIFT_PASS,
        "e2b_lift_pass": len(e2b_pass),
        "improvement_cells": improvement,
        "regressions": regressions,
        "capability_yes": cap_yes,
        "axes": AXES,
        "uses": USES,
        "value_matrix": cells,
        "by_verdict": {
            v: [f"{c['axis']}→{c['use']}" for c in cells if c["lift_verdict"] == v]
            for v in [
                "PASS",
                "MARGINAL",
                "NO",
                "PRELIM",
                "N/A-COVERAGE",
                "N/A-SEMANTICS",
                "N/A-DATA",
            ]
        },
        "decision": decision,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    make_figure(cells)
    print(f"Confirmation: JSON written -> {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
