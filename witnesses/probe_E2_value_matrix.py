"""PROBE E2 — 4-USE VALUE MATRIX for the enrichment-v2 decision-axes.

WHY — DE1-v2 proved 4 enrichment-v2 axes are discriminable + non-redundant vs both
metadata AND L1 (`e2_trend_phase_actual`, `e2_outfit_role` SALVAGEABLE; `e2_value_gap`,
`e2_trend_gap` non-redundant but behaviorally inert). This probe fills the value matrix
(design §5): rows = those 4 axes, cols = 4 use-cases. Each cell carries TWO separate
values — (a) CAPABILITY gain (does the axis expose a decision dimension metadata lacks,
established by construction + DE1 non-redundancy) and (b) measured behavioral DECISION-
LIFT vs a metadata baseline (with significance). The thesis being characterized is
"LLM-attribute value is not prediction but discriminable decision-AXES"; the matrix is
PREDICTED to come back capability-dense / lift-sparse, and the verdict treats that as
THESIS-CONFIRMING, not failure.

Columns: ① Faceted/control (D3-style steering) · ② Trend lead-time (lead-lag) ·
③ Merchandising (sell-through eta excess + placebo) · ④ Marketing audience (segment
behavioral divergence). Reuses D3/D5/DE1 machinery. CPU-only, seed 42, no API/$.

Usage:
    cd <repo> && PYTHONPATH=. python -u witnesses/probe_E2_value_matrix.py [--quick] [n_users]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation.cohorts import discovery_map  # noqa: E402
from src.features.lead_lag import lead_lag_vs_baseline, monthly_attribute_share  # noqa: E402
from witnesses._probe_common import OUT_DIR, load_variants  # noqa: E402
from witnesses.probe_D5_value_matrix import _effective_k, _repurchase_rate  # noqa: E402
from witnesses.probe_DE1_attribute_screen import _cramers_v, _eta  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42

MATRIX_PATH = ROOT / "data/knowledge/enrichment_v2/matrix_axes.parquet"
ARTICLES_PATH = ROOT / "data/processed/articles.parquet"
TRAIN_TXN = ROOT / "data/processed/train_transactions.parquet"
IMMEDIATE_GT = ROOT / "data/processed/immediate_ground_truth.json"
RESULT_PATH = OUT_DIR / "probe_E2_result.json"
FIG_PATH = ROOT / "results/figures/E2_value_matrix.png"

META_COLS = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
    "index_name",
    "graphical_appearance_name",
]

AXES = ["e2_trend_phase_actual", "e2_outfit_role", "e2_value_gap", "e2_trend_gap"]
BEHAVIORAL_AXES = {"e2_trend_phase_actual", "e2_outfit_role"}
USES = ["faceted", "leadlag", "merch", "audience"]

# Steering (① — D3 spine)
K, POOL, RECENT_DATE, MIN_HIST = 12, 5000, "2020-05-01", 3
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
STEER_COST_MODEST = 0.30
# Thresholds (pre-registered)
ETA_EXCESS_MIN = 0.05
LEADLAG_DELTA_MIN = 0.10


# ===========================================================================
# Data
# ===========================================================================
def _matched_meta_col(labels: np.ndarray, art: pd.DataFrame) -> str:
    """Metadata column that best predicts the axis (Cramér's V) — the DE1 baseline."""
    best, bv = META_COLS[0], -1.0
    for mc in META_COLS:
        if mc not in art.columns:
            continue
        v = _cramers_v(labels, art[mc].astype(str).to_numpy())
        if v > bv:
            bv, best = v, mc
    return best


def _onehot(art: pd.DataFrame, cols: list[str]) -> np.ndarray:
    cols = [c for c in cols if c in art.columns]
    return pd.get_dummies(art[cols].astype("category"), dummy_na=False).to_numpy(dtype=np.float32)


# ===========================================================================
# ③ Merchandising — sell-through eta excess (+ placebo)
# ===========================================================================
def cell_merch(
    axis: str,
    df: pd.DataFrame,
    art: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
    power: str,
) -> dict:
    sub = df[df[axis].notna() & df["velocity"].notna()].copy()
    if len(sub) < 200:
        return _na(axis, "merch", "N/A-DATA")
    labels = sub[axis].astype(str).to_numpy()
    y = np.log1p(sub["velocity"].to_numpy(dtype=float))
    art_sub = art.set_index("article_id").reindex(sub["article_id"]).reset_index()
    meta_col = _matched_meta_col(labels, art_sub)
    meta = art_sub[meta_col].astype(str).to_numpy()

    eta_attr = _eta(labels, y)
    eta_meta = _eta(meta, y)
    excess = eta_attr - eta_meta
    # placebo: random partition with same #groups (guards the axis-derived↔sales coupling)
    g = len(np.unique(labels))
    placebo = _eta(rng.integers(0, g, size=len(labels)).astype(str), y)
    # bootstrap CI of excess (subsample ≤20K items/boot for speed; CI is stable)
    bn = min(20000, len(labels))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(labels), size=bn)
        boot[b] = _eta(labels[i], y[i]) - _eta(meta[i], y[i])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    sig = bool(excess >= ETA_EXCESS_MIN and lo > 0 and placebo < ETA_EXCESS_MIN)
    verdict = "PASS" if sig else "MARGINAL" if excess >= ETA_EXCESS_MIN else "NO"
    if power == "PRELIMINARY" and verdict == "PASS":
        verdict = "PRELIM"
    coupling = (
        " [axis derived from sales/co-purchase → coupling; excess is vs metadata, placebo≈0]"
        if axis in BEHAVIORAL_AXES
        else ""
    )
    return {
        "axis": axis,
        "use": "merch",
        "capability": "YES",
        "capability_basis": "co-purchase/momentum role not in metadata (DE1 nonredundant)",
        "lift_metric": "sellthrough_eta_excess_vs_metadata",
        "metadata_baseline": f"eta({meta_col}, log velocity)={round(eta_meta,3)}",
        "lift_value": round(float(excess), 4),
        "lift_ci": [round(float(lo), 4), round(float(hi), 4)],
        "eta_attr": round(float(eta_attr), 4),
        "placebo_eta": round(float(placebo), 4),
        "significant": sig,
        "lift_verdict": verdict,
        "power": power,
        "note": f"eta(attr)={round(eta_attr,3)} vs meta {round(eta_meta,3)}; placebo {round(placebo,3)}{coupling}",
    }


# ===========================================================================
# ④ Marketing audience — between-segment behavioral divergence
# ===========================================================================
def _segment_divergence(labels: np.ndarray, y: np.ndarray) -> float:
    """Std of per-segment mean(y), weighted by segment size (behavioral spread)."""
    vals, inv = np.unique(labels, return_inverse=True)
    means = np.array([y[inv == i].mean() for i in range(len(vals))])
    sizes = np.array([(inv == i).sum() for i in range(len(vals))], dtype=float)
    w = sizes / sizes.sum()
    grand = (w * means).sum()
    return float(np.sqrt((w * (means - grand) ** 2).sum()))


def cell_audience(
    axis: str, df: pd.DataFrame, art: pd.DataFrame, rng: np.random.Generator, power: str
) -> dict:
    # Behavioral var = repurchase_rate (an INDEPENDENT loyalty KPI), NOT velocity — the
    # behavior-derived axes are defined from sales/co-purchase, so velocity-divergence
    # would be tautological. Repurchase loyalty is the honest "audience distinctness" test.
    sub = df[df[axis].notna() & df["repurchase_rate"].notna()].copy()
    if len(sub) < 200:
        return _na(axis, "audience", "N/A-DATA")
    labels = sub[axis].astype(str).to_numpy()
    y = sub["repurchase_rate"].to_numpy(dtype=float)
    n_groups = len(np.unique(labels))

    div_axis = _segment_divergence(labels, y)
    eff_axis = _effective_k(labels)
    # metadata partition: KMeans on metadata one-hot at the same cardinality
    art_sub = art.set_index("article_id").reindex(sub["article_id"]).reset_index()
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    Xm = StandardScaler(with_mean=True).fit_transform(_onehot(art_sub, META_COLS).astype(float))
    meta_lab = KMeans(n_clusters=n_groups, random_state=SEED, n_init=4).fit_predict(Xm)
    div_meta = _segment_divergence(meta_lab.astype(str), y)
    eff_meta = _effective_k(meta_lab)

    # permutation null for the axis divergence
    perm = np.array([_segment_divergence(rng.permutation(labels), y) for _ in range(500)])
    p_perm = float((perm >= div_axis).mean())

    # practical margin (honesty guard vs large-n trivial deltas, mirrors D5 PRACTICAL_MARGIN):
    # the divergence improvement over metadata must be ≥10% of the outcome's std.
    margin = 0.10 * float(np.std(y))
    delta = div_axis - div_meta
    sig = bool(delta >= margin and p_perm < 0.05 and eff_axis >= eff_meta - 0.5)
    verdict = "PASS" if sig else ("MARGINAL" if delta > 0 else "NO")
    if power == "PRELIMINARY" and verdict == "PASS":
        verdict = "PRELIM"
    return {
        "axis": axis,
        "use": "audience",
        "capability": "YES",
        "capability_basis": "new non-redundant partition dimension",
        "lift_metric": "segment_repurchase_divergence_vs_metadata",
        "metadata_baseline": f"metadata k-means (k={n_groups}) divergence={round(div_meta,4)}",
        "lift_value": round(float(div_axis - div_meta), 4),
        "lift_ci": None,
        "perm_p": round(p_perm, 4),
        "eff_k_axis": round(eff_axis, 3),
        "eff_k_meta": round(eff_meta, 3),
        "significant": sig,
        "lift_verdict": verdict,
        "power": power,
        "note": f"div(axis)={round(div_axis,4)} vs meta {round(div_meta,4)}; perm p={round(p_perm,3)}",
    }


# ===========================================================================
# ② Trend lead-time — lead-lag vs permutation null
# ===========================================================================
def cell_leadlag(axis: str, df: pd.DataFrame, con, lags: tuple, n_boot: int, power: str) -> dict:
    import duckdb  # noqa: F401

    if axis == "e2_trend_phase_actual":
        target_vals = ["Emerging", "Rising"]
        axis_df = df[["article_id", axis]].dropna()
    elif axis == "e2_trend_gap":
        # sleeper/risk: positive gap = looks-fresh-but-declining; bucket to sign labels
        d = df[["article_id", axis]].dropna().copy()
        d["lab"] = np.where(
            d[axis].astype(float) > 0, "pos", np.where(d[axis].astype(float) < 0, "neg", "zero")
        )
        axis_df, axis, target_vals = (
            d.rename(columns={"lab": "e2_trend_gap_sign"}),
            "e2_trend_gap_sign",
            ["pos"],
        )
    else:
        return _na(axis, "leadlag", "N/A-SEMANTICS", note="no momentum semantics (static role/gap)")

    real = monthly_attribute_share(con, TRAIN_TXN, ARTICLES_PATH, axis_df, axis, target_vals)
    if real.empty or real["cat"].nunique() < 3:
        return _na(axis, "leadlag", "N/A-DATA")
    # permutation null: shuffle the per-item label, rebuild share (a non-momentum control)
    rng = np.random.default_rng(SEED)
    shuf = axis_df.copy()
    shuf[axis] = rng.permutation(shuf[axis].to_numpy())
    null = monthly_attribute_share(con, TRAIN_TXN, ARTICLES_PATH, shuf, axis, target_vals)
    res = lead_lag_vs_baseline(real, null, lags=lags, n_boot=n_boot, seed=SEED)

    delta, lo = res["delta"], res["ci_lo"]
    has = delta == delta and lo == lo  # not NaN
    sig = bool(
        has and res["best_lag"] and res["best_lag"] >= 1 and delta >= LEADLAG_DELTA_MIN and lo > 0
    )
    verdict = "PASS" if sig else ("MARGINAL" if has and delta > 0 else "NO")
    if power == "PRELIMINARY" and verdict == "PASS":
        verdict = "PRELIM"
    return {
        "axis": axis,
        "use": "leadlag",
        "capability": "YES",
        "capability_basis": "sales-momentum dimension absent from metadata",
        "lift_metric": "leadlag_corr_excess_vs_permutation_null",
        "metadata_baseline": f"permuted-label share lead-lag (r={res['r_meta']})",
        "lift_value": delta,
        "lift_ci": [res["ci_lo"], res["ci_hi"]],
        "best_lag": res["best_lag"],
        "r_attr": res["r_attr"],
        "n_categories": res["n_categories"],
        "significant": sig,
        "lift_verdict": verdict,
        "power": power,
        "note": f"best_lag={res['best_lag']} r_attr={res['r_attr']} vs null {res['r_meta']}; {res['n_categories']} cats",
    }


# ===========================================================================
# ① Faceted/control — D3-style oracle steering (behavioral axes) or coverage-fact (gaps)
# ===========================================================================
def _build_steer(df: pd.DataFrame, rng: np.random.Generator, n_users: int):
    import duckdb

    canon_ids, V = load_variants(["L1"])
    l1 = V["L1"]
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}
    con = duckdb.connect()
    pop = con.execute(
        f"SELECT article_id FROM read_parquet('{TRAIN_TXN}') WHERE t_dat >= DATE '{RECENT_DATE}' "
        # deterministic tie-break on article_id so the pool (hence steering) is reproducible
        f"GROUP BY article_id ORDER BY count(*) DESC, article_id LIMIT {POOL}"
    ).fetchall()
    pool_ids = [str(a) for (a,) in pop if str(a) in id_to_idx]
    pool_idx = np.array([id_to_idx[a] for a in pool_ids])
    pool_l1 = l1[pool_idx]
    P = len(pool_ids)
    gt = {str(k): [str(x) for x in v] for k, v in json.loads(IMMEDIATE_GT.read_text()).items()}
    gt_users = pd.DataFrame({"customer_id": list(gt)})
    con.register("gt_users", gt_users)
    hist_rows = con.execute(
        f"SELECT t.customer_id, list(DISTINCT t.article_id) FROM read_parquet('{TRAIN_TXN}') t "
        f"JOIN gt_users g ON t.customer_id=g.customer_id GROUP BY t.customer_id"
    ).fetchall()
    con.close()
    train_history = {str(c): set(str(a) for a in items) for c, items in hist_rows}
    pool_set = set(pool_ids)
    all_users = list(gt)
    rng.shuffle(all_users)
    eligible = []
    for u in all_users:
        if len(eligible) >= n_users:
            break
        hist = train_history.get(u)
        if not hist:
            continue
        if sum(1 for h in hist if h in id_to_idx) < MIN_HIST:
            continue
        if any(g for g in gt[u] if g not in hist and g in pool_set):
            eligible.append(u)
    base_scores, owned_masks = {}, {}
    for u in eligible:
        hist = train_history[u]
        c = l1[[id_to_idx[h] for h in hist if h in id_to_idx]].mean(0)
        c = c / (np.linalg.norm(c) + 1e-9)
        base_scores[u] = pool_l1 @ c
        owned_masks[u] = np.fromiter((a in hist for a in pool_ids), dtype=bool, count=P)
    sub_gt = {u: gt[u] for u in eligible}
    sub_hist = {u: train_history[u] for u in eligible}
    return pool_ids, P, eligible, base_scores, owned_masks, sub_gt, sub_hist, gt


def _steer_axis(
    axis_map: dict,
    pool_ids,
    P,
    eligible,
    base_scores,
    owned_masks,
    sub_gt,
    sub_hist,
    gt,
    rng: np.random.Generator,
) -> dict:
    pool_vals = [axis_map.get(a, "NA") for a in pool_ids]
    vocab = sorted(set(pool_vals) - {"NA"})
    mask = {v: np.fromiter((pv == v for pv in pool_vals), dtype=bool, count=P) for v in vocab}
    ctx, off = {}, {}
    for u in eligible:
        new_in_pool = [g for g in sub_gt[u] if g not in sub_hist[u] and axis_map.get(g) in vocab]
        ctx[u] = axis_map[new_in_pool[0]] if new_in_pool else (vocab[0] if vocab else "NA")
        owned_vals = {axis_map.get(g) for g in gt[u]}
        choices = [v for v in vocab if v not in owned_vals]
        off[u] = str(rng.choice(choices)) if choices else (vocab[0] if vocab else "NA")

    def topk(score, owned):
        sc = np.where(owned, -np.inf, score)
        order = np.argpartition(-sc, K)[:K]
        return order[np.argsort(-sc[order])]

    def run(alpha):
        pc, po, predc, predo = [], [], {}, {}
        for u in eligible:
            base, owned = base_scores[u], owned_masks[u]
            mc = mask.get(ctx[u], np.zeros(P, bool))
            oc = topk(base + alpha * mc.astype(float), owned)
            predc[u] = [pool_ids[i] for i in oc]
            pc.append(float(mc[oc].mean()))
            mo = mask.get(off[u], np.zeros(P, bool))
            oo = topk(base + alpha * mo.astype(float), owned)
            predo[u] = [pool_ids[i] for i in oo]
            po.append(float(mo[oo].mean()))
        return {
            "alpha": alpha,
            "precision_ctx": float(np.mean(pc)),
            "precision_off": float(np.mean(po)),
            "dmap_ctx": float(discovery_map(predc, sub_gt, sub_hist, k=K).map_at_k),
            "dmap_off": float(discovery_map(predo, sub_gt, sub_hist, k=K).map_at_k),
        }

    sweep = [run(a) for a in ALPHAS]
    baseline = sweep[0]["dmap_ctx"]
    hi = [s for s in sweep if s["precision_off"] >= 0.8]
    if hi:
        s = min(hi, key=lambda x: x["alpha"])
        reachable = True
    else:
        s = max(sweep, key=lambda x: x["precision_off"])
        reachable = False
    cost_rel = (baseline - s["dmap_off"]) / baseline if baseline > 0 else float("inf")
    ctx_best = max((x["dmap_ctx"] for x in sweep if x["alpha"] > 0), default=baseline)
    ctx_helps = ctx_best > baseline
    return {
        "sweep": sweep,
        "baseline_dmap": baseline,
        "precision_reachable": reachable,
        "off_cost_rel": float(cost_rel),
        "alpha_at_0.8": s["alpha"] if reachable else None,
        "context_best_dmap": ctx_best,
        "context_helps": bool(ctx_helps),
    }


def cell_faceted(axis: str, df: pd.DataFrame, steer, rng: np.random.Generator, power: str) -> dict:
    if axis not in BEHAVIORAL_AXES:
        # gap axes: facet-coverage FACT (capability), no steering (pilot pool too thin)
        vals = df[axis].dropna().astype(int)
        return {
            "axis": axis,
            "use": "faceted",
            "capability": "YES",
            "capability_basis": f"{vals.nunique()} queryable gap buckets ({int(vals.min())}..{int(vals.max())})",
            "lift_metric": "n/a (coverage fact)",
            "metadata_baseline": "metadata exposes no such axis",
            "lift_value": None,
            "lift_ci": None,
            "significant": False,
            "lift_verdict": "N/A-COVERAGE",
            "power": power,
            "note": f"facet exposes {vals.nunique()} buckets; pilot pool too thin for steering",
        }
    amap = dict(zip(df["article_id"].astype(str), df[axis].astype(str)))
    r = _steer_axis(amap, *steer, rng)
    reach = r["precision_reachable"] and r["off_cost_rel"] <= STEER_COST_MODEST
    sig = bool(reach and r["context_helps"])
    verdict = "PASS" if sig else ("MARGINAL" if r["precision_reachable"] else "NO")
    base = r["baseline_dmap"]
    ctx_gain = (r["context_best_dmap"] - base) / base if base > 0 else 0.0  # higher=better
    return {
        "axis": axis,
        "use": "faceted",
        "capability": "YES",
        "capability_basis": "steerable facet absent from metadata (DE1 nonredundant)",
        "lift_metric": "context_steer_discovery_map_rel_gain (steerable @ off_cost_rel)",
        "metadata_baseline": "unsteered/off-target discovery_map",
        "lift_value": round(float(ctx_gain), 4),
        "lift_ci": None,
        "alpha_at_0.8": r["alpha_at_0.8"],
        "off_cost_rel": round(r["off_cost_rel"], 4),
        "precision_reachable": r["precision_reachable"],
        "context_helps": r["context_helps"],
        "context_best_dmap": round(r["context_best_dmap"], 6),
        "baseline_dmap": round(base, 6),
        "significant": sig,
        "lift_verdict": verdict,
        "power": power,
        "note": f"steer reachable@α={r['alpha_at_0.8']} off_cost_rel={r['off_cost_rel']:.2f}; "
        f"ctx_gain={ctx_gain:+.2f} (ctx_helps={r['context_helps']})",
    }


def _na(axis: str, use: str, verdict: str, note: str = "") -> dict:
    return {
        "axis": axis,
        "use": use,
        "capability": "YES" if "COVERAGE" in verdict else "PARTIAL",
        "capability_basis": "",
        "lift_metric": "n/a",
        "metadata_baseline": "n/a",
        "lift_value": None,
        "lift_ci": None,
        "significant": False,
        "lift_verdict": verdict,
        "power": "n/a",
        "note": note,
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
    glyph = {
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
            cap = {"YES": "✓", "PARTIAL": "~", "NO": "✗"}.get(c.get("capability", ""), "")
            if isinstance(v, (int, float)) and v == v:
                lift[i, j] = v
                annot[i, j] = f"cap {cap}\n{glyph.get(lv,'')} {v:+.2f}"
            else:
                annot[i, j] = f"cap {cap}\n{glyph.get(lv,'·')}"
    fig, axfig = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        np.nan_to_num(lift),
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        center=0,
        xticklabels=[
            "① Faceted/\ncontrol",
            "② Trend\nlead-time",
            "③ Merch\naudit",
            "④ Mkt\naudience",
        ],
        yticklabels=[a.replace("e2_", "").replace("_actual", "") for a in AXES],
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "decision-lift (Δ vs metadata)"},
        ax=axfig,
        annot_kws={"fontsize": 10},
    )
    axfig.set_title(
        "Enrichment-v2 Value Matrix — capability (cap) × measured decision-lift\n"
        "✓ pass · ~ marginal · ✗ no · ≈ prelim · ·/— n/a",
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
    lags = (1, 2) if quick else (1, 2, 3, 4)
    n_boot = 200 if quick else 1000

    print("=" * 80, flush=True)
    print(f"PROBE E2 — value matrix (seed 42, CPU, no $) {'[QUICK]' if quick else ''}", flush=True)
    print("=" * 80, flush=True)

    df = pd.read_parquet(MATRIX_PATH)
    df["article_id"] = df["article_id"].astype(str)
    art = pd.read_parquet(ARTICLES_PATH, columns=["article_id", *META_COLS])
    art["article_id"] = art["article_id"].astype(str)
    # independent loyalty KPI for ④ audience (NOT sales-derived → avoids tautology)
    print("computing per-item repurchase_rate (loyalty KPI) ...", flush=True)
    df["repurchase_rate"] = _repurchase_rate(df["article_id"].to_numpy())
    print(
        f"matrix items: {len(df)} | gap-axis items: {int(df['e2_value_gap'].notna().sum())} | "
        f"repurchase_rate non-null: {int(df['repurchase_rate'].notna().sum())}",
        flush=True,
    )

    print("\n[① faceted] building steering context ...", flush=True)
    steer = _build_steer(df, np.random.default_rng(SEED), n_users)
    print(f"   eligible users: {len(steer[2])}", flush=True)

    import duckdb

    con = duckdb.connect()
    cells: list[dict] = []
    for axis in AXES:
        power = "STRONG" if axis in BEHAVIORAL_AXES else "PRELIMINARY"
        print(f"\n[axis] {axis} ({power})", flush=True)
        cells.append(cell_faceted(axis, df, steer, np.random.default_rng(SEED), power))
        cells.append(cell_leadlag(axis, df, con, lags, n_boot, power))
        cells.append(cell_merch(axis, df, art, np.random.default_rng(SEED), n_boot, power))
        cells.append(cell_audience(axis, df, art, np.random.default_rng(SEED), power))
        for c in cells[-4:]:
            print(
                f"   {c['use']:9s} cap={c['capability']:4s} lift={c.get('lift_value')} "
                f"[{c['power']}] -> {c['lift_verdict']}",
                flush=True,
            )
    con.close()

    # ---- verdict ----
    cap_yes = sum(1 for c in cells if c["capability"] == "YES")
    strong_pass = [c for c in cells if c["power"] == "STRONG" and c["lift_verdict"] == "PASS"]
    n_lift_pass = len(strong_pass)
    outfit_merch = next(
        (c for c in cells if c["axis"] == "e2_outfit_role" and c["use"] == "merch"), {}
    )
    outfit_pass = outfit_merch.get("lift_verdict") == "PASS"
    if cap_yes >= 12 and n_lift_pass >= 2 and outfit_pass:
        decision = "E2 GO — axes are decision-axes with ≥2 behavioral lift wins"
    elif cap_yes >= 8 and n_lift_pass <= 1:
        decision = (
            f"E2 PARTIAL (thesis-CONFIRMING) — capability ✓ in {cap_yes}/16 cells but behavioral "
            f"lift in only {n_lift_pass} cell(s): value is interpretive/steerable structure, not predictive lift"
        )
    elif cap_yes < 8:
        decision = f"E2 NO-GO — axes failed to expose distinct dimensions (capability {cap_yes}/16) [bug red-flag]"
    else:
        decision = f"E2 MIXED — capability {cap_yes}/16, {n_lift_pass} strong lift PASS"

    pass_str = (
        ", ".join(f"{c['axis'].replace('e2_', '')}→{c['use']}" for c in strong_pass) or "none"
    )
    print("\n" + "=" * 80, flush=True)
    print(f"  DECISION: {decision}", flush=True)
    print(
        f"  capability YES: {cap_yes}/16 | strong lift PASS: {n_lift_pass} ({pass_str})", flush=True
    )
    print("=" * 80, flush=True)

    result = {
        "probe": "E2_value_matrix",
        "seed": SEED,
        "quick": quick,
        "n_users_faceted": len(steer[2]),
        "n_matrix_items": len(df),
        "thresholds": {
            "eta_excess_min": ETA_EXCESS_MIN,
            "leadlag_delta_min": LEADLAG_DELTA_MIN,
            "steer_cost_modest": STEER_COST_MODEST,
        },
        "axes": AXES,
        "uses": USES,
        "value_matrix": cells,
        "summary": {
            "capability_yes": cap_yes,
            "n_cells": len(cells),
            "strong_lift_pass": n_lift_pass,
            "strong_pass_cells": [f"{c['axis']}→{c['use']}" for c in strong_pass],
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
        },
        "decision": decision,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    make_figure(cells)
    print(f"Confirmation: JSON written -> {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
