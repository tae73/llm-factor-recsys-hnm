"""PROBE E2d — GAP-AXIS *FUTURE* DECISION-LIFT sweep (C-backlog b / Contribution E2-5).

WHY — the two perception×behavior gap axes (`e2_value_gap` = price_look − rank(price_tier);
`e2_trend_gap` = rank(trend_look) − rank(trend_phase)) PASSED capability/non-redundancy in
E2-1 (meta_p 0.35/0.13, l1_p 0.55/0.24 — orthogonal to metadata AND L1) but were
behaviorally INERT as *prediction* axes across all four uses in E2/E2b (faceted N/A,
leadlag N/A/≈0, merch NO, audience NO). Those probes measured the gap as a grouping var for
a TRAIN-window outcome. The gap's actual interpretive claim — "hidden-value / trend-risk" —
is that the *mismatch itself* is a forward-looking DECISION signal. This probe gives that
claim its single most-favorable, genuinely-untested test, on two pillars no prior probe did:

  (1) FUTURE-held-out outcomes (val 2020-07+), via build_article_future_outcomes (train-frozen).
  (2) INCREMENTAL OVER OWN CONSTITUENTS — the baseline is the gap's two component axes entered
      as ONE-HOT factors (+ product_group), NOT the η-vs-metadata template (which would silently
      re-measure the known NO). Because gap = c1 − c2 is *deterministically collinear* with the
      ordinals, the constituents are one-hot so the directional mismatch (sign + magnitude) is a
      genuine interaction contrast the additive levels cannot span.

Two readouts per (axis, decision): (i) incremental paired-fold macro-F1 of [base + gap] over
[base], + accuracy-increment bootstrap CI; (ii) deployable decision-rule precision@flag vs a
matched-N constituents-model flag. FIVE pre-registered gates (margin · paired-sig · CI>0 ·
within-group placebo ≈0 · sign-randomization KILLS it) before any positive; replication in
both cohorts (n_val≥5 and ≥20). Power is PRELIMINARY by construction (gap axes are the ~5.3K
LLM pilot) → verdicts emit PRELIM, never PASS, and always report effect-size + CI.

HONEST PRIOR: gap axes failed every prior behavioral screen → a clean negative is most-likely
and SHARPENS the thesis ("non-redundant interpretive coordinates, provably not forecasters").
Never touches probe_E2/E2b JSON (asserts mtime unchanged). CPU, seed 42, no API/$.

Usage:  PYTHONPATH=. python3 -u witnesses/probe_E2d_gap_decision.py [--quick] [--repro]
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

from src.features.enrichment_matrix import build_article_future_outcomes  # noqa: E402
from src.knowledge.enrichment_v2.schema import PRICE_TIER_RANK  # noqa: E402
from witnesses._probe_common import OUT_DIR, bootstrap_delta  # noqa: E402
from witnesses.probe_D5_value_matrix import _paired_sig, _quantile_bucket  # noqa: E402
from witnesses.probe_E2_value_matrix import ARTICLES_PATH, MATRIX_PATH, SEED  # noqa: E402

warnings.filterwarnings("ignore")

RESULT_PATH = OUT_DIR / "probe_E2d_gap_decision.json"
FIG_PATH = ROOT / "results/figures/E2d_gap_decision.png"
E2_JSON = ROOT / "witnesses/probe_E2_result.json"
E2B_JSON = ROOT / "witnesses/probe_E2b_result.json"
LLM_PATH = ROOT / "data/knowledge/enrichment_v2/enrichment_v2_llm.parquet"
DATA_DIR = ROOT / "data/processed"

# gap col -> (LLM constituent, behavioral constituent)
GAP_COMPONENTS = {
    "e2_value_gap": ("e2_price_look", "e2_price_tier_actual"),
    "e2_trend_gap": ("e2_trend_look", "e2_trend_phase_actual"),
}
# extra continuous baseline feature (absorbs mean-reversion for trend)
GAP_EXTRA_CONT = {"e2_trend_gap": "e2_trend_momentum"}

# Decisions per axis: a continuous FUTURE outcome + the "realized decision" tertile + the flag
# direction (gap·flag_sign ≥ MAG_THRESH). directional ⇒ the sign-randomization gate applies.
DECISIONS = {
    "e2_value_gap": [
        {
            "name": "markdown_risk",
            "outcome": "fut_price_drop",
            "realized_tertile": 0,  # deepest future price erosion
            "flag_sign": -1,  # gap ≤ −2 (looks cheaper than priced → overpriced)
            "hyp": "gap<0 (overpriced-look) → deeper future price drop",
        },
        {
            "name": "hidden_gem",
            "outcome": "fut_velocity",
            "realized_tertile": 2,  # fastest future sell-through
            "flag_sign": 1,  # gap ≥ +2 (looks pricey, priced low → deal)
            "hyp": "gap>0 (hidden deal) → faster future sell-through",
        },
    ],
    "e2_trend_gap": [
        {
            "name": "overhype_sleeper",
            "outcome": "fut_momentum_change",
            "realized_tertile": 0,  # future momentum decay (overhype)
            "flag_sign": 1,  # gap ≥ +2 (looks trendier than behavior → overhype)
            "hyp": "gap>0 (overhype) → future momentum decay; gap<0 (sleeper) → accelerate",
        },
    ],
}

# Pre-registered margins (parity with project conventions).
PRED_MARGIN = 0.01  # incremental macro-F1 (D5 PRACTICAL_MARGIN / E2c PRED_MARGIN)
PREC_MARGIN = 0.05  # decision-rule precision-points over matched-model flag (E2 ETA_EXCESS_MIN)
LIFT_MIN = 1.10  # decision rule must beat base rate by ≥10% (E2b DEPLOY_LIFT_MIN)
MAG_THRESH = 2  # |gap| ≥ 2 = the deployable extreme-mismatch flag
MIN_VAL_PRIMARY = 5  # primary cohort: items with ≥5 val purchases
MIN_VAL_ROBUST = 20  # robustness cohort (must replicate here for PRELIM)
N_SPLITS = 5


# ===========================================================================
# Feature builders
# ===========================================================================
def _oh(series: pd.Series) -> np.ndarray:
    """One-hot a categorical series (deterministic column order via sorted categories)."""
    return pd.get_dummies(series.astype(str).astype("category"), dummy_na=False).to_numpy(
        dtype=np.float32
    )


def _build_matrices(sub: pd.DataFrame, axis: str) -> dict[str, np.ndarray]:
    """base / full / sat design matrices for one (masked) axis sub-frame.

    base = onehot(c1) ⊕ onehot(c2) ⊕ onehot(product_group) [⊕ std(momentum) for trend].
    full = base ⊕ [sign(gap), |gap|]  (the directional mismatch the additive levels can't span).
    sat  = onehot(c1×c2 interaction) ⊕ onehot(product_group) [⊕ momentum] — upper bound; the gap
           is a coarsening of c1×c2 so f1_full ≤ f1_sat by construction (sanity check).
    """
    c1, c2 = GAP_COMPONENTS[axis]
    pg = _oh(sub["product_group_name"])
    base_parts = [_oh(sub[c1]), _oh(sub[c2]), pg]
    sat_parts = [_oh(sub[c1].astype(str) + "|" + sub[c2].astype(str)), pg]
    extra = GAP_EXTRA_CONT.get(axis)
    if extra is not None:
        cont = sub[extra].astype(float).to_numpy().reshape(-1, 1)
        cont = np.nan_to_num(cont, nan=float(np.nanmean(cont)))
        base_parts.append(cont)
        sat_parts.append(cont)
    base = np.concatenate(base_parts, axis=1).astype(np.float32)
    gap = sub[axis].astype(float).to_numpy()
    gap_feat = np.column_stack([np.sign(gap), np.abs(gap)]).astype(np.float32)
    full = np.concatenate([base, gap_feat], axis=1)
    sat = np.concatenate(sat_parts, axis=1).astype(np.float32)
    return {"base": base, "full": full, "sat": sat, "gap": gap}


def _logreg():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(solver="lbfgs", C=1.0, max_iter=200, tol=5e-3, random_state=SEED)


def _nested_cv(X_base: np.ndarray, X_full: np.ndarray, y: np.ndarray) -> dict:
    """Paired CV on IDENTICAL folds: per-fold macro-F1 (base, full) + per-item OOF correctness."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    f1_base, f1_full = [], []
    oof_base = np.zeros(len(y))
    oof_full = np.zeros(len(y))
    for tr, te in skf.split(X_base, y):
        for X, folds, oof in ((X_base, f1_base, oof_base), (X_full, f1_full, oof_full)):
            sc = StandardScaler(with_mean=True, with_std=True)
            Xtr = sc.fit_transform(X[tr].astype(np.float64))
            Xte = sc.transform(X[te].astype(np.float64))
            clf = _logreg().fit(Xtr, y[tr])
            pred = clf.predict(Xte)
            folds.append(round(f1_score(y[te], pred, average="macro"), 4))
            oof[te] = (pred == y[te]).astype(float)
    return {
        "f1_base": f1_base,
        "f1_full": f1_full,
        "oof_base": oof_base,
        "oof_full": oof_full,
    }


def _f1_only(X: np.ndarray, y: np.ndarray) -> float:
    """Mean macro-F1 of one design matrix (for the saturated upper-bound sanity check)."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    out = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler(with_mean=True, with_std=True)
        Xtr = sc.fit_transform(X[tr].astype(np.float64))
        Xte = sc.transform(X[te].astype(np.float64))
        clf = _logreg().fit(Xtr, y[tr])
        out.append(f1_score(y[te], clf.predict(Xte), average="macro"))
    return float(np.mean(out))


def _oof_proba(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OOF P(y=1) on the same fold scheme — the matched-baseline decision-model score."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler(with_mean=True, with_std=True)
        clf = _logreg().fit(sc.fit_transform(X[tr].astype(np.float64)), y[tr])
        p[te] = clf.predict_proba(sc.transform(X[te].astype(np.float64)))[:, 1]
    return p


# ===========================================================================
# Readouts
# ===========================================================================
def _incremental(mats: dict, y: np.ndarray, *, with_signrand: bool, c2_labels, n_boot, rng) -> dict:
    """Readout (i): paired incremental macro-F1 of full over base + placebos + CI.

    within-group placebo: shuffle the gap within product_group (here within the c2 constituent
    bucket — preserves the gap marginal, destroys the item-level link) → expect Δ≈0.
    sign-randomization: keep |gap|, randomize sign within c2 bucket → a DIRECTIONAL effect must
    vanish; if it persists, the increment was magnitude/constituent-driven, not the mismatch.
    """
    cv = _nested_cv(mats["base"], mats["full"], y)
    sig = _paired_sig(cv["f1_full"], cv["f1_base"], min_delta=PRED_MARGIN)
    ci = bootstrap_delta(cv["oof_base"], cv["oof_full"], n_boot=n_boot, seed=SEED)
    f1_sat = _f1_only(mats["sat"], y)

    def _placebo(gap_perm: np.ndarray) -> float:
        full_p = np.concatenate(
            [
                mats["base"],
                np.column_stack([np.sign(gap_perm), np.abs(gap_perm)]).astype(np.float32),
            ],
            axis=1,
        )
        cvp = _nested_cv(mats["base"], full_p, y)
        return round(float(np.mean(cvp["f1_full"]) - np.mean(cvp["f1_base"])), 4)

    gap = mats["gap"]
    shuffle_delta = _placebo(_perm_within(gap, c2_labels, rng, mode="shuffle"))
    signrand_delta = (
        _placebo(_perm_within(gap, c2_labels, rng, mode="signrand")) if with_signrand else None
    )
    return {
        "incr_delta_f1": sig["mean_delta"],
        "incr_p": sig["p_value"],
        "incr_significant": sig["significant"],
        "acc_incr": round(ci["delta"], 4),
        "acc_ci": [round(ci["ci_lo"], 4), round(ci["ci_hi"], 4)],
        "f1_base": round(float(np.mean(cv["f1_base"])), 4),
        "f1_full": round(float(np.mean(cv["f1_full"])), 4),
        "f1_sat": round(f1_sat, 4),
        "sat_dominates": bool(f1_sat + 1e-9 >= np.mean(cv["f1_full"])),
        "placebo_shuffle_delta": shuffle_delta,
        "placebo_signrand_delta": signrand_delta,
    }


def _perm_within(
    gap: np.ndarray, buckets: np.ndarray, rng: np.random.Generator, *, mode: str
) -> np.ndarray:
    """Permute the gap within each constituent bucket. mode='shuffle' permutes values;
    mode='signrand' keeps |gap| and randomizes sign (Rademacher)."""
    out = gap.copy()
    for b in np.unique(buckets):
        idx = np.where(buckets == b)[0]
        if mode == "shuffle":
            out[idx] = gap[rng.permutation(idx)]
        else:  # signrand
            signs = rng.choice([-1.0, 1.0], size=len(idx))
            out[idx] = np.abs(gap[idx]) * signs
    return out


def _decision_rule(
    gap: np.ndarray, realized: np.ndarray, X_base: np.ndarray, *, flag_sign, n_boot, rng
) -> dict:
    """Readout (ii): precision@flag of an extreme-gap rule vs a matched-N constituents-model flag."""
    flag = (gap * flag_sign) >= MAG_THRESH
    n_flag = int(flag.sum())
    base_rate = float(realized.mean())
    if n_flag < 30 or base_rate == 0:
        return {"n_flag": n_flag, "rule_verdict": "INSUFFICIENT", "base_rate": round(base_rate, 4)}
    prec_gap = float(realized[flag].mean())
    model_p = _oof_proba(X_base, realized.astype(int))
    model_flag = np.zeros(len(realized), dtype=bool)
    model_flag[np.argsort(-model_p)[:n_flag]] = True
    prec_model = float(realized[model_flag].mean())
    dprec = prec_gap - prec_model
    lift = prec_gap / base_rate
    boots = []
    n = len(realized)
    for _ in range(n_boot):
        bi = rng.integers(0, n, n)
        fb, mb = flag[bi], model_flag[bi]
        if fb.sum() and mb.sum():
            boots.append(float(realized[bi][fb].mean() - realized[bi][mb].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return {
        "n_flag": n_flag,
        "base_rate": round(base_rate, 4),
        "prec_gap": round(prec_gap, 4),
        "prec_model": round(prec_model, 4),
        "dprec": round(dprec, 4),
        "dprec_ci": [round(float(lo), 4), round(float(hi), 4)],
        "lift": round(lift, 3),
        "rule_verdict": (
            "PASS"
            if (lift >= LIFT_MIN and dprec >= PREC_MARGIN and lo > 0)
            else ("MARGINAL" if lift >= LIFT_MIN else "NO")
        ),
    }


def _robustness_continuous(sub: pd.DataFrame, axis: str, outcome: str) -> dict:
    """Confirmatory robustness for a directional cell, answering the two strongest counter-claims:

    (a) "tertile-binning hid the effect" → continuous Ridge ΔR² of [base+gap] over [base] on the
        RAW outcome (KFold, seed42). If even continuous regression shows no/negative ΔR², binning
        is not the culprit.
    (b) "the raw gap↔outcome correlation is real" → partial Pearson r of the SIGNED gap with the
        outcome BEFORE vs AFTER partialling out the continuous constituent (momentum for trend,
        tier-rank for value). A correlation that collapses to ~0 was mean-reversion of the
        constituent leaking through the gap, not the directional mismatch.
    """
    from scipy import stats
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    y = sub[outcome].to_numpy(dtype=float)
    mats = _build_matrices(sub, axis)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    def _oof_r2(x: np.ndarray) -> float:
        oof = np.zeros(len(y))
        for tr, te in kf.split(x):
            sc = StandardScaler().fit(x[tr].astype(np.float64))
            m = Ridge(alpha=1.0, random_state=SEED).fit(
                sc.transform(x[tr].astype(np.float64)), y[tr]
            )
            oof[te] = m.predict(sc.transform(x[te].astype(np.float64)))
        return float(r2_score(y, oof))

    r2_base, r2_full = _oof_r2(mats["base"]), _oof_r2(mats["full"])
    # partial correlation: residualize signed gap and outcome on the continuous constituent z
    cont = GAP_EXTRA_CONT.get(axis)
    if cont is not None and cont in sub.columns:
        z = sub[cont].astype(float).to_numpy()
    else:
        z = sub[GAP_COMPONENTS[axis][1]].map(PRICE_TIER_RANK).astype(float).to_numpy()
    z = np.nan_to_num(z, nan=float(np.nanmean(z)))
    g = mats["gap"]
    zc = np.column_stack([np.ones_like(z), z])

    def _resid(v: np.ndarray) -> np.ndarray:
        beta = np.linalg.lstsq(zc, v, rcond=None)[0]
        return v - zc @ beta

    raw = float(stats.pearsonr(g, y)[0])
    partial = float(stats.pearsonr(_resid(g), _resid(y))[0])
    return {
        "ridge_r2_base": round(r2_base, 4),
        "ridge_r2_full": round(r2_full, 4),
        "ridge_dR2": round(r2_full - r2_base, 4),
        "corr_gap_raw": round(raw, 4),
        "corr_gap_partial_on_constituent": round(partial, 4),
    }


# ===========================================================================
# Cells
# ===========================================================================
def _verdict(incr: dict, robust_sig: bool, directional: bool) -> str:
    """Five-gate pre-registered rule. PRELIM cap (gap power); PASS never emitted."""
    sig = incr["incr_significant"]
    ci_pos = incr["acc_ci"][0] > 0
    shuffle_ok = incr["placebo_shuffle_delta"] < PRED_MARGIN
    sr = incr["placebo_signrand_delta"]
    # directional effect must VANISH under sign-randomization; if it persists it is
    # magnitude/constituent-driven (a confound), not the directional mismatch → NO.
    signrand_ok = (sr is None) or (sr < PRED_MARGIN)
    all_gates = sig and ci_pos and shuffle_ok and signrand_ok and robust_sig
    if all_gates:
        return "PRELIM"  # PRELIMINARY power ⇒ candidate, never PASS
    if directional and sig and not signrand_ok:
        return "NO"  # survived sign-randomization = confound-driven false positive
    if incr["incr_delta_f1"] > 0:
        return "MARGINAL"
    return "NO"


def gap_decision_cell(axis, dec, merged, fut, n_boot, rng) -> dict:
    """One (axis, decision) cell: future-held-out incremental + decision-rule, 5 gates, 2 cohorts."""
    c1, c2 = GAP_COMPONENTS[axis]
    name, outcome = dec["name"], dec["outcome"]
    df = merged.merge(fut[["article_id", outcome, "fut_val_n"]], on="article_id", how="left")
    base_mask = df[axis].notna() & df[c1].notna() & df["product_group_name"].notna()

    def _cohort(min_val):
        m = base_mask & (df["fut_val_n"] >= min_val) & df[outcome].notna()
        sub = df[m].reset_index(drop=True)
        if len(sub) < 500:
            return None, None, None
        y = _quantile_bucket(sub[outcome].to_numpy(dtype=float), 3, np.ones(len(sub), bool))
        return sub, y, len(sub)

    sub, y, n_prim = _cohort(MIN_VAL_PRIMARY)
    if sub is None:
        return _cell_dict(axis, name, dec, "PRELIMINARY-INSUFFICIENT", None, None, n_prim, 0)

    mats = _build_matrices(sub, axis)
    c2_labels = sub[c2].astype(str).to_numpy()
    incr = _incremental(mats, y, with_signrand=True, c2_labels=c2_labels, n_boot=n_boot, rng=rng)
    realized = (y == dec["realized_tertile"]).astype(float)
    rule = _decision_rule(
        mats["gap"], realized, mats["base"], flag_sign=dec["flag_sign"], n_boot=n_boot, rng=rng
    )

    # robustness cohort (≥20): incremental significance only (replication gate)
    sub_r, y_r, n_rob = _cohort(MIN_VAL_ROBUST)
    robust_sig = False
    if sub_r is not None:
        mats_r = _build_matrices(sub_r, axis)
        cv_r = _nested_cv(mats_r["base"], mats_r["full"], y_r)
        sig_r = _paired_sig(cv_r["f1_full"], cv_r["f1_base"], min_delta=PRED_MARGIN)
        robust_sig = sig_r["significant"]
        incr["robust_delta_f1"] = sig_r["mean_delta"]
        incr["robust_n"] = int(n_rob)

    verdict = _verdict(incr, robust_sig, directional=True)
    cell = _cell_dict(axis, name, dec, verdict, incr, rule, n_prim, n_rob)
    cell["robustness"] = _robustness_continuous(sub, axis, outcome)
    return cell


def survival_cell(axis, merged, fut, n_boot, rng) -> dict:
    """Does an extreme gap predict SURVIVAL into the val window at all (non-directional)?

    Catches & correctly attributes selection effects (so a survival signal is never mis-read
    as markdown/momentum). Binary outcome over ALL gap-axis items; no sign-randomization gate.
    """
    c1, c2 = GAP_COMPONENTS[axis]
    df = merged.merge(fut[["article_id", "has_val_sale"]], on="article_id", how="left")
    m = (
        df[axis].notna()
        & df[c1].notna()
        & df["product_group_name"].notna()
        & df["has_val_sale"].notna()
    )
    sub = df[m].reset_index(drop=True)
    if len(sub) < 500 or sub["has_val_sale"].nunique() < 2:
        return _cell_dict(
            axis,
            "survival",
            {"hyp": "extreme gap → survives into val"},
            "PRELIMINARY-INSUFFICIENT",
            None,
            None,
            len(sub),
            0,
        )
    y = sub["has_val_sale"].astype(int).to_numpy()
    mats = _build_matrices(sub, axis)
    c2_labels = sub[c2].astype(str).to_numpy()
    incr = _incremental(mats, y, with_signrand=False, c2_labels=c2_labels, n_boot=n_boot, rng=rng)
    flag = np.abs(mats["gap"]) >= MAG_THRESH
    rule = _decision_rule(
        np.abs(mats["gap"]), y.astype(float), mats["base"], flag_sign=1, n_boot=n_boot, rng=rng
    )
    rule["flag_basis"] = "|gap|>=2 (extreme mismatch, either sign)"
    verdict = _verdict(incr, robust_sig=incr["incr_significant"], directional=False)
    return _cell_dict(
        axis,
        "survival",
        {"hyp": "extreme |gap| → survives into val", "outcome": "has_val_sale"},
        verdict,
        incr,
        rule,
        len(sub),
        int(flag.sum()),
    )


def _cell_dict(axis, decision, dec, verdict, incr, rule, n_prim, n_rob) -> dict:
    """E2-schema cell dict (capability/lift_value/lift_ci/significant/lift_verdict/power/note)."""
    c1, c2 = GAP_COMPONENTS[axis]
    lift_val = incr["incr_delta_f1"] if incr else None
    lift_ci = incr["acc_ci"] if incr else None
    note = f"{dec.get('hyp', '')}"
    if incr:
        note = (
            f"{dec.get('hyp','')} | incrΔF1={incr['incr_delta_f1']:+.4f} p={incr['incr_p']} "
            f"acc_incr={incr['acc_incr']:+.4f}{incr['acc_ci']} | "
            f"placebo shuffle={incr['placebo_shuffle_delta']:+.4f} "
            f"signrand={incr['placebo_signrand_delta']} | sat={incr['f1_sat']} | "
            f"rule lift={rule.get('lift')} dprec={rule.get('dprec')}"
        )
    return {
        "axis": axis,
        "use": f"decision:{decision}",
        "capability": "YES",
        "capability_basis": f"perception×behavior mismatch ({c1} vs {c2}) — non-redundant (E2-1)",
        "lift_metric": "incremental macro-F1 ([constituents-onehot + gap] vs [constituents-onehot]) on FUTURE outcome",
        "metadata_baseline": f"one-hot({c1}) + one-hot({c2}) + one-hot(product_group)"
        + (" + momentum" if axis in GAP_EXTRA_CONT else ""),
        "lift_value": lift_val,
        "lift_ci": lift_ci,
        "significant": bool(incr["incr_significant"]) if incr else False,
        "lift_verdict": verdict,
        "power": "PRELIMINARY",
        "n_primary": int(n_prim) if n_prim else 0,
        "n_robust": int(n_rob) if n_rob else 0,
        "incremental": incr,
        "decision_rule": rule,
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
    glyph = {"PRELIM": "≈", "MARGINAL": "~", "NO": "✗", "PRELIMINARY-INSUFFICIENT": "·"}
    labels = [
        c["axis"].replace("e2_", "") + "\n→" + c["use"].replace("decision:", "") for c in cells
    ]
    vals = [c["lift_value"] if isinstance(c["lift_value"], (int, float)) else 0.0 for c in cells]
    annot = []
    for c in cells:
        v = c["lift_value"]
        vt = f"{v:+.3f}" if isinstance(v, (int, float)) else "n/a"
        sr = (c.get("incremental") or {}).get("placebo_signrand_delta")
        annot.append(f"{glyph.get(c['lift_verdict'], '?')} {vt}\nn={c['n_primary']} sr={sr}")
    fig, ax = plt.subplots(figsize=(8.5, 1.0 + 0.9 * len(cells)))
    arr = np.array(vals).reshape(-1, 1)
    sns.heatmap(
        arr,
        annot=np.array(annot).reshape(-1, 1),
        fmt="",
        cmap="RdYlGn",
        center=0,
        vmin=-0.05,
        vmax=0.05,
        xticklabels=["incr. ΔmacroF1\n(gap over constituents, FUTURE)"],
        yticklabels=labels,
        linewidths=1.4,
        linecolor="white",
        cbar_kws={"label": "incremental decision-lift"},
        ax=ax,
        annot_kws={"fontsize": 9},
    )
    ax.set_title(
        "E2d — gap-axis FUTURE decision-lift over OWN constituents (PRELIMINARY power)\n"
        "≈ prelim · ~ marginal · ✗ no · · insufficient  (sr = sign-randomization placebo Δ)",
        fontsize=10,
    )
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_PATH}", flush=True)


# ===========================================================================
# Main
# ===========================================================================
def _run_cells(merged, fut, n_boot) -> list[dict]:
    cells: list[dict] = []
    for axis, decs in DECISIONS.items():
        for dec in decs:
            cells.append(
                gap_decision_cell(axis, dec, merged, fut, n_boot, np.random.default_rng(SEED))
            )
    for axis in GAP_COMPONENTS:
        cells.append(survival_cell(axis, merged, fut, n_boot, np.random.default_rng(SEED)))
    return cells


def main() -> None:
    quick = "--quick" in sys.argv
    repro = "--repro" in sys.argv
    n_boot = 200 if quick else 1000
    e2_mtime, e2b_mtime = E2_JSON.stat().st_mtime, E2B_JSON.stat().st_mtime

    print("=" * 80, flush=True)
    print(
        f"PROBE E2d — gap-axis FUTURE decision-lift (seed 42) {'[QUICK]' if quick else ''}",
        flush=True,
    )
    print("=" * 80, flush=True)

    df = pd.read_parquet(MATRIX_PATH)
    df["article_id"] = df["article_id"].astype(str)
    llm = pd.read_parquet(LLM_PATH, columns=["article_id", "e2_price_look", "e2_trend_look"])
    llm["article_id"] = llm["article_id"].astype(str)
    art = pd.read_parquet(ARTICLES_PATH, columns=["article_id", "product_group_name"])
    art["article_id"] = art["article_id"].astype(str)
    merged = df.merge(llm, on="article_id", how="left").merge(art, on="article_id", how="left")

    gap_ids = merged.loc[
        merged["e2_value_gap"].notna() | merged["e2_trend_gap"].notna(), "article_id"
    ].tolist()
    fut = build_article_future_outcomes(gap_ids, DATA_DIR)
    print(
        f"matrix items: {len(merged)} | value_gap: {int(merged['e2_value_gap'].notna().sum())} | "
        f"trend_gap: {int(merged['e2_trend_gap'].notna().sum())} | val-survivors: {int(fut['has_val_sale'].sum())}",
        flush=True,
    )

    cells = _run_cells(merged, fut, n_boot)
    if repro:
        cells2 = _run_cells(merged, fut, n_boot)
        assert json.dumps(cells, default=str) == json.dumps(cells2, default=str), "REPRO mismatch!"
        print("[repro] byte-identical double run ✓", flush=True)

    for c in cells:
        print(
            f"   {c['axis'].replace('e2_',''):14s} {c['use']:24s} "
            f"lift={c['lift_value']} -> {c['lift_verdict']} (n={c['n_primary']})",
            flush=True,
        )

    prelim = [f"{c['axis']}→{c['use']}" for c in cells if c["lift_verdict"] == "PRELIM"]
    if prelim:
        decision = (
            f"E2d PRELIM candidate(s) at {prelim} — gap mismatch shows FUTURE decision-lift over its "
            f"own constituents surviving all 5 gates + both cohorts, BUT pilot-power PRELIMINARY → "
            f"needs catalog-scale confirmation before any claim."
        )
    else:
        decision = (
            "E2d CLEAN NEGATIVE — neither gap axis carries incremental FUTURE decision-lift over its "
            "own constituents (one-hot levels) on markdown/sell-through/momentum/survival, under 5 "
            "pre-registered gates. The mismatch is a non-redundant INTERPRETIVE coordinate, not a "
            "forecasting one — sharpens the E2 thesis (value = decision-axis, not prediction)."
        )
    print("\n" + "=" * 80 + f"\n  {decision}\n" + "=" * 80, flush=True)

    result = {
        "probe": "E2d_gap_decision",
        "seed": SEED,
        "quick": quick,
        "thresholds": {
            "pred_margin": PRED_MARGIN,
            "prec_margin": PREC_MARGIN,
            "lift_min": LIFT_MIN,
            "mag_thresh": MAG_THRESH,
            "min_val_primary": MIN_VAL_PRIMARY,
            "min_val_robust": MIN_VAL_ROBUST,
        },
        "n_val_survivors": int(fut["has_val_sale"].sum()),
        "axes": list(GAP_COMPONENTS),
        "value_matrix": cells,
        "summary": {
            "n_cells": len(cells),
            "prelim_candidates": prelim,
            "by_verdict": {
                v: [f"{c['axis']}→{c['use']}" for c in cells if c["lift_verdict"] == v]
                for v in ["PRELIM", "MARGINAL", "NO", "PRELIMINARY-INSUFFICIENT"]
            },
        },
        "decision": decision,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    make_figure(cells)
    assert (
        E2_JSON.stat().st_mtime == e2_mtime and E2B_JSON.stat().st_mtime == e2b_mtime
    ), "E2/E2b canonical JSON modified!"
    print(f"Confirmation: JSON -> {RESULT_PATH}; E2/E2b canonical untouched", flush=True)


if __name__ == "__main__":
    main()
