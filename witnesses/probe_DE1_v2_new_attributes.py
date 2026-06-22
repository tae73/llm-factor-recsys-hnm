"""PROBE DE1-v2 — RE-SCREEN of the NEW enrichment-v2 decision-axes.

WHY — DE1 screened the v1 L1/L2/L3 attributes and only 2/20 survived (0/12 for L2/L3):
the v1 prompt SHOWED the LLM the item's metadata, so it recoded it (REDUNDANT), or the
attribute collapsed onto one value (CONCENTRATED, e.g. occasion 81% "Everyday"). The v2
axes were designed to PASS where the old ones failed:
  * behavior-derived (price-tier / trend-phase / outfit-role): computed from
    transactions, discriminable by construction, no metadata to recode.
  * LLM (occasion / fit-intent / care / perceived price+trend): extracted multimodally
    WITHOUT showing metadata, orthogonal to the v1 field each could recode.
  * gap (value_gap = price_look − price_tier; trend_gap): residual of two
    metadata-orthogonal inputs → DE1-safe by construction.

This re-screen REUSES the exact DE1 engine (same three scores, same pre-registered
thresholds, same SEED=42) so verdicts are directly comparable to ``probe_DE1_result.json``.
The only change is the input attributes and a TWO-POPULATION split:

  POP_FULL  — behavior-derived axes screened on the FULL catalog (~105K) → STRONG power.
  POP_PILOT — LLM + gap axes screened on the ~500-code pilot subset → discrimination &
              non-redundancy fully measured; behavioral flagged PRELIMINARY (limited
              item diversity, though the pilot's popular items give many transactions).

If the pilot LLM parquet is absent, the behavioral axes alone are screened (an early,
spend-free de-risk read).

Usage:
    cd <repo> && PYTHONPATH=. OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python -u \
        witnesses/probe_DE1_v2_new_attributes.py
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

from witnesses._probe_common import OUT_DIR  # noqa: E402
from witnesses.probe_DE1_attribute_screen import (  # noqa: E402  (reuse the EXACT engine)
    BEHAV_MIN,
    DISC_MIN,
    META_COLS,
    REDUNDANT_L1_LIFT,
    REDUNDANT_META_LIFT,
    SEED,
    TOP1_MAX,
    BehaviorTables,
    _cramers_v,
    _cv_lift_logreg,
    _first,
    _knn_lift_l1,
    _load_aligned,
    _load_l1_bge,
    _onehot_meta,
    behavioral_signal,
    discrimination,
)

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(SEED)

BEHAVIORAL_PARQUET = ROOT / "data/knowledge/enrichment_v2/behavioral_axes.parquet"
LLM_PARQUET = ROOT / "data/knowledge/enrichment_v2/enrichment_v2_llm.parquet"
TRAIN_TXN = ROOT / "data/processed/train_transactions.parquet"
RESULT_PATH = OUT_DIR / "probe_DE1_v2_result.json"

# Attribute groups (the new e2 axes) -----------------------------------------
BEHAVIORAL_ATTRS = ["e2_price_tier_actual", "e2_trend_phase_actual", "e2_outfit_role"]
LLM_ATTRS = [
    "e2_occasion_primary",
    "e2_occasion_secondary",
    "e2_occasion_formality",
    "e2_fit_intent",
    "e2_body_ease",
    "e2_care_burden",
    "e2_care_flags",
    "e2_price_look",
    "e2_trend_look",
]
GAP_ATTRS = ["e2_value_gap", "e2_trend_gap"]
LIST_COLS = {"e2_occasion_secondary", "e2_care_flags"}

# Explicit "no-signal" sentinels excluded from an attribute's screen population.
EXCLUDE_VALUES = {"e2_trend_phase_actual": {"Insufficient"}}

POWER_MIN_ITEMS = 20_000  # >= => STRONG behavioral power, else PRELIMINARY

# Gap rank maps (project onto a shared 1..5 scale; see schema.py).
PRICE_TIER_RANK = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5}
TREND_LOOK_RANK = {"Dated": 1, "Classic": 2, "Current": 4, "Emerging": 5}
TREND_PHASE_RANK = {"Declining": 1, "Mature": 2, "Peak": 3, "Rising": 4, "Emerging": 5}


# ---------------------------------------------------------------------------
# Data assembly: merge e2 columns onto the L1-aligned catalog + compute gaps
# ---------------------------------------------------------------------------
def _assemble(canon: np.ndarray) -> tuple[pd.DataFrame, bool]:
    """Return an e2 DataFrame aligned to ``canon`` order; (df, has_llm)."""
    e2 = pd.DataFrame({"article_id": canon})

    bh = pd.read_parquet(BEHAVIORAL_PARQUET)
    bh["article_id"] = bh["article_id"].astype(str)
    e2 = e2.merge(bh[["article_id", *BEHAVIORAL_ATTRS]], on="article_id", how="left")

    has_llm = LLM_PARQUET.exists()
    if has_llm:
        llm = pd.read_parquet(LLM_PARQUET)
        llm["article_id"] = llm["article_id"].astype(str)
        llm_cols = [c for c in LLM_ATTRS if c in llm.columns]
        e2 = e2.merge(llm[["article_id", *llm_cols]], on="article_id", how="left")
        # gaps (only where both inputs present)
        price_rank = bh.set_index("article_id")["e2_price_tier_actual"].map(PRICE_TIER_RANK)
        e2 = e2.merge(price_rank.rename("_price_rank"), on="article_id", how="left")
        if "e2_price_look" in e2.columns:
            e2["e2_value_gap"] = (e2["e2_price_look"] - e2["_price_rank"]).astype("Int64")
        phase_rank = bh.set_index("article_id")["e2_trend_phase_actual"].map(TREND_PHASE_RANK)
        e2 = e2.merge(phase_rank.rename("_phase_rank"), on="article_id", how="left")
        if "e2_trend_look" in e2.columns:
            look_rank = e2["e2_trend_look"].map(TREND_LOOK_RANK)
            e2["e2_trend_gap"] = (look_rank - e2["_phase_rank"]).astype("Int64")
    # ensure canon order
    e2 = e2.set_index("article_id").reindex(canon).reset_index()
    return e2, has_llm


# ---------------------------------------------------------------------------
# Per-population behavior tables (filter the in-memory transaction frame)
# ---------------------------------------------------------------------------
def _load_tx() -> pd.DataFrame:
    tx = pd.read_parquet(TRAIN_TXN, columns=["t_dat", "article_id"])
    tx["article_id"] = tx["article_id"].astype(str)
    tx["month"] = pd.to_datetime(tx["t_dat"]).dt.month
    return tx[["article_id", "month"]]


def _bt_for(
    tx: pd.DataFrame, article_ids: np.ndarray, txn_sample: int = 3_000_000
) -> tuple[BehaviorTables, dict]:
    """Build BehaviorTables restricted to ``article_ids`` (a screen population)."""
    sub = tx[tx["article_id"].isin(set(article_ids.tolist()))]
    item_total = sub["article_id"].value_counts().to_dict()
    if len(sub) > txn_sample:
        sub = sub.sample(txn_sample, random_state=SEED)
    bt = BehaviorTables(
        month_by_item={},
        item_total=item_total,
        txn_sample_articles=sub["article_id"].to_numpy(),
        txn_sample_months=sub["month"].to_numpy(),
    )
    return bt, item_total


# ---------------------------------------------------------------------------
# Screen one attribute on its (already subset) population
# ---------------------------------------------------------------------------
def _screen_one(
    attr: str,
    prim: np.ndarray,
    canon_sub: np.ndarray,
    meta_oh: np.ndarray,
    l1_bge: np.ndarray,
    art_sub: pd.DataFrame,
    bt: BehaviorTables,
    item_total_arr: np.ndarray,
    power: str,
) -> dict:
    disc, top1, n_vals = discrimination(prim)
    meta_pred = _cv_lift_logreg(meta_oh, prim)
    l1_pred = _knn_lift_l1(l1_bge, prim)
    nonredund = 1.0 - max(meta_pred["lift"], 0.0)

    best_meta_col, best_v = None, -1.0
    for mc in META_COLS:
        if mc not in art_sub.columns:
            continue
        v = _cramers_v(prim, art_sub[mc].astype(str).to_numpy())
        if v > best_v:
            best_v, best_meta_col = v, mc
    matched_meta = art_sub[best_meta_col].astype(str).to_numpy()

    attr_primary = {a: p for a, p in zip(canon_sub, prim)}
    behav = behavioral_signal(attr_primary, bt, canon_sub, item_total_arr, matched_meta)

    concentrated = disc < DISC_MIN or top1 > TOP1_MAX
    redundant = meta_pred["lift"] >= REDUNDANT_META_LIFT or l1_pred["lift"] >= REDUNDANT_L1_LIFT
    has_behav = behav["behavioral_signal"] >= BEHAV_MIN
    # STRONG gates = discrimination + non-redundancy (always measured at full power on
    # the screen population). Behavioral is PRELIMINARY for LLM axes (low item diversity).
    passes_strong = (not concentrated) and (not redundant)
    if passes_strong and has_behav:
        verdict = "SALVAGEABLE"
    elif concentrated:
        verdict = "CONCENTRATED"
    elif redundant:
        verdict = "REDUNDANT"
    else:
        verdict = "WEAK"

    return {
        "attribute": attr,
        "n_items_screened": int(len(prim)),
        "n_values": n_vals,
        "discrimination": round(disc, 4),
        "top1_share": round(top1, 4),
        "metadata_predictability": round(meta_pred["lift"], 4),
        "l1_predictability": round(l1_pred["lift"], 4),
        "nonredundancy": round(float(nonredund), 4),
        "matched_meta_col": best_meta_col,
        **behav,
        "behavioral_power": power,
        "passes_strong_gates": bool(passes_strong),
        "verdict": verdict,
        "_flags": {
            "concentrated": bool(concentrated),
            "redundant": bool(redundant),
            "has_behavioral": bool(has_behav),
        },
    }


def _run_group(
    attrs: list[str],
    e2: pd.DataFrame,
    canon: np.ndarray,
    art: pd.DataFrame,
    tx: pd.DataFrame,
) -> list[dict]:
    out: list[dict] = []
    for attr in attrs:
        if attr not in e2.columns:
            continue
        col = e2[attr]
        prim_all = (
            col.map(_first).to_numpy() if attr in LIST_COLS else col.astype("object").to_numpy()
        )
        # population = non-null & not a no-signal sentinel
        notnull = col.notna().to_numpy()
        excl = EXCLUDE_VALUES.get(attr, set())
        keep = notnull & np.array([str(v) not in excl for v in prim_all])
        if keep.sum() < 50:
            print(f"  [skip] {attr}: only {keep.sum()} usable rows", flush=True)
            continue
        canon_sub = canon[keep]
        art_sub = art.loc[keep].reset_index(drop=True)
        meta_oh = _onehot_meta(art_sub)
        l1_bge = _load_l1_bge(canon_sub)
        prim = np.array([str(v) for v in prim_all[keep]])
        bt, item_total = _bt_for(tx, canon_sub)
        item_total_arr = np.array([item_total.get(a, 0) for a in canon_sub], dtype=np.float64)
        power = "STRONG" if len(canon_sub) >= POWER_MIN_ITEMS else "PRELIMINARY"
        r = _screen_one(attr, prim, canon_sub, meta_oh, l1_bge, art_sub, bt, item_total_arr, power)
        out.append(r)
        print(
            f"  {attr:26s} n={r['n_items_screened']:>6d} disc={r['discrimination']:.2f} "
            f"top1={r['top1_share']:.2f} meta_p={r['metadata_predictability']:.2f} "
            f"l1_p={r['l1_predictability']:.2f} behav={r['behavioral_signal']:.3f} "
            f"[{power[:4]}] -> {r['verdict']}",
            flush=True,
        )
    return out


def _go_no_go(screen: list[dict], has_llm: bool) -> dict:
    """Pre-registered GO/NO-GO: >=3 new axes pass STRONG gates AND >=1 behavioral axis
    is fully SALVAGEABLE (all 3 tests, STRONG power). Beats old baseline (2/20; 0/12)."""
    strong_pass = [r["attribute"] for r in screen if r["passes_strong_gates"]]
    behav_salv = [
        r["attribute"]
        for r in screen
        if r["attribute"] in BEHAVIORAL_ATTRS
        and r["verdict"] == "SALVAGEABLE"
        and r["behavioral_power"] == "STRONG"
    ]
    n_strong = len(strong_pass)
    if not has_llm:
        decision = "PARTIAL (behavioral-only; LLM axes pending extraction)"
    elif n_strong >= 3 and len(behav_salv) >= 1:
        decision = "GO"
    elif n_strong == 2:
        decision = "CONDITIONAL-GO (scale extraction to resolve behavioral)"
    else:
        decision = "NO-GO (new axes recoded metadata like the old ones)"
    return {
        "decision": decision,
        "n_pass_strong_gates": n_strong,
        "pass_strong_gates": strong_pass,
        "behavioral_salvageable": behav_salv,
        "old_baseline": "2/20 salvageable (0/12 for L2/L3)",
    }


def main() -> None:
    print("=" * 80, flush=True)
    print("PROBE DE1-v2 — re-screen of NEW enrichment-v2 axes (seed=42, CPU, no $)", flush=True)
    print("  reuses the DE1 engine + thresholds; two-population split (full vs pilot)", flush=True)
    print("=" * 80, flush=True)

    canon, _fk, art = _load_aligned()
    print(f"L1-aligned catalog: {len(canon)} items", flush=True)
    e2, has_llm = _assemble(canon)
    print(f"LLM pilot parquet present: {has_llm}", flush=True)

    print("Loading train transactions (once) ...", flush=True)
    tx = _load_tx()

    print("\n[POP_FULL] behavior-derived axes (full catalog → STRONG):", flush=True)
    screen = _run_group(BEHAVIORAL_ATTRS, e2, canon, art, tx)

    if has_llm:
        print(
            "\n[POP_PILOT] LLM + gap axes (pilot subset → discrimination/non-redund fully powered):",
            flush=True,
        )
        screen += _run_group(LLM_ATTRS + GAP_ATTRS, e2, canon, art, tx)
    else:
        print("\n[POP_PILOT] skipped — LLM extraction not yet run.", flush=True)

    gng = _go_no_go(screen, has_llm)

    print("\n" + "=" * 80, flush=True)
    print(f"  DECISION: {gng['decision']}", flush=True)
    print(
        f"  pass STRONG gates ({gng['n_pass_strong_gates']}): "
        f"{', '.join(gng['pass_strong_gates']) or '(none)'}",
        flush=True,
    )
    print(
        f"  behavioral SALVAGEABLE: {', '.join(gng['behavioral_salvageable']) or '(none)'}",
        flush=True,
    )
    print("=" * 80, flush=True)

    result = {
        "probe": "DE1_v2_new_attributes",
        "seed": SEED,
        "has_llm_pilot": has_llm,
        "thresholds": {
            "discrimination_min": DISC_MIN,
            "top1_share_max": TOP1_MAX,
            "redundant_meta_lift": REDUNDANT_META_LIFT,
            "redundant_l1_lift": REDUNDANT_L1_LIFT,
            "behavioral_min": BEHAV_MIN,
            "power_min_items": POWER_MIN_ITEMS,
        },
        "per_attribute": {r["attribute"]: r for r in screen},
        "summary": {
            "SALVAGEABLE": [r["attribute"] for r in screen if r["verdict"] == "SALVAGEABLE"],
            "CONCENTRATED": [r["attribute"] for r in screen if r["verdict"] == "CONCENTRATED"],
            "REDUNDANT": [r["attribute"] for r in screen if r["verdict"] == "REDUNDANT"],
            "WEAK": [r["attribute"] for r in screen if r["verdict"] == "WEAK"],
        },
        "go_no_go": gng,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Confirmation: JSON written -> {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
