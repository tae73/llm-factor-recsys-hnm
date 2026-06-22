"""PROBE 01 — Incremental value of LLM layers (MAKE-OR-BREAK, Gate 0).

WHAT GENERALIZES
    The central claim of the project is that LLM-extracted L2 (perceptual) and L3
    (theory) attribute layers add recommendation value BEYOND H&M metadata and
    beyond L1 (product attributes). This probe tests that with content-based
    centroid-kNN retrieval (train-history centroid -> cosine over the full catalog
    -> top-12) on the held-out validation window, walking the confound-free ladder:
        META  ->  META+L1 (l1)  ->  META+L1+L2 (l1_l2)  ->  +L3 (l1_l2_l3)
    All variants share one item_id order and one fixed user sample, so deltas are
    per-user paired and get bootstrap 95% CIs. No model training is involved.

THE RESULT (boxed; filled at runtime, persisted to probe_01_result.json)
    +-----------------------------------------------------------------------+
    |  C1  META       -> META+L1+L2     (does the stack beat raw metadata?)  |
    |  C2  META+L1    -> META+L1+L2     (incremental L2 over L1)             |
    |  C3  META+L1+L2 -> META+L1+L2+L3  (incremental L3 over L1+L2, pooled)  |
    |  Reported pooled AND on the 2-4 (sparse cold-start) bracket.           |
    +-----------------------------------------------------------------------+

HONEST reduces_check
    Pre-registered, KILL-capable thresholds (fixed BEFORE running):
      C1 GO : HR@12 rel-gain >= +5% AND bootstrap CI excludes 0
              C1 KILL : l1_l2 <= META or CI includes 0  -> whole novelty dies
      C2 GO : HR@12 rel-gain >= +5% AND NDCG delta > 0 AND CI excludes 0
              C2 weak : rel-gain < +2% or CI includes 0  -> demote L2 to cold-start-only
      C3 GO : HR@12 rel-gain >= +3% AND CI excludes 0 (pooled)
              C3 KILL : delta <= 0 or CI includes 0  -> L3-as-accuracy dies, route to probe_02 (C4)
    Prior recorded signal is MIXED (cold-start 2-4: L1+L2 best, +L3 hurts), so the
    probe must be able to kill L3; that path is pre-committed above.

VERDICT
    Emits C1/C2/C3 GO/KILL flags. Project-level gate (combined with probe_02 C4):
      GO = C1 & C2 & (C3 or C4) ; PARTIAL = C1 & C2 & ~C3 & C4 ; NO-GO = ~C1.

Usage:
    uv run python witnesses/probe_01_incremental_layer_value.py [sample_users]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from witnesses._probe_common import (  # noqa: E402
    OUT_DIR,
    ScoreResult,
    bootstrap_delta,
    bracket_mask,
    build_fixed_users,
    load_variants,
    per_bracket_means,
    pooled_means,
    score_variant,
)

LADDER = ["META", "L1", "L1+L2", "L1+L2+L3"]
SEED = 42
N_BOOT = 1000


def _compare(
    name_a: str, name_b: str, sa: ScoreResult, sb: ScoreResult, fixed
) -> dict:
    """Paired comparison b vs a: pooled HR/NDCG bootstrap + 2-4 bracket HR."""
    m24 = bracket_mask(fixed, "2-4")
    return {
        "from": name_a,
        "to": name_b,
        "hr_pooled": bootstrap_delta(sa.hr, sb.hr, None, N_BOOT, seed=101),
        "ndcg_pooled": bootstrap_delta(sa.ndcg, sb.ndcg, None, N_BOOT, seed=102),
        "hr_bracket_2_4": bootstrap_delta(sa.hr, sb.hr, m24, N_BOOT, seed=103),
    }


def main() -> None:
    sample_users = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000

    print(f"[probe_01] loading ladder variants: {LADDER}")
    canon_ids, variants = load_variants(LADDER)
    n_items = len(canon_ids)
    print(f"[probe_01] aligned {n_items} items across {len(LADDER)} variants")

    fixed = build_fixed_users(canon_ids, sample_users=sample_users, seed=SEED)
    print(f"[probe_01] fixed eval users: {fixed.n_users}")
    bracket_counts = {b: int((fixed.brackets == b).sum()) for b in
                      sorted(set(fixed.brackets.tolist()))}
    print(f"[probe_01] bracket counts: {bracket_counts}")

    scores: dict[str, ScoreResult] = {}
    summary: dict[str, dict] = {}
    for nm in LADDER:
        print(f"[probe_01] scoring {nm} ...")
        res = score_variant(variants[nm], canon_ids, fixed, k=12)
        scores[nm] = res
        summary[nm] = {
            "pooled": pooled_means(res),
            "per_bracket": per_bracket_means(res, fixed),
        }
        p = summary[nm]["pooled"]
        b24 = summary[nm]["per_bracket"]["2-4"]
        print(f"   {nm}: pooled HR@12={p['hr_at_12']:.4f} NDCG={p['ndcg_at_12']:.4f}"
              f" | 2-4 HR@12={b24['hr_at_12']:.4f} (n={b24['n']})")

    # Pre-registered comparisons
    c1 = _compare("META", "L1+L2", scores["META"], scores["L1+L2"], fixed)
    c2 = _compare("L1", "L1+L2", scores["L1"], scores["L1+L2"], fixed)
    c3 = _compare("L1+L2", "L1+L2+L3", scores["L1+L2"], scores["L1+L2+L3"], fixed)

    # GO / KILL logic (pre-registered)
    def _ci_excl_0(d: dict) -> bool:
        return d["ci_lo"] > 0.0

    def _ci_incl_0(d: dict) -> bool:
        return d["ci_lo"] <= 0.0 <= d["ci_hi"]

    c1_hr = c1["hr_pooled"]
    c2_hr = c2["hr_pooled"]
    c3_hr = c3["hr_pooled"]

    c1_go = (c1_hr["rel_gain"] >= 0.05) and _ci_excl_0(c1_hr)
    c1_kill = (c1_hr["delta"] <= 0.0) or _ci_incl_0(c1_hr)

    c2_go = (c2_hr["rel_gain"] >= 0.05) and (c2["ndcg_pooled"]["delta"] > 0.0) and _ci_excl_0(c2_hr)
    c2_weak = (c2_hr["rel_gain"] < 0.02) or _ci_incl_0(c2_hr)

    c3_go = (c3_hr["rel_gain"] >= 0.03) and _ci_excl_0(c3_hr)
    c3_kill = (c3_hr["delta"] <= 0.0) or _ci_incl_0(c3_hr)

    flags = {
        "C1_go": bool(c1_go), "C1_kill": bool(c1_kill),
        "C2_go": bool(c2_go), "C2_weak": bool(c2_weak),
        "C3_go": bool(c3_go), "C3_kill": bool(c3_kill),
    }

    # Provisional project signal (C4 from probe_02 still pending)
    if c1_kill:
        project_signal = "NO-GO (C1 kill: metadata suffices, LLM layers add nothing)"
    elif c1_go and c2_go and c3_go:
        project_signal = "GO (full novelty: L1+L2+L3 all incremental)"
    elif c1_go and c2_go and c3_kill:
        project_signal = "PARTIAL pending probe_02 C4 (L3 accuracy killed; check L3 diversity/cold-start)"
    elif c1_go and c2_weak:
        project_signal = "PARTIAL (L2 weak; demote to cold-start-only) pending probe_02"
    else:
        project_signal = "AMBIGUOUS (see deltas; pending probe_02)"

    result = {
        "probe": "probe_01_incremental_layer_value",
        "sample_users": sample_users,
        "n_eval_users": fixed.n_users,
        "n_items": n_items,
        "seed": SEED,
        "n_boot": N_BOOT,
        "bracket_counts": bracket_counts,
        "variant_summary": summary,
        "comparisons": {"C1": c1, "C2": c2, "C3": c3},
        "flags": flags,
        "project_signal": project_signal,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "probe_01_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    def _fmt(d: dict) -> str:
        return (f"delta={d['delta']:+.5f} rel={d['rel_gain']*100:+.1f}% "
                f"CI=[{d['ci_lo']:+.5f},{d['ci_hi']:+.5f}] (n={d['n']})")

    print("\n" + "=" * 72)
    print("  PROBE 01 RESULT — incremental layer value (HR@12, pooled)")
    print("=" * 72)
    print(f"  C1  META     -> META+L1+L2     {_fmt(c1_hr)}")
    print(f"      [2-4]                       {_fmt(c1['hr_bracket_2_4'])}")
    print(f"  C2  META+L1  -> META+L1+L2     {_fmt(c2_hr)}  NDCGdelta={c2['ndcg_pooled']['delta']:+.5f}")
    print(f"      [2-4]                       {_fmt(c2['hr_bracket_2_4'])}")
    print(f"  C3  META+L1+L2 -> +L3          {_fmt(c3_hr)}")
    print(f"      [2-4]                       {_fmt(c3['hr_bracket_2_4'])}")
    print("-" * 72)
    print(f"  flags: {flags}")
    print(f"  VERDICT: {project_signal}")
    print("=" * 72)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
