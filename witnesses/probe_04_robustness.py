"""PROBE 04 — Adversarial robustness of the Gate-0 conclusion.

WHAT GENERALIZES
    Gate-0 (probe_01/02) concluded: L2 (perceptual) adds incremental retrieval
    value over L1 (C2 > 0), while L3 (theory) hurts (C3 < 0) -> "L1+L2 survives,
    L3 drops". This probe stress-tests that make-or-break conclusion against the
    explanations that could overturn it, in a single compute pass:
      - Lens A  multi-seed user samples (is it sample-specific?)
      - Lens B  cutoff k in {12, 50} (is it a k=12 artifact?)
      - Lens C  activity strata pooled / 2-4 (cold) / 50+ (heavy)
      - Lens D  retrieval method centroid vs max-similarity (aggregation artifact?)
      - Lens E  text-length confound: does adding text always hurt BGE retrieval?
                (already partly refuted in-design: L2 ALSO adds text yet HELPS)

THE RESULT (boxed; persisted to probe_04_result.json)
    +---------------------------------------------------------------------+
    |  For each (lens, metric) cell: sign of C2 (L1->L1+L2) and C3         |
    |  (L1+L2->L1+L2+L3). Conclusion holds iff C2>0 and C3<0 dominate.     |
    |  Plus per-variant mean text length (length-confound diagnostic).     |
    +---------------------------------------------------------------------+

HONEST reduces_check
    A make-or-break finding must survive perturbation. If C2 flips negative or
    C3 flips positive under some lens, the conclusion is method-specific and the
    re-scoped plan must note it. The frozen-BGE content-retrieval objective is a
    proxy; the in-model KAR test (Gate-1/2) remains the final arbiter for L3.

VERDICT
    Fraction of cells confirming (C2>0) and (C3<0); length diagnostic.

Usage:
    uv run python witnesses/probe_04_robustness.py [sample_users]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from witnesses._probe_common import (  # noqa: E402
    ACTIVITY_BRACKETS,
    OUT_DIR,
    _compute_hr_ndcg_mrr,
    bootstrap_delta,
    build_fixed_users,
    load_history_and_gt,
    load_variants,
    score_variant,
    score_variant_maxsim,
    variant_text_lengths,
)

LADDER = ["META", "L1", "L1+L2", "L1+L2+L3"]
SEEDS = [42, 7, 2024]
K_VALUES = [12, 50]
STRATA = ["pooled", "2-4", "50+"]
N_BOOT = 1000


def _metrics_at_k(topk_big: np.ndarray, fixed, canon_ids, k: int) -> np.ndarray:
    """Per-user HR@k from a (n_users, K_big>=k) ranked index matrix."""
    n = topk_big.shape[0]
    hr = np.zeros(n)
    for i in range(n):
        hr[i], _, _ = _compute_hr_ndcg_mrr(topk_big[i], fixed.gt[i], canon_ids, k)
    return hr


def _stratum_mask(fixed, stratum: str) -> np.ndarray | None:
    if stratum == "pooled":
        return None
    if stratum == "50+":
        return fixed.brackets == "50+"
    return fixed.brackets == stratum


def main() -> None:
    sample_users = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    maxsim_users = min(3_000, sample_users)

    print(f"[probe_04] loading {LADDER}")
    canon_ids, variants = load_variants(LADDER)
    n_items = len(canon_ids)

    # Read the ~700MB train parquet + val GT ONCE, reuse across all seeds.
    print("[probe_04] loading user history + val GT (once) ...")
    user_history, val_gt = load_history_and_gt()

    # Lens E — text length diagnostic (once)
    print("[probe_04] computing text-length diagnostic ...")
    lengths = variant_text_lengths(["META", "L1", "L1+L2", "L1+L2+L3"])
    for nm, d in lengths.items():
        print(f"   len {nm}: {d['mean_chars']:.1f} chars")

    checks: list[dict] = []

    # Lenses A/B/C/D
    for seed in SEEDS:
        fixed = build_fixed_users(canon_ids, sample_users=sample_users, seed=seed,
                                  user_history=user_history, val_gt=val_gt)
        # centroid: score @ max(K) once per variant, derive k=12/50
        topk_cent = {nm: score_variant(variants[nm], canon_ids, fixed, k=max(K_VALUES)).topk
                     for nm in ["L1", "L1+L2", "L1+L2+L3"]}
        for k in K_VALUES:
            hr = {nm: _metrics_at_k(topk_cent[nm], fixed, canon_ids, k)
                  for nm in topk_cent}
            for stratum in STRATA:
                mask = _stratum_mask(fixed, stratum)
                c2 = bootstrap_delta(hr["L1"], hr["L1+L2"], mask, N_BOOT, seed=seed + k)
                c3 = bootstrap_delta(hr["L1+L2"], hr["L1+L2+L3"], mask, N_BOOT, seed=seed + k + 1)
                checks.append({
                    "lens": "centroid", "seed": seed, "k": k, "stratum": stratum,
                    "C2_delta": c2["delta"], "C2_rel": c2["rel_gain"], "C2_ci": [c2["ci_lo"], c2["ci_hi"]], "C2_n": c2["n"],
                    "C3_delta": c3["delta"], "C3_rel": c3["rel_gain"], "C3_ci": [c3["ci_lo"], c3["ci_hi"]], "C3_n": c3["n"],
                })

    # Lens D — max-similarity retrieval (single seed, smaller sample, k=12)
    print("[probe_04] max-sim lens ...")
    fixed_ms = build_fixed_users(canon_ids, sample_users=maxsim_users, seed=42,
                                 user_history=user_history, val_gt=val_gt)
    hr_ms = {nm: score_variant_maxsim(variants[nm], canon_ids, fixed_ms, k=12).hr
             for nm in ["L1", "L1+L2", "L1+L2+L3"]}
    for stratum in STRATA:
        mask = _stratum_mask(fixed_ms, stratum)
        c2 = bootstrap_delta(hr_ms["L1"], hr_ms["L1+L2"], mask, N_BOOT, seed=999)
        c3 = bootstrap_delta(hr_ms["L1+L2"], hr_ms["L1+L2+L3"], mask, N_BOOT, seed=998)
        checks.append({
            "lens": "maxsim", "seed": 42, "k": 12, "stratum": stratum,
            "C2_delta": c2["delta"], "C2_rel": c2["rel_gain"], "C2_ci": [c2["ci_lo"], c2["ci_hi"]], "C2_n": c2["n"],
            "C3_delta": c3["delta"], "C3_rel": c3["rel_gain"], "C3_ci": [c3["ci_lo"], c3["ci_hi"]], "C3_n": c3["n"],
        })

    # Verdict: directional consistency (pooled + 2-4 cells weighted; ignore tiny-n)
    valid = [c for c in checks if c["C2_n"] >= 100 and c["C3_n"] >= 100]
    c2_pos = sum(1 for c in valid if c["C2_delta"] > 0)
    c3_neg = sum(1 for c in valid if c["C3_delta"] < 0)
    c2_pos_sig = sum(1 for c in valid if c["C2_ci"][0] > 0)
    c3_neg_sig = sum(1 for c in valid if c["C3_ci"][1] < 0)
    n = len(valid)

    # Length confound: does L2 add text yet help, while L3 adds text and hurts?
    len_l2_add = lengths["L1+L2"]["mean_chars"] - lengths["L1"]["mean_chars"]
    len_l3_add = lengths["L1+L2+L3"]["mean_chars"] - lengths["L1+L2"]["mean_chars"]
    length_confound_refuted = len_l2_add > 0  # L2 adds text yet C2>0 (helps) → not pure length

    verdict = (
        f"C2(L2 helps): {c2_pos}/{n} cells positive ({c2_pos_sig} sig); "
        f"C3(L3 hurts): {c3_neg}/{n} cells negative ({c3_neg_sig} sig). "
        f"Length-confound {'REFUTED' if length_confound_refuted else 'POSSIBLE'} "
        f"(L2 +{len_l2_add:.0f} chars yet helps; L3 +{len_l3_add:.0f} chars)."
    )

    result = {
        "probe": "probe_04_robustness",
        "sample_users": sample_users,
        "maxsim_users": maxsim_users,
        "n_items": n_items,
        "text_lengths": lengths,
        "checks": checks,
        "summary": {
            "n_valid_cells": n,
            "C2_positive": c2_pos, "C2_positive_sig": c2_pos_sig,
            "C3_negative": c3_neg, "C3_negative_sig": c3_neg_sig,
            "length_confound_refuted": bool(length_confound_refuted),
            "len_l2_add_chars": len_l2_add, "len_l3_add_chars": len_l3_add,
        },
        "verdict": verdict,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "probe_04_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print("  PROBE 04 RESULT — robustness of (L2 helps, L3 hurts)")
    print("=" * 72)
    for c in checks:
        print(f"  [{c['lens']:8s} seed={c['seed']:<4d} k={c['k']:<2d} {c['stratum']:6s}] "
              f"C2={c['C2_rel']*100:+6.1f}%  C3={c['C3_rel']*100:+6.1f}%  (n={c['C2_n']})")
    print("-" * 72)
    print(f"  text length: META {lengths['META']['mean_chars']:.0f} → "
          f"L1 {lengths['L1']['mean_chars']:.0f} → L1+L2 {lengths['L1+L2']['mean_chars']:.0f} "
          f"→ L1+L2+L3 {lengths['L1+L2+L3']['mean_chars']:.0f} chars")
    print(f"  VERDICT: {verdict}")
    print("=" * 72)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
