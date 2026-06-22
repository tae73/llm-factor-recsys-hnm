"""PROBE 02 — L3 isolation + diversity/coverage fallback (Gate 0, C4/C5).

WHAT GENERALIZES
    probe_01 tests incremental ACCURACY (HR@12) of the layer ladder. probe_02
    asks the layered-novelty fallback question: if L3 (and maybe L2) do not help
    accuracy, do they help DISCOVERY — catalog coverage@12 and intra-list
    diversity@12 — which is what the project actually claims (87% single-purchase
    -> discovery-oriented, not repeat prediction)? It also isolates L3's marginal
    contribution and its redundancy with L2.

    Same content-based centroid-kNN engine as probe_01; diversity is measured on a
    FIXED reference geometry (item_bge_embeddings) so a variant cannot inflate its
    own diversity via its own space. No model training.

THE RESULT (boxed; persisted to probe_02_result.json)
    +-----------------------------------------------------------------------+
    |  C4  L1+L2 -> L1+L2+L3 : diversity@12 / coverage@12 gain, HR guard     |
    |  C5  L1+L3 vs L1, vs L1+L2 : does L3 carry less than L2? redundant?    |
    |  L3-marginal: META->L3, L2->L2+L3 (does L3 add anything alone?)        |
    +-----------------------------------------------------------------------+

HONEST reduces_check (pre-registered)
    C4 GO (save L3 as diversity-only):
        intra-list diversity@12 rel-gain >= +10% OR coverage@12 rel-gain >= +15%,
        AND pooled HR@12 loss <= 2% relative (does not wreck accuracy).
    If C4 fails AND probe_01 C3 killed L3 -> DROP L3 entirely (L1+L2 method),
    or (if C2 also weak) DROP to L1-only + reframe as "metadata-as-text" study.
    Diversity measured on fixed reference space to prevent self-inflation.

VERDICT
    Emits C4 GO/FAIL and the L3 disposition (diversity-only / dropped).

Usage:
    uv run python witnesses/probe_02_l3_isolation_and_diversity.py [sample_users]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from witnesses._probe_common import (  # noqa: E402
    OUT_DIR,
    bootstrap_delta,
    build_fixed_users,
    catalog_coverage,
    intra_list_diversity,
    load_reference,
    load_variants,
    per_bracket_means,
    pooled_means,
    score_variant,
)

VARIANTS = ["META", "L1", "L2", "L3", "L1+L2", "L1+L3", "L2+L3", "L1+L2+L3"]
SEED = 42
N_BOOT = 1000


def main() -> None:
    sample_users = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000

    print(f"[probe_02] loading variants: {VARIANTS}")
    canon_ids, variants = load_variants(VARIANTS)
    n_items = len(canon_ids)
    ref_emb = load_reference(canon_ids)
    print(f"[probe_02] aligned {n_items} items; reference geometry loaded")

    fixed = build_fixed_users(canon_ids, sample_users=sample_users, seed=SEED)
    print(f"[probe_02] fixed eval users: {fixed.n_users}")

    scores = {}
    summary = {}
    for nm in VARIANTS:
        print(f"[probe_02] scoring {nm} ...")
        res = score_variant(variants[nm], canon_ids, fixed, k=12)
        div = intra_list_diversity(res.topk, ref_emb)
        cov = catalog_coverage(res.topk, n_items)
        scores[nm] = (res, div)
        summary[nm] = {
            "pooled": pooled_means(res),
            "per_bracket": per_bracket_means(res, fixed),
            "diversity_at_12": float(div.mean()),
            "coverage_at_12": cov,
        }
        s = summary[nm]
        print(f"   {nm}: HR@12={s['pooled']['hr_at_12']:.4f} "
              f"div={s['diversity_at_12']:.4f} cov={s['coverage_at_12']:.4f}")

    # ----- C4: L3 as diversity/coverage contributor (L1+L2 -> L1+L2+L3) -----
    res_a, div_a = scores["L1+L2"]
    res_b, div_b = scores["L1+L2+L3"]
    div_boot = bootstrap_delta(div_a, div_b, None, N_BOOT, seed=201)
    cov_a = summary["L1+L2"]["coverage_at_12"]
    cov_b = summary["L1+L2+L3"]["coverage_at_12"]
    cov_rel = (cov_b - cov_a) / cov_a if cov_a > 0 else 0.0
    hr_a = summary["L1+L2"]["pooled"]["hr_at_12"]
    hr_b = summary["L1+L2+L3"]["pooled"]["hr_at_12"]
    hr_rel_loss = (hr_b - hr_a) / hr_a if hr_a > 0 else 0.0  # negative = loss

    c4_div_go = (div_boot["rel_gain"] >= 0.10) and (div_boot["ci_lo"] > 0.0)
    c4_cov_go = cov_rel >= 0.15
    c4_hr_guard = hr_rel_loss >= -0.02
    c4_go = (c4_div_go or c4_cov_go) and c4_hr_guard

    # ----- C5: L3 isolation / redundancy -----
    c5 = {
        "L1->L1+L3_hr": bootstrap_delta(scores["L1"][0].hr, scores["L1+L3"][0].hr, None, N_BOOT, 202),
        "L1+L2_vs_L1+L3_hr": {
            "l1_l2": summary["L1+L2"]["pooled"]["hr_at_12"],
            "l1_l3": summary["L1+L3"]["pooled"]["hr_at_12"],
        },
        "L2->L2+L3_hr": bootstrap_delta(scores["L2"][0].hr, scores["L2+L3"][0].hr, None, N_BOOT, 203),
        "META->L3_hr": bootstrap_delta(scores["META"][0].hr, scores["L3"][0].hr, None, N_BOOT, 204),
    }

    if c4_go:
        l3_disposition = "KEEP L3 as diversity/coverage-only contributor (layered-novelty fallback)"
    else:
        l3_disposition = "DROP L3 (no accuracy, no diversity/coverage gain) -> L1(+L2) method"

    result = {
        "probe": "probe_02_l3_isolation_and_diversity",
        "sample_users": sample_users,
        "n_eval_users": fixed.n_users,
        "n_items": n_items,
        "seed": SEED,
        "n_boot": N_BOOT,
        "variant_summary": summary,
        "C4": {
            "diversity": div_boot,
            "coverage_l1_l2": cov_a,
            "coverage_l1_l2_l3": cov_b,
            "coverage_rel_gain": cov_rel,
            "hr_rel_change": hr_rel_loss,
            "div_go": bool(c4_div_go),
            "cov_go": bool(c4_cov_go),
            "hr_guard_ok": bool(c4_hr_guard),
            "C4_go": bool(c4_go),
        },
        "C5": c5,
        "l3_disposition": l3_disposition,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "probe_02_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print("  PROBE 02 RESULT — L3 isolation + diversity/coverage (C4/C5)")
    print("=" * 72)
    print(f"  C4 diversity L1+L2 -> +L3: {div_boot['mean_a']:.4f} -> {div_boot['mean_b']:.4f}"
          f"  rel={div_boot['rel_gain']*100:+.1f}% CI=[{div_boot['ci_lo']:+.4f},{div_boot['ci_hi']:+.4f}]")
    print(f"  C4 coverage  L1+L2 -> +L3: {cov_a:.4f} -> {cov_b:.4f}  rel={cov_rel*100:+.1f}%")
    print(f"  C4 HR guard  (rel change): {hr_rel_loss*100:+.1f}%  (ok if >= -2%)")
    print(f"  C4 GO = {c4_go}  (div_go={c4_div_go}, cov_go={c4_cov_go}, hr_guard={c4_hr_guard})")
    print("-" * 72)
    print(f"  C5 L1->L1+L3 HR delta: {c5['L1->L1+L3_hr']['delta']:+.5f} "
          f"(rel {c5['L1->L1+L3_hr']['rel_gain']*100:+.1f}%)")
    print(f"  C5 HR  L1+L2={c5['L1+L2_vs_L1+L3_hr']['l1_l2']:.4f}  "
          f"L1+L3={c5['L1+L2_vs_L1+L3_hr']['l1_l3']:.4f}")
    print("-" * 72)
    print(f"  VERDICT: {l3_disposition}")
    print("=" * 72)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
