"""PROBE 06 — Does content/LLM fill the NEW-item discovery gap? (hybrid thesis)

WHAT GENERALIZES
    probe_05 showed: on the immediate next week, repurchase nails the ~4% repeat
    purchases (MAP@12~0.024) but its discovery_map ~0 (it cannot predict the 96%
    NEW items). The hybrid thesis: LLM/content (KAR item attributes) is the lever
    for that 96% discovery + new users. This probe tests it cheaply (no training):
    content-based centroid-kNN over the ablation layer embeddings, evaluated with
    discovery_map (NEW-item-only GT) on the IMMEDIATE-next-week split, with the
    user's already-purchased items excluded from the recommendation slate.

THE RESULT (boxed; persisted to probe_06_result.json)
    +-----------------------------------------------------------------------+
    | discovery MAP@12 / HR@12 per layer (META, L1, L1+L2, L1+L2+L3)         |
    | vs the global-popularity discovery floor (~0.0027) and repurchase (~0).|
    +-----------------------------------------------------------------------+

HONEST reduces_check
    discovery_map isolates the 96% NEW-item portion where repurchase cannot help.
    If content-kNN beats the popularity discovery floor AND L1/L2 add over META,
    the LLM-for-discovery thesis has direct, fair support; if not, LLM's recsys
    value is limited even in its designated niche. Frozen-BGE proxy caveat (as in
    Gate-0) still applies; in-model KAR is the final arbiter.

VERDICT
    content discovery vs popularity floor; layer increments on the discovery task.

Usage:
    uv run python witnesses/probe_06_content_discovery.py [sample_users]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.cold_start import _build_user_purchase_history  # noqa: E402
from witnesses._probe_common import (  # noqa: E402
    OUT_DIR,
    bootstrap_delta,
    build_fixed_users,
    load_variants,
    score_variant,
)

LADDER = ["META", "L1", "L1+L2", "L1+L2+L3"]
IMMEDIATE_GT = Path("data/processed/immediate_ground_truth.json")
TRAIN_TXN = Path("data/processed/train_transactions.parquet")
SEED = 42
K = 12
TOPN = 80  # over-fetch then drop history items -> discovery slate of K


def _discovery_metrics(topn_idx, canon_ids, fixed, hist_idx_sets):
    """Per-user discovery MAP@K / HR@K: drop history items from the slate, eval on
    NEW-only ground truth (GT items not in the user's train history)."""
    n = fixed.n_users
    ap = np.zeros(n)
    hr = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        gt = fixed.gt[i]
        hist = hist_idx_sets[i]
        new_gt = {a for a in gt if a not in fixed.user_hist_ids[i]}  # article ids not bought
        if not new_gt:
            continue
        valid[i] = True
        # build discovery slate: top items excluding already-purchased indices
        slate = [int(j) for j in topn_idx[i] if int(j) not in hist][:K]
        hits = [canon_ids[j] in new_gt for j in slate]
        # AP@K
        num_rel = min(len(new_gt), K)
        score = 0.0
        nhit = 0
        for rank, h in enumerate(hits, 1):
            if h:
                nhit += 1
                score += nhit / rank
        ap[i] = score / num_rel if num_rel > 0 else 0.0
        hr[i] = float(any(hits))
    return ap, hr, valid


def main() -> None:
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 40_000

    canon_ids, variants = load_variants(LADDER)
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    user_history = _build_user_purchase_history(TRAIN_TXN)
    raw_gt = json.loads(IMMEDIATE_GT.read_text())
    immediate_gt = {u: set(v) for u, v in raw_gt.items()}

    from types import SimpleNamespace

    fixed = build_fixed_users(canon_ids, sample_users=sample, seed=SEED,
                              user_history=user_history, val_gt=immediate_gt)
    hist_idx_sets = [set(int(j) for j in hi) for hi in fixed.hist_indices]
    user_hist_ids = [set(user_history.get(u, [])) for u in fixed.user_ids]
    # FixedUsers is a NamedTuple; extend with per-user purchased article-id sets
    fx = SimpleNamespace(**fixed._asdict(), user_hist_ids=user_hist_ids)

    print(f"[probe_06] immediate-split discovery, eval users={fx.n_users}")

    results = {}
    ap_by_variant = {}
    for nm in LADDER:
        res = score_variant(variants[nm], canon_ids, fx, k=TOPN)
        ap, hr, valid = _discovery_metrics(res.topk, canon_ids, fx, hist_idx_sets)
        ap_by_variant[nm] = (ap, valid)
        results[nm] = {
            "discovery_map_at_12": float(ap[valid].mean()),
            "discovery_hr_at_12": float(hr[valid].mean()),
            "n_eval": int(valid.sum()),
        }
        print(f"   {nm}: discovery MAP@12={results[nm]['discovery_map_at_12']:.5f} "
              f"HR@12={results[nm]['discovery_hr_at_12']:.5f} (n={results[nm]['n_eval']})")

    # layer increments on discovery (paired bootstrap, common valid users)
    def paired(a_nm, b_nm):
        a_ap, a_v = ap_by_variant[a_nm]
        b_ap, b_v = ap_by_variant[b_nm]
        m = a_v & b_v
        return bootstrap_delta(a_ap[m], b_ap[m], None, 1000, seed=11)

    comps = {
        "META->L1": paired("META", "L1"),
        "L1->L1+L2": paired("L1", "L1+L2"),
        "L1+L2->L1+L2+L3": paired("L1+L2", "L1+L2+L3"),
    }

    out = {
        "probe": "probe_06_content_discovery",
        "eval": "immediate_next_week + discovery_map (new-item-only, history excluded)",
        "k": K, "topn_fetch": TOPN, "sample": sample,
        "popularity_discovery_floor": 0.00268,  # global_pop discovery_map (probe foundation)
        "repurchase_discovery": 0.00042,
        "variants": results,
        "layer_increments": comps,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_06_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("  PROBE 06 — content discovery (immediate split, NEW-item GT)")
    print("=" * 70)
    print(f"  popularity discovery floor = 0.00268 | repurchase discovery = 0.00042")
    for nm in LADDER:
        print(f"  {nm:10s} discovery MAP@12 = {results[nm]['discovery_map_at_12']:.5f}")
    for k, d in comps.items():
        print(f"  {k:16s} delta={d['delta']:+.5f} rel={d['rel_gain']*100:+.1f}% "
              f"CI=[{d['ci_lo']:+.5f},{d['ci_hi']:+.5f}]")
    print("=" * 70)
    print(f"  saved -> {OUT_DIR / 'probe_06_result.json'}")


if __name__ == "__main__":
    main()
