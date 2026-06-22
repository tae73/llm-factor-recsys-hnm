"""PROBE 14 — Root cause: are L2/L3 just a FUNCTION of L1? (product-internal redescription)

WHAT GENERALIZES
    The diagnosis for why L2/L3 add ~0 over L1: the LLM was a "product describer", not a
    "knowledge bringer" — L2 ("Casual/Minimalist") and L3 ("I-line/Monochromatic") are
    INFERRED from the same product that determines L1 ("cotton/slim/crew"). If so, L2/L3
    are predictable from L1 (the redundancy is by-construction, not a fixable measurement
    bug). This probe quantifies that: predict each L2/L3 attribute from the L1 embedding via
    k-NN (the nearest items in L1-space) and measure accuracy vs a majority-class baseline.
    High predictability = L2/L3 is a near-function of L1.

THE RESULT (boxed; persisted to probe_14_result.json)
    +-----------------------------------------------------------------------+
    | per L2/L3 attribute: kNN(L1)->attr accuracy vs majority baseline + lift |
    | (lift = fraction of the gap-to-perfect that L1 closes)                 |
    +-----------------------------------------------------------------------+

HONEST reduces_check
    Uses the l1 embedding (metadata+L1 text) as the predictor — exactly "what L1 gives us".
    kNN (no training) is a conservative predictor; a trained model would do >=. High lift =
    L2/L3 inferable from L1 = product-internal redescription = inevitable redundancy.

VERDICT
    diagnosis CONFIRMED iff L1 predicts L2/L3 well above majority (mean lift high).

Usage:
    uv run python witnesses/probe_14_layer_predictability.py [n_items]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from witnesses._probe_common import OUT_DIR, load_variants  # noqa: E402

FK = Path("data/knowledge/factual/factual_knowledge.parquet")
ATTRS = ["l2_style_mood", "l2_perceived_quality", "l2_trendiness", "l2_season_fit",
         "l2_versatility", "l3_color_harmony", "l3_tone_season", "l3_coordination_role",
         "l3_visual_weight", "l3_style_lineage"]
KNN = 15
SEED = 42


def _first(v) -> str:
    if isinstance(v, (list, np.ndarray)):
        return str(v[0]) if len(v) else "NA"
    return str(v)


def main() -> None:
    n_items = int(sys.argv[1]) if len(sys.argv) > 1 else 25_000
    canon_ids, V = load_variants(["L1"])
    emb = V["L1"]  # (n, 768) L2-normalized
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    fk = pd.read_parquet(FK, columns=["article_id"] + ATTRS)
    fk["article_id"] = fk["article_id"].astype(str)
    fk = fk[fk["article_id"].isin(id_to_idx)].reset_index(drop=True)
    idx = np.array([id_to_idx[a] for a in fk["article_id"]])

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(fk))
    if n_items < len(perm):
        perm = perm[:n_items]
    split = int(0.8 * len(perm))
    tr, te = perm[:split], perm[split:]
    tr_emb, te_emb = emb[idx[tr]], emb[idx[te]]
    # kNN in L1 space: for each test item, top-KNN train neighbors
    sims = te_emb @ tr_emb.T  # (n_te, n_tr)
    nn = np.argpartition(-sims, KNN, axis=1)[:, :KNN]  # (n_te, KNN) train-local indices

    results = {}
    lifts = []
    for attr in ATTRS:
        y = fk[attr].map(_first).to_numpy()
        ytr, yte = y[tr], y[te]
        # majority baseline (train mode)
        vals, cnts = np.unique(ytr, return_counts=True)
        majority = vals[cnts.argmax()]
        maj_acc = float((yte == majority).mean())
        # kNN majority vote
        preds = []
        for r in range(len(te)):
            neigh = ytr[nn[r]]
            v, c = np.unique(neigh, return_counts=True)
            preds.append(v[c.argmax()])
        knn_acc = float((np.array(preds) == yte).mean())
        lift = (knn_acc - maj_acc) / (1 - maj_acc) if maj_acc < 1 else 0.0
        results[attr] = {"knn_acc": knn_acc, "majority_acc": maj_acc, "lift": lift,
                         "n_classes": int(len(vals))}
        lifts.append(lift)

    mean_lift = float(np.mean(lifts))
    l2_lift = float(np.mean([results[a]["lift"] for a in ATTRS if a.startswith("l2")]))
    l3_lift = float(np.mean([results[a]["lift"] for a in ATTRS if a.startswith("l3")]))
    verdict = (f"DIAGNOSIS CONFIRMED — L1 predicts L2/L3 (mean lift={mean_lift:.2f}): "
               "L2/L3 are product-internal redescriptions of L1, redundancy is by-construction"
               if mean_lift >= 0.3 else
               f"WEAK — L1 predicts L2/L3 only at lift={mean_lift:.2f}")

    out = {"probe": "probe_14_layer_predictability", "n_train": len(tr), "n_test": len(te),
           "knn": KNN, "per_attribute": results,
           "mean_lift": mean_lift, "L2_mean_lift": l2_lift, "L3_mean_lift": l3_lift,
           "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_14_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 70)
    print("  PROBE 14 — is L2/L3 a function of L1? (kNN(L1) -> attribute)")
    print("=" * 70)
    print(f"  n_train={len(tr)} n_test={len(te)} kNN={KNN}")
    print(f"  {'attribute':24s} {'kNN_acc':>8s} {'majority':>9s} {'lift':>6s} {'classes':>8s}")
    for a in ATTRS:
        r = results[a]
        print(f"  {a:24s} {r['knn_acc']:>8.3f} {r['majority_acc']:>9.3f} {r['lift']:>6.2f} {r['n_classes']:>8d}")
    print("-" * 70)
    print(f"  mean lift = {mean_lift:.2f}  (L2 {l2_lift:.2f} / L3 {l3_lift:.2f})")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
