"""PROBE 10 — Optimal per-layer weighting: do L2/L3 ever earn weight? (layer-gating)

WHAT GENERALIZES
    Design flaw #3: KAR has no LAYER gate (only factual-vs-reasoning). This probe
    grants the maximal version of a layer gate: grid-search GLOBAL per-layer weights
    (w_l1, w_l2, w_l3) on the late-fused per-layer cosine re-ranker, choosing the
    weights that MAXIMIZE discovery MAP@12. If the optimum still puts ~0 on L2/L3
    (i.e. best weights ~ (1,0,0) and best ~ L1-only), then even a perfect layer gate
    cannot make L2/L3 help single-item discovery — refuting flaw #3 as a rescue.

    Same immediate-week discovery setup (popular pool, NEW-only GT). Frozen embeddings.

THE RESULT (boxed; persisted to probe_10_result.json)
    +-----------------------------------------------------------------------+
    | best per-layer weights + discovery MAP@12 vs L1-only                  |
    +-----------------------------------------------------------------------+

HONEST reduces_check
    Global (not per-user) weights + grid is an upper bound on a static layer gate and
    a lower bound on a context-dependent learned gate; but if L2/L3 get ~0 weight at
    the GLOBAL optimum, a static gate clearly can't rescue them on this task.

VERDICT
    GO (gating absence was the issue) iff best weights give L2 or L3 non-trivial
    weight AND beat L1-only (CI excl 0).

Usage:
    uv run python witnesses/probe_10_layer_weighted_rerank.py [sample_users]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.cold_start import _build_user_purchase_history  # noqa: E402
from witnesses._probe_common import OUT_DIR, bootstrap_delta, load_variants  # noqa: E402

TRAIN_TXN = Path("data/processed/train_transactions.parquet")
IMMEDIATE_GT = Path("data/processed/immediate_ground_truth.json")
LAYERS = ["L1", "L2", "L3"]
K = 12
POOL = 300
RECENT_DAYS = 30
SEED = 42


def _ap_from_order(order, cand_ids, new_gt, k=K):
    slate = [cand_ids[j] for j in order[:k]]
    hits = [a in new_gt for a in slate]
    nr = min(len(new_gt), k)
    s = 0.0
    nh = 0
    for r, h in enumerate(hits, 1):
        if h:
            nh += 1
            s += nh / r
    return s / nr if nr else 0.0


def main() -> None:
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    canon_ids, V = load_variants(LAYERS)
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    txn = pd.read_parquet(TRAIN_TXN, columns=["customer_id", "article_id", "t_dat"])
    txn["customer_id"] = txn["customer_id"].astype(str)
    txn["article_id"] = txn["article_id"].astype(str)
    tmax = txn["t_dat"].max()
    pop = txn[txn["t_dat"] > tmax - pd.Timedelta(days=RECENT_DAYS)]["article_id"].value_counts()
    pop_pool_ids = [a for a in pop.index.tolist() if a in id_to_idx][:POOL * 3]

    user_history = _build_user_purchase_history(TRAIN_TXN)
    gt = {u: set(v) for u, v in json.loads(IMMEDIATE_GT.read_text()).items()}
    rng = np.random.default_rng(SEED)
    users = [u for u in gt if u in user_history]
    samp = list(rng.choice(users, size=min(sample, len(users)), replace=False))

    # precompute per-user (cand_ids, per-layer cosine vectors, new_gt)
    rows = []
    for u in samp:
        hist = set(user_history[u])
        new_gt = {a for a in gt[u] if a not in hist}
        if not new_gt:
            continue
        cand_ids = [a for a in pop_pool_ids if a not in hist][:POOL]
        cand_idx = np.asarray([id_to_idx[a] for a in cand_ids])
        hidx = [id_to_idx[a] for a in user_history[u] if a in id_to_idx]
        if not hidx or not cand_ids:
            continue
        cos = {}
        for nm in LAYERS:
            c = V[nm][hidx].mean(0)
            c = c / (np.linalg.norm(c) + 1e-9)
            cos[nm] = c @ V[nm][cand_idx].T
        rows.append((cand_ids, np.stack([cos["L1"], cos["L2"], cos["L3"]]), new_gt))  # (3, n_cand)

    def eval_weights(w):
        aps = np.empty(len(rows))
        for i, (cand_ids, cmat, new_gt) in enumerate(rows):
            score = w @ cmat
            order = np.argsort(-score)
            aps[i] = _ap_from_order(order, cand_ids, new_gt)
        return aps

    # grid over simplex (step 0.1)
    grid = [(a, b, 1 - a - b) for a in np.arange(0, 1.01, 0.1) for b in np.arange(0, 1.01 - a, 0.1)]
    best_w, best_aps, best_mean = None, None, -1.0
    l1_aps = eval_weights(np.array([1.0, 0.0, 0.0]))
    l1_mean = float(l1_aps.mean())
    for w in grid:
        wa = np.array(w)
        aps = eval_weights(wa)
        m = float(aps.mean())
        if m > best_mean:
            best_mean, best_w, best_aps = m, wa, aps

    d = bootstrap_delta(l1_aps, best_aps, None, 1000, seed=10)
    nontrivial = best_w[1] + best_w[2] >= 0.2  # L2+L3 weight >= 0.2
    go = nontrivial and d["ci_lo"] > 0
    verdict = ("GO — optimal layer weighting gives L2/L3 weight and beats L1-only"
               if go else
               f"NO-GO — optimal weights={np.round(best_w,2).tolist()} (L1,L2,L3); best≈L1-only")

    out = {"probe": "probe_10_layer_weighted_rerank", "n_eval": len(rows),
           "L1_only_map": l1_mean, "best_weights_L1L2L3": best_w.tolist(),
           "best_map": best_mean, "best_vs_L1only": {"delta": d["delta"], "rel": d["rel_gain"],
           "ci": [d["ci_lo"], d["ci_hi"]]}, "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_10_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 66)
    print("  PROBE 10 — optimal per-layer weighting (discovery re-ranker)")
    print("=" * 66)
    print(f"  n_eval={len(rows)}")
    print(f"  L1-only discovery MAP@12 = {l1_mean:.5f}")
    print(f"  BEST weights (L1,L2,L3) = {np.round(best_w,2).tolist()}  MAP@12 = {best_mean:.5f}")
    print(f"  best vs L1-only: delta={d['delta']:+.5f} rel={d['rel_gain']*100:+.1f}% "
          f"CI=[{d['ci_lo']:+.5f},{d['ci_hi']:+.5f}]")
    print(f"  VERDICT: {verdict}")
    print("=" * 66)


if __name__ == "__main__":
    main()
