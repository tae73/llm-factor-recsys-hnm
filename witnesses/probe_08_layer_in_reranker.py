"""PROBE 08 — Does L2 (and L3) add value AS A RE-RANKER on discovery? (L2 framing)

WHAT GENERALIZES
    Gate-0 (probe_01) measured layer value under buy-similar full-catalog retrieval,
    where L1 (concrete product attrs) dominates and L2 (perceptual) is only weakly
    incremental. But the validated hybrid role for LLM is RE-RANKING a popular
    candidate pool (probe_07: L1+L2 rerank = +93% over popularity on discovery).
    This probe decomposes that re-ranker by layer (META / L1 / L1+L2 / L1+L2+L3) to
    test whether L2's incremental value is LARGER in the discovery/re-ranking
    context (matching user style/occasion to NEW trending items) than in buy-similar
    retrieval. If so, L2's contribution has a clear "home" (discovery personalization).

THE RESULT (boxed; persisted to probe_08_result.json)
    +-----------------------------------------------------------------------+
    | discovery MAP@12 of the content RE-RANKER, per layer variant          |
    | -> L1->L1+L2 increment in the re-ranker context (vs Gate-0's +8%/+15%) |
    +-----------------------------------------------------------------------+

HONEST reduces_check
    Same frozen-BGE proxy + hand-fixed candidate pool as probe_07; lower-bounds the
    trainable KAR. But isolates whether L2 carries discovery-specific re-ranking
    signal beyond L1, which directly informs how to frame L2's contribution.

Usage:
    uv run python witnesses/probe_08_layer_in_reranker.py [sample_users]
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
VARIANTS = ["META", "L1", "L1+L2", "L1+L2+L3"]
K = 12
POOL = 300
RECENT_DAYS = 30
SEED = 42


def _ap(slate_ids, new_gt, k=K):
    hits = [a in new_gt for a in slate_ids[:k]]
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
    canon_ids, variants = load_variants(VARIANTS)
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    txn = pd.read_parquet(TRAIN_TXN, columns=["customer_id", "article_id", "t_dat"])
    txn["customer_id"] = txn["customer_id"].astype(str)
    txn["article_id"] = txn["article_id"].astype(str)
    tmax = txn["t_dat"].max()
    pop_rank = txn[txn["t_dat"] > tmax - pd.Timedelta(days=RECENT_DAYS)]["article_id"].value_counts()
    pop_pool_ids = [a for a in pop_rank.index.tolist() if a in id_to_idx][:POOL * 3]

    user_history = _build_user_purchase_history(TRAIN_TXN)
    gt = {u: set(v) for u, v in json.loads(IMMEDIATE_GT.read_text()).items()}
    rng = np.random.default_rng(SEED)
    users = [u for u in gt if u in user_history]
    samp = list(rng.choice(users, size=min(sample, len(users)), replace=False))

    # precompute per-user candidate pool + new_gt + history centroid indices (shared)
    per_user = []
    for u in samp:
        hist = set(user_history[u])
        new_gt = {a for a in gt[u] if a not in hist}
        if not new_gt:
            continue
        cand_ids = [a for a in pop_pool_ids if a not in hist][:POOL]
        hidx = [id_to_idx[a] for a in user_history[u] if a in id_to_idx]
        if not hidx:
            continue
        per_user.append((cand_ids, [id_to_idx[a] for a in cand_ids], new_gt, hidx))

    pop_ap = float(np.mean([_ap(c[0], c[2]) for c in per_user]))

    res = {"popularity": pop_ap}
    ap_by_variant = {}
    for nm in VARIANTS:
        emb = variants[nm]
        aps = np.zeros(len(per_user))
        for i, (cand_ids, cand_idx, new_gt, hidx) in enumerate(per_user):
            c = emb[hidx].mean(0)
            c = c / (np.linalg.norm(c) + 1e-9)
            sims = emb[np.asarray(cand_idx)] @ c
            order = np.argsort(-sims)
            aps[i] = _ap([cand_ids[j] for j in order], new_gt)
        ap_by_variant[nm] = aps
        res[nm] = float(aps.mean())

    def inc(a, b):
        d = bootstrap_delta(ap_by_variant[a], ap_by_variant[b], None, 1000, seed=7)
        return {"delta": d["delta"], "rel": d["rel_gain"], "ci": [d["ci_lo"], d["ci_hi"]]}

    increments = {
        "META->L1": inc("META", "L1"),
        "L1->L1+L2": inc("L1", "L1+L2"),
        "L1+L2->L1+L2+L3": inc("L1+L2", "L1+L2+L3"),
    }
    out = {"probe": "probe_08_layer_in_reranker", "n_eval": len(per_user),
           "popularity": pop_ap, "rerank_discovery_map": {k: res[k] for k in VARIANTS},
           "increments": increments}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_08_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 66)
    print("  PROBE 08 — layer decomposition of the content RE-RANKER (discovery)")
    print("=" * 66)
    print(f"  n_eval={len(per_user)}  popularity={pop_ap:.5f}")
    for nm in VARIANTS:
        print(f"  rerank({nm:9s}) discovery MAP@12 = {res[nm]:.5f}  ({(res[nm]/pop_ap-1)*100:+.0f}% vs pop)")
    for k, d in increments.items():
        print(f"  {k:18s} delta={d['delta']:+.5f} rel={d['rel']*100:+.1f}% CI=[{d['ci'][0]:+.5f},{d['ci'][1]:+.5f}]")
    print("=" * 66)


if __name__ == "__main__":
    main()
