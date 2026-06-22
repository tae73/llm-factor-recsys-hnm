"""PROBE 09 — Does SEPARATE-layer encoding rescue L2/L3? (single-text bottleneck)

WHAT GENERALIZES
    probe_08 used the BLENDED l1_l2_l3 embedding (one BGE vector for all layers) as
    the discovery re-ranker, and found L2/L3 add ~0 over L1 — but the blend lets L1's
    concrete terms dominate a single 768-d vector (design flaw #1). This probe gives
    L2/L3 their OWN embedding space: late-fuse PER-LAYER cosines (cos in l1-space +
    cos in l2-space + cos in l3-space) instead of one blended cosine. If separate-
    layer fusion beats L1-only (and a metadata-matched control), the "redundancy" was
    a single-text-bottleneck artifact, not intrinsic.

    Same immediate-week discovery setup as probe_07/08 (popular candidate pool,
    NEW-item-only GT, history excluded). No training. Per-layer embeddings already on
    disk: ablation/{meta,l1,l2,l3,l1_l2_l3}.npz (each = metadata + that layer).

THE RESULT (boxed; persisted to probe_09_result.json)
    +-----------------------------------------------------------------------+
    | discovery MAP@12: L1-only | blended L1+L2+L3 | separate-sum L1+L2+L3   |
    |   + control (L1 + 2*meta) to rule out the extra-metadata confound       |
    +-----------------------------------------------------------------------+

HONEST reduces_check
    Per-layer npz each contain metadata, so a 3-term sum over-weights metadata; the
    L1+2*meta control has the SAME term count, isolating L2/L3's marginal effect. Sum-
    of-cosines equal-weight late fusion is a lower bound on a learned layer gate
    (probe_10). Frozen BGE caveat unchanged.

VERDICT
    GO (design suppressed L2/L3) iff separate-sum > L1-only AND > L1+2*meta control
    (paired CI excludes 0).

Usage:
    uv run python witnesses/probe_09_separate_layer_rerank.py [sample_users]
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
LAYERS = ["META", "L1", "L2", "L3", "L1+L2+L3"]
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
    canon_ids, V = load_variants(LAYERS)  # each L2-normalized
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

    def centroid(emb, hidx):
        c = emb[hidx].mean(0)
        return c / (np.linalg.norm(c) + 1e-9)

    schemes = ["L1_only", "blended_L1L2L3", "sep_L1L2", "sep_L1L2L3", "ctrl_L1_2meta"]
    ap = {s: [] for s in schemes}
    n_eval = 0
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
        n_eval += 1
        cos = {nm: centroid(V[nm], hidx) @ V[nm][cand_idx].T for nm in LAYERS}  # each (n_cand,)

        def rank_ap(score):
            order = np.argsort(-score)
            return _ap([cand_ids[j] for j in order], new_gt)

        ap["L1_only"].append(rank_ap(cos["L1"]))
        ap["blended_L1L2L3"].append(rank_ap(cos["L1+L2+L3"]))
        ap["sep_L1L2"].append(rank_ap(cos["L1"] + cos["L2"]))
        ap["sep_L1L2L3"].append(rank_ap(cos["L1"] + cos["L2"] + cos["L3"]))
        ap["ctrl_L1_2meta"].append(rank_ap(cos["L1"] + 2 * cos["META"]))

    means = {s: float(np.mean(ap[s])) for s in schemes}
    arr = {s: np.asarray(ap[s]) for s in schemes}

    def cmp(a, b):
        d = bootstrap_delta(arr[a], arr[b], None, 1000, seed=9)
        return {"from": a, "to": b, "delta": d["delta"], "rel": d["rel_gain"], "ci": [d["ci_lo"], d["ci_hi"]]}

    comparisons = {
        "sep_L1L2L3_vs_L1only": cmp("L1_only", "sep_L1L2L3"),
        "sep_L1L2L3_vs_control": cmp("ctrl_L1_2meta", "sep_L1L2L3"),
        "sep_L1L2L3_vs_blended": cmp("blended_L1L2L3", "sep_L1L2L3"),
        "sep_L1L2_vs_L1only": cmp("L1_only", "sep_L1L2"),
    }
    go = (comparisons["sep_L1L2L3_vs_L1only"]["ci"][0] > 0
          and comparisons["sep_L1L2L3_vs_control"]["ci"][0] > 0)
    verdict = ("GO — separate-layer encoding rescues L2/L3 (single-text bottleneck confirmed)"
               if go else
               "NO-GO — L2/L3 add ~0 even with separate-layer encoding")

    out = {"probe": "probe_09_separate_layer_rerank", "n_eval": n_eval,
           "rerank_discovery_map": means, "comparisons": comparisons, "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_09_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 68)
    print("  PROBE 09 — separate-layer vs blended re-ranker (discovery)")
    print("=" * 68)
    print(f"  n_eval={n_eval}")
    for s in schemes:
        print(f"  {s:18s} discovery MAP@12 = {means[s]:.5f}")
    print("-" * 68)
    for k, d in comparisons.items():
        print(f"  {k:26s} delta={d['delta']:+.5f} rel={d['rel']*100:+.1f}% CI=[{d['ci'][0]:+.5f},{d['ci'][1]:+.5f}]")
    print(f"  VERDICT: {verdict}")
    print("=" * 68)


if __name__ == "__main__":
    main()
