"""PROBE 07 — Does content PERSONALIZATION of popular candidates beat popularity? (KAR gate)

WHAT GENERALIZES
    probe_06: frozen content-kNN alone LOSES to popularity on new-item discovery
    (0.0013 < 0.0027). But discovery in a hybrid is not "content vs popularity" —
    it is "can content RE-RANK / personalize the popular candidate pool to do
    better than raw popularity?". If even this cheap content+popularity hybrid
    cannot beat the popularity floor, an expensive trainable KAR is unlikely to,
    and the LLM's recsys-accuracy value must be repositioned (diversity / specific
    cold cohorts). If it CAN, the hybrid (and KAR) thesis is worth the training.

    Setup (immediate next week, discovery = NEW-item-only GT, history excluded):
      candidate pool = top-N recent-popular NEW items per user; rank by
        (a) popularity (floor), (b) content-sim to user's L1+L2 history centroid,
        (c) hybrid = (1-a)*rank_pop + a*rank_content  (rank-blend over the pool).
    No training.

THE RESULT (boxed; persisted to probe_07_result.json)
    +-----------------------------------------------------------------------+
    | discovery MAP@12: popularity vs content-rerank vs hybrid-blend (pool)  |
    +-----------------------------------------------------------------------+

HONEST reduces_check
    This is still a frozen-BGE proxy and a hand-tuned blend, so it lower-bounds
    what a trainable KAR could learn. But it directly answers the gating question:
    does LLM content carry ANY new-item signal on top of popularity?

VERDICT
    GO for KAR training iff content-rerank or hybrid beats the popularity floor.

Usage:
    uv run python witnesses/probe_07_hybrid_discovery.py [sample_users]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.cold_start import _build_user_purchase_history  # noqa: E402
from witnesses._probe_common import OUT_DIR, load_variants  # noqa: E402

TRAIN_TXN = Path("data/processed/train_transactions.parquet")
IMMEDIATE_GT = Path("data/processed/immediate_ground_truth.json")
K = 12
POOL = 300          # candidate pool size (recent-popular NEW items per user)
RECENT_DAYS = 30    # popularity window
SEED = 42


def _ap_hr(slate_ids, new_gt, k=K):
    hits = [a in new_gt for a in slate_ids[:k]]
    num_rel = min(len(new_gt), k)
    s = 0.0
    nh = 0
    for r, h in enumerate(hits, 1):
        if h:
            nh += 1
            s += nh / r
    return (s / num_rel if num_rel else 0.0), float(any(hits))


def main() -> None:
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    # content embedding: best content variant (L1+L2)
    canon_ids, variants = load_variants(["L1+L2"])
    emb = variants["L1+L2"]
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    txn = pd.read_parquet(TRAIN_TXN, columns=["customer_id", "article_id", "t_dat"])
    txn["customer_id"] = txn["customer_id"].astype(str)
    txn["article_id"] = txn["article_id"].astype(str)
    tmax = txn["t_dat"].max()
    recent = txn[txn["t_dat"] > tmax - pd.Timedelta(days=RECENT_DAYS)]
    pop_rank = recent["article_id"].value_counts()  # ordered desc
    pop_pool_ids = [a for a in pop_rank.index.tolist() if a in id_to_idx][:POOL * 3]
    pop_pool_idx = np.array([id_to_idx[a] for a in pop_pool_ids])
    pop_pool_emb = emb[pop_pool_idx]  # (P, d)

    user_history = _build_user_purchase_history(TRAIN_TXN)
    gt = {u: set(v) for u, v in json.loads(IMMEDIATE_GT.read_text()).items()}

    rng = np.random.default_rng(SEED)
    users = [u for u in gt if u in user_history]
    samp = list(rng.choice(users, size=min(sample, len(users)), replace=False))

    rows = {"popularity": [], "content_rerank": [], "hybrid_blend": []}
    n_eval = 0
    for u in samp:
        hist = set(user_history[u])
        new_gt = {a for a in gt[u] if a not in hist}
        if not new_gt:
            continue
        n_eval += 1
        # candidate pool: top recent-popular NEW items (exclude history)
        cand_mask = np.array([a not in hist for a in pop_pool_ids])
        cand_idx = np.where(cand_mask)[0][:POOL]
        cand_ids = [pop_pool_ids[j] for j in cand_idx]
        # (a) popularity order (already sorted)
        rows["popularity"].append(_ap_hr(cand_ids, new_gt)[0])
        # content centroid
        hidx = [id_to_idx[a] for a in user_history[u] if a in id_to_idx]
        if not hidx:
            rows["content_rerank"].append(rows["popularity"][-1])
            rows["hybrid_blend"].append(rows["popularity"][-1])
            continue
        c = emb[hidx].mean(0)
        c = c / (np.linalg.norm(c) + 1e-9)
        sims = pop_pool_emb[cand_idx] @ c  # (n_cand,)
        # (b) content rerank of the pool
        order_c = np.argsort(-sims)
        rows["content_rerank"].append(_ap_hr([cand_ids[j] for j in order_c], new_gt)[0])
        # (c) hybrid rank-blend: pop rank index vs content rank index
        pop_rk = np.arange(len(cand_ids))
        con_rk = np.empty(len(cand_ids)); con_rk[order_c] = np.arange(len(cand_ids))
        blend = 0.5 * pop_rk + 0.5 * con_rk
        order_h = np.argsort(blend)
        rows["hybrid_blend"].append(_ap_hr([cand_ids[j] for j in order_h], new_gt)[0])

    result = {
        "probe": "probe_07_hybrid_discovery",
        "eval": "immediate_week discovery_map (NEW-only, history excluded)",
        "pool": POOL, "recent_days": RECENT_DAYS, "n_eval": n_eval,
        "popularity_discovery_map": float(np.mean(rows["popularity"])),
        "content_rerank_discovery_map": float(np.mean(rows["content_rerank"])),
        "hybrid_blend_discovery_map": float(np.mean(rows["hybrid_blend"])),
    }
    floor = result["popularity_discovery_map"]
    result["content_rerank_rel_vs_pop"] = result["content_rerank_discovery_map"] / floor - 1
    result["hybrid_rel_vs_pop"] = result["hybrid_blend_discovery_map"] / floor - 1
    result["verdict"] = (
        "GO (content adds over popularity)" if max(
            result["content_rerank_discovery_map"], result["hybrid_blend_discovery_map"]
        ) > floor else "NO-GO (content adds nothing over popularity for discovery)"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_07_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 64)
    print("  PROBE 07 — hybrid content+popularity discovery (candidate pool)")
    print("=" * 64)
    print(f"  n_eval={n_eval}  pool={POOL}  recent={RECENT_DAYS}d")
    print(f"  popularity      discovery MAP@12 = {result['popularity_discovery_map']:.5f}")
    print(f"  content-rerank  discovery MAP@12 = {result['content_rerank_discovery_map']:.5f} "
          f"({result['content_rerank_rel_vs_pop']*100:+.1f}% vs pop)")
    print(f"  hybrid-blend    discovery MAP@12 = {result['hybrid_blend_discovery_map']:.5f} "
          f"({result['hybrid_rel_vs_pop']*100:+.1f}% vs pop)")
    print(f"  VERDICT: {result['verdict']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
