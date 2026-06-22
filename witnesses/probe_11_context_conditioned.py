"""PROBE 11 — Does query-time CONTEXT (season/occasion) make L2 useful? (task mismatch 5a)

WHAT GENERALIZES
    Design flaw #5: L2 season_fit/occasion are CONTEXT-conditional attributes, but our
    eval is single-item retrieval with NO query-time context. The immediate eval window
    is Jul 1-7 = SUMMER (a known context the static task ignores). This probe injects
    that context: re-rank the popular candidate pool with a season-match boost (in-season
    items up, off-season down) and an occasion-continuity boost (match the user's recent
    occasion mix). If context-aware re-ranking beats content-only, L2's value is real but
    inaccessible to the context-free task — confirming the mismatch rather than redundancy.

    Same immediate-week discovery setup. No training. Uses l2_season_fit / l2_occasion
    from factual_knowledge.parquet + L1 content embedding.

THE RESULT (boxed; persisted to probe_11_result.json)
    +-----------------------------------------------------------------------+
    | discovery MAP@12: content(L1) | +season-boost | +occasion-boost | both |
    | + season-only baseline (does season_fit carry standalone signal?)      |
    +-----------------------------------------------------------------------+

HONEST reduces_check
    Catalog is season-homogeneous (All-season 56%) and occasion-homogeneous (Everyday
    dominant), so the ceiling for context signal is low BY DESIGN. A single summer week
    cannot test cross-season generalization. Still, if even the in-season boost moves
    discovery, context-conditioning is the missing ingredient.

VERDICT
    GO (task mismatch — context unlocks L2) iff a context boost beats content-only (CI excl 0).

Usage:
    uv run python witnesses/probe_11_context_conditioned.py [sample_users]
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
FK = Path("data/knowledge/factual/factual_knowledge.parquet")
K = 12
POOL = 300
RECENT_DAYS = 30
SEED = 42
CURRENT_SEASON = "Summer"  # Jul 1-7
SEASON_SCORE = {"Summer": 1.0, "All-season": 0.5, "Spring": 0.0, "Fall": -0.5, "Winter": -1.0}


def _ap(order, cand_ids, new_gt, k=K):
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
    canon_ids, V = load_variants(["L1"])
    emb = V["L1"]
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    fk = pd.read_parquet(FK, columns=["article_id", "l2_occasion", "l2_season_fit"])
    fk["article_id"] = fk["article_id"].astype(str)
    item_season = dict(zip(fk["article_id"], fk["l2_season_fit"].astype(str)))
    item_occ = {a: set(o.tolist() if hasattr(o, "tolist") else ([o] if isinstance(o, str) else []))
                for a, o in zip(fk["article_id"], fk["l2_occasion"])}

    txn = pd.read_parquet(TRAIN_TXN, columns=["customer_id", "article_id", "t_dat"])
    txn["customer_id"] = txn["customer_id"].astype(str)
    txn["article_id"] = txn["article_id"].astype(str)
    tmax = txn["t_dat"].max()
    pop = txn[txn["t_dat"] > tmax - pd.Timedelta(days=RECENT_DAYS)]["article_id"].value_counts()
    pop_pool_ids = [a for a in pop.index.tolist() if a in id_to_idx][:POOL * 3]
    season_arr = np.array([SEASON_SCORE.get(item_season.get(a, "All-season"), 0.0) for a in pop_pool_ids])

    user_history = _build_user_purchase_history(TRAIN_TXN)
    last_txn = (txn.sort_values("t_dat").groupby("customer_id")["article_id"]
                .apply(lambda s: s.tolist()[-5:]))
    gt = {u: set(v) for u, v in json.loads(IMMEDIATE_GT.read_text()).items()}
    rng = np.random.default_rng(SEED)
    users = [u for u in gt if u in user_history]
    samp = list(rng.choice(users, size=min(sample, len(users)), replace=False))

    LAM = 0.15
    schemes = ["content_L1", "content+season", "content+occasion", "content+both", "season_only"]
    ap = {s: [] for s in schemes}
    for u in samp:
        hist = set(user_history[u])
        new_gt = {a for a in gt[u] if a not in hist}
        if not new_gt:
            continue
        cmask = np.array([a not in hist for a in pop_pool_ids])
        cidx_pool = np.where(cmask)[0][:POOL]
        cand_ids = [pop_pool_ids[j] for j in cidx_pool]
        cand_eidx = np.asarray([id_to_idx[a] for a in cand_ids])
        cand_season = season_arr[cidx_pool]
        hidx = [id_to_idx[a] for a in user_history[u] if a in id_to_idx]
        if not hidx or not cand_ids:
            continue
        c = emb[hidx].mean(0)
        c = c / (np.linalg.norm(c) + 1e-9)
        content = c @ emb[cand_eidx].T  # (n_cand,)
        # recent occasion mix
        recent_occ = set()
        for a in last_txn.get(u, []):
            recent_occ |= item_occ.get(a, set())
        occ_match = np.array([1.0 if (item_occ.get(a, set()) & recent_occ) else 0.0 for a in cand_ids])

        ap["content_L1"].append(_ap(np.argsort(-content), cand_ids, new_gt))
        ap["content+season"].append(_ap(np.argsort(-(content + LAM * cand_season)), cand_ids, new_gt))
        ap["content+occasion"].append(_ap(np.argsort(-(content + LAM * occ_match)), cand_ids, new_gt))
        ap["content+both"].append(_ap(np.argsort(-(content + LAM * cand_season + LAM * occ_match)), cand_ids, new_gt))
        ap["season_only"].append(_ap(np.argsort(-cand_season), cand_ids, new_gt))

    means = {s: float(np.mean(ap[s])) for s in schemes}
    arr = {s: np.asarray(ap[s]) for s in schemes}

    def cmp(b):
        d = bootstrap_delta(arr["content_L1"], arr[b], None, 1000, seed=11)
        return {"delta": d["delta"], "rel": d["rel_gain"], "ci": [d["ci_lo"], d["ci_hi"]]}

    comps = {b: cmp(b) for b in ["content+season", "content+occasion", "content+both"]}
    go = any(c["ci"][0] > 0 for c in comps.values())
    verdict = ("GO — query-time context (season/occasion) unlocks L2 (task mismatch confirmed)"
               if go else
               "NO-GO — context boost does not beat content-only")

    out = {"probe": "probe_11_context_conditioned", "n_eval": len(arr["content_L1"]),
           "current_season": CURRENT_SEASON, "lambda": LAM,
           "discovery_map": means, "vs_content": comps, "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_11_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 66)
    print("  PROBE 11 — context-conditioned (season/occasion) re-ranker")
    print("=" * 66)
    print(f"  n_eval={out['n_eval']}  season={CURRENT_SEASON}  lambda={LAM}")
    for s in schemes:
        print(f"  {s:18s} discovery MAP@12 = {means[s]:.5f}")
    for b, c in comps.items():
        print(f"  {b:18s} vs content: delta={c['delta']:+.5f} rel={c['rel']*100:+.1f}% CI=[{c['ci'][0]:+.5f},{c['ci'][1]:+.5f}]")
    print(f"  VERDICT: {verdict}")
    print("=" * 66)


if __name__ == "__main__":
    main()
