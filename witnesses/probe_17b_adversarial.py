"""PROBE 17b — adversarial verify of probe_17 NO-GO (before killing option-1)

The probe_17 NO-GO (external user-profile loses to L1 on discovery) could be an artifact of
(a) MEAN-pooling washing out local complementarity — a trained attention/selective Expert
    would NOT mean-pool; approximate its ceiling with MAX-SIM aggregation:
        score(cand) = max_{h in history} cos(external_emb[h], L1[cand])
    "does this candidate complement ANY single item I already own?"
(b) TASK framing — external predicts COMPLEMENTS; if most next-NEW purchases are same-category
    refreshes, L1 wins overall by construction. Decompose discovery GT into same-PG vs
    cross-PG (complementary) next-purchases; external should win on the cross-PG subset if the
    signal is real.

If MAX-SIM external AND the cross-PG subset BOTH still favor L1 -> NO-GO is robust, option-1
(full external-KAR for discovery) is dead even with a trained Expert. If either flips ->
external knowledge has a narrower viable surface (outfit completion / cross-sell), reported.

VERDICT
    NO-GO ROBUST iff external (max-sim) <= L1 overall AND external <= L1 on the cross-PG subset.

Usage:
    uv run python witnesses/probe_17b_adversarial.py [n_users]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_16_external_knowledge as p16  # noqa: E402

from src.evaluation.metrics import compute_map_at_k  # noqa: E402
from witnesses._probe_common import OUT_DIR, bootstrap_delta, load_variants  # noqa: E402

TRAIN_TXN = Path("data/processed/train_transactions.parquet")
ARTICLES = Path("data/processed/articles.parquet")
GT_PATH = Path("data/processed/immediate_ground_truth.json")
K = 12
POOL = 5000
MIN_COVERED = 3
SEED = 42


def _disc_ap(pred: dict, gt: dict, hist: dict, users: list, restrict_pg=None, pg_of=None, user_pgs=None) -> np.ndarray:
    out = []
    for u in users:
        new_gt = [g for g in gt[u] if g not in hist[u]]
        if restrict_pg == "same":
            new_gt = [g for g in new_gt if pg_of.get(g) in user_pgs[u]]
        elif restrict_pg == "cross":
            new_gt = [g for g in new_gt if pg_of.get(g) not in user_pgs[u]]
        if not new_gt:
            continue
        out.append(compute_map_at_k({u: pred[u]}, {u: new_gt}, K))
    return np.array(out)


def main() -> None:
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    canon_ids, V = load_variants(["L1"])
    l1 = V["L1"]
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    art = pd.read_parquet(ARTICLES, columns=["article_id", "product_group_name"])
    art["article_id"] = art["article_id"].astype(str)
    pg_of = dict(zip(art["article_id"], art["product_group_name"]))

    ck = pd.read_parquet(p16.CACHE)
    comp_text = dict(zip(ck["article_id"].astype(str), ck["complement_text"]))
    from sentence_transformers import SentenceTransformer
    ext_ids = [a for a in comp_text if a in id_to_idx]
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:1")
    model.max_seq_length = 512
    arr = model.encode([comp_text[a] for a in ext_ids], batch_size=128,
                        normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    ext_emb = {a: arr[i] for i, a in enumerate(ext_ids)}
    ext_set = set(ext_ids)

    import duckdb
    con = duckdb.connect()
    pop = con.execute(
        f"SELECT article_id, count(*) c FROM read_parquet('{TRAIN_TXN}') WHERE t_dat >= DATE '2020-05-01' "
        f"GROUP BY article_id ORDER BY c DESC LIMIT {POOL}"
    ).fetchall()
    pool_ids = [str(a) for a, _ in pop if str(a) in id_to_idx]
    pool_idx = np.array([id_to_idx[a] for a in pool_ids])
    pool_l1 = l1[pool_idx]
    pool_ext = np.stack([ext_emb[a] if a in ext_emb else np.zeros(l1.shape[1], np.float32) for a in pool_ids])

    gt = {str(k): [str(x) for x in v] for k, v in json.loads(GT_PATH.read_text()).items()}
    gt_users = pd.DataFrame({"customer_id": list(gt)})
    con.register("gt_users", gt_users)
    hist_rows = con.execute(
        f"SELECT t.customer_id, list(DISTINCT t.article_id) FROM read_parquet('{TRAIN_TXN}') t "
        f"JOIN gt_users g ON t.customer_id=g.customer_id GROUP BY t.customer_id"
    ).fetchall()
    train_history = {str(cid): set(str(a) for a in items) for cid, items in hist_rows}

    rng = np.random.default_rng(SEED)
    eligible = [u for u, h in train_history.items() if len(h & ext_set) >= MIN_COVERED and u in gt]
    rng.shuffle(eligible)
    eligible = eligible[:n_users]
    user_pgs = {u: {pg_of.get(h) for h in train_history[u]} for u in eligible}

    preds = {s: {} for s in ["L1_profile", "ext_meansim", "ext_maxsim", "L1_maxsim", "ext_maxsim+L1"]}
    for u in eligible:
        hist = train_history[u]
        cov = [h for h in hist if h in ext_set]
        hist_l1 = l1[[id_to_idx[h] for h in cov]]          # (nc,768)
        hist_ext = np.stack([ext_emb[h] for h in cov])      # (nc,768)
        l1_prof = hist_l1.mean(0); l1_prof /= np.linalg.norm(l1_prof) + 1e-9
        ext_prof = hist_ext.mean(0); ext_prof /= np.linalg.norm(ext_prof) + 1e-9
        s_l1 = pool_l1 @ l1_prof
        s_extmean = pool_l1 @ ext_prof
        s_extmax = (hist_ext @ pool_l1.T).max(0)            # complement ANY owned item
        s_l1max = (hist_l1 @ pool_l1.T).max(0)              # similar to ANY owned item
        owned = np.array([(a in hist) for a in pool_ids])

        def topk(score):
            sc = np.where(owned, -np.inf, score)
            order = np.argpartition(-sc, K)[:K]
            return [pool_ids[i] for i in order[np.argsort(-sc[order])]]
        preds["L1_profile"][u] = topk(s_l1)
        preds["ext_meansim"][u] = topk(s_extmean)
        preds["ext_maxsim"][u] = topk(s_extmax)
        preds["L1_maxsim"][u] = topk(s_l1max)
        zmax = (s_extmax - s_extmax.mean()) / (s_extmax.std() + 1e-9)
        zl1 = (s_l1 - s_l1.mean()) / (s_l1.std() + 1e-9)
        preds["ext_maxsim+L1"][u] = topk(zmax + zl1)

    def agg(ap):
        return float(np.mean(ap)) if len(ap) else float("nan")

    overall = {s: agg(_disc_ap(preds[s], gt, train_history, eligible)) for s in preds}
    same = {s: agg(_disc_ap(preds[s], gt, train_history, eligible, "same", pg_of, user_pgs))
            for s in ["L1_profile", "ext_maxsim"]}
    cross = {s: agg(_disc_ap(preds[s], gt, train_history, eligible, "cross", pg_of, user_pgs))
             for s in ["L1_profile", "ext_maxsim"]}

    ap_l1 = _disc_ap(preds["L1_profile"], gt, train_history, eligible)
    ap_max = _disc_ap(preds["ext_maxsim"], gt, train_history, eligible)
    cmp = bootstrap_delta(ap_l1, ap_max, None, 1000, 16)
    robust = (overall["ext_maxsim"] <= overall["L1_profile"]) and (cross["ext_maxsim"] <= cross["L1_profile"])
    verdict = ("NO-GO ROBUST — external (max-sim, selective) <= L1 overall AND on cross-category subset: "
               "option-1 (full external-KAR for discovery) dead even with a trained Expert"
               if robust else
               "PARTIAL — external wins on a subset (narrower outfit-completion surface), reported")

    out = {"probe": "probe_17b_adversarial", "n_eligible": len(eligible),
           "overall_disc_map": overall, "same_pg_disc_map": same, "cross_pg_disc_map": cross,
           "ext_maxsim_vs_L1": {"delta": cmp["delta"], "rel": cmp["rel_gain"], "ci": [cmp["ci_lo"], cmp["ci_hi"]]},
           "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_17b_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 76)
    print("  PROBE 17b — adversarial verify of probe_17 NO-GO (max-sim + PG decomposition)")
    print("=" * 76)
    print(f"  eligible={len(eligible)}")
    print(f"  {'scorer':16s} {'overall':>10s}")
    for s in preds:
        print(f"  {s:16s} {overall[s]:>10.5f}")
    print(f"  -- discovery decomposition (L1_profile vs ext_maxsim) --")
    print(f"  same-PG (similar) : L1 {same['L1_profile']:.5f}  ext_max {same['ext_maxsim']:.5f}")
    print(f"  cross-PG (complmt): L1 {cross['L1_profile']:.5f}  ext_max {cross['ext_maxsim']:.5f}")
    print(f"  ext_maxsim vs L1: delta={cmp['delta']:+.5f} rel={cmp['rel_gain']*100:+.1f}% "
          f"CI=[{cmp['ci_lo']:+.5f},{cmp['ci_hi']:+.5f}]")
    print(f"  VERDICT: {verdict}")
    print("=" * 76)


if __name__ == "__main__":
    main()
