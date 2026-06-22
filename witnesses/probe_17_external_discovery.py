"""PROBE 17 — Does external styling knowledge beat L1 at USER-LEVEL discovery? (option-1 gate)

WHAT GENERALIZES
    probe_16 showed external knowledge beats product-similarity on seed->complement PAIR
    ranking. The real recommendation task is USER-LEVEL discovery: given a user's wardrobe,
    predict the NEW items they actually buy next. This probe is the make-or-break gate for
    the full external-knowledge-KAR build: build a user profile two ways —
      L1-profile        = mean L1 embedding of history items   ("recommend SIMILAR to my wardrobe")
      external-profile  = mean external-knowledge embedding    ("recommend COMPLEMENTS of my wardrobe")
    — re-rank a shared popular candidate pool, exclude owned items, and score with
    discovery_map (NEW-only GT, the R-4 infra). If external-profile > L1-profile on
    discovery_map, the styling-knowledge signal survives aggregation into a real recommender
    and the expensive full extraction + end-to-end KAR training is justified. If not, the
    complementarity result (probe_16) is the ceiling and we stop honestly.

EXTRACTION / COST
    Extends the probe_16 cache to the top-N most-purchased items (these dominate histories),
    GPT-4.1-nano, only items NOT already cached. Appends to
    data/knowledge/external/complement_knowledge.parquet (no re-spend on re-run).

THE RESULT (boxed; persisted to probe_17_result.json)
    +----------------------------------------------------------------------------+
    | discovery MAP@12 / HR@12: popularity | L1-profile | external-profile | blend |
    +----------------------------------------------------------------------------+

HONEST reduces_check
    Frozen (no training) — a lower bound on what a trainable external-Expert can do.
    Popular-item coverage biases toward head items (reported). Same candidate pool +
    same owned-item exclusion for all scorers, so the comparison is apples-to-apples.

VERDICT
    GO (build full) iff external-profile discovery_map > L1-profile (CI excludes 0).

Usage:
    uv run python witnesses/probe_17_external_discovery.py [n_items_extract] [n_users]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # to import probe_16

import probe_16_external_knowledge as p16  # noqa: E402

from src.config import EvalConfig  # noqa: E402
from src.evaluation.cohorts import discovery_map  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from witnesses._probe_common import OUT_DIR, bootstrap_delta, load_variants  # noqa: E402

TRAIN_TXN = Path("data/processed/train_transactions.parquet")
ARTICLES = Path("data/processed/articles.parquet")
GT_PATH = Path("data/processed/immediate_ground_truth.json")
CACHE = p16.CACHE
K = 12
POOL = 5000
MIN_COVERED = 3  # user needs >=3 history items with external knowledge
SEED = 42


def _ensure_extracted(top_items: list[str], meta: dict) -> dict[str, str]:
    """Extend external-knowledge cache to cover top_items (extract only missing)."""
    have: dict[str, str] = {}
    if CACHE.exists():
        ck = pd.read_parquet(CACHE)
        have = dict(zip(ck["article_id"].astype(str), ck["complement_text"]))
    missing = [a for a in top_items if a not in have and a in meta]
    if missing:
        p16._load_env_key()
        import os
        assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not found (.env)"
        print(f"[probe_17] extending external knowledge: {len(missing)} new items (have {len(have)})...")
        seeds_meta = [{"article_id": a, **meta[a]} for a in missing]
        new = asyncio.run(p16._extract(seeds_meta))
        have.update(new)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"article_id": list(have), "complement_text": list(have.values())}).to_parquet(CACHE)
        print(f"[probe_17] cache now {len(have)} items (+{len(new)})")
    else:
        print(f"[probe_17] cache already covers all {len(top_items)} top items ({len(have)} total)")
    return have


def main() -> None:
    n_extract = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    n_users = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    canon_ids, V = load_variants(["L1"])
    l1 = V["L1"]
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    art = pd.read_parquet(ARTICLES, columns=["article_id", "product_type_name", "colour_group_name",
                                             "section_name", "product_group_name", "detail_desc"])
    art["article_id"] = art["article_id"].astype(str)
    meta = art.set_index("article_id").to_dict("index")

    import duckdb
    con = duckdb.connect()

    # top items (dominate histories) -> ensure external knowledge
    top = con.execute(
        f"SELECT article_id FROM read_parquet('{TRAIN_TXN}') GROUP BY article_id "
        f"ORDER BY count(*) DESC LIMIT {n_extract}"
    ).fetchall()
    top_items = [str(a) for (a,) in top if str(a) in id_to_idx]
    comp_text = _ensure_extracted(top_items, meta)

    # embed external texts (BGE, free)
    from sentence_transformers import SentenceTransformer
    ext_ids = [a for a in comp_text if a in id_to_idx]
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:1")
    model.max_seq_length = 512
    arr = model.encode([comp_text[a] for a in ext_ids], batch_size=128,
                        normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    ext_emb = {a: arr[i] for i, a in enumerate(ext_ids)}
    ext_set = set(ext_ids)

    # candidate pool = recent popular items (discovery candidates)
    pop = con.execute(
        f"SELECT article_id, count(*) c FROM read_parquet('{TRAIN_TXN}') WHERE t_dat >= DATE '2020-05-01' "
        f"GROUP BY article_id ORDER BY c DESC LIMIT {POOL}"
    ).fetchall()
    pool_ids = [str(a) for a, _ in pop if str(a) in id_to_idx]
    pool_count = np.array([c for a, c in pop if str(a) in id_to_idx], float)
    pool_idx = np.array([id_to_idx[a] for a in pool_ids])
    pool_l1 = l1[pool_idx]  # (POOL, 768)

    # ground truth + train history (GT users only)
    gt = {str(k): [str(x) for x in v] for k, v in json.loads(GT_PATH.read_text()).items()}
    gt_users = pd.DataFrame({"customer_id": list(gt)})
    con.register("gt_users", gt_users)
    hist_rows = con.execute(
        f"SELECT t.customer_id, list(DISTINCT t.article_id) FROM read_parquet('{TRAIN_TXN}') t "
        f"JOIN gt_users g ON t.customer_id=g.customer_id GROUP BY t.customer_id"
    ).fetchall()
    train_history = {str(cid): set(str(a) for a in items) for cid, items in hist_rows}

    rng = np.random.default_rng(SEED)
    # eligible users: >=MIN_COVERED history items with external knowledge
    eligible = [u for u, h in train_history.items()
                if len(h & ext_set) >= MIN_COVERED and u in gt]
    rng.shuffle(eligible)
    eligible = eligible[:n_users]
    coverage = len(eligible) / max(1, len(gt))
    print(f"[probe_17] eligible users (>= {MIN_COVERED} covered hist items): {len(eligible)} "
          f"({coverage*100:.1f}% of {len(gt)} GT users)")

    pool_pos = {a: i for i, a in enumerate(pool_ids)}
    preds = {s: {} for s in ["popularity", "L1_profile", "external_profile", "blend"]}
    for u in eligible:
        hist = train_history[u]
        cov = [h for h in hist if h in ext_set]
        l1_prof = l1[[id_to_idx[h] for h in cov]].mean(0)
        l1_prof /= np.linalg.norm(l1_prof) + 1e-9
        ext_prof = np.stack([ext_emb[h] for h in cov]).mean(0)
        ext_prof /= np.linalg.norm(ext_prof) + 1e-9
        s_l1 = pool_l1 @ l1_prof
        s_ext = pool_l1 @ ext_prof
        s_pop = pool_count
        owned = np.array([(a in hist) for a in pool_ids])
        def topk(score):
            sc = np.where(owned, -np.inf, score)  # exclude owned (discovery)
            order = np.argpartition(-sc, K)[:K]
            order = order[np.argsort(-sc[order])]
            return [pool_ids[i] for i in order]
        preds["popularity"][u] = topk(s_pop)
        preds["L1_profile"][u] = topk(s_l1)
        preds["external_profile"][u] = topk(s_ext)
        # blend: z-normalized L1 + external
        zl = (s_l1 - s_l1.mean()) / (s_l1.std() + 1e-9)
        ze = (s_ext - s_ext.mean()) / (s_ext.std() + 1e-9)
        preds["blend"][u] = topk(zl + ze)

    cfg = EvalConfig(k=K)
    sub_gt = {u: gt[u] for u in eligible}
    sub_hist = {u: train_history[u] for u in eligible}
    res = {}
    for s in preds:
        d = discovery_map(preds[s], sub_gt, sub_hist, k=K)
        a = evaluate(preds[s], sub_gt, cfg)  # all-GT (incl. repurchase) for context
        res[s] = {"discovery_map": d.map_at_k, "discovery_hr": d.hr_at_k,
                  "all_map": a.map_at_k, "all_hr": a.hr_at_k}

    # per-user discovery AP for bootstrap CI (external vs L1)
    from src.evaluation.metrics import compute_map_at_k

    def per_user_disc_ap(pred):
        out = []
        for u in eligible:
            new_gt = [g for g in gt[u] if g not in train_history[u]]
            if not new_gt:
                continue
            out.append(compute_map_at_k({u: pred[u]}, {u: new_gt}, K))
        return np.array(out)

    ap_l1 = per_user_disc_ap(preds["L1_profile"])
    ap_ext = per_user_disc_ap(preds["external_profile"])
    cmp = bootstrap_delta(ap_l1, ap_ext, None, 1000, 16)
    go = res["external_profile"]["discovery_map"] > res["L1_profile"]["discovery_map"] and cmp["ci_lo"] > 0
    verdict = ("GO — external (complement) profile beats L1 (similar) profile on user-level discovery "
               "-> full external-KAR build justified"
               if go else
               "NO-GO — external profile does not beat L1 on user-level discovery; probe_16 (pair-level) is the ceiling")

    out = {"probe": "probe_17_external_discovery", "n_eligible_users": len(eligible),
           "coverage_frac": coverage, "n_extracted": len(ext_ids), "pool": POOL,
           "scorers": res, "external_vs_L1_discovery": {"delta": cmp["delta"], "rel": cmp["rel_gain"],
                                                        "ci": [cmp["ci_lo"], cmp["ci_hi"]]},
           "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_17_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 76)
    print("  PROBE 17 — external styling knowledge as USER-LEVEL discovery re-ranker")
    print("=" * 76)
    print(f"  eligible={len(eligible)} ({coverage*100:.1f}% GT)  extracted={len(ext_ids)}  pool={POOL}")
    print(f"  {'scorer':18s} {'disc_MAP@12':>11s} {'disc_HR@12':>11s} {'all_MAP':>9s}")
    for s in ["popularity", "L1_profile", "external_profile", "blend"]:
        r = res[s]
        print(f"  {s:18s} {r['discovery_map']:>11.5f} {r['discovery_hr']:>11.5f} {r['all_map']:>9.5f}")
    print(f"  external vs L1 (discovery): delta={cmp['delta']:+.5f} rel={cmp['rel_gain']*100:+.1f}% "
          f"CI=[{cmp['ci_lo']:+.5f},{cmp['ci_hi']:+.5f}]")
    print(f"  VERDICT: {verdict}")
    print("=" * 76)


if __name__ == "__main__":
    main()
