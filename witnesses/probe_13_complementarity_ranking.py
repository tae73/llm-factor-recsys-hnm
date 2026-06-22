"""PROBE 13 — A make-or-break: can L3 RANK complementary items? (outfit recommendation)

WHAT GENERALIZES
    probe_12 showed co-purchased cross-category items SHARE L3 attributes more than random
    (correlational). The make-or-break for the complementarity/outfit-recommendation
    direction (Track A) is stronger: given a SEED item, can an L3-based compatibility
    scorer RANK the actual co-purchased complement above realistic distractors better
    than popularity or L1-similarity? If yes, L3 powers a recommendation task L1 cannot
    -> Track A GO. If no, the coordination signal is too weak to rank -> reconsider.

    Task: held-out same-basket cross-category pairs (seed, complement). Candidate pool =
    [complement] + top popular cross-category distractors (product_group != seed's). Rank
    candidates by each scorer; measure where the true complement lands (HR@12, MRR). No
    training.

THE RESULT (boxed; persisted to probe_13_result.json)
    +-----------------------------------------------------------------------+
    | HR@12 / MRR of the true complement, per scorer:                        |
    |   popularity | L1-cos(sim) | L3-cos | harmony-match | combined(L3+attr)|
    +-----------------------------------------------------------------------+

HONEST reduces_check
    Same-basket co-purchase is a proxy for "outfit". Distractors are popular cross-cat
    items (realistic, hard). Frozen embeddings + hand scorers lower-bound a trained
    compatibility model. L1-cos is the "recommend SIMILAR" control that should FAIL at
    complementarity if the task is genuinely about matching-not-similarity.

VERDICT
    Track A GO iff an L3-based scorer beats BOTH popularity and L1-cos on HR@12/MRR.

Usage:
    uv run python witnesses/probe_13_complementarity_ranking.py [n_pairs]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from witnesses._probe_common import OUT_DIR, load_variants  # noqa: E402

TRAIN_TXN = Path("data/processed/train_transactions.parquet")
FK = Path("data/knowledge/factual/factual_knowledge.parquet")
ARTICLES = Path("data/processed/articles.parquet")
K = 12
N_DISTRACT = 100
SEED = 42


def main() -> None:
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    canon_ids, V = load_variants(["L1", "L3"])
    l1, l3 = V["L1"], V["L3"]
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    fk = pd.read_parquet(FK, columns=["article_id", "l3_color_harmony", "l3_style_lineage", "l3_tone_season"])
    fk["article_id"] = fk["article_id"].astype(str)
    ch = dict(zip(fk["article_id"], fk["l3_color_harmony"].astype(str)))
    sl = dict(zip(fk["article_id"], fk["l3_style_lineage"].astype(str)))

    art = pd.read_parquet(ARTICLES, columns=["article_id", "product_group_name"])
    art["article_id"] = art["article_id"].astype(str)
    pg = dict(zip(art["article_id"], art["product_group_name"].astype(str)))

    import duckdb

    rng = np.random.default_rng(SEED)
    # popularity (recent) for distractor pool + popularity scorer
    pop = duckdb.connect().execute(
        f"SELECT article_id, count(*) c FROM read_parquet('{TRAIN_TXN}') "
        f"WHERE t_dat >= DATE '2020-05-01' GROUP BY article_id ORDER BY c DESC LIMIT 4000"
    ).fetchall()
    pop_ids = [str(a) for a, _ in pop if str(a) in id_to_idx]
    pop_count = {str(a): c for a, c in pop}
    pool_idx = np.array([id_to_idx[a] for a in pop_ids])
    pool_pg = np.array([pg.get(a, "") for a in pop_ids])

    baskets = duckdb.connect().execute(
        f"SELECT list(article_id) items FROM read_parquet('{TRAIN_TXN}') "
        f"WHERE t_dat >= DATE '2020-06-01' GROUP BY customer_id, t_dat "
        f"HAVING count(*) BETWEEN 2 AND 6 LIMIT 120000"
    ).fetchall()
    pairs = []
    for (items,) in baskets:
        its = list(dict.fromkeys(str(a) for a in items if str(a) in id_to_idx))
        for i in range(len(its)):
            for j in range(len(its)):
                if i != j and pg.get(its[i]) != pg.get(its[j]):
                    pairs.append((its[i], its[j]))  # (seed, complement) directional
        if len(pairs) >= n_pairs:
            break
    pairs = pairs[:n_pairs]
    print(f"[probe_13] eval (seed,complement) pairs: {len(pairs)}")

    scorers = ["popularity", "L1_cos_sim", "L3_cos", "harmony_match",
               "L1_plus_L3", "L1_plus_harmony"]
    hr = {s: [] for s in scorers}
    rr = {s: [] for s in scorers}
    for seed, comp in pairs:
        s_pg = pg.get(seed)
        # distractor pool: popular items in a DIFFERENT product group than seed, != comp
        mask = (pool_pg != s_pg)
        cand_pool = [pop_ids[k] for k in np.where(mask)[0] if pop_ids[k] != comp][:N_DISTRACT]
        cand = [comp] + cand_pool
        cidx = np.array([id_to_idx[a] for a in cand])
        si1, si3 = l1[id_to_idx[seed]], l3[id_to_idx[seed]]
        l1c, l3c = l1[cidx] @ si1, l3[cidx] @ si3
        harm = np.array([(ch.get(a) == ch.get(seed) and ch.get(seed) != "nan")
                         + (sl.get(a) == sl.get(seed) and sl.get(seed) != "nan")
                         for a in cand], float)
        sc = {
            "popularity": np.array([pop_count.get(a, 0) for a in cand], float),
            "L1_cos_sim": l1c,
            "L3_cos": l3c,
            "harmony_match": harm + 0.01 * l3c,
            "L1_plus_L3": l1c + l3c,                 # does L3 ADD to L1?
            "L1_plus_harmony": l1c + 0.5 * harm,     # does coordination-match ADD to L1?
        }
        for s in scorers:
            order = np.argsort(-sc[s])
            rank = int(np.where(order == 0)[0][0]) + 1  # complement is index 0
            hr[s].append(1.0 if rank <= K else 0.0)
            rr[s].append(1.0 / rank)

    from witnesses._probe_common import bootstrap_delta
    res = {s: {"hr_at_12": float(np.mean(hr[s])), "mrr": float(np.mean(rr[s]))} for s in scorers}
    # KEY incremental test: does adding L3 to L1 beat L1 alone? (paired CI)
    incr = {
        "L1+L3 vs L1": bootstrap_delta(np.array(hr["L1_cos_sim"]), np.array(hr["L1_plus_L3"]), None, 1000, 13),
        "L1+harmony vs L1": bootstrap_delta(np.array(hr["L1_cos_sim"]), np.array(hr["L1_plus_harmony"]), None, 1000, 14),
    }
    l1_hr = res["L1_cos_sim"]["hr_at_12"]
    go = any(c["ci_lo"] > 0 for c in incr.values())  # L3 adds incrementally over L1
    verdict = ("Track A GO — L3/coordination adds incremental ranking value OVER L1 (CI excl 0)"
               if go else
               "NO-GO — L3 adds ~0 over L1 even for complementarity; L1 dominates this task too")

    out = {"probe": "probe_13_complementarity_ranking", "n_pairs": len(pairs),
           "n_distractors": N_DISTRACT, "scorers": res,
           "incremental_over_L1": {k: {"delta": v["delta"], "rel": v["rel_gain"],
                                       "ci": [v["ci_lo"], v["ci_hi"]]} for k, v in incr.items()},
           "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_13_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 64)
    print("  PROBE 13 — complementarity ranking (predict the co-purchased complement)")
    print("=" * 64)
    print(f"  n_pairs={len(pairs)}  distractors={N_DISTRACT} (popular cross-category)")
    for s in scorers:
        print(f"  {s:16s} HR@12={res[s]['hr_at_12']:.4f}  MRR={res[s]['mrr']:.4f}")
    print("-" * 64)
    for k, v in incr.items():
        print(f"  {k:18s} delta={v['delta']:+.5f} rel={v['rel_gain']*100:+.1f}% "
              f"CI=[{v['ci_lo']:+.5f},{v['ci_hi']:+.5f}]")
    print(f"  VERDICT: {verdict}")
    print("=" * 64)


if __name__ == "__main__":
    main()
