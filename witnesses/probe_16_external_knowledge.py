"""PROBE 16 — Does LLM EXTERNAL (styling) knowledge beat product-similarity? (KAR's real premise)

WHAT GENERALIZES
    Root-cause (probe_14): L1/L2/L3 are product-INTERNAL re-descriptions, so they cannot
    add over L1. KAR's actual premise is OPEN-WORLD knowledge the product/interactions lack.
    This probe realizes it for the first time: ask the LLM, as a stylist, what items
    COMPLETE an outfit with a seed (external styling knowledge NOT in the product image),
    embed that complement-description, and test whether it ranks the ACTUAL co-purchased
    complement above realistic distractors BETTER than L1 product-similarity (which won in
    probe_13). If yes, the LLM's external styling knowledge carries complementarity signal
    that product-content does not -> the KAR concept is rescued via external knowledge.

CACHING / COST
    New, separate extraction (~600 seed items, GPT-4.1-nano, ~$0.5). Cached to
    data/knowledge/external/complement_knowledge.parquet — re-runs do NOT re-spend.
    Touches NOTHING existing (L1/L2/L3, embeddings, data all untouched).

THE RESULT (boxed; persisted to probe_16_result.json)
    +-----------------------------------------------------------------------+
    | complement HR@12 / MRR: popularity | L1_cos(product-sim) | external_kn |
    +-----------------------------------------------------------------------+

VERDICT
    GO (external knowledge works) iff external-knowledge ranking beats L1_cos and popularity.

Usage:
    uv run python witnesses/probe_16_external_knowledge.py [n_seeds] [n_pairs]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from witnesses._probe_common import OUT_DIR, bootstrap_delta, load_variants  # noqa: E402

TRAIN_TXN = Path("data/processed/train_transactions.parquet")
ARTICLES = Path("data/processed/articles.parquet")
CACHE = Path("data/knowledge/external/complement_knowledge.parquet")
MODEL = "gpt-4.1-nano"
K = 12
N_DISTRACT = 100
SEED = 42

SYSTEM = (
    "You are a professional fashion stylist with broad knowledge of outfit coordination, "
    "current trends, and styling rules. Given ONE product a customer bought, describe the "
    "COMPLEMENTARY items (in OTHER categories) that complete an outfit with it. Use external "
    "styling knowledge — do NOT just re-describe the given product. Name the complementary "
    "product types, colors, materials, and the styling rationale. Be concrete (2-3 sentences)."
)


def _load_env_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                return


async def _extract(seeds_meta: list[dict]) -> dict[str, str]:
    import openai

    client = openai.AsyncOpenAI()
    sem = asyncio.Semaphore(24)

    async def one(m: dict) -> tuple[str, str]:
        desc = (f"Product type: {m.get('product_type_name')}; Color: {m.get('colour_group_name')}; "
                f"Section: {m.get('section_name')}; Group: {m.get('product_group_name')}. "
                f"{str(m.get('detail_desc') or '')[:200]}")
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.responses.create(
                        model=MODEL,
                        input=[{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": f"The customer bought: {desc}\n\nComplementary items to complete the outfit:"}],
                    )
                    return m["article_id"], r.output_text.strip()
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        return m["article_id"], ""
                    await asyncio.sleep(2 ** attempt)
        return m["article_id"], ""

    out = await asyncio.gather(*[one(m) for m in seeds_meta])
    return {aid: txt for aid, txt in out if txt}


def _ap_hr(order, cand_ids, comp, k=K):
    rank = int(np.where(order == 0)[0][0]) + 1  # complement is index 0
    return (1.0 if rank <= k else 0.0), 1.0 / rank


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    canon_ids, V = load_variants(["L1"])
    l1 = V["L1"]
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    art = pd.read_parquet(ARTICLES, columns=["article_id", "product_type_name", "colour_group_name",
                                             "section_name", "product_group_name", "detail_desc"])
    art["article_id"] = art["article_id"].astype(str)
    meta = {r.article_id: r._asdict() for r in art.itertuples(index=False)} if False else \
        art.set_index("article_id").to_dict("index")
    pg = {a: m["product_group_name"] for a, m in meta.items()}

    import duckdb

    rng = np.random.default_rng(SEED)
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
        f"WHERE t_dat >= DATE '2020-06-01' GROUP BY customer_id, t_dat HAVING count(*) BETWEEN 2 AND 6 LIMIT 120000"
    ).fetchall()
    pairs = []
    for (items,) in baskets:
        its = list(dict.fromkeys(str(a) for a in items if str(a) in id_to_idx))
        for i in range(len(its)):
            for j in range(len(its)):
                if i != j and pg.get(its[i]) != pg.get(its[j]):
                    pairs.append((its[i], its[j]))
        if len(pairs) >= n_pairs:
            break
    pairs = pairs[:n_pairs]

    # choose seeds (most frequent seeds in pairs) and extract external knowledge (cached)
    seed_counts = pd.Series([s for s, _ in pairs]).value_counts()
    seeds = [s for s in seed_counts.index if s in meta][:n_seeds]
    if CACHE.exists():
        ck = pd.read_parquet(CACHE)
        comp_text = dict(zip(ck["article_id"].astype(str), ck["complement_text"]))
        print(f"[probe_16] loaded cached external knowledge: {len(comp_text)}")
    else:
        _load_env_key()
        assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not found (.env)"
        seeds_meta = [{"article_id": s, **meta[s]} for s in seeds]
        print(f"[probe_16] extracting external complement knowledge for {len(seeds_meta)} seeds (GPT-4.1-nano)...")
        comp_text = asyncio.run(_extract(seeds_meta))
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"article_id": list(comp_text), "complement_text": list(comp_text.values())}).to_parquet(CACHE)
        print(f"[probe_16] extracted + cached: {len(comp_text)}")

    # embed complement texts with BGE (free)
    from sentence_transformers import SentenceTransformer
    ext_ids = [s for s in seeds if s in comp_text]
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:1")
    model.max_seq_length = 512
    ext_emb_arr = model.encode([comp_text[s] for s in ext_ids], batch_size=128,
                               normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    ext_emb = {s: ext_emb_arr[i] for i, s in enumerate(ext_ids)}

    # eval pairs whose seed has external knowledge
    eval_pairs = [(s, c) for s, c in pairs if s in ext_emb][:n_pairs]
    print(f"[probe_16] eval pairs with external knowledge: {len(eval_pairs)}")
    scorers = ["popularity", "L1_cos_sim", "external_knowledge"]
    hr = {x: [] for x in scorers}
    rr = {x: [] for x in scorers}
    for seed, comp in eval_pairs:
        s_pg = pg.get(seed)
        cand_pool = [pop_ids[k] for k in np.where(pool_pg != s_pg)[0] if pop_ids[k] != comp][:N_DISTRACT]
        cand = [comp] + cand_pool
        cidx = np.array([id_to_idx[a] for a in cand])
        cand_l1 = l1[cidx]
        sc = {
            "popularity": np.array([pop_count.get(a, 0) for a in cand], float),
            "L1_cos_sim": cand_l1 @ l1[id_to_idx[seed]],          # product similarity
            "external_knowledge": cand_l1 @ ext_emb[seed],        # LLM styling-knowledge -> complement
        }
        for x in scorers:
            h, m = _ap_hr(np.argsort(-sc[x]), cand, comp)
            hr[x].append(h)
            rr[x].append(m)

    res = {x: {"hr_at_12": float(np.mean(hr[x])), "mrr": float(np.mean(rr[x]))} for x in scorers}
    cmp = bootstrap_delta(np.array(hr["L1_cos_sim"]), np.array(hr["external_knowledge"]), None, 1000, 16)
    go = res["external_knowledge"]["hr_at_12"] > res["L1_cos_sim"]["hr_at_12"] and cmp["ci_lo"] > 0
    verdict = ("GO — LLM external styling knowledge beats product-similarity for complementarity "
               "(KAR open-world-knowledge premise realized)"
               if go else
               "NO-GO — external knowledge does not beat L1 product-similarity")

    out = {"probe": "probe_16_external_knowledge", "n_seeds_extracted": len(ext_ids),
           "n_eval_pairs": len(eval_pairs), "scorers": res,
           "external_vs_L1": {"delta": cmp["delta"], "rel": cmp["rel_gain"], "ci": [cmp["ci_lo"], cmp["ci_hi"]]},
           "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_16_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 70)
    print("  PROBE 16 — external LLM styling knowledge vs product-similarity")
    print("=" * 70)
    print(f"  seeds={len(ext_ids)}  eval_pairs={len(eval_pairs)}  distractors={N_DISTRACT}")
    for x in scorers:
        print(f"  {x:18s} HR@12={res[x]['hr_at_12']:.4f}  MRR={res[x]['mrr']:.4f}")
    print(f"  external vs L1: delta={cmp['delta']:+.5f} rel={cmp['rel_gain']*100:+.1f}% "
          f"CI=[{cmp['ci_lo']:+.5f},{cmp['ci_hi']:+.5f}]")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)
    # sample external knowledge for inspection
    if ext_ids:
        s0 = ext_ids[0]
        print(f"\n  sample seed {s0} ({meta[s0]['product_type_name']}, {meta[s0]['colour_group_name']}):")
        print(f"    external knowledge: {comp_text[s0][:240]}")


if __name__ == "__main__":
    main()
