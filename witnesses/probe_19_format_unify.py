"""PROBE 19 — does UNIFYING the knowledge FORMAT (before encoding) beat post-hoc fusion?

Reviewer's design point: probe_17's frozen NO-GO came from a TEXT-GENRE mismatch — external
knowledge is prose ("pairs with high-waisted trousers...") while L1 is an attribute list
("Type: Trousers; Color: Beige; ..."). BGE puts different genres in different sub-spaces, so
raw cosine fails and learned fusion only partly repairs it (probe_18). A cleaner design:
UNIFY the format at the SOURCE — render external knowledge in the SAME attribute-list template
as L1, encode, then the spaces are coherent from the start (no genre confound to bridge).

This probe re-extracts external knowledge as STRUCTURED complement attributes (product_type /
colour / material / style), renders them in L1's attribute template, BGE-encodes, and asks:
does format-unified external knowledge flip the FROZEN user-level discovery NO-GO (probe_17:
external −60% vs L1) — i.e. was the failure a format artifact fixable at the source?

EXTRACTION / COST
    Structured re-extraction of the cached seed items (GPT-4.1-nano, ~$0.5), cached to
    data/knowledge/external/complement_structured.parquet (no re-spend on re-run).

THE RESULT (boxed; probe_19_result.json)
    +----------------------------------------------------------------------------+
    | frozen discovery_map@12: L1 | ext_prose (probe_17) | ext_UNIFIED (this)     |
    +----------------------------------------------------------------------------+

VERDICT
    FORMAT MATTERS iff ext_unified >> ext_prose (the genre mismatch was the killer);
    FROZEN REVIVE iff ext_unified >= L1 (unification alone fixes it, no heavy fusion needed).

Usage:
    uv run python witnesses/probe_19_format_unify.py [n_users]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_16_external_knowledge as p16  # noqa: E402

from src.evaluation.cohorts import discovery_map  # noqa: E402
from witnesses._probe_common import OUT_DIR, bootstrap_delta, load_variants  # noqa: E402

TRAIN_TXN = Path("data/processed/train_transactions.parquet")
ARTICLES = Path("data/processed/articles.parquet")
PROSE_CACHE = p16.CACHE
STRUCT_CACHE = Path("data/knowledge/external/complement_structured.parquet")
MODEL = "gpt-4.1-nano"
K = 12
POOL = 5000
MIN_COVERED = 3
SEED = 42

SYS_STRUCT = (
    "You are a professional fashion stylist. Given ONE product a customer bought, output the "
    "complementary items (different categories) that complete an outfit with it, using external "
    "styling knowledge. Return 1-3 complement items as STRUCTURED attributes (not prose)."
)
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "complements": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "product_type": {"type": "string"},
                    "colour": {"type": "string"},
                    "material": {"type": "string"},
                    "style": {"type": "string"},
                },
                "required": ["product_type", "colour", "material", "style"],
            },
        }
    },
    "required": ["complements"],
}


def _render(comps: list[dict]) -> str:
    """Render structured complements in L1's attribute-list genre."""
    blocks = []
    for c in comps:
        blocks.append(f"Type: {c.get('product_type','')}; Color: {c.get('colour','')}; "
                      f"[Product] Material: {c.get('material','')}; Style: {c.get('style','')}")
    return " . ".join(blocks)


async def _extract_struct(seeds_meta: list[dict]) -> dict[str, str]:
    import openai
    client = openai.AsyncOpenAI()
    sem = asyncio.Semaphore(24)

    async def one(m: dict):
        desc = (f"Product type: {m.get('product_type_name')}; Color: {m.get('colour_group_name')}; "
                f"Section: {m.get('section_name')}; Group: {m.get('product_group_name')}. "
                f"{str(m.get('detail_desc') or '')[:200]}")
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.responses.create(
                        model=MODEL,
                        input=[{"role": "system", "content": SYS_STRUCT},
                               {"role": "user", "content": f"The customer bought: {desc}"}],
                        text={"format": {"type": "json_schema", "name": "complements",
                                         "schema": SCHEMA, "strict": True}},
                    )
                    comps = json.loads(r.output_text).get("complements", [])
                    return m["article_id"], _render(comps)
                except Exception:  # noqa: BLE001
                    if attempt == 3:
                        return m["article_id"], ""
                    await asyncio.sleep(2 ** attempt)
        return m["article_id"], ""

    out = await asyncio.gather(*[one(m) for m in seeds_meta])
    return {a: t for a, t in out if t}


def _encode(texts: list[str], dev: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev)
    enc.max_seq_length = 512
    return enc.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def main() -> None:
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    import torch
    dev = "cuda:1" if torch.cuda.is_available() else "cpu"

    canon_ids, V = load_variants(["L1"])
    l1 = V["L1"].astype(np.float32)
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    prose = pd.read_parquet(PROSE_CACHE)
    prose_text = dict(zip(prose["article_id"].astype(str), prose["complement_text"]))
    seed_ids = [a for a in prose_text if a in id_to_idx]

    art = pd.read_parquet(ARTICLES, columns=["article_id", "product_type_name", "colour_group_name",
                                             "section_name", "product_group_name", "detail_desc"])
    art["article_id"] = art["article_id"].astype(str)
    meta = art.set_index("article_id").to_dict("index")

    # structured re-extraction in L1 format (cached)
    if STRUCT_CACHE.exists():
        sc = pd.read_parquet(STRUCT_CACHE)
        struct_text = dict(zip(sc["article_id"].astype(str), sc["unified_text"]))
        print(f"[probe_19] loaded structured cache: {len(struct_text)}")
    else:
        p16._load_env_key()
        assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not found"
        sm = [{"article_id": a, **meta[a]} for a in seed_ids if a in meta]
        print(f"[probe_19] structured re-extraction (L1 format) for {len(sm)} items...")
        struct_text = asyncio.run(_extract_struct(sm))
        STRUCT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"article_id": list(struct_text), "unified_text": list(struct_text.values())}).to_parquet(STRUCT_CACHE)
        print(f"[probe_19] extracted + cached: {len(struct_text)}")

    ids = [a for a in seed_ids if a in struct_text]
    prose_emb = {a: e for a, e in zip(ids, _encode([prose_text[a] for a in ids], dev))}
    uni_emb = {a: e for a, e in zip(ids, _encode([struct_text[a] for a in ids], dev))}
    cov_set = set(ids)
    print(f"[probe_19] sample unified text: {struct_text[ids[0]][:160]}")

    import duckdb
    con = duckdb.connect()
    pop = con.execute(
        f"SELECT article_id FROM read_parquet('{TRAIN_TXN}') WHERE t_dat >= DATE '2020-05-01' "
        f"GROUP BY article_id ORDER BY count(*) DESC LIMIT {POOL}"
    ).fetchall()
    pool_ids = [str(a) for (a,) in pop if str(a) in id_to_idx]
    pool_l1 = np.stack([l1[id_to_idx[a]] for a in pool_ids])

    gt = {str(k): [str(x) for x in v] for k, v in json.loads((Path('data/processed/immediate_ground_truth.json')).read_text()).items()}
    con.register("gt_users", pd.DataFrame({"customer_id": list(gt)}))
    hist_rows = con.execute(
        f"SELECT t.customer_id, list(DISTINCT t.article_id) FROM read_parquet('{TRAIN_TXN}') t "
        f"JOIN gt_users g ON t.customer_id=g.customer_id GROUP BY t.customer_id"
    ).fetchall()
    train_history = {str(cid): set(str(a) for a in items) for cid, items in hist_rows}

    rng = np.random.default_rng(SEED)
    eligible = [u for u, h in train_history.items() if u in gt and len(h & cov_set) >= MIN_COVERED]
    rng.shuffle(eligible)
    eligible = eligible[:n_users]
    print(f"[probe_19] eligible={len(eligible)}")

    def profile_preds(emb_src, use_l1=False):
        preds = {}
        for u in eligible:
            cov = [h for h in train_history[u] if h in cov_set]
            if use_l1:
                p = l1[[id_to_idx[h] for h in cov]].mean(0)
            else:
                p = np.stack([emb_src[h] for h in cov]).mean(0)
            p = p / (np.linalg.norm(p) + 1e-9)
            s = pool_l1 @ p
            owned = np.array([a in train_history[u] for a in pool_ids])
            s = np.where(owned, -np.inf, s)
            order = np.argpartition(-s, K)[:K]
            preds[u] = [pool_ids[i] for i in order[np.argsort(-s[order])]]
        return preds

    sub_gt = {u: gt[u] for u in eligible}
    sub_hist = {u: train_history[u] for u in eligible}
    res = {}
    for name, (src, ul) in {"L1_profile": (None, True), "ext_prose": (prose_emb, False),
                            "ext_unified": (uni_emb, False)}.items():
        d = discovery_map(profile_preds(src, ul), sub_gt, sub_hist, k=K)
        res[name] = {"discovery_map": d.map_at_k, "discovery_hr": d.hr_at_k}

    # per-user AP for CI (unified vs prose, unified vs L1)
    from src.evaluation.metrics import compute_map_at_k
    pr = {n: profile_preds(s, u) for n, (s, u) in
          {"L1_profile": (None, True), "ext_prose": (prose_emb, False), "ext_unified": (uni_emb, False)}.items()}

    def ap(pred):
        out = []
        for u in eligible:
            ng = [g for g in gt[u] if g not in train_history[u]]
            if ng:
                out.append(compute_map_at_k({u: pred[u]}, {u: ng}, K))
        return np.array(out)
    ap_l1, ap_pr, ap_un = ap(pr["L1_profile"]), ap(pr["ext_prose"]), ap(pr["ext_unified"])
    uni_vs_prose = bootstrap_delta(ap_pr, ap_un, None, 1000, 16)
    uni_vs_l1 = bootstrap_delta(ap_l1, ap_un, None, 1000, 16)

    fmt_matters = res["ext_unified"]["discovery_map"] > res["ext_prose"]["discovery_map"] and uni_vs_prose["ci_lo"] > 0
    frozen_revive = res["ext_unified"]["discovery_map"] >= res["L1_profile"]["discovery_map"]
    verdict = (("FORMAT MATTERS + FROZEN REVIVE — unified-format external >= L1 with no learned fusion: "
                "the frozen NO-GO was a genre artifact; unify-at-source fixes it"
                if frozen_revive else
                "FORMAT MATTERS (partial) — unified >> prose but still < L1 frozen; format helps, "
                "learned fusion (probe_18) still adds")
               if fmt_matters else
               "FORMAT NEUTRAL — unifying format does not beat prose; genre was not the (main) cause")

    out = {"probe": "probe_19_format_unify", "n_eligible": len(eligible), "n_extracted": len(ids),
           "frozen_discovery_map": res,
           "unified_vs_prose": {"delta": uni_vs_prose["delta"], "rel": uni_vs_prose["rel_gain"],
                                "ci": [uni_vs_prose["ci_lo"], uni_vs_prose["ci_hi"]]},
           "unified_vs_L1": {"delta": uni_vs_l1["delta"], "rel": uni_vs_l1["rel_gain"],
                             "ci": [uni_vs_l1["ci_lo"], uni_vs_l1["ci_hi"]]},
           "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_19_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 78)
    print("  PROBE 19 — format-unified external knowledge (frozen, user-level discovery)")
    print("=" * 78)
    print(f"  eligible={len(eligible)}  extracted={len(ids)}")
    for n in ["L1_profile", "ext_prose", "ext_unified"]:
        print(f"  {n:14s} frozen discovery_map@12 = {res[n]['discovery_map']:.5f}")
    print(f"  unified vs prose: {uni_vs_prose['rel_gain']*100:+.1f}% CI=[{uni_vs_prose['ci_lo']:+.5f},{uni_vs_prose['ci_hi']:+.5f}]")
    print(f"  unified vs L1:    {uni_vs_l1['rel_gain']*100:+.1f}% CI=[{uni_vs_l1['ci_lo']:+.5f},{uni_vs_l1['ci_hi']:+.5f}]")
    print(f"  VERDICT: {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
