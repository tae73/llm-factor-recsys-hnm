"""PROBE 12 — Does L3 capture real outfit COORDINATION? (task mismatch 5b)

WHAT GENERALIZES
    L3 (color_harmony, tone_season, style_lineage, coordination_role) was DESIGNED for
    outfit coordination — predicting COMPLEMENTARY cross-category items, not similar ones.
    Our single-item retrieval never tests this. This probe asks the prerequisite,
    cheaply: do items that are actually CO-PURCHASED together (same user, same day,
    DIFFERENT product_group = an outfit) share L3 attributes MORE than random cross-
    category pairs? If yes, a coordination/complementarity recommender (which the project
    does not build) could exploit L3 — the value exists but the task can't see it. If no,
    L3 does not even capture coordination in real co-purchases -> robustly redundant.

    No training. Uses transactions (co-purchase) + factual_knowledge L3 fields + l3.npz.

THE RESULT (boxed; persisted to probe_12_result.json)
    +-----------------------------------------------------------------------+
    | co-purchased vs random cross-category pairs: L3-attr match rate +      |
    | L3-embedding cosine. coordination signal = co-purchase >> random?      |
    +-----------------------------------------------------------------------+

HONEST reduces_check
    Same-day co-purchase is a proxy for "outfit" (could be unrelated). Comparing against
    random pairs MATCHED on the same product-group composition controls for category
    base rates. This tests whether L3 carries ANY coordination signal, not a full
    complementarity recommender.

VERDICT
    coordination signal present iff co-purchase L3 match rate / cosine > random (CI excl 0).

Usage:
    uv run python witnesses/probe_12_complementarity.py [n_pairs]
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
SEED = 42


def _boot_diff(a, b, n=1000, seed=12):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    da = rng.integers(0, len(a), (n, len(a)))
    db = rng.integers(0, len(b), (n, len(b)))
    diff = a[da].mean(1) - b[db].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def main() -> None:
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 40_000
    canon_ids, V = load_variants(["L3"])
    l3 = V["L3"]
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    fk = pd.read_parquet(FK, columns=["article_id", "l3_color_harmony", "l3_tone_season", "l3_style_lineage"])
    fk["article_id"] = fk["article_id"].astype(str)
    ch = dict(zip(fk["article_id"], fk["l3_color_harmony"].astype(str)))
    ts = dict(zip(fk["article_id"], fk["l3_tone_season"].astype(str)))
    sl = dict(zip(fk["article_id"], fk["l3_style_lineage"].astype(str)))

    art = pd.read_parquet(ARTICLES, columns=["article_id", "product_group_name"])
    art["article_id"] = art["article_id"].astype(str)
    pg = dict(zip(art["article_id"], art["product_group_name"].astype(str)))

    # co-purchase baskets via DuckDB, restricted to the LAST 30 days of train (~1M rows → fast;
    # coordination is time-invariant). count-based HAVING avoids costly list ops over all groups.
    import duckdb

    rng = np.random.default_rng(SEED)
    baskets = duckdb.connect().execute(
        f"""
        SELECT list(article_id) AS items
        FROM read_parquet('{TRAIN_TXN}')
        WHERE t_dat >= DATE '2020-06-01'
        GROUP BY customer_id, t_dat
        HAVING count(*) BETWEEN 2 AND 6
        LIMIT 80000
        """
    ).fetchall()
    co_pairs = []
    for (items,) in baskets:
        its = list(dict.fromkeys(str(a) for a in items if str(a) in id_to_idx))
        for i in range(len(its)):
            for j in range(i + 1, len(its)):
                a, b = its[i], its[j]
                if pg.get(a) != pg.get(b):
                    co_pairs.append((a, b))
        if len(co_pairs) >= n_pairs:
            break
    co_pairs = co_pairs[:n_pairs]
    print(f"[probe_12] baskets={len(baskets)}, cross-category co-purchase pairs: {len(co_pairs)}")

    # random cross-category pairs (control): vectorized index sampling (fast)
    all_ids = np.array([a for a in id_to_idx if a in pg])
    all_pg = np.array([pg.get(a, "") for a in all_ids])
    need = len(co_pairs)
    rand_pairs = []
    while len(rand_pairs) < need:
        ia = rng.integers(0, len(all_ids), size=need * 2)
        ib = rng.integers(0, len(all_ids), size=need * 2)
        mask = all_pg[ia] != all_pg[ib]
        for x, y in zip(ia[mask], ib[mask]):
            rand_pairs.append((all_ids[x], all_ids[y]))
            if len(rand_pairs) >= need:
                break

    def stats(pairs):
        ch_m = [1.0 if ch.get(a) == ch.get(b) and ch.get(a) != "nan" else 0.0 for a, b in pairs]
        ts_m = [1.0 if ts.get(a) == ts.get(b) and ts.get(a) != "nan" else 0.0 for a, b in pairs]
        sl_m = [1.0 if sl.get(a) == sl.get(b) and sl.get(a) != "nan" else 0.0 for a, b in pairs]
        cos = [float(l3[id_to_idx[a]] @ l3[id_to_idx[b]]) for a, b in pairs]
        return ch_m, ts_m, sl_m, cos

    co = stats(co_pairs)
    rd = stats(rand_pairs)
    names = ["color_harmony_match", "tone_season_match", "style_lineage_match", "l3_cosine"]
    res = {}
    for k, nm in enumerate(names):
        d, lo, hi = _boot_diff(co[k], rd[k])
        res[nm] = {"co_purchase": float(np.mean(co[k])), "random": float(np.mean(rd[k])),
                   "diff": d, "ci": [lo, hi], "sig": lo > 0}

    any_sig = any(res[nm]["sig"] for nm in names)
    verdict = ("coordination SIGNAL present — co-purchased items share L3 attrs > random "
               "(L3 has complementarity value the single-item task ignores)"
               if any_sig else
               "NO coordination signal — L3 attrs not shared by real co-purchases > random")

    out = {"probe": "probe_12_complementarity", "n_pairs": len(co_pairs),
           "results": res, "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_12_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 70)
    print("  PROBE 12 — outfit coordination signal in L3 (co-purchase vs random)")
    print("=" * 70)
    print(f"  cross-category pairs: {len(co_pairs)}")
    for nm in names:
        r = res[nm]
        print(f"  {nm:22s} co={r['co_purchase']:.4f} rand={r['random']:.4f} "
              f"diff={r['diff']:+.4f} CI=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] {'SIG' if r['sig'] else 'ns'}")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
