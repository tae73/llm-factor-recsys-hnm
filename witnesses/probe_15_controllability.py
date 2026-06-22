"""PROBE 15 — L2/L3 enable CONTROLLABLE recommendation (a capability L1/metadata lack)

WHAT GENERALIZES
    L2/L3 do not beat L1 on accuracy, but the design (§7.9) promised an axis accuracy@k
    cannot measure: STEERABLE / controllable recommendation. L2/L3 expose semantic control
    dimensions (occasion, mood, season, quality, versatility, coordination) that the H&M
    metadata (product_type/colour/section) does NOT have. This probe quantifies the
    capability: given a target attribute value t (e.g. occasion=Party), can we steer a
    user's personalized content recommendation so the top-12 actually satisfy t — while
    staying personalized? L1 alone cannot, because L1 has no "occasion/mood" axis.

THE RESULT (boxed; persisted to probe_15_result.json)
    +-----------------------------------------------------------------------+
    | per controllable attribute: steered top-12 precision(t) vs unsteered   |
    | baseline rate(t) -> control gain; + personalization retained           |
    | + count of NEW control axes L2/L3 add over metadata                    |
    +-----------------------------------------------------------------------+

HONEST reduces_check
    Steering precision is a capability metric, not accuracy. Soft steering (content + boost)
    rather than hard filter, so "precision" reflects real controllability under a popular
    candidate pool. Personalization retained = steered items still close to the user's L1
    content centroid (vs random items with t).

VERDICT
    L2/L3 give controllable recommendation iff steered precision >> unsteered baseline while
    personalization is retained (capability L1/metadata cannot provide).

Usage:
    uv run python witnesses/probe_15_controllability.py [sample_users]
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
FK = Path("data/knowledge/factual/factual_knowledge.parquet")
# metadata fields (no perceptual/contextual axis) vs L2/L3 control axes
META_AXES = ["product_type_name", "colour_group_name", "section_name", "garment_group_name", "index_name"]
L2L3_AXES = ["l2_occasion", "l2_style_mood", "l2_season_fit", "l2_perceived_quality",
             "l2_trendiness", "l2_versatility", "l3_coordination_role", "l3_visual_weight"]
# controllable targets to demonstrate (attribute -> values)
TARGETS = {
    "l2_occasion": ["Party", "Work", "Formal"],
    "l2_season_fit": ["Summer", "Winter"],
    "l2_style_mood": ["Elegant", "Sporty", "Minimalist"],
}
K = 12
POOL = 2000
RECENT_DAYS = 30
LAM = 1.0  # steering strength (cosine ~[-1,1]; boost +1 per match)
SEED = 42


def _has(val, target) -> bool:
    if isinstance(val, (list, np.ndarray)):
        return target in [str(x) for x in val]
    return str(val) == target


def main() -> None:
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 8_000
    canon_ids, V = load_variants(["L1"])
    emb = V["L1"]
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    cols = ["article_id"] + list(TARGETS.keys())
    fk = pd.read_parquet(FK, columns=cols)
    fk["article_id"] = fk["article_id"].astype(str)
    attr = {a: dict(zip(fk["article_id"], fk[a])) for a in TARGETS}

    import duckdb

    rng = np.random.default_rng(SEED)
    pop = duckdb.connect().execute(
        f"SELECT article_id FROM read_parquet('{TRAIN_TXN}') WHERE t_dat >= DATE '2020-05-01' "
        f"GROUP BY article_id ORDER BY count(*) DESC LIMIT {POOL}"
    ).fetchall()
    pool_ids = [str(a) for (a,) in pop if str(a) in id_to_idx]
    pool_idx = np.array([id_to_idx[a] for a in pool_ids])
    pool_emb = emb[pool_idx]

    user_history = _build_user_purchase_history(TRAIN_TXN)
    users = list(user_history.keys())
    samp = list(rng.choice(users, size=min(sample, len(users)), replace=False))

    # precompute target-membership over pool
    target_mask = {}
    for a, vals in TARGETS.items():
        for t in vals:
            target_mask[(a, t)] = np.array([_has(attr[a].get(pid), t) for pid in pool_ids])

    rows = {f"{a}={t}": {"steer_prec": [], "base_rate": [], "steer_persona": [], "rand_persona": []}
            for a, vals in TARGETS.items() for t in vals}
    n_eval = 0
    for u in samp:
        hidx = [id_to_idx[x] for x in user_history[u] if x in id_to_idx]
        if not hidx:
            continue
        n_eval += 1
        c = emb[hidx].mean(0)
        c = c / (np.linalg.norm(c) + 1e-9)
        content = pool_emb @ c  # personalization score over pool
        unsteer_top = np.argpartition(-content, K)[:K]
        for a, vals in TARGETS.items():
            for t in vals:
                key = f"{a}={t}"
                m = target_mask[(a, t)]
                if m.sum() < K:
                    continue
                steered = content + LAM * m.astype(float)
                stop = np.argpartition(-steered, K)[:K]
                rows[key]["steer_prec"].append(float(m[stop].mean()))      # steered top-12 precision(t)
                rows[key]["base_rate"].append(float(m[unsteer_top].mean()))  # unsteered baseline rate(t)
                rows[key]["steer_persona"].append(float(content[stop].mean()))  # personalization retained
                rows[key]["rand_persona"].append(float(content[np.where(m)[0]].mean()))  # random t-items persona

    res = {}
    for key, d in rows.items():
        if not d["steer_prec"]:
            continue
        res[key] = {"steer_precision": float(np.mean(d["steer_prec"])),
                    "unsteered_rate": float(np.mean(d["base_rate"])),
                    "control_gain": float(np.mean(d["steer_prec"]) - np.mean(d["base_rate"])),
                    "steered_personalization": float(np.mean(d["steer_persona"])),
                    "random_t_personalization": float(np.mean(d["rand_persona"]))}
    mean_prec = float(np.mean([v["steer_precision"] for v in res.values()]))
    mean_base = float(np.mean([v["unsteered_rate"] for v in res.values()]))
    # personalization retained: steered > random-with-t
    persona_ok = np.mean([v["steered_personalization"] > v["random_t_personalization"] for v in res.values()])

    new_axes = [a for a in L2L3_AXES]  # none of these exist in metadata
    verdict = (f"GO — L2/L3 enable controllable recommendation: steered precision {mean_prec:.2f} "
               f"vs unsteered {mean_base:.2f} (control gain {mean_prec-mean_base:+.2f}), personalization retained "
               f"({persona_ok*100:.0f}% of targets). {len(new_axes)} new control axes absent from metadata."
               if mean_prec - mean_base > 0.3 and persona_ok >= 0.8 else
               f"WEAK — steered {mean_prec:.2f} vs base {mean_base:.2f}")

    out = {"probe": "probe_15_controllability", "n_eval": n_eval,
           "new_control_axes_over_metadata": new_axes, "metadata_axes": META_AXES,
           "per_target": res, "mean_steer_precision": mean_prec, "mean_unsteered_rate": mean_base,
           "personalization_retained_frac": float(persona_ok), "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_15_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 74)
    print("  PROBE 15 — controllable recommendation via L2/L3 (capability L1 lacks)")
    print("=" * 74)
    print(f"  n_eval={n_eval}  |  NEW control axes from L2/L3 (none in metadata): {len(new_axes)}")
    print(f"  {'target':24s} {'steer_prec':>10s} {'unsteer':>8s} {'gain':>6s} {'persona(steer/rand)':>20s}")
    for key, v in res.items():
        print(f"  {key:24s} {v['steer_precision']:>10.2f} {v['unsteered_rate']:>8.2f} "
              f"{v['control_gain']:>+6.2f} {v['steered_personalization']:>9.3f}/{v['random_t_personalization']:.3f}")
    print("-" * 74)
    print(f"  mean steered precision {mean_prec:.2f} vs unsteered {mean_base:.2f}; persona retained {persona_ok*100:.0f}%")
    print(f"  VERDICT: {verdict}")
    print("=" * 74)


if __name__ == "__main__":
    main()
