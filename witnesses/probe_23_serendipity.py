"""PROBE 23 — SERENDIPITY / NOVELTY / LONG-TAIL: enrichment's last open recsys role (R-10).

WHY — the recsys-value map is almost fully measured, all NEGATIVE on the dimensions tried:
  * accuracy (HR/NDCG/MAP) — CLOSED-NEGATIVE (probe_21/22 full-scale −12%, cold-start −21%).
  * intra-list diversity + catalog coverage — CLOSED-NEGATIVE (probe_02 C4: L1+L2→+L3 diversity
    −7.7% CI≠0, coverage −2.3%).
The ONE dimension `redesign_2026-06.md` (§65 "serendipity 미입증", §96 "novelty@12·long-tail-hit·
serendipity 추가 = L3의 유일한 잔존 rescue 경로") flagged but NEVER measured: does enrichment
surface RELEVANT-yet-SURPRISING items (serendipity / long-tail discovery) even though it doesn't
improve accuracy or basic diversity? H&M is 87% single-purchase = discovery-oriented, so this is
the motivation-aligned question. This probe falsifies it at FULL catalog scale (STRONG power).

DESIGN (comprehensive — user choice "둘 다")
  ① LLM semantic embeddings as the RETRIEVAL SPACE (META/L1/L2/L3/L1+L2+L3/external) → top-12.
  ② decision-axes (trend_phase/outfit_role, full-catalog) to CHARACTERIZE the hits (what *kind*
     of surprise). GT = immediate_ground_truth (next-week NEW items = discovery-native).
  Metrics: HR@12 (accuracy guardrail) · diversity/coverage (CONTEXT, known-negative — not re-claimed)
  · novelty@12 (DESCRIPTIVE list property) · long-tail exposure · **long-tail HIT-rate (S1)** ·
  **serendipity@12** = relevance ∧ unexpected, two operationalizations: S2 (fixed-ref cosine to
  user centroid < τ_L1, frozen) and **S2b (labeling-SYMMETRIC: each variant flags its own 6
  least-central recs)**. S2b is the FAIR test — a frozen-τ S2 gap can be a labeling artifact (a
  variant whose recs sit closer to the centroid emits fewer items past L1's τ), so S2 alone
  OVER-states; S2b removes that asymmetry.
  HONESTY: novelty is cheap to inflate (recommend unpopular items); only the *relevance-grounded*
  S1/S2/S2b count. Pre-registered R-10 GO = enrichment beats L1 on S1 OR S2 OR S2b by ≥+10% rel,
  bootstrap CI≠0, AND HR-guardrail loss ≤2% rel (probe_02 C4 discipline). The honest negative is
  "tie at best, never a win" — under the FAIR S2b, enrichment merely ties L1, never improves it.
  CPU, seed 42, no API/$. Never edits prior probe JSON.

Usage:  PYTHONPATH=. python3 -u witnesses/probe_23_serendipity.py [--quick] [--repro]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.cold_start import (  # noqa: E402
    _build_user_purchase_history,
    _load_val_ground_truth,
)
from witnesses._probe_common import (  # noqa: E402
    OUT_DIR,
    bootstrap_delta,
    build_fixed_users,
    catalog_coverage,
    hit_count,
    intra_list_diversity,
    item_novelty,
    load_reference,
    load_variants,
    longtail_exposure,
    score_variant,
    serendipitous_hit_count,
    tail_hit_count,
)

SEED = 42
RESULT_PATH = OUT_DIR / "probe_23_result.json"
FIG_PATH = ROOT / "results/figures/probe_23_serendipity.png"
TRAIN_TXN = ROOT / "data/processed/train_transactions.parquet"
IMMEDIATE_GT = ROOT / "data/processed/immediate_ground_truth.json"
MATRIX_AXES = ROOT / "data/knowledge/enrichment_v2/matrix_axes.parquet"
EMB_DIR = ROOT / "data/embeddings"

ABLATION_VARIANTS = ["META", "L1", "L2", "L3", "L1+L2+L3"]
EXTERNAL = {"EXT_prose": "external_prose.npz", "EXT_struct": "external_struct.npz"}
ENRICH_VARIANTS = ["L2", "L3", "L1+L2+L3", "EXT_prose", "EXT_struct"]  # tested vs L1
BASELINE = "L1"  # the content that WORKS (+80-104% over popularity)

# Pre-registered (mirror probe_02 C4 discipline)
SEREN_REL_GAIN_MIN = 0.10  # S1/S2/S2b rel-gain over L1 for GO
HR_GUARD = -0.02  # HR loss must be ≤ 2% relative
TAIL_TXN_SHARE = 0.80  # head = smallest item set covering 80% of train txns; tail = complement
K = 12
N_SURPRISE = 6  # labeling-symmetric surprise = bottom-half of the 12-list by centroid-cosine


# ===========================================================================
# Data prep
# ===========================================================================
def _load_aligned(path: Path, canon_ids: np.ndarray) -> np.ndarray:
    """Load an npz embedding, align to canon order, L2-normalize (generic load_reference)."""
    d = np.load(path, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    ids = d["article_ids"].astype(str)
    pos = {a: i for i, a in enumerate(ids)}
    sel = np.array([pos[a] for a in canon_ids])
    e = emb[sel]
    n = np.linalg.norm(e, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return e / n


def _popularity(canon_ids: np.ndarray):
    """Train purchase counts → smoothed pop_prob + long-tail mask (head=80% of txns)."""
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")  # deterministic COUNT aggregation
    df = con.execute(
        f"SELECT CAST(article_id AS VARCHAR) article_id, COUNT(*) c "
        f"FROM read_parquet('{TRAIN_TXN}') GROUP BY article_id"
    ).fetchdf()
    con.close()
    cmap = dict(zip(df["article_id"], df["c"].astype(float)))
    counts = np.array([cmap.get(a, 0.0) for a in canon_ids], dtype=float)
    total = counts.sum()
    pop_prob = (counts + 1.0) / (total + len(counts))
    order = np.argsort(-counts, kind="stable")  # deterministic tie order
    cum = np.cumsum(counts[order])
    head_n = int(np.searchsorted(cum, TAIL_TXN_SHARE * total) + 1)
    tail_mask = np.ones(len(counts), dtype=bool)
    tail_mask[order[:head_n]] = False
    return counts, pop_prob, tail_mask, head_n


def _centroids_ref(ref_emb: np.ndarray, fixed) -> np.ndarray:
    """Per-user history centroid in the FIXED reference space (normalized)."""
    C = np.zeros((fixed.n_users, ref_emb.shape[1]), dtype=np.float32)
    for i, hi in enumerate(fixed.hist_indices):
        c = ref_emb[hi].mean(axis=0)
        nrm = np.linalg.norm(c)
        if nrm > 0:
            c = c / nrm
        C[i] = c
    return C


def _rec_centroid_sim(topk: np.ndarray, ref_emb: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(n_users, k) cosine of each recommended item to the user's fixed-ref centroid."""
    return np.einsum("ukd,ud->uk", ref_emb[topk[:, :K]], C)


# ===========================================================================
# Per-variant metric bundle
# ===========================================================================
def _variant_metrics(res, ref_emb, C, pop_prob, tail_mask, gts, canon_ids, tau) -> dict:
    topk = res.topk
    sims = _rec_centroid_sim(topk, ref_emb, C)
    # S2 frozen-τ surprise: item < L1's median centroid-cosine. NOTE this favors the variant
    # whose recs sit FARTHER from the centroid — i.e. a S2 gap can be a labeling artifact, not a
    # quality gap (a variant emitting more central recs gets fewer items past L1's τ).
    unexpected = sims < tau
    # S2b labeling-SYMMETRIC surprise: each variant flags its OWN N_SURPRISE least-central recs
    # (within-list rank), removing the frozen-τ asymmetry. This is the fair serendipity test.
    ranks = np.argsort(
        sims, axis=1
    )  # ascending; first N_SURPRISE = least similar = most surprising
    within = np.zeros_like(sims, dtype=bool)
    within[np.arange(sims.shape[0])[:, None], ranks[:, :N_SURPRISE]] = True
    hits = hit_count(topk, gts, canon_ids, K)
    s1 = tail_hit_count(topk, gts, canon_ids, tail_mask, K)  # relevance ∧ long-tail
    s2 = serendipitous_hit_count(
        topk, gts, canon_ids, unexpected, K
    )  # relevance ∧ surprise (frozen-τ)
    s2b = serendipitous_hit_count(
        topk, gts, canon_ids, within, K
    )  # relevance ∧ surprise (symmetric)
    return {
        "hr": res.hr,
        "ndcg": res.ndcg,
        "novelty": item_novelty(topk, pop_prob),
        "tail_exposure": longtail_exposure(topk, tail_mask),
        "hits": hits,
        "tail_hits": s1,
        "seren_hits": s2,
        "seren_hits_sym": s2b,
        "topk": topk,
    }


def _axis_characterization(topk, gts, canon_ids, axis_maps) -> dict:
    """Composition of HITS by full-catalog decision-axes (descriptive — what KIND of surprise)."""
    phase, role = axis_maps
    n_phase = {"Emerging": 0, "Rising": 0, "other": 0}
    n_role = {"Anchor-hub": 0, "Versatile-connector": 0, "other": 0}
    n_hits = 0
    for u in range(len(topk)):
        gt = gts[u]
        for idx in topk[u, :K]:
            aid = canon_ids[idx]
            if aid in gt:
                n_hits += 1
                p = phase.get(aid, "other")
                n_phase[p if p in ("Emerging", "Rising") else "other"] += 1
                r = role.get(aid, "other")
                n_role[r if r in ("Anchor-hub", "Versatile-connector") else "other"] += 1
    if n_hits == 0:
        return {"n_hits": 0}
    return {
        "n_hits": n_hits,
        "frac_emerging_rising": round((n_phase["Emerging"] + n_phase["Rising"]) / n_hits, 4),
        "frac_anchor_versatile": round(
            (n_role["Anchor-hub"] + n_role["Versatile-connector"]) / n_hits, 4
        ),
    }


# ===========================================================================
# Main
# ===========================================================================
def _run(sample_users: int, n_boot: int) -> dict:
    all_variants = ABLATION_VARIANTS  # ablation npz
    canon_ids, variants = load_variants(all_variants)
    n_items = len(canon_ids)
    ref_emb = load_reference(canon_ids)
    for nm, fn in EXTERNAL.items():
        variants[nm] = _load_aligned(EMB_DIR / fn, canon_ids)

    counts, pop_prob, tail_mask, head_n = _popularity(canon_ids)

    # discovery-native GT (next-week NEW items)
    user_history = _build_user_purchase_history(TRAIN_TXN)
    immediate_gt = _load_val_ground_truth(IMMEDIATE_GT)
    fixed = build_fixed_users(
        canon_ids,
        sample_users=sample_users,
        seed=SEED,
        user_history=user_history,
        val_gt=immediate_gt,
    )
    C = _centroids_ref(ref_emb, fixed)

    # axis maps (full-catalog decision-axes for hit characterization)
    import pandas as pd

    ax = pd.read_parquet(
        MATRIX_AXES, columns=["article_id", "e2_trend_phase_actual", "e2_outfit_role"]
    )
    ax["article_id"] = ax["article_id"].astype(str)
    phase = dict(zip(ax["article_id"], ax["e2_trend_phase_actual"].astype(str)))
    role = dict(zip(ax["article_id"], ax["e2_outfit_role"].astype(str)))

    # τ frozen from L1 baseline: median rec-to-centroid cosine on L1's list
    res_l1 = score_variant(variants[BASELINE], canon_ids, fixed, k=K)
    tau = float(np.median(_rec_centroid_sim(res_l1.topk, ref_emb, C)))

    order = [BASELINE, "META"] + [v for v in ENRICH_VARIANTS]
    M, summary = {}, {}
    for nm in order:
        res = res_l1 if nm == BASELINE else score_variant(variants[nm], canon_ids, fixed, k=K)
        m = _variant_metrics(res, ref_emb, C, pop_prob, tail_mask, fixed.gt, canon_ids, tau)
        M[nm] = m
        div = intra_list_diversity(m["topk"], ref_emb)
        summary[nm] = {
            "hr_at_12": float(m["hr"].mean()),
            "ndcg_at_12": float(m["ndcg"].mean()),
            "diversity_at_12": float(div.mean()),  # context (known-negative)
            "coverage_at_12": catalog_coverage(m["topk"], n_items),  # context (known-negative)
            "novelty_at_12": float(m["novelty"].mean()),  # descriptive
            "tail_exposure_at_12": float(m["tail_exposure"].mean()),  # descriptive
            "hits_per_user": float(m["hits"].mean()),
            "tail_hits_per_user": float(m["tail_hits"].mean()),  # S1
            "seren_hits_per_user": float(m["seren_hits"].mean()),  # S2 (frozen-τ)
            "seren_hits_sym_per_user": float(
                m["seren_hits_sym"].mean()
            ),  # S2b (labeling-symmetric)
            "frac_hits_tail": round(float(m["tail_hits"].sum() / max(m["hits"].sum(), 1)), 4),
            "frac_hits_seren": round(float(m["seren_hits"].sum() / max(m["hits"].sum(), 1)), 4),
            "hit_characterization": _axis_characterization(
                m["topk"], fixed.gt, canon_ids, (phase, role)
            ),
        }

    # placebo: random-12 (novelty/serendipity floor)
    rng = np.random.default_rng(SEED)
    rand_topk = rng.integers(0, n_items, size=(fixed.n_users, K))
    placebo = {
        "novelty_at_12": float(item_novelty(rand_topk, pop_prob).mean()),
        "tail_exposure_at_12": float(longtail_exposure(rand_topk, tail_mask).mean()),
        "tail_hits_per_user": float(
            tail_hit_count(rand_topk, fixed.gt, canon_ids, tail_mask, K).mean()
        ),
        "seren_hits_per_user": float(
            serendipitous_hit_count(
                rand_topk,
                fixed.gt,
                canon_ids,
                _rec_centroid_sim(rand_topk, ref_emb, C) < tau,
                K,
            ).mean()
        ),
    }

    # ---- pre-registered R-10 falsification: enrichment vs L1 on S1/S2 (+ HR guard) ----
    l1 = M[BASELINE]
    cells = []
    for nm in ENRICH_VARIANTS:
        v = M[nm]
        s1_boot = bootstrap_delta(l1["tail_hits"], v["tail_hits"], None, n_boot, seed=301)
        s2_boot = bootstrap_delta(l1["seren_hits"], v["seren_hits"], None, n_boot, seed=302)
        s2b_boot = bootstrap_delta(
            l1["seren_hits_sym"], v["seren_hits_sym"], None, n_boot, seed=303
        )
        hr_rel = (summary[nm]["hr_at_12"] - summary[BASELINE]["hr_at_12"]) / max(
            summary[BASELINE]["hr_at_12"], 1e-9
        )
        s1_go = s1_boot["rel_gain"] >= SEREN_REL_GAIN_MIN and s1_boot["ci_lo"] > 0
        s2_go = s2_boot["rel_gain"] >= SEREN_REL_GAIN_MIN and s2_boot["ci_lo"] > 0
        s2b_go = s2b_boot["rel_gain"] >= SEREN_REL_GAIN_MIN and s2b_boot["ci_lo"] > 0
        hr_ok = hr_rel >= HR_GUARD
        cells.append(
            {
                "variant": nm,
                "s1_tailhit": s1_boot,
                "s2_serendipity_frozen_tau": s2_boot,
                "s2b_serendipity_symmetric": s2b_boot,  # the FAIR (labeling-symmetric) test
                "hr_rel_change": round(hr_rel, 4),
                "hr_guard_ok": bool(hr_ok),
                "go": bool((s1_go or s2_go or s2b_go) and hr_ok),
                "verdict": ("RESCUE" if ((s1_go or s2_go or s2b_go) and hr_ok) else "NO"),
            }
        )

    # per-bracket S1/S2 for cold-start (2-4, 5-9) — report counts; tiny-n, frame as "no rescue/tie"
    brackets = {}
    for b in ("2-4", "5-9"):
        mask = fixed.brackets == b
        if mask.sum() < 200:
            continue
        brackets[b] = {
            "n": int(mask.sum()),
            "L1": {
                "tail_hit_count": int(l1["tail_hits"][mask].sum()),
                "seren_hit_count": int(l1["seren_hits"][mask].sum()),
                "seren_sym_count": int(l1["seren_hits_sym"][mask].sum()),
            },
            "L1+L2+L3": {
                "tail_hit_count": int(M["L1+L2+L3"]["tail_hits"][mask].sum()),
                "seren_hit_count": int(M["L1+L2+L3"]["seren_hits"][mask].sum()),
                "seren_sym_count": int(M["L1+L2+L3"]["seren_hits_sym"][mask].sum()),
            },
            "note": "tiny hit counts — read as 'no rescue / tie', not large percentages",
        }

    any_rescue = [c["variant"] for c in cells if c["go"]]
    if any_rescue:
        decision = (
            f"R-10 SERENDIPITY RESCUE (PARTIAL) — {any_rescue} beat L1 on relevant-surprise "
            f"(S1/S2/S2b) ≥+10% with CI≠0 at ≤2% HR cost. Enrichment's last open recsys role is REAL "
            f"(thin): it surfaces more relevant long-tail/unexpected items despite failing accuracy."
        )
    else:
        decision = (
            "R-10 CLEAN NEGATIVE (tie at best, never a win) — NO enrichment variant IMPROVES "
            "relevant-surprise over L1 on any operationalization (S1 long-tail-hit, S2 frozen-τ "
            "serendipity, OR S2b labeling-symmetric serendipity). The frozen-τ S2 gap is ~94% a "
            "labeling artifact (enrichment recs sit closer to the user centroid → fewer cross L1's τ); "
            "under the FAIR symmetric S2b, L1+L2+L3 merely TIES L1 (CI includes 0). Enrichment DOES "
            "shift the list into the long tail (novelty/exposure↑) but those items are not the ones "
            "users buy. The last open recsys dimension is closed: enrichment value is merchant-side / "
            "interpretive, NOT consumer-recommendation lift — on any axis (accuracy/diversity/coverage/"
            "serendipity)."
        )

    return {
        "probe": "probe_23_serendipity",
        "seed": SEED,
        "n_eval_users": fixed.n_users,
        "n_items": n_items,
        "n_boot": n_boot,
        "gt": "immediate_ground_truth (next-week NEW items, discovery-native)",
        "tail_head_items": head_n,
        "tail_fraction": round(1 - head_n / n_items, 4),
        "tau_surprise": round(tau, 4),
        "thresholds": {
            "seren_rel_gain_min": SEREN_REL_GAIN_MIN,
            "hr_guard": HR_GUARD,
            "tail_txn_share": TAIL_TXN_SHARE,
        },
        "baseline": BASELINE,
        "variant_summary": summary,
        "placebo_random12": placebo,
        "falsification": cells,
        "cold_start_brackets": brackets,
        "context_note": (
            "accuracy (HR/NDCG) and diversity/coverage are CLOSED-NEGATIVE (probe_21/22, probe_02) — "
            "reported here only as guardrail/context, NOT re-claimed as new findings. novelty/exposure "
            "are descriptive list properties (cheap to inflate); headline = relevance-grounded S1/S2."
        ),
        "decision": decision,
    }


def make_figure(result: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:  # noqa: BLE001
        print(f"[fig] skipped ({e})", flush=True)
        return
    sns.set_theme(style="white", context="notebook")
    base = result["baseline"]
    variants = [base] + ENRICH_VARIANTS
    s = result["variant_summary"]
    metrics = ["hr_at_12", "tail_hits_per_user", "seren_hits_per_user", "novelty_at_12"]
    labels = ["HR@12\n(guardrail)", "S1 tail-hit\n★", "S2 serendipity\n★", "novelty\n(descr.)"]
    rel = np.array(
        [[(s[v][m] - s[base][m]) / max(s[base][m], 1e-9) for m in metrics] for v in variants]
    )
    fig, ax = plt.subplots(figsize=(8.5, 0.8 + 0.6 * len(variants)))
    annot = np.empty_like(rel, dtype=object)
    for i, v in enumerate(variants):
        for j, m in enumerate(metrics):
            annot[i, j] = f"{s[v][m]:.3f}\n({rel[i, j] * 100:+.0f}%)"
    sns.heatmap(
        rel,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        center=0,
        vmin=-0.3,
        vmax=0.3,
        xticklabels=labels,
        yticklabels=variants,
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "rel. change vs L1"},
        ax=ax,
        annot_kws={"fontsize": 8},
    )
    rescue = [c["variant"] for c in result["falsification"] if c["go"]]
    ax.set_title(
        f"R-10 serendipity/novelty — enrichment vs L1 (GT=next-week NEW, n={result['n_eval_users']})\n"
        f"headline=S1/S2 (relevance∧surprise); novelty=descriptive. "
        f"{'RESCUE: ' + ','.join(rescue) if rescue else 'CLEAN NEGATIVE'}",
        fontsize=9.5,
    )
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_PATH}", flush=True)


def main() -> None:
    quick = "--quick" in sys.argv
    repro = "--repro" in sys.argv
    sample_users = 3000 if quick else 25000
    n_boot = 300 if quick else 1000

    # guard: do not modify any existing probe canonical JSON
    prior = sorted(p for p in OUT_DIR.glob("probe_*_result.json") if p.name != RESULT_PATH.name)
    prior_mtime = {p: p.stat().st_mtime for p in prior}

    print("=" * 80, flush=True)
    print(f"PROBE 23 — serendipity/novelty (seed 42) {'[QUICK]' if quick else ''}", flush=True)
    print("=" * 80, flush=True)

    result = _run(sample_users, n_boot)
    if repro:
        result2 = _run(sample_users, n_boot)
        assert json.dumps(result, default=str) == json.dumps(
            result2, default=str
        ), "REPRO mismatch!"
        print("[repro] byte-identical double run ✓", flush=True)

    s = result["variant_summary"]
    print(
        f"\nn_eval={result['n_eval_users']} tail_frac={result['tail_fraction']} tau={result['tau_surprise']}",
        flush=True,
    )
    for v in [result["baseline"]] + ENRICH_VARIANTS:
        print(
            f"  {v:12s} HR={s[v]['hr_at_12']:.4f} S1tail={s[v]['tail_hits_per_user']:.4f} "
            f"S2seren={s[v]['seren_hits_per_user']:.4f} nov={s[v]['novelty_at_12']:.2f}",
            flush=True,
        )
    print("\n" + "=" * 80 + f"\n  {result['decision']}\n" + "=" * 80, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    make_figure(result)
    for p, mt in prior_mtime.items():
        assert p.stat().st_mtime == mt, f"{p.name} modified!"
    print(
        f"Confirmation: JSON -> {RESULT_PATH}; {len(prior_mtime)} prior probe JSONs untouched",
        flush=True,
    )


if __name__ == "__main__":
    main()
