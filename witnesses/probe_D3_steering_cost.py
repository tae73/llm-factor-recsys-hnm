"""PROBE D3 — The ACCURACY COST of controllable/steerable recommendation.

WHAT THIS DE-RISKS
    probe_15 proved steering PRECISION: a soft content+boost can drive the top-12 to
    occasion=Party at precision 1.00. But it NEVER measured what accuracy you pay for that
    control. This is the make-or-break for the "controllability" value cell: a steering knob
    that destroys discovery accuracy is a toy, not a product feature. D3 builds the
    precision-vs-accuracy trade-off curve over steering strength alpha, in TWO regimes:

      (a) CONTEXT-AWARE (oracle) steer — target = the attribute VALUE of the user's actual
          held-out next NEW item (they will buy a Party item -> steer to Party). Question:
          does steering toward the RIGHT context HELP or HURT discovery_map vs unsteered?
      (b) OFF-TARGET (random) steer — target = a random attribute value the user's GT does
          NOT have. This is the pure "price of control": how fast does discovery_map fall as
          we force the list toward an irrelevant attribute?

SETUP (immediate-split discovery, identical spine to probe_17)
    user L1 content centroid -> cosine over a shared top-5000 recent-popular pool -> top-12,
    exclude owned (discovery). Steering: score += alpha * 1[item carries target attribute].
    Accuracy = discovery_map@12 (NEW-only GT, the R-4 infra). Precision = fraction of the
    steered top-12 that carry the target attribute.

HONEST reduces_check
    Context-aware uses an ORACLE (we peek at the held-out item's attribute) — it is an
    upper bound on what a real context-predictor could realise, and an honest test of whether
    "right-context steering" is even directionally helpful. Off-target is the worst case and
    is the number to quote as the price of control. Same pool + same owned exclusion + same
    user sample across all alphas, so every comparison is paired/apples-to-apples.

VERDICT
    "D3 GO (controllability is cheap)" iff high steering precision (>=0.8) is reachable at a
    MODEST accuracy cost in the OFF-TARGET regime AND context-aware steering does NOT hurt
    (ideally helps) discovery_map. Otherwise the control knob is expensive and the value cell
    is weak/conditional.

Usage:
    uv run python witnesses/probe_D3_steering_cost.py [n_users]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.cohorts import discovery_map  # noqa: E402
from src.evaluation.metrics import compute_map_at_k  # noqa: E402
from witnesses._probe_common import OUT_DIR, load_variants  # noqa: E402

TRAIN_TXN = Path("data/processed/train_transactions.parquet")
FK = Path("data/knowledge/factual/factual_knowledge.parquet")
GT_PATH = Path("data/processed/immediate_ground_truth.json")

K = 12
POOL = 5000
RECENT_DATE = "2020-05-01"
MIN_HIST = 3
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
# primary single-attribute control axis (probe_15's lead demo) + a 2nd for composition
PRIMARY_ATTR = "l2_occasion"
FALLBACK_ATTR = "l2_style_mood"  # used if l2_occasion catalog coverage is thin (<50%)
SECOND_ATTR = "l2_season_fit"
COMPOSE_ALPHA = 0.5  # fixed alpha for the 2-attribute composition test
SEED = 42


def _values(val) -> list[str]:
    """Normalize an attribute cell (list / ndarray / scalar / NaN) to a list of str values."""
    if isinstance(val, (list, np.ndarray)):
        return [str(x) for x in val]
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    return [str(val)]


def main() -> None:
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    rng = np.random.default_rng(SEED)

    # ---- L1 item content embeddings (precomputed; CPU is fine) -------------------------
    canon_ids, V = load_variants(["L1"])
    l1 = V["L1"]
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    # ---- L2 attribute maps (lists) -----------------------------------------------------
    fk = pd.read_parquet(FK, columns=["article_id", PRIMARY_ATTR, SECOND_ATTR])
    fk["article_id"] = fk["article_id"].astype(str)
    attr_p = {a: _values(v) for a, v in zip(fk["article_id"], fk[PRIMARY_ATTR])}
    attr_s = {a: _values(v) for a, v in zip(fk["article_id"], fk[SECOND_ATTR])}

    # vocabulary of values per attribute (for off-target random sampling)
    vocab_p = sorted({x for vs in attr_p.values() for x in vs})
    vocab_s = sorted({x for vs in attr_s.values() for x in vs})
    print(f"[D3] {PRIMARY_ATTR} vocab ({len(vocab_p)}): {vocab_p}")
    print(f"[D3] {SECOND_ATTR} vocab ({len(vocab_s)}): {vocab_s}")

    # fallback primary axis if l2_occasion coverage over the catalog is thin
    occ_cov = np.mean([len(vs) > 0 for vs in attr_p.values()])
    if occ_cov < 0.5:
        print(f"[D3] {PRIMARY_ATTR} coverage {occ_cov*100:.1f}% < 50% -> falling back to "
              f"{FALLBACK_ATTR}")
        fb = pd.read_parquet(FK, columns=["article_id", FALLBACK_ATTR])
        fb["article_id"] = fb["article_id"].astype(str)
        attr_p = {a: _values(v) for a, v in zip(fb["article_id"], fb[FALLBACK_ATTR])}
        vocab_p = sorted({x for vs in attr_p.values() for x in vs})
        STEER_ATTR_NAME = FALLBACK_ATTR
    else:
        STEER_ATTR_NAME = PRIMARY_ATTR
    print(f"[D3] primary steering attribute = {STEER_ATTR_NAME} (coverage {occ_cov*100:.1f}%)")

    import duckdb

    con = duckdb.connect()

    # ---- candidate pool = top-5000 recent-popular items --------------------------------
    pop = con.execute(
        f"SELECT article_id FROM read_parquet('{TRAIN_TXN}') WHERE t_dat >= DATE '{RECENT_DATE}' "
        f"GROUP BY article_id ORDER BY count(*) DESC LIMIT {POOL}"
    ).fetchall()
    pool_ids = [str(a) for (a,) in pop if str(a) in id_to_idx]
    pool_idx = np.array([id_to_idx[a] for a in pool_ids])
    pool_l1 = l1[pool_idx]  # (P, 768), normalized
    P = len(pool_ids)
    print(f"[D3] candidate pool: {P} items (top recent-popular since {RECENT_DATE})")

    # per-pool-item membership masks per attribute value (vectorized steering boost)
    pool_p_vals = [set(attr_p.get(a, [])) for a in pool_ids]
    pool_s_vals = [set(attr_s.get(a, [])) for a in pool_ids]

    def pool_mask(attr_pool_vals: list[set], value: str) -> np.ndarray:
        return np.fromiter((value in s for s in attr_pool_vals), dtype=bool, count=P)

    mask_p = {v: pool_mask(pool_p_vals, v) for v in vocab_p}  # cache primary masks
    mask_s = {v: pool_mask(pool_s_vals, v) for v in vocab_s}

    # ---- ground truth (immediate split) + train history (GT users only) ---------------
    gt = {str(k): [str(x) for x in v] for k, v in json.loads(GT_PATH.read_text()).items()}
    gt_users = pd.DataFrame({"customer_id": list(gt)})
    con.register("gt_users", gt_users)
    hist_rows = con.execute(
        f"SELECT t.customer_id, list(DISTINCT t.article_id) FROM read_parquet('{TRAIN_TXN}') t "
        f"JOIN gt_users g ON t.customer_id=g.customer_id GROUP BY t.customer_id"
    ).fetchall()
    train_history = {str(cid): set(str(a) for a in items) for cid, items in hist_rows}

    # ---- build the eval user sample ----------------------------------------------------
    # A user is eligible if:
    #   - >= MIN_HIST history items with an L1 embedding (content centroid is meaningful)
    #   - has at least one NEW GT item (not owned) that is IN the candidate pool and carries
    #     a PRIMARY-attr value -> gives a recoverable oracle context AND a hittable target.
    pool_set = set(pool_ids)
    eligible: list[str] = []
    user_ctx_p: dict[str, str] = {}   # oracle context value for primary attr
    user_ctx_s: dict[str, str] = {}   # oracle context value for second attr (for composition)
    all_users = list(gt.keys())
    rng.shuffle(all_users)
    for u in all_users:
        if len(eligible) >= n_users:
            break
        hist = train_history.get(u)
        if not hist:
            continue
        hidx = [id_to_idx[h] for h in hist if h in id_to_idx]
        if len(hidx) < MIN_HIST:
            continue
        new_gt = [g for g in gt[u] if g not in hist]
        # NEW GT items that are scoreable in the pool
        new_in_pool = [g for g in new_gt if g in pool_set]
        if not new_in_pool:
            continue
        # oracle context: a primary-attr value carried by one of the user's new-in-pool GT items
        ctx_p_candidates = [v for g in new_in_pool for v in attr_p.get(g, [])]
        if not ctx_p_candidates:
            continue
        user_ctx_p[u] = ctx_p_candidates[0]
        ctx_s_candidates = [v for g in new_in_pool for v in attr_s.get(g, [])]
        user_ctx_s[u] = ctx_s_candidates[0] if ctx_s_candidates else (vocab_s[0] if vocab_s else "")
        eligible.append(u)

    coverage = len(eligible) / max(1, len(gt))
    print(f"[D3] eligible users (>= {MIN_HIST} hist, NEW GT in pool w/ {PRIMARY_ATTR}): "
          f"{len(eligible)} ({coverage*100:.1f}% of {len(gt)} GT users)")

    sub_gt = {u: gt[u] for u in eligible}
    sub_hist = {u: train_history[u] for u in eligible}

    # off-target value per user (random attr value the user's GT does NOT carry).
    # Precomputed ONCE per user so the off-target target is FIXED across the alpha sweep
    # (paired comparison; rng.choice must not advance inside the per-alpha loops).
    user_off_p: dict[str, str] = {}
    user_off_s: dict[str, str] = {}
    for u in eligible:
        owned_vals_p = {v for g in gt[u] for v in attr_p.get(g, [])}
        choices_p = [v for v in vocab_p if v not in owned_vals_p]
        user_off_p[u] = str(rng.choice(choices_p)) if choices_p else vocab_p[0]
        owned_vals_s = {v for g in gt[u] for v in attr_s.get(g, [])}
        choices_s = [v for v in vocab_s if v not in owned_vals_s]
        user_off_s[u] = (
            str(rng.choice(choices_s)) if choices_s else (vocab_s[0] if vocab_s else "")
        )

    # precompute per-user base content scores + owned mask over the pool (paired across alpha)
    print("[D3] precomputing per-user content scores over pool ...")
    base_scores: dict[str, np.ndarray] = {}
    owned_masks: dict[str, np.ndarray] = {}
    for u in eligible:
        hist = train_history[u]
        c = l1[[id_to_idx[h] for h in hist if h in id_to_idx]].mean(0)
        c = c / (np.linalg.norm(c) + 1e-9)
        base_scores[u] = pool_l1 @ c  # (P,)
        owned_masks[u] = np.fromiter((a in hist for a in pool_ids), dtype=bool, count=P)

    def topk_ids(score: np.ndarray, owned: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Top-K pool ids (owned excluded), and the boolean index of those K into the pool."""
        sc = np.where(owned, -np.inf, score)
        order = np.argpartition(-sc, K)[:K]
        order = order[np.argsort(-sc[order])]
        return [pool_ids[i] for i in order], order

    # ---- sweep alpha: single-attribute, context-aware & off-target ---------------------
    def run_single(alpha: float) -> dict:
        preds_ctx: dict[str, list[str]] = {}
        preds_off: dict[str, list[str]] = {}
        prec_ctx, prec_off = [], []
        for u in eligible:
            base = base_scores[u]
            owned = owned_masks[u]
            # context-aware
            tv = user_ctx_p[u]
            m = mask_p[tv]
            ids_c, order_c = topk_ids(base + alpha * m.astype(float), owned)
            preds_ctx[u] = ids_c
            prec_ctx.append(float(m[order_c].mean()))
            # off-target
            ov = user_off_p[u]
            mo = mask_p[ov]
            ids_o, order_o = topk_ids(base + alpha * mo.astype(float), owned)
            preds_off[u] = ids_o
            prec_off.append(float(mo[order_o].mean()))
        dmap_ctx = discovery_map(preds_ctx, sub_gt, sub_hist, k=K).map_at_k
        dmap_off = discovery_map(preds_off, sub_gt, sub_hist, k=K).map_at_k
        return {
            "alpha": alpha,
            "precision_ctx": float(np.mean(prec_ctx)),
            "precision_off": float(np.mean(prec_off)),
            "discovery_map_context": float(dmap_ctx),
            "discovery_map_offtarget": float(dmap_off),
        }

    print("[D3] sweeping alpha (single attribute) ...")
    single = [run_single(a) for a in ALPHAS]
    baseline_dmap = single[0]["discovery_map_context"]  # alpha=0 -> ctx==off (no steer)

    # ---- 2-attribute composition (context-aware + off-target) at sweep -----------------
    # joint precision = fraction of top-12 carrying BOTH targets.
    def run_compose(alpha: float) -> dict:
        preds_ctx: dict[str, list[str]] = {}
        preds_off: dict[str, list[str]] = {}
        jprec_ctx, jprec_off = [], []
        for u in eligible:
            base = base_scores[u]
            owned = owned_masks[u]
            # context-aware: steer toward the GT item's (occasion, season) jointly
            tp, ts = user_ctx_p[u], user_ctx_s[u]
            mp, ms = mask_p[tp], mask_s.get(ts, np.zeros(P, bool))
            boost = alpha * (mp.astype(float) + ms.astype(float))
            ids_c, order_c = topk_ids(base + boost, owned)
            preds_ctx[u] = ids_c
            jprec_ctx.append(float((mp[order_c] & ms[order_c]).mean()))
            # off-target: steer toward a random (occasion, season) pair the GT lacks
            # (targets FIXED per user across the alpha sweep — see user_off_p/user_off_s)
            op = user_off_p[u]
            os_ = user_off_s[u]
            mpo, mso = mask_p[op], mask_s.get(os_, np.zeros(P, bool))
            boost_o = alpha * (mpo.astype(float) + mso.astype(float))
            ids_o, order_o = topk_ids(base + boost_o, owned)
            preds_off[u] = ids_o
            jprec_off.append(float((mpo[order_o] & mso[order_o]).mean()))
        dmap_ctx = discovery_map(preds_ctx, sub_gt, sub_hist, k=K).map_at_k
        dmap_off = discovery_map(preds_off, sub_gt, sub_hist, k=K).map_at_k
        return {
            "alpha": alpha,
            "joint_precision_ctx": float(np.mean(jprec_ctx)),
            "joint_precision_off": float(np.mean(jprec_off)),
            "discovery_map_context": float(dmap_ctx),
            "discovery_map_offtarget": float(dmap_off),
        }

    print("[D3] sweeping alpha (2-attribute composition) ...")
    compose = [run_compose(a) for a in ALPHAS]
    compose_fixed = next(c for c in compose if abs(c["alpha"] - COMPOSE_ALPHA) < 1e-9)

    # ---- verdict logic -----------------------------------------------------------------
    # high precision reachable cheaply in OFF-TARGET regime?
    off_high = [s for s in single if s["precision_off"] >= 0.8]
    if off_high:
        # smallest alpha reaching >=0.8 off-target precision; its accuracy cost
        s_hi = min(off_high, key=lambda s: s["alpha"])
        off_cost_abs = baseline_dmap - s_hi["discovery_map_offtarget"]
        off_cost_rel = off_cost_abs / baseline_dmap if baseline_dmap > 0 else float("inf")
        precision_reachable = True
    else:
        s_hi = max(single, key=lambda s: s["precision_off"])
        off_cost_abs = baseline_dmap - s_hi["discovery_map_offtarget"]
        off_cost_rel = off_cost_abs / baseline_dmap if baseline_dmap > 0 else float("inf")
        precision_reachable = False

    # does context-aware steering hurt? compare best context-aware dmap across alpha>0 vs baseline
    ctx_best = max((s["discovery_map_context"] for s in single if s["alpha"] > 0), default=baseline_dmap)
    ctx_helps = ctx_best > baseline_dmap
    ctx_hurts = ctx_best < baseline_dmap * 0.95  # >5% drop counts as a hurt

    MODEST = 0.30  # off-target accuracy cost <=30% relative => "modest"
    cheap = precision_reachable and (off_cost_rel <= MODEST) and (not ctx_hurts)

    if cheap and ctx_helps:
        verdict = (f"D3 GO (controllability is cheap) — off-target precision >=0.8 reachable at "
                   f"alpha={s_hi['alpha']} for {off_cost_rel*100:.0f}% relative discovery_map cost, AND "
                   f"context-aware steering HELPS (best ctx dmap {ctx_best:.5f} > baseline {baseline_dmap:.5f}).")
    elif cheap:
        verdict = (f"D3 GO (controllability is cheap) — off-target precision >=0.8 reachable at "
                   f"alpha={s_hi['alpha']} for {off_cost_rel*100:.0f}% relative discovery_map cost; "
                   f"context-aware steering does NOT hurt (best ctx dmap {ctx_best:.5f} vs baseline {baseline_dmap:.5f}).")
    elif precision_reachable:
        verdict = (f"D3 CONDITIONAL — off-target precision >=0.8 reachable at alpha={s_hi['alpha']} but at "
                   f"{off_cost_rel*100:.0f}% relative discovery_map cost (>{MODEST*100:.0f}%): control is real "
                   f"but EXPENSIVE; context-aware best dmap {ctx_best:.5f} vs baseline {baseline_dmap:.5f}.")
    else:
        verdict = (f"D3 NO-GO — precision >=0.8 not reachable in off-target regime (max "
                   f"{s_hi['precision_off']:.2f}); the control knob is weak.")

    out = {
        "probe": "probe_D3_steering_cost",
        "setup": {
            "split": "immediate", "pool": POOL, "recent_since": RECENT_DATE, "k": K,
            "min_hist": MIN_HIST, "primary_attr": STEER_ATTR_NAME, "second_attr": SECOND_ATTR,
            "compose_alpha": COMPOSE_ALPHA, "n_eligible_users": len(eligible),
            "coverage_frac": coverage,
        },
        "baseline_discovery_map": baseline_dmap,
        "alphas": ALPHAS,
        "single_attribute": single,
        "two_attribute": compose,
        "two_attribute_fixed_alpha": compose_fixed,
        "accuracy_cost_summary": {
            "offtarget_precision_ge_0.8_at_alpha": s_hi["alpha"] if precision_reachable else None,
            "offtarget_dmap_at_that_alpha": s_hi["discovery_map_offtarget"],
            "offtarget_cost_abs": off_cost_abs,
            "offtarget_cost_rel": off_cost_rel,
            "context_best_dmap": ctx_best,
            "context_best_rel_gain": (ctx_best - baseline_dmap) / baseline_dmap if baseline_dmap > 0 else 0.0,
            "context_2attr_best_dmap": max(c["discovery_map_context"] for c in compose),
            "context_2attr_best_rel_gain": (
                (max(c["discovery_map_context"] for c in compose) - baseline_dmap) / baseline_dmap
                if baseline_dmap > 0 else 0.0
            ),
            "context_helps": bool(ctx_helps),
            "context_hurts": bool(ctx_hurts),
        },
        "verdict": verdict,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_D3_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ---- boxed trade-off table ---------------------------------------------------------
    W = 86
    print()
    print("+" + "-" * W + "+")
    print("|" + "  PROBE D3 — accuracy COST of controllable recommendation (steering trade-off)".ljust(W) + "|")
    print("|" + f"  immediate-split discovery | pool={POOL} | n_users={len(eligible)} | "
          f"baseline dmap={baseline_dmap:.5f}".ljust(W) + "|")
    print("+" + "-" * W + "+")
    def _rel(x: float) -> float:
        return (x - baseline_dmap) / baseline_dmap * 100 if baseline_dmap > 0 else 0.0

    hdr = (f"  {'alpha':>5s} | {'prec_ctx':>8s} {'prec_off':>8s} | "
           f"{'dmap_CTX':>9s} {'dmap_OFF':>9s} | {'CTX Δ% rel':>11s} {'OFF Δ% rel':>11s}")
    print("|" + hdr.ljust(W) + "|")
    print("|" + ("  " + "-" * (W - 4)).ljust(W) + "|")
    print("|" + f"  SINGLE ATTRIBUTE ({STEER_ATTR_NAME})  [Δ% = relative to baseline]".ljust(W) + "|")
    for s in single:
        row = (f"  {s['alpha']:>5.2f} | {s['precision_ctx']:>8.2f} {s['precision_off']:>8.2f} | "
               f"{s['discovery_map_context']:>9.5f} {s['discovery_map_offtarget']:>9.5f} | "
               f"{_rel(s['discovery_map_context']):>+10.1f}% {_rel(s['discovery_map_offtarget']):>+10.1f}%")
        print("|" + row.ljust(W) + "|")
    print("|" + ("  " + "-" * (W - 4)).ljust(W) + "|")
    print("|" + "  TWO ATTRIBUTES (occasion + season) — prec = JOINT (both present)".ljust(W) + "|")
    for c in compose:
        row = (f"  {c['alpha']:>5.2f} | {c['joint_precision_ctx']:>8.2f} {c['joint_precision_off']:>8.2f} | "
               f"{c['discovery_map_context']:>9.5f} {c['discovery_map_offtarget']:>9.5f} | "
               f"{_rel(c['discovery_map_context']):>+10.1f}% {_rel(c['discovery_map_offtarget']):>+10.1f}%")
        print("|" + row.ljust(W) + "|")
    print("+" + "-" * W + "+")
    print("|" + f"  @ alpha={COMPOSE_ALPHA}: joint prec ctx={compose_fixed['joint_precision_ctx']:.2f} "
          f"off={compose_fixed['joint_precision_off']:.2f} | "
          f"dmap ctx={compose_fixed['discovery_map_context']:.5f} "
          f"off={compose_fixed['discovery_map_offtarget']:.5f}".ljust(W) + "|")
    print("+" + "-" * W + "+")
    # wrap verdict
    print("  VERDICT:")
    words = verdict.split()
    line = "   "
    for w in words:
        if len(line) + len(w) + 1 > W:
            print(line)
            line = "   "
        line += " " + w
    if line.strip():
        print(line)
    print("=" * (W + 2))


if __name__ == "__main__":
    main()
