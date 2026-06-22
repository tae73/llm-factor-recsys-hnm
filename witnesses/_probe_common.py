"""Shared engine for Track-A de-risk probes (Gate 0).

Reuses src/analysis/cold_start.py retrieval primitives but retains PER-USER
metrics so that paired bootstrap CIs over variant deltas can be computed.

Design invariants (paired falsification):
  * All variants are aligned to ONE canonical article_id order.
  * All variants are evaluated on the IDENTICAL fixed user sample, in the same
    order, so metric arrays are per-user paired across variants.
  * History indices / ground-truth / bracket are computed ONCE (they depend on
    item identity, not on the embedding variant).

Nothing here decides GO/NO-GO; probe_01 / probe_02 apply the pre-registered
criteria on top of these primitives.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.analysis.cold_start import (
    ACTIVITY_BRACKETS,
    _build_user_purchase_history,
    _compute_hr_ndcg_mrr,
    _load_val_ground_truth,
)

ABL_DIR = Path("data/embeddings/ablation")
ITEM_REF_PATH = Path("data/embeddings/item_bge_embeddings.npz")
TRAIN_TXN = Path("data/processed/train_transactions.parquet")
VAL_GT_PATH = Path("data/processed/val_ground_truth.json")
OUT_DIR = Path("witnesses")

# Logical variant name -> npz filename. "META" is the metadata-only baseline
# built by build_meta_embedding.py (the missing confound-free reference).
VARIANT_FILE: dict[str, str] = {
    "META": "meta.npz",
    "L1": "l1.npz",
    "L2": "l2.npz",
    "L3": "l3.npz",
    "L1+L2": "l1_l2.npz",
    "L1+L3": "l1_l3.npz",
    "L2+L3": "l2_l3.npz",
    "L1+L2+L3": "l1_l2_l3.npz",
}


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return x / n


def load_variants(names: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load named variants and align them to a common article_id order.

    Returns (canonical_ids, {name: emb_f32_normalized aligned to canonical_ids}).
    """
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for nm in names:
        path = ABL_DIR / VARIANT_FILE[nm]
        d = np.load(path, allow_pickle=True)
        emb = d["embeddings"].astype(np.float32)
        ids = d["article_ids"].astype(str)
        raw[nm] = (emb, ids)

    base_ids = raw[names[0]][1]
    all_same = all(np.array_equal(base_ids, raw[nm][1]) for nm in names)

    if all_same:
        canon = base_ids
        out = {nm: _l2norm(raw[nm][0]) for nm in names}
        return canon, out

    # Fallback: intersect, preserving base order.
    common: set[str] = set(base_ids)
    for nm in names:
        common &= set(raw[nm][1])
    canon = np.array([i for i in base_ids if i in common])
    out = {}
    for nm in names:
        emb, ids = raw[nm]
        pos = {a: k for k, a in enumerate(ids)}
        sel = np.array([pos[a] for a in canon])
        out[nm] = _l2norm(emb[sel])
    return canon, out


def load_reference(canon_ids: np.ndarray) -> np.ndarray:
    """Load item_bge reference embeddings, aligned+normalized to canon order.

    Used as a FIXED geometry for intra-list diversity so a variant cannot
    inflate its own diversity via its own embedding space.
    """
    d = np.load(ITEM_REF_PATH, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    ids = d["article_ids"].astype(str)
    pos = {a: k for k, a in enumerate(ids)}
    sel = np.array([pos[a] for a in canon_ids])
    return _l2norm(emb[sel])


class FixedUsers(NamedTuple):
    user_ids: list[str]
    hist_indices: list[np.ndarray]  # per-user item indices into canon order
    gt: list[set]
    brackets: np.ndarray  # (n_users,) bracket name per user
    n_purchases: np.ndarray  # (n_users,)
    n_users: int


def load_history_and_gt() -> tuple[dict[str, list[str]], dict[str, set]]:
    """Load (user_history, val_ground_truth) once — expensive parquet read.

    Reuse across many build_fixed_users calls (multi-seed probes) to avoid
    re-reading the ~700MB train_transactions parquet per call.
    """
    return _build_user_purchase_history(TRAIN_TXN), _load_val_ground_truth(VAL_GT_PATH)


def build_fixed_users(
    canon_ids: np.ndarray,
    sample_users: int | None = 50_000,
    seed: int = 42,
    user_history: dict[str, list[str]] | None = None,
    val_gt: dict[str, set] | None = None,
) -> FixedUsers:
    """Fix the evaluation user sample ONCE (shared across all variants)."""
    if user_history is None or val_gt is None:
        user_history, val_gt = load_history_and_gt()
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    eval_users = [u for u in val_gt if u in user_history and len(user_history[u]) > 0]
    rng = np.random.default_rng(seed)
    if sample_users is not None and len(eval_users) > sample_users:
        eval_users = list(rng.choice(eval_users, size=sample_users, replace=False))

    uids: list[str] = []
    hidx: list[np.ndarray] = []
    gts: list[set] = []
    brks: list[str] = []
    nps: list[int] = []
    for u in eval_users:
        hist = user_history[u]
        hi = [id_to_idx[a] for a in hist if a in id_to_idx]
        if not hi:
            continue
        n = len(hist)
        bracket = "50+"
        for bname, (lo, hi2) in ACTIVITY_BRACKETS.items():
            if lo <= n <= hi2:
                bracket = bname
                break
        uids.append(u)
        hidx.append(np.asarray(hi, dtype=np.int64))
        gts.append(val_gt[u])
        brks.append(bracket)
        nps.append(n)

    return FixedUsers(
        user_ids=uids,
        hist_indices=hidx,
        gt=gts,
        brackets=np.asarray(brks),
        n_purchases=np.asarray(nps),
        n_users=len(uids),
    )


class ScoreResult(NamedTuple):
    hr: np.ndarray  # (n_users,)
    ndcg: np.ndarray  # (n_users,)
    mrr: np.ndarray  # (n_users,)
    topk: np.ndarray  # (n_users, k) item indices into canon order


def score_variant(
    emb: np.ndarray,
    canon_ids: np.ndarray,
    fixed: FixedUsers,
    k: int = 12,
    chunk: int = 2000,
) -> ScoreResult:
    """Centroid-kNN retrieval, returning PER-USER metrics + top-k indices."""
    n_users = fixed.n_users
    d = emb.shape[1]

    centroids = np.zeros((n_users, d), dtype=np.float32)
    for i, hi in enumerate(fixed.hist_indices):
        c = emb[hi].mean(axis=0)
        nrm = np.linalg.norm(c)
        if nrm > 0:
            c = c / nrm
        centroids[i] = c

    hr = np.zeros(n_users, dtype=np.float64)
    ndcg = np.zeros(n_users, dtype=np.float64)
    mrr = np.zeros(n_users, dtype=np.float64)
    topk = np.zeros((n_users, k), dtype=np.int64)

    embT = np.ascontiguousarray(emb.T)  # (d, n_items)
    for s in range(0, n_users, chunk):
        e = min(s + chunk, n_users)
        scores = centroids[s:e] @ embT  # (b, n_items)
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        order = np.argsort(-scores[rows, part], axis=1)
        tk = part[rows, order]  # (b, k) sorted desc
        topk[s:e] = tk
        for j in range(e - s):
            h, n, m = _compute_hr_ndcg_mrr(tk[j], fixed.gt[s + j], canon_ids, k)
            hr[s + j] = h
            ndcg[s + j] = n
            mrr[s + j] = m

    return ScoreResult(hr=hr, ndcg=ndcg, mrr=mrr, topk=topk)


def score_variant_maxsim(
    emb: np.ndarray,
    canon_ids: np.ndarray,
    fixed: FixedUsers,
    k: int = 12,
    max_hist: int = 30,
) -> ScoreResult:
    """Robustness lens: max-similarity retrieval (best-matching purchased item)
    instead of centroid. score(item) = max over user's history of cos(item, hist).
    History is capped to the last ``max_hist`` items to bound per-user cost for
    heavy users (the conclusion is unaffected — recent items dominate).
    """
    n_users = fixed.n_users
    embT = np.ascontiguousarray(emb.T)
    hr = np.zeros(n_users)
    ndcg = np.zeros(n_users)
    mrr = np.zeros(n_users)
    topk = np.zeros((n_users, k), dtype=np.int64)
    for i, hi in enumerate(fixed.hist_indices):
        h = hi[-max_hist:] if len(hi) > max_hist else hi
        sims = emb[h] @ embT  # (<=max_hist, n_items)
        scores = sims.max(axis=0)  # (n_items,)
        part = np.argpartition(-scores, k)[:k]
        tk = part[np.argsort(-scores[part])]
        topk[i] = tk
        hr[i], ndcg[i], mrr[i] = _compute_hr_ndcg_mrr(tk, fixed.gt[i], canon_ids, k)
    return ScoreResult(hr=hr, ndcg=ndcg, mrr=mrr, topk=topk)


def variant_text_lengths(
    variant_names: list[str], sample: int = 5000, seed: int = 42
) -> dict[str, dict[str, float]]:
    """Mean composed-text length (chars/words) per variant — length-confound diagnostic.

    Recomposes the ablation texts from factual_knowledge + articles via the same
    text_composer used to build the embeddings. Samples ``sample`` items (mean
    length is stable) to avoid a slow 105K-row Python recomposition loop.
    """
    import pandas as pd

    from src.knowledge.factual.text_composer import build_all_ablation_texts, construct_factual_text

    fk = pd.read_parquet("data/knowledge/factual/factual_knowledge.parquet")
    meta_cols = [
        "article_id",
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "graphical_appearance_name",
        "section_name",
    ]
    articles = pd.read_parquet("data/processed/articles.parquet", columns=meta_cols)
    fk["article_id"] = fk["article_id"].astype(str)
    articles["article_id"] = articles["article_id"].astype(str)
    merged = articles.merge(fk, on="article_id", how="inner")
    if sample and len(merged) > sample:
        merged = merged.sample(sample, random_state=seed)

    acc: dict[str, list[int]] = {nm: [] for nm in variant_names}
    for _, row in merged.iterrows():
        article_meta = {c: row.get(c) for c in meta_cols[1:]}
        knowledge = {kk: row[kk] for kk in row.index if kk.startswith(("l1_", "l2_", "l3_"))}
        super_cat = row.get("super_category", "Apparel")
        combos = build_all_ablation_texts(article_meta, knowledge, super_cat)
        meta_text = construct_factual_text(article_meta, None, None, None, super_cat)
        for nm in variant_names:
            txt = meta_text if nm == "META" else combos.get(nm, "")
            acc[nm].append(len(txt))
    return {
        nm: {"mean_chars": float(np.mean(v)), "mean_words": float(np.mean([c / 5.5 for c in v]))}
        for nm, v in acc.items()
    }


def intra_list_diversity(topk: np.ndarray, ref_emb: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Per-user intra-list diversity@k on a FIXED reference space.

    diversity = mean over unique pairs of (1 - cosine). ref_emb is normalized,
    so cosine = dot. Returns (n_users,) array.
    """
    n_users, k = topk.shape
    iu, ju = np.triu_indices(k, k=1)
    out = np.zeros(n_users, dtype=np.float64)
    for s in range(0, n_users, chunk):
        e = min(s + chunk, n_users)
        vecs = ref_emb[topk[s:e]]  # (b, k, d)
        sim = np.einsum("bkd,bjd->bkj", vecs, vecs)  # (b, k, k)
        pair_sims = sim[:, iu, ju]  # (b, n_pairs)
        out[s:e] = (1.0 - pair_sims).mean(axis=1)
    return out


def catalog_coverage(topk: np.ndarray, n_items: int) -> float:
    """Fraction of the catalog that appears in any user's top-k."""
    return float(len(np.unique(topk)) / n_items)


# ---------------------------------------------------------------------------
# Serendipity / novelty / long-tail metrics (R-10 — the OPEN non-accuracy axis)
#
# probe_02 already CLOSED classical intra-list diversity + coverage (negative).
# These add the untested "L3 rescue path": novelty (popularity-based), long-tail
# exposure, and RELEVANCE-grounded surprise (long-tail-hit + serendipity). Pure
# functions over a top-k index matrix; the probe computes popularity / tail / unexpected
# masks and passes them in. Honesty: novelty/exposure are list properties (cheap to
# inflate); the headline is the relevance-grounded *hit* metrics.
# ---------------------------------------------------------------------------
def item_novelty(topk: np.ndarray, pop_prob: np.ndarray) -> np.ndarray:
    """Per-user mean self-information −log2(pop_prob) of the top-k items (higher = less popular).

    pop_prob is aligned to the canonical item order (same index space as topk). Pure list
    property — no relevance; report as DESCRIPTIVE only.
    """
    nov = -np.log2(np.clip(pop_prob[topk], 1e-12, None))  # (n_users, k)
    return nov.mean(axis=1)


def longtail_exposure(topk: np.ndarray, tail_mask: np.ndarray) -> np.ndarray:
    """Per-user fraction of top-k items that are long-tail (tail_mask aligned to canon order)."""
    return tail_mask[topk].mean(axis=1)


def hit_count(topk: np.ndarray, gts: list[set], canon_ids: np.ndarray, k: int = 12) -> np.ndarray:
    """Per-user count of top-k items that are in the user's ground-truth set (total relevance)."""
    out = np.zeros(len(topk))
    for u in range(len(topk)):
        gt = gts[u]
        out[u] = sum(1 for idx in topk[u, :k] if canon_ids[idx] in gt)
    return out


def tail_hit_count(
    topk: np.ndarray, gts: list[set], canon_ids: np.ndarray, tail_mask: np.ndarray, k: int = 12
) -> np.ndarray:
    """Per-user count of top-k HITS that are also long-tail items (relevance on the tail)."""
    out = np.zeros(len(topk))
    for u in range(len(topk)):
        gt = gts[u]
        out[u] = sum(1 for idx in topk[u, :k] if canon_ids[idx] in gt and tail_mask[idx])
    return out


def serendipitous_hit_count(
    topk: np.ndarray, gts: list[set], canon_ids: np.ndarray, unexpected: np.ndarray, k: int = 12
) -> np.ndarray:
    """Per-user count of top-k HITS flagged unexpected (relevance ∧ surprise).

    ``unexpected`` is a (n_users, k) bool aligned to topk positions (the probe computes it from
    fixed-reference cosine of each recommended item to the user's history centroid).
    """
    out = np.zeros(len(topk))
    for u in range(len(topk)):
        gt = gts[u]
        out[u] = sum(
            1 for j, idx in enumerate(topk[u, :k]) if canon_ids[idx] in gt and unexpected[u, j]
        )
    return out


def bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 123,
) -> dict[str, float]:
    """Paired bootstrap of (b - a). Positive delta means b > a.

    Returns mean_a, mean_b, delta, rel_gain (delta/mean_a), ci_lo, ci_hi (95%),
    and n.
    """
    if mask is not None:
        a = a[mask]
        b = b[mask]
    n = len(a)
    if n == 0:
        return {
            "mean_a": 0.0,
            "mean_b": 0.0,
            "delta": 0.0,
            "rel_gain": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "n": 0,
        }
    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    mean_a = float(a.mean())
    delta = float(diff.mean())
    return {
        "mean_a": mean_a,
        "mean_b": float(b.mean()),
        "delta": delta,
        "rel_gain": float(delta / mean_a) if mean_a > 0 else 0.0,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": int(n),
    }


def bracket_mask(fixed: FixedUsers, bracket: str) -> np.ndarray:
    return fixed.brackets == bracket


def pooled_means(res: ScoreResult) -> dict[str, float]:
    return {
        "hr_at_12": float(res.hr.mean()),
        "ndcg_at_12": float(res.ndcg.mean()),
        "mrr": float(res.mrr.mean()),
    }


def per_bracket_means(res: ScoreResult, fixed: FixedUsers) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for bname in ACTIVITY_BRACKETS:
        m = fixed.brackets == bname
        nb = int(m.sum())
        if nb == 0:
            out[bname] = {"hr_at_12": 0.0, "ndcg_at_12": 0.0, "mrr": 0.0, "n": 0}
        else:
            out[bname] = {
                "hr_at_12": float(res.hr[m].mean()),
                "ndcg_at_12": float(res.ndcg[m].mean()),
                "mrr": float(res.mrr[m].mean()),
                "n": nb,
            }
    return out
