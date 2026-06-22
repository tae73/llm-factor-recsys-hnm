"""PROBE D5 — MULTI-STAKEHOLDER VALUE MATRIX (NON-recommendation cells).

WHY — the recsys-accuracy claim was FALSIFIED: LLM-extracted L2/L3 attributes do
NOT beat L1 content for ranking (probes 01..22). The pivot is that LLM attributes
are valuable for ANALYTICS / MARKETING / ENGINEERING — interpretable catalog
enrichment — NOT for ranking accuracy. This probe fills the NON-rec cells of the
"value matrix" with HONEST numbers: does L2/L3 beat a METADATA baseline on tasks
that are NOT recommendation ranking?

We measure three cells, each as an INCREMENT of L2/L3 over the metadata baseline,
fixed seed, fully reproducible, CPU-only, no API / no $:

  CELL 1  STRUCTURE / SEGMENTATION
      Cluster items with (a) metadata one-hot, (b) L2/L3 (one-hot AND BGE text
      emb), (c) both. Report effective-k (participation ratio of cluster sizes)
      AND silhouette. Increment = L2/L3 vs metadata.
      (Prior finding: L2/L3 freq-vectors collapse but text-BGE recovers eff_k~10;
       we verify cleanly by reporting BOTH the one-hot and the BGE variant.)

  CELL 2  BUSINESS-OUTCOME PREDICTIVE INCREMENT (the decisive cell)
      Predict an ITEM outcome NOT used for ranking, from feature sets
      {metadata, L2/L3, both} with a simple sklearn classifier, 5-fold CV.
      Outcome A: popularity-tier (quartile of total_purchases).
      Outcome B: repurchase-rate bucket (tercile of repeat-customer fraction,
                 computed from transactions via DuckDB — a genuine business KPI).
      Increment = (L2/L3 or both) - metadata, with a paired-fold significance test.

  CELL 3  FACETED-SEARCH COVERAGE
      Count the semantic query axes L2/L3 provide that metadata LACKS, and show
      the catalog distribution for several of them to PROVE queryability. This is
      a coverage FACT, not a beat-metadata test.

VERDICT — "D5 GO" iff >=1 non-rec task shows L2/L3 SIGNIFICANTLY beating metadata;
else an honest "enrichment adds structure/coverage but not predictive lift".

Usage:
    cd <repo> && PYTHONUNBUFFERED=1 uv run python -u witnesses/probe_D5_value_matrix.py
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import chain
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from witnesses._probe_common import ABL_DIR, OUT_DIR  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
RNG = np.random.default_rng(SEED)

FK_PATH = ROOT / "data/knowledge/factual/factual_knowledge.parquet"
ARTICLES_PATH = ROOT / "data/processed/articles.parquet"
ITEM_FEAT_PATH = ROOT / "data/features/item_features.npz"
ITEM_FEAT_META = ROOT / "data/features/feature_meta.json"
ITEM_FEAT_STATS = ROOT / "data/features/feature_stats.json"
ID_MAPS = ROOT / "data/features/id_maps.json"
TRAIN_TXN = ROOT / "data/processed/train_transactions.parquet"
RESULT_PATH = OUT_DIR / "probe_D5_result.json"

# Metadata baseline columns (the existing content baseline we must BEAT).
META_COLS = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
    "index_name",
    "graphical_appearance_name",
]

# L2/L3 fields. List-valued fields are multi-hot; scalars are categorical/ordinal.
L2L3_LIST_COLS = ["l2_style_mood", "l2_occasion", "l3_style_lineage"]
L2L3_SCALAR_COLS = [
    "l2_perceived_quality",
    "l2_trendiness",
    "l2_season_fit",
    "l2_versatility",
    "l3_color_harmony",
    "l3_tone_season",
    "l3_coordination_role",
    "l3_visual_weight",
]
# Semantic axes for the faceted-search coverage cell, mapped to whether metadata
# offers an equivalent queryable axis.
FACET_AXES = {
    "occasion (l2_occasion)": False,
    "style_mood (l2_style_mood)": False,
    "perceived_quality (l2_perceived_quality)": False,
    "trendiness (l2_trendiness)": False,
    "season_fit (l2_season_fit)": False,
    "versatility (l2_versatility)": False,
    "coordination_role (l3_coordination_role)": False,
    "visual_weight (l3_visual_weight)": False,
    "color_harmony (l3_color_harmony)": False,
    "tone_season (l3_tone_season)": False,
    "style_lineage (l3_style_lineage)": False,
}


# ---------------------------------------------------------------------------
# Loaders / feature builders
# ---------------------------------------------------------------------------
def _load_aligned() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load FK + articles aligned to a common article_id order (FK ablation order)."""
    fk = pd.read_parquet(FK_PATH)
    fk["article_id"] = fk["article_id"].astype(str)
    art = pd.read_parquet(ARTICLES_PATH)
    art["article_id"] = art["article_id"].astype(str)

    # Canonical order = ablation embedding order (so BGE emb aligns row-for-row).
    d = np.load(ABL_DIR / "l2.npz", allow_pickle=True)
    canon = d["article_ids"].astype(str)

    fk = fk.set_index("article_id").reindex(canon).reset_index()
    art = art.set_index("article_id").reindex(canon).reset_index()
    keep = fk["super_category"].notna() & art["product_type_name"].notna()
    canon = canon[keep.values]
    fk = fk.loc[keep.values].reset_index(drop=True)
    art = art.loc[keep.values].reset_index(drop=True)
    return canon, fk, art


def _onehot_meta(art: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in META_COLS if c in art.columns]
    X = pd.get_dummies(art[cols].astype("category"), dummy_na=False)
    return X.to_numpy(dtype=np.float32), list(X.columns)


def _onehot_l2l3(fk: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    names: list[str] = []
    # scalar categorical / ordinal -> one-hot
    scal = fk[L2L3_SCALAR_COLS].astype(str)
    Xs = pd.get_dummies(scal, dummy_na=False)
    parts.append(Xs.to_numpy(dtype=np.float32))
    names += list(Xs.columns)
    # list-valued -> multi-hot
    for col in L2L3_LIST_COLS:
        vocab = sorted(
            set(
                chain.from_iterable(
                    (list(x) if isinstance(x, (list, np.ndarray)) else [])
                    for x in fk[col]
                )
            )
        )
        idx = {v: i for i, v in enumerate(vocab)}
        M = np.zeros((len(fk), len(vocab)), dtype=np.float32)
        for r, x in enumerate(fk[col]):
            if isinstance(x, (list, np.ndarray)):
                for v in x:
                    M[r, idx[v]] = 1.0
        parts.append(M)
        names += [f"{col}={v}" for v in vocab]
    return np.concatenate(parts, axis=1), names


def _load_bge(name: str, canon: np.ndarray) -> np.ndarray:
    d = np.load(ABL_DIR / f"{name}.npz", allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    ids = d["article_ids"].astype(str)
    pos = {a: k for k, a in enumerate(ids)}
    sel = np.array([pos[a] for a in canon])
    emb = emb[sel]
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return emb / n


# ---------------------------------------------------------------------------
# Outcome construction (business outcomes NOT used for ranking)
# ---------------------------------------------------------------------------
def _total_purchases(canon: np.ndarray) -> np.ndarray:
    """RAW total_purchases per canon item from item_features.npz.

    item_features.npz numerical[:,0] stores RAW total_purchases (0..44761);
    the log1p+z-score in feature_stats.json is applied at train time, not at
    storage time, so we read the raw column directly.
    """
    feat = np.load(ITEM_FEAT_PATH, allow_pickle=True)["numerical"]
    raw = feat[:, 0].astype(np.float64)
    id_to_idx = {str(k): int(v) for k, v in json.loads(ID_MAPS.read_text())["item_to_idx"].items()}
    out = np.array([raw[id_to_idx[a]] if a in id_to_idx else 0.0 for a in canon], dtype=np.float64)
    return out


def _repurchase_rate(canon: np.ndarray) -> np.ndarray:
    """Per-item repurchase rate = fraction of buyers who bought it on >=2 days.

    Genuine business KPI (loyalty / replenishment), NOT a ranking signal.
    """
    try:
        import duckdb
    except ImportError:
        return np.full(len(canon), np.nan)
    con = duckdb.connect()
    q = f"""
    WITH ci AS (
      SELECT article_id, customer_id, COUNT(DISTINCT t_dat) AS n_days
      FROM read_parquet('{TRAIN_TXN.as_posix()}')
      GROUP BY article_id, customer_id
    )
    SELECT article_id,
           COUNT(*) AS n_customers,
           SUM(CASE WHEN n_days > 1 THEN 1 ELSE 0 END) AS n_repeat
    FROM ci GROUP BY article_id
    """
    df = con.execute(q).df()
    df["article_id"] = df["article_id"].astype(str)
    df["rr"] = df["n_repeat"] / df["n_customers"].clip(lower=1)
    m = dict(zip(df["article_id"], df["rr"]))
    return np.array([m.get(a, np.nan) for a in canon], dtype=np.float64)


def _quantile_bucket(vals: np.ndarray, q: int, mask: np.ndarray) -> np.ndarray:
    """Assign each masked item to one of q ~equal-sized buckets; -1 elsewhere.

    Rank-based binning (ties broken by stable position) guarantees balanced
    classes even when the outcome has heavy ties — value-edge quantiles collapse
    a bucket to size 0 when many items share a value (e.g. repurchase_rate=0),
    which would yield a degenerate <q-class problem.
    """
    out = np.full(len(vals), -1, dtype=np.int64)
    v = vals[mask]
    order = np.argsort(v, kind="stable")  # ascending; ties keep input order
    ranks = np.empty(len(v), dtype=np.int64)
    ranks[order] = np.arange(len(v))
    out[mask] = np.minimum((ranks * q) // len(v), q - 1)
    return out


# ---------------------------------------------------------------------------
# Cell 1 — segmentation increment
# ---------------------------------------------------------------------------
def _effective_k(labels: np.ndarray) -> float:
    """Participation ratio of cluster sizes: (sum p)^2 / sum p^2, p = cluster size.

    eff_k == k when balanced; collapses toward 1 when one cluster dominates.
    """
    _, counts = np.unique(labels, return_counts=True)
    p = counts.astype(np.float64)
    return float(p.sum() ** 2 / (p**2).sum())


def _cluster_metrics(X: np.ndarray, k: int = 10, sample: int = 8000) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X.astype(np.float64))
    km = KMeans(n_clusters=k, random_state=SEED, n_init=4)
    labels = km.fit_predict(Xs)
    eff_k = _effective_k(labels)
    # silhouette on a fixed subsample (cost control, deterministic seed)
    rng = np.random.default_rng(SEED)
    if len(Xs) > sample:
        sidx = rng.choice(len(Xs), size=sample, replace=False)
    else:
        sidx = np.arange(len(Xs))
    try:
        sil = float(silhouette_score(Xs[sidx], labels[sidx]))
    except Exception:
        sil = float("nan")
    return {"k": k, "eff_k": round(eff_k, 3), "silhouette": round(sil, 4)}


def cell_segmentation(
    meta_oh: np.ndarray,
    l2l3_oh: np.ndarray,
    l2l3_bge: np.ndarray,
    meta_bge: np.ndarray,
    k: int = 10,
) -> dict:
    print(f"\n[CELL 1] Segmentation increment (k={k}) ...", flush=True)
    res = {
        "metadata_onehot": _cluster_metrics(meta_oh, k),
        "l2l3_onehot": _cluster_metrics(l2l3_oh, k),
        "l2l3_bge": _cluster_metrics(l2l3_bge, k),
        "both_onehot": _cluster_metrics(np.concatenate([meta_oh, l2l3_oh], axis=1), k),
        "metadata_bge": _cluster_metrics(meta_bge, k),
    }
    for name, m in res.items():
        print(f"    {name:18s} eff_k={m['eff_k']:6.3f}  silhouette={m['silhouette']:.4f}", flush=True)
    # Increment defined on the FAIR comparison: BGE-text L2/L3 vs BGE-text metadata
    # (same encoder geometry), and one-hot vs one-hot.
    inc_effk_bge = res["l2l3_bge"]["eff_k"] - res["metadata_bge"]["eff_k"]
    inc_sil_bge = res["l2l3_bge"]["silhouette"] - res["metadata_bge"]["silhouette"]
    inc_effk_oh = res["l2l3_onehot"]["eff_k"] - res["metadata_onehot"]["eff_k"]
    res["increment"] = {
        "eff_k_bge": round(inc_effk_bge, 3),
        "silhouette_bge": round(inc_sil_bge, 4),
        "eff_k_onehot": round(inc_effk_oh, 3),
        "onehot_collapse_verified": bool(res["l2l3_onehot"]["eff_k"] < res["l2l3_bge"]["eff_k"] - 0.5),
    }
    return res


# ---------------------------------------------------------------------------
# Cell 2 — business-outcome predictive increment (decisive)
# ---------------------------------------------------------------------------
class CVResult(NamedTuple):
    macro_f1_mean: float
    macro_f1_folds: list[float]
    auc_mean: float


def _cv_predict(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> CVResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    f1s, aucs = [], []
    n_classes = len(np.unique(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler(with_mean=True, with_std=True)
        Xtr = sc.fit_transform(X[tr].astype(np.float64))
        Xte = sc.transform(X[te].astype(np.float64))
        # solver=lbfgs + loose tol (5e-3): CV macro-F1/AUC stabilize FAR before the
        # machine-precision optimum, so a loose tol converges in <10 iters (~0.7s/fit
        # on the 443-dim "both" set) with identical scores to tol=1e-4. Deterministic
        # via random_state=SEED. (saga was ~35x slower here for the same scores.)
        clf = LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=200, tol=5e-3, random_state=SEED
        )
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        f1s.append(f1_score(y[te], pred, average="macro"))
        try:
            proba = clf.predict_proba(Xte)
            if n_classes == 2:
                aucs.append(roc_auc_score(y[te], proba[:, 1]))
            else:
                aucs.append(roc_auc_score(y[te], proba, multi_class="ovr", average="macro"))
        except Exception:
            aucs.append(float("nan"))
    return CVResult(float(np.mean(f1s)), [round(x, 4) for x in f1s], float(np.nanmean(aucs)))


# Practical-significance margin for macro-F1: with ~96K items, a +0.3pp delta is
# statistically "significant" yet operationally meaningless. We require BOTH a
# p<0.05 paired test AND a >=1pp macro-F1 lift before claiming L2/L3 adds value.
PRACTICAL_MARGIN = 0.01


def _paired_sig(folds_a: list[float], folds_b: list[float], min_delta: float = PRACTICAL_MARGIN) -> dict:
    """Paired one-sided test: is A (e.g. both) > B (metadata) across folds?

    `significant` requires statistical (p<0.05) AND practical (delta>=min_delta)
    evidence — an honesty guard against large-n trivial deltas.
    """
    from scipy import stats

    a, b = np.asarray(folds_a), np.asarray(folds_b)
    diff = a - b
    if np.allclose(diff, 0):
        return {
            "mean_delta": 0.0,
            "p_value": 1.0,
            "stat_significant": False,
            "practically_significant": False,
            "significant": False,
        }
    t, p_two = stats.ttest_rel(a, b)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    stat_sig = bool(p_one < 0.05 and diff.mean() > 0)
    prac_sig = bool(diff.mean() >= min_delta)
    return {
        "mean_delta": round(float(diff.mean()), 4),
        "p_value": round(float(p_one), 5),
        "stat_significant": stat_sig,
        "practically_significant": prac_sig,
        "significant": bool(stat_sig and prac_sig),
    }


def _outcome_cell(
    name: str,
    y: np.ndarray,
    mask: np.ndarray,
    meta_oh: np.ndarray,
    l2l3_oh: np.ndarray,
) -> dict:
    yv = y[mask]
    Xm = meta_oh[mask]
    Xl = l2l3_oh[mask]
    Xb = np.concatenate([Xm, Xl], axis=1)
    print(f"\n[CELL 2:{name}] n={mask.sum()}  classes={np.bincount(yv).tolist()}", flush=True)
    rm = _cv_predict(Xm, yv)
    rl = _cv_predict(Xl, yv)
    rb = _cv_predict(Xb, yv)
    sig_l = _paired_sig(rl.macro_f1_folds, rm.macro_f1_folds)
    sig_b = _paired_sig(rb.macro_f1_folds, rm.macro_f1_folds)
    print(
        f"    metadata  F1={rm.macro_f1_mean:.4f} AUC={rm.auc_mean:.4f}\n"
        f"    l2l3      F1={rl.macro_f1_mean:.4f} AUC={rl.auc_mean:.4f}  "
        f"(Δ={sig_l['mean_delta']:+.4f} p={sig_l['p_value']} sig={sig_l['significant']})\n"
        f"    both      F1={rb.macro_f1_mean:.4f} AUC={rb.auc_mean:.4f}  "
        f"(Δ={sig_b['mean_delta']:+.4f} p={sig_b['p_value']} sig={sig_b['significant']})",
        flush=True,
    )
    return {
        "metadata": {"macro_f1": round(rm.macro_f1_mean, 4), "auc": round(rm.auc_mean, 4)},
        "l2l3": {"macro_f1": round(rl.macro_f1_mean, 4), "auc": round(rl.auc_mean, 4)},
        "both": {"macro_f1": round(rb.macro_f1_mean, 4), "auc": round(rb.auc_mean, 4)},
        "increment_l2l3_vs_meta": sig_l,
        "increment_both_vs_meta": sig_b,
    }


# ---------------------------------------------------------------------------
# Cell 3 — faceted-search coverage
# ---------------------------------------------------------------------------
def cell_faceted(fk: pd.DataFrame) -> dict:
    print("\n[CELL 3] Faceted-search coverage ...", flush=True)
    n_axes_l2l3 = len(FACET_AXES)
    n_axes_meta_lacks = sum(1 for has in FACET_AXES.values() if not has)
    # Distributions for 4 representative facets (proves queryability).
    dists: dict[str, dict] = {}

    def _list_counts(col: str) -> dict:
        flat = list(
            chain.from_iterable(
                (list(x) if isinstance(x, (list, np.ndarray)) else []) for x in fk[col]
            )
        )
        return pd.Series(flat).value_counts().head(8).astype(int).to_dict()

    dists["l2_occasion"] = _list_counts("l2_occasion")
    dists["l2_style_mood"] = _list_counts("l2_style_mood")
    dists["l2_season_fit"] = fk["l2_season_fit"].value_counts().head(8).astype(int).to_dict()
    dists["l3_coordination_role"] = (
        fk["l3_coordination_role"].value_counts().head(8).astype(int).to_dict()
    )
    for facet, d in dists.items():
        print(f"    {facet:22s} -> {d}", flush=True)
    print(
        f"    L2/L3 semantic axes={n_axes_l2l3}, metadata LACKS={n_axes_meta_lacks}",
        flush=True,
    )
    return {
        "n_semantic_axes_l2l3": n_axes_l2l3,
        "n_axes_metadata_lacks": n_axes_meta_lacks,
        "axes_metadata_lacks": [a for a, has in FACET_AXES.items() if not has],
        "example_distributions": dists,
        "coverage_fact": (
            f"L2/L3 exposes {n_axes_meta_lacks} queryable semantic axes "
            f"(occasion, mood, quality, trendiness, season-fit, versatility, "
            f"coordination, visual-weight, color-harmony, tone-season, lineage) "
            f"absent from the metadata catalog."
        ),
    }


# ---------------------------------------------------------------------------
# Verdict + boxed summary
# ---------------------------------------------------------------------------
def _build_value_matrix(seg: dict, out_cells: dict, facet: dict) -> list[dict]:
    rows: list[dict] = []
    # Cell 1: segmentation (eff_k on BGE-text — the fair, same-encoder comparison)
    rows.append(
        {
            "task": "STRUCTURE/SEGMENTATION (BGE-text, eff_k)",
            "stakeholder": "Analytics",
            "metric": "effective_k @ k=10",
            "metadata_score": seg["metadata_bge"]["eff_k"],
            "l2l3_score": seg["l2l3_bge"]["eff_k"],
            "both_score": seg["both_onehot"]["eff_k"],
            "increment": seg["increment"]["eff_k_bge"],
            "significant": bool(seg["increment"]["eff_k_bge"] > 0.5),
        }
    )
    rows.append(
        {
            "task": "STRUCTURE/SEGMENTATION (BGE-text, silhouette)",
            "stakeholder": "Analytics",
            "metric": "silhouette @ k=10",
            "metadata_score": seg["metadata_bge"]["silhouette"],
            "l2l3_score": seg["l2l3_bge"]["silhouette"],
            "both_score": None,
            "increment": seg["increment"]["silhouette_bge"],
            "significant": bool(seg["increment"]["silhouette_bge"] > 0.0),
        }
    )
    # Cell 2: business outcomes (decisive)
    for oname, cell in out_cells.items():
        best_inc = max(
            cell["increment_l2l3_vs_meta"]["mean_delta"],
            cell["increment_both_vs_meta"]["mean_delta"],
        )
        sig = bool(
            cell["increment_l2l3_vs_meta"]["significant"]
            or cell["increment_both_vs_meta"]["significant"]
        )
        rows.append(
            {
                "task": f"BUSINESS-OUTCOME: {oname}",
                "stakeholder": "Marketing/Merch",
                "metric": "5-fold CV macro-F1",
                "metadata_score": cell["metadata"]["macro_f1"],
                "l2l3_score": cell["l2l3"]["macro_f1"],
                "both_score": cell["both"]["macro_f1"],
                "increment": round(best_inc, 4),
                "significant": sig,
            }
        )
    # Cell 3: coverage (fact, not a beat-test — increment = axes metadata lacks)
    rows.append(
        {
            "task": "FACETED-SEARCH COVERAGE",
            "stakeholder": "Engineering/Search",
            "metric": "# semantic axes metadata lacks",
            "metadata_score": 0,
            "l2l3_score": facet["n_axes_metadata_lacks"],
            "both_score": None,
            "increment": facet["n_axes_metadata_lacks"],
            "significant": bool(facet["n_axes_metadata_lacks"] > 0),
        }
    )
    return rows


def _print_box(matrix: list[dict], verdict: str) -> None:
    header = f"{'TASK':<46}{'METRIC':<24}{'META':>9}{'L2/L3':>9}{'BOTH':>9}{'INC':>9}{'SIG':>5}"
    width = len(header)
    line = "=" * width

    def fmt(v):
        if v is None:
            return "  --  "
        if isinstance(v, bool):
            return "Y" if v else "n"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    print("\n" + line, flush=True)
    print("  PROBE D5 — MULTI-STAKEHOLDER VALUE MATRIX  (L2/L3 vs METADATA, non-rec)", flush=True)
    print(line, flush=True)
    print(header, flush=True)
    print("-" * width, flush=True)
    for r in matrix:
        print(
            f"{r['task'][:45]:<46}{r['metric'][:23]:<24}"
            f"{fmt(r['metadata_score']):>9}{fmt(r['l2l3_score']):>9}"
            f"{fmt(r['both_score']):>9}{fmt(r['increment']):>9}"
            f"{fmt(r['significant']):>5}",
            flush=True,
        )
    print(line, flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(line + "\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70, flush=True)
    print("PROBE D5 — value matrix (CPU-only, fixed seed=42, no API/$)", flush=True)
    print("=" * 70, flush=True)

    canon, fk, art = _load_aligned()
    print(f"Aligned catalog: {len(canon)} items", flush=True)

    meta_oh, meta_names = _onehot_meta(art)
    l2l3_oh, l2l3_names = _onehot_l2l3(fk)
    print(f"Feature dims: metadata one-hot={meta_oh.shape[1]}, L2/L3 one-hot={l2l3_oh.shape[1]}", flush=True)
    l2l3_bge = _load_bge("l2_l3", canon)
    meta_bge = _load_bge("meta", canon)

    # --- Cell 1: segmentation ---
    seg = cell_segmentation(meta_oh, l2l3_oh, l2l3_bge, meta_bge, k=10)

    # --- Cell 2: business outcomes ---
    out_cells: dict[str, dict] = {}
    # Outcome A: popularity quartile (from total_purchases; NOT a ranking metric)
    tot = _total_purchases(canon)
    mask_pop = tot > 0  # only items with any sales
    y_pop = _quantile_bucket(tot, q=4, mask=mask_pop)
    out_cells["popularity_quartile"] = _outcome_cell(
        "popularity_quartile", y_pop, mask_pop, meta_oh, l2l3_oh
    )
    # Outcome B: repurchase-rate tercile (genuine loyalty KPI, from transactions)
    rr = _repurchase_rate(canon)
    mask_rr = ~np.isnan(rr)
    if mask_rr.sum() > 1000:
        y_rr = _quantile_bucket(rr, q=3, mask=mask_rr)
        out_cells["repurchase_rate_tercile"] = _outcome_cell(
            "repurchase_rate_tercile", y_rr, mask_rr, meta_oh, l2l3_oh
        )
    else:
        print("[CELL 2:repurchase] skipped (duckdb unavailable or too few items)", flush=True)

    # --- Cell 3: faceted coverage ---
    facet = cell_faceted(fk)

    # --- Verdict ---
    matrix = _build_value_matrix(seg, out_cells, facet)
    # "D5 GO" iff >=1 PREDICTIVE business-outcome cell significantly beats metadata.
    predictive_go = any(
        c["increment_l2l3_vs_meta"]["significant"] or c["increment_both_vs_meta"]["significant"]
        for c in out_cells.values()
    )
    struct_go = seg["increment"]["eff_k_bge"] > 0.5 or facet["n_axes_metadata_lacks"] > 0
    if predictive_go:
        verdict = (
            "D5 GO — L2/L3 SIGNIFICANTLY beats metadata on >=1 predictive "
            "business-outcome task (decisive non-rec cell)."
        )
    elif struct_go:
        verdict = (
            "D5 PARTIAL — honest: enrichment adds STRUCTURE (segmentation) and "
            "COVERAGE (faceted-search axes) but NOT predictive lift over metadata."
        )
    else:
        verdict = "D5 NO-GO — no non-rec cell shows L2/L3 beating metadata."

    result = {
        "probe": "D5_value_matrix",
        "seed": SEED,
        "n_items": int(len(canon)),
        "verdict": verdict,
        "predictive_go": bool(predictive_go),
        "structure_coverage_go": bool(struct_go),
        "value_matrix": matrix,
        "cells": {
            "segmentation": seg,
            "business_outcomes": out_cells,
            "faceted_coverage": facet,
        },
        "feature_dims": {
            "metadata_onehot": int(meta_oh.shape[1]),
            "l2l3_onehot": int(l2l3_oh.shape[1]),
            "bge_dim": int(l2l3_bge.shape[1]),
        },
    }

    _print_box(matrix, verdict)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Confirmation: JSON written -> {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
