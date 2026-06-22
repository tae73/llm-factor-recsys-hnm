"""PROBE DE1 — ATTRIBUTE SCREEN: which LLM-extracted catalog attributes are SALVAGEABLE?

WHY — the recsys-ranking claim was falsified (probes 01..22): LLM L2/L3 attributes do
NOT beat L1 content for ranking. The surviving pivot is "interpretable catalog
enrichment" (D5: PARTIAL — structure+coverage but no predictive lift). But two failure
modes were diagnosed:
  * probe_14  — L1 PREDICTS L2/L3 (mean lift=0.38): L2/L3 are product-internal
                redescriptions of L1 (redundancy is by-construction).
  * D5        — extreme CONCENTRATION (l2_occasion 88% "Everyday", l3_coordination_role
                67% "Foundation") and no predictive lift over metadata.

Before spending $ on NEW extraction we SCREEN every existing L1/L2/L3 attribute on three
design principles and decide keep-vs-newly-extract. We also surface the DECISION-AXES
that are missing/weak (gaps to fill).

THREE SCORES per attribute (all 0..1, higher = better, fully reproducible, CPU-only, no $):

  1. DISCRIMINATION = normalized Shannon entropy of the value distribution,
     H / log(n_values). 0 = one value dominates (useless, e.g. occasion 88% Everyday).
     Also report top-1 value share.

  2. NON-REDUNDANCY = 1 - metadata_predictability. We PREDICT the attribute from the
     7 metadata fields (one-hot) with a LogisticRegression, 5-fold CV, and report a
     "lift over majority" exactly like probe_14: lift = (acc - maj) / (1 - maj). HIGH
     metadata-predictability => the attribute is REDUNDANT with metadata (low
     non-redundancy). We ALSO report L1-BGE predictability (kNN on data/embeddings/
     ablation/l1.npz) to separate "redundant with METADATA" vs "redundant with the
     PRODUCT itself" (probe_14 lens).

  3. BEHAVIORAL-SIGNAL = does the attribute carry purchase-behavior structure BEYOND
     metadata? Two cheap proxies on train_transactions, each normalized 0..1 and then
     CONTROLLED against a matched metadata field (the metadata col that best predicts
     the attribute), so we credit only the *excess* structure:
       (a) SEASONALITY  — Cramer's V between attribute value and purchase-month.
       (b) SELL-THROUGH — eta (ANOVA correlation ratio) between attribute value and
                          log1p(item total_purchases) bucket.
     behavioral_signal = max(0, raw_signal - matched_metadata_signal).

VERDICT TAG per attribute:
  SALVAGEABLE — discriminative AND non-redundant AND has behavioral signal.
  CONCENTRATED — low discrimination (one value dominates) — first failure mode.
  REDUNDANT — high metadata OR L1 predictability — second failure mode.
  WEAK — passes thresholds nowhere decisively (kept-but-marginal).

GAPS — decision-axes that are missing or weak across the surviving set (e.g. no good
"trend-phase / hype-cycle", "wardrobe-role / outfit-slot", "fine-grained occasion",
"price-tier / value-perception"), to guide new extraction.

Usage:
    cd <repo> && OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 uv run python -u \
        witnesses/probe_DE1_attribute_screen.py [n_items]
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
TRAIN_TXN = ROOT / "data/processed/train_transactions.parquet"
L1_EMB = ABL_DIR / "l1.npz"
RESULT_PATH = OUT_DIR / "probe_DE1_result.json"

# Metadata baseline = the existing content catalog the LLM attributes must improve on.
META_COLS = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
    "index_name",
    "graphical_appearance_name",
]

# All LLM-extracted L1/L2/L3 attributes we screen. We EXCLUDE the two free-text fields
# with ~20K-26K unique values (l1_material_detail, l2_target_impression) from the
# *categorical* screen — they have no usable value distribution as a facet — but we
# flag them explicitly in the output as "free-text (not a facet)".
FREETEXT_COLS = ["l1_material_detail", "l2_target_impression"]

# List-valued fields: discrimination uses the PRIMARY (first) value; behavioral signal
# uses the EXPLODED multi-hot membership. We state this in the output.
LIST_COLS = ["l1_design_details", "l2_style_mood", "l2_occasion", "l3_style_lineage"]

# Logical layer -> which metadata col is the natural "matched" control for behavioral
# signal. We control each attribute's behavioral signal against the metadata col that
# best predicts it (computed at runtime), so this map is only a documentation hint.
SCREEN_ATTRS = [
    # L1 — product/material/cut
    "l1_material", "l1_closure", "l1_design_details",
    "l1_slot4", "l1_slot5", "l1_slot6", "l1_slot7",
    # L2 — perceived
    "l2_style_mood", "l2_occasion", "l2_perceived_quality", "l2_trendiness",
    "l2_season_fit", "l2_versatility",
    # L3 — theory
    "l3_color_harmony", "l3_coordination_role", "l3_visual_weight",
    "l3_style_lineage", "l3_slot6", "l3_slot7", "l3_tone_season",
]

# Decision-axes we audit for COVERAGE in the GAPS summary. has=True means the existing
# catalog already covers it WELL (discriminative + usable); we mark False at runtime
# when no surviving attribute serves the axis.
DECISION_AXES = {
    "material / fabric": ["l1_material"],
    "cut / silhouette (fine)": ["l1_slot6", "l3_slot6", "l3_slot7"],
    "neckline / sleeve detail": ["l1_slot4", "l1_slot5", "l1_design_details"],
    "color harmony / palette": ["l3_color_harmony", "l3_tone_season"],
    "style mood / aesthetic": ["l2_style_mood", "l3_style_lineage"],
    "occasion (fine-grained)": ["l2_occasion"],
    "seasonality": ["l2_season_fit"],
    "perceived quality / value-tier": ["l2_perceived_quality"],
    "trend-phase / hype-cycle": ["l2_trendiness"],
    "wardrobe-role / outfit-slot": ["l3_coordination_role", "l2_versatility"],
    "visual weight / proportion": ["l3_visual_weight"],
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _first(v) -> str:
    if isinstance(v, (list, np.ndarray)):
        return str(v[0]) if len(v) else "NA"
    return str(v)


def _load_aligned() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """FK + articles aligned to the L1-ablation embedding order (so L1 BGE aligns)."""
    fk = pd.read_parquet(FK_PATH)
    fk["article_id"] = fk["article_id"].astype(str)
    art = pd.read_parquet(ARTICLES_PATH)
    art["article_id"] = art["article_id"].astype(str)

    d = np.load(L1_EMB, allow_pickle=True)
    canon = d["article_ids"].astype(str)

    fk = fk.set_index("article_id").reindex(canon).reset_index()
    art = art.set_index("article_id").reindex(canon).reset_index()
    keep = fk["super_category"].notna() & art["product_type_name"].notna()
    canon = canon[keep.values]
    fk = fk.loc[keep.values].reset_index(drop=True)
    art = art.loc[keep.values].reset_index(drop=True)
    return canon, fk, art


def _load_l1_bge(canon: np.ndarray) -> np.ndarray:
    d = np.load(L1_EMB, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    ids = d["article_ids"].astype(str)
    pos = {a: k for k, a in enumerate(ids)}
    sel = np.array([pos[a] for a in canon])
    emb = emb[sel]
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return emb / n


def _onehot_meta(art: pd.DataFrame) -> np.ndarray:
    cols = [c for c in META_COLS if c in art.columns]
    X = pd.get_dummies(art[cols].astype("category"), dummy_na=False)
    return X.to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# SCORE 1 — Discrimination (normalized Shannon entropy, top-1 share)
# ---------------------------------------------------------------------------
def discrimination(values: np.ndarray) -> tuple[float, float, int]:
    """Return (normalized_entropy, top1_share, n_values) on the PRIMARY-value series."""
    vals, cnts = np.unique(values, return_counts=True)
    p = cnts / cnts.sum()
    H = float(-(p * np.log(p)).sum())
    n = len(vals)
    norm_H = H / np.log(n) if n > 1 else 0.0
    top1 = float(cnts.max() / cnts.sum())
    return float(norm_H), top1, int(n)


# ---------------------------------------------------------------------------
# SCORE 2 — Non-redundancy (1 - metadata predictability; + L1-BGE predictability)
# ---------------------------------------------------------------------------
def _cv_lift_logreg(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5, max_classes: int = 60
) -> dict:
    """5-fold CV accuracy + majority + lift for predicting y from X (LogisticRegression).

    To bound cost & keep the multinomial well-posed, classes are capped: rare classes
    (beyond ``max_classes`` by frequency) are folded into one "OTHER" bucket. Reported
    n_classes is the post-fold count. Returns acc, majority_acc, lift.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    # cap classes
    vals, cnts = np.unique(y, return_counts=True)
    if len(vals) > max_classes:
        keep = set(vals[np.argsort(-cnts)[: max_classes - 1]])
        y = np.array([v if v in keep else "OTHER" for v in y])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler(with_mean=False)  # X is sparse-ish one-hot; keep nonneg ok
        Xtr = sc.fit_transform(X[tr].astype(np.float64))
        Xte = sc.transform(X[te].astype(np.float64))
        clf = LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=120, tol=5e-3, random_state=SEED, n_jobs=1
        )
        clf.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xte)))
    acc = float(np.mean(accs))
    mvals, mcnts = np.unique(y, return_counts=True)
    maj = float(mcnts.max() / mcnts.sum())
    lift = (acc - maj) / (1 - maj) if maj < 1 else 0.0
    return {"acc": acc, "majority_acc": maj, "lift": float(lift), "n_classes": int(len(mvals))}


def _knn_lift_l1(
    emb: np.ndarray, y: np.ndarray, knn: int = 15, n_sample: int = 20_000
) -> dict:
    """probe_14-style L1-BGE predictability: kNN-vote lift over majority.

    Subsamples to ``n_sample`` items for the O(n^2) sim; the lift estimate is stable.
    """
    n = len(y)
    perm = RNG.permutation(n)
    if n_sample < n:
        perm = perm[:n_sample]
    e = emb[perm]
    yy = y[perm]
    split = int(0.8 * len(perm))
    tr, te = np.arange(split), np.arange(split, len(perm))
    sims = e[te] @ e[tr].T
    nn = np.argpartition(-sims, knn, axis=1)[:, :knn]
    ytr, yte = yy[tr], yy[te]
    vals, cnts = np.unique(ytr, return_counts=True)
    maj = float((yte == vals[cnts.argmax()]).mean())
    preds = []
    for r in range(len(te)):
        v, c = np.unique(ytr[nn[r]], return_counts=True)
        preds.append(v[c.argmax()])
    acc = float((np.array(preds) == yte).mean())
    lift = (acc - maj) / (1 - maj) if maj < 1 else 0.0
    return {"acc": acc, "majority_acc": maj, "lift": float(lift)}


# ---------------------------------------------------------------------------
# SCORE 3 — Behavioral signal (seasonality + sell-through, controlled vs metadata)
# ---------------------------------------------------------------------------
def _cramers_v(cat: np.ndarray, other: np.ndarray) -> float:
    """Bias-corrected Cramer's V between two categorical arrays."""
    ct = pd.crosstab(cat, other).to_numpy()
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    n = ct.sum()
    row = ct.sum(1, keepdims=True)
    col = ct.sum(0, keepdims=True)
    exp = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((ct - exp) ** 2 / exp)
    r, k = ct.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def _eta(groups: np.ndarray, y: np.ndarray) -> float:
    """Correlation ratio eta (sqrt of between-group variance fraction), 0..1."""
    ybar = y.mean()
    ss_tot = ((y - ybar) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        m = groups == g
        ng = m.sum()
        if ng:
            ss_between += ng * (y[m].mean() - ybar) ** 2
    return float(np.sqrt(ss_between / ss_tot))


class BehaviorTables(NamedTuple):
    month_by_item: dict  # article_id -> month (mode of purchases) sample
    item_total: dict  # article_id -> total purchases
    txn_sample_articles: np.ndarray  # sampled per-purchase article ids
    txn_sample_months: np.ndarray  # matched purchase month


def _build_behavior_tables(canon: np.ndarray, txn_sample: int = 3_000_000) -> BehaviorTables:
    """Read train transactions once; build a per-purchase month sample (for seasonality)
    and per-item total purchases (for sell-through)."""
    tx = pd.read_parquet(TRAIN_TXN, columns=["t_dat", "article_id"])
    tx["article_id"] = tx["article_id"].astype(str)
    canon_set = set(canon.tolist())
    tx = tx[tx["article_id"].isin(canon_set)]
    # per-item total purchases
    item_total = tx["article_id"].value_counts().to_dict()
    # per-purchase month sample (seasonality is per-transaction)
    if len(tx) > txn_sample:
        tx = tx.sample(txn_sample, random_state=SEED)
    months = pd.to_datetime(tx["t_dat"]).dt.month.to_numpy()
    return BehaviorTables(
        month_by_item={},
        item_total=item_total,
        txn_sample_articles=tx["article_id"].to_numpy(),
        txn_sample_months=months,
    )


def behavioral_signal(
    attr_primary: dict,  # article_id -> primary attr value (str)
    bt: BehaviorTables,
    canon: np.ndarray,
    item_total_arr: np.ndarray,
    matched_meta: np.ndarray,  # per-item matched metadata value (str), aligned to canon
) -> dict:
    """Behavioral signal = excess of (seasonality Cramer's V) + (sell-through eta) over
    the SAME metrics computed on a matched metadata field. Returns components + score."""
    # ---- seasonality (per purchase) ----
    attr_per_txn = np.array([attr_primary.get(a, "NA") for a in bt.txn_sample_articles])
    season_attr = _cramers_v(attr_per_txn, bt.txn_sample_months)
    # matched metadata seasonality
    meta_map = {a: m for a, m in zip(canon, matched_meta)}
    meta_per_txn = np.array([meta_map.get(a, "NA") for a in bt.txn_sample_articles])
    season_meta = _cramers_v(meta_per_txn, bt.txn_sample_months)

    # ---- sell-through (per item, only items with sales) ----
    sold = item_total_arr > 0
    attr_per_item = np.array([attr_primary.get(a, "NA") for a in canon])
    y_log = np.log1p(item_total_arr[sold])
    sell_attr = _eta(attr_per_item[sold], y_log)
    sell_meta = _eta(matched_meta[sold], y_log)

    season_excess = max(0.0, season_attr - season_meta)
    sell_excess = max(0.0, sell_attr - sell_meta)
    # normalize each to 0..1 (Cramer's V & eta already 0..1) and average the two
    score = float((season_excess + sell_excess) / 2.0)
    return {
        "seasonality_cramers_v": round(float(season_attr), 4),
        "seasonality_meta": round(float(season_meta), 4),
        "seasonality_excess": round(float(season_excess), 4),
        "sellthrough_eta": round(float(sell_attr), 4),
        "sellthrough_meta": round(float(sell_meta), 4),
        "sellthrough_excess": round(float(sell_excess), 4),
        "behavioral_signal": round(score, 4),
    }


# ---------------------------------------------------------------------------
# Per-attribute screen
# ---------------------------------------------------------------------------
# Thresholds (pre-registered) for the verdict tags.
DISC_MIN = 0.55  # below => CONCENTRATED (one value dominates the entropy budget)
TOP1_MAX = 0.65  # above => CONCENTRATED (a single value owns >65% mass)
REDUNDANT_META_LIFT = 0.55  # metadata lift above => REDUNDANT with metadata
REDUNDANT_L1_LIFT = 0.55  # L1-BGE lift above => REDUNDANT with product
BEHAV_MIN = 0.02  # below => no excess behavioral structure over metadata


def screen_attribute(
    attr: str,
    fk: pd.DataFrame,
    canon: np.ndarray,
    meta_oh: np.ndarray,
    l1_bge: np.ndarray,
    bt: BehaviorTables,
    item_total_arr: np.ndarray,
    art: pd.DataFrame,
) -> dict:
    is_list = attr in LIST_COLS
    prim = fk[attr].map(_first).to_numpy()  # primary value (first elem if list)

    # SCORE 1 — discrimination on primary value
    disc, top1, n_vals = discrimination(prim)

    # SCORE 2 — non-redundancy: metadata predictability + L1-BGE predictability
    meta_pred = _cv_lift_logreg(meta_oh, prim)
    l1_pred = _knn_lift_l1(l1_bge, prim)
    nonredund = 1.0 - max(meta_pred["lift"], 0.0)

    # matched metadata control = the metadata col that best predicts the attribute
    # (single-col lift), used as the seasonality/sell-through control
    best_meta_col, best_meta_lift = None, -1.0
    for mc in META_COLS:
        if mc not in art.columns:
            continue
        mv = art[mc].astype(str).to_numpy()
        # cheap single-col predictability = 1 - normalized conditional entropy proxy:
        # use Cramer's V between attr primary and this metadata col as the match score
        v = _cramers_v(prim, mv)
        if v > best_meta_lift:
            best_meta_lift, best_meta_col = v, mc
    matched_meta = art[best_meta_col].astype(str).to_numpy()

    # SCORE 3 — behavioral signal (controlled vs matched metadata)
    attr_primary = {a: p for a, p in zip(canon, prim)}
    behav = behavioral_signal(attr_primary, bt, canon, item_total_arr, matched_meta)

    # composite salvageability: geometric-ish blend of the 3 principles (penalize any
    # near-zero pillar). behavioral is on a smaller scale, so we rescale it to ~0..1.
    behav_norm = min(1.0, behav["behavioral_signal"] / 0.15)
    composite = float((disc * nonredund * (0.25 + behav_norm)) ** (1 / 3))

    # ---- verdict ----
    concentrated = disc < DISC_MIN or top1 > TOP1_MAX
    redundant = meta_pred["lift"] >= REDUNDANT_META_LIFT or l1_pred["lift"] >= REDUNDANT_L1_LIFT
    has_behav = behav["behavioral_signal"] >= BEHAV_MIN
    discriminative = not concentrated
    nonredund_ok = not redundant

    if discriminative and nonredund_ok and has_behav:
        verdict = "SALVAGEABLE"
    elif concentrated:
        verdict = "CONCENTRATED"
    elif redundant:
        verdict = "REDUNDANT"
    else:
        verdict = "WEAK"

    return {
        "attribute": attr,
        "layer": attr.split("_")[0].upper(),
        "is_list": is_list,
        "n_values": n_vals,
        "discrimination": round(disc, 4),
        "top1_share": round(top1, 4),
        "metadata_predictability": round(meta_pred["lift"], 4),
        "metadata_acc": round(meta_pred["acc"], 4),
        "l1_predictability": round(l1_pred["lift"], 4),
        "nonredundancy": round(float(nonredund), 4),
        "matched_meta_col": best_meta_col,
        **behav,
        "composite": round(composite, 4),
        "verdict": verdict,
        "_flags": {
            "concentrated": bool(concentrated),
            "redundant": bool(redundant),
            "has_behavioral": bool(has_behav),
        },
    }


# ---------------------------------------------------------------------------
# GAPS summary
# ---------------------------------------------------------------------------
def build_gaps(screen: list[dict]) -> dict:
    by_attr = {r["attribute"]: r for r in screen}
    axes: list[dict] = []
    for axis, attrs in DECISION_AXES.items():
        present = [a for a in attrs if a in by_attr]
        # an axis is COVERED if any serving attr is SALVAGEABLE; WEAK if only WEAK/REDUNDANT
        verdicts = [by_attr[a]["verdict"] for a in present]
        best = (
            "SALVAGEABLE" if "SALVAGEABLE" in verdicts
            else "WEAK" if ("WEAK" in verdicts and verdicts)
            else "REDUNDANT" if "REDUNDANT" in verdicts
            else "CONCENTRATED" if "CONCENTRATED" in verdicts
            else "MISSING"
        )
        max_disc = max((by_attr[a]["discrimination"] for a in present), default=0.0)
        axes.append({
            "axis": axis,
            "serving_attrs": present,
            "best_verdict": best,
            "max_discrimination": round(max_disc, 4),
            "status": (
                "COVERED" if best == "SALVAGEABLE"
                else "WEAK/REDESIGN" if best in ("WEAK",)
                else "REDUNDANT-COLLAPSE" if best in ("REDUNDANT", "CONCENTRATED")
                else "MISSING"
            ),
        })
    # axes that NO existing catalog attribute serves at all (true gaps to extract NEW)
    missing_axes = [
        "trend-phase / hype-cycle (vs static 'Current/Classic')",
        "price-tier / value-perception (real $ tier, not 1-5 quality)",
        "fine-grained occasion (work/date/gym/formal — not 88% 'Everyday')",
        "outfit-pairing / co-purchase role (data-driven, not LLM-guessed 'Foundation')",
        "body-fit / size-intent (oversized/true-to-size/petite)",
        "care / practicality (machine-wash, wrinkle, durability)",
    ]
    weak_axes = [a["axis"] for a in axes if a["status"] in ("WEAK/REDESIGN", "REDUNDANT-COLLAPSE")]
    return {"per_axis": axes, "weak_or_collapsed_axes": weak_axes, "missing_decision_axes": missing_axes}


# ---------------------------------------------------------------------------
# Boxed table
# ---------------------------------------------------------------------------
def print_box(screen: list[dict], gaps: dict) -> None:
    ranked = sorted(screen, key=lambda r: -r["composite"])
    header = (
        f"{'ATTRIBUTE':<22}{'L':<3}{'DISC':>6}{'TOP1':>6}{'META_P':>8}"
        f"{'L1_P':>6}{'NONRED':>8}{'BEHAV':>7}{'COMP':>7}  VERDICT"
    )
    width = len(header) + 2
    line = "=" * width
    print("\n" + line, flush=True)
    print("  PROBE DE1 — CATALOG ATTRIBUTE SCREEN  (discrimination x non-redundancy x behavior)", flush=True)
    print(line, flush=True)
    print("  " + header, flush=True)
    print("  " + "-" * len(header), flush=True)
    for r in ranked:
        print(
            "  "
            f"{r['attribute']:<22}{r['layer'][1]:<3}"
            f"{r['discrimination']:>6.2f}{r['top1_share']:>6.2f}"
            f"{r['metadata_predictability']:>8.2f}{r['l1_predictability']:>6.2f}"
            f"{r['nonredundancy']:>8.2f}{r['behavioral_signal']:>7.3f}"
            f"{r['composite']:>7.3f}  {r['verdict']}",
            flush=True,
        )
    print(line, flush=True)

    salv = [r["attribute"] for r in ranked if r["verdict"] == "SALVAGEABLE"]
    conc = [r["attribute"] for r in ranked if r["verdict"] == "CONCENTRATED"]
    redu = [r["attribute"] for r in ranked if r["verdict"] == "REDUNDANT"]
    weak = [r["attribute"] for r in ranked if r["verdict"] == "WEAK"]

    print(f"  SALVAGEABLE ({len(salv)}): {', '.join(salv) if salv else '(none)'}", flush=True)
    print(f"  CONCENTRATED ({len(conc)}): {', '.join(conc) if conc else '(none)'}", flush=True)
    print(f"  REDUNDANT   ({len(redu)}): {', '.join(redu) if redu else '(none)'}", flush=True)
    print(f"  WEAK        ({len(weak)}): {', '.join(weak) if weak else '(none)'}", flush=True)
    print(f"  FREE-TEXT (not a facet, excluded): {', '.join(FREETEXT_COLS)}", flush=True)
    print("  " + "-" * len(header), flush=True)
    print("  GAPS — decision-axes weak/collapsed in existing catalog:", flush=True)
    for ax in gaps["per_axis"]:
        if ax["status"] != "COVERED":
            print(
                f"     [{ax['status']:<18}] {ax['axis']:<32} "
                f"(serving: {', '.join(ax['serving_attrs']) or 'none'}; best={ax['best_verdict']})",
                flush=True,
            )
    print("  GAPS — MISSING decision-axes to NEWLY extract:", flush=True)
    for m in gaps["missing_decision_axes"]:
        print(f"     - {m}", flush=True)
    print(line + "\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    n_items = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print("=" * 78, flush=True)
    print("PROBE DE1 — attribute screen (CPU-only, fixed seed=42, no API/$)", flush=True)
    print("  list-valued -> PRIMARY (first) value for discrimination/predictability;", flush=True)
    print("                 EXPLODED per-purchase for seasonality.", flush=True)
    print("=" * 78, flush=True)

    canon, fk, art = _load_aligned()
    if n_items is not None and n_items < len(canon):
        idx = RNG.permutation(len(canon))[:n_items]
        idx.sort()
        canon = canon[idx]
        fk = fk.iloc[idx].reset_index(drop=True)
        art = art.iloc[idx].reset_index(drop=True)
    print(f"Aligned catalog: {len(canon)} items", flush=True)

    meta_oh = _onehot_meta(art)
    l1_bge = _load_l1_bge(canon)
    print(f"Feature dims: metadata one-hot={meta_oh.shape[1]}, L1-BGE={l1_bge.shape[1]}", flush=True)

    print("Building behavior tables (train_transactions) ...", flush=True)
    bt = _build_behavior_tables(canon)
    item_total_arr = np.array([bt.item_total.get(a, 0) for a in canon], dtype=np.float64)
    print(
        f"  per-purchase month sample n={len(bt.txn_sample_months)}; "
        f"items with sales={(item_total_arr > 0).sum()}",
        flush=True,
    )

    screen: list[dict] = []
    for i, attr in enumerate(SCREEN_ATTRS, 1):
        r = screen_attribute(attr, fk, canon, meta_oh, l1_bge, bt, item_total_arr, art)
        screen.append(r)
        print(
            f"  [{i:2d}/{len(SCREEN_ATTRS)}] {attr:24s} "
            f"disc={r['discrimination']:.2f} top1={r['top1_share']:.2f} "
            f"meta_p={r['metadata_predictability']:.2f} l1_p={r['l1_predictability']:.2f} "
            f"behav={r['behavioral_signal']:.3f} -> {r['verdict']}",
            flush=True,
        )

    gaps = build_gaps(screen)
    print_box(screen, gaps)

    ranked = sorted(screen, key=lambda r: -r["composite"])
    result = {
        "probe": "DE1_attribute_screen",
        "seed": SEED,
        "n_items": int(len(canon)),
        "list_handling": "primary(first) value for discrimination/predictability; exploded per-purchase for seasonality",
        "freetext_excluded": FREETEXT_COLS,
        "thresholds": {
            "discrimination_min": DISC_MIN,
            "top1_share_max": TOP1_MAX,
            "redundant_meta_lift": REDUNDANT_META_LIFT,
            "redundant_l1_lift": REDUNDANT_L1_LIFT,
            "behavioral_min": BEHAV_MIN,
        },
        "per_attribute": {r["attribute"]: r for r in screen},
        "ranked": [r["attribute"] for r in ranked],
        "summary": {
            "SALVAGEABLE": [r["attribute"] for r in ranked if r["verdict"] == "SALVAGEABLE"],
            "CONCENTRATED": [r["attribute"] for r in ranked if r["verdict"] == "CONCENTRATED"],
            "REDUNDANT": [r["attribute"] for r in ranked if r["verdict"] == "REDUNDANT"],
            "WEAK": [r["attribute"] for r in ranked if r["verdict"] == "WEAK"],
        },
        "gaps": gaps,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Confirmation: JSON written -> {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
