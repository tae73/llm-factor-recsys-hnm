"""PROBE 21 — GATE-2': does an EXTERNAL-KNOWLEDGE KAR beat L1 and popularity on discovery?

MODEL — canonical Two-Tower (DSSM) content-based retrieval backbone.
    user-tower(content profile)  +  item-tower(L1 + external multi-view, item-side augmentation)
        -> shared d=64 embedding space  -> dot-product (cosine) scoring.
    This is the STANDARD content-based retrieval backbone:
        * DSSM — deep structured semantic two-tower (Huang et al., 2013).
        * YouTube two-tower deep candidate generation (Covington et al., 2016).
        * Sampled-softmax two-tower retrieval with in-batch negatives + logQ
          correction (Yi et al., 2019).
    We adopt the two-tower retrieval backbone (rather than a CF ranking backbone)
    because CF backbones FAIL on this Triple-Sparsity discovery task in full-catalog
    scoring: DeepFM MAP@12 = 0.0002 < popularity. The two-tower's content towers
    learn from item-side text/attribute features, so cold/sparse items with no CF
    signal still receive a meaningful embedding — exactly the regime this probe tests.

Path B. The de-risk two-tower (probe_18/20) VALIDATED that the learned BGE->rec projection +
item-side external augmentation (C_both) carries real, capacity-controlled discovery signal.
This probe SCALES that validated recipe into the proper Gate-2' test: a trainable two-tower
(DSSM) over the FULL external-knowledge catalog (prose + structured, item-side augmentation,
the validated C_both configuration), and asks the decisive research question —

    Does external knowledge (KAR_external) beat L1_only AND popularity on discovery_map@12,
    and does the win hold for the cold/sparse cohort (the central cold-start research claim)?

TRAINING OBJECTIVE — canonical two-tower in-batch sampled-softmax (Yi et al., 2019; default).
    For a batch of B (user, positive-item) pairs we score every user against ALL B in-batch
    positive items (the other B-1 act as shared negatives) and apply cross-entropy with the
    diagonal as the target. Logits are temperature-scaled dot products (tau=0.05). An optional
    logQ popularity correction subtracts log(P(item)) from the logits to debias toward popular
    in-batch negatives (Yi et al., 2019, eq. for corrected sampled softmax). A `--objective bpr`
    fallback retains the prior 1-pos/1-neg BPR loss for comparison.

EXTERNAL KNOWLEDGE SOURCE (auto-detect — forward-compat with the running FULL extraction)
    FULL    data/knowledge/external/external_knowledge_full.parquet
            (article_id, product_code, prose_text, structured_text)  -> full catalog coverage
    DERISK  fall back to the de-risk caches:
            complement_knowledge.parquet   (complement_text = prose)
            complement_structured.parquet  (unified_text    = structured)
            joined on article_id  (~3426-item coverage — smoke only)
    prose_text + structured_text are BGE-encoded SEPARATELY and CACHED to
    data/embeddings/external_prose.npz / external_struct.npz (keyed by article_id); reruns that
    already cover the source skip encoding entirely.

CONFIGS  (Two-Tower (DSSM), item-side augmentation, in-batch sampled-softmax, ~200 epochs, seeds=3)
    popularity      rank pool by recent-pop count          (static baseline, no training)
    L1_only         two-tower([l1])                  both sides
    KAR_external    two-tower([l1, prose, struct])   both sides (= validated C_both)

EVAL
    discovery_map@12 (src.evaluation.cohorts.discovery_map) on held-out users, mean over seeds.
    Broken down by COHORT: active (>=5 train purchases) vs sparse/cold (1-4 train purchases).

VERDICT
    GATE-2' GO iff KAR_external > L1_only AND KAR_external > popularity (mean over seeds), and
    KAR_external > L1_only in >=2/3 seeds.

Usage:
    uv run python witnesses/probe_21_gate2_external_kar.py [n_users] [epochs] [n_seeds] [objective]
        objective in {softmax (default), bpr}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation.cohorts import discovery_map  # noqa: E402
from witnesses._probe_common import OUT_DIR, load_variants  # noqa: E402

TRAIN_TXN = Path("data/processed/train_transactions.parquet")
GT_PATH = Path("data/processed/immediate_ground_truth.json")
ARTICLES = Path("data/processed/articles.parquet")

FULL_SRC = Path("data/knowledge/external/external_knowledge_full.parquet")
DERISK_PROSE = Path("data/knowledge/external/complement_knowledge.parquet")
DERISK_STRUCT = Path("data/knowledge/external/complement_structured.parquet")
PROSE_CACHE = Path("data/embeddings/external_prose.npz")
STRUCT_CACHE = Path("data/embeddings/external_struct.npz")

K = 12
POOL = 5000
MIN_COVERED = 3
ACTIVE_MIN = 5  # >=5 train purchases -> active; 1-4 -> sparse/cold

BGE_MODEL = "BAAI/bge-base-en-v1.5"

# Two-tower (DSSM) retrieval objective hyperparameters (Yi et al., 2019).
TAU = 0.05          # temperature on cosine/dot logits
SOFTMAX_BATCH = 512  # B: (user, positive-item) pairs per in-batch-softmax step
USE_LOGQ = True      # logQ popularity correction for in-batch sampled softmax


def _load_external_source(id_to_idx: dict[str, int]) -> tuple[dict[str, str], dict[str, str], str]:
    """Auto-detect the external-knowledge source.

    Returns ({aid: prose_text}, {aid: struct_text}, source_used) restricted to aids in the
    canonical L1 order. FULL source is keyed on product_code -> expand to every article_id
    sharing that product_code (knowledge is product-level). DERISK caches are keyed on article_id.
    """
    if FULL_SRC.exists():
        df = pd.read_parquet(FULL_SRC)
        cols = set(df.columns)
        art = pd.read_parquet(ARTICLES, columns=["article_id", "product_code"])
        art["article_id"] = art["article_id"].astype(str)
        art["product_code"] = art["product_code"].astype(str)
        if "article_id" in cols and df["article_id"].notna().any():
            # article_id-keyed full table
            df["article_id"] = df["article_id"].astype(str)
            prose = dict(zip(df["article_id"], df["prose_text"].astype(str)))
            struct = dict(zip(df["article_id"], df["structured_text"].astype(str)))
        else:
            # product_code-keyed -> expand to article_ids
            df["product_code"] = df["product_code"].astype(str)
            pc_prose = dict(zip(df["product_code"], df["prose_text"].astype(str)))
            pc_struct = dict(zip(df["product_code"], df["structured_text"].astype(str)))
            prose, struct = {}, {}
            for aid, pc in zip(art["article_id"], art["product_code"]):
                if pc in pc_prose:
                    prose[aid] = pc_prose[pc]
                    struct[aid] = pc_struct.get(pc, "")
        prose = {a: t for a, t in prose.items() if a in id_to_idx and t}
        struct = {a: struct.get(a, "") for a in prose}
        return prose, struct, "full"

    # de-risk fallback
    pr = pd.read_parquet(DERISK_PROSE)
    st = pd.read_parquet(DERISK_STRUCT)
    pr["article_id"] = pr["article_id"].astype(str)
    st["article_id"] = st["article_id"].astype(str)
    merged = pr.merge(st, on="article_id", how="inner")
    prose = {a: t for a, t in zip(merged["article_id"], merged["complement_text"].astype(str))
             if a in id_to_idx and t}
    struct = dict(zip(merged["article_id"], merged["unified_text"].astype(str)))
    struct = {a: struct.get(a, "") for a in prose}
    return prose, struct, "derisk-cache"


def _encode_cached(
    text_map: dict[str, str], cache: Path, dev: str, D: int
) -> dict[str, np.ndarray]:
    """BGE-encode ``text_map`` (keyed by article_id), reusing ``cache`` if it covers the source.

    Cache covers the source iff every needed article_id is present. Otherwise (re)encode and
    persist. Returns {article_id: normalized f32 vector}.
    """
    need = set(text_map)
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        ids = d["article_ids"].astype(str)
        arr = d["embeddings"].astype(np.float32)
        cached = {a: arr[i] for i, a in enumerate(ids)}
        if need <= set(cached) and arr.shape[1] == D:
            print(f"  [cache HIT] {cache.name}: {len(cached)} vecs cover {len(need)} needed")
            return {a: cached[a] for a in need}
        print(f"  [cache STALE] {cache.name}: covers {len(set(cached) & need)}/{len(need)} -> re-encode")

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(BGE_MODEL, device=dev)
    enc.max_seq_length = 512
    ids = list(text_map)
    arr = enc.encode([text_map[a] for a in ids], batch_size=128, normalize_embeddings=True,
                     show_progress_bar=False).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, embeddings=arr, article_ids=np.array(ids, dtype=object))
    print(f"  [encoded + cached] {cache.name}: {len(ids)} vecs")
    return {a: arr[i] for i, a in enumerate(ids)}


def main() -> None:
    # positional: [n_users] [epochs] [n_seeds] [objective]; --objective {softmax,bpr} also accepted
    argv = [a for a in sys.argv[1:]]
    objective = "softmax"
    pos = []
    for a in argv:
        if a.startswith("--objective"):
            objective = a.split("=", 1)[1] if "=" in a else "softmax"
        elif a in ("softmax", "bpr"):
            objective = a
        else:
            pos.append(a)
    n_users = int(pos[0]) if len(pos) > 0 else 12000
    epochs = int(pos[1]) if len(pos) > 1 else 200
    n_seeds = int(pos[2]) if len(pos) > 2 else 3
    if objective not in ("softmax", "bpr"):
        raise ValueError(f"objective must be 'softmax' or 'bpr', got {objective!r}")
    print(f"[probe_21] Two-Tower (DSSM) objective = {objective}")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    canon_ids, V = load_variants(["L1"])
    l1 = V["L1"].astype(np.float32)
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}
    D = l1.shape[1]

    prose_text, struct_text, source_used = _load_external_source(id_to_idx)
    print(f"[probe_21] external source = {source_used}  texts={len(prose_text)}")
    prose_emb = _encode_cached(prose_text, PROSE_CACHE, dev, D)
    struct_emb = _encode_cached(struct_text, STRUCT_CACHE, dev, D)
    cov_set = set(prose_emb) & set(struct_emb)
    print(f"[probe_21] external coverage (prose AND struct) = {len(cov_set)} items")

    # ---- immediate-split discovery setup (identical to probe_20) -----------------------------
    import duckdb
    con = duckdb.connect()
    pop_rows = con.execute(
        f"SELECT article_id, count(*) c FROM read_parquet('{TRAIN_TXN}') "
        f"WHERE t_dat >= DATE '2020-05-01' GROUP BY article_id ORDER BY c DESC LIMIT {POOL}"
    ).fetchall()
    pool_ids = [str(a) for (a, _) in pop_rows if str(a) in id_to_idx]
    pop_count = {str(a): float(c) for (a, c) in pop_rows}
    P = len(pool_ids)
    pool_l1 = np.stack([l1[id_to_idx[a]] for a in pool_ids])
    pool_pr = np.stack([prose_emb.get(a, np.zeros(D, np.float32)) for a in pool_ids])
    pool_st = np.stack([struct_emb.get(a, np.zeros(D, np.float32)) for a in pool_ids])
    pool_pop = np.array([pop_count.get(a, 0.0) for a in pool_ids], dtype=np.float32)
    pool_pos = {a: i for i, a in enumerate(pool_ids)}
    # logQ correction term (Yi et al., 2019): logits -= log P(item) so frequent
    # in-batch negatives are penalized. P(item) estimated from recent-pop counts.
    _pp = pool_pop + 1.0  # Laplace-smooth zero-pop pool items
    pool_logq = np.log(_pp / _pp.sum()).astype(np.float32)

    gt = {str(k): [str(x) for x in v] for k, v in json.loads(GT_PATH.read_text()).items()}
    con.register("gt_users", pd.DataFrame({"customer_id": list(gt)}))
    hist_rows = con.execute(
        f"SELECT t.customer_id, list(DISTINCT t.article_id) FROM read_parquet('{TRAIN_TXN}') t "
        f"JOIN gt_users g ON t.customer_id=g.customer_id GROUP BY t.customer_id"
    ).fetchall()
    train_history = {str(cid): set(str(a) for a in items) for cid, items in hist_rows}

    rng0 = np.random.default_rng(42)
    eligible = []
    for u, h in train_history.items():
        if u in gt and len(h & cov_set) >= MIN_COVERED and [g for g in gt[u] if g not in h and g in pool_pos]:
            eligible.append(u)
    rng0.shuffle(eligible)
    eligible = eligible[:n_users]
    cut = int(0.7 * len(eligible))
    tr_users, te_users = eligible[:cut], eligible[cut:]
    cov = {u: [h for h in train_history[u] if h in cov_set] for u in eligible}

    def prof(emb, u):
        p = np.stack([emb[h] for h in cov[u]]).mean(0)
        return (p / (np.linalg.norm(p) + 1e-9)).astype(np.float32)

    l1p = {u: prof({h: l1[id_to_idx[h]] for h in cov[u]}, u) for u in eligible}
    prp = {u: prof(prose_emb, u) for u in eligible}
    stp = {u: prof(struct_emb, u) for u in eligible}
    tr_pos = {u: [pool_pos[g] for g in gt[u] if g not in train_history[u] and g in pool_pos] for u in tr_users}
    te_owned = {u: np.array([a in train_history[u] for a in pool_ids]) for u in te_users}
    ulist = [u for u in tr_users if tr_pos[u]]
    pos_lists = [np.asarray(tr_pos[u]) for u in ulist]

    # cohort masks on the EVAL users (by full train-history size)
    cohort = {u: ("active" if len(train_history[u]) >= ACTIVE_MIN else "sparse") for u in te_users}
    te_active = [u for u in te_users if cohort[u] == "active"]
    te_sparse = [u for u in te_users if cohort[u] == "sparse"]
    print(f"[probe_21] eligible={len(eligible)} train={len(ulist)} eval={len(te_users)} "
          f"(active={len(te_active)} sparse={len(te_sparse)})  "
          f"pool_ext_cov={int((np.abs(pool_pr).sum(1) > 0).sum())}/{P}")

    # config -> (item_feat blocks, user_prof blocks); "popularity" is special-cased (no training)
    CONFIGS = {
        "L1_only": ([pool_l1], [l1p]),
        "KAR_external": ([pool_l1, pool_pr, pool_st], [l1p, prp, stp]),
    }

    class Tower(nn.Module):
        """One side of the Two-Tower (DSSM) retrieval model: MLP([din])->128->d=64.

        Both user-tower and item-tower share this architecture; they map their
        respective content features into a SHARED d-dim space where relevance is the
        dot product (Huang et al., 2013; Covington et al., 2016; Yi et al., 2019).
        """

        def __init__(self, din, d=64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, d))

        def forward(self, x):
            return self.net(x)

    def _discovery_by_cohort(preds: dict[str, list[str]]) -> dict[str, float]:
        """discovery_map@12 overall + per cohort from a single prediction dict."""
        out = {}
        for name, users in (("overall", te_users), ("active", te_active), ("sparse", te_sparse)):
            if not users:
                out[name] = float("nan")
                continue
            d = discovery_map({u: preds[u] for u in users}, {u: gt[u] for u in users},
                              {u: train_history[u] for u in users}, k=K)
            out[name] = d.map_at_k
        return out

    def eval_popularity() -> dict[str, float]:
        """Static popularity ranking of the pool (no training, seed-invariant)."""
        order_all = np.argsort(-pool_pop)
        preds = {}
        for u in te_users:
            owned = te_owned[u]
            rec = [pool_ids[i] for i in order_all if not owned[i]][:K]
            preds[u] = rec
        return _discovery_by_cohort(preds)

    logq_t = torch.tensor(pool_logq, device=dev)

    def _predict(utow, itow, item_feat, uf_te) -> dict[str, list[str]]:
        utow.eval()
        itow.eval()
        with torch.no_grad():
            i_emb = itow(item_feat)
            ue_te = utow(torch.tensor(uf_te, device=dev))
            sc = (ue_te @ i_emb.T).cpu().numpy()
        preds = {}
        for bi, u in enumerate(te_users):
            s = np.where(te_owned[u], -np.inf, sc[bi])
            order = np.argpartition(-s, K)[:K]
            preds[u] = [pool_ids[i] for i in order[np.argsort(-s[order])]]
        return preds

    def train_eval(item_blocks, user_blocks, seed) -> dict[str, float]:
        """Train one Two-Tower (DSSM) config and return discovery_map@12 per cohort."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        item_feat = torch.cat([torch.tensor(b, device=dev) for b in item_blocks], 1)
        din = D * len(item_blocks)
        uf = np.stack([np.concatenate([ub[u] for ub in user_blocks]) for u in ulist])
        uf_te = np.stack([np.concatenate([ub[u] for ub in user_blocks]) for u in te_users])
        utow, itow = Tower(din).to(dev), Tower(din).to(dev)
        opt = torch.optim.Adam(list(utow.parameters()) + list(itow.parameters()),
                               lr=1e-3, weight_decay=1e-5)
        uvecs = torch.tensor(uf, device=dev)
        Bu = len(ulist)

        if objective == "bpr":
            # Fallback: prior 1-pos / 1-neg BPR over all train users each epoch.
            for _ in range(epochs):
                pos_idx = torch.tensor([int(pl[rng.integers(len(pl))]) for pl in pos_lists],
                                       device=dev)
                neg_idx = torch.tensor(rng.integers(0, P, size=Bu), device=dev)
                u_emb = utow(uvecs)
                pemb = itow(item_feat[pos_idx])
                nemb = itow(item_feat[neg_idx])
                loss = -torch.log(
                    torch.sigmoid((u_emb * pemb).sum(1) - (u_emb * nemb).sum(1)) + 1e-9
                ).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            return _discovery_by_cohort(_predict(utow, itow, item_feat, uf_te))

        # Canonical two-tower in-batch sampled-softmax (Yi et al., 2019).
        # Build a flat list of (user-row, pool-item) discovery-positive pairs.
        pair_u = np.repeat(np.arange(Bu), [len(pl) for pl in pos_lists])
        pair_i = np.concatenate(pos_lists).astype(np.int64)
        n_pairs = len(pair_u)
        B = min(SOFTMAX_BATCH, n_pairs)
        steps_per_epoch = max(1, n_pairs // B)
        for _ in range(epochs):
            perm = rng.permutation(n_pairs)
            for s in range(steps_per_epoch):
                sel = perm[s * B:(s + 1) * B]
                if len(sel) < 2:
                    continue
                ui = torch.tensor(pair_u[sel], device=dev)
                ii = torch.tensor(pair_i[sel], device=dev)
                u_emb = utow(uvecs[ui])              # (b, d)
                i_emb = itow(item_feat[ii])          # (b, d)  in-batch positives = negatives
                # cosine logits / temperature
                u_n = torch.nn.functional.normalize(u_emb, dim=1)
                i_n = torch.nn.functional.normalize(i_emb, dim=1)
                logits = (u_n @ i_n.T) / TAU         # (b, b)
                if USE_LOGQ:
                    logits = logits - logq_t[ii].unsqueeze(0)  # subtract log P(item) per column
                target = torch.arange(len(sel), device=dev)    # diagonal = the true positive
                loss = torch.nn.functional.cross_entropy(logits, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return _discovery_by_cohort(_predict(utow, itow, item_feat, uf_te))

    # ---- run -------------------------------------------------------------------------------
    cohorts = ("overall", "active", "sparse")
    per = {c: {coh: [] for coh in cohorts} for c in ["popularity", *CONFIGS]}

    pop_res = eval_popularity()  # seed-invariant; replicate across seeds for paired structure
    for coh in cohorts:
        per["popularity"][coh] = [pop_res[coh]] * n_seeds

    for seed in range(n_seeds):
        for c, (ib, ub) in CONFIGS.items():
            r = train_eval(ib, ub, seed)
            for coh in cohorts:
                per[c][coh].append(r[coh])
        line = "  ".join(f"{c}(o/a/s)={per[c]['overall'][-1]:.5f}/{per[c]['active'][-1]:.5f}/"
                         f"{per[c]['sparse'][-1]:.5f}" for c in CONFIGS)
        print(f"  seed {seed}: {line}")

    def _summ(vals):
        a = np.array([v for v in vals if not np.isnan(v)], dtype=float)
        if a.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "per_seed": [float(x) for x in vals]}
        return {"mean": float(a.mean()), "std": float(a.std()), "per_seed": [float(x) for x in vals]}

    config_summary = {
        c: {coh: _summ(per[c][coh]) for coh in cohorts} for c in ["popularity", *CONFIGS]
    }

    kar = np.array(per["KAR_external"]["overall"])
    l1o = np.array(per["L1_only"]["overall"])
    pop_o = config_summary["popularity"]["overall"]["mean"]
    kar_mean = config_summary["KAR_external"]["overall"]["mean"]
    l1_mean = config_summary["L1_only"]["overall"]["mean"]

    kar_vs_l1_delta = kar_mean - l1_mean
    kar_vs_l1_rel = (kar_vs_l1_delta / l1_mean) if l1_mean > 0 else 0.0
    kar_vs_pop_delta = kar_mean - pop_o
    kar_vs_pop_rel = (kar_vs_pop_delta / pop_o) if pop_o > 0 else 0.0
    sign_kar_l1 = int((kar > l1o).sum())

    go = (kar_mean > l1_mean and kar_mean > pop_o and sign_kar_l1 >= 2)
    verdict = (
        "GATE-2' GO — external-knowledge KAR beats L1 and popularity on discovery "
        f"(KAR {kar_mean:.5f} > L1 {l1_mean:.5f} [{kar_vs_l1_rel * 100:+.1f}%], > pop {pop_o:.5f}; "
        f"KAR>L1 in {sign_kar_l1}/{n_seeds} seeds)"
        if go else
        "GATE-2' NO-GO — external-knowledge KAR does not robustly beat L1/popularity on discovery "
        f"(KAR {kar_mean:.5f} vs L1 {l1_mean:.5f}, pop {pop_o:.5f}; KAR>L1 {sign_kar_l1}/{n_seeds} seeds)"
    )

    out = {
        "probe": "probe_21_gate2_external_kar",
        "model": "two_tower_dssm",
        "objective": objective,
        "tau": TAU if objective == "softmax" else None,
        "logq_correction": bool(USE_LOGQ) if objective == "softmax" else None,
        "softmax_batch": SOFTMAX_BATCH if objective == "softmax" else None,
        "source_used": source_used,
        "coverage": len(cov_set),
        "n_eligible": len(eligible),
        "n_train": len(ulist),
        "n_eval": len(te_users),
        "n_eval_active": len(te_active),
        "n_eval_sparse": len(te_sparse),
        "epochs": epochs,
        "n_seeds": n_seeds,
        "configs": config_summary,
        "KAR_external_vs_L1_only": {
            "delta": kar_vs_l1_delta, "rel": kar_vs_l1_rel, "sign_seeds": sign_kar_l1,
        },
        "KAR_external_vs_popularity": {
            "delta": kar_vs_pop_delta, "rel": kar_vs_pop_rel,
        },
        "verdict": verdict,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_21_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    obj_tag = (f"in-batch sampled-softmax (tau={TAU}, logQ={USE_LOGQ}, B={SOFTMAX_BATCH})"
               if objective == "softmax" else "BPR 1-pos/1-neg")
    print("=" * 84)
    print("  PROBE 21 — GATE-2': external-knowledge KAR vs L1 vs popularity (discovery)")
    print("  MODEL: Two-Tower (DSSM)  |  OBJECTIVE: " + obj_tag)
    print("=" * 84)
    print(f"  source={source_used}  coverage={len(cov_set)}  "
          f"eligible={len(eligible)}  eval={len(te_users)} (active={len(te_active)} sparse={len(te_sparse)})")
    hdr = f"  {'config':14s} | {'overall':>16s} | {'active':>16s} | {'sparse/cold':>16s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in ["popularity", "L1_only", "KAR_external"]:
        cells = []
        for coh in cohorts:
            s = config_summary[c][coh]
            cells.append(f"{s['mean']:.5f}±{s['std']:.5f}")
        print(f"  {c:14s} | {cells[0]:>16s} | {cells[1]:>16s} | {cells[2]:>16s}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  KAR_external - L1_only     = {kar_vs_l1_delta:+.5f}  ({kar_vs_l1_rel * 100:+.1f}%)  "
          f"sign {sign_kar_l1}/{n_seeds}")
    print(f"  KAR_external - popularity  = {kar_vs_pop_delta:+.5f}  ({kar_vs_pop_rel * 100:+.1f}%)")
    print(f"  VERDICT: {verdict}")
    print("=" * 84)


if __name__ == "__main__":
    main()
