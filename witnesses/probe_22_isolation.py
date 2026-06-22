"""PROBE 22 — CONTROLLED ISOLATION: population bias vs coverage artifact.

WHY — external-knowledge KAR FLIPPED verdict between two de-risk regimes:
    * DE-RISK   (3.2% external coverage, 3426 popular items) -> GO  (KAR_external +27..43% over L1)
    * FULL      (100% external coverage, 105494 items)       -> NO-GO (KAR 0.00424 vs L1 0.00482, -12%, 0/3)
Two changes were CONFOUNDED between those runs:
    (1) COVERAGE   3.2% -> 100% of items carry external vectors (changes the candidate-pool item features).
    (2) POPULATION the "eligible" users differ. De-risk eligible = users with >=3 history items AMONG the
        3426 covered (popular) items -> biased toward HEAVY buyers of POPULAR items. Full eligible = all
        users with >=3 history items -> representative.

THIS PROBE runs "cell B" of the 2x2: the DE-RISK POPULATION but at FULL external coverage, to isolate
POPULATION from COVERAGE. It ADAPTS probe_21 (same data loading, same Two-Tower (DSSM) sampled-softmax
training, same discovery_map eval, same cohort logic, same multi-seed) and changes ONLY the
eligible-user filter via --population:

    population="derisk":  eligible = users whose history contains >=3 items in the ORIGINAL DE-RISK SET
        (the article_ids in complement_knowledge.parquet, ~3426 items) — replicating the de-risk eligible
        criterion — BUT external embeddings are loaded from the FULL source (external_knowledge_full.parquet)
        and the candidate pool / profiles use FULL coverage. Everything else identical to probe_21 full.
    population="full":    same as probe_21 full (eligible = users with >=3 history items in FULL cov_set).
        Sanity-check that it reproduces NO-GO.

DECISION LOGIC
    * derisk-population GO  while full-population NO-GO  -> POPULATION bias is the driver (the de-risk
      eligible subset was special — heavy buyers of popular items).
    * derisk-population ALSO NO-GO at full coverage      -> the original de-risk GO was a COVERAGE ARTIFACT
      (28%-pool-coverage: external acted as a coverage/popularity proxy), NOT a real population effect.

Usage:
    uv run python witnesses/probe_22_isolation.py [n_users] [epochs] [n_seeds]
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
DERISK_PROSE = Path("data/knowledge/external/complement_knowledge.parquet")  # the ORIGINAL de-risk set
DERISK_STRUCT = Path("data/knowledge/external/complement_structured.parquet")
PROSE_CACHE = Path("data/embeddings/external_prose.npz")
STRUCT_CACHE = Path("data/embeddings/external_struct.npz")

K = 12
POOL = 5000
MIN_COVERED = 3
ACTIVE_MIN = 5  # >=5 train purchases -> active; 1-4 -> sparse/cold

BGE_MODEL = "BAAI/bge-base-en-v1.5"

# Two-tower (DSSM) retrieval objective hyperparameters (Yi et al., 2019) — identical to probe_21.
TAU = 0.05
SOFTMAX_BATCH = 512
USE_LOGQ = True

# The two populations to compare (cell B vs the probe_21-full sanity replicate).
POPULATIONS = ("derisk", "full")


def _load_full_external(id_to_idx: dict[str, int]) -> tuple[dict[str, str], dict[str, str]]:
    """Load the FULL external-knowledge source (FULL coverage), keyed by article_id.

    Identical loading to probe_21's `_load_external_source` FULL branch. FULL source is
    product_code-level knowledge -> expand to every article_id sharing that product_code.
    Returns ({aid: prose_text}, {aid: struct_text}) restricted to aids in the canonical L1 order.
    """
    df = pd.read_parquet(FULL_SRC)
    cols = set(df.columns)
    art = pd.read_parquet(ARTICLES, columns=["article_id", "product_code"])
    art["article_id"] = art["article_id"].astype(str)
    art["product_code"] = art["product_code"].astype(str)
    if "article_id" in cols and df["article_id"].notna().any():
        df["article_id"] = df["article_id"].astype(str)
        prose = dict(zip(df["article_id"], df["prose_text"].astype(str)))
        struct = dict(zip(df["article_id"], df["structured_text"].astype(str)))
    else:
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
    return prose, struct


def _load_derisk_set(id_to_idx: dict[str, int]) -> set[str]:
    """The ORIGINAL de-risk set — article_ids in complement_knowledge.parquet (~3426 items).

    This is the set the DE-RISK eligible criterion was defined against. Restricted to canonical L1 order.
    """
    pr = pd.read_parquet(DERISK_PROSE)
    pr["article_id"] = pr["article_id"].astype(str)
    return {a for a in pr["article_id"] if a in id_to_idx}


def _encode_cached(
    text_map: dict[str, str], cache: Path, dev: str, D: int
) -> dict[str, np.ndarray]:
    """BGE-encode ``text_map`` (keyed by article_id), reusing ``cache`` if it covers the source.

    Identical to probe_21._encode_cached.
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
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_users = int(pos[0]) if len(pos) > 0 else 40000
    epochs = int(pos[1]) if len(pos) > 1 else 200
    n_seeds = int(pos[2]) if len(pos) > 2 else 3
    print(f"[probe_22] controlled isolation — Two-Tower (DSSM) sampled-softmax "
          f"(tau={TAU}, logQ={USE_LOGQ}, B={SOFTMAX_BATCH})")
    print(f"[probe_22] n_users={n_users} epochs={epochs} n_seeds={n_seeds}")

    import torch
    import torch.nn as nn

    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"[probe_22] device = {dev}")

    canon_ids, V = load_variants(["L1"])
    l1 = V["L1"].astype(np.float32)
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}
    D = l1.shape[1]

    # FULL external coverage for BOTH populations (the controlled variable held at FULL).
    prose_text, struct_text = _load_full_external(id_to_idx)
    print(f"[probe_22] external source = full  texts={len(prose_text)}")
    prose_emb = _encode_cached(prose_text, PROSE_CACHE, dev, D)
    struct_emb = _encode_cached(struct_text, STRUCT_CACHE, dev, D)
    full_cov_set = set(prose_emb) & set(struct_emb)
    print(f"[probe_22] FULL external coverage (prose AND struct) = {len(full_cov_set)} items")

    # The ORIGINAL de-risk set — used ONLY for the derisk-population eligibility filter.
    derisk_set = _load_derisk_set(id_to_idx) & full_cov_set
    print(f"[probe_22] ORIGINAL de-risk set (eligibility filter) = {len(derisk_set)} items")

    # ---- immediate-split discovery setup (identical to probe_21) -----------------------------
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
    # FULL coverage on the candidate pool for BOTH populations.
    pool_pr = np.stack([prose_emb.get(a, np.zeros(D, np.float32)) for a in pool_ids])
    pool_st = np.stack([struct_emb.get(a, np.zeros(D, np.float32)) for a in pool_ids])
    pool_pop = np.array([pop_count.get(a, 0.0) for a in pool_ids], dtype=np.float32)
    pool_pos = {a: i for i, a in enumerate(pool_ids)}
    _pp = pool_pop + 1.0
    pool_logq = np.log(_pp / _pp.sum()).astype(np.float32)
    pool_full_cov_frac = float((np.abs(pool_pr).sum(1) > 0).mean())
    pool_derisk_cov_frac = float(np.mean([1.0 if a in derisk_set else 0.0 for a in pool_ids]))
    print(f"[probe_22] candidate pool P={P}  external-cov(full)={pool_full_cov_frac * 100:.1f}%  "
          f"(of which in de-risk set={pool_derisk_cov_frac * 100:.1f}%)")

    gt = {str(k): [str(x) for x in v] for k, v in json.loads(GT_PATH.read_text()).items()}
    con.register("gt_users", pd.DataFrame({"customer_id": list(gt)}))
    hist_rows = con.execute(
        f"SELECT t.customer_id, list(DISTINCT t.article_id) FROM read_parquet('{TRAIN_TXN}') t "
        f"JOIN gt_users g ON t.customer_id=g.customer_id GROUP BY t.customer_id"
    ).fetchall()
    train_history = {str(cid): set(str(a) for a in items) for cid, items in hist_rows}

    class Tower(nn.Module):
        """One side of the Two-Tower (DSSM): MLP([din])->128->d=64. Identical to probe_21."""

        def __init__(self, din, d=64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, d))

        def forward(self, x):
            return self.net(x)

    logq_t = torch.tensor(pool_logq, device=dev)

    def run_population(elig_set: set[str], pop_name: str) -> dict:
        """Train+eval BOTH configs for one eligibility set; return the full per-population result.

        ONLY DIFFERENCE between populations is ``elig_set`` (the >=MIN_COVERED membership set used to
        decide eligibility AND to build the content profile `cov[u]`):
            * derisk -> the ORIGINAL de-risk set (~3426 popular items)
            * full   -> the FULL external coverage set (~105K items)  == probe_21 full
        Embeddings, candidate pool, training objective, eval are FULL/identical in both.
        """
        rng0 = np.random.default_rng(42)
        eligible = []
        for u, h in train_history.items():
            if (u in gt and len(h & elig_set) >= MIN_COVERED
                    and [g for g in gt[u] if g not in h and g in pool_pos]):
                eligible.append(u)
        rng0.shuffle(eligible)
        eligible = eligible[:n_users]
        cut = int(0.7 * len(eligible))
        tr_users, te_users = eligible[:cut], eligible[cut:]
        # content profile built from the user's history items that fall in the eligibility set
        # (mirrors probe_21, where cov[u] uses cov_set). For derisk this = de-risk items; for full
        # this = full-covered items — but external embeddings themselves are FULL in both.
        cov = {u: [h for h in train_history[u] if h in elig_set] for u in eligible}

        def prof(emb, u):
            p = np.stack([emb[h] for h in cov[u]]).mean(0)
            return (p / (np.linalg.norm(p) + 1e-9)).astype(np.float32)

        l1p = {u: prof({h: l1[id_to_idx[h]] for h in cov[u]}, u) for u in eligible}
        prp = {u: prof(prose_emb, u) for u in eligible}
        stp = {u: prof(struct_emb, u) for u in eligible}
        tr_pos = {u: [pool_pos[g] for g in gt[u] if g not in train_history[u] and g in pool_pos]
                  for u in tr_users}
        te_owned = {u: np.array([a in train_history[u] for a in pool_ids]) for u in te_users}
        ulist = [u for u in tr_users if tr_pos[u]]
        pos_lists = [np.asarray(tr_pos[u]) for u in ulist]

        cohort = {u: ("active" if len(train_history[u]) >= ACTIVE_MIN else "sparse") for u in te_users}
        te_active = [u for u in te_users if cohort[u] == "active"]
        te_sparse = [u for u in te_users if cohort[u] == "sparse"]
        print(f"[probe_22:{pop_name}] eligible={len(eligible)} train={len(ulist)} eval={len(te_users)} "
              f"(active={len(te_active)} sparse={len(te_sparse)})")

        CONFIGS = {
            "L1_only": ([pool_l1], [l1p]),
            "KAR_external": ([pool_l1, pool_pr, pool_st], [l1p, prp, stp]),
        }

        def _discovery_by_cohort(preds: dict[str, list[str]]) -> dict[str, float]:
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
            order_all = np.argsort(-pool_pop)
            preds = {}
            for u in te_users:
                owned = te_owned[u]
                rec = [pool_ids[i] for i in order_all if not owned[i]][:K]
                preds[u] = rec
            return _discovery_by_cohort(preds)

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

            # Canonical two-tower in-batch sampled-softmax (Yi et al., 2019). Identical to probe_21.
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
                    u_emb = utow(uvecs[ui])
                    i_emb = itow(item_feat[ii])
                    u_n = torch.nn.functional.normalize(u_emb, dim=1)
                    i_n = torch.nn.functional.normalize(i_emb, dim=1)
                    logits = (u_n @ i_n.T) / TAU
                    if USE_LOGQ:
                        logits = logits - logq_t[ii].unsqueeze(0)
                    target = torch.arange(len(sel), device=dev)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            return _discovery_by_cohort(_predict(utow, itow, item_feat, uf_te))

        cohorts = ("overall", "active", "sparse")
        per = {c: {coh: [] for coh in cohorts} for c in ["popularity", *CONFIGS]}
        pop_res = eval_popularity()
        for coh in cohorts:
            per["popularity"][coh] = [pop_res[coh]] * n_seeds
        for seed in range(n_seeds):
            for c, (ib, ub) in CONFIGS.items():
                r = train_eval(ib, ub, seed)
                for coh in cohorts:
                    per[c][coh].append(r[coh])
            line = "  ".join(f"{c}(o/a/s)={per[c]['overall'][-1]:.5f}/{per[c]['active'][-1]:.5f}/"
                             f"{per[c]['sparse'][-1]:.5f}" for c in CONFIGS)
            print(f"  [{pop_name}] seed {seed}: {line}")

        def _summ(vals):
            a = np.array([v for v in vals if not np.isnan(v)], dtype=float)
            if a.size == 0:
                return {"mean": float("nan"), "std": float("nan"),
                        "per_seed": [float(x) for x in vals]}
            return {"mean": float(a.mean()), "std": float(a.std()),
                    "per_seed": [float(x) for x in vals]}

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

        return {
            "population": pop_name,
            "n_eligible": len(eligible),
            "n_train": len(ulist),
            "n_eval": len(te_users),
            "n_eval_active": len(te_active),
            "n_eval_sparse": len(te_sparse),
            "configs": config_summary,
            "KAR_external_vs_L1_only": {
                "delta": kar_vs_l1_delta, "rel": kar_vs_l1_rel, "sign_seeds": sign_kar_l1,
                "sign": ("+" if kar_vs_l1_delta > 0 else ("-" if kar_vs_l1_delta < 0 else "0")),
            },
            "KAR_external_vs_popularity": {
                "delta": kar_vs_pop_delta, "rel": kar_vs_pop_rel,
                "sign": ("+" if kar_vs_pop_delta > 0 else ("-" if kar_vs_pop_delta < 0 else "0")),
            },
            "go": bool(go),
            "verdict": ("GO" if go else "NO-GO"),
        }

    # ---- run BOTH populations at FULL coverage --------------------------------------------
    elig_sets = {"derisk": derisk_set, "full": full_cov_set}
    results = {pop: run_population(elig_sets[pop], pop) for pop in POPULATIONS}

    # ---- decision logic --------------------------------------------------------------------
    derisk_go = results["derisk"]["go"]
    full_go = results["full"]["go"]
    derisk_sign = results["derisk"]["KAR_external_vs_L1_only"]["sign"]
    full_sign = results["full"]["KAR_external_vs_L1_only"]["sign"]

    if derisk_go and not full_go:
        conclusion = (
            "POPULATION BIAS — at FULL external coverage the DE-RISK population is GO while the FULL "
            "population is NO-GO. The de-risk eligible subset (heavy buyers of the ~3426 popular items) "
            "is special: external-knowledge KAR beats L1 on THAT population but not on the representative "
            "population. The verdict flip is driven by POPULATION, not coverage."
        )
    elif (not derisk_go) and (not full_go):
        conclusion = (
            "COVERAGE ARTIFACT — at FULL external coverage BOTH populations are NO-GO (de-risk KAR-vs-L1 "
            f"sign='{derisk_sign}', full sign='{full_sign}'). Holding the population fixed at the de-risk "
            "eligible subset does NOT recover the de-risk GO once coverage is full. Therefore the original "
            "de-risk GO was driven by the ~28%-pool-coverage artifact (external acted as a "
            "coverage/popularity PROXY in a sparsely-augmented pool), NOT a real population effect."
        )
    elif derisk_go and full_go:
        conclusion = (
            "BOTH GO at full coverage — neither population nor coverage suppresses the external signal in "
            "this run; the probe_21 full NO-GO is not reproduced here. Investigate run/seed/config drift "
            "before concluding."
        )
    else:
        conclusion = (
            "AMBIGUOUS / inverted — full population GO but de-risk population NO-GO at full coverage. The "
            "de-risk eligible subset is HARDER for external KAR at full coverage; the original de-risk GO "
            "cannot be a population effect in the assumed direction. Treat as coverage-driven and inspect."
        )

    out = {
        "probe": "probe_22_isolation",
        "question": "population bias vs coverage artifact behind the de-risk GO -> full NO-GO flip",
        "model": "two_tower_dssm",
        "objective": "softmax",
        "tau": TAU,
        "logq_correction": bool(USE_LOGQ),
        "softmax_batch": SOFTMAX_BATCH,
        "coverage_held_at": "full",
        "full_coverage_items": len(full_cov_set),
        "derisk_set_items": len(derisk_set),
        "pool_size": P,
        "pool_external_cov_frac_full": pool_full_cov_frac,
        "pool_in_derisk_set_frac": pool_derisk_cov_frac,
        "epochs": epochs,
        "n_seeds": n_seeds,
        "populations": results,
        "summary": {
            "derisk": {
                "KAR_external": results["derisk"]["configs"]["KAR_external"]["overall"]["mean"],
                "L1_only": results["derisk"]["configs"]["L1_only"]["overall"]["mean"],
                "delta": results["derisk"]["KAR_external_vs_L1_only"]["delta"],
                "rel": results["derisk"]["KAR_external_vs_L1_only"]["rel"],
                "sign": derisk_sign,
                "sign_seeds": results["derisk"]["KAR_external_vs_L1_only"]["sign_seeds"],
                "verdict": results["derisk"]["verdict"],
                "n_eligible": results["derisk"]["n_eligible"],
            },
            "full": {
                "KAR_external": results["full"]["configs"]["KAR_external"]["overall"]["mean"],
                "L1_only": results["full"]["configs"]["L1_only"]["overall"]["mean"],
                "delta": results["full"]["KAR_external_vs_L1_only"]["delta"],
                "rel": results["full"]["KAR_external_vs_L1_only"]["rel"],
                "sign": full_sign,
                "sign_seeds": results["full"]["KAR_external_vs_L1_only"]["sign_seeds"],
                "verdict": results["full"]["verdict"],
                "n_eligible": results["full"]["n_eligible"],
            },
        },
        "conclusion": conclusion,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_22_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ---- print comparison table ------------------------------------------------------------
    print("=" * 92)
    print("  PROBE 22 — CONTROLLED ISOLATION: population bias vs coverage artifact")
    print("  Two-Tower (DSSM) | coverage HELD at FULL for both populations")
    print("=" * 92)
    print(f"  full coverage={len(full_cov_set)} items | de-risk set={len(derisk_set)} items | "
          f"pool P={P} ext-cov={pool_full_cov_frac * 100:.0f}%")
    hdr = (f"  {'population':12s} | {'n_elig':>7s} | {'KAR_external':>14s} | {'L1_only':>14s} | "
           f"{'delta':>11s} | {'rel%':>7s} | {'sign':>5s} | verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pop in POPULATIONS:
        s = out["summary"][pop]
        print(f"  {pop:12s} | {s['n_eligible']:>7d} | {s['KAR_external']:>14.6f} | "
              f"{s['L1_only']:>14.6f} | {s['delta']:>+11.6f} | {s['rel'] * 100:>+6.1f}% | "
              f"{s['sign_seeds']}/{n_seeds:<2d}| {s['verdict']}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  CONCLUSION: {conclusion}")
    print("=" * 92)
    print(f"  [written] {OUT_DIR / 'probe_22_result.json'}")


if __name__ == "__main__":
    main()
