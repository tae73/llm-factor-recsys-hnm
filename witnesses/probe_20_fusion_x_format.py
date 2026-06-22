"""PROBE 20 — does format-unification ADD on top of learned fusion? (Q: fusion x format)

probe_18: learned fusion of PROSE external -> +12.6% on discovery (genre mismatch bridged by
the learned projection). probe_19: format-unified external under FROZEN cosine -> no help.
Open question: under LEARNED fusion, does adding the format-unified (attribute) external on
top of the prose external give a real synergy, or is prose alone sufficient?

CONFIGS (learned two-tower, both-side augmentation; multi-seed)
    A            L1 only
    C_prose      L1 + ext_prose                 (= probe_18 config C)
    C_unified    L1 + ext_unified               (attribute format only)
    C_both       L1 + ext_prose + ext_unified   (both — but +1 block of capacity)
    C_prose_dup  L1 + ext_prose + ext_prose     (capacity-matched control for C_both)

VERDICT
    UNIFY ADDS iff C_both > C_prose AND C_both > C_prose_dup (real info beyond capacity).
    FORMAT IRRELEVANT-UNDER-FUSION iff C_unified ~ C_prose (learned projection bridges genre).

Usage:
    uv run python witnesses/probe_20_fusion_x_format.py [n_users] [epochs] [n_seeds]
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

import probe_16_external_knowledge as p16  # noqa: E402

from src.evaluation.cohorts import discovery_map  # noqa: E402
from witnesses._probe_common import OUT_DIR, load_variants  # noqa: E402

TRAIN_TXN = Path("data/processed/train_transactions.parquet")
GT_PATH = Path("data/processed/immediate_ground_truth.json")
STRUCT_CACHE = Path("data/knowledge/external/complement_structured.parquet")
K = 12
POOL = 5000
MIN_COVERED = 3


def main() -> None:
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    import torch
    import torch.nn as nn

    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    canon_ids, V = load_variants(["L1"])
    l1 = V["L1"].astype(np.float32)
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}
    D = l1.shape[1]

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev)
    enc.max_seq_length = 512

    prose = pd.read_parquet(p16.CACHE)
    prose_text = dict(zip(prose["article_id"].astype(str), prose["complement_text"]))
    uni = pd.read_parquet(STRUCT_CACHE)
    uni_text = dict(zip(uni["article_id"].astype(str), uni["unified_text"]))
    ids = [a for a in prose_text if a in id_to_idx and a in uni_text]
    pe = enc.encode([prose_text[a] for a in ids], batch_size=128, normalize_embeddings=True,
                    show_progress_bar=False).astype(np.float32)
    ue = enc.encode([uni_text[a] for a in ids], batch_size=128, normalize_embeddings=True,
                    show_progress_bar=False).astype(np.float32)
    prose_emb = {a: pe[i] for i, a in enumerate(ids)}
    uni_emb = {a: ue[i] for i, a in enumerate(ids)}
    cov_set = set(ids)

    import duckdb
    con = duckdb.connect()
    pop = con.execute(
        f"SELECT article_id FROM read_parquet('{TRAIN_TXN}') WHERE t_dat >= DATE '2020-05-01' "
        f"GROUP BY article_id ORDER BY count(*) DESC LIMIT {POOL}"
    ).fetchall()
    pool_ids = [str(a) for (a,) in pop if str(a) in id_to_idx]
    P = len(pool_ids)
    pool_l1 = np.stack([l1[id_to_idx[a]] for a in pool_ids])
    pool_pr = np.stack([prose_emb.get(a, np.zeros(D, np.float32)) for a in pool_ids])
    pool_un = np.stack([uni_emb.get(a, np.zeros(D, np.float32)) for a in pool_ids])
    pool_pos = {a: i for i, a in enumerate(pool_ids)}

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
    unp = {u: prof(uni_emb, u) for u in eligible}
    tr_pos = {u: [pool_pos[g] for g in gt[u] if g not in train_history[u] and g in pool_pos] for u in tr_users}
    te_owned = {u: np.array([a in train_history[u] for a in pool_ids]) for u in te_users}
    ulist = [u for u in tr_users if tr_pos[u]]
    pos_lists = [np.asarray(tr_pos[u]) for u in ulist]
    print(f"[probe_20] eligible={len(eligible)} train={len(ulist)} eval={len(te_users)} "
          f"pool_ext_cov={int((np.abs(pool_pr).sum(1) > 0).sum())}")

    # config -> (item_feat blocks, user_prof blocks)
    CONFIGS = {
        "A": ([pool_l1], [l1p]),
        "C_prose": ([pool_l1, pool_pr], [l1p, prp]),
        "C_unified": ([pool_l1, pool_un], [l1p, unp]),
        "C_both": ([pool_l1, pool_pr, pool_un], [l1p, prp, unp]),
        "C_prose_dup": ([pool_l1, pool_pr, pool_pr], [l1p, prp, prp]),
    }
    t_blocks = {"pool_l1": torch.tensor(pool_l1, device=dev),
                "pool_pr": torch.tensor(pool_pr, device=dev),
                "pool_un": torch.tensor(pool_un, device=dev)}

    class Tower(nn.Module):
        def __init__(self, din, d=64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, d))
        def forward(self, x):
            return self.net(x)

    def train_eval(item_blocks, user_blocks, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        item_feat = torch.cat([torch.tensor(b, device=dev) for b in item_blocks], 1)
        din = D * len(item_blocks)
        uf = np.stack([np.concatenate([ub[u] for ub in user_blocks]) for u in ulist])
        uf_te = np.stack([np.concatenate([ub[u] for ub in user_blocks]) for u in te_users])
        utow, itow = Tower(din).to(dev), Tower(din).to(dev)
        opt = torch.optim.Adam(list(utow.parameters()) + list(itow.parameters()), lr=1e-3, weight_decay=1e-5)
        uvecs = torch.tensor(uf, device=dev)
        Bu = len(ulist)
        for _ in range(epochs):
            pos_idx = torch.tensor([int(pl[rng.integers(len(pl))]) for pl in pos_lists], device=dev)
            neg_idx = torch.tensor(rng.integers(0, P, size=Bu), device=dev)
            u_emb = utow(uvecs)
            pemb = itow(item_feat[pos_idx]); nemb = itow(item_feat[neg_idx])
            loss = -torch.log(torch.sigmoid((u_emb * pemb).sum(1) - (u_emb * nemb).sum(1)) + 1e-9).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        utow.eval(); itow.eval()
        with torch.no_grad():
            i_emb = itow(item_feat)
            ue_te = utow(torch.tensor(uf_te, device=dev))
            sc = (ue_te @ i_emb.T).cpu().numpy()
        preds = {}
        for bi, u in enumerate(te_users):
            s = np.where(te_owned[u], -np.inf, sc[bi])
            order = np.argpartition(-s, K)[:K]
            preds[u] = [pool_ids[i] for i in order[np.argsort(-s[order])]]
        d = discovery_map(preds, {u: gt[u] for u in te_users}, {u: train_history[u] for u in te_users}, k=K)
        return d.map_at_k

    per = {c: [] for c in CONFIGS}
    for seed in range(n_seeds):
        for c, (ib, ub) in CONFIGS.items():
            ib_arr = [b for b in ib]  # numpy blocks; train_eval tensors them
            per[c].append(train_eval(ib_arr, ub, seed))
        print(f"  seed {seed}: " + "  ".join(f"{c}={per[c][-1]:.5f}" for c in CONFIGS))

    m = {c: float(np.mean(v)) for c, v in per.items()}
    s = {c: float(np.std(v)) for c, v in per.items()}
    both_vs_prose = m["C_both"] - m["C_prose"]
    both_vs_dup = m["C_both"] - m["C_prose_dup"]   # capacity-controlled info gain of unified
    uni_vs_prose = m["C_unified"] - m["C_prose"]
    sign_both_dup = int((np.array(per["C_both"]) > np.array(per["C_prose_dup"])).sum())
    unify_adds = both_vs_prose > 0 and both_vs_dup > 0 and sign_both_dup >= max(2, n_seeds - 1)
    verdict = ("UNIFY ADDS — format-unified external adds real info on top of prose under learned fusion "
               f"(C_both−C_prose_dup={both_vs_dup:+.5f}, {sign_both_dup}/{n_seeds} seeds)"
               if unify_adds else
               "NO SYNERGY — unified format adds nothing beyond prose under learned fusion "
               f"(C_both−C_prose_dup={both_vs_dup:+.5f}); learned projection already bridges genre")

    out = {"probe": "probe_20_fusion_x_format", "n_eligible": len(eligible), "n_eval": len(te_users),
           "epochs": epochs, "n_seeds": n_seeds,
           "mean": m, "std": s,
           "C_both_minus_C_prose": both_vs_prose, "C_both_minus_C_prose_dup": both_vs_dup,
           "C_unified_minus_C_prose": uni_vs_prose, "sign_both_gt_dup": sign_both_dup,
           "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_20_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 78)
    print("  PROBE 20 — learned fusion x external format (does unify add on top?)")
    print("=" * 78)
    for c in CONFIGS:
        print(f"  {c:13s} discovery_map@12 = {m[c]:.5f} ± {s[c]:.5f}")
    print(f"  C_unified - C_prose   = {uni_vs_prose:+.5f}   [format-only under fusion]")
    print(f"  C_both    - C_prose   = {both_vs_prose:+.5f}")
    print(f"  C_both    - C_prose_dup = {both_vs_dup:+.5f}  (sign {sign_both_dup}/{n_seeds}) [capacity-controlled]")
    print(f"  VERDICT: {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
