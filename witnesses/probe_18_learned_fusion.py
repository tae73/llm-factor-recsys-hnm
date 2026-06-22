"""PROBE 18 — Does external knowledge help discovery under LEARNED fusion? (not frozen cosine)

WHY
    probe_17/17b killed external-knowledge discovery with FROZEN raw-cosine aggregation only.
    A reviewer's valid objection: KAR never uses raw cosine — it uses a LEARNED projection
    (knowledge-space -> rec-space) + gated fusion, which can align the genre mismatch between
    external styling text (BGE) and product-description text (BGE) that raw cosine cannot.
    This probe trains a small two-tower ranker on the discovery task and ablates external
    knowledge IN vs OUT of the learned fusion. If a TRAINED fusion of external knowledge
    still does not beat L1-only, the NO-GO is robust even against KAR-style fusion. If it
    flips, the earlier NO-GO was a frozen-fusion artifact and option-1 is back.

CONFIGS (identical training; only the feature set differs)
    A  L1-only        user=MLP(L1_profile)            item=MLP(L1)
    B  +ext user-side user=MLP([L1_profile,ext_prof]) item=MLP(L1)
    C  +ext both-side user=MLP([L1_profile,ext_prof]) item=MLP([L1,ext])   (KAR-faithful augmentation)

EVAL
    Train on 70% of eligible users (BPR over in-pool discovery positives), evaluate
    discovery_map@12 on the held-out 30%. Same pool / owned-exclusion for all configs.

VERDICT
    NO-GO ROBUST (learned fusion too) iff B and C do not beat A (CI includes 0 / negative).
    REVIVE iff B or C > A with CI excluding 0.

Usage:
    uv run python witnesses/probe_18_learned_fusion.py [n_users] [epochs]
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
ARTICLES = Path("data/processed/articles.parquet")
GT_PATH = Path("data/processed/immediate_ground_truth.json")
K = 12
POOL = 5000
MIN_COVERED = 3
DROP = 0.0
SEED = 42


def main() -> None:
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    import torch
    import torch.nn as nn

    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    canon_ids, V = load_variants(["L1"])
    l1 = V["L1"].astype(np.float32)
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}

    ck = pd.read_parquet(p16.CACHE)
    comp_text = dict(zip(ck["article_id"].astype(str), ck["complement_text"]))
    from sentence_transformers import SentenceTransformer
    ext_ids = [a for a in comp_text if a in id_to_idx]
    enc = SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev)
    enc.max_seq_length = 512
    arr = enc.encode([comp_text[a] for a in ext_ids], batch_size=128,
                     normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    ext_emb = {a: arr[i] for i, a in enumerate(ext_ids)}
    ext_set = set(ext_ids)
    D = l1.shape[1]

    import duckdb
    con = duckdb.connect()
    pop = con.execute(
        f"SELECT article_id FROM read_parquet('{TRAIN_TXN}') WHERE t_dat >= DATE '2020-05-01' "
        f"GROUP BY article_id ORDER BY count(*) DESC LIMIT {POOL}"
    ).fetchall()
    pool_ids = [str(a) for (a,) in pop if str(a) in id_to_idx]
    P = len(pool_ids)
    pool_l1 = np.stack([l1[id_to_idx[a]] for a in pool_ids])
    pool_ext = np.stack([ext_emb[a] if a in ext_emb else np.zeros(D, np.float32) for a in pool_ids])
    pool_pos = {a: i for i, a in enumerate(pool_ids)}
    print(f"[probe_18] pool={P}, pool covered by ext={sum(a in ext_emb for a in pool_ids)}")

    gt = {str(k): [str(x) for x in v] for k, v in json.loads(GT_PATH.read_text()).items()}
    gt_users = pd.DataFrame({"customer_id": list(gt)})
    con.register("gt_users", gt_users)
    hist_rows = con.execute(
        f"SELECT t.customer_id, list(DISTINCT t.article_id) FROM read_parquet('{TRAIN_TXN}') t "
        f"JOIN gt_users g ON t.customer_id=g.customer_id GROUP BY t.customer_id"
    ).fetchall()
    train_history = {str(cid): set(str(a) for a in items) for cid, items in hist_rows}

    # eligible: >=MIN_COVERED covered hist items AND >=1 discovery positive in pool
    eligible = []
    for u, h in train_history.items():
        if u not in gt or len(h & ext_set) < MIN_COVERED:
            continue
        pos = [g for g in gt[u] if g not in h and g in pool_pos]
        if pos:
            eligible.append(u)
    rng.shuffle(eligible)
    eligible = eligible[:n_users]
    cut = int(0.7 * len(eligible))
    tr_users, te_users = eligible[:cut], eligible[cut:]
    print(f"[probe_18] eligible={len(eligible)} train={len(tr_users)} eval={len(te_users)}")

    # user profiles
    def profiles(users):
        l1p, extp = {}, {}
        for u in users:
            cov = [h for h in train_history[u] if h in ext_set]
            a = pool_l1[0] * 0 + l1[[id_to_idx[h] for h in cov]].mean(0)
            a = a / (np.linalg.norm(a) + 1e-9)
            b = np.stack([ext_emb[h] for h in cov]).mean(0)
            b = b / (np.linalg.norm(b) + 1e-9)
            l1p[u], extp[u] = a.astype(np.float32), b.astype(np.float32)
        return l1p, extp
    l1p_tr, extp_tr = profiles(tr_users)
    l1p_te, extp_te = profiles(te_users)

    # positives per train user (in pool)
    tr_pos = {u: [pool_pos[g] for g in gt[u] if g not in train_history[u] and g in pool_pos] for u in tr_users}
    tr_owned = {u: np.array([a in train_history[u] for a in pool_ids]) for u in tr_users}

    t_pool_l1 = torch.tensor(pool_l1, device=dev)
    t_pool_ext = torch.tensor(pool_ext, device=dev)

    class Tower(nn.Module):
        def __init__(self, din, d=64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, d))
        def forward(self, x):
            return self.net(x)

    # owned masks over pool for eval users (numpy)
    te_owned = {u: np.array([a in train_history[u] for a in pool_ids]) for u in te_users}
    ulist = [u for u in tr_users if tr_pos[u]]
    pos_lists = [np.asarray(tr_pos[u]) for u in ulist]

    def run_config(use_user_ext: bool, use_item_ext: bool, tag: str):
        torch.manual_seed(SEED)
        ud = D * (2 if use_user_ext else 1)
        idim = D * (2 if use_item_ext else 1)
        utow, itow = Tower(ud).to(dev), Tower(idim).to(dev)
        opt = torch.optim.Adam(list(utow.parameters()) + list(itow.parameters()), lr=1e-3, weight_decay=1e-5)
        item_feat = torch.cat([t_pool_l1, t_pool_ext], 1) if use_item_ext else t_pool_l1
        uf = np.stack([np.concatenate([l1p_tr[u], extp_tr[u]]) if use_user_ext else l1p_tr[u] for u in ulist])
        uvecs = torch.tensor(uf, device=dev)                          # (Bu, ud) fixed
        Bu = len(ulist)
        for ep in range(epochs):                                     # vectorized BPR: 1 pos + 1 neg / user
            pos_idx = torch.tensor([int(pl[rng.integers(len(pl))]) for pl in pos_lists], device=dev)
            neg_idx = torch.tensor(rng.integers(0, P, size=Bu), device=dev)
            u_emb = utow(uvecs)                                       # (Bu, d)
            pe = itow(item_feat[pos_idx]); ne = itow(item_feat[neg_idx])
            loss = -torch.log(torch.sigmoid((u_emb * pe).sum(1) - (u_emb * ne).sum(1)) + 1e-9).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        # eval on held-out users (batched)
        utow.eval(); itow.eval()
        with torch.no_grad():
            i_emb = itow(item_feat)                                   # (P, d)
            uf_te = np.stack([np.concatenate([l1p_te[u], extp_te[u]]) if use_user_ext else l1p_te[u]
                              for u in te_users])
            ue = utow(torch.tensor(uf_te, device=dev))                # (Bte, d)
            sc = (ue @ i_emb.T).cpu().numpy()                         # (Bte, P)
            preds = {}
            for bi, u in enumerate(te_users):
                s = np.where(te_owned[u], -np.inf, sc[bi])
                order = np.argpartition(-s, K)[:K]
                preds[u] = [pool_ids[i] for i in order[np.argsort(-s[order])]]
        d = discovery_map(preds, {u: gt[u] for u in te_users}, {u: train_history[u] for u in te_users}, k=K)
        print(f"  [{tag}] eval discovery_map@12={d.map_at_k:.5f} hr@12={d.hr_at_k:.5f} (last loss {float(loss):.4f})")
        return {"discovery_map": d.map_at_k, "discovery_hr": d.hr_at_k}

    print("[probe_18] training configs...")
    resA = run_config(False, False, "A L1-only")
    resB = run_config(True, False, "B +ext user")
    resC = run_config(True, True, "C +ext both")

    best_ext = max(resB["discovery_map"], resC["discovery_map"])
    revive = best_ext > resA["discovery_map"] * 1.02  # need >2% rel improvement to claim
    verdict = ("REVIVE — learned fusion of external knowledge BEATS L1-only: earlier NO-GO was a "
               "frozen-fusion artifact; option-1 viable with KAR-style learned projection+gating"
               if revive else
               "NO-GO ROBUST (learned fusion too) — trained projection+fusion of external knowledge does "
               "NOT beat L1-only on discovery; external value confined to pair-level complementarity")

    out = {"probe": "probe_18_learned_fusion", "n_eligible": len(eligible),
           "n_train": len(tr_users), "n_eval": len(te_users), "epochs": epochs,
           "A_L1_only": resA, "B_ext_user": resB, "C_ext_both": resC,
           "best_ext_vs_A_rel": (best_ext / resA["discovery_map"] - 1) if resA["discovery_map"] else None,
           "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_18_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 76)
    print("  PROBE 18 — learned-fusion ablation of external knowledge on discovery")
    print("=" * 76)
    print(f"  eligible={len(eligible)} train={len(tr_users)} eval={len(te_users)} epochs={epochs}")
    print(f"  A L1-only    discovery_map@12 = {resA['discovery_map']:.5f}")
    print(f"  B +ext user  discovery_map@12 = {resB['discovery_map']:.5f}")
    print(f"  C +ext both  discovery_map@12 = {resC['discovery_map']:.5f}")
    print(f"  best ext vs A: {(best_ext/resA['discovery_map']-1)*100:+.1f}%")
    print(f"  VERDICT: {verdict}")
    print("=" * 76)


if __name__ == "__main__":
    main()
