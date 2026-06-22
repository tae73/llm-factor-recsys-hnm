"""PROBE 18b — is the learned-fusion REVIVE real knowledge, or just capacity? (placebo control)

probe_18 found config C (+ext both-side, KAR-style) beats L1-only by +12.6% under learned
fusion. But C has 2x the input dims -> more capacity. The gain could be capacity/regularization,
NOT the external knowledge's content. Decisive controls, multi-seed:

    A         L1-only
    C_real    +ext both-side (true external knowledge)
    C_shuffle +ext both-side, but ext vectors PERMUTED across items (same distribution, same
              capacity, item<->knowledge correspondence DESTROYED)
    C_noise   +ext both-side, ext replaced by random Gaussian (pure capacity)

If C_real > C_shuffle and C_real > C_noise consistently across seeds -> the SPECIFIC external
knowledge carries discovery signal -> REVIVE is real. If C_real ~ C_shuffle -> it was capacity,
NO-GO stands.

VERDICT
    REVIVE REAL iff mean(C_real) > mean(C_shuffle) and > mean(C_noise) with per-seed paired
    sign consistency (>=4/5) and paired CI excluding 0.

Usage:
    uv run python witnesses/probe_18b_placebo.py [n_users] [epochs] [n_seeds]
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
K = 12
POOL = 5000
MIN_COVERED = 3


def main() -> None:
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    import torch
    import torch.nn as nn

    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    canon_ids, V = load_variants(["L1"])
    l1 = V["L1"].astype(np.float32)
    id_to_idx = {str(a): i for i, a in enumerate(canon_ids)}
    D = l1.shape[1]

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

    import duckdb
    con = duckdb.connect()
    pop = con.execute(
        f"SELECT article_id FROM read_parquet('{TRAIN_TXN}') WHERE t_dat >= DATE '2020-05-01' "
        f"GROUP BY article_id ORDER BY count(*) DESC LIMIT {POOL}"
    ).fetchall()
    pool_ids = [str(a) for (a,) in pop if str(a) in id_to_idx]
    P = len(pool_ids)
    pool_l1 = np.stack([l1[id_to_idx[a]] for a in pool_ids])
    pool_ext_real = np.stack([ext_emb[a] if a in ext_emb else np.zeros(D, np.float32) for a in pool_ids])
    pool_pos = {a: i for i, a in enumerate(pool_ids)}

    gt = {str(k): [str(x) for x in v] for k, v in json.loads(GT_PATH.read_text()).items()}
    gt_users = pd.DataFrame({"customer_id": list(gt)})
    con.register("gt_users", gt_users)
    hist_rows = con.execute(
        f"SELECT t.customer_id, list(DISTINCT t.article_id) FROM read_parquet('{TRAIN_TXN}') t "
        f"JOIN gt_users g ON t.customer_id=g.customer_id GROUP BY t.customer_id"
    ).fetchall()
    train_history = {str(cid): set(str(a) for a in items) for cid, items in hist_rows}

    rng0 = np.random.default_rng(SEED := 42)
    eligible = []
    for u, h in train_history.items():
        if u not in gt or len(h & ext_set) < MIN_COVERED:
            continue
        if [g for g in gt[u] if g not in h and g in pool_pos]:
            eligible.append(u)
    rng0.shuffle(eligible)
    eligible = eligible[:n_users]
    cut = int(0.7 * len(eligible))
    tr_users, te_users = eligible[:cut], eligible[cut:]
    cov = {u: [h for h in train_history[u] if h in ext_set] for u in eligible}
    l1p = {u: (lambda a: a / (np.linalg.norm(a) + 1e-9))(l1[[id_to_idx[h] for h in cov[u]]].mean(0)).astype(np.float32)
           for u in eligible}
    tr_pos = {u: [pool_pos[g] for g in gt[u] if g not in train_history[u] and g in pool_pos] for u in tr_users}
    te_owned = {u: np.array([a in train_history[u] for a in pool_ids]) for u in te_users}
    ulist = [u for u in tr_users if tr_pos[u]]
    pos_lists = [np.asarray(tr_pos[u]) for u in ulist]
    print(f"[probe_18b] eligible={len(eligible)} train={len(ulist)} eval={len(te_users)} "
          f"pool_ext_cov={int((np.abs(pool_ext_real).sum(1) > 0).sum())}")

    def ext_source(kind: str, seed: int):
        """Return (item_ext_pool[P,D], user_extp dict) for the ext feature variant."""
        if kind == "real":
            src = ext_emb
        elif kind == "shuffle":
            r = np.random.default_rng(1000 + seed)
            perm = r.permutation(len(ext_ids))
            src = {a: ext_emb[ext_ids[perm[i]]] for i, a in enumerate(ext_ids)}
        elif kind == "noise":
            r = np.random.default_rng(2000 + seed)
            noise = r.standard_normal((len(ext_ids), D)).astype(np.float32)
            noise /= np.linalg.norm(noise, axis=1, keepdims=True) + 1e-9
            src = {a: noise[i] for i, a in enumerate(ext_ids)}
        else:
            raise ValueError(kind)
        pool_e = np.stack([src[a] if a in src else np.zeros(D, np.float32) for a in pool_ids])
        extp = {u: (lambda b: b / (np.linalg.norm(b) + 1e-9))(np.stack([src[h] for h in cov[u]]).mean(0)).astype(np.float32)
                for u in eligible}
        return pool_e, extp

    class Tower(nn.Module):
        def __init__(self, din, d=64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, d))
        def forward(self, x):
            return self.net(x)

    t_pool_l1 = torch.tensor(pool_l1, device=dev)

    def train_eval(use_ext: bool, ext_kind: str, seed: int) -> float:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        if use_ext:
            pool_e, extp = ext_source(ext_kind, seed)
            item_feat = torch.cat([t_pool_l1, torch.tensor(pool_e, device=dev)], 1)
            uf = np.stack([np.concatenate([l1p[u], extp[u]]) for u in ulist])
            uf_te = np.stack([np.concatenate([l1p[u], extp[u]]) for u in te_users])
            din = D * 2
        else:
            item_feat = t_pool_l1
            uf = np.stack([l1p[u] for u in ulist])
            uf_te = np.stack([l1p[u] for u in te_users])
            din = D
        utow, itow = Tower(din).to(dev), Tower(D * 2 if use_ext else D).to(dev)
        opt = torch.optim.Adam(list(utow.parameters()) + list(itow.parameters()), lr=1e-3, weight_decay=1e-5)
        uvecs = torch.tensor(uf, device=dev)
        Bu = len(ulist)
        for _ in range(epochs):
            pos_idx = torch.tensor([int(pl[rng.integers(len(pl))]) for pl in pos_lists], device=dev)
            neg_idx = torch.tensor(rng.integers(0, P, size=Bu), device=dev)
            u_emb = utow(uvecs)
            pe = itow(item_feat[pos_idx]); ne = itow(item_feat[neg_idx])
            loss = -torch.log(torch.sigmoid((u_emb * pe).sum(1) - (u_emb * ne).sum(1)) + 1e-9).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        utow.eval(); itow.eval()
        with torch.no_grad():
            i_emb = itow(item_feat)
            ue = utow(torch.tensor(uf_te, device=dev))
            sc = (ue @ i_emb.T).cpu().numpy()
        preds = {}
        for bi, u in enumerate(te_users):
            s = np.where(te_owned[u], -np.inf, sc[bi])
            order = np.argpartition(-s, K)[:K]
            preds[u] = [pool_ids[i] for i in order[np.argsort(-s[order])]]
        d = discovery_map(preds, {u: gt[u] for u in te_users}, {u: train_history[u] for u in te_users}, k=K)
        return d.map_at_k

    configs = {"A": (False, None), "C_real": (True, "real"), "C_shuffle": (True, "shuffle"), "C_noise": (True, "noise")}
    per_seed = {c: [] for c in configs}
    for seed in range(n_seeds):
        for c, (ue, kind) in configs.items():
            m = train_eval(ue, kind, seed)
            per_seed[c].append(m)
        print(f"  seed {seed}: " + "  ".join(f"{c}={per_seed[c][-1]:.5f}" for c in configs))

    summ = {c: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": [float(x) for x in v]}
            for c, v in per_seed.items()}
    real = np.array(per_seed["C_real"]); A = np.array(per_seed["A"])
    shuf = np.array(per_seed["C_shuffle"]); noise = np.array(per_seed["C_noise"])
    d_real_A = real - A
    d_real_shuf = real - shuf
    d_real_noise = real - noise
    sign_shuf = int((d_real_shuf > 0).sum())
    sign_A = int((d_real_A > 0).sum())
    # knowledge-specific gain = real - shuffle (capacity-controlled)
    know_gain = float(d_real_shuf.mean())
    cap_gain = float((shuf - A).mean())  # capacity-only gain (shuffle over L1)
    revive = (real.mean() > shuf.mean() and real.mean() > noise.mean()
              and sign_shuf >= max(4, n_seeds - 1) and know_gain > 0)
    verdict = ("REVIVE REAL — external knowledge content (not just capacity) adds to discovery under "
               f"learned fusion: C_real>C_shuffle in {sign_shuf}/{n_seeds} seeds, knowledge-gain {know_gain:+.5f}"
               if revive else
               f"NO-GO / CAPACITY — C_real does not robustly beat C_shuffle (sign {sign_shuf}/{n_seeds}, "
               f"knowledge-gain {know_gain:+.5f}); the probe_18 lift was capacity, not knowledge")

    out = {"probe": "probe_18b_placebo", "n_eligible": len(eligible), "n_eval": len(te_users),
           "epochs": epochs, "n_seeds": n_seeds, "configs": summ,
           "C_real_minus_A_mean": float(d_real_A.mean()), "sign_real_gt_A": sign_A,
           "C_real_minus_shuffle_mean": know_gain, "sign_real_gt_shuffle": sign_shuf,
           "capacity_only_gain_shuffle_minus_A": cap_gain,
           "C_real_minus_noise_mean": float(d_real_noise.mean()), "verdict": verdict}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_18b_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("=" * 78)
    print("  PROBE 18b — placebo control: is the learned-fusion lift knowledge or capacity?")
    print("=" * 78)
    for c in configs:
        print(f"  {c:10s} mean={summ[c]['mean']:.5f} ± {summ[c]['std']:.5f}")
    print(f"  C_real - A         = {d_real_A.mean():+.5f}  (sign {sign_A}/{n_seeds})   [total gain]")
    print(f"  C_shuffle - A      = {cap_gain:+.5f}            [capacity-only]")
    print(f"  C_real - C_shuffle = {know_gain:+.5f}  (sign {sign_shuf}/{n_seeds})   [KNOWLEDGE-specific]")
    print(f"  C_real - C_noise   = {d_real_noise.mean():+.5f}")
    print(f"  VERDICT: {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
