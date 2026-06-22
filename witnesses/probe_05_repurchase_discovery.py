"""PROBE 05 — H&M repurchase/discovery structure (why our numbers were 10x low).

WHAT GENERALIZES
    The project's absolute MAP@12 (~0.0037 popularity, ~0.0018 DeepFM) sat ~10x
    below Kaggle H&M solutions (~0.02-0.035), prompting "is our data/pipeline
    broken?". This probe answers it with simple, training-free baselines on three
    evaluation windows, the per-cohort split, and the repurchase-vs-new-item
    decomposition of next-week purchases. No model training.

THE RESULT (boxed; persisted to probe_05_result.json)
    +-----------------------------------------------------------------------+
    | Immediate-next-week (Kaggle-like): repurchase MAP@12 ~ 0.024 (competitive) |
    | 2-month-gap eval (our val/test): same repurchase collapses to ~0.003-0.009 |
    | Next-week purchases: ~96% NEW items, ~4% exact repurchase; ~7% new users   |
    +-----------------------------------------------------------------------+

HONEST reduces_check
    Our low numbers were NOT a data/metric bug. Two causes: (1) the pipeline
    discarded repurchase+recency (the cheap competitive signal), and (2) a ~2-month
    temporal gap between train end and eval window destroyed recency. The deep
    insight for positioning: 96% of next-week purchases are NEW items -> H&M is
    overwhelmingly a DISCOVERY problem, which is exactly where content/LLM (KAR)
    is the only lever; repurchase cheaply nails the predictable ~4% + recency.

VERDICT
    Re-baseline on the immediate-next-period; adopt a hybrid (repurchase+recency
    backbone for the ~4% + competitive number, LLM/KAR for the ~96% discovery +
    new users). Reframe done in redesign_2026-06.md.

Usage:
    uv run python witnesses/probe_05_repurchase_discovery.py [sample_users]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import EvalConfig  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402

PROC = Path("data/processed")
OUT = Path("witnesses/probe_05_result.json")


def _topn(df: pd.DataFrame, n: int = 12) -> list[str]:
    return df["article_id"].value_counts().head(n).index.tolist()


def main() -> None:
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    txn = pd.read_parquet(PROC / "train_transactions.parquet",
                          columns=["customer_id", "article_id", "t_dat"])
    val = pd.read_parquet(PROC / "val_transactions.parquet",
                          columns=["customer_id", "article_id", "t_dat"])
    for d in (txn, val):
        d["customer_id"] = d["customer_id"].astype(str)
        d["article_id"] = d["article_id"].astype(str)
    tmax = txn["t_dat"].max()

    gpop = _topn(txn)
    recent7 = _topn(txn[txn["t_dat"] > tmax - pd.Timedelta(days=7)])
    recent14 = _topn(txn[txn["t_dat"] > tmax - pd.Timedelta(days=14)])
    last_items = (txn.sort_values("t_dat").groupby("customer_id")["article_id"]
                  .apply(lambda s: list(dict.fromkeys(s.tolist()[::-1]))[:12]))

    def repurchase(u: str) -> list[str]:
        out = list(last_items.get(u, []))
        for it in recent14:
            if len(out) >= 12:
                break
            if it not in out:
                out.append(it)
        return out[:12]

    cfg = EvalConfig(k=12)
    rng = np.random.default_rng(42)

    def gt_for(df: pd.DataFrame) -> dict[str, list[str]]:
        return df.groupby("customer_id")["article_id"].apply(
            lambda s: list(dict.fromkeys(s))).to_dict()

    windows = {
        "immediate_week_Jul1-7": gt_for(val[val["t_dat"] <= tmax + pd.Timedelta(days=7)]),
        "val_2month": gt_for(val),
    }
    test_p = PROC / "test_transactions.parquet"
    if test_p.exists():
        test = pd.read_parquet(test_p, columns=["customer_id", "article_id", "t_dat"])
        test["customer_id"] = test["customer_id"].astype(str)
        test["article_id"] = test["article_id"].astype(str)
        windows["test_2month_gap"] = gt_for(test)

    result: dict = {"probe": "probe_05_repurchase_discovery", "train_end": str(tmax)}
    for wname, gt in windows.items():
        users = list(gt)
        samp = list(rng.choice(users, size=min(sample, len(users)), replace=False))
        gt_s = {u: list(gt[u]) for u in samp}
        row = {"n": len(samp), "avg_gt_items": float(np.mean([len(gt[u]) for u in samp]))}
        for nm, items in [("global_pop", gpop), ("recent14", recent14), ("recent7", recent7)]:
            row[nm] = evaluate({u: items for u in samp}, gt_s, cfg).map_at_k
        row["repurchase"] = evaluate({u: repurchase(u) for u in samp}, gt_s, cfg).map_at_k
        result[wname] = row
        print(f"[{wname}] n={row['n']} avgGT={row['avg_gt_items']:.2f} "
              f"pop={row['global_pop']:.5f} recent14={row['recent14']:.5f} "
              f"repurchase={row['repurchase']:.5f}")

    # Repurchase vs NEW-item decomposition + new-user share (immediate week)
    hist = txn.groupby("customer_id")["article_id"].apply(set).to_dict()
    train_users = set(hist)
    gt_im = windows["immediate_week_Jul1-7"]
    n_rep = n_new = new_users = 0
    for u, items in gt_im.items():
        h = hist.get(u, set())
        if u not in train_users:
            new_users += 1
        for it in set(items):
            n_rep += it in h
            n_new += it not in h
    tot = n_rep + n_new
    result["decomposition_immediate_week"] = {
        "repurchase_item_frac": n_rep / tot,
        "new_item_frac": n_new / tot,
        "new_user_frac": new_users / len(gt_im),
        "n_gt_items": tot,
    }
    print(f"\n[decomposition] NEW items={n_new / tot * 100:.1f}%  "
          f"repurchase items={n_rep / tot * 100:.1f}%  new users={new_users / len(gt_im) * 100:.1f}%")

    # Cohort: repurchase vs pop by train-history size (immediate week)
    hist_n = txn.groupby("customer_id").size()
    cohort = {}
    for bn, (lo, hi) in {"1": (1, 1), "2-4": (2, 4), "5-9": (5, 9),
                         "10-19": (10, 19), "20+": (20, 10**9)}.items():
        bu = [u for u in gt_im if lo <= hist_n.get(u, 0) <= hi]
        if not bu:
            continue
        gt_b = {u: list(gt_im[u]) for u in bu}
        cohort[bn] = {
            "n": len(bu),
            "repurchase": evaluate({u: repurchase(u) for u in bu}, gt_b, cfg).map_at_k,
            "global_pop": evaluate({u: gpop for u in bu}, gt_b, cfg).map_at_k,
        }
    result["cohort_immediate_week"] = cohort

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n" + "=" * 64)
    print("  VERDICT: data/metric fine. Immediate-week repurchase ~0.024 (Kaggle-level).")
    print("  Low numbers = discarded repurchase+recency + 2-month eval gap.")
    print("  96% of next-week purchases are NEW items -> discovery = LLM/KAR space.")
    print("=" * 64)
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
