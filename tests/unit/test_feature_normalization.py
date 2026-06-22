"""Unit tests for FIX A: numerical log-transform before z-score.

Heavy long-tailed count columns (e.g. item total_purchases, max ~44761 ≈ 61σ)
are log1p'd before z-scoring so the tail no longer dominates the embedding-table
gradients. These tests assert the transform tames the max-abs of a heavy column
vs plain z-score, and that the persisted stats reproduce the transform.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sigma


def _log1p_zscore(x: np.ndarray, log_idx: list[int]) -> np.ndarray:
    x = x.astype(np.float32).copy()
    if log_idx:
        x[:, log_idx] = np.log1p(np.maximum(x[:, log_idx], 0.0))
    return _zscore(x)


def _heavy_tail_total_purchases(n: int = 5000, seed: int = 0) -> np.ndarray:
    """Synthesize a power-law-ish count column with an extreme max (≈44761)."""
    rng = np.random.default_rng(seed)
    col = rng.integers(0, 50, size=n).astype(np.float32)  # most items tail
    col[0] = 44761.0  # the real H&M max
    return col.reshape(-1, 1)


def test_log1p_reduces_max_abs_vs_plain_zscore():
    """log1p+zscore must have a much smaller max |z| than plain zscore."""
    counts = _heavy_tail_total_purchases()

    plain = _zscore(counts)
    logged = _log1p_zscore(counts, log_idx=[0])

    plain_max = float(np.max(np.abs(plain)))
    logged_max = float(np.max(np.abs(logged)))

    # Plain z-score leaves a huge ~60σ outlier; log1p collapses it to a few σ.
    assert plain_max > 30.0, f"expected heavy tail, got max|z|={plain_max:.1f}"
    assert logged_max < plain_max / 3.0, (
        f"log1p did not tame tail: plain={plain_max:.1f} logged={logged_max:.1f}"
    )
    assert logged_max < 10.0


def test_log1p_only_applied_to_named_columns():
    """A non-heavy column should be unaffected by which columns are log1p'd."""
    rng = np.random.default_rng(1)
    counts = _heavy_tail_total_purchases()
    other = rng.normal(0.5, 0.1, size=(counts.shape[0], 1)).astype(np.float32)
    mat = np.concatenate([counts, other], axis=1)

    logged = _log1p_zscore(mat, log_idx=[0])  # only column 0
    plain_other = _zscore(other)

    # Column 1 (not log1p'd) must equal its own plain z-score.
    np.testing.assert_allclose(logged[:, 1], plain_other[:, 0], rtol=1e-5, atol=1e-5)


def test_persisted_stats_roundtrip_reproduces_transform():
    """feature_stats.json (mean/std/log1p_cols) must reproduce the train transform."""
    num_names = ["total_purchases", "avg_price"]
    counts = _heavy_tail_total_purchases()
    rng = np.random.default_rng(2)
    price = rng.normal(0.05, 0.01, size=(counts.shape[0], 1)).astype(np.float32)
    raw = np.concatenate([counts, price], axis=1)

    log_cols = ["total_purchases"]
    log_idx = [num_names.index(c) for c in log_cols]

    # Reproduce the trainer's transform + stat persistence.
    work = raw.astype(np.float32).copy()
    work[:, log_idx] = np.log1p(np.maximum(work[:, log_idx], 0.0))
    mu = work.mean(axis=0, keepdims=True)
    sigma = work.std(axis=0, keepdims=True) + 1e-8
    train_norm = (work - mu) / sigma

    stats = {
        "num_names": num_names,
        "log1p_cols": log_cols,
        "mean": mu.squeeze(0).tolist(),
        "std": sigma.squeeze(0).tolist(),
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "feature_stats.json"
        p.write_text(json.dumps(stats))
        loaded = json.loads(p.read_text())

    # Inference path: apply log1p to named cols, then (x - mean) / std from stats.
    infer = raw.astype(np.float32).copy()
    li = [loaded["num_names"].index(c) for c in loaded["log1p_cols"]]
    infer[:, li] = np.log1p(np.maximum(infer[:, li], 0.0))
    infer_norm = (infer - np.array(loaded["mean"])) / np.array(loaded["std"])

    np.testing.assert_allclose(infer_norm, train_norm, rtol=1e-5, atol=1e-5)
