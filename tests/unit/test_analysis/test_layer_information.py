"""Unit tests for src/analysis/layer_information.py."""

import numpy as np
import pytest

from src.analysis.layer_information import (
    CKAResult,
    CoherenceResult,
    SeparationResult,
    cka_results_to_matrix,
    compute_linear_cka,
    compute_purchase_coherence,
    compute_purchase_separation_auc,
)


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------


class TestLinearCKA:
    def test_identical_representations(self):
        """CKA of identical matrices = 1.0."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        cka = compute_linear_cka(X, X)
        assert cka == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_representations(self):
        """CKA of orthogonal representations ≈ 0."""
        n = 100
        X = np.zeros((n, 10))
        Y = np.zeros((n, 10))
        X[:50, :5] = np.eye(5)[:, :5].repeat(10, axis=0)
        Y[50:, 5:] = np.eye(5)[:, :5].repeat(10, axis=0)
        cka = compute_linear_cka(X, Y)
        assert cka < 0.3  # Near-orthogonal

    def test_scaled_representations(self):
        """CKA is invariant to isotropic scaling."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        Y = X * 5.0
        cka = compute_linear_cka(X, Y)
        assert cka == pytest.approx(1.0, abs=1e-6)

    def test_different_dimensions(self):
        """CKA works with different dimensionalities."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        Y = rng.standard_normal((50, 20))
        cka = compute_linear_cka(X, Y)
        assert 0 <= cka <= 1.0


class TestCKAResultsToMatrix:
    def test_matrix_format(self):
        results = [
            CKAResult("L1", "L1", 1.0),
            CKAResult("L1", "L2", 0.8),
            CKAResult("L2", "L1", 0.8),
            CKAResult("L2", "L2", 1.0),
        ]
        mat = cka_results_to_matrix(results)
        assert mat.shape == (2, 2)
        assert mat.loc["L1", "L2"] == 0.8
        assert mat.loc["L1", "L1"] == 1.0


# ---------------------------------------------------------------------------
# Purchase Coherence
# ---------------------------------------------------------------------------


class TestPurchaseCoherence:
    def test_high_coherence(self):
        """Users buying similar items → high coherence."""
        n_items = 20
        d = 8
        rng = np.random.default_rng(42)

        # Items 0-4 are similar (same cluster)
        emb = rng.standard_normal((n_items, d)).astype(np.float32)
        emb[:5] = emb[0] + rng.standard_normal((5, d)) * 0.1
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        item_ids = np.array([f"i{i}" for i in range(n_items)])
        user_history = {"u1": ["i0", "i1", "i2", "i3", "i4"]}

        results = compute_purchase_coherence(
            emb, item_ids, user_history, "test", sample_users=None,
        )
        # u1 has 5 purchases → "sparse (1-4)" or "light (5-9)"
        light = [r for r in results if r.activity_bracket == "light (5-9)"]
        assert len(light) == 1
        assert light[0].mean_coherence > 0.8

    def test_single_purchase_user_excluded(self):
        """Users with < 2 purchases are excluded (need pairs)."""
        emb = np.eye(5, dtype=np.float32)
        item_ids = np.array(["a", "b", "c", "d", "e"])
        user_history = {"u1": ["a"]}
        results = compute_purchase_coherence(emb, item_ids, user_history, "test")
        assert all(r.n_users == 0 for r in results)


# ---------------------------------------------------------------------------
# Purchase Separation AUC
# ---------------------------------------------------------------------------


class TestSeparationAUC:
    def test_perfect_separation(self):
        """When purchased items are nearest to centroid, AUC → 1."""
        d = 8
        n_items = 50
        rng = np.random.default_rng(42)

        # Create a cluster of items 0-4 that are very similar
        base = rng.standard_normal(d)
        base /= np.linalg.norm(base)
        emb = rng.standard_normal((n_items, d)).astype(np.float32)
        for i in range(5):
            emb[i] = base + rng.standard_normal(d) * 0.05
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        item_ids = np.array([f"i{i}" for i in range(n_items)])
        # User bought items 0-2, val ground truth is items 3-4
        user_history = {"u1": ["i0", "i1", "i2"]}
        val_gt = {"u1": {"i3", "i4"}}

        result = compute_purchase_separation_auc(
            emb, item_ids, user_history, val_gt, "test",
            n_neg_per_user=20, sample_users=None,
        )
        assert isinstance(result, SeparationResult)
        assert result.auc > 0.7  # Should be high given cluster structure
        assert result.mean_pos_sim > result.mean_neg_sim

    def test_random_embeddings(self):
        """Random embeddings → AUC ≈ 0.5."""
        rng = np.random.default_rng(42)
        n_items = 100
        d = 8
        emb = rng.standard_normal((n_items, d)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        item_ids = np.array([f"i{i}" for i in range(n_items)])
        user_history = {f"u{j}": [f"i{j*5+k}" for k in range(5)] for j in range(10)}
        val_gt = {f"u{j}": {f"i{50+j}"} for j in range(10)}

        result = compute_purchase_separation_auc(
            emb, item_ids, user_history, val_gt, "test",
            n_neg_per_user=10, sample_users=None,
        )
        # Random → AUC should be near 0.5
        assert 0.3 < result.auc < 0.7
