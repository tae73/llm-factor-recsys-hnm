"""Unit tests for LightGCN model, BPR loss, and L2 regularization.

Tests model initialization, graph propagation, scoring, gradient flow,
adjacency matrix construction, and auxiliary losses.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from src.config import LightGCNConfig
from src.losses import binary_cross_entropy, bpr_loss, embedding_l2_reg
from src.models.lightgcn import LightGCN, LightGCNInput, build_normalized_adj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return LightGCNConfig(
        d_embed=8,
        n_layers=2,
        dropout_rate=0.0,
        l2_reg=1e-4,
    )


@pytest.fixture
def tiny_graph():
    """Small bipartite graph: 5 users, 10 items, 20 edges."""
    rng = np.random.default_rng(42)
    n_users, n_items, n_edges = 5, 10, 20
    user_idx = rng.integers(0, n_users, size=n_edges).astype(np.int32)
    item_idx = rng.integers(0, n_items, size=n_edges).astype(np.int32)
    return user_idx, item_idx, n_users, n_items


@pytest.fixture
def adj_matrix(tiny_graph):
    user_idx, item_idx, n_users, n_items = tiny_graph
    return build_normalized_adj(user_idx, item_idx, n_users, n_items)


@pytest.fixture
def model(small_config, tiny_graph, adj_matrix):
    _, _, n_users, n_items = tiny_graph
    rngs = nnx.Rngs(params=42, dropout=43)
    return LightGCN(n_users, n_items, adj_matrix, small_config, rngs=rngs)


@pytest.fixture
def sample_input(tiny_graph):
    """Small batch of 4 user-item pairs."""
    _, _, n_users, n_items = tiny_graph
    rng = np.random.default_rng(42)
    return LightGCNInput(
        user_idx=jnp.array(rng.integers(0, n_users, size=4), dtype=jnp.int32),
        item_idx=jnp.array(rng.integers(0, n_items, size=4), dtype=jnp.int32),
    )


@pytest.fixture
def sample_labels():
    return jnp.array([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# BPR Loss Tests
# ---------------------------------------------------------------------------


class TestBPRLoss:
    def test_perfect_ranking(self):
        """Perfect ranking: pos >> neg → near-zero loss."""
        pos = jnp.array([10.0, 10.0, 10.0])
        neg = jnp.array([-10.0, -10.0, -10.0])
        loss = bpr_loss(pos, neg)
        assert float(loss) < 0.001

    def test_reversed_ranking(self):
        """Reversed ranking: pos << neg → high loss."""
        pos = jnp.array([-10.0, -10.0])
        neg = jnp.array([10.0, 10.0])
        loss = bpr_loss(pos, neg)
        assert float(loss) > 5.0

    def test_equal_scores(self):
        """Equal scores → loss = log(2) ≈ 0.693."""
        pos = jnp.array([0.0, 0.0])
        neg = jnp.array([0.0, 0.0])
        loss = bpr_loss(pos, neg)
        assert abs(float(loss) - np.log(2)) < 0.001

    def test_scalar_output(self):
        """Loss should be a scalar."""
        pos = jnp.array([1.0, 2.0, 3.0])
        neg = jnp.array([0.0, 1.0, 0.0])
        loss = bpr_loss(pos, neg)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# L2 Regularization Tests
# ---------------------------------------------------------------------------


class TestEmbeddingL2Reg:
    def test_positive_value(self):
        """L2 reg should be positive."""
        u = jnp.ones((4, 8))
        i = jnp.ones((4, 8))
        reg = embedding_l2_reg(u, i, weight=1e-4)
        assert float(reg) > 0.0

    def test_zero_embeddings(self):
        """Zero embeddings → zero reg."""
        u = jnp.zeros((4, 8))
        i = jnp.zeros((4, 8))
        reg = embedding_l2_reg(u, i, weight=1e-4)
        assert float(reg) == 0.0

    def test_weight_scaling(self):
        """Doubling weight should double reg."""
        u = jnp.ones((4, 8))
        i = jnp.ones((4, 8))
        reg1 = embedding_l2_reg(u, i, weight=1e-4)
        reg2 = embedding_l2_reg(u, i, weight=2e-4)
        np.testing.assert_allclose(float(reg2), float(reg1) * 2.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Adjacency Matrix Tests
# ---------------------------------------------------------------------------


class TestBuildNormalizedAdj:
    def test_shape(self, tiny_graph, adj_matrix):
        """Adjacency matrix should be (N, N) where N = n_users + n_items."""
        _, _, n_users, n_items = tiny_graph
        N = n_users + n_items
        assert adj_matrix.shape == (N, N)

    def test_symmetric(self, adj_matrix):
        """D^{-1/2} A D^{-1/2} should be symmetric."""
        dense = adj_matrix.todense()
        np.testing.assert_allclose(dense, dense.T, atol=1e-6)

    def test_normalized_values(self, adj_matrix):
        """All values should be finite and in reasonable range."""
        dense = adj_matrix.todense()
        assert jnp.all(jnp.isfinite(dense))
        assert jnp.all(jnp.abs(dense) <= 1.0)

    def test_no_self_loops(self, adj_matrix):
        """Diagonal should be zero (bipartite graph has no self-loops)."""
        dense = adj_matrix.todense()
        diag = jnp.diag(dense)
        np.testing.assert_allclose(diag, jnp.zeros_like(diag), atol=1e-6)


# ---------------------------------------------------------------------------
# LightGCN Model Tests
# ---------------------------------------------------------------------------


class TestLightGCNModel:
    def test_output_shape(self, model, sample_input):
        """Forward pass should return (B,) logits."""
        logits = model(sample_input)
        assert logits.shape == (4,)

    def test_output_is_finite(self, model, sample_input):
        """No NaN/Inf in output."""
        logits = model(sample_input)
        assert jnp.all(jnp.isfinite(logits))

    def test_predict_proba_range(self, model, sample_input):
        """Probabilities should be in [0, 1]."""
        model.eval()
        probs = model.predict_proba(sample_input)
        model.train()
        assert jnp.all(probs >= 0.0)
        assert jnp.all(probs <= 1.0)

    def test_get_all_embeddings_shape(self, model, tiny_graph):
        """get_all_embeddings() should return correct shapes."""
        _, _, n_users, n_items = tiny_graph
        user_embeds, item_embeds = model.get_all_embeddings()
        assert user_embeds.shape == (n_users, 8)
        assert item_embeds.shape == (n_items, 8)

    def test_get_all_embeddings_finite(self, model):
        """Propagated embeddings should be finite."""
        user_embeds, item_embeds = model.get_all_embeddings()
        assert jnp.all(jnp.isfinite(user_embeds))
        assert jnp.all(jnp.isfinite(item_embeds))

    def test_propagation_changes_embeddings(self, model):
        """After propagation, embeddings should differ from initial."""
        user_embeds, item_embeds = model.get_all_embeddings()
        initial_user = model.user_embedding.embedding[...]
        # Propagated should not be identical to initial
        assert not jnp.allclose(user_embeds, initial_user, atol=1e-6)

    def test_get_initial_embeddings(self, model, sample_input):
        """get_initial_embeddings should return non-propagated embeddings."""
        u_e0, i_e0 = model.get_initial_embeddings(sample_input)
        assert u_e0.shape == (4, 8)
        assert i_e0.shape == (4, 8)

    @pytest.mark.parametrize("B", [1, 4, 16])
    def test_batch_size_invariance(self, small_config, tiny_graph, adj_matrix, B):
        """Model should work with different batch sizes."""
        _, _, n_users, n_items = tiny_graph
        rngs = nnx.Rngs(params=42, dropout=43)
        m = LightGCN(n_users, n_items, adj_matrix, small_config, rngs=rngs)

        rng = np.random.default_rng(42)
        inp = LightGCNInput(
            user_idx=jnp.array(rng.integers(0, n_users, size=B), dtype=jnp.int32),
            item_idx=jnp.array(rng.integers(0, n_items, size=B), dtype=jnp.int32),
        )
        logits = m(inp)
        assert logits.shape == (B,)


# ---------------------------------------------------------------------------
# Training Step Tests
# ---------------------------------------------------------------------------


class TestLightGCNTrainStep:
    def test_gradient_flow(self, model, sample_input, sample_labels):
        """Trainable parameters should receive non-zero gradients."""

        def loss_fn(m):
            logits = m(sample_input)
            return binary_cross_entropy(logits, sample_labels)

        _, grads = nnx.value_and_grad(loss_fn)(model)
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, "shape"))
        assert has_nonzero, "All gradients are zero"

    def test_loss_decreases(self, small_config, tiny_graph, adj_matrix):
        """Loss should decrease over multiple training steps."""
        _, _, n_users, n_items = tiny_graph
        rngs = nnx.Rngs(params=42, dropout=43)
        m = LightGCN(n_users, n_items, adj_matrix, small_config, rngs=rngs)
        opt = nnx.Optimizer(m, optax.adam(learning_rate=1e-2), wrt=nnx.Param)

        rng = np.random.default_rng(42)
        inp = LightGCNInput(
            user_idx=jnp.array(rng.integers(0, n_users, size=16), dtype=jnp.int32),
            item_idx=jnp.array(rng.integers(0, n_items, size=16), dtype=jnp.int32),
        )
        labels = jnp.array(rng.choice([0.0, 1.0], size=16), dtype=jnp.float32)

        losses = []
        for _ in range(20):

            def loss_fn(m):
                logits = m(inp)
                return binary_cross_entropy(logits, labels)

            loss, grads = nnx.value_and_grad(loss_fn)(m)
            opt.update(m, grads)
            losses.append(float(loss))

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_jit_compatible(self, model, sample_input, sample_labels):
        """Model should work under nnx.jit."""
        opt = nnx.Optimizer(model, optax.adam(learning_rate=1e-3), wrt=nnx.Param)
        from src.training.trainer import make_lightgcn_train_step

        step_fn = make_lightgcn_train_step(l2_reg=1e-4)
        batch = {
            "user_idx": sample_input.user_idx,
            "item_idx": sample_input.item_idx,
            "labels": sample_labels,
        }
        loss = step_fn(model, opt, batch)
        assert jnp.isfinite(loss)

    def test_l2_reg_in_loss(self, model, sample_input, sample_labels):
        """Loss with L2 reg should be greater than without."""

        def loss_no_reg(m):
            logits = m(sample_input)
            return binary_cross_entropy(logits, sample_labels)

        def loss_with_reg(m):
            logits = m(sample_input)
            bce = binary_cross_entropy(logits, sample_labels)
            u_e0, i_e0 = m.get_initial_embeddings(sample_input)
            reg = embedding_l2_reg(u_e0, i_e0, weight=1.0)
            return bce + reg

        loss1 = loss_no_reg(model)
        loss2 = loss_with_reg(model)
        assert float(loss2) > float(loss1)
