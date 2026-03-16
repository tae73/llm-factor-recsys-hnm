"""LightGCN recommendation model in Flax NNX.

Architecture:
    Init:
      user_embed: Embed(n_users, d_embed) → E_u^(0)
      item_embed: Embed(n_items, d_embed) → E_i^(0)
      adj: BCOO sparse (N, N) where N = n_users + n_items
           D^{-1/2} A D^{-1/2} (symmetric normalized)

    Propagation (K layers, no trainable weights/activation):
      E^(0) = concat([E_u^(0), E_i^(0)]) → (N, d)
      E^(k) = adj @ E^(k-1)               → sparse-dense matmul
      E_final = mean(E^(0)...E^(K))        → (N, d)

    Scoring:
      u_embed = E_final[:n_users][user_idx]  → (B, d)
      i_embed = E_final[n_users:][item_idx]  → (B, d)
      logits = sum(u * i, axis=-1)           → (B,) dot product

Reference: He et al. "LightGCN: Simplifying and Powering Graph Convolution Network
           for Recommendation" (SIGIR 2020)

Device-agnostic: no sharding logic inside the model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy.sparse import coo_matrix

from src.config import LightGCNConfig


class LightGCNInput(NamedTuple):
    """Batched input for LightGCN (index-only, no features)."""

    user_idx: jax.Array  # (B,) int32
    item_idx: jax.Array  # (B,) int32


class LightGCN(nnx.Module):
    """Light Graph Convolution Network.

    Args:
        n_users: Number of users.
        n_items: Number of items.
        adj_matrix: Normalized adjacency BCOO sparse matrix (N, N).
        config: LightGCNConfig hyperparameters.
        rngs: NNX random number generators.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        adj_matrix: jsparse.BCOO,
        config: LightGCNConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self._n_users = n_users
        self._n_items = n_items
        self._n_layers = config.n_layers
        self._d_embed = config.d_embed

        # Trainable embeddings (only parameters in LightGCN)
        self.user_embedding = nnx.Embed(
            num_embeddings=n_users, features=config.d_embed, rngs=rngs
        )
        self.item_embedding = nnx.Embed(
            num_embeddings=n_items, features=config.d_embed, rngs=rngs
        )

        # Store adjacency as non-trainable (nnx.Variable for pytree compat)
        self.adj_indices = nnx.Variable(adj_matrix.indices)
        self.adj_data = nnx.Variable(adj_matrix.data)
        self._adj_shape = adj_matrix.shape

    def _get_adj(self) -> jsparse.BCOO:
        """Reconstruct BCOO from stored components."""
        return jsparse.BCOO((self.adj_data[...], self.adj_indices[...]), shape=self._adj_shape)

    def get_all_embeddings(self) -> tuple[jax.Array, jax.Array]:
        """Run K-layer propagation and return final user/item embeddings.

        Returns:
            (user_embeddings, item_embeddings): each (n_users, d) and (n_items, d).
        """
        # Initial embeddings
        user_e0 = self.user_embedding.embedding[...]  # (n_users, d)
        item_e0 = self.item_embedding.embedding[...]  # (n_items, d)
        e_all = jnp.concatenate([user_e0, item_e0], axis=0)  # (N, d)

        adj = self._get_adj()

        # Layer propagation: E^(k) = adj @ E^(k-1)
        layers = [e_all]
        e_k = e_all
        for _ in range(self._n_layers):
            e_k = adj @ e_k  # sparse-dense matmul: (N, N) @ (N, d) → (N, d)
            layers.append(e_k)

        # Mean aggregation: E_final = mean(E^(0), ..., E^(K))
        e_final = jnp.mean(jnp.stack(layers, axis=0), axis=0)  # (N, d)

        user_embeddings = e_final[: self._n_users]  # (n_users, d)
        item_embeddings = e_final[self._n_users :]  # (n_items, d)
        return user_embeddings, item_embeddings

    def embed(self, x: LightGCNInput) -> tuple[jax.Array, jax.Array]:
        """Compute propagated user/item embeddings for a batch.

        Returns:
            (user_embed, item_embed): each (B, d).
        """
        user_embeds, item_embeds = self.get_all_embeddings()
        u = user_embeds[x.user_idx]  # (B, d)
        i = item_embeds[x.item_idx]  # (B, d)
        return u, i

    def predict_from_embedding(
        self, user_embed: jax.Array, item_embed: jax.Array
    ) -> jax.Array:
        """Predict logits from pre-computed embeddings via dot product.

        Args:
            user_embed: (B, d) user embeddings.
            item_embed: (B, d) item embeddings.

        Returns:
            logits (B,).
        """
        return jnp.sum(user_embed * item_embed, axis=-1)

    def __call__(self, x: LightGCNInput) -> jax.Array:
        """Forward pass returning logits (B,) via dot product.

        Runs full propagation → looks up user/item embeddings → dot product.
        """
        u, i = self.embed(x)
        return self.predict_from_embedding(u, i)

    def predict_proba(self, x: LightGCNInput) -> jax.Array:
        """Inference: returns sigmoid probabilities (B,)."""
        return jax.nn.sigmoid(self.__call__(x))

    def get_initial_embeddings(self, x: LightGCNInput) -> tuple[jax.Array, jax.Array]:
        """Return initial (non-propagated) embeddings for L2 regularization.

        Args:
            x: LightGCNInput with user_idx and item_idx.

        Returns:
            (user_embed_0, item_embed_0): each (B, d), from E^(0).
        """
        u = self.user_embedding(x.user_idx)  # (B, d)
        i = self.item_embedding(x.item_idx)  # (B, d)
        return u, i


# ---------------------------------------------------------------------------
# Adjacency Matrix Construction
# ---------------------------------------------------------------------------


def build_normalized_adj(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    n_users: int,
    n_items: int,
) -> jsparse.BCOO:
    """Build symmetric normalized adjacency matrix for bipartite graph.

    Constructs D^{-1/2} A D^{-1/2} from (user, item) interaction pairs,
    where A is the bipartite adjacency matrix:

        A = [[0,     R    ],
             [R^T,   0    ]]

    with R being the user-item interaction matrix.

    Args:
        user_idx: User indices from training pairs (N_edges,) int.
        item_idx: Item indices from training pairs (N_edges,) int.
        n_users: Total number of users.
        n_items: Total number of items.

    Returns:
        BCOO sparse matrix (N, N) where N = n_users + n_items.
    """
    N = n_users + n_items

    # Shift item indices to [n_users, n_users + n_items)
    shifted_items = item_idx + n_users

    # Bipartite edges: user→item and item→user (symmetric)
    rows = np.concatenate([user_idx, shifted_items])
    cols = np.concatenate([shifted_items, user_idx])
    data = np.ones(len(rows), dtype=np.float32)

    # Build scipy sparse → compute degree normalization
    adj_coo = coo_matrix((data, (rows, cols)), shape=(N, N))

    # Degree: D_ii = sum of row i
    degree = np.array(adj_coo.sum(axis=1)).flatten()
    # D^{-1/2}, handle zero-degree nodes
    d_inv_sqrt = np.where(degree > 0, np.power(degree, -0.5), 0.0)

    # Normalized: D^{-1/2} A D^{-1/2}
    # For COO: norm_val[k] = d_inv_sqrt[row[k]] * data[k] * d_inv_sqrt[col[k]]
    norm_data = d_inv_sqrt[adj_coo.row] * adj_coo.data * d_inv_sqrt[adj_coo.col]

    # Convert to JAX BCOO
    indices = np.stack([adj_coo.row, adj_coo.col], axis=-1)  # (nnz, 2)
    return jsparse.BCOO(
        (jnp.array(norm_data, dtype=jnp.float32), jnp.array(indices, dtype=jnp.int32)),
        shape=(N, N),
    )
