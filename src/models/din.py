"""DIN (Deep Interest Network) recommendation model in Flax NNX.

Architecture:
    Input: user_cat(B,3) + user_num(B,8) + item_cat(B,5) + item_num(B,2)
           + history(B,T) + hist_len(B,)

    1. Embed target item + history items (shared embedding table)
    2. MLP Attention(target, history) → weighted sum → user interest (B, d_embed)
    3. Concat [user_interest, target_embed, user_static_features, item_static_features]
    4. DNN → logits (B,)

    Attention: f(q, k, q-k, q*k) → sigmoid weights (not softmax, per original DIN)

Reference: Zhou et al. "Deep Interest Network for Click-Through Rate Prediction"
           (KDD 2018)

Device-agnostic: no sharding logic inside the model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from src.config import DINConfig


class DINInput(NamedTuple):
    """Batched input for DIN."""

    user_cat: jax.Array   # (B, 3) int32
    user_num: jax.Array   # (B, 8) float32
    item_cat: jax.Array   # (B, 5) int32 — target item
    item_num: jax.Array   # (B, 2) float32 — target item
    history: jax.Array    # (B, T) int32 — item index sequence (0=PAD)
    hist_len: jax.Array   # (B,) int32 — actual lengths


class DIN(nnx.Module):
    """Deep Interest Network.

    Args:
        field_dims: Vocabulary sizes for each categorical field (user + item).
        n_numerical: Total number of numerical features (user_num + item_num).
        n_items: Total number of items (for sequence embedding, +1 for PAD at idx 0).
        max_seq_len: Maximum sequence length.
        config: DINConfig hyperparameters.
        rngs: NNX random number generators.
    """

    def __init__(
        self,
        field_dims: list[int],
        n_numerical: int,
        n_items: int,
        max_seq_len: int,
        config: DINConfig,
        *,
        rngs: nnx.Rngs,
    ):
        d_embed = config.d_embed
        self._d_embed = d_embed
        self._n_numerical = n_numerical
        self._n_fields = len(field_dims)
        self._max_seq_len = max_seq_len

        # --- Static feature embeddings (same pattern as DeepFM) ---
        self.cat_embeddings = nnx.List(
            [nnx.Embed(num_embeddings=dim, features=d_embed, rngs=rngs) for dim in field_dims]
        )
        self.num_projection = nnx.Linear(
            n_numerical, n_numerical * d_embed, use_bias=False, rngs=rngs
        )

        # --- Sequence item embedding (shared for history and target) ---
        # n_items + 1: index 0 is PAD
        self.item_seq_embedding = nnx.Embed(
            num_embeddings=n_items + 1, features=d_embed, rngs=rngs
        )

        # --- Attention MLP: f(q, k, q-k, q*k) → scalar ---
        att_input_dim = d_embed * 4  # [q, k, q-k, q*k]
        att_layers: list[nnx.Module] = []
        in_dim = att_input_dim
        for hidden_dim in config.attention_hidden_dims:
            att_layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
        att_layers.append(nnx.Linear(in_dim, 1, rngs=rngs))
        self.attention_layers = nnx.List(att_layers)

        # --- DNN ---
        # Input: user_interest(d) + target_embed(d) + static features
        static_dim = (self._n_fields + n_numerical) * d_embed
        dnn_input_dim = d_embed + d_embed + static_dim  # interest + target + static
        dnn_layers: list[nnx.Module] = []
        in_dim = dnn_input_dim
        for hidden_dim in config.dnn_hidden_dims:
            dnn_layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            if config.use_batch_norm:
                dnn_layers.append(nnx.BatchNorm(hidden_dim, rngs=rngs))
            dnn_layers.append(nnx.Dropout(rate=config.dropout_rate, rngs=rngs))
            in_dim = hidden_dim
        self.dnn_layers = nnx.List(dnn_layers)
        self.dnn_output = nnx.Linear(in_dim, 1, rngs=rngs)

        # --- Global bias ---
        self.bias = nnx.Param(jnp.zeros(()))

    def _attention(
        self,
        query: jax.Array,
        keys: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        """MLP-based attention: returns weighted sum of keys.

        Args:
            query: Target item embedding (B, d).
            keys: History item embeddings (B, T, d).
            mask: Padding mask (B, T) — True for valid positions.

        Returns:
            Weighted history representation (B, d).
        """
        T = keys.shape[1]
        # Expand query to (B, T, d)
        q_expanded = jnp.broadcast_to(query[:, None, :], keys.shape)

        # Concat [q, k, q-k, q*k] → (B, T, 4*d)
        att_input = jnp.concatenate(
            [q_expanded, keys, q_expanded - keys, q_expanded * keys],
            axis=-1,
        )

        # MLP → (B, T, 1)
        h = att_input
        for layer in self.attention_layers[:-1]:
            h = nnx.relu(layer(h))
        att_scores = self.attention_layers[-1](h).squeeze(-1)  # (B, T)

        # Apply sigmoid (DIN uses sigmoid, not softmax) + mask
        att_weights = jax.nn.sigmoid(att_scores)  # (B, T)
        att_weights = att_weights * mask  # zero out padding

        # Normalize by number of valid items (avoid division by zero)
        valid_counts = jnp.maximum(mask.sum(axis=-1, keepdims=True), 1.0)  # (B, 1)
        att_weights = att_weights / valid_counts  # (B, T)

        # Weighted sum: (B, T, 1) * (B, T, d) → sum → (B, d)
        user_interest = jnp.sum(att_weights[:, :, None] * keys, axis=1)
        return user_interest

    def embed(self, x: DINInput) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute user interest, target query, and static flat embeddings.

        Returns:
            (user_interest, target_query, static_flat):
            - user_interest: (B, d_embed) attention-weighted history.
            - target_query: (B, d_embed) target item query.
            - static_flat: (B, static_dim) flattened static features.
        """
        cat_indices = jnp.concatenate([x.user_cat, x.item_cat], axis=-1)
        num_features = jnp.concatenate([x.user_num, x.item_num], axis=-1)

        cat_embeds = [emb(cat_indices[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        num_proj = self.num_projection(num_features)
        num_embeds = num_proj.reshape(num_proj.shape[0], self._n_numerical, self._d_embed)
        num_embed_list = [num_embeds[:, i, :] for i in range(self._n_numerical)]

        all_static = cat_embeds + num_embed_list
        stacked_static = jnp.stack(all_static, axis=1)
        static_flat = stacked_static.reshape(stacked_static.shape[0], -1)

        # Target query from item categorical embeddings
        item_cat_embeds = [emb(x.item_cat[:, i]) for i, emb in enumerate(self.cat_embeddings[3:])]
        target_query = jnp.mean(jnp.stack(item_cat_embeds, axis=1), axis=1)

        hist_embeds = self.item_seq_embedding(x.history)
        mask = (x.history > 0).astype(jnp.float32)
        user_interest = self._attention(target_query, hist_embeds, mask)

        return user_interest, target_query, static_flat

    def predict_from_embedding(
        self,
        user_interest: jax.Array,
        target_query: jax.Array,
        static_flat: jax.Array,
    ) -> jax.Array:
        """Predict logits from pre-computed embeddings.

        Args:
            user_interest: (B, d_embed) attention-weighted history.
            target_query: (B, d_embed) target item embedding.
            static_flat: (B, static_dim) flattened static features.

        Returns:
            logits (B,).
        """
        dnn_input = jnp.concatenate([user_interest, target_query, static_flat], axis=-1)
        h = dnn_input
        for layer in self.dnn_layers:
            if isinstance(layer, nnx.Linear):
                h = layer(h)
                h = nnx.relu(h)
            elif isinstance(layer, (nnx.BatchNorm, nnx.Dropout)):
                h = layer(h)
        logits = self.dnn_output(h) + self.bias[...]
        return logits.squeeze(-1)

    def __call__(self, x: DINInput) -> jax.Array:
        """Forward pass returning logits (B,). Sigmoid applied in loss."""
        user_interest, target_query, static_flat = self.embed(x)
        return self.predict_from_embedding(user_interest, target_query, static_flat)

    def predict_proba(self, x: DINInput) -> jax.Array:
        """Inference: returns sigmoid probabilities (B,)."""
        return jax.nn.sigmoid(self.__call__(x))

    def get_attention_weights(
        self,
        x: DINInput,
    ) -> jax.Array:
        """Return attention weights for analysis. Shape (B, T)."""
        # Target query
        item_cat_embeds = [emb(x.item_cat[:, i]) for i, emb in enumerate(self.cat_embeddings[3:])]
        target_query = jnp.mean(jnp.stack(item_cat_embeds, axis=1), axis=1)

        hist_embeds = self.item_seq_embedding(x.history)
        mask = (x.history > 0).astype(jnp.float32)

        q_expanded = jnp.broadcast_to(target_query[:, None, :], hist_embeds.shape)
        att_input = jnp.concatenate(
            [q_expanded, hist_embeds, q_expanded - hist_embeds, q_expanded * hist_embeds],
            axis=-1,
        )
        h = att_input
        for layer in self.attention_layers[:-1]:
            h = nnx.relu(layer(h))
        att_scores = self.attention_layers[-1](h).squeeze(-1)
        att_weights = jax.nn.sigmoid(att_scores) * mask
        valid_counts = jnp.maximum(mask.sum(axis=-1, keepdims=True), 1.0)
        return att_weights / valid_counts
