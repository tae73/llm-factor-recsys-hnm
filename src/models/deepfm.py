"""DeepFM recommendation model in Flax NNX.

Architecture:
    Input: user_cat(B,3) + user_num(B,8) + item_cat(B,5) + item_num(B,2)

    First-order:  Σ bias_embed(cat_i) + Linear(num)  → (B,)
    FM 2nd-order: 0.5*(sum²−sum_of_sq) over all field embeddings → (B,)
    DNN:          concat all field embeddings → FC layers → (B,)

    Output: bias + first_order + fm_second_order + dnn_output → logits (B,)

Device-agnostic: no sharding logic inside the model. Sharding is the caller's
responsibility (src/training/trainer.py).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from src.config import DeepFMConfig


class DeepFMInput(NamedTuple):
    """Batched input for DeepFM."""

    user_cat: jax.Array  # (B, 3) int32
    user_num: jax.Array  # (B, 8) float32
    item_cat: jax.Array  # (B, 5) int32
    item_num: jax.Array  # (B, 2) float32


class DeepFM(nnx.Module):
    """Factorization Machine + Deep Neural Network.

    Args:
        field_dims: Vocabulary sizes for each categorical field.
                    e.g. [8, 5, 5, 132, 51, 22, 57, 11] for 3 user + 5 item cats.
        n_numerical: Total number of numerical features (user_num + item_num).
        config: DeepFMConfig hyperparameters.
        rngs: NNX random number generators.
    """

    def __init__(
        self,
        field_dims: list[int],
        n_numerical: int,
        config: DeepFMConfig,
        *,
        rngs: nnx.Rngs,
    ):
        n_fields = len(field_dims)
        d_embed = config.d_embed

        # --- First-order: bias embeddings for categorical fields ---
        # Use nnx.List for Flax >= 0.12 pytree compatibility
        self.first_order_embeddings = nnx.List(
            [nnx.Embed(num_embeddings=dim, features=1, rngs=rngs) for dim in field_dims]
        )
        # First-order linear for numerical features
        self.first_order_num = nnx.Linear(n_numerical, 1, use_bias=False, rngs=rngs)

        # --- FM second-order: interaction embeddings ---
        self.fm_embeddings = nnx.List(
            [nnx.Embed(num_embeddings=dim, features=d_embed, rngs=rngs) for dim in field_dims]
        )
        # Numerical → d_embed projection (one embedding per numerical field)
        self.num_projection = nnx.Linear(n_numerical, n_numerical * d_embed, use_bias=False, rngs=rngs)

        # --- DNN ---
        total_embed_dim = (n_fields + n_numerical) * d_embed
        dnn_layers: list[nnx.Module] = []
        in_dim = total_embed_dim
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

        # Store config for reference
        self._n_numerical = n_numerical
        self._n_fields = n_fields
        self._d_embed = d_embed

    def embed(self, x: DeepFMInput) -> tuple[jax.Array, jax.Array]:
        """Compute embeddings before FM/DNN prediction.

        Returns:
            (stacked, first_order): stacked is (B, n_total_fields, d_embed),
            first_order is (B, 1).
        """
        cat_indices = jnp.concatenate([x.user_cat, x.item_cat], axis=-1)
        num_features = jnp.concatenate([x.user_num, x.item_num], axis=-1)

        # First-order
        first_order_cat = jnp.concatenate(
            [emb(cat_indices[:, i]) for i, emb in enumerate(self.first_order_embeddings)],
            axis=-1,
        )
        first_order_num = self.first_order_num(num_features)
        first_order = jnp.sum(first_order_cat, axis=-1, keepdims=True) + first_order_num

        # FM embeddings
        cat_embeds = [emb(cat_indices[:, i]) for i, emb in enumerate(self.fm_embeddings)]
        num_proj = self.num_projection(num_features)
        num_embeds_reshaped = num_proj.reshape(
            num_proj.shape[0], self._n_numerical, self._d_embed
        )
        num_embed_list = [num_embeds_reshaped[:, i, :] for i in range(self._n_numerical)]

        all_embeds = cat_embeds + num_embed_list
        stacked = jnp.stack(all_embeds, axis=1)  # (B, n_total_fields, d_embed)
        return stacked, first_order

    def predict_from_embedding(
        self, stacked: jax.Array, first_order: jax.Array
    ) -> jax.Array:
        """Predict logits from pre-computed embeddings.

        Args:
            stacked: (B, n_total_fields, d_embed) field embeddings.
            first_order: (B, 1) first-order output.

        Returns:
            logits (B,).
        """
        # FM second-order
        sum_of_embeds = jnp.sum(stacked, axis=1)
        sum_squared = sum_of_embeds**2
        squared_sum = jnp.sum(stacked**2, axis=1)
        fm_second = 0.5 * jnp.sum(sum_squared - squared_sum, axis=-1, keepdims=True)

        # DNN
        dnn_input = stacked.reshape(stacked.shape[0], -1)
        h = dnn_input
        for layer in self.dnn_layers:
            if isinstance(layer, nnx.Linear):
                h = layer(h)
                h = nnx.relu(h)
            elif isinstance(layer, (nnx.BatchNorm, nnx.Dropout)):
                h = layer(h)
        dnn_out = self.dnn_output(h)

        logits = self.bias[...] + first_order + fm_second + dnn_out
        return logits.squeeze(-1)

    def __call__(self, x: DeepFMInput) -> jax.Array:
        """Forward pass returning logits (B,). Sigmoid applied in loss."""
        stacked, first_order = self.embed(x)
        return self.predict_from_embedding(stacked, first_order)

    def predict_proba(self, x: DeepFMInput) -> jax.Array:
        """Inference: returns sigmoid probabilities (B,)."""
        return jax.nn.sigmoid(self.__call__(x))
