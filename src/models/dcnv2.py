"""DCN-v2 (Deep & Cross Network v2) recommendation model in Flax NNX.

Architecture:
    Input: user_cat(B,3) + user_num(B,8) + item_cat(B,5) + item_num(B,2)
           (same DeepFMInput interface)

    Embedding Layer:
      cat: 8 fields × Embed(vocab, d_embed) → (B, 8, d_embed)
      num: Linear(10, 10*d_embed) → reshape → (B, 10, d_embed)
      x0 = flatten → (B, D)  where D = 18 * d_embed

    Cross Network v2 (n_cross_layers):
      CrossLayerV2: x_{l+1} = x0 ⊙ (MoE(x_l) + b) + x_l
      MoE: Σ_k g_k · U_k · V_k^T · x_l (low-rank mixture of experts)
      → x_cross: (B, D)

    DNN (FC + BN + Dropout):
      → x_deep: (B, h_last)

    Output: concat([x_cross, x_deep]) → Linear → logits (B,)

Reference: Wang et al. "DCN V2: Improved Deep & Cross Network" (WWW 2021)

Device-agnostic: no sharding logic inside the model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.config import DCNv2Config
from src.models.deepfm import DeepFMInput


class CrossLayerV2(nnx.Module):
    """Single cross layer with Mixture-of-Experts low-rank decomposition.

    x_{l+1} = x0 ⊙ (MoE(x_l) + b) + x_l

    MoE(x_l) = Σ_k g_k(x_l) · U_k · V_k^T · x_l
    where g_k are softmax gating weights.
    """

    def __init__(
        self,
        d_input: int,
        n_experts: int,
        d_low_rank: int,
        *,
        rngs: nnx.Rngs,
    ):
        self._n_experts = n_experts
        self._d_input = d_input
        self._d_low_rank = d_low_rank

        # V: (n_experts, d_low_rank, d_input) — projects x_l down
        self.V = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (n_experts, d_low_rank, d_input))
        )
        # U: (n_experts, d_input, d_low_rank) — projects back up
        self.U = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (n_experts, d_input, d_low_rank))
        )
        # Bias: (d_input,)
        self.bias = nnx.Param(jnp.zeros((d_input,)))

        # Gating network: Linear(d_input → n_experts)
        self.gate = nnx.Linear(d_input, n_experts, rngs=rngs)

    def __call__(self, x0: jax.Array, xl: jax.Array) -> jax.Array:
        """Forward: x_{l+1} = x0 ⊙ (MoE(x_l) + b) + x_l.

        Args:
            x0: Original input (B, D).
            xl: Input from previous layer (B, D).

        Returns:
            x_{l+1}: (B, D).
        """
        # V^T @ x_l: (n_experts, d_low_rank, D) × (B, D) → (B, n_experts, d_low_rank)
        vx = jnp.einsum("eld,bd->bel", self.V[...], xl)

        # U @ (V^T @ x_l): (n_experts, D, d_low_rank) × (B, n_experts, d_low_rank) → (B, n_experts, D)
        uvx = jnp.einsum("edl,bel->bed", self.U[...], vx)

        # Gating: softmax(Linear(x_l)) → (B, n_experts)
        gate_weights = jax.nn.softmax(self.gate(xl), axis=-1)  # (B, n_experts)

        # Weighted sum: (B, n_experts, 1) * (B, n_experts, D) → sum → (B, D)
        expert_out = jnp.sum(gate_weights[:, :, None] * uvx, axis=1)  # (B, D)

        # Cross: x0 ⊙ (expert_out + bias) + xl (element-wise product + residual)
        return x0 * (expert_out + self.bias[...]) + xl


class DCNv2(nnx.Module):
    """Deep & Cross Network v2.

    Args:
        field_dims: Vocabulary sizes for each categorical field.
        n_numerical: Total number of numerical features (user_num + item_num).
        config: DCNv2Config hyperparameters.
        rngs: NNX random number generators.
    """

    def __init__(
        self,
        field_dims: list[int],
        n_numerical: int,
        config: DCNv2Config,
        *,
        rngs: nnx.Rngs,
    ):
        n_fields = len(field_dims)
        d_embed = config.d_embed

        # --- Embedding layer (same pattern as DeepFM) ---
        self.cat_embeddings = nnx.List(
            [nnx.Embed(num_embeddings=dim, features=d_embed, rngs=rngs) for dim in field_dims]
        )
        self.num_projection = nnx.Linear(
            n_numerical, n_numerical * d_embed, use_bias=False, rngs=rngs
        )

        # Total flattened dimension
        total_dim = (n_fields + n_numerical) * d_embed

        # --- Cross Network v2 ---
        self.cross_layers = nnx.List(
            [
                CrossLayerV2(total_dim, config.n_experts, config.d_low_rank, rngs=rngs)
                for _ in range(config.n_cross_layers)
            ]
        )

        # --- DNN ---
        dnn_layers: list[nnx.Module] = []
        in_dim = total_dim
        for hidden_dim in config.dnn_hidden_dims:
            dnn_layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            if config.use_batch_norm:
                dnn_layers.append(nnx.BatchNorm(hidden_dim, rngs=rngs))
            dnn_layers.append(nnx.Dropout(rate=config.dropout_rate, rngs=rngs))
            in_dim = hidden_dim
        self.dnn_layers = nnx.List(dnn_layers)

        # --- Output: concat(cross, deep) → logits ---
        self.output_linear = nnx.Linear(total_dim + in_dim, 1, rngs=rngs)

        # --- Global bias ---
        self.bias = nnx.Param(jnp.zeros(()))

        # Store dims
        self._n_numerical = n_numerical
        self._n_fields = n_fields
        self._d_embed = d_embed

    def embed(self, x: DeepFMInput) -> jax.Array:
        """Compute flattened field embeddings before cross/DNN.

        Returns:
            stacked: (B, n_total_fields, d_embed).
        """
        cat_indices = jnp.concatenate([x.user_cat, x.item_cat], axis=-1)
        num_features = jnp.concatenate([x.user_num, x.item_num], axis=-1)

        cat_embeds = [emb(cat_indices[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        num_proj = self.num_projection(num_features)
        num_embeds = num_proj.reshape(num_proj.shape[0], self._n_numerical, self._d_embed)
        num_embed_list = [num_embeds[:, i, :] for i in range(self._n_numerical)]

        all_embeds = cat_embeds + num_embed_list
        stacked = jnp.stack(all_embeds, axis=1)
        return stacked

    def predict_from_embedding(self, stacked: jax.Array) -> jax.Array:
        """Predict logits from pre-computed embeddings.

        Args:
            stacked: (B, n_total_fields, d_embed) field embeddings.

        Returns:
            logits (B,).
        """
        x0 = stacked.reshape(stacked.shape[0], -1)

        # Cross Network
        x_cross = x0
        for cross_layer in self.cross_layers:
            x_cross = cross_layer(x0, x_cross)

        # DNN
        h = x0
        for layer in self.dnn_layers:
            if isinstance(layer, nnx.Linear):
                h = layer(h)
                h = nnx.relu(h)
            elif isinstance(layer, (nnx.BatchNorm, nnx.Dropout)):
                h = layer(h)

        combined = jnp.concatenate([x_cross, h], axis=-1)
        logits = self.output_linear(combined) + self.bias[...]
        return logits.squeeze(-1)

    def __call__(self, x: DeepFMInput) -> jax.Array:
        """Forward pass returning logits (B,). Sigmoid applied in loss."""
        stacked = self.embed(x)
        return self.predict_from_embedding(stacked)

    def predict_proba(self, x: DeepFMInput) -> jax.Array:
        """Inference: returns sigmoid probabilities (B,)."""
        return jax.nn.sigmoid(self.__call__(x))
