"""SASRec (Self-Attentive Sequential Recommendation) model in Flax NNX.

Architecture:
    Input: history(B, T) int32, hist_len(B,) int32

    1. Item embedding + learnable position embedding → (B, T, d)
    2. N transformer blocks:
       - Causal multi-head self-attention (lower-triangular + padding mask)
       - Point-wise feed-forward (2-layer MLP)
       - LayerNorm + residual + dropout
    3. Extract last valid position → user representation (B, d)
    4. Dot product with target item embedding → logits (B,)
    5. Full catalog scoring: user_embed @ all_item_embeds.T → (n_items,)

Reference: Kang & McAuley "Self-Attentive Sequential Recommendation" (ICDM 2018)

Device-agnostic: no sharding logic inside the model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from src.config import SASRecConfig


class SASRecInput(NamedTuple):
    """Batched input for SASRec."""

    history: jax.Array    # (B, T) int32 — item index sequence (0=PAD)
    hist_len: jax.Array   # (B,) int32 — actual lengths


class TransformerBlock(nnx.Module):
    """Single transformer block: causal self-attention + FFN.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout_rate: Dropout probability.
        rngs: NNX random number generators.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            decode=False,
            rngs=rngs,
        )
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ffn1 = nnx.Linear(d_model, d_model * 4, rngs=rngs)
        self.ffn2 = nnx.Linear(d_model * 4, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """Forward pass with causal + padding mask.

        Args:
            x: Input embeddings (B, T, d).
            mask: Combined mask (B, 1, T, T) — True means attend, False means mask.

        Returns:
            Output embeddings (B, T, d).
        """
        # Self-attention + residual
        attn_out = self.attention(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + residual
        ffn_out = self.ffn2(self.dropout(nnx.relu(self.ffn1(x))))
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class SASRec(nnx.Module):
    """Self-Attentive Sequential Recommendation.

    Args:
        n_items: Total number of items (+1 for PAD at idx 0 internally).
        config: SASRecConfig hyperparameters.
        rngs: NNX random number generators.
    """

    def __init__(
        self,
        n_items: int,
        config: SASRecConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self._n_items = n_items
        self._d_embed = config.d_embed
        self._max_seq_len = config.max_seq_len

        # Item embedding: n_items + 1 (index 0 = PAD)
        self.item_embedding = nnx.Embed(
            num_embeddings=n_items + 1,
            features=config.d_embed,
            rngs=rngs,
        )

        # Learnable position embedding
        self.position_embedding = nnx.Embed(
            num_embeddings=config.max_seq_len,
            features=config.d_embed,
            rngs=rngs,
        )

        self.input_norm = nnx.LayerNorm(config.d_embed, rngs=rngs)
        self.input_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        # Transformer blocks
        self.blocks = nnx.List([
            TransformerBlock(
                d_model=config.d_embed,
                n_heads=config.n_heads,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )
            for _ in range(config.n_blocks)
        ])

    def _build_mask(self, history: jax.Array) -> jax.Array:
        """Build combined causal + padding mask.

        Args:
            history: (B, T) int32 item indices (0=PAD).

        Returns:
            mask: (B, 1, T, T) bool — True for positions to attend to.
        """
        T = history.shape[1]

        # Causal mask: lower triangular (T, T)
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))  # (T, T)

        # Padding mask: True for non-PAD positions
        pad_mask = (history > 0)  # (B, T)

        # Key mask: which keys are valid → (B, 1, 1, T)
        key_mask = pad_mask[:, None, None, :]

        # Combined: causal AND key_padding
        # causal_mask is (T, T), key_mask is (B, 1, 1, T) → broadcast to (B, 1, T, T)
        combined = causal_mask[None, None, :, :] & key_mask

        return combined

    def encode(self, x: SASRecInput) -> jax.Array:
        """Encode sequence → per-position representations.

        Args:
            x: SASRecInput with history and hist_len.

        Returns:
            Sequence representations (B, T, d).
        """
        B, T = x.history.shape

        # Item + position embeddings
        item_emb = self.item_embedding(x.history)  # (B, T, d)
        positions = jnp.arange(T)[None, :]  # (1, T)
        pos_emb = self.position_embedding(positions)  # (1, T, d)

        h = self.input_norm(item_emb + pos_emb)
        h = self.input_dropout(h)

        # Build mask
        mask = self._build_mask(x.history)  # (B, 1, T, T)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, mask)

        return h  # (B, T, d)

    def get_user_embedding(self, x: SASRecInput) -> jax.Array:
        """Get user representation from last valid position.

        Args:
            x: SASRecInput with history and hist_len.

        Returns:
            User embeddings (B, d).
        """
        h = self.encode(x)  # (B, T, d)

        # Extract last valid position: hist_len - 1 (0-indexed)
        # For left-padded sequences, the last valid position is always T-1
        # if the sequence has at least one item. Use hist_len to find it.
        B, T, _ = h.shape
        # Last valid index: for left-padded, it's T - 1 if hist_len > 0
        # For sequences where hist_len > 0, the last valid item is at position T-1
        # (since we left-pad in sequences.py)
        last_idx = jnp.full((B,), T - 1, dtype=jnp.int32)

        # Gather: (B, d)
        user_embed = h[jnp.arange(B), last_idx]
        return user_embed

    def embed(
        self, x: SASRecInput, target_item_idx: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array | None]:
        """Compute user and target item embeddings.

        Args:
            x: SASRecInput.
            target_item_idx: (B,) int32 target item index (1-based).

        Returns:
            (user_embed, target_embed): user_embed (B, d), target_embed (B, d) or None.
        """
        user_embed = self.get_user_embedding(x)
        target_embed = self.item_embedding(target_item_idx) if target_item_idx is not None else None
        return user_embed, target_embed

    def predict_from_embedding(
        self, user_embed: jax.Array, target_embed: jax.Array
    ) -> jax.Array:
        """Predict logits from pre-computed embeddings via dot product.

        Args:
            user_embed: (B, d) user representation.
            target_embed: (B, d) target item embedding.

        Returns:
            logits (B,).
        """
        return jnp.sum(user_embed * target_embed, axis=-1)

    def __call__(self, x: SASRecInput, target_item_idx: jax.Array | None = None) -> jax.Array:
        """Forward pass returning logits (B,).

        Args:
            x: SASRecInput with history and hist_len.
            target_item_idx: (B,) int32 — target item index (1-based, 0=PAD).
                If None, returns user embeddings instead.

        Returns:
            logits (B,) — dot product of user_embed and target_embed.
        """
        user_embed, target_embed = self.embed(x, target_item_idx)

        if target_embed is None:
            return user_embed

        return self.predict_from_embedding(user_embed, target_embed)

    def predict_proba(self, x: SASRecInput, target_item_idx: jax.Array) -> jax.Array:
        """Inference: returns sigmoid probabilities (B,)."""
        return jax.nn.sigmoid(self.__call__(x, target_item_idx))

    def score_all_items(self, x: SASRecInput) -> jax.Array:
        """Score all items for a batch of users.

        Args:
            x: SASRecInput.

        Returns:
            Scores (B, n_items+1) — dot product with all item embeddings.
        """
        user_embed = self.get_user_embedding(x)  # (B, d)
        all_item_embeds = self.item_embedding.embedding[...]  # (n_items+1, d)
        scores = user_embed @ all_item_embeds.T  # (B, n_items+1)
        return scores
