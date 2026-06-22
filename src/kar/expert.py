"""Expert MLP modules for KAR.

Two instances of the same Expert class:
- FactualExpert: item BGE (768) → d_rec (64)
- ReasoningExpert: user BGE (768) → d_rec (64)

Each expert is a 2-layer ReLU MLP with dropout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.config import ExpertConfig


class Expert(nnx.Module):
    """2-layer ReLU MLP: (B, d_enc) → (B, d_rec).

    Args:
        config: ExpertConfig with d_enc, d_hidden, d_rec, n_layers, dropout_rate.
        rngs: NNX random number generators.
    """

    def __init__(self, config: ExpertConfig, *, rngs: nnx.Rngs):
        self._config = config

        layers: list[nnx.Module] = []
        in_dim = config.d_enc
        for _ in range(config.n_layers - 1):
            layers.append(nnx.Linear(in_dim, config.d_hidden, rngs=rngs))
            layers.append(nnx.Dropout(rate=config.dropout_rate, rngs=rngs))
            in_dim = config.d_hidden
        layers.append(nnx.Linear(in_dim, config.d_rec, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, h: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            h: Input embeddings (B, d_enc=768).

        Returns:
            Expert output (B, d_rec=64).
        """
        x = h
        for layer in self.layers:
            if isinstance(layer, nnx.Linear):
                x = layer(x)
                # Apply ReLU for hidden layers, not the last projection
                if layer is not self.layers[-1]:
                    x = nnx.relu(x)
            elif isinstance(layer, nnx.Dropout):
                x = layer(x)
        return x
