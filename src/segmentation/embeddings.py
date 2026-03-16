"""Backward-compat re-exports. Use src.embeddings directly."""

from src.embeddings import compute_item_embeddings, compute_user_embeddings, load_embeddings

__all__ = ["compute_item_embeddings", "compute_user_embeddings", "load_embeddings"]
