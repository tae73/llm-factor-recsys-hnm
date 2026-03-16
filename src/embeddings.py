"""BGE-base embedding computation for items and users.

Computes sentence embeddings from factual_text_full (items) and
reasoning_text (users) using SentenceTransformer. Saved as float16
.npz for Phase 4 KAR text_encoder reuse.

User embeddings use chunked streaming: model loaded once, texts read
in 100K chunks via PyArrow, MPS cache cleared between chunks to
prevent memory accumulation.

Shared across segmentation (Phase 3) and KAR text_encoder (Phase 4).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EmbeddingConfig

logger = logging.getLogger(__name__)


def compute_item_embeddings(
    fk_path: Path,
    output_path: Path,
    config: EmbeddingConfig = EmbeddingConfig(),
) -> np.ndarray:
    """Encode factual_text_full for all items using BGE-base.

    Args:
        fk_path: Path to factual_knowledge.parquet.
        output_path: Path to save item_bge_embeddings.npz.
        config: Embedding config.

    Returns:
        Embeddings array (n_items, 768) float16.
    """
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet(fk_path, columns=["article_id", "factual_text_full"])
    texts = df["factual_text_full"].fillna("").tolist()
    article_ids = df["article_id"].values

    logger.info("Encoding %d item texts with %s", len(texts), config.model_name)
    model = SentenceTransformer(config.model_name, device=config.device)
    model.max_seq_length = config.max_seq_length

    embeddings = model.encode(
        texts,
        batch_size=config.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings_f16 = embeddings.astype(np.float16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings_f16,
        article_ids=article_ids,
    )
    logger.info("Saved item embeddings: shape=%s, path=%s", embeddings_f16.shape, output_path)
    return embeddings_f16


def compute_user_embeddings(
    rk_path: Path,
    output_path: Path,
    config: EmbeddingConfig = EmbeddingConfig(),
) -> np.ndarray:
    """Encode reasoning_text for all users using BGE-base.

    Single-process streaming: loads model once, reads texts in 100K
    chunks via PyArrow, encodes each chunk, saves to disk, and clears
    MPS/GPU cache between chunks to prevent memory accumulation.

    Supports resume: completed chunk .npy files are skipped.

    Args:
        rk_path: Path to user_profiles.parquet.
        output_path: Path to save user_bge_embeddings.npz.
        config: Embedding config.

    Returns:
        Embeddings array (n_users, 768) float16.
    """
    import gc

    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

    customer_ids = pd.read_parquet(rk_path, columns=["customer_id"])["customer_id"].values
    n_total = len(customer_ids)
    chunk_size = 100_000
    tmp_dir = output_path.parent / "_user_emb_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (n_total + chunk_size - 1) // chunk_size

    # Clean up any stale .tmp files from interrupted writes
    stale = {f for f in tmp_dir.glob("*.tmp.npy")} | {f for f in tmp_dir.glob("*.npy.tmp.npy")}
    for tmp_file in stale:
        logger.warning("Removing stale temp file: %s", tmp_file.name)
        tmp_file.unlink()

    existing = sorted(tmp_dir.glob("chunk_*.npy"))
    if existing:
        logger.info("Resuming: %d/%d chunks already completed", len(existing), n_chunks)

    logger.info(
        "Encoding %d user texts with %s (streaming, chunk=%d, %d chunks)",
        n_total, config.model_name, chunk_size, n_chunks,
    )

    # Load model once (CPU avoids MPS memory leak on large batches)
    model = SentenceTransformer(config.model_name, device=config.device)
    model.max_seq_length = config.max_seq_length
    logger.info("Model loaded on device: %s", model.device)

    # Stream texts from Parquet and encode in chunks
    pf = pq.ParquetFile(rk_path)
    buffer: list[str] = []
    chunk_idx = 0
    chunk_offset = 0

    for batch in pf.iter_batches(batch_size=10_000, columns=["reasoning_text"]):
        buffer.extend(s.as_py() or "" for s in batch.column("reasoning_text"))

        while len(buffer) >= chunk_size:
            chunk_texts = buffer[:chunk_size]
            buffer = buffer[chunk_size:]
            _encode_and_save_chunk(
                model, chunk_texts, chunk_idx, chunk_offset, n_chunks, tmp_dir, config,
            )
            chunk_offset += chunk_size
            chunk_idx += 1
            del chunk_texts
            gc.collect()

    # Handle remaining texts
    if buffer:
        _encode_and_save_chunk(
            model, buffer, chunk_idx, chunk_offset, n_chunks, tmp_dir, config,
        )
        del buffer
        gc.collect()

    # Unload model before merge to free memory
    del model
    gc.collect()
    try:
        import torch
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except (ImportError, AttributeError):
        pass

    # Merge all chunks -> final .npz
    logger.info("Merging %d chunks into %s", n_chunks, output_path)
    chunk_files = sorted(tmp_dir.glob("chunk_*.npy"))
    embeddings = np.concatenate([np.load(p) for p in chunk_files], axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        customer_ids=customer_ids,
    )
    logger.info("Saved user embeddings: shape=%s, path=%s", embeddings.shape, output_path)

    # Cleanup temp chunks
    shutil.rmtree(tmp_dir)
    logger.info("Cleaned up temp dir: %s", tmp_dir)
    return embeddings


def _encode_and_save_chunk(
    model,
    texts: list[str],
    chunk_idx: int,
    chunk_offset: int,
    n_chunks: int,
    tmp_dir: Path,
    config: EmbeddingConfig,
) -> None:
    """Encode a single chunk of texts and save embeddings to disk."""
    import gc

    chunk_path = tmp_dir / f"chunk_{chunk_offset:07d}.npy"
    if chunk_path.exists():
        expected_size = len(texts) * 768 * 2 + 128  # float16 + npy header
        actual_size = chunk_path.stat().st_size
        if actual_size >= expected_size:
            logger.info(
                "  Chunk %d/%d [%d:%d] already exists (%d bytes), skipping",
                chunk_idx + 1, n_chunks, chunk_offset, chunk_offset + len(texts), actual_size,
            )
            return
        else:
            logger.warning(
                "  Chunk %d/%d [%d:%d] corrupt (%d < %d bytes), re-encoding",
                chunk_idx + 1, n_chunks, chunk_offset, chunk_offset + len(texts),
                actual_size, expected_size,
            )
            chunk_path.unlink()

    logger.info(
        "  Chunk %d/%d [%d:%d] encoding %d texts...",
        chunk_idx + 1, n_chunks, chunk_offset, chunk_offset + len(texts), len(texts),
    )

    # Sub-chunk encoding to prevent MPS memory accumulation.
    # MPS leaks memory within a single model.encode() call for large text lists.
    # Encoding 1K texts at a time + cache clearing keeps memory stable.
    sub_size = 1_000
    sub_results: list[np.ndarray] = []
    for sub_start in range(0, len(texts), sub_size):
        sub_texts = texts[sub_start : sub_start + sub_size]
        sub_emb = model.encode(
            sub_texts,
            batch_size=config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        sub_results.append(sub_emb.astype(np.float16))
        del sub_emb, sub_texts
        gc.collect()
        try:
            import torch
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass
    logger.info(
        "  Chunk %d/%d encoded %d sub-batches",
        chunk_idx + 1, n_chunks, len(sub_results),
    )

    emb_f16 = np.concatenate(sub_results, axis=0)
    del sub_results
    gc.collect()

    tmp_path = chunk_path.with_name(chunk_path.stem + ".tmp.npy")
    np.save(tmp_path, emb_f16)  # .npy suffix preserved, no auto-append
    tmp_path.rename(chunk_path)  # atomic on same filesystem
    logger.info("  Saved chunk %d: shape=%s", chunk_idx + 1, emb_f16.shape)

    del emb_f16
    gc.collect()
    try:
        import torch
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except (ImportError, AttributeError):
        pass


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings .npz -> (embeddings float32, ids).

    Converts float16 back to float32 for computation.
    """
    data = np.load(path, allow_pickle=True)
    key = "article_ids" if "article_ids" in data else "customer_ids"
    return data["embeddings"].astype(np.float32), data[key]
