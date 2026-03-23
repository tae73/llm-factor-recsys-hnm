"""Generate 7 ablation-variant item embeddings for Layer analysis.

Produces one .npz per layer combo (L1, L2, L3, L1+L2, L1+L3, L2+L3, L1+L2+L3)
by composing ablation texts via text_composer and encoding with BGE-base.
BGE model is loaded once and reused across all variants.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.knowledge.factual.text_composer import LAYER_COMBOS, build_all_ablation_texts

logger = logging.getLogger(__name__)


def _combo_to_filename(combo: str) -> str:
    """Convert combo name to a safe filename: 'L1+L2+L3' → 'l1_l2_l3'."""
    return combo.lower().replace("+", "_")


def generate_ablation_texts(
    fk_path: Path,
    articles_path: Path,
) -> dict[str, list[str]]:
    """Generate 7 ablation text variants for all items.

    Args:
        fk_path: Path to factual_knowledge.parquet.
        articles_path: Path to articles.parquet.

    Returns:
        Dict of combo_name → list of texts (same order as article_ids).
    """
    fk = pd.read_parquet(fk_path)
    articles = pd.read_parquet(articles_path, columns=[
        "article_id", "product_type_name", "product_group_name",
        "colour_group_name", "graphical_appearance_name", "section_name",
    ])
    fk["article_id"] = fk["article_id"].astype(str)
    articles["article_id"] = articles["article_id"].astype(str)

    merged = articles.merge(fk, on="article_id", how="inner")
    logger.info("Merged %d items for ablation text generation", len(merged))

    # Initialize result containers
    combo_texts: dict[str, list[str]] = {c: [] for c in LAYER_COMBOS}

    for _, row in merged.iterrows():
        article_meta = {
            "product_type_name": row.get("product_type_name"),
            "product_group_name": row.get("product_group_name"),
            "colour_group_name": row.get("colour_group_name"),
            "graphical_appearance_name": row.get("graphical_appearance_name"),
            "section_name": row.get("section_name"),
        }
        knowledge = {k: row[k] for k in row.index if k.startswith(("l1_", "l2_", "l3_"))}
        super_cat = row.get("super_category", "Apparel")

        variants = build_all_ablation_texts(article_meta, knowledge, super_cat)
        for combo in LAYER_COMBOS:
            combo_texts[combo].append(variants.get(combo, ""))

    return combo_texts


def encode_and_save_ablation_embeddings(
    fk_path: Path,
    articles_path: Path,
    output_dir: Path,
    model_name: str = "BAAI/bge-base-en-v1.5",
    device: str = "mps",
    batch_size: int = 256,
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Generate and save ablation embeddings for all 7 combos.

    Args:
        fk_path: Path to factual_knowledge.parquet.
        articles_path: Path to articles.parquet.
        output_dir: Directory to save .npz files.
        model_name: BGE model name.
        device: Compute device.
        batch_size: Encoding batch size.
        skip_existing: Skip combos that already have .npz files.

    Returns:
        Dict of combo_name → Path to saved .npz file.
    """
    from sentence_transformers import SentenceTransformer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check which combos need generation
    combos_to_generate = []
    for combo in LAYER_COMBOS:
        out_path = output_dir / f"{_combo_to_filename(combo)}.npz"
        if skip_existing and out_path.exists():
            logger.info("Skipping %s (already exists: %s)", combo, out_path)
        else:
            combos_to_generate.append(combo)

    if not combos_to_generate:
        logger.info("All ablation embeddings already exist, nothing to do")
        return {c: output_dir / f"{_combo_to_filename(c)}.npz" for c in LAYER_COMBOS}

    # Generate texts
    logger.info("Generating ablation texts for %d combos...", len(combos_to_generate))
    all_texts = generate_ablation_texts(fk_path, articles_path)

    # Get article_ids for alignment
    fk = pd.read_parquet(fk_path, columns=["article_id"])
    articles = pd.read_parquet(articles_path, columns=["article_id"])
    articles["article_id"] = articles["article_id"].astype(str)
    fk["article_id"] = fk["article_id"].astype(str)
    merged_ids = articles.merge(fk[["article_id"]], on="article_id", how="inner")
    article_ids = merged_ids["article_id"].values

    # Load BGE model once
    logger.info("Loading BGE model: %s on %s", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 512

    saved_paths: dict[str, Path] = {}

    for combo in LAYER_COMBOS:
        out_path = output_dir / f"{_combo_to_filename(combo)}.npz"
        if skip_existing and out_path.exists():
            saved_paths[combo] = out_path
            continue

        texts = all_texts[combo]
        logger.info("Encoding %s: %d texts...", combo, len(texts))

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings_f16 = embeddings.astype(np.float16)

        np.savez_compressed(
            out_path,
            embeddings=embeddings_f16,
            article_ids=article_ids,
        )
        logger.info("Saved %s: shape=%s → %s", combo, embeddings_f16.shape, out_path)
        saved_paths[combo] = out_path

    return saved_paths


def load_ablation_embeddings(
    ablation_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all ablation embeddings from directory.

    Returns:
        Dict of combo_name → (embeddings float32, article_ids).
    """
    result = {}
    for combo in LAYER_COMBOS:
        path = ablation_dir / f"{_combo_to_filename(combo)}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            emb = data["embeddings"].astype(np.float32)
            ids = data["article_ids"]
            result[combo] = (emb, ids)
            logger.info("Loaded %s: shape=%s", combo, emb.shape)
        else:
            logger.warning("Missing ablation embeddings: %s", path)
    return result
