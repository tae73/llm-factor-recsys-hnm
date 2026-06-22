"""Build the MISSING metadata-only item embedding (de-risk probe prerequisite).

WHAT GENERALIZES
    The 7 ablation embeddings on disk (l1, l2, l3, l1_l2, ...) ALL confound
    H&M metadata, because src/knowledge/factual/text_composer.construct_factual_text
    ALWAYS prepends _compose_metadata_text (text_composer.py:195-198). There is no
    metadata-only embedding, so the core RQ "do LLM L2/L3 layers add value BEYOND
    metadata" is untestable from disk. This script composes the metadata-only text
    (construct_factual_text(meta, None, None, None, ...)) and encodes it with the
    SAME BGE pipeline as the other variants, producing data/embeddings/ablation/meta.npz
    aligned (same article_ids order) with l1.npz ... l1_l2_l3.npz.

THE RESULT
    +-------------------------------------------------------------+
    |  meta.npz : (n_items, 768) f16, L2-normalized, article_ids  |
    |  ladder now complete: META -> META+L1 (l1) -> +L2 -> +L3    |
    +-------------------------------------------------------------+

HONEST reduces_check
    This does NOT by itself test incremental value; it only manufactures the missing
    baseline so probe_01 can. If meta.npz cannot be produced (missing articles/fk),
    the whole Track-A probe is blocked.

VERDICT
    Artifact-builder, not a GO/NO-GO probe. Success = meta.npz written with shape
    matching the existing ablation embeddings.

Usage:
    uv run python witnesses/build_meta_embedding.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.knowledge.factual.text_composer import construct_factual_text

FK_PATH = Path("data/knowledge/factual/factual_knowledge.parquet")
ARTICLES_PATH = Path("data/processed/articles.parquet")
OUT_PATH = Path("data/embeddings/ablation/meta.npz")

META_COLS = [
    "article_id",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "graphical_appearance_name",
    "section_name",
]


def build_metadata_texts() -> tuple[list[str], np.ndarray]:
    """Compose metadata-only text per item, aligned to the ablation merge order.

    Mirrors src/analysis/ablation_embeddings.generate_ablation_texts: inner-join
    articles (LEFT) with fk so the resulting article_id order matches l1.npz etc.
    """
    fk = pd.read_parquet(FK_PATH, columns=["article_id", "super_category"])
    articles = pd.read_parquet(ARTICLES_PATH, columns=META_COLS)
    fk["article_id"] = fk["article_id"].astype(str)
    articles["article_id"] = articles["article_id"].astype(str)

    merged = articles.merge(fk, on="article_id", how="inner")
    print(f"[meta] merged items: {len(merged)}")

    texts: list[str] = []
    for _, row in merged.iterrows():
        article_meta = {
            "product_type_name": row.get("product_type_name"),
            "product_group_name": row.get("product_group_name"),
            "colour_group_name": row.get("colour_group_name"),
            "graphical_appearance_name": row.get("graphical_appearance_name"),
            "section_name": row.get("section_name"),
        }
        super_cat = row.get("super_category", "Apparel")
        # metadata-only: L1/L2/L3 all None
        texts.append(construct_factual_text(article_meta, None, None, None, super_cat))

    article_ids = merged["article_id"].to_numpy()
    return texts, article_ids


def main() -> None:
    from sentence_transformers import SentenceTransformer

    if OUT_PATH.exists():
        print(f"[meta] {OUT_PATH} already exists; skipping.")
        return

    texts, article_ids = build_metadata_texts()
    n_empty = sum(1 for t in texts if not t.strip())
    print(f"[meta] composed {len(texts)} texts ({n_empty} empty)")
    print(f"[meta] sample text: {texts[0][:160]!r}")

    # GPU 0 is nearly full on this host; prefer cuda:1 (falls back to cuda:0/cpu).
    device = "cuda:1"
    print(f"[meta] loading BGE on {device} ...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    model.max_seq_length = 512

    emb = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float16)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, embeddings=emb, article_ids=article_ids)
    print(f"[meta] saved {OUT_PATH}: shape={emb.shape}")


if __name__ == "__main__":
    main()
