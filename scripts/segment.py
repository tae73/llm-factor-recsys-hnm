"""CLI entry point for customer/product segmentation.

Computes BGE embeddings, structured vectors (L1/L2/L3), and runs
5-level customer segmentation + product clustering.

Usage:
    # Full pipeline
    python scripts/segment.py \
        --fk-dir data/knowledge/factual \
        --rk-dir data/knowledge/reasoning \
        --data-dir data/processed \
        --output-dir data/segmentation

    # Embeddings only (Phase 4 preparation)
    python scripts/segment.py ... --embeddings-only

    # Skip embeddings (already computed)
    python scripts/segment.py ... --skip-embeddings
"""

import logging
from pathlib import Path

import typer

from src.config import EmbeddingConfig, SegmentationConfig

app = typer.Typer(help="Customer/product segmentation using LLM-extracted attributes")
logger = logging.getLogger(__name__)


@app.command()
def main(
    fk_dir: Path = typer.Option("data/knowledge/factual", help="Factual knowledge directory"),
    rk_dir: Path = typer.Option("data/knowledge/reasoning", help="Reasoning knowledge directory"),
    data_dir: Path = typer.Option("data/processed", help="Processed data directory"),
    output_dir: Path = typer.Option("data/segmentation", help="Output directory"),
    embeddings_dir: Path = typer.Option("data/embeddings", help="Embeddings output directory"),
    bge_model: str = typer.Option("BAAI/bge-base-en-v1.5", help="BGE model name"),
    bge_batch_size: int = typer.Option(256, help="BGE encoding batch size"),
    customer_method: str = typer.Option("kmeans", help="Clustering method"),
    embeddings_only: bool = typer.Option(False, help="Only compute embeddings"),
    skip_embeddings: bool = typer.Option(False, help="Skip embedding computation"),
    random_seed: int = typer.Option(42, help="Random seed"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
) -> None:
    """Run customer/product segmentation pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    emb_config = EmbeddingConfig(
        model_name=bge_model,
        batch_size=bge_batch_size,
    )
    seg_config = SegmentationConfig(
        customer_method=customer_method,
        random_seed=random_seed,
    )

    fk_path = fk_dir / "factual_knowledge.parquet"
    rk_path = rk_dir / "user_profiles.parquet"
    txn_path = data_dir / "transactions.parquet"
    articles_path = data_dir / "articles.parquet"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    item_emb_path = embeddings_dir / "item_bge_embeddings.npz"
    user_emb_path = embeddings_dir / "user_bge_embeddings.npz"

    # --- Step 1: Embeddings ---
    if not skip_embeddings:
        from src.embeddings import compute_item_embeddings, compute_user_embeddings

        if item_emb_path.exists():
            print(f"[segment] Item embeddings already exist: {item_emb_path}")
        else:
            print("[segment] Computing item embeddings...")
            item_emb = compute_item_embeddings(fk_path, item_emb_path, emb_config)
            print(f"  Item embeddings: {item_emb.shape}")

        if user_emb_path.exists():
            print(f"[segment] User embeddings already exist: {user_emb_path}")
        else:
            print("[segment] Computing user embeddings...")
            user_emb = compute_user_embeddings(rk_path, user_emb_path, emb_config)
            print(f"  User embeddings: {user_emb.shape}")
    else:
        print("[segment] Skipping embeddings (--skip-embeddings)")

    if embeddings_only:
        print("[segment] Embeddings only mode — done.")
        return

    # --- Step 2: Customer Segmentation ---
    from src.segmentation.customer import run_customer_segmentation

    print("[segment] Running customer segmentation (5 levels)...")
    cust_result = run_customer_segmentation(
        rk_path=rk_path,
        txn_path=txn_path,
        fk_path=fk_path,
        user_emb_path=user_emb_path,
        output_dir=output_dir,
        config=seg_config,
    )

    print("\n=== Customer Segmentation Summary ===")
    print(f"  L1: k={cust_result.l1_cluster.k}, silhouette={cust_result.l1_cluster.silhouette:.4f}")
    print(f"  L2: k={cust_result.l2_cluster.k}, silhouette={cust_result.l2_cluster.silhouette:.4f}")
    print(f"  L3: k={cust_result.l3_cluster.k}, silhouette={cust_result.l3_cluster.silhouette:.4f}")
    print(f"  Semantic: k={cust_result.semantic_cluster.k}, silhouette={cust_result.semantic_cluster.silhouette:.4f}")
    print(f"  Topic: n_topics={cust_result.topic.n_topics}, outliers={cust_result.topic.outlier_count}")

    # --- Step 3: Product Clustering ---
    from src.segmentation.product import run_product_clustering

    print("[segment] Running product clustering...")
    prod_result = run_product_clustering(
        item_emb_path=item_emb_path,
        articles_path=articles_path,
        output_dir=output_dir,
        config=seg_config,
    )

    print("\n=== Product Clustering Summary ===")
    print(f"  Clusters: {prod_result.cluster.k}")
    print(f"  Silhouette: {prod_result.cluster.silhouette:.4f}")
    print(f"  ARI vs native: {prod_result.ari_vs_native:.4f}")
    print(f"  Cross-category pairs: {len(prod_result.cross_category_pairs)}")

    # --- Step 4: Analysis ---
    from src.segmentation.analysis import (
        compute_segment_statistics,
        cross_layer_ari,
        profile_segments,
    )

    print("[segment] Computing analysis...")
    import pandas as pd

    profiles_df = pd.read_parquet(rk_path)
    segments_df = cust_result.segments_df

    # Profile each level
    all_profiles = {}
    for level in ["l1", "l2", "l3", "semantic", "topic"]:
        try:
            profs = profile_segments(segments_df, profiles_df, level=level)
            all_profiles[level] = [
                {"segment_id": p.segment_id, "size": p.size, "fraction": p.fraction, "label": p.label, "top_attributes": p.top_attributes}
                for p in profs
            ]
        except Exception as e:
            logger.warning("Could not profile level %s: %s", level, e)

    import json

    with open(output_dir / "segment_profiles.json", "w") as f:
        json.dump(all_profiles, f, indent=2)

    # Cross-layer ARI
    ari_df = cross_layer_ari(segments_df)
    print("\n=== Cross-Layer ARI Matrix ===")
    print(ari_df.to_string(float_format="%.3f"))

    # Per-level statistics
    for level in ["l1", "l2", "l3"]:
        try:
            stats = compute_segment_statistics(segments_df, profiles_df, level=level)
            stats.to_csv(output_dir / f"segment_stats_{level}.csv")
        except Exception as e:
            logger.warning("Could not compute stats for %s: %s", level, e)

    print(f"\n[segment] All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
