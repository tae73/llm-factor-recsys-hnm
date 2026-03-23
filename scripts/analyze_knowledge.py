"""CLI entry point for Knowledge-Purchase analysis.

Runs attribute-level information theory and sparsity analysis
to validate LLM-extracted L1/L2/L3 attribute value for recommendations.

Usage:
    # Generate ablation embeddings first (7 variants, ~14min)
    python scripts/analyze_knowledge.py --component ablation-emb

    # Run all analyses
    python scripts/analyze_knowledge.py --component all

    # Run specific component only
    python scripts/analyze_knowledge.py --component mi
    python scripts/analyze_knowledge.py --component cold-start
    python scripts/analyze_knowledge.py --component layer-info
    python scripts/analyze_knowledge.py --component diversity
"""

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(help="Knowledge-Purchase analysis (MI, Cold-Start, Layer Info, Diversity)")
logger = logging.getLogger(__name__)


@app.command()
def main(
    data_dir: Path = typer.Option("data/processed", help="Processed data directory"),
    fk_dir: Path = typer.Option("data/knowledge/factual", help="Factual knowledge directory"),
    features_dir: Path = typer.Option("data/features", help="Features directory"),
    embeddings_dir: Path = typer.Option("data/embeddings", help="Embeddings directory"),
    output_dir: Path = typer.Option("results/analysis", help="Output directory"),
    component: str = typer.Option(
        "all",
        help="Component: all, mi, cold-start, layer-info, diversity, ablation-emb",
    ),
    mi_sample_size: int = typer.Option(10_000_000, help="MI analysis: subsample train pairs"),
    cs_sample_users: int = typer.Option(50_000, help="Cold-start: max users to evaluate"),
    div_sample_users: int = typer.Option(100_000, help="Diversity: max users to analyze"),
    bge_device: str = typer.Option("mps", help="BGE device (mps/cpu/cuda)"),
    random_seed: int = typer.Option(42, help="Random seed"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
) -> None:
    """Run Knowledge-Purchase analysis pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fk_path = fk_dir / "factual_knowledge.parquet"
    articles_path = data_dir / "articles.parquet"
    train_txn_path = data_dir / "train_transactions.parquet"
    val_gt_path = data_dir / "val_ground_truth.json"
    ablation_dir = embeddings_dir / "ablation"

    components = (
        ["mi", "cold-start", "layer-info", "diversity"]
        if component == "all"
        else [component]
    )

    # --- Ablation Embeddings ---
    if "ablation-emb" in components:
        logger.info("=" * 60)
        logger.info("Generating Ablation Embeddings (7 variants)")
        logger.info("=" * 60)
        from src.analysis.ablation_embeddings import encode_and_save_ablation_embeddings

        encode_and_save_ablation_embeddings(
            fk_path=fk_path,
            articles_path=articles_path,
            output_dir=ablation_dir,
            model_name="BAAI/bge-base-en-v1.5",
            device=bge_device,
            skip_existing=True,
        )

    # --- Component A: Mutual Information ---
    if "mi" in components:
        logger.info("=" * 60)
        logger.info("Component A: Attribute-Purchase Mutual Information")
        logger.info("=" * 60)
        from src.analysis.mutual_information import (
            compute_attribute_mi,
            compute_conditional_mi,
            mi_results_to_dataframe,
        )

        mi_results = compute_attribute_mi(
            features_dir=features_dir,
            fk_path=fk_path,
            articles_path=articles_path,
            sample_size=mi_sample_size,
            random_seed=random_seed,
        )
        mi_df = mi_results_to_dataframe(mi_results)
        mi_df.to_csv(output_dir / "mi_results.csv", index=False)
        logger.info("Saved MI results to %s", output_dir / "mi_results.csv")

        # Conditional MI
        cond_mi = {}
        cond_mi["MI(L2|L1)"] = compute_conditional_mi(
            features_dir, fk_path, articles_path, "l1", "l2",
            sample_size=mi_sample_size, random_seed=random_seed,
        )
        cond_mi["MI(L3|L1+L2)"] = compute_conditional_mi(
            features_dir, fk_path, articles_path, "l2", "l3",
            sample_size=mi_sample_size, random_seed=random_seed,
        )
        cond_mi["MI(L2|metadata)"] = compute_conditional_mi(
            features_dir, fk_path, articles_path, "metadata", "l2",
            sample_size=mi_sample_size, random_seed=random_seed,
        )
        (output_dir / "conditional_mi.json").write_text(
            json.dumps(cond_mi, indent=2)
        )
        logger.info("Conditional MI: %s", cond_mi)

    # --- Component D: Cold-Start ---
    if "cold-start" in components:
        logger.info("=" * 60)
        logger.info("Component D: Cold-Start Knowledge Utility")
        logger.info("=" * 60)
        from src.analysis.ablation_embeddings import load_ablation_embeddings
        from src.analysis.cold_start import bracket_results_to_dataframe, run_all_combos

        embeddings_by_combo = load_ablation_embeddings(ablation_dir)
        if not embeddings_by_combo:
            logger.error("No ablation embeddings found. Run --component ablation-emb first.")
            raise typer.Exit(1)

        cs_results = run_all_combos(
            embeddings_by_combo=embeddings_by_combo,
            train_txn_path=train_txn_path,
            val_gt_path=val_gt_path,
            k=12,
            sample_users=cs_sample_users,
            random_seed=random_seed,
        )
        cs_df = bracket_results_to_dataframe(cs_results)
        cs_df.to_csv(output_dir / "cold_start_results.csv", index=False)
        logger.info("Saved cold-start results to %s", output_dir / "cold_start_results.csv")

    # --- Component B: Layer Information ---
    if "layer-info" in components:
        logger.info("=" * 60)
        logger.info("Component B: Layer Incremental Information")
        logger.info("=" * 60)
        from src.analysis.ablation_embeddings import load_ablation_embeddings
        from src.analysis.layer_information import (
            cka_results_to_matrix,
            compute_cka_matrix,
            compute_purchase_coherence,
            compute_purchase_separation_auc,
        )

        ablation_data = load_ablation_embeddings(ablation_dir)
        if not ablation_data:
            logger.error("No ablation embeddings found. Run --component ablation-emb first.")
            raise typer.Exit(1)

        # CKA matrix (embeddings only, no ids needed)
        emb_only = {c: emb for c, (emb, _) in ablation_data.items()}
        cka_results = compute_cka_matrix(emb_only, sample_size=5_000, random_seed=random_seed)
        cka_mat = cka_results_to_matrix(cka_results)
        cka_mat.to_csv(output_dir / "cka_matrix.csv")
        logger.info("Saved CKA matrix to %s", output_dir / "cka_matrix.csv")

        # Purchase Coherence + Separation AUC per combo
        user_history = _load_user_history(train_txn_path)
        val_gt = _load_val_gt(val_gt_path)

        all_coherence = []
        all_auc = []
        for combo, (emb, ids) in ablation_data.items():
            coherence = compute_purchase_coherence(
                emb, ids, user_history, combo, sample_users=10_000, random_seed=random_seed,
            )
            all_coherence.extend(coherence)

            auc = compute_purchase_separation_auc(
                emb, ids, user_history, val_gt, combo,
                sample_users=10_000, random_seed=random_seed,
            )
            all_auc.append(auc._asdict())

        import pandas as pd
        pd.DataFrame([c._asdict() for c in all_coherence]).to_csv(
            output_dir / "purchase_coherence.csv", index=False,
        )
        pd.DataFrame(all_auc).to_csv(output_dir / "separation_auc.csv", index=False)
        logger.info("Saved coherence and AUC results")

    # --- Component C: Preference Diversity ---
    if "diversity" in components:
        logger.info("=" * 60)
        logger.info("Component C: Preference Diversity")
        logger.info("=" * 60)
        from src.analysis.preference_diversity import (
            compute_preference_diversity,
            diversity_results_to_dataframe,
        )

        div_results = compute_preference_diversity(
            train_txn_path=train_txn_path,
            fk_path=fk_path,
            articles_path=articles_path,
            sample_users=div_sample_users,
            random_seed=random_seed,
        )
        div_df = diversity_results_to_dataframe(div_results)
        div_df.to_csv(output_dir / "diversity_results.csv", index=False)
        logger.info("Saved diversity results to %s", output_dir / "diversity_results.csv")

    logger.info("=" * 60)
    logger.info("Analysis complete. Results in %s", output_dir)


def _load_user_history(train_txn_path: Path) -> dict[str, list[str]]:
    """Helper to load user purchase history."""
    import pandas as pd
    df = pd.read_parquet(train_txn_path, columns=["customer_id", "article_id"])
    return df.groupby("customer_id")["article_id"].apply(list).to_dict()


def _load_val_gt(val_gt_path: Path) -> dict[str, set[str]]:
    """Helper to load validation ground truth."""
    raw = json.loads(val_gt_path.read_text())
    return {uid: set(items) for uid, items in raw.items()}


if __name__ == "__main__":
    app()
