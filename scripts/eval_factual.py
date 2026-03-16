"""CLI entry point for factual knowledge evaluation.

Runs structural checks (coverage, schema, domain, distribution, token budget)
and optionally LLM-as-Judge (5 dimensions) on extracted factual knowledge.

Usage:
    # Structural only (no LLM cost)
    python scripts/eval_factual.py \
        --data-dir data/processed \
        --knowledge-dir data/knowledge/factual \
        --output-dir results/eval/factual \
        --skip-judge

    # Full evaluation with LLM-as-Judge
    python scripts/eval_factual.py \
        --data-dir data/processed \
        --knowledge-dir data/knowledge/factual \
        --images-dir data/h-and-m-personalized-fashion-recommendations/images \
        --output-dir results/eval/factual \
        --sample-size 50

    # Custom judge model
    python scripts/eval_factual.py \
        --data-dir data/processed \
        --knowledge-dir data/knowledge/factual \
        --output-dir results/eval/factual \
        --judge-model gpt-4.1-mini \
        --sample-size 100
"""

import asyncio
import logging
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(help="Evaluate factual knowledge extraction quality.")


@app.command()
def main(
    data_dir: Path = typer.Option(
        ..., help="Processed data directory (articles.parquet)"
    ),
    knowledge_dir: Path = typer.Option(
        ..., help="Factual knowledge directory (factual_knowledge.parquet)"
    ),
    images_dir: Path | None = typer.Option(
        None, help="Product images directory (for multimodal judge)"
    ),
    output_dir: Path = typer.Option(
        ..., help="Output directory for evaluation report"
    ),
    sample_size: int = typer.Option(
        50, help="Number of items for LLM-as-Judge"
    ),
    judge_model: str = typer.Option(
        "gpt-4.1-mini", help="LLM model for judge evaluation"
    ),
    skip_judge: bool = typer.Option(
        False, help="Skip LLM-as-Judge (structural only)"
    ),
    verbose: bool = typer.Option(
        False, help="Enable verbose logging"
    ),
) -> None:
    """Run factual knowledge evaluation pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from src.eval_prompt.factual import FactualEvalConfig, run_factual_eval
    from src.eval_prompt.judge import JudgeConfig
    from src.eval_prompt.report import build_go_no_go, print_go_no_go, save_eval_report

    # Load data
    articles_path = data_dir / "articles.parquet"
    knowledge_path = knowledge_dir / "factual_knowledge.parquet"

    if not articles_path.exists():
        typer.echo(f"Error: {articles_path} not found", err=True)
        raise typer.Exit(1)
    if not knowledge_path.exists():
        typer.echo(f"Error: {knowledge_path} not found", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading articles from {articles_path}...")
    articles_df = pd.read_parquet(articles_path)

    typer.echo(f"Loading knowledge from {knowledge_path}...")
    knowledge_df = pd.read_parquet(knowledge_path)

    typer.echo(f"Articles: {len(articles_df):,}, Knowledge: {len(knowledge_df):,}")

    # Configure evaluation
    judge_config = JudgeConfig(
        model=judge_model,
        sample_size=sample_size,
    )
    config = FactualEvalConfig(
        judge_config=judge_config,
        run_judge=not skip_judge,
    )

    # Run evaluation
    report = asyncio.run(
        run_factual_eval(
            knowledge_df=knowledge_df,
            articles_df=articles_df,
            images_dir=images_dir,
            config=config,
        )
    )

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "factual_eval_report.json"
    save_eval_report(report, report_path)

    # Go/No-Go assessment
    go_no_go = build_go_no_go(report)
    all_passed = print_go_no_go(go_no_go)

    if not all_passed:
        typer.echo("Some criteria failed. Review the report for details.")
        raise typer.Exit(1)

    typer.echo("All criteria passed!")


if __name__ == "__main__":
    app()
