"""CLI entry point for user reasoning knowledge evaluation.

Runs structural checks (coverage, completeness, discriminability, token budget)
and optionally LLM-as-Judge (5 dimensions) on generated user reasoning knowledge.

Usage:
    # Structural only (no LLM cost)
    python scripts/eval_reasoning.py \
        --data-dir data/processed \
        --rk-dir data/knowledge/reasoning \
        --knowledge-dir data/knowledge/factual \
        --output-dir results/eval/reasoning \
        --skip-judge

    # Full evaluation with LLM-as-Judge
    python scripts/eval_reasoning.py \
        --data-dir data/processed \
        --rk-dir data/knowledge/reasoning \
        --knowledge-dir data/knowledge/factual \
        --output-dir results/eval/reasoning \
        --sample-size 50

    # Custom judge model
    python scripts/eval_reasoning.py \
        --data-dir data/processed \
        --rk-dir data/knowledge/reasoning \
        --knowledge-dir data/knowledge/factual \
        --output-dir results/eval/reasoning \
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

app = typer.Typer(help="Evaluate user reasoning knowledge quality.")


@app.command()
def main(
    data_dir: Path = typer.Option(
        ..., help="Processed data directory (transactions.parquet)"
    ),
    rk_dir: Path = typer.Option(
        ..., help="Reasoning knowledge directory (user_profiles.parquet)"
    ),
    knowledge_dir: Path = typer.Option(
        ..., help="Factual knowledge directory (factual_knowledge.parquet)"
    ),
    output_dir: Path = typer.Option(
        ..., help="Output directory for evaluation report"
    ),
    sample_size: int = typer.Option(
        50, help="Number of profiles for LLM-as-Judge"
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
    """Run user reasoning knowledge evaluation pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from src.eval_prompt.judge import JudgeConfig
    from src.eval_prompt.reasoning import ReasoningEvalConfig, run_reasoning_eval
    from src.eval_prompt.report import (
        REASONING_CRITERIA,
        build_go_no_go,
        print_go_no_go,
        save_eval_report,
    )

    # Load data
    txn_path = data_dir / "transactions.parquet"
    rk_path = rk_dir / "user_profiles.parquet"
    fk_path = knowledge_dir / "factual_knowledge.parquet"

    for path, name in [(txn_path, "transactions"), (rk_path, "reasoning"), (fk_path, "knowledge")]:
        if not path.exists():
            typer.echo(f"Error: {path} not found", err=True)
            raise typer.Exit(1)

    typer.echo(f"Loading reasoning profiles from {rk_path}...")
    profiles_df = pd.read_parquet(rk_path)
    typer.echo(f"Profiles: {len(profiles_df):,}")

    # Configure evaluation
    judge_config = JudgeConfig(
        model=judge_model,
        sample_size=sample_size,
    )
    config = ReasoningEvalConfig(
        judge_config=judge_config,
        run_judge=not skip_judge,
    )

    # Run evaluation
    report = asyncio.run(
        run_reasoning_eval(
            profiles_df=profiles_df,
            txn_path=txn_path,
            fk_path=fk_path,
            config=config,
        )
    )

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "reasoning_eval_report.json"
    save_eval_report(report, report_path)

    # Go/No-Go assessment
    go_no_go = build_go_no_go(report, REASONING_CRITERIA)
    all_passed = print_go_no_go(go_no_go)

    if not all_passed:
        typer.echo("Some criteria failed. Review the report for details.")
        raise typer.Exit(1)

    typer.echo("All criteria passed!")


if __name__ == "__main__":
    app()
