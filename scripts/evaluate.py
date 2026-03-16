"""CLI entry point for recommendation evaluation.

Computes MAP@12, HR@12, NDCG@12, MRR from prediction and ground truth JSON files.

Usage:
    python scripts/evaluate.py \
        --predictions-path results/predictions/userknn_val.json \
        --ground-truth-path data/processed/val_ground_truth.json \
        --output-path results/metrics/userknn_val.json
"""

import json
from pathlib import Path

import typer

from src.config import EvalConfig
from src.evaluation.metrics import evaluate

app = typer.Typer(help="Evaluate recommendation predictions")


@app.command()
def main(
    predictions_path: Path = typer.Option(..., help="Path to predictions JSON"),
    ground_truth_path: Path = typer.Option(..., help="Path to ground truth JSON"),
    output_path: Path = typer.Option(..., help="Path to save metrics JSON"),
    k: int = typer.Option(12, help="Cutoff K for metrics"),
) -> None:
    """Compute evaluation metrics and save results."""
    predictions = json.loads(predictions_path.read_text())
    ground_truth = json.loads(ground_truth_path.read_text())

    print(f"[evaluate] Predictions: {len(predictions):,} users")
    print(f"[evaluate] Ground truth: {len(ground_truth):,} users")

    config = EvalConfig(k=k)
    result = evaluate(predictions, ground_truth, config)

    metrics = {
        f"map_at_{k}": round(result.map_at_k, 6),
        f"hr_at_{k}": round(result.hr_at_k, 6),
        f"ndcg_at_{k}": round(result.ndcg_at_k, 6),
        "mrr": round(result.mrr, 6),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n=== Metrics (k={k}) ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    app()
