"""CLI entry point for the merchandising decision-support scenarios (C-1, no API).

Productizes the 3 PASS value-matrix cells into batch merchandising briefs and writes them
to ``results/tables/merch_scenarios/`` (parquet + csv). Confidence numbers are loaded from
the canonical probe JSONs (single source of truth) and the full honest value-matrix posture
(what is / isn't productized) is printed alongside. CPU/DuckDB only — no API spend.

Usage:
    PYTHONPATH=. python scripts/serve_scenarios.py --scenario all \
        --output-dir results/tables/merch_scenarios
    PYTHONPATH=. python scripts/serve_scenarios.py --scenario trend-leadtime
"""

import logging
import sys
from pathlib import Path

import typer

from src.serving.merch_scenarios import (
    ScenarioBrief,
    ScenarioConfig,
    build_all_briefs,
    build_brief,
    value_matrix_posture,
)

app = typer.Typer(help="Build merchandising decision-support briefs (3 PASS value-matrix cells)")

# CLI scenario name (kebab) → engine scenario name (snake).
_SCENARIO_ALIASES = {
    "trend-leadtime": "trend_leadtime",
    "launch-signal": "launch_signal",
    "copurchase-velocity": "copurchase_velocity",
}


def _save_brief(brief: ScenarioBrief, output_dir: Path) -> Path:
    """Persist a brief's operational table → parquet + csv. Returns the parquet path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pq = output_dir / f"{brief.name}.parquet"
    brief.table.to_parquet(pq, index=False)
    brief.table.to_csv(output_dir / f"{brief.name}.csv", index=False)
    return pq


def _print_brief(brief: ScenarioBrief) -> None:
    c = brief.confidence
    typer.echo(f"\n=== {brief.name} :: {brief.title} ===")
    typer.echo(
        f"  confidence [{c.verdict}] {c.cell}: {c.metric}\n"
        f"    value={c.value} vs baseline={c.baseline}  CI[{c.ci_lo}, {c.ci_hi}]"
        + (f"  (lag={c.best_lag}mo)" if c.best_lag else "")
    )
    typer.echo(f"  caveat: {brief.caveat}")
    typer.echo(f"  rows={len(brief.table)}; top:")
    typer.echo("  " + brief.table.head(5).to_string(index=False).replace("\n", "\n  "))


def _print_posture(cfg: ScenarioConfig) -> None:
    df = value_matrix_posture(cfg)
    typer.echo("\n=== HONEST VALUE-MATRIX POSTURE (canonical E2b) ===")
    typer.echo(
        f"  capability {df.attrs['capability_yes']}/{df.attrs['n_cells']} cells; "
        f"behavior-validated lift PASS {df.attrs['lift_pass']}/{df.attrs['n_cells']}"
    )
    typer.echo(f"  PRODUCTIZED (this build): {df.attrs['pass_cells']}")
    not_prod = (
        df[~df["productized"]]
        .apply(lambda r: f"{r['axis']}→{r['use']}[{r['lift_verdict']}]", axis=1)
        .tolist()
    )
    typer.echo(f"  NOT productized (context only): {not_prod}")
    typer.echo(f"  {df.attrs['recsys_negative']}")


@app.command()
def main(
    scenario: str = typer.Option(
        "all",
        help="Scenario: all | trend-leadtime | launch-signal | copurchase-velocity",
    ),
    matrix_path: Path = typer.Option(
        "data/knowledge/enrichment_v2/matrix_axes.parquet", help="matrix_axes.parquet path"
    ),
    data_dir: Path = typer.Option(
        "data/processed", help="Directory with train_transactions.parquet + articles.parquet"
    ),
    output_dir: Path = typer.Option(
        "results/tables/merch_scenarios", help="Output directory for brief tables"
    ),
    top_k: int = typer.Option(
        50, help="Max rows per item-level brief (trend-leadtime always lists all categories)"
    ),
    verbose: bool = typer.Option(False, help="Verbose logging"),
) -> None:
    """Build merchandising briefs and persist them, printing the honest posture."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    cfg = ScenarioConfig(
        matrix_path=matrix_path,
        train_path=data_dir / "train_transactions.parquet",
        articles_path=data_dir / "articles.parquet",
    )

    if scenario == "all":
        briefs = build_all_briefs(cfg, item_top_k=top_k)
    else:
        name = _SCENARIO_ALIASES.get(scenario)
        if name is None:
            raise typer.BadParameter(
                f"unknown scenario {scenario!r}; choose all|{'|'.join(_SCENARIO_ALIASES)}"
            )
        # category-level lead-time has only ~10 rows; keep all. Item-level → top_k.
        k = None if name == "trend_leadtime" else top_k
        briefs = [build_brief(name, cfg, top_k=k)]

    for brief in briefs:
        path = _save_brief(brief, output_dir)
        _print_brief(brief)
        typer.echo(f"  → wrote {path} (+ .csv)")

    _print_posture(cfg)
    typer.echo(f"\nDone. {len(briefs)} brief(s) → {output_dir}")


if __name__ == "__main__":
    app()
