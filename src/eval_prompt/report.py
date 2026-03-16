"""Evaluation report generation, persistence, and Go/No-Go decision support.

Provides unified report serialization and threshold-based pass/fail assessment
for both factual knowledge and user profile evaluation domains.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.eval_prompt.factual import FactualEvalReport
from src.eval_prompt.reasoning import ReasoningEvalReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report Serialization
# ---------------------------------------------------------------------------


def _namedtuple_to_dict(obj: Any) -> Any:
    """Recursively convert NamedTuples to dicts for JSON serialization."""
    if obj is None:
        return None
    if hasattr(obj, "_asdict"):
        return {k: _namedtuple_to_dict(v) for k, v in obj._asdict().items()}
    if isinstance(obj, list):
        return [_namedtuple_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _namedtuple_to_dict(v) for k, v in obj.items()}
    return obj


def save_eval_report(
    report: FactualEvalReport | ReasoningEvalReport,
    output_path: Path,
) -> None:
    """Save evaluation report as JSON.

    Args:
        report: FactualEvalReport or ReasoningEvalReport.
        output_path: Path to write the JSON report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = _namedtuple_to_dict(report)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Report saved to %s", output_path)


# ---------------------------------------------------------------------------
# Go/No-Go Decision
# ---------------------------------------------------------------------------

# Default criteria for factual knowledge
FACTUAL_CRITERIA: dict[str, tuple[str, float, str]] = {
    "coverage": ("coverage.overall_coverage", 0.90, ">="),
    "schema_valid": ("schema.n_invalid", 0.05, "rate_<="),
    "domain_errors": ("domain.n_error_violations", 0.02, "rate_<="),
    "token_budget": ("token_budget.pct_over_budget", 0.05, "<="),
    "judge_overall": ("judge.overall_mean", 3.5, ">="),
    "judge_pass_rate": ("judge.pass_rate", 0.70, ">="),
}

# Default criteria for user profiles
REASONING_CRITERIA: dict[str, tuple[str, float, str]] = {
    "completeness": ("completeness.overall_completeness", 0.85, ">="),
    "generic_rate": ("completeness.n_generic", 0.10, "rate_<="),
    "discriminability": ("discriminability.mean_pairwise_sim", 0.60, "<="),
    "token_budget": ("token_budget.pct_over_budget", 0.05, "<="),
    "judge_overall": ("judge.overall_mean", 3.5, ">="),
    "judge_pass_rate": ("judge.pass_rate", 0.70, ">="),
}


def _resolve_value(report_dict: dict, path: str) -> Any:
    """Resolve a dotted path in a nested dict."""
    parts = path.split(".")
    current = report_dict
    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def build_go_no_go(
    report: FactualEvalReport | ReasoningEvalReport,
    criteria: dict[str, tuple[str, float, str]] | None = None,
) -> dict[str, tuple[bool, str]]:
    """Evaluate Go/No-Go criteria against a report.

    Args:
        report: Evaluation report.
        criteria: Dict of {criterion_name: (metric_path, threshold, operator)}.
            Operators: ">=" (value >= threshold), "<=" (value <= threshold),
            "rate_<=" (value / n_items <= threshold).

    Returns:
        Dict of {criterion_name: (passed, reason_string)}.
    """
    if criteria is None:
        if isinstance(report, FactualEvalReport):
            criteria = FACTUAL_CRITERIA
        else:
            criteria = REASONING_CRITERIA

    report_dict = _namedtuple_to_dict(report)
    results: dict[str, tuple[bool, str]] = {}

    for name, (path, threshold, op) in criteria.items():
        value = _resolve_value(report_dict, path)

        if value is None:
            results[name] = (True, f"SKIP: {path} not available (judge disabled?)")
            continue

        if op == ">=":
            passed = float(value) >= threshold
            reason = f"{path}={value:.3f} {'>=':s} {threshold:.3f}"
        elif op == "<=":
            passed = float(value) <= threshold
            reason = f"{path}={value:.3f} {'<=':s} {threshold:.3f}"
        elif op == "rate_<=":
            # Compute rate using n_items from coverage or completeness
            n_items = (
                _resolve_value(report_dict, "coverage.n_items")
                or _resolve_value(report_dict, "completeness.n_short")
                or 1
            )
            # For schema checks, n_items = n_valid + n_invalid
            if "schema" in path:
                n_total = (
                    (_resolve_value(report_dict, "schema.n_valid") or 0)
                    + (_resolve_value(report_dict, "schema.n_invalid") or 0)
                )
                n_items = n_total if n_total > 0 else 1
            rate = float(value) / max(float(n_items), 1)
            passed = rate <= threshold
            reason = f"{path}={value} (rate={rate:.3f}) <= {threshold:.3f}"
        else:
            passed = True
            reason = f"Unknown operator: {op}"

        results[name] = (passed, f"{'PASS' if passed else 'FAIL'}: {reason}")

    return results


def print_go_no_go(go_no_go: dict[str, tuple[bool, str]]) -> bool:
    """Print Go/No-Go results and return overall pass status.

    Args:
        go_no_go: Output from build_go_no_go().

    Returns:
        True if all criteria passed, False otherwise.
    """
    all_passed = True
    print("\n" + "=" * 60)
    print("GO / NO-GO ASSESSMENT")
    print("=" * 60)

    for name, (passed, reason) in go_no_go.items():
        status = "PASS" if passed else "FAIL"
        marker = "[+]" if passed else "[X]"
        print(f"  {marker} {name:.<30s} {status}: {reason}")
        if not passed and "SKIP" not in reason:
            all_passed = False

    print("-" * 60)
    overall = "GO" if all_passed else "NO-GO"
    print(f"  Overall: {overall}")
    print("=" * 60 + "\n")

    return all_passed
