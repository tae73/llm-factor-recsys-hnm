"""Prompt Output Evaluation Framework.

Dual-layer evaluation for LLM-generated factual knowledge and user profiles:
- Structural (Programmatic): deterministic schema, coverage, token budget checks
- LLM-as-Judge: semantic 5-dimension scoring (accuracy, specificity, coherence,
  source_alignment, informativeness)
"""

from src.eval_prompt.judge import (
    DIMENSION_NAMES,
    JudgeConfig,
    JudgeDimension,
    JudgeReport,
    JudgeResult,
)
from src.eval_prompt.report import build_go_no_go, print_go_no_go, save_eval_report
from src.eval_prompt.structural import (
    CoverageResult,
    TokenBudgetResult,
    check_token_budget,
    compute_coverage,
)

__all__ = [
    "DIMENSION_NAMES",
    "JudgeConfig",
    "JudgeDimension",
    "JudgeReport",
    "JudgeResult",
    "CoverageResult",
    "TokenBudgetResult",
    "build_go_no_go",
    "check_token_budget",
    "compute_coverage",
    "print_go_no_go",
    "save_eval_report",
]
