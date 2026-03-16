"""Builder script for 01_factual_eval.ipynb.

Generates a notebook that loads a pre-generated JSON report
(from scripts/eval_factual.py) and visualizes/interprets the results.

4 sections:
1. Setup & Prerequisites
2. Structural Analysis (coverage, schema, domain, distributions, token budget)
3. LLM-as-Judge Analysis (per-dimension scores, heatmap, low-scorers, boxplot)
4. Go/No-Go Summary

Usage:
    conda run -n llm-factor-recsys-hnm python notebooks/builders/build_01_factual_eval.py
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "01_factual_eval.ipynb"


def make_cell(source: str, cell_type: str = "code") -> dict:
    """Create a notebook cell."""
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def build_notebook() -> dict:
    """Build the complete notebook."""
    cells = []

    # ===================================================================
    # Section 1: Setup & Prerequisites
    # ===================================================================
    cells.append(make_cell(
        "# 01 — Factual Knowledge Evaluation\n\n"
        "Loads a pre-generated evaluation report (`factual_eval_report.json`) "
        "produced by `scripts/eval_factual.py` and visualizes the results.\n\n"
        "**Sections:**\n"
        "1. Setup & Prerequisites\n"
        "2. Structural Analysis (coverage, schema, domain, distributions, token budget)\n"
        "3. LLM-as-Judge Analysis (5 dimensions)\n"
        "4. Go/No-Go Summary",
        "markdown",
    ))

    cells.append(make_cell("""\
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

PROJECT_ROOT = Path('.').absolute().parent
sys.path.insert(0, str(PROJECT_ROOT))"""))

    cells.append(make_cell(
        "### Prerequisites\n\n"
        "Run these scripts **before** this notebook:\n\n"
        "```bash\n"
        "# 1. Preprocess (skip if already done)\n"
        "python scripts/preprocess.py \\\n"
        "    --raw-dir data/h-and-m-personalized-fashion-recommendations \\\n"
        "    --output-dir data/processed\n\n"
        "# 2. Extract factual knowledge (skip if already done)\n"
        "python scripts/extract_factual_knowledge.py \\\n"
        "    --data-dir data/processed \\\n"
        "    --images-dir data/h-and-m-personalized-fashion-recommendations/images \\\n"
        "    --output-dir data/knowledge/factual \\\n"
        "    --batch-api\n\n"
        "# 3. Run evaluation (generates this notebook's input)\n"
        "python scripts/eval_factual.py \\\n"
        "    --data-dir data/processed \\\n"
        "    --knowledge-dir data/knowledge/factual \\\n"
        "    --output-dir results/eval/factual \\\n"
        "    --skip-judge\n\n"
        "# Or with LLM-as-Judge:\n"
        "python scripts/eval_factual.py \\\n"
        "    --data-dir data/processed \\\n"
        "    --knowledge-dir data/knowledge/factual \\\n"
        "    --images-dir data/h-and-m-personalized-fashion-recommendations/images \\\n"
        "    --output-dir results/eval/factual \\\n"
        "    --sample-size 50\n"
        "```",
        "markdown",
    ))

    cells.append(make_cell("""\
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

# Load report
REPORT_PATH = PROJECT_ROOT / "results" / "eval" / "factual" / "factual_eval_report.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eval" / "factual"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH) as f:
    report = json.load(f)

print(f"Report timestamp: {report.get('timestamp', 'N/A')}")
print(f"Sections: {list(report.keys())}")"""))

    # ===================================================================
    # Section 2: Structural Analysis
    # ===================================================================
    cells.append(make_cell("## 2. Structural Analysis", "markdown"))

    # --- 2.1 Coverage ---
    cells.append(make_cell("### 2.1 Coverage", "markdown"))
    cells.append(make_cell("""\
cov = report["coverage"]
print(f"Overall coverage: {cov['overall_coverage']:.1%}")
print(f"Items evaluated: {cov['n_items']:,}")

fig, ax = plt.subplots(figsize=(10, 5))
fields = list(cov["field_coverage"].keys())
values = list(cov["field_coverage"].values())
colors = ["#2ecc71" if v >= 0.9 else "#e74c3c" for v in values]
ax.barh(fields, values, color=colors)
ax.axvline(0.9, color="red", linestyle="--", alpha=0.5, label="90% threshold")
ax.set_xlabel("Coverage")
ax.set_title("Factual Knowledge: Field Coverage")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_coverage.png", dpi=150, bbox_inches="tight")
plt.show()"""))

    cells.append(make_cell(
        "**Interpretation:** Coverage measures the fraction of items with non-null values "
        "per field. Fields below 90% may indicate extraction failures or missing source "
        "data. Category-specific fields (e.g., `l1_slot4`–`l1_slot7`) are expected to "
        "have lower coverage as they only apply to certain product categories.",
        "markdown",
    ))

    # --- 2.2 Schema Validation ---
    cells.append(make_cell("### 2.2 Schema Validation", "markdown"))
    cells.append(make_cell("""\
schema = report["schema"]
n_total = schema["n_valid"] + schema["n_invalid"]
valid_rate = schema["n_valid"] / max(n_total, 1)

print(f"Valid: {schema['n_valid']:,} / {n_total:,} ({valid_rate:.1%})")
print(f"Invalid: {schema['n_invalid']:,}")

if schema["error_counts"]:
    print("\\nTop errors:")
    errors_sorted = sorted(schema["error_counts"].items(), key=lambda x: -x[1])
    for err, cnt in errors_sorted[:10]:
        print(f"  {cnt:>5d}  {err}")

if schema["warning_counts"]:
    print("\\nTop warnings:")
    warnings_sorted = sorted(schema["warning_counts"].items(), key=lambda x: -x[1])
    for warn, cnt in warnings_sorted[:10]:
        print(f"  {cnt:>5d}  {warn}")"""))

    cells.append(make_cell(
        "**Interpretation:** Schema validation checks type correctness, enum membership, "
        "and required field presence. A valid rate > 99% indicates the extraction prompt "
        "and structured output schema are well-calibrated. Common errors typically involve "
        "out-of-enum values for edge-case products.",
        "markdown",
    ))

    # --- 2.3 Domain Consistency ---
    cells.append(make_cell("### 2.3 Domain Consistency", "markdown"))
    cells.append(make_cell("""\
domain = report["domain"]
print(f"Items with violations: {domain['n_items_with_violations']:,}")
print(f"Error violations: {domain['n_error_violations']:,}")
print(f"Warning violations: {domain['n_warning_violations']:,}")

if domain["rule_counts"]:
    # Rule violation table
    rule_df = (
        pd.DataFrame(
            [{"rule": r, "count": c} for r, c in domain["rule_counts"].items()]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    display(rule_df)

    # Severity distribution pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [domain["n_error_violations"], domain["n_warning_violations"]]
    labels = [f"Error ({sizes[0]:,})", f"Warning ({sizes[1]:,})"]
    colors_pie = ["#e74c3c", "#f39c12"]
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%", startangle=90)
    ax.set_title("Domain Violations: Severity Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_domain_severity.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No domain violations detected.")"""))

    cells.append(make_cell(
        "**Interpretation:** Domain consistency checks 12 cross-attribute rules "
        "(e.g., `silhouette_x_visual_weight`, `coordination_x_harmony`). "
        "Error-level violations indicate genuine contradictions; warning-level ones "
        "indicate fashion-ambiguous cases (e.g., Basic coordination with Analogous harmony). "
        "After `correct_visual_weight()` post-processing, the error rate should be < 2%.",
        "markdown",
    ))

    # --- 2.4 Enum Distributions ---
    cells.append(make_cell("### 2.4 Enum Distributions", "markdown"))
    cells.append(make_cell("""\
dist = report["distributions"]

# Entropy summary table
entropy_df = (
    pd.DataFrame([
        {"field": f, "entropy": dist["entropy"][f], "n_unique": dist["n_unique"].get(f, 0)}
        for f in sorted(dist["entropy"], key=lambda x: -dist["entropy"][x])
    ])
)
display(entropy_df)"""))

    cells.append(make_cell("""\
# Top-6 fields by entropy — value distribution bar plots
top_fields = sorted(dist["entropy"].items(), key=lambda x: -x[1])[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (field, ent) in zip(axes.flat, top_fields):
    counts = dist["value_counts"].get(field, {})
    if counts:
        top_vals = sorted(counts.items(), key=lambda x: -x[1])[:10]
        labels, vals = zip(*top_vals)
        ax.barh(list(labels), list(vals))
        ax.set_title(f"{field}\\n(H={ent:.2f})")
for ax in axes.flat[len(top_fields):]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_distributions.png", dpi=150, bbox_inches="tight")
plt.show()"""))

    cells.append(make_cell(
        "**Interpretation:** Shannon entropy measures value diversity. "
        "Very low entropy (< 0.5) suggests a collapsed distribution where most items "
        "share the same value, reducing the attribute's discriminative power for "
        "recommendation. Very high entropy with many unique values (e.g., `style_lineage`) "
        "is expected for rich semantic fields and is fine for BGE embedding.",
        "markdown",
    ))

    # --- 2.5 Token Budget ---
    cells.append(make_cell("### 2.5 Token Budget", "markdown"))
    cells.append(make_cell("""\
tb = report["token_budget"]
token_df = pd.DataFrame([{
    "Metric": "Mean tokens",
    "Value": f"{tb['mean_tokens']:.0f}",
}, {
    "Metric": "Median tokens",
    "Value": f"{tb['median_tokens']:.0f}",
}, {
    "Metric": "P95 tokens",
    "Value": f"{tb['p95_tokens']:.0f}",
}, {
    "Metric": "P99 tokens",
    "Value": f"{tb['p99_tokens']:.0f}",
}, {
    "Metric": "Max tokens",
    "Value": str(tb["max_tokens"]),
}, {
    "Metric": "Over budget",
    "Value": f"{tb['n_over_budget']:,} ({tb['pct_over_budget']:.1%})",
}, {
    "Metric": "Budget limit",
    "Value": str(tb["budget_limit"]),
}])
display(token_df)"""))

    cells.append(make_cell(
        "**Interpretation:** Token budget checks that composed factual text fits within "
        "the BGE-base encoder's 512-token context window. P95 should be well under 512; "
        "any items over budget need text truncation or field pruning.",
        "markdown",
    ))

    # ===================================================================
    # Section 3: LLM-as-Judge Analysis
    # ===================================================================
    cells.append(make_cell(
        "## 3. LLM-as-Judge Analysis\n\n"
        "This section visualizes LLM-as-Judge results if they were included in the report "
        "(i.e., `scripts/eval_factual.py` was run **without** `--skip-judge`).",
        "markdown",
    ))

    cells.append(make_cell("""\
judge = report.get("judge")
if judge is None:
    print("LLM-as-Judge was not included in this report.")
    print("Re-run: python scripts/eval_factual.py ... (without --skip-judge)")"""))

    # Per-dimension mean barh
    cells.append(make_cell("""\
if judge is not None:
    print(f"Evaluated: {judge['n_evaluated']}")
    print(f"Overall mean: {judge['overall_mean']:.2f}")
    print(f"Pass rate: {judge['pass_rate']:.1%}")

    # Per-dimension mean bar chart
    dim_means = judge["per_dimension_mean"]
    fig, ax = plt.subplots(figsize=(10, 5))
    dims = list(dim_means.keys())
    means = list(dim_means.values())
    colors_bar = ["#2ecc71" if m >= 3.5 else "#e74c3c" for m in means]
    ax.barh(dims, means, color=colors_bar)
    ax.axvline(3.5, color="red", linestyle="--", alpha=0.5, label="Pass threshold (3.5)")
    ax.set_xlabel("Mean Score (1-5)")
    ax.set_title("LLM-as-Judge: Per-Dimension Mean Scores")
    ax.set_xlim(0, 5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_judge_scores.png", dpi=150, bbox_inches="tight")
    plt.show()"""))

    # Per-item score heatmap
    cells.append(make_cell("""\
if judge is not None:
    results = judge["results"]
    dim_names = list(judge["per_dimension_mean"].keys())

    # Build score matrix (items x dimensions)
    item_ids = [r["item_id"] for r in results]
    score_matrix = np.array([[r["scores"][d] for d in dim_names] for r in results])

    fig, ax = plt.subplots(figsize=(10, max(6, len(item_ids) * 0.3)))
    sns.heatmap(
        score_matrix,
        xticklabels=dim_names,
        yticklabels=item_ids,
        annot=True,
        fmt="d",
        cmap="RdYlGn",
        vmin=1,
        vmax=5,
        ax=ax,
    )
    ax.set_title("Per-Item Score Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_judge_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()"""))

    # Low-scoring items table
    cells.append(make_cell("""\
if judge is not None:
    # Items with any dimension score <= 2
    low_scorers = []
    for r in results:
        low_dims = [d for d, s in r["scores"].items() if s <= 2]
        if low_dims:
            for d in low_dims:
                low_scorers.append({
                    "item_id": r["item_id"],
                    "dimension": d,
                    "score": r["scores"][d],
                    "justification": r["justifications"].get(d, ""),
                })

    if low_scorers:
        low_df = pd.DataFrame(low_scorers).sort_values("score")
        print(f"Low-scoring items (score <= 2): {len(low_df)}")
        display(low_df)
    else:
        print("No items scored <= 2 on any dimension.")"""))

    # Dimension score distribution boxplot
    cells.append(make_cell("""\
if judge is not None:
    score_data = []
    for r in results:
        for dim in dim_names:
            score_data.append({"dimension": dim, "score": r["scores"][dim]})
    score_df = pd.DataFrame(score_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=score_df, x="dimension", y="score", ax=ax, palette="Set2")
    ax.axhline(3.5, color="red", linestyle="--", alpha=0.5, label="Pass threshold")
    ax.set_ylabel("Score (1-5)")
    ax.set_title("LLM-as-Judge: Score Distribution per Dimension")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_judge_boxplot.png", dpi=150, bbox_inches="tight")
    plt.show()"""))

    cells.append(make_cell(
        "**Interpretation:** The LLM-as-Judge evaluates 5 dimensions on a 1-5 scale:\n"
        "- **Accuracy**: Are L1/L2/L3 attributes factually correct?\n"
        "- **Specificity**: Are attributes specific to this item, not generic?\n"
        "- **Coherence**: Do attributes form a non-contradictory picture?\n"
        "- **Source Alignment**: Do attributes match the image/metadata?\n"
        "- **Informativeness**: Are attributes rich enough for recommendation?\n\n"
        "Pass threshold is 3.5 per dimension. Items scoring <= 2 need manual review.",
        "markdown",
    ))

    # ===================================================================
    # Section 4: Go/No-Go Summary
    # ===================================================================
    cells.append(make_cell("## 4. Go/No-Go Summary", "markdown"))

    cells.append(make_cell("""\
from src.eval_prompt.report import FACTUAL_CRITERIA

# Build Go/No-Go table from report data
def resolve_value(data, path):
    \"\"\"Resolve a dotted path in a nested dict.\"\"\"
    for part in path.split("."):
        if data is None or not isinstance(data, dict):
            return None
        data = data.get(part)
    return data

go_no_go_rows = []
all_passed = True

for name, (path, threshold, op) in FACTUAL_CRITERIA.items():
    value = resolve_value(report, path)

    if value is None:
        go_no_go_rows.append({
            "Criterion": name,
            "Status": "SKIP",
            "Value": "N/A",
            "Threshold": f"{threshold}",
            "Detail": f"{path} not available (judge disabled?)",
        })
        continue

    if op == ">=":
        passed = float(value) >= threshold
        detail = f"{path}={value:.3f} >= {threshold:.3f}"
    elif op == "<=":
        passed = float(value) <= threshold
        detail = f"{path}={value:.3f} <= {threshold:.3f}"
    elif op == "rate_<=":
        n_items = resolve_value(report, "coverage.n_items") or 1
        if "schema" in path:
            n_total = (resolve_value(report, "schema.n_valid") or 0) + (
                resolve_value(report, "schema.n_invalid") or 0
            )
            n_items = n_total if n_total > 0 else 1
        rate = float(value) / max(float(n_items), 1)
        passed = rate <= threshold
        detail = f"{path}={value} (rate={rate:.3f}) <= {threshold:.3f}"
    else:
        passed = True
        detail = f"Unknown operator: {op}"

    status = "PASS" if passed else "FAIL"
    if not passed and "SKIP" not in detail:
        all_passed = False

    go_no_go_rows.append({
        "Criterion": name,
        "Status": status,
        "Value": f"{value}" if isinstance(value, (int, str)) else f"{value:.4f}",
        "Threshold": f"{threshold}",
        "Detail": detail,
    })

go_df = pd.DataFrame(go_no_go_rows)
display(go_df)

overall = "GO" if all_passed else "NO-GO"
print(f"\\nOverall: {overall}")"""))

    # Build notebook structure
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def main() -> None:
    nb = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
