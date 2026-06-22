"""Builder script for 06_merch_scenario.ipynb.

Generates the merchandising decision-support scenario notebook — the *build* (product-design)
leg of enrichment-v2 that productizes the 3 PASS value-matrix cells into batch briefs. The
notebook is a THIN presentation layer over ``src/serving/merch_scenarios.py`` (no analysis
logic is duplicated here) and is posture-honest: it leads with the 3 wins but explicitly
shows the cells it does NOT productize (① automatic faceted-control, ④ audience, gap axes)
and the separate recommendation-accuracy negative.

Parts:
0. Framing — pivot + value matrix (capability 14/16, lift 3/16); what is / isn't productized.
1. Scenario A — Trend lead-time (3-month category demand early-warning; r=0.472).
2. Scenario B — Launch first-week sell-through by trend-phase (η=0.673 vs 0.223).
3. Scenario C — Co-purchase velocity / bundle-anchor by outfit-role (η=0.631 vs 0.534).
4. Honest boundary — what we deliberately do NOT productize, and why.

Usage:
    PYTHONPATH=. python notebooks/builders/build_06_merch_scenario.py
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "06_merch_scenario.ipynb"

_cell_counter = 0


def make_cell(source: str, cell_type: str = "code") -> dict:
    """Create a notebook cell with a unique id."""
    global _cell_counter
    _cell_counter += 1
    return {
        "cell_type": cell_type,
        "id": f"cell-{_cell_counter:04d}",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def build_notebook() -> dict:
    cells = []

    # ------------------------------------------------------------------ Title
    cells.append(
        make_cell(
            "# 06 — Merchandising Decision-Support Scenarios\n\n"
            "**C 백로그 (a): 3 PASS cell 실서비스 시나리오화** — enrichment-v2 연구의 *build* leg.\n\n"
            'H&M 프로젝트는 "LLM 속성으로 추천 정확도 향상" → **"interpretable multi-purpose '
            'catalog enrichment"** 로 pivot했다. Value matrix(`witnesses/probe_E2*.py`)는 16셀 중 '
            "**행동-검증 decision-lift가 정확히 3셀**임을 정직하게 확정했고, 이 노트북은 그 3셀을 "
            "머천다이저가 쓰는 **batch 의사결정-지원 brief**로 제품화한다.\n\n"
            "| # | 축 (모두 **행동-파생**, LLM 인식축 아님) | use | canonical |\n"
            "|---|---|---|---|\n"
            "| A | `trend_phase` | lead-time | hot-share(t)→sales(t+**3mo**) **r=0.472** vs null 0.062 |\n"
            "| B | `trend_phase` | merch | first_week_sell_through **η=0.673** vs meta 0.223 |\n"
            "| C | `outfit_role` | merch | velocity **η=0.631** vs meta 0.534 |\n\n"
            "**정직성 원칙:** 운영 brief는 데이터에서 fresh 계산하되, *confidence 수치*(r·η·CI)는 "
            "canonical probe JSON에서 **로드**한다(재계산해 새 숫자 안 만듦) → value matrix가 single "
            "source of truth. 안 되는 셀(① automatic lift·④ audience·gap축)과 추천-정확도 negative는 "
            "Part 4에서 *명시적으로* 맥락화한다(숨기지 않음).",
            "markdown",
        )
    )

    # ------------------------------------------------------------------ Boilerplate
    cells.append(
        make_cell(
            "%load_ext autoreload\n"
            "%autoreload 2\n\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "# Project root (1 level up from notebooks/)\n"
            "PROJECT_ROOT = Path('.').absolute().parent\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import seaborn as sns\n\n"
            "from src.serving.merch_scenarios import (\n"
            "    ScenarioConfig, load_confidence_cards, build_all_briefs,\n"
            "    trend_leadtime_brief, launch_signal_brief, copurchase_velocity_brief,\n"
            "    value_matrix_posture,\n"
            ")\n\n"
            "sns.set_theme(style='whitegrid', context='notebook')\n"
            "FIG_DIR = PROJECT_ROOT / 'results' / 'figures'\n"
            "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
            "DPI = 150\n\n"
            "cfg = ScenarioConfig(\n"
            "    matrix_path=PROJECT_ROOT / 'data/knowledge/enrichment_v2/matrix_axes.parquet',\n"
            "    train_path=PROJECT_ROOT / 'data/processed/train_transactions.parquet',\n"
            "    articles_path=PROJECT_ROOT / 'data/processed/articles.parquet',\n"
            "    canonical_e2=PROJECT_ROOT / 'witnesses/probe_E2_result.json',\n"
            "    canonical_e2b=PROJECT_ROOT / 'witnesses/probe_E2b_result.json',\n"
            ")\n"
            "cards = load_confidence_cards(cfg)\n"
            "cards"
        )
    )

    # ================================================================== Part 0
    cells.append(
        make_cell(
            "## Part 0 — Framing: 정직한 value matrix\n\n"
            "**capability(metadata 없는 결정 축)는 14/16로 dense하지만, 행동-검증된 decision-lift는 "
            "3/16으로 sparse하다.** 이것이 연구의 thesis: *LLM/enrichment 속성의 가치는 예측력이 "
            "아니라 metadata가 없는 해석 가능한 결정-축*. 아래 heatmap은 16셀 전체를 보여주고, "
            "제품화하는 3 PASS만 초록 테두리로 강조한다. capability YES인데 lift는 NO/MARGINAL인 "
            "셀들이 thesis의 핵심 증거다.",
            "markdown",
        )
    )
    cells.append(
        make_cell(
            "posture = value_matrix_posture(cfg)\n"
            "print(f\"capability {posture.attrs['capability_yes']}/{posture.attrs['n_cells']}  |  \"\n"
            "      f\"behavior-validated lift PASS {posture.attrs['lift_pass']}/{posture.attrs['n_cells']}\")\n"
            "print('PRODUCTIZED:', posture.attrs['pass_cells'])\n"
            "display(posture)\n\n"
            "# --- value-matrix heatmap (verdict-colored, PASS highlighted) ---\n"
            "USES = ['faceted', 'leadlag', 'merch', 'audience']\n"
            "AXES = ['e2_trend_phase_actual', 'e2_outfit_role', 'e2_value_gap', 'e2_trend_gap']\n"
            "VCODE = {'PASS': 3, 'MARGINAL': 2, 'NO': 1}\n"
            "p = posture.copy()\n"
            "p['axis_norm'] = p['axis'].str.replace('_sign', '', regex=False)\n"
            "p['code'] = p['lift_verdict'].map(VCODE).fillna(0)  # N/A-* -> 0\n"
            "grid = p.pivot_table(index='axis_norm', values='code', columns='use', aggfunc='first').reindex(index=AXES, columns=USES)\n"
            "labels = p.pivot_table(index='axis_norm', values='lift_verdict', columns='use', aggfunc='first').reindex(index=AXES, columns=USES)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 5.0))\n"
            "cmap = sns.color_palette(['#bdbdbd', '#e06666', '#ffd966', '#93c47d'])\n"
            "sns.heatmap(grid, annot=labels.fillna('N/A').values, fmt='', cmap=cmap, vmin=0, vmax=3,\n"
            "            cbar=False, linewidths=1.2, linecolor='white', ax=ax,\n"
            "            annot_kws={'fontsize': 9})\n"
            "# green border on the 3 productized PASS cells\n"
            "for _, r in p[p['productized']].iterrows():\n"
            "    yi, xi = AXES.index(r['axis_norm']), USES.index(r['use'])\n"
            "    ax.add_patch(plt.Rectangle((xi, yi), 1, 1, fill=False, edgecolor='#1b5e20', lw=3))\n"
            "ax.set_title('Enrichment-v2 value matrix — capability 14/16, lift PASS 3/16\\n(green = productized in this build)')\n"
            "ax.set_xlabel('use case'); ax.set_ylabel('decision axis')\n"
            "plt.tight_layout(); plt.savefig(FIG_DIR / '06_value_matrix_posture.png', dpi=DPI, bbox_inches='tight')\n"
            "plt.show()"
        )
    )

    # ================================================================== Part 1
    cells.append(
        make_cell(
            "## Part 1 — Scenario A · Trend lead-time (3-month 카테고리 수요 조기경보)\n\n"
            "**페르소나: category buyer.** 한 카테고리의 *hot(Emerging+Rising) 아이템 share* 가 그 "
            "카테고리 매출을 ~3개월 **선행**한다(canonical r=0.472, permutation null 0.062, lag=3mo). "
            "운영 brief는 현재 hot-share가 자기 이력 대비 얼마나 *상승*했는지(z-score)로 카테고리를 "
            "랭킹 → '3개월 뒤 수요 상승' 조기경보.\n\n"
            "⚠️ *modest* 상관(22개월·10 카테고리, CI 넓음). 카테고리 수준 신호이지 per-item 예측이 "
            "아니다. lag-3가 lag-1/2/4보다 corr이 높다는 점이 '선행' 구조의 증거.",
            "markdown",
        )
    )
    cells.append(
        make_cell(
            "briefA = trend_leadtime_brief(cfg)\n"
            "print(briefA.title); print('confidence:', briefA.confidence); print(briefA.caveat)\n"
            "display(briefA.table)\n\n"
            "by_lag = briefA.extra['lead_lag_by_lag']; best = briefA.extra['best_lag']\n"
            "share = briefA.extra['share_timeseries']\n"
            "null_r = briefA.confidence.baseline\n"
            "fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))\n"
            "# (a) lead-lag corr by horizon vs permutation null\n"
            "lags = sorted(by_lag); vals = [by_lag[k] for k in lags]\n"
            "bars = axes[0].bar([str(k) for k in lags], vals, color=['#93c47d' if k==best else '#9fc5e8' for k in lags])\n"
            "axes[0].axhline(null_r, ls='--', color='#e06666', label=f'permutation null r={null_r}')\n"
            "axes[0].set_title(f'Lead-lag corr: hot-share(t) → sales(t+k)\\nbest lag={best}mo, r={briefA.confidence.value}')\n"
            "axes[0].set_xlabel('lead horizon k (months)'); axes[0].set_ylabel('mean corr across categories'); axes[0].legend()\n"
            "# (b) early-warning ranking by current hot-share z-score\n"
            "t = briefA.table.sort_values('share_zscore')\n"
            "colors = {'high': '#93c47d', 'elevated': '#ffd966', 'watch': '#bdbdbd'}\n"
            "axes[1].barh(t['category'], t['share_zscore'], color=[colors.get(w, '#bdbdbd') for w in t['early_warning']])\n"
            "axes[1].axvline(1.0, ls=':', color='gray'); axes[1].set_xlabel('current hot-share z-score (vs own history)')\n"
            "axes[1].set_title('3-month demand early-warning (by category)')\n"
            "plt.tight_layout(); plt.savefig(FIG_DIR / '06_trend_leadtime.png', dpi=DPI, bbox_inches='tight'); plt.show()"
        )
    )

    # ================================================================== Part 2
    cells.append(
        make_cell(
            "## Part 2 — Scenario B · Launch first-week sell-through (트렌드-페이즈 런칭 신호)\n\n"
            "**페르소나: launch planner.** trend_phase가 신규 아이템의 **first-week sell-through**를 "
            "설명한다(η=0.673, product_group residualize, metadata 0.223). E2-2의 velocity(=sales-rate, "
            "momentum과 tautology)를 first-week sell-through로 교체해 정당화된 셀. 운영 brief는 hot "
            "cohort(Emerging+Rising, 최소구매 floor)를 예상 sell-through 티어로 스코어링 → 재고배분·우선순위.",
            "markdown",
        )
    )
    cells.append(
        make_cell(
            "briefB = launch_signal_brief(cfg, top_k=25)\n"
            "print(briefB.title); print('confidence:', briefB.confidence); print(briefB.caveat)\n"
            "ps = briefB.extra['phase_sellthrough']\n"
            "display(ps)\n"
            "display(briefB.table.head(15))\n\n"
            "c = briefB.confidence\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.6))\n"
            "ps2 = ps.sort_values('mean_first_week_sell_through')\n"
            "tier_color = {'High': '#93c47d', 'Medium': '#ffd966', 'Low': '#e06666'}\n"
            "ax.barh(ps2['e2_trend_phase_actual'], ps2['mean_first_week_sell_through'],\n"
            "        color=[tier_color.get(t, '#bdbdbd') for t in ps2['sell_through_tier']])\n"
            "ax.set_xlabel('mean first-week sell-through'); ax.set_ylabel('trend phase')\n"
            "ax.set_title(f'Launch signal: first-week sell-through by trend-phase\\nη={c.value} (resid product_group) vs metadata {c.baseline}')\n"
            "plt.tight_layout(); plt.savefig(FIG_DIR / '06_launch_signal.png', dpi=DPI, bbox_inches='tight'); plt.show()"
        )
    )

    # ================================================================== Part 3
    cells.append(
        make_cell(
            "## Part 3 — Scenario C · Co-purchase velocity / bundle-anchor (아웃핏-역할)\n\n"
            "**페르소나: cross-sell / bundling merchandiser.** same-basket cross-group co-purchase로 "
            "도출한 **outfit_role**이 판매 velocity를 설명한다(η=0.631, product_group residualize, "
            "metadata 0.534). 역할별 velocity 티어 + 머천다이징 번들 라벨(Anchor-hub=cross-sell 앵커 등). "
            "운영 brief는 anchor 역할(Anchor-hub·Versatile-connector) 아이템을 velocity로 랭킹 → 번들 "
            "앵커 재고 우선순위.",
            "markdown",
        )
    )
    cells.append(
        make_cell(
            "briefC = copurchase_velocity_brief(cfg, top_k=25)\n"
            "print(briefC.title); print('confidence:', briefC.confidence); print(briefC.caveat)\n"
            "rv = briefC.extra['role_velocity']\n"
            "display(rv[['e2_outfit_role', 'n_items', 'mean_velocity', 'median_velocity', 'velocity_tier', 'bundle_role']])\n"
            "display(briefC.table.head(15))\n\n"
            "c = briefC.confidence\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.6))\n"
            "rv2 = rv.sort_values('mean_velocity')\n"
            "tier_color = {'High': '#93c47d', 'Medium': '#ffd966', 'Low': '#e06666'}\n"
            "ax.barh(rv2['e2_outfit_role'], rv2['mean_velocity'],\n"
            "        color=[tier_color.get(t, '#bdbdbd') for t in rv2['velocity_tier']])\n"
            "ax.set_xlabel('mean velocity (purchases / lifespan day)'); ax.set_ylabel('outfit role')\n"
            "ax.set_title(f'Co-purchase velocity by outfit-role\\nη={c.value} (resid product_group) vs metadata {c.baseline}')\n"
            "plt.tight_layout(); plt.savefig(FIG_DIR / '06_copurchase_velocity.png', dpi=DPI, bbox_inches='tight'); plt.show()"
        )
    )

    # ================================================================== Part 4
    cells.append(
        make_cell(
            "## Part 4 — 정직한 경계: 제품화하지 *않는* 것\n\n"
            "honesty-first posture의 핵심. 3 PASS만 제품화하고, 아래는 **의도적으로 제외**하되 "
            "맥락화한다:\n\n"
            "- **① faceted-control (automatic lift) = NO**: D3에서 *oracle* steered precision 1.00 vs "
            "0.14(capability YES)지만, *배포 가능한* history-predictor의 automatic gain은 0.0. → "
            "human-in-the-loop 필터 표면으로는 capability가 있으나 자동 개인화 lift는 없음. *제품화 X.*\n"
            "- **④ audience = NO**: 축이 buyer-demographic divergence에서 metadata k-means를 못 넘음 "
            "(축은 category-직교). 마케팅 audience 세그먼터로 부적합.\n"
            "- **gap축(`value_gap`·`trend_gap`) = NO**: capability는 비중복이나 행동적으로 inert "
            "(decision-lift 없음). 별도 research probe(백로그 b)로 hidden-value 해석 축 가치를 재검증 예정.\n"
            "- **추천 정확도 = 별도 negative**: 이 매트릭스 밖. LLM L2/L3/외부지식이 L1 대비 정확도 "
            "개선 못함(full-scale −12%). 제품화 X.\n\n"
            "**요약: capability vs lift는 다르다.** 제품화는 lift가 행동적으로 검증된 셀에만 한다.",
            "markdown",
        )
    )
    cells.append(
        make_cell(
            "not_prod = posture[~posture['productized']].copy()\n"
            "not_prod['cell'] = not_prod['axis'] + '→' + not_prod['use']\n"
            "display(not_prod[['cell', 'capability', 'lift_verdict', 'lift_value']]\n"
            "        .sort_values('lift_verdict').reset_index(drop=True))\n"
            "print('\\n추천-정확도 negative (value matrix 밖, 제품화 X):')\n"
            "print(posture.attrs['recsys_negative'])\n\n"
            "# persist all three operational briefs for downstream use\n"
            "OUT = PROJECT_ROOT / 'results' / 'tables' / 'merch_scenarios'\n"
            "OUT.mkdir(parents=True, exist_ok=True)\n"
            "for b in [briefA, briefB, briefC]:\n"
            "    b.table.to_parquet(OUT / f'{b.name}.parquet', index=False)\n"
            "    b.table.to_csv(OUT / f'{b.name}.csv', index=False)\n"
            "print('briefs →', OUT)"
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (llm-factor-recsys-hnm)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.14",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build_notebook()
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook written to {NOTEBOOK_PATH}")
    print(f"  {len(nb['cells'])} cells")
