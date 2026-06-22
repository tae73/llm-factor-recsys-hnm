# H&M LLM-Factor RecSys

> 3-Layer Attribute Taxonomy + KAR Hybrid-Expert Adaptor for H&M Fashion Recommendation
> — a **falsification-first** study that disproved its own hypothesis, diagnosed why, and pivoted to where the value actually is.

---

## TL;DR — 정직한 결론

원래 가설 **"LLM이 추출한 다층 속성(L1 제품 / L2 체감 / L3 이론)이 추천 *정확도*를 높인다"** 는 13+ probe + 적대검증으로 **반증**됐다 (L2/L3는 L1 대비 정확도 증분 ≈ 0, multi-task robust). 그리고 여기서 멈추지 않고 **원인 진단 → pivot → 가치 재정의**까지 갔다:

- **추천 가치지도는 *전 축* negative** — 정확도(full-scale −12%), diversity/coverage(−7.7%/−2.3%), serendipity/novelty/long-tail(tie at best)까지 enrichment는 L1 content를 넘지 못한다.
- **그러나 가치는 분명하다 — 예측이 아니라 *해석 가능한 결정-축*에 있다:** ① **merchant 의사결정-지원**(행동-검증 3 cell 제품화), ② **interpretable decision-axes**(metadata·L1과 직교, capability 14/16), ③ **human-in-the-loop 제어**(steered precision 1.00 vs 0.14).
- **셀링포인트는 단일 정확도 수치가 아니라 *연구 성숙도*** — falsification rigor + mechanism diagnosis + de-risk→scale 자기반증 + honest negative. 모든 수치는 `witnesses/probe_*_result.json`(고정 seed·bootstrap CI)에서 그대로 인용.

> *capability vs prediction 분리*가 이 연구의 핵심 기여다. (단일 데이터셋·negative 결과·비-신규 방법 → 출판 논문은 아니며, 강점은 과정의 엄정성·정직성.)

---

## Problem: Triple-Sparsity in Fashion Recommendation

H&M 데이터셋(~105K items, ~1.37M users, ~31M transactions)은 세 차원에서 동시에 희소성이 발현되어 협업 필터링(CF)만으로는 유의미한 개인화가 구조적으로 불가능하다.

| Sparsity Dimension | Metric | Impact |
|--------------------|--------|--------|
| User-side | 32.1% (436K) users have only 1-4 purchases | <0.004% catalog interaction per user |
| Matrix-side | 99.98% sparse interaction matrix | MF/GNN signal propagation fails |
| Signal quality | Popularity > UserKNN > BPR-MF | More personalization = worse performance |

87% of user-item pairs are single-purchase → repeat-purchase prediction is structurally limited; **discovery-oriented recommendation** is essential. 이 환경에서 CF를 넘는 보완 신호로 **LLM 추출 다층 속성**을 시도한 것이 출발점이다.

---

## The Research Arc: Falsification → Diagnosis → Pivot

### 1. The Bet — 원래 가설
- **3-Layer Attribute Taxonomy** — L1(제품: material/fit/neckline) + L2(체감: mood/occasion/quality) + L3(이론: color_harmony/silhouette/lineage).
- **KAR Hybrid-Expert** — factual(L1+L2+L3) + reasoning 2-Expert.
- **검증 가능한 주장:** "L2/L3 추상 속성이 L1·metadata 대비 추천 정확도·다양성·cold-start를 *증분* 향상한다."

### 2. The Falsification — 가설을 스스로 반증
싼 probe로 핵심 가정을 make-or-break (falsification-first). META→L1은 강함(+130.9%)이나 **L1→L2 −0.8%(CI 0 포함)·L2→L3 −4.6%**; re-ranker·encoding·gating·complementarity 4종 fairness fix를 줘도 살아나지 않음. 적대검증(5-skeptic)으로 자기 보고("L2 생존")까지 교정 → **L2/L3 정확도 증분 0 (multi-task robust negative)** 확정.

### 3. The Diagnosis — 왜 실패했나 (probe_14)
프롬프트가 LLM을 *제품 그 자체*에 한정 → L2/L3는 L1을 결정하는 같은 제품에서 추론되어 **by-construction redundant**. 측정: **L1 임베딩만으로 L2/L3를 평균 lift 0.38로 예측**(style_lineage 44-class 0.51, style_mood 21-class 0.71). → 정확도 redundancy는 고칠 수 있는 버그가 아니라 **추출 방향의 구조적 귀결**.

### 4. The Pivot — interpretable multi-purpose enrichment
진단이 가리키는 수정을 de-risk + full-scale 검증:
- **Controllability (probe_15) = GO.** L2/L3는 metadata엔 없는 8 의미축을 노출 → 추천을 target으로 soft-steer 시 **steered precision 1.00 vs 무제어 0.14**(개인화 100% 유지). L1만으론 "Party occasion 추천" 표현 불가 = L2/L3 고유 capability.
- **External knowledge (probe_16→21·22) = 정직한 NO-GO.** pair-level 보완 랭킹에선 +12.2%(CI 유의)였으나, **full-scale 100% coverage 빌드에서 KAR 0.00424 vs L1 0.00482 = −12%(0/3)**. probe_22가 원인을 **population-selection bias**로 격리(de-risk subset = 인기 heavy buyer 편향). de-risk positive(+17%)를 스스로 반증한 과정이 무결성 가치.
- **Value matrix (E2):** 4 결정-축 × 4 use = 16 cell, **capability 14/16(dense) vs 행동-검증 lift 3/16(sparse)**.

### 5. 추천 가치지도 — 전 축 negative (정직 종결)
| axis | verdict | evidence |
|---|---|---|
| accuracy (HR/NDCG/MAP) | **NEGATIVE** | full-scale −12% (probe_21), cold-start −21% |
| diversity / coverage | **NEGATIVE** | probe_02 −7.7% / −2.3% |
| serendipity / novelty / long-tail | **NEGATIVE (tie at best)** | R-10 / probe_23 (5/5 NO; placebo가 novelty 최고지만 serendipitous hit ≈0) |
| gap decision-lift (markdown/sell-through/…) | **NEGATIVE** | E2-5 / probe_E2d (5/5 margin 미달) |

→ LLM enrichment는 *소비자 추천*을 어느 축에서도 개선하지 않는다. 가치는 아래에.

---

## What's Actually Valuable

| 가치 | 무엇 | 근거 |
|---|---|---|
| **Merchant 의사결정-지원** (제품화) | ① trend lead-time 3개월 조기경보 · ② launch first-week sell-through 스코어카드 · ③ co-purchase velocity/anchor 랭킹 | r=**0.472** / η=**0.673** / η=**0.631** (`src/serving/merch_scenarios.py`, `notebooks/06,07`) |
| **Interpretable decision-axes** | metadata·L1과 직교하는 결정/감사/제어 차원 (capability **14/16**) | E2-1 (meta_p·l1_p 직교) |
| **Human-in-the-loop 제어** | 의미축 steering 표면 (metadata엔 없음) | D3 / probe_15 (precision 1.00 vs 0.14) |
| **견고한 baseline** | content **L1**이 popularity를 **+80~104%** 능가 | probe_07/08/21 |
| **연구 무결성 (과정)** | de-risk→scale 자기반증·population-bias 격리·capability vs prediction 분리 | STORY.md, contribution_notes.md |

---

## Key Results

### Baseline (Phase 0, Validation Set, k=12)
| Model | MAP@12 | HR@12 | NDCG@12 | MRR |
|-------|--------|-------|---------|-----|
| **Popularity Global** | **0.003783** | **0.044994** | **0.008122** | **0.015481** |
| UserKNN (ALS) | 0.003036 | 0.033901 | 0.006319 | 0.012228 |
| BPR-MF | 0.001308 | 0.016069 | 0.002839 | 0.004924 |

*"More personalization = worse"* — content **L1** Two-Tower만이 popularity를 +80~104% 능가(견고한 positive). LLM L2/L3·외부지식의 추가 정확도 증분은 정직한 0/negative.

### Value Matrix (E2) — capability vs decision-lift
- **capability 14/16 (dense)** · **measured decision-lift 3/16 (sparse)** → 제품화한 3 PASS는 모두 *행동-파생 축*(trend_phase·outfit_role); LLM 인식축은 L1과 redundant라 제외.
- Figures: `results/figures/{E2_value_matrix,probe_23_serendipity,E2d_gap_decision}.png`; demo: `notebooks/07_demo.ipynb`.

---

## Architecture

```
[Offline]  LLM/VLM Attribute Extraction → BGE Encoding → behavioral/perception axes (matrix_axes.parquet)
           → value-matrix probes (capability × decision-lift) → confidence cards (canonical JSON)
[Serving]  merch_scenarios engine → batch decision-support briefs (trend / launch / co-purchase)
           confidence는 canonical에서 LOAD(재계산 X) = single source of truth
```

---

## Quick Start

```bash
# 0. Preprocess: raw CSV → DuckDB/Parquet
python scripts/preprocess.py --raw-dir data/h-and-m-personalized-fashion-recommendations --output-dir data/processed

# 1. Extract attributes (L1+L2+L3 factual; enrichment-v2 perception/behavioral axes)
python scripts/extract_factual_knowledge.py --data-dir data/processed --output-dir data/knowledge/factual --batch-api
python scripts/extract_enrichment_v2.py     --data-dir data/processed --output-dir data/knowledge/enrichment_v2

# 2. Build the matrix-ready axes + run the value-matrix probes
python scripts/build_enrichment_matrix.py --data-dir data/processed --e2-dir data/knowledge/enrichment_v2
PYTHONPATH=. python witnesses/probe_E2_value_matrix.py
PYTHONPATH=. python witnesses/probe_23_serendipity.py     # R-10 serendipity (non-accuracy axis)

# 3. Serve the merchandising decision-support briefs (the deployable product)
python scripts/serve_scenarios.py --scenario all

# 4. See it all together — demo
PYTHONPATH=. jupyter nbconvert --to notebook --execute --inplace notebooks/07_demo.ipynb
```

자세한 CLI·데이터 흐름은 [Scripts Tutorial](docs/scripts_tutorial.md) 참고.

---

## Project Structure

```
llm-factor-recsys-hnm/
├── src/                    # Core library
│   ├── knowledge/          #   factual (L1/L2/L3) · enrichment_v2 (perception/behavioral) · reasoning
│   ├── features/           #   enrichment_matrix, lead_lag, behavioral/audience axes
│   ├── serving/            #   merch_scenarios (decision-support engine)
│   ├── models/ · kar/      #   5 backbones + KAR (Expert/Gating/Fusion)
├── scripts/                # CLI entry points (preprocess, extract, build, serve_scenarios, ...)
├── witnesses/              # Falsification probes + canonical *_result.json (the evidence)
├── notebooks/              # 00–05 analysis · 06 merch scenarios · 07 end-to-end demo
├── results/                # figures · tables · notebooks/07_demo.html
├── docs/research_design/   # STORY · contribution_notes · enrichment_v2_design · redesign · unified design
├── mlops/                  # FastAPI serving · Prometheus+Grafana · DVC · Docker · K8s
└── tests/                  # unit + integration
```

---

## Tech Stack

JAX + Flax NNX + Optax | DuckDB + Parquet | BGE-base-en-v1.5 | GPT-4.1-nano (multimodal) | FastAPI + Redis | Prometheus + Grafana | DVC | Docker + K8s | W&B | Typer + Hydra

---

## Documentation

- [STORY](docs/research_design/STORY.md) — Falsification → Diagnosis → Pivot → honest scale-refutation (the narrative)
- [Contribution Notes](docs/research_design/contribution_notes.md) — per-phase contributions + cumulative numbers (canonical-cited)
- [Enrichment v2 Design](docs/research_design/enrichment_v2_design.md) — interpretable catalog-enrichment value matrix
- [Research Design](docs/research_design/hm_unified_project_design.md) — full project design (architecture, experiments, roadmap)
- [Cold-Start Analysis](docs/cold_start_analysis.md) — Triple-Sparsity analysis and resolution strategies
- [Scripts Tutorial](docs/scripts_tutorial.md) — CLI usage and data flow
</content>
