# H&M LLM-Factor RecSys

> 3-Layer Attribute Taxonomy + KAR Hybrid-Expert Adaptor for H&M Fashion Recommendation

---

## Problem: Triple-Sparsity in Fashion Recommendation

H&M 데이터셋(~105K items, ~1.37M users, ~31M transactions)은 세 가지 차원에서 동시에 희소성이 발현되어 협업 필터링(CF)만으로는 유의미한 개인화가 구조적으로 불가능하다.

| Sparsity Dimension | Metric | Impact |
|--------------------|--------|--------|
| User-side | 32.1% (436K) users have only 1-4 purchases | <0.004% catalog interaction per user |
| Matrix-side | 99.98% sparse interaction matrix | MF/GNN signal propagation fails |
| Signal quality | Popularity > UserKNN > BPR-MF | More personalization = worse performance |

87% of user-item pairs are single-purchase, making repeat-purchase prediction structurally limited. **Discovery-oriented recommendation** is essential.

---

## Why Not Just Existing Metadata?

H&M articles.csv에는 이미 풍부한 메타데이터(product_type 253종, colour 50종, detail_desc 텍스트)가 있다. DeepFM + 기존 메타데이터만으로도 순수 CF보다 나은 Content-Enhanced CF가 가능하다.

본 프로젝트가 검증하는 것은 **LLM 추출 속성의 증분 가치**:

| Level | 구성 | LLM 필요 |
|-------|------|---------|
| 0 | UserKNN, BPR-MF (ID only) | X |
| 1 | DeepFM + 기존 메타데이터 피처 | X |
| 2 | DeepFM + detail_desc 인코딩 | X |
| 3 | DeepFM + KAR(L1 구조화) | O |
| 4 | DeepFM + KAR(L1+L2+L3) | O |

Level 1→4의 증분 가치를 정량화하는 것이 핵심.

---

## Solution: Content-Based Multi-Layer Attribute Augmentation

### 3-Layer Attribute Taxonomy

- **L1 (Product)**: Material, fit, neckline, closure, design details — from metadata + LLM/VLM extraction
- **L2 (Perceptual)**: Style mood, occasion, perceived quality, trendiness — LLM world knowledge
- **L3 (Theory-grounded)**: Silhouette, color harmony, personal color tone, style lineage — fashion design theory

### KAR Hybrid-Expert Adaptor

- **Factual Expert**: L1+L2+L3 integrated text → BGE-base 768-dim → MLP → recommendation space
- **Reasoning Expert**: LLM Factorization Prompting → latent preference inference → recommendation space
- **Gating Network**: Dynamic combination of factual/reasoning signals per user-item pair

### Cold-Start Resolution

- **Full catalog scoring**: 105K items × JAX vmap → 전체 카탈로그 직접 스코어링 (~15ms)
- **1-purchase query**: Single purchase → attribute vector → personalized scoring (CF needs 5-10+)
- **Semantic bridge**: L2+L3 attributes connect sparse history to rich item descriptions

---

## Key Results

### Baseline (Phase 0, Validation Set, k=12)

| Model | MAP@12 | HR@12 | NDCG@12 | MRR | Level |
|-------|--------|-------|---------|-----|-------|
| **Popularity Global** | **0.003783** | **0.044994** | **0.008122** | **0.015481** | - |
| UserKNN (ALS) | 0.003036 | 0.033901 | 0.006319 | 0.012228 | 0 |
| BPR-MF | 0.001308 | 0.016069 | 0.002839 | 0.004924 | 0 |
| DeepFM + 기존 메타 | TBD | TBD | TBD | TBD | 1 |
| DeepFM + KAR(L1+L2+L3) | TBD | TBD | TBD | TBD | 4 |

---

## Architecture

```
[Offline] LLM Attribute Extraction → BGE Encoding → Expert + Gating → Pre-store (.npz)
[Online]  Pre-store lookup → JAX 전체 카탈로그 스코어링 → Top-12 (~15ms)
```

---

## Quick Start

```bash
# Environment setup
conda activate llm-factor-recsys-hnm
pip install -e ".[dev]"

# Pipeline (abbreviated)
python scripts/preprocess.py --raw-dir data/h-and-m-personalized-fashion-recommendations --output-dir data/processed
python scripts/extract_attributes.py --data-dir data/processed --output-dir data/attributes --layers l1,l2,l3
python scripts/extract_reasoning_knowledge.py --data-dir data/processed --attr-dir data/attributes --output-dir data/knowledge/reasoning
python scripts/build_features.py --data-dir data/processed --attr-dir data/attributes --profile-dir data/knowledge/reasoning --output-dir data/features
python scripts/train.py --data-dir data/features --prestore-dir data/prestore --model-dir results/models --backbone deepfm
python scripts/evaluate.py --model-dir results/models --data-dir data/features --metrics map@12,hr@12,ndcg@12,mrr
```

---

## Project Structure

```
llm-factor-recsys-hnm/
├── src/                    # Core library (attributes, profiles, features, models, kar, serving)
├── scripts/                # CLI entry points (preprocess, extract, train, evaluate, serve)
├── mlops/                  # MLOps (FastAPI serving, Prometheus+Grafana, DVC, Docker, K8s)
├── tests/                  # Unit + integration tests
├── notebooks/              # Analysis notebooks (00_eda.ipynb)
├── configs/                # Hydra YAML configs
├── data/                   # Dataset (git-ignored)
├── results/                # Models, figures, tables
└── docs/                   # Research design docs
```

---

## Tech Stack

JAX + Flax NNX + Optax | DuckDB + Parquet | BGE-base-en-v1.5 | GPT-4o-mini | FastAPI + Redis | Prometheus + Grafana | DVC | Docker + K8s | W&B | Typer + Hydra

---

## Documentation

- [Research Design](docs/research_design/hm_unified_project_design.md) — Full project design (architecture, experiments, roadmap)
- [Cold-Start Analysis](docs/cold_start_analysis.md) — Triple-Sparsity analysis and resolution strategies
- [MLOps Design](docs/mlops_design/) — Serving, monitoring, deployment design
- [Scripts Tutorial](docs/scripts_tutorial.md) — CLI usage and data flow
- [Config Design](docs/config_design.md) — Hydra config structure
