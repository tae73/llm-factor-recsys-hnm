# H&M LLM-Factor RecSys — MLOps 설계

> 3-Layer Attribute Taxonomy + KAR Hybrid-Expert Adaptor 기반 H&M 패션 추천 시스템의 MLOps 인프라 설계. DVC 파이프라인, 온라인 서빙, 관측성, 컨테이너화, CI/CD까지 연구 재현성과 엔지니어링 역량 시연을 동시에 달성하는 기술 명세.

---

## 1. 설계 원칙 및 범위

### 1.1 설계 원칙

본 프로젝트의 MLOps 설계는 세 가지 핵심 원칙에 기반한다.

**원칙 1 — 연구 재현성(Reproducibility):** 모든 실험은 `dvc exp run` 한 줄로 재현 가능해야 한다. 데이터 전처리부터 평가까지의 전 과정이 DVC DAG로 추적되며, 하이퍼파라미터는 `params.yaml`에서 Git으로 버전 관리된다. Fix-and-Vary Ablation(7종 Layer 조합 × 4종 Gating × 4종 Fusion × 3종 Encoder × 5종 Backbone)의 체계적 실행과 비교가 핵심 요구사항이다.

**원칙 2 — 엔지니어링 역량 시연(Engineering Demonstration):** 연구 프로젝트이지만, 프로덕션 수준의 서빙 아키텍처를 설계·구현하여 포트폴리오 시각 증거를 확보한다. FastAPI + Redis + Prometheus + Grafana 풀스택을 실제 동작하는 수준으로 구현하되, 실트래픽 운영은 범위 밖이다.

**원칙 3 — 과잉 방지(No Over-engineering):** 1인 연구 프로젝트에 적합한 단순 스택을 유지한다. Feature Store(Feast)는 DuckDB + Parquet + .npz로 대체하고, 모델 서빙(ONNX Runtime)은 JAX JIT 네이티브로 대체하며, 별도 ML 플랫폼(MLflow, Kubeflow)은 W&B + DVC 조합으로 대체한다.

### 1.2 범위 정의

| 구분 | 항목 | 상태 |
|------|------|------|
| **구현** | DVC 파이프라인 (7 stages) | 완전 구현 |
| **구현** | FastAPI 서빙 앱 (5 endpoints) | 완전 구현 |
| **구현** | Redis 캐싱 + Graceful Degradation | 완전 구현 |
| **구현** | Prometheus 커스텀 메트릭 (6종) | 완전 구현 |
| **구현** | Grafana 대시보드 (2종, JSON provisioning) | 완전 구현 |
| **구현** | Docker + docker-compose (4 services) | 완전 구현 |
| **구현** | CI/CD (GitHub Actions, 4 stages) | 완전 구현 |
| **구현** | W&B 모델 레지스트리 | 완전 구현 |
| **구현** | Locust 부하 테스트 | 완전 구현 |
| **구현** | Unit + Integration 테스트 | 완전 구현 |
| **시연** | K8s 매니페스트 (Kustomize) | 매니페스트만, 실제 클러스터 불필요 |
| **시연** | HPA (Horizontal Pod Autoscaler) | 매니페스트만 |
| **범위 밖** | 실트래픽 운영 | 연구 프로젝트 |
| **범위 밖** | Canary/Blue-Green 배포 | 불필요 |
| **범위 밖** | A/B 테스트 프레임워크 | 불필요 |

### 1.3 기존 설계와의 관계

본 문서는 `docs/research_design/hm_unified_project_design.md`의 다음 섹션을 MLOps 관점에서 상세 확장한다.

- **Section 7.8 (추론 파이프라인 및 서빙 아키텍처):** Pre-store 파이프라인과 2-Stage 추천 서빙의 구현 수준 명세를 본 문서 Section 3으로 확장.
- **Section 10 (기술 스택):** 데이터 저장(10.4), 추천 모델(10.3)의 기술 결정을 인프라 구현 수준으로 확장.
- **Section 9 (구현 로드맵) Phase 7:** 서빙 파이프라인 구현의 세부 태스크를 본 문서 전체로 분해.

일관성 보장 포인트:
- Latency 목표: ~15ms (Section 7.8.2 일치)
- Pre-store: .npz in-memory numpy (Section 10.4 일치)
- Candidate Gen 4종 → Ranking top-12 (Section 7.8.2 일치)
- 텍스트 인코더: BGE-base-en-v1.5 (Section 7.2 일치)
- 실험 추적: W&B (Section 10.3 일치)

---

## 2. DVC 파이프라인

### 2.1 DAG 구조

DVC(Data Version Control) 파이프라인은 7개 스테이지로 구성되며, 데이터 전처리부터 모델 평가까지 전 과정을 재현 가능한 DAG(Directed Acyclic Graph)로 관리한다.

```
┌──────────────┐
│  preprocess   │  Raw CSV → DuckDB/Parquet
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│extract_factual_knowledge│  LLM/VLM L1+L2+L3 속성 추출
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  extract_reasoning_knowledge   │  유저 프로파일 추론
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  build_features   │  Feature engineering + 시간 분할
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│   prestore    │  Augmented Vector 사전 계산
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    train      │  추천 모델 학습
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   evaluate    │  MAP@12, HR@12, NDCG@12, MRR
└──────────────┘
```

각 스테이지는 `scripts/` 디렉토리의 CLI 엔트리포인트를 직접 호출하며, DVC는 재현성 래퍼(wrapper)로만 기능한다. 실제 배치/학습 로직은 모두 `scripts/`와 `src/`에 구현된다.

### 2.2 각 Stage 상세 명세

#### 2.2.1 preprocess

Raw CSV 파일을 DuckDB에 로드하고 Parquet 형식으로 변환한다. 가격 5분위 변환, 결측치 처리, 데이터 타입 최적화를 수행한다.

- **cmd:** `python scripts/preprocess.py --raw-dir data/h-and-m-personalized-fashion-recommendations --output-dir data/processed`
- **deps:** `scripts/preprocess.py`, `src/data/preprocessing.py`, `data/h-and-m-personalized-fashion-recommendations/`
- **outs:** `data/processed/` (articles.parquet, customers.parquet, transactions.parquet)

#### 2.2.2 extract_factual_knowledge

GPT-4.1-nano(멀티모달) + Structured Output을 사용하여 ~47K 고유 제품의 L1+L2+L3 속성을 Per-Item 통합 프롬프트로 추출한다. product_code 기반 캐싱으로 색상 변형을 재활용하여 105K SKU에 전파한다.

- **cmd:** `python scripts/extract_factual_knowledge.py --data-dir data/processed --images-dir data/h-and-m-personalized-fashion-recommendations/images --output-dir data/knowledge/factual --batch-api --max-cost ${extract.max_cost_usd}`
- **deps:** `scripts/extract_factual_knowledge.py`, `src/knowledge/factual/extractor.py`, `src/knowledge/factual/prompts.py`, `src/knowledge/factual/cache.py`, `data/processed/articles.parquet`
- **params:** `extract.model`, `extract.max_cost_usd`
- **outs:** `data/knowledge/factual/` (factual_knowledge.parquet, quality_report.json, checkpoint/)

#### 2.2.3 extract_reasoning_knowledge

유저 프로파일을 추론한다. L1은 거래 데이터에서 직접 집계하고, L2/L3는 LLM Factorization Prompting으로 추론한다.

- **cmd:** `python scripts/extract_reasoning_knowledge.py --data-dir data/processed --attr-dir data/knowledge/factual --output-dir data/knowledge/reasoning --min-purchases ${profile.min_purchases}`
- **deps:** `scripts/extract_reasoning_knowledge.py`, `src/knowledge/reasoning/extractor.py`, `src/knowledge/reasoning/prompts.py`, `data/processed/transactions.parquet`, `data/knowledge/factual/`
- **params:** `profile.min_purchases`
- **outs:** `data/knowledge/reasoning/` (user_profiles.parquet, factual_texts.parquet, reasoning_texts.parquet)

#### 2.2.4 build_features

DuckDB 기반 피처 엔지니어링과 시간 분할(train/val/test)을 수행한다.

- **cmd:** `python scripts/build_features.py --data-dir data/processed --attr-dir data/knowledge/factual --profile-dir data/knowledge/reasoning --output-dir data/features`
- **deps:** `scripts/build_features.py`, `src/features/engineering.py`, `src/data/splitter.py`, `data/processed/`, `data/knowledge/factual/`, `data/knowledge/reasoning/`
- **outs:** `data/features/` (train.parquet, val.parquet, test.parquet, feature_store/)

#### 2.2.5 prestore

학습된 KAR 모듈(Text Encoder + Expert + Gating)을 사용하여 전체 아이템·유저의 Augmented Vector를 사전 계산한다.

- **cmd:** `python scripts/prestore.py --attr-dir data/knowledge/factual --profile-dir data/knowledge/reasoning --model-dir results/models/kar --output-dir data/prestore`
- **deps:** `scripts/prestore.py`, `src/serving/prestore.py`, `src/kar/text_encoder.py`, `src/kar/expert.py`, `src/kar/gating.py`, `data/knowledge/factual/`, `data/knowledge/reasoning/`, `results/models/kar/`
- **outs:** `data/prestore/` (item_store.npz, user_store.npz)

#### 2.2.6 train

추천 모델을 학습한다. Multi-stage(기본) 또는 End-to-End 학습 전략을 지원한다.

- **cmd:** `python scripts/train.py --data-dir data/features --prestore-dir data/prestore --model-dir results/models --backbone ${train.backbone} --gating ${train.gating} --fusion ${train.fusion} --stage ${train.stage} --lr ${train.lr} --batch-size ${train.batch_size} --epochs ${train.epochs}`
- **deps:** `scripts/train.py`, `src/models/`, `src/kar/`, `src/losses.py`, `data/features/`, `data/prestore/`
- **params:** `train.backbone`, `train.gating`, `train.fusion`, `train.stage`, `train.lr`, `train.batch_size`, `train.epochs`
- **outs:** `results/models/${train.backbone}/` (model checkpoint, config)
- **plots:** `results/plots/training_curves.json`

#### 2.2.7 evaluate

학습된 모델의 추천 성능을 평가한다.

- **cmd:** `python scripts/evaluate.py --model-dir results/models --data-dir data/features --metrics map@12,hr@12,ndcg@12,mrr`
- **deps:** `scripts/evaluate.py`, `results/models/`, `data/features/test.parquet`
- **metrics:** `results/metrics/eval_metrics.json` (MAP@12, HR@12, NDCG@12, MRR)

### 2.3 dvc.yaml 전체 내용

```yaml
stages:
  preprocess:
    cmd: >
      python scripts/preprocess.py
        --raw-dir data/h-and-m-personalized-fashion-recommendations
        --output-dir data/processed
    deps:
      - scripts/preprocess.py
      - src/data/preprocessing.py
      - data/h-and-m-personalized-fashion-recommendations/articles.csv
      - data/h-and-m-personalized-fashion-recommendations/customers.csv
      - data/h-and-m-personalized-fashion-recommendations/transactions_train.csv
    outs:
      - data/processed/

  extract_factual_knowledge:
    cmd: >
      python scripts/extract_factual_knowledge.py
        --data-dir data/processed
        --images-dir data/h-and-m-personalized-fashion-recommendations/images
        --output-dir data/knowledge/factual
        --batch-api
        --max-cost ${extract.max_cost_usd}
        --model ${extract.model}
    deps:
      - scripts/extract_factual_knowledge.py
      - src/knowledge/factual/extractor.py
      - src/knowledge/factual/prompts.py
      - src/knowledge/factual/cache.py
      - data/processed/articles.parquet
    params:
      - extract.model
      - extract.max_cost_usd
    outs:
      - data/knowledge/factual/

  extract_reasoning_knowledge:
    cmd: >
      python scripts/extract_reasoning_knowledge.py
        --data-dir data/processed
        --attr-dir data/knowledge/factual
        --output-dir data/knowledge/reasoning
        --min-purchases ${profile.min_purchases}
    deps:
      - scripts/extract_reasoning_knowledge.py
      - src/knowledge/reasoning/extractor.py
      - src/knowledge/reasoning/prompts.py
      - data/processed/transactions.parquet
      - data/knowledge/factual/
    params:
      - profile.min_purchases
    outs:
      - data/knowledge/reasoning/

  build_features:
    cmd: >
      python scripts/build_features.py
        --data-dir data/processed
        --attr-dir data/knowledge/factual
        --profile-dir data/knowledge/reasoning
        --output-dir data/features
    deps:
      - scripts/build_features.py
      - src/features/engineering.py
      - src/data/splitter.py
      - data/processed/
      - data/knowledge/factual/
      - data/knowledge/reasoning/
    outs:
      - data/features/

  prestore:
    cmd: >
      python scripts/prestore.py
        --attr-dir data/knowledge/factual
        --profile-dir data/knowledge/reasoning
        --model-dir results/models/kar
        --output-dir data/prestore
    deps:
      - scripts/prestore.py
      - src/serving/prestore.py
      - src/kar/text_encoder.py
      - src/kar/expert.py
      - src/kar/gating.py
      - data/knowledge/factual/
      - data/knowledge/reasoning/
      - results/models/kar/
    outs:
      - data/prestore/

  train:
    cmd: >
      python scripts/train.py
        --data-dir data/features
        --prestore-dir data/prestore
        --model-dir results/models
        --backbone ${train.backbone}
        --gating ${train.gating}
        --fusion ${train.fusion}
        --stage ${train.stage}
        --lr ${train.lr}
        --batch-size ${train.batch_size}
        --epochs ${train.epochs}
    deps:
      - scripts/train.py
      - src/models/
      - src/kar/
      - src/losses.py
      - data/features/
      - data/prestore/
    params:
      - train
    outs:
      - results/models/${train.backbone}/
    plots:
      - results/plots/training_curves.json:
          x: epoch
          y: loss

  evaluate:
    cmd: >
      python scripts/evaluate.py
        --model-dir results/models
        --data-dir data/features
        --metrics map@12,hr@12,ndcg@12,mrr
    deps:
      - scripts/evaluate.py
      - results/models/
      - data/features/
    metrics:
      - results/metrics/eval_metrics.json:
          cache: false
```

### 2.4 params.yaml 전체 내용

```yaml
# === 속성 추출 ===
extract:
  layers: "l1,l2,l3"        # 변형: "l1", "l2", "l3", "l1,l2", "l1,l3", "l2,l3", "l1,l2,l3"
  model: "gpt-4o-mini"
  batch_size: 100

# === 유저 프로파일 ===
profile:
  min_purchases: 5           # 활성 유저 최소 구매 수

# === 모델 학습 ===
train:
  backbone: "deepfm"         # 변형: deepfm, sasrec, lightgcn, dcnv2, din
  gating: "g2"               # 변형: g1, g2, g3, g4
  fusion: "f2"               # 변형: f1, f2, f3, f4
  encoder: "bge-base-frozen" # 변형: bge-base-frozen, bge-base-finetune, tfidf-projection
  stage: "multi"             # 변형: multi, e2e
  lr: 1.0e-3
  lr_finetune: 1.0e-4        # Stage 3 fine-tuning 학습률
  batch_size: 2048
  seq_length: 50             # SASRec 전용
  epochs: 10
  d_rec: 64                  # 추천 임베딩 차원
  d_hidden: 256              # Expert 은닉층 차원
  d_enc: 384                 # 텍스트 인코더 출력 차원
  neg_ratio: 4               # 네거티브 샘플링 비율
  alpha: 0.1                 # ℒ_align 가중치
  beta: 0.01                 # ℒ_div 가중치
  patience: 3                # Early stopping patience

# === 서빙 ===
serve:
  candidate_pool_size: 500
  top_k: 12
  faiss_nprobe: 32
```

### 2.5 실험 관리

DVC Experiments를 활용하여 Fix-and-Vary Ablation을 체계적으로 실행한다.

**Layer Ablation (7 변형):**

```bash
# L1 Only
dvc exp run -S extract.layers="l1"

# L2 Only
dvc exp run -S extract.layers="l2"

# L3 Only
dvc exp run -S extract.layers="l3"

# L1+L2
dvc exp run -S extract.layers="l1,l2"

# L1+L3
dvc exp run -S extract.layers="l1,l3"

# L2+L3
dvc exp run -S extract.layers="l2,l3"

# Full (기본)
dvc exp run -S extract.layers="l1,l2,l3"
```

**Gating / Fusion / Backbone 변형:**

```bash
# Gating 변형
dvc exp run -S train.gating="g1"
dvc exp run -S train.gating="g3"
dvc exp run -S train.gating="g4"

# Fusion 변형
dvc exp run -S train.fusion="f1"
dvc exp run -S train.fusion="f3"
dvc exp run -S train.fusion="f4"

# Backbone 변형
dvc exp run -S train.backbone="sasrec" -S train.batch_size=256
dvc exp run -S train.backbone="lightgcn"
dvc exp run -S train.backbone="dcnv2"
dvc exp run -S train.backbone="din"
```

**실험 결과 비교:**

```bash
# MAP@12 기준 정렬
dvc exp show --sort-by results/metrics/eval_metrics.json:map_at_12 --drop .*

# 특정 파라미터 컬럼만 표시
dvc exp show --include-params train.backbone,train.gating,train.fusion,extract.layers

# 최적 실험 체크아웃
dvc exp apply <exp-name>
```

### 2.6 원격 스토리지

DVC remote를 설정하여 대용량 데이터와 모델 아티팩트를 원격 저장소에서 관리한다.

```bash
# S3 호환 스토리지 설정 (선택적)
dvc remote add -d storage s3://hm-recsys/dvc-store
dvc remote modify storage endpointurl https://storage.example.com

# 로컬 개발 시 (기본)
dvc remote add -d local /tmp/dvc-store

# 데이터 push/pull
dvc push
dvc pull
```

`.dvc` 파일은 Git에 커밋되어 데이터 버전을 추적한다. Raw 데이터(`data/h-and-m-personalized-fashion-recommendations/`)는 `.gitignore`에 포함되고 DVC로만 관리된다.

---

## 3. 온라인 서빙 아키텍처

### 3.1 전체 요청 흐름

```
Client (POST /recommend)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  FastAPI App (app.py)                            │
│                                                   │
│  1. Request Validation (Pydantic)                │
│     │                                             │
│  2. Redis Cache Check                            │
│     │                                             │
│     ├── HIT ──→ Return cached result (~0.5ms)    │
│     │                                             │
│     └── MISS                                      │
│          │                                        │
│  3. Candidate Generation (~5ms)                  │
│     │  ├── Popularity: Top-100 (recent 1w)       │
│     │  ├── CF: UserKNN/ItemKNN (~100)            │
│     │  ├── Attribute: Faiss ANN (~200)           │
│     │  └── Sequential: SASRec top (~100)         │
│     │  → Union: ~300-500 candidates              │
│     │                                             │
│  4. Ranking (~10ms)                              │
│     │  ├── numpy lookup: e_aug (item/user)       │
│     │  ├── Embedding Fusion                       │
│     │  ├── JAX JIT Model Inference (batch)       │
│     │  └── Top-12 selection                      │
│     │                                             │
│  5. Redis Cache Store (TTL 1h)                   │
│     │                                             │
│  6. Return RecommendResponse                     │
│                                                   │
│  Total: ~15ms (cache miss)                       │
└─────────────────────────────────────────────────┘
```

### 3.2 FastAPI 앱 구조

`mlops/serving/app.py`는 FastAPI 앱의 핵심 엔트리포인트이다.

#### 3.2.1 엔드포인트 명세

| Method | Path | Description | Latency Target |
|--------|------|-------------|----------------|
| POST | `/recommend` | 유저 ID 기반 Top-12 추천 | ~15ms (miss), ~0.5ms (hit) |
| POST | `/recommend/explain` | 추천 결과 + Gating 기반 설명 | ~20ms |
| GET | `/health` | Liveness probe | <1ms |
| GET | `/ready` | Readiness probe (모델·Redis 상태) | <5ms |
| GET | `/metrics` | Prometheus 메트릭 엔드포인트 | <1ms |

#### 3.2.2 앱 설계

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from mlops.serving.dependencies import create_app_state, AppState

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 모델, prestore, Faiss, Redis 로드
    app.state.deps = await create_app_state()
    yield
    # Shutdown: Redis pool 정리
    await app.state.deps.redis_pool.close()

app = FastAPI(
    title="H&M LLM-Factor RecSys",
    version="0.1.0",
    lifespan=lifespan,
)

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    ...

@app.post("/recommend/explain", response_model=ExplainResponse)
async def recommend_explain(request: ExplainRequest):
    ...

@app.get("/health", response_model=HealthResponse)
async def health():
    ...

@app.get("/ready", response_model=ReadyResponse)
async def ready():
    ...
```

### 3.3 Lifespan 관리

`mlops/serving/dependencies.py`는 앱 시작 시 필요한 리소스를 메모리에 로드하고, 종료 시 정리한다.

#### 3.3.1 AppState 설계

```python
from typing import NamedTuple
import numpy as np
import jax
import redis.asyncio as aioredis

class AppState(NamedTuple):
    # Pre-store (in-memory numpy)
    item_vectors: np.ndarray     # shape: [N_items, d_rec], ~50MB
    item_ids: np.ndarray         # shape: [N_items]
    item_id_to_idx: dict         # {article_id: idx}
    item_gating: np.ndarray      # shape: [N_items, 2] (g_fact, g_reason)
    user_vectors: np.ndarray     # shape: [N_users, d_rec], ~500MB
    user_ids: np.ndarray         # shape: [N_users]
    user_id_to_idx: dict         # {customer_id: idx}
    user_gating: np.ndarray      # shape: [N_users, 2]

    # Model (JAX JIT-compiled)
    model_fn: callable           # JIT-compiled ranking function
    model_params: dict           # Flax NNX model state

    # Faiss index (Attribute-based ANN)
    faiss_index: object          # faiss.IndexHNSWFlat

    # Redis connection pool
    redis_pool: aioredis.ConnectionPool
```

#### 3.3.2 로드 순서

```python
async def create_app_state() -> AppState:
    # 1. Pre-store 로드 (~1s)
    item_store = np.load("data/prestore/item_store.npz")
    user_store = np.load("data/prestore/user_store.npz")
    item_id_to_idx = {int(id_): i for i, id_ in enumerate(item_store["ids"])}
    user_id_to_idx = {str(id_): i for i, id_ in enumerate(user_store["ids"])}

    # 2. Faiss HNSW 인덱스 구축 (~2s)
    import faiss
    d = item_store["e_aug"].shape[1]
    index = faiss.IndexHNSWFlat(d, 32)  # M=32
    index.hnsw.efSearch = 64
    index.add(item_store["e_aug"].astype(np.float32))

    # 3. JAX 모델 로드 + JIT 컴파일 (~3s)
    model, params = load_model("results/models/deepfm/")
    model_fn = nnx.jit(model.__call__)
    # Warmup JIT
    dummy = jax.numpy.zeros((1, d))
    _ = model_fn(dummy)

    # 4. Redis 연결 풀 (~0.1s)
    pool = aioredis.ConnectionPool.from_url(
        "redis://localhost:6379",
        max_connections=20,
        socket_timeout=0.1,  # 100ms timeout
        retry_on_timeout=True,
    )

    return AppState(
        item_vectors=item_store["e_aug"],
        item_ids=item_store["ids"],
        item_id_to_idx=item_id_to_idx,
        item_gating=item_store["gating"],
        user_vectors=user_store["e_aug"],
        user_ids=user_store["ids"],
        user_id_to_idx=user_id_to_idx,
        user_gating=user_store["gating"],
        model_fn=model_fn,
        model_params=params,
        faiss_index=index,
        redis_pool=pool,
    )
```

### 3.4 Pydantic 스키마

`mlops/serving/schemas.py`는 모든 요청/응답의 타입을 정의한다.

```python
from pydantic import BaseModel, Field

class RecommendRequest(BaseModel):
    user_id: str
    n_items: int = Field(default=12, ge=1, le=50)
    exclude_items: list[str] = Field(default_factory=list)

class RecommendItem(BaseModel):
    article_id: str
    score: float
    rank: int

class RecommendResponse(BaseModel):
    user_id: str
    items: list[RecommendItem]
    cached: bool
    latency_ms: float

class ExplainRequest(BaseModel):
    user_id: str
    article_id: str

class ExplainResponse(BaseModel):
    user_id: str
    article_id: str
    g_fact: float               # Gating weight for factual
    g_reason: float             # Gating weight for reasoning
    explanation: str            # LLM-generated explanation
    l1_match: dict              # L1 속성 매칭 점수
    l2_match: dict              # L2 속성 매칭 점수
    l3_match: dict              # L3 속성 매칭 점수

class HealthResponse(BaseModel):
    status: str                 # "healthy" | "unhealthy"

class ReadyResponse(BaseModel):
    status: str                 # "ready" | "not_ready"
    model_loaded: bool
    prestore_loaded: bool
    faiss_ready: bool
    redis_connected: bool
```

### 3.5 추천 파이프라인

요청 처리의 전체 흐름을 pseudocode로 기술한다.

```python
async def recommend(request: RecommendRequest, state: AppState) -> RecommendResponse:
    start = time.perf_counter()

    # 1. Redis 캐시 체크
    cached = await cache_get(state.redis_pool, request.user_id)
    if cached is not None:
        CACHE_HIT.inc()
        return RecommendResponse(
            user_id=request.user_id, items=cached,
            cached=True, latency_ms=(time.perf_counter() - start) * 1000,
        )
    CACHE_MISS.inc()

    # 2. 유저 벡터 조회
    user_idx = state.user_id_to_idx.get(request.user_id)
    if user_idx is None:
        # Cold-start: 인기도 기반 fallback
        return popularity_fallback(request, state)
    e_aug_user = state.user_vectors[user_idx]

    # 3. Candidate Generation (~5ms)
    with RECOMMENDATION_LATENCY.labels(stage="candidate_gen").time():
        candidates = generate_candidates(
            e_aug_user, state, pool_size=500,
            exclude=request.exclude_items,
        )
    CANDIDATE_POOL_SIZE.observe(len(candidates))

    # 4. Ranking (~10ms)
    with RECOMMENDATION_LATENCY.labels(stage="ranking").time():
        scores = batch_rank(
            state.model_fn, state.model_params,
            e_aug_user, candidates, state,
        )

    # 5. Top-K 선택
    top_indices = jax.numpy.argsort(scores)[::-1][:request.n_items]
    items = [
        RecommendItem(
            article_id=candidates[i].article_id,
            score=float(scores[i]),
            rank=rank + 1,
        )
        for rank, i in enumerate(top_indices)
    ]

    # 6. Redis 캐시 저장
    await cache_set(state.redis_pool, request.user_id, items, ttl=3600)

    latency = (time.perf_counter() - start) * 1000
    RECOMMENDATION_LATENCY.labels(stage="total").observe(latency / 1000)

    return RecommendResponse(
        user_id=request.user_id, items=items,
        cached=False, latency_ms=latency,
    )
```

---

## 4. Redis 캐싱 전략

### 4.1 키 패턴 및 데이터 구조

| 키 패턴 | 데이터 타입 | TTL | 설명 |
|---------|------------|-----|------|
| `rec:{user_id}` | JSON String | 1시간 | 추천 결과 (Top-12 아이템 + 점수) |
| `session:{user_id}` | List (LPUSH) | 24시간 | 세션 컨텍스트 (최근 조회/클릭 아이템) |

```python
# 추천 결과 캐싱
key = f"rec:{user_id}"
value = json.dumps([item.model_dump() for item in items])
await redis.set(key, value, ex=3600)  # TTL 1h

# 세션 컨텍스트 (실시간 행동 추적용)
session_key = f"session:{user_id}"
await redis.lpush(session_key, article_id)
await redis.ltrim(session_key, 0, 49)  # 최근 50건 유지
await redis.expire(session_key, 86400)  # TTL 24h
```

### 4.2 연결 풀 설정

```python
REDIS_CONFIG = {
    "url": "redis://localhost:6379/0",
    "max_connections": 20,
    "socket_timeout": 0.1,       # 100ms — 빠른 실패
    "socket_connect_timeout": 0.5,
    "retry_on_timeout": True,
    "retry": Retry(ExponentialBackoff(), retries=2),
    "health_check_interval": 30,
}
```

### 4.3 Graceful Degradation

Redis 장애 시 추천 서비스가 중단되지 않도록 numpy fallthrough 경로를 제공한다.

```python
async def cache_get(pool, user_id: str) -> list | None:
    try:
        redis = aioredis.Redis(connection_pool=pool)
        data = await asyncio.wait_for(
            redis.get(f"rec:{user_id}"),
            timeout=0.1,  # 100ms hard timeout
        )
        if data:
            return json.loads(data)
        return None
    except (aioredis.ConnectionError, asyncio.TimeoutError, Exception):
        # Redis 장애 → 캐시 미스로 처리, numpy path로 fallthrough
        return None

async def cache_set(pool, user_id: str, items: list, ttl: int = 3600):
    try:
        redis = aioredis.Redis(connection_pool=pool)
        await asyncio.wait_for(
            redis.set(f"rec:{user_id}", json.dumps(items), ex=ttl),
            timeout=0.1,
        )
    except (aioredis.ConnectionError, asyncio.TimeoutError, Exception):
        # Redis 저장 실패 → 무시 (다음 요청에서 재계산)
        pass
```

### 4.4 캐시 무효화 전략

| 이벤트 | 무효화 방법 | 근거 |
|--------|------------|------|
| 모델 업데이트 (재학습) | `FLUSHDB` 전체 초기화 | 모든 추천 결과가 구모델 기반 |
| Pre-store 갱신 | `FLUSHDB` 전체 초기화 | Augmented Vector 변경 |
| 유저 새 구매 | `DEL rec:{user_id}` 개별 삭제 | 해당 유저 추천만 갱신 필요 |
| TTL 만료 | 자동 (1시간) | 주기적 자연 갱신 |

---

## 5. 관측성 (Observability)

### 5.1 Prometheus 메트릭 설계

`mlops/serving/metrics.py`에서 6종 커스텀 메트릭을 정의한다.

#### 5.1.1 메트릭 명세

| # | Name | Type | Labels | Buckets/Range | 설명 |
|---|------|------|--------|---------------|------|
| 1 | `recommendation_latency_seconds` | Histogram | `stage` (candidate_gen, ranking, total) | [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25] | 추천 파이프라인 단계별 레이턴시 |
| 2 | `candidate_pool_size` | Histogram | — | [50, 100, 200, 300, 500, 1000] | 후보 풀 크기 분포 |
| 3 | `recommendation_cache_hits_total` | Counter | — | — | 캐시 히트 누적 횟수 |
| 4 | `recommendation_cache_misses_total` | Counter | — | — | 캐시 미스 누적 횟수 |
| 5 | `model_prediction_score` | Histogram | — | [0.0, 0.1, 0.2, ..., 0.9, 1.0] | 모델 예측 점수 분포 |
| 6 | `feature_drift_ks_statistic` | Gauge | `feature_name` | 0.0 ~ 1.0 | KS test 드리프트 점수 |

#### 5.1.2 구현

```python
from prometheus_client import Counter, Histogram, Gauge

RECOMMENDATION_LATENCY = Histogram(
    "recommendation_latency_seconds",
    "Recommendation pipeline latency by stage",
    labelnames=["stage"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

CANDIDATE_POOL_SIZE = Histogram(
    "candidate_pool_size",
    "Number of candidates generated per request",
    buckets=[50, 100, 200, 300, 500, 1000],
)

CACHE_HIT = Counter(
    "recommendation_cache_hits_total",
    "Total number of recommendation cache hits",
)

CACHE_MISS = Counter(
    "recommendation_cache_misses_total",
    "Total number of recommendation cache misses",
)

MODEL_PREDICTION_SCORE = Histogram(
    "model_prediction_score",
    "Distribution of model prediction scores",
    buckets=[i / 10 for i in range(11)],
)

DRIFT_SCORE = Gauge(
    "feature_drift_ks_statistic",
    "KS test statistic for feature drift detection",
    labelnames=["feature_name"],
)
```

#### 5.1.3 FastAPI Instrumentator

기본 HTTP 메트릭(request count, latency, status codes)은 `prometheus-fastapi-instrumentator`로 자동 수집한다.

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

### 5.2 Grafana 대시보드 설계

2종 대시보드를 JSON provisioning으로 자동 배포한다.

#### 5.2.1 Serving Overview 대시보드

| # | 패널 | 쿼리 (PromQL) | 시각화 |
|---|------|--------------|--------|
| 1 | Request Rate | `rate(http_requests_total[5m])` | Time Series |
| 2 | Error Rate | `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])` | Stat + Threshold |
| 3 | Latency p50/p95/p99 | `histogram_quantile(0.5/0.95/0.99, rate(recommendation_latency_seconds_bucket{stage="total"}[5m]))` | Time Series |
| 4 | Stage Breakdown | `histogram_quantile(0.95, rate(recommendation_latency_seconds_bucket[5m]))` by stage | Stacked Bar |
| 5 | Cache Hit Ratio | `rate(recommendation_cache_hits_total[5m]) / (rate(recommendation_cache_hits_total[5m]) + rate(recommendation_cache_misses_total[5m]))` | Gauge |
| 6 | Candidate Pool Size | `histogram_quantile(0.5, rate(candidate_pool_size_bucket[5m]))` | Time Series |
| 7 | Throughput | `rate(http_requests_total{handler="/recommend"}[5m])` | Stat |

#### 5.2.2 Model Health 대시보드

| # | 패널 | 쿼리 (PromQL) | 시각화 |
|---|------|--------------|--------|
| 1 | Prediction Score Distribution | `histogram_quantile(0.25/0.5/0.75, rate(model_prediction_score_bucket[5m]))` | Time Series |
| 2 | Drift KS Statistics | `feature_drift_ks_statistic` by feature_name | Time Series + Threshold (0.1 warning, 0.2 critical) |
| 3 | Cold-start User Ratio | custom metric | Stat |
| 4 | Gating Weight Distribution | custom metric (g_fact/g_reason 평균) | Pie Chart |
| 5 | Feature Importance | custom metric (top features by prediction contribution) | Bar Chart |

### 5.3 알림 규칙

| 조건 | 심각도 | 대응 |
|------|--------|------|
| Latency p99 > 50ms (5분 지속) | Warning | Faiss nprobe 조정, 후보 풀 크기 축소 |
| Latency p99 > 100ms (5분 지속) | Critical | 서비스 스케일 아웃, 캐시 TTL 연장 |
| Error Rate > 1% (5분 지속) | Warning | 로그 확인, Redis 연결 상태 점검 |
| Error Rate > 5% (5분 지속) | Critical | 롤백, 모델 상태 확인 |
| Drift KS > 0.1 (any feature) | Warning | 드리프트 리포트 생성, 재학습 검토 |
| Drift KS > 0.2 (any feature) | Critical | 즉시 재학습 트리거, W&B alert 발송 |

---

## 6. 모델 버전 관리

### 6.1 W&B Artifacts 워크플로우

`mlops/tracking/model_registry.py`에서 W&B Artifacts 기반 모델 버전 관리를 구현한다.

#### 6.1.1 모델 저장

```python
def save_model(
    model_dir: str,
    backbone: str,
    gating: str,
    fusion: str,
    layer_combo: str,
    metrics: dict,
    run: wandb.Run,
) -> str:
    """모델 체크포인트를 W&B Artifact로 저장."""
    artifact_name = f"model-{backbone}-{gating}-{fusion}-{layer_combo}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata={
            "backbone": backbone,
            "gating": gating,
            "fusion": fusion,
            "layer_combo": layer_combo,
            **metrics,
        },
    )
    artifact.add_dir(model_dir)
    run.log_artifact(artifact, aliases=["latest"])
    return artifact_name
```

#### 6.1.2 모델 로드

```python
def load_model(
    artifact_name: str,
    alias: str = "best",
    download_dir: str = "results/models/",
) -> str:
    """W&B Artifact에서 모델 다운로드."""
    api = wandb.Api()
    artifact = api.artifact(f"hm-recsys/{artifact_name}:{alias}")
    return artifact.download(root=download_dir)
```

#### 6.1.3 모델 프로모션

```python
def promote_model(
    artifact_name: str,
    from_alias: str = "best",
    to_alias: str = "production",
):
    """모델을 production alias로 프로모션."""
    api = wandb.Api()
    artifact = api.artifact(f"hm-recsys/{artifact_name}:{from_alias}")
    artifact.aliases.append(to_alias)
    artifact.save()
```

### 6.2 아티팩트 네이밍 컨벤션

```
model-{backbone}-{gating}-{fusion}-{layer_combo}

예시:
  model-deepfm-g2-f2-l1l2l3        # 기본 설정 (Full)
  model-deepfm-g2-f2-l1             # L1 Only ablation
  model-sasrec-g2-f2-l1l2l3         # SASRec backbone
  model-deepfm-g3-f2-l1l2l3         # Context-conditioned gating
```

### 6.3 실험 그룹 구조

W&B 실험을 논리적 그룹으로 조직하여 비교 분석을 용이하게 한다.

| Group | 고정 | 변동 | 실험 수 |
|-------|------|------|---------|
| `layer_ablation` | DeepFM, G2, F2, BGE-frozen | extract.layers (7종) | 7 |
| `gating_search` | DeepFM, F2, BGE-frozen, Full | train.gating (4종) | 4 |
| `fusion_search` | DeepFM, G2*, BGE-frozen, Full | train.fusion (4종) | 4 |
| `encoder_search` | DeepFM, G2*, F2*, Full | train.encoder (3종) | 3 |
| `backbone_search` | G2*, F2*, BGE-frozen*, Full | train.backbone (5종) | 5 |

\* 이전 단계에서 결정된 최적 설정 사용

```python
# W&B 실험 그룹 설정
wandb.init(
    project="hm-recsys",
    group="layer_ablation",
    name=f"layers-{layer_combo}",
    config={...},
)
```

### 6.4 모델 비교 및 프로모션 워크플로우

```
1. 실험 실행 (dvc exp run)
   └── W&B에 메트릭 + 아티팩트 자동 로깅

2. 실험 비교 (W&B Dashboard)
   └── MAP@12, HR@12, NDCG@12 비교 테이블

3. 최적 모델 선정
   └── dvc exp apply <best-exp>
   └── promote_model("model-deepfm-g2-f2-l1l2l3", "latest", "best")

4. 프로덕션 배포
   └── promote_model("model-deepfm-g2-f2-l1l2l3", "best", "production")
   └── Pre-store 재계산 (scripts/prestore.py)
   └── Docker 이미지 빌드 + 배포
```

---

## 7. 피처 드리프트 감지

### 7.1 KS Test 기반 감지

`mlops/monitoring/drift_detector.py`에서 Kolmogorov-Smirnov test를 사용하여 학습 시점 대비 서빙 시점의 피처 분포 변화를 감지한다.

#### 7.1.1 감지 대상 피처

| 피처 | 레퍼런스 분포 | 서빙 분포 | 근거 |
|------|-------------|----------|------|
| `prediction_score` | 학습 데이터 예측 점수 | 최근 N건 예측 점수 | 모델 성능 변화 조기 감지 |
| `e_aug_norm` | 학습 시 e_aug L2 norm | 서빙 시 e_aug L2 norm | 속성 벡터 분포 변화 |
| `price` | 학습 기간 가격 분포 | 서빙 기간 가격 분포 | 카탈로그 구성 변화 |
| `category_dist` | 학습 기간 카테고리 분포 | 서빙 기간 카테고리 분포 | 상품 구성 변화 |

#### 7.1.2 구현

```python
from scipy.stats import ks_2samp
from typing import NamedTuple

class DriftResult(NamedTuple):
    feature_name: str
    ks_statistic: float
    p_value: float
    is_warning: bool     # KS > 0.1
    is_critical: bool    # KS > 0.2

def detect_drift(
    reference: np.ndarray,
    serving: np.ndarray,
    feature_name: str,
    warning_threshold: float = 0.1,
    critical_threshold: float = 0.2,
) -> DriftResult:
    stat, p_value = ks_2samp(reference, serving)

    # Prometheus gauge 업데이트
    DRIFT_SCORE.labels(feature_name=feature_name).set(stat)

    return DriftResult(
        feature_name=feature_name,
        ks_statistic=stat,
        p_value=p_value,
        is_warning=stat > warning_threshold,
        is_critical=stat > critical_threshold,
    )
```

### 7.2 임계값 및 대응

| KS Statistic | 수준 | 대응 |
|--------------|------|------|
| < 0.1 | 정상 | Prometheus gauge 기록만 |
| 0.1 ~ 0.2 | Warning | Prometheus gauge + Grafana 경고 표시 |
| > 0.2 | Critical | W&B alert 발송 + 재학습 트리거 권고 |

```python
def handle_drift(result: DriftResult, wandb_run=None):
    if result.is_critical and wandb_run:
        wandb.alert(
            title=f"Feature Drift Critical: {result.feature_name}",
            text=f"KS statistic: {result.ks_statistic:.4f}, p-value: {result.p_value:.6f}",
            level=wandb.AlertLevel.ERROR,
        )
    elif result.is_warning and wandb_run:
        wandb.alert(
            title=f"Feature Drift Warning: {result.feature_name}",
            text=f"KS statistic: {result.ks_statistic:.4f}",
            level=wandb.AlertLevel.WARN,
        )
```

### 7.3 감지 주기

| 모드 | 주기 | 구현 | 용도 |
|------|------|------|------|
| 배치 | 일 1회 (cron) | 학습 데이터 vs 전일 서빙 로그 | 장기 트렌드 감지 |
| 실시간 | 1,000건 누적 시 | 서빙 중 버퍼 축적 → 자동 감지 | 급격한 분포 변화 조기 경보 |

---

## 8. 컨테이너화

### 8.1 Dockerfile

Multi-stage 빌드로 이미지 크기를 최소화한다.

```dockerfile
# Stage 1: Dependencies
FROM python:3.10-slim AS deps

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[serve]" \
    && rm -rf /root/.cache/pip

# Stage 2: Runtime
FROM python:3.10-slim AS runtime

# 보안: non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Dependencies 복사
COPY --from=deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# 소스 코드 복사
COPY src/ src/
COPY mlops/ mlops/

# 데이터 디렉토리 (런타임에 volume mount)
RUN mkdir -p data/prestore results/models \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### 8.2 docker-compose.yml 전체 구성

```yaml
version: "3.8"

services:
  api:
    build:
      context: .
      dockerfile: mlops/docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./data/prestore:/app/data/prestore:ro
      - ./results/models:/app/results/models:ro
    environment:
      - REDIS_URL=redis://redis:6379/0
      - MODEL_DIR=/app/results/models/deepfm
      - PRESTORE_DIR=/app/data/prestore
      - WANDB_API_KEY=${WANDB_API_KEY}
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "2.0"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  prometheus:
    image: prom/prometheus:v2.48.0
    ports:
      - "9090:9090"
    volumes:
      - ./mlops/monitoring-config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=7d"

  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    volumes:
      - ./mlops/monitoring-config/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./mlops/monitoring-config/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/serving-overview.json

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

### 8.3 환경 변수 및 시크릿 관리

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `REDIS_URL` | Redis 연결 URL | `redis://localhost:6379/0` |
| `MODEL_DIR` | 모델 체크포인트 경로 | `results/models/deepfm` |
| `PRESTORE_DIR` | Pre-store .npz 경로 | `data/prestore` |
| `WANDB_API_KEY` | W&B API 키 (시크릿) | — |
| `LOG_LEVEL` | 로깅 레벨 | `INFO` |
| `WORKERS` | Uvicorn 워커 수 | `1` |

시크릿(`WANDB_API_KEY`)은 docker-compose에서 `${WANDB_API_KEY}`로 호스트 환경 변수에서 주입한다. K8s에서는 Secret 리소스를 사용한다.

---

## 9. K8s 매니페스트

### 9.1 Kustomize 구조

```
mlops/k8s/
├── base/
│   ├── kustomization.yaml
│   ├── deployment-api.yaml
│   ├── service-api.yaml
│   ├── deployment-redis.yaml
│   ├── service-redis.yaml
│   ├── configmap.yaml
│   └── hpa.yaml
└── overlays/
    └── dev/
        ├── kustomization.yaml
        └── resource-limits-patch.yaml
```

### 9.2 base/ 매니페스트

#### 9.2.1 deployment-api.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hm-recsys-api
  labels:
    app: hm-recsys
    component: api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hm-recsys
      component: api
  template:
    metadata:
      labels:
        app: hm-recsys
        component: api
    spec:
      containers:
        - name: api
          image: hm-recsys-api:latest
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: hm-recsys-config
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 15
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          volumeMounts:
            - name: prestore-data
              mountPath: /app/data/prestore
              readOnly: true
            - name: model-data
              mountPath: /app/results/models
              readOnly: true
      volumes:
        - name: prestore-data
          persistentVolumeClaim:
            claimName: prestore-pvc
        - name: model-data
          persistentVolumeClaim:
            claimName: model-pvc
```

#### 9.2.2 service-api.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: hm-recsys-api
  labels:
    app: hm-recsys
    component: api
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
  selector:
    app: hm-recsys
    component: api
```

#### 9.2.3 deployment-redis.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hm-recsys-redis
  labels:
    app: hm-recsys
    component: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hm-recsys
      component: redis
  template:
    metadata:
      labels:
        app: hm-recsys
        component: redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          command: ["redis-server", "--appendonly", "yes", "--maxmemory", "256mb"]
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
```

#### 9.2.4 configmap.yaml

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hm-recsys-config
data:
  REDIS_URL: "redis://hm-recsys-redis:6379/0"
  MODEL_DIR: "/app/results/models/deepfm"
  PRESTORE_DIR: "/app/data/prestore"
  LOG_LEVEL: "INFO"
  WORKERS: "1"
```

#### 9.2.5 hpa.yaml

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hm-recsys-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hm-recsys-api
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

#### 9.2.6 kustomization.yaml (base)

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment-api.yaml
  - service-api.yaml
  - deployment-redis.yaml
  - service-redis.yaml
  - configmap.yaml
  - hpa.yaml

commonLabels:
  project: hm-recsys
```

### 9.3 overlays/dev/

개발 환경에서 낮은 리소스 제한을 적용한다.

#### 9.3.1 resource-limits-patch.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hm-recsys-api
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: api
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
```

#### 9.3.2 kustomization.yaml (dev overlay)

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

patches:
  - path: resource-limits-patch.yaml

namespace: hm-recsys-dev
```

### 9.4 매니페스트 검증

CI에서 매니페스트의 문법적·의미적 유효성을 검증한다.

```bash
# Kustomize 빌드 + kubectl dry-run 검증
kustomize build mlops/k8s/overlays/dev | kubectl apply --dry-run=client -f -
```

---

## 10. CI/CD

### 10.1 GitHub Actions 워크플로우 설계

2개 워크플로우를 운영한다.

| 워크플로우 | 트리거 | 역할 |
|-----------|--------|------|
| `ci.yml` | Push/PR to main | 코드 품질 + 테스트 + 빌드 + 검증 |
| `dvc-repro.yml` | Weekly (cron) | DVC DAG 유효성 검증 |

### 10.2 ci.yml 상세

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # === Stage 1: Code Quality ===
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install ruff black mypy
      - name: Ruff lint
        run: ruff check src/ mlops/ scripts/ tests/
      - name: Black format check
        run: black --check --line-length 100 src/ mlops/ scripts/ tests/
      - name: MyPy type check
        run: mypy src/ mlops/ --ignore-missing-imports

  # === Stage 2: Tests ===
  unit-tests:
    needs: quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e ".[test]"
      - name: Unit tests
        run: pytest tests/unit/ -v --tb=short

  integration-tests:
    needs: quality
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 3
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e ".[test]"
      - name: Integration tests
        run: pytest tests/integration/ -v --tb=short
        env:
          REDIS_URL: redis://localhost:6379/0

  # === Stage 3: Build ===
  docker-build:
    needs: [unit-tests, integration-tests]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -f mlops/docker/Dockerfile -t hm-recsys-api:${{ github.sha }} .

  # === Stage 4: Validate ===
  k8s-validate:
    needs: [unit-tests, integration-tests]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install kustomize
        run: |
          curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
          sudo mv kustomize /usr/local/bin/
      - name: Install kubectl
        uses: azure/setup-kubectl@v3
      - name: Validate K8s manifests
        run: kustomize build mlops/k8s/overlays/dev | kubectl apply --dry-run=client -f -
```

### 10.3 dvc-repro.yml 상세

```yaml
name: DVC Pipeline Validation

on:
  schedule:
    - cron: "0 6 * * 1"  # 매주 월요일 06:00 UTC
  workflow_dispatch:       # 수동 트리거 허용

jobs:
  validate-dag:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install dvc

      - name: Validate DVC DAG
        run: |
          cd mlops/pipeline
          dvc dag --dot  # DAG 구조 유효성 확인

      - name: Check DVC status
        run: |
          cd mlops/pipeline
          dvc status  # 파이프라인 상태 확인
```

---

## 11. 부하 테스트

### 11.1 Locust 시나리오

`mlops/loadtest/locustfile.py`에서 추천 엔드포인트의 부하 테스트를 정의한다.

```python
from locust import HttpUser, task, between
import random

# 테스트용 유저 ID 풀
TEST_USER_IDS = [f"user_{i:06d}" for i in range(1000)]

class RecommendUser(HttpUser):
    """추천 엔드포인트 부하 테스트."""
    wait_time = between(0.5, 2.0)
    weight = 9  # 전체 트래픽의 90%

    @task
    def get_recommendations(self):
        user_id = random.choice(TEST_USER_IDS)
        self.client.post(
            "/recommend",
            json={"user_id": user_id, "n_items": 12},
        )

class HealthCheckUser(HttpUser):
    """헬스체크 엔드포인트 부하 테스트."""
    wait_time = between(1.0, 3.0)
    weight = 1  # 전체 트래픽의 10%

    @task
    def check_health(self):
        self.client.get("/health")
```

### 11.2 성능 목표

| 메트릭 | 목표 | 근거 |
|--------|------|------|
| p50 Latency | < 5ms | 캐시 히트 포함 가중 평균 |
| p95 Latency | < 20ms | 캐시 미스 대부분 포함 |
| p99 Latency | < 50ms | 드문 고비용 요청 포함 |
| Throughput | > 100 RPS | 단일 인스턴스 기준 |
| Error Rate | < 0.1% | 정상 부하 조건 |

Latency 예산 분해 (캐시 미스 경로):
- numpy 벡터 조회: <0.01ms
- Faiss HNSW ANN 검색: ~5ms
- JAX JIT 배치 추론 (500 쌍): ~10ms
- Redis 저장: <0.5ms
- **총합: ~15ms**

### 11.3 테스트 프로파일

```bash
# Warm-up (30초, 10 users → 캐시 채우기)
locust -f mlops/loadtest/locustfile.py \
    --host http://localhost:8000 \
    --users 10 --spawn-rate 2 --run-time 30s --headless

# Steady-state (5분, 50 users)
locust -f mlops/loadtest/locustfile.py \
    --host http://localhost:8000 \
    --users 50 --spawn-rate 10 --run-time 5m --headless

# Spike test (50 → 200 users, 3분)
locust -f mlops/loadtest/locustfile.py \
    --host http://localhost:8000 \
    --users 200 --spawn-rate 50 --run-time 3m --headless
```

---

## 12. 테스트 전략

### 12.1 테스트 구조

```
tests/
├── unit/                    # 외부 의존성 없는 단위 테스트
│   ├── test_schemas.py      # Pydantic 스키마 검증
│   ├── test_metrics.py      # Prometheus 메트릭 등록
│   ├── test_drift.py        # 드리프트 계산 로직
│   └── test_registry.py     # 모델 레지스트리 네이밍
├── integration/             # Redis sidecar 필요
│   ├── test_cache.py        # 캐시 set/get/TTL/degradation
│   └── test_recommend.py    # 전체 추천 흐름
└── conftest.py              # 공유 fixtures
```

### 12.2 Unit 테스트

외부 서비스(Redis, W&B, GPU) 의존성 없이 실행 가능한 테스트이다.

```python
# tests/unit/test_schemas.py
def test_recommend_request_defaults():
    req = RecommendRequest(user_id="u001")
    assert req.n_items == 12
    assert req.exclude_items == []

def test_recommend_request_validation():
    with pytest.raises(ValidationError):
        RecommendRequest(user_id="u001", n_items=0)  # ge=1 위반

# tests/unit/test_drift.py
def test_no_drift_identical_distributions():
    ref = np.random.normal(0, 1, 1000)
    result = detect_drift(ref, ref, "test_feature")
    assert result.ks_statistic < 0.05
    assert not result.is_warning

def test_drift_detected_shifted_distribution():
    ref = np.random.normal(0, 1, 1000)
    shifted = np.random.normal(1, 1, 1000)  # 평균 1만큼 이동
    result = detect_drift(ref, shifted, "test_feature")
    assert result.is_critical  # KS > 0.2

# tests/unit/test_registry.py
def test_artifact_naming():
    name = build_artifact_name("deepfm", "g2", "f2", "l1l2l3")
    assert name == "model-deepfm-g2-f2-l1l2l3"

# tests/unit/test_metrics.py
def test_metrics_registered():
    from mlops.serving.metrics import (
        RECOMMENDATION_LATENCY, CACHE_HIT, CACHE_MISS,
        CANDIDATE_POOL_SIZE, MODEL_PREDICTION_SCORE, DRIFT_SCORE,
    )
    assert RECOMMENDATION_LATENCY is not None
    assert CACHE_HIT is not None
```

### 12.3 Integration 테스트

Redis sidecar가 필요한 통합 테스트이다. CI에서 Redis 서비스 컨테이너를 함께 기동한다.

```python
# tests/integration/test_cache.py
@pytest.mark.integration
async def test_cache_set_get(redis_pool):
    items = [{"article_id": "001", "score": 0.95, "rank": 1}]
    await cache_set(redis_pool, "test_user", items, ttl=60)

    result = await cache_get(redis_pool, "test_user")
    assert result is not None
    assert result[0]["article_id"] == "001"

@pytest.mark.integration
async def test_cache_ttl_expiry(redis_pool):
    await cache_set(redis_pool, "ttl_user", [{"id": "1"}], ttl=1)
    await asyncio.sleep(2)
    result = await cache_get(redis_pool, "ttl_user")
    assert result is None

@pytest.mark.integration
async def test_graceful_degradation():
    # 잘못된 Redis URL → cache_get이 None 반환 (예외 없음)
    bad_pool = aioredis.ConnectionPool.from_url("redis://nonexistent:6379")
    result = await cache_get(bad_pool, "any_user")
    assert result is None

# tests/integration/test_recommend.py
@pytest.mark.integration
async def test_full_recommend_flow(test_client):
    response = await test_client.post(
        "/recommend",
        json={"user_id": "test_user_001", "n_items": 12},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) <= 12
    assert data["latency_ms"] > 0
```

### 12.4 Fixtures 및 conftest.py

```python
# tests/conftest.py
import pytest
import redis.asyncio as aioredis

@pytest.fixture
async def redis_pool():
    """Integration 테스트용 Redis 연결 풀."""
    pool = aioredis.ConnectionPool.from_url(
        "redis://localhost:6379/1",  # DB 1 사용 (테스트 격리)
        max_connections=5,
    )
    yield pool
    # Cleanup: 테스트 DB 초기화
    r = aioredis.Redis(connection_pool=pool)
    await r.flushdb()
    await pool.disconnect()

@pytest.fixture
def sample_item_store(tmp_path):
    """테스트용 item_store.npz 생성."""
    n_items, d_rec = 100, 64
    np.savez(
        tmp_path / "item_store.npz",
        e_aug=np.random.randn(n_items, d_rec).astype(np.float32),
        ids=np.arange(n_items),
        gating=np.random.rand(n_items, 2).astype(np.float32),
    )
    return tmp_path / "item_store.npz"

@pytest.fixture
def sample_user_store(tmp_path):
    """테스트용 user_store.npz 생성."""
    n_users, d_rec = 50, 64
    np.savez(
        tmp_path / "user_store.npz",
        e_aug=np.random.randn(n_users, d_rec).astype(np.float32),
        ids=np.array([f"user_{i:06d}" for i in range(n_users)]),
        gating=np.random.rand(n_users, 2).astype(np.float32),
    )
    return tmp_path / "user_store.npz"
```

---

## 부록: rtb_ipinyou 프로젝트와의 차이점

이전 프로젝트(rtb_ipinyou)에서의 경험을 반영하여, 본 프로젝트에 적합한 기술 선택을 한 근거를 정리한다.

| 항목 | rtb_ipinyou | 본 프로젝트 | 변경 이유 |
|------|-------------|------------|----------|
| Feature Store | Redis (primary) + Feast | In-memory numpy + Redis (cache) | 105K+500K 규모가 RAM에 충분, Redis는 request-level 캐시로만 사용 |
| Model Format | ONNX Runtime | JAX JIT | JAX→ONNX 변환 미성숙, `nnx.jit` 자체로 충분한 추론 성능 |
| Pipeline | 없음 (scripts 직접) | DVC DAG | 연구 재현성 + Fix-and-Vary ablation 자동화 필수 |
| Monitoring | 설계만 (구현 없음) | Prometheus + Grafana (구현) | 포트폴리오 시각 증거 확보 |
| Deployment | Canary 점진 배포 | K8s manifests (역량 시연) | 연구 프로젝트, 실트래픽 없음 |
| Experiment Tracking | 없음 | W&B (Experiments + Artifacts) | 체계적 ablation 실험 관리 필수 |
