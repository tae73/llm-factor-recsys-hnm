# H&M LLM-Factor RecSys — CLAUDE.md

**IMPORTANT**: Git commit 시 Co-Authored-By 라인을 포함하지 않는다.

## Project Overview

3-Layer Attribute Taxonomy + KAR Hybrid-Expert Adaptor 기반 H&M 패션 추천 시스템 연구 프로젝트.
H&M 데이터의 Triple-Sparsity(유저 32.1% 희소, 아이템 57.3% tail, 행렬 99.98% sparse, CF 시그널 품질 저하) 환경에서, 기존 메타데이터 Content-Based를 넘어 LLM 추출 다층 속성(L1+L2+L3)의 증분 가치를 검증한다.

**IMPORTANT**: 매 작업 완료 시 `PLAN.md` 업데이트:
1. 완료된 Task 상태를 `[x]`로 변경
2. Key Findings에 주요 발견사항 추가
3. Next Steps 업데이트

**IMPORTANT**: 매 Phase 분석 완료(노트북 실행, eval 리포트 생성 등) 시 → `docs/research_design/contribution_notes.md` 업데이트:
- 해당 Phase의 Contribution 항목 추가 (번호 체계: `Contribution {Phase}-{N}`)
- 반드시 수치(%, 건수, 비율, 점수)를 포함 — 정성적 서술만으로는 불충분
- MD 시사점과 DS 시사점을 분리 기술
- Research Motivation(Triple-Sparsity)에 대비한 의미를 명시
- "누적 수치 요약" 테이블에 해당 Phase 행 업데이트

**IMPORTANT**: `src/` 또는 `scripts/` 변경 시 → `docs/scripts_tutorial.md` 업데이트:
- 각 스크립트의 사용법 (CLI 인자, 예시 명령어)
- src/ 모듈의 주요 함수 시그니처 및 사용 패턴
- 파이프라인 실행 순서 및 데이터 흐름 설명

**IMPORTANT**: `src/` ↔ `scripts/` 양방향 의존성 유지 — **동시 업데이트 필수**:
- `scripts/`는 CLI 래퍼(Typer → `src/` 호출)이고, `src/`는 핵심 로직을 담당한다
- **절대 규칙**: `src/` 모듈의 public 함수 시그니처를 변경하면, 해당 함수를 호출하는 `scripts/` 엔트리포인트를 **같은 커밋**에서 동기화해야 한다. 역방향도 동일.
- 변경 시 체크리스트:
  1. **Import 경로**: `src/` 모듈명·패키지 변경 → `scripts/`의 `from src.xxx import` 전부 업데이트
  2. **함수 시그니처**: 매개변수 추가/삭제/이름변경 → `scripts/`의 호출부 + CLI 인자 동기화
  3. **NamedTuple 필드**: `src/config.py`의 Config/Result 필드 변경 → `scripts/`에서 해당 필드 접근하는 코드 전부 업데이트
  4. **반환 타입**: `src/` 함수의 반환 타입 변경 → `scripts/`에서 결과를 사용하는 코드 업데이트
  5. **새 모듈 추가**: `src/xxx/` 신규 모듈 → 대응하는 `scripts/xxx.py` CLI 엔트리포인트 생성 (또는 기존 스크립트에 통합)
- **검증**: `scripts/` 변경 시 반드시 `python scripts/xxx.py --help`로 CLI 인자 목록 확인
- **docs 동기화**: 위 변경 발생 시 `docs/scripts_tutorial.md`도 같은 커밋에서 업데이트

**IMPORTANT**: Python 환경:
- Conda 환경: `conda activate llm-factor-recsys-hnm` (Python 3.11)
- 의존성 설치: `pip install -e ".[dev]"` (conda 환경 내에서 실행)
- 패키지 추가 시 `pyproject.toml`에 명시 후 `pip install -e ".[dev]"` 재실행

**Research Motivation**: H&M 데이터는 Triple-Sparsity(32.1% 유저 희소 + 99.98% 행렬 sparse + CF 시그널 품질 저하)로 인해 협업 필터링만으로는 구조적 한계가 있다. 아이템 측에서도 20.7% 아이템이 80% 거래를 차지하는 극심한 인기도 편중(Gini=0.7586)으로 57.3% tail 아이템의 CF 시그널이 불충분하여, 이 Triple-Sparsity를 더욱 증폭시킨다. 기존 메타데이터 Content-Based도 대안이나, LLM 추출 L2(체감)/L3(이론) 속성이 기존 메타데이터 대비 어떤 증분 가치를 제공하는지가 핵심 연구 질문이다. 87% 유저-아이템 쌍이 단일 구매이므로 반복 구매 예측이 아닌 발견 지향 추천이 필요하다.

**Research Question**: LLM으로 추출한 다층 속성(L1 제품/L2 체감/L3 이론 기반)을 KAR의 factual+reasoning 2-Expert 구조에 통합하여, H&M 패션 추천의 정확도·다양성·Cold-start 성능을 어떻게 향상시킬 것인가?

**핵심 혁신:**
1. **3-Layer Attribute Taxonomy**: L1(제품) + L2(체감) + L3(이론 기반) 속성 체계 최초 제안 및 체계적 실증
2. **KAR Hybrid-Expert Adaptor**: factual(L1+L2+L3 통합 텍스트) + reasoning(LLM 추론) 2-Expert 구조 채택
3. **텍스트 수준 Ablation**: Expert 아키텍처 변경 없이 Factual 텍스트 구성 변형(7종)으로 각 Layer 기여도 정량화
4. **Model-agnostic 속성 증강**: 5종 백본(DeepFM, SASRec, LightGCN, DCNv2, DIN)에서 범용 효과 검증
5. **Pre-store 서빙**: 오프라인 Augmented Vector 사전 계산 → 온라인 전체 카탈로그 스코어링 ~15ms
6. **Cold-start 해결 아키텍처**: Triple-Sparsity 환경에서 속성 벡터 + Reasoning Expert로 CF 시그널 없이 추천

---

## Project Structure

```
llm-factor-recsys-hnm/
├── src/                        # Core library
│   ├── data/                   # preprocessing.py, splitter.py
│   ├── knowledge/              # KAR knowledge extraction
│   │   ├── factual/            # extractor.py, prompts.py, batch.py, cache.py, validator.py, text_composer.py, image_utils.py
│   │   └── reasoning/          # extractor.py, prompts.py, batch.py, cache.py
│   ├── features/               # engineering.py, store.py
│   ├── models/                 # deepfm.py, sasrec.py, lightgcn.py, dcnv2.py, din.py
│   ├── kar/                    # text_encoder.py, expert.py, gating.py, fusion.py
│   ├── segmentation/           # clustering.py, affinity.py
│   ├── serving/                # candidate_gen.py, ranker.py, prestore.py
│   ├── embeddings.py            # BGE embedding computation (shared: segmentation + KAR)
│   ├── losses.py               # BCE, BPR, align, diversity losses
│   └── config.py               # Global configs (NamedTuples)
├── scripts/                    # CLI entry points
│   ├── preprocess.py           # Raw CSV → DuckDB/Parquet
│   ├── extract_factual_knowledge.py  # LLM/VLM 속성 추출 (L1+L2+L3)
│   ├── extract_reasoning_knowledge.py  # LLM 유저 Reasoning 추출
│   ├── build_features.py       # Feature engineering + Split
│   ├── prestore.py             # Augmented Vector 사전 계산
│   ├── train.py                # 추천 모델 학습
│   ├── evaluate.py             # 평가 (MAP@12, HR@12, NDCG@12)
│   └── serve.py                # 추천 서빙
├── mlops/                      # MLOps infrastructure
│   ├── serving/                # app.py, dependencies.py, schemas.py, cache.py, metrics.py
│   ├── monitoring/             # drift_detector.py, health.py
│   ├── tracking/               # wandb_logger.py, model_registry.py
│   ├── pipeline/               # dvc.yaml, params.yaml
│   ├── docker/                 # Dockerfile, docker-compose.yml (api+redis+prometheus+grafana)
│   ├── k8s/                    # Kustomize manifests (base + overlays/dev)
│   │   ├── base/               # deployment-api, service-api, deployment-redis, configmap, hpa
│   │   └── overlays/dev/       # Dev-specific resource limits
│   ├── monitoring-config/      # prometheus.yml, grafana provisioning + dashboards
│   └── loadtest/               # locustfile.py
├── .github/workflows/          # CI (lint+test+build+validate)
│   ├── ci.yml                  # Main: ruff, mypy, black, pytest, docker build, k8s validate
│   └── dvc-repro.yml           # Weekly: DVC DAG validation
├── tests/
│   ├── unit/                   # 외부 의존성 없는 단위 테스트
│   └── integration/            # Redis sidecar 필요한 통합 테스트
├── notebooks/                  # Analysis notebooks (00-xx)
├── data/                       # Dataset (git-ignored)
│   └── h-and-m-personalized-fashion-recommendations/
│       ├── articles.csv        # ~105K items
│       ├── customers.csv       # ~1.37M users
│       ├── transactions_train.csv  # ~31M transactions
│       └── images/             # Product images
├── results/                    # Models, figures, tables
├── configs/                    # YAML configs
├── docs/                       # Research design docs
│   └── research_design/        # hm_unified_project_design.md
├── PLAN.md                     # Progress tracking
└── pyproject.toml
```

---

## Dataset

**H&M Personalized Fashion Recommendations** — Kaggle 데이터셋.

- Location: `data/h-and-m-personalized-fashion-recommendations/`
- Period: 2018.09 ~ 2020.09 (약 2년)
- Scale: ~105K articles, ~1.37M customers, ~31M transactions

| File | Records | Key Fields |
|------|---------|------------|
| articles.csv | ~105,542 | article_id, product_type_name, colour_group_name, detail_desc |
| customers.csv | ~1,371,980 | customer_id, age, club_member_status, fashion_news_frequency |
| transactions_train.csv | ~31,788,324 | t_dat, customer_id, article_id, price, sales_channel_id |
| images/ | ~105K | {article_id}.jpg (흰 배경 상품 사진) |

**시간 분할:**
- Train: 2018.09 ~ 2020.06
- Validation: 2020.07 ~ 2020.08
- Test: 2020.09 첫 주

**고객 필터링:** 5건+ 구매 "활성 유저" + 1-4건 "희소 유저"(Cold-start 실험용) 분리.

---

## 핵심 기술 결정

| 항목 | 결정 | 근거 |
|------|------|------|
| 속성 체계 | 3-Layer Taxonomy (L1/L2/L3) | 제품→체감→이론 기반 다층 속성으로 추천 정보 극대화 |
| 속성 추출 모델 | GPT-4.1-nano (멀티모달) + Structured Output + Batch API | 비용 효율적 (~$10), 텍스트+이미지 동시 처리, Per-Item 통합 프롬프트 |
| KAR 구조 | factual + reasoning 2-Expert | KAR 원 논문의 검증된 아키텍처 그대로 채택 |
| 텍스트 인코더 | BGE-base-en-v1.5 (Frozen, 기본) | KAR 원 논문에서 효과적, 학습 비용 없음 |
| Gating | Expert-conditioned (G2, 기본) | KAR 원 논문 검증 방식 |
| Embedding Fusion | Addition (F2, 기본) | 차원 불변, 백본 수정 불필요 |
| 추천 백본 | DeepFM (기본) + 4종 대비 실험 | 풍부한 카테고리 피처 활용에 적합 |
| 학습 프레임워크 | JAX + Flax NNX + Optax | NNX의 Pythonic 뮤터블 API, nnx.jit JIT 컴파일 |
| 학습 루프 | `src/training/trainer.py` (핵심 로직) + `scripts/train.py` (CLI 래퍼) | 학습 로직을 src/에 분리하여 HPO·테스트·노트북에서 재사용 |
| 데이터 저장 | DuckDB + Parquet + .npz | 1인 연구 프로젝트에 적합한 단순 스택 |
| 스코어링 방식 | 전체 카탈로그 직접 스코어링 | 105K 규모에서 JAX vmap brute-force 가능 (~15ms) |
| 실험 추적 | Weights & Biases | 체계적 실험 관리 |
| 시각화 | matplotlib + seaborn (기본), Plotly + Streamlit (인터랙티브 대시보드만) | 가벼운 정적 시각화 기본, 대시보드에서만 Plotly |
| API Framework | FastAPI + Uvicorn | 비동기, 자동 OpenAPI docs, Pydantic 통합 |
| Model Serving | JAX JIT (native) | JAX→ONNX 미성숙, `nnx.jit` 자체가 충분 |
| Feature Lookup | In-memory numpy (.npz) | 105K+500K ≈ 550MB, dict lookup <0.01ms |
| Request Cache | Redis | 추천 결과 TTL 캐싱, 세션 컨텍스트, rate limiting |
| Model Registry | W&B Artifacts | 이미 W&B 사용, 별도 MLflow 불필요 |
| Metrics | Prometheus + Grafana | 산업 표준 관측성 스택, 대시보드 시각 증거 |
| Drift Detection | scipy KS test → Prometheus gauge | 통계적 엄밀성 + 운영 대시보드 통합 |
| Pipeline | DVC | 재현성 보장, DAG 시각화, `dvc exp run`으로 ablation |
| Container | Docker + docker-compose | 풀스택: API + Redis + Prometheus + Grafana |
| Orchestration | K8s manifests (Kustomize) | 프로덕션 배포 역량 시연 (실제 클러스터 불필요) |
| CI/CD | GitHub Actions (multi-stage) | lint → test → docker build → k8s validate |
| Load Test | Locust | ~15ms 레이턴시 타겟 검증, 논문 서빙 평가 데이터 |
| Config 관리 | Hydra Config Groups + OmegaConf | Config Group별 파일 분리, Compose API로 Typer 호환, DVC Grid Search 연동 |
| CLI 프레임워크 | Typer + Hydra Compose API | 타입 힌트 기반 CLI, @hydra.main() 없이 config composition |
| Data Loader | Grain (`grain.python`) | JAX 네이티브, deterministic, 멀티프로세스 prefetch, sharding 호환 |
| Distributed Training | `jax.sharding` + `nnx.jit` (single/multi-device auto) | jax.pmap 대비 통합 API, 단일/멀티 디바이스 코드 동일 |
| Hyperparameter Tuning | W&B Sweeps (Bayesian) + Hydra overrides | 이미 W&B 사용, DVC exp run 호환, scripts/ CLI 인자로 전달 |

---

## 주요 파일

| File | Description |
|------|-------------|
| `src/knowledge/factual/extractor.py` | 추출 엔진: product_code 그룹핑 → API 호출 → 파싱 (GPT-4.1-nano) |
| `src/knowledge/factual/prompts.py` | Super-Category별 통합 system prompt (3종) + JSON Schema |
| `src/knowledge/factual/batch.py` | OpenAI Batch API 래퍼 (JSONL 생성 → 제출 → 폴링 → 파싱) |
| `src/knowledge/factual/cache.py` | product_code 기반 캐싱 (dict + Parquet 체크포인트) |
| `src/knowledge/factual/validator.py` | 스키마 검증 (enum 범위, 필수 필드, 타입 체크) |
| `src/knowledge/factual/text_composer.py` | Ablation용 Factual 텍스트 조합 (7종 Layer 변형) |
| `src/knowledge/factual/image_utils.py` | 이미지 로드 + base64 인코딩 + 리사이즈 |
| `src/knowledge/reasoning/extractor.py` | 유저 Reasoning 추출 (L1 직접 집계 + L2/L3 LLM 추론) |
| `src/knowledge/reasoning/prompts.py` | LLM Factorization Prompting |
| `src/kar/text_encoder.py` | 속성 텍스트 인코더 (BGE-base / E5-base) |
| `src/kar/expert.py` | Factual + Reasoning Expert MLP |
| `src/kar/gating.py` | Gating Network (G1~G4 변형) |
| `src/kar/fusion.py` | Embedding Fusion (F1~F4 변형) |
| `src/models/deepfm.py` | DeepFM backbone (FM + DNN) |
| `src/models/sasrec.py` | SASRec backbone (Self-Attention Sequential) |
| `src/models/lightgcn.py` | LightGCN backbone (Graph Convolution) |
| `src/models/dcnv2.py` | DCNv2 backbone (Cross Network) |
| `src/models/din.py` | DIN backbone (Deep Interest Network) |
| `src/embeddings.py` | BGE embedding computation: `compute_item_embeddings()`, `compute_user_embeddings()`, `load_embeddings()` |
| `src/losses.py` | BCE, BPR, Align, Diversity losses |
| `src/features/engineering.py` | DuckDB 기반 피처 엔지니어링 |
| `src/serving/prestore.py` | Augmented Vector 사전 계산 (.npz) |
| `src/serving/candidate_gen.py` | 후보 생성 전략 (Future Work: 대규모 카탈로그 확장 시) |
| `mlops/serving/app.py` | FastAPI 앱 (/recommend, /health, /ready, /metrics) |
| `mlops/serving/dependencies.py` | Lifespan: 모델·prestore·Redis 메모리 로드 |
| `mlops/serving/schemas.py` | Pydantic 요청/응답 스키마 |
| `mlops/serving/cache.py` | Redis 추천 결과 캐싱 (TTL 1h, graceful degradation) |
| `mlops/serving/metrics.py` | Prometheus 커스텀 메트릭 (latency histogram, cache hit/miss, drift gauge) |
| `mlops/monitoring/drift_detector.py` | KS test → Prometheus gauge push + W&B alert |
| `mlops/monitoring/health.py` | 헬스/레디니스 체크 (모델·prestore·Redis 상태) |
| `mlops/tracking/wandb_logger.py` | W&B 래퍼: 메트릭·아티팩트 로깅 |
| `mlops/tracking/model_registry.py` | W&B Artifacts 모델 버전 관리 (save/load/promote) |
| `mlops/pipeline/dvc.yaml` | DVC 파이프라인 DAG (7 stages) |
| `mlops/pipeline/params.yaml` | 하이퍼파라미터 (DVC 추적, git 버전 관리) |
| `mlops/docker/docker-compose.yml` | 풀스택: API + Redis + Prometheus + Grafana |
| `mlops/k8s/base/deployment-api.yaml` | FastAPI Deployment (2 replicas, probes, resource limits) |
| `mlops/k8s/base/hpa.yaml` | HorizontalPodAutoscaler (CPU 70%, 2→8 replicas) |
| `mlops/loadtest/locustfile.py` | /recommend 엔드포인트 부하 테스트 |

---

## Pipeline Usage

```bash
# 0. Preprocess — Raw CSV → DuckDB/Parquet
python scripts/preprocess.py \
    --raw-dir data/h-and-m-personalized-fashion-recommendations \
    --output-dir data/processed

# 1. Extract factual knowledge — LLM/VLM L1+L2+L3 속성 추출
python scripts/extract_factual_knowledge.py \
    --data-dir data/processed \
    --images-dir data/h-and-m-personalized-fashion-recommendations/images \
    --output-dir data/knowledge/factual \
    --batch-api \
    --max-cost 15.0

# 2. Extract reasoning knowledge — LLM 유저 Reasoning 추출
python scripts/extract_reasoning_knowledge.py \
    --data-dir data/processed \
    --fk-dir data/knowledge/factual \
    --output-dir data/knowledge/reasoning \
    --min-purchases 5

# 3. Build features — Feature engineering + 시간 분할
python scripts/build_features.py \
    --data-dir data/processed \
    --fk-dir data/knowledge/factual \
    --rk-dir data/knowledge/reasoning \
    --output-dir data/features

# 4. Pre-store — Augmented Vector 사전 계산
python scripts/prestore.py \
    --fk-dir data/knowledge/factual \
    --rk-dir data/knowledge/reasoning \
    --model-dir results/models/kar \
    --output-dir data/prestore

# 5. Train — 추천 모델 학습 (Multi-stage)
python scripts/train.py \
    --data-dir data/features \
    --prestore-dir data/prestore \
    --model-dir results/models \
    --backbone deepfm \
    --gating g2 \
    --fusion f2 \
    --stage multi

# 6. Evaluate — 성능 평가
python scripts/evaluate.py \
    --model-dir results/models \
    --data-dir data/features \
    --metrics map@12,hr@12,ndcg@12,mrr

# 7. Serve — 추천 서빙
python scripts/serve.py \
    --model-dir results/models \
    --prestore-dir data/prestore \
    --port 8000
```

---

## MLOps

### A. 아키텍처 개관

```
[오프라인 — DVC Pipeline]
preprocess → extract_factual_knowledge → extract_reasoning_knowledge → build_features
    → prestore → train → evaluate
    (각 stage: deps/params/outs 추적, W&B 아티팩트 기록)

[온라인 — FastAPI + Redis + Prometheus]
Client → Redis 캐시 체크 → MISS → numpy lookup + JAX 전체 카탈로그 스코어링 → Redis 저장 → 응답
                           → HIT → 즉시 응답 (~0.5ms)

[관측성 — Prometheus + Grafana]
FastAPI /metrics → Prometheus (15s scrape) → Grafana 대시보드 2종
```

### B. DVC 파이프라인

`dvc.yaml` — 7 stages (preprocess, extract_factual_knowledge, extract_reasoning_knowledge, build_features, prestore, train, evaluate), 각 stage에 deps/params/outs/metrics 정의.

`params.yaml` — backbone, gating, fusion, layer combo, lr, batch_size 등 하이퍼파라미터.

Fix-and-Vary ablation:
```bash
dvc exp run -S extract.layers="l1"
dvc exp run -S extract.layers="l1,l2,l3"
dvc exp run -S train.backbone="sasrec"
dvc exp show --sort-by results/metrics/eval_metrics.json:map_at_12
```

### C. 온라인 서빙

- FastAPI 엔드포인트: `POST /recommend`, `POST /recommend/explain`, `GET /health`, `GET /ready`, `GET /metrics`
- Lifespan 로드: item_store.npz (~50MB), user_store.npz (~500MB), JIT-compiled Flax model
- Redis 캐싱: `rec:{user_id}` → JSON (TTL 1h), `session:{user_id}` → List (TTL 24h)
- Redis 장애 시 graceful degradation (socket_timeout=100ms, 실패 시 numpy path로 fallthrough)
- Latency: ~0.5ms (캐시 히트), ~15ms (캐시 미스: 전체 카탈로그 스코어링)

### D. Prometheus 커스텀 메트릭

```python
RECOMMENDATION_LATENCY = Histogram("recommendation_latency_seconds", ..., labelnames=["stage"])
CANDIDATE_POOL_SIZE = Histogram("candidate_pool_size", ...)
CACHE_HIT = Counter("recommendation_cache_hits_total")
CACHE_MISS = Counter("recommendation_cache_misses_total")
MODEL_PREDICTION_SCORE = Histogram("model_prediction_score", ...)
DRIFT_SCORE = Gauge("feature_drift_ks_statistic", ..., labelnames=["feature_name"])
```

### E. Grafana 대시보드 2종

1. **Serving Overview**: Request rate, error rate, latency p50/p95/p99, stage별 breakdown, cache hit ratio
2. **Model Health**: Prediction score distribution, drift KS statistics, cold-start user ratio

### F. 모델 버전 관리 (W&B Artifacts)

- 네이밍: `model-{backbone}-{gating}-{fusion}-{layer_combo}`
- 실험 그룹: `layer_ablation`, `gating_search`, `fusion_search`, `backbone_search`
- promote_model("best") → "production" alias

### G. K8s Manifests (Kustomize)

```
k8s/base/: deployment-api (2 replicas, probes, limits), service-api, deployment-redis,
           service-redis, configmap, hpa (CPU 70%, 2→8)
k8s/overlays/dev/: 낮은 resource limits for local kind/minikube
```
CI에서 `kustomize build | kubectl --dry-run=client` 검증.

### H. CI/CD (GitHub Actions)

```
Stage 1 (Quality): ruff + black --check + mypy
Stage 2 (Tests): pytest unit + pytest integration (Redis sidecar)
Stage 3 (Build): Docker build + K8s manifest validation
Stage 4 (Weekly): DVC DAG validation
```

### I. Docker 구성

```yaml
services:
  api:        # FastAPI + Uvicorn (port 8000)
  redis:      # Redis 7 Alpine (port 6379)
  prometheus: # Prometheus (port 9090, scrape api:8000/metrics)
  grafana:    # Grafana (port 3000, provisioned dashboards)
```

### J. rtb_ipinyou와의 차이점

| 항목 | rtb_ipinyou | 본 프로젝트 | 이유 |
|------|-------------|------------|------|
| Feature Store | Redis (primary) + Feast | In-memory numpy (primary) + Redis (cache) | RAM에 충분, Redis는 request-level 캐시로만 |
| Model Format | ONNX Runtime | JAX JIT | JAX→ONNX 미성숙 |
| Pipeline | 없음 (scripts 직접) | DVC DAG | 재현성 + ablation 자동화 |
| Monitoring | 설계만 (구현 없음) | Prometheus + Grafana (구현) | 포트폴리오 시각 증거 |
| Deployment | Canary 점진 배포 | K8s manifests (역량 시연) | 연구 프로젝트, 실트래픽 없음 |

---

## Coding Style

### Functional Programming
- Use `functools`, `itertools`, `NamedTuple`
- Prefer immutable data and pure functions
- Pandas: method chaining with `.assign()`, `.pipe()`, lambdas
- Prefer `map()` with lambdas over explicit loops

### NamedTuple Patterns
Use `NamedTuple` for both configuration and result objects:

```python
# Config objects (immutable settings)
class ExpertConfig(NamedTuple):
    d_enc: int = 384
    d_hidden: int = 256
    d_rec: int = 64
    n_layers: int = 2

# Result objects (structured returns)
class AblationResult(NamedTuple):
    layer_combo: str          # e.g., "L1+L2+L3"
    map_at_12: float
    hr_at_12: float
    ndcg_at_12: float
    g_fact: float             # Gating weight
    g_reason: float
```

### Type Hints
- All public functions must have type hints
- Common types: `jax.Array`, `np.ndarray`, `pd.DataFrame`, `Dict[str, ...]`, `Optional[T]`
- Return `NamedTuple` for structured outputs
- Flax NNX modules: `nnx.Module` subclass type hints

### Naming Conventions
| Type | Pattern | Example |
|------|---------|---------|
| Functions | verb_noun | `extract_factual_knowledge`, `build_profile`, `compute_map` |
| Constants | UPPER_SNAKE | `LAYER_NAMES`, `DEFAULT_CONFIG`, `GATING_VARIANTS` |
| Private helpers | _prefix | `_encode_factual_text`, `_compute_gating_weights` |
| Abbreviations | Domain standard | kar, aug, enc, rec, fact, reason, l1/l2/l3, emb |

### JAX/Flax NNX Conventions
```python
# Module pattern
class FactualExpert(nnx.Module):
    def __init__(self, config: ExpertConfig, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(config.d_enc, config.d_hidden, rngs=rngs)
        self.linear2 = nnx.Linear(config.d_hidden, config.d_rec, rngs=rngs)

    def __call__(self, h_fact: jax.Array) -> jax.Array:
        return nnx.relu(self.linear2(nnx.relu(self.linear1(h_fact))))

# JIT compilation
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch.features)
        return compute_loss(logits, batch.labels)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

# Optax optimizer
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3))
```

### Error Handling
- **try-except**: Edge case fallbacks (e.g., missing images, empty descriptions)
- **Conditional imports**: Optional dependencies with fallback
  ```python
  try:
      import faiss
      FAISS_AVAILABLE = True
  except ImportError:
      FAISS_AVAILABLE = False
  ```
- **ValueError**: Explicit validation for invalid inputs (unknown layer names, invalid gating variants)

### Data Processing Conventions
- **DuckDB SQL**: 오프라인 피처 계산, 집계 쿼리 (거래 로그 → 유저/아이템 통계)
- **Parquet**: DuckDB 네이티브 연동, 거래/메타데이터/속성 저장
- **.npz**: Augmented Vector + Gating 가중치 저장 (Pre-store)
- **Pandas chaining**: `.assign()`, `.pipe()`, `.query()` 체이닝
- Aggregations: use `.agg(**{...})` with explicit column naming
- Derived columns: chain `.assign()` calls with lambdas

### Notebook Conventions
- 모든 노트북 첫 셀에 아래 보일러플레이트 삽입:
  ```python
  %load_ext autoreload
  %autoreload 2

  import sys
  from pathlib import Path

  # Project root (1 level up from notebooks/)
  PROJECT_ROOT = Path('.').absolute().parent
  sys.path.insert(0, str(PROJECT_ROOT))
  ```
- 경로는 `PROJECT_ROOT` 기준 상대 경로 사용 (`Path('.').absolute().parent`로 프로젝트 루트 자동 탐지)

### Visualization Patterns
- **기본**: matplotlib + seaborn (노트북, 논문 figure, 정적 분석)
- **인터랙티브 대시보드만**: Plotly + Streamlit
- Define reusable plot functions
- Use config objects (`NamedTuple`) for consistent styling
- Prefer `fig, axes = plt.subplots()` with `list(map())` for batch plotting

```python
class PlotConfig(NamedTuple):
    width: float = 9.0
    height: float = 6.0
    style: str = "whitegrid"
    context: str = "notebook"
    dpi: int = 150

def plot_gating_distribution(
    gating_weights: pd.DataFrame,
    config: PlotConfig = PlotConfig(),
) -> plt.Figure:
    ...
```

### Modeling Conventions

#### Data Loading (Grain)
- 학습 데이터 로딩에 `grain.python` 사용 (JAX 네이티브 데이터 로더)
- `grain.python.DataLoader` — 멀티프로세스 prefetch, deterministic 셔플링
- `TrainPairsSource` → `FeatureLookupTransform` → `grain.Batch` → `grain.DataLoader`:
  ```python
  from src.training.data_loader import TrainPairsSource, FeatureLookupTransform

  source = TrainPairsSource(features_dir)  # __getitem__ → {user_idx, item_idx, label}
  user_features = load_user_features(features_dir)
  item_features = load_item_features(features_dir)

  sampler = grain.IndexSampler(
      num_records=len(source),
      num_epochs=1,
      shard_options=grain.ShardByJaxProcess(),
      shuffle=True,
      seed=config.random_seed,
  )
  loader = grain.DataLoader(
      data_source=source,
      sampler=sampler,
      operations=[
          FeatureLookupTransform(user_features, item_features),
          grain.Batch(batch_size=config.batch_size, drop_remainder=True),
      ],
      worker_count=config.num_workers,
      worker_buffer_size=config.prefetch_buffer_size,
  )
  ```
- `num_epochs=1`: 매 epoch마다 `seed + epoch`로 새 loader 생성 → 에포크별 다른 셔플
- `grain.ShardByJaxProcess()`: 멀티 호스트 학습 시 자동 데이터 분할 (single device = no-op)
- Deterministic: 동일 seed → 동일 배치 순서 (재현성 보장)

#### Distributed Training
- 모든 모델(`src/models/`)은 **device-agnostic**으로 작성한다
- `jax.sharding.NamedSharding` + `jax.device_put`으로 데이터/모델 샤딩
- `nnx.jit`은 자동으로 available devices를 활용 (단일 GPU ↔ 멀티 GPU 코드 변경 없음)
- 학습 루프(`src/training/`)에서 디바이스 감지 + 배치 분할 담당:
  ```python
  devices = jax.devices()
  n_devices = len(devices)
  mesh = jax.sharding.Mesh(np.array(devices), ("data",))
  data_sharding = NamedSharding(mesh, PartitionSpec("data"))
  replicated = NamedSharding(mesh, PartitionSpec())  # model params
  ```
- 모델 forward/backward에 디바이스 로직을 넣지 않는다 — 샤딩은 caller 책임
- CPU-only 환경(macOS)에서도 동일 코드 동작 (n_devices=1 → sharding no-op)

**IMPORTANT**: `src/training/` 내부 일반 학습(single-device)과 분산 학습(multi-device) 로직은 **동일 코드 경로**를 사용해야 한다:
- Mesh + NamedSharding + `jax.device_put()`는 항상 적용한다 (n_devices=1일 때 sharding은 자동 no-op)
- `train_step()` 함수는 sharded/unsharded 입력을 구분하지 않는다 — `@nnx.jit`이 자동 처리
- 조건 분기(`if n_devices > 1:`)로 학습 로직을 분리하지 않는다
- 새로운 백본 모델 추가 시에도 동일 원칙 적용: 모델은 device-agnostic, 샤딩은 caller(`src/training/`) 책임
- 검증: macOS CPU (n_devices=1)에서 전체 학습 파이프라인이 코드 변경 없이 동작해야 한다

#### Hyperparameter Tuning
- 모든 tunable 하이퍼파라미터는 `src/config.py` NamedTuple + `configs/` YAML에 정의
- `scripts/` CLI가 개별 하이퍼파라미터를 인자로 노출 (HPO 도구가 CLI로 주입 가능):
  ```bash
  # W&B Sweep agent가 호출하는 형태
  python scripts/train.py --backbone deepfm --learning-rate 0.001 --d-embed 16
  ```
- W&B Sweeps 설정: `configs/sweep/deepfm.yaml` (Bayesian search space 정의)
- `scripts/train.py`에서 `wandb.init(config=...)` → sweep agent가 config override
- DVC 연동: `dvc exp run -S train.learning_rate=0.0001` 형태로도 실행 가능
- 튜닝 우선순위: d_embed → learning_rate → dnn_hidden_dims → dropout_rate → batch_size

### Parallelization
- 논리적으로 병렬화 가능한 작업은 적극적으로 병렬화한다
- I/O-bound (파일 읽기, API 호출, 네트워크): `concurrent.futures.ThreadPoolExecutor` 사용
- CPU-bound (행렬 연산, 대규모 집계): `ray` 사용
- GPU-bound (모델 학습·추론, 배치 임베딩): JAX 병렬화 사용
- 배치 처리 (유저별 예측, 아이템별 속성 추출 등)는 기본적으로 병렬로 구현
- 단일 스레드/프로세스로 충분히 빠른 경우(< 1초)에는 병렬화 불필요

#### CPU/I/O 병렬화 패턴
- ThreadPool: 다수 유저 예측 생성, 파일 I/O 병렬화
- ray: 대규모 배치 속성 추출, 분산 모델 학습, 대규모 evaluation

#### JAX GPU 병렬화 패턴
- `jax.vmap`: 단일 디바이스 배치 벡터화 (유저/아이템 임베딩, Expert forward 등)
- `jax.pmap`: 멀티 GPU 데이터 병렬 학습 (각 디바이스에 배치 분할)
- `nnx.jit` + `jax.vmap` 조합: JIT 컴파일된 배치 추론
- `jax.lax.scan`: 시퀀셜 모델(SASRec 등)의 효율적 루프 대체
- 예시:
  ```python
  # vmap: 배치 유저 임베딩 계산
  batched_encode = jax.vmap(model.encode_user)
  user_embeddings = batched_encode(user_features)  # (B, d)

  # pmap: 멀티 GPU 학습
  @nnx.jit
  def train_step(model, optimizer, batch):
      ...
  parallel_train = jax.pmap(train_step, axis_name="devices")

  # vmap + jit: Expert forward 배치 처리
  @nnx.jit
  def batch_expert_forward(expert, h_batch):
      return jax.vmap(expert)(h_batch)
  ```
- GPU 메모리 고려: 대규모 배치는 `jax.lax.map` 또는 청크 분할로 OOM 방지

### CLI Convention
- `scripts/` 엔트리포인트는 `typer` + Hydra Compose API를 사용한다
- `@hydra.main()` 데코레이터는 사용하지 않음 (Typer와 충돌)
- 대신 `hydra.compose()` + `initialize_config_dir()`로 config composition
- `typer.Option()`으로 config_dir, config_name, overrides 등 CLI 인자 관리
- 예시:
  ```python
  import typer
  from pathlib import Path
  from hydra import compose, initialize_config_dir
  from omegaconf import DictConfig

  app = typer.Typer()

  @app.command()
  def main(
      config_dir: Path = typer.Option("configs", help="Hydra config 디렉토리"),
      config_name: str = typer.Option("config", help="Config 이름"),
      overrides: list[str] = typer.Option([], "--override", "-o", help="Hydra overrides"),
      output_dir: Path = typer.Option(..., help="출력 디렉토리"),
  ) -> None:
      with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
          cfg: DictConfig = compose(config_name=config_name, overrides=overrides)
      # cfg.split.train_end, cfg.baseline.factors 등으로 접근
      ...

  if __name__ == "__main__":
      app()
  ```

### Tool Config
- **Black**: `line-length=100`, `target-version=py310`
- **Ruff**: `line-length=100`, `select=["E","F","I","N","W"]`, `ignore=["E501"]`
- **MyPy**: `python_version=3.10`, `ignore_missing_imports=true`

---

## Implementation Roadmap

| Phase | Period | Description | Key Deliverables |
|-------|--------|-------------|------------------|
| 0 | Week 1-2 | 데이터 준비 + Baseline | EDA, 데이터 분할, 인기도/UserKNN/BPR-MF 벤치마크 |
| 1 | Week 3-5 | Factual Knowledge 추출 | Per-Item 통합 L1+L2+L3 (GPT-4.1-nano, ~47K 제품) |
| 2 | Week 6-7 | 유저 프로파일 | L1 집계 + L2/L3 Factorization Prompting, Attribute Store |
| 3 | Week 8-10 | 세그멘테이션 & 분석 | 고객/상품 클러스터링, Affinity Matrix, 대시보드 |
| 4 | Week 11-13 | 추천 모델 구현 | KAR 모듈 (Encoder+Expert+Gating+Fusion) + DeepFM + Multi-stage 학습 |
| 5 | Week 14-17 | 체계적 실험 | Fix-and-Vary Ablation + 5종 백본 + Cold-start 분석 |
| 6 | Week 18-19 | 서빙 파이프라인 | 전체 카탈로그 스코어링 서빙 + Latency 프로파일링 |
| 7 | Week 20-22 | 결과 정리 | 전체 실험 정리, 시각화, 논문 집필, 코드 공개 |

### Fix-and-Vary 실험 우선순위

기본 설정: **Full L1+L2+L3** / Frozen BGE / G2 Gating / F2 Fusion / DeepFM / Multi-stage

1. **Layer Ablation** (7 변형): L1 Only → L2 Only → L3 Only → L1+L2 → L1+L3 → L2+L3 → Full
2. **Gating** (4 변형): G1 (Fixed) → G2 (Expert-conditioned) → G3 (Context) → G4 (Cross)
3. **Fusion** (4 변형): F1 (Concat) → F2 (Addition) → F3 (Gated) → F4 (Cross-Attention)
4. **Encoder** (3 변형): Frozen BGE → Fine-tuned BGE → TF-IDF+Projection
5. **Backbone** (5 변형): DeepFM → SASRec → LightGCN → DCNv2 → DIN

### 평가 메트릭

| Metric | Description |
|--------|-------------|
| MAP@12 | Kaggle 원 대회 메트릭 (primary) |
| HR@12 | Hit Rate at 12 |
| NDCG@12 | Normalized Discounted Cumulative Gain |
| MRR | Mean Reciprocal Rank |
