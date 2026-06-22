# Scripts & Modules Tutorial

Phase 0-4 구현에서 생성된 모든 src/ 모듈과 scripts/ 엔트리포인트의 사용법.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Preprocess raw data (CSV → Parquet + temporal split)
python scripts/preprocess.py \
    --raw-dir data/h-and-m-personalized-fashion-recommendations \
    --output-dir data/processed

# 3. Train a baseline model
python scripts/train.py \
    --data-dir data/processed \
    --model-dir results/models \
    --backbone userknn

# 3b. Build features + train neural backbone
python scripts/build_features.py \
    --data-dir data/processed \
    --output-dir data/features
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone deepfm --no-wandb

# 3c. Train DCN-v2 or LightGCN (same features, different architecture)
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone dcnv2 --no-wandb
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone lightgcn --no-wandb

# 3d. Build sequential features + train DIN or SASRec
python scripts/build_features.py \
    --data-dir data/processed \
    --output-dir data/features \
    --build-sequences --max-seq-len 50
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone din --no-wandb
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone sasrec --no-wandb

# 3e. KAR training (Knowledge-Augmented Recommendation)
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone deepfm --use-kar \
    --embeddings-dir data/embeddings \
    --gating g2 --fusion f2 \
    --no-wandb

# 3f. Pre-store expert outputs for serving
python scripts/prestore.py \
    --model-dir results/models \
    --features-dir data/features \
    --embeddings-dir data/embeddings \
    --output-dir data/prestore \
    --backbone deepfm

# 4. Extract factual knowledge (L1+L2+L3)
python scripts/extract_factual_knowledge.py \
    --data-dir data/processed \
    --images-dir data/h-and-m-personalized-fashion-recommendations/images \
    --output-dir data/knowledge/factual \
    --pilot

# 5. Build user profiles (reasoning knowledge)
python scripts/extract_reasoning_knowledge.py \
    --data-dir data/processed \
    --fk-dir data/knowledge/factual \
    --output-dir data/knowledge/reasoning \
    --pilot

# 6. Evaluate predictions
python scripts/evaluate.py \
    --predictions-path results/predictions/userknn_val.json \
    --ground-truth-path data/processed/val_ground_truth.json \
    --output-path results/metrics/userknn_val.json

# 7. Evaluate factual knowledge quality (structural only)
python scripts/eval_factual.py \
    --data-dir data/processed \
    --knowledge-dir data/knowledge/factual \
    --output-dir results/eval/factual \
    --skip-judge

# 8. Evaluate user profile quality (structural only)
python scripts/eval_reasoning.py \
    --data-dir data/processed \
    --profile-dir data/knowledge/reasoning \
    --knowledge-dir data/knowledge/factual \
    --output-dir results/eval/reasoning \
    --skip-judge
```

---

## Data Preprocessing (`scripts/preprocess.py`)

Converts raw H&M CSV files to Parquet and creates temporal train/val/test splits.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--raw-dir` | Path | (required) | Raw CSV directory |
| `--output-dir` | Path | (required) | Output directory for Parquet files |
| `--active-min` | int | 5 | Minimum purchases for active user |
| `--train-end` | str | "2020-06-30" | Train period end date |
| `--val-start` | str | "2020-07-01" | Validation start date |
| `--val-end` | str | "2020-08-31" | Validation end date |
| `--test-start` | str | "2020-09-01" | Test start date |
| `--test-end` | str | "2020-09-07" | Test end date |
| `--eval-horizon-days` | int | 7 | Immediate-next-period eval window length (days after `train_end`) |
| `--build-immediate` / `--no-build-immediate` | bool | True | Build `immediate_ground_truth.json` for `(train_end, train_end+horizon]` |
| `--verbose` | bool | False | Print detailed statistics |

> **Immediate-next-period eval (Kaggle-comparable).** The 2-month `val`/`test` splits leave a ~2-month gap between `train_end` and the eval window, which destroys recency — the dominant H&M signal — and made absolute MAP@12 look ~10x below Kaggle (see `redesign_2026-06.md` §8-9). `--build-immediate` (default on) additionally writes `immediate_ground_truth.json` for the window **`(train_end, train_end + eval_horizon_days]`** (the week right after training, by default Jul 1-7). On this split a trivial repurchase baseline scores MAP@12 ≈ 0.024 (Kaggle-competitive). The existing 2-month splits are untouched (backward compatible).

### Output Files

```
data/processed/
├── articles.parquet              # Cleaned articles (article_id as VARCHAR)
├── customers.parquet             # Cleaned customers (nulls filled)
├── transactions.parquet          # Cleaned transactions (sorted by t_dat)
├── train_transactions.parquet    # Train split
├── val_transactions.parquet      # Validation split (2-month)
├── test_transactions.parquet     # Test split (2-month, +2-month gap)
├── val_ground_truth.json         # {customer_id: [article_ids]} (2-month)
├── test_ground_truth.json        # {customer_id: [article_ids]} (2-month)
├── immediate_ground_truth.json   # {customer_id: [article_ids]} for (train_end, +horizon] — Kaggle-comparable
├── active_customer_ids.json      # Active users (5+ purchases)
└── sparse_customer_ids.json      # Sparse users (1-4 purchases)
```

### Internal Calls

```
scripts/preprocess.py
  → src.data.preprocessing.run_preprocessing(DataPaths)
    → validate_raw_data() — DuckDB validation
    → load_and_convert_{articles,customers,transactions}() — ThreadPool parallel
  → src.data.splitter.run_split(processed_dir, output_dir, SplitConfig, FilterConfig, build_immediate=True)
    → split_transactions_temporal() — DuckDB WHERE on t_dat
    → filter_customers_by_activity() — GROUP BY + COUNT
    → build_ground_truth() — Deduplicated purchase lists (val + test)
    → build_immediate_eval() — GT for (train_end, train_end+eval_horizon_days] window
```

`build_immediate_eval(processed_dir, output_dir, train_end, horizon_days=7)` can also be called directly (independent of full preprocessing) to (re)build `immediate_ground_truth.json` from an existing `val_transactions.parquet`:

```python
from pathlib import Path
from src.data.splitter import build_immediate_eval

build_immediate_eval(
    processed_dir=Path("data/processed"),
    output_dir=Path("data/processed"),
    train_end="2020-06-30",
    horizon_days=7,
)
```

---

## Factual Knowledge Extraction (`scripts/extract_factual_knowledge.py`)

Extracts L1+L2+L3 structured attributes from product descriptions and images using GPT-4.1-nano.
Per-item integrated prompt extracts all three layers in a single API call per product_code.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | "data/processed" | Directory containing articles.parquet |
| `--images-dir` | Path | "data/h-and-m-.../images" | Product image directory |
| `--output-dir` | Path | "data/knowledge/factual" | Output directory |
| `--model` | str | "gpt-4.1-nano" | OpenAI model name |
| `--batch-api` | bool | False | Use Batch API (50% discount, 24h turnaround) |
| `--max-concurrent` | int | 5 | Real-time API concurrent requests |
| `--max-cost` | float | 15.0 | Cost limit in USD |
| `--tpm-limit` | int | 200000 | Tokens-per-minute limit for real-time API |
| `--pilot` | bool | False | Extract pilot sample only (500 products) |
| `--resume` | bool | False | Resume from checkpoint |
| `--batch-id` | str | "" | Poll existing batch ID |
| `--verbose` | bool | False | Verbose logging |

### Example Commands

```bash
# Pilot (500 products, real-time API, quality verification)
python scripts/extract_factual_knowledge.py \
    --data-dir data/processed \
    --images-dir data/h-and-m-personalized-fashion-recommendations/images \
    --output-dir data/knowledge/factual \
    --pilot

# Full batch (~47K products, Batch API, 50% discount)
# Sequential pipeline: splits into ~500-request chunks, submits one at a time
# to stay within org-level enqueued token limit (2M tokens for gpt-4.1-nano)
# Single command runs all: prepare → submit→poll×N → process
python scripts/extract_factual_knowledge.py \
    --data-dir data/processed \
    --images-dir data/h-and-m-personalized-fashion-recommendations/images \
    --output-dir data/knowledge/factual \
    --batch-api \
    --max-cost 15.0

# Resume (auto-detects batch_ids.json, skips completed chunks)
python scripts/extract_factual_knowledge.py \
    --data-dir data/processed \
    --images-dir data/h-and-m-personalized-fashion-recommendations/images \
    --output-dir data/knowledge/factual \
    --batch-api

# Poll single legacy batch ID (backwards compatible)
python scripts/extract_factual_knowledge.py \
    --data-dir data/processed \
    --images-dir data/h-and-m-personalized-fashion-recommendations/images \
    --output-dir data/knowledge/factual \
    --batch-api \
    --batch-id batch_abc123
```

### Output Files

```
data/knowledge/factual/
├── factual_knowledge.parquet    # 105K rows, 22 attribute columns (21 LLM + tone_season rule-based)
│                                #   L1 shared: material, closure, design_details, material_detail
│                                #   L1 specific: 4 per category (canonical slots l1_slot4-7)
│                                #   L2: 7 universal perceptual fields
│                                #   L3 shared: color_harmony, coordination_role, visual_weight, style_lineage
│                                #   L3 specific: 2 per category (canonical slots l3_slot6-7)
│                                #   L3 post-processed: tone_season (COLOR_TO_TONE mapping)
├── extraction_log.jsonl         # Per-call logs (tokens, cost, latency)
├── quality_report.json          # Coverage + validation statistics
├── checkpoint/                  # Resume-friendly checkpoints
│   └── checkpoint.parquet
└── batch/                       # Batch API files
    ├── input_000.jsonl          # Chunked JSONL (multi-chunk, <150MB each)
    ├── input_001.jsonl
    ├── ...
    ├── batch_ids.json           # Multi-batch manifest (batch IDs, resume support)
    ├── output_000.jsonl         # Per-chunk results
    ├── output_001.jsonl
    └── ...
```

### Internal Calls

```
scripts/extract_factual_knowledge.py
  → src.knowledge.factual.extractor.group_by_product_code()
    → garment_group_name → Super-Category routing
    → product_code grouping + representative SKU selection
  → src.knowledge.factual.extractor.extract_pilot() [real-time]
    → _extract_single_product() — GPT-4.1-nano Structured Output
    → src.knowledge.factual.validator.validate_knowledge()
    → src.knowledge.factual.cache.ProductCodeCache.put()
  → src.knowledge.factual.batch.prepare_batch_jsonl_chunked() [batch, multi-chunk]
    → max_requests (default 500) + max_bytes (default 150MB) dual limit
  → src.knowledge.factual.batch.run_batch_pipeline() [sequential submit→poll]
    → submit_batch() → poll_batch() per chunk (one at a time)
    → _save_pipeline_manifest() after each chunk (resume support)
    → load_batch_manifest() for stale manifest detection
  → src.knowledge.factual.batch.parse_batch_results(list[Path])
  → src.knowledge.factual.extractor.correct_visual_weight()
    → silhouette/fit/coordination → visual_weight 범위 교집합 clamp (Apparel only)
  → src.knowledge.factual.extractor.propagate_to_variants()
    → update_color_knowledge() — color-dependent L3 updates
  → src.knowledge.factual.text_composer.construct_factual_text()
```

### Rate Limiting: TPM 초과 방지

#### 문제

OpenAI 실시간 API는 **TPM(Tokens Per Minute) 200K** 한도를 적용한다.
`asyncio.Semaphore`만으로는 동시 요청 수만 제한할 뿐 분당 토큰 처리량을 제어하지 못한다.
동시 5건이라도 응답이 빠르면 burst가 누적되어 429 에러가 발생하고,
429 에러 후 대기 중이던 요청이 동시에 재시도하는 thundering-herd 문제까지 겹친다.

#### 해법 — 2중 제어 아키텍처

| 계층 | 메커니즘 | 역할 |
|------|----------|------|
| **동시성 제한** | `asyncio.Semaphore(max_concurrent)` | 동시 in-flight 요청 수 상한 (기본 5) |
| **처리량 제한** | `TokenRateLimiter(tpm_limit)` | 분당 토큰 소비 상한 (기본 200K) |

두 제어가 독립적으로 작동하며, 요청은 Semaphore와 TokenRateLimiter를 **모두** 통과해야 API를 호출한다.

#### TokenRateLimiter 동작

`TokenRateLimiter`는 sliding-window 방식으로 60초 구간의 토큰 소비를 추적한다.

**3-Phase 흐름:**

1. **acquire** — 현재 window 사용량 + 예상 토큰(러닝 평균, 초기 2500)이 budget을 초과하면,
   충분한 토큰이 window에서 만료될 때까지 자동 sleep
2. **record** — API 응답 수신 후 실제 토큰(`input_tokens + output_tokens`)을 window에 기록,
   러닝 평균 갱신
3. **backpressure** — 429 에러 수신 시 글로벌 pause를 설정하여
   모든 대기 중인 `acquire()` 호출이 해당 기간 동안 sleep (thundering-herd 방지)

#### Data Flow

```
process_product()
  │
  ├── async with semaphore          ← 동시성 제한 (max_concurrent)
  │     │
  │     └── _extract_single_product()
  │           │
  │           ├── rate_limiter.acquire()    ← window 여유 확인, 필요 시 sleep
  │           │
  │           ├── client.responses.create() ← OpenAI API 호출
  │           │
  │           ├── rate_limiter.record()     ← 실제 토큰 사용량 기록
  │           │
  │           └── (on 429)
  │                 └── rate_limiter.backpressure(wait)  ← 글로벌 pause
  │                 └── asyncio.sleep(wait)
```

#### CLI 파라미터 역할 차이

| 파라미터 | 기본값 | 제어 대상 | 설명 |
|----------|--------|-----------|------|
| `--max-concurrent` | 5 | `Semaphore` | 동시에 in-flight 상태인 API 요청 수 상한 |
| `--tpm-limit` | 200000 | `TokenRateLimiter` | 60초 sliding window 내 총 토큰 소비 상한 |

두 파라미터는 독립적으로 조정 가능하다.
Tier가 올라가 TPM이 증가하면 `--tpm-limit`만 높이면 되고,
API 서버 부하를 줄이려면 `--max-concurrent`를 낮추면 된다.

---

## External Knowledge Extraction (`scripts/extract_external_knowledge.py`)

KAR의 open-world(외부 스타일링) 지식 가설을 실현하는 모듈. probe_16(prose) +
probe_19(structured format-unify)에서 검증된 프롬프트를 **단일 호출로 결합**한다.
**PRODUCT_CODE 단위**(article 단위 아님)로 GPT-4.1-nano structured-output을 1회 호출해
(1) `prose` 보완 설명과 (2) `complements` 구조화 속성(product_type / colour / material /
style)을 **동시에** 추출한다. 구조화 결과는 L1 속성 리스트 형식으로 렌더링되어
(`render_external_text`) BGE가 L1과 같은 sub-space에 임베딩하도록 한다(format-unify).
결합 호출로 API 호출량을 절반으로 줄인다(prose/structured 따로 호출 대비).

`product_code` 결과는 해당 product_code의 모든 article SKU로 propagate되어
`external_knowledge_full.parquet`(columns: `article_id, product_code, prose_text,
structured_text`)로 저장된다. 체크포인트(`external_checkpoint.parquet`)는 product_code
단위 resume-safe이다 — 재실행 시 이미 추출된 product_code는 건너뛴다.

**중요**: 이 스크립트 실행은 비용을 발생시킨다(OpenAI API 호출). import 시점에는 API 호출이
없다. `--dry-run`은 articles 그룹핑 + 비용 추정만 수행하고 API를 호출하지 않는다.
제출 전 `--max-cost` 가드가 추정 비용을 초과하면 중단한다.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | `data/processed` | `articles.parquet` 위치 |
| `--output-dir` | Path | `data/knowledge/external` | 출력 디렉토리 |
| `--model` | str | `gpt-4.1-nano` | OpenAI 모델명 |
| `--concurrency` | int | 32 | async 동시 요청 수 (Semaphore) |
| `--max-retries` | int | 4 | 아이템당 retry/backoff 횟수 |
| `--max-cost` | float | 12.0 | 비용 가드 (USD). 추정 초과 시 중단 |
| `--pilot` | int | None | 처음 N개 product_code만 추출 (smoke test) |
| `--limit` | int | None | 이번 실행에서 처리할 product_code 상한 |
| `--dry-run` | flag | False | 그룹핑 + 비용 추정만, **API 호출 없음** |
| `--verbose` | flag | False | DEBUG 로깅 |

### Example Commands

```bash
# Dry-run — 그룹핑 + 비용 추정만 (API 호출 없음, 비용 0)
python scripts/extract_external_knowledge.py \
    --data-dir data/processed \
    --output-dir data/knowledge/external \
    --dry-run

# Pilot — 3개 product_code만 추출 (smoke test, 극소 비용)
python scripts/extract_external_knowledge.py \
    --data-dir data/processed \
    --output-dir data/knowledge/external \
    --pilot 3

# FULL — 전체 ~47K product_code 추출 (비용 가드 $12)
python scripts/extract_external_knowledge.py \
    --data-dir data/processed \
    --output-dir data/knowledge/external \
    --max-cost 12.0
```

### Output Files

| File | Description |
|------|-------------|
| `external_knowledge_full.parquet` | article 단위 결과 (article_id, product_code, prose_text, structured_text) |
| `external_checkpoint.parquet` | product_code 단위 resume-safe 체크포인트 |
| `quality_report.json` | product_code 수, article 수, API 호출 수, 토큰, 비용, prose/structured 커버리지 |

### 비용 추정

GPT-4.1-nano 실시간 가격($0.10/1M input, $0.40/1M output) 기준, 결합 호출당
약 240 input + 220 output 토큰 추정. **~47,224 product_code 전체 ≈ $5.29**
(`estimate_external_cost(47224)`), 기본 가드 $12 이내.

### Internal Calls

```
scripts/extract_external_knowledge.py
  └─ src.knowledge.external.extractor.extract_external_knowledge()  [async]
       ├─ group_representatives()         # product_code별 첫 article을 대표로
       ├─ load_checkpoint()               # resume (이미 추출된 product_code skip)
       ├─ estimate_external_cost()        # 비용 가드 (제출 전)
       ├─ _extract_one() × N              # 결합 호출 (responses.create + json_schema)
       │    └─ prompts.build_messages() / RESPONSE_FORMAT / render_external_text()
       ├─ save_checkpoint()               # 체크포인트 (checkpoint_interval마다)
       └─ propagate_to_articles()         # product_code → 모든 article SKU 전파
```

`src/config.py`의 `ExternalExtractionConfig`(model, concurrency, max_retries,
max_cost_usd, ...)와 `ExternalExtractionResult`(요약 NamedTuple)를 사용한다.

---

## Enrichment v2 — Behavioral Axes (`scripts/build_behavioral_axes.py`)

Catalog enrichment v2의 **행동파생 3축**(LLM/이미지 불필요, API 비용 0)을 전체 카탈로그에서
DuckDB로 계산한다. `behavioral_axes.parquet`(key=`article_id`)에 저장:
`e2_price_tier_actual`(product_group내 가격 quintile T1–T5), `e2_trend_phase_actual`(월별
매출 momentum→Emerging/Rising/Peak/Mature/Declining/Insufficient), `e2_outfit_role`(same-basket
cross-group co-purchase 그래프의 degree·방향·다양성을 product_group에 residualize→Anchor-hub/
Versatile-connector/Complement-addon/Niche-pair/Standalone). DE1 lesson: 이 축들은 LLM이
*관측할 수 없는* 행동 신호라 metadata 재코딩이 아니다.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | `data/processed` | `train_transactions.parquet` + `articles.parquet` |
| `--output-dir` | Path | `data/knowledge/enrichment_v2` | `behavioral_axes.parquet` 출력 |
| `--verbose` | flag | False | DEBUG 로깅 |

```bash
python scripts/build_behavioral_axes.py \
    --data-dir data/processed \
    --output-dir data/knowledge/enrichment_v2
```

`src/features/behavioral_axes.py`의 `compute_price_tier` / `compute_trend_phase` /
`compute_outfit_role` / `build_behavioral_axes`를 호출한다. 전체 105K 카탈로그 ~45초.

---

## Enrichment v2 — Multimodal LLM Axes (`scripts/extract_enrichment_v2.py`)

Catalog enrichment v2의 **LLM 인식축**을 추출하는 **anti-recoding 멀티모달** 파이프라인.
v1 factual 프롬프트가 metadata를 LLM에 주입해 재코딩을 유발한 것(DE1)을 교정 — 이 모듈은
**이미지 + 상품명 + (material/care/fabric-word strip된) detail_desc만** 보여주고 categorical
metadata는 절대 노출하지 않는다(`build_e2_messages`). PRODUCT_CODE 단위 GPT-4.1-nano
structured-output 1회 호출로 occasion(primary/secondary/formality)·fit_intent·body_ease·
care_burden·care_flags·price_look·trend_look을 추출, 모든 SKU로 propagate.

두 개의 sub-command: `build-sample`(층화 pilot 표본 freeze, API 0) + `extract`(추출/`--dry-run`).

### `build-sample` Arguments (no API)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | `data/processed` | 처리된 parquet 위치 |
| `--output-dir` | Path | `data/knowledge/enrichment_v2` | `pilot_sample.csv` + manifest |
| `--n-codes` | int | 500 | 목표 product_code 수 |
| `--floor` | int | 10 | product_code당 최소 train 구매(행동 power floor) |
| `--per-group-floor` | int | 3 | garment_group당 최소 code 수(breadth) |
| `--seed` | int | 42 | tie-break seed |

### `extract` Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | `data/processed` | `articles.parquet` 위치 |
| `--sample-file` | Path | `…/pilot_sample.csv` | freeze된 pilot 표본 |
| `--images-dir` | Path | `data/h-and-m-…/images` | H&M 이미지 디렉토리 |
| `--output-dir` | Path | `data/knowledge/enrichment_v2` | `enrichment_v2_llm.parquet` |
| `--model` | str | `gpt-4.1-nano` | OpenAI 모델명 |
| `--concurrency` | int | 16 | async 동시 요청 |
| `--max-cost` | float | 0.5 | 비용 가드(USD), 추정 초과 시 중단 |
| `--dry-run` | flag | False | 그룹핑 + 비용추정 + **이미지 hit-rate + metadata-leak 검사**, API 0 |

```bash
# 1) 층화 pilot 표본 freeze (API 0)
python scripts/extract_enrichment_v2.py build-sample \
    --data-dir data/processed --output-dir data/knowledge/enrichment_v2

# 2) dry-run — 이미지 hit-rate + leak 검사 (API 0)
python scripts/extract_enrichment_v2.py extract \
    --sample-file data/knowledge/enrichment_v2/pilot_sample.csv --dry-run

# 3) live 멀티모달 추출 (~$0.1, 가드 $0.5)
python scripts/extract_enrichment_v2.py extract \
    --sample-file data/knowledge/enrichment_v2/pilot_sample.csv \
    --images-dir data/h-and-m-personalized-fashion-recommendations/images \
    --max-cost 0.5
```

### Output Files

| File | Description |
|------|-------------|
| `pilot_sample.csv` + `pilot_sample_manifest.json` | freeze된 층화 표본(seed 42, resume-safe) |
| `enrichment_v2_llm.parquet` | article 단위 LLM 축 (article_id, product_code, e2_*) |
| `checkpoint_llm/checkpoint.parquet` | product_code 단위 resume 체크포인트 |
| `quality_report.json` | code/article/call 수, 이미지 사용 수, 토큰, 비용, coverage |

`src/knowledge/enrichment_v2/`(schema·prompts·validator·extractor·sampling) + `src/config.py`의
`EnrichmentV2Config`/`EnrichmentV2Result`를 사용한다. factual의 `cache.py`(ProductCodeCache)·
`image_utils.py`를 재사용한다.

### DE1 v2 re-screen (`witnesses/probe_DE1_v2_new_attributes.py`)

추출 후 **DE1 엔진·임계값을 그대로 재사용**해 새 축의 변별력·비중복·행동신호를 재검정(seed 42,
two-population: 행동축=full 105K STRONG / LLM·gap축=pilot PRELIMINARY). `behavioral_axes.parquet`
+ `enrichment_v2_llm.parquet`를 article_id로 left-join하고 gap축(value_gap/trend_gap)을 계산해
`probe_DE1_v2_result.json`(per-attribute verdict + GO/NO-GO)을 출력한다. LLM parquet 부재 시
행동축만 채점한다(spend-free 조기 read).

```bash
PYTHONPATH=. python -u witnesses/probe_DE1_v2_new_attributes.py
```

> **실행 주의(이 환경):** conda 미설치·system Python. `scripts/`·`witnesses/`는 repo root에서
> `PYTHONPATH=. python …`로 실행한다(미설정 시 `ModuleNotFoundError: src`).

---

## Enrichment v2 — Value Matrix (`scripts/build_enrichment_matrix.py` + `witnesses/probe_E2_value_matrix.py`)

E2-2: DE1-v2를 통과한 **4 결정-축**(trend_phase·outfit_role·value_gap·trend_gap)의 가치를
**4-use value matrix**(①faceted/control·②trend lead-time·③merchandising·④marketing)로 특성화한다.
각 cell = (a) **capability**(metadata 없는 결정 차원인가) + (b) **measured decision-lift**(행동지표로
metadata baseline을 이기나, 유의성 포함)를 **분리** 보고. API $0, seed 42 재현.

### 1) Matrix-ready 테이블 (`scripts/build_enrichment_matrix.py`, no API)

gap축(value_gap/trend_gap)을 영속화(DE1-v2 probe에서 transient 계산하던 것 승격)하고 per-item
sell-through(velocity = total_purchases/lifespan_days, buyer_concentration) + **E2-3 merch 신호**
(`compute_merch_signals`: markdown_depth·first_week_sell_through·online_ratio)를 더해
`matrix_axes.parquet`(key=article_id)로 저장. `src/features/enrichment_matrix.py`의
`compute_value_gap`/`compute_trend_gap`/`compute_sell_through`/`compute_merch_signals`/`build_matrix_table` 호출.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | `data/processed` | `train_transactions.parquet` |
| `--e2-dir` | Path | `data/knowledge/enrichment_v2` | `behavioral_axes.parquet` + `enrichment_v2_llm.parquet` → `matrix_axes.parquet` |

```bash
python scripts/build_enrichment_matrix.py --data-dir data/processed --e2-dir data/knowledge/enrichment_v2
```

### 2) Value-matrix probe (`witnesses/probe_E2_value_matrix.py`, no API)

D3(steering)·D5(`_effective_k`/`_repurchase_rate`)·DE1(`_eta`) 엔진과 신규 `src/features/lead_lag.py`
(월별 attribute-share→sales(t+k) lead-lag + permutation-null + block-bootstrap)를 재사용해 16 cell을
채점하고 `witnesses/probe_E2_result.json` + `results/figures/E2_value_matrix.png`(heatmap)를 출력한다.
`--quick`(lags 1-2, n_boot 200, 2k users)로 빠른 검증. 결과: **E2 GO**(capability 14/16, strong lift
PASS 2/16 = trend_phase→②lead-time·outfit_role→③merch).

```bash
PYTHONPATH=. python -u witnesses/probe_E2_value_matrix.py          # full (8k users, lags 1-4)
PYTHONPATH=. python -u witnesses/probe_E2_value_matrix.py --quick  # fast wiring check
```

### 3) Value-matrix 강화 probe (`witnesses/probe_E2b_value_matrix.py`, E2-3, no API)

E2-2 matrix(lift 2/16)를 *정직하게* 강화 — 임계값 불변, 더 나은 target/granularity/outcome로만.
컬럼당 PRIMARY 1개(나머지 descriptive): ① deployable history-predictor(oracle 대신) · ② weekly+
continuous momentum(secondary; monthly가 primary) · ③ rich outcome(first_week_sell_through·markdown,
product_group residualize) · ④ **buyer-population** divergence(`src/features/audience_signals.py`,
txn⋈customers age/channel). `probe_E2_result.json`은 **건드리지 않고** `probe_E2b_result.json` +
`results/figures/E2b_value_matrix.png`(★=new PASS) 출력 → before/after. 결과: **lift 2→3**(③
`trend_phase`→merch NEW PASS; ①②④는 정직 반증).

```bash
PYTHONPATH=. python -u witnesses/probe_E2b_value_matrix.py          # full
PYTHONPATH=. python -u witnesses/probe_E2b_value_matrix.py --quick  # fast wiring check
```

### 4) User-side value probe (`src/features/user_axes.py` + `witnesses/probe_E2c_user_value.py`, E2-4, no API)

E2-4는 KAR의 두 번째(user/reasoning) leg를 ①control·④audience에 붙여 **2-source × 4-use 분해**를 완성한다.
`src/features/user_axes.py`:
- `build_user_representations(...)` → 공통 `customer_id` 순서로 정렬된 train-derived 유저 표현 3종:
  **reasoning_bge**(`data/embeddings/user_bge_embeddings.npz` 768→PCA-50 + BGE isotropy), **reasoning_fields**
  (`reasoning_json` 9 prose 필드 → TF-IDF/SVD-50 + L1-aggregate numeric), **demographic**(11-d baseline =
  `data/features/user_features.npz` 8 numeric + 3 one-hot, `src/features/engineering.py` 레이아웃).
- `build_future_outcomes(val_txn, articles, immediate_gt, cohort)` → **FUTURE** 라벨(val/immediate만 touch):
  `fut_price_tier`(val 평균가 → **train-frozen** quintile edges), `fut_top_group`(MODE product_group 19),
  `fut_online`/`fut_repurchase`/`fut_bought`/`fut_n_types`.

`witnesses/probe_E2c_user_value.py`는 E2 steer spine·D5 `_cv_predict`/`_paired_sig`/`_effective_k`·
`audience_signals.segment_divergence_weighted`를 재사용해 {reasoning_bge, reasoning_fields} × ①②③④를
채점하고 `witnesses/probe_E2c_user_value.json` + `results/figures/E2c_user_value.png`(2-source×4-use heatmap)를
출력한다. `probe_E2_result.json`/`probe_E2b_result.json` **mtime 불변 assert**. 결과: **KAR-SYMMETRY CONFIRMED**
(④ audience modest PASS 둘 다 — `fut_top_group` Δ+0.0117/+0.0145; ① control NO 둘 다; ②③ N/A-SEMANTICS).

```bash
PYTHONPATH=. python -u witnesses/probe_E2c_user_value.py          # full (n_cohort 40k, perm 1000)
PYTHONPATH=. python -u witnesses/probe_E2c_user_value.py --quick  # fast wiring check (2k, perm 300)
```

### 5) gap축 FUTURE decision-lift probe (`witnesses/probe_E2d_gap_decision.py`, E2-5, no API)

gap축(value_gap/trend_gap)이 **자기 구성축 + product_group 대비** 미래(val 2020-07~08) 결정값을 더하는지
검증. 신규 src 함수 `src/features/enrichment_matrix.py::build_article_future_outcomes(article_ids, data_dir)`
— per-article held-out FUTURE outcome(`compute_sell_through`/`compute_merch_signals`를 val window에 재사용,
train-frozen reference price·`PRAGMA threads=1`로 AVG 결정성). 반환 컬럼: `fut_price_drop`·`fut_markdown_depth`·
`fut_velocity`·`fut_first_week_st`·`fut_momentum_change`·`fut_val_n`·`has_val_sale`(left-join → canonical order,
absent→NaN/0). probe는 2 gap축 × 4 결정(markdown-risk·hidden-gem·overhype/sleeper·survival) × 2 readout
(incremental paired-fold macro-F1 of [one-hot 구성축 + gap] over [one-hot 구성축] · decision-rule precision@flag),
placebo 2종(within-group shuffle·**sign-randomization**)·Ridge ΔR²·partial-corr robustness로 채점. **구성축은
one-hot**(ordinal로 넣으면 gap=c1−c2 collinearity로 Δ≡0 거짓-NO) — D5 `_cv_predict`/`_paired_sig`/`_quantile_bucket`·
`_probe_common.bootstrap_delta` 재사용. `probe_E2/E2b_result.json` **mtime 불변 assert**. 결과: **CLEAN NEGATIVE**
(5/5 cell 0.01 macro-F1 margin 미달, PRELIM 0 → gap = 비중복 *해석* 좌표이나 *예측/결정* 축 아님 확정).
`--repro`로 byte-identical 이중실행 검증. 출력 `witnesses/probe_E2d_gap_decision.json` +
`results/figures/E2d_gap_decision.png`.

```bash
PYTHONPATH=. python3 -u witnesses/probe_E2d_gap_decision.py            # full (n_boot 1000)
PYTHONPATH=. python3 -u witnesses/probe_E2d_gap_decision.py --quick --repro  # fast + byte-identical check
```

### 6) Serendipity/novelty probe (`witnesses/probe_23_serendipity.py`, R-10, no API)

enrichment(L2/L3/external 임베딩)이 *정확도*가 아닌 **serendipity·novelty·long-tail-hit**(redesign §65/§96의 "L3 마지막 rescue 경로")를 향상하나 검증. 정확도(probe_21)·diversity/coverage(probe_02)는 이미 닫힌 negative라 guardrail-context로만 재현. 신규 `witnesses/_probe_common.py` 순수 metric 함수: `item_novelty(topk, pop_prob)`·`longtail_exposure(topk, tail_mask)`·`hit_count`/`tail_hit_count`/`serendipitous_hit_count(topk, gts, canon_ids, ...)` (relevance-grounded). probe는 7 variant(META/L1/L2/L3/L1+L2+L3/external_prose/external_struct)를 `score_variant`로 top-12 retrieval → 6 metric(HR guardrail·div/cov context·novelty·tail-exposure·**S1 long-tail-hit·S2 serendipity**) + **S2b(labeling-symmetric serendipity, fair test)** + placebo(random-12) + 결정-축 hit 특성화. surprise threshold τ는 **L1 baseline서 frozen**(variant가 자기 임계값 못 고름); GT=`immediate_ground_truth`(차주 NEW = discovery-native). `bootstrap_delta` CI, prior probe JSON **mtime 불변 assert**. **결과: CLEAN NEGATIVE (tie at best)** — 5/5 enrichment variant가 S1/S2/S2b 어디서도 L1 못 넘음(matched-HR L1+L2+L3 S2b 동률); **placebo가 novelty 최고지만 serendipitous hit ≈0**(novelty 함정 실증). 출력 `witnesses/probe_23_result.json` + `results/figures/probe_23_serendipity.png`.

```bash
PYTHONPATH=. python3 -u witnesses/probe_23_serendipity.py                  # full (25k users, STRONG)
PYTHONPATH=. python3 -u witnesses/probe_23_serendipity.py --quick --repro  # fast (3k) + byte-identical
```

---

## Merchandising Scenarios (`scripts/serve_scenarios.py`)

value matrix가 닫은 **lift PASS 3 cell**(모두 행동-파생 축)을 머천다이저용 **batch 의사결정-지원 brief**로
제품화하는 product-design build (C 백로그 a). 운영 brief 테이블은 `matrix_axes.parquet` + 거래에서 fresh
계산하되, *confidence 수치*(r·η·CI·verdict)는 canonical `witnesses/probe_E2*.json`에서 **로드**(재계산 안 함)
→ value matrix가 single source of truth. CPU/DuckDB only, **API $0**.

### CLI Arguments

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--scenario` | str | `all` | `all` \| `trend-leadtime` \| `launch-signal` \| `copurchase-velocity` |
| `--matrix-path` | Path | `data/knowledge/enrichment_v2/matrix_axes.parquet` | 입력 매트릭스 |
| `--data-dir` | Path | `data/processed` | `train_transactions.parquet` + `articles.parquet` |
| `--output-dir` | Path | `results/tables/merch_scenarios` | brief 테이블 출력 (parquet + csv) |
| `--top-k` | int | `50` | item-level brief 최대 행 (trend-leadtime은 항상 전체 ~10 카테고리) |
| `--verbose` | flag | off | 상세 로깅 |

### Example Commands

```bash
# 3 시나리오 모두 산출 + 정직한 posture 출력
PYTHONPATH=. python scripts/serve_scenarios.py --scenario all --output-dir results/tables/merch_scenarios

# 단일 시나리오
PYTHONPATH=. python scripts/serve_scenarios.py --scenario trend-leadtime
```

### 3 PASS-cell briefs

- **A. Trend lead-time** (`trend_phase`→②): 카테고리 hot(Emerging+Rising) share z-score로 3개월 수요-상승
  조기경보. 근거 r=**0.472** vs null 0.062 (lag 3mo, CI[0.194,0.640]).
- **B. Launch signal** (`trend_phase`→③): hot cohort(최소구매 10) 예상 first_week_sell_through 스코어카드.
  근거 η=**0.673** (resid product_group) vs metadata 0.223.
- **C. Co-purchase velocity** (`outfit_role`→③): anchor 역할(Anchor-hub·Versatile-connector) velocity 랭킹 +
  번들 역할 라벨. 근거 η=**0.631** (resid product_group) vs metadata 0.534.

`value_matrix_posture()`가 **capability 14/16 vs lift 3/16** 전체를 노출하고, 제품화하지 않는 셀(① automatic
lift·④ audience·gap축·recsys-accuracy negative)을 명시 맥락화한다.

### Internal Calls

`src/serving/merch_scenarios.py`:
- `load_confidence_cards(cfg) → dict[str, ConfidenceCard]` — canonical E2b JSON에서 3 PASS cell의 r/η/CI/verdict 로드.
- `trend_leadtime_brief(cfg, cards=None, top_k=None)` — `lead_lag.monthly_attribute_share` + `lead_lag.lead_lag_corr`
  재사용(lag-3 r=0.4723 deterministic 재현), 카테고리별 hot-share z-score 랭킹.
- `launch_signal_brief(cfg, cards=None, top_k=50, min_purchases=10)` — `matrix_axes`의 `first_week_sell_through`로
  trend_phase별 sell-through 티어 + hot cohort 스코어카드.
- `copurchase_velocity_brief(cfg, cards=None, top_k=50)` — `matrix_axes`의 `velocity`로 outfit_role 티어 + anchor 랭킹.
- `build_all_briefs(cfg, item_top_k=50) → list[ScenarioBrief]`, `value_matrix_posture(cfg) → pd.DataFrame`(attrs:
  capability_yes·lift_pass·pass_cells·recsys_negative).
- NamedTuple: `ScenarioConfig`(paths+seed), `ConfidenceCard`(cell·metric·value·baseline·ci·verdict·best_lag),
  `ScenarioBrief`(name·title·table·confidence·caveat·extra).

노트북 `notebooks/06_merch_scenario.ipynb`(builder `notebooks/builders/build_06_merch_scenario.py`)는 위 엔진을
**호출만** 하는 thin 프레젠테이션 — Part 0 framing(value-matrix heatmap) → A/B/C 시나리오(figure `results/figures/06_*.png`)
→ Part 4 정직한 경계. 테스트 `tests/unit/test_merch_scenarios.py`(8 PASS): canonical 일치 가드 + lead-lag 재현 +
brief well-formedness.

---

## Model Training (`scripts/train.py`)

Trains baseline models or neural backbones (DeepFM, DCN-v2, LightGCN, DIN, SASRec) and saves predictions as JSON.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | (required) | Preprocessed data directory |
| `--model-dir` | Path | (required) | Model artifacts directory |
| `--predictions-dir` | Path | "results/predictions" | Predictions output directory |
| `--backbone` | str | (required) | Model type (see below) |
| `--k` | int | 12 | Number of recommendations per user |
| `--split` | str | "val" | Split to predict on: val or test |
| `--eval-split` | str | None | Eval GT split: val \| test \| immediate (overrides `--split` for baseline GT; `immediate` = Kaggle-comparable next-period window) |
| `--train-end` | str | "2020-06-30" | Train cut-off date (repurchase/recent_popularity recency window) |
| `--recent-days` | int | 14 | Recent-popularity window in days (repurchase/recent_popularity) |
| `--features-dir` | Path | None | Feature directory (required for neural backbones) |
| `--learning-rate` | float | 0.001 | Learning rate |
| `--batch-size` | int | 2048 | Batch size |
| `--max-epochs` | int | 50 | Max training epochs |
| `--patience` | int | 3 | Early stopping patience |
| `--d-embed` | int | 16 | Embedding dimension |
| `--dropout-rate` | float | 0.1 | Dropout rate |
| `--use-id-embed` | bool | False | Add per-user/per-item id embeddings (deepfm/dcnv2 CF capacity; items sharing metadata get distinct scores) |
| `--bce-pos-weight` | float | 1.0 | Positive-class weight in BCE (>1 up-weights positives; counters neg-sampling skew; deepfm/dcnv2) |
| `--no-wandb` | bool | False | Disable W&B logging |
| `--random-seed` | int | 42 | Random seed |
| `--num-workers` | int | 4 | Grain data loader workers |
| `--prefetch-buffer-size` | int | 2 | Batches to prefetch per worker |
| `--val-sample-users` | int | 50000 | Epoch-end validation users (drives early stop; batched scoring) |
| `--midval-sample-users` | int | 5000 | Mid-epoch validation users (cheap signal) |
| `--pred-chunk-users` | int | 32 | Users per batched full-catalog scoring chunk (lower if OOM) |
| `--n-cross-layers` | int | 3 | Number of cross layers (dcnv2) |
| `--n-experts` | int | 4 | Number of MoE experts per cross layer (dcnv2) |
| `--d-low-rank` | int | 64 | Low-rank dimension per expert (dcnv2) |
| `--n-gcn-layers` | int | 3 | Number of GCN propagation layers (lightgcn) |
| `--l2-reg` | float | 0.0001 | L2 regularization on embeddings (lightgcn) |
| `--attention-hidden-dims` | str | "64,32" | Attention MLP hidden dims, comma-separated (din) |
| `--n-heads` | int | 2 | Number of attention heads (sasrec) |
| `--n-blocks` | int | 2 | Number of transformer blocks (sasrec) |
| `--max-seq-len` | int | 50 | Max sequence length (din, sasrec) |
| `--use-kar` | bool | False | Enable KAR knowledge-augmented recommendation |
| `--embeddings-dir` | Path | None | BGE embeddings directory (required if --use-kar) |
| `--gating` | str | "g2" | KAR gating variant: g1\|g2\|g3\|g4 |
| `--fusion` | str | "f2" | KAR fusion variant: f1\|f2\|f3\|f4 |
| `--layer-combo` | str | "L1+L2+L3" | Attribute layer combination |
| `--d-rec` | int | 64 | Expert output dimension |
| `--align-weight` | float | 0.1 | Alignment loss weight |
| `--diversity-weight` | float | 0.01 | Diversity loss weight |
| `--stage1-epochs` | int | 2 | Stage 1 backbone pre-train epochs |
| `--stage2-epochs` | int | 5 | Stage 2 expert adaptor epochs |
| `--stage3-epochs` | int | 3 | Stage 3 end-to-end epochs |
| `--stage3-lr-factor` | float | 0.1 | LR multiplier for stage 3 |

> **검증 scoring (batched, Track B)** — `validate_sample`/`generate_predictions`(및 KAR `generate_predictions_kar`)는 유저당 per-user forward 대신 `--pred-chunk-users`명씩 묶어 전체 카탈로그를 한 번에 스코어링한다(`jax.lax.top_k`). 이전 `batch_size=1` per-user 루프(검증이 학습보다 ~5× 느렸던 병목)를 제거했다. mid-epoch는 `--midval-sample-users`(싼 신호), epoch-end는 `--val-sample-users`(early stopping 기준), 최종 eval만 전체 ground-truth로 1회 수행한다. OOM 시 `--pred-chunk-users`를 낮춘다(feature-based 메모리 ≈ chunk×105K×dims, KAR은 그 ~2배).

> **공정 eval + cohort 리포팅 (FIX C)** — 최종 평가는 `split_eval_cohorts()`로 유저를 `feature_capable`(=`user_to_idx`에 존재, 모델이 실제 스코어링 가능) / `cold_start`(train feature 없음)로 분할하고, **headline 메트릭은 `feature_capable` cohort**로 보고한다. baseline(`scripts/train.py`의 popularity 등)도 `--features-dir`를 주면 **동일한 `feature_capable` 필터**로 headline을 계산하므로 DeepFM과 apples-to-apples 비교가 된다. `{backbone}_metrics.json`에는 `headline`/`cohorts`/`cohort_sizes`/`all_users`가 함께 저장된다(flattened headline 키도 backward-compat 유지).

> **Numerical 정규화 + stats 영속화 (FIX A)** — `run_training`은 z-score 전에 heavy count 컬럼(user: `n_purchases`, `days_since_first_purchase`, `days_since_last_purchase`; item: `total_purchases`, max≈44761≈61σ)에 `np.log1p`를 적용해 극단 tail을 완화한다(컬럼은 `feature_meta` 이름으로 매핑). per-column mean/std/log1p_cols는 `data/features/feature_stats.json`에 저장되어 inference가 동일 변환을 재현한다. in-place 공유로 train/eval scorer가 동일 정규화를 사용한다.

> **Hybrid foundation (repurchase + immediate eval).** `repurchase`/`recent_popularity` baselines + `--eval-split immediate`는 H&M의 지배 신호(repurchase + recency)를 측정 가능한 형태로 복원한다. `src/baselines/repurchase.py`가 단일 source of truth (witnesses/probe_05 로직 그대로):
> - `recent_popularity(train_txn, train_end, days=14, k=12) -> list[str]` — 마지막 `days`일 카운트 top-k (`t_dat`는 datetime.date/datetime64 모두 허용).
> - `repurchase_predict(train_txn, users, k=12, fill_recent=None) -> dict[user->list]` — 유저별 distinct 구매 아이템(reverse-recency) → `fill_recent`로 패딩.
> - `hybrid_predict(train_txn, users, train_end, k=12, recent_days=14)` — `recent_popularity` fill을 계산해 `repurchase_predict`에 위임(동일 predictor).
>
> discovery-quality / cohort 측정은 `src/evaluation/cohorts.py` (per-user AP@k는 `src/evaluation/metrics.evaluate` 재사용, 중복 없음):
> - `activity_cohorts(train_txn) -> dict[user->bracket]` — {new, 1, 2-4, 5-9, 10-19, 20+} by train 구매 수.
> - `evaluate_cohorts(predictions, ground_truth, train_history, k=12) -> dict[bracket->EvalResult]`.
> - `discovery_map(predictions, ground_truth, train_history, k=12) -> EvalResult` — **핵심 신규 메트릭**: GT를 유저 train history에 *없는* NEW 아이템으로 제한해 MAP/HR/NDCG@k 측정. repurchase는 구조상 ~0이며 LLM/content가 메워야 할 discovery gap을 격리한다(new-GT 빈 유저는 스킵).
> - `repurchase_vs_new_decomposition(ground_truth, train_history) -> DecompositionResult(repurchase_frac, new_frac, n_gt_items)`.

### Available Backbones

| Backbone | Description | Key Hyperparameters |
|----------|-------------|---------------------|
| `popularity_global` | Top-K most purchased items (all time) | k |
| `popularity_recent` | Top-K most purchased items (last 7 days, DuckDB) | k, window_days=7 |
| `recent_popularity` | Top-K most purchased items in last `--recent-days` (pandas, hybrid fill list) | k, recent_days=14, train_end |
| `repurchase` | **Hybrid backbone**: per-user own recent items (reverse-recency) padded with recent-popularity. Kaggle-competitive on `--eval-split immediate` (MAP@12 ≈ 0.024) | k, recent_days=14, train_end |
| `userknn` | ALS collaborative filtering (implicit) | factors=128, reg=0.01, iter=15 |
| `bprmf` | BPR matrix factorization (implicit) | factors=128, lr=0.01, iter=100 |
| `deepfm` | DeepFM (FM + DNN, Flax NNX) — Level 1 metadata baseline | d_embed, learning_rate, batch_size, dropout_rate |
| `dcnv2` | DCN-v2 (Deep & Cross Network v2, Flax NNX) — MoE cross layers | d_embed, n_cross_layers, n_experts, d_low_rank, dropout_rate |
| `lightgcn` | LightGCN (Graph Convolution, Flax NNX) — CF graph baseline | d_embed(64), n_gcn_layers, l2_reg |
| `din` | DIN (Deep Interest Network, Flax NNX) — Target-aware attention over purchase history | d_embed, attention_hidden_dims, dropout_rate (requires `--build-sequences`) |
| `sasrec` | SASRec (Self-Attentive Sequential, Flax NNX) — Causal transformer for sequences | d_embed(64), n_heads, n_blocks, max_seq_len, dropout_rate (requires `--build-sequences`) |

### Example Commands

```bash
# Train all four baselines
python scripts/train.py --data-dir data/processed --model-dir results/models --backbone popularity_global
python scripts/train.py --data-dir data/processed --model-dir results/models --backbone popularity_recent
python scripts/train.py --data-dir data/processed --model-dir results/models --backbone userknn
python scripts/train.py --data-dir data/processed --model-dir results/models --backbone bprmf

# Hybrid foundation baselines on the immediate-next-period (Kaggle-comparable) split.
# repurchase ≈ MAP@12 0.024 (reproduces witnesses/probe_05); recent_popularity ≈ 0.003.
python scripts/train.py \
    --data-dir data/processed \
    --model-dir results/models \
    --backbone repurchase \
    --eval-split immediate --train-end 2020-06-30 --recent-days 14
python scripts/train.py \
    --data-dir data/processed \
    --model-dir results/models \
    --backbone recent_popularity \
    --eval-split immediate --train-end 2020-06-30 --recent-days 14

# DeepFM (Level 1: metadata baseline)
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --predictions-dir results/predictions \
    --backbone deepfm \
    --no-wandb

# DeepFM with custom hyperparameters
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone deepfm \
    --learning-rate 0.0005 \
    --d-embed 32 \
    --batch-size 4096

# DeepFM single-process debugging (no Grain workers)
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone deepfm \
    --no-wandb \
    --num-workers 0

# DeepFM with per-user/per-item id embeddings + positive up-weighting
# (id embeds give metadata-identical items distinct scores; pos-weight counters
#  the 4:1 neg-sampling skew). Works for dcnv2 too.
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone deepfm \
    --use-id-embed \
    --bce-pos-weight 4.0

# DCN-v2 (same features as DeepFM, different architecture)
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --predictions-dir results/predictions \
    --backbone dcnv2 \
    --no-wandb

# DCN-v2 with custom cross network hyperparameters
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone dcnv2 \
    --n-cross-layers 4 \
    --n-experts 6 \
    --d-low-rank 32

# LightGCN (graph-based, index-only — no feature lookup)
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --predictions-dir results/predictions \
    --backbone lightgcn \
    --no-wandb

# LightGCN with custom hyperparameters
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone lightgcn \
    --d-embed 128 \
    --n-gcn-layers 4 \
    --l2-reg 0.001

# DIN (requires sequential features: --build-sequences)
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --predictions-dir results/predictions \
    --backbone din \
    --no-wandb

# DIN with custom attention
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone din \
    --attention-hidden-dims "128,64,32"

# SASRec (requires sequential features: --build-sequences)
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --predictions-dir results/predictions \
    --backbone sasrec \
    --no-wandb

# SASRec with custom transformer config
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone sasrec \
    --d-embed 128 \
    --n-heads 4 \
    --n-blocks 3 \
    --max-seq-len 100

# KAR: DeepFM + KAR (3-stage multi-stage training)
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --predictions-dir results/predictions \
    --backbone deepfm --use-kar \
    --embeddings-dir data/embeddings \
    --no-wandb

# KAR: Custom gating/fusion/layer combo
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone deepfm --use-kar \
    --embeddings-dir data/embeddings \
    --gating g4 --fusion f3 \
    --layer-combo "L1+L2" \
    --d-rec 32 \
    --align-weight 0.2 --diversity-weight 0.05

# KAR: LightGCN backbone
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --backbone lightgcn --use-kar \
    --embeddings-dir data/embeddings

# Generate test predictions
python scripts/train.py --data-dir data/processed --model-dir results/models --backbone userknn --split test
```

### Internal Calls (Neural Backbones)

```
scripts/train.py (backbone=deepfm|dcnv2|lightgcn|din|sasrec)
  → src.training.trainer.run_training(backbone_name=...)
    → create_train_state(backbone_name, ...) — model + optimizer init
      → src.models.get_backbone() — BackboneSpec dispatch
      → Feature-based (deepfm/dcnv2): field_dims + n_numerical → model init
      → Graph-based (lightgcn): build_normalized_adj() + n_users/n_items → model init
    → Mesh + NamedSharding — device setup (same code path for 1 or N devices)
    → make_train_step(backbone_name, model_config) — JIT-compiled step factory
      → Feature-based: BCE loss on DeepFMInput
      → LightGCN: BCE + L2 reg on initial embeddings
    → src.training.data_loader.create_train_loader(backbone_name=...)
      → TrainPairsSource — (user_idx, item_idx, label) RandomAccessDataSource
      → Feature-based: FeatureLookupTransform — index → feature lookup
      → Graph-based: IndexOnlyTransform — pass indices only
      → grain.Batch — drop_remainder batching
    → jax.device_put(batch, data_sharding) — numpy → sharded jax.Array
    → step_fn(model, optimizer, batch) — JIT-compiled train step
    → validate_sample(backbone_name=...) — sampled MAP@12 check
    → score_full_catalog(backbone_name=...)
      → Feature-based: user broadcast × all items → predict_proba
      → Graph-based: get_all_embeddings() → u @ I.T dot product
    → generate_predictions() → evaluate() — final metrics
    → _save_model_state() / _load_model_state() — .npz checkpoints

scripts/train.py (--use-kar, backbone=deepfm|dcnv2|lightgcn|din|sasrec)
  → src.training.trainer.run_kar_training(backbone_name=..., kar_config=...)
    → create_kar_train_state() — backbone + KARModel + optimizer init
      → src.kar.hybrid.KARModel(backbone, expert, gating, fusion)
      → src.kar.hybrid.compute_d_backbone() — backbone-specific dim
    → src.kar.embedding_index.build_aligned_embeddings() — BGE .npz aligned to feature indices
    → src.training.data_loader.create_train_loader(use_kar=True, ...)
      → KARFeatureLookupTransform / KARDINLookupTransform / ... — base + h_fact + h_reason
    → Stage 1: make_kar_train_step_stage1() — BCE only (backbone pre-train)
    → Stage 2: make_kar_train_step_stage2() — align + diversity (backbone frozen)
    → Stage 3: make_kar_train_step_stage3() — BCE + align + diversity (all unfrozen, LR×0.1)
    → score_full_catalog_kar() — single user × full catalog scoring
    → generate_predictions() → evaluate() — final metrics
```

---

## Evaluation (`scripts/evaluate.py`)

Computes MAP@12, HR@12, NDCG@12, MRR from predictions and ground truth.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--predictions-path` | Path | (required) | Predictions JSON file |
| `--ground-truth-path` | Path | (required) | Ground truth JSON file |
| `--output-path` | Path | (required) | Output metrics JSON file |
| `--k` | int | 12 | Cutoff K for metrics |

### Metrics

| Metric | Description |
|--------|-------------|
| MAP@12 | Mean Average Precision at 12 (Kaggle official, primary metric) |
| HR@12 | Hit Rate at 12 — fraction of users with at least one hit |
| NDCG@12 | Normalized Discounted Cumulative Gain at 12 |
| MRR | Mean Reciprocal Rank |

---

## src/ Module API Reference

### `src/config.py`

Global NamedTuple definitions:

| NamedTuple | Fields | Usage |
|------------|--------|-------|
| `DataPaths` | raw_dir, processed_dir, {csv filenames} | Path configuration |
| `SplitConfig` | train_end, val_start, val_end, test_start, test_end | Temporal split boundaries |
| `FilterConfig` | active_min, sparse_min | Customer activity thresholds |
| `EvalConfig` | k, metrics | Evaluation settings |
| `BaselineConfig` | als_*, bpr_*, popularity_* | Baseline hyperparameters |
| `PreprocessResult` | paths + row counts | Preprocessing output |
| `SplitResult` | paths + counts + cold-start stats | Split output |
| `EvalResult` | map_at_k, hr_at_k, ndcg_at_k, mrr | Metric results |
| `InteractionData` | matrix (CSR), user/item index mappings | Sparse interaction matrix |
| `ExtractionConfig` | model, use_batch_api, max_concurrent, batch_max_requests(500), ... | LLM extraction settings |
| `ExtractionResult` | output_path, n_products, n_api_calls, cost, coverage | Extraction run summary |
| `ReasoningConfig` | model, use_batch_api, max_concurrent, batch_max_requests(500), min_purchases(5), recent_items_limit(20), l1_time_weight_halflife_days(90), max_cost_usd(120.0), ... | User profiling settings |
| `ReasoningResult` | output_path, n_active_users, n_sparse_users, n_api_calls, cost | Profile run summary |
| `ExternalExtractionConfig` | model(gpt-4.1-nano), concurrency(32), max_retries(4), max_cost_usd(12.0), checkpoint_interval(1000), max_complements(3), detail_desc_chars(200) | External (open-world styling) knowledge extraction settings |
| `ExternalExtractionResult` | output_path, n_product_codes, n_articles, n_api_calls, n_cache_hits, tokens, cost, coverage_prose, coverage_structured | External extraction run summary |
| `FeatureConfig` | neg_sample_ratio(4), reference_date, age_bins, age_labels, random_seed(42), chunk_size | Feature engineering settings |
| `FeatureResult` | output_dir, n_users, n_items, n_train_pairs, feature counts, vocab sizes | Feature engineering output |
| `DeepFMConfig` | d_embed(16), dnn_hidden_dims(256,128,64), dropout_rate(0.1), use_batch_norm | DeepFM hyperparameters |
| `DCNv2Config` | d_embed(16), n_cross_layers(3), n_experts(4), d_low_rank(64), dnn_hidden_dims, dropout_rate(0.1), use_batch_norm | DCN-v2 hyperparameters |
| `LightGCNConfig` | d_embed(64), n_layers(3), dropout_rate(0.0), l2_reg(1e-4) | LightGCN hyperparameters |
| `SequenceConfig` | max_seq_len(50), random_seed(42) | Sequential feature settings |
| `DINConfig` | d_embed(16), attention_hidden_dims(64,32), dnn_hidden_dims(256,128,64), dropout_rate(0.1), use_batch_norm | DIN hyperparameters |
| `SASRecConfig` | d_embed(64), n_heads(2), n_blocks(2), max_seq_len(50), dropout_rate(0.2) | SASRec hyperparameters |
| `TrainConfig` | learning_rate(1e-3), batch_size(2048), max_epochs(50), patience(3), val_every_n_steps, use_wandb, num_workers(4), prefetch_buffer_size(2) | Training loop settings |
| `TrainResult` | model_dir, best_epoch, best metrics, total_train_steps, total_train_time_seconds, n_devices | Training run summary |
| `ExpertConfig` | d_enc(768), d_hidden(256), d_rec(64), n_layers(2), dropout_rate(0.1) | KAR Expert MLP settings |
| `GatingConfig` | variant("g2"), d_context(0) | KAR Gating variant + context dim (G3) |
| `FusionConfig` | variant("f2"), alpha_init(0.1), n_heads(4) | KAR Fusion variant + F4 heads |
| `KARConfig` | expert, gating, fusion, layer_combo, align_weight(0.1), diversity_weight(0.01), stage1/2/3_epochs, stage3_lr_factor(0.1) | Full KAR configuration |

### `src/data/preprocessing.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `validate_raw_data` | `(con, raw_dir) → dict` | Validate CSV files |
| `load_and_convert_articles` | `(con, raw_dir, output_dir) → Path` | articles.csv → Parquet |
| `load_and_convert_customers` | `(con, raw_dir, output_dir) → Path` | customers.csv → Parquet |
| `load_and_convert_transactions` | `(con, raw_dir, output_dir) → Path` | transactions.csv → Parquet |
| `run_preprocessing` | `(DataPaths) → PreprocessResult` | Main entry point |

### `src/data/splitter.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `split_transactions_temporal` | `(con, txn_path, output_dir, SplitConfig) → (Path, Path, Path)` | Temporal split |
| `filter_customers_by_activity` | `(con, train_path, FilterConfig) → (list, list)` | Active/sparse filtering |
| `build_ground_truth` | `(con, txn_path) → dict[str, list[str]]` | Build eval ground truth |
| `compute_split_statistics` | `(con, train, val, test) → dict[str, int]` | Split descriptive stats |
| `run_split` | `(processed_dir, output_dir, SplitConfig, FilterConfig) → SplitResult` | Main entry point |

### `src/evaluation/metrics.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_ap_at_k` | `(predicted, actual, k) → float` | AP@K for single user |
| `compute_map_at_k` | `(predictions, ground_truth, k) → float` | MAP@K aggregate |
| `compute_hr_at_k` | `(predictions, ground_truth, k) → float` | HR@K aggregate |
| `compute_ndcg_at_k` | `(predictions, ground_truth, k) → float` | NDCG@K aggregate |
| `compute_mrr` | `(predictions, ground_truth, k) → float` | MRR aggregate |
| `evaluate` | `(predictions, ground_truth, EvalConfig) → EvalResult` | All metrics (parallel) |
| `evaluate_by_cohort` | `(predictions, ground_truth, cohorts, EvalConfig) → dict[str, EvalResult]` | Per-cohort eval |

### `src/features/`

| Module | Key Functions/Classes |
|--------|----------------------|
| `engineering.py` | `UserFeatures`, `ItemFeatures` (NamedTuples), `compute_user_features()`, `compute_item_features()`, `build_id_maps()`, `generate_train_pairs()`, `run_feature_engineering()` |
| `store.py` | `save_features()`, `load_train_pairs()`, `load_user_features()`, `load_item_features()`, `load_feature_meta()`, `load_id_maps()`, `load_cat_vocab()` |
| `sequences.py` | `build_sequences(data_dir, features_dir, SequenceConfig) → dict`, `load_sequences(features_dir) → dict[str, np.ndarray]` — time-ordered item sequences for DIN/SASRec |

### `src/losses.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `binary_cross_entropy` | `(logits: jax.Array, labels: jax.Array) → jax.Array` | Numerically stable BCE from logits |
| `bpr_loss` | `(pos_scores: jax.Array, neg_scores: jax.Array) → jax.Array` | BPR: -mean(log(sigmoid(pos - neg))) |
| `embedding_l2_reg` | `(user_embeds, item_embeds, weight) → jax.Array` | L2 reg: weight * (‖e_u‖² + ‖e_i‖²) / (2B) |
| `align_loss` | `(e_expert, x_backbone_sg) → jax.Array` | MSE between expert output and stop_gradient(backbone embed) |
| `diversity_loss` | `(e_fact, e_reason) → jax.Array` | Mean cosine similarity (minimize for complementarity) |
| `kar_total_loss` | `(logits, labels, e_fact, x_item_sg, e_reason, x_user_sg, ...) → (total, dict)` | Combined BCE + align + diversity (stage-aware) |

### `src/models/__init__.py`

| Class/Function | Description |
|----------------|-------------|
| `BackboneSpec` | NamedTuple: model_cls, input_cls, config_cls, needs_graph, needs_sequence |
| `BACKBONE_REGISTRY` | dict mapping backbone name → BackboneSpec |
| `get_backbone` | `(name: str) → BackboneSpec` — lookup with validation |
| `is_kar_model` | `(model) → bool` — check if model is KARModel |

### `src/models/deepfm.py`

| Class/Function | Description |
|----------------|-------------|
| `DeepFMInput` | NamedTuple: user_cat, user_num, item_cat, item_num |
| `DeepFM` | Flax NNX Module: FM + DNN, `__call__` → logits (B,), `predict_proba` → sigmoid (B,), `embed()` → (stacked, first_order), `predict_from_embedding()` → logits |

### `src/models/dcnv2.py`

| Class/Function | Description |
|----------------|-------------|
| `CrossLayerV2` | Flax NNX Module: MoE low-rank cross layer, x_{l+1} = x0 ⊙ (MoE(x_l) + b) + x_l |
| `DCNv2` | Flax NNX Module: Cross Network v2 + DNN, reuses `DeepFMInput`, `__call__` → logits (B,), `embed()` → stacked, `predict_from_embedding()` → logits |

### `src/models/lightgcn.py`

| Class/Function | Description |
|----------------|-------------|
| `LightGCNInput` | NamedTuple: user_idx, item_idx (index-only, no features) |
| `LightGCN` | Flax NNX Module: K-layer graph propagation, `__call__` → dot-product logits (B,), `get_all_embeddings()` → (user_emb, item_emb), `embed()` → (u_emb, i_emb), `predict_from_embedding()` → logits |
| `build_normalized_adj` | `(user_idx, item_idx, n_users, n_items) → BCOO` — D^{-1/2}AD^{-1/2} sparse adjacency |

### `src/models/din.py`

| Class/Function | Description |
|----------------|-------------|
| `DINInput` | NamedTuple: user_cat, user_num, item_cat, item_num, history, hist_len |
| `DIN` | Flax NNX Module: MLP attention over purchase history + static features + DNN, `__call__` → logits (B,), `predict_proba` → sigmoid (B,), `get_attention_weights` → (B, T), `embed()` → (user_interest, target_query, static_flat), `predict_from_embedding()` → logits |

### `src/models/sasrec.py`

| Class/Function | Description |
|----------------|-------------|
| `SASRecInput` | NamedTuple: history, hist_len (sequence-only, no static features) |
| `TransformerBlock` | Flax NNX Module: Causal self-attention + FFN with LayerNorm + residual |
| `SASRec` | Flax NNX Module: N transformer blocks, `__call__(x, target) → logits (B,)`, `get_user_embedding` → (B, d), `score_all_items` → (B, n_items+1), `embed()` → (user_emb, target_emb), `predict_from_embedding()` → logits |

### `src/training/data_loader.py`

| Class/Function | Signature | Description |
|----------------|-----------|-------------|
| `TrainPairsSource` | `(features_dir: Path)` | RandomAccessDataSource: `__getitem__` → {user_idx, item_idx, label} |
| `FeatureLookupTransform` | `(user_features, item_features)` | grain.MapTransform: index → feature lookup (DeepFM, DCNv2) |
| `IndexOnlyTransform` | `()` | grain.MapTransform: pass indices + label only (LightGCN) |
| `DINLookupTransform` | `(user_features, item_features, sequences, seq_lengths)` | grain.MapTransform: static features + sequence lookup (DIN) |
| `SASRecTransform` | `(sequences, seq_lengths)` | grain.MapTransform: sequence + target item index (SASRec) |
| `KARFeatureLookupTransform` | `(user_features, item_features, item_emb, user_emb)` | grain.MapTransform: base features + h_fact + h_reason (DeepFM, DCNv2) |
| `KARIndexTransform` | `(item_emb, user_emb)` | grain.MapTransform: indices + h_fact + h_reason (LightGCN) |
| `KARDINLookupTransform` | `(user_features, item_features, sequences, seq_lengths, item_emb, user_emb)` | grain.MapTransform: DIN features + h_fact + h_reason |
| `KARSASRecTransform` | `(sequences, seq_lengths, item_emb, user_emb)` | grain.MapTransform: SASRec features + h_fact + h_reason |
| `create_train_loader` | `(features_dir, batch_size, seed, ..., backbone_name="deepfm", use_kar=False, item_embeddings=None, user_embeddings=None) → DataLoader` | Grain DataLoader factory (per-epoch, backbone-aware, KAR-aware) |
| `steps_per_epoch` | `(features_dir, batch_size) → int` | Pre-compute steps (grain DataLoader has no `__len__`) |

### `src/training/trainer.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `create_train_state` | `(backbone_name, model_config, TrainConfig, feature_meta, features_dir?) → (Module, Optimizer)` | Multi-backbone model + optimizer init |
| `make_train_step` | `(backbone_name, model_config) → JIT-compiled step fn` | Factory: returns backbone-specific train step |
| `train_step` | `(model, optimizer, batch: dict) → loss` | Backward-compatible feature-based train step |
| `score_full_catalog` | `(model, user_idx, ..., backbone_name="deepfm") → list[int]` | Single user × full catalog (feature or graph dispatch) |
| `generate_predictions` | `(model, target_users, ..., backbone_name="deepfm") → dict[str, list[str]]` | Batch prediction generation |
| `validate_sample` | `(model, ..., backbone_name="deepfm") → dict[str, float]` | Quick validation on sampled users |
| `run_training` | `(model_config, TrainConfig, features_dir, data_dir, model_dir, predictions_dir, split, backbone_name) → TrainResult` | Full multi-backbone training pipeline |
| `create_kar_train_state` | `(backbone_name, model_config, kar_config, train_config, feature_meta, features_dir) → (KARModel, Optimizer)` | KARModel + optimizer init |
| `make_kar_train_step_stage1` | `(backbone_name) → JIT-compiled step fn` | Stage 1: BCE only |
| `make_kar_train_step_stage2` | `(backbone_name, align_w, div_w) → JIT-compiled step fn` | Stage 2: align + diversity, backbone frozen |
| `make_kar_train_step_stage3` | `(backbone_name, align_w, div_w) → JIT-compiled step fn` | Stage 3: BCE + align + diversity, all unfrozen |
| `score_full_catalog_kar` | `(model, user_idx, ..., backbone_name) → list[int]` | Single user × full catalog KAR scoring |
| `run_kar_training` | `(backbone_name, model_config, kar_config, train_config, ...) → TrainResult` | Full 3-stage KAR training pipeline |

### `src/kar/`

| Module | Key Classes/Functions |
|--------|----------------------|
| `expert.py` | `Expert(config: ExpertConfig, *, rngs)` — 2-layer ReLU MLP with dropout: (B, 768) → (B, d_rec). Same class for factual + reasoning (independent params) |
| `gating.py` | `G1FixedGating` (learnable scalar), `G2ExpertGating` (Softmax(W·[e_f;e_r]), default), `G3ContextGating` (+context), `G4CrossGating` (element-wise), `create_gating(config, d_rec, *, rngs)` factory |
| `fusion.py` | `F1ConcatFusion` (dim doubles), `F2AdditionFusion` (x + α·proj(e), default), `F3GatedFusion` (sigmoid gate), `F4CrossAttentionFusion` (multi-head + residual), `create_fusion(config, d_backbone, d_rec, *, rngs)` factory |
| `hybrid.py` | `KARInput` (NamedTuple: base_input, h_fact, h_reason, context, target_item_idx), `KARModel` (composition: backbone + experts + gating + fusion), `compute_d_backbone(backbone_name, backbone)` |
| `embedding_index.py` | `build_aligned_embeddings(features_dir, embeddings_dir) → (np.ndarray, np.ndarray)` — feature-index-aligned (n_items, 768) + (n_users, 768) |

### `src/serving/prestore.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_prestore` | `(model, item_emb, user_emb, output_dir, batch_size) → (Path, Path)` | Pre-compute expert outputs as .npz |
| `_batch_expert_forward` | `(expert, embeddings, batch_size) → np.ndarray` | Batched expert MLP forward |
| `load_prestore` | `(output_dir) → (np.ndarray, np.ndarray)` | Load pre-computed expert outputs |

### `src/knowledge/factual/`

| Module | Key Functions/Classes |
|--------|----------------------|
| `prompts.py` | `SUPER_CATEGORY_MAP`, `SCHEMA_MAP`, `MATERIAL_VALUES_*`, `CLOSURE_VALUES_*`, `STYLE_LINEAGE_VALUES`, `build_user_message()`, `get_prompt_and_schema()`, `map_to_canonical_slots()`. Schema: 21 LLM fields (tone_season excluded, material_detail added). |
| `extractor.py` | `group_by_product_code()`, `extract_pilot()`, `propagate_to_variants()`, `update_color_knowledge()`, `correct_visual_weight()` |
| `batch.py` | `prepare_batch_jsonl()`, `prepare_batch_jsonl_chunked(max_requests=)`, `submit_batch()`, `submit_multi_batch()`, `poll_batch()`, `poll_multi_batch()`, `run_batch_pipeline()`, `parse_batch_results(Path\|list[Path])`, `load_batch_manifest()` |
| `cache.py` | `ProductCodeCache` (get, put, save_checkpoint, load_checkpoint) |
| `validator.py` | `validate_knowledge()` (21 LLM fields), `validate_final_knowledge()` (22 fields w/ tone_season), `validate_domain_consistency()` → `list[DomainViolation]` (12 cross-attribute rules). Array fields accept `list`/`np.ndarray`. |
| `text_composer.py` | `construct_factual_text()`, `build_all_ablation_texts()`, `LAYER_COMBOS` |
| `image_utils.py` | `find_article_image()`, `load_and_encode_image()`, `get_image_for_article()` |

### `src/baselines/`

| Module | Key Functions |
|--------|---------------|
| `utils.py` | `build_interaction_matrix(con, train_path) → InteractionData` |
| `utils.py` | `predict_from_implicit_model(model, idata, user_ids, k) → dict` |
| `popularity.py` | `compute_global_popularity(con, train_path, k) → list[str]` |
| `popularity.py` | `compute_recent_popularity(con, train_path, k, window_days) → list[str]` |
| `popularity.py` | `predict_popularity(popular_items, user_ids) → dict` |
| `userknn.py` | `train_als(interaction_data, BaselineConfig) → AlternatingLeastSquares` |
| `bprmf.py` | `train_bpr(interaction_data, BaselineConfig) → BayesianPersonalizedRanking` |

### `src/knowledge/reasoning/`

| Module | Key Functions/Classes |
|--------|----------------------|
| `extractor.py` | `aggregate_l1_profiles()` (DuckDB bulk, exp decay), `get_recent_items_batch()`, `compute_l3_distributions_batch()`, `build_sparse_user_profiles()`, `compose_sparse_reasoning_text()` |
| `prompts.py` | `SYSTEM_PROMPT` (price vs quality 구분 규칙 포함), `REASONING_SCHEMA` (9-field JSON, quality_price_tendency에 price quintile 우선 명시), `_L3_SLOT_DISPLAY` (Slot6/Slot7→의미 레이블 매핑), `build_reasoning_user_message()` (semantic slot labels 사용), `compose_reasoning_text()`, `build_reasoning_request_line()` |
| `batch.py` | `prepare_reasoning_batch_jsonl_chunked()` (reuses factual batch infra) |
| `cache.py` | `CustomerCache` (get, put, save_checkpoint, load_checkpoint) |

---

## User Profiling (`scripts/extract_reasoning_knowledge.py`)

Builds per-user reasoning_text for the KAR Reasoning Expert. Active users (5+) get LLM profiling, sparse users (1-4) get template-based profiles.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | "data/processed" | Directory with train_transactions.parquet, articles.parquet, customer IDs |
| `--fk-dir` | Path | "data/knowledge/factual" | Directory with factual_knowledge.parquet |
| `--output-dir` | Path | "data/knowledge/reasoning" | Output directory |
| `--model` | str | "gpt-4.1-nano" | OpenAI model name |
| `--batch-api` | bool | False | Use Batch API (50% discount, 24h) |
| `--max-cost` | float | 120.0 | Cost limit in USD |
| `--min-purchases` | int | 5 | Min purchases for active user |
| `--pilot` | bool | False | Pilot mode (200 users, real-time API) |
| `--resume` | bool | False | Resume from checkpoint |
| `--retry-failed` | bool | False | Retry failed batch responses + template fallback + assemble |
| `--verbose` | bool | False | Verbose logging |

### Example Commands

```bash
# Pilot (200 users, real-time API, quality verification)
python scripts/extract_reasoning_knowledge.py \
    --data-dir data/processed \
    --fk-dir data/knowledge/factual \
    --output-dir data/knowledge/reasoning \
    --pilot

# Full batch (876K active users, Batch API, 50% discount)
python scripts/extract_reasoning_knowledge.py \
    --data-dir data/processed \
    --fk-dir data/knowledge/factual \
    --output-dir data/knowledge/reasoning \
    --batch-api \
    --max-cost 120

# Resume interrupted batch
# - Fast-path: skips L1/input prep if all chunks exist
# - Processes already-downloaded output_*.jsonl → checkpoint immediately
# - Handles KeyboardInterrupt gracefully (saves progress on Ctrl+C)
# - Deduplicates: only processes newly-downloaded outputs after pipeline
python scripts/extract_reasoning_knowledge.py \
    --data-dir data/processed \
    --fk-dir data/knowledge/factual \
    --output-dir data/knowledge/reasoning \
    --batch-api \
    --resume

# Retry failed batch results + assemble final output
# - Loads checkpoint, identifies uncached active users (~2,845 failed)
# - Prepares LLM input for failed users only → batch/retry/ subdirectory
# - Submits retry batch via Batch API
# - Template fallback for still-failed users (profile_source="template_fallback")
# - Builds sparse profiles + assembles user_profiles.parquet & reasoning_texts.parquet
# - Mutually exclusive with --pilot and --resume
python scripts/extract_reasoning_knowledge.py \
    --data-dir data/processed \
    --fk-dir data/knowledge/factual \
    --output-dir data/knowledge/reasoning \
    --batch-api \
    --retry-failed
```

### Output Files

```
data/knowledge/reasoning/
├── user_profiles.parquet       # Full profiles: L1 stats + reasoning_text (all users)
│                                #   customer_id, n_purchases, is_active
│                                #   top_categories_json, top_colors_json, top_materials_json
│                                #   avg_price_quintile, online_ratio, category_diversity
│                                #   reasoning_json (9-field structured), reasoning_text
│                                #   profile_source ("llm" / "template")
├── reasoning_texts.parquet     # KAR input: customer_id → reasoning_text
├── quality_report.json         # Coverage + statistics
├── checkpoint/                 # Resume-friendly checkpoints
│   └── checkpoint.parquet
└── batch/                      # Batch API files
    ├── input_000.jsonl         # Chunked JSONL
    ├── batch_ids.json          # Manifest (resume support)
    ├── output_000.jsonl        # Per-chunk results
    ├── retry/                  # Retry batch files (--retry-failed)
    │   ├── input_000.jsonl
    │   ├── output_000.jsonl
    │   └── ...
    └── ...
```

### Internal Calls

```
scripts/extract_reasoning_knowledge.py
  Stage A: L1 Aggregation (all users, DuckDB, no LLM)
    → src.knowledge.reasoning.extractor.aggregate_l1_profiles()
      → DuckDB JOIN txn × articles × fk, GROUP BY customer_id
      → Exponential decay weighting (halflife=90 days)
      → Category/color/material distributions + diversity score
  Stage B: LLM Reasoning (active users, GPT-4.1-nano)
    → src.knowledge.reasoning.extractor.get_recent_items_batch() — recent 20 items with L2
    → src.knowledge.reasoning.extractor.compute_l3_distributions_batch() — L3 distributions
    → src.knowledge.reasoning.prompts.build_reasoning_user_message() — 3-section prompt
    → src.knowledge.reasoning.batch.prepare_reasoning_batch_jsonl_chunked() — JSONL prep
    → src.knowledge.factual.batch.run_batch_pipeline() — sequential submit→poll (reuse)
    → src.knowledge.factual.batch.parse_batch_results() — parse results (reuse)
    → src.knowledge.reasoning.prompts.compose_reasoning_text() — JSON → natural language
    → src.knowledge.reasoning.cache.CustomerCache — checkpoint/resume
  Stage C: Sparse Fallback (1-4 purchases, template, no LLM)
    → src.knowledge.reasoning.extractor.build_sparse_user_profiles()
    → src.knowledge.reasoning.extractor.compose_sparse_reasoning_text()
  Assembly: merge L1 stats + reasoning → user_profiles.parquet + reasoning_texts.parquet
  Retry (--retry-failed):
    → _collect_failed_ids() — active IDs not in checkpoint cache
    → _prepare_user_data() — recent items + L3 distributions (failed users only)
    → prepare_reasoning_batch_jsonl_chunked() → batch/retry/ subdirectory
    → run_batch_pipeline() → submit + poll retry batch
    → _apply_template_fallback() — build_sparse_user_profiles() for still-failed users
    → Stage C + Assembly (same as normal flow)
```

---

## Factual Knowledge Evaluation (`scripts/eval_factual.py`)

Runs structural checks and optionally LLM-as-Judge on extracted factual knowledge.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | (required) | Processed data directory (articles.parquet) |
| `--knowledge-dir` | Path | (required) | Factual knowledge directory (factual_knowledge.parquet) |
| `--images-dir` | Path | None | Product images directory (for multimodal judge) |
| `--output-dir` | Path | (required) | Output directory for evaluation report |
| `--sample-size` | int | 50 | Number of items for LLM-as-Judge |
| `--judge-model` | str | "gpt-4.1-mini" | LLM model for judge evaluation |
| `--skip-judge` | bool | False | Skip LLM-as-Judge (structural only) |
| `--verbose` | bool | False | Verbose logging |

### Example Commands

```bash
# Structural only (no LLM cost)
python scripts/eval_factual.py \
    --data-dir data/processed \
    --knowledge-dir data/knowledge/factual \
    --output-dir results/eval/factual \
    --skip-judge

# Full evaluation with LLM-as-Judge (multimodal)
python scripts/eval_factual.py \
    --data-dir data/processed \
    --knowledge-dir data/knowledge/factual \
    --images-dir data/h-and-m-personalized-fashion-recommendations/images \
    --output-dir results/eval/factual \
    --sample-size 50
```

### Output Files

```
results/eval/factual/
└── factual_eval_report.json    # Combined structural + judge results
```

### Internal Calls

```
scripts/eval_factual.py
  → src.eval_prompt.structural.compute_coverage()
  → src.eval_prompt.structural.run_schema_checks()
    → src.knowledge.factual.validator.validate_final_knowledge()
  → src.eval_prompt.structural.run_domain_checks()
    → src.knowledge.factual.validator.validate_domain_consistency()
  → src.eval_prompt.structural.compute_distributions()
  → src.eval_prompt.structural.check_token_budget()
  → src.eval_prompt.factual.run_factual_judge() [optional]
    → src.eval_prompt.judge.evaluate_batch()
  → src.eval_prompt.report.save_eval_report()
  → src.eval_prompt.report.build_go_no_go() + print_go_no_go()
```

---

## User Profile Evaluation (`scripts/eval_reasoning.py`)

Runs structural checks and optionally LLM-as-Judge on generated user profiles.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | (required) | Processed data directory (transactions.parquet) |
| `--profile-dir` | Path | (required) | Reasoning knowledge directory (user_profiles.parquet) |
| `--knowledge-dir` | Path | (required) | Factual knowledge directory |
| `--output-dir` | Path | (required) | Output directory for evaluation report |
| `--sample-size` | int | 50 | Number of profiles for LLM-as-Judge |
| `--judge-model` | str | "gpt-4.1-mini" | LLM model for judge evaluation |
| `--skip-judge` | bool | False | Skip LLM-as-Judge (structural only) |
| `--verbose` | bool | False | Verbose logging |

### Example Commands

```bash
# Structural only (no LLM cost)
python scripts/eval_reasoning.py \
    --data-dir data/processed \
    --profile-dir data/knowledge/reasoning \
    --knowledge-dir data/knowledge/factual \
    --output-dir results/eval/reasoning \
    --skip-judge

# Full evaluation with LLM-as-Judge
python scripts/eval_reasoning.py \
    --data-dir data/processed \
    --profile-dir data/knowledge/reasoning \
    --knowledge-dir data/knowledge/factual \
    --output-dir results/eval/reasoning \
    --sample-size 50
```

### Output Files

```
results/eval/reasoning/
└── reasoning_eval_report.json    # Combined structural + judge results
```

### Internal Calls

```
scripts/eval_reasoning.py
  → src.eval_prompt.structural.compute_coverage()
  → src.eval_prompt.structural.check_completeness()
  → src.eval_prompt.structural.check_discriminability()
  → src.eval_prompt.structural.check_token_budget()
  → src.eval_prompt.reasoning.run_reasoning_judge() [optional]
    → src.knowledge.reasoning.extractor.get_recent_items_batch()
    → src.knowledge.reasoning.extractor.compute_l3_distributions_batch()
    → src.eval_prompt.judge.evaluate_batch()
  → src.eval_prompt.report.save_eval_report()
  → src.eval_prompt.report.build_go_no_go() + print_go_no_go()
```

---

## `src/eval_prompt/` Module API Reference

| Module | Key Functions/Classes |
|--------|----------------------|
| `judge.py` | `JudgeConfig`, `JudgeDimension`, `JudgeResult`, `JudgeReport`, `DIMENSION_NAMES`, `build_judge_schema()`, `evaluate_batch()`, `build_judge_system_prompt()` |
| `structural.py` | `compute_coverage()`, `check_token_budget()`, `run_schema_checks()`, `run_domain_checks()`, `compute_distributions()`, `check_completeness()`, `check_discriminability()` |
| `factual.py` | `FACTUAL_DIMENSIONS`, `FactualEvalConfig`, `FactualEvalReport`, `build_factual_judge_message()`, `run_factual_judge()`, `run_factual_eval()` |
| `reasoning.py` | `REASONING_DIMENSIONS`, `ReasoningEvalConfig`, `ReasoningEvalReport`, `build_reasoning_judge_message()`, `run_reasoning_judge()`, `run_reasoning_eval()` |
| `report.py` | `save_eval_report()`, `build_go_no_go()`, `print_go_no_go()`, `FACTUAL_CRITERIA`, `REASONING_CRITERIA` |

---

## Analysis Notebooks

Notebooks load pre-generated JSON reports from evaluation scripts and provide
visualizations, tables, and interpretive markdown. They do **not** run
structural checks or LLM-as-Judge directly.

### `notebooks/01_factual_eval.ipynb`

Analyzes `results/eval/factual/factual_eval_report.json` produced by
`scripts/eval_factual.py`.

| Section | Content |
|---------|---------|
| 1. Setup & Prerequisites | Boilerplate, prerequisite script commands, JSON load |
| 2. Structural Analysis | Coverage barh, schema valid/invalid, domain violation table + pie, enum distributions (entropy table + top-6 barh subplots), token budget table |
| 3. LLM-as-Judge | Per-dimension mean barh, per-item heatmap, low-scorers table, score boxplot |
| 4. Go/No-Go Summary | Threshold comparison table, overall GO/NO-GO verdict |

**Regenerate:** `conda run -n llm-factor-recsys-hnm python notebooks/builders/build_01_factual_eval.py`

### `notebooks/02_reasoning_eval.ipynb`

Analyzes `results/eval/reasoning/reasoning_eval_report.json` produced by
`scripts/eval_reasoning.py`.

| Section | Content |
|---------|---------|
| 1. Setup & Prerequisites | Boilerplate, prerequisite script commands, JSON load |
| 2. Structural Analysis | Coverage printout, completeness 9-field barh + generic/short counts, discriminability stats + per-field unique ratio, token budget table |
| 3. LLM-as-Judge | Per-dimension mean barh, per-item heatmap, low-scorers table, score boxplot |
| 4. Go/No-Go Summary | Threshold comparison table, overall GO/NO-GO verdict |

**Regenerate:** `conda run -n llm-factor-recsys-hnm python notebooks/builders/build_02_reasoning_eval.py`

---

## Feature Engineering (`scripts/build_features.py`)

Builds user/item features from preprocessed data using DuckDB aggregation (train split only, no data leakage). Generates negative samples for model training.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | "data/processed" | Preprocessed data directory |
| `--output-dir` | Path | "data/features" | Output directory for feature matrices |
| `--neg-sample-ratio` | int | 4 | Negative samples per positive |
| `--neg-strategy` | str | "uniform" | Negative sampling distribution: `uniform` \| `popularity` (∝ item train popularity) \| `mixed` |
| `--neg-mixed-pop-frac` | float | 0.5 | Fraction of popularity-proportional negatives when `--neg-strategy=mixed` |
| `--random-seed` | int | 42 | Random seed for negative sampling |
| `--verbose` | bool | False | Verbose logging |
| `--build-sequences` | bool | False | Build sequential features for DIN/SASRec |
| `--max-seq-len` | int | 50 | Max sequence length (requires `--build-sequences`) |

### Feature Specification

| User Numerical (8) | Source |
|---|---|
| n_purchases, avg_price, price_std | train_transactions |
| n_unique_categories, n_unique_colors | train_txn JOIN articles |
| days_since_first_purchase, days_since_last_purchase | train_txn (ref: 2020-06-30) |
| online_purchase_ratio | train_txn (sales_channel_id=2) |

| User Categorical (3) | Source |
|---|---|
| age_group (7 bins) | customers.parquet |
| club_member_status (4 values) | customers.parquet |
| fashion_news_frequency (4 values) | customers.parquet |

| Item Numerical (2) | Source |
|---|---|
| total_purchases, avg_price | train_txn COUNT/AVG |

| Item Categorical (5) | Source |
|---|---|
| product_type_name (~131), colour_group_name (~50), garment_group_name (~21), section_name (~56), index_name (~10) | articles.parquet |

### Example Commands

```bash
# Build features (only needs preprocessed data, no knowledge dependencies)
python scripts/build_features.py \
    --data-dir data/processed \
    --output-dir data/features

# Custom negative sampling ratio
python scripts/build_features.py \
    --data-dir data/processed \
    --output-dir data/features \
    --neg-sample-ratio 2

# Popularity-aware (hard) negatives — sample negatives ∝ item train popularity
python scripts/build_features.py \
    --data-dir data/processed \
    --output-dir data/features \
    --neg-strategy popularity

# Mixed: 50% popularity-proportional + 50% uniform negatives
python scripts/build_features.py \
    --data-dir data/processed \
    --output-dir data/features \
    --neg-strategy mixed --neg-mixed-pop-frac 0.5

# Build features + sequential features for DIN/SASRec
python scripts/build_features.py \
    --data-dir data/processed \
    --output-dir data/features \
    --build-sequences --max-seq-len 50
```

### Output Files

```
data/features/
├── train_pairs.npz       # user_idx, item_idx, labels (int32, int32, float32)
├── user_features.npz     # numerical (n_users, 8), categorical (n_users, 3)
├── item_features.npz     # numerical (n_items, 2), categorical (n_items, 5)
├── feature_meta.json     # Feature names, vocab sizes, counts
├── id_maps.json          # user↔idx, item↔idx bidirectional
├── cat_vocab.json        # Categorical vocabulary dictionaries
└── train_sequences.npz   # sequences (n_users, max_seq_len) int32, seq_lengths (n_users,) int32 [if --build-sequences]
```

### Internal Calls

```
scripts/build_features.py
  → src.features.engineering.run_feature_engineering()
    → compute_user_features() — DuckDB aggregation (train split only)
    → compute_item_features() — Full catalog (all articles)
    → build_id_maps() — Bidirectional ID ↔ index
    → generate_train_pairs() — Positive + negative sampling
  → src.features.store.save_features() — .npz + .json output
```

---

## Pre-store (`scripts/prestore.py`)

Pre-computes Expert MLP outputs for all items and users, saving as `.npz` for fast inference without re-running Expert MLPs at serving time.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-dir` | Path | (required) | Directory with trained KAR model |
| `--features-dir` | Path | (required) | Feature directory (for id maps) |
| `--embeddings-dir` | Path | (required) | BGE embeddings directory |
| `--output-dir` | Path | (required) | Output directory for prestore .npz |
| `--backbone` | str | "deepfm" | Backbone model name |
| `--batch-size` | int | 4096 | Batch size for expert forward |

### Example Commands

```bash
# Pre-compute expert outputs (DeepFM backbone)
python scripts/prestore.py \
    --model-dir results/models \
    --features-dir data/features \
    --embeddings-dir data/embeddings \
    --output-dir data/prestore \
    --backbone deepfm

# LightGCN backbone, smaller batch
python scripts/prestore.py \
    --model-dir results/models \
    --features-dir data/features \
    --embeddings-dir data/embeddings \
    --output-dir data/prestore \
    --backbone lightgcn \
    --batch-size 2048
```

### Output Files

```
data/prestore/
├── item_expert.npz    # expert_outputs (n_items, d_rec) — factual expert output
└── user_expert.npz    # expert_outputs (n_users, d_rec) — reasoning expert output
```

### Internal Calls

```
scripts/prestore.py
  → src.kar.embedding_index.build_aligned_embeddings() — feature-index-aligned BGE .npz
  → src.training.trainer.create_kar_train_state() — initialize KARModel
  → src.training.trainer._load_model_state() — load trained weights
  → src.serving.prestore.compute_prestore() — batched expert forward + save .npz
```

---

## Data Flow Diagram

```
Raw CSV (data/h-and-m-personalized-fashion-recommendations/)
  │
  ├── articles.csv (105K) ──┐
  ├── customers.csv (1.37M) ├── scripts/preprocess.py ──→ data/processed/
  └── transactions.csv (31M)┘         │                    ├── *.parquet (cleaned)
                                       │                    ├── {train,val,test}_transactions.parquet
                                       │                    ├── {val,test}_ground_truth.json
                                       │                    └── {active,sparse}_customer_ids.json
                                       │
  images/ (105K .jpg) ─────────────────┤
                                       │
                                       ▼
                    scripts/extract_factual_knowledge.py ──→ data/knowledge/factual/
                                       │                     ├── factual_knowledge.parquet (105K rows)
                                       │                     ├── quality_report.json
                                       │                     └── checkpoint/
                                       │
                                       ├── scripts/eval_factual.py ──→ results/eval/factual/
                                       │                                └── factual_eval_report.json
                                       ▼
                    scripts/extract_reasoning_knowledge.py ──→ data/knowledge/reasoning/
                                       │          ├── user_profiles.parquet (1.3M users)
                                       │          ├── reasoning_texts.parquet (KAR input)
                                       │          └── quality_report.json
                                       │
                                       ├── scripts/eval_reasoning.py ──→ results/eval/reasoning/
                                       │                                 └── reasoning_eval_report.json
                                       ▼
                    scripts/build_features.py ──→ data/features/
                                       │          ├── train_pairs.npz
                                       │          ├── user_features.npz
                                       │          ├── item_features.npz
                                       │          ├── feature_meta.json
                                       │          ├── id_maps.json
                                       │          └── cat_vocab.json
                                       ▼
                              scripts/train.py ──→ results/predictions/{backbone}_{split}.json
                                       │            results/models/{backbone}_best/params.npz
                                       │
  data/embeddings/                     │
  ├── item_bge_embeddings.npz ────────┤
  └── user_bge_embeddings.npz ────────┘
                                       │  (--use-kar)
                                       ├── scripts/train.py --use-kar ──→ results/models/kar_{backbone}_best/
                                       │                                   results/predictions/kar_{backbone}_{split}.json
                                       │
                                       ├── scripts/prestore.py ──→ data/prestore/
                                       │                           ├── item_expert.npz
                                       │                           └── user_expert.npz
                                       ▼
                              scripts/evaluate.py ──→ results/metrics/{backbone}_{split}.json
```

---

## 8. Segmentation (`scripts/segment.py`)

Phase 3: 고객/상품 세그멘테이션. BGE 임베딩 계산 + 5-level 고객 세그멘테이션 + 상품 클러스터링.

### Usage

```bash
# Full pipeline (embeddings + segmentation + analysis)
python scripts/segment.py \
    --fk-dir data/knowledge/factual \
    --rk-dir data/knowledge/reasoning \
    --data-dir data/processed \
    --output-dir data/segmentation

# Embeddings only (Phase 4 KAR 준비)
python scripts/segment.py \
    --fk-dir data/knowledge/factual \
    --rk-dir data/knowledge/reasoning \
    --output-dir data/segmentation \
    --embeddings-only

# Skip embeddings (이미 계산된 경우)
python scripts/segment.py \
    --fk-dir data/knowledge/factual \
    --rk-dir data/knowledge/reasoning \
    --data-dir data/processed \
    --output-dir data/segmentation \
    --skip-embeddings
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--fk-dir` | `data/knowledge/factual` | Factual knowledge 디렉토리 |
| `--rk-dir` | `data/knowledge/reasoning` | Reasoning knowledge 디렉토리 |
| `--data-dir` | `data/processed` | 전처리된 데이터 디렉토리 |
| `--output-dir` | `data/segmentation` | 출력 디렉토리 |
| `--embeddings-dir` | `data/embeddings` | 임베딩 출력 디렉토리 |
| `--bge-model` | `BAAI/bge-base-en-v1.5` | BGE 모델명 |
| `--bge-batch-size` | `256` | 인코딩 배치 크기 |
| `--bge-device` | `mps` | BGE 디바이스: mps \| cpu \| cuda \| cuda:1 (CUDA 호스트에서 임베딩 재생성 시 `cuda:1`) |
| `--customer-method` | `kmeans` | 클러스터링 방법 |
| `--embeddings-only` | `False` | 임베딩만 계산 |
| `--skip-embeddings` | `False` | 임베딩 계산 건너뛰기 |
| `--random-seed` | `42` | 랜덤 시드 |

### Source Modules

| Module | Key Functions |
|--------|--------------|
| `src/embeddings.py` | `compute_item_embeddings()`, `compute_user_embeddings()`, `load_embeddings()` — shared across segmentation & KAR |
| `src/segmentation/embeddings.py` | Re-exports from `src.embeddings` (backward compat) |
| `src/segmentation/vectorizer.py` | `vectorize_l1()`, `vectorize_l2()`, `vectorize_l3()` |
| `src/segmentation/clustering.py` | `select_k()`, `fit_clusters()`, `reduce_pca(standardize, whiten)`, `compute_umap_2d()` |
| `src/segmentation/topics.py` | `fit_topics()` — BERTopic-style UMAP+HDBSCAN+c-TF-IDF |
| `src/segmentation/customer.py` | `run_customer_segmentation()` — 5-level orchestration (BGE isotropy correction for Semantic/Topic) |
| `src/segmentation/product.py` | `run_product_clustering()` — BGE clusters + ARI + cross-category (BGE isotropy correction) |
| `src/segmentation/analysis.py` | `profile_segments()`, `profile_segments_discriminative()`, `cross_layer_ari()`, `compute_segment_statistics()`, `compute_effective_k()`, `compute_l3_segment_heatmap_data()`, `compute_cross_category_excess_similarity()`, `run_topic_sensitivity()` |

### Output Artifacts

```
data/embeddings/
├── item_bge_embeddings.npz        # (105494, 768) float16 — shared across segmentation & KAR
└── user_bge_embeddings.npz        # (1298206, 768) float16 — shared across segmentation & KAR

data/segmentation/
├── customer_l1_vectors.npz        # (~89D structured)
├── customer_l2_vectors.npz        # (~49D structured)
├── customer_l3_vectors.npz        # (~37D structured)
├── customer_segments.parquet      # customer_id + 5 segment columns
├── product_clusters.parquet       # article_id + cluster_id + metadata
├── cross_category_pairs.parquet   # similar items across product types
├── segment_profiles.json          # per-level segment profiles
├── clustering_meta.json           # k, silhouette, topic counts
└── segment_stats_{level}.csv      # per-segment statistics
```

### Pipeline Position

```
data/knowledge/factual/factual_knowledge.parquet ──┐
data/knowledge/reasoning/user_profiles.parquet ────┤
data/processed/transactions.parquet ───────────────┤
data/processed/articles.parquet ───────────────────┘
                    │
                    ▼
          scripts/segment.py
                    │
                    ├──→ data/embeddings/item_bge_embeddings.npz     (→ Phase 4 KAR)
                    ├──→ data/embeddings/user_bge_embeddings.npz     (→ Phase 4 KAR)
                    ├──→ data/segmentation/customer_segments.parquet
                    ├──→ data/segmentation/product_clusters.parquet
                    └──→ data/segmentation/clustering_meta.json
```

---

## 9. Knowledge-Purchase Analysis (`scripts/analyze_knowledge.py`)

세그멘테이션 보완 분석: LLM 추출 L1/L2/L3 속성의 구매 예측 가치를 정보이론적·임베딩적으로 검증.

### CLI Usage

```bash
# 전체 분석 (MI + Cold-Start + Layer Info + Diversity)
python scripts/analyze_knowledge.py \
    --data-dir data/processed \
    --fk-dir data/knowledge/factual \
    --features-dir data/features \
    --embeddings-dir data/embeddings \
    --output-dir results/analysis

# 특정 컴포넌트만 실행
python scripts/analyze_knowledge.py ... --component mi
python scripts/analyze_knowledge.py ... --component cold-start
python scripts/analyze_knowledge.py ... --component layer-info
python scripts/analyze_knowledge.py ... --component diversity

# 파라미터 조정
python scripts/analyze_knowledge.py \
    --mi-sample-size 5000000 \
    --cs-sample-users 30000 \
    --div-sample-users 50000 \
    --verbose
```

### 4 Components

| Component | Module | Description |
|-----------|--------|-------------|
| A. MI | `src/analysis/mutual_information.py` | 속성별 NMI, PMI, Conditional MI |
| B. Layer Info | `src/analysis/layer_information.py` | CKA, Purchase Coherence, Separation AUC |
| C. Diversity | `src/analysis/preference_diversity.py` | User JSD, Entropy, Temporal Stability, RVI |
| D. Cold-Start | `src/analysis/cold_start.py` | 구매수 구간별 Content-Based Retrieval |

### Key Functions

```python
# Component A: Mutual Information
from src.analysis.mutual_information import compute_attribute_mi, compute_pmi_by_value, compute_conditional_mi
mi_results = compute_attribute_mi(features_dir, fk_path, articles_path, sample_size=10_000_000)
# Returns: list[MIResult(attribute, layer, mi, nmi, n_values)]

# Component B: Layer Information
from src.analysis.layer_information import compute_linear_cka, compute_purchase_coherence, compute_purchase_separation_auc
cka = compute_linear_cka(X_l1, X_l2)  # CKA between two embedding matrices
# Returns: float in [0, 1]

# Component C: Preference Diversity
from src.analysis.preference_diversity import compute_preference_diversity
div_results = compute_preference_diversity(train_txn_path, fk_path, articles_path)
# Returns: list[DiversityResult(attribute, layer, mean_user_entropy, mean_pairwise_jsd, temporal_stability, recommendation_value_index)]

# Component D: Cold-Start
from src.analysis.cold_start import compute_contentbased_retrieval, run_all_combos
bracket_results = compute_contentbased_retrieval(embeddings, item_ids, user_history, val_gt, "L1+L2+L3")
# Returns: list[BracketResult(bracket, layer_combo, hr_at_12, ndcg_at_12, mrr, n_users)]
```

### Output

```
results/analysis/
├── mi_results.csv           # NMI per attribute (sorted)
├── conditional_mi.json      # MI(L2|L1), MI(L3|L1+L2)
├── cold_start_results.csv   # HR@12 per bracket × layer combo
├── separation_auc.json      # AUC per layer combo
└── diversity_results.csv    # Entropy, JSD, RVI per attribute
```

### Pipeline Position

```
data/features/train_pairs.npz ─────────────┐
data/features/id_maps.json ────────────────┤
data/knowledge/factual/factual_knowledge.parquet ─┤
data/processed/articles.parquet ───────────┤
data/processed/train_transactions.parquet ─┤
data/processed/val_ground_truth.json ──────┤
data/embeddings/item_bge_embeddings.npz ───┘
                    │
                    ▼
        scripts/analyze_knowledge.py
                    │
                    └──→ results/analysis/
```

---

## 서버 동기화 (`scripts/sync.sh`)

로컬 ↔ 원격 서버 파일 동기화. **PyCharm SFTP auto-upload 대체** — `rsync`(SSH 1연결 배치 + delta 전송 + 압축 + exclude)로 코드 동기화가 증분 시 1초 미만. Python 스크립트가 아닌 bash 유틸이며, 접속 설정은 프로젝트 루트의 `.sync.env`(git-ignored)에서 읽는다.

### 접속 설정 (`.sync.env`)

```sh
REMOTE_USER=mail-agent
REMOTE_HOST=3.38.195.121
REMOTE_PORT=5040
SSH_KEY=                                          # 비우면 기본 키/패스워드 (키 인증이면 비움)
REMOTE_DIR=/home/mail-agent/llm-factor-recsys-hnm
REMOTE_RSYNC=/home/mail-agent/.local/bin/rsync    # 서버 userland rsync (시스템 rsync 있으면 비움)
```

서버에 시스템 rsync가 없을 경우(no-sudo) userland 설치:
```bash
ssh -p 5040 mail-agent@3.38.195.121
cd /tmp && curl -sSO http://security.ubuntu.com/ubuntu/pool/main/r/rsync/rsync_3.2.7-0ubuntu0.22.04.7_amd64.deb
curl -sSO http://archive.ubuntu.com/ubuntu/pool/main/libp/libpopt/libpopt0_1.18-3build1_amd64.deb
for d in rsync_*.deb libpopt0_*.deb; do dpkg-deb -x "$d" ~/.local/rsync-pkg; done
cat > ~/.local/bin/rsync <<'EOF'
#!/bin/sh
P="$HOME/.local/rsync-pkg"
export LD_LIBRARY_PATH="$P/usr/lib/x86_64-linux-gnu:$P/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
exec "$P/usr/bin/rsync" "$@"
EOF
chmod +x ~/.local/bin/rsync
# 또는 sudo 가능하면:  sudo apt-get update && sudo apt-get install -y rsync  (이후 REMOTE_RSYNC 비움)
```

### 서브커맨드

| 명령 | 방향 | 대상 | 비고 |
|------|------|------|------|
| `push` | local→server | 코드(`src scripts configs tests mlops notebooks docs *.toml *.md`) | 가장 빠름. 존재하는 경로만 전송 |
| `push-knowledge` | local→server | `data/knowledge/` (~1.6G) | LLM 산출물(재생성 불가) |
| `push-data` | local→server | `processed/features/embeddings/segmentation/knowledge` | raw 32G 제외 |
| `pull` | server→local | `results/` | 모델·예측 회수. `--delete` 미사용(로컬 보호) |
| `remote "<cmd>"` | — | 서버에서 명령 실행 | 예: `remote "whoami && pwd"` |

### 플래그

- `-n` / `--dry-run`: 실제 전송 없이 미리보기 (특히 `--delete` 영향 사전확인)
- `--delete`: 대상에서 원본에 없는 파일 삭제 (push 미러링용; **기본 off**, opt-in)

### 예시

```bash
# 접속·원격 경로 확인
./scripts/sync.sh remote "whoami && pwd"

# 코드만 빠르게 업로드 (PyCharm auto-upload 대체)
./scripts/sync.sh push

# 변경 미리보기 후 업로드
./scripts/sync.sh push -n
./scripts/sync.sh push

# LLM 지식 데이터 업로드 (1회성, 이후 델타)
./scripts/sync.sh push-knowledge

# 서버에서 학습 후 결과 회수
./scripts/sync.sh pull

# 서버에서 학습 트리거
./scripts/sync.sh remote "cd ~/llm-factor-recsys-hnm && python scripts/train.py --backbone deepfm"
```

### 제외 규칙 (`.rsync-exclude`)

VCS/IDE/캐시(`.git/`, `.idea/`, `__pycache__/`, `*.pyc`, …), 실험 부산물(`wandb/`, `ray_results/`), **raw 데이터**(`data/h-and-m-personalized-fashion-recommendations/` 32G — 서버에서 kaggle로 직접 받기), **비밀/로컬 파일**(`*.pem`, `.env`, `.sync.env`)을 전송에서 제외.

### 데이터 전략 (집 업로드 9~32GB 회피)

| Tier | 대상 | 처리 |
|------|------|------|
| 반드시 업로드 | 코드 + `data/knowledge/` (~1.6G) | `push` / `push-knowledge` (재생성 불가) |
| 서버 재생성 권장 | `processed`/`features`/`embeddings`/`segmentation` (~7.8G) | 서버에서 파이프라인 재실행(GPU가 빠름), 또는 `push-data` |
| 안 올림 | raw (`data/h-and-m-.../` 32G) | 서버에서 `kaggle competitions download` (Kaggle 토큰 필요) |
