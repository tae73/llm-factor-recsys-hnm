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

# 3f. GBDT Re-Ranker (2-stage baseline)
# Base mode (score + user/item features only)
python scripts/train_reranker.py \
    --stage1-model-dir results/models \
    --stage1-backbone deepfm \
    --data-dir data/processed \
    --features-dir data/features \
    --output-dir results/reranker \
    --mode base --no-wandb
# Full mode (Base + L1/L2/L3 attributes + BGE similarity)
python scripts/train_reranker.py \
    --stage1-model-dir results/models \
    --stage1-backbone deepfm \
    --data-dir data/processed \
    --features-dir data/features \
    --fk-dir data/knowledge/factual \
    --embeddings-dir data/embeddings \
    --output-dir results/reranker \
    --mode full --no-wandb

# 3g. Pre-store expert outputs for serving
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
| `--verbose` | bool | False | Print detailed statistics |

### Output Files

```
data/processed/
├── articles.parquet              # Cleaned articles (article_id as VARCHAR)
├── customers.parquet             # Cleaned customers (nulls filled)
├── transactions.parquet          # Cleaned transactions (sorted by t_dat)
├── train_transactions.parquet    # Train split
├── val_transactions.parquet      # Validation split
├── test_transactions.parquet     # Test split
├── val_ground_truth.json         # {customer_id: [article_ids]}
├── test_ground_truth.json        # {customer_id: [article_ids]}
├── active_customer_ids.json      # Active users (5+ purchases)
└── sparse_customer_ids.json      # Sparse users (1-4 purchases)
```

### Internal Calls

```
scripts/preprocess.py
  → src.data.preprocessing.run_preprocessing(DataPaths)
    → validate_raw_data() — DuckDB validation
    → load_and_convert_{articles,customers,transactions}() — ThreadPool parallel
  → src.data.splitter.run_split(processed_dir, output_dir, SplitConfig, FilterConfig)
    → split_transactions_temporal() — DuckDB WHERE on t_dat
    → filter_customers_by_activity() — GROUP BY + COUNT
    → build_ground_truth() — Deduplicated purchase lists
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
| `--features-dir` | Path | None | Feature directory (required for neural backbones) |
| `--learning-rate` | float | 0.001 | Learning rate |
| `--batch-size` | int | 2048 | Batch size |
| `--max-epochs` | int | 50 | Max training epochs |
| `--patience` | int | 3 | Early stopping patience |
| `--d-embed` | int | 16 | Embedding dimension |
| `--dropout-rate` | float | 0.1 | Dropout rate |
| `--no-wandb` | bool | False | Disable W&B logging |
| `--random-seed` | int | 42 | Random seed |
| `--num-workers` | int | 4 | Grain data loader workers |
| `--prefetch-buffer-size` | int | 2 | Batches to prefetch per worker |
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

### Available Backbones

| Backbone | Description | Key Hyperparameters |
|----------|-------------|---------------------|
| `popularity_global` | Top-K most purchased items (all time) | k |
| `popularity_recent` | Top-K most purchased items (last 7 days) | k, window_days=7 |
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
