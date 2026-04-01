# Model Architecture Reference

> 프로젝트 내 모든 추천 모델의 아키텍처, 학습 파이프라인, 서빙 경로를 정리한다.

---

## 1. 아키텍처 개관

### 전체 파이프라인

```
[오프라인 학습]
Features(User 11D + Item 7D) + BGE Embeddings(768D)
    │
    ├─ Baseline: Backbone 단독 학습 (BCE) ──→ 전체 카탈로그 스코어링 → top-12
    │
    ├─ KAR 1-Stage: Backbone+Expert+Gating+Fusion 3-Stage 학습
    │      → 전체 카탈로그 스코어링 → top-12
    │
    └─ KAR 2-Stage (ReRank):
           Stage 1: KAR 모델 + 휴리스틱(repurchase/age_pop/recency) → top-100 후보
           Stage 2: LightGBM 재정렬 (KAR Expert 피처 + interaction 피처) → top-12
```

### 1-Stage vs 2-Stage

| 항목 | 1-Stage (KAR direct) | 2-Stage (KAR → LightGBM) |
|------|---------------------|--------------------------|
| 스코어링 범위 | 전체 카탈로그 (~105K) | Stage 1: 105K → 100, Stage 2: 100 → 12 |
| LLM 속성 활용 | BGE → Expert → Gating → Fusion | Stage 2: Expert output 130D dense |
| 추가 시그널 | 없음 | repurchase, age_pop, recency, interaction features |
| 연구 기여 귀인 | 깔끔 (KAR 단독) | 모호 (KAR + LightGBM) |
| 적합 시나리오 | Ablation 실험, 논문 | Kaggle-style 최적화 |

---

## 2. 백본 모델 (5종)

> 모든 백본은 logits `(B,)` 출력. sigmoid는 loss에서 적용 (모델 내부 X).
> `src/models/` 디렉토리, `BackboneRegistry`로 dispatch.

### 2.1 DeepFM (`src/models/deepfm.py`)

**입력**: `DeepFMInput(user_cat: (B,3), user_num: (B,8), item_cat: (B,5), item_num: (B,2))`

**아키텍처**:
- **First-order**: Embedding(cat) + Linear(num) → `(B, 1)`
- **FM**: `0.5 * (sum(e)^2 - sum(e^2))` over 18 fields × d_embed → `(B, 1)`
- **DNN**: concat(embeddings) → FC(256→128→64→1) + BatchNorm + Dropout
- **LayerNorm**: FM, DNN 출력에 각각 적용 (logit 폭발 방지 필수)
- 최종: `FM + DNN + first_order`

**하이퍼파라미터**: d_embed=16, dnn_hidden=(256,128,64), dropout=0.1

**안정성**: LayerNorm 없으면 초기 logit 100~700 → loss 폭발. v1(0.001773) → v17(0.002941) +66% 개선.

### 2.2 DCNv2 (`src/models/dcnv2.py`)

**입력**: DeepFM과 동일

**아키텍처**:
- **Cross Network v2** (3 layers): `x_{l+1} = x0 * (MoE(x_l) + b) + x_l`
  - MoE: low-rank (U_k, V_k) + gating network, 4 experts, d_low_rank=64
  - LayerNorm per layer (residual 누적 방지)
- **DNN**: 병렬 분기, DeepFM과 동일 구조
- 최종: `concat(x_cross, x_deep) → Linear → (B, 1)`

**하이퍼파라미터**: n_cross_layers=3, n_experts=4, d_low_rank=64, d_embed=16

**성능**: MAP@12=0.003361 (Popularity 88.9%), DeepFM 대비 +14%

### 2.3 LightGCN (`src/models/lightgcn.py`)

**입력**: `LightGCNInput(user_idx: (B,), item_idx: (B,))`

**아키텍처**:
- User/Item Embedding만 학습 (weight, activation 없음)
- Bipartite graph propagation: `D^{-1/2} A D^{-1/2}` × K layers
- Layer aggregation: `mean(E^(0), E^(1), ..., E^(K))`
- 스코어링: dot product `u · i`

**하이퍼파라미터**: d_embed=64, n_layers=3, l2_reg=1e-4

**특징**: Feature engineering 불필요, 그래프 구조만 사용. Adjacency matrix 사전 구축 필요.

### 2.4 DIN (`src/models/din.py`)

**입력**: `DINInput(...static features..., history: (B,T), hist_len: (B,))`

**아키텍처**:
- Static embeddings (DeepFM과 동일)
- **MLP Attention**: `concat(q, k, q-k, q*k) → MLP → sigmoid` (softmax 아님)
  - 구매 이력에서 target과 관련된 아이템에 높은 가중치
- DNN: `concat(user_interest, target_embed, static) → FC(256→128→64→1)`

**하이퍼파라미터**: d_embed=16, attention_hidden=(64,32), max_seq_len=50

**특징**: Attention weight로 "왜 이 아이템을 추천했는가" 해석 가능.

### 2.5 SASRec (`src/models/sasrec.py`)

**입력**: `SASRecInput(history: (B,T), hist_len: (B,))`

**아키텍처**:
- Item + Position embedding → LayerNorm + Dropout
- N Transformer blocks (causal masking):
  - Multi-head self-attention (n_heads=2)
  - FFN (d → 4d → d)
  - LayerNorm + residuals + dropout
- 마지막 valid position → user representation `(B, d)`
- 스코어링: dot product with target

**하이퍼파라미터**: d_embed=64, n_heads=2, n_blocks=2, max_seq_len=50

**특징**: User/Item feature 없이 순서 정보만 사용. Causal masking으로 미래 정보 차단.

### 백본 비교

| 백본 | 패러다임 | 입력 | d_embed | 핵심 메커니즘 |
|------|---------|------|---------|-------------|
| DeepFM | Feature-based | cat+num | 16 | FM interaction + DNN |
| DCNv2 | Feature-based | cat+num | 16 | Cross Network (MoE) + DNN |
| LightGCN | Graph | user_idx, item_idx | 64 | GCN propagation |
| DIN | Sequential | features + history | 16 | MLP Attention |
| SASRec | Sequential | history only | 64 | Transformer (causal) |

---

## 3. KAR (Knowledge-Augmented Recommendation)

> `src/kar/` 디렉토리. KARModel이 backbone을 소유하며 모든 상호작용을 관리.

### 3.1 KARModel (`src/kar/hybrid.py`)

```
h_fact (B, 768)  ─→ Factual Expert  ─→ e_fact (B, 64)  ─┐
                                                          ├→ Gating → (g_fact, g_reason)
h_reason (B, 768) → Reasoning Expert → e_reason (B, 64) ─┘
                                                          │
                    e_aug = g_fact * e_fact + g_reason * e_reason  (B, 64)
                                                          │
base_input ──────→ Backbone.embed() → x_backbone (B, d_backbone)
                                                          │
                    Fusion(x_backbone, e_aug) → x_augmented
                                                          │
                    Backbone.predict(x_augmented) → logits (B,)
```

**입력**: `KARInput(base_input, h_fact: (B,768), h_reason: (B,768), context?, target_item_idx?)`

**`forward_with_intermediates()`** 반환:
- `e_fact (B, 64)`, `e_reason (B, 64)`, `g_fact (B, 1)`, `g_reason (B, 1)`, `x_backbone_flat (B, d_backbone)`

**백본별 d_backbone**:

| 백본 | d_backbone | 산출 |
|------|-----------|------|
| DeepFM / DCNv2 | 288 | 18 fields × 16 |
| LightGCN | 128 | 64 × 2 (user + item) |
| SASRec | 128 | 64 × 2 |
| DIN | ~1600 | 64 + 64 + static_dim |

### 3.2 Expert (`src/kar/expert.py`)

```
(B, 768) → Linear(768→256) → ReLU → Dropout(0.1)
         → Linear(256→64) → LayerNorm → (B, 64)
```

- Config: `ExpertConfig(d_enc=768, d_hidden=256, d_rec=64, n_layers=2, dropout_rate=0.1)`
- Factual Expert: 아이템 BGE 임베딩 → 아이템 속성의 밀집 표현
- Reasoning Expert: 유저 BGE 임베딩 → 유저 선호의 밀집 표현

### 3.3 Gating 4종 (`src/kar/gating.py`)

| Variant | 입력 | 로직 | 파라미터 |
|---------|------|------|---------|
| **G1 (Fixed)** | - | `sigmoid(learnable_scalar)` | 1 |
| **G2 (Expert, 기본)** | `[e_fact; e_reason]` (128D) | `softmax(Linear(128→2))` | 258 |
| **G3 (Context)** | `[e_fact; e_reason; context]` | `softmax(Linear(128+d_ctx→2))` | 258+d_ctx |
| **G4 (Cross)** | `e_fact * e_reason` (64D) | `softmax(Linear(64→2))` | 130 |

출력: `(g_fact, g_reason)` 각 `(B, 1)`, 합=1.0

### 3.4 Fusion 4종 (`src/kar/fusion.py`)

| Variant | 수식 | 출력 차원 |
|---------|------|----------|
| **F1 (Concat)** | `concat(x_backbone, proj(e_aug))` | 2 × d_backbone |
| **F2 (Addition, 기본)** | `x_backbone + alpha * proj(e_aug)` | d_backbone |
| **F3 (Gated)** | `gate * x_backbone + (1-gate) * proj(e_aug)` | d_backbone |
| **F4 (CrossAttn)** | `LayerNorm(x_backbone + MHA(Q=x, KV=proj(e_aug)))` | d_backbone |

`proj`: `Linear(d_rec → d_backbone)`

---

## 4. 학습 파이프라인

> `src/training/trainer.py` (핵심 로직) + `scripts/train.py` (CLI 래퍼)

### 4.1 Baseline 학습

```python
create_train_state(backbone_name, model_config, train_config, feature_meta, features_dir)
→ (model, optimizer)
```

- Loss: `binary_cross_entropy(logits, labels)`
- Optimizer: Adam, lr=1e-3
- Data: Grain DataLoader, batch=2048, neg_ratio=1 (24.4M positive → 48.7M total)
- Early stopping: MAP@12 on sampled 5000 users, patience=3
- Max epochs: 50

### 4.2 KAR 3-Stage 학습 (`run_kar_training`)

| Stage | Epochs | 학습 대상 | Loss | LR |
|-------|--------|----------|------|-----|
| **S1: Backbone Pre-train** | max 20 | Backbone only | BCE | 1e-3 |
| **S2: Expert Adaptor** | max 10 | Expert+Gating (backbone frozen via stop_gradient) | BCE + align + diversity | 1e-3 |
| **S3: End-to-End** | max 10 | 전체 | BCE + align + diversity | 1e-4 (0.1x) |

- S1 → S2: best S1 체크포인트 로드 후 backbone freeze
- S2 → S3: best S2 체크포인트 로드 후 전체 unfreeze + LR 감소
- 각 Stage에서 독립적 early stopping (MAP@12, patience=3)

### 4.3 Loss 함수 (`src/losses.py`)

| Loss | 수식 | 용도 |
|------|------|------|
| **BCE** | `-y*log(sigmoid(logits)) - (1-y)*log(1-sigmoid(logits))` | 모든 학습 |
| **Align** | `MSE(e_expert, stop_gradient(x_backbone_proj))` | Expert가 backbone 표현에 정렬 |
| **Diversity** | `mean(cosine(normalize(e_fact), normalize(e_reason)))` | 두 Expert 분화 유도 |
| **L2 Reg** | `||user_emb||^2 + ||item_emb||^2` | LightGCN 정규화 |

KAR 통합 loss: `BCE + 0.1 * align + 0.01 * diversity`

---

## 5. 2-Stage ReRanker

> `scripts/train_reranker.py` (CLI) + `src/models/reranker.py` (LightGBM) + `src/features/reranker_features.py` (피처)

### 5.1 Stage 1: 후보 생성

4개 소스에서 독립 추출 후 `blend_candidates()`로 합집합 → top-K (기본 100):

| Source | 함수 | 로직 | Top-K |
|--------|------|------|-------|
| **stage1** | `extract_stage1_candidates()` / `_kar()` | 모델 전체 카탈로그 스코어링 | --top-k |
| **repurchase** | `extract_repurchase_candidates()` | product_code JOIN으로 SKU 확장, recency 가중 | 50 |
| **age_pop** | `extract_age_popularity_candidates()` | 연령대별 구매 빈도 top-K | 50 |
| **recency** | `extract_recency_candidates()` | 최근 14일 글로벌 인기 (모든 유저 동일) | 50 |

`blend_candidates()`: 유저별 union → 같은 아이템은 max score 채택 → top-K 선택

`src/features/candidate_generation.py`

### 5.2 Stage 2: LightGBM 재정렬 (`src/models/reranker.py`)

- `LGBMClassifier(objective="binary", is_unbalance=True)`
- Train/Val: 유저 단위 80/20 split
- Early stopping: Val AUC, patience=50
- 출력: `predict_proba()[:, 1]` → 유저별 내림차순 → top-12

**하이퍼파라미터**: n_estimators=500, max_depth=6, lr=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8

### 5.3 피처 구성 (`src/features/reranker_features.py`)

#### mode=base (~24D)

| 그룹 | 차원 | 피처 |
|------|------|------|
| Score | 3 | stage1_score, rank_position, gap_to_rank1 |
| User | 11 | n_purchases, avg_price, price_std, n_unique_categories, n_unique_colors, days_since_first, days_since_last, online_ratio, age_group, club_status, fashion_news |
| Item | 7 | total_purchases, avg_price, product_type, colour, garment_group, section, index |
| Cross | 3 | age*section, price_ratio, recency*popularity |

#### mode=full (~133D)

base + L1/L2/L3 LabelEncoder 속성(~102D) + interaction(6D) + BGE cosine(1D)

**문제**: LabelEncoder는 의미적 관계를 파괴 ("casual"=3, "relaxed"=7 → 거리 무의미)

#### mode=kar (~161D)

base + **KAR Expert/Gating dense features(130D)** + interaction(6D) + BGE cosine(1D)

| 피처 | 차원 | 설명 |
|------|------|------|
| kar_e_fact | 64 | Factual Expert output — 아이템의 의미적 밀집 표현 |
| kar_e_reason | 64 | Reasoning Expert output — 유저-아이템 관계의 의미적 표현 |
| kar_g_fact | 1 | Factual gating weight |
| kar_g_reason | 1 | Reasoning gating weight |

의미적 유사성 보존: LabelEncoder 정수 코드 대신 학습된 dense vector 사용.

`--stage1-backbone kar_deepfm` + `--mode kar` 로 실행.

#### Interaction features (multi-source 사용 시, 6D)

| 피처 | 설명 |
|------|------|
| has_bought_before | 유저가 이 아이템을 산 적 있는가 |
| purchase_count | 구매 횟수 |
| days_since_item_purchase | 마지막 구매 이후 일수 |
| has_bought_category | 이 카테고리를 산 적 있는가 |
| category_purchase_count | 카테고리별 구매 횟수 |
| user_item_price_ratio | 유저 평균가 / 아이템 가격 |

`src/features/candidate_generation.py:build_interaction_data()`

---

## 6. 피처 시스템

> `src/features/store.py` (로딩) + `src/features/engineering.py` (생성)

### User Features (11D)

| 타입 | 차원 | 필드 |
|------|------|------|
| Numerical | 8 | n_purchases, avg_price, price_std, n_unique_categories, n_unique_colors, days_since_first, days_since_last, online_ratio |
| Categorical | 3 | age_group, club_status, fashion_news |

### Item Features (7D)

| 타입 | 차원 | 필드 |
|------|------|------|
| Numerical | 2 | total_purchases, avg_price |
| Categorical | 5 | product_type, colour, garment_group, section, index |

### BGE Embeddings (768D)

- 모델: BAAI/bge-base-en-v1.5
- 아이템: factual text (L1+L2+L3 속성 조합) → 768D
- 유저: reasoning text (LLM 추론 프로파일) → 768D
- 저장: float16 `.npz`, `src/kar/embedding_index.py:build_aligned_embeddings()`로 feature index 정렬

### 학습 데이터

- Train pairs: ~24.4M positive + ~24.4M negative (neg_ratio=1)
- Loader: `grain.python.DataLoader` (deterministic, multiprocess prefetch)
- `src/training/data_loader.py`

---

## 7. 평가

> `src/evaluation/metrics.py`

### 메트릭 (k=12)

| 메트릭 | 설명 |
|--------|------|
| **MAP@12** | Kaggle 공식 메트릭 (primary). Average Precision at 12 |
| **HR@12** | Hit Rate — 12개 추천 중 1개라도 맞으면 1 |
| **NDCG@12** | DCG/IDCG with log2 discount |
| **MRR** | 첫 번째 hit의 reciprocal rank |

### 실험 결과 요약 (전체 유저 평가, 413K test users)

| 모델 | MAP@12 | vs Popularity |
|------|--------|--------------|
| Popularity Global (baseline) | 0.003783 | 100% |
| ReRank-Base (DCNv2→LightGBM, 단일) | 0.004055 | 107% |
| ReRank-Base (DCNv2→LightGBM, multi-source) | - | - |
| DCNv2 + LayerNorm | 0.003361 | 88.9% |
| KAR v3 (DCNv2+KAR) | 0.003499 | 92.5% |
| DeepFM + LayerNorm | 0.002941 | 77.7% |

**핵심 발견**:
- 메타데이터 피처만으로는 Popularity 미달 → KAR 지식 필수
- 1000-user sample validation은 25~37% 과대추정 → 전체 평가 필수
- DCNv2 > DeepFM (+14%) — high-order interaction 학습 우위
- LayerNorm이 안정화의 핵심 — FM/Cross Network 모두 logit 폭발 방지 필수

---

## 8. 서빙

> `src/serving/prestore.py` (사전 계산) + `mlops/serving/app.py` (FastAPI)

### Pre-store

Expert 출력을 오프라인에서 사전 계산:
- `item_expert.npz`: `(n_items, 64)` — Factual Expert output
- `user_expert.npz`: `(n_users, 64)` — Reasoning Expert output
- 서빙 시 Expert MLP forward 생략 → 5~10x 속도 개선

### 온라인 스코어링

```
Request(user_id)
→ Redis 캐시 체크 (HIT → 즉시 반환, ~0.5ms)
→ MISS → numpy lookup(user/item vectors) + JAX 전체 카탈로그 스코어링 (~15ms)
→ Redis 저장 (TTL 1h)
→ 응답
```

- 전체 카탈로그 직접 스코어링: 105K 아이템 × 1 유저, JAX JIT
- 2-Stage 서빙 시: Stage 1 top-100 → LightGBM re-rank → top-12

---

## 코드 경로 참조

| 영역 | 파일 |
|------|------|
| 백본 모델 | `src/models/deepfm.py`, `dcnv2.py`, `lightgcn.py`, `din.py`, `sasrec.py` |
| KAR | `src/kar/hybrid.py`, `expert.py`, `gating.py`, `fusion.py` |
| ReRanker | `src/models/reranker.py`, `src/features/reranker_features.py` |
| 후보 생성 | `src/features/candidate_generation.py` |
| 학습 | `src/training/trainer.py`, `data_loader.py` |
| Loss | `src/losses.py` |
| 피처 | `src/features/store.py`, `engineering.py` |
| 임베딩 | `src/embeddings.py`, `src/kar/embedding_index.py` |
| 평가 | `src/evaluation/metrics.py` |
| 서빙 | `src/serving/prestore.py` |
| Config | `src/config.py` |
| CLI | `scripts/train.py`, `scripts/train_reranker.py` |
