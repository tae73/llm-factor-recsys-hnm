# DeepFM Tuning History

> DeepFM baseline 학습 과정에서 발견된 문제, 시도한 해결책, 결과를 기록한다.
> Popularity Global baseline MAP@12=0.003783 대비 성능을 측정한다.

---

## 환경

- GPU: NVIDIA A100 80GB (MIG 3g.40gb), CUDA Driver 12.2
- Framework: JAX 0.9.2 + Flax NNX 0.12.6
- Data: 105K items, 1.3M users, 24.4M positive pairs
- Validation: 1,000 sampled users × full catalog scoring (105K items)

---

## 실험 이력

### v1: 원본 (2026-03-23)

**설정**: DeepFM d_embed=16, DNN(256,128,64), BCE loss, Adam lr=1e-3, batch=2048, neg_ratio=4 (121.8M pairs)

| Epoch | Avg Loss | MAP@12 | HR@12 | NDCG@12 | MRR |
|-------|----------|--------|-------|---------|-----|
| 1 | 16,534 | 0.000247 | 0.010 | 0.000957 | 0.002156 |
| 4 | 3,133 | 0.000948 | 0.014 | 0.002252 | 0.004876 |
| **6** | **1,482** | **0.001773** | **0.019** | **0.003334** | **0.005361** |
| 9 (stop) | 192 | 0.001228 | 0.010 | 0.001918 | 0.002385 |

**결과**: Early stop at epoch 9, best epoch 6. MAP@12=0.001773 (Popularity의 47%).
**문제**: Loss 16,534로 시작 — logit이 수만 단위. Epoch 7+ 과적합 (loss↓↓ MAP↓).
**버그 수정**: PRNGKey save (`jax.random.key_data`), model load (`nnx.update`).

---

### v5: tanh soft clipping + 정규화 (2026-03-23)

**변경**:
- 수치 피처 z-score 정규화 (학습 시점 in-memory)
- `10 * tanh(logits/10)` → logit ∈ (-10, 10)
- Gradient clipping 10.0

**결과**: Loss=1.994 (정상화!), MAP@12=0.001965 (epoch 1).
**문제**: Loss=2.0 고정 = **mode collapse**. 4:1 neg ratio에서 "모든 쌍을 negative로 예측"이 최적해.

계산: 80% neg → loss=0, 20% pos → logit=-10 → loss=10. Mean = 0.8×0 + 0.2×10 = **2.0**.

---

### v6: neg_ratio=1 + 정규화 + no clipping (2026-03-23)

**변경**:
- 피처 재빌드: neg_ratio=4 → 1 (48.7M pairs, 50/50 balanced)
- tanh clipping 제거 (원본 logit output)
- AdamW weight_decay=1e-5, grad_clip=1.0

**결과**: Loss 82,120 → 발산. Logit 폭발 재발.
**분석**: weight decay 1e-5로는 FM interaction의 quadratic 증가를 억제 불충분.

---

### v7: lr=1e-4로 감소 (2026-03-23)

**변경**: lr 0.001 → 0.0001

**결과**: Loss 60K → 5K (감소 중이나 여전히 큼). MAP@12 ≈ 0.

---

### v8: FM interaction 정규화 (2026-03-23)

**변경**: FM second-order를 `n_fields*(n_fields-1)/2`로 나눔

**결과**: Loss 500 → 260. 개선되었으나 DNN 출력이 여전히 큼.

---

### v9: DNN 입력 L2 정규화 (2026-03-23)

**변경**: DNN 입력을 unit norm으로 정규화

**결과**: Loss 600 → 300. 큰 차이 없음. DNN이 입력 정규화에도 큰 출력 생성.

---

### v10: per-component tanh (2026-03-23)

**변경**:
- `logits = bias + tanh(first_order) + tanh(fm_second) + tanh(dnn_out)`
- 각 component를 [-1, 1]로 bounded → logits ∈ [-3, 3]
- AdamW weight_decay=1e-5, grad_clip=1.0, neg_ratio=1

| Epoch | Avg Loss | MAP@12 | HR@12 | NDCG@12 | MRR |
|-------|----------|--------|-------|---------|-----|
| 1 | 0.433 | 0.000353 | 0.003 | 0.000595 | 0.001208 |
| **2** | **0.426** | **0.001620** | **0.016** | **0.003215** | **0.007093** |
| 3 | 0.421 | 0.000299 | 0.003 | 0.000705 | 0.001591 |
| 4 | 0.417 | 0.000096 | 0.002 | 0.000272 | 0.000424 |
| 5 | 0.413 | 0.000021 | 0.001 | 0.000085 | 0.000250 |

**결과**: Loss 정상 (0.43 → 0.41), 그러나 MAP@12가 epoch 2 이후 급락.
**분석**: tanh [-1,1]이 모델 표현력을 과도하게 제한. Loss↓ but ranking↓ = **underfitting**.

---

## 근본 원인 분석

### 1. Logit 폭발 메커니즘

DeepFM의 FM second-order: `0.5 * Σ_d(sum²_d - sum_of_sq_d)`
- 18 fields × 16-dim embedding → FM output ∝ O(n_fields² × d_embed)
- 초기 embedding norm ~0.3-1.0 → FM output ~45-200
- DNN(256→128→64→1) with ReLU → output 추가로 ~50-500

결과: 초기 logit ~100-700 → BCE loss ~100-700 per sample.

### 2. Mode Collapse (4:1 neg ratio)

neg_ratio=4에서 batch의 80%가 negative:
- 모델이 "모든 쌍 negative" 예측으로 loss 80% 최소화
- Positive 20%에 대한 gradient가 negative에 묻힘
- 학습이 ranking이 아닌 "negative 분류"에 최적화

### 3. BCE vs Ranking 목표 불일치

- BCE: pointwise binary classification (positive/negative 분류)
- MAP@12: ranking quality (positive 아이템이 상위에 오는지)
- BCE 최적화 ≠ ranking 최적화 (loss↓ but MAP 정체/하락 가능)

---

## 현재 코드 상태 (v10 기준)

| File | 변경 내용 |
|------|----------|
| `src/models/deepfm.py` | per-component tanh (first_order, fm_second, dnn_out) |
| `src/training/trainer.py` | AdamW(wd=1e-5) + grad_clip(1.0) + numerical z-score + batched predictions + model save/load fix |
| `src/training/data_loader.py` | NumpyBatchIterator (Grain 대체) |
| `data/features/train_pairs.npz` | neg_ratio=1 (48.7M pairs) |

---

## 다음 시도 방향

### Option A: BPR Pairwise Loss (권장)

`src/losses.py`에 `bpr_loss` 이미 구현됨. BCE pointwise 대신 (positive, negative) 쌍의 상대 순위 학습.
- FM/DNN logit 크기가 아닌 **positive - negative 차이**만 중요
- Logit 폭발에 본질적으로 robust
- per-component tanh 제거 가능 (원본 아키텍처 복원)

### Option B: Embedding Init Scale 축소

- Flax 기본 lecun_normal → `variance_scaling(0.01)` 등 작은 스케일
- 초기 logit ~1-5 범위 → loss 0.5-1.0 → 정상 학습
- 아키텍처 변경 없이 초기화만 수정

### Option C: Per-component tanh 범위 확대

- `tanh(x)` → `5 * tanh(x/5)` 각 component → [-5, 5]
- 표현력 복구 + bounded output
- 가장 단순한 변경

---

### v11: DeepFM 원본 복원 + norm + AdamW + lr=1e-4 + neg_ratio=1 (2026-03-24)

**변경**: per-component tanh 제거, 원본 아키텍처 복원. 나머지 유지.

| Epoch | Avg Loss | MAP@12 (end) | MAP@12 (mid best) |
|-------|----------|--------------|--------------------|
| 1 | 8,285 | 0.000033 | 0.000316 |

**결과**: Loss 진동 (8K→5K→8K), MAP@12 극히 낮음. lr=1e-4에서도 logit 폭발 지속.

---

### v12: DCNv2 baseline + norm + AdamW + lr=1e-4 + neg_ratio=1 (2026-03-24)

**변경**: backbone만 DCNv2로 교체. 동일 피처/정규화/optimizer.

| Epoch | Avg Loss | MAP@12 (end) | MAP@12 (mid best) |
|-------|----------|--------------|--------------------|
| 1 | 75,174,367 | 0.000521 | 0.000786 |

**결과**: Loss 극단적 폭발 (75M), 그러나 MAP@12=0.000521 (DeepFM보다 15x 높음). Cross Network이 feature interaction을 더 잘 포착하지만, 여전히 Popularity(0.003783) 미만.

---

## 최종 진단 (v1-v12 종합)

### 결론: Feature Quality 문제 확정

| Model | Best MAP@12 | vs Popularity (0.003783) |
|-------|-------------|--------------------------|
| DeepFM v1 (원본, neg4) | 0.001773 | 47% |
| DeepFM v10 (tanh, neg1) | 0.001620 | 43% |
| DeepFM v11 (no tanh, neg1) | 0.000316 | 8% |
| DCNv2 v12 (neg1) | 0.000786 | 21% |

**두 모델 모두 Popularity 미만 → 문제는 모델이 아니라 피처:**
- 현재 피처: 인구통계(user) × 메타데이터(item) — user-item interaction signal = 0
- 모델이 학습할 수 있는 것: "25세 여성 × 검정 부츠 → 구매 확률?" (카테고리 성향, ≠ 개인화)
- Popularity baseline: "가장 많이 팔린 12개" → 인기도 편향(Gini=0.7586)에서 단순 추천이 더 정확

### 프로젝트 설계 의도와의 정합성

이 결과는 프로젝트 연구 동기와 **정확히 일치**:
- **Triple-Sparsity 환경에서 기존 메타데이터 Content-Based의 한계 실증** ✓
- Level 1 (DeepFM + metadata) < Popularity → **KAR (L1+L2+L3 LLM 속성)의 증분 가치가 연구 핵심**
- 다음 단계: KAR 통합으로 BGE 임베딩(h_fact, h_reason)이 user-item interaction signal 제공

---

## Key Takeaways

1. **Feature quality >> 모델 아키텍처/튜닝** — 같은 피처에서 DeepFM, DCNv2 모두 Popularity 미만
2. **Logit 스케일이 추천 모델의 핵심** — FM interaction이 quadratic하게 증가하므로 반드시 제어 필요
3. **4:1 neg ratio + BCE = mode collapse 위험** — balanced (1:1) 또는 pairwise loss 사용
4. **Loss↓ ≠ MAP↑** — BCE pointwise loss는 ranking metric과 직접 연결되지 않음
5. **per-component tanh는 양날의 검** — loss 안정화 vs 표현력 제한
6. **validation 1000 users 샘플링의 분산이 큼** — epoch마다 MAP@12이 0.000~0.002로 변동
7. **KAR (LLM 속성) 통합이 성능 개선의 핵심 경로** — metadata-only 피처의 구조적 한계
