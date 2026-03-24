# H&M LLM-Factor RecSys — Progress Tracking

> 실제 배치 처리(속성 추출, 유저 프로파일링)와 모델 학습은 `scripts/` CLI 엔트리포인트로 실행한다.
> DVC 파이프라인(`mlops/pipeline/dvc.yaml`)은 이 스크립트들을 래핑하여 재현성을 보장한다.

## Project Status: Phase 1 완료 → Phase 2 진행 예정

---

## Phase 0: 데이터 준비 + Baseline (Week 1-2) ✓

- [x] Kaggle 데이터 다운로드 및 검증 (articles 105K, customers 1.37M, transactions 31M)
- [x] EDA 노트북 작성 (00_eda.ipynb) — Plot 퀄리티 + 분석 고도화 완료 (20 figure files)
- [x] EDA 보완 분석 추가 (Data Quality, detail_desc 텍스트 품질, 상품 수명주기, Age×Category, 시즌별 카테고리+Basket, Val/Test Overlap+Recency)
- [x] 시간 분할 구현 (train ~2020-06-30 / val 07-01~08-31 / test 09-01~09-07)
- [x] 고객 필터링 (활성 876K / 희소 421K / cold-start val 46K)
- [x] Baseline: 인기도 Top-12 (Global + Recent)
- [x] Baseline: UserKNN (ALS, factors=128, iter=15)
- [x] Baseline: BPR-MF (factors=128, lr=0.01, iter=100)

## Phase 1: Factual Knowledge 추출 (Week 3-5) — Per-Item 통합 L1+L2+L3

- [x] `src/knowledge/factual/` 모듈 구현 (extractor, prompts, batch, cache, validator, text_composer, image_utils)
- [x] `scripts/extract_factual_knowledge.py` CLI 구현 (Typer)
- [x] `configs/extract/default.yaml` Hydra config
- [x] Super-Category별 통합 프롬프트 설계 (Apparel/Footwear/Accessories × L1+L2+L3)
- [x] JSON Schema 정의 (OpenAI Structured Output, 카테고리당 21 fields)
- [x] product_code 기반 캐싱 + Parquet 체크포인트 구현
- [x] 7종 Ablation 텍스트 조합 (`text_composer.py`)
- [x] 단위 테스트 (166 tests, `tests/unit/test_factual_knowledge/`)
- [x] `ExtractionConfig`, `ExtractionResult` NamedTuple 추가 (`src/config.py`)
- [x] 파일럿 검증 노트북 (`notebooks/01_pilot_extraction.ipynb`) — 7 sections, 15 figures, Go/No-Go 매트릭스
- [x] 500개 파일럿 추출 (실시간 API) + 품질 검증 — Go/No-Go **GO** (5/5 PASS)
- [x] 프롬프트 품질 심층 검증 (`notebooks/01a_prompt_quality_deep_dive.ipynb`) — 7 sections, 10 figures, Health Score 92.9% (v2), Go/No-Go **YELLOW** (Conditional Pass). Material match 96.2% (v2 enum hierarchy + 구조 키워드 제거로 거짓 불일치 해소)
- [x] 프롬프트 개선 v2: visual_weight 재정의, 교차속성 규칙 12개, material/closure enum화, style_lineage 45값 enum, tone_season LLM 제거 (규칙 전용), material_detail 추가, design_details minItems=1, 다양성 지시문
- [x] 전체 47K 제품 배치 추출 (Batch API, 95/95 청크 완료, 실패 0, ~$8.50)
- [x] product_code → SKU 변형 전파 + 색상 규칙 매핑 (`correct_visual_weight()` + `propagate_to_variants()` + `construct_factual_text()` 자동 적용)
- [x] 품질 리포트 생성 (coverage 100%, Error 0.53%, Warning 9.33%)

## Phase 2: 유저 프로파일 (Week 6-7)

- [x] `src/knowledge/reasoning/` 모듈 구현 (extractor, prompts, batch, cache)
- [x] `scripts/extract_reasoning_knowledge.py` CLI 구현 (Typer, pilot/batch/resume 모드)
- [x] `configs/reasoning/default.yaml` Hydra config
- [x] `ReasoningConfig`, `ReasoningResult` NamedTuple 추가 (`src/config.py`)
- [x] L1 직접 집계 파이프라인 (DuckDB 벌크, 지수 감쇠 가중)
- [x] L2+L3 Factorization Prompting 구현 (9-field 구조화 JSON, SYSTEM_PROMPT)
- [x] Sparse user fallback (1-4건 유저, 템플릿 기반 reasoning_text)
- [x] Batch API 통합 (factual/batch.py 재사용, 프로파일 JSONL 준비)
- [x] CustomerCache (Parquet 체크포인트, resume 지원)
- [x] 단위 테스트 (77 tests, `tests/unit/test_reasoning_knowledge/`)
- [x] 파일럿 추출 (200 유저, 실시간 API) + 품질 검증
- [x] 파일럿 품질 검증 노트북 (`notebooks/02_pilot_reasoning.ipynb`) — 10 sections, 41 cells, 9 figures, Go/No-Go 5/6 PASS (비용만 FAIL)
- [x] `docs/prompt_design.md` Factual + Reasoning 통합 문서화 — KAR 비대칭 2종 지식 체계 재구성, Part II 신규 추가 (설계 동기, 3-Stage Pipeline, System Prompt, User Message 3-Section, 9-Field Schema, Text Composition, Sparse Template, 비용 분석, 파일럿 검증 결과, 구현 파일 매핑)
- [x] 파일럿 프로파일 종합 검토 및 개선 — 프롬프트 3건 수정 (quality_price schema/system prompt/slot labels), LLM-as-Judge 평가 도입, Consistency 방법론 개선, 비용 분석 보완, Discriminability threshold 조정, 노트북 Findings 2개 섹션 완성
- [x] `src/eval_prompt/` 평가 프레임워크 구현 (judge, structural, factual, reasoning, report — 57 tests PASS)
- [x] `scripts/eval_factual.py` + `scripts/eval_reasoning.py` CLI 구현
- [x] `docs/evaluation_methodology.md` 평가 방법론 문서
- [x] `notebooks/01_factual_eval.ipynb` + `02_reasoning_eval.ipynb` 새 평가 노트북 (기존 01/01a/01b/02 대체)
- [x] Eval 노트북 JSON 리포트 기반 분석으로 전환 — 스크립트 결과 시각화 전용 (structural 함수 직접 호출 제거)
- [x] `01_factual_eval.ipynb` Markdown 셀 고도화 — Senior MD/DS 이중 관점 해석. Judge 200건 재실행(198건 평가), Overall 4.43/5.0, Pass Rate 90.9%, 6개 Go/No-Go 전부 PASS. 7개 markdown 셀 교체 + 1개 신규(Go/No-Go 최종 판정 + Phase 2 연결)
- [x] Knowledge Case Study 노트북 (`notebooks/03_knowledge_case_study.ipynb`) — 4 Parts (A~D), 51 cells, 4 figures. 실물 사례 기반 정성 분석: Per-Category Deep Dive, Metadata vs LLM 증분 가치, 7종 Ablation 텍스트, Edge Cases, Active/Heavy/Moderate/Niche 유저 프로파일 비교, Knowledge Flow 추적, L1-only vs Full L1+L2+L3 정보 밀도
- [x] `--retry-failed` CLI 플래그 구현 — 2,845건(0.32%) 배치 파싱 실패 재시도 + template_fallback + 최종 parquet 조립. 3개 헬퍼 추출(_prepare_user_data, _collect_failed_ids, _apply_template_fallback), quality_report에 n_active_template_fallback 추가
- [x] 전체 876K 활성 유저 배치 처리 (Batch API) — 873,943건 1차 완료 + 2,845건 `--retry-failed` 재시도 → 876,790건 전원 성공 (fallback 0건), 최종 1,298,206 유저 parquet 조립
- [x] 추론 지식 품질 분석 노트북 (`notebooks/02a_reasoning_quality_report.ipynb`) — 46 cells, 13 figures. Coverage 100%, Completeness 99.99%, Discriminability 0.259 (excellent), Token budget 0%, Judge 4.86/5.0 (n=200, 100% pass). Go/No-Go 6/6 **GO** (stale NO-GO → GO 전환)

## Phase 2.5: Feature Engineering + DeepFM Baseline (Week 7-8)

- [x] `src/features/engineering.py` 구현 — DuckDB 피처 계산 (유저 8수치+3범주, 아이템 2수치+5범주), 네거티브 샘플링
- [x] `src/features/store.py` 구현 — .npz/.json 피처 저장/로드
- [x] `scripts/build_features.py` CLI 구현 (Typer)
- [x] `src/config.py` — FeatureConfig, FeatureResult, DeepFMConfig, TrainConfig, TrainResult 추가
- [x] `configs/` — features/default.yaml, model/deepfm.yaml, train/default.yaml, sweep/deepfm.yaml
- [x] `src/losses.py` 구현 — numerically stable BCE loss (JAX)
- [x] `src/models/deepfm.py` 구현 — DeepFM (Flax NNX, FM+DNN, nnx.List 호환)
- [x] `src/training/trainer.py` 구현 — 학습 루프 (BatchIterator, train_step, score_full_catalog, early stopping, W&B)
- [x] `scripts/train.py` 확장 — deepfm backbone 추가 (--features-dir, --learning-rate 등 CLI 인자)
- [x] 단위 테스트 34개 — test_features/ (engineering 15, store 6) + test_deepfm.py (loss 5, model 5, train 3) ALL PASS
- [x] `CLAUDE.md` 업데이트 — Modeling Conventions (Grain/Distributed/HPO), src↔scripts 강화, 기술 결정 3행 추가
- [x] 피처 빌드 실행 (실제 데이터) — 1,298,206 유저, 105,542 아이템, 121.8M 학습 쌍 (24.4M pos + 97.5M neg)
- [x] Grain 데이터 로더 구현 → NumpyBatchIterator 교체 (`src/training/data_loader.py`) — Grain fork() 데드락 + per-sample __getitem__ 병목 해결, vectorized numpy fancy indexing (<1ms/batch)
- [x] 분산 학습 인프라 (`src/training/trainer.py`) — Mesh + NamedSharding + jax.device_put
- [x] TrainConfig 확장 (num_workers, prefetch_buffer_size) + configs 동기화
- [x] scripts/train.py CLI 인자 추가 (--num-workers, --prefetch-buffer-size)
- [x] CLAUDE.md 학습 일관성 규칙 추가
- [x] 단위 테스트 추가 (test_data_loader.py 21개, test_distributed.py 6개) + test_deepfm.py 수정 — 643 total ALL PASS
- [x] DCN-v2 모델 구현 (`src/models/dcnv2.py`) — CrossLayerV2 MoE + DNN, DeepFMInput 재사용
- [x] LightGCN 모델 구현 (`src/models/lightgcn.py`) — Graph propagation, BCOO sparse adj, index-only input
- [x] Multi-backbone 인프라 — BackboneRegistry (`src/models/__init__.py`), trainer 리팩토링, data_loader IndexOnlyTransform
- [x] `src/losses.py` 확장 — bpr_loss, embedding_l2_reg 추가
- [x] `src/config.py` 확장 — DCNv2Config, LightGCNConfig 추가
- [x] `scripts/train.py` 확장 — dcnv2, lightgcn backbone 분기 + 모델별 CLI 인자
- [x] Config YAML — `configs/model/dcnv2.yaml`, `configs/model/lightgcn.yaml`
- [x] 단위 테스트 — test_dcnv2.py (14), test_lightgcn.py (25), test_deepfm.py (13) ALL PASS (52 total)
- [x] Sequential 피처 파이프라인 구축 (`src/features/sequences.py`) — 유저별 시간순 아이템 시퀀스 `train_sequences.npz` (padded, max_seq_len=50)
- [x] DIN 모델 구현 (`src/models/din.py`) — Target-aware MLP attention over history + static features + DNN
- [x] SASRec 모델 구현 (`src/models/sasrec.py`) — Causal self-attention transformer, position embedding, dot-product scoring, full catalog scoring
- [x] Sequential backbone 인프라 — BackboneSpec `needs_sequence` 플래그, `DINLookupTransform` + `SASRecTransform` (data_loader), sequential train step/scoring (trainer)
- [x] DIN/SASRec CLI + Config YAML — `scripts/train.py` din/sasrec 분기, `configs/model/din.yaml`, `configs/model/sasrec.yaml`
- [x] DIN/SASRec 단위 테스트 — test_din.py (12), test_sasrec.py (16), test_sequences.py (6) — 100 total ALL PASS
- [x] DeepFM 학습 실행 — 9 epochs (early stop), best epoch 6, MAP@12=0.001773, 7,953초 (A100 MIG 3g.40gb)
- [x] PRNGKey save/load 버그 수정 (`_save_model_state`, `_load_model_state`)
- [ ] Level 1 baseline 전체 평가 (scoring 배치화 필요 — per-user 413K에 ~3.5시간)
- [ ] scoring 배치화 (`score_full_catalog` → batched vmap)

### Phase 2.5b: GBDT Re-Ranker Baseline (2-stage)
- [x] `src/config.py` — ReRankerConfig, ReRankerResult 추가
- [x] `src/models/reranker.py` — LightGBM wrapper (train/predict/save/load/feature_importance)
- [x] `src/features/reranker_features.py` — Attribute encoding + feature builder (Base 21D / Full ~127D)
- [x] `src/training/trainer.py` — `extract_stage1_candidates()` 추가 (기존 함수 무수정)
- [x] `scripts/train_reranker.py` — CLI (Typer, --mode base/full)
- [x] `configs/model/reranker.yaml` — LightGBM 하이퍼파라미터
- [x] `pyproject.toml` — lightgbm>=4.0.0 의존성 추가
- [x] 단위 테스트 — test_reranker.py (~20 tests)
- [ ] ReRank-Base vs ReRank-Full 실험 실행

## Phase 3: 세그멘테이션 & 분석 (Week 8-10)

### Tier 1 (완료)
- [x] SegmentationConfig / SegmentationResult NamedTuple (`src/config.py`)
- [x] BGE 임베딩 계산 — `src/embeddings.py` (item 105K, user 1.3M, float16; segmentation+KAR 공유 모듈로 리팩토링)
- [x] 구조화 벡터라이저 — `src/segmentation/vectorizer.py` (L1 ~89D, L2 49D, L3 37D)
- [x] 클러스터링 모듈 — `src/segmentation/clustering.py` (PCA + K-Means + UMAP + silhouette k 선택, StandardScaler + whiten 추가)
- [x] BERTopic 토픽 모델링 — `src/segmentation/topics.py` (UMAP+HDBSCAN+c-TF-IDF)
- [x] 고객 세그멘테이션 — `src/segmentation/customer.py` (5-level: L1/L2/L3/Semantic/Topic, BGE isotropy correction)
- [x] 상품 클러스터링 — `src/segmentation/product.py` (BGE clusters + ARI vs native + cross-category, BGE isotropy correction)
- [x] 분석 모듈 — `src/segmentation/analysis.py` (profiles, discriminative profiling, cross-layer ARI, statistics, effective k, L3 heatmap, excess similarity, topic sensitivity)
- [x] CLI 스크립트 — `scripts/segment.py` (Typer)
- [x] 단위 테스트 — `tests/unit/test_segmentation/` (85 tests ALL PASS)
- [x] 분석 노트북 — `notebooks/04_segmentation_analysis.ipynb` (62 cells, 9 Parts + Part D-2 진단/실험 12 cells)
- [x] Hydra config — `configs/segmentation/default.yaml`
- [x] L2/L3 구조화 벡터 붕괴 진단 — 표현 형식 한계 (정보 가치 아님). 아이템 엔트로피(L2 evenness 0.668, L3 0.760), CLT 수렴(L2 12x 분산 감소), PCA 집중도(PC1: L2 9.9%, L3 12.6%). **Semantic 대조**: reasoning_text(L2 5필드+L3 3필드)→BGE 768D의 eff_k=10.30이 L2/L3 정보 가치 직접 증명
- [x] 전처리 개선 실험 — TF-IDF/CLR/UMAP 3종 비교, best: CLR(L2 eff_k 2.72)/UMAP(L3 eff_k 2.63), 모두 <3.0 MARGINAL. 구조화 best 2.72 vs Semantic 10.30 = 3.8배 → 구조화 벡터 표현력 한계 정량 실증

### Tier 1.5 (Knowledge-Purchase 분석 — 세그멘테이션 보완)
- [x] `src/analysis/` 모듈 구현 (mutual_information, layer_information, preference_diversity, cold_start) — 41 tests ALL PASS
- [x] `scripts/analyze_knowledge.py` CLI (Typer, 5개 컴포넌트: mi, diversity, layer-info, cold-start, ablation-emb)
- [x] 단위 테스트 `tests/unit/test_analysis/` — 41 tests, 기존 테스트 무파괴 (684 total ALL PASS)
- [x] Component A (MI) 실행 완료 — Conditional MI: MI(L3|L1+L2)=0.185 > MI(L2|L1)=0.148. Raw MI: style_lineage(L3) 전체 3위
- [x] Component C (Diversity) 실행 완료 — RVI Top-5 전부 L2/L3: perceived_quality(0.535), season_fit(0.525), coordination_role(0.492)
- [x] `src/analysis/ablation_embeddings.py` — 7종 ablation BGE 임베딩 생성 모듈
- [x] Ablation 임베딩 생성 실행 (7종 × 105K items, ~924MB total)
- [x] Component B (Layer Info) 실행 — CKA: L1↔L3=0.788 (최대 차이), Separation AUC 0.69~0.71 (변형 간 비슷)
- [x] Component D (Cold-Start) 실행 — L1+L2가 2-4건 구간 HR@12=3.28% (최고), Sparse 유저에서 content-based 가장 유효
- [x] notebooks/05a_knowledge_purchase_analysis.ipynb (18 cells, 6 figures) — MI + CKA + Diversity 시각화
- [x] notebooks/05b_knowledge_sparsity_analysis.ipynb (11 cells, 2 figures) — Cold-Start HR@12 시각화
- [x] contribution_notes.md 업데이트 — Contribution 3-9(MI) + 3-10(CKA) + 3-11(Diversity) + 3-12(Cold-Start) + 누적 수치 요약

### Tier 2 (후속)
- [ ] Affinity Matrix 계산
- [ ] 카탈로그 갭 분석
- [ ] Streamlit 대시보드

## Phase 4: KAR 모듈 구현 (Week 11-13)

- [x] BGE 임베딩 모듈 리팩토링 — `src/embeddings.py` + `EmbeddingConfig` (segmentation→공유 모듈 분리)
- [x] KAR Config NamedTuples — `src/config.py` (ExpertConfig, GatingConfig, FusionConfig, KARConfig)
- [x] KAR 손실 함수 — `src/losses.py` (align_loss, diversity_loss, kar_total_loss)
- [x] 5종 백본 embed()/predict_from_embedding() 분리 — DeepFM, DCNv2, LightGCN, DIN, SASRec (backward-compatible)
- [x] BGE 임베딩 인덱스 정렬 — `src/kar/embedding_index.py` (feature index↔BGE id 매핑, 48개 누락 zero-pad)
- [x] Expert Network 구현 — `src/kar/expert.py` (2-layer ReLU MLP, 768→d_rec, dropout)
- [x] Gating Network 구현 — `src/kar/gating.py` (G1 Fixed, G2 Expert-conditioned, G3 Context, G4 Cross + factory)
- [x] Embedding Fusion 구현 — `src/kar/fusion.py` (F1 Concat, F2 Addition, F3 Gated, F4 Cross-Attention + factory)
- [x] KARModel 구현 — `src/kar/hybrid.py` (Composition: backbone+experts+gating+fusion, forward_with_intermediates, get_expert_outputs)
- [x] KAR 데이터 로더 — `src/training/data_loader.py` (4종 KAR Transform: Feature, Index, DIN, SASRec + BGE lookup)
- [x] 3-Stage 학습 파이프라인 — `src/training/trainer.py` (Stage1 BCE, Stage2 align+div backbone frozen, Stage3 end-to-end)
- [x] KAR CLI — `scripts/train.py` (--use-kar, --gating, --fusion, --embeddings-dir 등 12개 옵션)
- [x] Pre-store 파이프라인 — `src/serving/prestore.py` + `scripts/prestore.py` (Expert 출력 사전 계산)
- [x] KAR Config YAML — `configs/kar/default.yaml`
- [x] 단위 테스트 72개 — `tests/unit/test_kar/` (expert 10, gating 17, fusion 15, hybrid 14, losses 8, prestore 4, backbone_embed 5) ALL PASS
- [x] 기존 테스트 무파괴 확인 — 199 tests ALL PASS (72 new + 100 existing backbone + 21 features + 6 distributed)
- [ ] 첫 End-to-End 학습 실험 (DeepFM+KAR, L1+L2+L3) ← **서버에서 진행**

### 서버 실행 커맨드 (Phase 4 마무리)

```bash
# 0. 환경 세팅
conda activate llm-factor-recsys-hnm
pip install -e ".[dev]"

# 1. 단위 테스트 확인 (코드 무결성)
python -m pytest tests/unit/ -v --tb=short

# 2. 첫 E2E 학습 실험: DeepFM + KAR (L1+L2+L3, G2, F2, 3-stage)
python scripts/train.py \
    --data-dir data/processed \
    --features-dir data/features \
    --model-dir results/models \
    --predictions-dir results/predictions \
    --backbone deepfm --use-kar \
    --embeddings-dir data/embeddings \
    --gating g2 --fusion f2 \
    --layer-combo "L1+L2+L3" \
    --d-rec 64 \
    --align-weight 0.1 --diversity-weight 0.01 \
    --stage1-epochs 2 --stage2-epochs 5 --stage3-epochs 3 \
    --stage3-lr-factor 0.1 \
    --no-wandb

# 3. (선택) Pre-store 계산
python scripts/prestore.py \
    --model-dir results/models \
    --features-dir data/features \
    --embeddings-dir data/embeddings \
    --output-dir data/prestore \
    --backbone deepfm
```

### 확인 사항
- `data/embeddings/` — item_bge_embeddings.npz (132MB), user_bge_embeddings.npz (1.8GB) 존재 확인
- `data/features/` — train_pairs.npz, user_features.npz, item_features.npz, feature_meta.json 존재 확인
- Stage별 loss 감소 확인: Stage1(BCE↓) → Stage2(align+div↓) → Stage3(total↓)
- 최종 출력: `results/models/kar_deepfm_best/`, `results/predictions/kar_deepfm_val.json`

## Phase 5: 체계적 실험 (Week 14-17)

기본 설정: **Full L1+L2+L3 / Frozen BGE / G2 Gating / F2 Fusion / DeepFM / Multi-stage**

- [ ] Layer Ablation (7 변형) — L1 / L2 / L3 / L1+L2 / L1+L3 / L2+L3 / L1+L2+L3
- [ ] Gating 변형 (4 변형) — G1 Fixed / G2 Expert / G3 Context / G4 Cross
- [ ] Fusion 변형 (4 변형) — F1 Concat / F2 Addition / F3 Gated / F4 CrossAttention
- [ ] Encoder 변형 (3 변형) — Frozen BGE / Fine-tuned BGE / TF-IDF+Projection
- [ ] Backbone 변형 (5종) — DeepFM / DCNv2 / LightGCN / DIN / SASRec
- [ ] Cold-start 분석 — sparse 유저(1-4건) 성능 vs 활성 유저
- [ ] Gating Weight 분석 — g_fact/g_reason 분포, 유저 세그먼트별 차이
- [ ] 시간 축 분석 — train/val/test 시점별 성능 추이

### Phase 5 실행 커맨드 예시

```bash
# Layer Ablation (7종 — 기본 설정 고정, layer-combo만 변형)
for combo in "L1" "L2" "L3" "L1+L2" "L1+L3" "L2+L3" "L1+L2+L3"; do
  python scripts/train.py \
    --data-dir data/processed --features-dir data/features \
    --model-dir results/models --predictions-dir results/predictions \
    --backbone deepfm --use-kar --embeddings-dir data/embeddings \
    --layer-combo "$combo" --no-wandb
done

# Gating 변형 (4종 — Full L1+L2+L3 고정, gating만 변형)
for g in g1 g2 g3 g4; do
  python scripts/train.py \
    --data-dir data/processed --features-dir data/features \
    --model-dir results/models --predictions-dir results/predictions \
    --backbone deepfm --use-kar --embeddings-dir data/embeddings \
    --gating $g --no-wandb
done

# Fusion 변형 (4종)
for f in f1 f2 f3 f4; do
  python scripts/train.py \
    --data-dir data/processed --features-dir data/features \
    --model-dir results/models --predictions-dir results/predictions \
    --backbone deepfm --use-kar --embeddings-dir data/embeddings \
    --fusion $f --no-wandb
done

# Backbone 변형 (5종)
for bb in deepfm dcnv2 lightgcn din sasrec; do
  python scripts/train.py \
    --data-dir data/processed --features-dir data/features \
    --model-dir results/models --predictions-dir results/predictions \
    --backbone $bb --use-kar --embeddings-dir data/embeddings \
    --no-wandb
done
```

## Phase 6: 서빙 파이프라인 + MLOps (Week 18-19)

- [ ] DVC 파이프라인 (dvc.yaml, params.yaml)
- [ ] FastAPI 서빙 앱 구현
- [ ] Redis 캐싱 구현
- [ ] Prometheus + Grafana 관측성
- [ ] 전체 카탈로그 스코어링 서빙 파이프라인 구현
- [ ] Latency 프로파일링
- [ ] Docker + docker-compose 구성
- [ ] K8s 매니페스트 작성
- [ ] CI/CD (GitHub Actions)
- [ ] Locust 부하 테스트
- [ ] W&B 모델 레지스트리 통합

## Phase 7: 결과 정리 (Week 20-22)

- [ ] 전체 실험 결과 정리
- [ ] 시각화 (논문용 figure)
- [ ] 논문 집필
- [ ] 코드 공개 준비

---

## Parallelization Conventions

| 계층 | 도구 | 적용 대상 |
|------|------|-----------|
| I/O-bound | `ThreadPoolExecutor` | 파일 I/O, API 호출, 유저별 예측 생성 |
| CPU-bound | `ray` | 대규모 배치 속성 추출, 분산 evaluation |
| GPU single-device | `jax.vmap` + `nnx.jit` | 배치 임베딩, Expert forward, 모델 추론 |
| GPU multi-device | `jax.sharding` + `nnx.jit` | 데이터 병렬 학습 (Mesh + NamedSharding) |
| GPU sequential | `jax.lax.scan` | SASRec 등 시퀀셜 모델 루프 |

---

## Key Findings

### 연구 동기: Cold-Start 및 Triple-Sparsity

| 차원 | 수치 | 의미 |
|------|------|------|
| 유저 측 희소성 | 32.1% (436K) 유저가 1-4건 | 카탈로그의 <0.004% 상호작용 |
| 행렬 측 희소성 | 99.98% sparse | MF/GNN 시그널 전파 실패 |
| 시그널 품질 | Popularity > UserKNN > BPR-MF | 희소 → 노이즈 임베딩 → 랜덤 랭킹 |

**대안의 계층 구조:** CF 실패 → Content-Based 필요, 단 기존 메타데이터만으로도 CB 가능
- Level 1: DeepFM + 기존 메타데이터(product_type, colour, age 등) = 이미 Content-Enhanced CF
- Level 4: + KAR(L1+L2+L3) + Reasoning Expert = 본 프로젝트
- 핵심 질문: Level 1 → Level 4의 증분 가치 정량화

> 상세: `docs/cold_start_analysis.md`

### Phase 0 Baseline 성능 (Validation Set, k=12)

| Baseline | MAP@12 | HR@12 | NDCG@12 | MRR |
|----------|--------|-------|---------|-----|
| **Popularity Global** | **0.003783** | **0.044994** | **0.008122** | **0.015481** |
| Popularity Recent (7d) | 0.001917 | 0.029886 | 0.004531 | 0.009449 |
| UserKNN (ALS) | 0.003036 | 0.033901 | 0.006319 | 0.012228 |
| BPR-MF | 0.001308 | 0.016069 | 0.002839 | 0.004924 |

### Phase 0 EDA 주요 발견

| Dimension | Finding | Implication |
|-----------|---------|-------------|
| SKU Structure | 47K products → 105K SKUs (avg 2.2 variants), 51% 1-variant | product_code = L1 grouping key |
| Cold-start | 87% single-purchase pairs, 32.1% users have <5 purchases | Content-based (L1+L2+L3) attributes essential |
| Preference | Avg 7.6 unique types/user, Black dominates | Color/type bias consideration needed |
| Channel | In-store 70.4%, Online 29.6% | In-store 주도; Online(30%) 서브셋에서 sequential modeling 적용 가능 |
| Sparsity | 99.98%+ sparse interaction matrix | Collaborative filtering alone insufficient |
| Long Tail | Gini=0.7586, Pareto: 20.7% items → 80% txn, Top-100=4.6%, Top-1K=18.2%, Top-10K=59.6% | Head(100+ purchases)=42.7% items have CF signal; 57.3% tail items need content-based augmentation (L1+L2+L3 attribute vectors) |
| Segment | Heavy(28+) = 24.4% users → 73.5% txn, Light(1-4) = 32.1% → 3.2% | Segment-aware evaluation needed |
| Data Quality | Customer nulls (FN, Active, age), price Kaggle-normalized | Null-aware features, price = relative only |
| detail_desc | Rich material/construction text, near-complete coverage | LLM L1 extraction feasible for full catalog |
| Lifecycle | Short-lived seasonal ↔ perennial basics coexist | Short-lived = cold-start; new item rate = refresh cadence |
| Age×Category | Distinct age-group preferences visible in heatmap | Age-aware user profiling validates KAR reasoning expert |
| Basket | Multi-item same-day purchases common, outfit-level pairs | L3 attributes should capture complementary relationships |
| Split Overlap | New users/items in Val/Test quantified | Content-based essential for evaluation cold-start |

**관찰:**
- Popularity Global이 모든 메트릭에서 최고 성능 — H&M 데이터의 인기도 편향이 매우 강함
- UserKNN(ALS)이 BPR-MF보다 우수 — ALS의 암시적 피드백 최적화가 BPR보다 효과적
- BPR-MF train AUC 94.12%에도 불구하고 추천 성능 최저 — 과적합 가능성 또는 BPR 학습이 Top-K 추천에 직접 최적화되지 않음
- 전체적으로 낮은 MAP@12 수준 (~0.001-0.004) — H&M의 아이템 수(95K)와 유저 행동의 다양성 반영
- Popularity Recent(7일)이 Global보다 낮음 — 최근 7일 트렌드가 2개월 val 기간의 구매 패턴을 충분히 커버하지 못함

**데이터 범위 결정:**
- **전체 채널 사용** (Online + In-store) — 30% 데이터 손실 방지, Kaggle 원 대회와 동일 조건, cold-start 실험군 보존
- `sales_channel_id`는 피처로 활용 (유저별 온라인 구매 비율, 아이템별 채널 편향)

**시사점:**
- 인기도 편향을 극복하려면 개인화된 속성 기반 추천(KAR)이 필요
- Cold-start 유저 분석 완료 — Triple-Sparsity 분석 및 해결 전략은 `docs/cold_start_analysis.md` 참조
- BPR-MF의 하이퍼파라미터 튜닝 여지 있으나, Phase 0의 목적은 baseline 확보이므로 진행

### Category-Adaptive 3-Layer Taxonomy 설계

- 기존 L1/L3 속성이 의류 전용(neckline, sleeve_type, fit, length, silhouette, proportion_effect)으로 ~18% 비의류에 적용 불가
- `garment_group_name` 기반 3 Super-Category(Apparel/Footwear/Accessories) 라우팅 도입, Non-fashion(0.12%) 제외
- L1: 4 Shared(material, closure, design_details, material_detail) + 4 Category-Specific → 카테고리당 8개
- L2: 7개 전 카테고리 Universal (변경 없음)
- L3: 4 LLM Shared(color_harmony, coordination_role, visual_weight, style_lineage) + 1 Post-processed(tone_season, 규칙 기반) + 2 Category-Specific → 카테고리당 7개
- 7종 Layer Ablation과 완전 호환 — Layer 제거 시 해당 Shared + Category-Specific 동시 제거
- 프롬프트 3-way 분기, 비용 영향 없음 (GPT-4.1-nano + Batch API ~$10)

### Phase 1 파일럿 추출 검증 (500 제품, Go/No-Go)

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Coverage >95% per attribute | >95% | 100.0% (worst) | PASS |
| Validation pass rate >99% | >99% | 100.0% | PASS |
| Enum adherence >99% | >99% | 100.0% (worst) | PASS |
| Batch API cost within budget | <$15.00 | $6.99 | PASS |
| Pilot sample representativeness | No sampling bias | Stratified (intentional over-sample) | PASS |

- `validator.py`에서 `np.ndarray` 미지원 버그 수정 (Parquet roundtrip 시 list → ndarray 변환)
- L2 Perceived Quality: 거의 전부 3 (H&M mid-market). 변별력 제한적이나 데이터 정확성 문제는 아님
- L3 Style Lineage: 207 고유값, 의미적 중복 존재 (Contemporary/Minimalist 변형). BGE 임베딩에서 자연 흡수 예상
- L1+L2+L3 텍스트 max ~89 words ≈ ~120 tokens, BGE-base 512 토큰 한도 내 안전

### Phase 1 프롬프트 품질 심층 검증 (01a_prompt_quality_deep_dive)

**v2 프롬프트 (최종 채택):**

| Check | N Tested | Issues | Rate | Severity |
|-------|----------|--------|------|----------|
| Material vs detail_desc | 237 | 9 | 3.8% | Warning |
| Sleeve vs product_type | 58 | 9 | 15.5% | Warning |
| Season vs product_type | 11 | 0 | 0.0% | Warning |
| Domain Rules (Error) | 500 | 56 | 11.2% | Error |
| Domain Rules (Warning) | 500 | 16 | 3.2% | Warning |
| Empty design_details | 500 | 0 | 0.0% | Warning |
| Entropy Collapse (<0.5) | 310 | 82 | 26.5% | Warning |
| Enum Adherence | 1508 | 0 | 0.0% | Warning |
| Empty material_detail | 500 | 0 | 0.0% | Warning |

- **Health Score: 92.9%, Go/No-Go: YELLOW (Conditional Pass)**
- Material match rate **96.2%**, Enum Adherence **100%**, material_detail/design_details **100%** 완전성
- 주요 Error: `silhouette_x_visual_weight` **51건** (전체 Error의 91%)
- Color Override: Harmony LLM-Rule 일치율 65.4% → 규칙 override 필수

**v2.1 CHECKLIST 미니 파일럿 결과 (실패 → 롤백):**

| Metric | v2 | v2.1 | 변화 |
|--------|-----|------|------|
| Health Score | **92.9%** | 91.3% | -1.6pp |
| Domain Error rate | 11.2% | **13.0%** | +1.8pp |
| silhouette_x_visual_weight | 51 | **58** | +7 |
| fit_x_visual_weight | 3 | **8** | +5 |
| coordination_x_harmony | 16 | **25** | +9 |

- v2.1 CHECKLIST는 nano 모델에서 attention 분산 → 전반적 품질 악화
- **결론: v2 프롬프트 유지, visual_weight 불일치는 규칙 기반 post-processing으로 해결**

### Phase 1 전체 배치 추출 결과 (47,203 제품 / 105,494 아티클)

**배치 실행:** 95/95 청크 완료 (500 requests/chunk), 실패 0, Coverage 100%

| Metric | Pilot (pre-correction) | Full Batch (post-correction) |
|--------|:---:|:---:|
| Error rate | 11.2% | **0.53%** (561건) |
| Warning rate | 3.2% | **9.33%** (9,842건) |

**Error 0.53% 내역 (561건, 6 rules):**

| Rule | Count | Rate |
|------|------:|-----:|
| `coordination_x_visual_weight` | 286 | 0.27% |
| `wearing_x_size_scale` | 258 | 0.24% |
| `sole_x_season` | 10 | 0.01% |
| `function_x_form_factor` | 4 | <0.01% |
| `sleeve_x_season` | 2 | <0.01% |
| `neckline_x_sleeve` | 1 | <0.01% |

- Error 대폭 감소(11.2% → 0.53%)는 `correct_visual_weight()` post-processing 효과 (파일럿 91% 차지하던 `silhouette_x_visual_weight` 해소)
- Warning 9.33%의 99%는 `coordination_x_harmony` 9,738건 — Basic coordination role에 Analogous/Complementary harmony 배정 (패션 해석 모호 영역, BGE 임베딩 수준에서 영향 미미)
- **Validator 역매핑 이슈:** `validate_domain_consistency()`에 Parquet canonical slot (l1_slot4 등) 직접 입력 시 거짓 0% 반환. 역매핑(l1_slot4 → neckline 등) 후 정확한 결과 확보. Phase 5에서 validator에 자동 역매핑 로직 추가 고려

### KAR 원본 비대칭 구조 교정 + Category-Adaptive Downstream 분석

- 문서가 "KAR 원 논문과 동일"이라 하면서 4종 텍스트(Item Factual/Item Reasoning/User Factual/User Reasoning)를 정의하던 오류를 교정
- KAR 원 논문(Fig.3)의 **비대칭 2종 구조** 충실 반영: **Item → Factual Knowledge** (아이템 속성 기술), **User → Reasoning Knowledge** (유저 선호 추론)
- Pre-store 대상을 Gating 이전 Expert 출력으로 변경, Gating은 온라인에서 user-item 쌍별 계산 (linear+softmax, ~1ms 미만)
- Category-Adaptive 속성의 downstream 호환성 확인: BGE-base 인코더가 자연어 수준에서 스키마 이질성 해소, Expert→Gating→Fusion→Backbone 전 구간 차원·아키텍처 변경 불필요
- Category-Adaptive 장점: (1) N/A 패딩 대비 시그널-노이즈 비율 개선, (2) SASRec 시퀀스 내 카테고리 전환 시맨틱 반영, (3) LightGCN 그래프 전파로 교차 카테고리 속성 일관성 클러스터링, (4) User Reasoning에서 LLM의 카테고리 횡단적 형태 선호 종합 (시맨틱 브릿지)

### Phase 2 파일럿 프로파일 종합 검토

**LLM 프로파일 품질 (200명 파일럿):**

| 영역 | 평가 | 핵심 |
|------|------|------|
| LLM 출력 품질 | **Excellent** | 100% completeness, 0% generic, 개인화 우수 |
| 프롬프트 설계 | **Good → 수정 완료** | 3건 수정 (quality_price schema, system prompt rule, slot labels) |
| Consistency 측정 | **Weak → LLM-as-Judge 도입** | 키워드 매칭(0.281)의 구조적 한계, GPT-4.1-mini 평가자 도입 |
| Discriminability | **Excellent** | mean cosine sim 0.14 (threshold 0.3 하향) |
| 비용 분석 | **PASS** | Budget $50→$120 상향 ($105.11 추정, 유저당 $0.00012) |

**프롬프트 수정 3건:**
1. `quality_price_tendency` schema: perceived_quality 앵커링 문제 → price quintile 우선 명시
2. System prompt Rules: price vs quality 구분 규칙 추가
3. Slot6/Slot7 → 의미 레이블 (Silhouette, Proportion Effect 등)

**Cold-start 프로파일링 가치 입증:** Light-Active Buyer (7건)에서도 coherent한 정체성 구성 → LLM 프로파일링이 소량 구매에서도 유의미한 정보 생산

### Phase 3 세그멘테이션 전처리 개선

**전처리 파이프라인 교체 결과** (StandardScaler + PCA whiten + BGE isotropy correction):

| Level | Before (sil) | After (sil) | k Before → After | 변화 |
|-------|-------------|-------------|-------------------|------|
| L1 | 0.287 | 0.007 | 6 → 12 | 이전 고silhouette는 스케일 불균형에 의한 착시. 표준화 후 실제 구조 반영 |
| L2 | 0.204 | 0.472 | 4 → 4 | whitening으로 의미 있는 구조 강화 |
| L3 | 0.532 | 0.011 | 4 → 4 | 이전 0.532는 harmony/tone 지배적 주성분이 K-Means를 왜곡한 결과 |
| Semantic | 0.182 | 0.040 | 4 → 12 | BGE 비등방성 (mean cosine 0.794) 제거 후 더 세분화된 클러스터링 |
| Topic | 5 topics | 10 topics | — | isotropy correction으로 HDBSCAN이 더 많은 밀도 구조 발견 |
| Product | ARI 0.449 | ARI 0.522 | 25 → 30 | 아이템 BGE mean subtraction으로 카테고리 분리 개선 |

**분석 함수 추가**: discriminative profiling (over/under-represented ratio), effective k (entropy-based), L3 37D heatmap, cross-category excess similarity (baseline-corrected), topic min_cluster_size sensitivity

---

## Next Steps

- [x] `prompts.py` / `extractor.py` 데이터 기반 검증 — COLOR_TO_TONE/HARMONY 22개 색상 추가 (13.2% 커버리지 gap 해소), TOE_SHAPE_VALUES "N/A" 추가 (Socks/Tights 2,272건)
- [x] 파일럿 검증 완료 — `validator.py` ndarray 호환 버그 수정, 노트북 코드 버그 6건 수정, Findings 6셀 완성, Go/No-Go 5/5 PASS
- [x] 프롬프트 품질 심층 검증 완료 — 의미적 품질 YELLOW (Health 92.9%), Material match 96.2%, Enum 100%, Domain Error Rate 11.2% (silhouette_x_visual_weight 51건이 91%), Material Check 거짓 불일치 해소
- [x] 프롬프트 v2 개선 완료 — visual_weight 재정의(FORM+VOLUME), 교차속성 규칙 12개 코드화, material/closure enum화 (3종 카테고리별), style_lineage 45값 enum, tone_season LLM→규칙 전용, material_detail 2-Level Hybrid, design_details minItems=1, 다양성 지시문 + 대비 예시. 151 tests PASS. Health Score 목표 ≥90% (GREEN)
- [x] 프롬프트 v2.1 VERIFICATION CHECKLIST — WRONG/RIGHT 마이크로 예시 체크리스트 배치. **500건 미니 파일럿 결과 실패**: Health 91.3% (v2: 92.9%), Domain Error 13.0% (v2: 11.2%). nano 모델에서 attention 분산 유발 → **v2 프롬프트 유지, CHECKLIST 롤백 결정**
- [x] 01a 노트북 v2 스키마 업데이트 — 43→48셀 (enum 준수율 검증 + material_detail 분석 추가), validate_domain_consistency() import 사용, l3_tone_season LLM 비교 제거, SCALAR_ENUM_COLS/ENTROPY_COLS에서 tone_season 제거
- [x] v2.1 미니 파일럿 500건 재추출 + 01a 재실행 — **v2.1 실패**: Health Score 91.3% (v2: 92.9%), Domain Error 13.0% (v2: 11.2%), silhouette_x_visual_weight 58건 (v2: 51건). CHECKLIST가 nano 모델에서 attention 분산 유발. **v2 프롬프트가 우수 → v2로 전체 배치 진행, visual_weight 불일치는 규칙 기반 post-processing으로 해결**
- [x] v2.1 CHECKLIST 롤백 + 규칙 기반 post-processing 구현 — `correct_visual_weight()` (silhouette/fit/coordination → visual_weight 범위 교집합 clamp), VERIFICATION CHECKLIST 프롬프트에서 제거, 166 tests PASS
- [x] Multi-Chunk Batch API 구현 — JSONL 150MB 청크 분할 (200MB 업로드 한도 대비), 다중 배치 제출/폴링/결과 병합, batch_ids.json 재개 지원, 177 tests PASS
- [x] Sequential Batch Pipeline 구현 — org-level enqueued token limit (2M) 대응: 청크당 max 500 requests (1.5M tokens < 2M), `run_batch_pipeline()` 순차 submit→poll, 단일 명령 완전 자동화, resume 지원, 218 tests PASS
- ~~Phase 1 전체 배치~~ ✓ 완료 (95/95 청크, 47,203 제품, 105,494 아티클, Error 0.53%)
- ~~Phase 1 배치 결과 분석 노트북~~ ✓ 완료 (`notebooks/01b_batch_quality_report.ipynb`) — 10 sections, 43 cells, 15 figures. Coverage 100%, Error 0.53%, Warning 9.33%, 전 텍스트 BGE 512-token 이내. Go/No-Go **GREEN**
- ~~Phase 2 모듈 구현~~ ✓ 완료 (`src/knowledge/reasoning/` — extractor, prompts, batch, cache + scripts/extract_reasoning_knowledge.py + 77 tests PASS, 전체 295 tests PASS)
- ~~Phase 2 파일럿 검증~~ ✓ 완료 (`notebooks/02_pilot_reasoning.ipynb` — 10 sections, 41 cells, 9 figures)
  - 9-field completeness 100%, Generic 0.0%, Token 99th=357 (512 이내), Discriminability mean_sim=0.14
  - **비용 이슈**: 측정 기반 추정 $105 (Batch API) > $50 예산. avg_input=1287 tokens, avg_output=277 tokens
  - Consistency score 0.499 (keyword-based, semantic이 아닌 word-level 매칭)
- ~~Phase 2 파일럿 종합 검토~~ ✓ 완료 — Senior LLM Engineer + Fashion Customer 전문가 관점 검증
  - **프롬프트 3건 수정**: (B-1) quality_price_tendency schema에 price quintile 우선 명시, (B-2) Slot6/Slot7→의미 레이블(Silhouette/Proportion Effect 등), (B-3) System prompt에 price vs quality 구분 규칙 추가
  - **Consistency 검증 개선**: 키워드 매칭(0.281)의 구조적 한계 확인 → LLM-as-Judge 평가 도입 (GPT-4.1-mini, 20명 샘플)
  - **quality_price 혼동 정량 분석**: perceived_quality 앵커링으로 실제 price_quintile과 괴리 → 프롬프트 수정으로 해결
  - **비용 분석 보완**: Budget $50→$120 상향 (`ReasoningConfig.max_cost_usd`), 절감 전략 5종 시뮬레이션 추가
  - **Discriminability threshold**: 0.8→0.3 (mean_sim=0.14로 여전히 PASS, 0.8은 무작위 텍스트도 통과하는 너무 관대한 기준)
  - **Go/No-Go 재판정**: 6/6 PASS (비용 FAIL→PASS 전환)
- ~~Prompt Output Evaluation Framework~~ ✓ 완료 — `src/eval_prompt/` (judge.py, structural.py, factual.py, reasoning.py, report.py), `scripts/eval_factual.py`, `scripts/eval_reasoning.py`, `docs/evaluation_methodology.md`, 57 tests PASS. 기존 01/01a/01b/02 노트북 삭제 → 01_factual_eval.ipynb + 02_reasoning_eval.ipynb 대체
- ~~Phase 2: 배치 실패 2,845건 retry 실행~~ ✓ 완료 (`--retry-failed` → 3 retry 청크 Batch API 완료, 876,790건 전원 `llm`, fallback 0건, 최종 1,298,206 유저 parquet 조립, reasoning_coverage=1.0)
- ~~Phase 2: 추론 지식 품질 분석 노트북~~ ✓ 완료 (`notebooks/02a_reasoning_quality_report.ipynb` — 46 cells, 13 figures, 6/6 GO)
- Level 1 baseline 구현 (DeepFM + 기존 메타데이터 피처) — Content-Enhanced CF 기준선 확보
