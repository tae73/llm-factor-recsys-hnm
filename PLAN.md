# H&M LLM-Factor RecSys — Progress Tracking

> 실제 배치 처리(속성 추출, 유저 프로파일링)와 모델 학습은 `scripts/` CLI 엔트리포인트로 실행한다.
> DVC 파이프라인(`mlops/pipeline/dvc.yaml`)은 이 스크립트들을 래핑하여 재현성을 보장한다.

## Project Status: Phase 0–4 완료 → **2026-06 전면 재설계 진행 중** (→ `docs/research_design/redesign_2026-06.md`)

> **2026-06-14 재설계 (Falsification-First):** "성능·자원 문제로 중단" 재진단 → 자원 부족 아님, **단일 검증 scoring 병목**이 원인.
> - **Track B (병목 교정) 완료:** `validate_sample` per-user(batch_size=1) → caller-side 배치(`generate_predictions`/`generate_predictions_kar`, `jax.lax.top_k`). 신규 등가성 테스트 9건 포함 **686 unit pass, 회귀 0**.
> - **Track A Gate-0 (make-or-break) 완료 — 재확인+robustness+적대검증 통과:** L2/L3 증분 가치 probe(confound 해소 `meta.npz`). **C1 +130.9%**(LLM텍스트 ≫ metadata, 5/5 skeptic holds — 단 ~+114%p는 **L1** 단계), **C2 +8.0%**(L2 = **weak·regime-의존**, k=12 cold-start 비유의), **C3 −13.5%·C4 −7.7%**(L3 = **frozen content-retrieval에서만 drop**, learnable gate가 최종 심판). → 3-Layer "다 도움" → **L1 강 / L2 약 / L3 in-model 미결**로 honest reshape.
> - **데이터:** 세션 중 derived data 소실 → raw로 processed/features 재생성 + 사용자 factual knowledge 재업로드 + 본인 embeddings(cuda:1) 재생성 → Gate-0 동일 재현 ✅.
> - **백본 진단·정비 완료(§7):** DeepFM<Popularity는 실재 **FM-scale 버그**(logit 분산 99.2% 점유·sigmoid 포화) + item-collapse + 불공정 eval. 수정 완료(686+20 tests). 단 id-embed는 극단 sparsity에서 **발산**(loss≈4800) → KAR의 사전학습 BGE가 item-identity 메커니즘.
> - **★ pivotal 발견(§8):** Kaggle 대비 10배 낮은 건 데이터·metric 문제가 **아님**. 즉시-다음주 평가 시 **repurchase baseline=0.0243(Kaggle 수준)**. 원인: ① 프레이밍이 H&M 지배 신호 **repurchase+recency를 버림** ② **2개월 gap eval**. GT의 **95.9%가 새 아이템**(discovery). → **Hybrid 재설계 채택**.
> - **Hybrid foundation 구축(720 tests):** immediate-next-period split + repurchase/recent-pop baseline(repurchase=0.0237 재현) + `discovery_map`(새-아이템 GT) metric. (`src/data/splitter.py`, `src/baselines/repurchase.py`, `src/evaluation/cohorts.py`)
> - **★ LLM의 자리 확정(§9, probe_06→07):** content를 *standalone 검색기*로 쓰면 discovery에서 popularity에 짐(0.0013<0.0027). 그러나 **인기 후보 풀의 re-ranker로 쓰면 +93%(0.0038 vs 0.0020)** — content/LLM은 **candidate-gen + ranker 패턴의 ranker**가 제자리. KAR = 학습된 re-ranker. 상세: `redesign_2026-06.md` §7–9, `witnesses/probe_0{5,6,7}_result.json`, `deepfm_audit_result.json`.
> - **모든 KAR 전제 준비 완료:** item/user BGE(user_bge 재생성, mps→cuda 이식성 버그 수정 포함) + ablation + features + immediate-split. **다음 = Gate-2'(trainable KAR as re-ranker, discovery/cold cohort).**

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

### Backbone Overhaul (메타데이터 DeepFM < Popularity 교정 — 5-agent audit 후속)
- [x] Tier1 FM-scale 버그 수정 (`deepfm.py`: `embed_init_std`+`fm_norm`+`_fm_scale`) + 검증 (Task #8, 선행 완료)
- [x] **Tier1 eval 공정성 + cohort 리포팅** (Task #9) — `split_eval_cohorts()` 추가(`feature_capable`=user_to_idx 존재 / `cold_start`), headline=feature_capable, baseline(`scripts/train.py`)도 `--features-dir` 시 **동일 필터**로 cohort eval → apples-to-apples 비교. `evaluate_by_cohort` 연동, `{backbone}_metrics.json`에 headline/cohorts/cohort_sizes/all_users 저장. 단위 테스트 5건 추가
- [x] **Tier2 numerical log-transform + stats 영속화** (Task #10) — z-score 전 heavy count 컬럼(user 3종 + item `total_purchases` max≈44761≈61σ)에 `np.log1p`, `feature_meta` 이름 매핑. per-column mean/std/log1p_cols → `data/features/feature_stats.json` 영속화(inference 재현). in-place 공유로 train/eval 일치. 단위 테스트 3건(log1p가 max|z| ~60σ→<10σ 완화)
- [x] **Tier2 popularity-aware negative sampling + pos_weight** (Task #11) — `FeatureConfig.neg_strategy`(uniform|popularity|mixed) + `_build_item_popularity`/`_sample_negatives`, `scripts/build_features.py --neg-strategy/--neg-mixed-pop-frac`. `binary_cross_entropy(pos_weight=1.0)` + `TrainConfig.bce_pos_weight` + `scripts/train.py --bce-pos-weight`. 기본값으로 동작 불변. 단위 테스트 3건(pop이 인기 아이템 over-sample, 가중 BCE)
- [x] **Tier2 config-gated article_id/user_id 임베딩** (Task #12) — `use_id_embed=True` 시 DeepFM/DCNv2가 per-user/per-item Embed 테이블 획득(메타데이터 동일 아이템 93.5%가 구별 가능). `DeepFMInput`에 optional `user_idx/item_idx`, FM 2nd-order +2 fields(`_fm_scale` 재계산), first-order id bias. `create_train_state`가 `n_users/n_items` 주입, data_loader가 항상 idx 추가, 배치/per-user/KAR scorer 모두 idx 전달, batch→jax dtype heuristic 수정(idx→int32). `--use-id-embed` CLI. 단위 테스트 9건(메타데이터 동일+다른 item_idx→다른 logit, batched==per-user)
- [x] 검증: 타겟 트리플(test_deepfm/test_batched_scoring/test_dcnv2) 48 passed, 전체 unit 706 passed (기존 7 fail만 유지 — test_batch.py 6 + test_partial_coverage 1, 무관). `docs/scripts_tutorial.md` src↔scripts 동기화 완료

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

### ★ 재설계 핵심 발견 (2026-06): L2/L3 "실패"의 진단 + 검증된 pivot

| 단계 | 발견 | 수치 (canonical) |
|------|------|------------------|
| **현상** | L2/L3 정확도 증분 ≈ 0 (multi-task robust) | 13 probe + 적대검증: L1→L2 −0.8%(CI 0포함), complementarity L1+L3 −0.9%(CI 0포함) |
| **원인** | L2/L3 = L1의 (준)함수 = product-internal 재서술 | probe_14: L1→L2/L3 kNN 예측 평균 lift **0.38** (style_mood 0.71/21cls, style_lineage 0.51/44cls) |
| **수정 ①** | L2/L3 = controllable 추천 인터페이스 (L1 불가) | probe_15: steered 정밀도 **1.00 vs 0.14**(gain +0.86), 개인화 100% 유지, 8 제어축 |
| **수정 ②** | LLM 외부 styling 지식 > product-similarity *(de-risk, pair-level)* | probe_16: HR@12 **0.2366 vs 0.2108 +12.2%**(CI [+0.0103,+0.0413]) → KAR open-world 비전 de-risk |
| **⚠️ 정정 (R-9)** | full-scale에서 외부지식이 L1을 **못 넘음** = de-risk REVIVE 반증 | probe_21(100% coverage, $4.03 추출): KAR **0.00424 vs L1 0.00482 −12.0%(0/3)** NO-GO, 둘 다 pop +80~104%, cold-start L1 0.00719 > KAR 0.00568. probe_22: **population-selection bias**(de-risk-pop +14.1% GO vs full-pop −9.1% NO-GO) |

**해석:** "L2/L3 실패"는 3-Layer taxonomy의 무효가 아니라 *LLM 사용 방향(제품 묘사 vs 외부지식)의 문제*. 가치 축이
**정확도 → controllability + 외부지식 complementarity**로 재정의. 셀링포인트 = falsification rigor + diagnosis + validated pivot.
**⚠️ 정직한 final(R-9):** full-scale 빌드(probe 21·22)에서 de-risk REVIVE(외부지식 +17%)는 **population-selection bias false positive**로 반증됨 — LLM 외부지식은 대표·cold-start 유저 discovery를 L1 대비 개선하지 못한다. **견고한 양성 = content L1이 popularity를 +80~104% 능가**. 무결성 가치는 de-risk→scale 반증의 추적 자체.
> 상세: `docs/research_design/STORY.md`, `contribution_notes.md` R-7·**R-9**, `redesign_2026-06.md` §11·**§12 최종 정정**

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
- [x] **Hybrid foundation 구현 (Gate-1')** — immediate-next-period eval split + repurchase/recent-pop baselines + discovery/cohort 평가. probe_05 로직을 production `src/`로 승격.
  - `src/data/splitter.py`: `build_immediate_eval(processed_dir, output_dir, train_end, horizon_days=7)` → `(train_end, train_end+horizon]` 윈도우 GT (`immediate_ground_truth.json`, Kaggle-comparable). `SplitConfig.eval_horizon_days=7` 추가. `run_split(..., build_immediate=True)`, `scripts/preprocess.py --build-immediate/--no-build-immediate` (default on).
  - `src/baselines/repurchase.py`: `recent_popularity`/`repurchase_predict`/`hybrid_predict` (probe_05 그대로, t_dat date/datetime 호환). `scripts/train.py`에 backbone `repurchase`/`recent_popularity` + `--eval-split immediate` 추가.
  - `src/evaluation/cohorts.py`: `activity_cohorts`/`evaluate_cohorts`/`discovery_map`(핵심 신규)/`repurchase_vs_new_decomposition`. per-user AP@k는 `metrics.evaluate` 재사용.
  - **검증 (immediate split, 전체 85,648 유저)**: repurchase MAP@12=**0.02374** (probe_05 0.0243 재현), recent_pop=0.00307, global_pop=0.00351. `discovery_map(repurchase)`=0.00042 (≈0, discovery gap 격리). decomposition new=95.9%/repurchase=4.1%. 신규 단위테스트 23개 PASS.
- [x] **재설계 세션 (Falsification-First, 2026-06) — `docs/research_design/redesign_2026-06.md` §1–11**
  - [x] Gate-0 falsification (probe 01·04): META→L1 +130.9%(강함), **L1→L2 +8% weak, L2→L3 −13.5% harm**. 적대검증(5-skeptic): 자기 이전 "L2 생존" 과대주장 교정. 검증 병목(3.5h/epoch) 배치화(회귀 0).
  - [x] LLM 처분 (probe 06·07·08): content=standalone 검색기 ✗ / **re-ranker +93%** ✓, layer 분해 **L1→L2 −0.8%(CI 0포함)**, L2→L3 −4.6%. → L1=헤드라인.
  - [x] 백본 전면 정비: **DeepFM FM-scale 버그** 수정(logit std 24→1.9), numerical log-transform train/eval 일치, popularity-aware negative sampling, id-embed config-gate(희소 발산 → 비활성 결론).
  - [x] Hybrid 재포지셔닝(R-4): "10× 낮은 MAP" 원인=eval setup(2개월 gap)+repurchase 폐기. immediate-split repurchase MAP@12=**0.02374**(Kaggle-comparable), discovery_map(NEW-only) 신설, new=95.9%/repurchase=4.1%.
  - [x] L2/L3 Fair-Chance (probe 09·10·11·12·13): 4개 설계-fix(encoding·gating·context·complementarity) 전부 NO-GO → **multi-task robust redundancy 확정** (L1_cos 0.226 ≥ L1+L3 0.224).
  - [x] **근본원인 진단 (probe_14)**: L1→L2/L3 kNN 예측 평균 lift **0.38** → L2/L3=L1의 (준)함수(product-internal 재서술). redundancy는 추출 설계의 구조적 귀결.
  - [x] **Reframe ① controllability (probe_15)**: steered 정밀도 **1.00 vs 무제어 0.14**(gain +0.86), 개인화 100% 유지, metadata엔 없는 8 제어축. GO.
  - [x] **Reframe ② 외부지식 KAR (probe_16)**: external_knowledge HR@12=**0.2366 vs L1_cos 0.2108 +12.2%**(CI [+0.0103,+0.0413], 0배제) → KAR open-world-knowledge 비전 첫 실증 = 컨셉 rescue. GO.
  - [x] 문서 종합: contribution_notes **R-7** + 누적 수치 요약 `R+` 행 + 포트폴리오 서사 **`docs/research_design/STORY.md`**(falsification→diagnosis→pivot).
  - [x] **외부지식 일반화 saga (probe 17–20, R-8)**: pair GO(+12.2%) → **frozen NO-GO(−60%)** → **learned fusion REVIVE(+12.6%)** → **placebo control REAL(지식-고유 +0.00080, 5/5)** → format 통일 NEUTRAL(단독) / multi-view ADDS(+0.00045, 3/3). reviewer의 fusion·format 지적이 직접 견인.
  - [x] **Fusion 역할 = 핵심 레버**를 원 설계 `hm_unified_project_design.md` **§7.4.4** 에 명시(학습 Expert projection이 외부지식 가용성 1차 결정, item-side augmentation 결정적, 외부지식 multi-view 권장) + §7.4.2·§8.2.4 교차참조. redesign **§12**.
  - [x] **1b/1c full-scale 빌드 + Gate-2' (probe 21·22, R-9)** ⚠️ **de-risk REVIVE 반증**: 1b 외부지식 **full 추출(47,224 product_code → 105,542 article, 100% coverage, $4.03, gpt-4.1-nano)** → `src/knowledge/external/` + `scripts/extract_external_knowledge.py`. 1c **Two-Tower(DSSM)** 학습(DeepFM은 full-catalog discovery MAP@12=0.000202 ≪ pop 0.00351 실패). **Gate-2' NO-GO(probe_21)**: KAR 0.00424 vs L1 0.00482 **−12.0%(0/3)**, 둘 다 pop +80~104%, cold-start L1 0.00719 > KAR 0.00568. **population isolation(probe_22)**: de-risk-pop +14.1%(3/3) GO vs full-pop −9.1%(1/3) NO-GO → **POPULATION-SELECTION BIAS**(de-risk eligible=인기 3,426 heavy buyer). de-risk REVIVE는 population-편향 false positive. canonical: `witnesses/probe_2{1,2}_result.json`. → contribution_notes **R-9**, redesign **§12 최종 정정**, STORY **§4.3**, design **§7.4.4**.
  - [x] **연구 pivot → Enrichment v2 (`docs/research_design/enrichment_v2_design.md`)**: 가치를 추천정확도(negative)→ **interpretable multi-purpose catalog enrichment**(metadata 없는 *결정 축*; 분석·마케팅·엔지니어링 다각도). de-risk: **D3**(제어=user-intent-driven facet, context +42~190%/off-target −88~99%), **D5**(가치=capability 11 의미축, 예측 아님), **DE1**(기존 20속성 중 salvageable 2개; L2/L3 12개 중 0개; color_harmony/tone_season=metadata 재코딩 lift 1.0). 새 6 gap-axes(trend-phase·price-tier·fine-occasion·outfit-pairing-role·body-fit·care) + 4-use value matrix + H&M/음악 교차. canonical: `witnesses/probe_D{3,5}_result.json`, `probe_DE1_result.json`. **다음:** 새 속성 스키마 설계 + pilot 추출 + DE1 re-screen.
  - [x] **E2-1: 6축 스키마 + 멀티모달 pilot + DE1 re-screen = GO (2026-06-16)** — 새 모듈 `src/knowledge/enrichment_v2/`(schema·prompts(metadata-free 멀티모달, fabric-word strip)·validator·extractor·sampling) + `src/features/behavioral_axes.py`(price-tier·trend-phase·co-purchase outfit-role) + CLI `scripts/{extract_enrichment_v2,build_behavioral_axes}.py` + `witnesses/probe_DE1_v2_new_attributes.py`(DE1 엔진 재사용, two-population, power flag). 10 unit test·ruff/black clean. Kaggle 전체 이미지 105,100 → pilot **500 code·5,354 art·100% cov·$0.093**(gpt-4.1-nano). **DE1 re-screen GO**(5/12 strong-gate, 2 SALVAGEABLE, seed42 byte-identical): 행동파생 `trend_phase`(meta_p 0.10·behav 0.057)·`outfit_role`(meta_p 0.28·behav 0.155) **SALVAGEABLE**, `price_tier` WEAK(inert); **LLM 인식축 9/9 실패**(occasion l1_p 0.85 REDUNDANT·집중은 해결 top1 0.81→0.45; care_burden top1 0.77·trend_look 0.72 CONCENTRATED); gap축 `value_gap`/`trend_gap` 비중복 통과·행동 inert→WEAK. **핵심: metadata 숨기기는 metadata-재코딩·집중을 고쳤으나 LLM 인식축은 L1과 redundant(probe_14 생존); 비중복 결정-축 = 행동 grounding + 인식×행동 gap.** canonical: `witnesses/probe_DE1_v2_result.json`. → contribution_notes **E2-1**, design **§7**. **다음:** 4-use value matrix(통과축 trend_phase·outfit_role·gap) + gap축 행동-검증.
  - [x] **E2-2: 4-use value matrix = E2 GO (2026-06-16)** — 신규 `src/features/{enrichment_matrix,lead_lag}.py` + CLI `scripts/build_enrichment_matrix.py` + `witnesses/probe_E2_value_matrix.py`(4 cell: ①D3-steer·②lead-lag·③η-excess+placebo·④segment-divergence; capability×lift 분리, D3/D5/DE1 엔진 재사용). 6 unit test·ruff/black clean·seed42 재현·**API $0**. **capability 14/16, strong lift PASS 2/16** — 사전예측된 2 cell에만: **trend_phase→②lead-time**(share→sales(t+**3mo**) r=0.472 vs null 0.062, Δ=0.41 **CI[0.19,0.64]**) + **outfit_role→③merch**(η 0.623 vs meta 0.564, excess +0.059 CI[0.046,0.069], placebo 0.003). ①faceted MARGINAL(oracle ctx-steer +97%/+24%, 제어비용 큼); ④audience MARGINAL(practical-margin 미달, D5 재확인); gap축 NO/N/A. **핵심: GO지만 lift는 2 cell 국한 = thesis(가치=예측 아닌 결정-축) 정량 확정**; 실전가치 = momentum 3mo early-warning + co-purchase merch velocity + steerable intent(oracle 상한). 자기참조 차단(④ repurchase·③ placebo). canonical: `witnesses/probe_E2_result.json` + `results/figures/E2_value_matrix.png`. → contribution_notes **E2-2**, design **§5**. **다음:** lead-time/merch 실서비스 시나리오 + gap축 decision-lift 검증 + 음악 교차도메인.
  - [x] **E2-3: value matrix 강화 = lift 2→3 (+1 genuine, ①②④ 정직 반증) (2026-06-16)** — 컬럼당 contribution을 *정직하게*(임계값 불변, 더 나은 target/granularity/outcome) 높이려 재설계. 신규 `src/features/audience_signals.py`(buyer-population 3-way join) + `enrichment_matrix.compute_merch_signals`(markdown/first-week/online) + `lead_lag` weekly/continuous + `witnesses/probe_E2b_value_matrix.py`(컬럼당 PRIMARY 1개·secondary descriptive). 11 unit test·ruff/black·seed42 byte-identical·E2-2 canonical 불변. **lift PASS 2/16→3/16**: **③ `trend_phase`→merch NEW PASS**(tautological velocity NO→launch **first_week_sell_through** η excess **+0.45** CI[0.43,0.46]·placebo 0.008·product_group residualize → 두 행동축 모두 merch); **① 반증**(oracle +97% but deployable history-predictor gain **0.0** — past값 steer, discovery는 새 아이템); **② refinement 반증**(weekly+continuous noisier→monthly PASS 0.41 유지); **④ 반증**(buyer-age div 0.38/0.49 < metadata 1.16·online도 패배 → 축 category-직교라 audience 못 가름). **핵심: ③만 정당 강화, ①②④는 반증 — genuine lift는 축이 metadata 없는 sales/co-purchase 신호 담는 그 지점에만(merch/trend), 제어·audience엔 없음.** canonical: `witnesses/probe_E2b_result.json` + `results/figures/E2b_value_matrix.png`. → contribution **E2-3**, design **§5**. **다음:** 3 PASS cell 실서비스화 + ① trend-aware predictor 탐색 + 음악 교차.
  - [x] **E2-4: KAR user-side leg = 2-source × 4-use 분해 (2026-06-16)** — KAR 비대칭(item→Factual, user→Reasoning)의 user/reasoning leg를 ①control·④audience(둘 다 USER 결정)에 붙임. 신규 `src/features/user_axes.py`(reasoning_bge PCA-50·reasoning_fields TF-IDF+L1agg·demographic 11-feat baseline·FUTURE outcome) + `witnesses/probe_E2c_user_value.py`(11 test, E2/E2b spine 재사용·canonical mtime 불변 assert). CPU·seed42·**API $0**·byte-identical. outcome=held-out **FUTURE**(val 2020-07~08, tautology 차단), baseline=11 demographic. **결과 = KAR-SYMMETRY CONFIRMED**(item→②③ / user→④①): **④ audience modest PASS**(둘 다, STRONG — 오직 `fut_top_group` `reasoning_bge` Δ**+0.0117**(p6e-05)·`reasoning_fields` Δ**+0.0145**(p2.4e-04), ~+1pp at the 1pp bar; `fut_price_tier`는 바 아래·습관축(online/repurchase) NULL→demo 승·④b div 혼재 fields 1.973 PASS·bge 0.353 FAIL); **① control NO**(둘 다, STRONG); **②③ N/A-SEMANTICS**. **★ control① 최종 = capability-PASS / lift-NO (사용자 결정 A)**: D3 precision 1.00 vs 0.14 = 배포 가능 faceted 제어 표면(capability), 자동제어 lift는 item(E2-3 deployable 0.0)·user(E2-4 NO) 양쪽 반증. canonical: `witnesses/probe_E2c_user_value.json` + `results/figures/E2c_user_value.png`. → contribution **E2-4**, design **§5·§7·§8**. **다음:** 3 PASS cell 실서비스화 + gap축 decision-lift + 음악 교차.
  - [x] **C-1: 3 PASS cell → 머천다이징 의사결정-지원 엔진 (build, product-design) (2026-06-16)** — value matrix가 닫은 lift PASS 3 cell(모두 **행동-파생 축**)을 batch 머천다이징 brief로 제품화. 신규 `src/serving/merch_scenarios.py`(NamedTuple `ScenarioConfig`/`ConfidenceCard`/`ScenarioBrief` + `trend_leadtime_brief`/`launch_signal_brief`/`copurchase_velocity_brief` + `build_all_briefs`/`value_matrix_posture`; `lead_lag.py`·`enrichment_matrix.py` feature 함수 재사용) + CLI `scripts/serve_scenarios.py`(Typer, `--scenario all|trend-leadtime|launch-signal|copurchase-velocity`) + `notebooks/06_merch_scenario.ipynb`(builder `notebooks/builders/build_06_merch_scenario.py`, 4 figure) + `tests/unit/test_merch_scenarios.py`(8 test PASS). CPU/DuckDB·**API $0**·ruff/black clean. **정직성: confidence 수치(r·η·CI)는 canonical `probe_E2*.json`에서 로드(재계산 X) → value matrix=single source of truth; lead-lag lag-3 r=0.4723 deterministic 재현(테스트 가드); canonical(`probe_E2*.json`·`E2*.png`) mtime 불변.** 3 brief: **A** trend_phase→lead-time(hot-share z-score 카테고리 조기경보, r=**0.472** vs null 0.062, lag 3mo) · **B** trend_phase→merch(launch first_week_sell_through 스코어카드, η=**0.673** vs 0.223) · **C** outfit_role→merch(anchor 역할 velocity 랭킹+번들 라벨, η=**0.631** vs 0.534). **posture 전체 노출**(capability **14/16** vs lift **3/16**): ① automatic lift·④ audience·gap축(value_gap/trend_gap)·recsys-accuracy negative는 **제품화 X·맥락화 O**. 출력 `results/tables/merch_scenarios/{name}.parquet`+csv, figure `results/figures/06_*.png`. → contribution **C-1** + 누적표 C-1 행, `docs/scripts_tutorial.md`. **다음:** gap축 decision-lift 별도 probe(백로그 b) + 음악 교차도메인(백로그 c).
  - [x] **E2-5: gap축 FUTURE decision-lift = CLEAN NEGATIVE (백로그 b 종결) (2026-06-17)** — gap축(`value_gap`/`trend_gap`)이 E2-1서 capability/비중복은 통과했으나 행동 inert였던 것을, **두 미검증 각도**로 falsify: (1) **FUTURE-held-out** outcome(val 2020-07~08, train-frozen) (2) **자기 구성축 one-hot 대비 incremental**(η-vs-metadata 아님 — gap=c1−c2 collinearity를 one-hot으로 우회해 directional mismatch를 testable interaction으로). 신규 `src/features/enrichment_matrix.build_article_future_outcomes`(per-article FUTURE outcome, `compute_sell_through`/`compute_merch_signals` 재사용, `PRAGMA threads=1` 결정성) + `witnesses/probe_E2d_gap_decision.py`(2 gap축 × 4 결정[markdown-risk·hidden-gem·overhype/sleeper·survival] × 2 readout[incremental paired-fold + decision-rule precision@flag], placebo 2종[within-group·**sign-randomization**], Ridge ΔR²·partial-corr robustness, E2/E2b mtime 불변 assert) + `tests/unit/test_e2d_gap_decision.py`(8 test). CPU·seed42·**API $0**·`--repro` byte-identical·ruff/black clean. **결과 = CLEAN NEGATIVE(5/5 cell 0.01 margin 미달, PRELIM 0, n_val≥5 ~2,017/1,999)**: value→markdown incrΔF1 **+0.0031**(rule lift **0.728<1**·Ridge ΔR² −0.0039)·hidden-gem **−0.0081**; trend→overhype **+0.0087**(p=0.019이나 sub-margin·**sign-rand +0.0067=mean-reversion**·≥20 robust −0.0022·**raw corr +0.107→partial −0.015**); survival 둘 다 deployable flag lift ≤1.03(trend **0.888**=극단 gap 덜 생존). **gap축 = 비중복 *해석* 좌표이나 *예측/결정* 축 아님 확정**(capability YES·forecasting lift NO). 적대 audit 2종(false-negative·suppressed-effect) 독립 통과. canonical: `witnesses/probe_E2d_gap_decision.json` + `results/figures/E2d_gap_decision.png`. → contribution **E2-5** + 누적표, design **§5·§7·§8**, `docs/scripts_tutorial.md`. **다음:** 음악 교차도메인(백로그 c, D4 feasibility 먼저).
  - [x] **R-10: Serendipity/Novelty/Long-tail = enrichment의 마지막 열린 recsys 차원도 CLEAN NEGATIVE (2026-06-19)** — "enrichment가 *추천*으로서의 역할을 할 수 있나"에 답: 정확도(probe_21 −12%)·diversity/coverage(probe_02 −7.7%/−2.3%)는 이미 닫힌 negative라, `redesign §65/§96`이 미측정으로 남긴 **serendipity/novelty/long-tail-hit**("L3의 마지막 rescue 경로")만 열려 있었음 → full-catalog(105,494)·**25,000 user**·discovery-native GT(immediate)로 STRONG falsify. 신규 `witnesses/_probe_common.py`(`item_novelty`·`longtail_exposure`·`tail_hit_count`·`serendipitous_hit_count`) + `witnesses/probe_23_serendipity.py`(7 variant × 6 metric + placebo + 결정-축 특성화) + `tests/unit/test_serendipity_metrics.py`(7 test). seed42·`--repro` byte-identical·**API $0**·prior 29 probe JSON mtime 불변·ruff/black clean. **결과 = CLEAN NEGATIVE(tie at best, never a win, all-NO)**: novelty는 trivially 부풀므로(headline 금지) relevance-grounded **S1·S2·S2b**만 verdict; **적대 audit이 frozen-τ "−60%"가 ~94% labeling 산물임을 발견 → fair S2b(labeling-symmetric) 추가**. matched-HR(−1%) `L1+L2+L3`가 fair **S2b 동률**(rel −0.02, CI[−0.0010,+0.0010] 0포함); L2/L3/EXT는 S2b −0.44~−0.74 LOSE. **placebo random-12가 novelty 19.31·tail-exp 0.81 최고지만 serendipitous hit ≈0**(novelty 함정 실증). cold-start rescue 없음. 적대 audit(trustworthy·not flippable·~10 정의서 unflippable) 통과. canonical: `witnesses/probe_23_result.json` + `results/figures/probe_23_serendipity.png`. → contribution **R-10** + 누적표, redesign **§65·§96** 종결, STORY, `docs/scripts_tutorial.md`. **결론: enrichment 추천 가치지도 전 축 negative — 소비자-추천 역할 종결, 가치는 merchant(C-1)·해석(E2)·제어(probe_15).**
- Level 1 baseline 구현 (DeepFM + 기존 메타데이터 피처) — Content-Enhanced CF 기준선 확보
