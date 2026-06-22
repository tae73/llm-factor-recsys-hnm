# Redesign 2026-06 — Falsification-First 재설계 기록

**상태:** 진행 중 — Track A Gate-0 **완료**, Track B 병목 교정 **완료**, 백본 진단·정비(§7) **완료**. **★ pivotal: H&M repurchase-discovery 구조 발견(§8) → Hybrid 재포지셔닝 채택(§9).**
**작성:** 2026-06-14

> **한 줄 요약:** "성능·자원 문제로 중단" → 실제 원인은 (a) 검증 scoring 병목(교정 완료), (b) DeepFM FM-scale 버그(교정 완료), (c) **H&M 지배 신호인 repurchase+recency를 프레이밍이 버리고 2개월 gap eval로 측정 불가하게 만든 것**(§8). 데이터·metric은 정상(즉시-다음주 repurchase=0.024, Kaggle 수준). → repurchase 백본 + LLM-discovery **Hybrid**로 재설계.

이 문서는 "성능·자원 문제로 중단"된 프로젝트의 전면 점검·재설계 기록이다. research-design OS의 brownfield **revise** 경로로 진행했다. 4원칙(Probe-before-proof / Adversarial honesty / Canonical-invariance / Layered novelty) 채택.

---

## 1. 재진단 — "자원 문제"는 부분적으로만 사실

3-에이전트 정찰 + 직접 코드 검증 결과, 중단 원인에 대한 최초 진단을 **방향교정**한다:

| 최초 진단 | 정직한 재진단 (증거) |
|---|---|
| "성능·자원 문제로 중단" | 자원 부족 아님 — 머신 = 2× L40S(96GB), 48 CPU, 372GB RAM |
| (암묵) 코드 미완성 | 거의 완성 — `src/` 16,365 LOC, 단위테스트 686 pass(+ 신규 9) |
| (암묵) 알고리즘 한계 | 단일 엔지니어링 결함 — 검증 scoring 비배치(`validate_sample(batch_size=1)`) → 413K val users × 105K items = ~3.5h/epoch, 학습:검증 ≈ 1:5 |
| (암묵) 핵심 가설 검증됨 | **미실증 + 측정 불가** — 7개 ablation 임베딩 전부 metadata confound(`text_composer.py:195-198`), `meta.npz` 부재 → "L2/L3 메타데이터 대비 증분"이 디스크만으로 측정 불가 |

**결론:** 전면 재설계 = (A) 핵심 가설 falsification, (B) 병목 1개 교정, (C) 실험 재범위. MLOps는 연구 결과까지 연기.

---

## 2. Track A — Gate-0 de-risk probe (make-or-break) ★

**질문:** LLM L2(체감)/L3(이론)가 metadata·L1 대비 **증분** 추천 가치를 주는가? NO면 핵심 novelty 붕괴.

**방법(학습 0회):** 누락됐던 `meta.npz`를 생성(confound 해소)해 사다리 `META → META+L1 → META+L1+L2 → +L3` 완성. content-based centroid-kNN retrieval(train-history centroid → 전 카탈로그 cosine → top-12)로 held-out val 평가. 모든 variant를 **동일 유저 샘플**로 → per-user paired bootstrap 95% CI. 엔진은 기존 `src/analysis/cold_start.py` 재사용.
- witnesses: `probe_01_incremental_layer_value.py`(50K), `probe_02_l3_isolation_and_diversity.py`(20K), `probe_04_robustness.py`(seed×k×stratum×maxsim), `_probe_common.py`, `build_meta_embedding.py`, `gate0_adversarial_review.workflow.js`(5-skeptic judge panel).
- canonical: `witnesses/probe_01_result.json`, `witnesses/probe_02_result.json`.

**결과 [sketch·합성검증 — data snapshot 기준, 재업로드 후 재확인 필요]:**

pooled HR@12 (50K users): META **0.0069** → L1 **0.0148** → L1+L2 **0.0160** → L1+L2+L3 **0.0138** (비단조)
*(데이터 재업로드 후 동일 seed 재확인 완료 — C1 +130.9%, C2 +8.0%, C3 −13.5%로 재현. 미세차는 BGE GPU 인코딩 last-bit 비결정성.)*

| 사전등록 테스트 | 결과 (probe_01/02) |
|---|---|
| **C1** META → L1+L2 | HR@12 **+130.9%**, CI=[+0.0079,+0.0102] (n=49,998); 2-4 +156% |
| **C2** L1 → L1+L2 (증분 **L2**) | HR@12 **+8.0%**, CI=[+0.0002,+0.0021] (k=12 pooled) |
| **C3** L1+L2 → +L3 (증분 **L3**) | HR@12 **−13.5%**, CI=[−0.0030,−0.0013] |
| **C4** L3 다양성/coverage | diversity **−7.7%**(CI 0 제외), coverage −2.3%, HR guard 실패 |
| **C5** L1+L3(0.0129) vs L1+L2(0.0163) | L3 < L2, 부분 중복 |

**Robustness (probe_04, 3 seed × k{12,50} × stratum × maxsim, 21 cells):**
- **C3(L3 harm): 18/21 음수, 11 유의, 양수 cell 0개** — 매우 견고. *length-confound 반증*(L2 +177자에도 도움; L3 +147자에 해로움 → 부호는 content가 결정).
- **C2(L2 gain): 17/21 양수, 8 유의, 유의-음수 0** — 방향 견고하나 **modest·regime-의존**. heavy 50+ 일부 cell flip(seed7 −14%); **k=12 pooled에선 3/4 seed가 CI 0 포함**, k=50에서 견고해짐.

**적대적 검증 (probe_04 + 5-skeptic judge panel `gate0_adversarial_review`):**
| Claim | 판정 (holds/uncertain/refuted) | 정직한 처분 |
|---|---|---|
| **A** LLM텍스트 ≫ metadata | **5/0/0 holds** | **강한 win** — 단, +130%의 ~+114%p는 **META→L1 단계**(L2 아님). credit은 **L1**. |
| **B** L2 증분 | **2/3/0 → holds_with_caveat** | **weak-but-real** — 방향 견고하나 **배포 지점 k=12 cold-start(2-4)에서 비유의**. "+7.8% CI 0제외"는 n=49,998에서만. |
| **C** L3 drop | **4/1/0 → holds_with_caveat** | **frozen 단위가중 content-retrieval에서만 drop**. learnable gate(KAR Gate-2)가 g_L3→0 가능 → "프로젝트에서 L3 제거"는 **과대주장**. L3≠순수노이즈(L3-alone NDCG > META). |

**방향교정 정직 기록 (2건):**
1. 3K 스모크 C2 −6% → 50K +8% (underpowered 결론 금지 원칙 실증).
2. **본인 이전 보고 "L2 깔끔히 생존(+7.8% GO) / L3 drop"을 적대검증으로 교정**: L2는 *weak·regime-의존*(k=12 cold-start 비유의), L3는 *frozen content-retrieval에서만 drop이고 in-model은 미결*. 과대주장 철회.

**Honest scoping (probes가 *입증한 것* vs *못한 것*):**
- 입증: frozen BGE + centroid-kNN content-retrieval proxy에서 (1) L1이 metadata 대비 큰 증분(+114%, robust), (2) L2가 작은 양의 증분(regime-의존), (3) L3 단위가중 concat은 정확도·다양성 모두 해로움.
- **미입증**: in-model trainable KAR의 거동(probe는 모든 layer를 고정 weight 1.0 unweighted mean-pool로 투입 — 노이즈 subspace down-weight 불가); L3의 *learnable* 신호; k=12에서 L2 최종 유의성. (✅ **frozen-retrieval serendipity/novelty/long-tail은 R-10/probe_23서 측정 = CLEAN NEGATIVE**; in-model trainable 버전만 Phase 5 미실행으로 잔여 미입증.)

**재설계 함의 (reshaped, honest):**
- **HEADLINE (강):** LLM 추출 제품속성(**L1**)이 raw H&M metadata 대비 큰·robust 증분 retrieval (+114% L1 단독). Triple-Sparsity 동기 직결(cold-start 2-4 +0.020, CI 0제외).
- **SECONDARY (hedged):** L2(체감)는 작은·의미적으로 실재하나 regime-의존 증분 — "설정된 +7.8%"가 아닌 **weak-positive**로 보고.
- **METHOD-SCOPED NEGATIVE:** L3는 frozen 단위가중에서 net-harmful → **content-retrieval/pre-store 경로에서 제외**. "L3 무용"·"프로젝트에서 제거"로 표현 금지.
- **L3를 footnote가 아닌 사전등록 in-model falsification으로 전환**(아래 Track C Gate-2).

---

## 3. Track B — 엔지니어링 병목 교정 (완료, 회귀 0)

검증 scoring을 per-user → caller-side 배치로 교체. **모든 backbone에서 동등성 증명.**

- `src/training/trainer.py`: `_score_users_chunk` + `generate_predictions`(전 backbone 배치, `jax.lax.top_k`), `generate_predictions_kar`(KAR 배치, prestored item embedding 재사용 구조), `validate_sample`에서 `batch_size=1` 제거.
  - per-backbone: feature(deepfm/dcnv2/din) = (B×n_items) broadcast; graph(lightgcn) = `get_all_embeddings` 1회 + matmul; sasrec = `score_all_items` 배치.
- `src/config.py` `TrainConfig`: `val_sample_users`(50000, epoch-end), `midval_sample_users`(5000), `pred_chunk_users`(32) 추가.
- `scripts/train.py` + `configs/train/default.yaml` + `docs/scripts_tutorial.md` 동기화(CLAUDE.md 규약).
- **검증:** `tests/unit/test_batched_scoring.py` 신규 9건 — 4개 backbone(feature/DIN/graph/sequence) × batch_size {1,4,32}에서 배치 top-12 == per-user top-12. 전체 **686 pass**, Track B 회귀 0.

> 사전 존재 실패 7건(`test_factual_knowledge/test_batch.py` 6건 = OpenAI Batch API mock, `test_structural.py::test_partial_coverage` 1건 = coverage 로직) — Track B와 무관, 별도 정리 대상.

---

## 4. Track C — 재범위 (예정)

원 1,680 그리드(7×4×4×3×5)를 kill-gated 사다리(~21–32 runs)로 교체. **Gate-0 + 적대검증 반영.**

**사전등록 in-model falsification 규칙 (적대검증 권고):**
- **L3 probation→kill:** L3는 frozen proxy에서만 drop. in-model에서 **kill iff** (학습된 Gate-2 weight `g_L3 → ~0`) **AND** (Full L1+L2+L3 KAR가 cold-start strata(1, 2-4)에서 L1+L2 KAR를 MAP@12/HR@12로 못 넘음, CI 0제외). 그 전엔 L3 유지하되 default 설정은 L1+L2.
- **L2 demote 규칙:** in-model에서 Gate-2가 L2에 안정적 non-trivial weight를 주고 cold-start MAP@12/NDCG@12 lift(CI 0제외)를 못 내면, L2를 "cold-start-only/weak-positive"로 강등(C2_weak rule).
- **discovery-native 메트릭 추가:** HR@12(buy-similar proxy) 외 novelty@12·long-tail-hit·serendipity 측정(L3의 유일한 잔존 rescue 경로 확인). → ✅ **완료 (R-10/probe_23, frozen-retrieval, full-catalog 105K·25K user): CLEAN NEGATIVE (tie at best)** — 5/5 enrichment variant가 S1(long-tail-hit)·S2(frozen-τ serendipity)·**S2b(fair labeling-symmetric serendipity)** 어디서도 L1 못 넘김(matched-HR L1+L2+L3 S2b **동률** rel −0.02 CI[−0.0010,+0.0010]; frozen-τ "−60%"는 ~94% labeling 산물=적대 audit 교정). **placebo(random-12)가 novelty 19.31·tail-exp 0.81 최고지만 serendipitous hit ≈0** = novelty 함정 실증. **L3 rescue 경로 종결.**

| Gate | 내용 | KILL 기준 |
|---|---|---|
| 0 | Track A probe + 적대검증 ✅ | A=강한 win, B=weak-positive, C=frozen-only drop(in-model 미결) — 범위 잠금 완료 |
| 1 | 단일 E2E KAR (DeepFM, **L1+L2** default, G2, F2) | Popularity(0.00378) **및** metadata-DeepFM 못 넘으면 STOP, in-model 버그 진단 (현 DeepFM 0.00177 회귀 포함) |
| 2 | Layer ablation 7종 + `g_L3`/`g_L2` 게이트 가중 분석 | 위 L3/L2 사전등록 규칙으로 layer 처분 확정 |
| 3 | Gating+Fusion marginal | default 유의미 초과 없으면 동결 |
| 4 | Backbone 5종 | ≥2 non-DeepFM lift 없으면 "DeepFM-specific" 정직 보고 |

---

## 5. 데이터 복구 기록 (2026-06-14)

세션 중 `data/{processed,embeddings,features,knowledge}` 소실(사용자 업로드/재구성) → 복구 완료:
- **raw로 재생성(본인):** `data/processed`(preprocess), `data/features`(build_features, 121,871,535 pairs) — canonical 일치.
- **knowledge 재업로드(사용자):** `factual_knowledge.parquet` — fingerprint 이전과 **완전 동일**(105494×27, super_cat 분포·article_id 집합 일치).
- **knowledge로 재생성(본인, cuda:1):** ablation 7종 + `item_bge_embeddings.npz` + `meta.npz`.
- **Gate-0 재확인:** probe_01/02 재실행 → 동일 재현(C1 +130.9% / C2 +8.0% / C3 −13.5%).
- **미업로드:** `reasoning_texts.parquet` (Gate-1 KAR user_bge 필요) → 업로드 시 `segment.py --embeddings-only --bge-device cuda:1`로 user_bge 재생성.

---

## 6. Next Steps

1. **Gate-0 ✅ 완료**: 재확인 + robustness(probe_04) + 적대검증(judge panel) 모두 통과. canonical: `witnesses/probe_0{1,2,4}_result.json`, `gate0_adversarial_review_result.json`.
2. **(사용자 대기)** `reasoning_texts.parquet` 업로드 → user_bge 재생성 → **Gate-1** 가능.
3. **Gate-1**: 단일 E2E KAR(L1+L2 default, G2, F2). `pred_chunk_users` GPU 메모리 맞춰 조정(GPU0 점유 시 `CUDA_VISIBLE_DEVICES=1`).
4. **현 DeepFM 회귀 진단 → §7에서 규명 완료** (정정: 초기 "content-kNN HR@12 0.016이 DeepFM MAP 0.00177을 9배 능가"는 HR/MAP metric 혼용 오류. 철회).
5. **L2/L3 in-model 처분**: §4 사전등록 규칙(`g_L3→0` + cold-start CI 테스트)으로 Gate-2에서 확정.

---

## 7. DeepFM < Popularity 진단 — 실재 버그 + 백본 전면 정비 (완료)

**질문(사용자):** DeepFM(MAP@12=0.00177)이 trivial Popularity(0.00378)에도 진다 — 버그 아닌가?

**5-에이전트 코드 감사 + 직접 재현으로 확정 — 버그 맞음.** canonical: `witnesses/deepfm_audit_result.json`.
- **#1 [BUG]** FM 2차 항 init 스케일 폭주: fm_second std=24 vs dnn 0.34 → **logit 분산 99.2% 점유·sigmoid 포화**(std 0.46) → DNN(유일한 item-aware 경로) 145배 묻힘, gradient 소실. `deepfm.py:137`에 `1/√(n·d)` 스케일 없음.
- **#2 [DESIGN]** item_id/user_id 임베딩 부재 → **93.5% 아이템이 metadata로 구분 불가**(17,255 조합 / 105,542 아이템).
- **#3 [METRIC]** cold-start eval 불공정(feature 없는 11.3%에 `[]`) → 보정 후에도 DeepFM 0.00200 < Pop 0.00380.

**수정(완료, 686+신규20 tests, 회귀 0):** FM-scale 정규화(logit 24→1.9, sigmoid std 0.46→0.28) + id 임베딩(config-gated) + numerical log1p + popularity-aware negatives/pos_weight + eval 공정성·cohort.

**경험적 결과(중요):** id 임베딩 학습이 **발산**(loss≈4800, logit ±수천, val MAP→random). 원인: 1.4M sparse id 임베딩이 학습 중 FM 2차 항을 폭주(init `fm_scale`은 학습 중 성장 미보호). → **극단 sparsity(87% 단일구매)에서 learnable id 임베딩은 불안정**. 이 자체가 §8 reframe과 KAR(content-identity)의 정당화 근거.

---

## 8. ★ H&M Repurchase-Discovery 구조 발견 → Hybrid 재포지셔닝 (pivotal)

**질문(사용자):** Kaggle은 MAP@12 ~0.02–0.035인데 우리는 baseline조차 0.0038, 왜?

**training-free baseline로 직접 측정 → 데이터·metric은 정상.** canonical: `witnesses/probe_05_result.json`.

| eval window | global_pop | recent14 | **repurchase** |
|---|---|---|---|
| **즉시 다음주(Jul 1-7, Kaggle-like)** | 0.00346 | 0.0029 | **0.02427** (Kaggle 상위권 범위!) |
| val (2개월 gap) | 0.00359 | 0.0021 | 0.00855 |
| test (2개월 gap) | 0.00371 | 0.0003 | 0.00319 |

**구조 분해(즉시 다음주):** GT 아이템의 **95.9%가 새 아이템**(repurchase 불가), exact repurchase 4.1%; 구매자의 **6.9%가 완전 신규 유저**.

**왜 10배 낮았나(확정):**
1. **repurchase+recency를 버림** — 프로젝트 "발견 지향, repurchase 아님" 프레이밍이 H&M 지배 신호(즉시 다음주 repurchase=0.024)를 제외. Kaggle 점수 대부분이 이 cheap한 4%+recency에서 나옴.
2. **eval에 2개월 gap** — train 6/30 종료, test 9/1·val 7~8월. 직후 1주 대신 평가 → repurchase 0.024→0.003 (8배 붕괴).
3. 모델 버그(§7)는 부차적 — 신호가 제거된 뒤라 어떤 content 모델로도 복구 불가.

**Hybrid 재포지셔닝 (사용자 채택):**
- **repurchase+recency 백본**: 예측 쉬운 4% + recency를 cheap하게 nail → **~0.024(Kaggle 경쟁력)**. "warm/repeat head는 solved"로 정직 인정.
- **LLM/KAR**: 나머지 **95.9% 새 아이템 발견 + 신규 유저**의 유일한 lever — 실제 구매의 **다수**이자 연구 기여 공간. Gate-0의 L1/L2 content 가치가 정확히 이 자리.
- **eval setup 교정**: 2개월 gap 제거 → **immediate-next-period** 평가, cohort(신규유저·discovery-portion vs repeat) 층화. LLM 가치는 discovery portion에서 측정(pooled 아님).
- 프로젝트의 "발견 지향" 프레이밍은 **옳았음** — 단 ① 경쟁력용 repurchase 백본 누락 ② gap eval로 측정 불가가 문제였음.

---

## 9. Hybrid 재설계 — 실행 계획 (repurchase 백본 + LLM discovery)

**아키텍처 (Kaggle 후보생성+랭커 패턴 + 연구 novelty):**
```
[candidate generation]  repurchase(유저 최근구매) + recent-popularity(+trend) + content/LLM(KAR 아이템 유사)
            │  warm/repeat 4% + recency는 cheap하게 → 경쟁력 점수
            ▼
[ranker]  후보 풀 랭킹. cold/new 유저·새 아이템(95.9%) 구간에서 LLM 속성(L1/L2)이 lever
            │  cohort 층화 평가: 신규유저 / repeat-portion / discovery-portion
```

**즉시 실행 단계 (우선순위):**
1. **[FOUNDATION] eval setup 교정** — 2개월 gap 제거. `src/data/splitter.py`에 *immediate-next-period* 분할 옵션(train ≤ D → predict D+1..D+7). Kaggle-comparable. 기존 2개월 val/test는 보조로만.
2. **[BASELINE] repurchase + recent-pop candidate gen** — `src/baselines/`에 repurchase·recent-popularity·hybrid baseline(첫 목표 ~0.024). cohort·discovery-portion 층화 리포팅(probe_05 로직 → `src/evaluation`).
3. **[REPOSITION] KAR/LLM = discovery lever** — Gate-1을 "KAR가 Popularity를 이기나"가 아니라 **"hybrid에서 LLM discovery 후보·랭킹이 (a) 신규 유저 (b) 새-아이템 portion에서 repurchase-only 대비 증분을 주나"**로 재정의. Gate-0의 L1/L2 content 가치가 정확히 이 측정 대상.
4. **[MODEL] 백본 정비(§7) 활용** — content/discovery 랭커로 DeepFM+KAR(item BGE = 안정적 item-identity, id-embed 발산 회피). id-embed는 극단 sparsity에서 발산하므로 **KAR의 사전학습 BGE가 item-identity 메커니즘**.

**재정의된 kill-gate (Track C 대체):**
- Gate-1': immediate-next-period에서 **hybrid(repurchase+recency)** 가 ~0.02 도달(sanity, 데이터·파이프라인 정상 확인).
- Gate-2': **discovery portion / 신규유저 cohort**에서 LLM-augmented 후보·랭킹이 repurchase-only 대비 MAP/recall 증분(CI 0제외). NO면 LLM의 추천-정확도 기여는 cold/discovery에 한정 또는 다양성으로 재포지셔닝.
- 이후 layer/gating/fusion/backbone ablation은 **discovery cohort에서만** 의미.

**사전 측정 (probe_06, content-discovery, immediate split + discovery_map, frozen):**
| | discovery MAP@12 |
|---|---|
| **popularity floor** | **0.00268** |
| content META | 0.00028 |
| content L1 | 0.00113 (META→L1 **+308%**) |
| content L1+L2 | 0.00130 (L1→L2 +15%, CI 0 근접) |
| content L1+L2+L3 | 0.00114 (L2→L3 **−12.5%**) |
| repurchase | 0.00042 |

- **정직한(sobering) 발견:** frozen content-kNN(LLM 속성 포함)이 discovery에서 **popularity floor(0.0027)를 못 넘음**(L1+L2=0.0013). "과거 구매 유사 아이템 추천"은 새 아이템 예측엔 트렌드(popularity)보다 약함. LLM 속성이 content를 크게 끌어올리나(L1 +308% over metadata, Gate-0 layer 구조 재현: L1강·L2약·L3해로움) **content-kNN 자체가 discovery의 약한 메커니즘**.
- **함의:** content를 *standalone 전-카탈로그 검색기*로 쓰면 discovery에서 popularity에 짐. canonical: `witnesses/probe_06_result.json`.

**probe_07 (★ pessimism 반전 — hybrid re-ranker):** content를 *인기 후보 풀(top-300 recent-popular new items)의 re-ranker*로 쓰면:
| discovery MAP@12 | |
|---|---|
| popularity (pool top-12) | 0.00199 |
| **content-rerank (L1+L2)** | **0.00384 (+93.1%)** |
| hybrid-blend (0.5/0.5) | 0.00299 (+50.6%) |
- **결론: GO.** content/LLM은 *standalone 검색기*가 아니라 **candidate pool re-ranker**로 쓰면 discovery를 **거의 2배**(popularity 대비 +93%)로 만듦. "obscure 유사 아이템"이 아니라 **"트렌딩 아이템을 유저 스타일로 개인화"** 가 정답 — 정확히 Kaggle candidate-gen+ranker 패턴이자 LLM 속성의 제자리. canonical: `witnesses/probe_07_result.json`.
- **확정된 아키텍처 prescription:** KAR/LLM = **candidate pool(repurchase+recent-pop) 위의 학습된 re-ranker**(+ reasoning + CF), standalone 검색기 아님. frozen content-rerank(+93%)는 trainable KAR의 lower bound.

**probe_08 (★ L2 처분 결정 — re-ranker layer 분해):** re-ranker 맥락에서 layer별 discovery MAP@12:
| re-ranker 임베딩 | discovery MAP@12 | 직전 layer 대비 |
|---|---|---|
| popularity | 0.00199 | — |
| META | 0.00294 | (+48% vs pop) |
| **L1** | **0.00387** | META→L1 **+31.5%** (CI 0제외) |
| L1+L2 | 0.00384 | **L1→L2 −0.8% (CI [−0.0004,+0.0003] 0포함)** |
| L1+L2+L3 | 0.00366 | L2→L3 −4.6% |
- **결정:** L2는 buy-similar(Gate-0 +8% weak)에 이어 **re-ranker/discovery에서도 L1 대비 증분 ≈ 0**(−0.8%, CI 0포함). 즉 **두 맥락 모두에서 L2/L3는 L1과 redundant** — 정확도/discovery 증분 없음. (단 L2 단독은 metadata +66% = semantic signal 실재; 문제는 **L1과의 중복**.)
- **기전:** BGE가 텍스트 전체를 인코딩하고 L1 구체속성(material·fit·neckline)이 지각 인상을 이미 결정 → L2 지각 텍스트는 L1에서 추론 가능, 임베딩 신호 거의 안 더함.
- **Contribution framing (초안):** L1 = 헤드라인. L2/L3 = 추상화 redundancy falsification. → **단, "설계 문제 아닌가" 적대 점검(아래 §10)으로 이 결론을 더 정확히 교정함.** canonical: `witnesses/probe_08_result.json`.

---

## 10. ★ L2/L3 Fair-Chance 진단 — "redundant"가 우리 설계 인공물인가? (probe 09–12)

사용자 질문("우리 설계 문제 아닌가")에 따라, "L2/L3 redundant" 결론을 내리기 전 **5가지 설계 결함을 싸게 제거하고 L2/L3에 공정한 기회**를 줌. (전제: L2/L3는 내재적으로 무익하지 않음 — MI(L3|L1+L2)=0.185, RVI Top-5 전부 L2/L3, temporal stability 0.80–0.87, 추출 품질 L1보다 높음.)

| probe | 결함 검증 | 결과 | 판정 |
|---|---|---|---|
| **09** separate-layer 인코딩 | #1 single-text blend | sep_L1L2L3 vs L1-only **−1.6%(CI 0포함)**; vs metadata-control +8.8% | **NO-GO** — 인코딩 형식이 문제 아님, L1이 subsume |
| **10** optimal layer 가중 | #3 layer gating 부재 | best weights **[0.6,0.2,0.2]**, +5.5%(**CI 0포함**) | **NO-GO** — layer gate는 marginal·비유의 |
| **11** context(season/occasion) | #5a context-free task | season **−42%**(해로움), occasion +0.6%(중립) | **NO-GO** — popularity가 이미 계절성 포착·catalog 동질 |
| **12** coordination(co-purchase) | #5b single-item vs outfit | **co-purchase가 L3 속성 공유 ≫ random**: color_harmony +0.080, style_lineage 1.8×, tone_season +0.032, L3 cosine +0.025 (**전부 SIG**) | **GO** — L3는 실제 outfit coordination 포착 |

**★ 정직한 결론 (교정):**
- **single-item next-purchase retrieval(우리가 평가하는 task)에선 L2/L3가 L1과 redundant** — 설계 fix(separate encoding·layer gating·context)로도 안 살아남(09–11 NO-GO). 이유: L1이 similarity 신호를 subsume + H&M mid-market catalog 동질성(All-season 56%, Everyday 지배) + popularity가 context 포착.
- **그러나 L2/L3는 무익하지 않다 — task가 잘못됐다(probe_12 GO).** L3는 **실제 outfit coordination 구조**를 포착(co-purchased 아이템이 color_harmony/style_lineage를 유의하게 공유). 정보이론 신호(MI/RVI/stability)는 옳았음 — 단 그건 **coordination/complementarity 정보**이지 single-item-similarity 정보가 아님.
- **즉 진짜 설계 결함은 encoder/아키텍처(#1–4)가 아니라 TASK/EVAL 설계(#5).** L3는 outfit 코디용인데 우리는 single-item similarity만 평가 → L2/L3 가치가 task에 보이지 않음.

**probe_13 (★ Track A make-or-break — complementarity 랭킹):** probe_12의 coordination 신호(상관)가 실제 *랭킹*에 쓰이는지 검증. held-out co-purchase (seed,complement) cross-category, [complement]+인기 cross-cat distractor 100개 후보 풀, complement의 HR@12.
| scorer | HR@12 | |
|---|---|---|
| popularity | 0.010 | (인기 distractor를 위로 → random 이하) |
| **L1_cos (유사도)** | **0.226** | 최고 |
| L3_cos | 0.216 | < L1 |
| harmony_match | 0.192 | |
| **L1+L3** | 0.224 | **L1 대비 −0.9% (CI 0포함)** |
| L1+harmony | 0.185 | L1 대비 −18% (해로움) |
- **결과: Track A NO-GO.** complementarity(L3의 "본거지")에서도 **L1이 지배하고 L3는 L1 위에 증분 0**. probe_12의 coordination 신호는 실재했으나 **L1이 cross-category co-purchase 구조(일관된 스타일/품질)도 더 잘 잡아** L3가 랭킹에 보탬이 안 됨. canonical: `witnesses/probe_13_result.json`.

**→ 최종 contribution framing (probe 01–13 + 적대검증, 가장 robust한 negative):**
- **L1 = LLM-속성 기여.** 구체 제품속성이 metadata 대비 content 추천을 압도(+114% retrieval·+95% discovery·complementarity HR 0.226 vs popularity 0.010), **테스트한 모든 task에서 지배**.
- **L2/L3 = robust redundancy.** single-item discovery + **complementarity/coordination** + 4개 설계-fix(separate encoding·layer gating·context·재인코딩 가능) + 적대검증 — **전부에서 L1이 subsume**. 정보이론 비중복성(MI/RVI)·coordination 상관(probe_12)은 실재하나 **L1 대비 랭킹 증분으로 번역되지 않음**(homogeneous mid-market catalog에서 concrete 속성이 recommendation-relevant 신호를 대부분 보유).
- 즉 §10 초안의 "complementarity 미개척 방향"도 probe_13으로 **검증 후 NO-GO**. L2/L3는 "task-mismatch로 살릴 수 있음"이 아니라 **다중 task에서 검증된 redundant**. canonical: `witnesses/probe_0{9,10,11,12,13}_result.json`.

**미해결/대기:** `reasoning_texts.parquet` 업로드(user_bge → KAR user-side) ✅ 진행 중. Gate-2' = trainable KAR가 discovery에서 popularity를 넘는지(immediate split, discovery_map, cold/new cohort).

---

## 11. ★ 근본원인 진단 + 컨셉 재정의 — 왜 L2/L3가 실패했나, 그리고 무엇이 작동하나 (probe 14–16)

사용자 질문("B(redundant) 확정이면 내 연구가 실패한 건데 — 실패의 이유와 컨셉부터 다시")에 따라, "L2/L3 정확도 증분 0"이라는 **현상(what)** 을 넘어 **원인(why)** 을 규명하고, 그 원인에 대한 **수정(fix)** 을 싸게 de-risk했다. 세 reframe 모두 GO.

### 11.1 진단 — L2/L3는 "외부 지식"이 아니라 "제품-내부 재서술"이다 (probe_14)

근본 가설: L2/L3가 L1 대비 증분 0인 이유는 **프롬프트가 LLM을 "the product itself / the image"에 한정**(`knowledge/factual/prompts.py`)했기 때문. L2("Casual/Minimalist")·L3("I-line/Monochromatic")는 L1("cotton/slim/crew")을 결정하는 **같은 제품**에서 추론되므로, 임베딩 공간에서 **by-construction redundant**. → L2/L3가 L1의 (준)함수임을 직접 측정.

**방법:** L1 임베딩(`l1.npz`)으로 각 L2/L3 속성을 kNN(k=15, train 20K/test 5K)으로 예측 → majority-class baseline 대비 정확도·lift(= gap-to-perfect 중 L1이 메우는 비율).

| 속성 | kNN(L1) acc | majority | lift | classes |
|---|---|---|---|---|
| l2_style_mood | 0.710 | 0.448 | 0.48 | 21 |
| l2_season_fit | 0.758 | 0.564 | 0.44 | 5 |
| l3_color_harmony | 0.639 | 0.336 | 0.46 | 8 |
| l3_visual_weight | 0.736 | 0.507 | 0.46 | 5 |
| l3_style_lineage | 0.513 | 0.154 | 0.42 | 44 |
| (10개 평균) | — | — | **0.38** | — |

- **결과: 진단 CONFIRMED.** L1-최근접 아이템만 보고 L2/L3를 majority보다 **평균 lift 0.38**(L2 0.37 / L3 0.38)로 예측 — 44-클래스 style_lineage조차 0.51, 21-클래스 style_mood 0.71. **L2/L3는 L1의 (준)함수 = product-internal 재서술**이고, 따라서 정확도 redundancy는 **고칠 수 있는 측정 버그가 아니라 추출 설계의 구조적 귀결**. canonical: `witnesses/probe_14_result.json`.
- 즉 §9–10의 "redundant"에 대한 **mechanistic 설명 확보**: 인코더/아키텍처/task를 바꿔도 안 살아난 이유 = 애초에 L2/L3 텍스트가 L1과 같은 제품-내부 정보를 재서술. **KAR의 진짜 약속(open-world knowledge)은 한 번도 시도된 적 없음.**

### 11.2 Reframe ① — controllable/steerable recommendation (probe_15, 무료, 정확도-무관 축)

L2/L3의 가치를 **정확도가 아닌 제어(control) 축**에서 시연(설계 §7.9가 약속했으나 미측정). L1/metadata엔 없는 **8개 의미 제어축**(occasion·mood·season·quality·trendiness·versatility·coordination·visual_weight). 추천을 target 속성(예: occasion=Party)으로 soft-steer → top-12 precision vs 무제어 baseline rate, 개인화(유저 L1 centroid) 유지 여부.

| target | steered 정밀도 | 무제어 baseline | control gain | 개인화(steer>rand) |
|---|---|---|---|---|
| occasion=Party | 1.00 | 0.02 | +0.98 | ✓ (0.886>0.853) |
| occasion=Work | 1.00 | 0.09 | +0.91 | ✓ |
| season_fit=Summer | 1.00 | 0.13 | +0.87 | ✓ |
| style_mood=Minimalist | 1.00 | 0.48 | +0.52 | ✓ |
| (전체 평균) | **1.00** | **0.14** | **+0.86** | **100%** |

- **결과: GO.** L2/L3로 추천을 의미축으로 **steered 정밀도 1.00 vs 무제어 0.14**로 제어 가능하며, steered 아이템은 "그냥 attribute-t 아이템"보다 유저에 더 가까움(제어 + 개인화 동시, 100% target에서 유지). **metadata엔 없는 capability** — L1만으론 "Party occasion으로 추천"이 표현 불가. canonical: `witnesses/probe_15_result.json`.

### 11.3 Reframe ② — 외부지식 KAR (probe_16, ~$0.1, KAR 원래 비전 실현) ★

진단의 **근본 수정**: 제품 묘사가 아니라 *제품·상호작용에 없는 외부 styling 지식*을 LLM에서 끌어낸다. 새 프롬프트로 600 seed 아이템에 대해 LLM(GPT-4.1-nano)이 **보완 아이템(다른 카테고리)을 완성하는 styling 지식**을 생성 → BGE 임베딩 → held-out cross-category co-purchase 보완을 **L1 product-similarity(probe_13 우승) 대비** 랭킹.

| scorer | HR@12 | MRR |
|---|---|---|
| popularity | 0.0158 | 0.0139 |
| **L1_cos_sim** (제품유사도) | 0.2108 | 0.1045 |
| **external_knowledge** (LLM styling) | **0.2366** | **0.1096** |

- **결과: GO.** 외부 styling 지식이 product-similarity를 **+12.2%**(delta +0.0258, **bootstrap CI [+0.0103, +0.0413] — 0 배제, 유의**) 능가. 예: seed=검정 양말 → "pastel silk/satin slip dress, nightwear look" (양말 묘사에선 못 끌어내는 보완 지식). **진단(제품-내부라 실패)에 대한 수정(외부지식)이 실제 작동 = KAR open-world-knowledge 비전이 원리적으로 실현 가능** → 컨셉 rescue. canonical: `witnesses/probe_16_result.json`.
- 정직한 scope: 600 seeds·3606 pairs·complementarity 한정 **de-risk(개념증명)**. full KAR end-to-end 통합·스케일·다중 task 검증은 future work. 캐시(`data/knowledge/external/complement_knowledge.parquet`)로 재현 시 무과금.

### 11.4 종합 — "실패"가 아니라 falsify → 진단 → 검증된 pivot

| 단계 | probe | 결과 |
|---|---|---|
| **현상(what)** L2/L3 정확도 증분 0 | 01·04·06·08·09·10·11·12·13 + 적대검증 | multi-task robust redundancy |
| **원인(why)** 제품-내부 재서술 | **14** | L1→L2/L3 lift 0.38 (CONFIRMED) |
| **수정 ①** control 축 | **15** | steered 1.00 vs 0.14, 개인화 유지 (GO) |
| **수정 ②** 외부지식 | **16** | +12.2% CI-유의 vs L1-sim (GO) |

- 원래 가설("L2/L3 추상 속성이 정확도를 올린다")은 **반증**됐으나, 그 반증을 **mechanism까지 진단**하고, 진단이 가리키는 **두 수정(control·외부지식)이 모두 작동**함을 보였다. 이는 단순 negative보다 강한 결과 — **연구 성숙도(falsification rigor + diagnosis + validated pivot)** 자체가 포트폴리오 셀링포인트.
- contribution framing 최종: **R-7**(contribution_notes), 서사 문서 `STORY.md`.

---

## 12. ★ 외부지식 일반화 saga — pair GO → frozen NO-GO → learned-fusion REVIVE (probe 16–19, R-8)

option-1(외부지식 KAR for discovery)의 full 빌드($5-10+GPU) 정당화 여부를 falsification-first로 게이트. **방향이 두 번 뒤집힌 뒤 placebo로 확정** — 그 과정 자체가 방법론 사례.

| probe | 무엇 | 결과 | 판정 |
|---|---|---|---|
| **16** | pair-level (seed→complement 랭킹) | external HR@12 0.2366 vs L1 0.2108 **+12.2%**(CI 0제외) | GO |
| **17** | user-level discovery, **frozen** re-ranker | external 0.00104 vs L1 0.00265 **−60.7%** | NO-GO (겉보기) |
| **17b** | 적대 검증 (max-sim + cross-PG) | ext_maxsim 0.00101 vs L1 0.00289, cross-PG도 패배 | NO-GO robust (frozen 한정) |
| **18** | **learned fusion** two-tower (A/B/C) | A 0.00405 / B(+ext user) 0.00416 / **C(+ext both) 0.00456 +12.6%** | REVIVE |
| **18b** | ★ placebo control (5 seed) | **C_real−C_shuffle=+0.00080(5/5)**, C_shuffle−A=+0.00013(용량≈0) | **REVIVE REAL** |
| **19** | format 통일 (external→L1 속성, frozen) | ext_unified 0.00051 < ext_prose 0.00078 « L1 | FORMAT NEUTRAL |
| **20** | learned fusion × format (단독 vs multi-view) | C_unified 0.00390≈A / **C_both 0.00526 > C_prose_dup 0.00480**(3/3) | MULTI-VIEW ADDS |
| **21** | full-scale Gate-2' (100% coverage) | KAR 0.00424 vs L1 0.00482 **−12%**, 0/3 | NO-GO |
| **22** | population isolation | de-risk-pop **+14.1% GO** / full-pop **−9.1% NO-GO** | POPULATION BIAS |

**★ 결론:**
- **외부지식은 pair-level AND user-level discovery 모두에서 유효** — 단 *learned fusion(projection + item-side augmentation)* 전제. frozen raw-cosine NO-GO(−60%)는 **대표적이지 않은 fusion으로 인한 거짓 음성**이었음(probe_17→18 반전).
- 이득은 **용량이 아니라 외부지식의 *내용*** — C_shuffle(같은 차원·분포, item↔knowledge 대응만 파괴)은 이득 0(+0.00013), C_real만 +0.00080(5/5 seed) → placebo-controlled REAL.
- reviewer 가설("frozen 실패=styling/L1 텍스트 *장르 불일치*")은 합리적이었으나 **format 통일로는 안 풀림**(probe_19 NEUTRAL) — lever는 source-format이 아니라 learned projection.
- **방법론 교훈:** de-risk는 *대표적 fusion*으로 해야 하고(frozen으로 죽이면 거짓 NO-GO), lift는 *placebo(shuffle/noise)+multi-seed*로 capacity와 분리해야 한다. falsification-first + 대표성 + placebo.

**→ option-1 부활:** full 빌드(1b 47K 추출 + KAR external-Expert + 1c end-to-end 학습 + Gate-2') 정당화. de-risk 규모(3,426 추출·pool=5K·frozen BGE 위 소형 tower)에서 +17% 지식-고유이므로, full KAR로 확대 검증이 다음 단계. canonical: `witnesses/probe_1{6,7,7b,8,8b,9}_result.json`, `probe_20_result.json`.

**설계 반영:** fusion/Expert의 역할(학습 projection이 외부지식 가용성의 1차 레버, item-side augmentation 결정적, 외부지식은 multi-view 텍스트 권장)을 **원 설계 문서 `hm_unified_project_design.md` §7.4.4** 에 명시(+ §7.4.2·§8.2.4 교차참조). 원 설계의 "F2-Addition 기본"은 *결합 연산*으로 유효하나 가치는 그 앞단 Expert가 만든다는 점을 보강.

**자원 메모:** GPU1 free ~8.7GB(rtb_ipinyou train.py가 23GB 점유 중) — 소형 probe는 무관하나 1c full 학습은 cuda:1 8.7GB 내 맞춤 또는 rtb 잡 종료 대기 필요.

**★ 최종 정정 (full 빌드 후, probe 21·22):** 위 "→ option-1 부활" 결론은 **full-scale 빌드에서 반전됨**. 1b에서 외부지식을 **100% coverage($4.03, 47,224 product_code → 105,542 article)**로 추출하고 1c Two-Tower(DSSM)로 학습한 결과, **외부지식 KAR은 대표·cold-start 유저 discovery에서 L1을 능가하지 못한다** — probe_21(full coverage 105,494, eligible 40,000, 3 seed): **KAR 0.00424 vs L1 0.00482 = −12%(0/3) Gate-2' NO-GO**, cold-start cohort에서도 L1 0.00719 > KAR 0.00568. probe_22 isolation(coverage를 full로 고정·population만 변경)이 원인을 **population-selection bias**로 확정: de-risk eligible(인기 3,426 아이템 heavy buyer, active 11,888 vs sparse 112)에선 KAR +14.1%(3/3) GO이나, 동일 coverage의 full-population에선 −9.1%(1/3) NO-GO. **즉 de-risk REVIVE(+17%)는 population-selection-bias false positive였다** — R-8 placebo는 *capacity*만 통제했고 이 selection bias는 통제하지 못했다(full 빌드가 비로소 잡아냄). **견고하게 남는 양성 결과: content(L1 Two-Tower DSSM)가 popularity를 +80~104% 능가**한다는 사실이다(외부지식의 추가 가치가 아님). 정직한 final = LLM 외부지식은 본 데이터의 일반 discovery 추천기에 L1 대비 증분 가치를 주지 못하며, **de-risk→scale 반증의 추적 자체가 본 단계의 무결성 기여**다. canonical: `witnesses/probe_2{1,2}_result.json`, `data/knowledge/external/external_knowledge_full.parquet`.
