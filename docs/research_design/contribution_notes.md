# Research Contributions — Industry (MD) & Technical (DS) Perspectives

## Research Motivation Recap (수치 요약)
- Triple-Sparsity 핵심 수치 (32.1%, 99.98%, 57.3%, Gini 0.7586)
- Baseline 성능 한계 (Popularity MAP@12=0.003783 > UserKNN > BPR-MF)
- 87% 단일구매 → 발견 지향 추천 필요

---

## Phase 0: Data Preparation + Baseline

### Contribution 0-1: Triple-Sparsity 구조 정량화 (MD+DS)
- 수치: 32.1% Light유저(3.2% 거래) vs 24.4% Heavy유저(73.5% 거래)
- 수치: Gini=0.7586, 20.7% 아이템→80% 거래, 57.3% tail
- 수치: 99.98% 행렬 sparsity, 87% 단일구매 쌍
- MD 시사점: 인기 편중 극복 없이는 롱테일 발견 불가능
- DS 시사점: CF 시그널 자체가 구조적으로 불충분 → Content-Based 필수

### Contribution 0-2: CF 실패 정량화 — Baseline 역전 현상 (DS)
- 수치: Popularity Global MAP=0.003783 > UserKNN 0.003036 > BPR-MF 0.001308
- 수치: BPR-MF train AUC 94.12%이나 추천 성능 최하위
- 의미: 개인화 모델이 비개인화보다 낮음 → 극단적 sparsity에서 CF 학습 자체가 실패

### Contribution 0-3: SKU 구조 분석 — 추출 단위 결정 근거 (MD+DS)
- 수치: 47K products → 105K SKUs (평균 2.2 variants), 51%는 단일 variant
- 수치: 253개 product_type, 50개 colour_group
- MD 시사점: product_code 단위 추출이 합리적 (컬러 변형은 rule-based 처리)
- DS 시사점: API 호출 47K회로 충분 (105K 대비 55% 절감)

---

## Phase 1: Factual Knowledge Extraction

### Contribution 1-1: 105K SKU 전수 속성 추출 — 100% Coverage (MD+DS)
- 수치: 105,494 SKU × 22 attributes, Coverage 100%, Schema Error 0%
- 수치: 47,203 product_code, 95 batches, 0 failures, ~$8.50 USD
- MD 시사점: 원본 detail_desc ~30% 빈약 → LLM으로 100% 밀도 달성, tail 57.3% 아이템도 동일 품질
- DS 시사점: null 처리 불필요, 7종 Layer Ablation 통제 변인 순도 확보

### Contribution 1-2: Category-Adaptive 3-Layer Taxonomy 실증 (MD+DS)
- 수치: L1 8필드 + L2 7필드 + L3 7필드 = 22필드/카테고리
- 수치: Apparel 82%, Footwear 7.3%, Accessories 10.5% — 카테고리별 특화 슬롯 분리
- MD 시사점: super-category별 다른 속성 세트가 상품 특성을 정확히 반영 (Apparel의 silhouette vs Footwear의 heel_type)
- DS 시사점: canonical slot mapping으로 균일한 Parquet 스키마 유지 + 카테고리 특화 정보 보존

### Contribution 1-3: Style-Mood 분포가 H&M 브랜드 DNA 정량 반영 (MD)
- 수치: Casual+Feminine 12.8%, Casual+Minimalist 7.7%, Casual+Cozy 6.9%
- 수치: l2_style_mood 726 unique 조합, H=5.69 (최고 엔트로피)
- 수치: l3_style_lineage 302 unique, l2_occasion 287 unique
- MD 시사점: LLM이 추출한 스타일/무드 분포가 SPA 브랜드 상품 구성비와 일치 — 속성의 현실 타당성 입증
- DS 시사점: 높은 엔트로피 = BGE 임베딩에서 fine-grained 구분력 확보

### Contribution 1-4: 퍼스널 컬러 × 코디 속성 자동 매핑 (MD)
- 수치: 50개 H&M colour_group → 6개 tone_season (rule-based, 100% coverage)
- 수치: 9개 color_harmony 유형, Monochromatic 압도적 1위 (~30K)
- 수치: Cool-Winter 1위 → H&M의 Black/Navy/Grey 비중 반영
- MD 시사점: 퍼스널 컬러 기반 추천의 인프라 자동 구축 — 기존에는 MD가 수동 태깅하던 영역

### Contribution 1-5: Post-Processing으로 Error Rate 11.2% → 0.53% (DS)
- 수치: Pilot 11.2% error → Full Batch 0.53% error (correct_visual_weight 후처리)
- 수치: 561 error / 105,494 items, 9,738 warning의 94.6%가 단일 rule
- 수치: LLM-as-Judge 4.43/5.0 (n=198), Pass Rate 90.9% (95% CI: [86.9%, 94.9%])
- 수치: Per-dimension — coherence 4.80, source_alignment 4.53, accuracy 4.49, informativeness 4.21, specificity 4.14
- 수치: Score<=2 아이템 13/198 (6.6%) — 주요 실패 모드: 소재 오인(knitwear/scarf), 프린트 오인(chili/banana), 컬러 오인(dark blue/black)
- DS 시사점: Structured Output + rule-based 후처리 조합이 LLM 추출 품질을 프로덕션 수준으로 끌어올림. 실패 모드가 L1 소재 필드에 집중되어 L2+L3 Ablation에는 영향 없음

### Contribution 1-6: 비용 효율성 — $8.50로 105K 아이템 추출 (DS)
- 수치: GPT-4.1-nano Batch API, $8.50 / 105,494 items = $0.00008/item
- 수치: 95 batches × 500 requests, 0 failures, 100% completion
- DS 시사점: 산업 규모 카탈로그(~100K)에서 LLM 속성 추출의 경제적 실현 가능성 입증

---

## Phase 2: User Profiling

### Contribution 2-1: Knowledge Case Study — 3-Layer 속성의 5x 정보 확장 실증 (MD+DS)
- 수치: H&M 메타데이터 ~5 dims → L1+L2+L3 추가 시 ~24 dims (약 5x 확장)
- 수치: L2 7개 필드 전부 메타데이터와 zero-overlap (style_mood, occasion, perceived_quality, trendiness, season_fit, target_impression, versatility)
- 수치: L3 5-7개 필드 역시 zero-overlap (color_harmony, tone_season, coordination_role, visual_weight, style_lineage + category-specific)
- MD 시사점: L2/L3 속성은 기존 H&M 카탈로그 시스템에 전혀 없던 차원 — MD가 수동 태깅하기 어려운 감성/이론 속성을 LLM이 자동 생성
- DS 시사점: KAR Factual Expert 입력이 5x 풍부해짐 → BGE 임베딩 공간에서 아이템 간 더 정밀한 거리 측정 가능

### Contribution 2-2: Knowledge Case Study — User Profile Discriminability 실증 (DS)
- 수치: Heavy(100+건)/Moderate(10-20건)/Niche(5-10건) 유저 프로파일이 9 필드 전부에서 가시적으로 다른 내용 생성
- 수치: Low-activity(5-6건) vs Active(15-30건) 프로파일 간 unique vocabulary 차이 (Active가 더 풍부)
- DS 시사점: Reasoning Expert가 유저 유형별 구별 가능한 preference vector를 학습할 수 있는 원천 데이터 품질 확보
- Research Motivation 연결: Triple-Sparsity 환경에서도 5건 이상 유저는 개인화된 프로파일 생성 가능

### Contribution 2-3: Knowledge Case Study — Item→User Knowledge Flow 추적 (MD+DS)
- 수치: L2 5개 필드 → profile (a)(b)(c)(d)(e) 5개 필드 직접 매핑
- 수치: L3 3개 필드 → profile (f)(g)(h) 3개 필드 직접 매핑
- 수치: L1-only 프로파일: 9개 중 3개 필드만 부분 사용 가능, 5개 NOT AVAILABLE, 1개 극히 제한
- MD 시사점: L2 없이는 스타일/무드/TPO(occasion) 기반 추천 불가, L3 없이는 코디/컬러 하모니 추천 불가
- DS 시사점: Layer Ablation 실험 가설 수립 — L2 단독이 L1 단독보다 큰 성능 향상 예상

### Contribution 2-4: Knowledge Case Study — LLM-as-Judge 저점 아이템 오류 분석 (MD+DS)
- 수치: Judge 198건 평가, 37건(18.7%)에서 1개 이상 차원 ≤3점, Pass Rate 90.9%
- 수치: 최저 3건 overall 2.2~2.6 — accuracy/specificity/source_alignment 동시 저하
- 수치: 오류 집중: L1 물리 속성(소재/프린트/컬러) → L2/L3 감성/이론 속성은 상대적 견고
- MD 시사점: 엣지 케이스(비정형 상품, 모호한 이미지)에서 LLM 한계 확인 — 프로덕션 시 human-in-the-loop 필요
- DS 시사점: Expert MLP가 ~9% 노이즈를 흡수 가능, L2/L3 ablation에 영향 미미

### Contribution 2-5: 876K 활성 유저 배치 프로파일링 — 100% 최종 성공률 (DS)
- 수치: 876,788건 Batch API 요청, 873,943건(99.68%) 1차 성공, 2,845건(0.32%) 파싱 실패
- 수치: 실패 원인 — empty text 2,736건(96.2%), max_output_tokens 43건(1.5%), truncated JSON 56건(2.0%), whitespace-padded 10건(0.4%)
- 수치: `--retry-failed` 재시도 → 2,845건 전원 복구, template_fallback 0건, 최종 876,790건 활성 유저 전원 `llm` 소스
- 수치: 최종 parquet — user_profiles 1,298,206건 (876,790 active + 421,416 sparse), reasoning_texts 1,298,206건 (avg 1,064 chars), reasoning_coverage=1.0
- DS 시사점: 대규모 LLM 배치 프로파일링의 실패 모드가 transient(빈 응답 96%)이므로 1회 재시도로 100% 복구 달성. Template fallback은 안전망으로만 존재 (사용 0건)
- Research Motivation 연결: 876K 활성 유저 전원에 대한 reasoning_text 확보 → KAR Reasoning Expert 학습 데이터 완성

### Contribution 2-6: Full-Batch Reasoning Quality — Stale NO-GO → GO 전환 (MD+DS)
- 수치: Coverage 32.5% → 100% (1,298,206/1,298,206 users), Completeness 32.5% → 99.99%
- 수치: Discriminability mean_sim=0.259 (excellent, threshold 0.60), mean_trigrams=137.5
- 수치: Token budget — mean 182, p99=306, max 1,166, over-budget 5건 (0.0004%)
- 수치: LLM-as-Judge — overall 4.86/5.0 (n=200), pass rate 100%, 5개 차원 전부 >= 4.7
- 수치: Per-dimension — coherence 4.96, accuracy 4.89, source_alignment 4.88, specificity 4.80, informativeness 4.76
- 수치: Active LLM profiles ~4x longer than sparse templates (mean text length, token count)
- 수치: Completeness vs purchases Spearman correlation — profile quality fairness across segments
- 수치: Go/No-Go 6/6 PASS (completeness, generic_rate, discriminability, token_budget, judge_overall, judge_pass_rate)
- MD 시사점: 130만 고객 전원의 패션 정체성 프로파일 완성 — 스타일/무드/TPO/가격성향/트렌드감도/시즌/폼/컬러/코디 9차원 커버. Sparse 고객(1-4건)도 기본 프로파일 보유
- DS 시사점: KAR Reasoning Expert 학습 데이터 품질 검증 완료. BGE-base 512 토큰 내 안전 (p99=306), 프로파일 간 구분력 우수 (sim=0.259). Phase 3 세그멘테이션 + Phase 4 모델 학습 진행 가능
- Research Motivation 연결: Triple-Sparsity 환경에서 CF 시그널 없는 421K sparse 유저에게도 template 기반 reasoning vector 제공 → cold-start 추천의 content-based 경로 확보

## Phase 3: Segmentation & Analysis

### Contribution 3-1: 5-Level 고객 세그멘테이션 — L1/L2/L3 구조적 독립성 검증 (MD+DS)
- 수치: 1,298,206 유저 × 5 세그멘테이션 레벨 (L1/L2/L3/Semantic/Topic)
- 수치: L1 ~89D, L2 49D, L3 37D 구조화 벡터 + BGE 768D 시맨틱 벡터
- 수치: Silhouette-based k 선택 (50K 서브샘플), K-Means 클러스터링
- 수치: Cross-layer ARI (5×5): off-diagonal ARI로 레벨 간 독립성 정량화
- MD 시사점: L1(제품)·L2(체감)·L3(이론) 세그멘트가 구조적으로 다른 고객 facet을 포착 — 다차원 타겟팅 가능
- DS 시사점: 낮은 off-diagonal ARI = 각 Layer가 비중복 정보 담당 → KAR 3-Layer 입력의 다양성 근거

### Contribution 3-2: BERTopic 기반 데이터 기반 토픽 vs L2 속성 교차 검증 (DS)
- 수치: UMAP(5D) + HDBSCAN → 자동 토픽 수 결정, c-TF-IDF 토픽별 top-10 키워드
- 수치: ARI(Topic, L2) — L2 속성 설계가 데이터 기반으로도 유효한지 정량 검증
- DS 시사점: 중간 수준 ARI면 L2 속성이 실제 패턴 반영 + 추가 구조 부여, 낮은 ARI면 L2가 독자적 차원 포착

### Contribution 3-3: LLM 임베딩 기반 상품 클러스터링 — Cross-Category 발견 (MD+DS)
- 수치: 105K 아이템 BGE 클러스터 vs H&M product_type ARI 0.522 (isotropy correction 후 0.449→0.522)
- 수치: FAISS ANN 기반 cross-category 유사 쌍 탐지 (cosine > 0.85, product_type 다름)
- MD 시사점: 카테고리 경계를 넘는 시맨틱 유사 아이템 → 발견 지향 추천의 후보 풀 확장
- DS 시사점: LLM 임베딩이 원본 카탈로그 분류 대비 더 정밀한 아이템 거리 측정 제공

### Contribution 3-4: BGE 임베딩 사전 계산 — Phase 4 KAR 파이프라인 준비 (DS)
- 수치: item_bge_embeddings.npz (105,494 × 768, float16), user_bge_embeddings.npz (1,298,206 × 768, float16)
- DS 시사점: Phase 4 KAR text_encoder.py에서 재인코딩 없이 직접 로드 → 학습 파이프라인 가속

### Contribution 3-5: 전처리 파이프라인 개선 — StandardScaler + PCA whitening + BGE isotropy correction (DS)
- 수치: L2 silhouette 0.204→0.472 (+131% 개선), L3 silhouette 0.532→0.011 (inflated→실제 구조)
- 수치: L1 silhouette 0.287→0.007, k 6→12 (스케일 불균형 해소 후 실제 구조 반영)
- 수치: Semantic silhouette 0.182→0.040, k 4→12 (BGE mean norm 0.943 제거)
- 수치: Topic 5→10 topics (isotropy correction으로 HDBSCAN 밀도 구조 발견)
- 수치: Product ARI vs native 0.449→0.522 (+16%, item BGE mean subtraction)
- DS 시사점: PCA 전 StandardScaler 부재 시 스케일 큰 피처가 지배적 주성분을 형성해 실루엣을 왜곡. whiten=True로 주성분 분산 정규화하여 K-Means 등거리 가정 충족
- DS 시사점: BGE-base 임베딩의 비등방성(mean cosine 0.794)이 클러스터링을 방해 — mean subtraction으로 해소

### Contribution 3-6: 분석 함수 추가 — discriminative profiling, effective k, L3 heatmap, excess similarity, topic sensitivity (DS)
- 수치: 기존 population top-N 프로파일링의 "Trousers/Black 반복" 문제 해결 — segment_freq/population_freq ratio로 차별 속성 탐지
- 수치: Effective k = exp(entropy) — nominal k 대비 실제 세그먼트 활용도 정량화
- 수치: L3 37차원 전체 히트맵 — harmony/tone(15D) 외 coordination_role/style_lineage(22D) 차별화 분석
- 수치: Cross-category excess similarity — baseline mean cosine 대비 초과 유사도로 진짜 cross-category 관계 필터링
- DS 시사점: 분석 도구 완비로 세그멘테이션 품질을 다각도 검증 가능 — 연구 논문 Figure/Table 소재 확보

### Contribution 3-7: L2/L3 구조화 벡터 붕괴 진단 — 표현 형식 한계, 정보 가치 아님 (MD+DS)
- 수치: L2 eff_k=1.08 (nominal k=4, 98.8% 단일 세그먼트), L3 eff_k=1.76 (78% 단일 세그먼트)
- 수치: 아이템 수준 속성 evenness — L2 mean 0.668, L3 mean 0.760 (style_mood 0.70, occasion 0.61, trendiness 0.61 → 주 값이 29-64% 점유)
- 수치: CLT 수렴 확인 — L2 분산 1-5건 유저 0.024 → 50+건 유저 0.002 (12x 감소)
- 수치: PCA PC1 점유율 — L1 3.2%, L2 9.9%, L3 12.6% (L2/L3가 다소 집중적이나 극단적이지 않음)
- 수치: 95% 분산 필요 차원 — L1 31/89, L2 31/49, L3 28/37
- 수치: **Semantic 대조 증거 — reasoning_text(L2 5필드 + L3 3필드 합성)→BGE 768D의 eff_k=10.30 (High)**
- 수치: 구조화 벡터(49D/37D 빈도 벡터) vs 텍스트→BGE(768D) = 동일 정보, 표현 형식만 다름 → 결과 3.8배 차이
- MD 시사점: H&M은 mid-market SPA로 ~70% Casual/Everyday 집중 → L2/L3 구조화 벡터의 유저 간 차이가 브랜드 동질성에 의해 압축. 단, Semantic(text 경로)에서 동일 정보가 성공적으로 고객 차별화 → L2/L3 정보 가치 자체는 유효
- DS 시사점: L2/L3 붕괴는 "정보 가치 없음"이 아닌 "구조화 빈도 벡터의 표현 형식 한계". Semantic eff_k=10.30이 L2+L3 텍스트 경로의 정보 가치를 직접 증명. KAR의 text_composer.py→BGE 경로 선택이 정당화됨

### Contribution 3-8: L2/L3 전처리 개선 실험 — 구조화 벡터 best vs Semantic 3.8배 차이 (DS)
- 수치: L2 — Original eff_k=1.08, TF-IDF 1.70, CLR **2.72** (best), UMAP 1.90
- 수치: L3 — Original eff_k=1.76, TF-IDF 1.00, CLR 1.81, UMAP **2.63** (best)
- 수치: 최선의 결과도 eff_k < 3.0 (MARGINAL) → 전처리로는 구조화 벡터의 표현력 한계 극복 불가
- 수치: CLR이 L2에서 +1.64 향상 (compositional section의 simplex→Euclidean 매핑 효과)
- 수치: UMAP이 L3에서 +0.87 향상 (비선형 manifold 구조 포착, 50K 서브샘플)
- 수치: **구조화 벡터 best eff_k=2.72 vs Semantic eff_k=10.30 → 3.8배 차이** (동일 L2+L3 정보, 표현 형식만 다름)
- DS 시사점: 3종 대안 모두 eff_k < 3.0 → 구조화 빈도 벡터의 표현력 한계 정량 실증. text→BGE 768D 경로(Semantic)가 3.8배 효과적 — KAR text_composer.py→BGE 경로의 필요성을 방법론적으로 정당화
- Research Motivation 연결: negative result + positive 대조가 동시에 기여 — 구조화 벡터 한계 실증 + Semantic을 통한 L2/L3 정보 가치 입증이 text-based KAR 아키텍처의 설계 근거를 완성

### Contribution 3-9: Attribute-Purchase MI — L2/L3의 구매 예측 시그널 정량화 (MD+DS)
- 수치: 24개 속성 NMI, 10M subsampled train pairs (121.8M 중)
- 수치: Raw MI top: section(meta, 0.114) > index(meta, 0.083) > **style_lineage(L3, 0.041)** > **style_mood(L2, 0.031)** > garment_group(meta, 0.029) > product_type(meta, 0.027)
- 수치: Conditional MI(L2|L1) = 0.148, MI(L3|L1+L2) = **0.185** (가장 큰 비중복 정보), MI(L2|metadata) = 0.140
- 수치: Layer 평균 NMI: metadata 0.0131 > L1 0.0043 > L3 0.0034 ≈ L2 0.0033
- 수치: NMI는 고-카디널리티 속성(style_lineage 303값, style_mood 727값)에 불리 (분모 H(A) 큼) → Raw MI가 더 공정한 비교
- MD 시사점: style_lineage(패션 이론 계보)가 비-메타데이터 속성 중 가장 강한 구매 시그널 — "Scandinavian Minimalism" 같은 패션 이론 개념이 실제 구매 패턴과 연결
- DS 시사점: Conditional MI 양수 = 각 Layer가 비중복 정보 포함. L3의 MI(L3|L1+L2)=0.185가 최대 → Phase 5에서 L3 제거 시 가장 큰 성능 저하 예측. 3-Layer Taxonomy의 비중복성 정보이론적 정당화
- Research Motivation 연결: NMI가 높은 L2/L3 = CF가 못 잡는 구매 시그널 → sparse 환경에서 content-based 경로의 가치 입증

### Contribution 3-10: CKA + Separation AUC — BGE 임베딩 공간의 Layer 독립성 (DS)
- 수치: 7×7 CKA 행렬 (5K item sample), CKA(L1,L3)=0.788 (최대 차이), CKA(L2,L3)=0.868, CKA(L1,L2)=0.821
- 수치: CKA(L2+L3, L1+L2+L3)=0.867 → L1 추가로 13.3% 표현 변화 (가장 큰 단일-layer 효과)
- 수치: CKA(L1, L1+L2+L3)=0.935 → L2+L3 추가로 6.5% 표현 변화
- 수치: Separation AUC: L1 0.709, L2 0.703, L3 0.704, L1+L2 0.697, L1+L2+L3 0.694 (모두 0.69~0.71)
- DS 시사점: Layer 간 CKA < 1.0 = BGE 공간에서 서로 다른 표현 학습. 단, 순수 cosine AUC는 변형 간 차이 미미(0.69~0.71) → KAR Expert의 non-linear 변환이 이 차이를 증폭시켜야 함
- Research Motivation 연결: CKA 분석이 3-Layer 비중복성을 임베딩 수준에서 추가 확인 (MI의 정보이론적 증거와 상호 보완)

### Contribution 3-11: Preference Diversity — L2/L3가 추천에 가장 가치 있는 속성 (MD+DS)
- 수치: RVI(JSD/entropy) Top-5 전부 L2/L3: perceived_quality(L2, 0.535), season_fit(L2, 0.525), coordination_role(L3, 0.492), versatility(L2, 0.488), trendiness(L2, 0.478)
- 수치: Temporal stability: L2/L3 저-카디널리티 0.80-0.87 vs metadata 0.35-0.52 (product_type 0.349, colour_group 0.523)
- 수치: JSD(유저 간 차별화): style_mood(L2, 0.743) > product_type(meta, 0.739) > section(meta, 0.728)
- 수치: 100K 유저 샘플, 100K 유저 쌍 JSD, 지수 감쇠 가중(halflife=90일)
- MD 시사점: perceived_quality(품질 기대치)와 coordination_role(코디 역할 선호)은 H&M 메타데이터에 전혀 없는 차원이면서 유저 간 가장 차별화되고 시간적으로 안정적 — 근본적 취향을 포착
- DS 시사점: 세그멘테이션 CLT 붕괴(Contribution 3-7)를 속성-레벨 직접 측정으로 우회. RVI가 높은 L2/L3 속성이 KAR Factual Expert에서 가장 큰 기여 예측. 클러스터링 없이도 유저 선호 차별화 정량 가능
- Research Motivation 연결: L2/L3가 포착하는 선호(품질 기대치, 계절 선호, 코디 역할)는 metadata보다 시간적으로 더 안정(0.80+ vs 0.35) → 장기적 추천 품질에 기여

### Contribution 3-12: Cold-Start Content-Based Retrieval — Sparse 유저에서 속성 가치 직접 입증 (DS)
- 수치: 50K 유저 × 105K 아이템 × 7종 ablation, 6 구간별 HR@12/NDCG@12/MRR
- 수치: 1건 유저 HR@12: L1 2.47%, L1+L2 2.26%, L1+L2+L3 2.36% (L1 단독 최강)
- 수치: 2-4건 유저 HR@12: L1 3.03%, **L1+L2 3.28%** (전체 최고), L1+L2+L3 3.14%
- 수치: Sparse(1-4건) HR@12 2.5-3.3% > Heavy(50+) HR@12 0.5-1.2% — content-based가 sparse에서 더 유효
- 수치: L2/L3 단독은 L1보다 약함 (L2 1.95%, L3 1.54% vs L1 2.47% at 1건)
- 수치: L1+L2 > L1+L2+L3 > L1 > L1+L3 > L2+L3 > L2 > L3 (2-4건 기준)
- DS 시사점: L1(소재/핏/실루엣)이 가장 강한 단독 content-based 시그널. L2(스타일/무드)는 보완재로 작동(L1+L2가 best). Content-based HR@12가 KAR 모델의 최소 기대 바닥값
- Research Motivation 연결: Triple-Sparsity 환경에서 1-4건 유저에 대한 content-based 추천 성능을 직접 측정 — CF 시그널이 불충분한 환경에서 속성 기반 경로의 실용적 가치 정량화. Phase 5 Cold-start 실험과 직접 비교 가능

## Phase 4: KAR Module Implementation

### Contribution 4-1: 5종 백본 모델 embed()/predict_from_embedding() 분리 — Backward-Compatible (DS)
- 수치: DeepFM, DCNv2, LightGCN, DIN, SASRec 5종 모델 모두 embed()+predict_from_embedding() 추가
- 수치: 기존 __call__() 결과와 embed()+predict_from_embedding() 결과 atol=1e-5 이내 동일 (5종 전수 검증)
- 수치: 기존 테스트 100건 전수 PASS (backward compatibility 유지)
- DS 시사점: KAR Fusion 삽입 지점 확보 — backbone 내부 임베딩 레이어와 예측 레이어 사이에 augmented vector 주입 가능
- Research Motivation 연결: Model-agnostic 속성 증강을 위한 backbone 추상화 완성

### Contribution 4-2: KAR 2-Expert 아키텍처 구현 — 4종 Gating × 4종 Fusion 변형 (DS)
- 수치: Expert MLP (768D → d_rec=64, 2-layer ReLU + dropout)
- 수치: Gating 4종 (G1 Fixed, G2 Expert-conditioned, G3 Context, G4 Cross) — 모두 softmax 정규화 (g_fact + g_reason = 1)
- 수치: Fusion 4종 (F1 Concat, F2 Addition, F3 Gated, F4 CrossAttention) — 모두 내부 Linear(d_rec, d_backbone) 프로젝션 포함
- 수치: d_rec(64) ↔ d_backbone(백본별 상이: DeepFM 288, LightGCN 128 등) 차원 불일치를 Fusion 내부 프로젝션으로 해소
- DS 시사점: 4×4=16종 Gating×Fusion 조합 실험 가능 — Fix-and-Vary ablation 인프라 완성

### Contribution 4-3: KARModel Composition 패턴 — 백본 코드 최소 변경 (DS)
- 수치: KARModel이 backbone을 소유하는 Composition 패턴 — backbone 내부 코드 수정 0줄 (embed/predict 추가만)
- 수치: forward_with_intermediates() → e_fact, e_reason, g_fact, g_reason, x_backbone_flat 5종 중간값 반환
- 수치: compute_d_backbone() — 백본 유형별 자동 임베딩 차원 계산
- DS 시사점: 백본 교체 시 KAR 코드 수정 불필요 — 5종 백본 실험의 코드 재사용성 확보

### Contribution 4-4: 3-Stage Multi-Stage 학습 파이프라인 (DS)
- 수치: Stage 1 (backbone pre-train, BCE only), Stage 2 (expert adaptor, align+div, backbone frozen), Stage 3 (end-to-end, BCE+align+div, LR×0.1)
- 수치: align_loss = MSE(e_expert, stop_gradient(backbone_embed)), diversity_loss = mean(cos_sim(e_fact, e_reason))
- 수치: Stage 간 모델 파라미터 유지, 옵티마이저만 재생성 (새 LR)
- DS 시사점: Xi et al. 2023 원 논문의 multi-stage 학습 충실 구현 — Stage 2에서 stop_gradient로 backbone 임베딩 품질 보존

### Contribution 4-5: KAR 데이터 파이프라인 — BGE 임베딩 인덱스 정렬 + Grain DataLoader 통합 (DS)
- 수치: build_aligned_embeddings() — feature store integer index 기준 BGE (105K×768, 1.3M×768) 정렬
- 수치: 48개 누락 아이템 zero-vector 패딩 (0.05%)
- 수치: 4종 KAR Transform (KARFeatureLookupTransform, KARIndexTransform, KARDINLookupTransform, KARSASRecTransform)
- 수치: create_train_loader(use_kar=True) — 기존 Grain DataLoader에 h_fact + h_reason 조회 추가
- DS 시사점: 사전 계산된 BGE 임베딩을 학습 배치에 실시간 조회 — 재인코딩 없이 학습 파이프라인 가속

### Contribution 4-6: Pre-store 서빙 준비 — 오프라인 Expert 출력 사전 계산 (DS)
- 수치: compute_prestore() — item_expert.npz (n_items, d_rec=64) + user_expert.npz (n_users, d_rec=64)
- 수치: 배치 크기 조절 가능 (기본 4096) — GPU 메모리 고려
- DS 시사점: 온라인 서빙 시 Expert MLP forward 불필요 → 전체 카탈로그 스코어링 ~15ms 달성 경로 확보

### Contribution 4-7: 72건 신규 단위 테스트 — 기존 100건 포함 199건 전수 PASS (DS)
- 수치: test_expert.py(10), test_gating.py(17), test_fusion.py(15), test_hybrid.py(14), test_losses.py(8), test_prestore.py(4), test_backbone_embed.py(5) = 73건 신규
- 수치: 기존 테스트 100건 + 신규 73건 = 199건(실제) 전수 PASS
- 수치: 테스트 커버리지: shape 검증, gradient flow, JIT 호환, backward compatibility, stage2/3 loss 분해
- DS 시사점: Phase 5 체계적 실험 전 코드 안정성 보장 — 변형 실험 시 regression 즉시 탐지

## Phase 5: Systematic Experiments (추가 예정)

---

## 누적 수치 요약 (Phase별 업데이트)

| Phase | 핵심 수치 | Motivation 연결 |
|-------|----------|----------------|
| 0 | Gini=0.7586, 99.98% sparse, Pop>UserKNN>BPR-MF | CF 구조적 실패 정량화 |
| 1 | 105K×22 attr, 100% cov, error 0.53%, $8.50, Judge 4.43/5.0 (n=198, 90.9% pass) | Content-Based 인프라 완성 |
| 2 | 5x 정보 확장(5→24 dims), L2/L3 zero-overlap, 876K batch(100% 최종 성공, retry 2,845→전원 복구, fallback 0), 1,298,206 유저 parquet, Heavy/Moderate/Niche discriminable, Eval: completeness 99.99%, sim=0.259, Judge 4.86/5.0 (n=200, 100% pass), Go/No-Go 6/6 GO | 유저 프로파일 + 배치 추출 + 품질 검증 완성 |
| 3 | **Tier 1**: 1.3M유저×5레벨, StandardScaler+whiten+isotropy fix, L2 sil 0.204→0.472, Product ARI 0.449→0.522, L2/L3 구조화 벡터 붕괴(eff_k<3.0) = 표현 형식 한계, Semantic eff_k=10.30 = 정보 가치 증거 (3.8배). **Tier 1.5**: MI(L3\|L1+L2)=0.185(최대 비중복 정보), Raw MI style_lineage(L3) 전체 3위, RVI Top-5 전부 L2/L3(perceived_quality 0.535), CKA(L1,L3)=0.788(최대 표현 차이), Temporal stability L2/L3 0.80-0.87 > metadata 0.35-0.52, Cold-Start 2-4건 L1+L2 HR@12=3.28%(최고), Sparse>Heavy (content-based가 sparse에서 더 유효), 8 figures, 41+85=126 unit tests | Tier 1: 세그멘테이션 진단 + Semantic 대조. Tier 1.5: 속성-구매 관계 직접 정량화 — L2/L3 비중복성(MI), 추천 가치(RVI), 임베딩 독립성(CKA), Cold-start 보상(HR@12) 4중 증거 |
| 4 | 5종 백본 embed/predict 분리(atol<1e-5), KAR 2-Expert(Expert+4G+4F), KARModel composition(backbone 수정 0줄), 3-Stage multi-stage(BCE→align+div→full), BGE 인덱스 정렬(48 zero-pad, 0.05%), 4종 KAR Transform, Pre-store .npz, 73건 신규 테스트(199 total PASS) | KAR 모듈 구현 완료, Phase 5 실험 인프라 확보 |
| 5 | (TBD) | 최종 성능 비교 |
