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

## Phase R: 2026-06 Redesign (Falsification-First)

> 전체 점검·전면 재설계. 상세: `redesign_2026-06.md`. research-design OS brownfield revise.

### Contribution R-1: 검증 scoring 병목 교정 — Phase 5 unblock (DS)
- 수치: 중단 근본 원인 = `validate_sample(batch_size=1)` per-user scoring → 413K val users × 105K items = **~3.5h/epoch**, 학습:검증 ≈ **1:5** (자원 부족 아님 — 2× L40S, 48 CPU, 372GB RAM)
- 수치: caller-side 배치 scorer로 교체(`generate_predictions`/`generate_predictions_kar`, `jax.lax.top_k`), 전 backbone(feature/graph/sequence)
- 수치: 신규 등가성 테스트 **9건**(4 backbone × batch_size {1,4,32}, 배치 top-12 == per-user top-12) 포함 **686 unit pass, 회귀 0**
- DS 시사점: Phase 4 E2E·Phase 5 그리드를 막던 단일 결함 해소. config/scripts/YAML/docs 동기화(CLAUDE.md)
- Motivation 연결: "성능·자원 문제로 중단"의 실제 원인이 알고리즘·자원이 아닌 1개 엔지니어링 결함임을 규명

### Contribution R-2: 메타데이터 confound 해소 + Layer 증분 가치 falsification (MD+DS)
- 수치: 기존 7개 ablation 임베딩이 전부 metadata를 prepend(`text_composer.py:195-198`) → `meta.npz` 부재로 "메타데이터 대비 증분"이 **측정 불가**였음. `meta.npz` 생성으로 사다리 `META→L1→L1+L2→+L3` 완성
- 수치(probe_01, n=49,998, paired bootstrap CI): pooled HR@12 META **0.0069** → L1 **0.0148** → L1+L2 **0.0160** → +L3 **0.0138** (비단조)
- 수치: **C1** META→L1+L2 **+130.9%** (CI [+0.0079,+0.0102]); **C2** L1→L2 **+8.0%** (CI [+0.0002,+0.0021]); **C3** L1+L2→L3 **−13.5%** (CI [−0.0030,−0.0013]); **C4** L3 diversity **−7.7%**·coverage −2.3%
- MD 시사점: 3-Layer "다 도움" 주장 → **L1=강한 win / L2=weak-but-real / L3=frozen content-retrieval에서 net-harmful**로 reshape. Contribution 3-12("L1+L2=3.28% 최고")를 metadata-free baseline + 유의성으로 정직 보강(3-12엔 metadata 기준·CI 없었음)
- DS 시사점: 데이터 재업로드 후 동일 seed 재현(C1 +130.9 등). frozen unweighted mean-pool은 layer를 weight 1.0 고정 투입 → in-model gate 거동은 미입증
- Motivation 연결: Triple-Sparsity의 cold-start(2-4)에서 L1 content가 metadata 대비 +0.020(CI 0제외) — CF 약한 곳에서 LLM 텍스트의 가치 직접 입증

### Contribution R-3: 적대적 robustness + 5-skeptic 검증 — 결론 hardening (DS)
- 수치(probe_04, 3 seed × k{12,50} × stratum × maxsim = 21 cells): **C3(L3 harm) 18/21 음수·11 유의·양수 0개**(견고); **C2(L2 gain) 17/21 양수·8 유의·유의음수 0**(방향 견고하나 k=12 cold-start 비유의)
- 수치: **length-confound 반증** — L2 +177자에도 도움, L3 +147자에 해로움 → 부호는 텍스트 길이 아닌 content가 결정
- 수치(5-skeptic judge panel): **A holds 5/0/0**(단, +130%의 ~+114%p는 L1 단계), **B holds_with_caveat 2/3/0**(L2 weak·regime-의존), **C holds_with_caveat 4/1/0**(frozen-only drop, in-model 미결)
- DS 시사점: 본인 이전 보고 "L2 깔끔히 생존 / L3 drop"을 **과대주장으로 교정** — L2는 weak-positive, L3는 frozen에서만 drop이고 learnable gate(KAR Gate-2)가 최종 심판. 사전등록 in-model falsification 규칙 수립(`g_L3→0` + cold-start CI)
- Motivation 연결: make-or-break 결론을 적대적으로 hardening → 논문 주장의 honest scoping 확정

### Contribution R-4: Hybrid 평가·baseline foundation — immediate-split + repurchase + discovery 메트릭 (MD+DS)
- 수치(immediate-next-period split, 전체 85,648 유저; train_end=2020-06-30, window Jul1-7): **repurchase(hybrid) MAP@12=0.02374** (HR@12=0.09503, NDCG@12=0.03447) — probe_05의 0.0243을 production 함수로 재현(50K 샘플 vs 전수 차 ≈0.0005). 대조: recent_popularity(14d) MAP@12=0.00307, global_pop MAP@12=0.00351 → repurchase가 ~7× 우위
- 수치(discovery_map, NEW-items-only GT): **repurchase discovery MAP@12=0.00042** (HR@12=0.00694) ≈ 0 — repurchase는 구조상 새 아이템을 못 맞춤. global_pop discovery MAP@12=0.00268(>repurchase, pop이 대부분 유저에게 new). discovery gap을 격리해 LLM/KAR이 메워야 할 공간을 명시
- 수치(repurchase vs new decomposition): GT 305,061개 중 **new=95.9% / repurchase=4.1%** (probe_05 일치). 즉시-다음주 H&M = 압도적 discovery 미션
- 수치(구현): `build_immediate_eval`(splitter) + `SplitConfig.eval_horizon_days=7` + `--build-immediate`; `src/baselines/repurchase.py`(3함수, t_dat date/datetime 호환); `src/evaluation/cohorts.py`(activity_cohorts/evaluate_cohorts/**discovery_map**/decomposition, AP@k는 metrics.evaluate 재사용); `scripts/train.py` backbone `repurchase`/`recent_popularity` + `--eval-split immediate`. 신규 단위테스트 23건 PASS(720 total, 사전 실패 7건 외 회귀 0)
- MD 시사점: 10× 낮던 절대 MAP는 데이터·metric 결함이 아니라 **eval setup(2개월 gap)과 프레이밍(repurchase 폐기)** 때문임을 측정 가능한 형태로 확정. Kaggle-comparable 0.024 backbone 확보 → 이제 KAR 가치는 pooled가 아니라 **discovery portion에서만** 측정해야 함이 코드로 강제됨
- DS 시사점: probe_05(witness)의 일회성 로직을 재사용 가능한 src 모듈로 승격(단일 source of truth). Gate-2'(KAR=discovery lever) 측정 인프라 완비 — `discovery_map`이 핵심 신규 지표
- Motivation 연결: Triple-Sparsity + 87% 단일구매 + 95.9% new-item → **발견 지향 추천**이 본질임을 baseline으로 입증. repurchase가 싼 4%를 잡고, LLM/KAR이 96% discovery를 노리는 Hybrid 분업의 측정 기반 확립

### Contribution R-5: LLM = candidate-pool re-ranker + L1/L2/L3 정직한 처분 (MD+DS)
- 수치(probe_06, immediate split, discovery_map, NEW-only): content **standalone 전-카탈로그 검색기**는 popularity floor(0.00268)에 짐 — META 0.00028 → L1 0.00113 → L1+L2 0.00130 → L1+L2+L3 0.00114. (Gate-0 layer 구조 재현: META→L1 +308%, L1→L2 +15%(CI 근접), L2→L3 −12.5%)
- 수치(probe_07, re-ranker): content를 **인기 후보 풀 re-ranker**로 쓰면 popularity 0.00199 → **content-rerank(L1+L2) 0.00384 (+93.1%)**, hybrid-blend 0.00299(+51%). → **LLM의 역할 = standalone 검색기 ✗, candidate-pool re-ranker ✓** (Kaggle candidate-gen+ranker 패턴)
- 수치(probe_08, re-ranker layer 분해): META 0.00294 → **L1 0.00387 (META→L1 +31.5%, CI 0제외)** → L1+L2 0.00384 (**L1→L2 −0.8%, CI [−0.0004,+0.0003] 0포함**) → L1+L2+L3 0.00366 (L2→L3 −4.6%)
- **결정(framing):** **L1 = 헤드라인 기여**(LLM 구체 제품속성이 metadata 압도). **L2/L3 = "속성 추상화 redundancy" falsification** — 두 맥락(buy-similar retrieval + discovery re-ranker) 모두에서 L1 대비 추천 정확도 증분 ≈ 0. L2 단독은 metadata +66%(semantic signal 실재)이나 **L1과 중복**. ⚠️ R-2/R-3의 잠정 "L2 weak-positive"를 probe_08으로 **"정확도 증분 없음(redundant)"** 로 확정 — L2/L3를 정확도 기여로 주장하지 않음
- 기전: BGE가 텍스트 전체 인코딩 + L1 구체속성(material·fit·neckline)이 지각 인상 이미 결정 → L2 지각 텍스트는 L1에서 추론 가능, 신호 거의 무증분
- MD 시사점: "속성 많을수록 좋다" 가정의 반례 — 풍부한 perceptual/theory 속성도 concrete product 속성과 중복되면 추천엔 무의미. L2/L3의 잔존 가치는 diversity(미측정)·explainability·in-model gate(증거상 낮음)
- DS 시사점: 평가를 immediate-split + discovery_map으로 고정하면 LLM 가치가 frozen에선 L1에 집중됨이 일관 측정. Gate-2'(trainable KAR re-ranker)가 frozen +93% lower bound를 매치/초과하는지가 사활
- Motivation 연결: Triple-Sparsity discovery에서 LLM 제품속성(L1)이 metadata·popularity 대비 실질 가치 확정; 추상화 층(L2/L3)은 정직하게 redundant로 보고 (단 R-6에서 task-mismatch로 교정)

### Contribution R-6: L2/L3 Fair-Chance 진단 — "redundant"는 task-mismatch였음 (MD+DS)
- 동기: R-5의 "L2/L3 redundant"가 우리 설계 인공물인지 적대 점검(5 설계 결함: single-text blend·frozen encoder·layer-gating 부재·압축·task mismatch)
- 수치(probe_09, separate-layer 인코딩): sep_L1L2L3 vs L1-only **−1.6%(CI 0포함)**, vs metadata-control +8.8% → **인코딩 형식이 문제 아님**(결함 #1 반증)
- 수치(probe_10, optimal layer 가중): best weights **[0.6,0.2,0.2]**, L1-only 대비 +5.5%(**CI 0포함, 비유의**) → layer gate는 marginal(결함 #3 약함)
- 수치(probe_11, context): season-match boost **−42%**(해로움), occasion +0.6%(중립) → context-conditioning 무효(결함 #5a 반증: popularity가 계절성 포착·catalog 동질)
- 수치(probe_12, coordination 상관): 실제 co-purchase(cross-category outfit) 아이템이 L3 속성 공유 ≫ random — color_harmony +0.080, style_lineage **1.8×**, L3 cosine +0.025 (전부 SIG) → coordination 신호 *존재*
- 수치(probe_13, ★ coordination 랭킹 make-or-break): held-out co-purchase 예측에서 **L1_cos HR@12=0.226 > L3_cos 0.216**, **L1+L3 0.224(L1 대비 −0.9%, CI 0포함)**, L1+harmony 0.185(−18%) → **complementarity에서도 L1 지배, L3 증분 0**
- **결론 (최종, 교정):** probe_12 상관 신호는 실재했으나 **랭킹 가치로 번역 안 됨** — L1이 cross-category co-purchase 구조(일관 스타일/품질)도 더 잘 포착. 즉 R-5의 "redundant"는 **과소가 아니라 정확**했고, 이를 4개 설계-fix(09-11)+coordination task(12-13)로 적대 검증해 **다중 task robust redundancy로 확정**. 설계 결함(encoder/gating/task)은 실재하나 **고쳐도 L2/L3 안 살아남**.
- MD 시사점: H&M mid-market catalog에서 **concrete 제품속성(L1)이 recommendation-relevant 신호를 대부분 보유** → 추상 perceptual(L2)·theory(L3)는 풍부해도 redundant. "속성 추상화 = 추천 가치"는 이 도메인에서 성립 안 함(반직관적·강한 negative).
- DS 시사점: falsification-first의 정수 — single-item에서 "redundant"를 7개 probe(01/04/06/08/09/10/13)+적대검증+coordination task로 다각 검증. 단정 negative 전에 설계·task 공정성을 끝까지 점검해 robust화.
- Motivation 연결: 87% 단일구매·discovery 미션에서 **L1이 LLM-속성 기여의 전부**; L2/L3 추출 투자는 추천 정확도엔 미회수(향후 explainability·다른 도메인 catalog에서 재검토 여지)

### Contribution R-7: 근본원인 진단 + 컨셉 재정의 — 왜 실패했나 + 검증된 pivot (MD+DS) ★
- 동기: R-5/R-6의 "L2/L3 정확도 증분 0"이라는 **현상**을 넘어 **원인**을 규명하고(왜 안 됐나), 원인에 대한 **수정이 작동하는지** 싸게 de-risk. 사용자 요구 = "실패의 이유와 컨셉부터 다시" + "포트폴리오용이되 논문급 퀄리티".
- **진단(probe_14, kNN k=15, train 20K/test 5K):** L1 임베딩만으로 L2/L3 속성 예측 **평균 lift=0.38**(L2 0.37 / L3 0.38, majority 대비). l2_style_mood 0.71(21클래스)·l3_style_lineage 0.51(44클래스)·l3_color_harmony 0.64(8클래스)·l3_visual_weight 0.74 → **L2/L3는 L1의 (준)함수 = product-internal 재서술**. redundancy는 측정 버그가 아니라 **추출 설계의 구조적 귀결**(프롬프트가 LLM을 "the product itself"에 한정 → KAR open-world-knowledge 약속 미시도).
- **수정 ①(probe_15, control 축, 무료):** L2/L3로 추천을 의미축으로 **steered 정밀도 1.00 vs 무제어 0.14**(control gain +0.86), **개인화 100% 유지**(steered가 random-t 아이템보다 유저 L1 centroid에 가까움). metadata엔 없는 **8개 의미 제어축**(occasion·mood·season·quality·trendiness·versatility·coordination·visual_weight) — L1만으론 "occasion=Party로 추천"이 표현 불가.
- **수정 ②(probe_16, 외부지식 KAR, ~$0.1):** 새 프롬프트로 LLM이 *제품에 없는* 보완 styling 지식 생성(600 seeds) → cross-category co-purchase 보완 랭킹에서 **external_knowledge HR@12=0.2366 vs L1_cos 0.2108 = +12.2%**(delta +0.0258, **bootstrap CI [+0.0103,+0.0413], 0 배제**), popularity 0.0158. **진단이 가리키는 수정(외부지식)이 product-similarity를 유의하게 능가 = KAR 비전 원리적 실현 = 컨셉 rescue**. scope: complementarity 한정 de-risk(개념증명), full 통합은 future work.
- MD 시사점: "L2/L3 실패"는 **3-Layer taxonomy의 무효가 아니라 *추출 방향(제품 묘사 vs 외부지식)*의 문제**임을 mechanism으로 확정. 가치 축이 **정확도 → (a) controllability (b) 외부지식 complementarity**로 재정의됨. 풍부한 perceptual/theory 속성이 redundant인 것과, 그 속성을 *제어 인터페이스*·*외부지식 추출 타깃*으로 쓰는 것은 별개 — 후자에선 L2/L3 어휘가 핵심.
- DS 시사점: falsification-first의 완성형 — **현상(13 probe)→원인(probe_14)→수정 검증(15·16)**. negative를 mechanism까지 진단하고 진단이 함의하는 fix를 다시 적대 검증해 "단순 실패"를 "검증된 pivot"으로 전환. 모든 수치 고정 seed·canonical JSON·bootstrap CI로 재현(probe_16은 추출 캐시로 무과금 재현).
- Motivation 연결: Triple-Sparsity·96% discovery 환경에서 LLM의 가치는 **제품 재서술(L1로 충분)이 아니라 *상호작용·CF에 없는 외부 styling 지식*에서 나옴**(probe_16 +12.2%) — KAR 원 동기(open-world knowledge로 sparse 보완)를 본 도메인에서 처음 실증. controllability(probe_15)는 cold-start·발견 추천에서 유저 의도 주입 경로 제공.
- ➡️ **R-8 확장:** R-7의 probe_16(pair-level +12.2%)을 유저-레벨 discovery로 일반화 — **frozen에선 NO-GO(−60%)였으나 그것이 *fusion artifact*였고, learned fusion + placebo control에서 REVIVE REAL**(지식-고유 +17%, 5/5 seed). 외부지식 KAR이 discovery에서도 유효.

### Contribution R-8: 외부지식 일반화 게이트 — frozen NO-GO는 fusion artifact, learned fusion에선 REVIVE REAL (MD+DS) ★
- 동기: R-7 probe_16(pair-level +12.2%)이 *실제 유저-레벨 discovery 추천*으로 옮겨가는지 = full external-KAR 빌드($5-10+GPU) 정당화 여부의 make-or-break 게이트. falsification-first로 투자 전 de-risk. **핵심 교훈: de-risk는 *대표적 fusion*으로 해야 한다 — frozen으로 죽이면 거짓 NO-GO를 낼 수 있다(아래).**
- **게이트(probe_17, frozen, immediate-split discovery_map, 12K 유저):** 유저 프로파일을 두 방식으로 — L1-profile("유사 추천")=mean L1 / external-profile("내 옷장의 보완")=mean 외부지식emb — 공유 인기 풀 re-rank(owned 제외). **external 0.00104 vs L1 0.00265 = −60.7%**(CI [−0.00219,−0.00108]), popularity 0.00190, blend 0.00210. → 외부지식 집계 프로파일이 L1에 크게 짐.
- **적대 검증(probe_17b, 8K 유저):** mean-pool이 신호를 죽인 artifact인지 두 각도로 — ① **max-sim**(=학습 attention-Expert 근사, "옷장 중 *아무거나* 보완"): ext_maxsim 0.00101 vs L1_maxsim **0.00289**(L1이 max에서 더 강함) ② **cross-PG 분해**(보완의 본거지): cross-PG L1 0.00027 vs ext_max **0.00004**. **ext_maxsim vs L1 −58.8%**(CI [−0.00221,−0.00072]). → **둘 다 L1 우세 = NO-GO ROBUST**(학습된 selective Expert로도, 보완 부분집합에서도 외부지식이 L1에 짐).
- **fusion artifact 규명(probe_18, learned two-tower, 70/30 user split, held-out discovery_map):** raw-cosine 대신 *학습 projection*을 주면 — A(L1-only) 0.00405 / B(+ext user-side) 0.00416(+2.7%) / **C(+ext both-side, KAR식 item augmentation) 0.00456(+12.6%)**. frozen L1(0.00265)도 학습으로 0.00405로 상승 → **probe_17/17b의 −60%는 fusion 불일치 artifact**(styling 문장 vs 제품 묘사 텍스트 장르차를 raw-cosine이 못 정렬).
- **★ placebo control(probe_18b, 5 seed):** +12.6%가 *용량(2× 차원)* 인지 *지식 내용* 인지 분리 — A 0.00459 / **C_real 0.00552** / C_shuffle(ext를 아이템 간 뒤섞음=같은 분포·같은 capacity) 0.00472 / C_noise(랜덤) 0.00405. **C_shuffle−A=+0.00013(capacity만≈0)**, **C_real−C_shuffle=+0.00080(지식-고유, 5/5 seed 일관)**, C_real−A=+0.00093(5/5). → **REVIVE REAL: 이득은 외부지식의 *내용*이지 용량이 아님**(placebo로 확인).
- **format-통일 검증(probe_19):** "frozen 실패는 styling-산문 vs L1-속성 *장르 불일치* 탓 아닌가"를 직접 테스트 — external을 L1 속성 format으로 구조화 재추출($0.5)·재인코딩. frozen discovery_map: L1 0.00207 / ext_prose 0.00078 / **ext_unified 0.00051**(unified vs prose −34% 비유의, vs L1 −75%). → **FORMAT NEUTRAL: 장르 통일은 frozen 실패를 못 고침**(lever는 format이 아니라 learned fusion). caveat: 본 구현은 external→속성 *압축*(lossy)이라 format↔정보손실 혼재, 반대방향(L1→산문) 미검증이나 unified<prose는 통일에 불리한 증거. canonical: `witnesses/probe_19_result.json`.
- **fusion×format 시너지(probe_20, learned, 3 seed):** format 통일이 learned fusion 하에서 *추가*되는지 — A 0.00384 / C_prose 0.00484 / **C_unified 0.00390(≈A, 단독 무익)** / **C_both 0.00526** / C_prose_dup(capacity 대조) 0.00480. C_unified−C_prose=−0.00094(단독 unified는 prose보다 나쁨), **C_both−C_prose_dup=+0.00045(3/3, capacity 넘는 진짜 이득)**. → **단일 format 통일은 비권장이나, 산문+구조화 *multi-view*는 상보적**(같은 외부지식의 다른 format이 보완 신호). 설계 함의는 7.4.4에 반영.
- **★ fusion 역할 = 핵심 레버(설계 보강, 7.4.4):** 결합 연산(F1~F4) 선택보다 **Expert의 학습 projection 유무**가 외부지식 가용성을 1차 결정. frozen raw Addition은 외부지식 죽임(−60%), 학습 Expert+item-side augmentation이 살림(+17% placebo-confirmed). 외부지식 입력은 multi-view 권장.
- **결론(최종):** 외부지식은 **pair-level(+12.2%, probe_16) AND user-level discovery(learned fusion +17% 지식-고유, 5/5, probe_18b) 모두에서 유효**. frozen NO-GO(probe_17/17b −60%)는 **대표적이지 않은 fusion(raw-cosine)으로 인한 거짓 음성**이었음. → **option-1(외부지식 KAR for discovery) 부활 — full 빌드($5-10 추출 + KAR external-Expert + end-to-end 학습) 정당화.** scope: de-risk 규모(3,426 아이템 추출·pool=5K re-ranker·frozen BGE feature 위 소형 학습 tower), full end-to-end KAR은 1c.
- MD 시사점: LLM 외부지식(상품·CF에 *없는* styling 지식)이 **메인 discovery 추천을 실제로 향상**(+17% 지식-고유) — 단 *올바른 통합(learned projection + item augmentation)* 전제. "지식을 추출했다"가 아니라 "지식을 *정렬·증강*했다"가 가치를 가름. H&M mid-market에서도 외부지식이 cold-start discovery 레버로 성립.
- DS 시사점: **de-risk 방법론 자체의 교훈** — frozen raw-cosine de-risk는 learned-fusion 모델의 잠재력을 *과소평가*해 거짓 NO-GO를 낼 수 있음(probe_17→18 반전). 대표적 fusion으로 재검정하고, lift는 반드시 **placebo(shuffle/noise) + multi-seed**로 capacity와 분리해야 함(probe_18b). falsification-first에 *대표성*과 *placebo*를 추가한 정점 사례.
- Motivation 연결: Triple-Sparsity discovery의 해법으로 **외부지식 KAR이 실증됨**(learned fusion, placebo-controlled) — KAR 원 동기(open-world knowledge로 sparse 보완)를 본 도메인에서 정량 확인. LLM 기여 지도: **L1(제품속성) + controllability(probe_15) + 외부지식 discovery 증강(probe_18b)**.
- ⚠️ **방향교정 (R-9에서 반증, 이력 보존):** 위 de-risk REVIVE("learned fusion +17% 지식-고유")는 **full-scale(100% coverage) 빌드에서 재현되지 않았다**. R-9의 probe_21(Two-Tower DSSM, coverage=105,494, eligible=40,000, 3 seed)에서 **KAR_external 0.00424 vs L1 0.00482 = −12.0% (0/3 seed) = Gate-2' NO-GO**(둘 다 popularity는 +80~104% 능가). probe_22 isolation(coverage를 full로 고정·population만 변경)으로 원인이 **population-selection bias**임을 확정 — de-risk eligible은 인기 3,426 아이템의 heavy buyer로 편향(active 11,888 vs sparse 112), de-risk-population에선 +14.1%(3/3) GO이나 full-population에선 −9.1%(1/3) NO-GO로 뒤집힘. **따라서 R-8의 "REVIVE REAL / option-1 부활 / full 빌드 정당화" 결론은 R-9에서 정정됨** — de-risk REVIVE는 *heavy-buyer 편향 subset에서만* 성립한 false positive였다. 상세·근거는 R-9 참조. (probe 16–20 de-risk 이력은 *de-risk 기록*으로 유지.)

### Contribution R-9: Full-scale external-KAR 빌드 — Gate-2' NO-GO + population-bias 진단 (MD+DS) ★
- 동기: R-8의 de-risk REVIVE(+17% 지식-고유, learned fusion·placebo-controlled)가 *coverage 제약을 푼 full-scale 빌드*에서 대표·cold-start 유저로 일반화되는지 = full external-KAR 빌드의 make-or-break 게이트. de-risk 규모(3,426 아이템·pool=5K)의 GO를 스케일에서 적대 재검정.
- **1b 추출 (full external knowledge, $4.03):** 새 모듈 `src/knowledge/external/` + `scripts/extract_external_knowledge.py`로 **47,224 product_code → 105,542 article**에 대해 외부지식을 **100% coverage**(산문 prose + 구조화 structured)로 추출(Batch=realtime async, gpt-4.1-nano). `data/knowledge/external/external_knowledge_full.parquet` 저장. de-risk 3,426 아이템 → full 카탈로그로 확장 완료.
- **1c 모델 (Two-Tower / DSSM):** user tower(content profile) + item tower(L1 + external prose + structured, **item-side augmentation**), in-batch sampled-softmax(tau=0.05, logQ correction), 3 seed. **CF 백본(DeepFM)은 실패** — full-catalog 학습에서 discovery MAP@12=**0.000202 ≪ popularity 0.00351**(본 프로젝트의 원초적 "deep model < popularity" 문제 재현) → 검증된 content 백본인 Two-Tower/DSSM으로 학습.
- **★ Gate-2' NO-GO (probe_21, coverage=105,494, eligible=40,000, 3 seed):** popularity disc_map=**0.00236** / L1_only=**0.00482** / KAR_external=**0.00424**. **KAR_external − L1 = −12.0% (0/3 seed) = NO-GO**. 단, **두 content 모델 모두 popularity를 +80~104% 능가**(L1 +104%, KAR +80%). Cold-start cohort에서도 **L1 0.00719 vs KAR 0.00568**로 외부지식이 짐 — Triple-Sparsity cold-start에서 외부지식이 *가장 크게* 실패.
- **★ population isolation (probe_22, coverage를 full로 고정·population만 변경):** de-risk-population에선 KAR **0.004062 vs L1 0.003561 = +14.1% (3/3) GO**, 동일 coverage의 full-population에선 KAR **0.004363 vs L1 0.004800 = −9.1% (1/3) NO-GO**. → **원인은 coverage가 아니라 POPULATION-SELECTION BIAS**: de-risk eligible은 인기 3,426 아이템의 heavy buyer로 편향(active 11,888 vs sparse 112). full-population의 sparse/cold cohort에서 외부지식이 가장 강하게 패배(**KAR 0.00366 vs L1 0.00647**). R-8의 de-risk +27~43%는 *population effect(+14%) × 28% pool-coverage popularity-proxy*의 합성 산물이었다.
- MD 시사점: **LLM 외부지식은 대표·cold-start 유저의 discovery를 L1 대비 개선하지 못한다**(−12.0%, cold-start −21%). 본 데이터의 discovery lever는 **content L1**이며, 두 content 모델 모두 popularity를 **+80~104%** 능가한다(=content/L1이 견고한 양성 결과). 외부지식 item-side augmentation은 일반 discovery 추천기에 권장되지 않음.
- DS 시사점: **de-risk 방법론의 한계 사례** — coverage 제약 subset에서 de-risk하면 **population-selection bias**가 유입돼 false positive를 낼 수 있다. R-8의 placebo(shuffle)는 *capacity*만 통제했을 뿐 이 selection bias는 통제하지 못했다 — full-scale 빌드 + probe_22 isolation(coverage 고정·population 변경)이 비로소 잡아냈다. **교훈: de-risk subset의 대표성(population) 검정이 placebo·multi-seed만큼 필수.**
- Motivation 연결: Triple-Sparsity cold-start는 **외부지식이 가장 크게 실패하는 지점**(full-population sparse KAR 0.00366 vs L1 0.00647)이다 — 즉 외부지식 KAR은 본 연구의 핵심 동기인 cold-start 보완에 기여하지 못했고, 그 정직한 반증과 population-bias 진단 자체가 본 단계의 기여다.

### Contribution R-10: Serendipity/Novelty/Long-tail — enrichment의 마지막 열린 recsys 차원도 NO (tie at best) (MD+DS) ★
- 동기: recsys 가치지도가 거의 다 negative로 닫혔다 — 정확도(probe_21/22 full-scale **−12%**·cold-start −21%), intra-list diversity·coverage(probe_02 C4 **−7.7%/−2.3%**). `redesign §65/§96`이 *미측정*으로 남긴 마지막 차원 = **serendipity·novelty@12·long-tail-hit**("L3의 유일한 잔존 rescue 경로"). 87% 단일구매·96% discovery 미션이라 motivation-직결. full-catalog(105,494)·**25,000 user**·discovery-native GT(immediate, 95.6% 신규)로 STRONG-power falsify — gap probe(pilot)와 달리 power 충분.
- 정직성 핵심: novelty는 *비관련 인기-낮은 아이템 추천*으로 **trivially 부풀릴 수 있다**(headline 금지) → relevance-grounded **S1(long-tail-HIT)·S2(serendipity)**만 verdict. S2 surprise threshold τ는 **L1 baseline서 frozen**(variant가 자기 임계값 못 고름). **적대 audit이 "−60%" framing이 ~94% τ-labeling 산물임을 발견**(enrichment recs가 centroid에 더 가까워 L1의 τ를 덜 넘음, McNemar p<0.0001·seed 7/99 재현) → fair한 **S2b(labeling-symmetric: 각 variant가 자기 6개 least-central rec를 flag)** 추가. 정확도·diversity는 **이미 닫힌 negative 재현 = guardrail/context**(재주장 X). 신규 `witnesses/_probe_common.py`(`item_novelty`·`longtail_exposure`·`tail_hit_count`·`serendipitous_hit_count`) + `witnesses/probe_23_serendipity.py`(7 variant × 6 metric + placebo + 결정-축 특성화) + `tests/unit/test_serendipity_metrics.py`(7 test). seed42·`--repro` byte-identical·**API $0**·prior 29 probe JSON mtime 불변·적대 audit 2종 통과.
- **★ 결과 = CLEAN NEGATIVE (tie at best, never a win) — 5/5 enrichment variant가 어떤 operationalization(S1·S2·S2b)서도 L1을 못 넘김:**
  - **matched-accuracy 결정 증거:** `L1+L2+L3` HR **−1%**(0.0128 vs L1 0.0129 = 사실상 동률)인데 fair **S2b serendipity 동률**(0.0042 vs 0.0043, rel **−0.02**, CI[−0.0010,+0.0010] **0포함**). frozen-τ S2는 −60%로 보이나(0.0009 vs 0.0023) **그건 labeling 산물**(audit 확인) → *동률이지 품질저하 아님*.
  - **나머지 4 variant 전부 LOSE(S2b CI 0제외):** L2 **−0.57**·L3 **−0.74**·EXT_prose **−0.44**·EXT_struct **−0.64**, HR도 −0.31~−0.54.
  - **novelty 함정 실증(placebo):** random-12가 novelty **19.31**(최고)·tail-exposure **0.81**(최고)이나 serendipitous hit ≈**0.0003**(≈0). enrichment도 list를 tail로 밀지만(L1+L2+L3 nov 17.50>16.93·texp 0.512>0.420) **그 tail 아이템은 사용자가 사지 않는다**(S1/S2 하락).
  - **cold-start(2-4·5-9, motivation 지점) rescue 없음:** 2-4(n=1,415) L1+L2+L3 sym-serendip 10 vs L1 16, 5-9 18 vs 17 = no rescue(tiny n → '동률/무rescue'로 읽음). probe_21 외부지식 cold-start 실패와 동형.
  - **diversity≠serendipity 확인:** L2/L3/EXT는 intra-list diversity가 L1보다 *높지만*(EXT_struct 0.130 vs 0.071) serendipity는 낮음 → 다양성과 관련-놀라움은 별개(probe_02 닫힌 negative와도 anti-correlate).
- MD 시사점: **enrichment의 추천 가치지도 완성 = 전 축 negative.** 정확도·diversity·coverage·serendipity 어디서도 L1 content를 넘지 못한다(serendipity는 tie at best). 따라서 LLM enrichment의 *소비자 추천* 역할은 없고, 가치는 **merchant-side(C-1 merch scenarios)·interpretable 결정-축(E2)·human-in-the-loop 제어(probe_15 capability)**에 국한 — "enrichment가 추천으로서의 역할을 할 수 있나"에 대한 정직한 종결.
- DS 시사점: **honesty 장치 다중** — novelty(cheap)를 headline서 분리, relevance-grounded S1/S2/S2b, **frozen-τ 산물을 적대 audit이 잡아 labeling-symmetric S2b로 교정**([[probe-reproduce-before-verdict]]: 정의 디테일이 verdict *크기*를 바꿈 → effect-size+CI로, "−60%" 대신 "tie"), placebo가 novelty 함정 실증, 닫힌 diversity/coverage는 context로만, prior probe JSON mtime 불변. 재사용: `_probe_common` `score_variant`·`intra_list_diversity`·`bootstrap_delta`·`build_fixed_users`. canonical `witnesses/probe_23_result.json` + `results/figures/probe_23_serendipity.png`.
- Motivation 연결: Triple-Sparsity·discovery 미션에서 LLM enrichment가 *마지막 희망*(serendipity)에서도 L1을 못 넘음을 STRONG-power로 확정 → recsys-accuracy negative([[recsys-negative-established]])를 **모든 비-정확도 축(diversity·coverage·serendipity·novelty)으로 확장**해 닫음. 가치는 consumer-recsys lift가 아니라 merchant/해석/제어.

## Phase E2: Enrichment v2 (Interpretable Catalog Decision-Axes)

### Contribution E2-1: 6 결정-축 스키마 + 멀티모달 pilot + DE1 re-screen — 가치는 LLM 인식이 아니라 행동 grounding+gap (MD+DS) ★
- 동기: DE1이 기존 20속성 중 2개만 SALVAGEABLE(L2/L3 0/12)로 판정 — 원인은 v1 프롬프트가 **metadata를 LLM에 보여줘 재코딩**(color_harmony metadata-lift=1.00)·집중(occasion 81% "Everyday"). DE1 gap 6축(trend-phase·price-tier·fine-occasion·outfit-role·body-fit·care)을 *DE1을 통과하도록* 재설계하고, **동일 DE1 엔진·임계값(seed 42)**으로 적대 재검정.
- **스키마 설계 (3-family, 14 e2 컬럼):** ① LLM 추출 9축(occasion primary/secondary/formality·fit_intent·body_ease·care_burden·care_flags·price_look·trend_look) — **metadata 미노출 멀티모달**(이미지+상품명+material/care-strip된 detail_desc; 상품명 fabric-word도 제거 → leak-check clean). ② 행동파생 3축(price_tier=product_group내 quintile·trend_phase=매출 momentum·outfit_role=co-purchase 그래프 degree/방향/다양성을 product_group에 residualize). ③ gap 2축(value_gap=price_look−price_tier_rank·trend_gap=trend_look_rank−phase_rank) — 두 metadata-직교 입력의 잔차.
- **Pilot 추출:** Kaggle 전체 이미지 **105,100장** 확보 → 층화 표본 **500 product_code**(≥10 구매 floor, 21 strata, median 10,776 구매) → **5,354 article, 100% coverage, 전건 이미지 사용, $0.093**(gpt-4.1-nano). `src/knowledge/enrichment_v2/`(schema·prompts·validator·extractor·sampling) + `src/features/behavioral_axes.py` + `witnesses/probe_DE1_v2_new_attributes.py`(DE1 엔진 재사용, two-population, power flag), 10 unit test·ruff/black clean.
- **★ DE1 re-screen 결과 (DECISION=GO; 5/12 strong-gate 통과, 2 SALVAGEABLE; seed 42 byte-identical 재현):**
  - **행동파생(full catalog, STRONG):** `e2_trend_phase_actual` disc=0.7551·meta_p=0.0954·l1_p=0.0649·behav=0.0565 → **SALVAGEABLE**; `e2_outfit_role` disc=0.8518·meta_p=0.2812·l1_p=0.2409·behav=0.1547 → **SALVAGEABLE**(예측된 최고-위험 축을 product_group residualize로 구제); `e2_price_tier_actual` disc=1.0·top1=0.2001·behav=0.0 → **WEAK**(변별·비중복은 통과, 행동 inert).
  - **LLM 인식축(pilot 5,338, 행동 PRELIMINARY): 9축 전부 실패** — `e2_occasion_primary` meta_p=0.657·**l1_p=0.8459** → REDUNDANT(단 top1=0.4528로 **집중 문제는 해결**: 구 0.81→0.45, "Everyday" 삭제 효과), `e2_fit_intent` l1_p=0.778·`e2_care_flags` l1_p=0.7125 등 REDUNDANT(l1_p 0.71–0.85), `e2_care_burden` top1=0.7743·`e2_trend_look` top1=0.7241 → CONCENTRATED.
  - **gap축: 비중복 통과·행동 inert** — `e2_value_gap` disc=0.8077·meta_p=0.3506·l1_p=0.5496·top1=0.2819 / `e2_trend_gap` disc=0.8187·meta_p=0.1326·l1_p=0.2416 → 둘 다 strong-gate 통과(metadata·L1 직교) but behav≈0 → WEAK. 요약 카운트: SALVAGEABLE 2 / CONCENTRATED 2 / REDUNDANT 7 / WEAK 3.
- MD 시사점: **metadata 숨기기는 metadata-재코딩과 집중을 고쳤다**(occasion meta_p 0.66 vs 구 color_harmony 1.00; occasion top1 0.81→0.45). **그러나 LLM 인식축은 여전히 L1과 redundant**(l1_p 0.71–0.85) — 같은 이미지에서 뽑은 L1 content 임베딩이 이미 그 축을 담고 있다(**probe_14 "L2/L3=L1의 함수"가 metadata 제거 후에도 생존**, product-internal redescription). **비중복 결정-축은 LLM이 관측할 수 없는 것에서만 나온다**: 행동파생(매출 momentum·co-purchase)과 인식×행동 gap(value_gap/trend_gap, metadata·L1 양쪽에 직교). → enrichment 방향은 GO이나, 가치는 *LLM 이미지 인식이 아니라 행동 grounding + perception×behavior gap*.
- DS 시사점: 재사용 인프라 — DE1 엔진을 두-population(full STRONG / pilot PRELIMINARY)·power-flag로 확장, 행동축은 full 105K에서 screen해 인기-편향(pilot에서 outfit_role top1 0.80 붕괴) 회피. metadata-free 멀티모달 추출 모듈 + co-purchase 그래프 축 + gap 잔차축, seed 42 재현·$0.093.
- Motivation 연결: Triple-Sparsity에서 LLM 속성의 가치는 예측이 아니라 metadata가 못 가진 결정-축([[recsys-negative-established]]) — 이번 결과는 그 결정-축이 **LLM의 이미지 인식이 아니라 행동(LLM 미관측)·인식×행동 gap**에서 온다고 sharp하게 규명, recsys-accuracy negative의 메커니즘을 *L1-redundancy*로 한 단계 deepen. 다음: 통과 축(trend_phase·outfit_role·gap)으로 4-use value matrix 채움.

### Contribution E2-2: 4-use Value Matrix — capability는 dense(14/16), lift는 정확히 예측된 2 cell에만 (MD+DS) ★
- 동기: E2-1이 통과시킨 4축(trend_phase·outfit_role·value_gap·trend_gap)의 *가치를 특성화*. 행=4축, 열=4 use(①faceted/control·②trend lead-time·③merchandising·④marketing), 각 cell = (a) **capability**(metadata 없는 결정 차원인가, 구성+DE1 비중복으로) + (b) **measured decision-lift**(metadata baseline을 행동지표로 이기나, 유의성 포함). thesis "가치=예측 아닌 변별 가능한 결정-축"의 **falsification 도구**: capability-dense/lift-sparse로 돌아오도록 사전등록, 그 결과를 thesis-confirming으로 처리. DE1 하위지표로 cell별 기대를 사전 calibrate(outfit_role sellthrough excess +0.309 → ③ 예상 PASS; trend_phase seasonality excess +0.113 → ② 예상 real; gap축 excess 0 → capability-only).
- **방법(D3/D5/DE1 엔진 재사용):** ① D3 oracle-steering(attribute-agnostic, e2축 swap)·precision_ctx + discovery_map@12(immediate-split NEW-only). ② **신규** `src/features/lead_lag.py`: 월별 attribute-share(category)→category sales(t+k) Pearson, **permutation-null**(label shuffle) baseline, block-bootstrap over categories. ③ DE1 `_eta`(sell-through velocity) excess vs matched-metadata + **placebo**(random partition). ④ segment-divergence(독립 KPI **repurchase_rate**로 — 행동파생 축의 velocity 자기참조 회피) + practical-margin(≥0.10·std) guard. `src/features/enrichment_matrix.py`(gap축 영속화 + sell-through) + `witnesses/probe_E2_value_matrix.py`, 6 unit test·ruff/black clean·seed 42 재현·**API $0**.
- **★ 결과 = E2 GO (capability 14/16, strong lift PASS 2/16; `witnesses/probe_E2_result.json`):**
  - **②`trend_phase`→lead-time PASS:** share(t)가 category sales(t+3개월)를 lead, **r=0.472 vs permuted-null 0.062**(Δ=0.4107, **CI[0.194,0.640]** 0배제, best_lag=**3mo**, 10 categories). 속성 momentum이 판매를 *선행* 포착.
  - **③`outfit_role`→merch PASS:** sell-through **η=0.623 vs metadata 0.564**(excess +0.0592, **CI[0.046,0.069]**), **placebo η=0.003**(자기참조 아님). co-purchase 역할이 metadata 넘어 velocity 변별.
  - **①faceted = MARGINAL(둘 다):** oracle context-steering이 discovery를 **trend_phase +97%·outfit_role +24%** 끌어올림(intent 포착 = capability 강함)이나 임의 제어 비용 큼(off_cost_rel 0.92/0.96) → D3-style CONDITIONAL.
  - **④audience = MARGINAL(둘 다):** repurchase-divergence 0.008 vs metadata 0.006 — practical-margin(0.10·std) 미달(perm p=0이나 trivial) → **컬럼 ④는 사전등록대로 disappoint**(D5 eff_k −0.97 재확인). gap축 ③/④ = NO(excess<0)·①= N/A-COVERAGE·outfit/value_gap→② = N/A-SEMANTICS.
- MD 시사점: enrichment-v2 축은 **결정-AXES(capability 14/16)지만 행동지표로 metadata를 이기는 cell은 정확히 2개**(trend→lead-time, outfit-role→merch) — *가치는 broad prediction이 아니라 특정 결정-축의 lift*. 두 lift는 실무 직결: 속성 momentum 3개월 **early-warning**(buying/planning) + co-purchase 역할 **merchandising velocity** 신호. ①faceted는 oracle 상한(+97%)으론 강하나 배포 제어 비용이 큼(정직한 CONDITIONAL). gap축은 해석 axis(hidden-value/trend-risk)지 예측 axis 아님.
- DS 시사점: **capability와 lift를 cell마다 분리 보고**(fused score 금지), N/A를 1급 verdict로, pilot-5K gap축 PRELIM 격리. **자기참조 함정 2건 명시적 차단** — ④는 sales-derived velocity 대신 독립 repurchase_rate, ③은 placebo로 coupling 통제. large-n trivial-delta는 practical-margin(η excess≥0.05·Δcorr≥0.10·div≥0.10·std)으로 guard. 재사용 인프라: lead-lag + sell-through 모듈, value-matrix probe(figure 포함).
- Motivation 연결: **PARTIAL이 아니라 GO지만 "lift는 2 cell 국한"** = thesis("가치=예측 아닌 결정-축")를 정량 확정. Triple-Sparsity에서 LLM-증강 catalog의 실전 가치 = recsys 정확도(negative)도 broad prediction도 아닌 **(i) 행동 momentum의 early-warning lead-time + (ii) co-purchase 역할의 merchandising 신호 + (iii) steerable intent facet(oracle 상한)**. 다음: lead-time/merch cell 실서비스 시나리오화, gap축 decision-lift 별도 검증, 음악 교차도메인 복제.

### Contribution E2-3: Value Matrix 강화 — lift 2→3 (③ 진짜 강화 + ①②④ 정직 반증) (MD+DS) ★
- 동기: 각 use-컬럼의 contribution을 *정직하게* 높이려 재설계 — 임계값 완화(p-hacking) 금지, **더 나은 target·세밀한 granularity·decision-relevant outcome**로만. E2-2(`probe_E2_result.json`, lift 2/16) 불변 보존 + before/after(`probe_E2b_result.json`). 컬럼당 PRIMARY 1개 사전등록, 나머지 descriptive, 임계값 불변(η excess≥0.05·practical-margin·CI 0배제·placebo<0.05). seed42 byte-identical 재현·API $0. 신규 `src/features/audience_signals.py`(buyer-population) + `enrichment_matrix.compute_merch_signals`(markdown/first-week/online) + `lead_lag` weekly/continuous + `witnesses/probe_E2b_value_matrix.py`(11 unit test).
- **★ 결과 = lift PASS 2/16 → 3/16 (+1 genuine, 0 regression; capability 14/16 불변):**
  - **③ merch 진짜 강화 — `trend_phase`→merch NEW PASS**: E2-2의 tautological velocity(NO, excess −0.13)를 *결정-relevant* outcome **first_week_sell_through**(런칭 sell-through)로 교체 → η=0.673 excess **+0.45**(CI[0.43,0.46], **placebo 0.008**, product_group residualize). `outfit_role`→merch(velocity +0.097, placebo 0.003) 유지. **두 행동축 모두 merch 통과.**
  - **① faceted 반증(undeployable)**: +97%/+24% oracle ctx-gain은 *배포 불가* — 현실적 history-mode predictor의 deployable gain = **0.0**(recovery 0/0.97). predictor는 유저의 *과거* modal 축값으로 steer하나 discovery는 *새* 아이템(다른 값) → control은 oracle-only ceiling. **NO.**
  - **② refinement 반증(weekly noisier)**: weekly+continuous-momentum이 monthly-binary보다 **noisier**(연속 momentum permutation-null이 real share를 너무 닮음) → 개선 실패. monthly-binary PASS(0.41) 유지, weekly는 descriptive null.
  - **④ audience 반증(category-orthogonal)**: target을 item-repurchase→**buyer-population**(age/channel)으로 교정했으나 metadata k-means가 buyer를 **더 잘** 분리(trend_phase age-div 0.38 vs meta 1.16=0.33×; outfit_role 0.49 vs 1.16=0.43×; online도 둘 다 패배). buyer demographics는 product **category**가 결정 → 축은 category-직교(DE1)라 *누가 사는지*를 metadata만큼 못 가름. **NO.**
- MD 시사점: **contribution은 정직하게 1 컬럼만 올랐다(③).** 강화의 메커니즘 = "E2-2 velocity가 tautological(momentum=매출률)이라 NO였던 걸, momentum이 *진짜* 예측하는 launch-timing(first-week)으로 바꾸니 PASS." **나머지 3 컬럼은 반증** — ①제어는 oracle-only(배포불가), ②weekly는 noisier, ④축은 audience-segmenter가 아님(demographics=category-driven). **메타 발견: value matrix는 정직한 ceiling 근처 — genuine lift는 *축이 metadata 없는 sales/co-purchase 신호를 담는 정확히 그 지점*(trend momentum→lead-time·launch-merch; co-purchase→velocity-merch = 3 cell)에만 존재하고, 제어·audience엔 없다.** 이는 thesis를 한 단계 더 sharpen: 축은 **merchandising/trend 결정 신호이지 steering knob도 audience segment도 아니다.**
- DS 시사점: **정직성 장치로 강화의 진위 검증** — ④ 자기참조 교정(repurchase→demographics)에도 반증 유지, ③ placebo로 coupling 통제, ②는 established method를 primary로 보존(regression 회피), 임계값 불변, capability/lift 분리, E2-2 canonical mtime 불변 검증. 재사용: buyer-population 3-way join 모듈, weekly/continuous lead-lag, rich merch signals. `results/figures/E2b_value_matrix.png`(★=new PASS).
- Motivation 연결: "각 컬럼 contribution↑" 요청에 대한 정직한 답 = **③만 정당하게 강화(+1), ①②④는 반증** — 그리고 그 반증들이 *축의 가치 위치를 정밀하게 규명*(merch/trend 신호 O, 제어·audience X). Triple-Sparsity에서 LLM-증강의 실전 가치는 더 좁고 더 선명: **trend momentum의 lead-time + launch-merch, co-purchase의 velocity-merch.** 다음: 3 PASS cell 실서비스화 + ① 정교한 predictor(trend-aware) 탐색 + 음악 교차.

### Contribution E2-4: 2-Source × 4-Use Value Decomposition — KAR user/reasoning leg가 ④audience 도달, ①control은 capability-PASS/lift-NO (MD+DS) ★
- 동기: E2-2/E2-3 value matrix는 *item-side* enrichment축만 사용 → lift는 ②lead-time·③merch(아이템 결정)에만 났다. **①control·④audience는 둘 다 USER 결정인데 user-side 강화가 한 번도 투입된 적이 없었다.** KAR는 비대칭 — **Item→Factual expert, User→Reasoning expert** — 이고 지금까지 factual/item leg만 돌렸다. 자산은 이미 on-disk: `data/knowledge/reasoning/user_profiles.parquet`(1.3M 유저, 876,790 active LLM 프로파일, 9 선호필드 in `reasoning_json`) + `data/embeddings/user_bge_embeddings.npz`(1.3M×768, ReasoningExpert 입력). user-reasoning 강화를 정확히 ①④에 붙여 **2-source × 4-use 분해**로 KAR 완성.
- 정직성 핵심(make-or-break): `reasoning_json`은 유저의 **train 구매에서 도출**(L1 집계 + 구매 아이템 LLM prose) → "reasoning이 행동 예측"은 outcome이 train-derived면 **tautology**. **Fix(모든 cell):** outcome = **held-out FUTURE 행동**(val 2020-07-01~08-31), predictor = train-derived reasoning, baseline = train-derived **11 demographic features**(`src/features/engineering.py`, 8 numeric + 3 categorical) → reasoning-vs-demographics가 *미래예측 apples-to-apples*. **established negative 존중:** LLM L2/L3+외부지식은 recsys *ranking 정확도*를 개선 못함(multiply confirmed, reasoning 포함) → user-side ①④는 **CONTROL(steerability)·AUDIENCE(future-behavior segmentation)**로 엄격히 framing하고, "reasoning이 recsys 정확도 개선"으로 절대 frame 안 한다.
- 신규 `src/features/user_axes.py`(reasoning_bge 768→PCA-50 + isotropy / reasoning_fields TF-IDF+SVD-50+L1agg / demographic 11-feat baseline + FUTURE outcome) + `witnesses/probe_E2c_user_value.py`(11 unit test, E2/E2b steer·D5 cv-predict spine 재사용, E2/E2b JSON mtime 불변 assert). CPU, seed42, **API $0**, quick×2 byte-identical 재현.
- **★ 결과 = KAR-SYMMETRY CONFIRMED — item→②③ / user→④(modest)/①(NO); n_cohort=40,000, baseline=11 demographic features:**
  - **④ audience = modest PASS (둘 다, power STRONG)** — 오직 `fut_top_group`(미래 카테고리믹스, 19 product_group): `reasoning_bge` demo_f1 0.045→both 0.0567 **Δ+0.0117**(p=6e-05, significant) · `reasoning_fields` demo 0.045→both 0.0596 **Δ+0.0145**(p=2.4e-04, significant). 증분은 **LLM prose 귀속**(reason-alone<demo, demo+reason>demo). ④b divergence는 **혼재**: `reasoning_fields` ratio **1.973**(PASS≥1.20, eff_k 3.49) vs `reasoning_bge` ratio **0.353**(rep<demo, FAIL) → ④ PASS 판정은 ④a predictive에 근거.
  - **정직 단서(effect-size):** ④는 **small-but-robust, ~+1pp at the 1pp bar**, *카테고리믹스 한 outcome*에만. `fut_price_tier`는 바 *아래*(Δ+0.0077 bge / +0.0064 fields, practical-margin 미달). 습관축(`fut_online`/`fut_repurchase`)은 NULL→demo 승.
  - **① control = NO (둘 다, power STRONG)** — reasoning이 유저 미래 facet(`fut_outfit_role`/`fut_trend_phase`) 예측에서 demo 못 넘음(Δ+0.0003/+0.0006 bge, −0.0001/+0.0001 fields, ns).
  - **②③ = N/A-SEMANTICS** — user rep엔 item-momentum 없고(②) per-item η의 grouping var도 아님(③). symmetry(item→②③ / user→①④) 확인; 여기 non-NULL이면 leak flag.
- **★ control① 최종 처리(사용자 결정 A — capability-PASS 재정의):** ①은 *lift/예측* 셀로 **두 번 실패**(E2-3 deployable history-predictor gain **0.0**; E2-4 reasoning **NO**·power STRONG). 따라서 ①을 **capability-PASS / lift-NO**로 정직하게 닫는다 — enrichment축은 **배포 가능한 human-in-the-loop faceted 제어 표면**을 제공(이미 canonical: D3/probe_15 steered precision **1.00 vs 무제어 0.14**, gain +0.86, metadata엔 없는 8 제어축)하나 **automatic-personalization lift는 아니다**. thesis("가치=결정-축, 예측 아님")와 정합.
- MD 시사점: **2-source 분해가 KAR의 가치 위치를 깨끗이 분리** — 아이템강화→②③(merch/trend 신호), 사용자강화→④(modest audience). ①control은 *어느 source로도 lift가 안 나오는* 칸이고, 그 정직한 결론은 "①은 lift가 아니라 **capability**(제어 표면)"이다. ④조차 효과는 작고(~+1pp) *미래 카테고리믹스/가격(taste)* 쪽에만, **습관(채널·repurchase)은 demographics가 이미 잡는다** → "LLM-enrichment가 audience를 가른다"가 아니라 "미래 taste 위에 demographics 대비 thin하게 더한다".
- DS 시사점: **정직성 장치 다중** — outcome FUTURE-only(provenance assert), 두 rep 모두 train-derived, practical-margin 0.01·permutation null(300/1000)·eff-k guard, E2/E2b canonical mtime 불변 검증, ④는 binary가 아니라 **effect-size**로 보고([[probe-reproduce-before-verdict]]: 효과가 1pp 바 근처라 측정 디테일이 verdict를 뒤집을 수 있음 → DuckDB `ROW_NUMBER` tie-break·deterministic 재현 확정 후에만 보고). 재사용: E2 steer spine·D5 `_cv_predict`/`_paired_sig`/`_effective_k`·`audience_signals.segment_divergence_weighted`. canonical: `witnesses/probe_E2c_user_value.json` + `results/figures/E2c_user_value.png`(2-source×4-use heatmap).
- Motivation 연결: KAR factual+reasoning 2-Expert 가치를 **셀 단위로 위치확정** — Triple-Sparsity에서 LLM-증강의 실전 가치는 (i) 아이템: trend-momentum lead-time + launch/velocity merch, (ii) 유저: 미래 카테고리믹스 audience(thin), (iii) 제어: 배포 가능 faceted 제어 표면(**capability**), (iv) recsys 정확도·자동제어 lift는 정직한 **negative**. 다음: 3 PASS cell 실서비스화 + gap축 decision-lift + 음악 교차도메인.

### Contribution E2-5: gap축 FUTURE decision-lift — value_gap/trend_gap는 자기 구성축 대비 미래 결정값을 더하지 못한다 = 해석 axis 확정 (MD+DS) ★
- 동기: gap축(`value_gap`=price_look−rank(price_tier)·`trend_gap`=rank(trend_look)−rank(trend_phase))은 E2-1서 **capability/비중복은 통과**(meta_p 0.35/0.13·l1_p 0.55/0.24, metadata·L1 직교)했으나 E2-2/E2-3 4 use 전부 **행동 inert**(faceted N/A·leadlag N/A/≈0·merch NO·audience NO)였다. 그 prior probe들은 gap을 *train-window* outcome의 grouping var로만 봤다. gap의 진짜 해석 주장("hidden-value/trend-risk")은 **인식↔행동 mismatch 자체가 미래지향 결정 신호**라는 것 — 이를 *가장 유리한·미검증* 각도로 falsify. (C 백로그 b)
- 정직성 핵심: 이전 probe와 구별되는 **두 미검증 pillar**만 측정. (1) **FUTURE-held-out** outcome(val 2020-07~08, train-frozen reference — 신규 `build_article_future_outcomes`, `PRAGMA threads=1`로 AVG 결정성). (2) **자기 구성축 대비 incremental** — baseline=**one-hot(c1)+one-hot(c2)+one-hot(product_group)**(η-vs-metadata 템플릿 아님, 그건 알려진 NO 재측정). gap=c1−c2가 ordinal과 **결정론적 collinear**라서 구성축을 *one-hot*으로 넣어 directional mismatch(sign+|gap|)를 additive level이 못 span하는 interaction contrast로 만든다(이걸 ordinal로 넣으면 Δ≡0 기계적 거짓-NO). **5 게이트**(margin·paired-sig·CI>0·within-group placebo≈0·**sign-randomization이 효과를 죽여야**)+**두 cohort 복제**(n_val≥5·≥20). power는 pilot-only(~5.3K)라 **PRELIMINARY by construction** → verdict는 `PRELIM`만(PASS 미발행), 항상 effect-size+CI.
- 신규 `src/features/enrichment_matrix.build_article_future_outcomes`(per-article FUTURE outcome, `compute_sell_through`/`compute_merch_signals`를 val에 재사용, train-frozen edge) + `witnesses/probe_E2d_gap_decision.py`(2 gap축 × 4 결정[markdown-risk·hidden-gem·overhype/sleeper·survival], 2 readout[incremental paired-fold + decision-rule precision@flag], placebo 2종, Ridge ΔR²·partial-corr robustness, E2/E2b JSON mtime 불변 assert) + `tests/unit/test_e2d_gap_decision.py`(8 test). CPU·seed42·**API $0**·`--repro` byte-identical.
- **★ 결과 = CLEAN NEGATIVE — 5/5 cell이 0.01 macro-F1 practical margin 미달(PRELIM 0, n_val≥5 cohort ~2,017/1,999, val-survivor 2,823):**
  - **value_gap→markdown-risk = MARGINAL/NO**: incrΔF1 **+0.0031**(p=0.25, ns)·acc_CI[−0.016,+0.004] 0포함·decision-rule lift **0.728(<1)** → gap≤−2 "overpriced-look" flag가 미래 markdown을 base보다 *못* 맞춤. Ridge ΔR² **−0.0039**(연속 reframe도 음수).
  - **value_gap→hidden-gem = NO**: incrΔF1 **−0.0081**·rule lift 0.765. gap>0 "deal"이 미래 sell-through 가속 안 함.
  - **trend_gap→overhype/sleeper = MARGINAL**: incrΔF1 **+0.0087**(p=0.019 *통계*적 유의이나 0.01 practical 미달) — 그러나 (i) **sign-randomization placebo +0.0067**(효과 ~77% 잔존 → 방향 mismatch 아닌 trend_phase **mean-reversion**), (ii) **≥20 cohort robust −0.0022**(부호 뒤집힘, 비복제), (iii) `sat_dominates=True`, (iv) **raw corr +0.1066 → momentum partial-out 후 −0.0146**(평균회귀 붕괴), (v) Ridge ΔR² **−0.0019**(p≈0.92, 연속 reframe서 *악화*) — 네 독립 근거로 기각.
  - **survival(둘 다) = MARGINAL**: value `+0.0035`(p=0.36, shuffle placebo +0.0081≈효과=카테고리 노이즈)·trend `+0.006`(p=0.0007이나 large-n trivial). deployable flag |gap|≥2: value lift 1.033·trend lift **0.888**(P(survival) 0.508 vs base 0.572 — 극단 gap은 *덜* 생존) → 방향이 반대라 배포 불가.
- MD 시사점: **gap축은 비중복 *해석/진단* 좌표이지 *예측·결정* 축이 아님이 확정.** capability(metadata·L1 직교)는 있으나, 미래·구성축-통제·배포규칙 세 각도로 가장 유리하게 줘도 **자기 구성축을 못 넘는다**. 머천다이저에게 gap은 "이 아이템은 보이는 값/트렌드와 행동이 어긋난다"는 *설명 가능한 라벨*로는 쓰되, markdown·재고·hidden-gem **결정 자동화 신호로는 쓰지 말 것**(C-1이 이미 gap축을 제품화 X·맥락화 O로 처리한 것과 정합).
- DS 시사점: **정직성 장치 다중** — collinearity 함정 회피(one-hot levels, 단위테스트가 pipeline 생존 입증: 합성 gap 신호 +0.24 F1 검출), FUTURE-only outcome(train-frozen, leak 없음), **sign-randomization이 confound-driven 거짓양성을 NO로 강등**, 2-cohort 복제, Ridge/partial-corr robustness가 canonical JSON에 기록(재현). practical-margin 0.01은 D5·E2c와 **동일 코드**(`_paired_sig` import)라 사후조정 아님 — E2c도 같은 바로 +0.0077/p4e-05를 강등했다. 두 적대 audit(false-negative·suppressed-effect)이 독립 확인. canonical: `witnesses/probe_E2d_gap_decision.json` + `results/figures/E2d_gap_decision.png`. E2/E2b mtime 불변.
- Motivation 연결: Triple-Sparsity에서 LLM-enrichment의 가치 지도를 **셀 단위로 마감** — value matrix의 마지막 미검증 행(gap축)을 *미래-held-out·구성축-통제·배포규칙* 3중으로 닫아, "**capability(해석 좌표)는 있으나 forecasting lift는 없다**"는 thesis를 가장 강한 형태로 sharpen. 측정 lift는 여전히 item 행동축 3 cell(trend→lead-time·trend→merch·outfit→merch) + user ④audience(thin)에만; gap은 *descriptive transparency*에 그친다. 다음(C 백로그): (c) 음악 교차도메인(D4 feasibility 먼저).

## Phase C: Productization (3 PASS cell 실서비스 시나리오화)

### Contribution C-1: 3 PASS cell → 머천다이징 의사결정-지원 엔진 (build, product-design) ★
- 성격: **연구 probe가 아니라 build** — value matrix(E2-2/E2-3)가 정직하게 닫은 **behavior-validated lift 3/16 cell**을 머천다이저가 쓰는 **batch 의사결정-지원 brief**로 제품화. 3 PASS는 모두 **행동-파생 축**(LLM 인식축 9/9는 redundant라 제외): (A) `trend_phase`→②lead-time, (B) `trend_phase`→③merch, (C) `outfit_role`→③merch.
- 신규 `src/serving/merch_scenarios.py`(NamedTuple `ScenarioConfig`/`ConfidenceCard`/`ScenarioBrief` + 3 brief 함수 + `build_all_briefs`/`value_matrix_posture`, 기존 `lead_lag.py`·`enrichment_matrix.py` feature 함수 재사용) + `scripts/serve_scenarios.py`(Typer CLI) + `notebooks/06_merch_scenario.ipynb`(builder 생성, 4 figure) + `tests/unit/test_merch_scenarios.py`(8 test PASS). CPU/DuckDB, **API $0**.
- **정직성 원칙(drift guard):** 운영 brief 테이블은 데이터에서 fresh 계산하되, *confidence 수치*(r·η·CI·verdict)는 canonical `probe_E2*.json`에서 **로드**(재계산 안 함) → value matrix가 single source of truth. lead-lag는 reuse로 lag-3 **r=0.4723 deterministic 재현**(테스트 가드). canonical(`probe_E2*.json`·`E2*.png`) **mtime 불변**.
- **3 PASS cell 운영화 + 검증된 신호:**
  - **A. Trend lead-time** — 카테고리 hot(Emerging+Rising) share가 매출을 **3개월 선행**(r=**0.472** vs permutation null 0.062, CI[0.194,0.640], lag=3mo, 10 카테고리). 운영: 10 카테고리를 현재 hot-share z-score로 랭킹 → 3개월 수요-상승 조기경보 테이블.
  - **B. Launch signal** — trend_phase가 신규 런칭 first_week_sell_through 설명(η=**0.673** resid product_group vs metadata 0.223, CI[0.428,0.458], placebo 0.008). 운영: hot cohort(최소구매 floor 10) 예상 sell-through 티어 스코어카드.
  - **C. Co-purchase velocity** — outfit_role이 velocity 설명(η=**0.631** resid product_group vs metadata 0.534, CI[0.085,0.106], placebo 0.002). 운영: anchor 역할(Anchor-hub·Versatile-connector) 아이템을 velocity로 랭킹 + 번들 역할 라벨.
- **정직한 posture(제품화 안 하는 셀 명시):** capability **14/16** dense vs lift **3/16** sparse 전체를 노출. ① faceted automatic-personalization lift=NO(capability-PASS/lift-NO, deployable gain 0.0), ④ audience=NO(category-직교), gap축(value_gap·trend_gap)=NO(행동 inert), 추천-정확도=별도 negative(full-scale −12%) — 전부 *제품화 X, 맥락화 O*.
- MD 시사점: enrichment의 실전 가치를 **3개의 구체 머천다이징 워크플로**(category buyer 3mo 조기경보 / launch planner first-week 재고배분 / cross-sell merchandiser 번들 앵커)로 번역. capability(있음)와 deployable lift(3 cell만)를 시각·수치로 구분 — "쓸 수 있는 신호"와 "안 쓰는 신호"를 같은 화면에서 정직하게 제시.
- DS 시사점: 연구 canonical을 single source of truth로 두고 confidence를 로드(재계산 금지)해 문서·코드·figure 간 수치 drift 차단. brief 운영 로직은 feature 모듈 재사용으로 중복 0. 8 test가 (i) canonical 일치 가드 (ii) lead-lag deterministic 재현 (iii) brief well-formedness를 커버. ruff/black clean.
- Motivation 연결: Triple-Sparsity에서 CF·추천정확도가 막힌 자리(recsys negative)를 우회해, LLM/행동-enrichment의 *배포 가능한* 가치를 **머천다이징 의사결정-지원**(예측 아닌 결정-축)으로 실현 — thesis("가치=해석 가능한 결정-축")의 product-side 실증. 다음(C 백로그): (b) gap축 decision-lift 별도 probe, (c) 음악 교차도메인.

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
| R (재설계) | 병목 교정(검증 3.5h/epoch→배치, 686 pass·회귀 0·신규 9 등가성), confound 해소(meta.npz), Gate-0 falsification: C1 +130.9%·C2 +8.0%·C3 −13.5%(CI 0제외), robustness 21 cells(C3 18/21·11유의·양수0, C2 17/21·k12 cold-start 비유의), 적대검증 A=강한win·B=weak·C=frozen-only-drop, length-confound 반증. **Hybrid foundation(R-4)**: immediate-split repurchase MAP@12=0.02374(probe_05 0.0243 재현), recent_pop=0.00307·global_pop=0.00351, discovery_map(repurchase)=0.00042(≈0), decomposition new=95.9%/repurchase=4.1%, 신규 테스트 23 PASS(회귀 0). **LLM 처분(R-5)**: content standalone discovery는 popularity에 짐(probe_06), **re-ranker로는 +93%**(probe_07 0.00384 vs 0.00199), layer 분해(probe_08) L1=+95% vs pop·**L1→L2 −0.8%(CI 0포함)**·L2→L3 −4.6% → **L1=헤드라인, L2/L3=redundancy falsification** | "자원 문제" 재진단=1개 병목; **3-Layer→L1 강력 기여 / L2·L3는 L1과 redundant(정확도 증분 0)로 정직 확정**; LLM=candidate-pool re-ranker; Kaggle-comparable 0.024 backbone + discovery_map으로 Gate-2'(trainable KAR) 준비 |
| R+ (reframe) | **진단(probe_14)**: L1→L2/L3 kNN 예측 평균 lift=0.38(L2 0.37/L3 0.38), style_mood 0.71(21cls)·style_lineage 0.51(44cls) → L2/L3=L1의 (준)함수(product-internal). **control(probe_15)**: steered 정밀도 1.00 vs 무제어 0.14(gain +0.86), 개인화 100% 유지, metadata엔 없는 8 제어축. **외부지식(probe_16)**: external_knowledge HR@12=0.2366 vs L1_cos 0.2108 **+12.2%**(CI [+0.0103,+0.0413], 0배제) vs popularity 0.0158 → 외부 styling 지식이 product-sim 능가 | "L2/L3 실패"의 **원인=제품-내부 재서술** 규명 + **수정(control·외부지식) 작동** 검증. 가치 축 재정의: 정확도→controllability+외부지식 complementarity. KAR open-world-knowledge 비전 본 도메인 첫 실증(+12.2%). falsify→진단→검증된 pivot = 연구 성숙도 |
| 1b/1c (full build) | **추출(1b)**: 47,224 product_code → 105,542 article, **100% coverage**(prose+structured), **$4.03**(gpt-4.1-nano). **Gate-2'(probe_21, full coverage 105,494·eligible 40,000·3 seed)**: popularity 0.00236 / L1 **0.00482** / KAR_external **0.00424** → **KAR−L1=−12.0%(0/3) NO-GO**(둘 다 pop +80~104%), cold-start L1 0.00719 vs KAR 0.00568. **isolation(probe_22)**: de-risk-pop KAR 0.004062 vs L1 0.003561 **+14.1%(3/3) GO** / full-pop KAR 0.004363 vs L1 0.004800 **−9.1%(1/3) NO-GO** → **POPULATION-SELECTION BIAS**(de-risk eligible=인기 3,426 heavy buyer) | ⚠️ **방향교정**: full-scale 빌드가 R-8 de-risk REVIVE를 **반증** — LLM 외부지식은 **대표·cold-start 유저 discovery를 L1 대비 개선 못함**(Triple-Sparsity cold-start에서 가장 크게 실패: full-pop sparse KAR 0.00366 vs L1 0.00647). 견고한 양성 결과 = **content L1이 popularity +80~104% 능가**. de-risk subset의 population-selection bias가 false positive 유발 → **de-risk→scale 반증의 정직성**이 본 단계 기여 |
| E2 (enrichment v2) | **재설계 6축 DE1 re-screen = GO**(5/12 strong-gate, 2 SALVAGEABLE, seed42 재현). 행동파생: trend_phase(meta_p 0.10·l1_p 0.06·behav 0.057) + outfit_role(meta_p 0.28·behav 0.155) **SALVAGEABLE**, price_tier WEAK(inert). **LLM 인식축 9/9 실패**: occasion meta_p 0.66·**l1_p 0.85**(집중은 해결 top1 0.81→0.45) 등 REDUNDANT, care_burden top1 0.77·trend_look 0.72 CONCENTRATED. gap축 value_gap/trend_gap = 비중복 통과(meta_p 0.35/0.13)·행동 inert→WEAK. pilot 500 code·5,354 art·100% cov·이미지 105,100·$0.093 | **metadata 숨기기는 metadata-재코딩·집중을 고쳤으나 LLM 인식축은 L1과 redundant(probe_14 생존, product-internal redescription)**. 비중복 결정-축 = LLM 미관측 행동 grounding + 인식×행동 gap. recsys-negative 메커니즘을 L1-redundancy로 deepen; 가치는 LLM 인식 아닌 behavior-grounding |
| E2-2 (value matrix) | **4-use value matrix = E2 GO**(capability **14/16**, strong lift PASS **2/16**, seed42 재현·API $0). lift는 사전예측된 2 cell에만: **trend_phase→②lead-time**(share→sales(t+3mo) r=0.472 vs null 0.062, Δ=0.41 **CI[0.19,0.64]**) + **outfit_role→③merch**(η 0.623 vs meta 0.564, excess +0.059 CI[0.046,0.069], placebo 0.003). ①faceted=MARGINAL(oracle ctx-steer +97%/+24%, 제어비용 큼). ④audience=MARGINAL(repurchase-div practical-margin 미달, D5 재확인). gap축 ③/④=NO·①=N/A. 신규 `lead_lag.py`+`enrichment_matrix.py`+`probe_E2`, 6 test | **GO지만 lift는 2 cell 국한 = thesis("가치=예측 아닌 결정-축") 정량 확정**. 실전 가치 = 행동 momentum 3mo **early-warning** + co-purchase **merchandising velocity** + steerable intent(oracle 상한). gap=해석 axis(hidden-value/trend-risk), 예측 axis 아님. 자기참조 차단(④ repurchase·③ placebo)·practical-margin guard로 정직성 확보 |
| E2-3 (matrix 강화) | **lift 2/16 → 3/16**(+1 genuine, 0 regression, seed42 재현, E2-2 canonical 불변). **③ `trend_phase`→merch NEW PASS**: tautological velocity(NO −0.13)→launch **first_week_sell_through** η excess **+0.45**(CI[0.43,0.46]·placebo 0.008) → 두 행동축 모두 merch 통과. **① 반증**: oracle +97% but deployable history-predictor gain **0.0**(배포불가). **② refinement 반증**: weekly+continuous noisier→monthly PASS 유지. **④ 반증**: buyer-age div axis 0.38/0.49 < metadata 1.16(0.33/0.43×)·online도 패배 → 축 category-직교라 audience 못 가름. 신규 `audience_signals`+merch-signals+weekly lead-lag+`probe_E2b` | **"각 컬럼↑" 정직한 답 = ③만 정당 강화·①②④ 반증**. 강화 메커니즘=momentum의 *진짜* target(velocity 아닌 launch-timing). 메타: genuine lift는 *축이 metadata 없는 sales/co-purchase 신호 담는 정확히 그 지점*(3 cell)에만 — 제어·audience엔 없음. thesis sharpen: 축=merch/trend 신호, steering knob도 audience segment도 아님 |
| E2-4 (KAR user leg) | **2-source × 4-use 분해 = KAR-SYMMETRY CONFIRMED**(item→②③ / user→④①, n=40,000, seed42 byte-identical·API $0, E2/E2b canonical 불변). user-reasoning을 ①④에 투입, outcome=held-out **FUTURE**(val 2020-07~08)·baseline=11 demographic. **④ audience modest PASS(둘 다, STRONG)**: 오직 `fut_top_group` — `reasoning_bge` Δ**+0.0117**(p=6e-05)·`reasoning_fields` Δ**+0.0145**(p=2.4e-04), ~+1pp at the 1pp bar; `fut_price_tier`는 바 아래(Δ+0.0077/+0.0064), 습관축(online/repurchase) NULL→demo 승; ④b div 혼재(fields 1.973 PASS·bge 0.353 FAIL). **① control NO(둘 다, STRONG)**: reasoning이 미래 facet 예측서 demo 못 넘음(Δ≈0). **②③ N/A-SEMANTICS**. 신규 `src/features/user_axes.py`+`probe_E2c`, 11 test | **KAR 가치 위치 셀단위 확정**: item→merch/trend, user→audience(thin, 미래 taste만). **① 최종 = capability-PASS/lift-NO**(D3 precision 1.00 vs 0.14 = 배포 가능 제어 표면, 자동제어 lift는 두 번 실패). LLM-증강은 audience를 *가르지* 않고 미래 카테고리믹스에 demographics 위 thin하게 더함; 습관은 demographics가 잡음. effect-size 정직 보고 |
| C-1 (productization) | **3 PASS cell → 머천다이징 엔진**(build, API $0). `src/serving/merch_scenarios.py`+`scripts/serve_scenarios.py`+`notebooks/06_merch_scenario.ipynb`(4 fig)+8 test PASS, ruff/black clean. confidence는 canonical JSON에서 **로드**(재계산 X), lead-lag lag-3 **r=0.4723 deterministic 재현**, canonical mtime 불변. A:trend_phase→lead-time(r=**0.472** vs null 0.062, lag 3mo) · B:trend_phase→merch(first_week_ST η=**0.673** vs 0.223) · C:outfit_role→merch(velocity η=**0.631** vs 0.534). posture **capability 14/16 vs lift 3/16** 전체 노출, ①④·gap축·recsys-negative는 제품화 X·맥락화 O | **연구 canonical을 깨지 않고 lift PASS 3 cell만 배포 가능 머천다이징 워크플로로 제품화**. capability(있음)와 deployable lift(3 cell)를 정직 구분. recsys-negative 우회한 enrichment의 product-side 가치 실현 = thesis("가치=결정-축") 산업 측 실증 |
| E2-5 (gap decision-lift) | **CLEAN NEGATIVE — gap축 5/5 cell이 0.01 margin 미달**(seed42·`--repro` byte-identical·API $0·E2/E2b 불변). FUTURE-held-out(val 2020-07~08, train-frozen) × 자기 구성축 one-hot baseline incremental(η-vs-metadata 아님): value→markdown incrΔF1 +0.0031(rule lift **0.728**<1·Ridge ΔR² −0.0039)·hidden-gem −0.0081; trend→overhype +0.0087(p=0.019이나 sign-rand +0.0067=mean-reversion·≥20 robust −0.0022·**raw corr +0.107→partial −0.015**); survival 둘 다 deployable flag lift ≤1.03(trend 0.888=극단 gap 덜 생존). 신규 `build_article_future_outcomes`+`probe_E2d`, 8 test, 적대 audit 2종 통과 | **gap축 = 비중복 *해석* 좌표, *예측/결정* 축 아님 확정** — capability(metadata·L1 직교)는 있으나 미래·구성축통제·배포규칙 3중에도 자기 구성축 못 넘음. thesis("가치=해석 가능한 결정-축, 예측 아님")를 value matrix 마지막 행에서 가장 강하게 sharpen; gap은 descriptive transparency에 국한 |
| R-10 (serendipity/novelty) | **enrichment의 마지막 열린 recsys 차원도 CLEAN NEGATIVE (tie at best)** — full-catalog 105,494·25,000 user·discovery-native GT, seed42·`--repro` byte-identical·API $0·prior 29 probe JSON 불변. 5/5 enrichment variant가 S1·S2·**S2b(fair, labeling-symmetric)** 어디서도 L1 못 넘김: matched-HR(−1%) `L1+L2+L3` S2b **동률**(rel −0.02, CI[−0.0010,+0.0010] 0포함; frozen-τ "−60%"는 ~94% labeling 산물=audit 교정), L2/L3/EXT는 S2b −0.44~−0.74 LOSE. **placebo random-12가 novelty 19.31·tail-exp 0.81 최고지만 serendipitous hit ≈0**(novelty 함정 실증). diversity·coverage(probe_02)·accuracy(probe_21)는 guardrail-context로 재현. 신규 `_probe_common` 4 metric + `probe_23` + 7 test, 적대 audit 2종 통과 | **recsys-accuracy negative를 모든 비-정확도 축(diversity·coverage·serendipity·novelty)으로 확장해 닫음** = enrichment의 *소비자 추천* 역할 종결. 가치 = merchant-side(C-1)·해석 결정-축(E2)·human-in-the-loop 제어(probe_15), consumer-recsys lift 아님 |
| 5 | (TBD) | 최종 성능 비교 |
