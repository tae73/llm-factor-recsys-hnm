# H&M 데이터셋 Cold-Start 분석 및 해결 전략

> EDA 노트북(`notebooks/00_eda.ipynb`)에서 도출한 정량적 분석을 기반으로, H&M 데이터의 구조적 Cold-Start 문제와 본 프로젝트의 해결 전략을 상세히 기술한다.

---

## 1. 문제 정의: Triple-Sparsity

H&M 데이터셋은 세 가지 차원에서 동시에 희소성이 발현되는 **Triple-Sparsity** 문제를 보인다. 이 복합적 희소성이 협업 필터링(CF) 기반 추천의 구조적 한계를 만든다.

### 1.1 유저 측 희소성

| 세그먼트 | 유저 수 | 비율 | 거래 비율 |
|----------|---------|------|-----------|
| Heavy (28건+) | 332,870명 | 24.4% | 73.5% |
| Moderate (5-27건) | 602,387명 | 43.5% | 23.3% |
| Light (1-4건) | 436,723명 | 32.1% | 3.2% |

- 전체 유저의 **32.1% (436,723명)**가 1-4건 구매에 불과하며, 이들이 전체 거래의 **3.2%**만 차지한다.
- 구매 분포: median=9, mean=23.3 — 우편향(right-skewed) 분포로, 소수 Heavy 유저가 거래 대다수를 생성한다.
- Light 유저는 카탈로그의 **<0.004%** 아이템과만 상호작용하므로, 유저 임베딩이 극소수 아이템 시그널에만 의존한다.

### 1.2 행렬 측 희소성

- 유저-아이템 상호작용 행렬 희소도: **99.98% sparse**
- 1.37M 유저 x 105K 아이템 = ~1,440억 셀 중 ~31M 관측값만 존재
- MF(Matrix Factorization) 모델은 관측된 상호작용에서 잠재 인자를 추정하는데, 99.98% 미관측 셀에서는 그래디언트 시그널이 사실상 부재한다.
- GNN(Graph Neural Network) 모델은 이분 그래프에서 다중 홉 이웃 정보를 집계하는데, Light 유저 노드의 이웃 수가 1-4개로 극소하여 정보 전파가 빈약하다.

### 1.3 시그널 품질 저하

Baseline 실험(Phase 0) 결과가 시그널 품질 저하의 직접적 증거를 제공한다:

| Baseline | MAP@12 | HR@12 | NDCG@12 | MRR |
|----------|--------|-------|---------|-----|
| **Popularity Global** | **0.003783** | **0.044994** | **0.008122** | **0.015481** |
| UserKNN (ALS) | 0.003036 | 0.033901 | 0.006319 | 0.012228 |
| BPR-MF | 0.001308 | 0.016069 | 0.002839 | 0.004924 |

**Popularity > UserKNN > BPR-MF** — 개인화 수준이 높아질수록 성능이 오히려 하락한다. 이는 희소한 상호작용 데이터에서 학습된 유저/아이템 임베딩이 노이즈를 포함하여, 개인화된 랭킹이 비개인화 인기도보다 못한 결과를 산출함을 의미한다.

### 1.4 Triple-Sparsity 복합 효과

세 가지 희소성은 독립적이지 않고 cascading 효과를 만든다:

```
유저 측 희소성 (32.1% Light)
    → 행렬 측 희소성 (99.98% sparse)
        → 시그널 품질 저하 (노이즈 임베딩)
            → CF 모델 실패 (Popularity > BPR-MF)
```

**결론:** 협업 필터링만으로는 H&M 데이터에서 유의미한 개인화가 구조적으로 불가능하다. CF 시그널을 보완하는 콘텐츠 기반 시그널(속성 벡터)이 필수적이다.

### 1.5 아이템 측 인기도 편중

> Figure 참조: `results/figures/00_eda_item_popularity.png`

위의 Triple-Sparsity는 유저 측과 행렬 측 희소성에 초점을 맞추지만, **아이템 측 인기도 편중**이 이를 더욱 증폭시킨다.

**Pareto 분포:** 전체 아이템의 **20.7%가 80%의 거래를 차지**하며, Gini 계수 = **0.7586**으로 극심한 편중을 보인다.

**Top-K 집중도:**

| Top-K | 거래 비율 |
|-------|-----------|
| Top-100 | 4.6% |
| Top-1,000 | 18.2% |
| Top-10,000 | 59.6% |

**아이템 세그먼트:**

| 세그먼트 | 기준 | 아이템 비율 | CF 시그널 상태 |
|----------|------|-------------|---------------|
| Head | 100건+ 구매 | 42.7% | 충분한 CF 시그널 |
| Mid-tail | 10-99건 구매 | 37.0% | 부분적 CF 시그널 |
| Long-tail | 2-9건 구매 | 16.0% | 희박한 CF 시그널 |
| Extreme tail | 1건 구매 | 4.3% | CF 시그널 사실상 부재 |

**57.3%의 tail 아이템**(100건 미만)은 CF 시그널이 불충분하여 인기도 기반 추천에서 체계적으로 소외된다. 이 아이템들에 대해서는 상호작용 데이터에 의존하지 않는 콘텐츠 기반 속성 벡터(L1+L2+L3 → BGE-base 768-dim)가 유일한 밀집 표현이다.

**Triple-Sparsity 증폭 메커니즘:**

```
유저 측 희소성 (32.1% Light) ─┐
                               ├→ 행렬 측 희소성 (99.98% sparse)
아이템 측 인기도 편중 (57.3% tail) ─┘       │
                                        ↓
                               시그널 품질 저하 (노이즈 임베딩)
                                        ↓
                               CF 모델 실패 (Popularity > BPR-MF)
```

유저 측과 아이템 측의 희소성이 결합하면, 행렬의 대부분 셀이 "양쪽 모두 희소한" 영역에 해당한다. 32.1% Light 유저 × 57.3% tail 아이템 = 전체 셀의 ~18.4%가 양측 모두 CF 시그널이 극빈한 영역이며, 이 영역에서의 추천은 콘텐츠 기반 속성 벡터 없이는 사실상 불가능하다. 이것이 105K 전수 아이템에 대한 3-Layer 속성 추출의 직접적 근거이다.

---

## 2. CF 실패 이후의 대안: Content-Based 접근의 계층 구조

CF가 실패하면 자연스러운 대안은 콘텐츠 기반(Content-Based) 접근이다. 그러나 "콘텐츠 기반"에도 여러 수준이 존재하며, LLM 속성 추출이 유일한 해법은 아니다. 본 섹션에서는 기존 메타데이터만으로도 가능한 Content-Based 대안들을 정직하게 나열하고, 본 프로젝트가 검증해야 할 **증분 가치**를 명확히 한다.

### 2.1 기존 메타데이터 Content-Based (LLM 불필요)

H&M articles.csv에는 이미 풍부한 메타데이터가 존재한다:

- **product_type_name** (253종): Vest top, Trousers, Sweater 등
- **colour_group_name** (50종): Black, White, Blue 등
- **section_name, index_name**: Womens Everyday Basics, Ladieswear 등
- **graphical_appearance_name**: Solid, Stripe, Print 등
- **perceived_colour_value_name / perceived_colour_master_name**: 명도/채도 기반 색상 분류
- **detail_desc**: 자연어 상품 설명 (near-complete coverage, 1-2문장)

이것만으로도 TF-IDF, item-item cosine similarity 등 전통적 Content-Based 추천이 가능하다. Cold-start 유저에게도 유저 상호작용 없이 작동한다.

### 2.2 Content-Enhanced 백본 모델 (LLM 불필요)

DeepFM, DCNv2, DIN은 **순수 CF 모델이 아니다**. 기존 메타데이터를 피처로 직접 입력받는 Content-Enhanced CF이다:

| 피처 | 예시 | 차원 |
|------|------|------|
| product_type embedding | Vest top, Trousers (253종) | ~16-dim |
| colour embedding | Black, White (50종) | ~8-dim |
| section/index embedding | Womens Everyday Basics | ~8-dim |
| age_bucket embedding | 유저 연령대 | ~4-dim |
| price scalar | 정규화 가격 | 1-dim |

ID 임베딩 + 메타데이터 피처 = Content-Enhanced CF → 순수 CF(UserKNN, BPR-MF)보다 Cold-start에 강하다. **LLM 없이도** 신규 아이템에 대해 메타데이터 피처만으로 스코어링이 가능하다.

### 2.3 비교 수준: 본 프로젝트가 검증해야 할 증분 가치

| Level | 모델 구성 | Content 수준 | LLM 필요 |
|-------|----------|-------------|---------|
| 0 | UserKNN, BPR-MF | 없음 (ID only) | X |
| 1 | DeepFM + 기존 메타데이터 피처 | 기존 메타 | X |
| 2 | DeepFM + detail_desc 인코딩 | + 텍스트 | X |
| 3 | DeepFM + KAR(L1 구조화) | + LLM 구조화 | O |
| 4 | DeepFM + KAR(L1+L2+L3) | + L2/L3 추가 | O ← 본 프로젝트 |

**핵심 연구 질문:** "CF가 안 되니까 LLM이 필요하다"가 아니라, **"Content-Based가 필요한 상황에서, 기존 메타데이터 대비 LLM 추출 L2/L3가 얼마나 추가 가치를 주는가?"** Level 1→4의 증분 가치를 정량화하는 것이 핵심이지, Level 0→4가 아니다.

---

## 3. 속성 벡터의 Cold-Start 해결 메커니즘

### 3.1 속성 벡터의 CF 독립성

본 프로젝트는 105K 전체 아이템에 대해 LLM/VLM으로 3-Layer 속성(L1 제품 + L2 체감 + L3 이론 기반)을 추출하고, BGE-base-en-v1.5 텍스트 인코더로 **768차원 밀집 벡터**를 생성한다. 이 벡터는 유저-아이템 상호작용과 독립적으로 존재하므로, 행렬 희소도에 영향받지 않는다.

- 아이템 벡터: 105K x 768-dim — 전체 카탈로그 커버
- 유저 벡터: 구매 이력의 속성 집계 → 768-dim — 최소 1건 구매만으로 생성 가능

### 3.2 최소 유저 시그널로 작동

- **1건 구매 유저**: 해당 아이템의 속성 벡터를 기반으로 전체 카탈로그를 스코어링하여 유사 아이템 추천
- **0건 구매 유저(신규)**: 인구통계(age, club_status) 기반으로 가장 유사한 기존 세그먼트의 평균 속성 벡터를 프록시로 사용
- CF 기반 추천은 최소 5-10건 이상의 상호작용이 필요한 반면, 속성 기반 추천은 1건으로 충분하다.

### 3.3 전체 카탈로그 스코어링

105K 아이템 규모에서는 JAX `vmap` + JIT로 **전체 카탈로그를 직접 스코어링**할 수 있다 (~15ms). 대규모 카탈로그(1M+)에서 필요한 후보 생성(Candidate Generation) + 랭킹의 2-Stage 파이프라인은 이 규모에서 불필요하며, 오히려 recall loss라는 교란 변수를 도입한다.

```python
# 전체 카탈로그 직접 스코어링
scores = jax.vmap(model.score, in_axes=(None, 0))(user_feat, all_item_feats)  # (105K,)
top_12 = jax.lax.top_k(scores, 12)  # ~15ms on GPU
```

이 방식은 모든 Level(0~4)에서 동일 조건으로 공정 비교가 가능하다는 장점이 있다.

### 3.4 구체적 예시

유저가 cotton T-shirt 1벌만 구매한 경우:

```
구매 아이템 속성 (Factual 텍스트 → BGE-base 인코딩):
  L1: Category=T-shirt, Material=Cotton, Fit=Regular, Color=White
  L2: Style mood=Casual/Minimalist, Occasion=Everyday, Versatility=5/5
  L3: Silhouette=I-line, Coordination role=Basic, Style lineage=Scandinavian minimal

→ 유저 속성 벡터 생성 (구매 아이템 속성 집계)
→ KAR Expert + Gating → Augmented Vector
→ 전체 105K 아이템 스코어링 (~15ms)
→ Top-12: Casual I-line Basic 아이템 (T-shirt, Tank top, Simple pants 등)
```

CF 모델이 이 유저에게 빈 추천을 반환하거나 인기도 기반 폴백에 의존하는 것과 달리, 속성 벡터 기반 스코어링은 해당 유저의 실제 구매 취향에 기반한 개인화된 추천을 즉시 제공한다. 단, 기존 메타데이터(product_type, colour) 피처만으로도 Content-Enhanced CF가 가능하다는 점에서, **핵심 차별점은 L2/L3 속성이 제공하는 시맨틱 풍부함**이다.

---

## 4. Reasoning Expert

### 4.1 반복 구매 예측의 한계

EDA에서 확인된 핵심 수치: **87%의 유저-아이템 쌍이 단일 구매**이다.

이는 "이 유저가 이 아이템을 다시 구매할 것인가?"라는 전통적 추천 질문이 H&M 데이터에서 구조적으로 한계가 있음을 의미한다. 87% 쌍의 정답이 "No"이므로, majority class를 예측하는 것만으로 87% 정확도에 도달한다. 실질적으로 유의미한 추천은 "재구매"가 아닌 "새로운 발견"에 초점을 맞춰야 한다.

### 4.2 발견 지향 추천 (Discovery-Oriented Recommendation)

질문의 전환:

```
전통적: "이 유저가 이전에 구매한 아이템과 유사한 아이템은 무엇인가?"
본 프로젝트: "이 유저의 잠재 선호(스타일, 무드, 착용 상황)에 부합하지만 아직 발견하지 못한 아이템은 무엇인가?"
```

KAR의 **Reasoning Expert**가 이 질문에 답한다. LLM의 Factorization Prompting으로 유저의 구매 패턴에서 잠재 선호를 추론하고, 이를 밀집 벡터로 변환하여 추천 모델에 주입한다.

### 4.3 L2+L3 시맨틱 브릿지

Reasoning Expert의 핵심은 **L2(체감 속성)와 L3(이론 기반 속성)**가 희소한 구매 이력과 풍부한 아이템 설명 사이의 시맨틱 브릿지 역할을 하는 것이다:

```
희소 이력 (1-4건 구매)
    ↓ L2+L3 속성 추론
"이 유저는 Casual-Minimalist 무드 (L2)의
 I-line 실루엣 (L3)을 선호하는 패턴을 보인다"
    ↓ Reasoning Expert
유저의 잠재 선호 벡터 (d_rec 차원)
    ↓ Gating + Fusion
추천 백본에 주입
```

- L1(제품 속성)만으로는 "cotton T-shirt를 샀으니 다른 cotton T-shirt를 추천"하는 수준에 그친다.
- L2를 추가하면 "Casual Minimalist 무드의 다른 카테고리 아이템"으로 확장된다.
- L3를 추가하면 "I-line 실루엣 + Neutral 색조 + Basic 코디 역할"이라는 무의식적 선호까지 포착하여, 유저가 인지하지 못한 교차 카테고리 아이템(예: 같은 I-line의 knit vest)을 발견하게 한다.

---

## 5. 아키텍처 매핑

### 5.1 Triple-Sparsity → 컴포넌트 대응

| Triple-Sparsity 차원 | 해결 컴포넌트 | 메커니즘 |
|----------------------|--------------|----------|
| 유저 측 희소성 (32.1% Light) | 속성 벡터 기반 스코어링 | 1건 구매 → 속성 벡터 쿼리, 0건 → demographic 프록시 |
| 행렬 측 희소성 (99.98%) | 3-Layer 속성 벡터 (Factual Expert) | CF 시그널 독립, 105K 아이템 전수 커버 |
| 시그널 품질 저하 (CF < Popularity) | Reasoning Expert + Gating | 잠재 선호 추론, Factual/Reasoning 동적 결합 |

**백본 모델의 역할 명확화:** DeepFM, DCNv2, DIN은 순수 CF 모델이 아니라 Content-Enhanced CF이다. 기존 메타데이터 피처를 직접 입력받아 ID-only CF보다 Cold-start에 강하다. KAR의 역할은 이 백본 모델에 LLM 속성 벡터를 **추가 주입**하여 증분 가치를 제공하는 것이다. 105K 아이템 규모에서는 전체 카탈로그 직접 스코어링(~15ms)이 가능하므로, 별도의 후보 생성 단계 없이 1-Stage로 추천을 수행한다.

### 5.2 비교 수준별 아키텍처

| Level | 모델 구성 | 증강 |
|-------|----------|------|
| 0 | UserKNN, BPR-MF | 없음 (ID only) |
| 1 | DeepFM + 기존 메타 피처 | 메타데이터 임베딩 |
| 2 | DeepFM + detail_desc | + BGE 텍스트 인코딩 |
| 3 | DeepFM + KAR(L1) | + Factual Expert (L1) |
| 4 | DeepFM + KAR(L1+L2+L3) | + Factual Expert (L1+L2+L3) + Reasoning Expert |

모든 Level에서 전체 카탈로그 직접 스코어링으로 공정 비교한다.

### 5.3 Cold-Start 시나리오별 처리 경로

| 유저 상태 | 아이템 상태 | 처리 경로 |
|----------|-----------|----------|
| 신규 (0건) | 기존 | Demographic 프록시 → 인기도 + 세그먼트 평균 속성 벡터 기반 전체 스코어링 |
| 희소 (1-4건) | 기존 | 구매 아이템 속성 벡터 → 전체 카탈로그 스코어링 + Reasoning Expert 잠재 선호 |
| 기존 (5건+) | 기존 | Full pipeline: 속성 벡터 + CF 시그널 통합 → 전체 스코어링 |
| 기존 (5건+) | 신규 | LLM 속성 즉시 추출 → BGE 인코딩 → Expert + Gating → e_aug 생성 |
| 희소 (1-4건) | 신규 | 유저 속성 벡터 + 신규 아이템 속성 벡터 → 속성 공간 유사도 기반 스코어링 |

---

## 5.5 평가 시점의 Cold-Start: Temporal Split 분석

> Figure 참조: `results/figures/00_eda_temporal_split.png`, `results/figures/00_eda_split_overlap_recency.png`

Section 1~5에서 기술한 Triple-Sparsity는 Train 데이터 기준의 정적 분석이다. 그러나 실제 평가 시점(Val/Test)에서는 **신규 유저·아이템의 유입**과 **유저 활동의 시간적 감쇠**로 인해 cold-start 문제가 더욱 심화된다.

### 5.5.1 Split 설계 및 규모

| Split | 기간 | 거래 수 | 유저 수 | 아이템 수 | 비율 |
|-------|------|---------|---------|-----------|------|
| Train | 2018-09 ~ 2020-06 (22개월) | 28,401,361 | 1,298,206 | 95,909 | 89.3% |
| Val | 2020-07 ~ 2020-08 (2개월) | 2,588,694 | 413,408 | 37,033 | 8.2% |
| Test | 2020-09-01 ~ 09-07 (1주) | 798,269 | 189,510 | 26,252 | 2.5% |

엄격한 시간 순서(chronological) 분할로 미래 데이터 누수가 없으며, Test 1주 윈도우는 Kaggle 원 대회의 MAP@12 평가 조건과 동일하다. 7-day MA 기준 Val/Test 기간의 거래량이 유사하여 Val에서 튜닝한 하이퍼파라미터의 Test 전이가 유효하다.

### 5.5.2 유저·아이템 Cold-Start 유입

| 차원 | Val (Train에 없음) | Test (Train에 없음) |
|------|-------------------|---------------------|
| 유저 | 46,537명 (**11.3%**) | 21,761명 (**11.5%**) |
| 아이템 | 5,943개 (**16.0%**) | 7,884개 (**30.0%**) |

**유저 측:** Val과 Test 모두 ~11%의 유저가 Train 이력 없는 완전 cold-start이다. 이 유저들은 CF 시그널이 전무하므로, demographic 프록시(age, club_status 기반 세그먼트 평균 속성 벡터) 또는 인기도 폴백만 가능하다.

**아이템 측:** Test의 신규 아이템 비율(**30.0%**)이 Val(**16.0%**)의 거의 2배이다. 9월은 가을 시즌 시작으로 신상품이 대량 출시되는 시기이며, 이 7,884개 아이템은 Train에서 단 한 건의 거래도 없어 CF 임베딩이 존재하지 않는다. **L1+L2+L3 속성 벡터가 이 아이템들의 유일한 밀집 표현**이다.

이 Val→Test 신규 아이템 비율 급증은 시즌 전환이 Triple-Sparsity를 평가 시점에서 **동적으로 증폭**시킴을 보여준다.

### 5.5.3 유저 Recency 분포

Test 시작 시점(2020-09-01) 기준, Train+Val 기간 동안의 마지막 구매로부터의 경과일(recency):

| 지표 | 값 |
|------|-----|
| Median | 146일 (~5개월) |
| Mean | 228일 (~7.5개월) |
| IQR | [46일, 365일] |
| 30일 내 활동 | **18.5%** |
| 90일 내 활동 | **40.0%** |

**60%의 유저가 3개월 이상 비활동** 상태에서 Test 기간에 구매한다. 이는 두 가지 문제를 야기한다:

1. **CF 시그널 시효성**: Train에서 학습된 유저 임베딩이 수개월 전 행동에 기반하므로, 현재 선호와 괴리될 가능성이 높다 (preference drift)
2. **Recency-aware 추론 필요**: 최근 활동 유저(18.5%)와 장기 비활동 유저(60%+)에 대해 동일한 CF 임베딩을 적용하는 것은 부적절하다

Reasoning Expert는 LLM Factorization Prompting으로 유저의 **잠재 선호**를 추론하므로, 오래된 행동 데이터에서도 시간 불변적인 스타일 선호(L2 무드, L3 실루엣 등)를 포착할 수 있다. 이 시맨틱 수준의 선호 추론은 CF 임베딩의 시효성 문제를 완화하는 구조적 장점이 있다.

### 5.5.4 Triple-Sparsity의 동적 증폭

```
[정적 Triple-Sparsity]                [평가 시점 동적 증폭]
유저 32.1% Light ──────────────────→ + 11% 완전 신규 유저 (CF 시그널 전무)
아이템 57.3% tail ─────────────────→ + 30% 완전 신규 아이템 (Test, 시즌 전환)
행렬 99.98% sparse ────────────────→ + 60% 유저 recency > 90일 (시효성 저하)
```

Train 데이터 기준의 정적 희소성에 더해, 평가 시점에서 신규 유저·아이템 유입과 기존 유저의 활동 감쇠가 중첩되어 cold-start 문제가 가중된다. 이 동적 증폭은 **콘텐츠 기반 속성 벡터의 가치가 Val보다 Test에서 더 크게 발현**될 것임을 시사하며, Phase 6 실험에서 Val vs Test 성능 격차를 Layer Ablation별로 비교하여 검증할 수 있다.

---

## 6. 실험 계획: Cold-Start 검증

### 6.1 비교 수준별 실험 (Level 0~4)

Phase 6(체계적 실험)에서 5-Level 비교를 수행한다:

| Level | 실험 구성 | 목적 |
|-------|----------|------|
| 0 | UserKNN, BPR-MF (이미 완료) | ID-only CF baseline |
| 1 | DeepFM + 기존 메타데이터 피처 | Content-Enhanced CF baseline |
| 2 | DeepFM + detail_desc BGE 인코딩 | 텍스트 content의 기여 |
| 3 | DeepFM + KAR(L1 구조화) | LLM 구조화의 기여 |
| 4 | DeepFM + KAR(L1+L2+L3) | L2/L3 추가의 증분 가치 |

핵심 비교: Level 1 vs Level 4의 격차가 LLM 속성 추출의 정당성을 결정한다.

### 6.2 Cold-Start Deep Dive

- **구매 건수별 성능 곡선**: 1건, 2건, 3건, 5건, 10건, 20건+ 유저 그룹별 MAP@12, HR@12, NDCG@12
- **Gating Weight 분석**: Cold-start 유저에서 `g_reason`(Reasoning Expert 가중치)이 기존 유저 대비 증가하는지 확인
- **후보 생성 전략 비교**: 속성 기반 Only vs CF Only vs 인기도 Only vs 조합 — Cold-start 유저 그룹에서의 성능 차이
- **Layer 기여 분석**: Light 유저에서 L2/L3 속성 추가의 한계 기여(marginal contribution)가 Heavy 유저 대비 큰지 검증

### 6.3 예상 결과

- Cold-start 유저(1-4건)에서 속성 기반 추천이 인기도 기반 대비 **HR@12 50%+ 향상** 목표
- Gating Weight: Cold-start 유저에서 `g_reason` > 0.6 (기존 유저 대비 Reasoning Expert 의존도 증가)
- Layer Ablation: Light 유저에서 Full(L1+L2+L3) vs L1 Only의 성능 격차가 Heavy 유저 대비 크게 나타날 것으로 예상 (L2+L3 시맨틱 브릿지의 가치가 희소 상황에서 더 높음)
- **Level 1 vs Level 4 격차**: 기존 메타데이터 Content-Enhanced CF 대비 LLM 속성 증강의 증분 가치를 정량적으로 입증

---

## 참조

- EDA 노트북: `notebooks/00_eda.ipynb` — Triple-Sparsity 정량 분석 원본
- 프로젝트 설계: `docs/research_design/hm_unified_project_design.md` — Section 0 (연구 동기)
- Baseline 결과: `PLAN.md` — Key Findings > Phase 0 Baseline 성능
- 아키텍처 상세: `docs/research_design/hm_unified_project_design.md` — Section 7 (추천 시스템 아키텍처)
- 실험 설계: `docs/research_design/hm_unified_project_design.md` — Section 8 (실험 설계)
- Figure: `results/figures/` — EDA 시각화 (구매 분포, 세그먼트 분석 등)
