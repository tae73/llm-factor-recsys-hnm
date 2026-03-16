# H&M 데이터셋 기반 다층 속성 추출·분석·추천 통합 프로젝트 설계

> 3-Layer Attribute Taxonomy를 H&M Kaggle 데이터셋(~105K 아이템, ~1.37M 고객, ~31M 거래)에 적용하는 End-to-End 연구 설계. 속성 추출 → 고객/상품 분석·세그멘테이션 → 교차 타겟팅 → 추천 시스템 통합까지 전 과정을 하나의 문서에서 명세한다.

---

## 0. 연구 동기: Triple-Sparsity 문제와 콘텐츠 기반 속성 증강의 필요성

> EDA 분석(`notebooks/00_eda.ipynb`)에서 도출한 H&M 데이터의 구조적 문제를 정리하고, 이를 해결하기 위한 본 프로젝트의 핵심 전략을 개관한다. 상세 분석은 `docs/cold_start_analysis.md` 참조.

### 0.1 H&M 데이터의 Triple-Sparsity 문제

H&M 데이터셋은 세 가지 차원에서 동시에 희소성이 발현되는 **Triple-Sparsity** 문제를 보인다:

**유저 측 희소성:**

| 세그먼트 | 유저 수 | 비율 | 거래 비율 |
|----------|---------|------|-----------|
| Heavy (28건+) | 332,870명 | 24.4% | 73.5% |
| Moderate (5-27건) | 602,387명 | 43.5% | 23.3% |
| Light (1-4건) | 436,723명 | 32.1% | 3.2% |

전체 유저의 32.1%(436,723명)가 1-4건 구매에 불과하며, 이들은 카탈로그의 <0.004% 아이템과만 상호작용한다.

**행렬 측 희소성:** 유저-아이템 상호작용 행렬 희소도 **99.98%**. 1.37M x 105K = ~1,440억 셀 중 ~31M 관측값만 존재하여 MF/GNN의 시그널 전파가 구조적으로 실패한다.

**시그널 품질 저하:** Baseline 실험에서 **Popularity > UserKNN > BPR-MF** — 개인화 수준이 높아질수록 성능이 오히려 하락한다. 희소한 상호작용에서 학습된 유저/아이템 임베딩이 노이즈를 포함하여, 랜덤에 가까운 랭킹을 산출한다.

**아이템 측 인기도 편중:** 이 Triple-Sparsity를 더욱 증폭시키는 요인으로, 아이템 인기도가 극심하게 편중되어 있다. 전체 아이템의 **20.7%가 80%의 거래를 차지**(Gini=0.7586)하며, 아이템 세그먼트별로 CF 시그널 가용성이 크게 다르다:

| 아이템 세그먼트 | 기준 | 비율 | CF 시그널 |
|----------------|------|------|----------|
| Head | 100건+ | 42.7% | 충분 |
| Mid-tail | 10-99건 | 37.0% | 부분적 |
| Long-tail | 2-9건 | 16.0% | 희박 |
| Extreme tail | 1건 | 4.3% | 부재 |

**57.3%의 tail 아이템**은 CF 시그널이 불충분하여, 상호작용에 의존하지 않는 콘텐츠 기반 밀집 벡터(L1+L2+L3 → BGE-base 768-dim)가 필수적이다. 이것이 105K 전수 아이템에 대한 속성 추출의 직접적 근거이다.

세 가지 희소성은 독립적이지 않고 cascading 효과를 만든다: 유저 측 희소 + 아이템 측 편중 → 행렬 sparse → 노이즈 임베딩 → CF 실패. **결론:** 협업 필터링만으로는 H&M 데이터에서 유의미한 개인화가 구조적으로 불가능하며, CF 시그널을 보완하는 콘텐츠 기반 시그널이 필수적이다.

단, 기존 메타데이터만으로도 Content-Based 접근이 가능하다는 점을 간과해서는 안 된다. H&M articles.csv에는 product_type_name(253종), colour_group_name(50종), section_name, detail_desc(자연어 설명) 등 풍부한 메타데이터가 이미 존재한다. 또한 DeepFM, DCNv2, DIN 등 백본 모델은 순수 CF가 아닌 **Content-Enhanced CF**로, 기존 메타데이터 피처를 직접 입력받아 ID-only CF보다 Cold-start에 강하다. 따라서 핵심 연구 질문은 "CF가 안 되니까 LLM이 필요하다"가 아니라, **"Content-Based가 필요한 상황에서, 기존 메타데이터 대비 LLM 추출 L2/L3가 얼마나 추가 가치를 주는가?"**이다.

### 0.2 콘텐츠 기반 접근의 계층 구조와 LLM 속성의 증분 가치

CF 실패 후 콘텐츠 기반 접근에도 여러 수준이 존재한다. 본 프로젝트는 다음 5단계에서 LLM 속성의 증분 가치를 정량화한다:

| Level | 모델 구성 | Content 수준 | LLM 필요 |
|-------|----------|-------------|---------|
| 0 | UserKNN, BPR-MF | 없음 (ID only) | X |
| 1 | DeepFM + 기존 메타데이터 피처 | 기존 메타 | X |
| 2 | DeepFM + detail_desc 인코딩 | + 텍스트 | X |
| 3 | DeepFM + KAR(L1 구조화) | + LLM 구조화 | O |
| 4 | DeepFM + KAR(L1+L2+L3) | + L2/L3 추가 | O ← 본 프로젝트 |

105K 전체 아이템에 대해 LLM/VLM으로 3-Layer 속성(L1+L2+L3)을 추출하고, BGE-base-en-v1.5로 **768차원 밀집 벡터**를 생성한다. 이 벡터는 유저-아이템 상호작용과 독립적이므로, 행렬 희소도에 영향받지 않는다.

- 105K 아이템 전수 커버 (BGE-base 768-dim)
- **최소 1건 구매**로 속성 공간 쿼리 가능 (CF는 5-10건 이상 필요)
- 0건 유저(신규): 인구통계 기반 세그먼트 평균 속성 벡터를 프록시로 사용
- 105K 아이템 규모에서 전체 카탈로그 직접 스코어링 가능 (JAX vmap, ~15ms)

### 0.3 Reasoning Expert: 반복 구매 예측을 넘어서

EDA에서 확인: **87%의 유저-아이템 쌍이 단일 구매**. 이는 재구매 예측이 구조적으로 한계가 있으며, **발견 지향(discovery-oriented) 추천**이 필요함을 의미한다.

KAR의 Reasoning Expert가 LLM Factorization Prompting으로 유저의 잠재 선호를 추론한다. 특히 **L2(체감 속성)와 L3(이론 기반 속성)**가 희소한 구매 이력과 풍부한 아이템 설명 사이의 시맨틱 브릿지 역할을 한다:

- L1만: "cotton T-shirt를 샀으니 다른 cotton T-shirt를 추천" (카테고리 내 유사)
- L2 추가: "Casual Minimalist 무드의 다른 카테고리 아이템"으로 확장
- L3 추가: "I-line 실루엣 + Neutral 색조 + Basic 코디 역할"이라는 무의식적 선호까지 포착 → 교차 카테고리 발견

### 0.4 연구 질문의 구체화

Triple-Sparsity 문제에서 본 프로젝트의 5대 핵심 혁신이 각각 어떻게 대응하는지:

| Triple-Sparsity 차원 | 대응 혁신 | 메커니즘 |
|----------------------|----------|----------|
| 유저 측 희소성 | 속성 벡터 기반 스코어링 | 1건 구매 → 속성 벡터 쿼리, CF 의존성 우회 |
| 행렬 측 희소성 | 3-Layer Attribute Taxonomy | 105K 아이템 전수 768-dim 벡터, CF 시그널 독립 |
| 시그널 품질 저하 | Reasoning Expert + Gating | 잠재 선호 추론, Factual/Reasoning 동적 결합 |
| 아이템 측 인기도 편중 | 3-Layer 속성 전수 추출 | 57.3% tail 아이템 포함 105K 전수 768-dim 벡터 |
| 복합 효과 | KAR Hybrid-Expert Adaptor | Item Factual(L1+L2+L3) + User Reasoning 비대칭 2-Expert 구조 |
| Cold-start | Pre-store + 2-Stage 서빙 | 오프라인 벡터 사전 계산, 온라인 ~15ms 추론 |

**비교 수준별 증분 가치:**

| Level | 구성 | LLM 필요 | 검증 목적 |
|-------|------|---------|----------|
| 0 | UserKNN, BPR-MF (ID only) | X | 순수 CF baseline |
| 1 | DeepFM + 기존 메타데이터 | X | Content-Enhanced CF baseline |
| 2 | DeepFM + detail_desc 인코딩 | X | 텍스트 content 기여 |
| 3 | DeepFM + KAR(L1 구조화) | O | LLM 구조화 기여 |
| 4 | DeepFM + KAR(L1+L2+L3) | O | L2/L3 증분 가치 ← 본 프로젝트 |

핵심 비교: Level 1 vs Level 4의 격차가 LLM 속성 추출의 정당성을 결정한다.

이 연구 동기가 Section 7(추천 시스템 아키텍처)과 Section 8(실험 설계)의 근거를 형성한다. 특히 Section 8.4의 Cold-Start Deep Dive에서 Triple-Sparsity 해결 효과를 비교 수준(Level 0~4)별로 정량적으로 검증한다.

---

## 1. 데이터셋 상세 분석

### 1.1 데이터 구성

H&M Personalized Fashion Recommendations 데이터셋은 2018년 9월부터 2020년 9월까지 약 2년간의 실제 거래 데이터로 구성된다.

**articles.csv (~105,542 아이템)** 는 상품 메타데이터를 담고 있다. 각 아이템은 article_id로 식별되며, product_code와 prod_name으로 상품 자체를 기술한다. 카테고리 계층 구조는 product_type_name(e.g., "Vest top", "Trousers"), product_group_name(e.g., "Garment Upper body"), department_name(e.g., "Jersey Basic"), section_name(e.g., "Womens Everyday Basics"), index_name(e.g., "Ladieswear"), index_group_name, garment_group_name의 다중 계층으로 구성된다. 시각적 속성으로는 graphical_appearance_name(e.g., "Solid", "Stripe")과 colour_group_name(e.g., "Black", "White")이 있으며, 색상은 perceived_colour_value_name("Dark", "Dusty Light")과 perceived_colour_master_name으로 이중 분류된다. detail_desc 필드에는 자연어 상품 설명(e.g., "Fitted strap top in soft jersey with narrow shoulder straps.")이 포함된다.

**customers.csv (~1,371,980 고객)** 는 customer_id, FN(Fashion News 수신 여부), Active(활성 상태), club_member_status(ACTIVE, PRE-CREATE, LEFT CLUB 등), fashion_news_frequency(Regularly, Monthly, None), age, postal_code(익명화)를 포함한다.

**transactions_train.csv (~31,788,324 거래)** 는 t_dat(거래 날짜), customer_id, article_id, price(정규화 가격), sales_channel_id(1: 오프라인, 2: 온라인)를 포함한다.

**images/ 폴더** 에는 각 아이템의 상품 이미지가 포함되어 있으며, 대부분 흰 배경의 단독 상품 사진이다.

### 1.2 데이터 특성 및 도전 과제

**강점:** 이미지 + 텍스트 + 구조화 메타데이터 + 거래 로그의 멀티모달 조합, 105K 아이템의 현실적 비용-품질 트레이드오프 규모, 31M 거래의 통계적 신뢰성, 다중 카테고리 계층.

**도전 과제:** 구매만 존재하는 Implicit Feedback Ambiguity(열람/클릭/장바구니 부재), 2년간의 패션 트렌드·계절성 중첩, 유저당 구매 수 희소성(중앙값 ~10건, 상세 분석은 Section 0 참조), detail_desc의 짧은 길이(1-2문장).

### 1.3 데이터 전처리

**시간 분할:** 2018.09–2020.06(학습), 2020.07–2020.08(검증), 2020.09 첫 주(테스트). 속성 추출은 전체 아이템, 유저 추론은 학습 기간만 사용.

**고객 필터링:** 5건 이상 구매 "활성 유저" + 1-4건 "희소 유저"(Cold-start 실험용) 분리.

**아이템 분기:** detail_desc 결측/이미지 부재 여부에 따라 LLM 입력 모드(텍스트/이미지/멀티모달) 분기.

**가격:** 정규화된 price를 5분위로 변환하여 카탈로그 내 상대적 가격 포지션으로 사용.

---

## 2. 3-Layer 속성 체계: H&M 데이터 매핑

> **설계 원칙: 개념적 분류 vs 구현 구조의 분리**
>
> L1(제품 속성), L2(체감 속성), L3(이론 기반 속성)는 **"무엇을 추출할 것인가"를 체계적으로 조직하는 개념적 분류 체계(Conceptual Taxonomy)**이다. 이 3-Layer 분류는 LLM 프롬프트 설계, 속성 품질 검증, 도메인 간 비교 분석에서 핵심 역할을 한다. 그러나 **추천 모델 구현에서는** 이 세 층위를 별도 Expert로 분리하지 않고, KAR 원 논문(Xi et al., 2023, Fig.3)의 검증된 **비대칭 2-Expert 구조를 그대로 채택**한다: **Item → Factual Knowledge**(아이템의 L1+L2+L3 속성을 하나의 통합 텍스트로 결합), **User → Reasoning Knowledge**(LLM의 유저 선호 추론). Factual Expert는 아이템 측 지식을, Reasoning Expert는 유저 측 지식을 처리한다. 이 구조는 다음과 같은 이점을 갖는다: (1) KAR의 검증된 비대칭 아키텍처를 그대로 활용하여 구현 리스크 최소화, (2) 3-Layer 개념적 풍부함은 추출 단계에서 온전히 보존, (3) Ablation 실험에서 "Item Factual 텍스트에 어떤 Layer의 속성을 포함/제외하는가"로 각 Layer의 기여도를 여전히 정량화 가능.

### 2.0 Product Super-Category 라우팅

H&M 카탈로그는 의류 82%, 악세서리 10.5%, 신발+양말/타이츠 7.3%, Non-fashion 0.12%로 구성된다. 기존 L1 LLM-Enhanced 속성(neckline, sleeve_type, fit, length)과 L3 속성(silhouette, proportion_effect)이 **의류 전용**으로 정의되어 있어 ~18% 비의류 아이템에 적용 불가하다. 이를 해결하기 위해 `garment_group_name` 필드(19개 값)를 3+1 Super-Category로 매핑하고, 카테고리별 특화 속성을 정의한다.

| Super-Category | garment_group_name 매핑 | 아이템 수 | 비율 |
|----------------|------------------------|----------|------|
| Apparel | Garment Upper/Lower/Full body, Underwear, Nightwear, Swimwear | ~86K | 82% |
| Footwear | Shoes, Socks & Tights | ~7.7K | 7.3% |
| Accessories | Accessories (bags, hats, jewelry, scarves 등) | ~11K | 10.5% |
| **Excluded** | Non-fashion (cosmetics, furniture, stationery) | ~130 | 0.12% |

- **라우팅 기준:** `garment_group_name` 필드 — 19개 고유값을 3개 Super-Category로 그룹화
- **Non-fashion 제외 사유:** 패션 도메인 속성(소재, 핏, 실루엣 등) 적용 불가, 0.12%로 추천 성능에 무의미
- **속성 구조:** L1과 L3는 **Shared(공통) + Category-Specific(특화)** 이원 구조, L2는 전 카테고리 Universal

**Downstream 아키텍처 호환성:** Category-Specific 속성이 카테고리별로 다른 attribute name을 사용하더라도, 시맨틱 인코더(BGE-base)가 자연어 텍스트를 고정 d_enc 차원 벡터로 압축하는 시점에서 스키마 이질성이 해소된다. 이후 Expert→Gating→Fusion→Backbone 전 구간에서 차원·아키텍처 변경이 불필요하다. 단일 스키마에서 비의류에 "neckline: N/A" 같은 무의미 패딩을 삽입하는 것 대비, Category-Specific 텍스트는 모든 슬롯이 의미 있는 값을 가지므로 인코더 출력의 시그널-노이즈 비율이 개선된다. User Reasoning 텍스트에서는 LLM이 구매 이력의 Category-Specific 속성을 카테고리 횡단적으로 종합하므로(예: "Apparel I-line + Footwear Streamlined = 리니어 선호"), 교차 카테고리 추천의 시맨틱 브릿지 역할이 강화된다.

### 2.1 L1 제품 속성 (Product Attributes)

#### 2.1.1 기존 메타데이터 직접 활용 (No LLM)

product_type_name(253개 고유값), product_group_name(19개), section_name, index_name, garment_group_name, graphical_appearance_name(패턴), colour_group_name(색상 그룹), perceived_colour_value_name(명도/채도), perceived_colour_master_name(대표 색상), price 5분위, sales_channel_id.

#### 2.1.2 LLM/VLM 보강 L1 속성

detail_desc + 이미지로부터 추출하는 **카테고리당 7개 속성** (3 Shared + 4 Category-Specific).

**L1 Shared (전 카테고리 공통, 3개):**
- `material` — 소재 (Jersey, Cotton, Leather, Metal, Plastic 등)
- `closure` — 잠금 방식 (Buttons, Zipper, Pull-on, Clasp, Buckle, Magnetic 등)
- `design_details` — 디자인 디테일 태그 리스트 (러플, 포켓, 자수, 스티칭 등)

**L1 Category-Specific (카테고리별 4개):**

| Slot | Apparel | Footwear | Accessories |
|------|---------|----------|-------------|
| 4 | `neckline` (Scoop, V-neck, Crew, Mock 등) | `toe_shape` (Round, Pointed, Square, Open, Peep) | `form_factor` (Structured, Soft, Rigid, Flexible, Chain) |
| 5 | `sleeve_type` (Sleeveless, Short, Long 등) | `shaft_height` (Low/Ankle, Mid-calf, Knee-high, Over-knee) | `size_scale` (Mini, Small, Medium, Large, Oversized) |
| 6 | `fit` (Slim, Regular, Loose, Oversized) | `heel_type` (Flat, Low, Mid, High, Platform, Wedge) | `wearing_method` (Handheld, Shoulder, Crossbody, Wrist, Neck, Head, Ear, Finger, Waist) |
| 7 | `length` (Crop, Regular, Long, Maxi) | `sole_type` (Rubber, Leather, Foam, Textile) | `primary_function` (Carrying, Covering, Decorating, Protecting, Fastening) |

### 2.2 L2 체감 속성 (Perceptual Attributes)

기존 데이터 필드에는 전혀 존재하지 않으며, LLM의 세계 지식과 추론 능력으로 추출한다. style_mood(최대 3개: Casual, Formal, Sporty, Bohemian, Minimalist, Romantic, Edgy, Preppy, Streetwear, Vintage, Elegant, Playful, Professional, Cozy, Avant-garde), occasion(최대 3개: Everyday, Work/Office, Date, Party, Outdoor, Beach, Formal event, Lounging, Travel), perceived_quality(1-5), trendiness(Classic/Current/Emerging/Dated), season_fit(Spring, Summer, Fall, Winter, All-season), target_impression(1문장 자유 텍스트), versatility(1-5).

### 2.3 L3 이론 기반 속성 (Theory-grounded Attributes)

패션 디자인·색채 이론에 근거한 전문적 속성이다. **카테고리당 7개 속성** (5 Shared + 2 Category-Specific).

**L3 Shared (전 카테고리 공통, 5개):**
- `color_harmony` — Monochromatic/Analogous/Complementary/Triadic/Neutral
- `tone_season` — Spring/Summer/Autumn/Winter/Neutral (퍼스널 컬러)
- `coordination_role` — Statement/Basic/Accent/Layering
- `visual_weight` — 1-5
- `style_lineage` — 최대 2개 (Scandinavian minimal, French chic, Workwear heritage 등)

**L3 Category-Specific (카테고리별 2개):**

| Slot | Apparel | Footwear | Accessories |
|------|---------|----------|-------------|
| 6 | `silhouette` (A/H/X/Y/I/O-line) | `foot_silhouette` (Streamlined, Chunky, Sleek, Architectural) | `visual_form` (Structured, Organic, Geometric, Linear, Circular) |
| 7 | `proportion_effect` (허리 강조, 다리 연장 등) | `height_effect` (Leg-lengthening, Grounding, Neutral) | `styling_effect` (Focal point, Balance, Elongating, Framing) |

### 2.4 속성 추출 비용 계획

L1/L2/L3 모두 GPT-4.1-nano(멀티모달) + Structured Output으로 통일한다. Per-Item 통합 프롬프트로 아이템당 단일 API 호출에서 L1+L2+L3를 동시 추출한다. product_code 기반 캐싱(~47K 고유 제품, 색상 변형 재활용)으로 API 호출 수를 105K→47K로 절감. 파일럿(500개)은 실시간 API, 전체 배치는 Batch API(50% 할인)로 처리.

**Category-Adaptive 라우팅:** `garment_group_name` 기반으로 Apparel/Footwear/Accessories 3-way 프롬프트 분기를 적용한다. 각 Super-Category별 전용 system prompt를 사용하여 해당 카테고리의 L1 Category-Specific 속성(Apparel: neckline/sleeve_type/fit/length, Footwear: toe_shape/shaft_height/heel_type/sole_type, Accessories: form_factor/size_scale/wearing_method/primary_function)과 L3 Category-Specific 속성(Apparel: silhouette/proportion_effect, Footwear: foot_silhouette/height_effect, Accessories: visual_form/styling_effect)을 추출한다. L2는 전 카테고리 공통 프롬프트. 프롬프트 길이가 카테고리간 유사하므로 비용 영향 없음. 총 예상 비용 **~$10** (Batch API 50% 할인 적용).

---

## 3. 유저 속성 추론

### 3.1 유저 L1 선호 (행동 기반 직접 집계)

거래 데이터에서 SQL/Pandas로 직접 집계한다. 선호 카테고리 분포(product_type_name별 구매 비율), 선호 색상/패턴/소재/핏 분포, 가격 포지션 분포, 채널 선호, 구매 주기성(평균 간격, 월별 빈도), 카테고리 다양성 스코어.

### 3.2 유저 L2 선호 (LLM Factorization Prompting)

유저의 최근 20건 구매 아이템의 L2 속성을 LLM에 제시하고, 다음 요인별로 분리 추론(Factorization)한다: (a) 스타일 무드 선호, (b) 착용 상황 선호, (c) 품질-가격 성향, (d) 트렌드 민감도, (e) 계절 패턴, (f) 전반적 스타일 정체성 요약(1문장).

### 3.3 유저 L3 선호 (이론 기반 무의식적 패턴)

구매 아이템의 L3 속성 분포를 LLM에 제시하여 추론한다. Shared L3(색채 조화, 톤, 코디 역할, 시각적 무게, 스타일 계보)는 통합 분포로, Category-Specific L3는 카테고리별 분리 분포로 제시한다(예: 'Silhouette (Apparel): I-line 60%. Foot silhouette (Footwear): Streamlined 80%.'). LLM이 카테고리 횡단적 형태 선호 패턴(예: I-line + Streamlined = 전반적 슬림/리니어 선호)을 종합 추론한다: (a) 형태 선호 패턴(카테고리 횡단), (b) 퍼스널 컬러 성향, (c) 코디 성향, (d) 스타일 계보 일관성, (e) 시각적 무게 선호.

### 3.4 비용

활성 유저 약 50만 명 × GPT-4.1-nano(L2+L3 통합 프롬프트) × Batch API(50% 할인) ≈ ~$25. 희소 유저(1-4건)는 LLM 없이 아이템 속성 분포의 가중 평균으로 약식 프로파일 생성.

---

## 4. 고객 세그멘테이션

### 4.1 L1 기반: "무엇을 소비하는가"

product_type 분포(50차원) + colour 분포(20차원) + pattern 분포(15차원) + material 분포(12차원) + fit(4차원) + price(5차원) + channel(1차원) + diversity(1차원) ≈ 108차원 → PCA/UMAP 20-30차원 축소 → K-Means/GMM 클러스터링.

예상 세그먼트: "Basics 중심 실용형"(Jersey/T-shirt, Solid, Black/White), "Dress/Skirt 여성복 중심형", "아동복 구매형"(Baby/Children 인덱스 지배), "스포츠/아웃도어형", "다카테고리 탐색형".

### 4.2 L2 기반: "어떤 감성을 추구하는가"

style_mood 분포(15차원) + occasion 분포(9차원) + quality 평균/분산(2차원) + trendiness 분포(4차원) + season_fit 분포(5차원) + versatility(1차원) ≈ 36차원. Soft Clustering(GMM/Fuzzy C-means) 적용.

예상 세그먼트: "Casual Everyday 실용파", "Formal Professional", "Trendy Social", "Cozy Home", "Sporty Active".

### 4.3 L3 기반: "숨겨진 구조적 선호"

silhouette 분포 + tone_season 분포 + coordination_role 분포 + color_harmony 분포 + style_lineage 분포 + visual_weight 평균.

예상 세그먼트: "Scandinavian Minimalist"(H/I-line, Neutral, Basic/Layering), "Warm Tone Romantic"(A/X-line, Analogous, Autumn/Spring), "Bold Contrast"(Complementary, 높은 visual_weight, Statement), "Classic Proportion"(X-line, 허리 강조, French chic).

### 4.4 다층 통합

계층적(L1→L2→L3 점진 세분화), 동시적(Multi-view Clustering), 추천 모델 Gating 가중치 기반(g_fact/g_reason 비율 분포로 유저 유형 분류) 세 방식을 비교.

Cross-Layer 검증: "같은 L1 Basics 구매자 안에서 L2가 Casual vs Cozy로 나뉘는가?", "같은 L2 Casual 안에서 L3가 Scandinavian Minimalist vs Warm Romantic으로 나뉘는가?"

---

## 5. 상품 세그멘테이션 및 카탈로그 진단

### 5.1 상품 클러스터링

기존 H&M 카테고리 체계와 LLM 추출 속성 기반 클러스터링의 차이를 비교한다. 핵심은 product_type_name이 다르지만 L2/L3 속성이 유사한 "교차 카테고리 유사 아이템" 발견이다.

### 5.2 카탈로그 갭 분석

고객 세그먼트의 선호 분포와 상품의 속성 분포를 대조하여, 수요 대비 공급이 부족한 영역을 식별한다. H&M 특화 분석 질문: Autumn톤 × Formal × X-line 조합의 공급 부족 여부, 남성복의 Statement 아이템 비율, 아동복 L2 다양성.

### 5.3 시즌별 속성 트렌드

분기별 인기 아이템의 L2/L3 속성 분포 변화를 추적한다. 특히 2020년 상반기(COVID) Cozy/Lounging 무드와 Oversized 핏의 급증 여부를 속성 수준에서 검증.

---

## 6. 고객-상품 교차 분석 및 타겟팅

### 6.1 Affinity Matrix

행동 친화도 A_behavioral(s,c) = 세그먼트 s의 유저가 클러스터 c를 구매한 비율. 속성 잠재 친화도 A_latent(s,c) = cosine_similarity(세그먼트 센트로이드, 클러스터 센트로이드). 기회 점수 Opportunity(s,c) = A_latent - A_behavioral. 양수이면서 큰 셀이 미발견 타겟팅 기회.

### 6.2 타겟팅 전략 4종

고친화도 타겟팅(고값 셀 집중 노출), 잠재 기회 타겟팅(속성↑ 행동↓ 조합 proactive 노출), 교차 세그먼트 브릿징(공유 브릿지 클러스터로 세그먼트 확장), 맥락 기반 동적 타겟팅(시간/상황에 따라 L2 가중치 조정).

### 6.3 H&M 특화 시나리오

계절 전환기 프로모션(Autumn톤 신상 × 여름 구매 고객), Cross-selling(Basic 코디 역할 구매자 × 조화로운 Accent 아이템), 아동복→성인복 교차(부모 L2/L3 선호 기반 성인복 추천), 온/오프라인 채널 차별화.

---

## 7. 추천 시스템 아키텍처 설계

이 섹션이 본 프로젝트의 기술적 핵심이다. KAR(Xi et al., 2023)의 Hybrid-Expert Adaptor를 그대로 채택하되, 원 논문의 **비대칭 구조**(Item Factual + User Reasoning 2-Expert)를 충실히 따르면서 3-Layer 개념적 속성 체계를 통해 추출한 풍부한 속성을 Item Factual 텍스트로 통합한다. H&M 데이터의 특성에 맞게 구체적 아키텍처를 설계한다.

### 7.1 전체 구조 개관

추천 시스템은 크게 네 개의 모듈로 구성된다.

**모듈 A — 속성 텍스트 인코더(Attribute Text Encoder):** 아이템의 Factual 텍스트와 유저의 Reasoning 텍스트를 밀집 벡터로 변환한다.

**모듈 B — Hybrid-Expert Adaptor:** Factual Expert(아이템 측)와 Reasoning Expert(유저 측)가 각각의 지식 유형을 추천 공간으로 변환하고, Gating Network가 동적으로 결합한다.

**모듈 C — Embedding Fusion:** Expert Adaptor의 출력(Augmented Vector)을 추천 백본 모델의 원본 ID 임베딩과 결합한다.

**모듈 D — 추천 백본(Recommendation Backbone):** 결합된 임베딩을 입력으로 최종 추천 점수를 출력한다.

추론 시에는 모듈 A와 B의 Expert 출력을 사전 계산(Pre-store)하여 저장하므로, 실시간 추천에서는 LLM 호출이 불필요하다. Gating은 user-item 쌍별로 온라인에서 계산하나, 단순 linear+softmax이므로 latency 영향은 무시할 수 있다. 모듈 C와 D만 온라인으로 실행된다.

```
[오프라인 Pre-store 단계]

  아이템 Factual 텍스트           유저 Reasoning 텍스트
  (L1+L2+L3 속성 통합)           (LLM Factorization Prompting)
        │                            │
        ▼                            ▼
  ┌──────────────┐            ┌──────────────┐
  │ 모듈A: Text  │            │ 모듈A: Text  │
  │   Encoder    │            │   Encoder    │
  │  (Shared)    │            │  (Shared)    │
  └──────┬───────┘            └──────┬───────┘
         │ h_fact^item               │ h_reason^user
         ▼                           ▼
  ┌──────────────┐            ┌──────────────┐
  │ 모듈B:       │            │ 모듈B:       │
  │ Expert_fact  │            │ Expert_reason│
  └──────┬───────┘            └──────┬───────┘
         │ e_fact^item               │ e_reason^user
         ▼                           ▼
  ┌────────────────────────────────────────┐
  │  Attribute Store (Pre-stored)          │
  │  item_id → e_fact^item (d_rec차원)     │
  │  user_id → e_reason^user (d_rec차원)   │
  └────────────────────────────────────────┘

[온라인 추천 단계]

  ┌─────────────────────────────────────────────────┐
  │ 모듈B: Gating (온라인, user-item 쌍별)           │
  │  g = Gating(e_fact^item, e_reason^user)          │
  │  e_aug^item = g_fact · e_fact^item               │
  │  e_aug^user = g_reason · e_reason^user           │
  └────────────────────┬────────────────────────────┘
                       │
  ┌─────────────────────────────────────────────────┐
  │ 모듈C: Embedding Fusion                          │
  │  x'_item = x_item ⊕ e_aug^item                  │
  │  x'_user = x_user ⊕ e_aug^user                  │
  └────────────────────┬────────────────────────────┘
                       │
  ┌─────────────────────────────────────────────────┐
  │ 모듈D: 추천 백본 (DeepFM / SASRec / LightGCN)   │
  │  ŷ = f(x'_user, x'_item)                        │
  └─────────────────────────────────────────────────┘
```

### 7.2 모듈 A: 속성 텍스트 인코더 (Attribute Text Encoder)

#### 7.2.1 입력 구성 — KAR 원 논문의 Item Factual + User Reasoning 비대칭 구조

KAR 원 논문(Xi et al., 2023, Fig.3)의 **비대칭 2종 구조**를 그대로 채택한다: **Item → Factual Knowledge**(아이템의 객관적·기술적 속성), **User → Reasoning Knowledge**(유저 선호의 추론적 분석). Factual Expert는 아이템 측 지식을, Reasoning Expert는 유저 측 지식을 처리한다. L1/L2/L3에서 추출한 모든 아이템 속성은 Item Factual 텍스트에 통합되고, LLM의 유저 선호 추론이 User Reasoning 텍스트를 구성한다.

**아이템 Factual 텍스트 (L1+L2+L3 속성 통합) — Super-Category별 3종 예시:**

**(a) Apparel 예시:**
```
"Category: Vest top. Group: Garment Upper body. Section: Womens Everyday Basics.
Pattern: Solid. Color: Black (Dark). Material: Jersey cotton blend.
Neckline: Scoop. Sleeve: Sleeveless. Fit: Slim. Length: Regular.
Closure: Pull-on. Price position: 2/5.
Style mood: Casual, Minimalist. Occasion: Everyday, Lounging.
Perceived quality: 2/5. Trendiness: Classic. Season: All-season.
Impression: 편안하고 미니멀한 데일리 베이직. Versatility: 5/5.
Silhouette: I-line. Color harmony: Neutral (single dark).
Tone season: Winter (pure black). Proportion: Balanced.
Style lineage: Scandinavian minimal. Coordination role: Basic.
Visual weight: 1/5."
```

**(b) Footwear 예시:**
```
"Category: Sneakers. Group: Shoes. Section: Divided Shoes.
Pattern: Solid. Color: White (Light). Material: Canvas cotton.
Toe shape: Round. Shaft height: Low/Ankle. Heel type: Flat. Sole type: Rubber.
Closure: Lace-up. Price position: 2/5.
Style mood: Casual, Sporty. Occasion: Everyday, Outdoor.
Perceived quality: 3/5. Trendiness: Classic. Season: All-season.
Impression: 캐주얼 데일리 스니커즈. Versatility: 4/5.
Foot silhouette: Streamlined. Color harmony: Monochromatic (white).
Tone season: Spring (bright white). Height effect: Neutral.
Style lineage: Athletic heritage. Coordination role: Basic.
Visual weight: 2/5."
```

**(c) Accessories 예시:**
```
"Category: Bag. Group: Accessories. Section: Womens Small Accessories.
Pattern: Solid. Color: Black (Dark). Material: Faux leather.
Form factor: Structured. Size scale: Medium. Wearing method: Shoulder.
Primary function: Carrying. Closure: Zipper. Price position: 3/5.
Style mood: Minimalist, Professional. Occasion: Work/Office, Everyday.
Perceived quality: 3/5. Trendiness: Classic. Season: All-season.
Impression: 깔끔한 오피스 데일리 숄더백. Versatility: 4/5.
Visual form: Structured. Color harmony: Neutral (single dark).
Tone season: Winter (deep black). Styling effect: Balance.
Style lineage: Scandinavian minimal. Coordination role: Basic.
Visual weight: 3/5."
```

각 예시에서 L1(Category~Price), L2(Style mood~Versatility), L3(Color harmony~Visual weight) 속성이 자연스럽게 통합되어 있다. L1과 L3의 Category-Specific 슬롯(Apparel: Neckline/Sleeve/Fit/Length/Silhouette/Proportion, Footwear: Toe shape/Shaft height/Heel type/Sole type/Foot silhouette/Height effect, Accessories: Form factor/Size scale/Wearing method/Primary function/Visual form/Styling effect)이 Super-Category에 따라 분기된다. KAR에서 "factual knowledge"는 아이템의 객관적·기술적 정보를 의미하며, 본 프로젝트에서는 LLM이 추출한 체감/이론 속성까지 아이템의 "사실적 기술(description)"로 통합한다.

**유저 Reasoning 텍스트 (LLM Factorization Prompting 출력 — L2+L3 통합 추론):**
```
"(a) 스타일 무드: Casual-Minimalist 지배적, 간헐적 Cozy.
(b) 주요 착용 상황: Everyday 위주, 계절 전환기에 Layering 수요 증가.
(c) 품질-가격: 중저가 실용 지향, 소재 품질보다 범용성 우선.
(d) 트렌드: Classic/Current 중심, 유행 추종보다 꾸준한 기본템 선호.
(e) 계절: All-season 기본, 겨울 니트 약간 증가.
(f) 형태 선호: 슬림/리니어 형태 일관 — Apparel I-line, Footwear Streamlined 선호.
(g) 색채 성향: Neutral-Winter 톤 지배적, Monochromatic 색채 조화 선호.
(h) 코디 역할: Basic/Layering 중심, Statement 아이템 회피.
(i) 정체성: 모노톤 미니멀리스트, 편안함과 단순함을 최우선시하는 실용적 소비자."
```

이 텍스트는 LLM이 유저의 구매 이력에서 L2(체감) 속성과 L3(이론 기반) 속성의 분포를 종합하여 **유저 선호의 추론적 분석**을 수행한 결과이다. KAR 원 논문에서 "reasoning knowledge"는 사실(구매 이력)을 바탕으로 한 추론적 판단을 의미한다. L3 항목(f~h)에서는 카테고리별로 다른 Category-Specific 속성(Apparel silhouette, Footwear foot_silhouette 등)을 LLM이 카테고리 횡단적으로 종합하여 통합적 형태·색채·코디 선호를 추론한다.

#### 7.2.2 인코더 선택

KAR 원 논문과 동일한 세 후보를 비교한다.

**후보 A — Frozen Pre-trained Encoder (기본):** BGE-base-en-v1.5 또는 E5-base를 동결(freeze)하여 사용한다. 학습 파라미터가 없으므로 빠르고 안정적이다. KAR 원 논문에서 효과적이었다.

**후보 B — Fine-tuned Encoder:** Pre-trained Encoder를 추천 태스크와 함께 End-to-End로 파인튜닝한다. 속성 텍스트의 도메인 특화 의미를 더 잘 포착할 수 있으나, 학습 비용이 높고 과적합 위험이 있다.

**후보 C — Lightweight Projection:** 단순 TF-IDF 또는 Word2Vec 평균 벡터에 학습 가능한 Linear Projection을 적용하는 경량 접근이다. 비용이 매우 낮고 빠르지만 의미적 풍부함이 부족할 수 있다.

**실험 계획:** 세 후보를 모두 구현하여 추천 성능을 비교한다. KAR 원 논문에서 Frozen Encoder가 효과적이었으므로 후보 A를 기본으로 설정하되, 후보 B의 한계 이득이 학습 비용을 정당화하는지 검증한다. Item Factual 텍스트와 User Reasoning 텍스트는 동일한 인코더를 공유한다(KAR 원 논문과 동일).

> **Category-Adaptive 속성과 인코더 적합성:** Pre-trained 시맨틱 인코더(BGE-base)는 자연어 수준에서 속성명 이질성을 자연 처리한다 — "neckline: Scoop"과 "toe_shape: Round"는 각각 의미적 맥락에서 인코딩되므로, 카테고리별 속성명이 다르더라도 별도 처리가 불필요하다. 오히려 단일 스키마의 "neckline: N/A" 패딩 대비 모든 슬롯이 의미 있는 값을 갖는 Category-Specific 텍스트가 인코더 출력의 정보 밀도를 높인다. User Reasoning 측에서는 LLM이 카테고리별 속성을 교차 종합하여 자연어 수준에서 일관된 추론 텍스트를 생성하므로, 인코더 입력의 품질이 보장된다.

#### 7.2.3 인코더 출력

KAR 원 논문의 비대칭 구조에 따라, 인코더는 아이템 Factual과 유저 Reasoning 각각에 대해 고정 차원 밀집 벡터를 출력한다.

```
아이템 i에 대해:
  h_fact^item = Encoder(text_factual^item) ∈ ℝ^d_enc    (e.g., d_enc = 384)

유저 u에 대해:
  h_reason^user = Encoder(text_reasoning^user) ∈ ℝ^d_enc
```

### 7.3 모듈 B: Hybrid-Expert Adaptor (KAR 원 논문 구조)

KAR 원 논문의 핵심 구성요소인 Hybrid-Expert Adaptor를 그대로 채택한다. Factual Expert와 Reasoning Expert 2개가 각각의 지식 유형을 추천 공간으로 변환하고, Gating Network가 이를 동적으로 결합한다.

#### 7.3.1 Factual / Reasoning Expert Network

각 Expert는 해당 유형의 인코더 출력(d_enc 차원)을 추천 공간(d_rec 차원)으로 변환하는 MLP이다. d_rec는 추천 백본 모델의 임베딩 차원(예: DeepFM의 피처 임베딩 차원)에 맞춘다. KAR 원 논문의 비대칭 구조에 따라, Expert_fact는 **아이템** Factual 인코더 출력을, Expert_reason은 **유저** Reasoning 인코더 출력을 처리한다.

```
Expert_fact (Factual Expert — 아이템 측):
  e_fact^item = ReLU(W_f² · ReLU(W_f¹ · h_fact^item + b_f¹) + b_f²)
  W_f¹ ∈ ℝ^(d_hidden × d_enc), W_f² ∈ ℝ^(d_rec × d_hidden)

Expert_reason (Reasoning Expert — 유저 측):
  e_reason^user = ReLU(W_r² · ReLU(W_r¹ · h_reason^user + b_r¹) + b_r²)
  W_r¹ ∈ ℝ^(d_hidden × d_enc), W_r² ∈ ℝ^(d_rec × d_hidden)
```

여기서 d_hidden은 은닉층 차원(하이퍼파라미터, 예: 256)이며, d_rec는 추천 백본의 임베딩 차원(예: 64 또는 128)이다. 두 Expert는 독립된 파라미터를 갖되 동일한 아키텍처를 공유한다.

**Expert 깊이 변형:** 2-Layer MLP(기본), 3-Layer MLP(깊은 변환), Residual MLP(스킵 연결 추가)를 실험한다.

#### 7.3.2 Gating Network

두 Expert의 출력을 동적으로 결합하는 Gating Network를 설계한다. KAR 원 논문의 방식을 기본으로 하되, 변형을 탐색한다.

**Variant G1 — Input-independent Gating (Baseline):**

학습 가능한 고정 가중치로, 모든 유저-아이템 쌍에 동일한 게이팅을 적용한다.

```
g = Softmax(w_g)
w_g ∈ ℝ² (학습 파라미터)
e_aug^item = g_fact · e_fact^item
e_aug^user = g_reason · e_reason^user
```

이 변형은 가장 단순하며, 학습된 g_fact, g_reason 값 자체가 "전체 데이터셋에서 사실적 지식과 추론적 지식 중 어느 쪽이 더 중요한가?"의 글로벌 해석을 제공한다.

**Variant G2 — Expert-conditioned Gating (KAR 원 방식, 기본):**

아이템의 Factual Expert 출력과 유저의 Reasoning Expert 출력을 concat하여 게이팅 입력으로 사용한다. user-item 쌍별로 계산된다.

```
g = Softmax(W_g · [e_fact^item; e_reason^user] + b_g)
W_g ∈ ℝ^(2 × 2·d_rec)
e_aug^item = g_fact · e_fact^item
e_aug^user = g_reason · e_reason^user
```

이 변형은 Expert 출력의 상대적 "확신도"에 따라 게이팅이 적응적으로 변한다. KAR 원 논문에서 검증된 방식이다.

**Variant G3 — Context-conditioned Gating:**

유저 컨텍스트(연령, 멤버십, 시간대 등)에 따라 게이팅이 달라지는 변형이다.

```
context = [age_bucket, club_member_status, season, channel]
context_emb = Embed(context) ∈ ℝ^d_ctx
g = Softmax(W_g · [e_fact^item; e_reason^user; context_emb] + b_g)
```

이 변형은 "어떤 유저 상황에서 사실적 지식과 추론적 지식 중 어느 쪽이 더 중요한가?"의 조건부 해석을 가능하게 한다.

**Variant G4 — User-Item Cross Gating:**

유저의 Expert 출력과 아이템의 Expert 출력을 결합하여 게이팅을 결정한다.

```
cross = [e_fact^item ⊙ e_reason^user]
g = Softmax(W_g · cross + b_g)
```

여기서 ⊙은 원소별 곱이다. 이 변형은 "이 user-item 쌍에서 아이템의 사실적 속성과 유저의 추론적 선호 간 교차 시그널"을 포착한다.

**실험 계획:** G1 → G2 → G3 → G4 순으로 복잡도를 높이며 비교한다. G2를 기본(default)으로 설정한다(KAR 원 논문과 동일).

#### 7.3.3 Pre-store 및 온라인 Gating 출력

KAR 비대칭 구조에서 Pre-store 대상은 **Gating 이전의 Expert 출력**이며, Gating은 온라인에서 user-item 쌍별로 계산한다.

```
Pre-store (오프라인):
  item_id → e_fact^item = Expert_fact(Encoder(text_factual^item)) ∈ ℝ^d_rec
  user_id → e_reason^user = Expert_reason(Encoder(text_reasoning^user)) ∈ ℝ^d_rec

Online (user-item 쌍별):
  g = Gating(e_fact^item, e_reason^user)    ← G2: 쌍별 계산, 매우 경량 (linear+softmax)
  e_aug^item = g_fact · e_fact^item ∈ ℝ^d_rec
  e_aug^user = g_reason · e_reason^user ∈ ℝ^d_rec
```

전체 105K 아이템과 50만 유저의 Expert 출력을 사전 계산하여 Attribute Store에 저장한다. Gating은 온라인에서 쌍별로 계산하나, 단순 linear+softmax이므로 latency 영향은 무시할 수 있다(전체 카탈로그 105K 쌍에 대해서도 ~1ms 미만).

> **L1/L2/L3 개념적 Layer의 기여도 분석 방법:** 3-Layer를 구현 수준에서 분리하지 않더라도, Item Factual 텍스트의 **구성 요소를 변경하는 Ablation**으로 각 Layer의 기여를 정량화할 수 있다. 예를 들어 "Factual 텍스트에서 L3 속성(Silhouette~Visual weight)을 제거"하면 L3의 한계 기여를 측정할 수 있다. 이 접근은 구현 복잡성 없이 개념적 Layer의 실험적 분리를 달성한다(섹션 8.2.1 상세).

### 7.4 모듈 C: Embedding Fusion

Pre-store된 Augmented Vector를 추천 백본의 원본 ID 임베딩과 결합하는 방법을 설계한다.

#### 7.4.1 원본 ID 임베딩

추천 백본 모델은 유저 ID와 아이템 ID를 학습 가능한 임베딩 테이블에서 룩업한다.

```
x_item = ItemEmbedding(item_id) ∈ ℝ^d_rec
x_user = UserEmbedding(user_id) ∈ ℝ^d_rec
```

이 ID 임베딩은 collaborative filtering 시그널을 포착하는 전통적 방식이다. 속성 기반 Augmented Vector는 이와 독립적인 content-based 시그널을 추가한다.

#### 7.4.2 Fusion 전략

**Strategy F1 — Concatenation:**

```
x'_item = [x_item; e_aug^item] ∈ ℝ^(2·d_rec)
x'_user = [x_user; e_aug^user] ∈ ℝ^(2·d_rec)
```

가장 단순하며, 백본 모델이 두 시그널의 결합 방식을 자유롭게 학습한다. 단점은 입력 차원이 2배가 되어 백본 모델의 아키텍처 조정이 필요하다는 것이다.

**Strategy F2 — Addition:**

```
x'_item = x_item + α · e_aug^item ∈ ℝ^d_rec    (e_aug^item은 Item Factual 기반)
x'_user = x_user + α · e_aug^user ∈ ℝ^d_rec    (e_aug^user는 User Reasoning 기반)
```

여기서 α는 학습 가능한 스칼라 가중치(초기값 0.1)이다. 차원이 변하지 않으므로 백본 모델 수정이 불필요하며, α가 augmentation 강도를 제어한다. 비대칭 구조에서 e_aug^item은 아이템의 객관적 속성(L1+L2+L3)에서, e_aug^user는 유저의 추론적 선호에서 각각 유래하므로, 두 소스가 상보적인 시그널을 제공한다.

**Strategy F3 — Gated Addition:**

```
gate = σ(W_f · [x_item; e_aug^item] + b_f)
x'_item = x_item + gate ⊙ e_aug^item
```

원소별 게이팅으로, ID 임베딩의 각 차원이 속성 시그널을 선택적으로 수용한다.

**Strategy F4 — Cross-Attention Fusion:**

```
Q = W_Q · x_item,  K = W_K · e_aug^item,  V = W_V · e_aug^item
attention = Softmax(Q · K^T / √d_k) · V
x'_item = x_item + attention
```

가장 표현력이 높지만 계산 비용도 높다. 소규모 d_rec(64-128)에서는 부담이 크지 않다.

**실험 계획:** F2(Addition)를 기본으로 설정하고, F1/F3/F4를 대비 실험으로 수행한다. KAR 원 논문에서 Addition이 효과적이었다.

#### 7.4.3 H&M 추가 피처 통합

H&M 데이터에는 ID 임베딩과 속성 벡터 외에도 활용할 수 있는 피처가 있다.

**구조화 피처:** 유저의 age(5세 단위 버킷), club_member_status, fashion_news_frequency를 임베딩하여 유저 피처 벡터로, 아이템의 기존 카테고리 필드(product_type_no, section_no, index_code, colour_group_code 등)를 임베딩하여 아이템 피처 벡터로 구성한다.

**행동 집계 피처:** DuckDB에서 거래 로그를 집계하여 생성하는 파생 피처: 유저별(구매 빈도, 평균 가격, 카테고리 엔트로피, 최근 구매 경과일, 계절 분포), 아이템별(총 판매량, 최근 7/30/90일 판매 추이, 재구매율, 평균 구매자 연령).

```
x'_item = Fusion(x_item, e_aug^item, feat_item)
x'_user = Fusion(x_user, e_aug^user, feat_user)
```

### 7.5 모듈 D: 추천 백본 모델

속성 증강의 효과를 다양한 추천 백본에서 검증하여 model-agnostic 성능 향상을 입증한다.

#### 7.5.1 DeepFM (CTR 예측)

H&M의 풍부한 카테고리 피처를 활용하기에 적합한 모델이다. 피처 상호작용을 FM(Factorization Machine) 컴포넌트와 DNN 컴포넌트가 공동으로 학습한다.

```
입력 피처:
  - user_id embedding (d_rec)
  - item_id embedding (d_rec)
  - e_aug^user (d_rec)                   ← 속성 증강 (L1+L2+L3 통합)
  - e_aug^item (d_rec)                   ← 속성 증강 (L1+L2+L3 통합)
  - user_age_bucket embedding (d_feat)
  - user_club_status embedding (d_feat)
  - item_product_type embedding (d_feat)
  - item_index embedding (d_feat)
  - item_colour embedding (d_feat)
  - item_pattern embedding (d_feat)
  - price (scalar)
  - sales_channel (embedding)

DeepFM 아키텍처:
  FM Component: 모든 피처 임베딩 쌍의 내적 합산
  DNN Component: 피처 임베딩 concat → FC(512) → FC(256) → FC(128) → FC(1)
  Output: ŷ = σ(FM_output + DNN_output)
```

**H&M 적용 특이사항:** 거래 데이터가 구매만 포함하므로, BPR(Bayesian Personalized Ranking) 스타일의 네거티브 샘플링이 필요하다. 각 양성 샘플(구매)에 대해 4-5개의 음성 샘플(비구매)을 시간 범위 내에서 무작위로 샘플링한다. 음성 샘플링 시, 인기도 기반 가중 샘플링(인기 아이템을 더 높은 확률로 음성으로 사용)을 적용하면 학습 효과가 높아진다.

#### 7.5.2 SASRec (순차 추천)

H&M의 시간순 거래 데이터를 활용하여, 유저의 구매 시퀀스에서 다음 구매 아이템을 예측한다. Self-Attention 기반으로 시퀀스 내 장기/단기 의존성을 모두 포착한다.

```
입력 시퀀스 (유저 u의 최근 N건 구매):
  S_u = [item_1, item_2, ..., item_N]

각 아이템의 임베딩:
  x'_i = Fusion(ItemEmb(item_i), e_aug^item_i, feat_item_i)

시퀀스 인코딩:
  E = [x'_1 + PE_1, x'_2 + PE_2, ..., x'_N + PE_N]    (PE = Position Encoding)

Self-Attention Blocks (L layers):
  E' = MultiHead_SelfAttention(E)
  E'' = FFN(E')

예측 (마지막 타임스텝의 출력으로 다음 아이템 예측):
  ŷ_j = E''_N · x'_j^T    (후보 아이템 j와의 내적)
```

**속성 증강 통합 방식:** SASRec에서는 시퀀스의 각 아이템 임베딩에 Augmented Vector(e_aug^item, Item Factual 기반)가 이미 결합되어 있으므로, Self-Attention이 속성 시그널도 함께 학습한다. Augmented Vector는 L1+L2+L3 속성이 Factual Expert를 거쳐 하나의 벡터로 압축된 것이므로, 시퀀스 내에서 감성적/구조적 유사도에 기반한 장기 패턴을 포착할 수 있다. 추가로, 유저의 Augmented Vector(e_aug^user, User Reasoning 기반)를 시퀀스 인코딩 후의 출력에 결합하는 방식도 실험한다. Category-Adaptive 속성에서는 시퀀스 내 카테고리 전환(예: Apparel → Footwear → Accessories)이 e_aug^item의 시맨틱 변화로 자연스럽게 반영되어, Self-Attention이 카테고리 전환 패턴을 시맨틱 수준에서 포착할 수 있다.

```
변형 1: 아이템 수준만 Fusion
  x'_i = ItemEmb(item_i) + e_aug^item_i
  최종 유저 표현 = E''_N

변형 2: 아이템 + 유저 Fusion
  x'_i = ItemEmb(item_i) + e_aug^item_i
  최종 유저 표현 = E''_N + e_aug^user
```

**H&M 적용 특이사항:** 시퀀스 최대 길이 N은 50으로 설정한다(대부분의 유저가 2년간 50건 이내 구매). 시간 간격이 불규칙하므로, 위치 인코딩(PE) 대신 시간 간격 인코딩(Time-aware PE)을 사용하는 변형도 실험한다.

```
PE_i = TimeEmb(t_i - t_{i-1})
// 구매 간 경과 일수를 임베딩하여 위치 인코딩으로 사용
```

#### 7.5.3 LightGCN (그래프 기반)

유저-아이템 이분 그래프(bipartite graph)에서 다중 홉 이웃 정보를 집계하여 임베딩을 학습하는 모델이다. CF 시그널이 그래프 구조를 통해 전파된다.

```
그래프 구성:
  노드 = {유저} ∪ {아이템}
  엣지 = {(user_u, item_i) : u가 i를 구매}

임베딩 초기화:
  e_u^(0) = UserEmb(u) + e_aug^user    ← 속성 증강
  e_i^(0) = ItemEmb(i) + e_aug^item    ← 속성 증강

Layer-k 업데이트:
  e_u^(k) = Σ_{i∈N(u)} (1/√|N(u)|·√|N(i)|) · e_i^(k-1)
  e_i^(k) = Σ_{u∈N(i)} (1/√|N(i)|·√|N(u)|) · e_u^(k-1)

최종 임베딩 (K layer 평균):
  e_u = (1/(K+1)) · Σ_{k=0}^{K} e_u^(k)
  e_i = (1/(K+1)) · Σ_{k=0}^{K} e_i^(k)

예측:
  ŷ = e_u^T · e_i
```

**속성 증강 통합 방식:** 초기 임베딩에 Augmented Vector를 Addition하여, 그래프 전파 과정에서 속성 시그널이 이웃 노드로 확산되도록 한다. 이는 "비슷한 속성의 아이템을 구매한 유저끼리 더 유사해지는" 효과를 만든다. Category-Adaptive 속성에서는 다카테고리 유저의 이질적 e_aug^item이 그래프 전파를 통해 블렌딩되어, 카테고리 횡단적 속성 일관성을 가진 유저끼리 그래프 공간에서 근접하게 된다(예: "I-line Apparel + Structured Accessories = 미니멀 일관성" 유저 클러스터 형성).

**H&M 적용 특이사항:** 31M 거래로 인해 그래프가 매우 크므로, Mini-batch GCN 학습(GraphSAINT 또는 RandomWalk 샘플링)이 필요하다. K=3 layers, 임베딩 차원 d_rec=64를 기본으로 한다.

#### 7.5.4 DCNv2 (Cross Network)

피처 간의 명시적 교차(cross) 상호작용을 학습하는 모델이다. DeepFM보다 고차원 피처 상호작용을 효율적으로 포착한다.

```
입력: x₀ = [x_user; x_item; e_aug^user; e_aug^item; feat_user; feat_item]

Cross Layer l:
  x_{l+1} = x₀ · (W_l · x_l + b_l) + x_l

Deep Layer (병렬):
  h_{l+1} = ReLU(W_d_l · h_l + b_d_l)

출력: ŷ = σ(W_out · [x_L; h_L])
```

**가치:** Augmented Vector 내에 L1/L2/L3 속성이 통합되어 있으므로, Cross Layer가 이 풍부한 속성 시그널과 기존 피처 간의 교차 상호작용(예: "소재가 실크이면서 무드가 Elegant인 아이템"이 특정 유저에게 특히 효과적)을 명시적으로 모델링한다.

#### 7.5.5 DIN (Deep Interest Network)

유저의 과거 구매 아이템 중, 후보 아이템과 관련 높은 아이템에 주의(attention)를 기울이는 모델이다.

```
유저의 구매 이력: B_u = [b_1, b_2, ..., b_T]
각 이력 아이템 임베딩: e_b_t = ItemEmb(b_t) + e_aug^item_{b_t}
후보 아이템 임베딩: e_c = ItemEmb(c) + e_aug^item_c

Attention Weight:
  α_t = attention_net(e_b_t, e_c)    // e_b_t와 e_c의 관련도
  // attention_net = MLP([e_b_t; e_c; e_b_t ⊙ e_c; e_b_t - e_c])

유저의 관심 표현:
  v_u = Σ_t α_t · e_b_t

예측:
  ŷ = σ(MLP([v_u; e_c; e_aug^user; feat_user; feat_item]))
```

**속성 증강의 DIN 통합 가치:** Attention 계산에서 Augmented Vector가 포함됨으로써, "과거 구매한 캐시미어 니트(L1)와 동일 소재가 아니더라도, L2 무드(Cozy)가 유사한 후보 아이템에 높은 어텐션이 부여"되는 효과가 나타난다. 즉, 피상적 아이템 유사도가 아닌 감성적/구조적 유사도 기반의 관심 표현이 형성된다. 이는 Augmented Vector가 L1+L2+L3의 통합된 속성 정보를 담고 있기 때문에 가능하다.

### 7.6 손실 함수 설계

#### 7.6.1 추천 손실 (ℒ_rec)

**CTR 모델(DeepFM, DCNv2, DIN):**

```
ℒ_rec = -(1/N) Σ [y · log(ŷ) + (1-y) · log(1-ŷ)]    (Binary Cross-Entropy)
```

여기서 y ∈ {0,1}은 구매 여부, ŷ는 모델 예측이다. 네거티브 샘플링 비율(positive:negative = 1:4)을 사용한다.

**시퀀셜 모델(SASRec):**

```
ℒ_rec = -(1/|S|) Σ_{(u,S_u)} Σ_{t=1}^{|S_u|} [log σ(E''_t · x'_{s_{t+1}}) + log σ(-E''_t · x'_{neg})]
                                                  (BPR Loss per timestep)
```

**그래프 모델(LightGCN):**

```
ℒ_rec = Σ_{(u,i,j)} -log σ(e_u^T · e_i - e_u^T · e_j)    (BPR Loss)
```

여기서 i는 양성 아이템, j는 음성 아이템이다.

#### 7.6.2 속성 정렬 손실 (ℒ_align)

Expert 출력이 추천 임베딩 공간에 잘 정렬되도록 하는 보조 손실이다. KAR 원 논문의 접근을 그대로 적용한다.

```
ℒ_align = ‖e_fact^item − sg(x_item)‖² + ‖e_reason^user − sg(x_user)‖²
```

여기서 sg(·)는 stop-gradient 연산자이다. 비대칭 구조에서 Factual Expert 출력은 아이템 ID 임베딩과, Reasoning Expert 출력은 유저 ID 임베딩과 같은 벡터 공간에 위치하도록 유도한다.

#### 7.6.3 Expert 다양성 손실 (ℒ_div)

두 Expert의 출력이 서로 다른 정보를 포착하도록 유도하는 다양성 정규화이다. Factual Expert(아이템 측)와 Reasoning Expert(유저 측)가 중복된 정보를 인코딩하는 것을 방지한다.

```
ℒ_div = cos(e_fact^item, e_reason^user)
```

이 손실은 사실적 지식(아이템의 객관적 속성 기술)과 추론적 지식(유저 선호 패턴의 추론적 분석)이 벡터 공간에서 상보적인 역할을 유지하도록 한다.

#### 7.6.4 총 손실 함수

```
ℒ = ℒ_rec + α · ℒ_align + β · ℒ_div

α, β는 하이퍼파라미터 (기본값: α=0.1, β=0.01)
```

### 7.7 학습 파이프라인

#### 7.7.1 Multi-stage 학습 (안정적, 기본 설정)

**Stage 1 — 백본 Pre-training (2 epochs):** Augmented Vector 없이 원본 ID 임베딩 + 피처만으로 백본 모델을 사전 학습한다. 이 단계에서 ID 임베딩이 기본적인 CF 패턴을 학습한다.

**Stage 2 — Expert Adaptor 학습 (5 epochs):** 백본 파라미터를 고정(freeze)하고, Expert Network + Gating Network만 학습한다. ℒ_align + ℒ_div로 Expert 출력을 추천 공간에 정렬한다. 이 단계에서 속성 벡터가 추천 공간에 매핑된다.

**Stage 3 — End-to-End Fine-tuning (3 epochs):** 모든 파라미터(백본 + Expert + Gating)를 해제하고, ℒ_rec + α·ℒ_align + β·ℒ_div로 공동 학습한다. 학습률을 Stage 1의 1/10로 줄여 안정적으로 파인튜닝한다. 텍스트 인코더는 Frozen(후보 A 사용 시)이거나 매우 낮은 학습률로 파인튜닝한다.

#### 7.7.2 End-to-End 학습 (빠른, 대비 실험)

Multi-stage 없이 모든 파라미터를 처음부터 동시에 학습한다. 학습 초기에 불안정할 수 있으나, 하이퍼파라미터 튜닝이 간단하다. Multi-stage와 성능을 비교한다.

#### 7.7.3 학습 하이퍼파라미터

```
Optimizer: Adam (lr=1e-3 for Stage 1, 1e-4 for Stage 3)
Batch size: 2048 (DeepFM, DCNv2, DIN), 256 sequences (SASRec)
Embedding dimension d_rec: {32, 64, 128} — 그리드 서치
Expert hidden dim d_hidden: {128, 256, 512}
GCN layers K: {2, 3, 4}
SASRec sequence length N: {20, 50}
SASRec attention heads: 2
Negative sampling ratio: {1:4, 1:8}
Early stopping: Validation MAP@12, patience=3
```

### 7.8 추론 파이프라인 및 서빙 아키텍처

#### 7.8.1 Pre-store 파이프라인 (오프라인)

전체 아이템의 Factual Expert 출력과 유저의 Reasoning Expert 출력을 사전 계산하여 Attribute Store에 저장한다. Gating은 user-item 쌍별로 온라인에서 계산한다.

```
[오프라인 배치 잡]

1. 아이템 인코딩 배치:
   for each item in articles:
     h_fact = Encoder(text_factual^item)
     e_fact^item = Expert_fact(h_fact)
     STORE(item_id → e_fact^item)

2. 유저 인코딩 배치:
   for each user in customers:
     h_reason = Encoder(text_reasoning^user)
     e_reason^user = Expert_reason(h_reason)
     STORE(user_id → e_reason^user)

3. Gating: 온라인에서 (user, item) 쌍별 계산

4. 저장 형식:
   .npz: item_store.npz (e_fact: [N_items, d_rec], ids: [N_items])
         user_store.npz (e_reason: [N_users, d_rec], ids: [N_users])
   서빙 시: numpy 배열을 메모리에 로드, dict(id → index) 룩업 (<0.01ms)
```

Pre-store 소요 시간 추정: 아이템 105K × 1 텍스트 인코딩(BGE-base, GPU) = 105K 인퍼런스 (~8분) + Expert(GPU) ~2분. 유저 50만 명 × 1 텍스트 (~45분) + Expert ~5분. 일일 배치 작업으로 충분히 처리 가능하다.

#### 7.8.2 추천 서빙 (온라인)

105K 아이템 규모에서는 **전체 카탈로그를 직접 스코어링**하는 1-Stage 방식을 채택한다. 대규모 카탈로그(1M+)에서 필요한 Candidate Generation + Ranking의 2-Stage 파이프라인은 이 규모에서 불필요하며, recall loss라는 교란 변수를 도입한다.

```python
# Pre-store된 e_aug 룩업 (in-memory numpy, <0.01ms)
e_aug_user = user_store[user_id]       # (d_rec,)
e_aug_items = item_store[all_item_ids]  # (105K, d_rec)

# 전체 카탈로그 직접 스코어링 (JAX vmap + JIT, ~15ms)
scores = jax.vmap(model.score, in_axes=(None, 0))(
    user_features + e_aug_user,
    all_item_features + e_aug_items
)  # (105K,)

recommendations = jax.lax.top_k(scores, 12)
```

이 방식의 장점:
- 모든 Level(0~4) 비교 실험에서 **동일 조건**으로 공정 비교 가능
- 후보 생성 전략(인기도/CF/속성)에 따른 recall loss가 교란 변수로 작용하지 않음
- 아키텍처가 단순하여 구현·디버깅·실험 모두 간결

**Latency 예산:** Pre-store된 e_aug 룩업(in-memory numpy, <0.01ms) + 전체 카탈로그 스코어링(JAX vmap + JIT, ~15ms) ≈ 총 ~15ms per request.

> **Future Work:** 카탈로그가 1M+ 규모로 확장될 경우, Faiss HNSW 기반 2-Stage(후보 생성 → 랭킹) 파이프라인으로 전환 가능. 속성 벡터 기반, 인기도 기반, CF 기반, 시퀀셜 기반 등 다양한 후보 생성 전략의 조합을 검토할 수 있다.

#### 7.8.3 Augmented Vector 갱신 전략

유저의 Augmented Vector는 행동 축적에 따라 갱신이 필요하다.

**아이템:** 카탈로그 변동(신규 상품, 가격 변경, 계절 전환) 시 해당 아이템만 재계산. 일일 배치.

**유저:** 유저 프로파일의 증분 갱신 주기(N건 누적 또는 T일 경과)에 맞춰 재계산. 주간 배치가 기본이며, 활성 유저는 더 빈번하게 갱신 가능.

**신규 아이템 (Item Cold-start):** LLM 속성 추출 → 텍스트 인코딩 → Expert + Gating → e_aug^item 생성의 전 과정을 신규 아이템 입고 시 실시간으로 수행한다. 속성 기반 Augmented Vector가 존재하므로, 구매 이력이 전혀 없어도 Candidate Generation의 속성 기반 전략으로 즉시 추천 가능하다.

**신규 유저 (User Cold-start):** 초기에는 인구통계 피처(age, club_status)로부터 가장 유사한 기존 세그먼트의 평균 e_aug^user를 프록시로 사용한다. 첫 구매 후 L1 기반 약식 프로파일 → 5건 이상 축적 시 LLM 기반 L2/L3 추론의 점진적 고도화 전략을 적용한다.

### 7.9 설명 가능한 추천 (Explainability)

#### 7.9.1 Gating Weight 기반 설명

Gating Network의 g_fact, g_reason 가중치가 추천 이유의 지식 유형 수준 설명을 제공한다.

```
추천 아이템: "Slim fit cashmere sweater"
유저에 대한 Gating 가중치: g_fact=0.4, g_reason=0.6

설명 생성 (LLM):
"이 캐시미어 스웨터를 추천합니다. 고객님의 구매 패턴과
감성적 선호를 종합 분석한 결과(g_reason=0.6),
편안하면서도 고급스러운 느낌을 추구하시는 성향에
이 아이템의 소재와 핏이 잘 부합합니다."
```

#### 7.9.2 Factual 텍스트 내 L1/L2/L3 속성별 매칭 분석

Gating은 Item Factual / User Reasoning 2-way이지만, Item Factual 텍스트 내의 L1/L2/L3 속성을 개별 분석하여 세밀한 설명이 가능하다. Item Factual 텍스트 내에서 유저 프로파일과 아이템 속성 간 매칭 점수를 속성 그룹(L1/L2/L3)별로 분해한다.

```
L1 매칭 (제품 속성): 소재 유사 (캐시미어 → 유저 울/캐시미어 선호 45%)
L2 매칭 (체감 속성): 무드 일치 (Cozy+Elegant → 유저 Cozy 무드 선호 60%)
L3 매칭 (이론 속성): 실루엣 일치 (H-line → 유저 H/I-line 선호 65%)
```

이처럼 구현은 Item Factual + User Reasoning 비대칭 2-Expert이지만, L1/L2/L3 개념적 분류가 설명 가능성에서 여전히 핵심 역할을 한다.

#### 7.9.3 세그먼트 기반 설명

유저의 세그먼트 소속 정보와 Affinity Matrix를 활용하여 집단 수준의 설명도 가능하다.

```
"Scandinavian Minimalist 스타일을 선호하시는 고객님께, 같은
스타일 성향의 고객님들이 최근 가장 많이 구매하신 아이템입니다."
```

---

## 8. 실험 설계

### 8.1 Baseline 모델

**Non-personalized:** 전체 인기 Top-12, 최근 1주 인기 Top-12.

**Classical CF:** UserKNN (Implicit ALS), ItemKNN (TF-IDF 코사인), BPR-MF.

**Deep CF (속성 증강 없음):** DeepFM (기존 메타데이터만), SASRec (ID 임베딩만), LightGCN (ID 임베딩만), DCNv2 (기존 메타데이터만), DIN (ID 임베딩만).

**KAR 원본 재현:** Factorization Prompting으로 단일 수준 지식 추출 + 비대칭 2-Expert Adaptor(Item Factual + User Reasoning). 본 프로젝트의 구현 아키텍처와 동일한 비대칭 구조이나, 속성 추출이 3-Layer 체계 없이 단일 수준으로 수행된다.

**단일 텍스트 속성 Baseline:** detail_desc를 직접 텍스트 인코딩하여 단일 벡터로 결합 (LLM 속성 추출 없이).

### 8.2 제안 모델 변형

#### 8.2.1 Factual 텍스트 구성 Ablation — L1/L2/L3 기여도 분석 (7 변형)

구현은 Item Factual + User Reasoning 비대칭 2-Expert이지만, **Item Factual 텍스트에 포함하는 속성의 범위**를 변경하여 각 개념적 Layer의 기여를 정량화한다.

| 변형 | Item Factual 텍스트 구성 | User Reasoning 텍스트 | 목적 |
|------|--------------------------|----------------------|------|
| L1 Only | L1 속성만 | L1 기반 유저 추론만 | L1 단독 기여 |
| L2 Only | L2 속성만 | L2 기반 유저 추론만 | L2 단독 기여 |
| L3 Only | L3 속성만 | L3 기반 유저 추론만 | L3 단독 기여 |
| L1+L2 | L1+L2 통합 | L1+L2 기반 유저 추론 | L3 제외 영향 |
| L1+L3 | L1+L3 통합 | L1+L3 기반 유저 추론 | L2 제외 영향 |
| L2+L3 | L2+L3 통합 | L2+L3 기반 유저 추론 | L1 제외 영향 |
| Full (L1+L2+L3) | 전체 통합 | 전체 기반 유저 추론 | 기본 설정 |

이 접근의 핵심 가치: Expert 아키텍처를 변경하지 않으면서(항상 Item Factual + User Reasoning 비대칭 2-Expert), 입력 텍스트의 내용만 변경하여 각 개념적 Layer의 순수 기여도를 측정한다. 구현의 일관성을 유지하면서도 개념적 분석의 세밀함을 달성한다.

> **Category-Adaptive 호환성:** L1 Only 변형에서 제거되는 속성은 Super-Category별로 다르다(Apparel: neckline/sleeve_type/fit/length, Footwear: toe_shape/shaft_height/heel_type/sole_type, Accessories: form_factor/size_scale/wearing_method/primary_function). 마찬가지로 L3 Only 변형에서도 Category-Specific 슬롯(silhouette vs foot_silhouette vs visual_form 등)이 카테고리별로 다르다. 그러나 Layer 단위 ablation의 논리 구조는 변하지 않는다 — "L1을 제거한다"는 모든 카테고리에서 해당 Layer의 Shared + Category-Specific 속성을 동시에 제거하는 것이므로, 7종 변형 설계는 Category-Adaptive와 완전 호환된다.

#### 8.2.2 인코더 변형 (3 변형)

Frozen BGE-base (기본), Fine-tuned BGE-base, Lightweight TF-IDF+Projection.

#### 8.2.3 Gating 변형 (4 변형)

G1 (Fixed), G2 (Expert-conditioned, 기본, KAR 원 방식), G3 (Context-conditioned), G4 (User-Item Cross).

#### 8.2.4 Fusion 변형 (4 변형)

F1 (Concat), F2 (Addition, 기본), F3 (Gated Addition), F4 (Cross-Attention).

#### 8.2.5 백본 변형 (5 변형)

DeepFM + 속성 증강 (기본), SASRec + 속성 증강, LightGCN + 속성 증강, DCNv2 + 속성 증강, DIN + 속성 증강.

#### 8.2.6 학습 전략 변형 (2 변형)

Multi-stage (기본), End-to-End.

#### 8.2.7 실험 우선순위

모든 조합을 완전 탐색하면 7×3×4×4×5×2 = 3,360 실험이 되므로 비현실적이다. 다음과 같은 "고정-변동(Fix-and-Vary)" 전략으로 진행한다.

먼저 기본 설정(Full L1+L2+L3 Factual 텍스트, Frozen BGE, G2 Gating, F2 Fusion, DeepFM, Multi-stage)으로 Baseline 대비 성능 향상을 확인한다. 그 다음 한 번에 하나의 축만 변동하여 최적 설정을 찾는다: Factual 텍스트 구성 Ablation → 최적 Gating 선택 → 최적 Fusion 선택 → 최적 인코더 → 최적 백본. 최종 최적 설정으로 전체 실험을 수행한다.

### 8.3 평가 프로토콜

**메트릭:** MAP@12(Kaggle 원 대회 메트릭), HR@12, NDCG@12, MRR.

**검증 전략:** 시간 기반 분할. 2018.09–2020.06(학습), 마지막 7일(내부 검증), 2020.07–2020.08(최종 검증).

**Cold-start 별도 평가:** 검증 기간에 처음 등장하는 유저(user cold-start)와 아이템(item cold-start)에 대한 성능을 별도 보고.

**통계적 유의성:** 5-fold 시간 기반 교차 검증, Paired t-test.

### 8.4 심화 분석 실험

**Ablation Study:** Item Factual 텍스트 구성 변형(L1 Only~Full)에 의한 각 개념적 Layer의 한계 기여(marginal contribution) 정량화. L3 추출 비용 대비 성능 향상의 ROI 분석. 추가로, Super-Category별(Apparel/Footwear/Accessories) Layer 한계 기여를 세분화하여 보고한다 — 예를 들어 L3의 Category-Specific 속성(silhouette vs foot_silhouette vs visual_form)이 카테고리별로 다른 ROI를 보이는지 분석한다.

**Gating Weight Analysis:** g_fact, g_reason을 유저 세그먼트별, 아이템 카테고리별, Cold-start 여부별로 분석. "Cold-start 유저에서 g_reason(추론적 지식)이 더 중요해지는가?", "특정 카테고리에서 g_fact(사실적 지식)가 지배적인가?"

**Factual 텍스트 내 L1/L2/L3 속성 기여 분석:** Ablation 결과에서 "L1+L2 → Full(L1+L2+L3) 전환 시 성능 향상폭"으로 L3의 순수 한계 기여를 측정. 유저 세그먼트별로 L3 기여가 유의하게 다른지 분석(예: "Scandinavian Minimalist 세그먼트에서 L3 실루엣/퍼스널 컬러 속성이 특히 중요한가?").

**Attribute Quality Evaluation:** 무작위 500개 아이템 × 3인 어노테이터(Cohen's Kappa) + GPT-4 LLM-as-Judge 대규모 자동 평가.

**Segmentation Utility:** 세그먼트별 특화 모델 vs 단일 모델 성능 비교. Intra-segment vs Inter-segment 추천 성능 차이.

**Cold-Start Deep Dive:** 아이템 cold-start(LLM 속성 기반 클러스터 매핑 vs 인기도), 유저 cold-start(L1 초기 세그먼트 매핑 vs 인기도), 구매 건수별(1건, 3건, 5건, 10건) 성능 추이. Category-Adaptive 속성의 cold-start 이점도 정량화한다 — 신규 아이템이 Super-Category 전용 프롬프트로 즉시 category-appropriate 속성을 획득하므로, "neckline: N/A" 같은 무의미 패딩 없이 정보 밀도 높은 Factual 벡터를 생성할 수 있다. Super-Category별 cold-start 성능을 별도 보고하여 Footwear/Accessories 등 소규모 카테고리에서의 효과를 검증한다.

**시간 축 분석:** 분기별 모델 성능 변화, 속성 Augmentation의 시간적 안정성(2019년 학습 → 2020년 테스트에서도 속성 시그널이 유효한가?), COVID 기간(2020 상반기)에서의 속성 기반 모델 강건성.

---

## 9. 구현 로드맵

### Phase 0: 데이터 준비 (Week 1-2)

Kaggle 데이터 다운로드, EDA, 데이터 분할 확정. Baseline 추천 모델(인기도, UserKNN, BPR-MF) 구현 및 벤치마크 기록.

### Phase 1: L1 속성 추출 (Week 3-4)

메타데이터 파싱 파이프라인 + LLM/VLM 보강 L1 추출. 500개 파일럿 → 전체 105K 배치.

### Phase 2: L2/L3 속성 추출 (Week 5-7)

L2/L3 프롬프트 설계, 파일럿, 품질 검증, 전체 배치 추출, 500샘플 × 3인 어노테이션.

### Phase 3: 유저 프로파일 추론 (Week 8-9)

L1 직접 집계 + L2/L3 LLM Factorization Prompting 배치 처리. Attribute Store 구축.

### Phase 4: 세그멘테이션 & 분석 (Week 10-12)

고객/상품 세그멘테이션, 카탈로그 갭 분석, Affinity Matrix, 타겟팅 시나리오, 시각화 대시보드.

### Phase 5: 추천 모델 아키텍처 구현 (Week 13-15)

텍스트 인코더 파이프라인 구축. Expert Network + Gating Network 구현(JAX/Flax NNX). Embedding Fusion 모듈 구현. Pre-store 파이프라인 구축. DeepFM 백본 직접 구현 및 Multi-stage 학습 파이프라인 검증. 기본 설정으로 첫 End-to-End 실험 수행.

### Phase 6: 체계적 실험 수행 (Week 16-19)

Fix-and-Vary 전략에 따른 체계적 실험 수행. Layer 조합 Ablation → Gating 변형 → Fusion 변형 → 인코더 변형 순서로 탐색. 최적 설정 확정 후, 전체 백본(SASRec, LightGCN, DCNv2, DIN) 변형 실험. Cold-start, 시간 축 분석, 세그멘테이션 유틸리티 등 심화 분석 수행.

### Phase 7: 2-Stage 서빙 파이프라인 및 분석 (Week 20-21)

Candidate Generation 전략 구현(인기도, CF, 속성 기반, 시퀀셜). 2-Stage 파이프라인 성능 측정. Latency 프로파일링.

### Phase 8: 결과 정리 및 논문화 (Week 22-24)

전체 실험 결과 정리, 시각화. 주요 발견 정리(Layer별 기여도, Gating 해석, Cold-start 효과, 세그멘테이션 활용). 논문 집필, 재현 가능한 코드 공개 준비.

---

## 10. 기술 스택

### 10.1 속성 추출

L1/L2/L3 통합: GPT-4.1-nano(멀티모달) + Structured Output. 배치: OpenAI Batch API (50% 할인). 캐시: product_code 기반 dict + Parquet 체크포인트.

### 10.2 분석

scikit-learn, UMAP, hdbscan(클러스터링). BGE-base-en-v1.5(텍스트 인코딩). Plotly + Streamlit(대시보드).

### 10.3 추천 모델

JAX + Flax NNX(전체 모델 파이프라인). 5종 백본(DeepFM, SASRec, LightGCN, DCNv2, DIN)과 KAR 확장(Expert Adaptor, Gating Network, Embedding Fusion)을 모두 Flax NNX로 직접 구현한다. NNX의 Pythonic 뮤터블 API가 모듈 조립과 디버깅에 유리하며, `nnx.jit`으로 학습 루프를 JIT 컴파일하여 실험 처리량을 높인다. 105K 아이템 전체 카탈로그를 `jax.vmap`으로 직접 스코어링한다. Optax(옵티마이저). Weights & Biases(실험 관리).

### 10.4 데이터 저장

DuckDB(오프라인 피처 계산 및 분석 쿼리) + Parquet(DuckDB 네이티브 연동, 거래/메타데이터 저장). Attribute Store(e_aug 벡터 + Gating 가중치)는 .npz로 저장하고 서빙 시 in-memory numpy 배열로 로드한다(105K 아이템 × 128차원 ≈ 50MB, dict 룩업 <0.01ms). 별도의 벡터 DB(pgvector, ChromaDB 등)나 캐시(Redis)는 이 규모에서 불필요하며, 정식 Feature Store(Feast 등)도 1인 연구 프로젝트에서는 DuckDB + Parquet + .npz 조합이 동일 기능을 더 단순하게 제공하므로 도입하지 않는다.

---

## 11. 예상 결과 및 기여

### 11.1 정량적 예상

MAP@12 기준으로 Kaggle Silver Medal 수준(~0.029)에서 속성 증강을 통해 유의미한 개선을 목표로 한다. 특히 Cold-start 상황에서 인기도 기반 대비 50% 이상의 HR@12 향상을 기대한다. 각 Layer의 한계 기여를 정량화하되, L3의 성능 기여가 L2 대비 적을 수 있으며 이 경우 비용 대비 ROI를 투명하게 보고한다.

### 11.2 기술적 기여

첫째, 3-Layer Attribute Taxonomy의 최초 제안 및 체계적 실증. 둘째, KAR의 검증된 비대칭 2-Expert 구조(Item Factual + User Reasoning) 위에 3-Layer 개념적 속성 체계를 통합하는 방법론 제시(텍스트 수준 Ablation으로 개념적 Layer 기여 정량화). 셋째, 5종 추천 백본에서의 model-agnostic 속성 증강 효과 입증. 넷째, 속성 기반 Candidate Generation이 CF 기반 대비 다양성·커버리지에서 우위를 보이는지 검증. 다섯째, Pre-store + 2-Stage 서빙의 실시간 추천 가능성 시연.

### 11.3 분석적 기여

첫째, LLM 추출 속성 기반 다층 세그멘테이션이 기존 카테고리 기반 세그멘테이션 대비 추천 성능과 해석 품질에서 우위를 보이는지 검증. 둘째, Affinity Matrix 기반 "잠재 기회 타겟팅"의 발견 및 Hit Rate 측정. 셋째, H&M 카탈로그에서 교차 카테고리 유사성 발견과 카탈로그 갭 식별의 실용적 가치 시연.

---

## 12. Future Work

### 12.1 멀티모달 임베딩 통합

현재 설계에서는 이미지가 LLM 속성 추출의 입력으로만 사용되고, 추출 후 시각 정보가 텍스트로 변환되면서 소실된다(원단 질감, 디테일의 시각적 인상, 전체적 무드 등). CLIP ViT-L/14의 이미지 임베딩을 아이템 피처로 보존하여 추천 모델에 직접 통합하는 확장을 계획한다.

이를 별도 단계로 분리하는 이유: CLIP 시각 벡터(768차원)를 Embedding Fusion에 추가하면, 입력 차원과 모달리티 수가 변경되어 현재의 Fusion 전략(F1~F4), Expert 아키텍처, Gating 구조를 재설계해야 할 수 있다. 본 프로젝트의 1차 목표인 "3-Layer 텍스트 속성의 추천 기여 검증"이 깔끔하게 완료된 후, 멀티모달 확장을 독립 실험으로 수행한다.

계획하는 접근: (a) CLIP 시각 벡터를 별도 Expert로 추가하는 3-Expert 구조(Item Factual + User Reasoning + Item Visual), (b) Late Fusion — e_aug와 v_item을 독립적으로 계산 후 Ranking 단계에서 결합, (c) 시각 벡터를 Item Factual 텍스트 인코딩과 concat하여 기존 비대칭 2-Expert 구조를 유지하는 경량 접근. 각 접근의 아키텍처 변경 범위와 성능을 비교한다.
