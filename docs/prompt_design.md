# Prompt Design: KAR Knowledge Extraction (Factual + Reasoning)

## Design Background

H&M 데이터의 Triple-Sparsity(유저 32.1% 희소 + 99.98% 행렬 sparse + CF 시그널 품질 저하)로 인해
협업 필터링만으로는 구조적 한계가 있다. LLM 추출 속성을 통해 CF 시그널이 부족한 아이템/유저에 대한
콘텐츠 기반 보완 정보를 제공한다.

### KAR 비대칭 2종 지식 구조

KAR(Knowledge Augmented Recommendation)은 **아이템 측 Factual Knowledge**와 **유저 측 Reasoning Knowledge**를
비대칭적으로 추출하여 각각 Factual Expert와 Reasoning Expert에 입력한다.

| 지식 유형 | 대상 | 추출 방식 | Expert |
|-----------|------|-----------|--------|
| **Factual** (Part I) | 아이템 (~47K products) | LLM/VLM 속성 추출 (L1+L2+L3) | Factual Expert |
| **Reasoning** (Part II) | 유저 (~1.3M customers) | L1 집계 + LLM 추론 (Active) / Template (Sparse) | Reasoning Expert |

- **Factual**: 아이템의 객관적·주관적·이론적 속성을 구조화 → `factual_text`로 인코딩
- **Reasoning**: 유저의 구매 패턴에서 선호도를 추론 → `reasoning_text`로 인코딩
- 두 Expert의 출력은 Gating Network를 거쳐 백본 임베딩에 융합

---

## Part I: Item Factual Knowledge

### 3-Layer Attribute Taxonomy

#### L1 (Product) — 객관적 물리 속성
- **Shared 필드 (4)**: material, closure, design_details, material_detail
- **Category-specific 필드 (4x3)**: Apparel/Footwear/Accessories 각각 4개
- material, closure → **enum** (카테고리별) + material_detail **free-text** (2-Level Hybrid)
- 근거: 분류 일관성 확보 + 블렌드/가공 세부사항 보존

#### L2 (Perceptual) — 주관적 체감 속성
- **7개 유니버설 필드**: style_mood, occasion, perceived_quality, trendiness, season_fit, target_impression, versatility
- perceived_quality: H&M 브랜드 내 상대 평가 (1=Divided budget, 5=designer collab)
- 근거: 추천 시스템의 사용자-아이템 감성 매칭

#### L3 (Theory) — 패션 이론 기반 속성
- **Shared 필드 (4)**: color_harmony, coordination_role, visual_weight, style_lineage
- **Post-processed 필드 (1)**: tone_season (규칙 기반 COLOR_TO_TONE 매핑)
- **Category-specific 필드 (2x3)**: silhouette/proportion, foot_silhouette/height, visual_form/styling
- 근거: Color Theory, Silhouette Theory, Coordination Theory 반영

---

### Enum 설계 원칙

#### Enum화 (Enumeration)
- 자유형식 텍스트를 미리 정의된 허용 값 목록으로 제한
- OpenAI Structured Output strict 모드에서 LLM이 반드시 목록 내 값만 출력
- 장점: 일관성, 분석 용이, 엔트로피 붕괴 감소
- 단점: 미세한 표현 차이 손실

#### 2-Level Hybrid 패턴
- `l1_material`: enum(primary category) + `l1_material_detail`(free-text blend/finish details)
- 적용 이유: 소재 분류 일관성 + 블렌드/가공 세부사항 보존
- 예: material="Cotton blend", material_detail="60% cotton 40% polyester, brushed fleece finish"

#### Enum 값 선정 기준
- H&M 카탈로그 실 데이터 커버리지 (파일럿 500개 분석)
- 패션 도메인 표준 용어
- "Other" 이스케이프: material/closure에만 (style_lineage는 45값으로 충분한 커버리지)

#### 카테고리별 Enum
| 필드 | Apparel | Footwear | Accessories |
|------|---------|----------|-------------|
| material | 22값 (Cotton~Other) | 12값 (Leather~Other) | 13값 (Metal~Other) |
| closure | 11값 (Pullover~N/A) | 9값 (Lace-up~None) | 11값 (Clasp~N/A) |
| style_lineage | 45값 (공통) | 45값 (공통) | 45값 (공통) |

---

### Cross-Attribute Consistency Rules

visual_weight 의미 혼동 방지를 위한 교차 속성 일관성 규칙:

#### Error-severity (8개)
1. `coordination_x_visual_weight`: Basic→weight≤3, Statement→weight≥3
2. `silhouette_x_visual_weight`: I-line→weight≤3, O-line→weight≥3
3. `fit_x_visual_weight`: Slim/Skinny→weight≤3, Oversized/Loose/Boxy→weight≥3
4. `sleeve_x_season`: Sleeveless→season≠Winter
5. `neckline_x_sleeve`: Strapless→sleeve∈{Sleeveless, N/A}
6. `sole_x_season`: Foam→season≠Winter
7. `function_x_form_factor`: Storage→form_factor≠{Mini, Compact}
8. `wearing_x_size_scale`: Wrist/Finger→size∈{Petite, Small}

#### Warning-severity (4개)
9. `mood_x_occasion`: Bohemian→occasion≠{Work, Formal}
10. `heel_x_occasion`: Stiletto→occasion≠Outdoor
11. `coordination_x_harmony`: Basic→harmony∈{Mono, Neutral, Earth}, Accent→harmony≠Mono
12. `lineage_x_mood`: 모순 쌍 체크 (Punk+Classic, Grunge+Luxury 등)

구현: `validator.py`의 `validate_domain_consistency()` → `DomainViolation` NamedTuple 리스트

#### 프롬프트 내 배치 전략: v2.1 실패 → Rule-Based Post-Processing 전환

**v2.0 상태**: 교차속성 규칙이 `_COMMON_INSTRUCTIONS` 중간에 위치. Domain Error 11.6% (58/500건).

**v2.1 CHECKLIST 시도 (실패)**:
- **전략**: `_COMMON_INSTRUCTIONS`에 forward reference만 남기고, 각 카테고리 프롬프트 맨 끝에 VERIFICATION CHECKLIST (`[ ]` 체크박스 + WRONG/RIGHT 마이크로 예시) 배치
- **결과**: Health 91.3% (v2 대비 **-1.6pp 하락**), Domain Error 13.0%
- **원인**: GPT-4.1-nano의 attention 용량 한계 — 체크리스트 추가로 프롬프트 길이 증가 시 오히려 attention 분산, 핵심 규칙 주의력 약화
- **결론**: 소형 모델에서 프롬프트 내 검증 지시는 역효과. **v2 프롬프트로 롤백**.

**최종 해결: `correct_visual_weight()` 규칙 기반 후처리**:
- 프롬프트 수정 대신, 추출 후 규칙 기반으로 visual_weight를 보정
- silhouette / fit / coordination_role 3개 속성에서 허용 범위를 도출하여 교집합 clamping
- 상세: 아래 "Visual Weight Post-Processing" 섹션 참조

---

### 프롬프트 구조

#### System Prompt (v2 — 현행)
- `_COMMON_INSTRUCTIONS`: 공통 규칙, 교차속성 일관성 규칙 (visual_weight = FORM/VOLUME 정의), Per-Item Specificity 다양성 지시문
- Category-specific: Apparel/Footwear/Accessories 각각의 가이드 + 예시 JSON + 대비 사례(Contrast)
- v2.1 CHECKLIST는 시도 후 실패 → 제거됨 (위 "v2.1 실패" 참조)

#### User Message
- 9개 메타데이터 필드 + detail_desc + 이미지 (base64, low detail)
- 동적 suffix: `"Analyze THIS specific {product_type_name}'s unique visual details."`

#### JSON Schema
- OpenAI Structured Output strict 모드
- 21필드 (LLM 추출) + 1필드 (후처리 tone_season) = **22필드 최종**

---

### Color Override 시스템

| 필드 | LLM 추출 | 규칙 오버라이드 | 근거 |
|------|---------|---------------|------|
| tone_season | 제거 (95.2% 불일치) | COLOR_TO_TONE (50개 colour_group) | 규칙이 압도적 정확 |
| color_harmony | 유지 | COLOR_TO_HARMONY (fallback) | LLM 70.6% 일치, 규칙 보완 |

---

### Visual Weight Post-Processing

v2.1 CHECKLIST 실패 후 도입된 규칙 기반 보정. `extractor.py`의 `correct_visual_weight()`.

#### 동작 원리
1. silhouette, fit, coordination_role 각각의 허용 visual_weight 범위를 매핑 테이블에서 조회
2. 3개 범위의 **교집합**을 구하고, LLM 출력 값을 해당 범위로 clamp
3. 교집합이 공집합이면 **silhouette 범위를 우선** 적용 (fallback)
4. **Apparel only** — Footwear/Accessories는 silhouette/fit 필드가 없어 적용 대상 아님

#### 매핑 테이블 3종

**SILHOUETTE_WEIGHT_RANGE** (silhouette → min, max):
| Silhouette | Range | 근거 |
|-----------|-------|------|
| I-line | 1–3 | 직선형, 볼륨 없음 |
| H/X/A/V/Y-line | 2–4 | 중간 구조 |
| O-line, Trapeze, Cocoon | 3–5 | 볼륨 실루엣 |
| Empire | 2–4 | 상체 짧고 하체 유동 |

**FIT_WEIGHT_RANGE** (fit → min, max):
| Fit | Range | 근거 |
|-----|-------|------|
| Slim, Skinny, Bodycon | 1–3 | 몸에 밀착 |
| Tailored | 2–3 | 구조적이나 밀착 |
| Regular, Relaxed | 2–4 | 중간 |
| Wide, Loose, Boxy, Oversized | 3–5 | 볼륨 추가 |

**COORDINATION_WEIGHT_RANGE** (coordination_role → min, max):
| Role | Range | 근거 |
|------|-------|------|
| Basic, Foundation, Finishing | 1–3 | 눈에 띄지 않는 역할 |
| Layering, Accent | 2–4 | 중간 존재감 |
| Statement | 3–5 | 시각적 주도 |

#### 알고리즘
```python
# 범위 교집합
lo = max(range[0] for range in ranges)
hi = min(range[1] for range in ranges)
if lo > hi:  # 교집합 공집합
    lo, hi = silhouette_range  # silhouette 우선 fallback
clamped = clamp(weight, lo, hi)
```

#### 적용 시점
`_build_article_rows()` 내에서 `map_to_canonical_slots()` **이전**에 호출:
```
LLM 추출 → correct_visual_weight() → map_to_canonical_slots() → propagate_to_variants()
```

---

### Quality Assurance

- Pilot → Deep Dive → Health Score 기반 Go/No-Go
- 도메인 규칙 12개 자동 검증 (Error 8개 + Warning 4개)
- 엔트로피 모니터링: product_type x attribute 셀 단위 붕괴 감지
- 목표: Health Score >= 90%

#### 실측 결과 (v2 프롬프트 + Post-Processing)
- **v2 프롬프트 단독**: Health 92.9% (YELLOW), Domain Error 11.2%
- **v2.1 CHECKLIST (실패)**: Health 91.3%, Domain Error 13.0% → 롤백
- **Post-processing 후 목표**: Domain Error ~1% (visual_weight 위반 제거), Health ≥95% (GREEN)

---

### Text Composition for KAR

- 7개 ablation variant: L1, L2, L3, L1+L2, L1+L3, L2+L3, L1+L2+L3
- 메타데이터 항상 포함, `[Product]`, `[Perceptual]`, `[Theory]` 섹션 마커
- BGE-base-en-v1.5 인코딩 (512 토큰 제한, 실측 ~130 토큰)

---

### Factual 비용 분석

| 항목 | 값 |
|------|-----|
| 모델 | GPT-4.1-nano (Batch API, 50% discount) |
| System prompt | ~720 tokens |
| User message | ~350 tokens |
| Output (21 fields) | ~360 tokens |
| 요청당 합계 | ~1,430 tokens |
| 47K 전체 배치 비용 | ~$8.50 |
| 예산 한도 | $15.0 |

---

### 필드 수 요약

```
LLM 스키마: 21필드
  L1 shared (4): material, closure, design_details, material_detail
  L1 specific (4): category-dependent
  L2 (7): style_mood, occasion, perceived_quality, trendiness, season_fit, target_impression, versatility
  L3 shared (4): color_harmony, coordination_role, visual_weight, style_lineage
  L3 specific (2): category-dependent

후처리 보정 (2종):
  1. tone_season  — COLOR_TO_TONE rule (colour_group_name → 6 tones)
  2. visual_weight — correct_visual_weight() rule (silhouette×fit×coordination 범위 clamping, Apparel only)

최종 출력: 22필드 (visual_weight는 기존 필드의 보정이므로 필드 수 변경 없음)
```

---

## Part II: User Reasoning Knowledge

### 설계 동기

KAR 원 논문에서 Reasoning Knowledge는 유저의 구매 이력으로부터 **선호도 패턴을 LLM이 추론**하여 자연어로 표현한 것이다. Factual Knowledge가 아이템의 "무엇(what)"을 기술한다면, Reasoning Knowledge는 유저의 "왜(why)"를 포착한다.

**H&M 특수성**: 1.3M 유저 중 Active(5+ 구매) 876K명 + Sparse(1-4 구매) 421K명의 이원 분포. Sparse 유저는 구매 데이터가 너무 빈약하여 LLM 추론의 입력 신호가 불충분하고, 전수 LLM 호출은 비용($100+)이 과다하다. 따라서:

- **Active 유저 (5+ 구매)**: LLM Reasoning — 충분한 구매 이력에서 패턴 추론
- **Sparse 유저 (1-4 구매)**: Template Fallback — 구매 아이템 속성 직접 집계, LLM 미사용

두 경로 모두 동일한 9-field 구조 + `(a)~(i)` 텍스트 포맷을 출력하여, Reasoning Expert 입력이 일관적이다.

---

### 3-Stage Pipeline 개관

```
Stage A: L1 Direct Aggregation (DuckDB bulk, 전 유저)
    ↓ per-user L1 통계: 카테고리/컬러/소재 분포, 가격, 채널, 다양성
    ↓ + 최근 20건 아이템의 L2 속성
    ↓ + L3 속성 분포 (shared + category-specific)

Stage B: LLM Reasoning (Active 유저 876K명, GPT-4.1-nano)
    ↓ L1 통계 + L2 최근 아이템 + L3 분포 → 3-section user message
    ↓ System prompt + JSON Schema → 9-field reasoning_json
    ↓ → compose_reasoning_text() → reasoning_text

Stage C: Sparse Fallback (Sparse 유저 421K명, Template)
    ↓ 구매 아이템 L2/L3 속성 직접 집계
    ↓ → _build_single_sparse_profile() → reasoning_json
    ↓ → compose_sparse_reasoning_text() → reasoning_text
```

- **Stage A**는 모든 유저에 대해 수행 (DuckDB bulk aggregation)
- **Stage B**는 Active 유저에만 적용 (LLM API 호출)
- **Stage C**는 Sparse 유저에만 적용 (no LLM, template 로직)

---

### System Prompt

```
You are a fashion consumer analyst. Given a customer's purchase summary and
attribute patterns, produce a structured 9-dimensional preference profile.

**Input format:**
1. Customer Overview — purchase count, top categories, price position, online
   ratio, diversity score
2. Recent Items (L2 Attributes) — last 20 items with style mood, occasion,
   quality, trendiness, season, impression, versatility
3. Attribute Patterns (L3 Theory-Based) — distributions of color harmony, tone
   season, coordination role, visual weight, style lineage, plus
   category-specific silhouette/form patterns

**Rules:**
- Synthesize patterns across ALL purchased items, not just one or two.
- For each field, identify the DOMINANT tendency and note significant secondary
  patterns.
- Be specific to THIS customer — avoid generic descriptions.
- Consider cross-attribute relationships (e.g., high versatility + basic
  coordination → wardrobe builder).
- The identity_summary should be a single compelling sentence that captures the
  customer's unique fashion identity.
- Keep each field concise (1-2 sentences max).
- If data is sparse for a dimension, note the limitation rather than speculating.
```

**설계 의도 — 6개 규칙:**

| # | 규칙 | 목적 |
|---|------|------|
| 1 | Synthesize ALL items | 최근 편향 방지, 전체 구매 패턴 반영 |
| 2 | DOMINANT + secondary | 주요 경향 + 보조 패턴 모두 포착 |
| 3 | Specific to THIS customer | Generic 출력 방지 (개인화 핵심) |
| 4 | Cross-attribute relationships | L2↔L3 교차 패턴 추론 유도 |
| 5 | identity_summary single sentence | 압축적 정체성 표현 → 임베딩 품질 |
| 6 | Sparse → note limitation | 데이터 부족 시 환각 대신 솔직한 표현 |

---

### User Message 구조 (3-Section)

LLM에 전달되는 user message는 3개 섹션으로 구성된다. `build_reasoning_user_message()`이 L1 통계, L2 최근 아이템, L3 분포를 결합하여 생성.

#### Section 1: Customer Overview (L1 통계)

```
--- Customer Overview ---
Purchases: 113 items, 28 unique types, diversity 0.87
Top categories: Jersey 22%, Trousers 14%, Sweater 9%, T-shirt 8%, Dress 7%
Price position: avg 3.0/5, Online ratio: 62%
```

- `n_purchases`, `n_unique_types`, `category_diversity`: 구매 규모와 다양성
- `top_categories_json`: 시간 가중 카테고리 분포 (exponential decay, halflife=90일)
- `avg_price_quintile`: 1-5 가격 포지션
- `online_ratio`: 온라인 구매 비율

#### Section 2: Recent Items L2 Attributes (최근 20건)

```
--- Recent Items (L2 Attributes) ---
1. Apparel | Style: Casual, Minimalist | Occasion: Everyday | Quality: 3/5 | Trend: Classic | Season: All-season | Impression: Relaxed | Versatility: 4/5
2. Apparel | Style: Sporty | Occasion: Active | Quality: 3/5 | Trend: Current | Season: Spring-Summer | ...
...
20. Footwear | Style: Classic | Occasion: Work | Quality: 4/5 | ...
```

- 최근 20건 구매 아이템의 L2 속성 7개를 파이프(`|`) 구분 열거
- Factual Knowledge에서 추출된 L2 속성을 직접 활용

#### Section 3: Attribute Patterns L3 Theory-Based (분포)

```
--- Attribute Patterns (L3 Theory-Based) ---
Color Harmony: Monochromatic 35%, Neutral 28%, Analogous 18%, Earth 12%
Tone Season: Cool-Winter 42%, Cool-Summer 31%, Warm-Autumn 15%
Coordination Role: Basic 38%, Foundation 25%, Layering 20%
Style Lineage: Minimalist 30%, Classic 22%, Casual 18%
Visual Weight: mean 2.4, std 0.8
Category-Specific:
  Apparel (n=85): | Slot6: I-line 45%, H-line 25%, A-line 18% | Slot7: Slim 35%, Regular 30%
  Footwear (n=18): | Slot6: Low-top 55%, Ankle 30% | Slot7: Flat 60%, Low 30%
```

- Shared L3 필드: color_harmony, tone_season, coordination_role, style_lineage — 정규화 분포
- visual_weight: 수치형이므로 mean + std
- Category-specific: super_category별 l3_slot6, l3_slot7 분포

---

### JSON Schema — 9-Field Reasoning Profile

OpenAI Structured Output strict 모드로 9개 필드를 강제한다. `REASONING_SCHEMA`:

```json
{
  "type": "object",
  "properties": {
    "style_mood_preference": {
      "type": "string",
      "description": "Dominant style/mood tendencies. Synthesize from recurring L2 style_mood and target_impression patterns."
    },
    "occasion_preference": {
      "type": "string",
      "description": "Primary occasions this customer shops for. Infer from L2 occasion patterns and purchase frequency."
    },
    "quality_price_tendency": {
      "type": "string",
      "description": "Quality-price positioning. Consider perceived_quality distribution and price quintile."
    },
    "trend_sensitivity": {
      "type": "string",
      "description": "How trend-aware the customer is. Infer from L2 trendiness distribution."
    },
    "seasonal_pattern": {
      "type": "string",
      "description": "Seasonal shopping behavior. Combine L2 season_fit distribution with purchase timing."
    },
    "form_preference": {
      "type": "string",
      "description": "Cross-category form preferences. Synthesize from L3 silhouette/foot_silhouette/visual_form patterns."
    },
    "color_tendency": {
      "type": "string",
      "description": "Color preference pattern. Combine L3 color_harmony and tone_season distributions."
    },
    "coordination_tendency": {
      "type": "string",
      "description": "Outfit coordination style. Infer from L3 coordination_role and visual_weight distributions."
    },
    "identity_summary": {
      "type": "string",
      "description": "One-sentence identity summary capturing this customer's fashion identity."
    }
  },
  "required": ["style_mood_preference", "occasion_preference", "quality_price_tendency",
                "trend_sensitivity", "seasonal_pattern", "form_preference",
                "color_tendency", "coordination_tendency", "identity_summary"],
  "additionalProperties": false
}
```

**필드별 입력↔출력 매핑:**

| 출력 필드 | 주요 입력 소스 | 추론 방식 |
|-----------|---------------|-----------|
| style_mood_preference | L2 style_mood, target_impression | 반복 패턴 종합 |
| occasion_preference | L2 occasion, 구매 빈도 | 주요 용도 추론 |
| quality_price_tendency | L2 perceived_quality, price_quintile | 가격-품질 포지셔닝 |
| trend_sensitivity | L2 trendiness | 트렌드 민감도 |
| seasonal_pattern | L2 season_fit, 구매 시점 | 계절 행동 패턴 |
| form_preference | L3 slot6/slot7 (silhouette, fit 등) | 크로스 카테고리 형태 선호 |
| color_tendency | L3 color_harmony, tone_season | 색상 팔레트 경향 |
| coordination_tendency | L3 coordination_role, visual_weight | 코디네이션 스타일 |
| identity_summary | 전체 종합 | 한 문장 정체성 |

---

### Reasoning Text Composition

`compose_reasoning_text()`는 9-field JSON을 `(a)~(i)` 라벨링된 자연어 텍스트로 변환한다. 이 텍스트가 BGE-base-en-v1.5로 인코딩되어 Reasoning Expert에 입력된다.

**출력 포맷:**
```
(a) Style mood: Casual minimalist with occasional formal touches.
(b) Occasion: Everyday basics with weekend casual.
(c) Quality-price: Mid-range with selective premium purchases.
(d) Trend: Classic core with occasional current pieces.
(e) Season: Year-round basics buyer, heavier winter purchases.
(f) Form: Prefers streamlined I-line silhouettes, slim fits.
(g) Color: Monochromatic neutral palette, Cool-Winter tones.
(h) Coordination: Builds from basics, adds occasional statement pieces.
(i) Identity: A practical minimalist who values quality basics and neutral palettes.
```

**field_map 매핑:**
| 라벨 | JSON 키 | 표시명 |
|------|---------|--------|
| (a) | style_mood_preference | Style mood |
| (b) | occasion_preference | Occasion |
| (c) | quality_price_tendency | Quality-price |
| (d) | trend_sensitivity | Trend |
| (e) | seasonal_pattern | Season |
| (f) | form_preference | Form |
| (g) | color_tendency | Color |
| (h) | coordination_tendency | Coordination |
| (i) | identity_summary | Identity |

LLM 경로(`compose_reasoning_text`)와 Template 경로(`compose_sparse_reasoning_text`) 모두 동일한 `(a)~(i)` 포맷을 출력하여 Reasoning Expert 입력의 일관성을 보장한다.

---

### Sparse User Template 전략

**대상**: 1-4건 구매 유저 421K명 (`profile_source="template"`)

**LLM 미사용 근거:**
1. **입력 데이터 빈약**: 1-4건 구매로는 "패턴"이 아닌 개별 사건 — LLM 추론의 입력 신호 불충분
2. **비용**: 421K명 × ~1,500 tokens ≈ $30+ 추가 (Batch API 기준), 품질 대비 ROI 낮음
3. **System Prompt 규칙 6**: "If data is sparse, note the limitation rather than speculating" — LLM이 환각할 가능성

**9-field 매핑 로직 (`_build_single_sparse_profile`):**

| 필드 | 집계 방식 |
|------|-----------|
| style_mood_preference | L2 style_mood 빈도 top-3 |
| occasion_preference | L2 occasion 빈도 top-3 |
| quality_price_tendency | L2 perceived_quality 평균 → Budget/Mid-range/Quality-oriented |
| trend_sensitivity | L2 trendiness 최빈값 |
| seasonal_pattern | L2 season_fit 빈도 top-2 |
| form_preference | L3 slot6 빈도 top-2 |
| color_tendency | L3 color_harmony top-2 + tone_season top-2 |
| coordination_tendency | L3 coordination_role 빈도 top-2 |
| identity_summary | `"User with {n} purchase(s), limited history for deep analysis."` |

**`compose_sparse_reasoning_text()` 출력 포맷** — LLM과 동일한 `(a)~(i)`:
```
(a) Style mood: Casual, Minimalist. (b) Occasion: Everyday, Weekend.
(c) Quality-price: Mid-range (avg 3.2/5). (d) Trend: Classic.
(e) Season: All-season, Spring-Summer. (f) Form: I-line, Regular.
(g) Color: Monochromatic, Neutral; Cool-Winter, Cool-Summer.
(h) Coordination: Basic, Foundation.
(i) Identity: User with 3 purchase(s), limited history for deep analysis.
```

0건 구매 유저(트랜잭션 미발견)는 모든 필드를 `"Unknown"`으로 채우고 identity를 `"New user with no purchase history."`로 설정.

---

### Reasoning 비용 분석

파일럿 200명에서 측정한 토큰 통계 기반:

| 항목 | 값 |
|------|-----|
| 모델 | GPT-4.1-nano |
| System prompt | 243 tokens |
| Avg user message | ~1,045 tokens |
| **Avg input (system + user)** | **~1,288 tokens** |
| **Avg output (reasoning_json)** | **~277 tokens** |
| 요청당 합계 | ~1,565 tokens |

**비용 추정 (GPT-4.1-nano, $0.10/1M input, $0.40/1M output):**

| 범위 | 유저 수 | Input tokens | Output tokens | 비용 (Realtime) | 비용 (Batch API 50%) |
|------|---------|-------------|--------------|----------------|---------------------|
| Pilot (완료) | 200 | ~258K | ~55K | ~$0.05 | ~$0.02 |
| Full Active | 876K | ~1,128M | ~243M | ~$210 | **~$105** |
| Sparse Template | 421K | — | — | **$0** | **$0** |

**예산 대비**: Full batch $105 > 예산 $50 → 비용 초과. 대안:
- Active 유저 임계값 상향 (5→10+ 구매)으로 대상 축소
- 청킹 + 체크포인트로 분할 실행

---

### 파일럿 검증 결과 (200명)

`notebooks/02_pilot_profiles.ipynb`에서 200명 Active LLM 프로파일 품질 검증.

#### Go/No-Go 표

| 기준 | 임계값 | 측정값 | 판정 |
|------|--------|--------|------|
| 9-field completeness | ≥99% | **100.0%** | PASS |
| Generic response rate | <5% | **0.0%** | PASS |
| Consistency score (mean) | ≥0.4 | **0.50** | PASS |
| Token 99th percentile | ≤512 | **357** | PASS |
| Discriminability (mean sim) | <0.8 | **0.14** | PASS |
| Cost within budget ($50) | ≤$50 | **$105** | FAIL |

**결론**: 5 PASS + 1 FAIL (비용). 프로파일 품질은 우수하나 전체 배치 비용이 예산 초과.

#### 주요 수치

- **Completeness**: 9개 필드 모두 200명 전원 non-null, non-empty (100.0%)
- **Generic rate**: "Unknown", "N/A" 등 패턴 0건 / 1,800건 (0.0%)
- **Token length**: mean=277, median=277, 99th=357, max=367 — 전원 512 이내
- **Discriminability**: User-user TF-IDF cosine similarity mean=0.14, median=0.14
  - p10=0.10, p90=0.18 → 프로파일 간 높은 차별성
- **L1↔Reasoning consistency**: Category mention=0.28, Price consistency=0.72
- **Information richness**: LLM 프로파일 avg unique trigrams >> Template

---

### 구현 파일 매핑

| 파일 | 역할 |
|------|------|
| `src/knowledge/reasoning/prompts.py` | System prompt, JSON schema, user message 구성, reasoning text 변환 |
| `src/knowledge/reasoning/extractor.py` | L1 DuckDB 집계, recent items/L3 분포 조회, sparse template 생성 |
| `src/knowledge/reasoning/batch.py` | OpenAI Batch API 래퍼 (프로파일용) |
| `src/knowledge/reasoning/cache.py` | customer_id 기반 체크포인트 (500건 간격) |
| `src/config.py` | ReasoningConfig (모델/비용/임계값), ReasoningResult (실행 요약) |
| `scripts/extract_reasoning_knowledge.py` | CLI 엔트리포인트 (--pilot, --batch-api, --max-cost) |
