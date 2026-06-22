# STORY — Falsification → Diagnosis → Pivot → Honest Scale-Refutation

> H&M LLM-Factor RecSys 연구의 서사. 대부분의 포트폴리오가 *positive 결과만* 큐레이션하는 것과 달리, 이 프로젝트는
> **원래 가설을 스스로 반증하고, 실패를 mechanism까지 진단하고, 진단이 가리키는 수정을 de-risk했으며, 그 de-risk
> positive마저 full-scale 빌드로 다시 *반증*하고 원인(population-selection bias)을 격리**한 전 과정을 기록한다.
> 셀링포인트는 단일 정확도 수치가 아니라 **연구 *성숙도* — falsification rigor + diagnosis + de-risk→scale 자기반증 + honest negative**.
> (출판 가능한 *논문*은 아니다 — 단일 데이터셋·negative 결과·비-신규 방법. 강점은 *과정의 엄정성·정직성*이다.)
>
> 모든 수치는 `witnesses/probe_*_result.json` (canonical, 고정 seed, bootstrap CI)에서 그대로 인용한다.

---

## TL;DR

원래 가설 **"LLM이 추출한 다층 속성(L1 제품 / L2 체감 / L3 이론)이 추천 정확도를 높인다"** 는 13개 probe + 적대검증으로
**반증**됐다 — L2/L3는 L1 대비 정확도 증분 ≈ 0 (multi-task robust). 그러나 여기서 멈추지 않고:

1. **원인 진단(probe_14):** L1 임베딩만으로 L2/L3를 평균 lift **0.38** 로 예측 → L2/L3는 L1의 (준)함수, **product-internal 재서술**. 정확도 redundancy는 버그가 아니라 *추출 설계의 구조적 귀결*.
2. **수정 ① 검증(probe_15):** L2/L3를 *제어 인터페이스*로 쓰면 추천을 의미축(occasion·mood·season)으로 **steered 정밀도 1.00 vs 무제어 0.14**, 개인화 100% 유지 — L1/metadata엔 없는 capability.
3. **수정 ② — de-risk 후 *반증*(probe_16 → 21·22):** LLM 외부 styling 지식은 *pair-level* 보완 랭킹에선 product-similarity를 **+12.2%** 능가(probe_16)했고, learned-fusion de-risk(3.2% coverage)에서 +17%로 보였다. **그러나 full-scale 빌드(100% coverage, $4.03, Two-Tower DSSM)에서 L1을 못 이김 — KAR 0.00424 vs L1 0.00482, −12%(0/3) NO-GO**(probe_21). probe_22가 원인을 **population-selection bias**로 격리(de-risk eligible = 인기아이템 heavy buyer라 +14% GO이나 대표 유저 −9% NO-GO, cold-start에서 가장 크게 패배).

**결론(정직):** ① **controllability(probe_15)는 진짜 기여** — L2/L3로 추천을 의미축으로 제어(steered 1.00 vs 0.14). ② **정확도 축에선 LLM 지식(L2/L3 내부 OR 외부)이 대표·cold-start 유저 discovery를 L1 대비 개선 못 함** — 단일아이템·complementarity·외부지식·full-scale 모두에서 확인된 robust negative. content(L1) Two-Tower가 popularity +80~104%로 진짜 lever. ③ **비-정확도 축도 닫힘**: diversity·coverage(probe_02 −7.7%/−2.3%)에 더해 **serendipity·novelty·long-tail-hit(R-10/probe_23, full-catalog 105K·25K user, STRONG)**에서도 L1을 못 넘음 — serendipity는 *tie at best*(matched-HR L1+L2+L3가 fair S2b서 동률, frozen-τ "−60%"는 labeling 산물=적대 audit 교정), placebo(random)가 novelty 최고지만 hit ≈0. → **추천 가치지도 전 축 negative**, enrichment의 소비자-추천 역할 종결(가치는 merchant·제어·해석에 국한). **무결성 가치 = de-risk positive(+17%)를 스스로 반증하고 population bias로 진단**한 과정 자체.

---

## 1. The Bet — 원래 가설

H&M 데이터는 **Triple-Sparsity**(32.1% 유저 희소 + 99.98% 행렬 sparse + CF 시그널 품질 저하)에 87% 단일구매·96% discovery
미션. CF만으론 구조적 한계 → **LLM 추출 다층 속성**으로 보완한다는 가설:

- **3-Layer Attribute Taxonomy** — L1(제품: material/fit/neckline) + L2(체감: mood/occasion/quality) + L3(이론: color_harmony/silhouette/lineage).
- **KAR Hybrid-Expert** — factual(L1+L2+L3) + reasoning 2-Expert 구조.
- **검증 가능한 주장:** "L2/L3 추상 속성이 L1·metadata 대비 추천 정확도·다양성·cold-start를 *증분* 향상한다."

인프라는 실제로 구축됐다: **105K 아이템 × 22 속성 추출**(GPT-4.1-nano 멀티모달, $8.50, Judge 4.86/5.0), 1.3M 유저 프로파일,
세그멘테이션, 5종 백본 + KAR 모듈. (이 자산들은 결과와 무관하게 유효 — §6.)

---

## 2. The Falsification — 가설을 스스로 반증하다

"중단된 프로젝트 점검"에서 시작해, 먼저 **싼 probe로 핵심 가정을 make-or-break** 했다(falsification-first).

| 맥락 | probe | 결과 |
|---|---|---|
| buy-similar retrieval | 01·04 | META→L1 +130.9% (강함), **L1→L2 +8% weak, L2→L3 −13.5% harm** |
| discovery re-ranker | 06·07·08 | content=standalone 검색기 ✗ / re-ranker ✓(+93%), **L1→L2 −0.8% (CI 0포함)**, L2→L3 −4.6% |
| 설계 공정성 (4 fix) | 09·10·11 | separate-encoding −1.6%, optimal layer-weight +5.5%(비유의), context −42%/+0.6% — **고쳐도 안 살아남** |
| complementarity (L3의 "본거지") | 12·13 | coordination 상관은 실재(SIG)하나, 랭킹에선 **L1_cos 0.226 ≥ L1+L3 0.224 (−0.9%, CI 0포함)** |

**적대검증(5-skeptic judge panel):** 자기 이전 보고("L2 깔끔히 생존")를 과대주장으로 **교정** — L2는 weak/regime-의존,
L3는 frozen-only drop. length-confound 반증(부호는 텍스트 길이가 아닌 content가 결정).

→ **multi-task robust redundancy 확정.** 7개 probe + 적대검증 + 2개 task fairness fix를 거쳐 "L2/L3 정확도 증분 0" 은
견고한 negative다. *대부분의 연구가 여기서 멈추거나 결과를 숨긴다.*

---

## 3. The Diagnosis — 왜 실패했나 (probe_14)

사용자: *"B(redundant) 확정이면 내 연구가 실패한 건데 — 실패의 이유와 컨셉부터 다시."*

가설: 프롬프트가 LLM을 **"the product itself / the image"** 에 한정 → L2("Casual/Minimalist")·L3("Monochromatic")는
L1("cotton/slim/crew")을 결정하는 **같은 제품**에서 추론 → 임베딩 공간에서 **by-construction redundant**.

**측정:** L1 임베딩으로 각 L2/L3 속성을 kNN(k=15)으로 예측. majority 대비 lift = L1이 메우는 gap-to-perfect 비율.

| 속성 | kNN(L1) | majority | lift | classes |
|---|---|---|---|---|
| l2_style_mood | 0.710 | 0.448 | 0.48 | 21 |
| l3_color_harmony | 0.639 | 0.336 | 0.46 | 8 |
| l3_visual_weight | 0.736 | 0.507 | 0.46 | 5 |
| l3_style_lineage | 0.513 | 0.154 | 0.42 | **44** |
| **평균(10개)** | — | — | **0.38** | — |

**진단 CONFIRMED.** L1-최근접 아이템만 보고 44-클래스 style_lineage를 0.51, 21-클래스 style_mood를 0.71로 예측.
**L2/L3는 L1의 (준)함수** — 정확도 redundancy는 *고칠 수 있는 측정 버그가 아니라 추출 방향의 구조적 귀결*이고,
인코더/아키텍처/task를 바꿔도 안 살아난 이유가 mechanism으로 설명된다. **KAR의 진짜 약속(open-world knowledge)은
한 번도 시도된 적이 없었다.**

---

## 4. The Pivot — 진단이 가리키는 수정이 작동하다

진단은 두 방향을 함의한다: (a) L2/L3 어휘를 *정확도가 아닌 제어 축*으로, (b) LLM을 *제품 묘사자가 아닌 외부지식 주입자*로.

### 4.1 Controllability (probe_15, 무료)

L2/L3는 metadata엔 없는 **8개 의미 제어축**(occasion·mood·season·quality·trendiness·versatility·coordination·visual_weight)을 노출한다.
추천을 target 속성으로 soft-steer:

| target | steered 정밀도 | 무제어 baseline | gain |
|---|---|---|---|
| occasion=Party | **1.00** | 0.02 | +0.98 |
| occasion=Work | 1.00 | 0.09 | +0.91 |
| season=Summer | 1.00 | 0.13 | +0.87 |
| mood=Minimalist | 1.00 | 0.48 | +0.52 |
| **평균** | **1.00** | **0.14** | **+0.86** |

**GO.** steered 정밀도 1.00 vs 무제어 0.14이며, steered 아이템은 "그냥 attribute-t 아이템"보다 유저에 더 가까움
(**제어 + 개인화 동시, 100% target에서 유지**). L1만으론 "Party occasion으로 추천"이 표현 불가 — **L2/L3 고유 capability**.

### 4.2 External Knowledge KAR (probe_16, ~$0.1) ★

LLM에게 *제품에 없는* 보완 styling 지식을 생성시킨다(600 seed, 새 프롬프트). 예: seed=검정 양말 →
*"pastel silk/satin slip dress, nightwear look"* (양말 묘사에선 못 끌어내는 지식). 이를 임베딩해 cross-category co-purchase 보완 랭킹:

| scorer | HR@12 | MRR |
|---|---|---|
| popularity | 0.0158 | 0.0139 |
| L1_cos_sim (제품유사도) | 0.2108 | 0.1045 |
| **external_knowledge** | **0.2366** | **0.1096** |

**GO (pair-level).** 외부 styling 지식이 product-similarity(probe_13 우승 baseline)를 **+12.2%**(delta +0.0258,
**bootstrap CI [+0.0103, +0.0413] — 0 배제**) 능가.

### 4.3 일반화 saga — frozen NO-GO → learned-fusion REVIVE (probe 17–19) ★

pair-level GO가 *실제 유저-레벨 discovery 추천*으로 옮겨가는지(=full 빌드 정당화)를 게이트. **방향이 두 번 뒤집힘:**

| probe | setup | 결과 |
|---|---|---|
| 17/17b | frozen re-ranker | external −60% vs L1 (max-sim·cross-PG robust) → **겉보기 NO-GO** |
| 18 | **learned fusion** two-tower | +ext both-side **+12.6%** vs L1 → REVIVE |
| 18b | **placebo control** (5 seed) | **C_real−C_shuffle +0.00080 (5/5)**, 용량만(shuffle)은 +0.00013 → **REAL** |
| 19 | format 통일 | NEUTRAL (장르가 원인 아님) |

**결론:** 외부지식은 pair-level AND discovery 모두에서 유효 — **단 KAR식 learned projection+augmentation 전제**. frozen NO-GO는 *대표적이지 않은 fusion*의 거짓 음성이었고, placebo가 "용량이 아닌 지식 내용"임을 확정. **이 두-번-뒤집힘 자체가 방법론적 셀링포인트** — de-risk는 대표 fusion으로, lift는 placebo로 검증해야 함. (reviewer의 fusion/format 지적이 이 saga를 직접 견인.)

**진단(제품-내부라 실패)에 대한 수정(외부지식, 올바른 fusion)이 실제 작동 = KAR open-world-knowledge 비전 실증 = 컨셉 rescue.**

**⚠️ full-scale 후속 (정직한 정정, probe 21·22):** de-risk REVIVE는 **full coverage에서 재현되지 않았다**. 외부지식을 100% coverage($4.03, 47,224 product_code → 105,542 article)로 추출하고 Two-Tower(DSSM)로 학습하니 **KAR 0.00424 vs L1 0.00482 = −12%(0/3) NO-GO**(probe_21). probe_22 isolation(coverage 고정·population 변경)이 원인을 **population-selection bias**로 격리 — de-risk eligible은 인기 3,426 아이템의 heavy buyer라 +14.1% GO이나 대표 population에선 −9.1% NO-GO. **정직한 final: LLM 외부지식은 대표·cold-start 유저의 discovery를 L1 대비 개선하지 못한다**(de-risk REVIVE는 population-편향 false positive). 무결성 가치는 이 **de-risk→scale 반증** 자체에 있다. (placebo는 capacity만 통제 — population bias는 full 빌드가 잡아냄.) canonical: `witnesses/probe_2{1,2}_result.json`.

---

## 5. What This Demonstrates — 연구 성숙도

```
현상(what)         원인(why)           수정(fix) de-risk        full-scale 검증
L2/L3 정확도 0  →  product-internal →  ① control (GO)       →  control 유지 (진짜 기여)
(13 probe·적대)    재서술 lift 0.38     ② 외부지식 de-risk +17%  ② NO-GO −12% (probe_21)
                   (probe_14)          (probe_16·18b)          원인=population bias (probe_22)
```

- 원래 가설을 **반증**하고, 그 반증을 **mechanism까지 진단**(probe_14)하고, 진단이 함의하는 두 수정을 **de-risk**했다 —
  ① control은 검증됨(GO), ② 외부지식은 de-risk에서 +17%로 *보였으나* **full-scale 빌드가 그 positive마저 반증**(NO-GO)하고
  원인을 **population-selection bias**로 격리(probe_22).
- 셀링포인트는 **falsification rigor(자기 결론의 적대 재검정) + diagnosis(왜) + de-risk→scale 자기반증 + honest negative**.
  *validated pivot이 아니라*, de-risk positive를 스스로 무너뜨리고 그 이유까지 밝힌 **연구 무결성**이 핵심.
- 정직한 scope/한계: 단일 데이터셋(H&M)·negative 결과·비-신규 방법(Two-Tower/KAR/LLM 추출) → **출판 논문은 아님**.
  강점은 *과정의 엄정성*. probe_16은 추출 캐시로 재현 시 무과금; full 추출(1b)은 $4.03.

---

## 6. Accuracy-Independent Contributions — 결과와 무관하게 유효한 자산

- **3-Layer Attribute Taxonomy** (L1/L2/L3) 제안 + 105K×22 속성 멀티모달 추출 pipeline($8.50, Judge 4.86/5.0, cov 100%).
- **Triple-Sparsity 정량화** (Gini 0.7586, 99.98% sparse, 87% 단일구매) + 세그멘테이션/affinity.
- **Hybrid repurchase-discovery 재포지셔닝(R-4):** "10× 낮은 MAP"의 원인이 데이터가 아니라 *eval setup(2개월 gap) + repurchase 폐기*임을 규명 — immediate-split repurchase MAP@12 **0.02374**(Kaggle-comparable), discovery_map(NEW-only) 지표 신설. new=95.9%/repurchase=4.1%.
- **엔지니어링 교정:** "자원 문제로 중단"의 실제 원인 = 검증 병목(3.5h/epoch) → 배치화(회귀 0); **DeepFM FM-scale 버그** 수정(logit std 24→1.9); numerical log-transform train/eval 일치; popularity-aware negative sampling.
- **Falsification harness:** 16 probe + canonical JSON + 적대검증 + 고정 seed 재현.

---

## 7. Evidence Index — canonical witnesses

| 주장 | probe | canonical JSON |
|---|---|---|
| META→L1 강함, L1→L2/L3 redundant (retrieval) | 01·04 | `probe_0{1,4}_result.json` |
| content=re-ranker(+93%), layer 분해 | 06·07·08 | `probe_0{6,7,8}_result.json` |
| 설계 공정성 4 fix (encoding·gating·context) | 09·10·11 | `probe_{09,10,11}_result.json` |
| coordination 상관 실재 / 랭킹 NO-GO | 12·13 | `probe_1{2,3}_result.json` |
| **진단: L2/L3 = L1의 함수 (lift 0.38)** | **14** | `probe_14_result.json` |
| **수정 ①: controllability 1.00 vs 0.14** | **15** | `probe_15_result.json` |
| **수정 ②: 외부지식 +12.2% (CI-유의)** | **16** | `probe_16_result.json` |

상세 기록: `redesign_2026-06.md` §1–11, contribution `contribution_notes.md` R-1~R-7.
