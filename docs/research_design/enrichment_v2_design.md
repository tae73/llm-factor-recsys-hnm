# Enrichment v2 — Interpretable Multi-Purpose Catalog Enrichment (설계 정리)

**상태:** E2-3 완료 (E2-2 value matrix + 강화 라운드 = **lift 2→3**: ③만 정당 강화·①②④ 반증, 2026-06-16). 다음 = 3 PASS cell 실서비스 + ① trend-aware predictor + 음악 교차.
**작성:** 2026-06 세션 정리. 상위 플랜: `~/.claude/plans/federated-yawning-chipmunk.md`, 구현 플랜: `~/.claude/plans/research-design-enrichment-v2-encapsulated-platypus.md`.

---

## 1. Context — 왜 이 방향인가
직전 H&M 연구: **"LLM 다층/외부 지식이 추천 *정확도*를 L1 content 대비 개선한다"** → robustly **반증**(contribution R-9, probe_21/22: full-scale −12% NO-GO, population-selection bias). 정직 기록 완료(R-9/STORY).
**사용자 확장 의도:** 가치를 *추천 정확도*에 한정 말고 — 상품의 모든 속성(제품·체감·이론)을 추출·체계화하고 catalog를 확장해 **해석 가능(interpretable)** 하고 **분석·엔지니어링·마케팅 다각도**에서 쓰는 **데이터 증강(catalog enrichment)** 으로. + KAR 확장 + 산업·학계 novelty(moderate) + **H&M(패션) + Spotify(음악)** 교차도메인.
**두 렌즈:** research_design(falsification-first) + product_design(다중 stakeholder ROI·배포·UX).

## 2. Thesis
> LLM 속성 추출의 가치는 *예측력(추천 정확도·business-outcome)이 아니라*(반증됨), metadata가 못 가진 **해석 가능한 *결정 축*** 이다 — 변별력 있고 결정에 쓰이는 facet/제어/분석 축. 이를 **multi-stakeholder value matrix**로 특성화(추천 정확도 = 정직한 negative cell), **패션·음악 교차검증**.

## 3. ★ 확정된 증거 (de-risk 결과 — 이미 도는 것, 재실행 불필요)
| probe | 결과 | 함의 |
|---|---|---|
| R-9 (probe_21/22) | 추천 정확도 full-scale −12% NO-GO, population bias | 예측 가치 없음 (기록 완료) |
| **D5** (`probe_D5_result.json`) | 비-추천 *예측*(popularity·repurchase)도 L2/L3가 metadata 못 넘음. **살아남는 건 coverage = 11 의미축** | 가치=capability, not prediction |
| **D3** (`probe_D3_result.json`) | steering: **context-aware +42~190%** / **off-target −88~99%**, precision 0.80@α0.2 | 제어는 *user-intent-driven* facet 가치 (블라인드는 파괴적) |
| **DE1** (`probe_DE1_result.json`) ★ | 20속성 중 SALVAGEABLE=2(`l1_design_details`,`l2_season_fit`). **L2/L3 12개 중 0개**. color_harmony·tone_season = metadata 색상 **lift=1.00 결정론적 재코딩(정보 0)**. occasion 81%·perceived_quality 73%·coordination_role 67% = 집중. style_lineage·trendiness·style_mood = 행동신호 0 | **기존 L2/L3 대부분 폐기**; "LLM에 metadata 보여주고 도출시키면 재코딩"의 교훈 |

## 4. ★ Enrichment 설계 — 무엇을 새로 뽑을까 (DE1 gap 로드맵)
새로 추출할 **결정-축**(기존이 못 가진 것):
1. **trend-phase / hype-cycle** (정적 Current/Classic 대신 부상-단계)
2. **price-tier / value-perception** (1-5 추측 말고 실제 $ tier)
3. **fine occasion** (work/date/gym/formal — "Everyday 81%" 분해)
4. **outfit-pairing / co-purchase role** (LLM 추측 "Foundation" 말고 **거래 데이터-기반**)
5. **body-fit / size-intent** (oversized/true-to-size/petite)
6. **care / practicality** (세탁·내구)

**4 전략 매핑(사용자 채택):** B 결정-grounded(1,2,3,5,6) + C 행동-hybrid(4, +3 calibrate) + A 정제(3) + D 외부(1 일부).

**설계 원칙 (DE1이 강제):**
- **metadata를 LLM에 안 보여준다** (재코딩 방지 — color_harmony 교훈). 이미지/순수속성에서, 또는 metadata와 *직교*하게.
- **행동-grounded 우선** (outfit-role은 LLM 아닌 co-purchase에서 도출).
- **변별력 설계** (fine occasion은 분포 균형 목표 — top1 share 낮게).
- 추출 후 **DE1 screen 재적용**(변별력·비중복·행동신호 통과만 채택).

## 5. ★ Value Matrix v2 (headline 산출물)
행=속성층/속성, 열=4 use, **각 cell 2값**: ① *capability gain*(metadata 없는 축/intent) ② *측정된 decision-lift*(행동 검증). 목표(capability vs lift)는 **미정 — 둘 다 측정 후 증거로 결정.**

| use | non-trivial 주장 | 지표 (vs metadata baseline) | risk |
|---|---|---|---|
| **① Faceted/control** | LLM facet이 *intent*를 metadata proxy보다 잘 잡음 | held-out 맥락구매 retrieval P/R; D3 context-control(+42%) | 낮음 |
| **② Trend lead-time** | 속성 momentum이 판매보다 *먼저* 포착 | attribute-share(t)→sales(t+k) lead-time vs product_type | 중간 |
| **③ Merchandising 감사** | 속성 demand/supply gap = 미충족 수요 | gap ↔ per-item sell-through 상관 | 중간 |
| **④ Marketing audience** | 속성-세그가 metadata-세그보다 행동적으로 구별 | between-segment 행동 divergence | 높음(D5: 약할 수↑) |

**핵심 방법론:** correctness를 *인간평가 대신 행동신호*로 검증("Party 태그가 실제 파티맥락서 더 팔리나") → solo 가능 + trivial("필터 가능") → non-trivial("필터가 행동을 맞춘다").

**✅ 채움 완료 (E2-2, 2026-06-16 — `witnesses/probe_E2_value_matrix.py`, seed42, API $0).** 4 통과축(trend_phase·outfit_role·value_gap·trend_gap) × 4 use = 16 cell, 각 cell **(capability, decision-lift) 분리**:
- **결과 = E2 GO: capability 14/16(dense), 측정 lift PASS는 정확히 2 cell(sparse).** 결론 = **capability는 확실, lift는 특정 결정-축 2개에 국한** — thesis("가치=예측 아닌 결정-축") 정량 확정.
- **②`trend_phase`→lead-time = PASS**(★ 핵심): attribute share(t) → category sales(**t+3개월**) r=**0.472 vs permuted-null 0.062**(Δ=0.41, CI[0.19,0.64] 0배제, 10 cats). 속성 momentum이 판매 **선행**.
- **③`outfit_role`→merch = PASS**(★): sell-through η=**0.623 vs metadata 0.564**(excess +0.059, CI[0.046,0.069], placebo 0.003).
- **①faceted = MARGINAL**(둘 다): oracle context-steer가 discovery +97%/+24%(intent 포착=capability 강함)이나 임의 제어 비용 큼(off_cost_rel 0.92/0.96) → D3-style CONDITIONAL.
- **④audience = MARGINAL**(둘 다): repurchase-divergence가 practical-margin 미달 → D5의 "L2/L3 segmentation < metadata" 재확인. gap축 ③/④=NO(behaviorally inert)·①=N/A.
- **정직성 장치:** capability/lift 분리 보고, ④ 자기참조 차단(velocity 대신 독립 repurchase_rate), ③ placebo, practical-margin(η≥0.05·Δcorr≥0.10·div≥0.10·std). 상세 = `contribution_notes.md` **E2-2** + figure `results/figures/E2_value_matrix.png`.

**✅ 강화 라운드 (E2-3, 2026-06-16 — `witnesses/probe_E2b_value_matrix.py`, 임계값 불변, seed42 재현).** 컬럼당 contribution을 정직하게 높이려 재설계(더 나은 target·granularity·outcome). **결과 lift 2/16 → 3/16(+1 genuine):**
- **③만 정당 강화:** `trend_phase`→merch가 NEW PASS — E2-2의 tautological velocity(NO)를 *launch* outcome **first_week_sell_through**로 교체(η excess **+0.45**, placebo 0.008, product_group residualize) → 두 행동축 모두 merch 통과.
- **①②④ 반증(중요):** ① 제어는 oracle-only(deployable history-predictor gain **0.0** → 배포불가); ② weekly+continuous는 noisier(monthly PASS 유지, refinement null); ④ axes는 audience-segmenter 아님(buyer-age div 0.38/0.49 < metadata 1.16; demographics=category-driven, 축은 category-직교).
- **메타 발견:** value matrix는 정직한 ceiling 근처 — genuine lift는 *축이 metadata 없는 sales/co-purchase 신호 담는 정확히 그 지점*(trend momentum→lead-time·launch-merch; co-purchase→velocity-merch = 3 cell)에만, 제어·audience엔 없다. thesis sharpen: **축 = merch/trend 결정 신호, steering knob도 audience segment도 아님.** 상세 `contribution_notes.md` **E2-3** + figure `results/figures/E2b_value_matrix.png`.

**✅ KAR user-side leg = 2-source matrix (E2-4, 2026-06-16 — `witnesses/probe_E2c_user_value.py`, seed42 byte-identical, API $0, E2/E2b canonical 불변).** value matrix를 *2-source*(item-enrichment + user/reasoning-enrichment)로 확장 — KAR 비대칭(item→Factual, user→Reasoning)을 ①control·④audience(둘 다 USER 결정)에 붙임. outcome=held-out **FUTURE**(val 2020-07~08), baseline=11 demographic features, n_cohort=40,000.

|  | ① Control | ② Lead-time | ③ Merch | ④ Audience |
|---|---|---|---|---|
| **Item-enrichment**(E2-2/3) | capability-PASS / lift-NO | ✓ trend_phase | ✓ outfit_role·trend_phase | ✗ |
| **User-enrichment**(E2-4) | **NO**(reasoning≯demo) | N/A-SEMANTICS | N/A-SEMANTICS | **modest PASS**(둘 다) |

- **④ audience = modest PASS(둘 다, power STRONG)** — 오직 `fut_top_group`(미래 카테고리믹스): `reasoning_bge` Δ**+0.0117**(p=6e-05)·`reasoning_fields` Δ**+0.0145**(p=2.4e-04), ~+1pp at the 1pp bar. `fut_price_tier`는 바 아래(Δ+0.0077/+0.0064)·습관축(online/repurchase) NULL→demo 승·④b div 혼재(fields ratio 1.973 PASS·bge 0.353 FAIL). 증분=LLM prose 귀속.
- **① control = capability-PASS / lift-NO (사용자 결정 A):** user-reasoning도 ①에서 demo 못 넘음(NO·STRONG) → E2-3 deployable gain 0.0와 합쳐 ①은 *lift 셀*로 **두 번 실패**. ①을 **capability-PASS**(D3/probe_15 steered precision **1.00 vs 무제어 0.14**, gain +0.86, metadata엔 없는 8 제어축 = 배포 가능 human-in-the-loop faceted 제어 표면)로 정직하게 닫고, **automatic-personalization lift는 아님**을 명시.
- **결론 = KAR-SYMMETRY CONFIRMED:** item→②③(merch/trend), user→④(modest audience, 미래 taste에만); ①은 어느 source로도 lift 없음=capability. 상세 `contribution_notes.md` **E2-4** + figure `results/figures/E2c_user_value.png`.

**✅ gap축 FUTURE decision-lift = CLEAN NEGATIVE (E2-5, `witnesses/probe_E2d_gap_decision.py`, seed42·`--repro` byte-identical, API $0, E2/E2b canonical 불변).** value matrix 마지막 미검증 행(gap축)을, 이전 probe가 안 한 **두 각도**로 falsify: (1) **FUTURE-held-out** outcome(val 2020-07~08, train-frozen — 신규 `enrichment_matrix.build_article_future_outcomes`), (2) **자기 구성축 one-hot 대비 incremental**(η-vs-metadata 템플릿이 아니라 baseline=one-hot(price_look)+one-hot(price_tier)+one-hot(product_group); gap=c1−c2 collinearity를 one-hot으로 우회해 directional mismatch를 testable interaction으로). 2 gap축 × 4 결정(markdown-risk·hidden-gem·overhype/sleeper·survival), 5 게이트 + 2 cohort 복제.
- **5/5 cell이 0.01 macro-F1 practical margin 미달 → PRELIM 0.** value→markdown incrΔF1 +0.0031(decision-rule lift **0.728<1**·Ridge ΔR² −0.0039)·hidden-gem −0.0081; trend→overhype +0.0087(p=0.019 통계유의이나 sub-margin, **sign-rand placebo +0.0067=mean-reversion of trend_phase**·≥20 robust −0.0022 부호반전·**raw corr +0.107→momentum partial-out −0.015**); survival 둘 다 deployable flag lift ≤1.03(trend **0.888**=극단 gap이 *덜* 생존).
- **결론: gap축은 비중복 *해석/진단* 좌표(metadata·L1 직교 = capability YES)이나 *예측·결정* 축이 아님 = 해석 axis 확정.** 미래·구성축통제·배포규칙 3중에도 자기 구성축 못 넘음 → E2-2의 "gap축 ③/④=NO(train-window)"를 **FUTURE decision-lift도 NO**로 닫음. 적대 audit 2종(false-negative·suppressed-effect) 독립 확인. 상세 `contribution_notes.md` **E2-5** + figure `results/figures/E2d_gap_decision.png`.

## 6. 교차도메인 (H&M ↔ Music)
audio-features=L1, LLM mood/occasion/era=L2/L3, 신곡=cold-start(40-50%), "workout/chill"=controllability(음악서 더 강함). 데이터: MPD(AIcrowd)+Kaggle audio features / Last.fm-360K fallback. C(특성화)의 교차복제: "capability vs lift" 패턴이 도메인 의존적인가.

## 7. De-risk 진행/남은 것
- ✅ **D3**(제어 대가), **D5**(value matrix 코어), **DE1**(기존 속성 screen).
- ✅ **E2-1 완료 (2026-06-16):** ① 6축 스키마 설계(LLM 9 + 행동파생 3 + gap 2) → ② **멀티모달 pilot 추출 500 code·5,354 art·100% cov·$0.093**(Kaggle 이미지 105,100, metadata 미노출) → ③ **DE1 re-screen = GO**(`witnesses/probe_DE1_v2_result.json`, seed42 재현). 상세는 `contribution_notes.md` Contribution **E2-1**.
  - **결과 요약:** 행동파생 `trend_phase`·`outfit_role` **SALVAGEABLE**(meta_p 0.10/0.28, l1_p 0.06/0.24, behav 0.057/0.155); `price_tier` WEAK(inert). **LLM 인식축 9/9 실패** — metadata-재코딩·집중은 고쳤으나(occasion meta_p 0.66 vs 구 1.00, top1 0.81→0.45) **여전히 L1과 redundant**(l1_p 0.71–0.85 = product-internal redescription, probe_14 생존). gap축 `value_gap`/`trend_gap` 비중복 통과(meta_p 0.35/0.13)·행동 inert→WEAK.
  - **핵심 교훈:** 비중복 결정-축은 **LLM이 관측 못하는 것**(행동 momentum·co-purchase)과 **인식×행동 gap**에서만 나온다. enrichment 방향 GO이나 가치는 LLM 이미지 인식이 아니라 behavior-grounding + gap.
- ✅ **E2-2 완료 (2026-06-16):** ④ 4-use value matrix 채움 = **E2 GO**(capability 14/16, lift PASS 2/16 — §5 참조). lift는 trend_phase→②lead-time(3mo) + outfit_role→③merch에만; ①faceted MARGINAL(oracle 상한 강함)·④audience MARGINAL(D5 재확인). `witnesses/probe_E2_result.json`, contribution **E2-2**.
- ✅ **E2-3 강화 완료 (2026-06-16):** lift **2→3**(③ `trend_phase`→merch NEW PASS via first-week sell-through). **①②④ 정직 반증**(제어 undeployable·weekly noisier·축은 audience-segmenter 아님) — §5 참조. `witnesses/probe_E2b_result.json`, contribution **E2-3**.
- ✅ **E2-4 KAR user-side leg 완료 (2026-06-16):** 2-source × 4-use 분해 = **KAR-SYMMETRY CONFIRMED**(item→②③ / user→④①). **④ audience modest PASS**(둘 다 — `fut_top_group` Δ+0.0117/+0.0145, ~+1pp), **① control NO**(둘 다, STRONG) → **① 최종 = capability-PASS / lift-NO**(사용자 결정 A) — §5·§8 참조. `witnesses/probe_E2c_user_value.json`, contribution **E2-4**.
- ✅ **E2-5 gap축 FUTURE decision-lift 완료 (CLEAN NEGATIVE):** gap축 5/5 cell이 0.01 margin 미달 — FUTURE-held-out·자기 구성축 통제·배포규칙 3중에도 자기 구성축 못 넘음 → **gap = 해석 axis(비중복) but 예측/결정 axis 아님 확정**. trend→overhype의 통계유의(p=0.019)는 mean-reversion(raw corr +0.107→partial −0.015)·비복제로 기각. `witnesses/probe_E2d_gap_decision.json`, contribution **E2-5** — §5 참조.
- ⏭ **다음:** 음악 교차도메인 복제(보류 중 D4 음악 feasibility 먼저). (**① control은 capability-PASS로 닫힘** — automatic-personalization lift는 item·user 양쪽에서 두 번 반증; **gap축 decision-lift도 E2-5서 NO로 닫힘**.)
- 보류: D1(C 예측가능성), D2(A targeted), D4(음악 feasibility) — 위 결과 후.

## 8. Novelty + 정직 calibration
- **차별점:** ① multi-stakeholder value matrix(거의 빈 영역) ② capability vs prediction 분리 특성화 ③ 정직한 recsys-accuracy negative를 한 cell로 ④ 교차도메인 ⑤ behavior-validated enrichment.
- **⚠️ 정직:** 속성 *추출*은 commodity. lift cell 다수가 marginal로 나올 가능성 큼 → 결론은 "capability는 확실, lift는 ①faceted 정도 국한" 같은 **정직한 characterization**일 수 있음(그게 novel·정직한 기여). 단일 조직 데이터 → moderate academic(RecSys/SIGIR applied·industry·연구신뢰도 포트폴리오).
- **✅ 확정 결론 (E2-2/3/4 후, 방향교정):** 위 사전 hedge가 실제로 실현됨 — 측정 lift는 정확히 **item-side 3 cell**(trend→lead-time·trend→merch·outfit→merch) + **user-side ④audience modest**(~+1pp, 미래 카테고리믹스만)에 국한. **① control = capability-PASS / lift-NO** (사용자 결정 A): 배포 가능 faceted 제어 표면은 metadata엔 없으나(capability YES, D3 precision 1.00 vs 0.14) automatic-personalization lift는 item·user 양쪽에서 반증. recsys-accuracy는 별도 negative cell — **그리고 R-10(probe_23)이 이를 비-정확도 축으로 확장: diversity·coverage(probe_02)에 더해 serendipity·novelty·long-tail-hit도 전부 negative(serendipity는 tie at best, full-catalog STRONG) → enrichment의 *소비자-추천* 역할은 전 축에서 닫힘, 가치는 merchant·해석·제어에 국한([[recsys-negative-established]]).** **gap축(value_gap/trend_gap)은 E2-5서 FUTURE decision-lift도 NO**(미래·구성축통제·배포규칙 3중에도 자기 구성축 못 넘음) → capability(metadata·L1 직교)는 있으나 forecasting lift는 없는 **해석 좌표**로 확정 — value matrix 모든 행에서 "capability는 dense, lift는 merch/trend 3 cell + thin audience에 sparse"가 일관 확인. 최종 = "**capability(제어 표면·결정-축·해석 좌표)는 확실, 측정 lift는 merch/trend + thin audience에 국한**" — capability vs prediction 분리가 본 연구의 정직한 핵심 기여.

## 9. 재사용 자산 (경로)
- 카탈로그: `data/knowledge/factual/factual_knowledge.parquet`(105K×25), `external_knowledge_full.parquet`
- 추출 인프라: `src/knowledge/{factual,reasoning,external}/`, `scripts/extract_external_knowledge.py`
- 분석/마케팅: `src/segmentation/`, `results/analysis/*`, `notebooks/0{0,3,4,5a,5b}*`
- 평가: `src/evaluation/cohorts.py`(discovery_map), `src/data/splitter.py`(immediate split), `witnesses/probe_21_gate2_external_kar.py`(Two-Tower)
- de-risk: `witnesses/probe_D{3,5}_*.py`, `probe_DE1_attribute_screen.py` + `*_result.json`
- 진단/실패: `docs/research_design/{contribution_notes(R-1..9),STORY,redesign §11-12,hm_unified_project_design §7.4.4}.md`
