# LLM Prompt Output Evaluation Methodology

## 1. Overview

This document describes the evaluation framework for LLM-generated outputs in the H&M LLM-Factor RecSys project. Two types of prompt outputs are evaluated:

1. **Factual Knowledge Extraction** — L1/L2/L3 structured attributes extracted per product
2. **User Profile Reasoning** — 9-field structured reasoning profiles generated per user

The framework uses a **dual-layer** approach:
- **Structural (Programmatic)**: Deterministic, mathematical, and counting-based checks
- **LLM-as-Judge**: Semantic quality assessment on 5 shared dimensions

## 2. Design Principles

### 2.1 Structural: What LLMs Cannot Reliably Do
Structural checks cover operations requiring exact counting, mathematical computation, schema validation, and distribution analysis. These are deterministic and reproducible.

### 2.2 LLM-as-Judge: Overcoming Keyword Matching Limitations
Previous programmatic approaches (keyword matching for metadata cross-validation, L1↔reasoning consistency) hit structural ceilings (e.g., 0.281 keyword overlap). LLM-as-Judge replaces these with semantic evaluation that understands meaning beyond exact string matches.

### 2.3 Consistency: Shared Framework Across Domains
Both domains share:
- 5 identical dimension names
- Same scoring protocol (1-5 scale)
- Common NamedTuple structures (`JudgeConfig`, `JudgeResult`, `JudgeReport`)
- Shared structural functions (`compute_coverage`, `check_token_budget`)

## 3. Structural Verification

### 3.1 Common: Coverage & Token Budget

**Coverage** (`compute_coverage`): Measures non-null percentage per attribute field.
- Applied to both factual knowledge (16+ attribute fields) and profiles (reasoning_text, reasoning_json)
- Threshold: >= 90% overall coverage

**Token Budget** (`check_token_budget`): Ensures composed text fits within encoder limits.
- BGE-base-en-v1.5 maximum: 512 tokens
- Uses whitespace-split × 1.3 subword factor as approximation
- Reports mean, median, p95, p99, max, and over-budget count

### 3.2 Factual Knowledge: Schema, Domain Rules, Distribution

**Schema Validation** (`run_schema_checks`): Validates each item against the JSON schema.
- Type checking (string, integer, array)
- Enum range validation
- Required field presence (21 LLM fields + 1 post-processed tone_season = 22 total)
- Category-specific field validation (Apparel / Footwear / Accessories)

**Domain Consistency** (`run_domain_checks`): 12 cross-attribute rules.

| # | Rule | Category | Severity |
|---|------|----------|----------|
| 1 | coordination_role × visual_weight | All | Error |
| 2 | silhouette × visual_weight | Apparel | Error |
| 3 | fit × visual_weight | Apparel | Error |
| 4 | sleeve_type × season_fit | Apparel | Error |
| 5 | neckline × sleeve_type | Apparel | Error |
| 6 | sole_type × season_fit | Footwear | Error |
| 7 | primary_function × form_factor | Accessories | Error |
| 8 | wearing_method × size_scale | Accessories | Error |
| 9 | style_mood × occasion | All | Warning |
| 10 | heel_type × occasion | Footwear | Warning |
| 11 | coordination_role × color_harmony | All | Warning |
| 12 | style_lineage × style_mood | All | Warning |

**Distribution Quality** (`compute_distributions`): Entropy and value counts for enum fields.
- High entropy = good diversity (not collapsed to single value)
- Low entropy on unexpected fields = potential systematic bias

### 3.3 Profile: Completeness, Discriminability

**Completeness** (`check_completeness`): 9-field presence + generic detection.
- Checks each reasoning_json field for non-null, non-generic content
- Generic markers: "Unknown", "N/A", "Not available", "No data", "Insufficient"
- Short text detection: total reasoning text < 100 characters

**Discriminability** (`check_discriminability`): How distinct profiles are.
- TF-IDF cosine similarity between profile texts (lower = more discriminable)
- Unique trigram count per text (higher = richer information)
- Per-field unique value ratio

## 4. LLM-as-Judge Protocol

### 4.1 Model
Default: `gpt-4.1-mini` — fast, cost-effective, structured output support.

### 4.2 Five Dimensions

| # | Dimension | Description |
|---|-----------|-------------|
| 1 | **Accuracy** | Factual correctness of extracted/generated content |
| 2 | **Specificity** | Content is specific to the item/user, not generic |
| 3 | **Coherence** | Internal consistency across fields, no contradictions |
| 4 | **Source Alignment** | Consistency with provided source data |
| 5 | **Informativeness** | Usefulness for downstream recommendation |

### 4.3 Scoring Rubric (1-5 Scale)

| Score | Label | Description |
|-------|-------|-------------|
| 1 | Very Poor | Fundamentally incorrect or useless |
| 2 | Poor | Significant issues, limited value |
| 3 | Acceptable | Mostly correct but with notable gaps |
| 4 | Good | Minor issues, generally useful |
| 5 | Excellent | Accurate, specific, and highly useful |

Each dimension produces:
- **Score**: Integer 1-5
- **Justification**: One-sentence rationale

### 4.4 Sampling Strategy
- **Factual**: Stratified by super_category (Apparel/Footwear/Accessories), default 50 samples
- **Profile**: Random sample, default 50 samples
- Seed: 42 for reproducibility

### 4.5 Cost Estimation
- gpt-4.1-mini: ~$0.40/1M input + $1.60/1M output
- 50 samples × ~2K input tokens + ~500 output tokens ≈ $0.12 per run
- With multimodal (images): ~$0.05 additional per image token overhead

## 5. Factual Knowledge Evaluation

### 5.1 Structural Checks Applied
1. **Coverage**: 16+ attribute fields non-null percentage
2. **Schema Validation**: 22 fields (21 LLM + tone_season) per category
3. **Domain Consistency**: 12 cross-attribute rules
4. **Distribution Quality**: Enum value entropy and counts
5. **Token Budget**: factual_text against 512 limit

### 5.2 Domain-Specific Dimension Interpretations

| Dimension | Factual Interpretation |
|-----------|----------------------|
| Accuracy | L1/L2/L3 attributes factually correct for this product |
| Specificity | Attributes specific to this item (not generic defaults) |
| Coherence | L1/L2/L3 form non-contradictory picture |
| Source Alignment | Matches source image + metadata (detail_desc, colour_group) |
| Informativeness | Rich design_details, useful for recommendation matching |

### 5.3 Go/No-Go Criteria

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| Coverage | overall_coverage | >= 90% |
| Schema Valid | invalid rate | <= 5% |
| Domain Errors | error rate | <= 2% |
| Token Budget | over-budget rate | <= 5% |
| Judge Overall | mean score | >= 3.5 |
| Judge Pass Rate | items scoring >= 3.5 | >= 70% |

## 6. User Profile Evaluation

### 6.1 Structural Checks Applied
1. **Coverage**: reasoning_text, reasoning_json non-null
2. **Completeness**: 9-field presence + generic detection
3. **Discriminability**: TF-IDF similarity + trigram richness
4. **Token Budget**: reasoning_text against 512 limit

### 6.2 Domain-Specific Dimension Interpretations

| Dimension | Profile Interpretation |
|-----------|----------------------|
| Accuracy | Reasoning reflects purchase data (L1/L2/L3 patterns) |
| Specificity | Reasoning specific to this user (not boilerplate) |
| Coherence | 9 fields form coherent fashion identity |
| Source Alignment | Aligns with price quintile, L1 stats, L3 distributions |
| Informativeness | Meaningful fashion preference insights for recommendation |

### 6.3 Go/No-Go Criteria

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| Completeness | overall_completeness | >= 85% |
| Generic Rate | generic profile rate | <= 10% |
| Discriminability | mean_pairwise_sim | <= 0.60 |
| Token Budget | over-budget rate | <= 5% |
| Judge Overall | mean score | >= 3.5 |
| Judge Pass Rate | profiles scoring >= 3.5 | >= 70% |

## 7. Replaced Verification → LLM-as-Judge Mapping

Previous programmatic checks that have been replaced by LLM-as-Judge dimensions:

| Previous Check | Issue | Replaced By |
|---------------|-------|-------------|
| Metadata cross-validation (material vs desc keyword matching) | Keyword matching misses semantic equivalence (e.g., "polyester blend" vs "synthetic") | **Source Alignment** |
| Manual gallery inspection | Not scalable, subjective | **Accuracy** + **Source Alignment** |
| Free-text quality (design_details keyword counting) | Counts presence, not quality | **Specificity** + **Informativeness** |
| Targeted outlier review | Manual, inconsistent | **Coherence** |
| L1↔Reasoning keyword consistency (ceiling: 0.281) | Keyword overlap cannot capture semantic agreement | **Accuracy** |
| Price quintile ↔ quality_price direction (keyword-based) | Misses nuanced price-quality relationships | **Source Alignment** |

## 8. Limitations & Future Work

### 8.1 LLM-as-Judge Limitations
- **Self-evaluation bias**: GPT-4.1-mini may be lenient on outputs from GPT-4.1-nano
- **Cost scaling**: Full population evaluation is expensive; sampling introduces variance
- **Rubric sensitivity**: Scores depend on prompt engineering quality

### 8.2 Mitigation Strategies
- Cross-model evaluation (use different judge model from extraction model)
- Confidence intervals on sampled metrics
- Periodic human-in-the-loop calibration (annotate 20 items, compare with LLM scores)

### 8.3 Future Enhancements
- Inter-annotator agreement between LLM judge and human experts
- Automated rubric refinement based on failure mode analysis
- Continuous monitoring integration (drift detection on judge scores over batches)
