export const meta = {
  name: 'gate0-adversarial-review',
  description: 'Adversarially refute the Gate-0 make-or-break conclusion (L2 helps, L3 drops) via diverse skeptic lenses, then synthesize honest scoping.',
  phases: [
    { title: 'Refute', detail: 'one skeptic per refutation lens, refute-by-default' },
    { title: 'Synthesize', detail: 'per-claim disposition + honest scoping' },
  ],
}

const ARTIFACTS = `Read these files in the repo before judging:
- witnesses/probe_01_result.json  (incremental ladder META->L1->L1+L2->+L3, HR@12 + bootstrap CIs)
- witnesses/probe_02_result.json  (L3 isolation + diversity/coverage C4/C5)
- witnesses/probe_04_result.json  (robustness: multi-seed x k x stratum x maxsim + text-length confound)
- witnesses/probe_01_incremental_layer_value.py and witnesses/_probe_common.py  (the method)
The probes use frozen BGE-base text embeddings + content-based centroid-kNN retrieval (train-history centroid -> cosine over 105K catalog -> top-12), evaluated on held-out val purchases, per-user paired bootstrap.`

const CLAIMS = `The three claims under adversarial test:
- CLAIM A (C1): LLM attribute text (L1) gives LARGE incremental retrieval value over raw H&M metadata (~+130% HR@12, CI excludes 0).
- CLAIM B (C2): L2 (perceptual) gives incremental value over L1 (~+7.8% HR@12; note the bootstrap CI lower bound is very close to 0 — adversarially relevant).
- CLAIM C (C3+C4): L3 (theory) should be DROPPED for content retrieval — it hurts accuracy (~-13.5%) and diversity/coverage.
Also assess CAVEAT D: frozen-BGE content-retrieval is a PROXY; the in-model trainable KAR (Gate-1/2) is the final arbiter for L3.`

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'A_metadata_vs_llm', 'B_l2_incremental', 'C_drop_l3', 'strongest_objection', 'what_would_change_my_mind'],
  properties: {
    lens: { type: 'string' },
    A_metadata_vs_llm: {
      type: 'object', additionalProperties: false, required: ['verdict', 'objection'],
      properties: { verdict: { type: 'string', enum: ['holds', 'refuted', 'uncertain'] }, objection: { type: 'string' } },
    },
    B_l2_incremental: {
      type: 'object', additionalProperties: false, required: ['verdict', 'objection'],
      properties: { verdict: { type: 'string', enum: ['holds', 'refuted', 'uncertain'] }, objection: { type: 'string' } },
    },
    C_drop_l3: {
      type: 'object', additionalProperties: false, required: ['verdict', 'objection'],
      properties: { verdict: { type: 'string', enum: ['holds', 'refuted', 'uncertain'] }, objection: { type: 'string' } },
    },
    strongest_objection: { type: 'string' },
    what_would_change_my_mind: { type: 'string' },
  },
}

const LENSES = [
  { key: 'statistician', prompt: `You are a hostile STATISTICIAN. Attack the inference: are the paired bootstrap CIs valid? Is CLAIM B (C2 ~+7.8%) actually significant, given its CI lower bound is near 0 — is it knife-edge / multiple-comparison-inflated / driven by a few users? Check probe_04 multi-seed/k/stratum cells for B and C: do signs flip anywhere? Is the effect-size practically meaningful vs the META->L1 jump? Refute by default.` },
  { key: 'confound-hunter', prompt: `You are a CONFOUND HUNTER. Propose non-semantic explanations for the deltas: text length (does adding text mechanically shift BGE embeddings?), product-name vocabulary leakage into the metadata/L1 text, BGE tokenization/normalization artifacts. Check probe_04's text_lengths diagnostic: does L2 add text yet help (refuting pure length)? Could L3's harm be length-driven? Decide if confounds are actually ruled out. Refute by default.` },
  { key: 'construct-validity', prompt: `You are a CONSTRUCT-VALIDITY critic. Is content-based centroid-kNN over frozen BGE a valid proxy for the actual recommendation task? The project is discovery-oriented (87% single-purchase). Is HR@12 against next-window purchases measuring "buy-similar" rather than discovery — and could that bias AGAINST L2/L3 (abstract style) and FOR L1 (concrete product)? Does this threaten CLAIM C specifically? Refute by default.` },
  { key: 'l3-in-model-advocate', prompt: `You are an ADVOCATE that L3 may still help IN-MODEL. Argue that a trainable KAR Expert + gating can extract or down-weight L3 even if frozen-BGE retrieval hurts; "drop L3" from a frozen probe may be premature before Gate-1. Assess how strongly probe evidence licenses dropping L3 now vs keeping it as a pre-registered Gate-2 falsification target. Refute CLAIM C by default; be fair to A and B.` },
  { key: 'alternative-conclusion', prompt: `You are an ALTERNATIVE-CONCLUSION skeptic. Could the data support a different reading: "L1 dominates; L2's gain is within noise; L3 just needs a different encoder/composition not deletion"? Is the "2-layer (L1+L2)" reframe the best-supported reading, or an overreach? Stress-test whether B and C are the right conclusions vs alternatives. Refute by default.` },
]

phase('Refute')
const verdicts = (await parallel(LENSES.map(L => () =>
  agent(
    `${ARTIFACTS}\n\n${CLAIMS}\n\nYOUR LENS:\n${L.prompt}\n\nGround every objection in the actual numbers in the JSON files (quote them). A claim only 'refuted' if you have a concrete, evidence-backed objection — not a vibe. Return the structured verdict.`,
    { label: `refute:${L.key}`, phase: 'Refute', schema: VERDICT_SCHEMA }
  ).then(v => v ? { ...v, lens: L.key } : null)
))).filter(Boolean)

function tally(claimKey) {
  const vs = verdicts.map(v => v[claimKey]?.verdict)
  return {
    refuted: vs.filter(x => x === 'refuted').length,
    uncertain: vs.filter(x => x === 'uncertain').length,
    holds: vs.filter(x => x === 'holds').length,
    objections: verdicts.map(v => ({ lens: v.lens, verdict: v[claimKey]?.verdict, objection: v[claimKey]?.objection })),
  }
}
const tallies = { A: tally('A_metadata_vs_llm'), B: tally('B_l2_incremental'), C: tally('C_drop_l3') }

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['per_claim', 'honest_scoping', 'recommended_reframe', 'residual_risks'],
  properties: {
    per_claim: {
      type: 'object', additionalProperties: false, required: ['A', 'B', 'C'],
      properties: {
        A: { type: 'object', additionalProperties: false, required: ['disposition', 'rationale'], properties: { disposition: { type: 'string', enum: ['holds', 'holds_with_caveat', 'refuted'] }, rationale: { type: 'string' } } },
        B: { type: 'object', additionalProperties: false, required: ['disposition', 'rationale'], properties: { disposition: { type: 'string', enum: ['holds', 'holds_with_caveat', 'refuted'] }, rationale: { type: 'string' } } },
        C: { type: 'object', additionalProperties: false, required: ['disposition', 'rationale'], properties: { disposition: { type: 'string', enum: ['holds', 'holds_with_caveat', 'refuted'] }, rationale: { type: 'string' } } },
      },
    },
    honest_scoping: { type: 'string' },
    recommended_reframe: { type: 'string' },
    residual_risks: { type: 'array', items: { type: 'string' } },
  },
}

const synthesis = await agent(
  `${ARTIFACTS}\n\n${CLAIMS}\n\nFive adversarial skeptics returned verdicts. Tally (refuted/uncertain/holds counts + objections):\n${JSON.stringify(tallies, null, 2)}\n\nFull verdicts:\n${JSON.stringify(verdicts, null, 2)}\n\nAdjudicate each claim. A claim 'holds' if skeptics could not land an evidence-backed refutation; 'holds_with_caveat' if a legitimate scoping limit exists (e.g., proxy objective); 'refuted' if a majority landed a concrete objection. Then write: (1) honest_scoping — the precise statement of what the probes do and do NOT establish (esp. frozen-retrieval-proxy vs in-model for L3); (2) recommended_reframe — the contribution framing the evidence supports; (3) residual_risks the Gate-1/2 in-model test must resolve. Be conservative and honest.`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return { tallies, verdicts, synthesis }
