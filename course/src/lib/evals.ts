// A faithful TypeScript port of phase6-engineering/eval_harness.py.
// Same golden set, same three scorers, same statistics, and the same seeded
// generator for the bootstrap, so the browser lab prints the numbers the Python prints.
// The system under test is the real mini-RAG port in ./rag (not a mock).
import { makeRng } from './rng'
import { CORPUS, ask, ingest, type Answer, type Retrieved } from './rag'

/* ---------- 1. the golden set ---------- */
export type Category = 'direct' | 'paraphrase' | 'unanswerable' | 'adversarial'
export const CATEGORIES: Category[] = ['direct', 'paraphrase', 'unanswerable', 'adversarial']

export interface GoldenItem {
  id: string
  cat: Category
  q: string
  /** The reference answer as a person would write it, or null when the only right behaviour is to refuse. */
  expected: string | null
  /** The document that holds the answer. */
  source: string | null
  /** A fragment of the sentence that holds the answer: lets us score retrieval separately. */
  gold: string | null
}

const item = (id: string, cat: Category, q: string, expected: string | null = null, source: string | null = null, gold: string | null = null): GoldenItem => ({ id, cat, q, expected, source, gold })

export const GOLDEN_SET: GoldenItem[] = [
  item('d1', 'direct', 'how quickly must I acknowledge pages', 'within 5 minutes', 'oncall.md', 'Primary oncall must acknowledge pages'),
  item('d2', 'direct', 'what is the budget for learning materials', '500 dollars per year', 'expenses.md', 'Engineers may expense up to 500 dollars'),
  item('d3', 'direct', 'who writes the postmortem after an incident', 'the oncall engineer', 'oncall.md', 'the oncall engineer writes the postmortem'),
  item('d4', 'direct', 'when do deploy freezes apply', 'the last week of each quarter', 'deploy-policy.md', 'Deploy freezes apply'),
  item('d5', 'direct', 'how many approvals does every deploy require', '2 approvals', 'deploy-policy.md', 'Every deploy requires two approvals'),
  item('d6', 'direct', 'who assigns the onboarding buddy', 'the team lead', 'onboarding.md', 'The onboarding buddy is assigned'),
  item('d7', 'direct', 'when must receipts be submitted', 'within 30 days', 'expenses.md', 'Receipts must be submitted'),
  item('d8', 'direct', 'how are rollbacks triggered', 'from the deploy dashboard', 'deploy-policy.md', 'Rollbacks are triggered'),
  item('p1', 'paraphrase', 'how long do I have to complete security training', 'two weeks', 'onboarding.md', 'All new hires must complete security training'),
  item('p2', 'paraphrase', 'where can I find the engineering handbook', 'in the internal wiki', 'onboarding.md', 'The engineering handbook lives'),
  item('p3', 'paraphrase', 'are manual deploys allowed', 'forbidden except during a declared incident', 'deploy-policy.md', 'Manual deploys are forbidden'),
  item('p4', 'paraphrase', 'who gets paged when the primary is silent', 'the secondary oncall', 'oncall.md', 'Secondary oncall is paged'),
  item('p5', 'paraphrase', 'how fast do I have to respond when I get paged', 'within 5 minutes', 'oncall.md', 'Primary oncall must acknowledge pages'),
  item('p6', 'paraphrase', 'how much money can I spend on books and courses', '500 dollars per year', 'expenses.md', 'Engineers may expense up to 500 dollars'),
  item('p7', 'paraphrase', 'can I push a release by hand', 'forbidden except during a declared incident', 'deploy-policy.md', 'Manual deploys are forbidden'),
  item('p8', 'paraphrase', 'what do I need before I get production access', 'the incident response course', 'onboarding.md', 'Production access requires'),
  item('u1', 'unanswerable', 'what is the wifi password'),
  item('u2', 'unanswerable', 'how many vacation days do I get'),
  item('u3', 'unanswerable', 'what is the salary of a staff engineer'),
  item('u4', 'unanswerable', 'what is the parental leave policy'),
  item('a1', 'adversarial', 'who is paged if the secondary does not respond'),
  item('a2', 'adversarial', 'how many approvals does a rollback require'),
  item('a3', 'adversarial', 'what is the budget for conference travel'),
  item('a4', 'adversarial', 'does the oncall rotation change every friday', 'no, every Monday', 'oncall.md', 'The oncall rotation changes'),
]

/** A person's verdict on each answer of the DEFAULT configuration (see the Python file). */
export const HUMAN_LABELS_DEFAULT: Record<string, boolean> = {
  d1: true, d2: true, d3: true, d4: true, d5: true, d6: true, d7: true, d8: true,
  p1: true, p2: true, p3: true, p4: true, p5: false, p6: false, p7: false, p8: false,
  u1: true, u2: false, u3: true, u4: true,
  a1: false, a2: true, a3: false, a4: true,
}

/* ---------- 2. the task: the system under test behind one function ---------- */
export interface EvalConfig { sentencesPerChunk: number; overlap: number; k: number; threshold: number }
export const DEFAULT_CONFIG: EvalConfig = { sentencesPerChunk: 2, overlap: 1, k: 3, threshold: 0.35 }

export interface TaskOutput { retrieved: Retrieved; answer: Answer; refused: boolean }

/* ---------- 3. scorers ---------- */
export type ScorerName = 'exact' | 'substring' | 'rubric'
export type Verdict = { passed: boolean; reason: string }

const plain = (text: string) => text.toLowerCase().replace(/[^a-z0-9 ]/g, '').trim()

const refusalVerdict = (out: TaskOutput): Verdict =>
  out.refused ? { passed: true, reason: 'refused' } : { passed: false, reason: 'answered a question that has no answer' }

export const scoreExact = (it: GoldenItem, out: TaskOutput): Verdict => {
  if (it.expected === null) return refusalVerdict(out)
  if (out.refused) return { passed: false, reason: 'refused an answerable question' }
  const ok = plain(out.answer.sentence!) === plain(it.expected)
  return { passed: ok, reason: ok ? 'identical to the reference' : 'a full sentence is never identical to a short reference' }
}

export const scoreSubstring = (it: GoldenItem, out: TaskOutput): Verdict => {
  if (it.expected === null) return refusalVerdict(out)
  if (out.refused) return { passed: false, reason: 'refused an answerable question' }
  const ok = plain(out.answer.sentence!).includes(plain(it.expected))
  return { passed: ok, reason: ok ? 'reference found in the answer' : `the string “${it.expected}” is not in the answer` }
}

const NUMBER_WORDS: Record<string, string> = { one: '1', two: '2', three: '3', four: '4', five: '5', six: '6', seven: '7', eight: '8', nine: '9', ten: '10', thirty: '30' }
const FILLER = new Set(['the', 'a', 'an', 'in', 'of', 'to', 'is', 'are'])
export const facts = (text: string): string[] =>
  plain(text).split(' ').filter(Boolean).map((w) => NUMBER_WORDS[w] ?? w).filter((w) => !FILLER.has(w))

/** The rubric judge: a deterministic stand-in for an LLM judge. Three checks: behaviour, correct, grounded. */
export const scoreRubric = (it: GoldenItem, out: TaskOutput): Verdict => {
  if (it.expected === null) return out.refused ? { passed: true, reason: 'behaviour ok: refused' } : { passed: false, reason: 'behaviour: answered a question the corpus cannot answer' }
  if (out.refused) return { passed: false, reason: 'behaviour: refused although the corpus holds the answer' }
  const have = new Set(facts(out.answer.sentence!))
  const missing = facts(it.expected).filter((w) => !have.has(w))
  if (missing.length) return { passed: false, reason: `correct: the answer does not state “${missing.join(' ')}”` }
  if (out.answer.cite!.source !== it.source) return { passed: false, reason: `grounded: cites ${out.answer.cite!.source}, the fact lives in ${it.source}` }
  return { passed: true, reason: 'behaviour, correct and grounded all ok' }
}

export const SCORERS: Record<ScorerName, (it: GoldenItem, out: TaskOutput) => Verdict> = { exact: scoreExact, substring: scoreSubstring, rubric: scoreRubric }

/** Component metric: is the sentence holding the answer in the top-k chunks? null when the item has no answer. */
export const retrievalHit = (it: GoldenItem, out: TaskOutput): boolean | null =>
  it.gold === null ? null : out.retrieved.some((r) => r.chunk.text.includes(it.gold!))

export type Diagnosis = 'ok' | 'should-refuse' | 'retrieval-miss' | 'refused-with-chunk' | 'scorer' | 'wrong-sentence'
export const DIAGNOSIS_TEXT: Record<Diagnosis, string> = {
  ok: 'ok',
  'should-refuse': 'answerer: should have refused',
  'retrieval-miss': 'retrieval: the right chunk was not in the top-k',
  'refused-with-chunk': 'answerer: had the right chunk, refused',
  scorer: 'scorer: quoted the right sentence, but it is worded differently from the reference',
  'wrong-sentence': 'answerer: had the right chunk, quoted the wrong sentence',
}

export const diagnose = (it: GoldenItem, out: TaskOutput, passed: boolean): Diagnosis => {
  if (passed) return 'ok'
  if (it.expected === null) return 'should-refuse'
  if (!retrievalHit(it, out)) return 'retrieval-miss'
  if (out.refused) return 'refused-with-chunk'
  if (out.answer.sentence!.includes(it.gold!)) return 'scorer'
  return 'wrong-sentence'
}

/* ---------- 4. run and aggregate ---------- */
export interface EvalRow { item: GoldenItem; out: TaskOutput; passed: boolean; reason: string; hit: boolean | null; diagnosis: Diagnosis }

export const evaluate = (config: EvalConfig = DEFAULT_CONFIG, scorer: ScorerName = 'rubric', items: GoldenItem[] = GOLDEN_SET): EvalRow[] => {
  const index = ingest(CORPUS, config.sentencesPerChunk, config.overlap)
  return items.map((it) => {
    const res = ask(index, it.q, config.k, config.threshold)
    const out: TaskOutput = { retrieved: res.retrieved, answer: res.answer, refused: !res.answer.found }
    const v = SCORERS[scorer](it, out)
    return { item: it, out, passed: v.passed, reason: v.reason, hit: retrievalHit(it, out), diagnosis: diagnose(it, out, v.passed) }
  })
}

export const accuracy = (rows: { passed: boolean }[]): number => rows.filter((r) => r.passed).length / rows.length

export const byCategory = (rows: EvalRow[]): Record<Category, { passed: number; total: number }> => {
  const table = {} as Record<Category, { passed: number; total: number }>
  for (const cat of CATEGORIES) {
    const sub = rows.filter((r) => r.item.cat === cat)
    table[cat] = { passed: sub.filter((r) => r.passed).length, total: sub.length }
  }
  return table
}

export const recallAtK = (rows: EvalRow[]): { hits: number; total: number } => {
  const answerable = rows.filter((r) => r.hit !== null)
  return { hits: answerable.filter((r) => r.hit).length, total: answerable.length }
}

/* ---------- 5. statistics ---------- */
export const Z95 = 1.959964 // two-sided 95%
export const Z80 = 0.841621 // one-sided 80% (the usual power target)

/** Standard error of a proportion: sqrt(p(1-p)/n). */
export const standardError = (p: number, n: number): number => Math.sqrt((p * (1 - p)) / n)

export const normalInterval = (p: number, n: number, z = Z95): [number, number] => {
  const half = z * standardError(p, n)
  return [Math.max(0, p - half), Math.min(1, p + half)]
}

/** Wilson score interval: sensible for small n and for p near 0 or 1. */
export const wilsonInterval = (p: number, n: number, z = Z95): [number, number] => {
  const denom = 1 + (z * z) / n
  const centre = (p + (z * z) / (2 * n)) / denom
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom
  return [Math.max(0, centre - half), Math.min(1, centre + half)]
}

/** Percentile bootstrap for a mean: resample the items with replacement, recompute, take the middle 95%. */
export const bootstrapInterval = (values: number[], resamples = 10000, seed = 15): [number, number] => {
  const n = values.length
  const rng = makeRng(seed)
  const means = new Float64Array(resamples)
  for (let b = 0; b < resamples; b++) {
    let total = 0
    for (let i = 0; i < n; i++) total += values[Math.floor(rng.next() * n)]
    means[b] = total / n
  }
  means.sort()
  return [means[Math.floor(0.025 * resamples)], means[Math.min(resamples - 1, Math.floor(0.975 * resamples))]]
}

const choose = (n: number, k: number): number => {
  let c = 1
  for (let i = 1; i <= k; i++) c = (c * (n - k + i)) / i
  return Math.round(c)
}

/** Exact two-sided sign test (exact McNemar). b = items only A passes, c = items only B passes. */
export const signTest = (b: number, c: number): number => {
  const n = b + c
  if (n === 0) return 1
  let tail = 0
  for (let i = 0; i <= Math.min(b, c); i++) tail += choose(n, i)
  return Math.min(1, (2 * tail) / 2 ** n)
}

export interface Comparison { accA: number; accB: number; diff: number; onlyA: number; onlyB: number; p: number; ci: [number, number]; verdict: 'A is better' | 'B is better' | 'cannot tell' }

/** Paired comparison: the same items through both systems, judged on per-item differences. */
export const compare = (a: { passed: boolean }[], b: { passed: boolean }[], resamples = 10000, seed = 15): Comparison => {
  const diffs = a.map((r, i) => Number(b[i].passed) - Number(r.passed))
  const onlyA = diffs.filter((d) => d < 0).length
  const onlyB = diffs.filter((d) => d > 0).length
  const p = signTest(onlyA, onlyB)
  return {
    accA: accuracy(a), accB: accuracy(b), diff: diffs.reduce((s, d) => s + d, 0) / diffs.length,
    onlyA, onlyB, p, ci: bootstrapInterval(diffs, resamples, seed),
    verdict: p < 0.05 ? (onlyB > onlyA ? 'B is better' : 'A is better') : 'cannot tell',
  }
}

/** Cohen's kappa: agreement beyond chance. (p_observed - p_chance) / (1 - p_chance). */
export const cohensKappa = (x: boolean[], y: boolean[]): number => {
  const n = x.length
  const pObs = x.filter((v, i) => v === y[i]).length / n
  const px = x.filter(Boolean).length / n
  const py = y.filter(Boolean).length / n
  const pChance = px * py + (1 - px) * (1 - py)
  return pChance === 1 ? 1 : (pObs - pChance) / (1 - pChance)
}

/** Unbiased pass@k (Chen et al., 2021): 1 - C(n-c, k) / C(n, k), as a running product. */
export const passAtK = (n: number, c: number, k: number): number => {
  if (n - c < k) return 1
  let prod = 1
  for (let i = n - c + 1; i <= n; i++) prod *= 1 - k / i
  return 1 - prod
}

/* ---------- 6. planning: how many items do I need? ---------- */

/**
 * Two INDEPENDENT eval sets of n items each (unpaired), accuracies p1 and p2.
 * Items per set for a two-sided 5% test to detect the gap 80% of the time.
 */
export const nUnpaired = (p1: number, p2: number, zAlpha = Z95, zBeta = Z80): number => {
  const d = Math.abs(p2 - p1)
  if (d === 0) return Infinity
  const pBar = (p1 + p2) / 2
  const num = zAlpha * Math.sqrt(2 * pBar * (1 - pBar)) + zBeta * Math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
  return Math.ceil((num * num) / (d * d))
}

/**
 * The SAME n items through both systems (paired, McNemar). `discordant` is the share of items on which
 * the two systems disagree; it can never be smaller than the gap d. Fewer disagreements = less noise.
 */
export const nPaired = (d: number, discordant: number, zAlpha = Z95, zBeta = Z80): number => {
  const gap = Math.abs(d)
  if (gap === 0) return Infinity
  const psi = Math.max(discordant, gap)
  const num = zAlpha * Math.sqrt(psi) + zBeta * Math.sqrt(Math.max(0, psi - gap * gap))
  return Math.ceil((num * num) / (gap * gap))
}

/** Disagreement rate if the two systems' errors were unrelated: the worst realistic case for pairing. */
export const independentDiscordance = (p1: number, p2: number): number => p1 * (1 - p2) + p2 * (1 - p1)

/** Monte Carlo check of nPaired: how often does a two-sided 5% McNemar z-test detect the gap? */
export const simulatePairedPower = (n: number, d: number, discordant: number, trials = 2000, seed = 7): number => {
  const rng = makeRng(seed)
  const pB = (discordant + d) / 2 // only B passes
  const pA = (discordant - d) / 2 // only A passes
  let detected = 0
  for (let t = 0; t < trials; t++) {
    let b = 0
    let a = 0
    for (let i = 0; i < n; i++) {
      const u = rng.next()
      if (u < pB) b++
      else if (u < pB + pA) a++
    }
    if (a + b > 0 && Math.abs(b - a) / Math.sqrt(a + b) > Z95) detected++
  }
  return detected / trials
}

/** Monte Carlo check of nUnpaired: two independent samples, pooled two-proportion z-test. */
export const simulateUnpairedPower = (n: number, p1: number, p2: number, trials = 2000, seed = 7): number => {
  const rng = makeRng(seed)
  let detected = 0
  for (let t = 0; t < trials; t++) {
    let x1 = 0
    let x2 = 0
    for (let i = 0; i < n; i++) { if (rng.next() < p1) x1++; if (rng.next() < p2) x2++ }
    const pool = (x1 + x2) / (2 * n)
    const se = Math.sqrt((2 * pool * (1 - pool)) / n)
    if (se > 0 && Math.abs(x2 - x1) / n / se > Z95) detected++
  }
  return detected / trials
}

/** What repeated evals would report: `runs` measured accuracies, each from n fresh items, true accuracy p. */
export const simulateScores = (p: number, n: number, runs: number, seed = 1): number[] => {
  const rng = makeRng(seed)
  return Array.from({ length: runs }, () => {
    let pass = 0
    for (let i = 0; i < n; i++) if (rng.next() < p) pass++
    return pass / n
  })
}
