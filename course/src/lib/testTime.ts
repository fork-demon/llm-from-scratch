// Test-time compute, as a toy you can reason about exactly.
//
// THE TOY MODEL (a simplification, stated in the UI too):
//   - One "attempt" at a problem is correct with probability p.
//   - Otherwise it lands, uniformly at random, on one of m distinct wrong answers ("distractors").
//     So each distractor is produced with probability (1 - p) / m: the "distractor rate".
//   - Attempts are independent of each other.
//
// Three ways to spend N attempts:
//   1. single shot:     ignore N, accuracy = p.
//   2. majority vote:   return the most common answer. Ties are broken uniformly at random
//                       among the tied answers. (Self-consistency.)
//   3. best-of-N with a verifier of accuracy a:
//                       the verifier looks at each attempt independently. It ACCEPTS a correct
//                       attempt with probability a and ACCEPTS a wrong attempt with probability 1 - a.
//                       We return one attempt chosen uniformly among the accepted ones; if nothing
//                       was accepted we return one chosen uniformly among all N.
//                       a = 1 is a perfect checker (unit tests), a = 0.5 is a coin flip.
//
// Everything has a closed form or an exact dynamic programme, and a seeded Monte-Carlo
// simulation of the literal procedure is provided so the two can be checked against each other.
import type { Rng } from './rng'

export interface TestTimeParams {
  p: number // probability one attempt is correct, 0..1
  m: number // number of distinct wrong answers, >= 1
  a: number // verifier accuracy, 0..1
}

/** How often each individual wrong answer shows up. */
export const distractorRate = (p: number, m: number): number => (1 - p) / m

/** Majority vote converges to the right answer as N grows only if it is the single most likely answer. */
export const majorityCanWin = (p: number, m: number): boolean => p > distractorRate(p, m) + 1e-12

const factorials = (n: number): number[] => {
  const f = [1]
  for (let i = 1; i <= n; i++) f.push(f[i - 1] * i)
  return f
}

const binomPmf = (n: number, k: number, p: number, fact: number[]): number =>
  (fact[n] / (fact[k] * fact[n - k])) * p ** k * (1 - p) ** (n - k)

/**
 * P(correct answer wins | it received c votes and the other n votes fell uniformly on m distractors).
 * Exact. DP over distractors: state = (votes used so far, how many distractors tie with c).
 * Any distractor with more than c votes means we lose, so each distractor may take 0..c votes.
 * We count arrangements through the exponential generating function  prod_j  x^j / j!  and multiply
 * by n! / m^n at the end to turn the count into a probability.
 */
export const winGivenVotes = (c: number, n: number, m: number, fact = factorials(Math.max(c, n))): number => {
  if (n === 0) return 1
  if (c === 0) return 0 // at least one distractor has a vote, the correct answer has none
  // dp[u][t]
  let dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0))
  dp[0][0] = 1
  for (let bin = 0; bin < m; bin++) {
    const next: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0))
    for (let u = 0; u <= n; u++) {
      for (let t = 0; t <= bin; t++) {
        const v = dp[u][t]
        if (v === 0) continue
        const maxJ = Math.min(c, n - u)
        for (let j = 0; j <= maxJ; j++) next[u + j][t + (j === c ? 1 : 0)] += v / fact[j]
      }
    }
    dp = next
  }
  const scale = fact[n] / m ** n
  let win = 0
  for (let t = 0; t <= m; t++) win += (dp[n][t] * scale) / (t + 1) // t-way tie with the correct answer: win 1/(t+1)
  return win
}

/** Exact accuracy of majority vote (plurality, ties broken uniformly at random) over N attempts. */
export const majorityVoteAccuracy = (p: number, m: number, N: number): number => {
  if (N <= 0) return 0
  if (p >= 1) return 1
  if (p <= 0) return 0
  const fact = factorials(N)
  let acc = 0
  for (let c = 0; c <= N; c++) acc += binomPmf(N, c, p, fact) * winGivenVotes(c, N - c, m, fact)
  return Math.min(1, Math.max(0, acc))
}

/** P(the verifier accepts a random attempt). */
export const acceptRate = (p: number, a: number): number => p * a + (1 - p) * (1 - a)

/** P(an attempt is correct | the verifier accepted it). This is the ceiling best-of-N approaches as N grows. */
export const precisionWhenAccepted = (p: number, a: number): number => {
  const r = acceptRate(p, a)
  return r === 0 ? p : (p * a) / r
}

/**
 * Exact accuracy of best-of-N with the verifier described at the top of this file.
 *   P = P(something accepted) * P(correct | accepted) + P(nothing accepted) * P(correct | rejected)
 * With a = 1 this collapses to 1 - (1 - p)^N. With a = 0.5 it collapses to p.
 */
export const bestOfNAccuracy = (p: number, a: number, N: number): number => {
  if (N <= 0) return 0
  const r = acceptRate(p, a)
  const noneAccepted = (1 - r) ** N
  const correctGivenRejected = r >= 1 ? 0 : (p * (1 - a)) / (1 - r)
  return (1 - noneAccepted) * precisionWhenAccepted(p, a) + noneAccepted * correctGivenRejected
}

export interface Accuracies { single: number; majority: number; bestOfN: number }

export const analytic = ({ p, m, a }: TestTimeParams, N: number): Accuracies => ({
  single: p,
  majority: majorityVoteAccuracy(p, m, N),
  bestOfN: bestOfNAccuracy(p, a, N),
})

/** Accuracy of each strategy for N = 1..maxN. */
export const curves = (params: TestTimeParams, maxN: number): Accuracies[] =>
  Array.from({ length: maxN }, (_, i) => analytic(params, i + 1))

/** One set of N attempts. Answer 0 is the correct one; 1..m are the distractors. */
export const drawAttempts = ({ p, m }: TestTimeParams, N: number, rng: Rng): number[] =>
  Array.from({ length: N }, () => (rng.next() < p ? 0 : 1 + rng.int(m)))

/** Plurality winner with uniform random tie-breaking. */
export const majorityPick = (answers: number[], m: number, rng: Rng): number => {
  const counts = new Array(m + 1).fill(0)
  for (const x of answers) counts[x]++
  const best = Math.max(...counts)
  const tied = counts.map((c, i) => (c === best ? i : -1)).filter((i) => i >= 0)
  return tied[rng.int(tied.length)]
}

/** The verifier's verdict on each attempt: true = accepted. */
export const verify = (answers: number[], a: number, rng: Rng): boolean[] =>
  answers.map((x) => (x === 0 ? rng.next() < a : rng.next() < 1 - a))

export const verifierPick = (answers: number[], accepted: boolean[], rng: Rng): number => {
  const pool = answers.filter((_, i) => accepted[i])
  const from = pool.length ? pool : answers
  return from[rng.int(from.length)]
}

/** Monte-Carlo: literally run the three procedures `trials` times and count how often each is right. */
export const simulate = (params: TestTimeParams, N: number, trials: number, rng: Rng): Accuracies => {
  let single = 0
  let majority = 0
  let best = 0
  for (let t = 0; t < trials; t++) {
    const answers = drawAttempts(params, N, rng)
    if (answers[0] === 0) single++
    if (majorityPick(answers, params.m, rng) === 0) majority++
    if (verifierPick(answers, verify(answers, params.a, rng), rng) === 0) best++
  }
  return { single: single / trials, majority: majority / trials, bestOfN: best / trials }
}

/** Cost is linear: every attempt is a full generation. (Verifier cost is ignored here and said so in the UI.) */
export const tokensSpent = (N: number, tokensPerAttempt: number): number => N * tokensPerAttempt

/* ---------- the "direct vs with steps" puzzle ---------- */
export type Op = { kind: 'times' | 'plus' | 'minus'; by: number }

export interface TraceStep { text: string; before: number; after: number }

const OP_SIGN = { times: '×', plus: '+', minus: '−' } as const

export const applyOp = (x: number, op: Op): number =>
  op.kind === 'times' ? x * op.by : op.kind === 'plus' ? x + op.by : x - op.by

/** Write the puzzle out one operation per line, the way a reasoning trace would. */
export const buildTrace = (start: number, ops: Op[]): TraceStep[] => {
  let x = start
  return ops.map((op) => {
    const after = applyOp(x, op)
    const step = { text: `${x} ${OP_SIGN[op.kind]} ${op.by} = ${after}`, before: x, after }
    x = after
    return step
  })
}

/** Rough token count: one token per whitespace-separated piece. Good enough to compare 1 line with 5. */
export const roughTokens = (s: string): number => s.trim().split(/\s+/).filter(Boolean).length
