// Reward hacking in a toy you can read completely.
//
// The task: reply to “My UPI payment failed but the money left my account. How do I get a refund?”
// Every possible reply is described by three counts:
//   f = useful facts (0 to 3): the real steps the customer needs
//   p = padding sentences (0 to 6): “We truly value you as a customer...”
//   k = extra keyword repeats (0 to 6): “refund” said again for no reason
// That gives 4 × 7 × 7 = 196 possible replies.
//
// The PROXY grader is the kind of cheap automatic check a team might write in a hurry. It likes
// on-topic sentences, it likes length, and it likes the keyword “refund”:
//   a fact sentence            +1.0
//   any other sentence         +0.3   (length)
//   each extra “refund”        +0.3   (keyword)
//   proxy = f + 0.3 p + 0.3 k
// The TRUE quality is what the customer actually needs:
//   true = f − 0.4 p − 0.5 k   (padding and repetition waste the reader's time)
// The two agree about facts, so pushing the proxy up helps at first. Then they disagree.
//
// The “SFT model” (the reference) is a fixed distribution over the 196 replies: it usually writes
// one or two facts, a little politeness, and rarely repeats itself.
//
// The optimiser is idealised: exact expected-gradient steps (mirror ascent) on
//      objective = E[proxy] − β · KL(policy ‖ reference)
// Each step:  log π ← (1 − ηβ) log π + ηβ log ref + η proxy, then renormalise.
// With β = 0 after t steps this is exactly π ∝ ref · exp(η t · proxy): pressure grows without limit.
// With β > 0 it converges to π ∝ ref · exp(proxy / β): the KL leash caps how far it can go.
// Real RLHF samples replies and estimates this gradient noisily; the direction is the same.
import type { Vec } from './math'

export const MAX_F = 3
export const MAX_P = 6
export const MAX_K = 6

export interface Reply { f: number; p: number; k: number }

export const REPLIES: Reply[] = (() => {
  const out: Reply[] = []
  for (let f = 0; f <= MAX_F; f++) for (let p = 0; p <= MAX_P; p++) for (let k = 0; k <= MAX_K; k++) out.push({ f, p, k })
  return out
})()

export const proxyScore = (r: Reply): number => r.f + 0.3 * r.p + 0.3 * r.k
export const trueScore = (r: Reply): number => r.f - 0.4 * r.p - 0.5 * r.k

/** Reference (“SFT model”) marginals. The joint is their product. */
export const REF_F: Vec = [0.1, 0.3, 0.4, 0.2]
export const REF_P: Vec = [0.5, 0.3, 0.12, 0.05, 0.02, 0.008, 0.002]
export const REF_K: Vec = [0.75, 0.18, 0.05, 0.012, 0.005, 0.002, 0.001]

export const REFERENCE: Vec = REPLIES.map((r) => REF_F[r.f] * REF_P[r.p] * REF_K[r.k])

export const expectation = (probs: Vec, score: (r: Reply) => number): number =>
  probs.reduce((s, pi, i) => s + pi * score(REPLIES[i]), 0)

export const klDiv = (p: Vec, q: Vec): number =>
  p.reduce((s, pi, i) => (pi > 0 ? s + pi * Math.log(pi / q[i]) : s), 0)

const normaliseLog = (logp: Vec): Vec => {
  const m = Math.max(...logp)
  const e = logp.map((l) => Math.exp(l - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / s)
}

/** One idealised optimisation step. Returns the new policy. Requires η·β ≤ 1. */
export const step = (policy: Vec, beta: number, eta: number, ref: Vec = REFERENCE): Vec => {
  const logp = policy.map((pi, i) => (1 - eta * beta) * Math.log(pi) + eta * beta * Math.log(ref[i]) + eta * proxyScore(REPLIES[i]))
  return normaliseLog(logp)
}

/** The fixed point for β > 0: π ∝ ref · exp(proxy / β). */
export const optimum = (beta: number, ref: Vec = REFERENCE): Vec =>
  normaliseLog(ref.map((r, i) => Math.log(r) + proxyScore(REPLIES[i]) / beta))

/** π ∝ ref · exp(λ · proxy): the β = 0 policy after pressure λ = η · steps. */
export const tilted = (lambda: number, ref: Vec = REFERENCE): Vec =>
  normaliseLog(ref.map((r, i) => Math.log(r) + lambda * proxyScore(REPLIES[i])))

export interface Point { step: number; proxy: number; true: number; kl: number }

export const ETA = 0.1

/** Run the optimiser and record proxy, true quality and KL after every step (index 0 = reference). */
export const run = (beta: number, steps: number, eta = ETA): { points: Point[]; policies: Vec[] } => {
  let pi = REFERENCE.slice()
  const policies: Vec[] = [pi]
  const points: Point[] = [{ step: 0, proxy: expectation(pi, proxyScore), true: expectation(pi, trueScore), kl: 0 }]
  for (let t = 1; t <= steps; t++) {
    pi = step(pi, beta, eta)
    policies.push(pi)
    points.push({ step: t, proxy: expectation(pi, proxyScore), true: expectation(pi, trueScore), kl: klDiv(pi, REFERENCE) })
  }
  return { points, policies }
}

/** The single most likely reply under a policy. */
export const mostLikely = (policy: Vec): Reply => REPLIES[policy.indexOf(Math.max(...policy))]

/** Expected counts of facts, padding and repeats under a policy. */
export const expectedCounts = (policy: Vec): Reply => ({
  f: expectation(policy, (r) => r.f),
  p: expectation(policy, (r) => r.p),
  k: expectation(policy, (r) => r.k),
})

const FACTS = [
  'Open History in the app and check the refund status of that payment.',
  'If it still says “pending” after 48 hours, tap “Raise a refund dispute” on the same screen.',
  'Failed debits are normally reversed automatically, and you get an SMS when the refund lands.',
]
const PADDING = [
  'We truly value you as a customer.',
  'Your satisfaction is our top priority.',
  'We understand how frustrating this must be.',
  'Thank you for your patience and understanding.',
  'We are always here to help you.',
  'Rest assured, we take every concern seriously.',
]

/** Render a reply as text, so the learner can read what the optimiser is producing. */
export const renderReply = (r: Reply): string => {
  const parts: string[] = []
  if (r.p > 0) parts.push(PADDING[0])
  parts.push(...FACTS.slice(0, r.f))
  parts.push(...PADDING.slice(1, r.p))
  if (r.k > 0) parts.push(Array(r.k).fill('Refund.').join(' '))
  return parts.length === 0 ? 'Sorry to hear that.' : parts.join(' ')
}
