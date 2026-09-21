// Speculative decoding on toy distributions: a TypeScript mirror of
// phase6-engineering/speculative_demo.py. The "models" are tiny Markov chains: row i of P (target)
// or Q (draft) is the next-token distribution after token i.
import { softmax, type Mat, type Vec } from './math'
import { makeRng, type Rng } from './rng'

/** Where the target wants MORE probability than the draft gave: max(0, p - q), normalised. */
export const residual = (p: Vec, q: Vec): Vec => {
  const r = p.map((x, i) => Math.max(0, x - q[i]))
  const total = r.reduce((s, x) => s + x, 0)
  return total > 0 ? r.map((x) => x / total) : p.slice()
}

/** Distribution of the first token a round emits: kept proposals plus resampled ones. Equals p. */
export const oneStepOutputDistribution = (p: Vec, q: Vec): Vec => {
  const accept = p.map((x, i) => Math.min(x, q[i])) // q(x) * min(1, p(x) / q(x))
  const rejected = 1 - accept.reduce((s, x) => s + x, 0)
  const res = residual(p, q)
  return accept.map((a, i) => a + rejected * res[i])
}

/** alpha: the chance one proposal is accepted = sum_x min(p, q), averaged uniformly over contexts. */
export const acceptanceRate = (P: Mat, Q: Mat): number =>
  P.reduce((s, row, i) => s + row.reduce((t, x, j) => t + Math.min(x, Q[i][j]), 0), 0) / P.length

/** Exact expected tokens per target pass, averaged uniformly over the last token.
 *  M = min(P, Q): (M^i 1)[a] is the chance the first i proposals after token a all survive. */
export const expectedTokensPerPass = (P: Mat, Q: Mat, gamma: number): number => {
  const V = P.length
  const M = P.map((row, i) => row.map((x, j) => Math.min(x, Q[i][j])))
  let alive: Vec = Array(V).fill(1)
  const total: Vec = Array(V).fill(1)
  for (let k = 0; k < gamma; k++) {
    alive = M.map((row) => row.reduce((s, m, j) => s + m * alive[j], 0))
    alive.forEach((a, i) => { total[i] += a })
  }
  return total.reduce((s, x) => s + x, 0) / V
}

/** Leviathan et al. (2023), equation 1, assuming every proposal is accepted independently with probability alpha. */
export const iidTokensPerPass = (alpha: number, gamma: number): number => (alpha >= 1 ? gamma + 1 : (1 - alpha ** (gamma + 1)) / (1 - alpha))

const draw = (p: Vec, u: number): number => {
  let c = 0
  for (let i = 0; i < p.length; i++) { c += p[i]; if (u < c) return i }
  return p.length - 1
}

export interface ProposalStep { ctx: number; token: number; p: number; q: number; acceptProb: number; u: number; accepted: boolean }
export interface SpecRound {
  proposals: number[] // what the draft guessed (always gamma tokens)
  steps: ProposalStep[] // the target's verdicts, up to and including the first rejection
  resampled: number | null // replacement drawn from the residual after a rejection
  residual: Vec | null
  bonus: number | null // free token when everything was accepted
  out: number[] // tokens this round adds to the text: 1 to gamma + 1
}

/** One round = gamma cheap draft steps + ONE target pass. */
export const speculativeRound = (P: Mat, Q: Mat, last: number, rng: Rng, gamma: number): SpecRound => {
  const proposals: number[] = []
  let ctx = last
  for (let k = 0; k < gamma; k++) { const x = draw(Q[ctx], rng.next()); proposals.push(x); ctx = x }
  const steps: ProposalStep[] = []
  const out: number[] = []
  ctx = last
  for (const x of proposals) {
    const p = P[ctx][x], q = Q[ctx][x]
    const acceptProb = Math.min(1, p / q)
    const u = rng.next()
    const accepted = u < acceptProb
    steps.push({ ctx, token: x, p, q, acceptProb, u, accepted })
    if (!accepted) {
      const res = residual(P[ctx], Q[ctx])
      const y = draw(res, rng.next())
      out.push(y)
      return { proposals, steps, resampled: y, residual: res, bonus: null, out }
    }
    out.push(x)
    ctx = x
  }
  const bonus = draw(P[ctx], rng.next())
  out.push(bonus)
  return { proposals, steps, resampled: null, residual: null, bonus, out }
}

export const samplePlain = (P: Mat, start: number, n: number, rng: Rng): number[] => {
  const out: number[] = []
  for (let i = 0; i < n; i++) out.push(draw(P[out.length ? out[out.length - 1] : start], rng.next()))
  return out
}

export const sampleSpeculative = (P: Mat, Q: Mat, start: number, n: number, rng: Rng, gamma: number): { tokens: number[]; passes: number } => {
  const out: number[] = []
  let passes = 0
  while (out.length < n) { out.push(...speculativeRound(P, Q, out.length ? out[out.length - 1] : start, rng, gamma).out); passes++ }
  return { tokens: out.slice(0, n), passes }
}

/** The target's exact distribution over all V^n sequences, flattened (first token is the most significant digit). */
export const exactJoint = (P: Mat, start: number, n: number): Vec => {
  const V = P.length
  const joint: Vec = Array(V ** n).fill(0)
  for (let idx = 0; idx < joint.length; idx++) {
    let prob = 1, ctx = start, rem = idx
    for (let k = n - 1; k >= 0; k--) { const x = Math.floor(rem / V ** k); rem -= x * V ** k; prob *= P[ctx][x]; ctx = x }
    joint[idx] = prob
  }
  return joint
}

export const tvDistance = (a: Vec, b: Vec): number => 0.5 * a.reduce((s, x, i) => s + Math.abs(x - b[i]), 0)

/** The lab's toy models. Target: a peaked random chain. Draft: the target blended with an unrelated chain.
 *  eps = 0 is a perfect draft, eps = 1 an unrelated one. (Python uses Dirichlet rows; the idea is the same.) */
export const makeModels = (eps: number, V = 6, seed = 3): { P: Mat; Q: Mat } => {
  const rng = makeRng(seed)
  const chain = (): Mat => Array.from({ length: V }, () => softmax(Array.from({ length: V }, () => 1.6 * rng.normal())))
  const P = chain()
  const R = chain()
  return { P, Q: P.map((row, i) => row.map((x, j) => (1 - eps) * x + eps * R[i][j])) }
}
