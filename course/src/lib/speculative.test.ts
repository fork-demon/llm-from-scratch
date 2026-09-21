// Mirrors phase6-engineering/speculative_demo.py and its tests: the claim that matters is that
// speculative sampling follows the TARGET distribution exactly, whatever the draft is.
import { describe, expect, it } from 'vitest'
import { acceptanceRate, exactJoint, expectedTokensPerPass, iidTokensPerPass, makeModels, oneStepOutputDistribution, residual, samplePlain, sampleSpeculative, speculativeRound, tvDistance } from './speculative'
import { makeRng } from './rng'

const P = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]
const Q = [[0.4, 0.4, 0.2], [0.3, 0.4, 0.3], [0.3, 0.2, 0.5]]

describe('the exact results match the Python functions on a fixed pair of chains', () => {
  it('residual and the one-step identity', () => {
    expect(residual(P[0], Q[0]).map((x) => +x.toFixed(12))).toEqual([1, 0, 0])
    oneStepOutputDistribution(P[0], Q[0]).forEach((x, i) => expect(x).toBeCloseTo(P[0][i], 12))
  })
  it('acceptance rate and expected tokens per pass', () => {
    expect(acceptanceRate(P, Q)).toBeCloseTo(0.8, 12)
    expect(expectedTokensPerPass(P, Q, 4)).toBeCloseTo(3.340533333333333, 12)
  })
  it('the i.i.d. formula from Leviathan et al.', () => {
    expect(iidTokensPerPass(0.8, 4)).toBeCloseTo((1 - 0.8 ** 5) / 0.2, 12)
    expect(iidTokensPerPass(1, 4)).toBe(5)
    expect(iidTokensPerPass(0, 4)).toBe(1)
  })
})

describe('the one-step output distribution is exactly the target, for any draft', () => {
  for (const eps of [0, 0.3, 0.7, 1]) {
    it(`eps = ${eps}`, () => {
      const m = makeModels(eps)
      m.P.forEach((p, ctx) => oneStepOutputDistribution(p, m.Q[ctx]).forEach((x, i) => expect(x).toBeCloseTo(p[i], 12)))
    })
  }
})

describe('sampling', () => {
  const m = makeModels(0.6)
  const n = 40000
  const hist = (sampler: () => number[]) => {
    const h = Array(36).fill(0)
    for (let i = 0; i < n; i++) { const [a, b] = sampler(); h[a * 6 + b] += 1 / n }
    return h
  }
  it('speculative samples follow the target, not the draft', () => {
    const truth = exactJoint(m.P, 0, 2)
    expect(truth.reduce((s, x) => s + x, 0)).toBeCloseTo(1, 12)
    const rng = makeRng(5)
    const spec = hist(() => sampleSpeculative(m.P, m.Q, 0, 2, rng, 4).tokens)
    const plain = hist(() => samplePlain(m.P, 0, 2, rng))
    const draft = hist(() => samplePlain(m.Q, 0, 2, rng))
    // the TV distance an UNBIASED sampler shows at this n, from the binomial noise in each cell (about 0.008)
    const noise = 0.5 * truth.reduce((s, p) => s + Math.sqrt((2 * p * (1 - p)) / (Math.PI * n)), 0)
    expect(tvDistance(truth, spec)).toBeLessThan(2 * noise)
    expect(tvDistance(truth, plain)).toBeLessThan(2 * noise)
    expect(tvDistance(truth, draft)).toBeGreaterThan(0.15)
    // chi-square goodness of fit: 36 cells, 35 degrees of freedom, 99.99th percentile is about 72.6
    const chi2 = truth.reduce((s, t, i) => s + (t > 0 ? (n * (spec[i] - t) ** 2) / t : 0), 0)
    expect(chi2).toBeLessThan(75)
  })
  it('measured tokens per pass matches the exact expectation', () => {
    const rng = makeRng(1)
    const rounds = 30000
    let total = 0
    for (let i = 0; i < rounds; i++) total += speculativeRound(m.P, m.Q, rng.int(6), rng, 4).out.length
    expect(total / rounds).toBeCloseTo(expectedTokensPerPass(m.P, m.Q, 4), 1)
  })
  it('a round emits 1 to gamma + 1 tokens, and stops at the first rejection', () => {
    const rng = makeRng(9)
    for (let i = 0; i < 500; i++) {
      const r = speculativeRound(m.P, m.Q, 0, rng, 4)
      expect(r.out.length).toBeGreaterThanOrEqual(1)
      expect(r.out.length).toBeLessThanOrEqual(5)
      const rejected = r.steps.findIndex((s) => !s.accepted)
      if (rejected >= 0) { expect(r.steps.length).toBe(rejected + 1); expect(r.bonus).toBeNull(); expect(r.out.length).toBe(rejected + 1) }
      else { expect(r.resampled).toBeNull(); expect(r.out.length).toBe(5) }
    }
  })
})

describe('draft quality', () => {
  it('a perfect draft yields gamma + 1 tokens per pass; worse drafts yield fewer, never below 1', () => {
    const e = [0, 0.2, 0.5, 1].map((eps) => { const m = makeModels(eps); return expectedTokensPerPass(m.P, m.Q, 4) })
    expect(e[0]).toBeCloseTo(5, 12)
    expect(e[0]).toBeGreaterThan(e[1]); expect(e[1]).toBeGreaterThan(e[2]); expect(e[2]).toBeGreaterThan(e[3]); expect(e[3]).toBeGreaterThan(1)
  })
})
