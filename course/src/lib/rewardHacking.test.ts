import { describe, expect, it } from 'vitest'
import { ETA, REFERENCE, REPLIES, expectation, klDiv, mostLikely, optimum, proxyScore, renderReply, run, step, tilted, trueScore } from './rewardHacking'

describe('the toy world', () => {
  it('196 replies and a proper reference distribution', () => {
    expect(REPLIES.length).toBe(196)
    expect(REFERENCE.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(mostLikely(REFERENCE)).toEqual({ f: 2, p: 0, k: 0 })
  })
  it('scores printed in the lesson', () => {
    expect(proxyScore({ f: 2, p: 1, k: 0 })).toBeCloseTo(2.3, 12)
    expect(trueScore({ f: 2, p: 1, k: 0 })).toBeCloseTo(1.6, 12)
    expect(proxyScore({ f: 3, p: 0, k: 0 })).toBe(3)
    expect(proxyScore({ f: 3, p: 6, k: 6 })).toBeCloseTo(6.6, 12)
    expect(trueScore({ f: 3, p: 6, k: 6 })).toBeCloseTo(-2.4, 12)
    expect(expectation(REFERENCE, proxyScore)).toBeCloseTo(2.052, 3)
    expect(expectation(REFERENCE, trueScore)).toBeCloseTo(1.195, 3)
  })
  it('renders the hacked reply', () => {
    expect(renderReply({ f: 3, p: 6, k: 6 })).toMatch(/Refund\. Refund\. Refund\. Refund\. Refund\. Refund\.$/)
    expect(renderReply({ f: 0, p: 0, k: 0 }).length).toBeGreaterThan(0)
  })
})

describe('the optimiser', () => {
  it('with beta = 0, t steps equal exponential tilting by eta * t', () => {
    const { policies } = run(0, 30)
    const exact = tilted(ETA * 30)
    policies[30].forEach((p, i) => expect(p).toBeCloseTo(exact[i], 10))
  })
  it('with beta > 0 it converges to ref * exp(proxy / beta)', () => {
    let pi = REFERENCE.slice()
    for (let t = 0; t < 2000; t++) pi = step(pi, 1, ETA)
    const exact = optimum(1)
    pi.forEach((p, i) => expect(p).toBeCloseTo(exact[i], 8))
  })
  it('Goodhart: without a leash, true quality rises then collapses', () => {
    const pts = run(0, 100).points
    let best = 0
    pts.forEach((p, i) => { if (p.true > pts[best].true) best = i })
    expect(best).toBe(12)
    expect(pts[12].true).toBeCloseTo(1.599, 3)
    expect(pts[100].proxy).toBeCloseTo(6.50, 2)
    expect(pts[100].true).toBeCloseTo(-2.26, 2)
    expect(pts[100].kl).toBeCloseTo(13.43, 2)
    expect(mostLikely(run(0, 100).policies[100])).toEqual({ f: 3, p: 6, k: 6 })
    // proxy never stops rising
    for (let i = 1; i < pts.length; i++) expect(pts[i].proxy).toBeGreaterThan(pts[i - 1].proxy)
  })
  it('the leash: numbers quoted in the lesson', () => {
    const at = (beta: number) => run(beta, 100).points[100]
    expect(at(1).true).toBeCloseTo(1.584, 3)
    expect(at(1).kl).toBeCloseTo(0.426, 3)
    expect(at(0.8).true).toBeCloseTo(1.599, 3)
    expect(at(0.3).true).toBeCloseTo(0.543, 3)
    expect(at(3).true).toBeCloseTo(1.375, 3)
    expect(mostLikely(run(1, 100).policies[100])).toEqual({ f: 3, p: 0, k: 0 })
  })
  it('best beta after 100 steps is about 0.8', () => {
    let bestBeta = 0
    let bestTrue = -Infinity
    for (let b = 0; b <= 3.0001; b += 0.05) {
      const t = run(b, 100).points[100].true
      if (t > bestTrue) { bestTrue = t; bestBeta = b }
    }
    expect(bestBeta).toBeCloseTo(0.8, 5)
  })
  it('two-reply worked example: closed form', () => {
    const share = (beta: number) => { const a = 0.9 * Math.exp(3 / beta); const h = 0.1 * Math.exp(6.6 / beta); return h / (a + h) }
    expect(share(1)).toBeCloseTo(0.8026, 4)
    expect(share(3)).toBeCloseTo(0.2695, 4)
    expect(klDiv([1 - share(1), share(1)], [0.9, 0.1])).toBeCloseTo(1.372, 3)
    expect(klDiv([1 - share(3), share(3)], [0.9, 0.1])).toBeCloseTo(0.115, 3)
  })
})
