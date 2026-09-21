import { describe, expect, it } from 'vitest'
import { NUDGE_FUNCTIONS, PIPELINE, chainReport, convergence, nudgeDerivative, nudgeReport, runPipeline, samplePoints, wiggleTrace } from './calculus'

const fn = (id: string) => NUDGE_FUNCTIONS.find((f) => f.id === id)!

describe('nudgeDerivative reproduces math_primer.py section 7', () => {
  it('x^2 at x=3 measures 6', () => {
    expect(Math.abs(nudgeDerivative((x) => x ** 2, 3) - 6)).toBeLessThan(1e-3)
  })
  it('the formula table at x=2: 5x, x^2, e^x, ln x', () => {
    const table: [(x: number) => number, number][] = [[(x) => 5 * x, 5], [(x) => x ** 2, 4], [Math.exp, Math.exp(2)], [Math.log, 0.5]]
    for (const [f, expected] of table) expect(Math.abs(nudgeDerivative(f, 2) - expected)).toBeLessThan(1e-3)
  })
  it('the page example: f(3.001) = 9.006001, ratio 6.001', () => {
    const r = nudgeReport(fn('square'), 3, 0.001)
    expect(r.fxh).toBeCloseTo(9.006001, 9)
    expect(r.ratio).toBeCloseTo(6.001, 6)
    expect(r.exact).toBe(6)
  })
})

describe('numbers printed in the derivatives lesson', () => {
  it('x^2 at x=3: ratio settles on 6 as h shrinks', () => {
    const c = convergence(fn('square').f, 3).map((r) => r.ratio)
    ;[7, 6.1, 6.01, 6.001].forEach((v, i) => expect(c[i]).toBeCloseTo(v, 6))
  })
  it('exercise: x^3 at x=2 is 12', () => {
    const c = convergence(fn('cube').f, 2).map((r) => r.ratio)
    ;[19, 12.61, 12.0601, 12.006001].forEach((v, i) => expect(c[i]).toBeCloseTo(v, 6))
    expect(nudgeDerivative(fn('cube').f, 2)).toBeCloseTo(12, 3)
    expect(fn('cube').exact(2)).toBe(12)
  })
  it('a straight line has the same slope for every x and every h', () => {
    for (const x of [-2, 0, 1.5]) for (const h of [1, 0.01]) expect(nudgeDerivative(fn('line').f, x, h)).toBeCloseTo(3, 9)
  })
  it('max(0, x): slope 0 on the left, 1 on the right, undefined at the kink', () => {
    const relu = fn('relu')
    expect(nudgeDerivative(relu.f, -1, 0.01)).toBe(0)
    expect(nudgeDerivative(relu.f, 1, 0.01)).toBeCloseTo(1, 9)
    expect(Number.isNaN(relu.exact(0))).toBe(true)
    expect(nudgeDerivative(relu.f, 0, 0.01)).toBeCloseTo(1) // nudging right
    expect(nudgeDerivative(relu.f, 0, -0.01)).toBeCloseTo(0) // nudging left
  })
  it('negative slope: x^2 at x=-1.5 is -3', () => {
    expect(nudgeDerivative(fn('square').f, -1.5)).toBeCloseTo(-3, 4)
  })
  it('exact formulas agree with the nudge experiment everywhere we plot', () => {
    for (const f of NUDGE_FUNCTIONS) for (const x of [-2.5, -1, 0.5, 2]) expect(nudgeDerivative(f.f, x)).toBeCloseTo(f.exact(x), 3)
  })
})

describe('chain rule pipeline', () => {
  it('math_primer.py section 8: x -> square -> times 5 at x=2 gives 4 x 5 = 20', () => {
    const stages = [PIPELINE[0], { name: 'times 5', expr: '5u', apply: (u: number) => 5 * u, local: () => 5, localRule: 'always 5' }]
    const r = chainReport(2, stages)
    expect(r.product).toBe(20)
    expect(Math.abs(r.measured - 20)).toBeLessThan(1e-3)
  })
  it('x -> square -> +1 -> x3 at x=2: 4 x 1 x 3 = 12, output 15', () => {
    const r = chainReport(2)
    expect(r.stages.map((s) => s.local)).toEqual([4, 1, 3])
    expect(r.stages.map((s) => s.output)).toEqual([4, 5, 15])
    expect(r.output).toBe(15)
    expect(runPipeline(2)).toBe(15)
    expect(r.product).toBe(12)
    expect(Math.abs(r.measured - 12)).toBeLessThan(1e-3)
  })
  it('product equals the measured nudge for other inputs too', () => {
    for (const x of [-3, -0.5, 0, 1, 2.5]) {
      const r = chainReport(x)
      expect(r.product).toBeCloseTo(6 * x, 9)
      expect(r.measured).toBeCloseTo(r.product, 4)
    }
  })
  it('exercise: x -> x3 -> square -> -4 at x=1 is 3 x 6 x 1 = 18', () => {
    expect(nudgeDerivative((x) => (3 * x) ** 2 - 4, 1)).toBeCloseTo(18, 3)
  })
  it('wiggle trace: +0.001 becomes about 0.004, 0.004, 0.012', () => {
    const w = wiggleTrace(2, 0.001)
    ;[0.001, 0.004001, 0.004001, 0.012003].forEach((v, i) => expect(w[i]).toBeCloseTo(v, 9))
  })
})

describe('samplePoints', () => {
  it('covers the interval inclusively', () => {
    const pts = samplePoints((x) => x * x, -2, 2, 4)
    expect(pts).toEqual([[-2, 4], [-1, 1], [0, 0], [1, 1], [2, 4]])
  })
})
