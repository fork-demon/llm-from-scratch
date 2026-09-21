import { describe, expect, it } from 'vitest'
import { applyStep, bestFit, describeGradient, gdStep, lossAndGrads, makeLineData, maxStableLr, mse, numericalGrads, runGD, sampleBatch } from './gd'
import { makeRng } from './rng'

// Three points exactly on y = 3x - 1.5. Reference numbers come from running the
// Stage C formulas of phase1-foundations/gradient_descent.py in NumPy on these points.
const three = { xs: [1, 2, -1], ys: [1.5, 4.5, -4.5] }

describe('one update by hand (the lesson’s worked example)', () => {
  it('matches the NumPy numbers for step 0', () => {
    const g = lossAndGrads({ w: 0, b: 0 }, three)
    expect(g.errs).toEqual([-1.5, -4.5, 4.5])
    expect(g.loss).toBeCloseTo(14.25)
    expect(g.gradW).toBeCloseTo(-10)
    expect(g.gradB).toBeCloseTo(-1)
    const p = applyStep({ w: 0, b: 0 }, g, 0.1)
    expect(p.w).toBeCloseTo(1.0)
    expect(p.b).toBeCloseTo(0.1)
    expect(mse(p, three)).toBeCloseTo(6.2933333, 6)
  })
  it('matches NumPy for steps 1 and 2', () => {
    const { history, params } = runGD(three, { lr: 0.1, steps: 3 })
    expect(history[1].gradW).toBeCloseTo(-5.8666667, 6)
    expect(history[1].gradB).toBeCloseTo(0.5333333, 6)
    expect(history[2].loss).toBeCloseTo(3.4725926, 6)
    expect(params.w).toBeCloseTo(1.9457778, 6)
    expect(params.b).toBeCloseTo(-0.0742222, 6)
  })
  it('the exercise: two points on y = 2x, w = 1, lr = 0.1 gives w = 1.5', () => {
    const data = { xs: [1, 2], ys: [2, 4] }
    // model y = w*x: keep b out of it by looking only at gradW
    const g = lossAndGrads({ w: 1, b: 0 }, data)
    expect(g.gradW).toBeCloseTo(-5)
    expect(1 - 0.1 * g.gradW).toBeCloseTo(1.5)
  })
})

describe('analytic gradient = numerical gradient (Stage A vs Stage B)', () => {
  const data = makeLineData()
  for (const p of [{ w: 0, b: 0 }, { w: 4.2, b: -3 }, { w: -1, b: 2 }]) {
    it(`at w=${p.w}, b=${p.b}`, () => {
      const a = lossAndGrads(p, data)
      const n = numericalGrads(p, data)
      expect(n.gradW).toBeCloseTo(a.gradW, 3)
      expect(n.gradB).toBeCloseTo(a.gradB, 3)
    })
  }
})

describe('the data', () => {
  it('is deterministic and comes from y = 3x - 1.5', () => {
    const a = makeLineData()
    const b = makeLineData()
    expect(a).toEqual(b)
    expect(a.xs).toHaveLength(40)
    expect(Math.min(...a.xs)).toBeGreaterThanOrEqual(-2)
    expect(Math.max(...a.xs)).toBeLessThanOrEqual(2)
    const fit = bestFit(a)
    expect(fit.w).toBeGreaterThan(2.8)
    expect(fit.w).toBeLessThan(3.2)
    expect(fit.b).toBeGreaterThan(-1.7)
    expect(fit.b).toBeLessThan(-1.3)
  })
  it('the best fit has (almost) zero gradient', () => {
    const data = makeLineData()
    const g = lossAndGrads(bestFit(data), data)
    expect(Math.abs(g.gradW)).toBeLessThan(1e-9)
    expect(Math.abs(g.gradB)).toBeLessThan(1e-9)
  })
})

describe('running the loop', () => {
  const data = makeLineData()
  it('converges near w = 3, b = -1.5 with a sensible learning rate', () => {
    const { params, history, diverged } = runGD(data, { lr: 0.1, steps: 200 })
    expect(diverged).toBe(false)
    expect(params.w).toBeCloseTo(bestFit(data).w, 4)
    expect(params.b).toBeCloseTo(bestFit(data).b, 4)
    expect(Math.abs(params.w - 3)).toBeLessThan(0.2)
    expect(Math.abs(params.b + 1.5)).toBeLessThan(0.2)
    expect(history[history.length - 1].loss).toBeLessThan(history[0].loss / 50)
  })
  it('the loss falls on every step when lr is small', () => {
    const { history } = runGD(data, { lr: 0.05, steps: 50 })
    for (let i = 1; i < history.length; i++) expect(history[i].loss).toBeLessThanOrEqual(history[i - 1].loss)
  })
  it('diverges with a learning rate that is too large', () => {
    const { history, diverged } = runGD(data, { lr: 1.5, steps: 200 })
    expect(diverged).toBe(true)
    expect(history[10].loss).toBeGreaterThan(history[0].loss)
  })
  it('maxStableLr is the real boundary', () => {
    const edge = maxStableLr(data)
    expect(edge).toBeGreaterThan(0.5)
    expect(edge).toBeLessThan(1)
    const below = runGD(data, { lr: edge * 0.95, steps: 400 })
    expect(below.diverged).toBe(false)
    expect(below.history[399].loss).toBeLessThan(0.2)
    const above = runGD(data, { lr: edge * 1.05, steps: 400 })
    expect(above.history[above.history.length - 1].loss).toBeGreaterThan(above.history[0].loss)
  })
  it('mini-batches are noisy but still get close (Stage C settings)', () => {
    const { params, history } = runGD(data, { lr: 0.05, steps: 400, batchSize: 8, seed: 3 })
    expect(Math.abs(params.w - 3)).toBeLessThan(0.3)
    expect(Math.abs(params.b + 1.5)).toBeLessThan(0.3)
    expect(history[0].idx).toHaveLength(8)
    // the batch loss differs from the full loss: it is an estimate
    expect(history.some((r) => Math.abs(r.loss - r.fullLoss) > 1e-3)).toBe(true)
  })
  it('gdStep records the update with its actual numbers', () => {
    const r = gdStep({ w: 0, b: 0 }, three, 0.1, 0)
    expect(r.after.w).toBeCloseTo(r.before.w - r.lr * r.gradW)
    expect(r.idx).toBeNull()
  })
  it('sampleBatch stays in range', () => {
    const idx = sampleBatch(makeRng(1), 40, 16)
    expect(idx).toHaveLength(16)
    expect(idx.every((i) => i >= 0 && i < 40 && Number.isInteger(i))).toBe(true)
  })
})

describe('describeGradient', () => {
  it('positive gradient means turn the knob down', () => {
    expect(describeGradient(6).direction).toBe('down')
    expect(describeGradient(6).strength).toBe('strongly')
    expect(describeGradient(-2).direction).toBe('up')
    expect(describeGradient(0.01).direction).toBe('leave')
  })
})
