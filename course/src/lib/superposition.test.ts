import { describe, expect, it } from 'vitest'
import { analyse, createTrainer, evalLoss, importances, initModel, lossAndGrad, reconstruct, sampleBatch, train, trainReference, type ToyModel } from './superposition'
import { makeRng } from './rng'

describe('toy model pieces', () => {
  it('importance decays geometrically', () => {
    expect(importances(0.9).map((v) => +v.toFixed(4))).toEqual([1, 0.9, 0.81, 0.729, 0.6561])
  })
  it('sparsity controls how many features are zero', () => {
    const X = sampleBatch(makeRng(1), 4000, 0.9)
    const zeros = X.flat().filter((v) => v === 0).length / X.flat().length
    expect(zeros).toBeGreaterThan(0.88)
    expect(zeros).toBeLessThan(0.92)
    expect(sampleBatch(makeRng(1), 50, 0).flat().every((v) => v > 0 && v < 1)).toBe(true)
  })
  it('gradient matches a numerical derivative', () => {
    const model = initModel(4)
    const X = sampleBatch(makeRng(2), 32, 0.5)
    const I = importances(0.9)
    const g = lossAndGrad(model, X, I)
    const eps = 1e-6
    for (const [k, i] of [[0, 0], [1, 3], [0, 4]]) {
      const plus: ToyModel = { W: model.W.map((r) => r.slice()), b: model.b.slice() }
      const minus: ToyModel = { W: model.W.map((r) => r.slice()), b: model.b.slice() }
      plus.W[k][i] += eps
      minus.W[k][i] -= eps
      const num = (lossAndGrad(plus, X, I).loss - lossAndGrad(minus, X, I).loss) / (2 * eps)
      expect(g.dW[k][i]).toBeCloseTo(num, 5)
    }
    const plus = { W: model.W, b: model.b.map((v, i) => (i === 2 ? v + eps : v)) }
    const minus = { W: model.W, b: model.b.map((v, i) => (i === 2 ? v - eps : v)) }
    expect(g.db[2]).toBeCloseTo((lossAndGrad(plus, X, I).loss - lossAndGrad(minus, X, I).loss) / (2 * eps), 5)
  })
  it('the fast trainer gives the same result as the readable reference', () => {
    const t = createTrainer({ sparsity: 0.95, decay: 0.9, seed: 3 })
    t.step(20)
    const ref = trainReference({ sparsity: 0.95, decay: 0.9, seed: 3 }, 20)
    t.model.W.forEach((row, k) => row.forEach((w, i) => expect(w).toBeCloseTo(ref.W[k][i], 10)))
    t.model.b.forEach((v, i) => expect(v).toBeCloseTo(ref.b[i], 10))
  })
})

describe('real training runs (numbers quoted in the lesson)', () => {
  it('dense data (S = 0): only the two most important features get a direction, at right angles', () => {
    const { model } = train({ sparsity: 0, decay: 0.9, seed: 1 })
    const a = analyse(model)
    expect(a.shape).toBe('orthogonal-pair')
    expect(a.represented).toEqual([0, 1])
    expect(a.norms[0]).toBeCloseTo(1, 1)
    // dropped features are predicted as their average value, 0.5, through the bias
    for (const i of [2, 3, 4]) expect(model.b[i]).toBeGreaterThan(0.45)
    expect(evalLoss(model, 0, 0.9)).toBeCloseTo(0.184, 2)
  })
  it('sparse data (S = 0.95, seed 1): all five features, 72° apart, with negative biases', () => {
    const { model } = train({ sparsity: 0.95, decay: 0.9, seed: 1 })
    const a = analyse(model)
    expect(a.shape).toBe('pentagon')
    a.gaps.forEach((g) => expect(Math.abs(g - 72)).toBeLessThan(3))
    a.norms.forEach((v) => expect(v).toBeGreaterThan(1.1))
    model.b.forEach((v) => expect(v).toBeLessThan(-0.2))
    // feature 1 alone: itself comes back, the two neighbours leak about 0.14, the far two are cut by ReLU
    const out = reconstruct(model, [1, 0, 0, 0, 0]).out
    expect(out[0]).toBeGreaterThan(1)
    expect(out.filter((v) => v > 0.1 && v < 0.2).length).toBe(2)
    expect(out.filter((v) => v === 0).length).toBe(2)
    expect(evalLoss(model, 0.95, 0.9)).toBeCloseTo(0.0082, 3)
  })
  it('seed 2 gets stuck in a square, which has higher loss than the pentagon', () => {
    const sq = train({ sparsity: 0.95, decay: 0.9, seed: 2 }).model
    const pent = train({ sparsity: 0.95, decay: 0.9, seed: 1 }).model
    expect(analyse(sq).shape).toBe('square')
    expect(evalLoss(sq, 0.95, 0.9)).toBeGreaterThan(evalLoss(pent, 0.95, 0.9) * 1.5)
  })
  it('with seed 1, more sparsity means more features represented', () => {
    const count = (S: number) => analyse(train({ sparsity: S, decay: 0.9, seed: 1 }).model).represented.length
    expect(count(0)).toBe(2)
    expect(count(0.7)).toBe(4)
    expect(count(0.95)).toBe(5)
  })
})

describe('local minima across seeds (quoted in the lesson)', () => {
  it('at S = 0.95, 7 of 12 seeds find the pentagon and 5 get stuck in a square', () => {
    const shapes = Array.from({ length: 12 }, (_, k) => analyse(train({ sparsity: 0.95, decay: 0.9, seed: k + 1 }).model).shape)
    expect(shapes.filter((s) => s === 'pentagon').length).toBe(7)
    expect(shapes.filter((s) => s === 'square').length).toBe(5)
  })
  it('the phase change for seed 1 sits between S = 0.80 (square) and S = 0.82 (pentagon)', () => {
    expect(analyse(train({ sparsity: 0.8, decay: 0.9, seed: 1 }).model).shape).toBe('square')
    expect(analyse(train({ sparsity: 0.82, decay: 0.9, seed: 1 }).model).shape).toBe('pentagon')
  })
})
