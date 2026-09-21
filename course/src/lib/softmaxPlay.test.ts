import { describe, expect, it } from 'vitest'
import { crossEntropy, softmax } from './math'
import { makeRng } from './rng'
import { divideBySum, fmtBig, naiveSoftmax, rollCounts, softmaxSteps, stableSoftmax, surprise } from './softmaxPlay'

const close = (a: number[], b: number[], digits = 3) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], digits))

describe('numbers printed in the softmax lesson', () => {
  it('cat 4.2, dog 2.1, car -0.7', () => {
    const s = softmaxSteps([4.2, 2.1, -0.7])
    close(s.exps, [66.686, 8.166, 0.497])
    expect(s.expSum).toBeCloseTo(75.349, 2)
    close(s.probs, [0.885, 0.108, 0.007])
  })
  it('dividing by the sum breaks on negatives', () => {
    close(divideBySum([4.2, 2.1, -0.7]), [0.75, 0.375, -0.125])
  })
  it('exercise: softmax([2,1,0]) = [0.665, 0.245, 0.090]', () => {
    close(softmax([2, 1, 0]), [0.665, 0.245, 0.09])
  })
  it('temperature: T=2 flattens, T=0.5 sharpens', () => {
    close(softmax([2, 1, 0], 2), [0.506, 0.307, 0.186])
    close(softmax([2, 1, 0], 0.5), [0.867, 0.117, 0.016])
  })
  it('surprise table', () => {
    expect(surprise(0.9)).toBeCloseTo(0.105, 3)
    expect(surprise(0.5)).toBeCloseTo(0.693, 3)
    expect(surprise(0.01)).toBeCloseTo(4.605, 3)
    expect(surprise(1)).toBeCloseTo(0, 12)
    expect(crossEntropy([0.1, 0.9], 1)).toBeCloseTo(surprise(0.9), 6)
  })
})

describe('only differences matter', () => {
  it('adding 100 to every logit changes e^z enormously but not the probabilities', () => {
    const a = softmaxSteps([4.2, 2.1, -0.7], 1, 0)
    const b = softmaxSteps([4.2, 2.1, -0.7], 1, 100)
    expect(b.exps[0]).toBeGreaterThan(1e44)
    close(a.probs, b.probs, 10)
  })
})

describe('numerical stability, as in mlp_numpy.py::softmax', () => {
  it('the naive formula overflows to NaN on big logits', () => {
    expect(Math.exp(1000)).toBe(Infinity)
    expect(naiveSoftmax([1000, 999]).every(Number.isNaN)).toBe(true)
  })
  it('subtracting the max fixes it and changes nothing else', () => {
    close(stableSoftmax([1000, 999]), [0.731, 0.269])
    close(stableSoftmax([1000, 999, 998]), softmax([2, 1, 0]), 10)
    close(stableSoftmax([2, 1, 0]), naiveSoftmax([2, 1, 0]), 12)
    close(stableSoftmax([4.2, 2.1, -0.7]), softmax([4.2, 2.1, -0.7]), 12)
  })
})

describe('sampling = rolling a weighted die', () => {
  it('is deterministic for a seed and accumulates', () => {
    const p = [0.7, 0.2, 0.1]
    const a = rollCounts(p, 100, makeRng(42))
    expect(rollCounts(p, 100, makeRng(42))).toEqual(a)
    expect(a.reduce((s, v) => s + v, 0)).toBe(100)
    const more = rollCounts(p, 50, makeRng(1), a)
    expect(more.reduce((s, v) => s + v, 0)).toBe(150)
    expect(a.reduce((s, v) => s + v, 0)).toBe(100) // input not mutated
  })
  it('counts approach the probabilities with many rolls', () => {
    const p = [0.7, 0.2, 0.1]
    const c = rollCounts(p, 20000, makeRng(7))
    c.forEach((n, i) => expect(n / 20000).toBeCloseTo(p[i], 1))
  })
  it('a probability of exactly 0 is never drawn', () => {
    expect(rollCounts([0, 1, 0], 500, makeRng(3))).toEqual([0, 500, 0])
  })
})

describe('fmtBig', () => {
  it('formats ordinary, huge and infinite values', () => {
    expect(fmtBig(66.686)).toBe('66.686')
    expect(fmtBig(1234.56)).toBe('1234.6')
    expect(fmtBig(Math.exp(104.2))).toBe('1.79 × 10^45')
    expect(fmtBig(0.0000123)).toBe('1.23 × 10^−5')
    expect(fmtBig(Infinity)).toBe('∞ (overflow)')
  })
})
