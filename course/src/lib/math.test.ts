import { describe, expect, it } from 'vitest'
import { argmax, attention, attentionFromQKV, cosine, crossEntropy, dot, matmul, sampleIndex, softmax, topK, topP, transpose } from './math'

const sum = (v: number[]) => v.reduce((a, b) => a + b, 0)

describe('vectors', () => {
  it('dot product measures agreement', () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32)
    expect(dot([1, 0], [0, 1])).toBe(0)
    expect(dot([1, 0], [-1, 0])).toBe(-1)
  })
  it('cosine ignores length', () => {
    expect(cosine([1, 1], [5, 5])).toBeCloseTo(1)
    expect(cosine([1, 0], [0, 3])).toBeCloseTo(0)
    expect(cosine([0, 0], [1, 1])).toBe(0)
  })
  it('rejects mismatched lengths', () => {
    expect(() => dot([1], [1, 2])).toThrow()
  })
})

describe('matrices', () => {
  it('multiplies by rows-times-columns', () => {
    expect(matmul([[2, 0], [1, 3]], [[1], [2]])).toEqual([[2], [7]])
  })
  it('checks inner dimensions', () => {
    expect(() => matmul([[1, 2, 3]], [[1, 2, 3]])).toThrow(/inner/)
  })
  it('transposes', () => {
    expect(transpose([[1, 2, 3], [4, 5, 6]])).toEqual([[1, 4], [2, 5], [3, 6]])
  })
})

describe('softmax', () => {
  it('sums to one and keeps order', () => {
    const p = softmax([4.2, 2.1, -0.7])
    expect(sum(p)).toBeCloseTo(1)
    expect(p[0]).toBeGreaterThan(p[1])
    expect(p[1]).toBeGreaterThan(p[2])
    expect(p[0]).toBeCloseTo(0.8853, 3)
  })
  it('is unchanged by shifting all logits', () => {
    const a = softmax([1, 2, 3])
    const b = softmax([101, 102, 103])
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i]))
  })
  it('does not overflow on huge logits', () => {
    expect(sum(softmax([1000, 999]))).toBeCloseTo(1)
  })
  it('temperature sharpens and flattens', () => {
    const base = softmax([2, 1, 0])[0]
    expect(softmax([2, 1, 0], 0.5)[0]).toBeGreaterThan(base)
    expect(softmax([2, 1, 0], 2)[0]).toBeLessThan(base)
    expect(softmax([2, 1, 0], 0)).toEqual([1, 0, 0])
  })
  it('gives masked (-inf) entries exactly zero', () => {
    const p = softmax([1, -Infinity, 1])
    expect(p).toEqual([0.5, 0, 0.5])
  })
  it('cross-entropy is small when confident and right', () => {
    expect(crossEntropy([0.9, 0.1], 0)).toBeLessThan(crossEntropy([0.9, 0.1], 1))
    expect(crossEntropy([1, 0], 0)).toBeCloseTo(0)
  })
})

describe('attention', () => {
  const I = [[1, 0], [0, 1]]
  const x = [[1, 0], [0, 1], [1, 1]]
  it('rows of the weight matrix sum to one', () => {
    const { weights } = attention(x, I, I, I)
    weights.forEach((row) => expect(sum(row)).toBeCloseTo(1))
  })
  it('matches a hand calculation', () => {
    // Q = K = V = x. Row 0 raw scores: [1, 0, 1]; scaled by sqrt(2).
    const { raw, weights, out } = attention(x, I, I, I)
    expect(raw[0]).toEqual([1, 0, 1])
    const e1 = Math.exp(1 / Math.SQRT2)
    const expected = [e1, 1, e1].map((v) => v / (2 * e1 + 1))
    weights[0].forEach((w, j) => expect(w).toBeCloseTo(expected[j]))
    expect(out[0][0]).toBeCloseTo(expected[0] + expected[2])
  })
  it('causal mask blocks the future', () => {
    const { weights } = attention(x, I, I, I, { causal: true })
    expect(weights[0]).toEqual([1, 0, 0])
    expect(weights[1][2]).toBe(0)
    expect(sum(weights[2])).toBeCloseTo(1)
  })
  it('output is a blend of values: stays inside their range', () => {
    const V = [[10, 0], [0, 10], [5, 5]]
    const { out } = attentionFromQKV(x, x, V)
    out.forEach((row) => row.forEach((v) => { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(10) }))
  })
})

describe('sampling', () => {
  const p = [0.5, 0.3, 0.15, 0.05]
  it('top-k keeps k tokens', () => {
    const q = topK(p, 2)
    expect(q.filter((v) => v > 0)).toHaveLength(2)
    expect(sum(q)).toBeCloseTo(1)
    expect(q[0]).toBeCloseTo(0.625)
  })
  it('top-p keeps the smallest set covering p', () => {
    expect(topP(p, 0.8).filter((v) => v > 0)).toHaveLength(2)
    expect(topP(p, 0.81).filter((v) => v > 0)).toHaveLength(3)
    expect(topP(p, 0.1).filter((v) => v > 0)).toHaveLength(1)
  })
  it('sampleIndex follows the cumulative distribution', () => {
    expect(sampleIndex(p, 0.0)).toBe(0)
    expect(sampleIndex(p, 0.49)).toBe(0)
    expect(sampleIndex(p, 0.5)).toBe(1)
    expect(sampleIndex(p, 0.99)).toBe(3)
    expect(argmax(p)).toBe(0)
  })
})
