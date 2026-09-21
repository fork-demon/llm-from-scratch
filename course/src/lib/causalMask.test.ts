import { describe, expect, it } from 'vitest'
import { softmax } from './math'
import { cheatingQKV, futureWeight, leakOnAnswer, MASK_NEXT, MASK_TOKENS, maskAfterSoftmax, maskWithDiagonal, randomQKV, runMask, trainingExamples } from './causalMask'

const sum = (v: number[]) => v.reduce((a, b) => a + b, 0)

describe('causal mask', () => {
  it('masked: future weight is exactly 0 and rows still sum to 1', () => {
    const r = runMask(randomQKV(), true)
    expect(futureWeight(r.weights).every((w) => w === 0)).toBe(true)
    r.weights.forEach((row) => expect(sum(row)).toBeCloseTo(1, 10))
    expect(r.weights[0]).toEqual([1, 0, 0, 0, 0, 0]) // the first token can only see itself
  })

  it('unmasked: every row puts some weight on the future (except the last)', () => {
    const fw = futureWeight(runMask(randomQKV(), false).weights)
    fw.slice(0, -1).forEach((w) => expect(w).toBeGreaterThan(0))
    expect(fw[fw.length - 1]).toBe(0)
  })

  it('masking does not change the relative weights among the visible tokens', () => {
    const free = runMask(randomQKV(), false).weights
    const masked = runMask(randomQKV(), true).weights
    const t = 3
    const visible = sum(free[t].slice(0, t + 1))
    for (let j = 0; j <= t; j++) expect(masked[t][j]).toBeCloseTo(free[t][j] / visible, 10)
  })

  it('the cheating head copies the answer without a mask and cannot with one', () => {
    const free = runMask(cheatingQKV(6), false)
    leakOnAnswer(free.weights).forEach((w) => expect(w).toBeGreaterThan(0.98))
    // the value it reads is the one-hot of the NEXT token: a copy of the answer
    expect(free.out[2][3]).toBeGreaterThan(0.98)
    const masked = runMask(cheatingQKV(6), true)
    leakOnAnswer(masked.weights).forEach((w) => expect(w).toBe(0))
  })

  it('strength is the final score of the answer cell', () => {
    expect(runMask(cheatingQKV(3), false).scores[1][2]).toBeCloseTo(3, 10)
  })

  it('T tokens give T training examples', () => {
    const ex = trainingExamples(MASK_TOKENS, MASK_NEXT)
    expect(ex).toHaveLength(MASK_TOKENS.length)
    expect(ex[0]).toEqual({ context: ['the'], target: 'cat' })
    expect(ex[5].target).toBe('.')
  })

  it('bug 1, mask after softmax: rows no longer sum to 1', () => {
    const w = maskAfterSoftmax([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    expect(sum(w[0])).toBeCloseTo(softmax([1, 2, 3])[0], 10) // 0.09
    expect(sum(w[0])).toBeLessThan(0.1)
    expect(sum(w[2])).toBeCloseTo(1, 10)
  })

  it('bug 2, k=0: a token cannot see itself, and row 0 has nothing left', () => {
    const s = maskWithDiagonal([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    expect(s[0].every((v) => v === -Infinity)).toBe(true)
    expect(softmax(s[1])).toEqual([1, 0, 0])
  })
})
