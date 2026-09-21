import { describe, expect, it } from 'vitest'
import { attention, type Mat } from './math'
import { averagePatterns, concatHeads, HEAD_SENTENCE, identity, illustrativeHeads, multiHeadAttention, shapesDemo, splitHeads } from './multihead'

const sinMat = (r: number, c: number, a: number, b: number, s: number): Mat =>
  Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => Math.sin(a * i + b * j + s)))

describe('multiHeadAttention (port of attention_numpy.py::multi_head_attention)', () => {
  const x = sinMat(3, 4, 1, 2, 0.3)
  const Wq = sinMat(4, 4, 2, 1, 0.1)
  const Wk = sinMat(4, 4, 1, 3, 0.7)
  const Wv = sinMat(4, 4, 3, 2, 1.1)
  const Wo = sinMat(4, 4, 1, 1, 0.5)

  it('reproduces the numbers printed by the Python function (n_heads=2, causal)', () => {
    const r = multiHeadAttention(x, Wq, Wk, Wv, Wo, 2)
    const pyOut = [[0.164238, -1.052414, -1.301481, -0.353973], [0.188087, -0.595859, -0.831975, -0.303177], [0.401494, 0.15127, -0.238032, -0.408488]]
    const pyW = [
      [[1, 0, 0], [0.262374, 0.737626, 0], [0.080742, 0.297839, 0.62142]],
      [[1, 0, 0], [0.76199, 0.23801, 0], [0.531273, 0.27405, 0.194677]],
    ]
    r.out.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(pyOut[i][j], 5)))
    r.weights.forEach((h, k) => h.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(pyW[k][i][j], 5))))
  })

  it('equals single-head attention when h = 1 and Wo = I', () => {
    for (const causal of [true, false]) {
      const multi = multiHeadAttention(x, Wq, Wk, Wv, identity(4), 1, causal)
      const single = attention(x, Wq, Wk, Wv, { causal })
      multi.out.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(single.out[i][j], 10)))
      multi.weights[0].forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(single.weights[i][j], 10)))
    }
  })

  it('every row of every head sums to 1, and the future gets exactly 0', () => {
    for (const h of [1, 2, 4]) {
      const r = multiHeadAttention(x, Wq, Wk, Wv, Wo, h)
      expect(r.weights).toHaveLength(h)
      for (const head of r.weights) head.forEach((row, i) => {
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
        row.forEach((v, j) => { if (j > i) expect(v).toBe(0) })
      })
    }
  })

  it('shapes: (T,D) -> h x (T,D/h) -> (T,D)', () => {
    const r = multiHeadAttention(x, Wq, Wk, Wv, Wo, 2)
    expect(r.Qh).toHaveLength(2)
    expect([r.Qh[0].length, r.Qh[0][0].length]).toEqual([3, 2])
    expect([r.headOut[1].length, r.headOut[1][0].length]).toEqual([3, 2])
    expect([r.concat.length, r.concat[0].length]).toEqual([3, 4])
    expect([r.out.length, r.out[0].length]).toEqual([3, 4])
  })

  it('split then concat is the identity; bad head counts throw', () => {
    expect(concatHeads(splitHeads(x, 2))).toEqual(x)
    expect(() => splitHeads(x, 3)).toThrow()
  })

  it('the shapes demo gives heads that really differ', () => {
    const d = shapesDemo()
    const r = multiHeadAttention(d.x, d.Wq, d.Wk, d.Wv, d.Wo, 2)
    const diff = Math.max(...r.weights[0].flatMap((row, i) => row.map((v, j) => Math.abs(v - r.weights[1][i][j]))))
    expect(diff).toBeGreaterThan(0.1)
  })
})

describe('illustrative heads', () => {
  const heads = illustrativeHeads()
  const T = HEAD_SENTENCE.length
  it('are valid causal attention patterns', () => {
    expect(heads).toHaveLength(4)
    for (const h of heads) {
      expect(h.weights).toHaveLength(T)
      h.weights.forEach((row, i) => {
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
        row.forEach((v, j) => { if (j > i) expect(v).toBe(0) })
      })
    }
  })
  it('show the advertised patterns', () => {
    const it_ = HEAD_SENTENCE.indexOf('it')
    const animal = HEAD_SENTENCE.indexOf('animal')
    expect(heads[0].weights[it_][it_ - 1]).toBeGreaterThan(0.8)
    expect(heads[1].weights[it_][animal]).toBeGreaterThan(0.7)
    expect(heads[2].weights[it_][0]).toBeGreaterThan(0.8)
    expect(heads[3].weights[it_][0]).toBeCloseTo(1 / (it_ + 1))
  })
  it('averaging the four patterns blurs them: no pattern survives intact', () => {
    const avg = averagePatterns(heads.map((h) => h.weights))
    const it_ = HEAD_SENTENCE.indexOf('it')
    expect(avg[it_].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    expect(Math.max(...avg[it_])).toBeLessThan(0.5)
  })
})
