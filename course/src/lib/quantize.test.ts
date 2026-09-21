// Mirrors phase6-engineering/quantize_demo.py. The small fixed matrices below were run through the
// Python functions; the expected numbers are its output. (The big random matrices in the Python demo
// come from NumPy's generator, which we do not port, so for those we test the same orderings.)
import { describe, expect, it } from 'vitest'
import { bitsPerWeight, decodeTokPerSBound, demoMatrices, fakeQuantize, linear, modelGB, outputError, plantOutlier, qmax, quantizeBlock, relErr, roundHalfEven } from './quantize'

const W = [[0.021, -0.013, 0.004, 0.035, -0.028, 0.009, -0.002, 0.017], [0.011, 0.052, -0.019, -0.007, 0.003, -0.044, 0.026, 0.015]]
const X = [[1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 0.1, -0.3], [0.2, 0.4, -0.6, 0.8, -1.0, 1.2, -1.4, 1.6]]

describe('quantizeBlock', () => {
  it('the hand example from the Python file', () => {
    const { q, scale } = quantizeBlock(W[0], 4)
    expect(scale).toBeCloseTo(0.005, 12)
    expect(q).toEqual([4, -3, 1, 7, -6, 2, 0, 3])
    q.forEach((v, i) => expect(Math.abs(W[0][i] - v * scale)).toBeLessThanOrEqual(scale / 2 + 1e-12))
  })
  it('levels', () => { expect(qmax(8)).toBe(127); expect(qmax(4)).toBe(7) })
  it('rounds ties to even like numpy', () => { expect([0.5, 1.5, 2.5, -0.5, -1.5].map(roundHalfEven).map((x) => x + 0)).toEqual([0, 2, 2, 0, -2]) })
  it('an all-zero block does not divide by zero', () => { expect(quantizeBlock([0, 0], 4)).toEqual({ q: [0, 0], scale: 1 }) })
})

describe('fakeQuantize reproduces quantize_demo.fake_quantize', () => {
  const cases: [number, 'tensor' | 'row' | 'group', number, number, number, number][] = [
    [8, 'tensor', 0, 1, 0.005626870309857437, 0.004862327091432268],
    [4, 'tensor', 0, 1, 0.09516734760279869, 0.06468596452498912],
    [4, 'row', 0, 2, 0.0851308130924325, 0.0365614781307117],
    [4, 'group', 4, 4, 0.07517295564623866, 0.038619943515745964],
  ]
  for (const [bits, gran, group, nScales, wErr, oErr] of cases) {
    it(`${bits}-bit ${gran}`, () => {
      const r = fakeQuantize(W, bits, gran, group)
      expect(r.nScales).toBe(nScales)
      expect(relErr(W, r.What)).toBeCloseTo(wErr, 12)
      expect(outputError(W, r.What, X)).toBeCloseTo(oErr, 12)
    })
  }
  it('dequantized values, group of 4', () => {
    const r = fakeQuantize(W, 4, 'group', 4)
    expect(r.What[1].map((x) => Number(x.toFixed(6)) + 0)).toEqual([0.007429, 0.052, -0.022286, -0.007429, 0, -0.044, 0.025143, 0.012571])
    expect(Math.max(...r.Q.flat().map(Math.abs))).toBeLessThanOrEqual(7)
  })
  it('an outlier: finer scales confine the damage (Python numbers)', () => {
    const Wo = plantOutlier(W, 0, 3, 1.0)
    expect(outputError(Wo, fakeQuantize(Wo, 4, 'tensor').What, X)).toBeCloseTo(0.05691950495144625, 12)
    expect(outputError(Wo, fakeQuantize(Wo, 4, 'row').What, X)).toBeCloseTo(0.043285010107241645, 12)
    expect(outputError(Wo, fakeQuantize(Wo, 4, 'group', 4).What, X)).toBeCloseTo(0.012844244320152104, 12)
  })
})

describe('the lab matrix shows the same orderings as the Python demo', () => {
  const { W: Wd, X: Xd } = demoMatrices()
  const err = (M: number[][], bits: number, g: 'tensor' | 'row' | 'group', group = 8) => outputError(M, fakeQuantize(M, bits, g, group).What, Xd)
  it('int8 is about 1%, int4 improves with finer scales', () => {
    expect(err(Wd, 8, 'row')).toBeLessThan(0.015)
    expect(err(Wd, 4, 'tensor')).toBeGreaterThan(err(Wd, 4, 'row'))
    expect(err(Wd, 4, 'row')).toBeGreaterThan(err(Wd, 4, 'group', 8))
  })
  it('one planted weight wrecks per-tensor int4 and barely touches group-wise', () => {
    const Wo = plantOutlier(Wd, 2, 5, 0.5)
    expect(relErr(Wo, fakeQuantize(Wo, 4, 'tensor').What)).toBeGreaterThan(0.3)
    expect(fakeQuantize(Wo, 4, 'tensor').Q.flat().filter((q) => q === 0).length).toBeGreaterThan(0.9 * 8 * 32)
    // the damage is confined to the one group that contains the outlier
    const clean = fakeQuantize(Wd, 4, 'group', 8).What
    const dirty = fakeQuantize(Wo, 4, 'group', 8).What
    let changed = 0
    clean.forEach((row, i) => row.forEach((v, j) => { if (v !== dirty[i][j]) changed++ }))
    expect(changed).toBeLessThanOrEqual(8)
  })
  it('linear is x @ W.T', () => { expect(linear([[1, 2]], [[3, 4], [5, 6]])).toEqual([[11, 17]]) })
})

describe('memory arithmetic (section 6 of the Python file)', () => {
  it('bits per weight including scales', () => {
    expect(bitsPerWeight(4, 128)).toBeCloseTo(4.125, 12)
    expect(bitsPerWeight(4, 32)).toBeCloseTo(4.5, 12)
    expect(bitsPerWeight(8)).toBeCloseTo(8.0039, 4)
  })
  it('a 7B model', () => {
    expect(modelGB(7e9, 16)).toBeCloseTo(14, 9)
    expect(modelGB(7e9, bitsPerWeight(4, 128))).toBeCloseTo(3.609, 3)
    expect(decodeTokPerSBound(2e12, 14)).toBeCloseTo(142.9, 1)
    expect(decodeTokPerSBound(2e12, modelGB(7e9, bitsPerWeight(4, 128)))).toBeCloseTo(554.1, 1)
  })
})
