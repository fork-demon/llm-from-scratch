import { describe, expect, it } from 'vitest'
import type { Mat } from './math'
import { BLOCK_TOKENS, blockDemo, blockForward, countBlockNumbers, demoInput, depthScales, layerNorm, layerNormVec, makeBlockParams, meanOf, runStack, stdOf, swapRows, zeroBlockParams, type BlockParams } from './block'
import { makeRng } from './rng'

const sinMat = (r: number, c: number, a: number, b: number, s: number): Mat =>
  Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => Math.sin(a * i + b * j + s)))
const expectClose = (a: Mat, b: Mat, digits = 8) => a.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(b[i][j], digits)))

describe('layerNorm', () => {
  it('[2, 4, 6] -> [-1.22, 0, 1.22] (same as torch.nn.functional.layer_norm)', () => {
    const y = layerNormVec([2, 4, 6])
    expect(y[0]).toBeCloseTo(-1.224743, 4)
    expect(y[1]).toBeCloseTo(0, 6)
    expect(y[2]).toBeCloseTo(1.224743, 4)
  })
  it('every row comes out with mean 0 and std 1, whatever its scale', () => {
    const x = [[100, 250, -30, 7], [0.001, 0.002, -0.004, 0.003], [1, 2, 3, 4]]
    for (const row of layerNorm(x, undefined, undefined, 1e-12)) {
      expect(meanOf(row)).toBeCloseTo(0, 8)
      expect(stdOf(row)).toBeCloseTo(1, 5)
    }
  })
  it('gain and bias are applied after normalising', () => {
    expect(layerNormVec([2, 4, 6], [2, 2, 2], [1, 1, 1])[2]).toBeCloseTo(1 + 2 * 1.224743, 4)
  })
})

describe('blockForward', () => {
  it('matches tiny_gpt.py Block (PyTorch, GELU swapped for ReLU) on a fixed example', () => {
    const p: BlockParams = {
      ln1: { gain: Array(4).fill(1.5), bias: Array(4).fill(0.1) },
      attn: { Wq: sinMat(4, 4, 2, 1, 0.1), Wk: sinMat(4, 4, 1, 3, 0.7), Wv: sinMat(4, 4, 3, 2, 1.1), Wo: sinMat(4, 4, 1, 1, 0.5), bq: Array(4).fill(0.1), bk: Array(4).fill(0.2), bv: Array(4).fill(0.3), bo: Array(4).fill(0.05), nHeads: 2 },
      ln2: { gain: Array(4).fill(0.5), bias: Array(4).fill(-0.2) },
      ffn: { W1: sinMat(4, 16, 1, 2, 0.2), b1: Array(16).fill(-0.1), W2: sinMat(16, 4, 2, 3, 0.9), b2: Array(4).fill(0.2) },
    }
    const out = blockForward(sinMat(3, 4, 1, 2, 0.3), p).out
    expectClose(out, [[1.994865, -2.677154, -2.554438, -3.469074], [4.953594, -2.804626, 0.221619, -4.742956], [8.062619, -2.895175, 3.762047, -7.655278]], 5)
  })

  it('preserves the shape (T, D), so blocks can be stacked', () => {
    const x = demoInput(BLOCK_TOKENS, true)
    const t = blockForward(x, blockDemo().params)
    expect([t.out.length, t.out[0].length]).toEqual([3, 4])
    expect([t.ffn.act.length, t.ffn.act[0].length]).toEqual([3, 16])
    const rng = makeRng(1)
    const outs = runStack(x, [1, 2, 3].map(() => makeBlockParams(rng, 4, 2)))
    expect(outs).toHaveLength(3)
    expect([outs[2].length, outs[2][0].length]).toEqual([3, 4])
  })

  it('a block with zeroed sub-layer weights is the identity, thanks to the residuals', () => {
    const x = demoInput(BLOCK_TOKENS, true)
    expectClose(blockForward(x, zeroBlockParams(4, 2)).out, x)
    // without residuals the same block destroys everything
    expect(blockForward(x, zeroBlockParams(4, 2), { residual: false }).out.flat().every((v) => v === 0)).toBe(true)
  })

  it('no positions + no mask: swapping two input tokens just swaps the output rows', () => {
    const p = blockDemo().params
    const a = blockForward(demoInput(['dog', 'bites', 'man'], false), p, { causal: false }).out
    const b = blockForward(demoInput(['man', 'bites', 'dog'], false), p, { causal: false }).out
    expectClose(b, swapRows(a, 0, 2))
  })

  it('with positions the outputs genuinely change', () => {
    const p = blockDemo().params
    const a = blockForward(demoInput(['dog', 'bites', 'man'], true), p, { causal: false }).out
    const b = blockForward(demoInput(['man', 'bites', 'dog'], true), p, { causal: false }).out
    const diff = Math.max(...swapRows(a, 0, 2).flatMap((row, i) => row.map((v, j) => Math.abs(v - b[i][j]))))
    expect(diff).toBeGreaterThan(0.1)
  })

  it('no positions + causal mask: the last token cannot tell in which order the earlier tokens came', () => {
    const p = blockDemo().params
    const a = blockForward(demoInput(['dog', 'bites', 'man'], false), p).out
    const b = blockForward(demoInput(['bites', 'dog', 'man'], false), p).out
    a[2].forEach((v, j) => expect(v).toBeCloseTo(b[2][j], 8))
  })

  it('causal: changing a later token never changes an earlier row', () => {
    const p = blockDemo().params
    const x = demoInput(BLOCK_TOKENS, true)
    const y = x.map((r) => r.slice())
    y[2] = [9, -9, 9, -9]
    const a = blockForward(x, p).out
    const b = blockForward(y, p).out
    expectClose(a.slice(0, 2), b.slice(0, 2))
  })

  it('counts 12D² + 13D numbers per block', () => {
    expect(countBlockNumbers(makeBlockParams(makeRng(1), 4, 2))).toBe(12 * 16 + 13 * 4)
  })
})

describe('depthScales', () => {
  it('without normalisation the activations compound; with it they grow slowly', () => {
    const withNorm = depthScales(24, { norm: true, residual: true })
    const without = depthScales(24, { norm: false, residual: true })
    expect(withNorm).toHaveLength(25)
    expect(withNorm[24]).toBeLessThan(20)
    expect(without[24]).toBeGreaterThan(1000 * withNorm[24])
    expect(withNorm.every(Number.isFinite)).toBe(true)
  })
})
