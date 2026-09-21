import { describe, expect, it } from 'vitest'
import { GIB, KV_PRESETS, attentionScoreEntries, centre, compareKv, divisors, ffnParams, formatBytes, kvBytesPerToken, kvCacheBytes, layerNorm, moeFfnParams, rmsNorm } from './modernArch'

const preset = (id: string) => KV_PRESETS.find((p) => p.id === id)!

describe('KV cache size', () => {
  it('Llama-2 7B, fp16: 0.5 MiB per token, exactly 2 GiB at 4,096 tokens', () => {
    const p = preset('llama2-7b')
    expect(kvBytesPerToken({ ...p, bytesPerValue: 2 })).toBe(524288)
    expect(kvCacheBytes({ ...p, tokens: 4096, bytesPerValue: 2 })).toBe(2 * GIB)
  })
  it('70B with 8 K/V heads: 327,680 bytes per token, 1.25 GiB at 4,096 tokens, 8x less than MHA', () => {
    const p = preset('llama-70b')
    expect(kvBytesPerToken({ ...p, bytesPerValue: 2 })).toBe(327680)
    const c = compareKv({ ...p, tokens: 4096, bytesPerValue: 2 })
    expect(c.gqa).toBe(1.25 * GIB)
    expect(c.mha).toBe(10 * GIB)
    expect(c.mqa).toBe(0.15625 * GIB)
    expect(c.savingVsMha).toBe(8)
    expect(c.groupSize).toBe(8)
  })
  it('70B at 128k tokens (131,072): 40 GiB with GQA, 320 GiB with MHA', () => {
    const c = compareKv({ ...preset('llama-70b'), tokens: 131072, bytesPerValue: 2 })
    expect(c.gqa).toBe(40 * GIB)
    expect(c.mha).toBe(320 * GIB)
  })
  it('our tiny GPT: 2 x 4 x 4 x 32 x 64 x 4 bytes (fp32) = 262,144 bytes = 256 KiB', () => {
    expect(kvCacheBytes({ ...preset('tiny'), tokens: 64, bytesPerValue: 4 })).toBe(262144)
    expect(formatBytes(262144)).toBe('256 KiB')
  })
  it('saving factor is queryHeads / kvHeads; MHA equals GQA when every head has its own K/V', () => {
    const c = compareKv({ ...preset('llama2-7b'), tokens: 1000, bytesPerValue: 2 })
    expect(c.savingVsMha).toBe(1)
    expect(c.mha / c.mqa).toBe(32)
  })
  it('scales linearly with tokens and with precision', () => {
    const base = { layers: 10, kvHeads: 4, headDim: 64, bytesPerValue: 2 }
    expect(kvCacheBytes({ ...base, tokens: 2000 })).toBe(2 * kvCacheBytes({ ...base, tokens: 1000 }))
    expect(kvCacheBytes({ ...base, tokens: 1000, bytesPerValue: 1 })).toBe(kvCacheBytes({ ...base, tokens: 1000 }) / 2)
  })
  it('rejects K/V head counts that do not divide the query heads', () => {
    expect(() => compareKv({ layers: 1, queryHeads: 32, kvHeads: 5, headDim: 8, tokens: 1, bytesPerValue: 2 })).toThrow()
    expect(divisors(32)).toEqual([1, 2, 4, 8, 16, 32])
    expect(divisors(12)).toEqual([1, 2, 3, 4, 6, 12])
  })
  it('formats bytes', () => {
    expect(formatBytes(512)).toBe('512 bytes')
    expect(formatBytes(2 * GIB)).toBe('2.00 GiB')
    expect(formatBytes(1.25 * GIB)).toBe('1.25 GiB')
    expect(formatBytes(40 * GIB)).toBe('40.0 GiB')
    expect(formatBytes(320 * GIB)).toBe('320 GiB')
  })
})

describe('context length cost', () => {
  it('32x more tokens means 1,024x more attention scores', () => {
    expect(attentionScoreEntries(4096)).toBe(16777216)
    expect(attentionScoreEntries(131072)).toBe(17179869184)
    expect(attentionScoreEntries(131072) / attentionScoreEntries(4096)).toBe(1024)
  })
})

describe('LayerNorm vs RMSNorm', () => {
  const x = [2, 4, 6, 8]
  it('worked example from the lesson', () => {
    const ln = layerNorm(x, 0)
    expect(ln.mean).toBe(5)
    expect(ln.variance).toBe(5)
    expect(ln.out.map((v) => +v.toFixed(3))).toEqual([-1.342, -0.447, 0.447, 1.342])
    const rms = rmsNorm(x, 0)
    expect(rms.meanSquare).toBe(30)
    expect(rms.out.map((v) => +v.toFixed(3))).toEqual([0.365, 0.73, 1.095, 1.461])
  })
  it('LayerNorm output has mean 0 and variance 1; RMSNorm output has mean square 1', () => {
    const ln = layerNorm(x, 0).out
    expect(ln.reduce((s, v) => s + v, 0)).toBeCloseTo(0, 12)
    expect(ln.reduce((s, v) => s + v * v, 0) / 4).toBeCloseTo(1, 12)
    const r = rmsNorm(x, 0).out
    expect(r.reduce((s, v) => s + v * v, 0) / 4).toBeCloseTo(1, 12)
  })
  it('the two agree exactly when the input already has mean 0', () => {
    const c = centre(x)
    expect(c).toEqual([-3, -1, 1, 3])
    const a = layerNorm(c).out
    const b = rmsNorm(c).out
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 12))
  })
  it('RMSNorm keeps the direction of x (all ratios preserved); LayerNorm does not', () => {
    const r = rmsNorm(x, 0).out
    expect(r[1] / r[0]).toBeCloseTo(2, 12)
    const l = layerNorm(x, 0).out
    expect(l[1] / l[0]).not.toBeCloseTo(2, 3)
  })
  it('both ignore the overall scale of the input', () => {
    const big = x.map((v) => v * 100)
    rmsNorm(big, 0).out.forEach((v, i) => expect(v).toBeCloseTo(rmsNorm(x, 0).out[i], 10))
    layerNorm(big, 0).out.forEach((v, i) => expect(v).toBeCloseTo(layerNorm(x, 0).out[i], 10))
  })
  it('eps keeps an all-zero vector finite', () => {
    expect(rmsNorm([0, 0, 0, 0]).out).toEqual([0, 0, 0, 0])
    expect(layerNorm([3, 3, 3, 3]).out).toEqual([0, 0, 0, 0])
  })
})

describe('parameter counts', () => {
  it('SwiGLU with hidden = 8d/3 has about as many weights as the classic 4d MLP', () => {
    const d = 4096
    expect(ffnParams(d, 4 * d, false)).toBe(134217728)
    // Llama-2 7B uses hidden 11008 (8d/3 = 10922.67 rounded up to a multiple of 256)
    expect(ffnParams(d, 11008, true)).toBe(135266304)
    expect(ffnParams(d, 11008, true) / ffnParams(d, 4 * d, false)).toBeCloseTo(1.008, 3)
  })
  it('tiny GPT FFN: 2 x 128 x 512 = 131,072 weights', () => {
    expect(ffnParams(128, 512, false)).toBe(131072)
  })
  it('MoE: 8 experts, 2 active -> 8x stored, 2x computed', () => {
    expect(moeFfnParams(100, 8, 2)).toEqual({ total: 800, active: 200 })
  })
})
