import { describe, expect, it } from 'vitest'
import {
  KV_PRESETS, emptyCache, formatBytes, forwardFull, forwardStep, generateCached, generateNaive,
  kvBytesPerToken, kvCacheBytes, kvRowsForToken, kvRowsTotal, makeToyParams, maxAbsDiff,
} from './kvcache'

const preset = (id: string) => KV_PRESETS.find((p) => p.id === id)!

describe('cache size formula', () => {
  it('reproduces the number kv_cache_demo.py prints: 123.0 KB at T=123', () => {
    // python: floats = LAYERS * 2 * T * D ; floats * 4 / 1024
    expect(kvCacheBytes(preset('toy'), 123)).toBe(2 * 2 * 123 * 64 * 4)
    expect(kvCacheBytes(preset('toy'), 123) / 1024).toBeCloseTo(123.0, 5)
  })
  it('GPT-2 small: 36 KB per token, 36 MB at 1,024 tokens', () => {
    expect(kvBytesPerToken(preset('gpt2'))).toBe(36864)
    expect(kvCacheBytes(preset('gpt2'), 1024)).toBe(36 * 1024 * 1024)
  })
  it('7B-class: 0.5 MB per token, 2 GB at 4,096 tokens', () => {
    expect(kvBytesPerToken(preset('7b'))).toBe(524288)
    expect(kvBytesPerToken(preset('7b')) / 1024 / 1024).toBe(0.5)
    expect(kvCacheBytes(preset('7b'), 4096) / 1024 ** 3).toBe(2)
  })
  it('GQA with 8 kv heads is 4x smaller', () => {
    expect(kvBytesPerToken({ ...preset('7b'), kvHeads: 8 })).toBe(524288 / 4)
  })
  it('formats bytes', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(524288)).toBe('512 KB')
    expect(formatBytes(2 * 1024 ** 3)).toBe('2.00 GB')
    expect(formatBytes(36864)).toBe('36.0 KB')
  })
})

describe('work counts', () => {
  it('naive recomputes every row each step; cached computes each row once', () => {
    expect(kvRowsForToken(3, 1, false)).toBe(3)
    expect(kvRowsForToken(3, 4, false)).toBe(6)
    expect(kvRowsForToken(3, 1, true)).toBe(3) // prefill
    expect(kvRowsForToken(3, 4, true)).toBe(1) // decode
    expect(kvRowsTotal(3, 5, false)).toBe(3 + 4 + 5 + 6 + 7)
    expect(kvRowsTotal(3, 5, true)).toBe(3 + 4)
  })
  it('naive grows like T squared, cached like T', () => {
    expect(kvRowsTotal(1, 1000, false)).toBe((1000 * 1001) / 2)
    expect(kvRowsTotal(1, 1000, true)).toBe(1000)
  })
})

describe('toy transformer: cached generation equals naive generation', () => {
  const p = makeToyParams({ vocab: 50, dim: 16, heads: 4, layers: 2, maxLen: 64 }, 9)
  const prompt = [1, 7, 3]

  it('same tokens out (the assert from kv_cache_demo.py)', () => {
    const a = generateNaive(p, prompt, 20)
    const b = generateCached(p, prompt, 20)
    expect(a).toEqual(b)
    expect(a.length).toBe(23)
    expect(new Set(a.slice(3)).size).toBeGreaterThan(1) // not a degenerate constant output
  })

  it('same logits, and the recomputed K/V rows equal the cached rows', () => {
    const ids = [1, 7, 3, 12, 40, 5]
    const cache = emptyCache(p)
    let stepLogits: number[] = []
    ids.forEach((t, i) => { stepLogits = forwardStep(p, t, i, cache) })
    const full = forwardFull(p, ids)
    expect(maxAbsDiff([full.logits], [stepLogits])).toBeLessThan(1e-12)
    full.kv.forEach((layer, l) => {
      expect(cache[l].K.length).toBe(ids.length)
      expect(maxAbsDiff(layer.K, cache[l].K)).toBeLessThan(1e-12)
      expect(maxAbsDiff(layer.V, cache[l].V)).toBeLessThan(1e-12)
    })
  })

  it('old K/V rows do not change when a token is appended (the causal-mask insight)', () => {
    const short = forwardFull(p, [1, 7, 3]).kv
    const long = forwardFull(p, [1, 7, 3, 12]).kv
    short.forEach((layer, l) => {
      expect(maxAbsDiff(layer.K, long[l].K.slice(0, 3))).toBe(0)
      expect(maxAbsDiff(layer.V, long[l].V.slice(0, 3))).toBe(0)
    })
  })

  it('a cache that forgets to append gives different output (the debug exercise)', () => {
    const cache = emptyCache(p)
    const ids = [1, 7, 3, 12]
    let broken: number[] = []
    ids.forEach((t, i) => {
      broken = forwardStep(p, t, i, cache)
      if (i === 1) cache.forEach((c) => { c.K.pop(); c.V.pop() }) // "forgot" token 1
    })
    // positions are now misaligned; rebuild a consistent call to compare against
    expect(cache[0].K.length).toBe(3)
    expect(maxAbsDiff([forwardFull(p, ids).logits], [broken])).toBeGreaterThan(1e-6)
  })
})
