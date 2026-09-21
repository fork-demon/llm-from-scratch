import { describe, expect, it } from 'vitest'
import { CORPUS_BASE } from './bigram'
import { buildNgram, contextCounts, formatPow10, generateNgram, log10Contexts, longestCopiedRun, sparsity } from './ngram'
import { makeRng } from './rng'

describe('context explosion', () => {
  it('counts possible contexts in log space', () => {
    expect(log10Contexts(27, 10)).toBeCloseTo(Math.log10(27 ** 10), 9)
    expect(formatPow10(log10Contexts(27, 2))).toBe('729')
    expect(formatPow10(log10Contexts(27, 10))).toBe('2.1 × 10^14') // 205,891,132,094,649
    expect(formatPow10(log10Contexts(50000, 2))).toBe('2.5 × 10^9')
    expect(log10Contexts(50000, 20)).toBeGreaterThan(80) // more contexts than atoms in the universe
  })
})

// Reference values computed independently in Python (collections.Counter over the same text).
describe('data sparsity on the real toy corpus (one copy, 493 chars)', () => {
  it('matches the Python counts', () => {
    expect(CORPUS_BASE.length).toBe(493)
    const rows = [1, 2, 3, 4, 6, 8, 10].map((n) => sparsity(CORPUS_BASE, n, 27))
    expect(rows.map((r) => [r.distinct, r.once])).toEqual([[27, 3], [159, 71], [299, 218], [367, 310], [432, 392], [472, 462], [482, 480]])
    expect(rows[2].onceFraction).toBeCloseTo(0.7291, 4)
    expect(rows[5].onceFraction).toBeCloseTo(0.9788, 4)
    expect(rows[1].coverage).toBeCloseTo(159 / 729, 10)
  })
  it('contextCounts sums to the number of windows', () => {
    const total = Array.from(contextCounts(CORPUS_BASE, 3).values()).reduce((s, c) => s + c, 0)
    expect(total).toBe(493 - 3 + 1)
  })
})

describe('n-gram generator', () => {
  it('builds next-character counts per context', () => {
    const t = buildNgram('abcabd', 2)
    expect(Object.fromEntries(t.get('ab')!)).toEqual({ c: 1, d: 1 })
    expect(Object.fromEntries(t.get('bc')!)).toEqual({ a: 1 })
  })
  it('is deterministic per seed', () => {
    expect(generateNgram(CORPUS_BASE, 3, 80, makeRng(5).next).text).toBe(generateNgram(CORPUS_BASE, 3, 80, makeRng(5).next).text)
  })
  it('longer context means more copying and fewer real choices', () => {
    const run = (k: number) => {
      const g = generateNgram(CORPUS_BASE, k, 200, makeRng(11).next)
      return { copied: longestCopiedRun(g.text, CORPUS_BASE).length, forced: g.forced / g.steps }
    }
    const k1 = run(1), k3 = run(3), k6 = run(6)
    expect(k1.copied).toBeLessThan(k3.copied)
    expect(k3.copied).toBeLessThan(k6.copied)
    expect(k6.copied).toBeGreaterThan(40)
    expect(k1.forced).toBeLessThan(0.2)
    expect(k6.forced).toBeGreaterThan(0.85)
  })
  it('longestCopiedRun finds the verbatim stretch', () => {
    expect(longestCopiedRun('xx the quick zz', CORPUS_BASE)).toEqual({ start: 3, length: 10 })
    expect(longestCopiedRun('', CORPUS_BASE).length).toBe(0)
  })
})
