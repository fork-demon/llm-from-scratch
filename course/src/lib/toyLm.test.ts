import { describe, expect, it } from 'vitest'
import { CAT_LOGITS, CAT_JUNK_FROM, CAT_TOKENS, EOS, TOY_JUNK, TOY_VOCAB, detokenize, samplingStages, tallyDraws, toyGenerate, toyLogits } from './toyLm'

const sum = (v: number[]) => v.reduce((a, b) => a + b, 0)
const off = { temperature: 1, topK: 0, topP: 1 }

describe('samplingStages', () => {
  it('reproduces kv_cache_demo.py sample_demo', () => {
    const logits = [3.0, 2.5, 2.0, 1.8, 1.0, -1.0, -1.5, -2.0]
    const r2 = (v: number[]) => v.map((x) => x.toFixed(2))
    expect(r2(samplingStages(logits, off).final)).toEqual(['0.41', '0.25', '0.15', '0.12', '0.06', '0.01', '0.00', '0.00'])
    expect(r2(samplingStages(logits, { ...off, temperature: 0.3 }).final)).toEqual(['0.80', '0.15', '0.03', '0.01', '0.00', '0.00', '0.00', '0.00'])
    expect(r2(samplingStages(logits, { ...off, temperature: 3 }).final)).toEqual(['0.23', '0.19', '0.16', '0.15', '0.12', '0.06', '0.05', '0.04'])
    expect(r2(samplingStages(logits, { ...off, topK: 4 }).final)).toEqual(['0.44', '0.27', '0.16', '0.13', '0.00', '0.00', '0.00', '0.00'])
    expect(r2(samplingStages(logits, { ...off, topP: 0.9 }).final)).toEqual(['0.44', '0.27', '0.16', '0.13', '0.00', '0.00', '0.00', '0.00'])
  })

  it('every stage that is a distribution sums to 1', () => {
    const s = samplingStages(CAT_LOGITS, { temperature: 0.7, topK: 5, topP: 0.8 })
    for (const v of [s.probs, s.afterTopK, s.final]) expect(sum(v)).toBeCloseTo(1, 12)
    expect(s.scaled[0]).toBeCloseTo(4 / 0.7, 12)
    expect(s.afterTopK.filter((p) => p > 0).length).toBe(5)
  })

  it('temperature 0 is greedy', () => {
    const s = samplingStages(CAT_LOGITS, { ...off, temperature: 0 })
    expect(s.final[0]).toBe(1)
    expect(tallyDraws(s.final, 50, 1)[0]).toBe(50)
  })

  it('the numbers quoted in the lesson', () => {
    expect(CAT_TOKENS.length).toBe(CAT_LOGITS.length)
    const t1 = samplingStages(CAT_LOGITS, off).final
    expect(t1[0]).toBeCloseTo(0.403, 3)
    expect(sum(t1.slice(CAT_JUNK_FROM))).toBeCloseTo(0.0032, 4)
    const hot = samplingStages(CAT_LOGITS, { ...off, temperature: 3 }).final
    expect(sum(hot.slice(CAT_JUNK_FROM))).toBeCloseTo(0.078, 3)
    // exercise: softmax([2, 1, 0]) at T = 0.5
    expect(samplingStages([2, 1, 0], { ...off, temperature: 0.5 }).final[0]).toBeCloseTo(0.867, 3)
    expect(samplingStages([2, 1, 0], off).final[0]).toBeCloseTo(0.665, 3)
    // exercise: top-p 0.9 keeps 4 of these
    expect(samplingStages([0.5, 0.2, 0.15, 0.1, 0.05].map(Math.log), { ...off, topP: 0.9 }).final.filter((p) => p > 0).length).toBe(4)
  })

  it('tallies are seeded and add up', () => {
    const f = samplingStages(CAT_LOGITS, off).final
    expect(tallyDraws(f, 100, 5)).toEqual(tallyDraws(f, 100, 5))
    expect(sum(tallyDraws(f, 100, 5))).toBe(100)
  })
})

describe('toy next-word table', () => {
  it('has one logit per vocabulary entry', () => {
    for (const w of TOY_VOCAB) expect(toyLogits(w).length).toBe(TOY_VOCAB.length)
  })

  it('greedy decoding loops forever and only max-tokens stops it', () => {
    const g = toyGenerate(['the', 'cat'], { temperature: 0, topK: 0, topP: 1 }, 12, 1)
    expect(g.stopped).toBe('max-tokens')
    expect(detokenize(g.tokens)).toBe('sat on the cat sat on the cat sat on the cat')
    expect(toyGenerate(['the', 'cat'], { temperature: 0, topK: 0, topP: 1 }, 12, 99).tokens).toEqual(g.tokens)
  })

  it('sampling at T=1 usually reaches the end-of-sequence token, and EOS is never printed', () => {
    let eos = 0
    for (let seed = 1; seed <= 40; seed++) {
      const g = toyGenerate(['the', 'cat'], { temperature: 1, topK: 0, topP: 1 }, 40, seed)
      if (g.stopped === 'eos') eos++
      expect(g.tokens).not.toContain(EOS)
    }
    expect(eos).toBeGreaterThan(30)
  })

  it('high temperature lets junk through; top-k removes it again', () => {
    const junk = (s: { temperature: number; topK: number; topP: number }) => {
      let n = 0
      for (let seed = 1; seed <= 40; seed++) n += toyGenerate(['the', 'cat'], s, 20, seed).tokens.filter((t) => TOY_JUNK.has(t)).length
      return n
    }
    expect(junk({ temperature: 3, topK: 0, topP: 1 })).toBeGreaterThan(10)
    expect(junk({ temperature: 3, topK: 4, topP: 1 })).toBe(0)
    expect(junk({ temperature: 0.7, topK: 0, topP: 0.9 })).toBe(0)
  })

  it('is deterministic for a seed', () => {
    const s = { temperature: 1, topK: 0, topP: 1 }
    expect(toyGenerate(['the', 'cat'], s, 20, 3)).toEqual(toyGenerate(['the', 'cat'], s, 20, 3))
  })
})
