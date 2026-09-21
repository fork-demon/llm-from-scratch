import { describe, expect, it } from 'vitest'
import { CORPUS, buildVocab, countModel, countPairs, crossEntropyOfTable, generate, perplexity, rollNext, shiftedPairs, toIds } from './bigram'
import { makeRng } from './rng'

// Reference numbers come from running phase2-language/bigram_lm.py.
const vocab = buildVocab(CORPUS)
const V = vocab.chars.length
const ids = toIds(CORPUS, vocab)
const id = (c: string) => vocab.stoi.get(c)!

describe('count-table bigram matches bigram_lm.py', () => {
  it('corpus: 1479 chars, vocab 27', () => {
    expect(CORPUS.length).toBe(1479)
    expect(V).toBe(27)
  })

  it('cross-entropy 1.7518 and perplexity 5.76 with smoothing 0.01', () => {
    const ce = crossEntropyOfTable(countModel(ids, V), ids)
    expect(ce).toBeCloseTo(1.7518, 4)
    expect(perplexity(ce)).toBeCloseTo(5.76, 2)
  })

  it('heavier smoothing (1.0) makes the table worse: 1.9841 / 7.27', () => {
    const ce = crossEntropyOfTable(countModel(ids, V, 1.0), ids)
    expect(ce).toBeCloseTo(1.9841, 4)
    expect(perplexity(ce)).toBeCloseTo(7.27, 2)
  })

  it('rows are probabilities and match the Python table', () => {
    const table = countModel(ids, V)
    for (const row of table) expect(row.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 10)
    expect(table[id('t')][id('h')]).toBeCloseTo(0.5767, 4)
    expect(table[id('q')][id('u')]).toBeCloseTo(0.9585, 4)
    expect(table[id('h')][id('e')]).toBeCloseTo(0.7858, 4)
    expect(table[id(' ')][id('t')]).toBeCloseTo(0.2701, 4)
  })

  it('raw counts used in the lesson', () => {
    const counts = countPairs(ids, V)
    expect(counts[id('t')][id('h')]).toBe(78)
    expect(counts[id('t')].reduce((s, c) => s + c, 0)).toBe(135)
    expect(counts[id('q')][id('u')]).toBe(6)
  })
})

describe('generation', () => {
  const table = countModel(ids, V)
  it('is deterministic for a given seed and only emits known characters', () => {
    const a = generate(table, id('t'), 50, makeRng(3).next)
    const b = generate(table, id('t'), 50, makeRng(3).next)
    expect(a).toEqual(b)
    expect(a.every((i) => i >= 0 && i < V)).toBe(true)
  })
  it('rollNext picks the slice of the die that u falls into', () => {
    const q = id('q')
    expect(rollNext(table, q, 0.5).next).toBe(id('u'))
    expect(rollNext(table, id('t'), 0.99, true).next).toBe(id('h')) // greedy ignores u
  })
  it('greedy generation gets stuck in a loop', () => {
    const text = generate(table, id('t'), 40, makeRng(1).next, true).map((i) => vocab.chars[i]).join('')
    expect(text).toContain('he the the the')
  })
})

describe('shiftedPairs', () => {
  it('every position is a training example', () => {
    expect(shiftedPairs('the ')).toEqual([['t', 'h'], ['h', 'e'], ['e', ' ']])
  })
})
