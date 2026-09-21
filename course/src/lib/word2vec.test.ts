import { describe, expect, it } from 'vitest'
import { SENTENCES_BASE, buildDataset, initModel, nearest, similarity, trainStep } from './word2vec'

const data = buildDataset(SENTENCES_BASE)
const id = (w: string) => data.stoi.get(w)!

describe('dataset matches tiny_word2vec.py', () => {
  it('70 words, 492 pairs per copy of the corpus (9840 for the 20 copies Python uses)', () => {
    expect(data.words.length).toBe(70)
    expect(data.pairs.length / 2).toBe(9840 / 20)
    expect(buildDataset(['feed the cat'], 2).pairs.length / 2).toBe(6)
  })
})

describe('training creates geometry', () => {
  const m = initModel(data.words.length, 32, 1)
  const before = similarity(m, id('cat'), id('dog'))
  const firstLoss = trainStep(m, data)
  for (let s = 0; s < 799; s++) trainStep(m, data)

  it('loss falls from about ln(70)', () => {
    expect(firstLoss).toBeGreaterThan(4)
    expect(firstLoss).toBeLessThan(4.5)
    expect(m.loss).toBeLessThan(3)
    expect(m.step).toBe(800)
  })
  it('words used in similar contexts end up closer than unrelated words', () => {
    expect(Math.abs(before)).toBeLessThan(0.45) // random start
    expect(similarity(m, id('cat'), id('dog'))).toBeGreaterThan(0.4)
    expect(similarity(m, id('cat'), id('dog'))).toBeGreaterThan(similarity(m, id('cat'), id('computer')) + 0.4)
    expect(similarity(m, id('laptop'), id('computer'))).toBeGreaterThan(similarity(m, id('laptop'), id('cheese')) + 0.3)
    expect(similarity(m, id('bread'), id('cheese'))).toBeGreaterThan(similarity(m, id('bread'), id('software')))
  })
  it('nearest neighbour of cat is dog', () => {
    expect(nearest(m, data, 'cat', 3)[0].word).toBe('dog')
    expect(nearest(m, data, 'nope')).toEqual([])
  })
  it('is deterministic for a seed', () => {
    const a = initModel(70, 8, 4), b = initModel(70, 8, 4)
    for (let s = 0; s < 5; s++) { trainStep(a, data, 0.5, 32); trainStep(b, data, 0.5, 32) }
    expect(Array.from(a.E)).toEqual(Array.from(b.E))
  })
})
