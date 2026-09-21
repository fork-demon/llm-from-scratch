import { describe, expect, it } from 'vitest'
import { TOY_WORDS, analogy, neighbours, pairStats } from './embeddingToy'

const idx = (w: string) => TOY_WORDS.findIndex((x) => x.word === w)
const pos = (w: string) => TOY_WORDS[idx(w)].pos

describe('toy embedding layout', () => {
  it('pair statistics are computed correctly', () => {
    const s = pairStats([3, 4], [4, 3])
    expect(s.dot).toBe(24)
    expect(s.cosine).toBeCloseTo(0.96, 10)
    expect(s.distance).toBeCloseTo(Math.SQRT2, 10)
  })
  it('numbers quoted in the lesson', () => {
    const cd = pairStats(pos('cat'), pos('dog'))
    expect(cd.dot).toBeCloseTo(18.48, 10)
    expect(cd.cosine).toBeCloseTo(0.974, 3)
    expect(cd.distance).toBeCloseTo(1.0, 10)
    const cc = pairStats(pos('cat'), pos('car'))
    expect(cc.dot).toBeCloseTo(-3.68, 10)
    expect(cc.cosine).toBeCloseTo(-0.202, 3)
  })
  it('similar things are neighbours', () => {
    expect(neighbours(TOY_WORDS, idx('cat')).slice(0, 3).map((n) => n.word).sort()).toEqual(['dog', 'kitten', 'puppy'])
    expect(neighbours(TOY_WORDS, idx('car')).slice(0, 2).map((n) => n.word).sort()).toEqual(['bus', 'truck'])
    expect(neighbours(TOY_WORDS, idx('king')).slice(0, 3).map((n) => n.word)).toEqual(['man', 'queen', 'woman'])
  })
  it('king − man + woman lands near queen', () => {
    const r = analogy(TOY_WORDS, 'king', 'man', 'woman')
    expect(r.landing[0]).toBeCloseTo(3.4, 10)
    expect(r.landing[1]).toBeCloseTo(3.8, 10)
    expect(r.nearest.word).toBe('queen')
    expect(r.nearest.distance).toBeLessThan(0.3)
  })
})
