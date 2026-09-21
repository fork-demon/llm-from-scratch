import { describe, expect, it } from 'vitest'
import { END, EMBEDDINGS, PROMPT, UNK, VOCAB, decode, embed, encode, generateAll, generateStep, nextDistribution, toyLogits, tokenize, uniformAt } from './pipeline'
import { softmax } from './math'

describe('toy tokenizer', () => {
  it('splits the running prompt into 5 tokens with leading spaces', () => {
    expect(tokenize(PROMPT)).toEqual(['What', ' is', ' a', ' cat', '?'])
    expect(encode(PROMPT)).toEqual([1, 2, 3, 4, 5])
  })
  it('round-trips and maps unknown words to <unk>', () => {
    expect(decode(encode(PROMPT))).toBe(PROMPT)
    expect(encode('What is a zebra?')[3]).toBe(UNK)
  })
  it('has one embedding row of 4 numbers per vocabulary entry', () => {
    expect(EMBEDDINGS.length).toBe(VOCAB.length)
    expect(EMBEDDINGS.every((r) => r.length === 4)).toBe(true)
    expect(embed([4])[0]).toEqual(EMBEDDINGS[4])
  })
})

describe('toy next-token table', () => {
  it('every reachable distribution sums to 1', () => {
    for (let a = 0; a < VOCAB.length; a++) for (let b = 0; b < VOCAB.length; b++) {
      const s = nextDistribution([a, b]).reduce((x, y) => x + y, 0)
      expect(s).toBeCloseTo(1, 10)
    }
  })
  it('uses two tokens of context when it has a rule for them', () => {
    expect(nextDistribution(encode('What is a cat? A cat is furry'))[VOCAB.indexOf('.')]).toBe(1)
    expect(nextDistribution(encode('What is a cat? A cat is a small furry'))[VOCAB.indexOf('.')]).toBe(0)
  })
  it('softmax of the toy logits gives back the probabilities', () => {
    const probs = nextDistribution(encode(PROMPT))
    softmax(toyLogits(probs)).forEach((p, i) => expect(p).toBeCloseTo(probs[i], 10))
  })
})

describe('generation loop', () => {
  const prompt = encode(PROMPT)
  it('greedy always gives the same sentence', () => {
    expect(decode(generateAll(prompt, 'greedy', 1))).toBe('What is a cat? A cat is a small furry animal.')
    expect(generateAll(prompt, 'greedy', 99)).toEqual(generateAll(prompt, 'greedy', 1))
  })
  it('sampling is reproducible for a seed and differs across seeds', () => {
    expect(generateAll(prompt, 'sample', 7)).toEqual(generateAll(prompt, 'sample', 7))
    const outs = new Set(Array.from({ length: 20 }, (_, s) => decode(generateAll(prompt, 'sample', s))))
    expect(outs.size).toBeGreaterThan(3)
  })
  it('stepping one token at a time equals running the whole loop', () => {
    const ids = prompt.slice()
    for (let n = 0; n < 24 && ids[ids.length - 1] !== END; n++) ids.push(generateStep(ids, 'sample', uniformAt(3, n)).next)
    expect(ids).toEqual(generateAll(prompt, 'sample', 3))
  })
  it('only ever picks tokens with non-zero probability', () => {
    const ids = generateAll(prompt, 'sample', 11)
    for (let t = prompt.length; t < ids.length; t++) expect(nextDistribution(ids.slice(0, t))[ids[t]]).toBeGreaterThan(0)
  })
})
