import { describe, expect, it } from 'vitest'
import { HAND_VOCAB, handWeights, randomEmbedding, splitWords } from './attentionDemo'
import { attention } from './math'

// Reference numbers come from running phase3-transformers/attention_numpy.py (section 4).
describe('river bank demo matches the Python repo', () => {
  const run = (ctx: string[]) => {
    const { Wq, Wk, Wv } = handWeights()
    return attention(ctx.map((w) => HAND_VOCAB[w]), Wq, Wk, Wv, { causal: true })
  }
  it('"bank" attends mostly to "river" and becomes more watery', () => {
    const { weights, out } = run(['the', 'river', 'bank'])
    const bank = weights[2]
    expect(bank[1]).toBeGreaterThan(bank[0] * 5) // river >> the
    expect(bank[1]).toBeCloseTo(bank[2]) // bank's own key matches its query equally well
    expect(out[2][0]).toBeGreaterThan(out[2][1]) // watery > financial
  })
  it('the same word next to "money" becomes more financial', () => {
    const { out } = run(['the', 'money', 'bank'])
    expect(out[2][1]).toBeGreaterThan(out[2][0])
  })
  it('exact weights for the river context', () => {
    // q_bank = [2,2,0,0]; keys: the=[0,0..], river=[2,0..], bank=[1,1..]; scores/2 = [0, 2, 2]
    const { weights } = run(['the', 'river', 'bank'])
    const e = Math.exp(2)
    expect(weights[2][0]).toBeCloseTo(1 / (1 + 2 * e))
    expect(weights[2][1]).toBeCloseTo(e / (1 + 2 * e))
  })
})

describe('typed sentences', () => {
  it('splits and caps the number of tokens', () => {
    expect(splitWords('The cat, sat!')).toEqual(['the', 'cat', 'sat'])
    expect(splitWords('a b c d e f g h i j')).toHaveLength(8)
  })
  it('gives a word the same vector every time', () => {
    expect(randomEmbedding('cat')).toEqual(randomEmbedding('cat'))
    expect(randomEmbedding('cat')).not.toEqual(randomEmbedding('mat'))
  })
})
