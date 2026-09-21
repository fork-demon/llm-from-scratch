import { describe, expect, it } from 'vitest'
import { BPE_REPEAT, BPE_SAMPLE, UNK, UnknownCharError, applyMerge, bestPair, decode, encode, encodeSafe, initTraining, mergeStep, train, utf8Bytes } from './bpe'

// Reference values come from running phase2-language/bpe_tokenizer.py.
const TEXT = BPE_SAMPLE.repeat(BPE_REPEAT)

describe('BPE training reproduces bpe_tokenizer.py', () => {
  const tok = train(TEXT, 80)
  const learned = tok.merges.map((m) => [tok.vocab[m.pair[0]], tok.vocab[m.pair[1]], m.count])

  it('has 28 base characters and 52 merges at vocab_size=80', () => {
    expect(initTraining(TEXT).vocab.length).toBe(28)
    expect(tok.merges.length).toBe(52)
    expect(tok.vocab.length).toBe(80)
  })

  it('learns the same first merges, with the same counts and tie-breaks', () => {
    expect(learned.slice(0, 10)).toEqual([
      [' ', 't', 75], ['h', 'e', 64], [' t', 'he', 59], [' the', ' ', 47], ['i', 'n', 24],
      ['e', 'n', 20], ['f', 'o', 16], ['s', ' ', 16], ['e', 'r', 16], ['fo', 'x', 12],
    ])
    expect(tok.merges.slice(0, 3).map((m) => m.pair)).toEqual([[0, 21], [9, 6], [28, 29]])
    expect(tok.vocab[79]).toBe('quick brown fox jumps over the la')
  })

  it('encodes the sample sentence exactly like the Python file', () => {
    const sample = 'the fox jumps over the lazy tokenizer'
    const ids = encode(tok, sample)
    expect(ids).toEqual([21, 29, 0, 45, 11, 22, 14, 17, 38, 23, 46, 13, 2, 27, 26, 28, 16, 12, 33, 10, 27, 36])
    expect(ids.map((i) => tok.vocab[i])).toEqual(['t', 'he', ' ', 'fox ', 'j', 'u', 'm', 'p', 's o', 'v', 'er the ', 'l', 'a', 'z', 'y', ' t', 'o', 'k', 'en', 'i', 'z', 'er'])
    expect(decode(tok, ids)).toBe(sample)
  })

  it('splits unseen words into learned pieces', () => {
    expect(encode(tok, 'foxes').map((i) => tok.vocab[i])).toEqual(['fox', 'e', 's'])
    expect(encode(tok, 'unbelievable').length).toBe(12) // shatters into single characters
    expect(encode(tok, 'strawberry').map((i) => tok.vocab[i])).toEqual(['s', 't', 'r', 'a', 'w', 'b', 'er', 'r', 'y'])
  })

  it('stops when no pair repeats (single copy of the text: 31 merges)', () => {
    const once = train(BPE_SAMPLE, 80)
    expect(once.merges.length).toBe(31)
    expect(once.done).toBe(true)
    expect(once.vocab[once.vocab.length - 1]).toBe(' the token')
  })
})

describe('BPE mechanics', () => {
  it('bestPair breaks ties by first appearance and needs a count of 2', () => {
    expect(bestPair([1, 2, 3, 4, 3, 4, 1, 2])).toEqual({ pair: [1, 2], count: 2 })
    expect(bestPair([1, 2, 3])).toBeNull()
    expect(bestPair([])).toBeNull()
  })

  it('applyMerge is left to right and non-overlapping', () => {
    expect(applyMerge([5, 5, 5, 5, 5], [5, 5], 9)).toEqual([9, 9, 5])
  })

  it('mergeStep one at a time equals train()', () => {
    let s = initTraining(TEXT)
    for (let i = 0; i < 7; i++) s = mergeStep(s)
    expect(s.merges).toEqual(train(TEXT, 28 + 7).merges)
    expect(decode(s, s.ids)).toBe(TEXT)
  })

  it('round-trips any text made of known characters', () => {
    const tok = train(TEXT, 60)
    for (const t of ['', 'a', 'the the the', 'zzz...', 'god eht revo spmuj xof']) expect(decode(tok, encode(tok, t))).toBe(t)
  })

  it('replay order matters: reversed merges give a different, longer encoding', () => {
    const tok = train(TEXT, 80)
    const wrong = { ...tok, merges: tok.merges.slice().reverse() }
    const good = encode(tok, ' the fox')
    const bad = encode(wrong, ' the fox')
    expect(good.map((i) => tok.vocab[i])).toEqual([' the ', 'fox'])
    expect(bad.map((i) => tok.vocab[i])).toEqual([' t', 'he', ' ', 'fo', 'x'])
    expect(decode(tok, bad)).toBe(' the fox') // still lossless, just not what the model was trained on
  })

  it('unseen characters: encode throws like Python, encodeSafe marks <unk>', () => {
    const tok = train(TEXT, 80)
    expect(() => encode(tok, 'the Fox')).toThrow(UnknownCharError)
    const pieces = encodeSafe(tok, 'the é fox')
    expect(pieces.map((p) => p.text).join('')).toBe('the é fox')
    expect(pieces.filter((p) => p.id === UNK).map((p) => p.text)).toEqual(['é'])
    expect(encodeSafe(tok, 'the fox').map((p) => p.id)).toEqual(encode(tok, 'the fox'))
    expect(utf8Bytes('é')).toEqual([0xc3, 0xa9])
  })
})
