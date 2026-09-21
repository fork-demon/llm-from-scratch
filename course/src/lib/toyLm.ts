// Sampling, made visible. Two things live here:
//  1. samplingStages(): logits -> / T -> softmax -> top-k -> top-p, keeping every intermediate step.
//  2. A hand-made word-level "next token" table so the same settings can be seen changing real text.
//     It is a TOY lookup table (the next word depends only on the previous word), not a Transformer.
// Same policies as phase3-transformers/kv_cache_demo.py :: sample_demo.
import { sampleIndex, softmax, topK, topP, type Vec } from './math'
import { makeRng } from './rng'

/* ------------------------------------------------------------------ */
/* The pipeline from logits to the die that gets rolled                 */
/* ------------------------------------------------------------------ */

export interface SamplingSettings {
  temperature: number // 0 means greedy (argmax)
  topK: number // 0 means off
  topP: number // 1 means off
}

export interface SamplingStages {
  logits: Vec
  scaled: Vec // logits / T
  probs: Vec // softmax(scaled)
  afterTopK: Vec // k best kept, rest 0, renormalised
  final: Vec // then the smallest set reaching top-p kept, rest 0, renormalised: this is what gets sampled
}

export const samplingStages = (logits: Vec, s: SamplingSettings): SamplingStages => {
  const scaled = s.temperature > 0 ? logits.map((z) => z / s.temperature) : logits.slice()
  const probs = softmax(logits, s.temperature)
  const afterTopK = s.topK > 0 ? topK(probs, s.topK) : probs.slice()
  const final = s.topP < 1 ? topP(afterTopK, s.topP) : afterTopK.slice()
  return { logits, scaled, probs, afterTopK, final }
}

/** Draw `n` tokens from the final distribution with a seeded generator; returns how often each index came up. */
export const tallyDraws = (final: Vec, n: number, seed: number): number[] => {
  const rng = makeRng(seed)
  const counts = final.map(() => 0)
  for (let i = 0; i < n; i++) counts[sampleIndex(final, rng.next())]++
  return counts
}

/** A fixed, realistic-looking set of next-token logits for "The cat sat on the ___". The last three are junk. */
export const CAT_PROMPT = 'The cat sat on the'
export const CAT_TOKENS = ['mat', 'floor', 'sofa', 'bed', 'couch', 'roof', 'table', 'moon', 'zx@!', 'qq9', '###']
export const CAT_LOGITS = [4.0, 3.2, 3.0, 2.7, 2.4, 1.6, 1.4, 0.0, -1.5, -2.0, -2.5]
export const CAT_JUNK_FROM = 8 // index of the first junk token

/* ------------------------------------------------------------------ */
/* A toy next-word table                                                */
/* ------------------------------------------------------------------ */

export const EOS = '<eos>'
export const TOY_VOCAB = ['the', 'a', 'cat', 'dog', 'sat', 'slept', 'on', 'under', 'mat', 'sofa', 'and', '.', EOS, 'zx@!', 'qq9']
export const TOY_JUNK = new Set(['zx@!', 'qq9'])

const UNLIKELY = -3 // logit of any real word the table does not mention
const JUNK = -4 // logit of the junk tokens

/** previous word -> logits for the words that plausibly follow it */
const TABLE: Record<string, Record<string, number>> = {
  the: { cat: 3.0, mat: 2.9, dog: 2.6, sofa: 2.5 },
  a: { cat: 2.5, dog: 2.5, sofa: 2.2, mat: 2.0 },
  cat: { sat: 3.0, slept: 2.5, and: 1.0, '.': 0.5 },
  dog: { slept: 2.8, sat: 2.6, and: 1.0, '.': 0.5 },
  sat: { on: 3.2, under: 2.0, '.': 1.0, and: 0.8 },
  slept: { on: 2.8, under: 2.4, '.': 1.2, and: 0.8 },
  on: { the: 3.2, a: 2.2 },
  under: { the: 3.0, a: 2.4 },
  mat: { '.': 2.8, and: 2.2 },
  sofa: { '.': 3.0, and: 2.0 },
  and: { the: 3.0, a: 2.0, slept: 1.0, sat: 1.0 },
  '.': { [EOS]: 3.0, the: 1.8, a: 1.0 },
}

/** The toy "model": one logit per vocabulary entry, given only the previous word. */
export const toyLogits = (prev: string): Vec => {
  const row = TABLE[prev] ?? TABLE['.']
  return TOY_VOCAB.map((w) => row[w] ?? (TOY_JUNK.has(w) ? JUNK : UNLIKELY))
}

export interface ToyGeneration { tokens: string[]; stopped: 'eos' | 'max-tokens' }

/**
 * The generation loop: logits -> sampling policy -> one token -> append -> repeat.
 * Stops when the model emits the end-of-sequence token, or when maxTokens is reached, whichever comes first.
 */
export const toyGenerate = (prompt: string[], s: SamplingSettings, maxTokens: number, seed: number): ToyGeneration => {
  const rng = makeRng(seed)
  const tokens: string[] = []
  let prev = prompt[prompt.length - 1]
  for (let i = 0; i < maxTokens; i++) {
    const { final } = samplingStages(toyLogits(prev), s)
    const next = TOY_VOCAB[sampleIndex(final, rng.next())]
    if (next === EOS) return { tokens, stopped: 'eos' }
    tokens.push(next)
    prev = next
  }
  return { tokens, stopped: 'max-tokens' }
}

/** Join word tokens into readable text (no space before a full stop). */
export const detokenize = (tokens: string[]): string => tokens.join(' ').replace(/ \./g, '.')
