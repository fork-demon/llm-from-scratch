// Toy data for the "prompt to answer" pipeline explorer (lesson 0.1).
// Everything here is a hand-written stand-in: an 18-entry vocabulary, 4-number
// embeddings and a small table of next-token probabilities. A real LLM computes
// those probabilities with a Transformer; the *loop around them* is the real one.
import { argmax, sampleIndex, type Mat, type Vec } from './math'
import { makeRng } from './rng'

export const PROMPT = 'What is a cat?'

/** id -> token text. Tokens carry their leading space, as real tokenizers do. */
export const VOCAB: string[] = [
  '<end>', // 0: special token meaning "the answer is finished"
  'What', // 1
  ' is', // 2
  ' a', // 3
  ' cat', // 4
  '?', // 5
  ' A', // 6
  ' small', // 7
  ' furry', // 8
  ' animal', // 9
  '.', // 10
  ' pet', // 11
  ' dog', // 12
  ' It', // 13
  ' that', // 14
  ' purrs', // 15
  ' mammal', // 16
  '<unk>', // 17: anything the tokenizer has never seen
]
export const END = 0
export const UNK = 17

const ID_OF = new Map(VOCAB.map((t, i) => [t, i]))
export const idOf = (token: string): number => ID_OF.get(token) ?? UNK

/**
 * Toy tokenizer: a word keeps the space in front of it, punctuation stands alone.
 * (Real tokenizers learn their pieces from data: see the Tokenization lesson.)
 */
export const tokenize = (text: string): string[] => text.match(/ ?[A-Za-z]+|[^A-Za-z\s]/g) ?? []

export const encode = (text: string): number[] => tokenize(text).map(idOf)
export const decode = (ids: number[]): string => ids.filter((i) => i !== END).map((i) => VOCAB[i]).join('')

/** Hand-made embedding table: one row of 4 numbers per vocabulary entry. */
export const EMBED_DIMS = ['living thing', 'describes', 'grammar glue', 'punctuation'] as const
export const EMBEDDINGS: Mat = [
  [0.0, 0.0, 0.0, 1.0], // <end>
  [0.0, 0.1, 0.9, 0.0], // What
  [0.0, 0.0, 1.0, 0.0], // is
  [0.0, 0.1, 0.9, 0.0], // a
  [1.0, 0.2, 0.0, 0.0], // cat
  [0.0, 0.0, 0.2, 0.9], // ?
  [0.0, 0.1, 0.9, 0.0], // A
  [0.1, 0.9, 0.0, 0.0], // small
  [0.3, 0.9, 0.0, 0.0], // furry
  [0.9, 0.1, 0.0, 0.0], // animal
  [0.0, 0.0, 0.1, 1.0], // .
  [0.9, 0.2, 0.0, 0.0], // pet
  [1.0, 0.1, 0.0, 0.0], // dog
  [0.4, 0.0, 0.7, 0.0], // It
  [0.0, 0.0, 1.0, 0.0], // that
  [0.6, 0.6, 0.0, 0.0], // purrs
  [0.9, 0.1, 0.0, 0.0], // mammal
  [0.0, 0.0, 0.0, 0.0], // <unk>
]
export const embed = (ids: number[]): Mat => ids.map((i) => EMBEDDINGS[i])

/**
 * The stand-in for the Transformer: next-token probabilities keyed by the last
 * one or two tokens. Two-token keys win over one-token keys.
 */
const TABLE: Record<string, Record<string, number>> = {
  '?': { ' A': 0.7, ' It': 0.3 },
  ' A': { ' cat': 0.92, ' dog': 0.08 },
  ' cat': { ' is': 0.8, ' purrs': 0.2 },
  ' dog': { ' is': 1 },
  ' is': { ' a': 0.9, ' furry': 0.1 },
  ' is| furry': { '.': 1 },
  ' a': { ' small': 0.5, ' furry': 0.3, ' pet': 0.1, ' mammal': 0.1 },
  ' small': { ' furry': 0.5, ' animal': 0.2, ' pet': 0.2, ' mammal': 0.1 },
  ' furry': { ' animal': 0.5, ' pet': 0.3, ' mammal': 0.2 },
  ' animal': { '.': 0.7, ' that': 0.3 },
  ' pet': { '.': 0.7, ' that': 0.3 },
  ' mammal': { '.': 0.7, ' that': 0.3 },
  ' that': { ' purrs': 1 },
  ' purrs': { '.': 1 },
  '.': { '<end>': 0.8, ' It': 0.2 },
  ' It': { ' is': 0.7, ' purrs': 0.3 },
}

/** Probability for every vocabulary entry, given all the token ids so far. */
export const nextDistribution = (ids: number[]): Vec => {
  const last = VOCAB[ids[ids.length - 1]]
  const prev = VOCAB[ids[ids.length - 2]]
  const row = TABLE[`${prev}|${last}`] ?? TABLE[last] ?? { '<end>': 1 }
  const probs = VOCAB.map(() => 0)
  for (const [tok, p] of Object.entries(row)) probs[idOf(tok)] = p
  return probs
}

/**
 * Scores ("logits") that softmax turns back into `probs`. In a real model the
 * Transformer produces the logits and the probabilities follow; our toy goes
 * the other way round so the two stages show consistent numbers.
 */
export const toyLogits = (probs: Vec): Vec => {
  const logs = probs.filter((p) => p > 0).map(Math.log)
  const shift = 1 - Math.min(...logs) // smallest possible token gets a score of 1
  return probs.map((p) => (p > 0 ? Math.log(p) + shift : -Infinity))
}

/** The u-th uniform random number of a seeded stream: the same seed replays the same answer. */
export const uniformAt = (seed: number, n: number): number => {
  const rng = makeRng(seed)
  let u = rng.next()
  for (let i = 0; i < n; i++) u = rng.next()
  return u
}

export type PickMode = 'sample' | 'greedy'
export interface StepResult {
  probs: Vec
  u: number | null // the random number used (null for greedy)
  next: number // chosen token id
}

/** One turn of the loop: probabilities for the next token, then pick one. */
export const generateStep = (ids: number[], mode: PickMode, u: number): StepResult => {
  const probs = nextDistribution(ids)
  const next = mode === 'greedy' ? argmax(probs) : sampleIndex(probs, u)
  return { probs, u: mode === 'greedy' ? null : u, next }
}

export const MAX_NEW_TOKENS = 24

/** Run the whole loop. Returns prompt ids plus everything generated (including <end>, if reached). */
export const generateAll = (promptIds: number[], mode: PickMode, seed: number, maxNew = MAX_NEW_TOKENS): number[] => {
  const ids = promptIds.slice()
  for (let n = 0; n < maxNew; n++) {
    const { next } = generateStep(ids, mode, uniformAt(seed, n))
    ids.push(next)
    if (next === END) break
  }
  return ids
}
