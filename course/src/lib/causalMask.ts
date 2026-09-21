// Data and helpers for the CausalMaskLab. Pure functions, no DOM.
import { attentionFromQKV, softmax, type AttentionResult, type Mat } from './math'
import { makeRng } from './rng'

/** The model reads INPUT and, at every position, must predict the token that follows. */
export const MASK_TOKENS = ['the', 'cat', 'sat', 'on', 'a', 'mat']
export const MASK_NEXT = '.' // what follows the last input token in the training text

const D = 4

/** Seeded, untrained Q / K / V for the sentence: the pattern is meaningless, the mechanics are exact. */
export const randomQKV = (seed = 5, T = MASK_TOKENS.length): { Q: Mat; K: Mat; V: Mat } => {
  const rng = makeRng(seed)
  const m = (): Mat => Array.from({ length: T }, () => Array.from({ length: D }, () => Math.round(rng.normal() * 10) / 10))
  return { Q: m(), K: m(), V: m() }
}

/**
 * What training would discover if nothing stopped it: a Q/K pair in which the query of
 * position t matches the key of position t+1. `strength` = how far training has pushed it.
 * (Position t's query is the one-hot for t+1, scaled; keys are one-hot for their own position.)
 */
export const cheatingQKV = (strength: number, T = MASK_TOKENS.length): { Q: Mat; K: Mat; V: Mat } => {
  const onehot = (i: number) => Array.from({ length: T }, (_, j) => (i === j ? 1 : 0))
  const s = strength * Math.sqrt(T) // undo the division by sqrt(d) so `strength` is the final score
  return {
    Q: Array.from({ length: T }, (_, t) => onehot(t + 1).map((v) => v * s)),
    K: Array.from({ length: T }, (_, t) => onehot(t)),
    V: Array.from({ length: T }, (_, t) => onehot(t)), // value = "I am token number t": lets us read off what was copied
  }
}

export const runMask = (qkv: { Q: Mat; K: Mat; V: Mat }, causal: boolean): AttentionResult => attentionFromQKV(qkv.Q, qkv.K, qkv.V, { causal, scale: true })

/** For every position t that has a successor: how much weight does row t put on t+1, the token it is supposed to predict? */
export const leakOnAnswer = (weights: Mat): number[] => weights.slice(0, -1).map((row, t) => row[t + 1])

/** Total weight a row puts on any future position. Exactly 0 with a causal mask. */
export const futureWeight = (weights: Mat): number[] => weights.map((row, t) => row.slice(t + 1).reduce((a, b) => a + b, 0))

/** One sequence of T tokens is T training examples: (everything up to t) -> token t+1. */
export const trainingExamples = (tokens: string[], next: string): { context: string[]; target: string }[] =>
  tokens.map((_, t) => ({ context: tokens.slice(0, t + 1), target: t + 1 < tokens.length ? tokens[t + 1] : next }))

/* ---------- two classic bugs, used by the debug exercise ---------- */

/** BUG: softmax first, then zero the future. Rows no longer sum to 1. */
export const maskAfterSoftmax = (scores: Mat): Mat => scores.map((row, i) => softmax(row).map((w, j) => (j > i ? 0 : w)))

/** BUG: np.triu(..., k=0) also masks the diagonal, so a token cannot see itself and row 0 sees nothing at all. */
export const maskWithDiagonal = (scores: Mat): Mat => scores.map((row, i) => row.map((s, j) => (j >= i ? -Infinity : s)))
