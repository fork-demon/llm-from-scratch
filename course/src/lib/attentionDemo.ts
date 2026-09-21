// Data for the attention playground.
// 1) A hand-built example that mirrors demo_disambiguation() in
//    phase3-transformers/attention_numpy.py: 4-dimensional embeddings whose
//    dimensions we *chose* to mean [watery, financial, noun-ness, ambiguity].
// 2) A way to turn any typed sentence into (meaningless, untrained) vectors so
//    the mechanics can be explored on the learner's own text.
import { makeRng } from './rng'
import type { Mat, Vec } from './math'

export const DIMS = ['watery', 'financial', 'noun', 'ambiguous']

export const HAND_VOCAB: Record<string, Vec> = {
  the: [0, 0, 0.1, 0],
  river: [1, 0, 0.8, 0],
  money: [0, 1, 0.8, 0],
  bank: [0.5, 0.5, 0.9, 1],
}

const zeros = (n: number, m: number): Mat => Array.from({ length: n }, () => Array(m).fill(0))
const eye = (n: number): Mat => zeros(n, n).map((r, i) => r.map((_, j) => (i === j ? 1 : 0)))

/** Ambiguous tokens ask for "watery or financial"; keys advertise exactly those two properties. */
export const handWeights = () => {
  const Wq = zeros(4, 4)
  Wq[3][0] = 2
  Wq[3][1] = 2
  const Wk = zeros(4, 4)
  Wk[0][0] = 2
  Wk[1][1] = 2
  return { Wq, Wk, Wv: eye(4) }
}

export const splitWords = (text: string): string[] =>
  text.toLowerCase().replace(/[^a-z0-9'\s]/g, ' ').split(/\s+/).filter(Boolean).slice(0, 8)

const hash = (s: string): number => {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  return h >>> 0
}

/** The same word always gets the same vector (like a real embedding table), but the numbers are random: untrained. */
export const randomEmbedding = (word: string, d = 4): Vec => {
  const rng = makeRng(hash(word))
  return Array.from({ length: d }, () => Math.round(rng.normal() * 10) / 10)
}

export const randomWeights = (seed: number, d = 4) => {
  const rng = makeRng(seed)
  const m = (): Mat => Array.from({ length: d }, () => Array.from({ length: d }, () => Math.round((rng.normal() / Math.sqrt(d)) * 10) / 10))
  return { Wq: m(), Wk: m(), Wv: m() }
}
