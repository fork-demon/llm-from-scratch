// Multi-head attention: a port of multi_head_attention() in
// phase3-transformers/attention_numpy.py, plus the hand-made head patterns used
// by the MultiHeadLab. Pure functions, no DOM.
import { attentionFromQKV, matmul, softmax, type Mat } from './math'
import { makeRng } from './rng'

/** (T, D) -> h matrices of (T, D/h): head k gets columns k*hd .. (k+1)*hd. Same as reshape(T, h, hd).transpose(1, 0, 2). */
export const splitHeads = (M: Mat, nHeads: number): Mat[] => {
  const D = M[0].length
  if (D % nHeads !== 0) throw new Error(`splitHeads: D=${D} is not divisible by n_heads=${nHeads}`)
  const hd = D / nHeads
  return Array.from({ length: nHeads }, (_, h) => M.map((row) => row.slice(h * hd, (h + 1) * hd)))
}

/** h matrices of (T, hd) -> (T, h*hd): glue the head outputs side by side. */
export const concatHeads = (heads: Mat[]): Mat => heads[0].map((_, t) => heads.flatMap((h) => h[t]))

export interface MultiHeadResult {
  Q: Mat // (T, D)
  K: Mat
  V: Mat
  Qh: Mat[] // H x (T, hd)
  Kh: Mat[]
  Vh: Mat[]
  scores: Mat[] // H x (T, T) scaled and masked
  weights: Mat[] // H x (T, T) rows sum to 1
  headOut: Mat[] // H x (T, hd)
  concat: Mat // (T, D)
  out: Mat // (T, D) = concat @ Wo
}

/** Everything after the Q/K/V projections. Used directly by block.ts, which adds biases to the projections. */
export const multiHeadFromQKV = (Q: Mat, K: Mat, V: Mat, Wo: Mat, nHeads: number, causal = true): MultiHeadResult => {
  const Qh = splitHeads(Q, nHeads)
  const Kh = splitHeads(K, nHeads)
  const Vh = splitHeads(V, nHeads)
  // attentionFromQKV divides by sqrt(width of the keys it is given) = sqrt(hd), as the Python does
  const per = Qh.map((q, h) => attentionFromQKV(q, Kh[h], Vh[h], { causal, scale: true }))
  const headOut = per.map((r) => r.out)
  const concat = concatHeads(headOut)
  return { Q, K, V, Qh, Kh, Vh, scores: per.map((r) => r.scores), weights: per.map((r) => r.weights), headOut, concat, out: matmul(concat, Wo) }
}

/** Same signature and maths as attention_numpy.py::multi_head_attention. */
export const multiHeadAttention = (x: Mat, Wq: Mat, Wk: Mat, Wv: Mat, Wo: Mat, nHeads: number, causal = true): MultiHeadResult =>
  multiHeadFromQKV(matmul(x, Wq), matmul(x, Wk), matmul(x, Wv), Wo, nHeads, causal)

export const identity = (n: number): Mat => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))

/** Seeded random (rows, cols) matrix with entries ~ N(0, 1/rows), rounded so the UI can print it exactly. */
export const randomMat = (rng: ReturnType<typeof makeRng>, rows: number, cols: number, digits = 1): Mat => {
  const f = 10 ** digits
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.round((rng.normal() / Math.sqrt(rows)) * f) / f))
}

/** The small fixed example of the "shapes" view: T = 3 tokens, D = 4 numbers each. */
export const shapesDemo = (seed = 11) => {
  const rng = makeRng(seed)
  const x: Mat = [
    [1, 0, 0.5, -0.5],
    [0, 1, -0.5, 0.5],
    [0.5, 0.5, 1, 0],
  ]
  // entries ~ N(0, 1): larger than a real initialisation, so that the heads visibly disagree
  const big = (): Mat => randomMat(rng, 4, 4).map((r) => r.map((v) => Math.round(v * 20) / 10))
  return { tokens: ['the', 'river', 'bank'], x, Wq: big(), Wk: big(), Wv: randomMat(rng, 4, 4), Wo: randomMat(rng, 4, 4) }
}

/* ---------- hand-made, ILLUSTRATIVE head patterns ---------- */
export const HEAD_SENTENCE = ['The', 'animal', "didn't", 'cross', 'the', 'road', 'because', 'it', 'was', 'tired']

export interface IllustrativeHead { id: string; name: string; asks: string; note: string; weights: Mat }

/** Turn a hand-written table of scores into causal attention weights (mask, then softmax), exactly like the real thing. */
export const causalWeightsFromScores = (scores: Mat): Mat =>
  scores.map((row, i) => softmax(row.map((s, j) => (j > i ? -Infinity : s))))

/**
 * Four hand-written score tables. They are NOT taken from a trained model. They show the
 * kinds of pattern one head can hold, so the learner can see why one pattern is not enough.
 */
export const illustrativeHeads = (): IllustrativeHead[] => {
  const T = HEAD_SENTENCE.length
  const table = (f: (i: number, j: number) => number): Mat => Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => f(i, j)))
  // "who or what am I about?" : a few hand-picked links, everything else looks at itself
  const refers: Record<number, Record<number, number>> = {
    1: { 0: 2.5 }, // animal -> The
    3: { 1: 3 }, // cross -> animal (who crosses?)
    5: { 4: 2.5, 3: 1.5 }, // road -> the, cross
    7: { 1: 4, 5: 2 }, // it -> animal (mostly), road (a little)
    8: { 7: 2, 1: 2.5 }, // was -> it, animal
    9: { 1: 3.5, 7: 2.5 }, // tired -> animal, it
  }
  return [
    { id: 'prev', name: 'Head 1: previous token', asks: 'What came just before me?', note: 'Every token puts most of its weight on its left neighbour.', weights: causalWeightsFromScores(table((i, j) => (j === i - 1 ? 4 : 0))) },
    { id: 'refers', name: 'Head 2: what do I refer to?', asks: 'Which earlier word am I about?', note: '“it” puts most of its weight on “animal”.', weights: causalWeightsFromScores(table((i, j) => refers[i]?.[j] ?? (i === j ? 1 : 0))) },
    { id: 'first', name: 'Head 3: first token', asks: 'Nothing in particular', note: 'Everything looks at the first token. Trained models often contain heads like this: a place to put weight when the head has nothing useful to fetch.', weights: causalWeightsFromScores(table((_, j) => (j === 0 ? 3.5 : 0))) },
    { id: 'broad', name: 'Head 4: broad context', asks: 'What is the general topic so far?', note: 'Equal scores, so equal weights: a plain average of everything seen so far.', weights: causalWeightsFromScores(table(() => 0)) },
  ]
}

/** What a single head would have to do if it tried to hold all four patterns at once: average them. */
export const averagePatterns = (heads: Mat[]): Mat => heads[0].map((row, i) => row.map((_, j) => heads.reduce((s, h) => s + h[i][j], 0) / heads.length))
