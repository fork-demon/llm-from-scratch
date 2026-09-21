// KV cache: the size formula, the work counts, and a small port of
// phase3-transformers/kv_cache_demo.py (forward_full vs forward_step) so we can
// TEST that reading keys and values from a cache gives exactly the same output.
import { argmax, matmul, softmax, type Mat, type Vec } from './math'
import { makeRng } from './rng'

/* ------------------------------------------------------------------ */
/* Memory                                                               */
/* ------------------------------------------------------------------ */

export interface KvConfig {
  layers: number
  kvHeads: number // heads that store keys and values (equal to the number of attention heads unless GQA is used)
  headDim: number
  bytesPerNumber: number // fp32 = 4, fp16/bf16 = 2, int8 = 1
}

/** One key vector and one value vector, per head, per layer, for ONE token. */
export const kvBytesPerToken = (c: KvConfig): number => 2 * c.layers * c.kvHeads * c.headDim * c.bytesPerNumber

/** cache bytes = 2 x layers x kv_heads x head_dim x tokens x bytes_per_number */
export const kvCacheBytes = (c: KvConfig, tokens: number): number => kvBytesPerToken(c) * tokens

export const formatBytes = (b: number): string => {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let x = b
  while (x >= 1024 && i < units.length - 1) { x /= 1024; i++ }
  return `${x >= 100 || i === 0 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2)} ${units[i]}`
}

export interface KvPreset extends KvConfig { id: string; name: string; context: number; note: string }

export const KV_PRESETS: KvPreset[] = [
  { id: 'toy', name: 'The repo toy (kv_cache_demo.py)', layers: 2, kvHeads: 4, headDim: 16, bytesPerNumber: 4, context: 123, note: '2 layers, 4 heads of 16 numbers, counted at 4 bytes per number as the file does' },
  { id: 'gpt2', name: 'GPT-2 small', layers: 12, kvHeads: 12, headDim: 64, bytesPerNumber: 2, context: 1024, note: '12 layers, 12 heads of 64, 16-bit numbers' },
  { id: '7b', name: 'A 7B-class model', layers: 32, kvHeads: 32, headDim: 128, bytesPerNumber: 2, context: 4096, note: '32 layers, 32 heads of 128, 16-bit numbers, no GQA (the shape of Llama-2 7B)' },
]

/* ------------------------------------------------------------------ */
/* Work: how many key/value rows get computed (per layer)               */
/* ------------------------------------------------------------------ */

/** Rows of K (and of V) computed, per layer, to produce new token number `i` (1-based). */
export const kvRowsForToken = (promptLen: number, i: number, cached: boolean): number =>
  cached ? (i === 1 ? promptLen : 1) : promptLen + i - 1

/** Total rows computed, per layer, to generate `newTokens` tokens after a prompt. */
export const kvRowsTotal = (promptLen: number, newTokens: number, cached: boolean): number => {
  let total = 0
  for (let i = 1; i <= newTokens; i++) total += kvRowsForToken(promptLen, i, cached)
  return total
}

/* ------------------------------------------------------------------ */
/* A tiny causal Transformer, the same one as kv_cache_demo.py          */
/* ------------------------------------------------------------------ */

export interface ToyConfig { vocab: number; dim: number; heads: number; layers: number; maxLen: number }
export interface ToyLayer { Wq: Mat; Wk: Mat; Wv: Mat; Wo: Mat; ff1: Mat; ff2: Mat }
export interface ToyParams { cfg: ToyConfig; emb: Mat; pos: Mat; out: Mat; layers: ToyLayer[] }

/** Random weights, scaled like make_params(). We are measuring mechanics, not text quality. */
export const makeToyParams = (cfg: ToyConfig, seed = 9): ToyParams => {
  const rng = makeRng(seed)
  const rand = (r: number, c: number, s: number): Mat => Array.from({ length: r }, () => Array.from({ length: c }, () => s * rng.normal()))
  const D = cfg.dim
  return {
    cfg,
    emb: rand(cfg.vocab, D, 0.1),
    pos: rand(cfg.maxLen, D, 0.1),
    out: rand(D, cfg.vocab, 0.1),
    layers: Array.from({ length: cfg.layers }, () => ({
      Wq: rand(D, D, 1 / Math.sqrt(D)), Wk: rand(D, D, 1 / Math.sqrt(D)), Wv: rand(D, D, 1 / Math.sqrt(D)), Wo: rand(D, D, 1 / Math.sqrt(D)),
      ff1: rand(D, 4 * D, 1 / Math.sqrt(D)), ff2: rand(4 * D, D, 1 / Math.sqrt(4 * D)),
    })),
  }
}

const addRows = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((x, j) => x + b[i][j]))
const relu = (m: Mat): Mat => m.map((row) => row.map((x) => Math.max(0, x)))

/**
 * Multi-head attention of `Q` rows over `K`/`V` rows. Each head uses its own slice of the vectors.
 * `firstPos` is the sequence position of Q's first row: row i may look at keys 0 .. firstPos + i (the causal mask).
 */
const multiHead = (Q: Mat, K: Mat, V: Mat, heads: number, firstPos: number): Mat => {
  const hd = Q[0].length / heads
  return Q.map((q, i) => {
    const visible = firstPos + i + 1
    const outRow: Vec = []
    for (let h = 0; h < heads; h++) {
      const lo = h * hd
      const scores: Vec = []
      for (let j = 0; j < visible; j++) {
        let s = 0
        for (let d = 0; d < hd; d++) s += q[lo + d] * K[j][lo + d]
        scores.push(s / Math.sqrt(hd))
      }
      const w = softmax(scores)
      for (let d = 0; d < hd; d++) {
        let s = 0
        for (let j = 0; j < visible; j++) s += w[j] * V[j][lo + d]
        outRow.push(s)
      }
    }
    return outRow
  })
}

/** One K matrix and one V matrix per layer: a row per token seen so far. */
export type KvCache = { K: Mat; V: Mat }[]
export const emptyCache = (p: ToyParams): KvCache => p.layers.map(() => ({ K: [], V: [] }))

/** forward_full: run the WHOLE sequence. Returns the last position's logits, and every K/V row it had to compute. */
export const forwardFull = (p: ToyParams, ids: number[]): { logits: Vec; kv: KvCache } => {
  let x: Mat = ids.map((id, t) => p.emb[id].map((e, d) => e + p.pos[t][d]))
  const kv: KvCache = []
  for (const L of p.layers) {
    const Q = matmul(x, L.Wq)
    const K = matmul(x, L.Wk)
    const V = matmul(x, L.Wv)
    kv.push({ K, V })
    x = addRows(x, matmul(multiHead(Q, K, V, p.cfg.heads, 0), L.Wo))
    x = addRows(x, matmul(relu(matmul(x, L.ff1)), L.ff2))
  }
  return { logits: matmul([x[x.length - 1]], p.out)[0], kv }
}

/** forward_step: run ONE new token. Appends its k and v to the cache, then attends over the cache. */
export const forwardStep = (p: ToyParams, tokenId: number, posIdx: number, cache: KvCache): Vec => {
  let x: Mat = [p.emb[tokenId].map((e, d) => e + p.pos[posIdx][d])]
  p.layers.forEach((L, l) => {
    const q = matmul(x, L.Wq)
    const k = matmul(x, L.Wk)
    const v = matmul(x, L.Wv)
    cache[l].K.push(k[0]) // old rows never change, so we only ever append
    cache[l].V.push(v[0])
    // no mask needed: the cache only CONTAINS the past, so the new token may look at all of it
    x = addRows(x, matmul(multiHead(q, cache[l].K, cache[l].V, p.cfg.heads, cache[l].K.length - 1), L.Wo))
    x = addRows(x, matmul(relu(matmul(x, L.ff1)), L.ff2))
  })
  return matmul(x, p.out)[0]
}

export const generateNaive = (p: ToyParams, prompt: number[], n: number): number[] => {
  const ids = prompt.slice()
  for (let i = 0; i < n; i++) ids.push(argmax(forwardFull(p, ids).logits))
  return ids
}

export const generateCached = (p: ToyParams, prompt: number[], n: number): number[] => {
  const cache = emptyCache(p)
  let logits: Vec = []
  prompt.forEach((t, i) => { logits = forwardStep(p, t, i, cache) }) // prefill
  const ids = prompt.slice()
  for (let i = 0; i < n; i++) { // decode
    ids.push(argmax(logits))
    logits = forwardStep(p, ids[ids.length - 1], ids.length - 1, cache)
  }
  return ids
}

/** Largest absolute difference between two matrices of the same shape. */
export const maxAbsDiff = (a: Mat, b: Mat): number => {
  let worst = 0
  a.forEach((row, i) => row.forEach((x, j) => { worst = Math.max(worst, Math.abs(x - b[i][j])) }))
  return worst
}
