// Core numeric helpers shared by every interactive. Pure functions, no DOM.
// Conventions mirror the repo's NumPy code: a matrix is number[][] (rows),
// a token sequence is (T, D): T rows of D numbers.

export type Vec = number[]
export type Mat = number[][]

export const dot = (a: Vec, b: Vec): number => {
  if (a.length !== b.length) throw new Error(`dot: length mismatch ${a.length} vs ${b.length}`)
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

export const norm = (a: Vec): number => Math.sqrt(dot(a, a))

/** Cosine similarity: the dot product with both lengths divided out. 0 if either vector is all zeros. */
export const cosine = (a: Vec, b: Vec): number => {
  const d = norm(a) * norm(b)
  return d === 0 ? 0 : dot(a, b) / d
}

export const add = (a: Vec, b: Vec): Vec => a.map((v, i) => v + b[i])
export const scale = (a: Vec, k: number): Vec => a.map((v) => v * k)

export const shape = (m: Mat): [number, number] => [m.length, m[0]?.length ?? 0]

export const transpose = (m: Mat): Mat => m[0].map((_, j) => m.map((row) => row[j]))

/** (n, k) @ (k, m) -> (n, m). Cell (i, j) = row i of A dotted with column j of B. */
export const matmul = (a: Mat, b: Mat): Mat => {
  const [, k1] = shape(a)
  const [k2] = shape(b)
  if (k1 !== k2) throw new Error(`matmul: inner dimensions differ (${k1} vs ${k2})`)
  const bt = transpose(b)
  return a.map((row) => bt.map((col) => dot(row, col)))
}

/**
 * Softmax with temperature. Subtracts the max first so exp() never overflows;
 * the result is unchanged because softmax only cares about differences.
 * -Infinity logits get probability exactly 0 (that is how masking works).
 */
export const softmax = (logits: Vec, temperature = 1): Vec => {
  if (temperature <= 0) {
    // the limit T -> 0 is argmax
    const best = logits.indexOf(Math.max(...logits))
    return logits.map((_, i) => (i === best ? 1 : 0))
  }
  const scaled = logits.map((z) => z / temperature)
  const max = Math.max(...scaled)
  const exps = scaled.map((z) => (z === -Infinity ? 0 : Math.exp(z - max)))
  const sum = exps.reduce((s, v) => s + v, 0)
  return exps.map((e) => e / sum)
}

/** Cross-entropy for one example: minus the log of the probability given to the correct answer. */
export const crossEntropy = (probs: Vec, target: number): number => -Math.log(probs[target] + 1e-12)

export interface AttentionResult {
  Q: Mat
  K: Mat
  V: Mat
  raw: Mat // Q K^T
  scores: Mat // after scaling and masking
  weights: Mat // after softmax: rows sum to 1
  out: Mat // weights @ V
}

/**
 * Single-head self-attention, step by step. Same maths as
 * phase3-transformers/attention_numpy.py::attention.
 */
export const attention = (x: Mat, Wq: Mat, Wk: Mat, Wv: Mat, opts: { causal?: boolean; scale?: boolean } = {}): AttentionResult => {
  const { causal = false, scale: doScale = true } = opts
  const Q = matmul(x, Wq)
  const K = matmul(x, Wk)
  const V = matmul(x, Wv)
  return attentionFromQKV(Q, K, V, { causal, scale: doScale })
}

export const attentionFromQKV = (Q: Mat, K: Mat, V: Mat, opts: { causal?: boolean; scale?: boolean } = {}): AttentionResult => {
  const { causal = false, scale: doScale = true } = opts
  const d = K[0].length
  const raw = matmul(Q, transpose(K))
  const scores = raw.map((row, i) =>
    row.map((s, j) => (causal && j > i ? -Infinity : doScale ? s / Math.sqrt(d) : s)),
  )
  const weights = scores.map((row) => softmax(row))
  const out = matmul(weights, V)
  return { Q, K, V, raw, scores, weights, out }
}

/** Keep the k largest probabilities, zero the rest, renormalise. */
export const topK = (probs: Vec, k: number): Vec => {
  if (k >= probs.length) return probs.slice()
  const order = probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0])
  const keep = new Set(order.slice(0, Math.max(1, k)).map(([, i]) => i))
  return renormalise(probs.map((p, i) => (keep.has(i) ? p : 0)))
}

/** Nucleus sampling: keep the smallest set of tokens whose probabilities add up to at least p. */
export const topP = (probs: Vec, p: number): Vec => {
  const order = probs.map((pr, i) => [pr, i] as const).sort((a, b) => b[0] - a[0])
  const keep = new Set<number>()
  let cum = 0
  for (const [pr, i] of order) {
    keep.add(i)
    cum += pr
    if (cum >= p - 1e-12) break
  }
  return renormalise(probs.map((pr, i) => (keep.has(i) ? pr : 0)))
}

export const renormalise = (v: Vec): Vec => {
  const s = v.reduce((a, b) => a + b, 0)
  return s === 0 ? v.slice() : v.map((x) => x / s)
}

/** Draw an index from a probability distribution. `u` is a uniform random number in [0, 1). */
export const sampleIndex = (probs: Vec, u: number): number => {
  let cum = 0
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i]
    if (u < cum) return i
  }
  return probs.length - 1
}

export const argmax = (v: Vec): number => v.indexOf(Math.max(...v))

export const round = (x: number, digits = 2): number => {
  const f = 10 ** digits
  return Math.round(x * f) / f
}

/** Format a number for display in a matrix cell. */
export const fmt = (x: number, digits = 2): string => {
  if (x === -Infinity) return '−∞'
  if (x === Infinity) return '∞'
  const r = round(x, digits)
  return (Object.is(r, -0) ? 0 : r).toFixed(digits).replace('-', '−')
}
