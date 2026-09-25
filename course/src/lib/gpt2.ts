// REAL GPT-2 small (124M parameters) in the browser, over int8 weights.
// Weights: public/models/gpt2/, written by phase6-engineering/export_gpt2.py (int8 with one float32 scale per
// output row; LayerNorm and biases float32; 8 outlier columns of the tied embedding kept in float32).
//
// This is the same computation as trainedGpt.ts (pre-norm blocks, learned positions, tied head), at full size:
//   x = wte[ids] + wpe[0..T-1]
//   per block: x = x + c_proj(causal_attention(ln_1(x)));  x = x + mlp_proj(gelu(c_fc(ln_2(x))))
//   logits = ln_f(x) @ wte^T
// Two things are different because the model is 1,000 times bigger:
//   1. Weights stay int8 in memory. A matrix-vector product reads the int8 row and multiplies by the row's
//      scale once at the end: sum_i x_i * (q_i * s) = s * sum_i x_i * q_i. Nothing is expanded to float32,
//      so memory stays close to the 127 MB download.
//   2. A KV cache. Keys and values of earlier positions never change (causal mask), so extend() only
//      computes the NEW positions and reuses the stored K and V. That is also what makes the trace cheap to
//      grow token by token: every intermediate of an old position is already final.
// Pure functions, no DOM. The page runs them in a Web Worker (gpt2.worker.ts).

/* ---------------- weights ---------------- */

export interface Gpt2Config { vocabSize: number; nCtx: number; nEmbd: number; nHead: number; nLayer: number; eps: number }

export interface ManifestTensor { name: string; dtype: 'int8' | 'float32'; shape: number[]; chunk: number; offset: number; scale_offset?: number; columns?: number[] }
export interface Gpt2Manifest {
  format: string
  source: string
  license: string
  quantization: string
  config: { vocab_size: number; n_ctx: number; n_embd: number; n_head: number; n_layer: number; layer_norm_epsilon: number; activation: string; tied_head: boolean }
  n_params: number
  chunks: { file: string; bytes: number; sha256: string }[]
  total_bytes: number
  tokenizer: { file: string; bytes: number; n_merges: number; eot_id: number }
  tensors: ManifestTensor[]
}

/** An int8 matrix, (rows, cols) row-major, value = q * scale[row]. Optional float32 outlier columns. */
export interface QMat {
  rows: number
  cols: number
  q: Int8Array
  scale: Float32Array
  outCols?: Int32Array
  outVals?: Float32Array // (rows, outCols.length)
}
export interface Norm { g: Float32Array; b: Float32Array }
export interface Gpt2Block {
  ln1: Norm
  attnW: QMat // (3D, D): rows 0..D-1 make q, D..2D-1 make k, 2D..3D-1 make v
  attnB: Float32Array
  projW: QMat // (D, D)
  projB: Float32Array
  ln2: Norm
  fcW: QMat // (4D, D)
  fcB: Float32Array
  fc2W: QMat // (D, 4D)
  fc2B: Float32Array
}
export interface Gpt2Model {
  cfg: Gpt2Config
  nParams: number
  wte: QMat // (V, D): the token embedding AND the output head (tied)
  wpe: QMat // (nCtx, D)
  blocks: Gpt2Block[]
  lnF: Norm
}

/** Build the model as views into the downloaded chunks (no copies). Checks every shape. */
export const buildGpt2 = (man: Gpt2Manifest, chunks: ArrayBuffer[]): Gpt2Model => {
  const c = man.config
  const cfg: Gpt2Config = { vocabSize: c.vocab_size, nCtx: c.n_ctx, nEmbd: c.n_embd, nHead: c.n_head, nLayer: c.n_layer, eps: c.layer_norm_epsilon }
  if (c.activation !== 'gelu_new') throw new Error(`unsupported activation ${c.activation}`)
  if (chunks.length !== man.chunks.length) throw new Error(`expected ${man.chunks.length} weight chunks, got ${chunks.length}`)
  chunks.forEach((b, i) => { if (b.byteLength !== man.chunks[i].bytes) throw new Error(`chunk ${i} has ${b.byteLength} bytes, expected ${man.chunks[i].bytes}`) })
  const byName = new Map(man.tensors.map((t) => [t.name, t]))
  const D = cfg.nEmbd
  const entry = (name: string, shape: number[], dtype: string) => {
    const t = byName.get(name)
    if (!t) throw new Error(`checkpoint is missing tensor ${name}`)
    if (t.shape.join(',') !== shape.join(',')) throw new Error(`${name}: shape ${t.shape} but expected ${shape}`)
    if (t.dtype !== dtype) throw new Error(`${name}: ${t.dtype} but expected ${dtype}`)
    return t
  }
  const f32 = (name: string, shape: number[]) => {
    const t = entry(name, shape, 'float32')
    return new Float32Array(chunks[t.chunk], t.offset, shape.reduce((a, b) => a * b, 1))
  }
  const q8 = (name: string, rows: number, cols: number): QMat => {
    const t = entry(name, [rows, cols], 'int8')
    const m: QMat = { rows, cols, q: new Int8Array(chunks[t.chunk], t.offset, rows * cols), scale: new Float32Array(chunks[t.chunk], t.scale_offset!, rows) }
    const o = byName.get(`${name}.outliers`)
    if (o) {
      m.outCols = Int32Array.from(o.columns!)
      m.outVals = f32(`${name}.outliers`, [rows, o.columns!.length])
    }
    return m
  }
  const norm = (p: string): Norm => ({ g: f32(`${p}.weight`, [D]), b: f32(`${p}.bias`, [D]) })
  const blocks = Array.from({ length: cfg.nLayer }, (_, i): Gpt2Block => {
    const p = `h.${i}.`
    return {
      ln1: norm(p + 'ln_1'),
      attnW: q8(p + 'attn.c_attn.weight', 3 * D, D),
      attnB: f32(p + 'attn.c_attn.bias', [3 * D]),
      projW: q8(p + 'attn.c_proj.weight', D, D),
      projB: f32(p + 'attn.c_proj.bias', [D]),
      ln2: norm(p + 'ln_2'),
      fcW: q8(p + 'mlp.c_fc.weight', 4 * D, D),
      fcB: f32(p + 'mlp.c_fc.bias', [4 * D]),
      fc2W: q8(p + 'mlp.c_proj.weight', D, 4 * D),
      fc2B: f32(p + 'mlp.c_proj.bias', [D]),
    }
  })
  return { cfg, nParams: man.n_params, wte: q8('wte', cfg.vocabSize, D), wpe: q8('wpe', cfg.nCtx, D), blocks, lnF: norm('ln_f') }
}

/* ---------------- kernels ---------------- */

/** Dequantise one row into out[off..off+cols). */
export const rowOf = (m: QMat, r: number, out: Float32Array, off = 0): Float32Array => {
  const { q, cols } = m
  const s = m.scale[r]
  const base = r * cols
  for (let i = 0; i < cols; i++) out[off + i] = q[base + i] * s
  if (m.outCols && m.outVals) {
    const k = m.outCols.length
    for (let j = 0; j < k; j++) out[off + m.outCols[j]] = m.outVals[r * k + j]
  }
  return out
}

/**
 * out[t, o] = b[o] + scale[o] * sum_i x[t, i] * q[o, i]   for T rows of x (T x cols) -> (T x rows).
 * The loop reads each int8 weight row once and uses it for up to 4 positions at a time (register blocking),
 * then applies the row's scale once. Accumulators are JS doubles.
 */
export const linearQ = (x: Float32Array, T: number, W: QMat, b: Float32Array | null, out: Float32Array = new Float32Array(T * W.rows)): Float32Array => {
  const { rows: R, cols: C, q, scale } = W
  for (let o = 0; o < R; o++) {
    const wo = o * C
    const sc = scale[o]
    const bo = b ? b[o] : 0
    let t = 0
    for (; t + 3 < T; t += 4) {
      const x0 = t * C, x1 = x0 + C, x2 = x1 + C, x3 = x2 + C
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0
      for (let i = 0; i < C; i++) {
        const w = q[wo + i]
        s0 += x[x0 + i] * w
        s1 += x[x1 + i] * w
        s2 += x[x2 + i] * w
        s3 += x[x3 + i] * w
      }
      out[t * R + o] = bo + sc * s0
      out[(t + 1) * R + o] = bo + sc * s1
      out[(t + 2) * R + o] = bo + sc * s2
      out[(t + 3) * R + o] = bo + sc * s3
    }
    for (; t < T; t++) {
      const x0 = t * C
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0
      let i = 0
      for (; i + 3 < C; i += 4) {
        s0 += x[x0 + i] * q[wo + i]
        s1 += x[x0 + i + 1] * q[wo + i + 1]
        s2 += x[x0 + i + 2] * q[wo + i + 2]
        s3 += x[x0 + i + 3] * q[wo + i + 3]
      }
      for (; i < C; i++) s0 += x[x0 + i] * q[wo + i]
      out[t * R + o] = bo + sc * (s0 + s1 + s2 + s3)
    }
  }
  return out
}

/** The tied head for ONE position: logits[v] = h . wte[v] (int8 part + the float32 outlier columns). */
export const headLogits = (m: Gpt2Model, h: Float32Array, hOff = 0, out = new Float32Array(m.cfg.vocabSize)): Float32Array => {
  const W = m.wte
  const x = hOff ? h.subarray(hOff, hOff + W.cols) : h
  linearQ(x, 1, W, null, out)
  if (W.outCols && W.outVals) {
    const k = W.outCols.length
    const hx = Array.from(W.outCols, (c) => x[c])
    for (let v = 0; v < W.rows; v++) {
      let s = 0
      for (let j = 0; j < k; j++) s += W.outVals[v * k + j] * hx[j]
      out[v] += s
    }
  }
  return out
}

/** nn.LayerNorm over rows t0..t1 of x (biased variance, eps from the config). */
export const layerNorm = (x: Float32Array, t0: number, t1: number, D: number, n: Norm, eps: number, out: Float32Array, outT0 = t0): Float32Array => {
  for (let t = t0; t < t1; t++) {
    const o = t * D
    let mean = 0
    for (let i = 0; i < D; i++) mean += x[o + i]
    mean /= D
    let v = 0
    for (let i = 0; i < D; i++) { const d = x[o + i] - mean; v += d * d }
    const inv = 1 / Math.sqrt(v / D + eps)
    const oo = (t - t0 + outT0) * D
    for (let i = 0; i < D; i++) out[oo + i] = (x[o + i] - mean) * inv * n.g[i] + n.b[i]
  }
  return out
}

const GELU_C = Math.sqrt(2 / Math.PI)
/** GPT-2's GELU ("gelu_new"): the tanh approximation, 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). */
export const geluNew = (x: number): number => 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)))

/* ---------------- the running state: KV cache + trace ---------------- */

/** Everything computed for one layer, for positions 0..T-1. Row-major; stride = the state's capacity where noted. */
export interface LayerTrace {
  ln1: Float32Array // (T, D)
  q: Float32Array // (T, D), head h = columns h*64 .. h*64+63
  k: Float32Array // (T, D)  <- the KV cache
  v: Float32Array // (T, D)  <- the KV cache
  scores: Float32Array // (H, cap, cap): q.k / sqrt(64) for s <= t (entries s > t are masked and never used)
  attn: Float32Array // (H, cap, cap): softmax of the scores along s <= t; 0 above the diagonal
  heads: Float32Array // (T, D): the 12 head outputs side by side, before the output projection
  attnOut: Float32Array // (T, D): after the output projection c_proj
  resid1: Float32Array // (T, D): x + attnOut
  ln2: Float32Array // (T, D)
  act: Float32Array // (T, 4D): MLP hidden layer after GELU
  mlpOut: Float32Array // (T, D)
  resid2: Float32Array // (T, D): the block's output
}

export interface Gpt2State {
  m: Gpt2Model
  ids: number[]
  cap: number
  /** Most positions this state will ever hold (sizes the buffers). */
  limit: number
  /** true: keep every intermediate (for the explainer). false: keep only K and V (plain generation). */
  capture: boolean
  tokEmb: Float32Array // (T, D)
  posEmb: Float32Array // (T, D)
  x0: Float32Array // (T, D): tokEmb + posEmb, the input to block 1
  layers: LayerTrace[]
  lnF: Float32Array // (T, D)
}

const grow = (a: Float32Array, n: number): Float32Array => {
  if (a.length >= n) return a
  const b = new Float32Array(n)
  b.set(a)
  return b
}
/** Re-lay an (H, oldCap, oldCap) array into (H, newCap, newCap). */
const growSquare = (a: Float32Array, H: number, oldCap: number, newCap: number): Float32Array => {
  const b = new Float32Array(H * newCap * newCap)
  for (let h = 0; h < H; h++) for (let t = 0; t < oldCap; t++) b.set(a.subarray((h * oldCap + t) * oldCap, (h * oldCap + t + 1) * oldCap), (h * newCap + t) * newCap)
  return b
}

const emptyLayer = (): LayerTrace => {
  const z = () => new Float32Array(0)
  return { ln1: z(), q: z(), k: z(), v: z(), scores: z(), attn: z(), heads: z(), attnOut: z(), resid1: z(), ln2: z(), act: z(), mlpOut: z(), resid2: z() }
}

export const newState = (m: Gpt2Model, capture = true, limit = m.cfg.nCtx): Gpt2State => ({
  m, ids: [], cap: 0, limit: Math.min(limit, m.cfg.nCtx), capture,
  tokEmb: new Float32Array(0), posEmb: new Float32Array(0), x0: new Float32Array(0),
  layers: m.blocks.map(emptyLayer), lnF: new Float32Array(0),
})

const ensureCapacity = (s: Gpt2State, T: number) => {
  if (T <= s.cap) return
  const { nEmbd: D, nHead: H } = s.m.cfg
  const cap = Math.min(s.limit, Math.max(T, s.cap * 2, 16))
  const n = cap * D
  s.tokEmb = grow(s.tokEmb, n)
  s.posEmb = grow(s.posEmb, n)
  s.x0 = grow(s.x0, n)
  s.lnF = grow(s.lnF, n)
  for (const L of s.layers) {
    L.k = grow(L.k, n)
    L.v = grow(L.v, n)
    if (s.capture) {
      for (const key of ['ln1', 'q', 'heads', 'attnOut', 'resid1', 'ln2', 'mlpOut', 'resid2'] as const) L[key] = grow(L[key], n)
      L.act = grow(L.act, 4 * n)
      L.scores = s.cap ? growSquare(L.scores, H, s.cap, cap) : new Float32Array(H * cap * cap)
      L.attn = s.cap ? growSquare(L.attn, H, s.cap, cap) : new Float32Array(H * cap * cap)
    }
  }
  s.cap = cap
}

/**
 * Run GPT-2 on `newIds`, appended after the tokens already in the state (the KV cache).
 * Returns the logits at the LAST position. Every intermediate is kept in the state when capture is on.
 */
export const extend = (s: Gpt2State, newIds: number[]): Float32Array => {
  const { m } = s
  const { nEmbd: D, nHead: H, vocabSize: V, nCtx, eps } = m.cfg
  const hd = D / H
  const scale = 1 / Math.sqrt(hd)
  const T0 = s.ids.length
  const n = newIds.length
  const T = T0 + n
  if (n === 0) throw new Error('extend: no new tokens')
  if (T > nCtx) throw new Error(`GPT-2 reads at most ${nCtx} tokens; this would be ${T}`)
  if (T > s.limit) throw new Error(`this state holds at most ${s.limit} tokens; this would be ${T}`)
  for (const id of newIds) if (!(Number.isInteger(id) && id >= 0 && id < V)) throw new Error(`token id ${id} is outside the vocabulary`)
  ensureCapacity(s, T)
  const cap = s.cap

  // ---- embeddings for the new positions ----
  let x = new Float32Array(n * D) // the residual stream of the NEW positions only
  const row = new Float32Array(D)
  for (let j = 0; j < n; j++) {
    const t = T0 + j
    rowOf(m.wte, newIds[j], s.tokEmb, t * D)
    rowOf(m.wpe, t, s.posEmb, t * D)
    for (let i = 0; i < D; i++) x[j * D + i] = s.x0[t * D + i] = s.tokEmb[t * D + i] + s.posEmb[t * D + i]
  }

  const h = new Float32Array(n * D)
  const qkv = new Float32Array(n * 3 * D)
  const y = new Float32Array(n * D)
  const a = new Float32Array(n * D)
  const f = new Float32Array(n * 4 * D)
  const f2 = new Float32Array(n * D)
  const sc = new Float64Array(T)

  for (let l = 0; l < m.blocks.length; l++) {
    const b = m.blocks[l]
    const L = s.layers[l]
    // ---- attention: every new position looks at itself and all earlier positions ----
    layerNorm(x, 0, n, D, b.ln1, eps, h)
    linearQ(h, n, b.attnW, b.attnB, qkv)
    for (let j = 0; j < n; j++) {
      const t = T0 + j
      L.k.set(qkv.subarray(j * 3 * D + D, j * 3 * D + 2 * D), t * D)
      L.v.set(qkv.subarray(j * 3 * D + 2 * D, j * 3 * D + 3 * D), t * D)
      if (s.capture) { L.ln1.set(h.subarray(j * D, (j + 1) * D), t * D); L.q.set(qkv.subarray(j * 3 * D, j * 3 * D + D), t * D) }
    }
    for (let j = 0; j < n; j++) {
      const t = T0 + j
      const qo = j * 3 * D
      for (let hh = 0; hh < H; hh++) {
        const ho = hh * hd
        let max = -Infinity
        for (let u = 0; u <= t; u++) {
          let d = 0
          const ko = u * D + ho
          for (let i = 0; i < hd; i++) d += qkv[qo + ho + i] * L.k[ko + i]
          sc[u] = d * scale
          if (sc[u] > max) max = sc[u]
        }
        const rowOff = (hh * cap + t) * cap
        if (s.capture) for (let u = 0; u <= t; u++) L.scores[rowOff + u] = sc[u]
        let z = 0
        for (let u = 0; u <= t; u++) { sc[u] = Math.exp(sc[u] - max); z += sc[u] }
        for (let u = 0; u <= t; u++) sc[u] /= z // u > t is never touched: the causal mask
        if (s.capture) for (let u = 0; u <= t; u++) L.attn[rowOff + u] = sc[u]
        for (let i = 0; i < hd; i++) {
          let acc = 0
          for (let u = 0; u <= t; u++) acc += sc[u] * L.v[u * D + ho + i]
          y[j * D + ho + i] = acc
        }
      }
    }
    linearQ(y, n, b.projW, b.projB, a)
    for (let i = 0; i < n * D; i++) x[i] += a[i]
    if (s.capture) { L.heads.set(y, T0 * D); L.attnOut.set(a, T0 * D); L.resid1.set(x, T0 * D) }

    // ---- MLP: every position on its own, 768 -> 3072 -> GELU -> 768 ----
    layerNorm(x, 0, n, D, b.ln2, eps, h)
    linearQ(h, n, b.fcW, b.fcB, f)
    for (let i = 0; i < f.length; i++) f[i] = geluNew(f[i])
    linearQ(f, n, b.fc2W, b.fc2B, f2)
    const nx = new Float32Array(n * D)
    for (let i = 0; i < n * D; i++) nx[i] = x[i] + f2[i]
    x = nx
    if (s.capture) { L.ln2.set(h, T0 * D); L.act.set(f, T0 * 4 * D); L.mlpOut.set(f2, T0 * D); L.resid2.set(x, T0 * D) }
  }
  layerNorm(x, 0, n, D, m.lnF, eps, s.lnF, T0)
  s.ids = [...s.ids, ...newIds]
  layerNorm(x, n - 1, n, D, m.lnF, eps, row, 0)
  return headLogits(m, row)
}

/** Forget everything after the first `keep` tokens (the cache for those stays valid). */
export const truncate = (s: Gpt2State, keep: number) => { s.ids = s.ids.slice(0, keep) }

/* ---------------- probabilities, logit lens, sampling ---------------- */

export interface Ranked { id: number; p: number }

export const softmax = (logits: ArrayLike<number>): Float32Array => {
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i]
  const out = new Float32Array(logits.length)
  let z = 0
  for (let i = 0; i < logits.length; i++) { const e = Math.exp(logits[i] - max); out[i] = e; z += e }
  for (let i = 0; i < out.length; i++) out[i] /= z
  return out
}

/** The k largest entries, largest first (ties: lower id first). One pass, no full sort. */
export const topK = (v: ArrayLike<number>, k: number): { id: number; value: number }[] => {
  const best: { id: number; value: number }[] = []
  for (let i = 0; i < v.length; i++) {
    const x = v[i]
    if (best.length === k && x <= best[k - 1].value) continue
    let j = best.length < k ? best.length : k - 1
    while (j > 0 && best[j - 1].value < x) j--
    best.splice(j, 0, { id: i, value: x })
    if (best.length > k) best.pop()
  }
  return best
}

export interface LensRow { layer: number; top: Ranked[]; pFinal: number; rankFinal: number }
/**
 * Logit lens at the last position: after each block, push the residual stream through ln_f and the head,
 * as if the model stopped there. The row for the last block IS the model's real output.
 * pFinal/rankFinal track the token the full model ends up ranking first.
 */
export const logitLens = (s: Gpt2State, n = 5): LensRow[] => {
  const { m } = s
  const { nEmbd: D, eps } = m.cfg
  const T = s.ids.length
  if (!s.capture || T === 0) throw new Error('logit lens needs a captured run')
  const h = new Float32Array(D)
  const probs = s.layers.map((L) => {
    layerNorm(L.resid2, T - 1, T, D, m.lnF, eps, h, 0)
    return softmax(headLogits(m, h))
  })
  const last = probs[probs.length - 1]
  let fin = 0
  for (let i = 1; i < last.length; i++) if (last[i] > last[fin]) fin = i
  return probs.map((p, layer) => {
    let rank = 0
    for (let i = 0; i < p.length; i++) if (p[i] > p[fin]) rank++
    return { layer, top: topK(p, n).map((r) => ({ id: r.id, p: r.value })), pFinal: p[fin], rankFinal: rank + 1 }
  })
}

export interface SamplingOpts { temperature: number; topK: number; topP: number }
export interface Distribution {
  /** Token ids in order of logit, largest first (fixed for a given set of logits). */
  order: Int32Array
  /** Probability of each id in `order` after temperature, top-k and top-p (0 once cut). */
  probs: Float32Array
  /** Probability of each id in `order` after temperature only (before the cuts), for comparison. */
  before: Float32Array
  /** How many tokens survive the cuts. */
  kept: number
}

/** Sort the vocabulary by logit once; temperature never changes this order. */
export const orderByLogit = (logits: ArrayLike<number>): Int32Array => {
  const idx = Int32Array.from({ length: logits.length }, (_, i) => i)
  return idx.sort((a, b) => logits[b] - logits[a] || a - b)
}

/**
 * GPT-2 sampling, in the usual order: divide logits by the temperature, softmax, keep the top-k (0 = all),
 * then keep the smallest set of top tokens whose probability adds up to at least top-p (1 = all), renormalise.
 * Temperature 0 means greedy: all probability on the top token.
 */
export const distribution = (logits: ArrayLike<number>, o: SamplingOpts, order: Int32Array = orderByLogit(logits)): Distribution => {
  const N = order.length
  const before = new Float32Array(N)
  if (o.temperature <= 0) before[0] = 1
  else {
    const top = logits[order[0]]
    let z = 0
    for (let r = 0; r < N; r++) { const e = Math.exp((logits[order[r]] - top) / o.temperature); before[r] = e; z += e }
    for (let r = 0; r < N; r++) before[r] /= z
  }
  let kept = o.topK > 0 ? Math.min(o.topK, N) : N
  if (o.topP < 1) {
    let c = 0
    let r = 0
    for (; r < kept; r++) { c += before[r]; if (c >= o.topP) { r++; break } }
    kept = Math.max(1, Math.min(kept, r))
  }
  if (o.temperature <= 0) kept = 1
  const probs = new Float32Array(N)
  let z = 0
  for (let r = 0; r < kept; r++) z += before[r]
  for (let r = 0; r < kept; r++) probs[r] = before[r] / z
  return { order, probs, before, kept }
}

/** Pick a token with a uniform number u in [0, 1). */
export const sample = (d: Distribution, u: number): { id: number; p: number; rank: number } => {
  let c = 0
  for (let r = 0; r < d.kept; r++) {
    c += d.probs[r]
    if (u < c) return { id: d.order[r], p: d.probs[r], rank: r + 1 }
  }
  const r = d.kept - 1
  return { id: d.order[r], p: d.probs[r], rank: r + 1 }
}

/** Plain generation with the KV cache: prompt once, then one position per new token. */
export const generate = (m: Gpt2Model, prompt: number[], nNew: number, o: SamplingOpts, rand: () => number): number[] => {
  const s = newState(m, false)
  let logits = extend(s, prompt)
  const out: number[] = []
  for (let i = 0; i < nNew; i++) {
    const { id } = sample(distribution(logits, o), rand())
    out.push(id)
    if (i + 1 < nNew) logits = extend(s, [id])
  }
  return out
}

/* ---------------- small summaries for the page ---------------- */

export const norm2 = (v: Float32Array, off: number, len: number): number => {
  let s = 0
  for (let i = 0; i < len; i++) s += v[off + i] * v[off + i]
  return Math.sqrt(s)
}
