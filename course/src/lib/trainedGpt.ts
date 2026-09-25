// A REAL trained tiny GPT, run in the browser. The weights come from
// phase3-transformers/export_checkpoint.py, which trains class GPT from tiny_gpt.py on tiny Shakespeare
// and writes public/models/tiny-gpt.json (manifest) + tiny-gpt.bin (float16 weights).
//
// forward() is the exact computation of GPT.forward in eval mode (no dropout):
//   x = tok_emb[ids] + pos_emb[0..T-1]
//   per block: x = x + proj(causal_multi_head_attention(ln1(x)));  x = x + fc2(gelu(fc1(ln2(x))))
//   logits = ln_f(x) @ tok_emb^T                       (tied head)
// and it also records what happens inside: every head's attention weights, the residual stream after
// every block, and the "logit lens" (ln_f + head applied to each intermediate residual).
// Pure functions, no DOM. Plain loops over Float32Array: the model is small enough that this is fast.
import { makeRng } from './rng'

export interface TrainedConfig { vocabSize: number; contextLen: number; nEmbd: number; nHead: number; nLayer: number }

export interface TensorEntry { name: string; shape: number[]; offset: number; length: number }
export interface TrainingInfo {
  data: string
  data_chars: number
  steps: number
  batch_size: number
  lr: number
  schedule: string
  optimizer: string
  init: string
  dropout: number
  device: string
  train_seconds: number
  torch: string
  train_loss: number
  val_loss: number
  loss_note: string
}
export interface Manifest {
  format: string
  dtype: 'float16'
  weights: string
  config: { vocab_size: number; context_len: number; n_embd: number; n_head: number; n_layer: number; tied_head: boolean; gelu: string }
  vocab: string[]
  n_params: number
  training: TrainingInfo
  python_head_scores?: { previous_token: number[][]; induction: number[][] }
  tensors: TensorEntry[]
}

interface Norm { g: Float32Array; b: Float32Array }
interface BlockW {
  ln1: Norm
  qkvW: Float32Array // (3D, D) row-major, PyTorch Linear layout: y = W x + b
  qkvB: Float32Array
  projW: Float32Array // (D, D)
  projB: Float32Array
  ln2: Norm
  fcW: Float32Array // (4D, D)
  fcB: Float32Array
  fc2W: Float32Array // (D, 4D)
  fc2B: Float32Array
}
export interface TrainedGpt {
  cfg: TrainedConfig
  vocab: string[]
  stoi: Map<string, number>
  nParams: number
  training: TrainingInfo
  pythonHeadScores?: Manifest['python_head_scores']
  tokEmb: Float32Array // (V, D), also the head
  posEmb: Float32Array // (context, D)
  blocks: BlockW[]
  lnF: Norm
}

/* ---------------- loading ---------------- */

/** IEEE 754 half precision -> float32, one value. */
export const halfToFloat = (h: number): number => {
  const s = h & 0x8000 ? -1 : 1
  const e = (h >> 10) & 0x1f
  const f = h & 0x3ff
  if (e === 0) return s * f * 2 ** -24
  if (e === 31) return f ? NaN : s * Infinity
  return s * (1 + f / 1024) * 2 ** (e - 15)
}

/** Decode little-endian float16 bytes into a Float32Array. */
export const decodeFloat16 = (buf: ArrayBuffer): Float32Array => {
  const dv = new DataView(buf)
  const out = new Float32Array(buf.byteLength / 2)
  for (let i = 0; i < out.length; i++) out[i] = halfToFloat(dv.getUint16(i * 2, true))
  return out
}

/** Build a model from the manifest and the decoded weights. Checks every shape. */
export const buildModel = (man: Manifest, weights: Float32Array): TrainedGpt => {
  const c = man.config
  const cfg: TrainedConfig = { vocabSize: c.vocab_size, contextLen: c.context_len, nEmbd: c.n_embd, nHead: c.n_head, nLayer: c.n_layer }
  if (cfg.nEmbd % cfg.nHead !== 0) throw new Error('n_embd must be divisible by n_head')
  if (man.vocab.length !== cfg.vocabSize) throw new Error(`vocab has ${man.vocab.length} entries, config says ${cfg.vocabSize}`)
  const byName = new Map(man.tensors.map((t) => [t.name, t]))
  const D = cfg.nEmbd
  const get = (name: string, shape: number[]): Float32Array => {
    const t = byName.get(name)
    if (!t) throw new Error(`checkpoint is missing tensor ${name}`)
    if (t.shape.join(',') !== shape.join(',')) throw new Error(`${name}: shape ${t.shape} but expected ${shape}`)
    if (t.offset + t.length > weights.length) throw new Error(`${name}: runs past the end of the weights file`)
    return weights.subarray(t.offset, t.offset + t.length)
  }
  const norm = (p: string): Norm => ({ g: get(`${p}.weight`, [D]), b: get(`${p}.bias`, [D]) })
  const blocks: BlockW[] = Array.from({ length: cfg.nLayer }, (_, i) => {
    const p = `blocks.${i}.`
    return {
      ln1: norm(p + 'ln1'),
      qkvW: get(p + 'attn.qkv.weight', [3 * D, D]),
      qkvB: get(p + 'attn.qkv.bias', [3 * D]),
      projW: get(p + 'attn.proj.weight', [D, D]),
      projB: get(p + 'attn.proj.bias', [D]),
      ln2: norm(p + 'ln2'),
      fcW: get(p + 'ffn.fc.weight', [4 * D, D]),
      fcB: get(p + 'ffn.fc.bias', [4 * D]),
      fc2W: get(p + 'ffn.proj.weight', [D, 4 * D]),
      fc2B: get(p + 'ffn.proj.bias', [D]),
    }
  })
  return {
    cfg,
    vocab: man.vocab,
    stoi: new Map(man.vocab.map((ch, i) => [ch, i])),
    nParams: man.n_params,
    training: man.training,
    pythonHeadScores: man.python_head_scores,
    tokEmb: get('tok_emb', [cfg.vocabSize, D]),
    posEmb: get('pos_emb', [cfg.contextLen, D]),
    blocks,
    lnF: norm('ln_f'),
  }
}

type FetchLike = (url: string) => Promise<{ ok: boolean; status: number; json(): Promise<unknown>; arrayBuffer(): Promise<ArrayBuffer> }>

/** Fetch manifest + weights. `baseUrl` is the folder, e.g. `${import.meta.env.BASE_URL}models/`. */
export const loadTrainedGpt = async (baseUrl: string, fetchFn: FetchLike = fetch as unknown as FetchLike): Promise<TrainedGpt> => {
  const mRes = await fetchFn(`${baseUrl}tiny-gpt.json`)
  if (!mRes.ok) throw new Error(`could not load the model manifest (HTTP ${mRes.status})`)
  const man = (await mRes.json()) as Manifest
  if (man.dtype !== 'float16') throw new Error(`unsupported weight type ${man.dtype}`)
  const wRes = await fetchFn(`${baseUrl}${man.weights}`)
  if (!wRes.ok) throw new Error(`could not load the model weights (HTTP ${wRes.status})`)
  return buildModel(man, decodeFloat16(await wRes.arrayBuffer()))
}

/* ---------------- tokenizer ---------------- */

/** Character-level: one id per character. Characters the model never saw are dropped (and reported). */
export const encode = (m: TrainedGpt, text: string): { ids: number[]; dropped: string[] } => {
  const ids: number[] = []
  const dropped: string[] = []
  for (const ch of text) {
    const id = m.stoi.get(ch)
    if (id === undefined) dropped.push(ch)
    else ids.push(id)
  }
  return { ids, dropped }
}
export const decode = (m: TrainedGpt, ids: number[]): string => ids.map((i) => m.vocab[i]).join('')
/** Make whitespace visible in token chips. */
export const showTok = (ch: string): string => (ch === ' ' ? '␣' : ch === '\n' ? '↵' : ch === '\t' ? '⇥' : ch)

/* ---------------- maths ---------------- */

/** erf via the Chebyshev fit of erfc from Numerical Recipes (erfcc): |error| < 1.2e-7 everywhere. */
export const erf = (x: number): number => {
  const z = Math.abs(x)
  const t = 1 / (1 + 0.5 * z)
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
  return x >= 0 ? 1 - r : r - 1
}
/** nn.GELU() default: the exact (erf) form, x * Phi(x). */
export const gelu = (x: number): number => 0.5 * x * (1 + erf(x / Math.SQRT2))

const layerNormRows = (x: Float32Array, T: number, D: number, n: Norm): Float32Array => {
  const out = new Float32Array(T * D)
  for (let t = 0; t < T; t++) {
    const o = t * D
    let mean = 0
    for (let i = 0; i < D; i++) mean += x[o + i]
    mean /= D
    let v = 0
    for (let i = 0; i < D; i++) v += (x[o + i] - mean) ** 2
    const inv = 1 / Math.sqrt(v / D + 1e-5) // nn.LayerNorm: biased variance, eps 1e-5
    for (let i = 0; i < D; i++) out[o + i] = (x[o + i] - mean) * inv * n.g[i] + n.b[i]
  }
  return out
}

/** y = x W^T + b for every row. W is (outDim, inDim) row-major, as stored by PyTorch. */
const linear = (x: Float32Array, T: number, inDim: number, W: Float32Array, b: Float32Array, outDim: number): Float32Array => {
  const out = new Float32Array(T * outDim)
  for (let t = 0; t < T; t++) {
    const xo = t * inDim
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim
      // unrolled by 4 (inDim is a multiple of 4 here): about twice as fast in V8
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0
      let i = 0
      for (; i + 3 < inDim; i += 4) {
        s0 += x[xo + i] * W[wo + i]
        s1 += x[xo + i + 1] * W[wo + i + 1]
        s2 += x[xo + i + 2] * W[wo + i + 2]
        s3 += x[xo + i + 3] * W[wo + i + 3]
      }
      for (; i < inDim; i++) s0 += x[xo + i] * W[wo + i]
      out[t * outDim + o] = b[o] + s0 + s1 + s2 + s3
    }
  }
  return out
}

/** ln_f then the tied head: one row of V logits per position. */
const unembed = (m: TrainedGpt, x: Float32Array, T: number): Float32Array[] => {
  const { nEmbd: D, vocabSize: V } = m.cfg
  const h = layerNormRows(x, T, D, m.lnF)
  const rows: Float32Array[] = []
  for (let t = 0; t < T; t++) {
    const r = new Float32Array(V)
    for (let v = 0; v < V; v++) {
      let s = 0
      for (let i = 0; i < D; i++) s += h[t * D + i] * m.tokEmb[v * D + i]
      r[v] = s
    }
    rows.push(r)
  }
  return rows
}

/* ---------------- forward ---------------- */

export interface GptRun {
  ids: number[]
  /** attention[layer][head][query t][key s] (s <= t; the rest are 0). */
  attention: Float32Array[][][]
  /** residual[layer] = the (T, D) stream after block `layer`, row-major. residual has nLayer entries. */
  residual: Float32Array[]
  /** lens[layer][t] = V logits from ln_f + head applied to residual[layer] at position t. lens[nLayer-1] equals logits. */
  lens: Float32Array[][]
  /** Final logits (T rows of V). */
  logits: Float32Array[]
}

/** GPT.forward (eval mode) for one sequence. With `inside: false` it skips the logit lens (faster, for generation). */
export const forward = (m: TrainedGpt, ids: number[], opts: { inside?: boolean } = {}): GptRun => {
  const inside = opts.inside ?? true
  const { nEmbd: D, nHead: H, contextLen, vocabSize: V } = m.cfg
  const T = ids.length
  if (T === 0) throw new Error('forward: need at least one token')
  if (T > contextLen) throw new Error(`forward: ${T} tokens do not fit in the context window of ${contextLen}`)
  for (const id of ids) if (!(id >= 0 && id < V)) throw new Error(`forward: token id ${id} is outside the vocabulary`)
  const hd = D / H
  const scale = 1 / Math.sqrt(hd)

  let x = new Float32Array(T * D)
  ids.forEach((id, t) => {
    for (let i = 0; i < D; i++) x[t * D + i] = m.tokEmb[id * D + i] + m.posEmb[t * D + i]
  })

  const attention: Float32Array[][][] = []
  const residual: Float32Array[] = []
  const lens: Float32Array[][] = []

  for (const b of m.blocks) {
    // ---- attention: communicate ----
    const h = layerNormRows(x, T, D, b.ln1)
    const qkv = linear(h, T, D, b.qkvW, b.qkvB, 3 * D) // row t = [q (D) | k (D) | v (D)]
    const y = new Float32Array(T * D) // concatenated head outputs
    const layerAtt: Float32Array[][] = []
    for (let hh = 0; hh < H; hh++) {
      const qo = hh * hd
      const ko = D + hh * hd
      const vo = 2 * D + hh * hd
      const rows: Float32Array[] = []
      for (let t = 0; t < T; t++) {
        const w = new Float32Array(T)
        let max = -Infinity
        const sc = new Float64Array(t + 1)
        for (let s = 0; s <= t; s++) {
          let d = 0
          for (let i = 0; i < hd; i++) d += qkv[t * 3 * D + qo + i] * qkv[s * 3 * D + ko + i]
          sc[s] = d * scale
          if (sc[s] > max) max = sc[s]
        }
        let z = 0
        for (let s = 0; s <= t; s++) { sc[s] = Math.exp(sc[s] - max); z += sc[s] }
        for (let s = 0; s <= t; s++) w[s] = sc[s] / z // positions s > t stay 0: the causal mask
        for (let i = 0; i < hd; i++) {
          let acc = 0
          for (let s = 0; s <= t; s++) acc += (sc[s] / z) * qkv[s * 3 * D + vo + i]
          y[t * D + qo + i] = acc
        }
        rows.push(w)
      }
      layerAtt.push(rows)
    }
    attention.push(layerAtt)
    const a = linear(y, T, D, b.projW, b.projB, D)
    for (let i = 0; i < x.length; i++) x[i] += a[i]

    // ---- feed-forward: compute ----
    const h2 = layerNormRows(x, T, D, b.ln2)
    const f = linear(h2, T, D, b.fcW, b.fcB, 4 * D)
    for (let i = 0; i < f.length; i++) f[i] = gelu(f[i])
    const f2 = linear(f, T, 4 * D, b.fc2W, b.fc2B, D)
    const nx = new Float32Array(T * D)
    for (let i = 0; i < x.length; i++) nx[i] = x[i] + f2[i]
    x = nx
    residual.push(x)
    if (inside) lens.push(unembed(m, x, T))
  }
  const logits = inside ? lens[lens.length - 1] : unembed(m, x, T)
  return { ids, attention, residual, lens, logits }
}

/* ---------------- probabilities and sampling ---------------- */

export const softmaxRow = (logits: ArrayLike<number>, temperature = 1): number[] => {
  const t = Math.max(temperature, 1e-6)
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i] / t)
  const e = Array.from(logits, (l) => Math.exp(l / t - max))
  const z = e.reduce((a, c) => a + c, 0)
  return e.map((v) => v / z)
}

export interface Ranked { id: number; p: number }
export const topN = (probs: number[], n: number): Ranked[] =>
  probs.map((p, id) => ({ id, p })).sort((a, b) => b.p - a.p || a.id - b.id).slice(0, n)

/**
 * The next-token distribution after `ids`: crop to the context window, forward, last row, temperature,
 * then keep only the `topK` most likely (0 = keep all) and renormalise.
 */
export const nextDistribution = (m: TrainedGpt, ids: number[], temperature = 1, topK = 0): number[] => {
  const ctx = ids.slice(-m.cfg.contextLen)
  const { logits } = forward(m, ctx, { inside: false })
  const probs = softmaxRow(logits[logits.length - 1], temperature)
  if (topK > 0 && topK < probs.length) {
    const keep = new Set(topN(probs, topK).map((r) => r.id))
    const z = probs.reduce((a, p, i) => a + (keep.has(i) ? p : 0), 0)
    return probs.map((p, i) => (keep.has(i) ? p / z : 0))
  }
  return probs
}

/** Pick an index from `probs` with a uniform number u in [0, 1). */
export const pick = (probs: number[], u: number): number => {
  let c = 0
  for (let i = 0; i < probs.length; i++) {
    c += probs[i]
    if (u < c && probs[i] > 0) return i
  }
  for (let i = probs.length - 1; i >= 0; i--) if (probs[i] > 0) return i
  return probs.length - 1
}

export interface GenStep { id: number; p: number; top: Ranked[] }
/** One generation step: the distribution, the pick, and the top 5 alternatives it was chosen from. */
export const generateStep = (m: TrainedGpt, ids: number[], temperature: number, topK: number, u: number): GenStep => {
  const probs = nextDistribution(m, ids, temperature, topK)
  const id = pick(probs, u)
  return { id, p: probs[id], top: topN(probs, 5) }
}

/** GPT.generate with temperature and top-k. Deterministic for a given seed. */
export const generate = (m: TrainedGpt, ids: number[], maxNewTokens: number, opts: { temperature?: number; topK?: number; seed?: number } = {}): { ids: number[]; steps: GenStep[] } => {
  const rng = makeRng(opts.seed ?? 1)
  const out = ids.slice()
  const steps: GenStep[] = []
  for (let i = 0; i < maxNewTokens; i++) {
    const s = generateStep(m, out, opts.temperature ?? 1, opts.topK ?? 0, rng.next())
    steps.push(s)
    out.push(s.id)
  }
  return { ids: out, steps }
}

/* ---------------- head scores ---------------- */

/** A few lines from the validation part of tiny Shakespeare (text the model was NOT trained on). */
export const PROBE_TEXTS = [
  'ept his service.\n\nBAPTISTA:\nA thousand thanks, Signior Gremio.\nW',
  'en taught by any of my trade:\nAnd there it is in writing, fairly',
  'Tailor:\nShe says your worship means to make\na puppet of her.\n\nPE',
  'saw suffer: a brave vessel,\nWho had, no doubt, some noble creatu',
]

export interface HeadScores {
  /** [layer][head]: mean attention from each position to the one just before it, on real text. */
  previousToken: number[][]
  /** [layer][head]: on random lower-case letters repeated twice, mean attention from each token in the 2nd copy
   *  to the token that FOLLOWED its earlier occurrence. */
  induction: number[][]
  /** What a head that spreads attention evenly would score, for comparison. */
  uniformPrevious: number
  uniformInduction: number
  /** Mean loss (nats per character) on the random letters: first copy vs second copy. If the model
   *  copied from the earlier copy, the second number would be far lower. */
  repeatLoss: { first: number; second: number }
}

/** Mean attention to position t-1 over positions 1..T-1 of one run. */
export const previousTokenScore = (att: Float32Array[]): number => {
  let s = 0
  for (let t = 1; t < att.length; t++) s += att[t][t - 1]
  return s / Math.max(1, att.length - 1)
}
/** For a sequence [r, r] with r of length `half`: mean attention from t in the 2nd copy to t - half + 1. */
export const inductionScore = (att: Float32Array[], half: number): number => {
  let s = 0
  for (let t = half; t < 2 * half; t++) s += att[t][t - half + 1]
  return s / half
}

type ScoreOpts = { repeats?: number; half?: number; seed?: number }

/** The measurement as a generator that yields after every forward pass, so a page can spread it over frames. */
function* headScoreSteps(m: TrainedGpt, opts: ScoreOpts): Generator<void, HeadScores> {
  const { nLayer: L, nHead: H, contextLen } = m.cfg
  const half = Math.min(opts.half ?? 48, Math.floor(contextLen / 2))
  const repeats = opts.repeats ?? 6
  const rng = makeRng(opts.seed ?? 7)
  const zero = () => Array.from({ length: L }, () => Array(H).fill(0) as number[])
  const prev = zero()
  const ind = zero()
  const lower = encode(m, 'abcdefghijklmnopqrstuvwxyz').ids
  const texts = PROBE_TEXTS.map((t) => encode(m, t).ids.slice(0, contextLen))
  for (const ids of texts) {
    const run = forward(m, ids, { inside: false })
    for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) prev[l][h] += previousTokenScore(run.attention[l][h]) / texts.length
    yield
  }
  let uniformPrevious = 0
  for (let t = 1; t < texts[0].length; t++) uniformPrevious += 1 / (t + 1) / (texts[0].length - 1)
  let uniformInduction = 0
  for (let t = half; t < 2 * half; t++) uniformInduction += 1 / (t + 1) / half
  const repeatLoss = { first: 0, second: 0 }
  for (let r = 0; r < repeats; r++) {
    const letters = Array.from({ length: half }, () => lower[rng.int(lower.length)])
    const seq = [...letters, ...letters]
    const run = forward(m, seq, { inside: false })
    for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) ind[l][h] += inductionScore(run.attention[l][h], half) / repeats
    // position t predicts seq[t + 1]; first copy: t = 0..half-2, second copy: t = half..2*half-2
    for (let t = 0; t < 2 * half - 1; t++) {
      if (t === half - 1) continue // the jump from the end of copy 1 to the start of copy 2 is not predictable
      const loss = -Math.log(softmaxRow(run.logits[t])[seq[t + 1]] + 1e-12) / (half - 1) / repeats
      if (t < half) repeatLoss.first += loss
      else repeatLoss.second += loss
    }
    yield
  }
  return { previousToken: prev, induction: ind, uniformPrevious, uniformInduction, repeatLoss }
}

export const headScores = (m: TrainedGpt, opts: ScoreOpts = {}): HeadScores => {
  const it = headScoreSteps(m, opts)
  for (;;) {
    const r = it.next()
    if (r.done) return r.value
  }
}

/** Same result as headScores, but gives the browser a chance to paint between forward passes. */
export const headScoresAsync = async (m: TrainedGpt, opts: ScoreOpts = {}, pause: () => Promise<void> = () => new Promise((r) => setTimeout(r, 0))): Promise<HeadScores> => {
  const it = headScoreSteps(m, opts)
  for (;;) {
    const r = it.next()
    if (r.done) return r.value
    await pause()
  }
}

export type HeadLabel = 'previous-token' | 'induction' | null
/** Simple fixed thresholds, chosen as several times what an even spread of attention would score. */
export const PREV_THRESHOLD = 0.4
export const INDUCTION_THRESHOLD = 0.3
export const labelHead = (s: HeadScores, l: number, h: number): HeadLabel =>
  s.induction[l][h] >= INDUCTION_THRESHOLD ? 'induction' : s.previousToken[l][h] >= PREV_THRESHOLD ? 'previous-token' : null
