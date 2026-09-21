// A real, tiny character-level language model with manual backprop.
// Faithful port of phase2-language/bigram_lm.py :: train_context3 ("Model C"):
//   ctx previous characters -> embeddings -> concatenated -> ReLU hidden layer -> logits
// trained with plain mini-batch gradient descent on cross-entropy.
// Pure functions + typed arrays, no DOM: the TrainingPlayground drives it in small chunks.
import { makeRng, type Rng } from './rng'

/* ------------------------------------------------------------------ */
/* Data                                                                 */
/* ------------------------------------------------------------------ */

const SUBJECTS = ['the cat', 'the dog', 'a bird', 'the old man', 'a small child', 'the fisherman', 'my sister', 'the mailman']
const VERBS = ['sat on', 'slept under', 'walked to', 'looked at', 'ran past', 'jumped over', 'sang near', 'played by']
const OBJECTS = ['the mat', 'the warm sun', 'the river', 'the old mill', 'the stone bridge', 'the park', 'a tree', 'the sea', 'the quiet street', 'the lazy dog']
const TAILS = ['', '', ' in the morning', ' at dawn', ' every day', ' and sang']

/**
 * A small synthetic corpus: simple sentences drawn from a tiny grammar (3,200 possible sentences).
 * Because train and validation text come from the same grammar but are different sentences,
 * a model that learns the patterns does well on both; a model that memorises does well on train only.
 */
export const makeCorpus = (seed = 7, sentences = 200): string => {
  const rng = makeRng(seed)
  const pick = (a: string[]) => a[rng.int(a.length)]
  let out = ''
  for (let i = 0; i < sentences; i++) out += `${pick(SUBJECTS)} ${pick(VERBS)} ${pick(OBJECTS)}${pick(TAILS)}. `
  return out
}

export interface Dataset {
  chars: string[] // the vocabulary, sorted (same as sorted(set(text)) in Python)
  stoi: Record<string, number>
  train: Int32Array
  val: Int32Array
}

/** chars -> ids, then a 90/10 split exactly like tiny_gpt.py: `n = int(0.9 * len(data))`. */
export const makeDataset = (text: string, trainFraction = 0.9): Dataset => {
  const chars = Array.from(new Set(text)).sort()
  const stoi: Record<string, number> = {}
  chars.forEach((c, i) => (stoi[c] = i))
  const ids = Int32Array.from(Array.from(text, (c) => stoi[c]))
  const n = Math.floor(trainFraction * ids.length)
  return { chars, stoi, train: ids.slice(0, n), val: ids.slice(n) }
}

/* ------------------------------------------------------------------ */
/* Model                                                                */
/* ------------------------------------------------------------------ */

export interface TinyLmConfig { vocabSize: number; ctx: number; dim: number; hidden: number }

/** All weights, flat row-major. Shapes: E (V, dim), W1 (ctx*dim, hidden), b1 (hidden), W2 (hidden, V), b2 (V). */
export interface TinyLmParams { E: Float64Array; W1: Float64Array; b1: Float64Array; W2: Float64Array; b2: Float64Array }
export interface TinyLm { cfg: TinyLmConfig; p: TinyLmParams }

export const PARAM_NAMES = ['E', 'W1', 'b1', 'W2', 'b2'] as const

/** Same initialisation as the Python: 0.1 * normal for weights, zeros for biases. */
export const createModel = (cfg: TinyLmConfig, seed = 7): TinyLm => {
  const rng = makeRng(seed)
  const randn = (n: number) => Float64Array.from({ length: n }, () => 0.1 * rng.normal())
  return {
    cfg,
    p: {
      E: randn(cfg.vocabSize * cfg.dim),
      W1: randn(cfg.ctx * cfg.dim * cfg.hidden),
      b1: new Float64Array(cfg.hidden),
      W2: randn(cfg.hidden * cfg.vocabSize),
      b2: new Float64Array(cfg.vocabSize),
    },
  }
}

export const countParams = (cfg: TinyLmConfig): number =>
  cfg.vocabSize * cfg.dim + cfg.ctx * cfg.dim * cfg.hidden + cfg.hidden + cfg.hidden * cfg.vocabSize + cfg.vocabSize

interface ForwardCache { emb: Float64Array; h: Float64Array; probs: Float64Array; loss: number }

/**
 * Forward pass for n examples. X holds n rows of ctx token ids, Y the n correct next ids.
 *   emb    = E[xb].reshape(batch, -1)
 *   h      = np.maximum(0, emb @ W1 + b1)
 *   logits = h @ W2 + b2 ; probs = softmax(logits)
 *   loss   = -np.log(probs[range(batch), yb]).mean()
 */
const forward = (m: TinyLm, X: Int32Array, Y: Int32Array | null, n: number): ForwardCache => {
  const { vocabSize: V, ctx, dim, hidden } = m.cfg
  const { E, W1, b1, W2, b2 } = m.p
  const inDim = ctx * dim
  const emb = new Float64Array(n * inDim)
  const h = new Float64Array(n * hidden)
  const probs = new Float64Array(n * V)
  let loss = 0
  for (let i = 0; i < n; i++) {
    // embedding lookup + concatenation
    for (let c = 0; c < ctx; c++) {
      const row = X[i * ctx + c] * dim
      for (let d = 0; d < dim; d++) emb[i * inDim + c * dim + d] = E[row + d]
    }
    // hidden layer
    const hi = i * hidden
    for (let j = 0; j < hidden; j++) h[hi + j] = b1[j]
    for (let k = 0; k < inDim; k++) {
      const e = emb[i * inDim + k]
      if (e === 0) continue
      const wk = k * hidden
      for (let j = 0; j < hidden; j++) h[hi + j] += e * W1[wk + j]
    }
    for (let j = 0; j < hidden; j++) if (h[hi + j] < 0) h[hi + j] = 0 // ReLU
    // logits
    const pi = i * V
    for (let v = 0; v < V; v++) probs[pi + v] = b2[v]
    for (let j = 0; j < hidden; j++) {
      const a = h[hi + j]
      if (a === 0) continue
      const wj = j * V
      for (let v = 0; v < V; v++) probs[pi + v] += a * W2[wj + v]
    }
    // softmax (subtract the max so exp cannot overflow)
    let max = -Infinity
    for (let v = 0; v < V; v++) if (probs[pi + v] > max) max = probs[pi + v]
    let sum = 0
    for (let v = 0; v < V; v++) { const e = Math.exp(probs[pi + v] - max); probs[pi + v] = e; sum += e }
    for (let v = 0; v < V; v++) probs[pi + v] /= sum
    if (Y) loss += -Math.log(probs[pi + Y[i]] + 1e-12)
  }
  return { emb, h, probs, loss: loss / n }
}

/** Average cross-entropy of the model on n examples. No gradients: this is what eval_loss does. */
export const lossOf = (m: TinyLm, X: Int32Array, Y: Int32Array, n: number): number => forward(m, X, Y, n).loss

/** Next-token probabilities for one context of exactly ctx ids. */
export const predict = (m: TinyLm, context: ArrayLike<number>): Float64Array =>
  forward(m, Int32Array.from(context), null, 1).probs

/** Forward + backward. Returns the loss and the gradient of the loss with respect to every parameter. */
export const lossAndGrads = (m: TinyLm, X: Int32Array, Y: Int32Array, n: number): { loss: number; grads: TinyLmParams } => {
  const { vocabSize: V, ctx, dim, hidden } = m.cfg
  const { W1, W2 } = m.p
  const inDim = ctx * dim
  const { emb, h, probs, loss } = forward(m, X, Y, n)
  const g: TinyLmParams = {
    E: new Float64Array(m.p.E.length),
    W1: new Float64Array(W1.length),
    b1: new Float64Array(hidden),
    W2: new Float64Array(W2.length),
    b2: new Float64Array(V),
  }
  const dh = new Float64Array(hidden)
  for (let i = 0; i < n; i++) {
    const pi = i * V
    const hi = i * hidden
    // d_logits = (probs - onehot) / batch      (reuse the probs buffer)
    probs[pi + Y[i]] -= 1
    for (let v = 0; v < V; v++) probs[pi + v] /= n
    // d_W2 = h.T @ d_logits ; d_b2 = d_logits.sum(0) ; d_h = (d_logits @ W2.T) * (h > 0)
    for (let v = 0; v < V; v++) g.b2[v] += probs[pi + v]
    for (let j = 0; j < hidden; j++) {
      const a = h[hi + j]
      if (a === 0) { dh[j] = 0; continue }
      const wj = j * V
      let s = 0
      for (let v = 0; v < V; v++) { const dl = probs[pi + v]; g.W2[wj + v] += a * dl; s += dl * W2[wj + v] }
      dh[j] = s
      g.b1[j] += s
    }
    // d_W1 = emb.T @ d_h ; d_emb = d_h @ W1.T, scattered back into the embedding rows (np.add.at)
    for (let k = 0; k < inDim; k++) {
      const e = emb[i * inDim + k]
      const wk = k * hidden
      let s = 0
      for (let j = 0; j < hidden; j++) { const d = dh[j]; if (d === 0) continue; g.W1[wk + j] += e * d; s += d * W1[wk + j] }
      const c = (k / dim) | 0
      g.E[X[i * ctx + c] * dim + (k - c * dim)] += s
    }
  }
  return { loss, grads: g }
}

/** Plain gradient descent, as in bigram_lm.py: `W -= lr * d_W`. */
export const sgdStep = (m: TinyLm, grads: TinyLmParams, lr: number): void => {
  for (const name of PARAM_NAMES) {
    const w = m.p[name]
    const g = grads[name]
    for (let i = 0; i < w.length; i++) w[i] -= lr * g[i]
  }
}

/* ------------------------------------------------------------------ */
/* Batches, evaluation, generation                                      */
/* ------------------------------------------------------------------ */

/** Number of (context, next char) examples in a stretch of text: every position is an example. */
export const exampleCount = (ids: Int32Array, ctx: number): number => Math.max(0, ids.length - ctx)

const gather = (ids: Int32Array, ctx: number, starts: number[]): { X: Int32Array; Y: Int32Array } => {
  const X = new Int32Array(starts.length * ctx)
  const Y = new Int32Array(starts.length)
  starts.forEach((s, i) => {
    for (let c = 0; c < ctx; c++) X[i * ctx + c] = ids[s + c]
    Y[i] = ids[s + ctx]
  })
  return { X, Y }
}

/** A random mini-batch, like get_batch(): random start positions, y is the character after the window. */
export const sampleBatch = (ids: Int32Array, ctx: number, batchSize: number, rng: Rng): { X: Int32Array; Y: Int32Array } => {
  const n = exampleCount(ids, ctx)
  return gather(ids, ctx, Array.from({ length: batchSize }, () => rng.int(n)))
}

/** Deterministic loss over (up to maxExamples evenly spaced) examples. No weight is changed. */
export const evalLoss = (m: TinyLm, ids: Int32Array, maxExamples = 600): number => {
  const n = exampleCount(ids, m.cfg.ctx)
  if (n === 0) return NaN
  const take = Math.min(n, maxExamples)
  const starts = Array.from({ length: take }, (_, i) => Math.floor((i * n) / take))
  const { X, Y } = gather(ids, m.cfg.ctx, starts)
  return lossOf(m, X, Y, take)
}

/** The autoregressive loop: predict, draw from the weighted die, append, slide the window. */
export const generate = (m: TinyLm, ds: Dataset, prompt: string, n: number, rng: Rng): string => {
  const ctx = m.cfg.ctx
  const window = Array.from(prompt.slice(-ctx), (c) => ds.stoi[c] ?? 0)
  while (window.length < ctx) window.unshift(ds.stoi[' '] ?? 0)
  let out = ''
  for (let t = 0; t < n; t++) {
    const probs = predict(m, window)
    if (!Number.isFinite(probs[0])) return out + ' [the weights are NaN]'
    let u = rng.next()
    let next = probs.length - 1
    for (let v = 0; v < probs.length; v++) { u -= probs[v]; if (u < 0) { next = v; break } }
    out += ds.chars[next]
    window.push(next)
    window.shift()
  }
  return out
}

/* ------------------------------------------------------------------ */
/* The training loop, packaged so a UI can run it a few steps at a time */
/* ------------------------------------------------------------------ */

export interface TrainerOptions { ctx: number; dim: number; hidden: number; seed: number; tinyData: boolean }
export interface HistoryPoint { step: number; train: number; val: number }
export interface Sample { step: number; text: string }

export interface Trainer {
  opts: TrainerOptions
  ds: Dataset
  trainIds: Int32Array // what the model actually trains on (a small slice when tinyData is on)
  model: TinyLm
  rng: Rng
  step: number
  tokensSeen: number
  batchLoss: number
  history: HistoryPoint[]
  samples: Sample[]
  diverged: boolean
}

export const TINY_DATA_CHARS = 300
export const SAMPLE_PROMPT = 'the cat sat on the mat. '
export const baselineLoss = (vocabSize: number): number => Math.log(vocabSize)

const record = (t: Trainer, withSample: boolean) => {
  t.history.push({ step: t.step, train: evalLoss(t.model, t.trainIds), val: evalLoss(t.model, t.ds.val) })
  if (withSample) t.samples.push({ step: t.step, text: generate(t.model, t.ds, SAMPLE_PROMPT, 70, makeRng(1000 + t.step)) })
}

export const createTrainer = (opts: TrainerOptions, corpus: string = makeCorpus()): Trainer => {
  const ds = makeDataset(corpus)
  const trainIds = opts.tinyData ? ds.train.slice(0, TINY_DATA_CHARS) : ds.train
  const model = createModel({ vocabSize: ds.chars.length, ctx: opts.ctx, dim: opts.dim, hidden: opts.hidden }, opts.seed)
  const t: Trainer = { opts, ds, trainIds, model, rng: makeRng(opts.seed + 1), step: 0, tokensSeen: 0, batchLoss: NaN, history: [], samples: [], diverged: false }
  record(t, true)
  return t
}

/**
 * Run up to `steps` training steps. One step = one batch:
 *   get_batch -> forward + loss -> backward -> update.
 * Every `evalEvery` steps measure train and validation loss; every `sampleEvery` steps also write a text sample.
 * Stops (and sets `diverged`) if the loss stops being a finite number.
 */
export const trainSteps = (t: Trainer, steps: number, lr: number, batchSize: number, evalEvery = 25, sampleEvery = 100): void => {
  for (let s = 0; s < steps && !t.diverged; s++) {
    const { X, Y } = sampleBatch(t.trainIds, t.opts.ctx, batchSize, t.rng)
    const { loss, grads } = lossAndGrads(t.model, X, Y, batchSize)
    if (!Number.isFinite(loss)) { t.diverged = true; t.batchLoss = loss; return }
    sgdStep(t.model, grads, lr)
    t.step += 1
    t.tokensSeen += batchSize
    t.batchLoss = loss
    if (t.step % evalEvery === 0) {
      record(t, t.step % sampleEvery === 0)
      const last = t.history[t.history.length - 1]
      if (!Number.isFinite(last.train) || !Number.isFinite(last.val)) { t.diverged = true; return }
    }
  }
}
