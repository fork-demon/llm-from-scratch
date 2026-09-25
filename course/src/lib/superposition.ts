// The toy model of superposition (Elhage et al., 2022, "Toy Models of Superposition"),
// small enough to train in the browser with plain gradient descent.
//
//   x   : n features, each 0 with probability S (sparsity), otherwise uniform in [0, 1]
//   h   = W x                 (squeeze n numbers into m numbers; here n = 5, m = 2)
//   x'  = ReLU(Wᵀ h + b)      (try to rebuild the n features from the m numbers)
//   loss = mean over the batch of  Σᵢ Iᵢ (xᵢ − x'ᵢ)²      (Iᵢ = importance of feature i)
//
// Column i of W is the 2-D direction the model gives feature i. Plot the five columns
// as arrows and you can see whether the model keeps 2 features (dense data) or
// squeezes all 5 into a pentagon (sparse data).
import { makeRng, type Rng } from './rng'
import type { Mat, Vec } from './math'

export const N_FEATURES = 5
export const N_HIDDEN = 2

export interface ToyModel { W: Mat; b: Vec } // W is m×n (2×5), b has n entries

export interface ToyConfig {
  sparsity: number // probability that a feature is 0 in an example
  decay: number // importance of feature i is decay^i (feature 1 matters most)
  seed: number
  lr?: number
  batch?: number
}

export const DEFAULT_LR = 0.1
export const DEFAULT_BATCH = 256
export const DEFAULT_STEPS = 6000

export const importances = (decay: number, n = N_FEATURES): Vec => Array.from({ length: n }, (_, i) => decay ** i)

/** Small random start, seeded, so every run with the same seed is identical. */
export const initModel = (seed: number, n = N_FEATURES, m = N_HIDDEN, scale = 0.3): ToyModel => {
  const rng = makeRng(seed * 7919 + 17)
  return {
    W: Array.from({ length: m }, () => Array.from({ length: n }, () => rng.normal() * scale)),
    b: new Array(n).fill(0),
  }
}

/** One batch of synthetic data: each feature is present with probability 1 − S, with a size in [0, 1]. */
export const sampleBatch = (rng: Rng, batch: number, sparsity: number, n = N_FEATURES): Mat =>
  Array.from({ length: batch }, () =>
    Array.from({ length: n }, () => {
      const present = rng.next() >= sparsity
      const v = rng.next()
      return present ? v : 0
    }),
  )

/** h = W x: the 2 numbers the model is allowed to keep. */
export const hidden = (model: ToyModel, x: Vec): Vec => model.W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0))

/** x' = ReLU(Wᵀ h + b): the model's attempt to rebuild all features. */
export const reconstruct = (model: ToyModel, x: Vec): { h: Vec; pre: Vec; out: Vec } => {
  const h = hidden(model, x)
  const pre = model.b.map((bi, i) => bi + model.W.reduce((s, row, k) => s + row[i] * h[k], 0))
  return { h, pre, out: pre.map((v) => Math.max(0, v)) }
}

/** Weighted squared error, averaged over the batch, and its exact gradient. */
export const lossAndGrad = (model: ToyModel, X: Mat, I: Vec): { loss: number; dW: Mat; db: Vec } => {
  const m = model.W.length
  const n = model.b.length
  const B = X.length
  const dW = Array.from({ length: m }, () => new Array(n).fill(0))
  const db = new Array(n).fill(0)
  let loss = 0
  for (const x of X) {
    const { h, pre, out } = reconstruct(model, x)
    // dL/d(pre_i) for this example
    const d = out.map((y, i) => (pre[i] > 0 ? (2 / B) * I[i] * (y - x[i]) : 0))
    for (let i = 0; i < n; i++) loss += (I[i] * (out[i] - x[i]) ** 2) / B
    // pre = Wᵀ W x + b. W appears twice: once in h = W x, once in Wᵀ h.
    const Wd = model.W.map((row) => row.reduce((s, w, i) => s + w * d[i], 0)) // (W d), length m
    for (let k = 0; k < m; k++) {
      for (let i = 0; i < n; i++) dW[k][i] += h[k] * d[i] + Wd[k] * x[i]
    }
    for (let i = 0; i < n; i++) db[i] += d[i]
  }
  return { loss, dW, db }
}

/** Plain gradient descent: w ← w − lr · gradient. No momentum, no Adam. */
export const gdStep = (model: ToyModel, g: { dW: Mat; db: Vec }, lr: number): ToyModel => ({
  W: model.W.map((row, k) => row.map((w, i) => w - lr * g.dW[k][i])),
  b: model.b.map((bi, i) => bi - lr * g.db[i]),
})

/**
 * A resumable training run: call step(k) repeatedly (the lab does this in animation frames).
 * Same maths as sampleBatch + lossAndGrad + gdStep, written as one allocation-free loop
 * because the browser runs thousands of steps. A test checks the two give identical results.
 */
export const createTrainer = (cfg: ToyConfig) => {
  const n = N_FEATURES
  const m = N_HIDDEN
  const I = importances(cfg.decay)
  const rng = makeRng(cfg.seed * 104729 + 3)
  const lr = cfg.lr ?? DEFAULT_LR
  const B = cfg.batch ?? DEFAULT_BATCH
  const init = initModel(cfg.seed)
  const W = Float64Array.from(init.W.flat()) // W[k * n + i]
  const b = new Float64Array(n)
  const dW = new Float64Array(m * n)
  const db = new Float64Array(n)
  const x = new Float64Array(n)
  const h = new Float64Array(m)
  const d = new Float64Array(n)
  const Wd = new Float64Array(m)
  let steps = 0
  let loss = NaN
  const snapshot = (): ToyModel => ({
    W: Array.from({ length: m }, (_, k) => Array.from(W.subarray(k * n, k * n + n))),
    b: Array.from(b),
  })
  return {
    get model() { return snapshot() },
    get steps() { return steps },
    get loss() { return loss },
    step(count: number) {
      for (let t = 0; t < count; t++) {
        dW.fill(0)
        db.fill(0)
        let L = 0
        for (let e = 0; e < B; e++) {
          for (let i = 0; i < n; i++) {
            const present = rng.next() >= cfg.sparsity
            const v = rng.next()
            x[i] = present ? v : 0
          }
          for (let k = 0; k < m; k++) {
            let s = 0
            for (let i = 0; i < n; i++) s += W[k * n + i] * x[i]
            h[k] = s
          }
          for (let i = 0; i < n; i++) {
            let pre = b[i]
            for (let k = 0; k < m; k++) pre += W[k * n + i] * h[k]
            const y = pre > 0 ? pre : 0
            L += (I[i] * (y - x[i]) ** 2) / B
            d[i] = pre > 0 ? (2 / B) * I[i] * (y - x[i]) : 0
          }
          for (let k = 0; k < m; k++) {
            let s = 0
            for (let i = 0; i < n; i++) s += W[k * n + i] * d[i]
            Wd[k] = s
          }
          for (let k = 0; k < m; k++) for (let i = 0; i < n; i++) dW[k * n + i] += h[k] * d[i] + Wd[k] * x[i]
          for (let i = 0; i < n; i++) db[i] += d[i]
        }
        for (let j = 0; j < m * n; j++) W[j] -= lr * dW[j]
        for (let i = 0; i < n; i++) b[i] -= lr * db[i]
        loss = L
        steps++
      }
      return snapshot()
    },
  }
}

/** The slow, readable version of the same loop, used by the tests to check the fast one. */
export const trainReference = (cfg: ToyConfig, steps: number): ToyModel => {
  const I = importances(cfg.decay)
  const rng = makeRng(cfg.seed * 104729 + 3)
  let model = initModel(cfg.seed)
  for (let t = 0; t < steps; t++) {
    const X = sampleBatch(rng, cfg.batch ?? DEFAULT_BATCH, cfg.sparsity)
    model = gdStep(model, lossAndGrad(model, X, I), cfg.lr ?? DEFAULT_LR)
  }
  return model
}

export const train = (cfg: ToyConfig, steps = DEFAULT_STEPS): { model: ToyModel; loss: number } => {
  const t = createTrainer(cfg)
  t.step(steps)
  return { model: t.model, loss: t.loss }
}

/** Loss on a fixed, separately seeded evaluation set, so two models can be compared fairly. */
export const evalLoss = (model: ToyModel, sparsity: number, decay: number, size = 2000, seed = 999): number => {
  const X = sampleBatch(makeRng(seed), size, sparsity)
  return lossAndGrad(model, X, importances(decay)).loss
}

/** Column i of W: the arrow for feature i. */
export const column = (model: ToyModel, i: number): Vec => model.W.map((row) => row[i])

export const REPRESENTED = 0.5 // an arrow shorter than this counts as "not represented"

export interface Analysis {
  norms: Vec // length of each feature's arrow
  angles: Vec // direction of each arrow in degrees, 0..360
  represented: number[] // indices of features with norm > REPRESENTED
  interference: Vec // Σ over other features j of (unit_i · W_j)²: how much others leak into feature i's direction
  gaps: Vec // angles between neighbouring represented arrows, sorted round the circle
  shape: 'none' | 'one' | 'orthogonal-pair' | 'antipodal-pair' | 'triangle' | 'square' | 'pentagon' | 'other'
}

export const analyse = (model: ToyModel): Analysis => {
  const n = model.b.length
  const cols = Array.from({ length: n }, (_, i) => column(model, i))
  const norms = cols.map((c) => Math.hypot(...c))
  const angles = cols.map((c) => ((Math.atan2(c[1], c[0]) * 180) / Math.PI + 360) % 360)
  const represented = norms.map((v, i) => (v > REPRESENTED ? i : -1)).filter((i) => i >= 0)
  const interference = cols.map((ci, i) => {
    if (norms[i] < 1e-9) return 0
    const u = ci.map((v) => v / norms[i])
    return cols.reduce((s, cj, j) => (j === i ? s : s + (u[0] * cj[0] + u[1] * cj[1]) ** 2), 0)
  })
  const sorted = represented.map((i) => angles[i]).sort((a, b) => a - b)
  const gaps = sorted.map((a, k) => (k === sorted.length - 1 ? sorted[0] + 360 - a : sorted[k + 1] - a))
  const near = (g: number, target: number) => Math.abs(g - target) < 15
  let shape: Analysis['shape'] = 'other'
  const r = represented.length
  if (r === 0) shape = 'none'
  else if (r === 1) shape = 'one'
  else if (r === 2) shape = gaps.every((g) => near(g, 180)) ? 'antipodal-pair' : gaps.some((g) => near(g, 90)) ? 'orthogonal-pair' : 'other'
  else if (r === 3 && gaps.every((g) => near(g, 120))) shape = 'triangle'
  else if (r === 4 && gaps.every((g) => near(g, 90))) shape = 'square'
  else if (r === 5 && gaps.every((g) => near(g, 72))) shape = 'pentagon'
  return { norms, angles, represented, interference, gaps, shape }
}

export const SHAPE_TEXT: Record<Analysis['shape'], string> = {
  none: 'no feature is represented yet',
  one: 'one feature is represented',
  'orthogonal-pair': 'two features at right angles, the rest dropped: no superposition',
  'antipodal-pair': 'two features pointing in opposite directions',
  triangle: 'three features, 120° apart: superposition',
  square: 'four features in two opposite pairs: superposition (often a local minimum on the way to the pentagon)',
  pentagon: 'all five features, 72° apart: a pentagon. Superposition',
  other: 'an in-between arrangement: keep training, or try another seed',
}
