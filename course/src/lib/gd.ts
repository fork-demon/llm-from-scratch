// Gradient descent on a straight line, y = w*x + b, with mean squared error.
// Same algorithm as phase1-foundations/gradient_descent.py (Stage A, B and C).
// Pure functions, no DOM.
import { makeRng, type Rng } from './rng'

export interface LineData { xs: number[]; ys: number[] }
export interface LineParams { w: number; b: number }

export const TRUE_W = 3.0
export const TRUE_B = -1.5

/** Stage C data: x uniform in [-2, 2], y = 3x - 1.5 + gaussian noise. Seeded, so every learner sees the same points. */
export const makeLineData = (n = 40, seed = 42, noise = 0.3, w = TRUE_W, b = TRUE_B): LineData => {
  const rng = makeRng(seed)
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i < n; i++) {
    const x = -2 + 4 * rng.next()
    xs.push(x)
    ys.push(w * x + b + noise * rng.normal())
  }
  return { xs, ys }
}

export const predict = (p: LineParams, x: number): number => p.w * x + p.b

export interface GradResult {
  preds: number[]
  errs: number[] // pred - y
  loss: number // mean(err^2)
  gradW: number // mean(2 * err * x)
  gradB: number // mean(2 * err)
}

/** Forward pass, loss and the hand-derived gradients, exactly as in stage_c(). `idx` selects a mini-batch. */
export const lossAndGrads = (p: LineParams, data: LineData, idx?: number[]): GradResult => {
  const ids = idx ?? data.xs.map((_, i) => i)
  const n = ids.length
  const preds: number[] = []
  const errs: number[] = []
  let loss = 0
  let gradW = 0
  let gradB = 0
  for (const i of ids) {
    const pred = p.w * data.xs[i] + p.b // 1. forward
    const err = pred - data.ys[i]
    preds.push(pred)
    errs.push(err)
    loss += err * err // 2. loss
    gradW += 2 * err * data.xs[i] // 3. gradients
    gradB += 2 * err
  }
  return { preds, errs, loss: loss / n, gradW: gradW / n, gradB: gradB / n }
}

export const mse = (p: LineParams, data: LineData): number => lossAndGrads(p, data).loss

/** Stage B: the slope by nudging. (L(p + h) - L(p)) / h, no calculus. */
export const numericalGrads = (p: LineParams, data: LineData, h = 1e-5): { gradW: number; gradB: number } => {
  const base = mse(p, data)
  return {
    gradW: (mse({ w: p.w + h, b: p.b }, data) - base) / h,
    gradB: (mse({ w: p.w, b: p.b + h }, data) - base) / h,
  }
}

/** 4. update: step against the gradient. */
export const applyStep = (p: LineParams, g: { gradW: number; gradB: number }, lr: number): LineParams => ({
  w: p.w - lr * g.gradW,
  b: p.b - lr * g.gradB,
})

/** A random mini-batch drawn with replacement, like rng.integers(0, N, size=batch_size). */
export const sampleBatch = (rng: Rng, n: number, size: number): number[] => Array.from({ length: size }, () => rng.int(n))

export interface StepRecord extends GradResult {
  step: number
  idx: number[] | null // the mini-batch used (null = all data)
  before: LineParams
  after: LineParams
  lr: number
  fullLoss: number // loss on ALL data before the update
}

/** One complete iteration of the loop: forward, loss, gradients, update. */
export const gdStep = (p: LineParams, data: LineData, lr: number, step: number, idx: number[] | null = null): StepRecord => {
  const g = lossAndGrads(p, data, idx ?? undefined)
  const fullLoss = idx ? mse(p, data) : g.loss
  return { ...g, step, idx, before: p, after: applyStep(p, g, lr), lr, fullLoss }
}

export const hasDiverged = (p: LineParams): boolean => !Number.isFinite(p.w) || !Number.isFinite(p.b) || Math.abs(p.w) > 1e6 || Math.abs(p.b) > 1e6

/** Run the loop for a fixed number of steps. batchSize 0 means "use all the data every step". */
export const runGD = (data: LineData, opts: { lr: number; steps: number; batchSize?: number; seed?: number; start?: LineParams }): { params: LineParams; history: StepRecord[]; diverged: boolean } => {
  const { lr, steps, batchSize = 0, seed = 1, start = { w: 0, b: 0 } } = opts
  const rng = makeRng(seed)
  let p = start
  const history: StepRecord[] = []
  for (let s = 0; s < steps; s++) {
    const idx = batchSize > 0 ? sampleBatch(rng, data.xs.length, batchSize) : null
    const rec = gdStep(p, data, lr, s, idx)
    history.push(rec)
    p = rec.after
    if (hasDiverged(p)) return { params: p, history, diverged: true }
  }
  return { params: p, history, diverged: false }
}

/** The best possible line (ordinary least squares). Used only to reveal the solution on request. */
export const bestFit = (data: LineData): LineParams => {
  const n = data.xs.length
  const mx = data.xs.reduce((s, v) => s + v, 0) / n
  const my = data.ys.reduce((s, v) => s + v, 0) / n
  let sxy = 0
  let sxx = 0
  for (let i = 0; i < n; i++) {
    sxy += (data.xs[i] - mx) * (data.ys[i] - my)
    sxx += (data.xs[i] - mx) ** 2
  }
  const w = sxy / sxx
  return { w, b: my - w * mx }
}

/**
 * The largest learning rate for which full-batch gradient descent on this data still converges.
 * The loss is a bowl; its steepest curvature is the largest eigenvalue of 2 * [[mean(x^2), mean(x)], [mean(x), 1]].
 * Steps stay stable while lr < 2 / that curvature.
 */
export const maxStableLr = (data: LineData): number => {
  const n = data.xs.length
  const a = (2 * data.xs.reduce((s, x) => s + x * x, 0)) / n
  const c = (2 * data.xs.reduce((s, x) => s + x, 0)) / n
  const d = 2
  const lambdaMax = (a + d) / 2 + Math.sqrt(((a - d) / 2) ** 2 + c * c)
  return 2 / lambdaMax
}

/** Words for a gradient, for the "you are the optimizer" mode. */
export const describeGradient = (grad: number): { direction: 'up' | 'down' | 'leave'; strength: string } => {
  const a = Math.abs(grad)
  if (a < 0.05) return { direction: 'leave', strength: 'about right' }
  const strength = a > 4 ? 'strongly' : a > 1 ? 'moderately' : 'gently'
  // positive gradient = loss rises when the knob goes up, so turn it down
  return { direction: grad > 0 ? 'down' : 'up', strength }
}
