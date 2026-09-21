// LoRA in miniature. Conventions follow LoRALinear in
// phase4-modern-llms/finetune_tiny_gpt.py:
//   forward:  x @ W (frozen)  +  (x @ A) @ B * scale      A: (d_in, r)   B: (r, d_out)
//   A starts small and random, B starts at ZERO, so training begins exactly at the base model.
import { matmul, transpose, type Mat } from './math'
import { makeRng } from './rng'

/* ---------- parameter counting ---------- */
export const fullParams = (dIn: number, dOut: number) => dIn * dOut
export const loraParams = (dIn: number, dOut: number, r: number) => r * (dIn + dOut)
export const loraPercent = (dIn: number, dOut: number, r: number) => (100 * loraParams(dIn, dOut, r)) / fullParams(dIn, dOut)

/* ---------- forward pass ---------- */
const zeros = (n: number, m: number): Mat => Array.from({ length: n }, () => new Array<number>(m).fill(0))
const addM = (a: Mat, b: Mat, k = 1): Mat => a.map((row, i) => row.map((v, j) => v + k * b[i][j]))
export const frob = (m: Mat) => Math.sqrt(m.flat().reduce((s, v) => s + v * v, 0))

/** x: (n, d_in). Returns x @ W + (x @ A) @ B * scale. With r = 0 there is no bypass at all. */
export const loraForward = (x: Mat, W: Mat, A: Mat, B: Mat, scale = 1): Mat => {
  const base = matmul(x, W)
  if (A[0]?.length === 0 || B.length === 0) return base
  return addM(base, matmul(matmul(x, A), B), scale)
}

/** Fold the adapter into one ordinary matrix: W + scale * A @ B. Same outputs, no extra cost at inference. */
export const mergeLora = (W: Mat, A: Mat, B: Mat, scale = 1): Mat => (B.length === 0 ? W.map((r) => r.slice()) : addM(W, matmul(A, B), scale))

/* ---------- the live demo: can a rank-r patch reproduce a given change? ---------- */
export const D = 6
export type TargetKind = 'low-rank' | 'full-rank'

const randMat = (n: number, m: number, seed: number, std = 1): Mat => {
  const rng = makeRng(seed)
  return Array.from({ length: n }, () => Array.from({ length: m }, () => std * rng.normal()))
}

/** A fixed 6x6 "pretrained" matrix. */
export const baseW = (): Mat => randMat(D, D, 101, 0.6)

/**
 * The change fine-tuning "wants": W* - W.
 * 'low-rank': built as (6x2) @ (2x6), so it has rank exactly 2.
 * 'full-rank': 36 independent random numbers, scaled to the same overall size.
 */
export const targetDelta = (kind: TargetKind): Mat => {
  const low = matmul(randMat(D, 2, 202, 0.7), randMat(2, D, 303, 0.7))
  if (kind === 'low-rank') return low
  const full = randMat(D, D, 404)
  const k = frob(low) / frob(full)
  return full.map((r) => r.map((v) => v * k))
}

export interface LoraState { r: number; A: Mat; B: Mat; step: number }

export const initLora = (r: number, seed = 7): LoraState => ({
  r,
  A: r === 0 ? zeros(D, 0) : randMat(D, r, seed, 0.3),
  B: zeros(r, D), // zeros: the patch A @ B starts as exactly nothing
  step: 0,
})

/** Remaining error as a fraction of the change we wanted: |A@B - delta| / |delta|. 1 = nothing learned, 0 = perfect. */
export const relError = (s: LoraState, delta: Mat): number => (s.r === 0 ? 1 : frob(addM(matmul(s.A, s.B), delta, -1)) / frob(delta))

/**
 * Plain gradient descent on loss = 1/2 |A@B - delta|^2, updating A and B only. W is never touched.
 *   E = A@B - delta      dL/dA = E @ B^T      dL/dB = A^T @ E
 */
export const trainLora = (s: LoraState, delta: Mat, steps: number, lr = 0.05): LoraState => {
  if (s.r === 0) return { ...s, step: s.step + steps }
  let { A, B } = s
  for (let t = 0; t < steps; t++) {
    const E = addM(matmul(A, B), delta, -1)
    const gA = matmul(E, transpose(B))
    const gB = matmul(transpose(A), E)
    A = addM(A, gA, -lr)
    B = addM(B, gB, -lr)
  }
  return { r: s.r, A, B, step: s.step + steps }
}

/** Train from scratch to (near) convergence for every rank 0..6: the "error vs rank" curve. */
export const errorByRank = (kind: TargetKind, steps = 600): number[] => {
  const delta = targetDelta(kind)
  return [0, 1, 2, 3, 4, 5, 6].map((r) => relError(trainLora(initLora(r), delta, steps), delta))
}
