// Pure helpers for the SoftmaxPlayground (lesson 1.3): softmax with every
// intermediate value exposed, the naive version that overflows, weighted-die
// sampling, and "surprise" (cross-entropy for one example).
import { sampleIndex, softmax, type Vec } from './math'
import type { Rng } from './rng'

export interface SoftmaxSteps {
  shifted: Vec // logits after the optional "+shift to every logit"
  scaled: Vec // shifted / temperature
  exps: Vec // e^scaled, computed naively (can be astronomically large, or Infinity)
  expSum: number
  probs: Vec // computed with the stable softmax from math.ts
}

export const softmaxSteps = (logits: Vec, temperature = 1, shift = 0): SoftmaxSteps => {
  const shifted = logits.map((z) => z + shift)
  const scaled = shifted.map((z) => z / temperature)
  const exps = scaled.map((z) => Math.exp(z))
  return { shifted, scaled, exps, expSum: exps.reduce((s, v) => s + v, 0), probs: softmax(shifted, temperature) }
}

/** Softmax exactly as the formula reads, with no stability trick. exp(1000) is Infinity, and Infinity / Infinity is NaN. */
export const naiveSoftmax = (logits: Vec): Vec => {
  const exps = logits.map((z) => Math.exp(z))
  const sum = exps.reduce((s, v) => s + v, 0)
  return exps.map((e) => e / sum)
}

/**
 * The stable version, written the same way as `softmax` in phase1-foundations/mlp_numpy.py:
 *   z = logits - logits.max();  e = np.exp(z);  return e / e.sum()
 */
export const stableSoftmax = (logits: Vec): Vec => {
  const max = Math.max(...logits)
  const exps = logits.map((z) => Math.exp(z - max))
  const sum = exps.reduce((s, v) => s + v, 0)
  return exps.map((e) => e / sum)
}

/** "Why not just divide by the sum?" The tempting wrong answer. */
export const divideBySum = (scores: Vec): Vec => {
  const sum = scores.reduce((s, v) => s + v, 0)
  return scores.map((v) => v / sum)
}

/** Roll the weighted die n times and add the results to `counts` (returns a new array). */
export const rollCounts = (probs: Vec, n: number, rng: Rng, counts?: number[]): number[] => {
  const out = counts ? counts.slice() : probs.map(() => 0)
  for (let i = 0; i < n; i++) out[sampleIndex(probs, rng.next())]++
  return out
}

/** Surprise = minus the natural log of the probability given to what actually happened. */
export const surprise = (p: number): number => -Math.log(p)

/** Display helper: ordinary numbers as decimals, astronomically large ones in scientific notation. */
export const fmtBig = (x: number): string => {
  if (!Number.isFinite(x)) return x > 0 ? '∞ (overflow)' : String(x)
  if (x !== 0 && (Math.abs(x) >= 1e6 || Math.abs(x) < 1e-3)) {
    const [m, e] = x.toExponential(2).split('e')
    return `${m} × 10^${Number(e)}`.replace('-', '−')
  }
  return x.toFixed(x >= 100 ? 1 : 3)
}
