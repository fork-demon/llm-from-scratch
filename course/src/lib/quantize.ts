// Weight quantization: a TypeScript mirror of phase6-engineering/quantize_demo.py.
// Symmetric, absmax, round-to-nearest: the baseline every real method starts from.
import { matmul, transpose, type Mat, type Vec } from './math'
import { makeRng } from './rng'

export type Granularity = 'tensor' | 'row' | 'group'

/** Largest integer level: int8 -> 127, int4 -> 7. */
export const qmax = (bits: number): number => 2 ** (bits - 1) - 1

/** Round half to even, like numpy.round, so ties land where the Python file puts them. */
export const roundHalfEven = (x: number): number => {
  const f = Math.floor(x)
  const d = x - f
  if (d < 0.5) return f
  if (d > 0.5) return f + 1
  return f % 2 === 0 ? f : f + 1
}

const absmax = (w: Vec): number => w.reduce((m, x) => Math.max(m, Math.abs(x)), 0)

/** One scale for the whole array. Returns the stored integers and the scale. */
export const quantizeBlock = (w: Vec, bits: number): { q: Vec; scale: number } => {
  const scale = absmax(w) / qmax(bits)
  if (scale === 0) return { q: w.map(() => 0), scale: 1 }
  const lim = qmax(bits)
  return { q: w.map((x) => Math.min(lim, Math.max(-lim, roundHalfEven(x / scale))) + 0), scale }
}

export interface Quantized { What: Mat; Q: Mat; scales: number[]; nScales: number }

/** Quantize then dequantize, so the result can be compared with W directly.
 *  tensor: one scale for the matrix. row: one per output channel. group: one per `group` consecutive weights in a row. */
export const fakeQuantize = (W: Mat, bits: number, granularity: Granularity, group = 8): Quantized => {
  const cols = W[0].length
  if (granularity === 'tensor') {
    const { q, scale } = quantizeBlock(W.flat(), bits)
    const Q = W.map((_, i) => q.slice(i * cols, (i + 1) * cols))
    return { What: Q.map((r) => r.map((x) => x * scale)), Q, scales: [scale], nScales: 1 }
  }
  const size = granularity === 'row' ? cols : group
  if (cols % size !== 0) throw new Error('row length must be a multiple of the group size')
  const Q: Mat = []
  const What: Mat = []
  const scales: number[] = []
  for (const row of W) {
    const qRow: Vec = []
    const hRow: Vec = []
    for (let g = 0; g < cols; g += size) {
      const { q, scale } = quantizeBlock(row.slice(g, g + size), bits)
      scales.push(scale)
      qRow.push(...q)
      hRow.push(...q.map((x) => x * scale))
    }
    Q.push(qRow)
    What.push(hRow)
  }
  return { What, Q, scales, nScales: scales.length }
}

/** Relative error: size of the difference divided by the size of the original (Frobenius norm). */
export const relErr = (A: Mat, B: Mat): number => {
  let num = 0, den = 0
  A.forEach((row, i) => row.forEach((a, j) => { num += (a - B[i][j]) ** 2; den += a * a }))
  return Math.sqrt(num / den)
}

/** y = x @ W.T, the PyTorch Linear convention: W has one row per output channel. */
export const linear = (X: Mat, W: Mat): Mat => matmul(X, transpose(W))

export const outputError = (W: Mat, What: Mat, X: Mat): number => relErr(linear(X, W), linear(X, What))

/** Storage per weight including its share of the 16-bit scales. group = undefined means one scale per row. */
export const bitsPerWeight = (bits: number, group?: number, scaleBits = 16, rowLen = 4096): number => bits + scaleBits / (group ?? rowLen)

export const modelGB = (params: number, bpw: number): number => (params * bpw) / 8 / 1e9

/** Upper bound for one stream: every generated token reads every weight once. */
export const decodeTokPerSBound = (bandwidthBytesPerS: number, weightGB: number): number => bandwidthBytesPerS / (weightGB * 1e9)

/** A small seeded layer for the lab: W ~ N(0, 0.02), X ~ N(0, 1). */
export const demoMatrices = (rows = 8, cols = 32, nX = 16, seed = 17): { W: Mat; X: Mat } => {
  const rng = makeRng(seed)
  const W = Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0.02 * rng.normal()))
  const X = Array.from({ length: nX }, () => Array.from({ length: cols }, () => rng.normal()))
  return { W, X }
}

export const plantOutlier = (W: Mat, i: number, j: number, value: number): Mat => W.map((row, r) => (r === i ? row.map((x, c) => (c === j ? value : x)) : row))
