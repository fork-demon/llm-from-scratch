// Pure helpers for the MatrixPlayground (lesson 1.2): shape checking, the
// explanation of one result cell, and the "three loops" version of matmul.
// Mirrors sections 4 to 6 of phase1-foundations/math_primer.py.
import { shape, type Mat, type Vec } from './math'

export type Shape = [number, number]

export const SHAPES: Shape[] = [[2, 2], [2, 3], [3, 2]]
export const shapeLabel = (s: Shape): string => `(${s[0]}, ${s[1]})`

/** (n, k) @ (k, m): the inner numbers must match. */
export const canMultiply = (a: Shape, b: Shape): boolean => a[1] === b[0]

/** Output shape of a @ b, or null when the inner numbers differ. */
export const resultShape = (a: Shape, b: Shape): Shape | null => (canMultiply(a, b) ? [a[0], b[1]] : null)

export interface CellExplanation {
  row: Vec // row i of A
  col: Vec // column j of B
  terms: number[] // pairwise products
  sum: number
}

/** Result cell (i, j) = row i of A dotted with column j of B. */
export const explainCell = (A: Mat, B: Mat, i: number, j: number): CellExplanation => {
  const row = A[i]
  const col = B.map((r) => r[j])
  if (row.length !== col.length) throw new Error(`explainCell: inner dimensions differ (${row.length} vs ${col.length})`)
  const terms = row.map((v, t) => v * col[t])
  return { row, col, terms, sum: terms.reduce((s, v) => s + v, 0) }
}

/**
 * Matrix multiply written the long way: two loops choose the cell, the inner loop is a dot product.
 * This is the code shown in the lesson next to `A @ B`.
 */
export const matmulLoops = (A: Mat, B: Mat): Mat => {
  const [n, k] = shape(A)
  const [k2, m] = shape(B)
  if (k !== k2) throw new Error(`matmul: inner dimensions differ (${k} vs ${k2})`)
  const out: Mat = Array.from({ length: n }, () => Array(m).fill(0))
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++)
      for (let t = 0; t < k; t++) out[i][j] += A[i][t] * B[t][j]
  return out
}

/** Change a matrix to a new shape, keeping the numbers that still fit and filling new cells with 0. */
export const resize = (m: Mat, [rows, cols]: Shape): Mat =>
  Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => m[i]?.[j] ?? 0))

/** Which single transpose (if any) would make a @ b legal? */
export const transposeFix = (a: Shape, b: Shape): ('A' | 'B')[] => {
  const fixes: ('A' | 'B')[] = []
  if (canMultiply([a[1], a[0]], b)) fixes.push('A')
  if (canMultiply(a, [b[1], b[0]])) fixes.push('B')
  return fixes
}
