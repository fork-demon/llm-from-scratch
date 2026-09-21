import { describe, expect, it } from 'vitest'
import { matmul, transpose } from './math'
import { canMultiply, explainCell, matmulLoops, resize, resultShape, shapeLabel, transposeFix } from './matrixPlay'

// the worked example from phase1-foundations/math_primer.py section 5
const A = [[1, 2], [3, 0]]
const B = [[1, 0, 1], [2, 1, 0]]

describe('matmul reproduces math_primer.py', () => {
  it('A @ B = [[5,2,1],[3,0,3]] with both implementations', () => {
    expect(matmul(A, B)).toEqual([[5, 2, 1], [3, 0, 3]])
    expect(matmulLoops(A, B)).toEqual([[5, 2, 1], [3, 0, 3]])
  })
  it('W @ x from section 4: two dot products give [4, 4]', () => {
    const W = [[1, 0, 2], [0, 3, 1]]
    expect(matmul(W, [[2], [1], [1]])).toEqual([[4], [4]])
  })
  it('order matters: C@D != D@C', () => {
    const C = [[0, 1], [1, 0]]
    const D = [[2, 0], [0, 1]]
    expect(matmul(C, D)).toEqual([[0, 1], [2, 0]])
    expect(matmul(D, C)).toEqual([[0, 2], [1, 0]])
  })
  it('X @ X.T is the all-pairs similarity grid (section 6)', () => {
    const X = [[1, 0], [0, 1], [1, 1]]
    expect(matmul(X, transpose(X))).toEqual([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
  })
  it('three tokens through one layer (lesson example)', () => {
    const X = [[1, 0], [0, 1], [1, 1]]
    const W = [[2, 0, 1], [1, 3, 0]]
    expect(matmul(X, W)).toEqual([[2, 0, 1], [1, 3, 0], [3, 3, 1]])
  })
  it('exercise: [[2,0],[1,3]] @ [[1,4],[2,5]]', () => {
    expect(matmul([[2, 0], [1, 3]], [[1, 4], [2, 5]])).toEqual([[2, 8], [7, 19]])
  })
})

describe('explainCell', () => {
  it('cell (0,0) = [1,2] . [1,2] = 1 + 4 = 5', () => {
    expect(explainCell(A, B, 0, 0)).toEqual({ row: [1, 2], col: [1, 2], terms: [1, 4], sum: 5 })
  })
  it('cell (1,2) = [3,0] . [1,0] = 3', () => {
    expect(explainCell(A, B, 1, 2)).toEqual({ row: [3, 0], col: [1, 0], terms: [3, 0], sum: 3 })
  })
  it('agrees with matmul in every cell', () => {
    const out = matmul(A, B)
    out.forEach((r, i) => r.forEach((v, j) => expect(explainCell(A, B, i, j).sum).toBe(v)))
  })
  it('refuses mismatched shapes', () => {
    expect(() => explainCell(B, B, 0, 0)).toThrow()
    expect(() => matmulLoops(B, B)).toThrow(/inner/)
  })
})

describe('shape rule', () => {
  it('inner numbers must match, outer numbers are the answer', () => {
    expect(canMultiply([2, 3], [3, 2])).toBe(true)
    expect(canMultiply([2, 3], [2, 3])).toBe(false)
    expect(resultShape([5, 3], [3, 3])).toEqual([5, 3])
    expect(resultShape([4, 8], [8, 100])).toEqual([4, 100])
    expect(resultShape([2, 3], [2, 3])).toBeNull()
    expect(shapeLabel([2, 3])).toBe('(2, 3)')
  })
  it('suggests the transpose that fixes a mismatch', () => {
    expect(transposeFix([2, 3], [2, 3])).toEqual(['A', 'B'])
    expect(transposeFix([3, 2], [3, 2])).toEqual(['A', 'B'])
    expect(transposeFix([2, 2], [3, 2])).toEqual(['B'])
    expect(transposeFix([2, 3], [2, 2])).toEqual(['A'])
  })
  it('resize keeps what fits and pads with zeros', () => {
    expect(resize([[1, 2], [3, 4]], [2, 3])).toEqual([[1, 2, 0], [3, 4, 0]])
    expect(resize([[1, 2, 3], [4, 5, 6]], [3, 2])).toEqual([[1, 2], [4, 5], [0, 0]])
  })
})
