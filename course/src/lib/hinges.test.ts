import { describe, expect, it } from 'vitest'
import { bendAt, collapsedLine, curveError, grid, netOutput, relu, sigmoid, SOLUTIONS, START_NET, TARGETS, unitOutput, type HingeNet } from './hinges'
import { matmul } from './math'

describe('a single hinge', () => {
  it('relu is flat then straight', () => {
    expect(relu(-3)).toBe(0)
    expect(relu(2.5)).toBe(2.5)
  })
  it('sigmoid squashes into 0..1', () => {
    expect(sigmoid(0)).toBeCloseTo(0.5)
    expect(sigmoid(10)).toBeGreaterThan(0.9999)
    expect(sigmoid(-10)).toBeLessThan(0.0001)
  })
  it('the bias moves the bend', () => {
    expect(bendAt({ w: 1, b: -1, v: 1 })).toBe(1)
    expect(bendAt({ w: 2, b: 1, v: 1 })).toBe(-0.5)
    expect(bendAt({ w: 0, b: 1, v: 1 })).toBeNull()
  })
  it('worked example from the lesson: x = 2, w = 1.5, b = -1, v = 2', () => {
    expect(unitOutput({ w: 1.5, b: -1, v: 2 }, 2)).toBeCloseTo(4) // relu(3 - 1) * 2
    expect(unitOutput({ w: 1.5, b: -1, v: 2 }, 0)).toBe(0) // relu(-1) = 0
  })
})

describe('hand-made solutions', () => {
  it('three hinges make the bump exactly', () => {
    expect(curveError(SOLUTIONS.tent, TARGETS.tent.f)).toBeCloseTo(0, 12)
  })
  it('two hinges make |x| exactly', () => {
    expect(curveError(SOLUTIONS.abs, TARGETS.abs.f)).toBeCloseTo(0, 12)
  })
  it('four hinges approximate x² closely but not exactly', () => {
    const e = curveError(SOLUTIONS.parabola, TARGETS.parabola.f)
    expect(e).toBeGreaterThan(0)
    expect(e).toBeLessThan(0.04)
    for (const x of [-2, -1, 0, 1, 2]) expect(netOutput(SOLUTIONS.parabola, x)).toBeCloseTo(x * x)
  })
  it('the starting network is not already a solution', () => {
    expect(curveError(START_NET, TARGETS.tent.f)).toBeGreaterThan(0.1)
  })
})

describe('without the activation everything collapses to one line', () => {
  const nets: HingeNet[] = [START_NET, SOLUTIONS.tent, SOLUTIONS.parabola]
  it('output equals slope*x + intercept for every x', () => {
    for (const net of nets) {
      const { slope, intercept } = collapsedLine(net)
      for (const x of grid(9)) expect(netOutput(net, x, false)).toBeCloseTo(slope * x + intercept)
    }
  })
  it('so a line can never fit the bump: the best line is far worse than three hinges', () => {
    // the bump solution without ReLU is the flat line y = 0 (slope 1 - 2 + 1 = 0)
    expect(collapsedLine(SOLUTIONS.tent)).toEqual({ slope: 0, intercept: 0 })
    expect(curveError(SOLUTIONS.tent, TARGETS.tent.f, false)).toBeGreaterThan(0.05)
  })
  it('the 2×2 example printed in the lesson: W2(W1 x) = (W2 W1) x', () => {
    const W1 = [[2, 0], [1, 1]]
    const W2 = [[1, -1], [0, 3]]
    const x = [[1], [2]]
    const twoSteps = matmul(W2, matmul(W1, x))
    const merged = matmul(W2, W1)
    expect(merged).toEqual([[1, -1], [3, 3]])
    expect(matmul(W1, x)).toEqual([[2], [3]])
    expect(twoSteps).toEqual([[-1], [9]])
    expect(matmul(merged, x)).toEqual(twoSteps)
  })
})
