import { describe, expect, it } from 'vitest'
import { cosine, dot } from './math'
import { describeDot, fromScreen, projection, snap, toScreen, verdictOf } from './vectors2d'

describe('describeDot reproduces math_primer.py section 2', () => {
  it('shopping bill: [2,1,3] . [4,0,2] = 8 + 0 + 6 = 14', () => {
    const r = describeDot([2, 1, 3], [4, 0, 2])
    expect(r.terms).toEqual([8, 0, 6])
    expect(r.dot).toBe(14)
  })
  it('taste vectors: friend agrees, stranger opposes', () => {
    const you = [0.9, 0.8, -0.5]
    expect(dot(you, [0.8, 0.9, -0.4])).toBeCloseTo(1.64, 10)
    expect(dot(you, [-0.6, -0.8, 0.9])).toBeCloseTo(-1.63, 10)
    expect(cosine(you, [0.8, 0.9, -0.4])).toBeCloseTo(0.991, 3)
    expect(cosine(you, [-0.6, -0.8, 0.9])).toBeCloseTo(-0.929, 3)
    expect(verdictOf(you, [0.8, 0.9, -0.4])).toBe('agree')
    expect(verdictOf(you, [-0.6, -0.8, 0.9])).toBe('oppose')
  })
  it('length of [3,4] is 5', () => {
    expect(describeDot([3, 4], [1, 0]).lenA).toBe(5)
  })
})

describe('numbers printed in the vectors lesson', () => {
  it('same direction, perpendicular, opposite', () => {
    expect(describeDot([3, 1], [3, 1]).dot).toBe(10)
    const perp = describeDot([3, 1], [-1, 3])
    expect(perp.dot).toBe(0)
    expect(perp.angleDeg).toBeCloseTo(90)
    expect(perp.verdict).toBe('unrelated')
    const opp = describeDot([3, 1], [-3, -1])
    expect(opp.dot).toBe(-10)
    expect(opp.cosine).toBeCloseTo(-1)
    expect(opp.verdict).toBe('oppose')
  })
  it('worked example A=[3,1], B=[2,4]', () => {
    const r = describeDot([3, 1], [2, 4])
    expect(r.terms).toEqual([6, 4])
    expect(r.dot).toBe(10)
    expect(r.lenA).toBeCloseTo(3.162, 3)
    expect(r.lenB).toBeCloseTo(4.472, 3)
    expect(r.cosine).toBeCloseTo(0.707, 3)
    expect(r.angleDeg).toBeCloseTo(45, 6)
  })
  it('a long vector wins the dot product but not the cosine', () => {
    const q = [2, 1]
    expect(dot(q, [2, 1])).toBe(5)
    expect(dot(q, [4, -2])).toBe(6)
    expect(cosine(q, [2, 1])).toBeCloseTo(1)
    expect(cosine(q, [4, -2])).toBeCloseTo(0.6)
  })
  it('exercises: 4-dimensional examples', () => {
    expect(dot([1, 2, 0, -1], [3, 1, 5, 2])).toBe(3)
    expect(dot([1, -2, 3, 0], [4, 1, -1, 7])).toBe(-1)
    expect(dot([1, 0, 1, 0], [0, 1, 0, 1])).toBe(0)
  })
})

describe('edge cases and geometry', () => {
  it('a zero vector has no direction', () => {
    const r = describeDot([0, 0], [1, 2])
    expect(r.verdict).toBe('zero')
    expect(Number.isNaN(r.angleDeg)).toBe(true)
  })
  it('projection: the shadow of B on A', () => {
    expect(projection([2, 0], [3, 4])).toEqual([3, 0])
    expect(projection([0, 0], [3, 4])).toEqual([0, 0])
  })
  it('snap: grid, clamp, no negative zero, no float dust', () => {
    expect(snap(1.26, 0.5, 5)).toBe(1.5)
    expect(snap(9, 0.5, 5)).toBe(5)
    expect(Object.is(snap(-0.1, 0.5, 5), 0)).toBe(true)
    expect(snap(0.1 + 0.2, 0.1, 5)).toBe(0.3)
  })
  it('screen mapping round-trips and flips y', () => {
    const plane = { size: 360, limit: 5 }
    expect(toScreen([0, 0], plane)).toEqual([180, 180])
    expect(toScreen([5, 5], plane)).toEqual([360, 0])
    const [sx, sy] = toScreen([2.5, -1], plane)
    expect(fromScreen(sx, sy, plane)).toEqual([2.5, -1])
  })
})
