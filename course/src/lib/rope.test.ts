import { describe, expect, it } from 'vitest'
import { dot, norm } from './math'
import { makeRng } from './rng'
import { additiveScore, degToRad, rope2d, ropeFrequencies, ropeRotate, ropeScore, ropeScoreRelative, rotate2d, scoreByOffset } from './rope'

describe('rotate2d', () => {
  it('rotates [1, 0] by 90 degrees to [0, 1]', () => {
    const r = rotate2d([1, 0], degToRad(90))
    expect(r[0]).toBeCloseTo(0, 12)
    expect(r[1]).toBeCloseTo(1, 12)
  })
  it('never changes the length of a vector', () => {
    const rng = makeRng(1)
    for (let t = 0; t < 20; t++) {
      const v = [rng.normal(), rng.normal()]
      expect(norm(rotate2d(v, rng.next() * 10))).toBeCloseTo(norm(v), 10)
    }
  })
})

describe('RoPE score depends only on the position difference', () => {
  it('worked example from the lesson: q = k = [1, 0], theta = 30 degrees, positions 5 and 3 -> cos(60) = 0.5', () => {
    const th = degToRad(30)
    expect(ropeScore([1, 0], [1, 0], 5, 3, th)).toBeCloseTo(0.5, 12)
    expect(ropeScore([1, 0], [1, 0], 105, 103, th)).toBeCloseTo(0.5, 9)
    expect(ropeScore([1, 0], [1, 0], 5, 5, th)).toBeCloseTo(1, 12)
  })
  it('lab defaults: q = [2, 1], k = [1, 2], theta = 20 degrees, positions 3 and 7', () => {
    const th = degToRad(20)
    const s = ropeScore([2, 1], [1, 2], 3, 7, th)
    expect(s).toBeCloseTo(ropeScore([2, 1], [1, 2], 103, 107, th), 9)
    expect(s).toBeCloseTo(4 * Math.cos(degToRad(80)) - 3 * Math.sin(degToRad(80)), 10)
  })
  it('shifting both positions by any amount leaves the score unchanged (random vectors)', () => {
    const rng = makeRng(42)
    for (let t = 0; t < 50; t++) {
      const q = [rng.normal(), rng.normal()]
      const k = [rng.normal(), rng.normal()]
      const m = rng.int(50)
      const n = rng.int(50)
      const shift = rng.int(1000)
      const th = rng.next()
      expect(ropeScore(q, k, m + shift, n + shift, th)).toBeCloseTo(ropeScore(q, k, m, n, th), 8)
      expect(ropeScore(q, k, m, n, th)).toBeCloseTo(ropeScoreRelative(q, k, n - m, th), 8)
    }
  })
  it('changing the difference does change the score', () => {
    const th = degToRad(20)
    expect(Math.abs(ropeScore([2, 1], [1, 2], 3, 7, th) - ropeScore([2, 1], [1, 2], 3, 8, th))).toBeGreaterThan(0.1)
  })
  it('at offset 0 the score is the plain dot product', () => {
    expect(ropeScore([2, 1], [1, 2], 9, 9, 0.7)).toBeCloseTo(dot([2, 1], [1, 2]), 10)
  })
  it('rope2d at position 0 is the identity', () => {
    expect(rope2d([2, 1], 0, 0.3)).toEqual([2, 1])
  })
})

describe('additive positions (contrast case)', () => {
  it('is NOT shift invariant', () => {
    const th = degToRad(20)
    const a = additiveScore([2, 1], [1, 2], 3, 7, th)
    const b = additiveScore([2, 1], [1, 2], 8, 12, th)
    expect(Math.abs(a - b)).toBeGreaterThan(0.5)
  })
})

describe('scoreByOffset', () => {
  it('returns one point per integer offset and matches ropeScore', () => {
    const th = degToRad(20)
    const pts = scoreByOffset([2, 1], [1, 2], th, -10, 10)
    expect(pts).toHaveLength(21)
    expect(pts[10].offset).toBe(0)
    expect(pts[10].score).toBeCloseTo(4, 10)
    expect(pts[14].score).toBeCloseTo(ropeScore([2, 1], [1, 2], 3, 7, th), 10)
  })
})

describe('full RoPE on longer vectors', () => {
  it('uses frequencies base^(-2i/d): first pair fastest (1), later pairs slower', () => {
    const f = ropeFrequencies(4)
    expect(f[0]).toBe(1)
    expect(f[1]).toBeCloseTo(0.01, 12)
    expect(() => ropeFrequencies(3)).toThrow()
  })
  it('keeps the shift-invariance property in 8 dimensions and preserves length', () => {
    const rng = makeRng(7)
    const q = Array.from({ length: 8 }, () => rng.normal())
    const k = Array.from({ length: 8 }, () => rng.normal())
    const base = dot(ropeRotate(q, 4), ropeRotate(k, 9))
    expect(dot(ropeRotate(q, 504), ropeRotate(k, 509))).toBeCloseTo(base, 8)
    expect(Math.abs(dot(ropeRotate(q, 4), ropeRotate(k, 10)) - base)).toBeGreaterThan(1e-3)
    expect(norm(ropeRotate(q, 123))).toBeCloseTo(norm(q), 10)
  })
})
