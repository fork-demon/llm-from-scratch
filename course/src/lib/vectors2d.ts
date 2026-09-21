// Pure helpers for the VectorPlayground (lesson 1.1): the dot product spelled
// out term by term, a plain-words verdict, and plane <-> screen geometry.
// Same arithmetic as section 2 of phase1-foundations/math_primer.py.
import { cosine, dot, norm, type Vec } from './math'

export type Verdict = 'agree' | 'unrelated' | 'oppose' | 'zero'

export interface DotReport {
  terms: number[] // the pairwise products, one per slot
  dot: number
  lenA: number
  lenB: number
  cosine: number
  angleDeg: number // NaN when either vector is all zeros
  verdict: Verdict
}

/** |cosine| below this counts as "unrelated" in the plain-words verdict. */
export const UNRELATED_BAND = 0.15

export const verdictOf = (a: Vec, b: Vec): Verdict => {
  if (norm(a) === 0 || norm(b) === 0) return 'zero'
  const c = cosine(a, b)
  if (Math.abs(c) < UNRELATED_BAND) return 'unrelated'
  return c > 0 ? 'agree' : 'oppose'
}

export const describeDot = (a: Vec, b: Vec): DotReport => {
  const terms = a.map((v, i) => v * b[i])
  const c = Math.max(-1, Math.min(1, cosine(a, b)))
  const zero = norm(a) === 0 || norm(b) === 0
  return {
    terms,
    dot: dot(a, b),
    lenA: norm(a),
    lenB: norm(b),
    cosine: c,
    angleDeg: zero ? NaN : (Math.acos(c) * 180) / Math.PI,
    verdict: verdictOf(a, b),
  }
}

/** The "shadow" of b on the line through a: the part of b that points along a. */
export const projection = (a: Vec, b: Vec): Vec => {
  const aa = dot(a, a)
  if (aa === 0) return a.map(() => 0)
  const k = dot(a, b) / aa
  return a.map((v) => v * k)
}

/** Snap to a grid step and clamp to [-limit, limit]. Avoids -0 and float dust like 0.30000000000000004. */
export const snap = (x: number, step: number, limit: number): number => {
  const s = Math.round(x / step) * step
  const c = Math.max(-limit, Math.min(limit, s))
  const r = Math.round(c * 1e6) / 1e6
  return r === 0 ? 0 : r
}

export interface Plane { size: number; limit: number } // square svg of `size` units showing [-limit, limit]

export const toScreen = (p: Vec, plane: Plane): [number, number] => {
  const unit = plane.size / (2 * plane.limit)
  return [plane.size / 2 + p[0] * unit, plane.size / 2 - p[1] * unit]
}

export const fromScreen = (sx: number, sy: number, plane: Plane): Vec => {
  const unit = plane.size / (2 * plane.limit)
  return [(sx - plane.size / 2) / unit, (plane.size / 2 - sy) / unit]
}
