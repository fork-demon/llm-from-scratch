// Rotary position embeddings (RoPE), small enough to check by hand.
// The 2D case is the whole idea: rotate the query by (its position x theta) and the key by
// (its position x theta). Their dot product then depends only on the position DIFFERENCE.
// Real models do this to every consecutive pair of numbers in q and k, each pair with its own theta.
import { dot, type Vec } from './math'

export const degToRad = (deg: number): number => (deg * Math.PI) / 180

/** Rotate a 2D vector counter-clockwise by `angle` radians. Length is unchanged. */
export const rotate2d = (v: Vec, angle: number): Vec => {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  return [v[0] * c - v[1] * s, v[0] * s + v[1] * c]
}

/** RoPE in 2D: the vector at position `pos` is rotated by pos x theta (theta in radians). */
export const rope2d = (v: Vec, pos: number, theta: number): Vec => rotate2d(v, pos * theta)

/** Attention score (before scaling and softmax) between a query at position m and a key at position n. */
export const ropeScore = (q: Vec, k: Vec, m: number, n: number, theta: number): number =>
  dot(rope2d(q, m, theta), rope2d(k, n, theta))

/**
 * The same score written so that the positions only appear as a difference:
 * rotate the key by (n - m) x theta and leave the query alone.
 */
export const ropeScoreRelative = (q: Vec, k: Vec, offset: number, theta: number): number =>
  dot(q, rotate2d(k, offset * theta))

/**
 * A toy ADDITIVE scheme for contrast (what tiny_gpt.py does in spirit: x = tok_emb + pos_emb).
 * The position vector here is a point on the unit circle; a learned table has even less structure.
 * The score contains cross terms (q . p_n and p_m . k) that depend on absolute positions.
 */
export const positionVector = (pos: number, theta: number): Vec => [Math.cos(pos * theta), Math.sin(pos * theta)]

export const additiveScore = (q: Vec, k: Vec, m: number, n: number, theta: number): number => {
  const pm = positionVector(m, theta)
  const pn = positionVector(n, theta)
  return dot([q[0] + pm[0], q[1] + pm[1]], [k[0] + pn[0], k[1] + pn[1]])
}

/** Score as a function of the offset (key position minus query position), for plotting. */
export const scoreByOffset = (q: Vec, k: Vec, theta: number, minOffset: number, maxOffset: number): { offset: number; score: number }[] => {
  const out: { offset: number; score: number }[] = []
  for (let o = minOffset; o <= maxOffset; o++) out.push({ offset: o, score: ropeScoreRelative(q, k, o, theta) })
  return out
}

/** Per-pair rotation speeds used by real RoPE: theta_i = base^(-2i/d), i = 0 .. d/2 - 1. */
export const ropeFrequencies = (d: number, base = 10000): Vec => {
  if (d % 2 !== 0) throw new Error('rope: dimension must be even')
  return Array.from({ length: d / 2 }, (_, i) => base ** ((-2 * i) / d))
}

/** Full RoPE on an even-length vector: consecutive pairs (x0,x1), (x2,x3), ... each rotated at its own speed. */
export const ropeRotate = (x: Vec, pos: number, base = 10000): Vec => {
  const freqs = ropeFrequencies(x.length, base)
  const out: number[] = []
  freqs.forEach((f, i) => out.push(...rotate2d([x[2 * i], x[2 * i + 1]], pos * f)))
  return out
}
