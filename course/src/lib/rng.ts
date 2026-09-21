/** Small seeded random generator (mulberry32) so demos and tests are reproducible. */
export const makeRng = (seed: number) => {
  let a = seed >>> 0
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  /** Standard normal via Box-Muller. */
  const normal = () => {
    const u = Math.max(next(), 1e-12)
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next())
  }
  const int = (n: number) => Math.floor(next() * n)
  return { next, normal, int }
}
export type Rng = ReturnType<typeof makeRng>
