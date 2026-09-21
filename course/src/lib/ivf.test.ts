import { describe, expect, it } from 'vitest'
import { buildIvf, makeClusteredPoints, measure, searchExact, searchIvf, type P2 } from './ivf'
import { makeRng } from './rng'

describe('IVF index (the idea from vector_db.py, in 2-D)', () => {
  const points = makeClusteredPoints()
  const index = buildIvf(points, 8)

  it('every point lands in exactly one bucket', () => {
    expect(index.buckets.flat().sort((a, b) => a - b)).toEqual(points.map((_, i) => i))
    expect(index.buckets.every((b) => b.length > 0)).toBe(true)
  })
  it('is deterministic for a seed', () => {
    expect(buildIvf(points, 8).centroids).toEqual(index.centroids)
  })
  it('nprobe = all clusters gives exactly the exact-search result', () => {
    const rng = makeRng(5)
    for (let t = 0; t < 100; t++) {
      const q: P2 = [rng.next(), rng.next()]
      expect(searchIvf(index, q, 5, 8).result).toEqual(searchExact(points, q, 5))
    }
  })
  it('nprobe = 1 compares far fewer points, and sometimes misses', () => {
    const one = measure(index, 1)
    const all = measure(index, 8)
    expect(all.recall).toBe(1)
    expect(all.fraction).toBeCloseTo(1)
    expect(one.fraction).toBeLessThan(0.3)
    expect(one.recall).toBeLessThan(1)
    expect(one.recall).toBeGreaterThan(0.6)
  })
  it('recall never goes down as nprobe goes up', () => {
    const r = [1, 2, 3, 4, 5, 6, 7, 8].map((n) => measure(index, n).recall)
    for (let i = 1; i < r.length; i++) expect(r[i]).toBeGreaterThanOrEqual(r[i - 1])
  })
  it('probes the nearest centroid first and counts centroid comparisons', () => {
    const q: P2 = index.centroids[3]
    const r = searchIvf(index, q, 1, 1)
    expect(r.probed).toEqual([3])
    expect(r.compared).toBe(index.buckets[3].length + 8)
  })
})
