// The IVF idea from phase4-modern-llms/vector_db.py, in 2-D so it can be drawn:
// cluster the vectors once (k-means), then at query time compare the query with
// the centroids and brute-force ONLY inside the nprobe nearest clusters.
//
// One deliberate difference: vector_db.py normalises every vector and ranks by
// dot product (cosine). Here we rank by straight-line distance, because that is
// what the eye sees on a 2-D scatter plot. For unit-length vectors the two give
// the same ranking, so the algorithm and its trade-off are unchanged.
import { makeRng, type Rng } from './rng'

export type P2 = [number, number]

const d2 = (a: P2, b: P2) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

/** Clumpy data, like real embeddings (text clumps by topic). Coordinates stay inside [0, 1]. */
export const makeClusteredPoints = (seed = 11, nBlobs = 7, perBlob = 40, spread = 0.07): P2[] => {
  const rng = makeRng(seed)
  const clamp = (x: number) => Math.min(0.98, Math.max(0.02, x))
  const centers: P2[] = Array.from({ length: nBlobs }, () => [0.15 + 0.7 * rng.next(), 0.15 + 0.7 * rng.next()])
  const pts: P2[] = []
  for (let i = 0; i < nBlobs * perBlob; i++) {
    const c = centers[i % nBlobs]
    pts.push([clamp(c[0] + spread * rng.normal()), clamp(c[1] + spread * rng.normal())])
  }
  return pts
}

/** k-means, as in vector_db.py: assign each point to its nearest centroid, move centroids to the mean, repeat. */
export const kmeans = (X: P2[], k: number, rng: Rng, iters = 20): { centroids: P2[]; assign: number[] } => {
  const pool = X.map((_, i) => i)
  const centroids: P2[] = []
  for (let j = 0; j < k; j++) centroids.push(X[pool.splice(rng.int(pool.length), 1)[0]].slice() as P2) // k distinct points
  let assign = new Array<number>(X.length).fill(0)
  for (let it = 0; it < iters; it++) {
    assign = X.map((p) => nearestOrder(p, centroids)[0])
    for (let j = 0; j < k; j++) {
      const members = X.filter((_, i) => assign[i] === j)
      if (members.length) centroids[j] = [mean(members.map((m) => m[0])), mean(members.map((m) => m[1]))]
    }
  }
  assign = X.map((p) => nearestOrder(p, centroids)[0])
  return { centroids, assign }
}

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length

/** Indices of `targets`, nearest first. */
export const nearestOrder = (q: P2, targets: P2[]): number[] =>
  targets.map((t, i) => [d2(q, t), i] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(([, i]) => i)

export interface IvfIndex { points: P2[]; centroids: P2[]; assign: number[]; buckets: number[][] }

export const buildIvf = (points: P2[], nClusters: number, seed = 3): IvfIndex => {
  const { centroids, assign } = kmeans(points, nClusters, makeRng(seed))
  const buckets: number[][] = centroids.map(() => [])
  assign.forEach((c, i) => buckets[c].push(i))
  return { points, centroids, assign, buckets }
}

/** Exact search: compare the query with every point. */
export const searchExact = (points: P2[], q: P2, k = 1): number[] => nearestOrder(q, points).slice(0, k)

export interface IvfResult {
  probed: number[] // cluster ids searched, nearest centroid first
  compared: number // distance computations: all centroids + every point in the probed buckets
  candidates: number // points in the probed buckets
  result: number[] // point indices, nearest first
}

export const searchIvf = (index: IvfIndex, q: P2, k = 1, nprobe = 1): IvfResult => {
  const probed = nearestOrder(q, index.centroids).slice(0, Math.max(1, nprobe))
  const cand = probed.flatMap((c) => index.buckets[c])
  const result = nearestOrder(q, cand.map((i) => index.points[i])).slice(0, k).map((j) => cand[j])
  return { probed, candidates: cand.length, compared: cand.length + index.centroids.length, result }
}

/** Average recall@k and fraction of points compared, over a grid of queries. */
export const measure = (index: IvfIndex, nprobe: number, k = 1, grid = 15): { recall: number; fraction: number } => {
  let hit = 0
  let cand = 0
  let n = 0
  for (let a = 0; a < grid; a++) {
    for (let b = 0; b < grid; b++) {
      const q: P2 = [(a + 0.5) / grid, (b + 0.5) / grid]
      const truth = new Set(searchExact(index.points, q, k))
      const r = searchIvf(index, q, k, nprobe)
      hit += r.result.filter((i) => truth.has(i)).length / k
      cand += r.candidates
      n++
    }
  }
  return { recall: hit / n, fraction: cand / n / index.points.length }
}
