import { describe, expect, it } from 'vitest'
import { softmax } from './math'
import { TEACHER_LOGITS, chanceMissing, crossEntropySoft, drawIndex, entropy, gradient, kl, loss, oneHot, stepsToReach, train } from './distill'

describe('the quantities in the lesson', () => {
  const p = [0.7, 0.2, 0.1]
  const q = [0.5, 0.3, 0.2]
  it('worked KL example', () => {
    expect(kl(p, q)).toBeCloseTo(0.0851, 4)
    expect(kl(q, p)).toBeCloseTo(0.0920, 4) // KL is not symmetric
    expect(kl(p, p)).toBe(0)
  })
  it('soft cross-entropy = entropy of the teacher + KL', () => {
    expect(crossEntropySoft(p, q)).toBeCloseTo(0.8869, 4)
    expect(entropy(p)).toBeCloseTo(0.8018, 4)
    expect(crossEntropySoft(p, q)).toBeCloseTo(entropy(p) + kl(p, q), 12)
    expect(crossEntropySoft(oneHot(3, 0), q)).toBeCloseTo(Math.log(2), 12) // hard label: -ln 0.5
  })
  it('gradients: soft q - p, hard q - onehot', () => {
    const z = q.map(Math.log) // logits whose softmax is q
    gradient(z, p).forEach((g, i) => expect(g).toBeCloseTo([-0.2, 0.1, 0.1][i], 12))
    gradient(z, oneHot(3, 0)).forEach((g, i) => expect(g).toBeCloseTo([-0.5, 0.3, 0.2][i], 12))
  })
  it('temperature on the teacher logits [3, 1, -1]', () => {
    const r = (x: number[]) => x.map((v) => Number(v.toFixed(3)))
    expect(r(softmax([3, 1, -1], 1))).toEqual([0.867, 0.117, 0.016])
    expect(r(softmax([3, 1, -1], 2))).toEqual([0.665, 0.245, 0.09])
    expect(r(softmax(TEACHER_LOGITS, 1))).toEqual([0.702, 0.116, 0.095, 0.064, 0.021, 0.002])
    expect(r(softmax(TEACHER_LOGITS, 4))).toEqual([0.291, 0.186, 0.177, 0.16, 0.121, 0.065])
  })
  it('gradient matches a numerical derivative of the T^2-scaled loss', () => {
    const z = [0.3, -0.2, 0.5, 0.1]
    const target = softmax([1, 0, 2, -1], 3)
    const g = gradient(z, target, 3)
    const eps = 1e-6
    z.forEach((_, i) => {
      const up = z.slice(); up[i] += eps
      const dn = z.slice(); dn[i] -= eps
      expect((loss(up, target, 3) - loss(dn, target, 3)) / (2 * eps)).toBeCloseTo(g[i], 6)
    })
  })
  it('model collapse arithmetic', () => {
    expect(chanceMissing(0.01, 50)).toBeCloseTo(0.605, 3)
    expect(chanceMissing(0.01, 500)).toBeCloseTo(0.0066, 4)
  })
  it('drawIndex walks the cumulative distribution', () => {
    expect(drawIndex([0.2, 0.5, 0.3], 0.1)).toBe(0)
    expect(drawIndex([0.2, 0.5, 0.3], 0.69)).toBe(1)
    expect(drawIndex([0.2, 0.5, 0.3], 0.71)).toBe(2)
  })
})

describe('the student', () => {
  const hard = train('hard', 200)
  const sampled = train('sampled', 200)
  const soft1 = train('soft', 200, { T: 1 })
  const soft4 = train('soft', 200, { T: 4 })
  it('starts uniform, KL 0.802', () => {
    expect(hard.kl[0]).toBeCloseTo(0.802, 3)
    expect(soft4.kl[0]).toBe(hard.kl[0])
  })
  it('hard labels: KL falls, then rises as the student becomes overconfident', () => {
    const min = Math.min(...hard.kl)
    expect(hard.kl.indexOf(min)).toBe(7)
    expect(min).toBeCloseTo(0.098, 3)
    // at its best moment the leftover is spread evenly: "moon" gets as much as "sofa"
    const q7 = softmax(hard.logits[7])
    expect(q7[0]).toBeCloseTo(0.695, 3)
    expect(q7[1]).toBeCloseTo(q7[5], 12)
    expect(hard.kl[10]).toBeCloseTo(0.118, 3)
    expect(hard.kl[200]).toBeCloseTo(0.911, 3)
    expect(softmax(hard.logits[200])[0]).toBeGreaterThan(0.99)
    expect(stepsToReach(hard.kl, 0.05)).toBe(-1)
  })
  it('sampled labels: slow and noisy, but heading to the teacher', () => {
    expect(sampled.kl[200]).toBeLessThan(0.05)
    expect(stepsToReach(sampled.kl, 0.01)).toBe(-1)
    // in 200 draws the rare token "moon" (p = 0.0017) never appears
    const counts = [0, 0, 0, 0, 0, 0]
    sampled.targets.forEach((t) => counts[t.indexOf(1)]++)
    expect(counts).toEqual([135, 22, 20, 16, 7, 0])
  })
  it('soft targets reach the teacher; higher temperature gets there faster', () => {
    expect(stepsToReach(soft1.kl, 0.01)).toBe(94)
    expect(stepsToReach(train('soft', 200, { T: 2 }).kl, 0.01)).toBe(24)
    expect(stepsToReach(soft4.kl, 0.01)).toBe(18)
    // at T = 1 the rare token "moon" is still badly learned after 200 steps
    expect(softmax(soft1.logits[200])[5]).toBeCloseTo(0.008, 3)
    expect(softmax(soft4.logits[200])[5]).toBeCloseTo(0.0017, 4)
  })
  it('is deterministic', () => {
    expect(train('sampled', 50).kl).toEqual(train('sampled', 50).kl)
  })
})
