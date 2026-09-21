import { describe, expect, it } from 'vitest'
import { applyUpdate, backward, DEMO_NET, DEMO_TARGET, DEMO_X, forward, lossOf, numericalGrad, type TinyNet, type WeightRef } from './tinyMlp'
import { makeRng } from './rng'

// Reference numbers: phase1-foundations/mlp_numpy.py, MLP(sizes=(2, 2, 2)) with the same
// weights, X = [[1, 2]], y = [1], then net.forward / net.backward / net.step.
describe('reproduces mlp_numpy.py on the demo network', () => {
  const cache = forward(DEMO_NET, DEMO_X, DEMO_TARGET)
  const g = backward(DEMO_NET, cache, DEMO_TARGET)
  it('forward', () => {
    expect(cache.z1).toEqual([2, -0.5])
    expect(cache.h).toEqual([2, 0])
    expect(cache.logits).toEqual([1, -1])
    expect(cache.probs[0]).toBeCloseTo(0.880797, 6)
    expect(cache.probs[1]).toBeCloseTo(0.119203, 6)
    expect(cache.loss).toBeCloseTo(2.126928, 6)
  })
  it('backward', () => {
    expect(g.dLogits[0]).toBeCloseTo(0.880797, 6)
    expect(g.dLogits[1]).toBeCloseTo(-0.880797, 6)
    expect(g.dW2[0][0]).toBeCloseTo(1.761594, 6)
    expect(g.dW2[0][1]).toBeCloseTo(-1.761594, 6)
    expect(g.dW2[1][0]).toBeCloseTo(0, 12) // hidden unit 2 output 0, so its outgoing weights get no blame
    expect(g.dW2[1][1]).toBeCloseTo(0, 12)
    expect(g.dh[0]).toBeCloseTo(0.880797, 6)
    expect(g.dh[1]).toBeCloseTo(-0.440399, 6)
    expect(g.dz1[0]).toBeCloseTo(0.880797, 6)
    expect(g.dz1[1]).toBe(0) // the ReLU gate: hidden unit 2 was shut
    expect(g.dW1[0][0]).toBeCloseTo(0.880797, 6)
    expect(g.dW1[1][0]).toBeCloseTo(1.761594, 6)
    expect(g.dW1[0][1]).toBe(0)
    expect(g.db1[0]).toBeCloseTo(0.880797, 6)
  })
  it('update: the loss drops, by the same amounts as net.step in Python', () => {
    expect(lossOf(applyUpdate(DEMO_NET, g, 0.1), DEMO_X, DEMO_TARGET)).toBeCloseTo(1.155231, 5)
    expect(lossOf(applyUpdate(DEMO_NET, g, 0.5), DEMO_X, DEMO_TARGET)).toBeCloseTo(0.346742, 5)
  })
})

const allRefs: WeightRef[] = [1, 2].flatMap((layer) => [0, 1].flatMap((i) => [0, 1].map((j) => ({ layer: layer as 1 | 2, i, j }))))
const analytic = (g: ReturnType<typeof backward>, r: WeightRef) => (r.layer === 1 ? g.dW1 : g.dW2)[r.i][r.j]

describe('gradient check: backprop vs the numerical oracle', () => {
  it('agrees on every weight of the demo network', () => {
    const g = backward(DEMO_NET, forward(DEMO_NET, DEMO_X, DEMO_TARGET), DEMO_TARGET)
    for (const r of allRefs) expect(Math.abs(numericalGrad(DEMO_NET, DEMO_X, DEMO_TARGET, r).grad - analytic(g, r))).toBeLessThan(1e-7)
  })
  it('agrees on random networks, inputs and targets', () => {
    const rng = makeRng(11)
    for (let t = 0; t < 25; t++) {
      // modest weights: with huge logits the 1e-12 guard inside crossEntropy (kept from the Python file) would dominate
      const m = () => [[0.5 * rng.normal(), 0.5 * rng.normal()], [0.5 * rng.normal(), 0.5 * rng.normal()]]
      const net: TinyNet = { W1: m(), b1: [rng.normal(), rng.normal()], W2: m(), b2: [rng.normal(), rng.normal()] }
      const x = [rng.normal(), rng.normal()]
      const target = rng.int(2)
      const g = backward(net, forward(net, x, target), target)
      for (const r of allRefs) expect(Math.abs(numericalGrad(net, x, target, r).grad - analytic(g, r))).toBeLessThan(1e-6)
    }
  })
  it('catches the planted bug: forgetting the ReLU gate', () => {
    const buggy = backward(DEMO_NET, forward(DEMO_NET, DEMO_X, DEMO_TARGET), DEMO_TARGET, { reluGate: false })
    const ref: WeightRef = { layer: 1, i: 1, j: 1 } // feeds the shut hidden unit
    expect(numericalGrad(DEMO_NET, DEMO_X, DEMO_TARGET, ref).grad).toBeCloseTo(0, 8)
    expect(buggy.dW1[1][1]).toBeCloseTo(-0.880797, 5) // the buggy code blames a weight that did nothing
  })
})

describe('the miss', () => {
  it('d_logits = probs - one_hot sums to zero and is negative only at the target', () => {
    const c = forward(DEMO_NET, DEMO_X, 0)
    const g = backward(DEMO_NET, c, 0)
    expect(g.dLogits[0] + g.dLogits[1]).toBeCloseTo(0)
    expect(g.dLogits[0]).toBeLessThan(0)
    expect(g.dLogits[1]).toBeGreaterThan(0)
  })
  it('repeated updates at lr 0.1 lower the loss every time and keep hidden unit 1 alive (the sequence quoted in the lesson)', () => {
    let net = DEMO_NET
    const losses: number[] = []
    for (let s = 0; s < 50; s++) {
      const c = forward(net, DEMO_X, DEMO_TARGET)
      expect(c.h[0]).toBeGreaterThan(0)
      losses.push(c.loss)
      net = applyUpdate(net, backward(net, c, DEMO_TARGET), 0.1)
    }
    expect(losses.slice(0, 4).map((l) => l.toFixed(2))).toEqual(['2.13', '1.16', '0.81', '0.65'])
    for (let i = 1; i < losses.length; i++) expect(losses[i]).toBeLessThan(losses[i - 1])
    expect(losses[49]).toBeLessThan(0.1)
  })
  it('a step that is too large (lr 0.5) shuts both hinges: every weight gradient becomes 0', () => {
    const g0 = backward(DEMO_NET, forward(DEMO_NET, DEMO_X, DEMO_TARGET), DEMO_TARGET)
    const net = applyUpdate(DEMO_NET, g0, 0.5)
    const c = forward(net, DEMO_X, DEMO_TARGET)
    expect(c.h).toEqual([0, 0])
    const g = backward(net, c, DEMO_TARGET)
    expect([...g.dW1.flat(), ...g.dW2.flat()].every((v) => v === 0)).toBe(true)
  })
})
