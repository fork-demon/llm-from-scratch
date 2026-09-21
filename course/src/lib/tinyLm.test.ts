import { describe, expect, it } from 'vitest'
import { makeRng } from './rng'
import {
  PARAM_NAMES, baselineLoss, countParams, createModel, createTrainer, evalLoss, exampleCount, generate,
  lossAndGrads, lossOf, makeCorpus, makeDataset, predict, sampleBatch, sgdStep, trainSteps,
} from './tinyLm'

describe('data', () => {
  it('corpus is deterministic and splits 90/10 like tiny_gpt.py', () => {
    expect(makeCorpus(7)).toBe(makeCorpus(7))
    const text = makeCorpus()
    const ds = makeDataset(text)
    expect(ds.train.length).toBe(Math.floor(0.9 * text.length))
    expect(ds.train.length + ds.val.length).toBe(text.length)
    expect(ds.chars).toEqual([...ds.chars].sort())
    expect(ds.chars.length).toBe(27)
  })

  it('every position is an example: y is the character after the window', () => {
    const ds = makeDataset('hello world')
    expect(exampleCount(ds.train, 3)).toBe(ds.train.length - 3)
    const { X, Y } = sampleBatch(ds.train, 3, 8, makeRng(1))
    const ids = Array.from(ds.train)
    for (let i = 0; i < 8; i++) {
      const win = Array.from(X.slice(i * 3, i * 3 + 3))
      const at = ids.findIndex((_, s) => win.every((w, c) => ids[s + c] === w) && ids[s + 3] === Y[i])
      expect(at).toBeGreaterThanOrEqual(0)
    }
  })
})

describe('model', () => {
  const cfg = { vocabSize: 5, ctx: 2, dim: 3, hidden: 4 }

  it('counts parameters', () => {
    expect(countParams(cfg)).toBe(5 * 3 + 2 * 3 * 4 + 4 + 4 * 5 + 5)
    const m = createModel(cfg, 1)
    expect(PARAM_NAMES.reduce((s, n) => s + m.p[n].length, 0)).toBe(countParams(cfg))
  })

  it('probabilities sum to 1', () => {
    const p = predict(createModel(cfg, 1), [0, 3])
    expect(Array.from(p).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('analytic gradients match numerical gradients (gradient check)', () => {
    const m = createModel(cfg, 3)
    // bigger weights than the default init so ReLUs are firmly on or off and gradients are not tiny
    const rng = makeRng(11)
    for (const n of PARAM_NAMES) for (let i = 0; i < m.p[n].length; i++) m.p[n][i] = rng.normal()
    const X = Int32Array.from([0, 1, 2, 3, 4, 0, 1, 1, 3, 2])
    const Y = Int32Array.from([2, 4, 1, 0, 3])
    const { grads } = lossAndGrads(m, X, Y, 5)
    const eps = 1e-5
    let worst = 0
    for (const n of PARAM_NAMES) {
      for (let i = 0; i < m.p[n].length; i++) {
        const old = m.p[n][i]
        m.p[n][i] = old + eps
        const up = lossOf(m, X, Y, 5)
        m.p[n][i] = old - eps
        const down = lossOf(m, X, Y, 5)
        m.p[n][i] = old
        const numeric = (up - down) / (2 * eps)
        worst = Math.max(worst, Math.abs(numeric - grads[n][i]))
      }
    }
    expect(worst).toBeLessThan(1e-6)
  })

  it('an untrained model has loss close to ln(V)', () => {
    const ds = makeDataset(makeCorpus())
    const m = createModel({ vocabSize: ds.chars.length, ctx: 4, dim: 16, hidden: 64 }, 7)
    expect(baselineLoss(27)).toBeCloseTo(3.2958, 4)
    expect(Math.abs(evalLoss(m, ds.val) - Math.log(ds.chars.length))).toBeLessThan(0.05)
  })

  it('one SGD step on a batch lowers the loss on that batch', () => {
    const ds = makeDataset(makeCorpus())
    const m = createModel({ vocabSize: ds.chars.length, ctx: 3, dim: 8, hidden: 16 }, 7)
    const { X, Y } = sampleBatch(ds.train, 3, 32, makeRng(2))
    const { loss, grads } = lossAndGrads(m, X, Y, 32)
    sgdStep(m, grads, 0.3)
    expect(lossOf(m, X, Y, 32)).toBeLessThan(loss)
  })
})

describe('trainer', () => {
  const opts = { ctx: 4, dim: 16, hidden: 64, seed: 7, tinyData: false }

  it('loss decreases with a sane learning rate, on train and on validation', () => {
    const t = createTrainer(opts)
    trainSteps(t, 300, 0.3, 32)
    const first = t.history[0]
    const last = t.history[t.history.length - 1]
    expect(last.step).toBe(300)
    expect(t.tokensSeen).toBe(300 * 32)
    expect(last.train).toBeLessThan(first.train - 1)
    expect(last.val).toBeLessThan(first.val - 1)
    expect(t.samples.length).toBe(4) // steps 0, 100, 200, 300
  })

  it('is deterministic with the same seed, and different with another seed', () => {
    const a = createTrainer(opts); trainSteps(a, 60, 0.3, 16)
    const b = createTrainer(opts); trainSteps(b, 60, 0.3, 16)
    const c = createTrainer({ ...opts, seed: 8 }); trainSteps(c, 60, 0.3, 16)
    expect(a.history).toEqual(b.history)
    expect(a.samples).toEqual(b.samples)
    expect(a.history[a.history.length - 1].val).not.toBe(c.history[c.history.length - 1].val)
  })

  it('training in chunks gives the same result as training in one go', () => {
    const a = createTrainer(opts); trainSteps(a, 50, 0.3, 16)
    const b = createTrainer(opts); for (let i = 0; i < 10; i++) trainSteps(b, 5, 0.3, 16)
    expect(b.history).toEqual(a.history)
  })

  it('a tiny dataset overfits: train loss far below validation loss, and validation gets worse again', () => {
    const t = createTrainer({ ...opts, tinyData: true })
    trainSteps(t, 1000, 0.3, 32)
    const last = t.history[t.history.length - 1]
    const bestVal = Math.min(...t.history.map((h) => h.val))
    expect(last.train).toBeLessThan(0.5)
    expect(last.val).toBeGreaterThan(last.train + 1)
    expect(last.val).toBeGreaterThan(bestVal + 0.3)
  })

  it('a huge learning rate diverges and the trainer stops gracefully', () => {
    const t = createTrainer(opts)
    trainSteps(t, 500, 30, 32)
    expect(t.diverged).toBe(true)
    expect(t.step).toBeLessThan(500)
    expect(() => generate(t.model, t.ds, 'the ', 10, makeRng(1))).not.toThrow()
  })

  it('generates text of the requested length from the vocabulary', () => {
    const t = createTrainer(opts)
    const s = generate(t.model, t.ds, 'the cat ', 40, makeRng(3))
    expect(s.length).toBe(40)
    expect([...s].every((c) => t.ds.chars.includes(c))).toBe(true)
  })
})
