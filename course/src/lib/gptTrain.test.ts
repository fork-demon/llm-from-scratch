import { describe, expect, it } from 'vitest'
import {
  adamwStep, backward, countParams, createGpt, createSession, DEFAULT_SIZES, DEFAULT_TRAIN, forward, generate, gelu,
  getBatch, inspect, makeCharData, paramGroups, trainFor, type GptConfig,
} from './gptTrain'
import { makeRng } from './rng'

const TINY: GptConfig = { vocabSize: 7, contextLen: 5, nEmbd: 8, nHead: 2, nLayer: 2 }

describe('gptTrain: shapes and parameter count', () => {
  it('matches the PyTorch parameter count of tiny_gpt.py (tied head)', () => {
    // per block: 2 LN (4D) + qkv (3D*D + 3D) + proj (D*D + D) + fc (4D*D + 4D) + fc2 (4D*D + D) = 12D^2 + 13D
    const cfg: GptConfig = { vocabSize: 65, contextLen: 64, nEmbd: 128, nHead: 4, nLayer: 4 }
    const D = 128
    expect(countParams(cfg)).toBe(65 * D + 64 * D + 4 * (12 * D * D + 13 * D) + 2 * D)
    const groups = paramGroups(cfg)
    expect(groups[groups.length - 1].offset + groups[groups.length - 1].size).toBe(countParams(cfg))
  })

  it('GELU is the exact (erf) version used by nn.GELU()', () => {
    expect(gelu(0)).toBe(0)
    expect(gelu(1)).toBeCloseTo(0.8413447, 6)
    expect(gelu(-1)).toBeCloseTo(-0.1586553, 6)
  })
})

describe('gptTrain: gradients', () => {
  it('backward agrees with central finite differences for every parameter group', () => {
    const m = createGpt(TINY, 3, true)
    // bigger random weights than the 0.02 init so every path carries a real signal
    const rng = makeRng(11)
    for (const g of m.groups) for (let i = 0; i < g.size; i++) m.params[g.offset + i] += 0.3 * rng.normal()
    const B = 2, T = 5
    const x = Int32Array.from({ length: B * T }, () => rng.int(TINY.vocabSize))
    const y = Int32Array.from({ length: B * T }, () => rng.int(TINY.vocabSize))
    forward(m, x, B, T, y)
    backward(m, x, B, T, y)
    const analytic = Float64Array.from(m.grads)
    const h = 1e-5
    const checked: string[] = []
    for (const g of m.groups) {
      let worst = 0
      for (let k = 0; k < Math.min(g.size, 12); k++) {
        const i = g.offset + ((k * 7919) % g.size)
        const old = m.params[i]
        m.params[i] = old + h
        const lp = forward(m, x, B, T, y)
        m.params[i] = old - h
        const lm = forward(m, x, B, T, y)
        m.params[i] = old
        const num = (lp - lm) / (2 * h)
        const err = Math.abs(num - analytic[i]) / Math.max(1e-3, Math.abs(num) + Math.abs(analytic[i])) // floor: the key bias has an exactly-zero gradient (softmax ignores a shared shift)
        worst = Math.max(worst, err)
      }
      expect(worst, g.name).toBeLessThan(1e-6)
      checked.push(g.name)
    }
    expect(checked.length).toBe(2 + 12 * TINY.nLayer + 2)
  })

  it('attention rows are causal probability distributions', () => {
    const m = createGpt(TINY, 1)
    const { attn, probs } = inspect(m, [1, 2, 3, 4])
    expect(attn.length).toBe(2)
    expect(attn[0].length).toBe(2)
    for (const layer of attn) for (const head of layer) head.forEach((row, t) => {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5)
      row.forEach((p, t2) => { if (t2 > t) expect(p).toBe(0) })
    })
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5)
  })
})

describe('gptTrain: loss and training', () => {
  it('loss at init is close to ln(vocab) because the init is small', () => {
    const V = 40
    const m = createGpt({ ...DEFAULT_SIZES, vocabSize: V }, 1337)
    const rng = makeRng(5)
    const data = Int32Array.from({ length: 2000 }, () => rng.int(V))
    const { x, y } = getBatch(data, 8, 32, rng)
    const loss = forward(m, x, 8, 32, y)
    expect(Math.abs(loss - Math.log(V))).toBeLessThan(0.05)
  })

  it('the split is 90/10 and the vocabulary is sorted(set(text))', () => {
    const d = makeCharData('hello world, hello there!')
    expect(d.chars).toEqual([...new Set('hello world, hello there!')].sort())
    expect(d.train.length).toBe(Math.floor(0.9 * 25))
    expect(d.train.length + d.val.length).toBe(25)
  })

  it('AdamW with weight decay shrinks a parameter that has zero gradient', () => {
    const m = createGpt(TINY, 1)
    m.params[0] = 1
    m.grads.fill(0)
    adamwStep(m, { lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, clip: 1 })
    expect(m.params[0]).toBeCloseTo(1 - 0.1 * 0.01, 6)
  })

  it('a short training run lowers the loss, deterministically', () => {
    const text = 'abcabdabcabdabcabdabcabdabcabd '.repeat(40)
    const opts = { ...DEFAULT_TRAIN, sizes: { nLayer: 1, nHead: 2, nEmbd: 16, contextLen: 8 }, batchSize: 8, evalEvery: 20, sampleEvery: 1000, sampleLen: 20 }
    const run = () => {
      const s = createSession(text, opts)
      trainFor(s, Infinity, 60, () => 0)
      return s
    }
    const a = run(), b = run()
    const first = a.history[0], last = a.history[a.history.length - 1]
    expect(last.step).toBe(60)
    expect(first.train).toBeGreaterThan(1.2)
    expect(last.train).toBeLessThan(first.train * 0.5)
    expect(last.val).toBeLessThan(first.val * 0.5)
    expect(b.history).toEqual(a.history)
    expect(Array.from(b.model.params)).toEqual(Array.from(a.model.params))
  })

  it('generate crops to the context window and stays in the vocabulary', () => {
    const m = createGpt(TINY, 1)
    const out = generate(m, [0], 20, 1, makeRng(1))
    expect(out.length).toBe(21)
    expect(out.every((i) => i >= 0 && i < TINY.vocabSize)).toBe(true)
  })
})
