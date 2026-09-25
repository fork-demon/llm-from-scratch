import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  buildModel, decode, decodeFloat16, encode, erf, forward, generate, halfToFloat, headScores, headScoresAsync, inductionScore, loadTrainedGpt,
  nextDistribution, pick, previousTokenScore, softmaxRow, type Manifest,
} from './trainedGpt'

interface FixtureCase { prompt: string; logits: number[][]; attn_l1_h2: number[][]; lens_last: number[][] }
const fixture = JSON.parse(readFileSync(resolve(__dirname, '__fixtures__/trainedGpt.fixture.json'), 'utf8')) as { cases: FixtureCase[] }
const dir = resolve(__dirname, '../../public/models')
const manifest = JSON.parse(readFileSync(resolve(dir, 'tiny-gpt.json'), 'utf8')) as Manifest
const bin = readFileSync(resolve(dir, manifest.weights))
const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) as ArrayBuffer
const model = buildModel(manifest, decodeFloat16(buf))

const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
  return m
}

describe('float16 decoding', () => {
  it('decodes known half-precision bit patterns', () => {
    expect(halfToFloat(0x3c00)).toBe(1)
    expect(halfToFloat(0xc000)).toBe(-2)
    expect(halfToFloat(0x3555)).toBeCloseTo(0.33325, 5)
    expect(halfToFloat(0x0001)).toBeCloseTo(5.96e-8, 9) // smallest subnormal
    expect(halfToFloat(0x7c00)).toBe(Infinity)
    expect(halfToFloat(0x0000)).toBe(0)
  })
})

describe('erf', () => {
  it('matches known values', () => {
    expect(erf(0)).toBeCloseTo(0, 7)
    expect(erf(0.5)).toBeCloseTo(0.5204998778, 6)
    expect(erf(1)).toBeCloseTo(0.8427007929, 6)
    expect(erf(-2)).toBeCloseTo(-0.995322265, 6)
  })
})

describe('the checkpoint', () => {
  it('has the config and parameter count it claims', () => {
    const { vocabSize: V, contextLen: C, nEmbd: D, nLayer: L } = model.cfg
    const perBlock = 2 * D + (3 * D * D + 3 * D) + (D * D + D) + 2 * D + (4 * D * D + 4 * D) + (4 * D * D + D)
    expect(V * D + C * D + L * perBlock + 2 * D).toBe(manifest.n_params) // head is tied: no extra matrix
    expect(model.vocab.length).toBe(V)
    expect(manifest.training.val_loss).toBeLessThan(2)
  })
})

describe('forward pass matches PyTorch (fixture from export_checkpoint.py)', () => {
  for (const c of fixture.cases) {
    it(`logits, attention and logit lens for ${JSON.stringify(c.prompt.slice(0, 20))}`, () => {
      const { ids, dropped } = encode(model, c.prompt)
      expect(dropped).toEqual([])
      expect(decode(model, ids)).toBe(c.prompt)
      const run = forward(model, ids)
      expect(run.logits.length).toBe(ids.length)
      run.logits.forEach((row, t) => expect(maxAbsDiff(row, c.logits[t])).toBeLessThan(2e-3))
      run.attention[1][2].forEach((row, t) => expect(maxAbsDiff(row, c.attn_l1_h2[t])).toBeLessThan(1e-4))
      run.lens.forEach((rows, l) => expect(maxAbsDiff(rows[ids.length - 1], c.lens_last[l])).toBeLessThan(2e-3))
    })
  }

  it('attention rows are causal and sum to 1', () => {
    const run = forward(model, encode(model, 'To be, or not').ids)
    for (const layer of run.attention) for (const head of layer) head.forEach((row, t) => {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5)
      for (let s = t + 1; s < row.length; s++) expect(row[s]).toBe(0)
    })
  })

  it('the last logit-lens layer is the model output, and inside:false gives the same logits', () => {
    const ids = encode(model, 'KING RICHARD').ids
    const a = forward(model, ids)
    const b = forward(model, ids, { inside: false })
    expect(a.lens[model.cfg.nLayer - 1]).toBe(a.logits)
    expect(maxAbsDiff(a.logits[3], b.logits[3])).toBe(0)
    expect(b.lens).toEqual([])
  })

  it('rejects empty and over-long inputs', () => {
    expect(() => forward(model, [])).toThrow()
    expect(() => forward(model, Array(model.cfg.contextLen + 1).fill(1))).toThrow(/context window/)
  })
})

describe('the model has learned something', () => {
  it('predicts a likely character after a common prefix', () => {
    const probs = nextDistribution(model, encode(model, 'KING RICHAR').ids)
    expect(probs[model.stoi.get('D')!]).toBeGreaterThan(0.5)
  })
})

describe('sampling', () => {
  it('softmax sums to 1 and temperature sharpens', () => {
    const p = softmaxRow([1, 2, 3])
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    expect(softmaxRow([1, 2, 3], 0.5)[2]).toBeGreaterThan(p[2])
  })
  it('top-k keeps exactly k tokens', () => {
    const probs = nextDistribution(model, encode(model, 'the ').ids, 1, 5)
    expect(probs.filter((p) => p > 0).length).toBe(5)
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  })
  it('pick walks the cumulative distribution and never picks a zero', () => {
    expect(pick([0.2, 0.5, 0.3], 0.1)).toBe(0)
    expect(pick([0.2, 0.5, 0.3], 0.69)).toBe(1)
    expect(pick([0.2, 0.5, 0.3], 0.999)).toBe(2)
    expect(pick([0.5, 0.5, 0], 0.9999999999)).toBe(1)
  })
  it('generate is deterministic per seed and reports top-5', () => {
    const start = encode(model, 'ROMEO:\n').ids
    const a = generate(model, start, 12, { temperature: 0.8, topK: 5, seed: 3 })
    const b = generate(model, start, 12, { temperature: 0.8, topK: 5, seed: 3 })
    expect(a.ids).toEqual(b.ids)
    expect(a.ids.length).toBe(start.length + 12)
    a.steps.forEach((s) => { expect(s.top.length).toBe(5); expect(s.top.some((r) => r.id === s.id)).toBe(true) })
  })
  it('generation crops to the context window', () => {
    const long = encode(model, 'a'.repeat(model.cfg.contextLen + 10)).ids
    expect(generate(model, long, 2).ids.length).toBe(long.length + 2)
  })
})

describe('head scores', () => {
  it('score definitions on hand-made attention patterns', () => {
    const I = (n: number, f: (t: number) => number) => Array.from({ length: n }, (_, t) => { const r = new Float32Array(n); r[f(t)] = 1; return r })
    expect(previousTokenScore(I(5, (t) => Math.max(0, t - 1)))).toBe(1)
    expect(previousTokenScore(I(5, (t) => t))).toBe(0)
    expect(inductionScore(I(8, (t) => (t >= 4 ? t - 3 : 0)), 4)).toBe(1)
  })
  it('agrees in shape and range with the Python scores', () => {
    const s = headScores(model)
    expect(s.previousToken.length).toBe(model.cfg.nLayer)
    for (const row of [...s.previousToken, ...s.induction]) for (const v of row) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1) }
    // the strongest previous-token head should be the one Python found (different probe sequences, same definition)
    const argmax = (m: number[][]) => m.flat().indexOf(Math.max(...m.flat()))
    const py = manifest.python_head_scores!
    expect(argmax(s.previousToken)).toBe(argmax(py.previous_token))
  })
  it('the async version gives the same numbers', async () => {
    const opts = { repeats: 1 }
    expect(await headScoresAsync(model, opts, async () => undefined)).toEqual(headScores(model, opts))
  })
})

describe('loading', () => {
  it('reports HTTP failures clearly', async () => {
    const fail = async () => ({ ok: false, status: 404, json: async () => ({}), arrayBuffer: async () => new ArrayBuffer(0) })
    await expect(loadTrainedGpt('/models/', fail)).rejects.toThrow(/HTTP 404/)
  })
  it('loads through a fetch function', async () => {
    const ok = async (url: string) => ({ ok: true, status: 200, json: async () => manifest, arrayBuffer: async () => (url.endsWith('.bin') ? buf : new ArrayBuffer(0)) })
    const m = await loadTrainedGpt('/models/', ok)
    expect(m.nParams).toBe(manifest.n_params)
  })
  it('rejects a truncated weights file', () => {
    expect(() => buildModel(manifest, new Float32Array(100))).toThrow(/past the end/)
  })
})
