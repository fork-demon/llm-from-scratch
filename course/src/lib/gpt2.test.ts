// Checks the in-browser GPT-2 against PyTorch. The fixture is written by
// `python phase6-engineering/export_gpt2.py --fixtures`: transformers' GPT2LMHeadModel (fp32, eager attention)
// run on the SAME int8-dequantised weights this engine reads, plus the fp32 model's top predictions.
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeAll, describe, expect, it } from 'vitest'
import {
  buildGpt2, distribution, extend, generate, geluNew, linearQ, logitLens, newState, rowOf, sample, softmax, topK,
  type Gpt2Manifest, type Gpt2Model, type QMat,
} from './gpt2'
import fixture from './__fixtures__/gpt2Forward.fixture.json'

const dir = resolve(__dirname, '../../public/models/gpt2')
let m: Gpt2Model

const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  expect(a.length).toBe(b.length)
  let d = 0
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]))
  return d
}

beforeAll(() => {
  const man = JSON.parse(readFileSync(resolve(dir, 'manifest.json'), 'utf8')) as Gpt2Manifest
  const chunks = man.chunks.map((c) => {
    const b = readFileSync(resolve(dir, c.file))
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer
  })
  m = buildGpt2(man, chunks)
})

describe('kernels', () => {
  it('linearQ equals the dequantised matrix product', () => {
    const W: QMat = { rows: 3, cols: 5, q: Int8Array.from([1, -2, 3, 0, 127, -127, 5, 5, 5, 5, 0, 0, 0, 0, 1]), scale: Float32Array.from([0.5, 0.01, 2]) }
    const x = Float32Array.from([1, 2, 3, 4, 5, -1, 0.5, 0, 2, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 2, 1, 0, -1]) // 5 rows
    const b = Float32Array.from([0.1, 0.2, 0.3])
    const out = linearQ(x, 5, W, b)
    const row = new Float32Array(5)
    for (let t = 0; t < 5; t++) for (let o = 0; o < 3; o++) {
      rowOf(W, o, row)
      let s = b[o]
      for (let i = 0; i < 5; i++) s += x[t * 5 + i] * row[i]
      expect(Math.abs(out[t * 3 + o] - s)).toBeLessThan(1e-6 * Math.max(1, Math.abs(s))) // out is stored as float32
    }
  })
  it('gelu_new matches the tanh formula at known points', () => {
    expect(geluNew(0)).toBe(0)
    expect(geluNew(1)).toBeCloseTo(0.841192, 5)
    expect(geluNew(-1)).toBeCloseTo(-0.158808, 5)
  })
  it('sampling: temperature, top-k, top-p', () => {
    const logits = Float32Array.from([2, 1, 0, -1])
    const d = distribution(logits, { temperature: 1, topK: 0, topP: 1 })
    expect(Array.from(d.order)).toEqual([0, 1, 2, 3])
    expect(d.probs[0]).toBeCloseTo(softmax(logits)[0], 6)
    expect(distribution(logits, { temperature: 1, topK: 2, topP: 1 }).kept).toBe(2)
    // p = [0.644, 0.237, 0.087, 0.032]: 0.644 < 0.8 <= 0.881, so top-p 0.8 keeps 2
    const p = distribution(logits, { temperature: 1, topK: 0, topP: 0.8 })
    expect(p.kept).toBe(2)
    expect(p.probs[0] + p.probs[1]).toBeCloseTo(1, 6)
    expect(distribution(logits, { temperature: 0, topK: 0, topP: 1 }).probs[0]).toBe(1)
    expect(sample(p, 0.99).id).toBe(1)
    expect(topK([3, 9, 1, 9, 4], 3)).toEqual([{ id: 1, value: 9 }, { id: 3, value: 9 }, { id: 4, value: 4 }])
  })
})

describe('GPT-2 small vs PyTorch on the same int8 weights', () => {
  it('has 124M parameters and the tied embedding keeps its outlier columns', () => {
    expect(m.nParams).toBe(124439808)
    expect(m.wte.outCols?.length).toBe(8)
  })

  for (const c of fixture.cases) {
    it(`"${c.prompt}": logits, intermediates and logit lens match`, () => {
      const s = newState(m)
      const t0 = performance.now()
      const logits = extend(s, c.ids)
      const ms = performance.now() - t0
      console.log(`forward ${c.ids.length} tokens: ${ms.toFixed(0)} ms`)
      const D = 768
      const T = c.ids.length
      const last = T - 1

      // logits at the last position: absolute tolerance on values that are around -80 to -120
      const sel = c.logits.index.map((i) => logits[i])
      expect(maxDiff(sel, c.logits.value)).toBeLessThan(2e-3)
      let mean = 0
      for (let i = 0; i < logits.length; i++) mean += logits[i] / logits.length
      expect(Math.abs(mean - c.logits.mean)).toBeLessThan(1e-3)
      expect(topK(logits, 1)[0].id).toBe(c.logits.argmax)
      // and the int8 model agrees with fp32 GPT-2 on the top prediction
      expect(c.logits.argmax).toBe(c.fp32_top1)

      const L = s.layers
      const cap = s.cap
      expect(maxDiff(L[0].ln1.subarray(last * D, last * D + 16), c.ln1_l0_last16)).toBeLessThan(1e-4)
      expect(maxDiff(L[3].q.subarray(last * D + 320, last * D + 384), c.q_l3_h5_last)).toBeLessThan(2e-4)
      expect(maxDiff(L[3].k.subarray(320, 384), c.k_l3_h5_first)).toBeLessThan(2e-4)
      expect(maxDiff(L[3].v.subarray(last * D + 320, last * D + 384), c.v_l3_h5_last)).toBeLessThan(2e-4)
      const attnRow = (l: number, h: number, t: number) => L[l].attn.subarray((h * cap + t) * cap, (h * cap + t) * cap + T)
      expect(maxDiff(attnRow(0, 0, last), c.attn_l0_h0_last)).toBeLessThan(1e-5)
      expect(maxDiff(attnRow(5, 1, last), c.attn_l5_h1_last)).toBeLessThan(1e-5)
      const all11: number[] = []
      for (let t = 0; t < T; t++) all11.push(...attnRow(11, 7, t))
      expect(maxDiff(all11, c.attn_l11_h7_all)).toBeLessThan(1e-5)
      expect(maxDiff(L[7].attnOut.subarray(last * D, last * D + 16), c.attnout_l7_last16)).toBeLessThan(5e-4)
      expect(maxDiff(L[2].act.subarray(last * 4 * D, last * 4 * D + 16), c.act_l2_last16)).toBeLessThan(2e-4)
      c.block_last16.forEach((ref, l) => {
        // the residual stream grows to values in the hundreds in a few dimensions: relative tolerance
        const got = L[l].resid2.subarray(last * D, last * D + 16)
        ref.forEach((r, i) => expect(Math.abs(got[i] - r)).toBeLessThan(1e-3 + 2e-5 * Math.abs(r)))
      })
      expect(maxDiff(s.lnF.subarray(last * D, last * D + 16), c.lnf_last16)).toBeLessThan(2e-4)

      const lens = logitLens(s)
      lens.forEach((row, l) => {
        expect(row.top.map((r) => r.id)).toEqual(c.lens_top5[l].ids)
        expect(maxDiff(row.top.map((r) => r.p), c.lens_top5[l].p)).toBeLessThan(1e-4)
      })
      expect(lens[11].rankFinal).toBe(1)
    })
  }

  it('the KV cache gives the same logits as a full re-run, and greedy generation matches PyTorch', () => {
    const c = fixture.cases[0]
    const s = newState(m)
    extend(s, c.ids.slice(0, 3))
    const inc = extend(s, c.ids.slice(3))
    const full = extend(newState(m), c.ids)
    expect(maxDiff(inc, full)).toBeLessThan(1e-3)
    // incremental trace: attention rows of the early positions are unchanged, the later ones filled in
    expect(s.ids).toEqual(c.ids)

    for (const cc of fixture.cases) {
      const t0 = performance.now()
      const g = generate(m, cc.ids, 8, { temperature: 0, topK: 0, topP: 1 }, () => 0.5)
      console.log(`generate 8 tokens after ${cc.ids.length}: ${(performance.now() - t0).toFixed(0)} ms`)
      expect(g).toEqual(cc.greedy8)
    }
  })
})
