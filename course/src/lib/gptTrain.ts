// A real, trainable character-level GPT with hand-written forward AND backward passes.
// Same architecture as class GPT in phase3-transformers/tiny_gpt.py:
//   tok_emb + pos_emb -> n_layer x [ x + attn(ln1(x)) ; x + ffn(ln2(x)) ] -> ln_f -> head (tied to tok_emb)
//   attention: fused qkv Linear, causal multi-head softmax(q k^T / sqrt(hd)) v, output proj Linear
//   ffn: Linear(D, 4D) -> GELU -> Linear(4D, D)
// trained with cross-entropy, AdamW (PyTorch defaults: betas 0.9/0.999, eps 1e-8, weight decay 0.01 on every
// parameter) and global gradient-norm clipping.
// Differences from the Python, on purpose: no dropout; weights start at N(0, 0.02) like GPT-2 (so the loss at
// init is ~ln(vocab)), with the residual projections scaled by 1/sqrt(2 n_layer); a larger learning rate
// because the model is ~100x smaller.
// Everything is flat typed arrays and plain loops (the kernel layout follows Karpathy's llm.c). No DOM.
import { makeRng, type Rng } from './rng'

export type Arr = Float32Array | Float64Array

/* ------------------------------------------------------------------ */
/* Config and parameters                                                */
/* ------------------------------------------------------------------ */

export interface GptSizes { nLayer: number; nHead: number; nEmbd: number; contextLen: number }
export interface GptConfig extends GptSizes { vocabSize: number }

/** The default: loss falls visibly within ~30 s on a laptop (see the report in gptTrain.test.ts). */
export const DEFAULT_SIZES: GptSizes = { nLayer: 2, nHead: 4, nEmbd: 32, contextLen: 32 }

export interface ParamGroup { name: string; offset: number; size: number; shape: number[] }

const LAYER_KEYS = ['ln1.g', 'ln1.b', 'attn.qkv.w', 'attn.qkv.b', 'attn.proj.w', 'attn.proj.b', 'ln2.g', 'ln2.b', 'ffn.fc.w', 'ffn.fc.b', 'ffn.proj.w', 'ffn.proj.b'] as const

/** Every parameter tensor, in order, with its offset in the one flat parameter array. Linear weights are (out, in). */
export const paramGroups = (cfg: GptConfig): ParamGroup[] => {
  const { vocabSize: V, contextLen: C, nEmbd: D } = cfg
  const F = 4 * D
  const shapes: [string, number[]][] = [['tok_emb', [V, D]], ['pos_emb', [C, D]]]
  for (let l = 0; l < cfg.nLayer; l++) {
    const s: Record<(typeof LAYER_KEYS)[number], number[]> = {
      'ln1.g': [D], 'ln1.b': [D], 'attn.qkv.w': [3 * D, D], 'attn.qkv.b': [3 * D], 'attn.proj.w': [D, D], 'attn.proj.b': [D],
      'ln2.g': [D], 'ln2.b': [D], 'ffn.fc.w': [F, D], 'ffn.fc.b': [F], 'ffn.proj.w': [D, F], 'ffn.proj.b': [D],
    }
    for (const k of LAYER_KEYS) shapes.push([`blocks.${l}.${k}`, s[k]])
  }
  shapes.push(['ln_f.g', [D]], ['ln_f.b', [D]])
  let offset = 0
  return shapes.map(([name, shape]) => {
    const size = shape.reduce((a, b) => a * b, 1)
    const g = { name, offset, size, shape }
    offset += size
    return g
  })
}

/** Total number of trainable numbers (the head is tied to tok_emb, so it adds nothing). */
export const countParams = (cfg: GptConfig): number => paramGroups(cfg).reduce((s, g) => s + g.size, 0)

interface LayerP { ln1g: Arr; ln1b: Arr; qkvw: Arr; qkvb: Arr; projw: Arr; projb: Arr; ln2g: Arr; ln2b: Arr; fcw: Arr; fcb: Arr; fc2w: Arr; fc2b: Arr }
interface Views { wte: Arr; wpe: Arr; layers: LayerP[]; lnfg: Arr; lnfb: Arr }

const makeViews = (buf: Arr, groups: ParamGroup[], nLayer: number): Views => {
  const byName = new Map(groups.map((g) => [g.name, buf.subarray(g.offset, g.offset + g.size) as Arr]))
  const get = (n: string) => byName.get(n)!
  return {
    wte: get('tok_emb'),
    wpe: get('pos_emb'),
    layers: Array.from({ length: nLayer }, (_, l) => {
      const k = (s: string) => get(`blocks.${l}.${s}`)
      return {
        ln1g: k('ln1.g'), ln1b: k('ln1.b'), qkvw: k('attn.qkv.w'), qkvb: k('attn.qkv.b'), projw: k('attn.proj.w'), projb: k('attn.proj.b'),
        ln2g: k('ln2.g'), ln2b: k('ln2.b'), fcw: k('ffn.fc.w'), fcb: k('ffn.fc.b'), fc2w: k('ffn.proj.w'), fc2b: k('ffn.proj.b'),
      }
    }),
    lnfg: get('ln_f.g'),
    lnfb: get('ln_f.b'),
  }
}

interface Acts {
  B: number; T: number
  enc: Arr
  layers: { inp: Arr; ln1: Arr; ln1m: Arr; ln1r: Arr; qkv: Arr; att: Arr; aty: Arr; res2: Arr; ln2: Arr; ln2m: Arr; ln2r: Arr; fch: Arr; fcg: Arr; res3: Arr }[]
  lnf: Arr; lnfm: Arr; lnfr: Arr
  logits: Arr // becomes probabilities after the loss is computed
  // scratch for the backward pass
  dres: Arr; dln: Arr; dqkv: Arr; daty: Arr; dfch: Arr; dfcg: Arr; datt: Float64Array
}

export interface Gpt {
  cfg: GptConfig
  groups: ParamGroup[]
  f64: boolean
  params: Arr
  grads: Arr
  p: Views
  g: Views
  adamM: Arr
  adamV: Arr
  adamT: number
  ws: Map<string, Acts>
}

const alloc = (f64: boolean, n: number): Arr => (f64 ? new Float64Array(n) : new Float32Array(n))

/**
 * A fresh model with seeded random weights. `f64` switches every buffer to Float64Array (used by the
 * gradient check, where Float32 rounding would swamp the finite differences).
 */
export const createGpt = (cfg: GptConfig, seed = 1337, f64 = false): Gpt => {
  if (cfg.nEmbd % cfg.nHead !== 0) throw new Error(`nEmbd ${cfg.nEmbd} must divide by nHead ${cfg.nHead}`)
  const groups = paramGroups(cfg)
  const n = countParams(cfg)
  const params = alloc(f64, n)
  const grads = alloc(f64, n)
  const rng = makeRng(seed)
  const residStd = 0.02 / Math.sqrt(2 * cfg.nLayer)
  for (const g of groups) {
    const view = params.subarray(g.offset, g.offset + g.size)
    if (g.name.endsWith('.g')) view.fill(1) // LayerNorm gain
    else if (g.name.endsWith('.b')) view.fill(0) // biases, LayerNorm shift
    else {
      const std = g.name.endsWith('proj.w') ? residStd : 0.02
      for (let i = 0; i < view.length; i++) view[i] = std * rng.normal()
    }
  }
  return {
    cfg, groups, f64, params, grads,
    p: makeViews(params, groups, cfg.nLayer),
    g: makeViews(grads, groups, cfg.nLayer),
    adamM: alloc(f64, n), adamV: alloc(f64, n), adamT: 0,
    ws: new Map(),
  }
}

const workspace = (m: Gpt, B: number, T: number): Acts => {
  const key = `${B}x${T}`
  const hit = m.ws.get(key)
  if (hit) return hit
  const { nEmbd: D, nHead: H, nLayer: L, vocabSize: V } = m.cfg
  const BT = B * T
  const a = (n: number) => alloc(m.f64, n)
  const enc = a(BT * D)
  const layers: Acts['layers'] = []
  let inp = enc
  for (let l = 0; l < L; l++) {
    const res3 = a(BT * D)
    layers.push({
      inp, ln1: a(BT * D), ln1m: a(BT), ln1r: a(BT), qkv: a(BT * 3 * D), att: a(B * H * T * T), aty: a(BT * D), res2: a(BT * D),
      ln2: a(BT * D), ln2m: a(BT), ln2r: a(BT), fch: a(BT * 4 * D), fcg: a(BT * 4 * D), res3,
    })
    inp = res3
  }
  const acts: Acts = {
    B, T, enc, layers, lnf: a(BT * D), lnfm: a(BT), lnfr: a(BT), logits: a(BT * V),
    dres: a(BT * D), dln: a(BT * D), dqkv: a(BT * 3 * D), daty: a(BT * D), dfch: a(BT * 4 * D), dfcg: a(BT * 4 * D), datt: new Float64Array(T),
  }
  if (m.ws.size > 8) m.ws.clear() // inference with many prompt lengths: do not grow forever
  m.ws.set(key, acts)
  return acts
}

/* ------------------------------------------------------------------ */
/* Kernels (forward and backward). N = rows, C = in features, OC = out */
/* ------------------------------------------------------------------ */

const LN_EPS = 1e-5

function layernormForward(out: Arr, mean: Arr, rstd: Arr, inp: Arr, w: Arr, b: Arr, N: number, C: number) {
  for (let n = 0; n < N; n++) {
    const o = n * C
    let m = 0
    for (let i = 0; i < C; i++) m += inp[o + i]
    m /= C
    let v = 0
    for (let i = 0; i < C; i++) { const d = inp[o + i] - m; v += d * d }
    v /= C
    const s = 1 / Math.sqrt(v + LN_EPS)
    for (let i = 0; i < C; i++) out[o + i] = (inp[o + i] - m) * s * w[i] + b[i]
    mean[n] = m
    rstd[n] = s
  }
}

function layernormBackward(dinp: Arr, dw: Arr, db: Arr, dout: Arr, inp: Arr, w: Arr, mean: Arr, rstd: Arr, N: number, C: number) {
  for (let n = 0; n < N; n++) {
    const o = n * C
    const m = mean[n], s = rstd[n]
    let dnormMean = 0, dnormNormMean = 0
    for (let i = 0; i < C; i++) {
      const norm = (inp[o + i] - m) * s
      const dnorm = w[i] * dout[o + i]
      dnormMean += dnorm
      dnormNormMean += dnorm * norm
    }
    dnormMean /= C
    dnormNormMean /= C
    for (let i = 0; i < C; i++) {
      const norm = (inp[o + i] - m) * s
      const d = dout[o + i]
      db[i] += d
      dw[i] += norm * d
      dinp[o + i] += (w[i] * d - dnormMean - norm * dnormNormMean) * s
    }
  }
}

/** out (N, OC) = inp (N, C) @ w^T + bias, with w stored (OC, C) like torch.nn.Linear. Four outputs at a time. */
function matmulForward(out: Arr, inp: Arr, w: Arr, bias: Arr | null, N: number, C: number, OC: number) {
  const OC4 = OC - (OC % 4)
  for (let n = 0; n < N; n++) {
    const io = n * C, oo = n * OC
    let o = 0
    for (; o < OC4; o += 4) {
      const w0 = o * C, w1 = w0 + C, w2 = w1 + C, w3 = w2 + C
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0
      for (let i = 0; i < C; i++) {
        const x = inp[io + i]
        s0 += x * w[w0 + i]; s1 += x * w[w1 + i]; s2 += x * w[w2 + i]; s3 += x * w[w3 + i]
      }
      if (bias) { s0 += bias[o]; s1 += bias[o + 1]; s2 += bias[o + 2]; s3 += bias[o + 3] }
      out[oo + o] = s0; out[oo + o + 1] = s1; out[oo + o + 2] = s2; out[oo + o + 3] = s3
    }
    for (; o < OC; o++) {
      const wo = o * C
      let s = bias ? bias[o] : 0
      for (let i = 0; i < C; i++) s += inp[io + i] * w[wo + i]
      out[oo + o] = s
    }
  }
}

/** Accumulates (+=) into dinp, dw, dbias. */
function matmulBackward(dinp: Arr, dw: Arr, dbias: Arr | null, dout: Arr, inp: Arr, w: Arr, N: number, C: number, OC: number) {
  const OC4 = OC - (OC % 4), N4 = N - (N % 4)
  // dinp = dout @ w, four weight rows at a time
  for (let n = 0; n < N; n++) {
    const io = n * C, oo = n * OC
    let o = 0
    for (; o < OC4; o += 4) {
      const d0 = dout[oo + o], d1 = dout[oo + o + 1], d2 = dout[oo + o + 2], d3 = dout[oo + o + 3]
      const w0 = o * C, w1 = w0 + C, w2 = w1 + C, w3 = w2 + C
      for (let i = 0; i < C; i++) dinp[io + i] += d0 * w[w0 + i] + d1 * w[w1 + i] + d2 * w[w2 + i] + d3 * w[w3 + i]
    }
    for (; o < OC; o++) {
      const d = dout[oo + o], wo = o * C
      for (let i = 0; i < C; i++) dinp[io + i] += d * w[wo + i]
    }
  }
  // dw = dout^T @ inp, four rows of the batch at a time
  for (let o = 0; o < OC; o++) {
    const wo = o * C
    let db = 0
    let n = 0
    for (; n < N4; n += 4) {
      const d0 = dout[n * OC + o], d1 = dout[(n + 1) * OC + o], d2 = dout[(n + 2) * OC + o], d3 = dout[(n + 3) * OC + o]
      db += d0 + d1 + d2 + d3
      const i0 = n * C, i1 = i0 + C, i2 = i1 + C, i3 = i2 + C
      for (let i = 0; i < C; i++) dw[wo + i] += d0 * inp[i0 + i] + d1 * inp[i1 + i] + d2 * inp[i2 + i] + d3 * inp[i3 + i]
    }
    for (; n < N; n++) {
      const d = dout[n * OC + o], io = n * C
      db += d
      for (let i = 0; i < C; i++) dw[wo + i] += d * inp[io + i]
    }
    if (dbias) dbias[o] += db
  }
}

/** Causal multi-head attention. qkv (B, T, 3D) holds [q | k | v]; att (B, H, T, T) keeps the probabilities. */
function attentionForward(out: Arr, att: Arr, qkv: Arr, B: number, T: number, D: number, H: number) {
  const hs = D / H
  const scale = 1 / Math.sqrt(hs)
  const D3 = 3 * D
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < H; h++) {
        const q = (b * T + t) * D3 + h * hs
        const a = ((b * H + h) * T + t) * T
        let max = -Infinity
        for (let t2 = 0; t2 <= t; t2++) {
          const k = (b * T + t2) * D3 + D + h * hs
          let s = 0
          for (let i = 0; i < hs; i++) s += qkv[q + i] * qkv[k + i]
          s *= scale
          att[a + t2] = s
          if (s > max) max = s
        }
        let sum = 0
        for (let t2 = 0; t2 <= t; t2++) { const e = Math.exp(att[a + t2] - max); att[a + t2] = e; sum += e }
        const inv = 1 / sum
        for (let t2 = 0; t2 <= t; t2++) att[a + t2] *= inv
        for (let t2 = t + 1; t2 < T; t2++) att[a + t2] = 0 // the mask: the future gets exactly zero
        const o = (b * T + t) * D + h * hs
        for (let i = 0; i < hs; i++) out[o + i] = 0
        for (let t2 = 0; t2 <= t; t2++) {
          const v = (b * T + t2) * D3 + 2 * D + h * hs
          const p = att[a + t2]
          for (let i = 0; i < hs; i++) out[o + i] += p * qkv[v + i]
        }
      }
    }
  }
}

/** Accumulates into dqkv. datt is scratch of length T. */
function attentionBackward(dqkv: Arr, dout: Arr, qkv: Arr, att: Arr, datt: Float64Array, B: number, T: number, D: number, H: number) {
  const hs = D / H
  const scale = 1 / Math.sqrt(hs)
  const D3 = 3 * D
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < H; h++) {
        const a = ((b * H + h) * T + t) * T
        const o = (b * T + t) * D + h * hs
        const q = (b * T + t) * D3 + h * hs
        // out = sum_t2 att[t2] * v[t2]
        let dot = 0
        for (let t2 = 0; t2 <= t; t2++) {
          const v = (b * T + t2) * D3 + 2 * D + h * hs
          const p = att[a + t2]
          let s = 0
          for (let i = 0; i < hs; i++) { s += dout[o + i] * qkv[v + i]; dqkv[v + i] += p * dout[o + i] }
          datt[t2] = s
          dot += p * s
        }
        // softmax backward, then scores = scale * q . k
        for (let t2 = 0; t2 <= t; t2++) {
          const dpre = att[a + t2] * (datt[t2] - dot) * scale
          if (dpre === 0) continue
          const k = (b * T + t2) * D3 + D + h * hs
          for (let i = 0; i < hs; i++) { dqkv[q + i] += dpre * qkv[k + i]; dqkv[k + i] += dpre * qkv[q + i] }
        }
      }
    }
  }
}

// GELU as in torch.nn.GELU(): x * Phi(x), with erf from Abramowitz & Stegun 7.1.26 (error < 1.5e-7).
// The backward pass differentiates this exact approximation, so gradients match the forward to rounding.
const P = 0.3275911, A1 = 0.254829592, A2 = -0.284496736, A3 = 1.421413741, A4 = -1.453152027, A5 = 1.061405429
const erf = (x: number): number => {
  const ax = Math.abs(x)
  const t = 1 / (1 + P * ax)
  const y = 1 - ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t * Math.exp(-ax * ax)
  return x < 0 ? -y : y
}
/** d erf / dx of the approximation above (an even function). */
const erfPrime = (x: number): number => {
  const ax = Math.abs(x)
  const t = 1 / (1 + P * ax)
  const poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t
  const dpoly = (((5 * A5 * t + 4 * A4) * t + 3 * A3) * t + 2 * A2) * t + A1 // d poly / dt
  const e = Math.exp(-ax * ax)
  return e * (dpoly * P * t * t + 2 * ax * poly)
}
const INV_SQRT2 = 1 / Math.SQRT2
export const gelu = (x: number): number => 0.5 * x * (1 + erf(x * INV_SQRT2))

function geluForward(out: Arr, inp: Arr, N: number) {
  for (let i = 0; i < N; i++) out[i] = gelu(inp[i])
}
function geluBackward(dinp: Arr, inp: Arr, dout: Arr, N: number) {
  for (let i = 0; i < N; i++) {
    const x = inp[i], z = x * INV_SQRT2
    dinp[i] = dout[i] * (0.5 * (1 + erf(z)) + 0.5 * x * INV_SQRT2 * erfPrime(z))
  }
}

/* ------------------------------------------------------------------ */
/* Forward, loss, backward                                              */
/* ------------------------------------------------------------------ */

/**
 * GPT.forward on B sequences of T tokens (ids is (B, T) flattened). Fills the workspace; if targets are given,
 * turns logits into probabilities and returns the mean cross-entropy, else returns NaN.
 */
export function forward(m: Gpt, ids: ArrayLike<number>, B: number, T: number, targets?: ArrayLike<number>): number {
  const { nEmbd: D, nHead: H, vocabSize: V, contextLen } = m.cfg
  if (T > contextLen) throw new Error(`forward: ${T} tokens do not fit in a context window of ${contextLen}`)
  const w = workspace(m, B, T)
  const BT = B * T, F = 4 * D
  const { wte, wpe } = m.p
  // x = tok_emb(idx) + pos_emb(pos)
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const n = b * T + t, tok = ids[n]
      if (tok < 0 || tok >= V) throw new Error(`forward: token id ${tok} is outside the vocabulary`)
      for (let i = 0; i < D; i++) w.enc[n * D + i] = wte[tok * D + i] + wpe[t * D + i]
    }
  }
  for (let l = 0; l < m.cfg.nLayer; l++) {
    const p = m.p.layers[l], a = w.layers[l]
    layernormForward(a.ln1, a.ln1m, a.ln1r, a.inp, p.ln1g, p.ln1b, BT, D)
    matmulForward(a.qkv, a.ln1, p.qkvw, p.qkvb, BT, D, 3 * D)
    attentionForward(a.aty, a.att, a.qkv, B, T, D, H)
    matmulForward(a.res2, a.aty, p.projw, p.projb, BT, D, D)
    for (let i = 0; i < BT * D; i++) a.res2[i] += a.inp[i] // x = x + attn(ln1(x))
    layernormForward(a.ln2, a.ln2m, a.ln2r, a.res2, p.ln2g, p.ln2b, BT, D)
    matmulForward(a.fch, a.ln2, p.fcw, p.fcb, BT, D, F)
    geluForward(a.fcg, a.fch, BT * F)
    matmulForward(a.res3, a.fcg, p.fc2w, p.fc2b, BT, F, D)
    for (let i = 0; i < BT * D; i++) a.res3[i] += a.res2[i] // x = x + ffn(ln2(x))
  }
  const last = m.cfg.nLayer ? w.layers[m.cfg.nLayer - 1].res3 : w.enc
  layernormForward(w.lnf, w.lnfm, w.lnfr, last, m.p.lnfg, m.p.lnfb, BT, D)
  matmulForward(w.logits, w.lnf, wte, null, BT, D, V) // head, tied: logits = ln_f(x) @ tok_emb^T
  if (!targets) return NaN
  let loss = 0
  for (let n = 0; n < BT; n++) {
    const o = n * V
    let max = -Infinity
    for (let v = 0; v < V; v++) if (w.logits[o + v] > max) max = w.logits[o + v]
    let sum = 0
    for (let v = 0; v < V; v++) { const e = Math.exp(w.logits[o + v] - max); w.logits[o + v] = e; sum += e }
    for (let v = 0; v < V; v++) w.logits[o + v] /= sum
    loss -= Math.log(Math.max(w.logits[o + targets[n]], 1e-30))
  }
  return loss / BT
}

/** Backward pass for the last forward(..., targets). Overwrites m.grads. */
export function backward(m: Gpt, ids: ArrayLike<number>, B: number, T: number, targets: ArrayLike<number>) {
  const { nEmbd: D, nHead: H, vocabSize: V } = m.cfg
  const w = workspace(m, B, T)
  const BT = B * T, F = 4 * D
  m.grads.fill(0)
  // dlogits = (probs - onehot) / BT, written in place over the probabilities
  const dlogits = w.logits
  for (let n = 0; n < BT; n++) dlogits[n * V + targets[n]] -= 1
  for (let i = 0; i < BT * V; i++) dlogits[i] /= BT
  w.dln.fill(0)
  matmulBackward(w.dln, m.g.wte, null, dlogits, w.lnf, m.p.wte, BT, D, V)
  w.dres.fill(0)
  const last = m.cfg.nLayer ? w.layers[m.cfg.nLayer - 1].res3 : w.enc
  layernormBackward(w.dres, m.g.lnfg, m.g.lnfb, w.dln, last, m.p.lnfg, w.lnfm, w.lnfr, BT, D)
  for (let l = m.cfg.nLayer - 1; l >= 0; l--) {
    const p = m.p.layers[l], g = m.g.layers[l], a = w.layers[l]
    // res3 = res2 + fc2(gelu(fc(ln2(res2)))): dres is d/d res3 and flows straight through the '+'
    w.dfcg.fill(0)
    matmulBackward(w.dfcg, g.fc2w, g.fc2b, w.dres, a.fcg, p.fc2w, BT, F, D)
    geluBackward(w.dfch, a.fch, w.dfcg, BT * F)
    w.dln.fill(0)
    matmulBackward(w.dln, g.fcw, g.fcb, w.dfch, a.ln2, p.fcw, BT, D, F)
    layernormBackward(w.dres, g.ln2g, g.ln2b, w.dln, a.res2, p.ln2g, a.ln2m, a.ln2r, BT, D) // now d/d res2
    // res2 = inp + proj(attn(ln1(inp)))
    w.daty.fill(0)
    matmulBackward(w.daty, g.projw, g.projb, w.dres, a.aty, p.projw, BT, D, D)
    w.dqkv.fill(0)
    attentionBackward(w.dqkv, w.daty, a.qkv, a.att, w.datt, B, T, D, H)
    w.dln.fill(0)
    matmulBackward(w.dln, g.qkvw, g.qkvb, w.dqkv, a.ln1, p.qkvw, BT, D, 3 * D)
    layernormBackward(w.dres, g.ln1g, g.ln1b, w.dln, a.inp, p.ln1g, a.ln1m, a.ln1r, BT, D) // now d/d inp
  }
  // embeddings: the token table also received the head's gradient above (weight tying)
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const n = b * T + t, tok = ids[n]
      for (let i = 0; i < D; i++) { const d = w.dres[n * D + i]; m.g.wte[tok * D + i] += d; m.g.wpe[t * D + i] += d }
    }
  }
}

export interface AdamW { lr: number; beta1: number; beta2: number; eps: number; weightDecay: number; clip: number }
export const ADAMW_DEFAULTS: AdamW = { lr: 5e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, clip: 1 }

/** clip_grad_norm_(clip) then one torch.optim.AdamW step. Returns the gradient norm before clipping. */
export function adamwStep(m: Gpt, o: AdamW = ADAMW_DEFAULTS): number {
  const g = m.grads, p = m.params, M = m.adamM, Vv = m.adamV
  let ss = 0
  for (let i = 0; i < g.length; i++) ss += g[i] * g[i]
  const norm = Math.sqrt(ss)
  const k = o.clip > 0 && norm > o.clip ? o.clip / (norm + 1e-6) : 1
  m.adamT++
  const bc1 = 1 - Math.pow(o.beta1, m.adamT), bc2 = 1 - Math.pow(o.beta2, m.adamT)
  for (let i = 0; i < p.length; i++) {
    const gi = g[i] * k
    p[i] *= 1 - o.lr * o.weightDecay // decoupled weight decay
    M[i] = o.beta1 * M[i] + (1 - o.beta1) * gi
    Vv[i] = o.beta2 * Vv[i] + (1 - o.beta2) * gi * gi
    p[i] -= (o.lr * (M[i] / bc1)) / (Math.sqrt(Vv[i] / bc2) + o.eps)
  }
  return norm
}

/* ------------------------------------------------------------------ */
/* Data                                                                 */
/* ------------------------------------------------------------------ */

export interface CharData { chars: string[]; stoi: Map<string, number>; train: Int32Array; val: Int32Array }

/** chars = sorted(set(text)); 90/10 split exactly like tiny_gpt.py (n = int(0.9 * len(data))). */
export const makeCharData = (text: string, trainFraction = 0.9): CharData => {
  const chars = [...new Set(text)].sort()
  const stoi = new Map(chars.map((c, i) => [c, i]))
  const ids = Int32Array.from([...text], (c) => stoi.get(c)!)
  const n = Math.floor(trainFraction * ids.length)
  return { chars, stoi, train: ids.slice(0, n), val: ids.slice(n) }
}
export const encodeChars = (d: CharData, s: string): number[] => [...s].flatMap((c) => (d.stoi.has(c) ? [d.stoi.get(c)!] : []))
export const decodeChars = (d: CharData, ids: ArrayLike<number>): string => Array.from(ids, (i) => d.chars[i]).join('')

/** The smallest text the trainer accepts for a given context (both splits need at least one full window). */
export const minTextLength = (contextLen: number): number => 10 * (contextLen + 2)

/** get_batch: B random windows; y is x shifted by one. Start positions are 0 .. len - T - 2, as in the Python. */
export const getBatch = (data: Int32Array, B: number, T: number, rng: Rng, x: Int32Array = new Int32Array(B * T), y: Int32Array = new Int32Array(B * T)) => {
  const hi = data.length - T - 1
  if (hi < 1) throw new Error('getBatch: text shorter than one context window')
  for (let b = 0; b < B; b++) {
    const s = rng.int(hi)
    for (let t = 0; t < T; t++) { x[b * T + t] = data[s + t]; y[b * T + t] = data[s + t + 1] }
  }
  return { x, y }
}

/** Mean loss over fixed batches (no gradient). */
export const estimateLoss = (m: Gpt, batches: { x: Int32Array; y: Int32Array }[], B: number, T: number): number =>
  batches.reduce((s, b) => s + forward(m, b.x, B, T, b.y), 0) / batches.length

/* ------------------------------------------------------------------ */
/* Inference: attention maps and generation                              */
/* ------------------------------------------------------------------ */

export interface Inspect {
  /** attn[layer][head][t][t2]: how much position t attends to position t2 (rows sum to 1, zero above the diagonal) */
  attn: number[][][][]
  /** next-token probabilities at the last position */
  probs: number[]
}

/** Forward one sequence (cropped to the context window) and read out the attention weights of every head. */
export const inspect = (m: Gpt, ids: number[], temperature = 1): Inspect => {
  const seq = ids.slice(-m.cfg.contextLen)
  if (seq.length === 0) return { attn: [], probs: [] }
  const T = seq.length
  forward(m, seq, 1, T)
  const w = workspace(m, 1, T)
  const { nHead: H, nLayer: L, vocabSize: V } = m.cfg
  const attn = Array.from({ length: L }, (_, l) =>
    Array.from({ length: H }, (_, h) => Array.from({ length: T }, (_, t) => Array.from(w.layers[l].att.subarray((h * T + t) * T, (h * T + t + 1) * T)))),
  )
  return { attn, probs: softmaxRow(w.logits, (T - 1) * V, V, temperature) }
}

const softmaxRow = (a: Arr, o: number, V: number, temperature: number): number[] => {
  const tt = Math.max(temperature, 1e-4)
  let max = -Infinity
  for (let v = 0; v < V; v++) max = Math.max(max, a[o + v] / tt)
  const e = Array.from({ length: V }, (_, v) => Math.exp(a[o + v] / tt - max))
  const s = e.reduce((x, y) => x + y, 0)
  return e.map((x) => x / s)
}

/** GPT.generate: crop to the context window, forward, last position / temperature, softmax, sample, append. */
export const generate = (m: Gpt, prompt: number[], maxNew: number, temperature: number, rng: Rng): number[] => {
  const out = prompt.slice()
  const V = m.cfg.vocabSize
  for (let i = 0; i < maxNew; i++) {
    const seq = out.slice(-m.cfg.contextLen)
    const T = seq.length
    forward(m, seq, 1, T)
    const probs = softmaxRow(workspace(m, 1, T).logits, (T - 1) * V, V, temperature)
    let u = rng.next(), k = 0
    for (; k < V - 1; k++) { u -= probs[k]; if (u < 0) break }
    out.push(k)
  }
  return out
}

/* ------------------------------------------------------------------ */
/* A training session: what the UI (or its Web Worker) drives in chunks  */
/* ------------------------------------------------------------------ */

export interface TrainOptions {
  sizes: GptSizes
  batchSize: number
  optim: AdamW
  seed: number
  evalEvery: number // steps between loss estimates
  evalBatches: number // fixed batches per split for the estimate
  sampleEvery: number // steps between printed samples
  sampleLen: number
  temperature: number
}
export const DEFAULT_TRAIN: TrainOptions = {
  sizes: DEFAULT_SIZES, batchSize: 16, optim: ADAMW_DEFAULTS, seed: 1337,
  evalEvery: 25, evalBatches: 4, sampleEvery: 100, sampleLen: 120, temperature: 0.8,
}

export interface LossPoint { step: number; train: number; val: number; seconds: number }
export interface Sample { step: number; text: string }

export interface Session {
  opts: TrainOptions
  data: CharData
  model: Gpt
  rng: Rng // batch sampling
  sampleRng: Rng
  step: number
  seconds: number // time spent training (excluding pauses)
  lastLoss: number // loss of the most recent training batch
  history: LossPoint[]
  samples: Sample[]
  evalTrain: { x: Int32Array; y: Int32Array }[]
  evalVal: { x: Int32Array; y: Int32Array }[]
  x: Int32Array; y: Int32Array
}

export const createSession = (text: string, opts: TrainOptions = DEFAULT_TRAIN): Session => {
  const T = opts.sizes.contextLen
  if (text.length < minTextLength(T)) throw new Error(`Need at least ${minTextLength(T)} characters of text, got ${text.length}.`)
  const data = makeCharData(text)
  const model = createGpt({ ...opts.sizes, vocabSize: data.chars.length }, opts.seed)
  const evalRng = makeRng(opts.seed + 1)
  const B = opts.batchSize
  const fixed = (d: Int32Array) => Array.from({ length: opts.evalBatches }, () => getBatch(d, B, T, evalRng))
  const s: Session = {
    opts, data, model, rng: makeRng(opts.seed + 2), sampleRng: makeRng(opts.seed + 3),
    step: 0, seconds: 0, lastLoss: NaN, history: [], samples: [],
    evalTrain: fixed(data.train), evalVal: fixed(data.val),
    x: new Int32Array(B * T), y: new Int32Array(B * T),
  }
  recordEval(s)
  recordSample(s)
  return s
}

const recordEval = (s: Session) => {
  const { batchSize: B, sizes } = s.opts
  s.history.push({ step: s.step, train: estimateLoss(s.model, s.evalTrain, B, sizes.contextLen), val: estimateLoss(s.model, s.evalVal, B, sizes.contextLen), seconds: s.seconds })
}

export const sampleText = (s: Session, len = s.opts.sampleLen, temperature = s.opts.temperature, prompt?: string): string => {
  const start = prompt && encodeChars(s.data, prompt).length ? encodeChars(s.data, prompt) : [s.data.train[0]]
  return decodeChars(s.data, generate(s.model, start, len, temperature, s.sampleRng))
}
const recordSample = (s: Session) => s.samples.push({ step: s.step, text: sampleText(s) })

/** One optimisation step: batch, forward, backward, clip, AdamW. Returns the batch loss. */
export const trainStep = (s: Session): number => {
  const { batchSize: B, sizes } = s.opts
  const T = sizes.contextLen
  getBatch(s.data.train, B, T, s.rng, s.x, s.y)
  const loss = forward(s.model, s.x, B, T, s.y)
  backward(s.model, s.x, B, T, s.y)
  adamwStep(s.model, s.opts.optim)
  s.step++
  s.lastLoss = loss
  if (s.step % s.opts.evalEvery === 0) recordEval(s)
  if (s.step % s.opts.sampleEvery === 0) recordSample(s)
  return loss
}

/**
 * Train for up to `maxSteps` steps or until `budgetMs` has passed (checked after every step).
 * `now` is injectable so tests stay deterministic. Returns the number of steps taken.
 */
export const trainFor = (s: Session, budgetMs: number, maxSteps = Infinity, now: () => number = () => performance.now()): number => {
  const t0 = now()
  const base = s.seconds
  let n = 0
  while (n < maxSteps) {
    s.seconds = base + (now() - t0) / 1000 // so loss points record training time, not wall time
    trainStep(s)
    n++
    if (!Number.isFinite(s.lastLoss) || now() - t0 >= budgetMs) break
  }
  s.seconds = base + (now() - t0) / 1000
  return n
}
