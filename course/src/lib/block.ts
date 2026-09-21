// One Transformer block, step by step, mirroring class Block in
// phase3-transformers/tiny_gpt.py:
//     x = x + attn(ln1(x))
//     x = x + ffn(ln2(x))
// Pure functions, no DOM. One honest simplification: the feed-forward uses ReLU
// (so the numbers can be checked by hand) where tiny_gpt.py uses GELU, a smooth ReLU.
import { add, matmul, type Mat, type Vec } from './math'
import { multiHeadFromQKV, randomMat, type MultiHeadResult } from './multihead'
import { makeRng, type Rng } from './rng'

/* ---------- small pieces ---------- */

/** LayerNorm on ONE token vector: subtract its mean, divide by its standard deviation, then learned gain and bias. */
export const layerNormVec = (v: Vec, gain?: Vec, bias?: Vec, eps = 1e-5): Vec => {
  const mean = v.reduce((a, b) => a + b, 0) / v.length
  const variance = v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length
  const std = Math.sqrt(variance + eps)
  return v.map((x, i) => ((x - mean) / std) * (gain?.[i] ?? 1) + (bias?.[i] ?? 0))
}

/** LayerNorm over a (T, D) matrix: every row (token) is normalised separately. Same as nn.LayerNorm(D). */
export const layerNorm = (x: Mat, gain?: Vec, bias?: Vec, eps = 1e-5): Mat => x.map((row) => layerNormVec(row, gain, bias, eps))

export const meanOf = (v: Vec): number => v.reduce((a, b) => a + b, 0) / v.length
export const stdOf = (v: Vec): number => {
  const m = meanOf(v)
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length)
}
/** Root-mean-square of every number in a matrix: "how big are the activations?" */
export const rms = (m: Mat): number => {
  const flat = m.flat()
  return Math.sqrt(flat.reduce((a, b) => a + b * b, 0) / flat.length)
}

/** x @ W + b, the nn.Linear of the repo (W stored as (in, out) here, like the NumPy files). */
export const linear = (x: Mat, W: Mat, b?: Vec): Mat => matmul(x, W).map((row) => (b ? add(row, b) : row))

export const addMat = (a: Mat, b: Mat): Mat => a.map((row, i) => add(row, b[i]))

/** token embeddings + the position vector of the slot each token sits in. */
export const addPositions = (tok: Mat, posEmb: Mat): Mat => tok.map((row, t) => add(row, posEmb[t]))

export const swapRows = (m: Mat, i: number, j: number): Mat => m.map((row, r) => (r === i ? m[j] : r === j ? m[i] : row))

/* ---------- parameters ---------- */
export interface NormParams { gain: Vec; bias: Vec }
export interface AttnParams { Wq: Mat; Wk: Mat; Wv: Mat; Wo: Mat; bq: Vec; bk: Vec; bv: Vec; bo: Vec; nHeads: number }
export interface FfnParams { W1: Mat; b1: Vec; W2: Mat; b2: Vec }
export interface BlockParams { ln1: NormParams; attn: AttnParams; ln2: NormParams; ffn: FfnParams }

const zerosVec = (n: number): Vec => Array(n).fill(0)
const onesVec = (n: number): Vec => Array(n).fill(1)
const zerosMat = (r: number, c: number): Mat => Array.from({ length: r }, () => zerosVec(c))

/** Random, UNTRAINED block. LayerNorm starts as gain 1, bias 0 and biases start at 0, as in PyTorch. `gainScale` multiplies the weight matrices. */
export const makeBlockParams = (rng: Rng, D: number, nHeads: number, digits = 2, gainScale = 1): BlockParams => {
  const m = (r: number, c: number): Mat => randomMat(rng, r, c, digits).map((row) => row.map((v) => v * gainScale))
  return {
    ln1: { gain: onesVec(D), bias: zerosVec(D) },
    attn: { Wq: m(D, D), Wk: m(D, D), Wv: m(D, D), Wo: m(D, D), bq: zerosVec(D), bk: zerosVec(D), bv: zerosVec(D), bo: zerosVec(D), nHeads },
    ln2: { gain: onesVec(D), bias: zerosVec(D) },
    ffn: { W1: m(D, 4 * D), b1: zerosVec(4 * D), W2: m(4 * D, D), b2: zerosVec(D) },
  }
}

/** A block whose sub-layers output nothing. Thanks to the residuals the whole block is then the identity. */
export const zeroBlockParams = (D: number, nHeads: number): BlockParams => ({
  ln1: { gain: onesVec(D), bias: zerosVec(D) },
  attn: { Wq: zerosMat(D, D), Wk: zerosMat(D, D), Wv: zerosMat(D, D), Wo: zerosMat(D, D), bq: zerosVec(D), bk: zerosVec(D), bv: zerosVec(D), bo: zerosVec(D), nHeads },
  ln2: { gain: onesVec(D), bias: zerosVec(D) },
  ffn: { W1: zerosMat(D, 4 * D), b1: zerosVec(4 * D), W2: zerosMat(4 * D, D), b2: zerosVec(D) },
})

/** How many numbers a block holds: 4D²+4D (attention) + 8D²+5D (MLP) + 4D (two LayerNorms). */
export const countBlockNumbers = (p: BlockParams): number => {
  const mat = (m: Mat) => m.length * m[0].length
  const a = p.attn
  const f = p.ffn
  return mat(a.Wq) + mat(a.Wk) + mat(a.Wv) + mat(a.Wo) + a.bq.length + a.bk.length + a.bv.length + a.bo.length
    + mat(f.W1) + f.b1.length + mat(f.W2) + f.b2.length
    + p.ln1.gain.length + p.ln1.bias.length + p.ln2.gain.length + p.ln2.bias.length
}

/* ---------- sub-layers ---------- */
export const selfAttention = (x: Mat, p: AttnParams, causal = true): MultiHeadResult & { y: Mat } => {
  const r = multiHeadFromQKV(linear(x, p.Wq, p.bq), linear(x, p.Wk, p.bk), linear(x, p.Wv, p.bv), p.Wo, p.nHeads, causal)
  return { ...r, y: r.out.map((row) => add(row, p.bo)) }
}

export interface FfnTrace { pre: Mat; act: Mat; out: Mat }
/** Expand to 4D, bend (ReLU), project back to D. Every token (row) is processed on its own. */
export const feedForward = (x: Mat, p: FfnParams): FfnTrace => {
  const pre = linear(x, p.W1, p.b1)
  const act = pre.map((row) => row.map((v) => Math.max(0, v)))
  return { pre, act, out: linear(act, p.W2, p.b2) }
}

/* ---------- the block ---------- */
export interface BlockOpts {
  residual?: boolean // false: x = attn(ln1(x)) instead of x = x + attn(ln1(x))
  norm?: boolean // false: skip both LayerNorms
  causal?: boolean
}

export interface BlockTrace {
  input: Mat // (T, D)
  ln1: Mat // (T, D)
  attn: MultiHeadResult & { y: Mat }
  afterAttn: Mat // (T, D) input + attention output
  ln2: Mat // (T, D)
  ffn: FfnTrace // hidden is (T, 4D)
  out: Mat // (T, D) same shape as the input
}

export const blockForward = (x: Mat, p: BlockParams, opts: BlockOpts = {}): BlockTrace => {
  const { residual = true, norm = true, causal = true } = opts
  const ln1 = norm ? layerNorm(x, p.ln1.gain, p.ln1.bias) : x
  const attn = selfAttention(ln1, p.attn, causal)
  const afterAttn = residual ? addMat(x, attn.y) : attn.y
  const ln2 = norm ? layerNorm(afterAttn, p.ln2.gain, p.ln2.bias) : afterAttn
  const ffn = feedForward(ln2, p.ffn)
  const out = residual ? addMat(afterAttn, ffn.out) : ffn.out
  return { input: x, ln1, attn, afterAttn, ln2, ffn, out }
}

/** Blocks stack because output shape = input shape. Returns the residual stream after every block. */
export const runStack = (x: Mat, blocks: BlockParams[], opts: BlockOpts = {}): Mat[] => {
  const outs: Mat[] = []
  let cur = x
  for (const b of blocks) {
    cur = blockForward(cur, b, opts).out
    outs.push(cur)
  }
  return outs
}

/* ---------- the fixed example of the explorer: 3 tokens, 4 numbers each ---------- */
export const BLOCK_TOKENS = ['dog', 'bites', 'man']
export const blockDemo = (seed = 4) => {
  const tokEmb: Record<string, Vec> = {
    dog: [1, 0.2, -0.5, 0.3],
    bites: [-0.4, 1, 0.6, -0.2],
    man: [0.8, -0.3, 0.2, 1],
  }
  const posEmb: Mat = [
    [0.5, -0.5, 0, 0.3],
    [0, 0.4, -0.4, 0.2],
    [-0.5, 0.1, 0.4, -0.3],
  ]
  return { tokEmb, posEmb, params: makeBlockParams(makeRng(seed), 4, 2, 1) }
}

/** Build the block input for a token order, with or without positional information. */
export const demoInput = (tokens: string[], usePositions: boolean, demo = blockDemo()): Mat => {
  const tok = tokens.map((t) => demo.tokEmb[t])
  return usePositions ? addPositions(tok, demo.posEmb) : tok
}

/* ---------- depth experiment: how big do the numbers get after N blocks? ---------- */
export interface DepthOpts { norm: boolean; residual: boolean; weightScale?: number }
/** RMS size of the activations after 0, 1, ..., N blocks of random (untrained) weights. */
export const depthScales = (N: number, opts: DepthOpts, seed = 21, T = 4, D = 16): number[] => {
  const rng = makeRng(seed)
  const x: Mat = Array.from({ length: T }, () => Array.from({ length: D }, () => rng.normal()))
  const blocks = Array.from({ length: N }, () => makeBlockParams(rng, D, 4, 4, opts.weightScale ?? 1))
  return [rms(x), ...runStack(x, blocks, { norm: opts.norm, residual: opts.residual, causal: true }).map(rms)]
}
