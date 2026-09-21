// A miniature GPT forward pass with random (UNTRAINED) weights, mirroring class GPT in
// phase3-transformers/tiny_gpt.py: tok_emb + pos_emb -> blocks -> ln_f -> head (tied) -> logits.
// Built from block.ts. Pure functions, no DOM.
import { matmul, sampleIndex, softmax, transpose, type Mat, type Vec } from './math'
import { addPositions, blockForward, countBlockNumbers, layerNorm, makeBlockParams, type BlockParams, type NormParams } from './block'
import { makeRng } from './rng'
import type { GptConfig } from './params'

/* ---------- character-level tokenizer (same idea as tiny_gpt.py: sorted(set(text))) ---------- */
export const CHARS = [...new Set(" ',.?abcdefghijklmnopqrstuvwxyz")].sort()
const STOI = new Map(CHARS.map((c, i) => [c, i]))
/** Characters outside the vocabulary are dropped: a fixed vocabulary cannot represent them. */
export const encode = (text: string): number[] => [...text.toLowerCase()].flatMap((c) => (STOI.has(c) ? [STOI.get(c)!] : []))
export const decode = (ids: number[]): string => ids.map((i) => CHARS[i]).join('')
export const showChar = (c: string): string => (c === ' ' ? '␣' : c)

export const TRACER_CONFIG: GptConfig = { vocabSize: CHARS.length, contextLen: 16, nEmbd: 8, nHead: 2, nLayer: 2 }

export interface TinyGpt {
  cfg: GptConfig
  tokEmb: Mat // (V, D)
  posEmb: Mat // (context, D)
  blocks: BlockParams[]
  lnF: NormParams
  // no separate head matrix: the head reuses tokEmb (weight tying)
}

export const makeTinyGpt = (cfg: GptConfig = TRACER_CONFIG, seed = 1337): TinyGpt => {
  const rng = makeRng(seed)
  const emb = (rows: number): Mat => Array.from({ length: rows }, () => Array.from({ length: cfg.nEmbd }, () => Math.round(rng.normal() * 50) / 100))
  return {
    cfg,
    tokEmb: emb(cfg.vocabSize),
    posEmb: emb(cfg.contextLen),
    blocks: Array.from({ length: cfg.nLayer }, () => makeBlockParams(rng, cfg.nEmbd, cfg.nHead)),
    lnF: { gain: Array(cfg.nEmbd).fill(1), bias: Array(cfg.nEmbd).fill(0) },
  }
}

/** How many numbers the model actually holds. Must agree with params.ts::countParams (tied). */
export const countModelNumbers = (m: TinyGpt): number =>
  m.tokEmb.length * m.cfg.nEmbd + m.posEmb.length * m.cfg.nEmbd + m.blocks.reduce((s, b) => s + countBlockNumbers(b), 0) + m.lnF.gain.length + m.lnF.bias.length

export interface GptTrace {
  ids: number[] // (T)
  tok: Mat // (T, D) rows looked up in the embedding table
  pos: Mat // (T, D) position vectors 0..T-1
  x0: Mat // (T, D) tok + pos
  blockOuts: Mat[] // nLayer x (T, D)
  final: Mat // (T, D) after ln_f
  logits: Mat // (T, V): at EVERY position, a score for every possible next token
}

/** GPT.forward for one sequence (batch size 1). Throws if the sequence is longer than the context window. */
export const gptForward = (m: TinyGpt, ids: number[]): GptTrace => {
  if (ids.length === 0) throw new Error('gptForward: need at least one token')
  if (ids.length > m.cfg.contextLen) throw new Error(`gptForward: ${ids.length} tokens do not fit in a context window of ${m.cfg.contextLen}`)
  const tok = ids.map((i) => m.tokEmb[i])
  const pos = ids.map((_, t) => m.posEmb[t])
  const x0 = addPositions(tok, m.posEmb)
  const blockOuts: Mat[] = []
  let x = x0
  for (const b of m.blocks) {
    x = blockForward(x, b).out
    blockOuts.push(x)
  }
  const final = layerNorm(x, m.lnF.gain, m.lnF.bias)
  const logits = matmul(final, transpose(m.tokEmb)) // the head: tied to the embedding table
  return { ids, tok, pos, x0, blockOuts, final, logits }
}

/** idx[:, -context_len:] : keep only the most recent tokens that fit. */
export const cropToContext = (ids: number[], contextLen: number): number[] => ids.slice(-contextLen)

export interface NextToken { probs: Vec; id: number }
/** Last row of logits -> temperature -> softmax -> draw. `u` is a uniform random number in [0, 1). */
export const nextToken = (logits: Mat, temperature: number, u: number): NextToken => {
  const probs = softmax(logits[logits.length - 1], temperature)
  return { probs, id: sampleIndex(probs, u) }
}

/** GPT.generate: crop, forward, last position, sample, append, repeat. */
export const generate = (m: TinyGpt, ids: number[], maxNewTokens: number, temperature = 1, seed = 1): number[] => {
  const rng = makeRng(seed)
  let out = ids.slice()
  for (let i = 0; i < maxNewTokens; i++) {
    const { logits } = gptForward(m, cropToContext(out, m.cfg.contextLen))
    out = [...out, nextToken(logits, temperature, rng.next()).id]
  }
  return out
}
