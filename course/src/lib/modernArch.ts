// Arithmetic behind the "modern architecture" lesson: KV-cache sizes (MHA vs GQA vs MQA),
// LayerNorm vs RMSNorm on a single vector, and a few parameter / cost counts.
// Pure functions, no DOM.
import type { Vec } from './math'

/* ---------- KV cache ---------- */
export interface KvConfig {
  layers: number
  queryHeads: number
  kvHeads: number
  headDim: number
  tokens: number // tokens held in the cache (context length x concurrent sequences)
  bytesPerValue: number // 4 = fp32, 2 = fp16/bf16, 1 = int8, 0.5 = 4-bit
}

/** 2 (one K and one V) x layers x kv_heads x head_dim x tokens x bytes per number. */
export const kvCacheBytes = (c: Pick<KvConfig, 'layers' | 'kvHeads' | 'headDim' | 'tokens' | 'bytesPerValue'>): number =>
  2 * c.layers * c.kvHeads * c.headDim * c.tokens * c.bytesPerValue

export const kvBytesPerToken = (c: Pick<KvConfig, 'layers' | 'kvHeads' | 'headDim' | 'bytesPerValue'>): number =>
  kvCacheBytes({ ...c, tokens: 1 })

export interface KvComparison {
  mha: number // every query head has its own K/V head
  gqa: number // the chosen number of K/V heads
  mqa: number // one K/V head shared by all query heads
  savingVsMha: number // mha / gqa
  groupSize: number // query heads per K/V head
}

export const compareKv = (c: KvConfig): KvComparison => {
  if (c.kvHeads < 1 || c.queryHeads % c.kvHeads !== 0) throw new Error('kvHeads must divide queryHeads')
  const mha = kvCacheBytes({ ...c, kvHeads: c.queryHeads })
  const gqa = kvCacheBytes(c)
  const mqa = kvCacheBytes({ ...c, kvHeads: 1 })
  return { mha, gqa, mqa, savingVsMha: mha / gqa, groupSize: c.queryHeads / c.kvHeads }
}

/** Valid K/V head counts for a given number of query heads (the divisors). */
export const divisors = (n: number): number[] => {
  const out: number[] = []
  for (let i = 1; i <= n; i++) if (n % i === 0) out.push(i)
  return out
}

/** Binary units: 1 KiB = 1024 bytes. */
export const formatBytes = (bytes: number): string => {
  const units = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB']
  let v = bytes
  let u = 0
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024
    u++
  }
  return `${u === 0 ? v.toFixed(0) : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)} ${units[u]}`
}

export const GIB = 1024 ** 3

export interface KvPreset {
  id: string
  label: string
  note: string
  layers: number
  queryHeads: number
  kvHeads: number
  headDim: number
  context: number
}

/** Configurations taken from the public model cards / config files of each model. */
export const KV_PRESETS: KvPreset[] = [
  { id: 'tiny', label: 'Our tiny GPT', note: 'tiny_gpt.py Config: 4 layers, 4 heads, n_embd 128, so head_dim 32, context 64', layers: 4, queryHeads: 4, kvHeads: 4, headDim: 32, context: 64 },
  { id: 'gpt2', label: 'GPT-2 small', note: '12 layers, 12 heads, head_dim 64, context 1,024. Plain multi-head attention.', layers: 12, queryHeads: 12, kvHeads: 12, headDim: 64, context: 1024 },
  { id: 'llama2-7b', label: 'Llama-2 7B', note: '32 layers, 32 heads, 32 K/V heads (plain multi-head attention), head_dim 128, context 4,096', layers: 32, queryHeads: 32, kvHeads: 32, headDim: 128, context: 4096 },
  { id: 'llama3-8b', label: 'Llama-3 8B', note: '32 layers, 32 heads, 8 K/V heads (GQA), head_dim 128, context 8,192', layers: 32, queryHeads: 32, kvHeads: 8, headDim: 128, context: 8192 },
  { id: 'llama-70b', label: 'Llama-2 / Llama-3 70B', note: '80 layers, 64 heads, 8 K/V heads (GQA), head_dim 128. Context 4,096 (Llama-2) or 8,192 (Llama-3).', layers: 80, queryHeads: 64, kvHeads: 8, headDim: 128, context: 4096 },
]

/* ---------- context length ---------- */
/** Number of query-key scores for one head in one layer when processing T tokens at once (no mask shortcut). */
export const attentionScoreEntries = (tokens: number): number => tokens * tokens

/* ---------- normalisation ---------- */
export const mean = (x: Vec): number => x.reduce((s, v) => s + v, 0) / x.length

export interface NormSteps {
  mean: number
  variance: number // mean of (x - mean)^2
  meanSquare: number // mean of x^2
  out: Vec
}

/** LayerNorm without the learned gain and bias: (x - mean) / sqrt(variance + eps). Matches torch.nn.LayerNorm at init. */
export const layerNorm = (x: Vec, eps = 1e-5): NormSteps => {
  const m = mean(x)
  const variance = mean(x.map((v) => (v - m) ** 2))
  const meanSquare = mean(x.map((v) => v * v))
  return { mean: m, variance, meanSquare, out: x.map((v) => (v - m) / Math.sqrt(variance + eps)) }
}

/** RMSNorm without the learned gain: x / sqrt(mean(x^2) + eps). No mean subtraction, no bias. */
export const rmsNorm = (x: Vec, eps = 1e-5): NormSteps => {
  const m = mean(x)
  const variance = mean(x.map((v) => (v - m) ** 2))
  const meanSquare = mean(x.map((v) => v * v))
  return { mean: m, variance, meanSquare, out: x.map((v) => v / Math.sqrt(meanSquare + eps)) }
}

export const centre = (x: Vec): Vec => {
  const m = mean(x)
  return x.map((v) => v - m)
}

/* ---------- feed-forward parameter counts (weights only, no biases) ---------- */
/** Classic MLP: d -> hidden -> d (two matrices). Gated (SwiGLU): three matrices, d -> hidden twice, hidden -> d. */
export const ffnParams = (d: number, hidden: number, gated: boolean): number => (gated ? 3 : 2) * d * hidden

/* ---------- mixture of experts ---------- */
/** FFN parameters stored vs used per token in one MoE layer. */
export const moeFfnParams = (perExpert: number, experts: number, activePerToken: number): { total: number; active: number } => ({
  total: perExpert * experts,
  active: perExpert * activePerToken,
})
