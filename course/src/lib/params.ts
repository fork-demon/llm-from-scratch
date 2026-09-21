// Where do a GPT's parameters live? Counting rules for the architecture in
// phase3-transformers/tiny_gpt.py (which is GPT-2's architecture). Pure functions.

export interface GptConfig {
  vocabSize: number
  contextLen: number
  nEmbd: number // D
  nHead: number // does not change the count: heads split D, they do not add to it
  nLayer: number
}

export interface ParamBreakdown {
  tokenEmb: number // V * D
  posEmb: number // context * D
  attnPerBlock: number // 4D² + 4D   (Wq, Wk, Wv, Wo and their biases)
  mlpPerBlock: number // 8D² + 5D   (D -> 4D -> D and their biases)
  normPerBlock: number // 4D         (two LayerNorms: gain + bias each)
  attn: number // all blocks
  mlp: number
  norms: number // all block LayerNorms + the final one
  head: number // 0 when tied to the token embeddings, else V * D
  total: number
}

export const countParams = (c: GptConfig, opts: { tied?: boolean } = {}): ParamBreakdown => {
  const { tied = true } = opts
  const D = c.nEmbd
  const tokenEmb = c.vocabSize * D
  const posEmb = c.contextLen * D
  const attnPerBlock = 4 * D * D + 4 * D
  const mlpPerBlock = 8 * D * D + 5 * D
  const normPerBlock = 4 * D
  const attn = c.nLayer * attnPerBlock
  const mlp = c.nLayer * mlpPerBlock
  const norms = c.nLayer * normPerBlock + 2 * D
  const head = tied ? 0 : c.vocabSize * D // nn.Linear(D, V, bias=False)
  return { tokenEmb, posEmb, attnPerBlock, mlpPerBlock, normPerBlock, attn, mlp, norms, head, total: tokenEmb + posEmb + attn + mlp + norms + head }
}

export const headDim = (c: GptConfig): number | null => (c.nEmbd % c.nHead === 0 ? c.nEmbd / c.nHead : null)

export const PRESETS: { id: string; label: string; note: string; config: GptConfig }[] = [
  { id: 'tiny', label: 'Our tiny GPT', note: 'tiny_gpt.py on Shakespeare: 65 different characters', config: { vocabSize: 65, contextLen: 64, nEmbd: 128, nHead: 4, nLayer: 4 } },
  { id: 'gpt2', label: 'GPT-2 small', note: 'the 2019 model, “124M”', config: { vocabSize: 50257, contextLen: 1024, nEmbd: 768, nHead: 12, nLayer: 12 } },
  { id: 'gpt2-xl', label: 'GPT-2 XL', note: 'the largest GPT-2, “1.5B”', config: { vocabSize: 50257, contextLen: 1024, nEmbd: 1600, nHead: 25, nLayer: 48 } },
]

/** 809856 -> "0.81M", 124439808 -> "124.4M", 1557611200 -> "1.56B" */
export const humanCount = (n: number): string => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (n >= 1e5) return `${(n / 1e6).toFixed(n >= 1e8 ? 1 : 2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return String(n)
}
