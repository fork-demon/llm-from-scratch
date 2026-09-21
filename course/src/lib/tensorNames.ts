// Real tensor names and shapes, matched to the course concept that built each one.
// GPT-2 names and shapes are what phase6-engineering/inspect_hf_model.py prints.
// Llama 3 8B names and shapes were read from the safetensors header of the published checkpoint.
import { makeRng } from './rng'

export type Round = 'gpt2' | 'llama'

export interface TensorItem {
  id: string
  round: Round
  name: string
  shape: number[]
  concept: string // the right answer, in the course's words
  lesson: string
  explain: string
}

export const TENSORS: TensorItem[] = [
  { id: 'wte', round: 'gpt2', name: 'transformer.wte.weight', shape: [50257, 768], concept: 'Token embedding table (and, tied, the head)', lesson: 'embeddings',
    explain: 'One row of 768 numbers per token id: tok_emb in tiny_gpt.py. 50,257 is the vocabulary size. lm_head.weight is this same tensor, which is why it is not stored twice.' },
  { id: 'wpe', round: 'gpt2', name: 'transformer.wpe.weight', shape: [1024, 768], concept: 'Learned position table', lesson: 'transformer-block',
    explain: 'One row per position, added to the token vector: pos_emb in tiny_gpt.py. 1024 rows is the context window. Position 1024 has no row, so GPT-2 cannot read past it.' },
  { id: 'ln1', round: 'gpt2', name: 'transformer.h.0.ln_1.weight', shape: [768], concept: 'LayerNorm gain, applied before attention', lesson: 'transformer-block',
    explain: 'A vector, not a matrix: one gain per dimension. ln_1 comes before attention inside the block (pre-norm), exactly as Block.ln1 does. There is a matching ln_1.bias.' },
  { id: 'cattn', round: 'gpt2', name: 'transformer.h.0.attn.c_attn.weight', shape: [768, 2304], concept: 'Wq, Wk and Wv fused into one matrix', lesson: 'attention',
    explain: '2304 = 3 × 768. One multiply makes Q, K and V together, then the result is split in three: attn.qkv in tiny_gpt.py. Stored as (in, out) because GPT-2 uses Conv1D, the transpose of nn.Linear’s (out, in).' },
  { id: 'cproj', round: 'gpt2', name: 'transformer.h.0.attn.c_proj.weight', shape: [768, 768], concept: 'Wo: mixes the concatenated heads', lesson: 'masks-and-heads',
    explain: 'After the 12 heads are concatenated back to 768 numbers, this matrix lets them exchange information: attn.proj in tiny_gpt.py.' },
  { id: 'cfc', round: 'gpt2', name: 'transformer.h.0.mlp.c_fc.weight', shape: [768, 3072], concept: 'MLP, first layer: D up to 4D', lesson: 'transformer-block',
    explain: '3072 = 4 × 768. The feed-forward network widens each token’s vector, applies GELU, then narrows it again. This is the widening half.' },
  { id: 'mproj', round: 'gpt2', name: 'transformer.h.0.mlp.c_proj.weight', shape: [3072, 768], concept: 'MLP, second layer: 4D back down to D', lesson: 'transformer-block',
    explain: 'Same name as the attention c_proj, different parent: mlp, not attn. The shape settles it: 3072 in, 768 out. LoRA’s target_modules match on names, so this distinction matters.' },
  { id: 'lnf', round: 'gpt2', name: 'transformer.ln_f.weight', shape: [768], concept: 'Final LayerNorm, after the last block', lesson: 'build-gpt',
    explain: 'No “h.N” in the name: it is outside the blocks. It runs once, just before the head: GPT.ln_f.' },

  { id: 'embed', round: 'llama', name: 'model.embed_tokens.weight', shape: [128256, 4096], concept: 'Token embedding table', lesson: 'embeddings',
    explain: '128,256 tokens, 4,096 numbers each. Same job as wte. There is no position table next to it: RoPE rotates Q and K inside attention instead.' },
  { id: 'qproj', round: 'llama', name: 'model.layers.0.self_attn.q_proj.weight', shape: [4096, 4096], concept: 'Wq: 32 query heads of 128', lesson: 'attention',
    explain: 'Llama keeps Wq, Wk, Wv as separate matrices. nn.Linear stores (out, in): 32 heads × 128 = 4096 out, 4096 in. No bias tensor exists.' },
  { id: 'kproj', round: 'llama', name: 'model.layers.0.self_attn.k_proj.weight', shape: [1024, 4096], concept: 'Wk under GQA: only 8 key heads of 128', lesson: 'modern-architecture',
    explain: '1024 = 8 × 128. Grouped-query attention: 32 query heads share 8 key/value heads, so K and V are a quarter the size of Q, and so is the KV cache.' },
  { id: 'oproj', round: 'llama', name: 'model.layers.0.self_attn.o_proj.weight', shape: [4096, 4096], concept: 'Wo: mixes the concatenated heads', lesson: 'masks-and-heads',
    explain: 'Same shape as q_proj, so only the name tells them apart. o_proj is GPT-2’s attn.c_proj.' },
  { id: 'gate', round: 'llama', name: 'model.layers.0.mlp.gate_proj.weight', shape: [14336, 4096], concept: 'SwiGLU gate: the third MLP matrix', lesson: 'modern-architecture',
    explain: 'SwiGLU computes down( silu(gate(x)) × up(x) ). gate_proj and up_proj have the same shape; the gate decides how much of each of the 14,336 channels passes.' },
  { id: 'down', round: 'llama', name: 'model.layers.0.mlp.down_proj.weight', shape: [4096, 14336], concept: 'MLP, last layer: back down to D', lesson: 'transformer-block',
    explain: '(out, in) = (4096, 14336): from the wide middle back to the model width. It is GPT-2’s mlp.c_proj.' },
  { id: 'inorm', round: 'llama', name: 'model.layers.0.input_layernorm.weight', shape: [4096], concept: 'RMSNorm gain, applied before attention', lesson: 'modern-architecture',
    explain: 'The name says layernorm, the class is RMSNorm: a gain and no bias, so there is no matching .bias tensor. It plays the role of ln_1.' },
  { id: 'lmhead', round: 'llama', name: 'lm_head.weight', shape: [128256, 4096], concept: 'The head, as its own matrix (not tied)', lesson: 'build-gpt',
    explain: 'Same shape as embed_tokens, but tie_word_embeddings is false, so it is a separate 525M-number matrix stored in the file. In GPT-2 this tensor is absent from the file because it is tied.' },
]

export const itemsFor = (round: Round): TensorItem[] => TENSORS.filter((t) => t.round === round)

/** The answer options for a round: every concept once, in a fixed shuffled order (never the order of the questions). */
export const optionsFor = (round: Round): string[] => {
  const items = itemsFor(round)
  const rng = makeRng(round === 'gpt2' ? 11 : 23)
  const opts = items.map((t) => t.concept)
  for (let i = opts.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1))
    ;[opts[i], opts[j]] = [opts[j], opts[i]]
  }
  if (opts.every((o, i) => o === items[i].concept)) opts.push(opts.shift()!)
  return opts
}

export const grade = (round: Round, answers: Record<string, string>): { correct: number; total: number; perItem: Record<string, boolean> } => {
  const perItem: Record<string, boolean> = {}
  for (const t of itemsFor(round)) perItem[t.id] = answers[t.id] === t.concept
  return { correct: Object.values(perItem).filter(Boolean).length, total: itemsFor(round).length, perItem }
}

export const shapeText = (shape: number[]): string => `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`
export const numel = (shape: number[]): number => shape.reduce((a, b) => a * b, 1)
