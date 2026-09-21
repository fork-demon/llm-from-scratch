// Read a Hugging Face config.json and size the model from it: parameters by component,
// weight memory, KV cache, training memory. Pure functions.
//
// Two dialects are understood:
//   GPT-2 style  (n_embd, n_layer, n_head, n_positions): learned positions, LayerNorm with bias,
//                a two-matrix GELU MLP, biases everywhere, head tied to the token table.
//   Llama style  (hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
//                intermediate_size): RoPE (no position table), RMSNorm (gain only), a three-matrix
//                SwiGLU MLP, usually no biases, GQA, head usually untied.
// The GPT-2 rules are the ones in params.ts (tested there against 124,439,808); the tests check
// the Llama rules against the totals the Hub reports for real checkpoints.
import { kvBytesPerToken } from './kvcache'

export type Family = 'gpt2' | 'llama'

export interface ModelSpec {
  family: Family
  modelType: string
  vocab: number
  hidden: number // D
  layers: number
  heads: number
  kvHeads: number // equal to heads unless GQA is used
  headDim: number
  intermediate: number // width of the MLP's middle layer
  context: number
  tied: boolean // does the head share storage with the token table?
  positions: 'learned' | 'rope'
  norm: 'layernorm' | 'rmsnorm'
  mlp: 'gelu' | 'swiglu'
  qkvBias: boolean
  oBias: boolean
  mlpBias: boolean
  ropeTheta: number | null
  slidingWindow: number | null
  dtype: string | null
}

export interface FieldNote {
  key: string
  value: string
  meaning: string
  lesson: string | null // lesson id that taught the idea
  sizing: boolean // does this field change the parameter count or the memory?
}

export type ParseResult =
  | { ok: true; spec: ModelSpec; fields: FieldNote[]; warnings: string[] }
  | { ok: false; error: string }

/* ------------------------------------------------------------------ */
/* What each field means, in the words of earlier lessons               */
/* ------------------------------------------------------------------ */

type Note = [meaning: string, lesson: string | null, sizing?: boolean]

const NOTES: Record<string, Note> = {
  // shared
  architectures: ['which model class to build: the code that gives these tensors their meaning', 'build-gpt'],
  model_type: ['the architecture family; it picks the Python class that reads the rest of this file', 'build-gpt'],
  vocab_size: ['rows of the token embedding table (and of the head): one per token the tokenizer can produce', 'tokenization', true],
  bos_token_id: ['id of the special “beginning of sequence” token', 'tokenization'],
  eos_token_id: ['id of the special “end of sequence” token; generation stops when it is sampled', 'tokenization'],
  pad_token_id: ['id used to pad short sequences in a batch; padding is masked out', 'training-gpt'],
  initializer_range: ['standard deviation of the random starting weights; only matters when training from scratch', 'training-gpt'],
  tie_word_embeddings: ['true: the head reuses the token table (weight tying). false: the head is a second V × D matrix', 'build-gpt', true],
  torch_dtype: ['the number format the weights were saved in; bytes per parameter follow from it', 'inference-systems', true],
  dtype: ['the number format the weights were saved in (newer name for torch_dtype)', 'inference-systems', true],
  use_cache: ['use the KV cache during generation', 'inference'],
  transformers_version: ['version of the library that wrote this file; not a property of the model', null],
  // GPT-2 dialect
  n_embd: ['D: how many numbers represent one token (tiny_gpt.py: n_embd)', 'embeddings', true],
  n_layer: ['how many Transformer blocks are stacked (tiny_gpt.py: n_layer)', 'transformer-block', true],
  n_head: ['attention heads per block; each head gets D / n_head numbers (tiny_gpt.py: n_head)', 'masks-and-heads', true],
  n_positions: ['rows of the learned position table = the context window (tiny_gpt.py: context_len)', 'transformer-block', true],
  n_ctx: ['older duplicate of n_positions: the context window', 'transformer-block'],
  n_inner: ['width of the MLP’s middle layer; null means the default, 4 × D', 'transformer-block', true],
  activation_function: ['the MLP’s nonlinearity; gelu_new is GPT-2’s tanh approximation of GELU', 'neurons'],
  layer_norm_epsilon: ['the tiny constant under the square root in LayerNorm, so it never divides by zero', 'transformer-block'],
  attn_pdrop: ['dropout rate on attention weights; only active in train mode', 'training-gpt'],
  embd_pdrop: ['dropout rate on the embeddings; only active in train mode', 'training-gpt'],
  resid_pdrop: ['dropout rate on each sub-layer’s output; only active in train mode', 'training-gpt'],
  summary_activation: ['only read by an old classification variant of GPT-2; ignored when generating text', null],
  summary_first_dropout: ['only read by an old classification variant of GPT-2; ignored when generating text', null],
  summary_proj_to_labels: ['only read by an old classification variant of GPT-2; ignored when generating text', null],
  summary_type: ['only read by an old classification variant of GPT-2; ignored when generating text', null],
  summary_use_proj: ['only read by an old classification variant of GPT-2; ignored when generating text', null],
  task_specific_params: ['default generation settings for library pipelines; not part of the network', 'inference'],
  // Llama dialect
  hidden_size: ['D: how many numbers represent one token (GPT-2 calls it n_embd)', 'embeddings', true],
  num_hidden_layers: ['how many Transformer blocks are stacked (GPT-2: n_layer)', 'transformer-block', true],
  num_attention_heads: ['query heads per block (GPT-2: n_head)', 'masks-and-heads', true],
  num_key_value_heads: ['key/value heads per block. Fewer than the query heads means GQA: smaller K and V matrices, smaller KV cache', 'modern-architecture', true],
  head_dim: ['numbers per head, when it is not simply D / heads', 'masks-and-heads', true],
  intermediate_size: ['width of the MLP’s middle layer. With SwiGLU there are three matrices of this width, not two', 'modern-architecture', true],
  hidden_act: ['the MLP’s nonlinearity; silu inside a gated MLP is what “SwiGLU” means', 'modern-architecture'],
  max_position_embeddings: ['the context window the model was trained for. With RoPE there is no position table, so this adds no parameters', 'modern-architecture', true],
  rope_theta: ['the base of RoPE’s rotation speeds; larger bases are used for longer contexts', 'modern-architecture'],
  rope_scaling: ['optional recipe for stretching RoPE beyond the trained context', 'modern-architecture'],
  rms_norm_eps: ['the tiny constant under the square root in RMSNorm', 'modern-architecture'],
  attention_bias: ['whether the Q, K, V and output projections have bias vectors (Llama: no)', 'attention', true],
  mlp_bias: ['whether the MLP matrices have bias vectors (Llama: no)', 'transformer-block', true],
  attention_dropout: ['dropout rate on attention weights; 0 in most modern models', 'training-gpt'],
  sliding_window: ['each token attends only to this many previous tokens, which caps the KV cache', 'modern-architecture', true],
  use_sliding_window: ['whether the sliding window is actually switched on', 'modern-architecture', true],
  pretraining_tp: ['a leftover from how the training run was split across GPUs; 1 means ignore', null],
}

/* ------------------------------------------------------------------ */
/* Parsing                                                              */
/* ------------------------------------------------------------------ */

const isPosInt = (x: unknown): x is number => typeof x === 'number' && Number.isInteger(x) && x > 0

const show = (v: unknown): string => {
  if (v === null) return 'null'
  if (typeof v === 'object') return Array.isArray(v) ? JSON.stringify(v) : '{…}'
  return String(v)
}

const MOE_KEYS = ['num_local_experts', 'num_experts', 'n_routed_experts', 'moe_intermediate_size', 'num_experts_per_tok']

export const parseConfig = (input: string | Record<string, unknown>): ParseResult => {
  let c: Record<string, unknown>
  if (typeof input === 'string') {
    try {
      const parsed: unknown = JSON.parse(input)
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return { ok: false, error: 'A config.json is a JSON object: it must start with { and end with }.' }
      c = parsed as Record<string, unknown>
    } catch (e) {
      return { ok: false, error: `This is not valid JSON: ${(e as Error).message}` }
    }
  } else c = input

  const warnings: string[] = []
  const need = (keys: string[]): string | null => {
    const missing = keys.filter((k) => !isPosInt(c[k]))
    return missing.length ? `Missing or non-numeric field${missing.length > 1 ? 's' : ''}: ${missing.join(', ')}.` : null
  }
  const modelType = typeof c.model_type === 'string' ? c.model_type : 'unknown'
  const dtype = typeof c.dtype === 'string' ? c.dtype : typeof c.torch_dtype === 'string' ? c.torch_dtype : null
  let spec: ModelSpec

  if (isPosInt(c.n_embd) || modelType === 'gpt2') {
    const err = need(['n_embd', 'n_layer', 'n_head', 'n_positions', 'vocab_size'])
    if (err) return { ok: false, error: err }
    const D = c.n_embd as number
    const heads = c.n_head as number
    if (D % heads !== 0) return { ok: false, error: `n_embd (${D}) is not divisible by n_head (${heads}).` }
    if (modelType !== 'gpt2' && modelType !== 'unknown') warnings.push(`model_type is “${modelType}”; counted with GPT-2’s rules because the field names are GPT-2’s.`)
    spec = {
      family: 'gpt2', modelType, vocab: c.vocab_size as number, hidden: D, layers: c.n_layer as number, heads, kvHeads: heads, headDim: D / heads,
      intermediate: isPosInt(c.n_inner) ? c.n_inner : 4 * D, context: c.n_positions as number,
      tied: c.tie_word_embeddings !== false, // the library default is true, and GPT-2's file does not mention it
      positions: 'learned', norm: 'layernorm', mlp: 'gelu', qkvBias: true, oBias: true, mlpBias: true, ropeTheta: null, slidingWindow: null, dtype,
    }
  } else if (isPosInt(c.hidden_size)) {
    const err = need(['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size', 'vocab_size'])
    if (err) return { ok: false, error: err }
    const moe = MOE_KEYS.filter((k) => k in c)
    if (moe.length) return { ok: false, error: `This looks like a mixture-of-experts model (${moe.join(', ')}). This reader only sizes dense models: an MoE has many MLPs per block and uses a few per token.` }
    const D = c.hidden_size as number
    const heads = c.num_attention_heads as number
    const kvHeads = isPosInt(c.num_key_value_heads) ? c.num_key_value_heads : heads
    if (!isPosInt(c.head_dim) && D % heads !== 0) return { ok: false, error: `hidden_size (${D}) is not divisible by num_attention_heads (${heads}).` }
    if (heads % kvHeads !== 0) return { ok: false, error: `num_attention_heads (${heads}) must be a multiple of num_key_value_heads (${kvHeads}): each K/V head serves a whole group of query heads.` }
    const known = ['llama', 'mistral', 'qwen2']
    if (!known.includes(modelType)) warnings.push(`model_type is “${modelType}”. Counted with Llama’s rules (RoPE, RMSNorm, three-matrix MLP). Families differ in small ways (extra norms, biases), so treat the total as an estimate.`)
    if (typeof c.tie_word_embeddings !== 'boolean') warnings.push('tie_word_embeddings is not in the file; assumed false, the default for Llama-style models.')
    if (!isPosInt(c.max_position_embeddings)) warnings.push('max_position_embeddings is missing; the context window is shown as 0.')
    const attnBias = c.attention_bias === true
    const window = isPosInt(c.sliding_window) && c.use_sliding_window !== false ? c.sliding_window : null
    spec = {
      family: 'llama', modelType, vocab: c.vocab_size as number, hidden: D, layers: c.num_hidden_layers as number, heads, kvHeads,
      headDim: isPosInt(c.head_dim) ? c.head_dim : D / heads, intermediate: c.intermediate_size as number,
      context: isPosInt(c.max_position_embeddings) ? c.max_position_embeddings : 0,
      tied: c.tie_word_embeddings === true,
      positions: 'rope', norm: 'rmsnorm', mlp: 'swiglu',
      // Qwen2's code puts a bias on Q, K and V (not on the output projection) without a config field for it
      qkvBias: attnBias || modelType === 'qwen2', oBias: attnBias, mlpBias: c.mlp_bias === true,
      ropeTheta: typeof c.rope_theta === 'number' ? c.rope_theta : null, slidingWindow: window, dtype,
    }
  } else {
    return { ok: false, error: 'No n_embd and no hidden_size: this does not look like a decoder-only language model config.' }
  }

  const fields: FieldNote[] = Object.keys(c).map((key) => {
    const note = NOTES[key]
    return { key, value: show(c[key]), meaning: note ? note[0] : 'not needed to size the model; see the model’s documentation', lesson: note ? note[1] : null, sizing: !!note?.[2] }
  })
  return { ok: true, spec, fields, warnings }
}

/* ------------------------------------------------------------------ */
/* Counting                                                             */
/* ------------------------------------------------------------------ */

export interface SpecBreakdown {
  tokenEmb: number
  posEmb: number // 0 with RoPE: rotation needs no table
  qProj: number // per block
  kProj: number
  vProj: number
  oProj: number
  attnPerBlock: number
  mlpPerBlock: number
  normPerBlock: number
  attn: number // all blocks
  mlp: number
  norms: number // block norms plus the final one
  head: number // 0 when tied
  total: number
}

export const countSpec = (s: ModelSpec): SpecBreakdown => {
  const D = s.hidden
  const qOut = s.heads * s.headDim
  const kvOut = s.kvHeads * s.headDim // GQA: fewer K/V heads, so smaller K and V matrices
  const qProj = D * qOut + (s.qkvBias ? qOut : 0)
  const kProj = D * kvOut + (s.qkvBias ? kvOut : 0)
  const vProj = kProj
  const oProj = qOut * D + (s.oBias ? D : 0)
  const attnPerBlock = qProj + kProj + vProj + oProj
  const I = s.intermediate
  const mlpPerBlock = s.mlp === 'swiglu'
    ? 3 * D * I + (s.mlpBias ? 2 * I + D : 0) // gate, up, down
    : 2 * D * I + (s.mlpBias ? I + D : 0) // up, down
  const oneNorm = s.norm === 'layernorm' ? 2 * D : D // RMSNorm has a gain and no bias
  const normPerBlock = 2 * oneNorm
  const tokenEmb = s.vocab * D
  const posEmb = s.positions === 'learned' ? s.context * D : 0
  const head = s.tied ? 0 : s.vocab * D
  const attn = s.layers * attnPerBlock
  const mlp = s.layers * mlpPerBlock
  const norms = s.layers * normPerBlock + oneNorm
  return { tokenEmb, posEmb, qProj, kProj, vProj, oProj, attnPerBlock, mlpPerBlock, normPerBlock, attn, mlp, norms, head, total: tokenEmb + posEmb + attn + mlp + norms + head }
}

/* ------------------------------------------------------------------ */
/* Memory                                                               */
/* ------------------------------------------------------------------ */

export const BYTES_PER_PARAM = { fp32: 4, bf16: 2, int8: 1, int4: 0.5 } as const
export type Precision = keyof typeof BYTES_PER_PARAM

/** Weights only. int8/int4 ignore the small extra cost of the scales (see the inference-systems lesson). */
export const weightBytes = (params: number, precision: Precision): number => params * BYTES_PER_PARAM[precision]

/**
 * The standard estimate for mixed-precision training with Adam (Rajbhandari et al., ZeRO, 2019):
 * 2 (16-bit weights) + 2 (16-bit gradients) + 4 + 4 + 4 (32-bit master weights, Adam momentum, Adam variance).
 * Activations are extra and depend on batch size and sequence length.
 */
export const TRAIN_BYTES_PER_PARAM = 16
export const trainingBytes = (params: number): number => params * TRAIN_BYTES_PER_PARAM

/** K and V for one token, all layers, at `bytes` per number (2 for 16-bit). */
export const specKvBytesPerToken = (s: ModelSpec, bytes = 2): number =>
  kvBytesPerToken({ layers: s.layers, kvHeads: s.kvHeads, headDim: s.headDim, bytesPerNumber: bytes })

/** Tokens the cache can ever hold for one sequence: the context, or the sliding window if one is on. */
export const kvTokenCap = (s: ModelSpec): number => (s.slidingWindow ? Math.min(s.slidingWindow, s.context || s.slidingWindow) : s.context)

/** Decimal gigabytes, as the Hub shows file sizes: 8.03B parameters × 2 bytes = 16.06 GB. */
export const formatGB = (bytes: number): string => {
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(2)} TB`
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(bytes >= 1e11 ? 0 : bytes >= 1e10 ? 1 : 2)} GB`
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(bytes >= 1e8 ? 0 : 1)} MB`
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(bytes >= 1e5 ? 0 : 1)} KB`
  return `${bytes} B`
}

/* ------------------------------------------------------------------ */
/* Presets: the real files, verbatim, as fetched from the Hub           */
/* ------------------------------------------------------------------ */

export interface ConfigPreset { id: string; label: string; source: string; note: string; json: string }

export const CONFIG_PRESETS: ConfigPreset[] = [
  {
    id: 'gpt2',
    label: 'GPT-2 small',
    source: 'https://huggingface.co/openai-community/gpt2/blob/main/config.json',
    note: 'the model you loaded in inspect_hf_model.py',
    json: `{
  "activation_function": "gelu_new",
  "architectures": ["GPT2LMHeadModel"],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": { "text-generation": { "do_sample": true, "max_length": 50 } },
  "vocab_size": 50257
}`,
  },
  {
    id: 'llama3-8b',
    label: 'Llama 3 8B',
    source: 'https://huggingface.co/NousResearch/Meta-Llama-3-8B/blob/main/config.json',
    note: 'Meta’s own repository needs a login, so this is an ungated copy of the same file',
    json: `{
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.0.dev0",
  "use_cache": true,
  "vocab_size": 128256
}`,
  },
  {
    id: 'mistral-7b',
    label: 'Mistral 7B v0.1',
    source: 'https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json',
    note: 'same block as Llama, plus a sliding attention window',
    json: `{
  "architectures": ["MistralForCausalLM"],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 32768,
  "model_type": "mistral",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "sliding_window": 4096,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.34.0.dev0",
  "use_cache": true,
  "vocab_size": 32000
}`,
  },
  {
    id: 'qwen-0.5b',
    label: 'Qwen2.5 0.5B',
    source: 'https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json',
    note: 'a small modern model: Llama-style, but with a tied head',
    json: `{
  "architectures": ["Qwen2ForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 24,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": true,
  "use_mrope": false,
  "use_sliding_window": false,
  "vocab_size": 151936
}`,
  },
]
