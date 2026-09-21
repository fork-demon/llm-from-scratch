import { describe, expect, it } from 'vitest'
import { CONFIG_PRESETS, countSpec, formatGB, kvTokenCap, parseConfig, specKvBytesPerToken, trainingBytes, weightBytes, type ModelSpec } from './hfConfig'
import { countParams } from './params'
import { LESSONS } from '../data/curriculum'

const spec = (id: string): ModelSpec => {
  const r = parseConfig(CONFIG_PRESETS.find((p) => p.id === id)!.json)
  if (!r.ok) throw new Error(r.error)
  return r.spec
}

describe('parameter counts reproduce the totals the Hub reports for the real checkpoints', () => {
  it('GPT-2 small = 124,439,808, and agrees with params.ts component by component', () => {
    const b = countSpec(spec('gpt2'))
    expect(b.total).toBe(124_439_808)
    const ref = countParams({ vocabSize: 50257, contextLen: 1024, nEmbd: 768, nHead: 12, nLayer: 12 })
    expect([b.tokenEmb, b.posEmb, b.attn, b.mlp, b.norms, b.head]).toEqual([ref.tokenEmb, ref.posEmb, ref.attn, ref.mlp, ref.norms, ref.head])
    expect(b.qProj + b.kProj + b.vProj).toBe(768 * 2304 + 2304) // the fused c_attn
  })

  it('Llama 3 8B = 8,030,261,248 (published as 8.03B)', () => {
    const b = countSpec(spec('llama3-8b'))
    expect(b.total).toBe(8_030_261_248)
    expect(Math.abs(b.total - 8.03e9) / 8.03e9).toBeLessThan(0.01)
    expect(b.posEmb).toBe(0) // RoPE: no position table
    expect(b.qProj).toBe(4096 * 4096)
    expect(b.kProj).toBe(4096 * 1024) // GQA: 8 KV heads of 128, so K is a quarter the size of Q
    expect(b.mlpPerBlock).toBe(3 * 4096 * 14336) // SwiGLU: gate, up, down
    expect(b.normPerBlock).toBe(2 * 4096) // RMSNorm: gain only
    expect(b.head).toBe(128256 * 4096) // untied
  })

  it('Mistral 7B = 7,241,732,096 (published as 7.24B)', () => {
    expect(countSpec(spec('mistral-7b')).total).toBe(7_241_732_096)
  })

  it('Qwen2.5 0.5B = 494,032,768: tied head, and Q/K/V biases that the config does not mention', () => {
    const s = spec('qwen-0.5b')
    expect(s.tied).toBe(true)
    expect(s.slidingWindow).toBeNull() // use_sliding_window is false
    const b = countSpec(s)
    expect(b.head).toBe(0)
    expect(b.kProj).toBe(896 * 128 + 128)
    expect(b.total).toBe(494_032_768)
  })

  it('the rest of the GPT-2 family', () => {
    const fam = (n_embd: number, n_layer: number, n_head: number) => {
      const r = parseConfig({ model_type: 'gpt2', n_embd, n_layer, n_head, n_positions: 1024, vocab_size: 50257 })
      if (!r.ok) throw new Error(r.error)
      return countSpec(r.spec).total
    }
    expect(fam(1024, 24, 16)).toBe(354_823_168)
    expect(fam(1280, 36, 20)).toBe(774_030_080)
    expect(fam(1600, 48, 25)).toBe(1_557_611_200)
  })
})

describe('memory', () => {
  it('weights: bytes per parameter', () => {
    const n = countSpec(spec('llama3-8b')).total
    expect(weightBytes(n, 'fp32')).toBe(n * 4)
    expect(weightBytes(n, 'bf16')).toBe(16_060_522_496)
    expect(weightBytes(n, 'int4')).toBe(n / 2)
    expect(formatGB(weightBytes(n, 'bf16'))).toBe('16.1 GB')
  })
  it('training: 16 bytes per parameter before activations', () => {
    expect(trainingBytes(124_439_808)).toBe(1_991_036_928)
    expect(formatGB(trainingBytes(countSpec(spec('llama3-8b')).total))).toBe('128 GB')
  })
  it('KV cache per token: 2 x layers x kv_heads x head_dim x bytes', () => {
    expect(specKvBytesPerToken(spec('gpt2'))).toBe(2 * 12 * 12 * 64 * 2) // 36,864
    expect(specKvBytesPerToken(spec('llama3-8b'))).toBe(2 * 32 * 8 * 128 * 2) // 131,072 = 128 KiB
    // without GQA it would be four times larger
    const r = parseConfig(CONFIG_PRESETS[1].json.replace('"num_key_value_heads": 8', '"num_key_value_heads": 32'))
    expect(r.ok && specKvBytesPerToken(r.spec)).toBe(4 * 131_072)
  })
  it('a sliding window caps the cache', () => {
    expect(kvTokenCap(spec('mistral-7b'))).toBe(4096)
    expect(kvTokenCap(spec('llama3-8b'))).toBe(8192)
    expect(kvTokenCap(spec('gpt2'))).toBe(1024)
  })
})

describe('parsing', () => {
  it('explains every field of every preset and links only to lessons that exist', () => {
    const ids = new Set(LESSONS.map((l) => l.id))
    for (const p of CONFIG_PRESETS) {
      const r = parseConfig(p.json)
      expect(r.ok, p.id).toBe(true)
      if (!r.ok) continue
      for (const f of r.fields) {
        if (f.lesson) expect(ids.has(f.lesson), `${p.id}.${f.key} -> ${f.lesson}`).toBe(true)
        if (!['max_window_layers', 'use_mrope'].includes(f.key)) expect(f.meaning.startsWith('not needed'), `${p.id}.${f.key}`).toBe(false)
      }
    }
  })
  it('defaults: GPT-2 ties its head, a Llama-style file without the field does not', () => {
    const g = parseConfig({ n_embd: 8, n_layer: 1, n_head: 2, n_positions: 4, vocab_size: 10 })
    expect(g.ok && g.spec.tied).toBe(true)
    const l = parseConfig({ model_type: 'llama', hidden_size: 8, num_hidden_layers: 1, num_attention_heads: 2, intermediate_size: 16, vocab_size: 10, max_position_embeddings: 4 })
    expect(l.ok && l.spec.tied).toBe(false)
    expect(l.ok && l.warnings.some((w) => w.includes('tie_word_embeddings'))).toBe(true)
    expect(l.ok && l.spec.kvHeads).toBe(2) // no num_key_value_heads: plain multi-head attention
  })
  it('an explicit head_dim wins over hidden_size / heads', () => {
    const r = parseConfig({ model_type: 'llama', hidden_size: 12, head_dim: 5, num_hidden_layers: 1, num_attention_heads: 2, num_key_value_heads: 1, intermediate_size: 16, vocab_size: 10, max_position_embeddings: 4, tie_word_embeddings: true })
    expect(r.ok && countSpec(r.spec).qProj).toBe(12 * 10)
    expect(r.ok && countSpec(r.spec).kProj).toBe(12 * 5)
    expect(r.ok && countSpec(r.spec).oProj).toBe(10 * 12)
  })
  it('rejects what it cannot size, with a reason', () => {
    expect(parseConfig('{ not json').ok).toBe(false)
    expect(parseConfig('[1, 2]').ok).toBe(false)
    expect(parseConfig('{"foo": 1}').ok).toBe(false)
    const moe = parseConfig({ model_type: 'mixtral', hidden_size: 8, num_hidden_layers: 1, num_attention_heads: 2, intermediate_size: 16, vocab_size: 10, num_local_experts: 8 })
    expect(!moe.ok && moe.error).toMatch(/mixture-of-experts/)
    const bad = parseConfig({ model_type: 'llama', hidden_size: 8, num_hidden_layers: 1, num_attention_heads: 4, num_key_value_heads: 3, intermediate_size: 16, vocab_size: 10 })
    expect(!bad.ok && bad.error).toMatch(/multiple/)
    const missing = parseConfig({ model_type: 'gpt2', n_embd: 8 })
    expect(!missing.ok && missing.error).toMatch(/n_layer/)
  })
  it('warns when it is guessing the family', () => {
    const r = parseConfig({ model_type: 'gemma', hidden_size: 8, num_hidden_layers: 1, num_attention_heads: 2, intermediate_size: 16, vocab_size: 10, max_position_embeddings: 4, tie_word_embeddings: true })
    expect(r.ok && r.warnings.some((w) => w.includes('estimate'))).toBe(true)
  })
})
