import { describe, expect, it } from 'vitest'
import { countParams, headDim, humanCount, PRESETS } from './params'

const preset = (id: string) => PRESETS.find((p) => p.id === id)!.config

describe('countParams', () => {
  it('GPT-2 small = 124,439,808 (tied head, biases included)', () => {
    const b = countParams(preset('gpt2'))
    expect(b.total).toBe(124_439_808)
    expect(b.tokenEmb).toBe(38_597_376)
    expect(b.posEmb).toBe(786_432)
    expect(b.attnPerBlock).toBe(2_362_368)
    expect(b.mlpPerBlock).toBe(4_722_432)
    expect(b.head).toBe(0)
  })
  it('the other published GPT-2 sizes', () => {
    expect(countParams({ vocabSize: 50257, contextLen: 1024, nEmbd: 1024, nHead: 16, nLayer: 24 }).total).toBe(354_823_168)
    expect(countParams({ vocabSize: 50257, contextLen: 1024, nEmbd: 1280, nHead: 20, nLayer: 36 }).total).toBe(774_030_080)
    expect(countParams(preset('gpt2-xl')).total).toBe(1_557_611_200)
  })
  it('the repo tiny GPT = 809,856: what sum(p.numel() for p in GPT(cfg).parameters()) returns with vocab 65', () => {
    const b = countParams(preset('tiny'))
    expect(b.total).toBe(809_856)
    expect(humanCount(b.total)).toBe('0.81M') // tiny_gpt.py prints "params 0.81M"
    expect(b.mlp).toBe(526_848)
    expect(b.attn).toBe(264_192)
  })
  it('untying the head adds V*D', () => {
    const c = preset('gpt2')
    expect(countParams(c, { tied: false }).total - countParams(c).total).toBe(50257 * 768)
  })
  it('the number of heads does not change the count', () => {
    const c = preset('tiny')
    expect(countParams({ ...c, nHead: 1 }).total).toBe(countParams({ ...c, nHead: 8 }).total)
  })
  it('headDim', () => {
    expect(headDim(preset('gpt2'))).toBe(64)
    expect(headDim({ ...preset('gpt2'), nHead: 5 })).toBeNull()
  })
  it('humanCount', () => {
    expect(humanCount(124_439_808)).toBe('124.4M')
    expect(humanCount(1_557_611_200)).toBe('1.56B')
    expect(humanCount(8320)).toBe('8.3K')
  })
})

describe('numbers quoted in the build-gpt lesson', () => {
  it('exercise config: V=100, C=32, D=64, 2 layers -> 108,544', () => {
    expect(countParams({ vocabSize: 100, contextLen: 32, nEmbd: 64, nHead: 4, nLayer: 2 }).total).toBe(108_544)
  })
  it('tiny GPT with n_layer=6 -> 1,206,400; with n_embd=256 -> 3,192,576', () => {
    const tiny = preset('tiny')
    expect(countParams({ ...tiny, nLayer: 6 }).total).toBe(1_206_400)
    expect(countParams({ ...tiny, nEmbd: 256 }).total).toBe(3_192_576)
    expect(humanCount(1_206_400)).toBe('1.21M')
    expect(humanCount(3_192_576)).toBe('3.19M')
  })
  it('shares: tiny GPT MLP 65% / attention 33% / embeddings 2%; GPT-2 small embeddings 31%, XL 5%', () => {
    const t = countParams(preset('tiny'))
    expect(Math.round((t.mlp / t.total) * 100)).toBe(65)
    expect(Math.round((t.attn / t.total) * 100)).toBe(33)
    expect(Math.round(((t.tokenEmb + t.posEmb) / t.total) * 100)).toBe(2)
    const g = countParams(preset('gpt2'))
    expect(Math.round((g.tokenEmb / g.total) * 100)).toBe(31)
    expect(Math.round(g.mlp / 1e6)).toBe(57)
    expect(Math.round(g.attn / 1e6)).toBe(28)
    expect(Math.round((12 * 12 * 768 * 768) / 1e6)).toBe(85)
    const xl = countParams(preset('gpt2-xl'))
    expect(Math.round((xl.tokenEmb / xl.total) * 100)).toBe(5)
  })
})
