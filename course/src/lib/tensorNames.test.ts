import { describe, expect, it } from 'vitest'
import { grade, itemsFor, numel, optionsFor, shapeText, TENSORS } from './tensorNames'
import { CONFIG_PRESETS, countSpec, parseConfig } from './hfConfig'
import { LESSONS } from '../data/curriculum'

const breakdown = (id: string) => {
  const r = parseConfig(CONFIG_PRESETS.find((p) => p.id === id)!.json)
  if (!r.ok) throw new Error(r.error)
  return countSpec(r.spec)
}
const t = (id: string) => TENSORS.find((x) => x.id === id)!

describe('tensor shapes agree with the config-based parameter count', () => {
  it('GPT-2', () => {
    const b = breakdown('gpt2')
    expect(numel(t('wte').shape)).toBe(b.tokenEmb)
    expect(numel(t('wpe').shape)).toBe(b.posEmb)
    expect(numel(t('cattn').shape) + 2304).toBe(b.qProj + b.kProj + b.vProj) // weight + bias
    expect(numel(t('cproj').shape) + 768).toBe(b.oProj)
    expect(numel(t('cfc').shape) + 3072 + numel(t('mproj').shape) + 768).toBe(b.mlpPerBlock)
  })
  it('Llama 3 8B', () => {
    const b = breakdown('llama3-8b')
    expect(numel(t('embed').shape)).toBe(b.tokenEmb)
    expect(numel(t('qproj').shape)).toBe(b.qProj)
    expect(numel(t('kproj').shape)).toBe(b.kProj)
    expect(numel(t('oproj').shape)).toBe(b.oProj)
    expect(2 * numel(t('gate').shape) + numel(t('down').shape)).toBe(b.mlpPerBlock)
    expect(2 * numel(t('inorm').shape)).toBe(b.normPerBlock)
    expect(numel(t('lmhead').shape)).toBe(b.head)
  })
})

describe('the matching exercise', () => {
  it('every item links to a lesson that exists, ids and concepts are unique within a round', () => {
    const lessons = new Set(LESSONS.map((l) => l.id))
    expect(new Set(TENSORS.map((x) => x.id)).size).toBe(TENSORS.length)
    for (const x of TENSORS) expect(lessons.has(x.lesson), x.id).toBe(true)
    for (const round of ['gpt2', 'llama'] as const) {
      const concepts = itemsFor(round).map((x) => x.concept)
      expect(new Set(concepts).size).toBe(concepts.length)
      expect(optionsFor(round).slice().sort()).toEqual(concepts.slice().sort())
      expect(optionsFor(round)).not.toEqual(concepts) // never in question order
    }
  })
  it('grades', () => {
    const right = Object.fromEntries(itemsFor('llama').map((x) => [x.id, x.concept]))
    expect(grade('llama', right).correct).toBe(8)
    const swapped = { ...right, qproj: t('oproj').concept, oproj: t('qproj').concept }
    const g = grade('llama', swapped)
    expect(g.correct).toBe(6)
    expect(g.perItem.qproj).toBe(false)
    expect(grade('gpt2', {}).correct).toBe(0)
  })
  it('formats shapes the way PyTorch prints them', () => {
    expect(shapeText([768])).toBe('(768,)')
    expect(shapeText([768, 2304])).toBe('(768, 2304)')
  })
})
