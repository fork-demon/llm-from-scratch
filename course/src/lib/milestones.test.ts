import { describe, expect, it } from 'vitest'
import { MILESTONES, capstoneProgress, itemKey, parseChecked, toggle } from './milestones'

describe('capstone milestones', () => {
  it('has six milestones in build order with unique ids', () => {
    expect(MILESTONES.map((m) => m.id)).toEqual(['tokenizer', 'transformer', 'training', 'inference', 'rag', 'tools'])
    expect(MILESTONES.every((m) => m.checklist.length >= 4 && m.files.length >= 1)).toBe(true)
  })
  it('starts at zero and points at the first milestone', () => {
    const p = capstoneProgress({})
    expect([p.done, p.percent, p.milestonesComplete, p.current]).toEqual([0, 0, 0, 'tokenizer'])
    expect(p.total).toBe(MILESTONES.reduce((s, m) => s + m.checklist.length, 0))
  })
  it('toggle checks and unchecks without mutating', () => {
    const a = {}
    const b = toggle(a, itemKey('rag', 0))
    expect(a).toEqual({})
    expect(b).toEqual({ 'rag:0': true })
    expect(toggle(b, 'rag:0')).toEqual({})
  })
  it('counts a milestone complete only when every item is checked', () => {
    let c = {}
    MILESTONES[0].checklist.forEach((_, i) => { c = toggle(c, itemKey('tokenizer', i)) })
    const p = capstoneProgress(c)
    expect(p.perMilestone[0].complete).toBe(true)
    expect(p.milestonesComplete).toBe(1)
    expect(p.current).toBe('transformer')
    expect(p.percent).toBe(Math.round((MILESTONES[0].checklist.length / p.total) * 100))
  })
  it('reaches 100% with nothing left', () => {
    let c = {}
    for (const m of MILESTONES) m.checklist.forEach((_, i) => { c = toggle(c, itemKey(m.id, i)) })
    const p = capstoneProgress(c)
    expect([p.percent, p.current, p.milestonesComplete]).toEqual([100, null, 6])
  })
  it('survives corrupt or stale storage', () => {
    expect(parseChecked(null)).toEqual({})
    expect(parseChecked('not json')).toEqual({})
    expect(parseChecked('[1,2]')).toEqual({})
    expect(parseChecked('{"rag:0":true,"rag:99":true,"gone:0":true,"rag:1":"yes"}')).toEqual({ 'rag:0': true })
  })
})
