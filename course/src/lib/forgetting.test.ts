import { describe, expect, it } from 'vitest'
import { ADAPTER_RANK, TASK_A, TASK_B, V, ftStep, initFt, logitsOf, lossOn, measureFt, pretrain, trainableCount } from './forgetting'

const base = pretrain()
const run = (mode: 'full' | 'adapter' | 'replay') => ftStep(initFt(base, mode), 10, 300)

describe('catastrophic forgetting in a bigram LM', () => {
  it('task tables are probability tables', () => {
    for (const t of [TASK_A, TASK_B]) expect(t.P.flat().reduce((a, b) => a + b, 0)).toBeCloseTo(1)
  })
  it('pretraining makes the model good at A and worse at B (but better than uniform guessing)', () => {
    const a = lossOn(base, TASK_A)
    const b = lossOn(base, TASK_B)
    expect(a).toBeLessThan(2.1)
    expect(b).toBeGreaterThan(a + 0.5)
    expect(b).toBeLessThan(Math.log(V))
  })
  it('full fine-tune on B: loss on B falls, loss on A rises', () => {
    const before = measureFt(initFt(base, 'full'))
    const after = measureFt(run('full'))
    expect(after.b).toBeLessThan(before.b - 0.4)
    expect(after.a).toBeGreaterThan(before.a + 0.25)
  })
  it('an adapter starts exactly at the base model (B = 0)', () => {
    const s = initFt(base, 'adapter')
    expect(logitsOf(s)).toEqual(base)
  })
  it('frozen base + adapter: unplug the adapter and task A performance is exactly the base', () => {
    const s = run('adapter')
    expect(s.W).toEqual(base)
    expect(measureFt(s).aBaseOnly).toBe(lossOn(base, TASK_A))
    expect(measureFt(s).b).toBeLessThan(lossOn(base, TASK_B) - 0.2) // and it did learn some of B
  })
  it('replay forgets less than full fine-tuning, at a small cost on B', () => {
    const full = measureFt(run('full'))
    const replay = measureFt(run('replay'))
    expect(replay.a).toBeLessThan(full.a - 0.1)
    expect(replay.b).toBeGreaterThan(full.b)
    expect(replay.b).toBeLessThan(full.b + 0.1)
  })
  it('the numbers quoted in the lesson (lr 10, 300 steps)', () => {
    const r = (x: number) => Number(x.toFixed(2))
    expect([r(lossOn(base, TASK_A)), r(lossOn(base, TASK_B))]).toEqual([2.02, 2.73])
    const f = measureFt(run('full'))
    const a = measureFt(run('adapter'))
    const p = measureFt(run('replay'))
    expect([r(f.a), r(f.b)]).toEqual([2.37, 2.12])
    expect([r(a.a), r(a.b), r(a.aBaseOnly)]).toEqual([2.19, 2.47, 2.02])
    expect([r(p.a), r(p.b)]).toEqual([2.24, 2.15])
  })
  it('trainable parameter counts', () => {
    expect(trainableCount('full')).toBe(V * V)
    expect(trainableCount('adapter')).toBe(2 * V * ADAPTER_RANK)
    expect([V * V, 2 * V * ADAPTER_RANK]).toEqual([729, 108])
  })
})
