import { describe, expect, it } from 'vitest'
import { DEFAULT_PARAMS, naiveBilledInput, naiveOverflowStep, simulate, type BudgetParams } from './contextBudget'

const P: BudgetParams = { ...DEFAULT_PARAMS }
const fixed = P.system + P.toolDefs + P.docs + P.task
const growth = P.output + P.toolResult

describe('the naive loop', () => {
  it('billed input equals the closed form n*F + g*n(n-1)/2', () => {
    const sim = simulate(P)
    expect(sim.rows).toHaveLength(12)
    expect(sim.billedInput).toBe(naiveBilledInput(12, fixed, growth))
    expect(sim.billedInput).toBe(12 * 9200 + 2800 * 66) // 295,200
    expect(sim.billedOutput).toBe(12 * 300)
    for (const n of [1, 2, 5, 30]) expect(simulate({ ...P, steps: n, window: 1e9 }).billedInput).toBe(naiveBilledInput(n, fixed, growth))
  })
  it('the worked example of the lesson: 10 steps of 1,000 on a 2,000 prefix', () => {
    const sim = simulate({ ...P, system: 2000, toolDefs: 0, docs: 0, task: 0, output: 200, toolResult: 800, steps: 10, window: 1e9 })
    expect(sim.rows.map((r) => r.input)).toEqual([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000])
    expect(sim.billedInput).toBe(65000)
    expect(simulate({ ...P, system: 2000, toolDefs: 0, docs: 0, task: 0, output: 200, toolResult: 800, steps: 20, window: 1e9 }).billedInput).toBe(230000)
  })
  it('doubling the steps roughly quadruples the appended part of the bill', () => {
    const appended = (n: number) => simulate({ ...P, steps: n, window: 1e9 }).billedInput - n * fixed
    expect(appended(40) / appended(20)).toBeCloseTo((40 * 39) / (20 * 19), 10)
  })
  it('each step adds the previous output and tool result, and the prefix never changes', () => {
    const { rows } = simulate(P)
    expect(rows[0].input).toBe(fixed)
    expect(rows[3].segments).toMatchObject({ system: 2000, toolDefs: 3000, docs: 4000, history: 900, toolResults: 7500 })
    expect(rows[3].input - rows[2].input).toBe(growth)
  })
  it('finds the overflow step', () => {
    const sim = simulate(P)
    // 9200 + (i-1)*2800 + 300 > 32000  ->  i - 1 > 8.03  ->  step 10
    expect(sim.overflowAt?.step).toBe(10)
    expect(naiveOverflowStep(12, fixed, growth, P.output, P.window)).toBe(10)
    expect(sim.rows.filter((r) => r.overflow).map((r) => r.step)).toEqual([10, 11, 12])
    expect(simulate({ ...P, window: 200000 }).overflowAt).toBeNull()
    expect(simulate({ ...P, window: 9000 }).overflowAt?.step).toBe(1)
  })
})

describe('prompt caching', () => {
  it('changes the bill, not the tokens or the window', () => {
    const sim = simulate(P)
    expect(sim.cachedInput).toBe(sim.billedInput - fixed - 11 * growth) // every call re-reads the previous call's input
    expect(sim.costCached).toBeLessThan(sim.cost)
    expect(sim.cost).toBeCloseTo((295200 * 3 + 3600 * 15) / 1e6, 10)
    expect(sim.costCached).toBeCloseTo(((sim.cachedInput * 0.1 + (295200 - sim.cachedInput) * 1.25) * 3 + 3600 * 15) / 1e6, 10)
    expect(simulate({ ...P, cacheRead: 1, cacheWrite: 1 }).costCached).toBeCloseTo(sim.cost, 10)
    expect(sim.peak).toBe(simulate({ ...P, cacheRead: 0.5 }).peak)
  })
})

describe('strategies', () => {
  const naive = simulate(P)
  it('truncating tool results reduces billed tokens and delays overflow', () => {
    const sim = simulate({ ...P, truncateTo: 500 })
    expect(sim.billedInput).toBe(naiveBilledInput(12, fixed, 800))
    expect(sim.billedInput).toBeLessThan(naive.billedInput)
    expect(sim.overflowAt).toBeNull()
    expect(simulate({ ...P, truncateTo: 99999 }).billedInput).toBe(naive.billedInput)
  })
  it('compaction bounds the window, costs extra calls and resets the cache', () => {
    const sim = simulate({ ...P, summariseEvery: 4 })
    expect(sim.rows.filter((r) => r.kind === 'compaction')).toHaveLength(2) // after steps 4 and 8, not after the last
    expect(sim.overflowAt).toBeNull()
    expect(sim.billedInput).toBeLessThan(naive.billedInput)
    expect(sim.peak).toBe(fixed + 500 + 4 * growth + 500) // the second compaction call: old summary + four steps in, new summary out
    const afterCompaction = sim.rows.find((r) => r.kind === 'step' && r.step === 5)!
    expect(afterCompaction.segments).toMatchObject({ summary: 500, history: 0, toolResults: 0 })
    expect(afterCompaction.cached).toBe(fixed) // only the fixed prefix is still a cache hit
    expect(sim.billedOutput).toBe(12 * 300 + 2 * 500)
  })
  it('a sub-agent isolates the research steps in their own window', () => {
    const sim = simulate({ ...P, subAgentSteps: 6 })
    const sub = sim.rows.filter((r) => r.agent === 'sub')
    const main = sim.rows.filter((r) => r.agent === 'main')
    expect([sub.length, main.length]).toEqual([6, 6])
    expect(sub[0].segments.docs).toBe(0)
    expect(sub[5].output).toBe(500) // its last output is the report
    expect(main[0].segments.summary).toBe(500) // the parent sees the report only
    expect(sim.billedInput).toBeLessThan(naive.billedInput)
    expect(sim.overflowAt).toBeNull()
    expect(sim.billedInput).toBe(naiveBilledInput(6, fixed - 4000, growth) + naiveBilledInput(6, fixed + 500, growth))
  })
  it('never moves all steps into the sub-agent', () => {
    expect(simulate({ ...P, subAgentSteps: 99 }).rows.filter((r) => r.agent === 'main')).toHaveLength(1)
  })
})
