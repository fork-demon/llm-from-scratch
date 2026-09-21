import { describe, expect, it } from 'vitest'
import { TOOLS, runAgent, QUESTIONS } from './agent'
import { EXAMPLE_PRICES, QUESTIONS_V2, callCost, estimateTokens, runBudgeted, sharedPrefix, validateArgs, type ModelSpan, type ToolSpan } from './agentBudget'

const models = (r: ReturnType<typeof runBudgeted>) => r.trace.filter((s): s is ModelSpan => s.kind === 'model')
const tools = (r: ReturnType<typeof runBudgeted>) => r.trace.filter((s): s is ToolSpan => s.kind === 'tool')
const FREE = { maxSteps: 8, maxRepeats: 0, contextBudget: 1e9 }

describe('metering', () => {
  it('estimates tokens as characters / 4, rounded up', () => {
    expect([estimateTokens(''), estimateTokens('abcd'), estimateTokens('abcde')]).toEqual([0, 1, 2])
    expect(sharedPrefix('abcdef', 'abcxyz')).toBe(3)
  })
  it('cost arithmetic with and without caching', () => {
    expect(callCost(1000, 0, 100, EXAMPLE_PRICES, false)).toBeCloseTo((1000 * 3 + 100 * 15) / 1e6, 12)
    expect(callCost(1000, 800, 100, EXAMPLE_PRICES, true)).toBeCloseTo(((800 * 0.1 + 200 * 1.25) * 3 + 100 * 15) / 1e6, 12)
  })
})

describe('matches agent_budget.py span for span', () => {
  it('the three-step oncall run', () => {
    const run = runBudgeted(QUESTIONS_V2.oncall)
    expect(run.stopped).toBe('answer')
    expect(run.answer).toMatch(/^165/)
    expect(models(run).map((s) => s.inputTokens)).toEqual([137, 184, 215])
    expect(models(run).map((s) => s.cachedTokens)).toEqual([0, 136, 184])
    expect(models(run).map((s) => s.outputTokens)).toEqual([28, 28, 24])
    expect(tools(run).map((s) => [s.name, s.resultTokens])).toEqual([['search_docs', 18], ['calculator', 1]])
    expect(run.trace.map((s) => s.latencyMs)).toEqual([787, 120, 772, 5, 690])
    expect(run.totals).toMatchObject({ modelCalls: 3, toolCalls: 2, inputTokens: 536, cachedTokens: 320, outputTokens: 80, latencyMs: 2374 })
    expect(run.totals.cost).toBeCloseTo(0.002106, 9)
    expect(run.totals.costWithoutCache).toBeCloseTo(0.002808, 9)
  })
  it('validation: the wrong argument name is rejected, then repaired', () => {
    const run = runBudgeted(QUESTIONS_V2.sloppy)
    expect(tools(run).map((s) => s.status)).toEqual(['invalid_args', 'ok'])
    expect(tools(run)[0].resultTokens).toBe(37)
    expect(run.answer).toBe('161')
    expect(run.totals).toMatchObject({ inputTokens: 509, cachedTokens: 302, outputTokens: 48, latencyMs: 1613 })
    expect(run.totals.cost).toBeCloseTo(0.00158685, 9)
  })
  it('approval gate: denied by default, sent when approved, paused when nobody has decided', () => {
    const denied = runBudgeted(QUESTIONS_V2.email)
    expect(tools(denied).map((s) => s.status)).toEqual(['ok', 'denied'])
    expect(denied.outbox).toEqual([])
    expect(denied.answer).toContain('did not send')
    expect(denied.totals).toMatchObject({ inputTokens: 548, cachedTokens: 311, outputTokens: 91, latencyMs: 2593 })
    const ok = runBudgeted(QUESTIONS_V2.email, { approve: () => true })
    expect(ok.outbox).toEqual([{ to: 'team-lead@example.com', body: 'The oncall rotation has 4 engineers.' }])
    expect(ok.totals).toMatchObject({ inputTokens: 530, outputTokens: 71, latencyMs: 2490 })
    expect(ok.totals.cost).toBeCloseTo(0.00197955, 9)
    const paused = runBudgeted(QUESTIONS_V2.email, { approve: () => 'pending' })
    expect(paused.stopped).toBe('awaiting_approval')
    expect(paused.pending?.name).toBe('send_email')
    expect(paused.outbox).toEqual([])
  })
  it('context growth, truncation and compaction', () => {
    const naive = runBudgeted(QUESTIONS_V2.loop, FREE)
    expect(models(naive).map((s) => s.inputTokens)).toEqual([126, 910, 1694, 2479, 3263, 4047, 4831, 5616])
    expect(naive.totals).toMatchObject({ inputTokens: 22966, cachedTokens: 17345, outputTokens: 168, latencyMs: 8432 })
    expect(naive.totals.cost).toBeCloseTo(0.02880225, 9)
    expect(naive.totals.costWithoutCache).toBeCloseTo(0.071418, 9)
    const trimmed = runBudgeted(QUESTIONS_V2.loop, { ...FREE, maxResultTokens: 100 })
    expect(trimmed.totals).toMatchObject({ inputTokens: 5159, cachedTokens: 3990 })
    expect(Object.keys(trimmed.offloaded)).toHaveLength(8)
    const compacted = runBudgeted(QUESTIONS_V2.loop, { maxSteps: 8, maxRepeats: 0 })
    expect(compacted.totals).toMatchObject({ inputTokens: 11315, cachedTokens: 6435 })
    expect(compacted.totals.cachedTokens / compacted.totals.inputTokens).toBeLessThan(naive.totals.cachedTokens / naive.totals.inputTokens)
  })
  it('every brake stops the run with its own reason', () => {
    const loop = runBudgeted(QUESTIONS_V2.loop, { maxSteps: 8 })
    expect([loop.stopped, loop.totals.modelCalls, loop.totals.inputTokens]).toEqual(['loop_detected', 3, 2730])
    const tok = runBudgeted(QUESTIONS_V2.loop, { ...FREE, maxTokens: 5000 })
    expect(tok.stopped).toBe('max_tokens')
    expect(tok.detail).toBe('step 4 needs 2479 input tokens; 2793 of 5000 already used')
    const cost = runBudgeted(QUESTIONS_V2.loop, { ...FREE, maxCost: 0.01 })
    expect(cost.stopped).toBe('max_cost')
    expect(cost.totals.cost).toBeLessThanOrEqual(0.01)
    const steps = runBudgeted(QUESTIONS_V2.loop, { maxSteps: 4, maxRepeats: 0 })
    expect([steps.stopped, steps.totals.modelCalls]).toEqual(['max_steps', 4])
  })
})

describe('closed form and safety', () => {
  it('billed input grows with the square of the step count', () => {
    const ins = models(runBudgeted(QUESTIONS_V2.loop, FREE)).map((s) => s.inputTokens)
    const grow = ins[1] - ins[0]
    const closed = 8 * ins[0] + (grow * 8 * 7) / 2
    expect(Math.abs(ins.reduce((a, b) => a + b, 0) - closed)).toBeLessThanOrEqual(8)
  })
  it('caching changes the bill, not the window', () => {
    const on = runBudgeted(QUESTIONS_V2.oncall)
    const off = runBudgeted(QUESTIONS_V2.oncall, { caching: false })
    expect(on.totals.inputTokens).toBe(off.totals.inputTokens)
    expect(off.totals.cachedTokens).toBe(0)
    expect(on.totals.cost).toBeLessThan(off.totals.cost)
  })
  it('validation messages say what is wrong, what was expected and what to do', () => {
    expect(validateArgs('calculator', { expression: '1+1' })).toBeNull()
    expect(validateArgs('calculator', { expr: '1+1' })).toBe('ERROR: invalid arguments for calculator: missing "expression"; unexpected "expr". Expected {"expression": str}. Fix ARGS and call the tool again.')
    expect(validateArgs('calculator', { expression: 7 })).toContain('must be a str')
  })
  it('does not modify the shared agent module', () => {
    runBudgeted(QUESTIONS_V2.email, { approve: () => true })
    expect(Object.keys(TOOLS)).toEqual(['calculator', 'search_docs'])
    expect(runAgent(QUESTIONS.oncall).answer).toMatch(/^165/)
  })
})
