// A small simulator of what fills an agent's context window, step by step, and what it costs.
// No model is involved: this is bookkeeping. Every model call re-sends the whole context, so
//   input(step i) = fixed prefix + everything the loop has appended so far.
// Three strategies change what gets appended or kept: truncating tool results, compacting the
// history every k steps, and running the first m steps in a sub-agent with its own window.

export interface BudgetParams {
  system: number // system prompt tokens
  toolDefs: number // tool definitions (names, descriptions, schemas)
  docs: number // documents stuffed into the prompt up front
  task: number // the user's request
  output: number // tokens the model writes per step (thought + tool call)
  toolResult: number // average tokens one tool result adds
  steps: number
  window: number // context window, in tokens
  priceIn: number // dollars per million input tokens (example number)
  priceOut: number // dollars per million output tokens (example number)
  cacheRead: number // multiplier for cached input tokens, e.g. 0.1
  cacheWrite: number // multiplier for input tokens written to the cache, e.g. 1.25 (1 = no premium)
  truncateTo: number | null // cap each tool result at this many tokens
  summariseEvery: number | null // compact the history after every k steps
  summaryTokens: number // size of a compaction summary or a sub-agent's report
  subAgentSteps: number // run the first m steps in a sub-agent (0 = off)
}

export const DEFAULT_PARAMS: BudgetParams = {
  system: 2000, toolDefs: 3000, docs: 4000, task: 200, output: 300, toolResult: 2500, steps: 12, window: 32000,
  priceIn: 3, priceOut: 15, cacheRead: 0.1, cacheWrite: 1.25,
  truncateTo: null, summariseEvery: null, summaryTokens: 500, subAgentSteps: 0,
}

export interface Segments { system: number; toolDefs: number; docs: number; task: number; summary: number; history: number; toolResults: number }
export const SEGMENT_KEYS: (keyof Segments)[] = ['system', 'toolDefs', 'docs', 'task', 'summary', 'history', 'toolResults']

export interface CallRow {
  agent: 'main' | 'sub'
  kind: 'step' | 'compaction'
  step: number // step number within its agent (compaction rows carry the step they follow)
  segments: Segments
  input: number
  output: number
  cached: number // input tokens that were an unchanged prefix of the previous call
  overflow: boolean // input + output does not fit the window
}

export interface Simulation {
  rows: CallRow[]
  overflowAt: CallRow | null // first call that does not fit
  peak: number // largest input + output of any call
  billedInput: number
  billedOutput: number
  cachedInput: number
  cost: number // no caching
  costCached: number
}

const total = (s: Segments) => s.system + s.toolDefs + s.docs + s.task + s.summary + s.history + s.toolResults

const runLoop = (agent: 'main' | 'sub', steps: number, fixed: Pick<Segments, 'system' | 'toolDefs' | 'docs' | 'task'>, startSummary: number, p: BudgetParams, lastOutput: number | null): CallRow[] => {
  const rows: CallRow[] = []
  const result = p.truncateTo === null ? p.toolResult : Math.min(p.toolResult, p.truncateTo)
  const fixedTokens = fixed.system + fixed.toolDefs + fixed.docs + fixed.task
  let summary = startSummary
  let history = 0
  let toolResults = 0
  let cachedPrefix = 0 // how much of the next call is an unchanged prefix of the previous one
  for (let step = 1; step <= steps; step++) {
    const segments: Segments = { ...fixed, summary, history, toolResults }
    const input = total(segments)
    const output = step === steps && lastOutput !== null ? lastOutput : p.output
    rows.push({ agent, kind: 'step', step, segments, input, output, cached: Math.min(cachedPrefix, input), overflow: input + output > p.window })
    cachedPrefix = input
    history += output
    toolResults += result
    if (p.summariseEvery !== null && step % p.summariseEvery === 0 && step < steps) {
      // Compaction is itself a model call: it reads the whole context and writes a summary.
      const seg: Segments = { ...fixed, summary, history, toolResults }
      const cInput = total(seg)
      rows.push({ agent, kind: 'compaction', step, segments: seg, input: cInput, output: p.summaryTokens, cached: Math.min(cachedPrefix, cInput), overflow: cInput + p.summaryTokens > p.window })
      summary = p.summaryTokens
      history = 0
      toolResults = 0
      cachedPrefix = fixedTokens // everything after the fixed prefix was rewritten: those cache entries are useless now
    }
  }
  return rows
}

export const simulate = (p: BudgetParams): Simulation => {
  const m = Math.max(0, Math.min(p.subAgentSteps, p.steps - 1))
  const rows: CallRow[] = []
  if (m > 0) {
    // The sub-agent gets its own window: system prompt, tools and a brief. Not the parent's documents or history.
    rows.push(...runLoop('sub', m, { system: p.system, toolDefs: p.toolDefs, docs: 0, task: p.task }, 0, p, p.summaryTokens))
  }
  // The parent sees only the sub-agent's report, never its working.
  rows.push(...runLoop('main', p.steps - m, { system: p.system, toolDefs: p.toolDefs, docs: p.docs, task: p.task }, m > 0 ? p.summaryTokens : 0, p, null))

  const billedInput = rows.reduce((s, r) => s + r.input, 0)
  const billedOutput = rows.reduce((s, r) => s + r.output, 0)
  const cachedInput = rows.reduce((s, r) => s + r.cached, 0)
  const cost = (billedInput * p.priceIn + billedOutput * p.priceOut) / 1e6
  const costCached = ((cachedInput * p.cacheRead + (billedInput - cachedInput) * p.cacheWrite) * p.priceIn + billedOutput * p.priceOut) / 1e6
  return {
    rows, overflowAt: rows.find((r) => r.overflow) ?? null, peak: Math.max(...rows.map((r) => r.input + r.output)),
    billedInput, billedOutput, cachedInput, cost, costCached,
  }
}

/** Closed form for the naive loop: n calls, each re-sending a fixed prefix F plus (i - 1) appended steps of size g. */
export const naiveBilledInput = (n: number, fixed: number, growth: number): number => n * fixed + (growth * n * (n - 1)) / 2

/** First step of the naive loop whose input plus output no longer fits, or null if all n steps fit. */
export const naiveOverflowStep = (n: number, fixed: number, growth: number, output: number, window: number): number | null => {
  for (let i = 1; i <= n; i++) if (fixed + (i - 1) * growth + output > window) return i
  return null
}
