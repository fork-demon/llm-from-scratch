// A TypeScript port of phase6-engineering/agent_budget.py, built ON TOP of ./agent (which is not modified):
// the same loop, wrapped from the outside with token and cost metering, prompt-cache accounting,
// hard budgets, loop detection, tool-argument validation, an approval gate and a structured trace.
// Differences from the Python are marked "PORT NOTE".
import { TOOLS, parseAction, runAgent, scriptedModel, BUDGET_EXHAUSTED, type Model, type Registry } from './agent'

/* ---------- 1. tokens and money ---------- */

/** Roughly 4 characters per token for English. An estimate, as in the Python. */
export const estimateTokens = (text: string): number => Math.ceil(text.length / 4)

export interface Prices { inputPerMtok: number; outputPerMtok: number; cacheReadMultiplier: number; cacheWriteMultiplier: number }
/** Example prices in dollars per million tokens. Parameters, not any vendor's price list. */
export const EXAMPLE_PRICES: Prices = { inputPerMtok: 3, outputPerMtok: 15, cacheReadMultiplier: 0.1, cacheWriteMultiplier: 1.25 }

export const callCost = (inputTokens: number, cachedTokens: number, outputTokens: number, prices: Prices, caching: boolean): number => {
  const fresh = inputTokens - cachedTokens
  const billedIn = caching ? cachedTokens * prices.cacheReadMultiplier + fresh * prices.cacheWriteMultiplier : inputTokens
  return (billedIn * prices.inputPerMtok + outputTokens * prices.outputPerMtok) / 1e6
}

export const sharedPrefix = (a: string, b: string): number => {
  const n = Math.min(a.length, b.length)
  let i = 0
  while (i < n && a[i] === b[i]) i++
  return i
}

/** Latency placeholders, so the trace has a time axis. Invented constants with a plausible shape. */
export const modelLatencyMs = (fresh: number, cached: number, output: number): number => Math.floor(200 + 0.2 * fresh + 0.02 * cached + 20 * output + 0.5)
export const TOOL_LATENCY_MS: Record<string, number> = { calculator: 5, search_docs: 120, read_log: 250, send_email: 300 }

/* ---------- 2. two more tools: one risky, one bulky ---------- */
export const readLog = (service: string): string => {
  const lines: string[] = []
  for (let i = 0; i < 39; i++) lines.push(`2026-01-15T10:${String(i).padStart(2, '0')}:00Z ${service} INFO request_id=${1000 + i} status=200 latency_ms=${40 + ((i * 7) % 50)}`)
  lines.push(`2026-01-15T10:39:00Z ${service} ERROR request_id=1039 status=500 upstream timeout`)
  return lines.join('\n')
}

type ArgType = 'str'
export const TOOL_SCHEMAS: Record<string, Record<string, ArgType>> = {
  calculator: { expression: 'str' },
  search_docs: { query: 'str' },
  send_email: { to: 'str', body: 'str' },
  read_log: { service: 'str' },
}
export const IRREVERSIBLE = new Set(['send_email'])

/** Null if the arguments fit the schema, else an error message the model can act on. */
export const validateArgs = (name: string, args: Record<string, unknown>): string | null => {
  const schema = TOOL_SCHEMAS[name]
  const problems = [
    ...Object.keys(schema).filter((k) => !(k in args)).map((k) => `missing "${k}"`),
    ...Object.keys(args).filter((k) => !(k in schema)).map((k) => `unexpected "${k}"`),
    ...Object.keys(schema).filter((k) => k in args && typeof args[k] !== 'string').map((k) => `"${k}" must be a str`),
  ]
  if (!problems.length) return null
  const expected = '{' + Object.entries(schema).map(([k, t]) => `"${k}": ${t}`).join(', ') + '}'
  return `ERROR: invalid arguments for ${name}: ${problems.join('; ')}. Expected ${expected}. Fix ARGS and call the tool again.`
}

/* ---------- 3. a scripted model with three bad habits ---------- */
export const scriptedModelV2: Model = (context) => {
  const question = /USER QUESTION: (.+)/.exec(context)?.[1] ?? ''

  if (question.toLowerCase().includes('email')) {
    if (!context.includes('RESULT:')) return 'Thought: I need the rotation size first.\nTOOL: search_docs\nARGS: {"query": "oncall rotation"}'
    if (!context.includes('TOOL: send_email')) return 'Thought: Now send it.\nTOOL: send_email\nARGS: {"to": "team-lead@example.com", "body": "The oncall rotation has 4 engineers."}'
    if (context.includes('declined')) return 'Thought: A human said no. I must not retry.\nANSWER: I did not send the email because the approval was declined. The rotation has 4 engineers.'
    return 'Thought: Done.\nANSWER: Email sent. The rotation has 4 engineers.'
  }
  if (question.toLowerCase().includes('checkout')) return 'Thought: Let me look at the logs again.\nTOOL: read_log\nARGS: {"service": "checkout"}'
  if (question.startsWith('What is 23*7?')) {
    if (!context.includes('RESULT:')) return 'Thought: Use the calculator.\nTOOL: calculator\nARGS: {"expr": "23*7"}'
    if (!context.includes('RESULT: 161')) return 'Thought: The argument is called expression.\nTOOL: calculator\nARGS: {"expression": "23*7"}'
    return 'Thought: I have it.\nANSWER: 161'
  }
  return scriptedModel(context)
}

export const QUESTIONS_V2 = {
  oncall: 'What is 23*7 plus the number of engineers on the oncall rotation?',
  loop: 'Why is checkout slow?',
  sloppy: 'What is 23*7?',
  email: 'Email the size of the oncall rotation to the team lead.',
} as const

/* ---------- 4. the harness ---------- */
export type StopReason = 'answer' | 'max_steps' | 'max_tokens' | 'max_cost' | 'loop_detected' | 'awaiting_approval'
export type ToolStatus = 'ok' | 'invalid_args' | 'denied' | 'unknown_tool'

export interface ModelSpan {
  span: number; step: number; kind: 'model'; name: 'model'
  inputTokens: number; cachedTokens: number; outputTokens: number
  cost: number; costWithoutCache: number; latencyMs: number
  text: string
}
export interface ToolSpan {
  span: number; step: number; kind: 'tool'; name: string
  args: Record<string, unknown>; status: ToolStatus
  resultTokens: number; fullResultTokens: number; cost: 0; latencyMs: number
  result: string
}
export type Span = ModelSpan | ToolSpan

export interface Totals { modelCalls: number; toolCalls: number; inputTokens: number; cachedTokens: number; outputTokens: number; cost: number; costWithoutCache: number; latencyMs: number }

export interface BudgetedRun {
  question: string; answer: string | null; stopped: StopReason; detail: string
  trace: Span[]; totals: Totals; offloaded: Record<string, string>; outbox: { to: string; body: string }[]
  /** Set when stopped === 'awaiting_approval': the call a human must decide on. */
  pending?: { name: string; args: Record<string, unknown> }
}

/** true = approved, false = declined, 'pending' = nobody has decided yet (PORT NOTE: the Python has no pause, it asks a callback). */
export type Approve = (name: string, args: Record<string, unknown>) => boolean | 'pending'
export const denyAll: Approve = () => false

export interface BudgetOptions {
  model?: Model
  maxSteps?: number
  maxTokens?: number | null
  maxCost?: number | null
  maxRepeats?: number
  maxResultTokens?: number | null
  validate?: boolean
  approve?: Approve
  caching?: boolean
  prices?: Prices
  contextBudget?: number
}

class Stop extends Error {
  constructor(public reason: StopReason, public detail: string, public pending?: BudgetedRun['pending']) { super(detail) }
}

export const runBudgeted = (question: string, opts: BudgetOptions = {}): BudgetedRun => {
  const {
    model = scriptedModelV2, maxSteps = 6, maxTokens = null, maxCost = null, maxRepeats = 3, maxResultTokens = null,
    validate = true, approve = denyAll, caching = true, prices = EXAMPLE_PRICES, contextBudget = 3500,
  } = opts
  const trace: Span[] = []
  const offloaded: Record<string, string> = {}
  const outbox: { to: string; body: string }[] = []
  const state = { step: 0, prevContext: '', tokens: 0, cost: 0, calls: [] as string[] }

  // ---- the metered model: count, enforce budgets, then call the real one ----
  const meteredModel: Model = (context) => {
    state.step += 1
    const inputTokens = estimateTokens(context)
    const cached = caching ? Math.floor(sharedPrefix(state.prevContext, context) / 4) : 0
    const floorCost = callCost(inputTokens, cached, 0, prices, caching)
    if (maxTokens !== null && state.tokens + inputTokens > maxTokens)
      throw new Stop('max_tokens', `step ${state.step} needs ${inputTokens} input tokens; ${state.tokens} of ${maxTokens} already used`)
    if (maxCost !== null && state.cost + floorCost > maxCost)
      throw new Stop('max_cost', `step ${state.step} would cost at least $${floorCost.toFixed(6)}; $${state.cost.toFixed(6)} of $${maxCost.toFixed(6)} already spent`)

    const text = model(context)
    const outputTokens = estimateTokens(text)
    const cost = callCost(inputTokens, cached, outputTokens, prices, caching)
    state.prevContext = context
    state.tokens += inputTokens + outputTokens
    state.cost += cost
    trace.push({
      span: trace.length + 1, step: state.step, kind: 'model', name: 'model', inputTokens, cachedTokens: cached, outputTokens,
      cost, costWithoutCache: callCost(inputTokens, 0, outputTokens, prices, false), latencyMs: modelLatencyMs(inputTokens - cached, cached, outputTokens), text,
    })

    // loop detection: the same tool with the same arguments, again and again
    const action = parseAction(text)
    if (action.kind === 'tool') {
      const call = `${action.name} ${action.rawArgs}`
      state.calls.push(call)
      if (maxRepeats > 0 && state.calls.length >= maxRepeats && state.calls.slice(-maxRepeats).every((c) => c === call))
        throw new Stop('loop_detected', `${action.name} called ${maxRepeats} times in a row with identical arguments`)
    }
    return text
  }

  // ---- guarded tools: validate, ask for approval, run, trim, record ----
  const str = (v: unknown) => (typeof v === 'string' ? v : String(v))
  const raw: Registry = {
    ...TOOLS,
    send_email: {
      desc: 'sends an email. args: {"to": str, "body": str}',
      fn: (a) => { outbox.push({ to: str(a.to), body: str(a.body) }); return `sent to ${str(a.to)}` },
    },
    read_log: { desc: 'returns recent log lines of a service. args: {"service": str}', fn: (a) => readLog(str(a.service)) },
  }
  const guarded: Registry = {}
  for (const [name, tool] of Object.entries(raw)) {
    guarded[name] = {
      desc: tool.desc,
      fn: (args) => {
        let status: ToolStatus = 'ok'
        let result: string
        const problem = validate ? validateArgs(name, args) : null
        if (problem) { status = 'invalid_args'; result = problem }
        else if (IRREVERSIBLE.has(name)) {
          const decision = approve(name, args)
          if (decision === 'pending') throw new Stop('awaiting_approval', `${name} is irreversible: waiting for a human`, { name, args })
          if (decision) result = tool.fn(args)
          else { status = 'denied'; result = `ERROR: a human declined to approve ${name}. Do not retry. Tell the user what you would have done.` }
        } else result = tool.fn(args)
        const fullResultTokens = estimateTokens(result)
        if (maxResultTokens !== null && fullResultTokens > maxResultTokens) {
          const ref = `result_${trace.length + 1}.txt`
          offloaded[ref] = result
          result = result.slice(0, maxResultTokens * 4) + `\n[truncated: first ${maxResultTokens} of ${fullResultTokens} tokens. Full result saved as ${ref}. Ask for a narrower slice.]`
        }
        trace.push({
          span: trace.length + 1, step: state.step, kind: 'tool', name, args, status, resultTokens: estimateTokens(result), fullResultTokens,
          cost: 0, latencyMs: status === 'ok' ? TOOL_LATENCY_MS[name] ?? 100 : 0, result,
        })
        return result
      },
    }
  }

  let answer: string | null = null
  let stopped: StopReason = 'answer'
  let detail = ''
  let pending: BudgetedRun['pending']
  try {
    const run = runAgent(question, { model: meteredModel, maxSteps, contextBudget, tools: guarded })
    answer = run.answer
    if (run.stopped === 'budget' || run.answer === BUDGET_EXHAUSTED) { stopped = 'max_steps'; detail = `no final answer after ${maxSteps} model calls` }
  } catch (e) {
    if (!(e instanceof Stop)) throw e
    stopped = e.reason; detail = e.detail; pending = e.pending
  }

  const models = trace.filter((s): s is ModelSpan => s.kind === 'model')
  const sum = (f: (s: ModelSpan) => number) => models.reduce((t, s) => t + f(s), 0)
  const totals: Totals = {
    modelCalls: models.length, toolCalls: trace.length - models.length,
    inputTokens: sum((s) => s.inputTokens), cachedTokens: sum((s) => s.cachedTokens), outputTokens: sum((s) => s.outputTokens),
    cost: sum((s) => s.cost), costWithoutCache: sum((s) => s.costWithoutCache), latencyMs: trace.reduce((t, s) => t + s.latencyMs, 0),
  }
  return { question, answer, stopped, detail, trace, totals, offloaded, outbox, pending }
}
