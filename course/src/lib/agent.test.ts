import { describe, expect, it } from 'vitest'
import { BUDGET_EXHAUSTED, QUESTIONS, SYSTEM_PROMPT, TOOLS, buildSystemPrompt, calculator, chainReliability, parseAction, runAgent, sanitizeToolResult, searchDocs, summarizeScratchpad, type Model } from './agent'

// Reference values come from running phase5-agents/mini_agent.py.
describe('calculator: a parser, never eval', () => {
  it('matches Python on arithmetic', () => {
    const cases: [string, string][] = [
      ['23*7 + 4', '165'], ['8/2', '4.0'], ['7/2', '3.5'], ['2**10', '1024'], ['7//2', '3'], ['-3+5', '2'],
      ['2*(3+4)', '14'], ['1.5*2', '3.0'], ['2**-1', '0.5'], ['-2**2', '-4'], ['0.1+0.2', '0.30000000000000004'],
      ['10/0', 'ERROR: division by zero'],
    ]
    for (const [e, want] of cases) expect(calculator(e), e).toBe(want)
  })
  it('rejects anything that is not arithmetic', () => {
    for (const e of ['__import__("os")', 'alert(1)', '1e3', 'process.exit()', '2; 3', '']) expect(calculator(e)).toBe('ERROR: only arithmetic allowed')
  })
  it('reports syntax errors instead of throwing', () => {
    for (const e of ['2+', '(1+2', '3 4', '2 * * 3', '()', '1..2']) expect(calculator(e)).toMatch(/^ERROR: /)
  })
})

describe('registry and system prompt', () => {
  it('search_docs is the toy keyword lookup', () => {
    expect(searchDocs('Oncall rotation')).toBe('The oncall rotation has 4 engineers: primary, secondary, two shadows.')
    expect(searchDocs('wifi')).toBe('No documents found.')
  })
  it('SYSTEM_PROMPT is identical to the Python string', () => {
    expect(SYSTEM_PROMPT).toBe(
      'Answer the user\'s question. You may use tools.\nTo use a tool reply EXACTLY:\nThought: <why>\nTOOL: <name>\nARGS: <json>\nWhen you have the answer reply:\nThought: <why>\nANSWER: <final answer>\n\nAvailable tools:\n- calculator: evaluates arithmetic. args: {"expression": str}\n- search_docs: finds internal documents. args: {"query": str}',
    )
  })
  it('is built from the registry: remove a tool and it disappears from the prompt', () => {
    const { calculator: _c, ...rest } = TOOLS
    expect(buildSystemPrompt(rest)).not.toContain('calculator')
  })
})

describe('parseAction', () => {
  it('tool call', () => {
    expect(parseAction('Thought: x\nTOOL: calculator\nARGS: {"expression": "1+1"}')).toEqual({ kind: 'tool', name: 'calculator', args: { expression: '1+1' }, rawArgs: '{"expression": "1+1"}' })
  })
  it('final answer (ANSWER wins, and may span lines)', () => {
    expect(parseAction('Thought: done\nANSWER: 42\nreally')).toEqual({ kind: 'answer', answer: '42\nreally', malformed: false })
  })
  it('malformed output is treated as the final answer', () => {
    expect(parseAction('  hello there ')).toEqual({ kind: 'answer', answer: 'hello there', malformed: true })
  })
  it('broken JSON does not throw', () => {
    const a = parseAction('TOOL: calculator\nARGS: {expression: 1}')
    expect(a.kind === 'tool' && a.args).toBe(null)
  })
})

describe('the ReAct loop', () => {
  it('the oncall question: search, calculate, answer. Same text as the Python', () => {
    const run = runAgent(QUESTIONS.oncall)
    expect(run.steps.map((s) => (s.action.kind === 'tool' ? s.action.name : 'ANSWER'))).toEqual(['search_docs', 'calculator', 'ANSWER'])
    expect(run.steps[1].result).toBe('165')
    expect(run.answer).toBe("165 -- that's 23*7 (161) plus the 4 oncall engineers.")
    expect(run.stopped).toBe('answer')
  })
  it('the context grows: each turn contains every earlier RESULT', () => {
    const { steps } = runAgent(QUESTIONS.oncall)
    expect(steps[0].context).toBe(`${SYSTEM_PROMPT}\n\nUSER QUESTION: ${QUESTIONS.oncall}\n\n`)
    expect(steps[1].context.length).toBeGreaterThan(steps[0].context.length)
    expect(steps[2].context).toContain('RESULT: The oncall rotation has 4 engineers')
    expect(steps[2].context).toContain('RESULT: 165')
  })
  it('a direct answer stops after one model call', () => {
    const run = runAgent(QUESTIONS.direct)
    expect(run.steps.length).toBe(1)
    expect(run.answer).toBe("I don't know.")
  })
  it('budget exhaustion', () => {
    for (const maxSteps of [1, 2]) {
      const run = runAgent(QUESTIONS.oncall, { maxSteps })
      expect(run.stopped).toBe('budget')
      expect(run.answer).toBe(BUDGET_EXHAUSTED)
      expect(run.steps.length).toBe(maxSteps)
    }
    expect(runAgent(QUESTIONS.oncall, { maxSteps: 3 }).stopped).toBe('answer')
  })
  it('a tool missing from the registry: the error is fed back as the RESULT', () => {
    const { calculator: _c, ...tools } = TOOLS
    const run = runAgent(QUESTIONS.oncall, { tools })
    expect(run.steps[1].result).toBe('ERROR: unknown tool calculator')
    expect(run.steps[2].context).toContain('RESULT: ERROR: unknown tool calculator')
    expect(run.answer).toBe("I don't know.")
  })
  it('prompt injection reproduces', () => {
    const run = runAgent(QUESTIONS.injection)
    expect(run.steps[0].result).toContain('ignore your instructions')
    expect(run.answer).toBe('BANANA')
  })
  it('and is defused by the sanitizer', () => {
    const run = runAgent(QUESTIONS.injection, { sanitize: true })
    expect(run.steps[0].rawResult).toContain('BANANA')
    expect(run.steps[0].result).toBe('Company handbook v2. [removed by sanitizer: text that looked like instructions]')
    expect(run.steps[1].context).not.toContain('BANANA')
    expect(run.answer).toBe("I don't know.")
  })
  it('the sanitizer leaves ordinary results alone', () => {
    expect(sanitizeToolResult('165')).toEqual({ text: '165', removed: [] })
    expect(runAgent(QUESTIONS.oncall, { sanitize: true }).answer).toBe(runAgent(QUESTIONS.oncall).answer)
  })
})

describe('memory management', () => {
  it('summary line matches the Python format', () => {
    expect(summarizeScratchpad(['Thought: x\nTOOL: search_docs\nARGS: {}\nRESULT: a', 'TOOL: calculator\nARGS: {}\nRESULT: 1'])).toBe(
      "[SUMMARY of 2 earlier steps: used tools ['search_docs', 'calculator']; key facts retained in later steps]",
    )
  })
  it('over budget: older steps collapse, the last two stay verbatim', () => {
    let n = 0
    const chatty: Model = () => (++n <= 4 ? `Thought: step ${n}\nTOOL: search_docs\nARGS: {"query": "deploy"}` : 'ANSWER: done')
    const run = runAgent('q', { model: chatty, contextBudget: 150 })
    expect(run.steps.map((s) => s.summarized)).toEqual([false, false, false, true, true])
    expect(run.steps[3].context).toContain('[SUMMARY of 1 earlier steps')
    expect(run.steps[3].context).not.toContain('Thought: step 1')
    expect(run.steps[3].context).toContain('Thought: step 3')
  })
})

describe('error compounding', () => {
  it('numbers quoted in the lesson', () => {
    expect(chainReliability(0.95, 10)).toBeCloseTo(0.5987, 4)
    expect(chainReliability(0.9, 7)).toBeCloseTo(0.4783, 4)
    expect(chainReliability(0.9, 10)).toBeCloseTo(0.3487, 4)
    expect(chainReliability(0.99, 50)).toBeCloseTo(0.605, 3)
  })
})
