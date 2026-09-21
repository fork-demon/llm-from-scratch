// A budgeted agent run shown as a trace: one span per model call and per tool call, on a time axis,
// with tokens and cost per span. Budgets stop the run, the validator rejects a malformed call,
// and the approval gate pauses on an irreversible tool. Logic: src/lib/agentBudget.ts.
import { useId, useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { QUESTIONS_V2, runBudgeted, type Span, type StopReason } from '../lib/agentBudget'

type Scenario = keyof typeof QUESTIONS_V2
const SCENARIOS: { id: Scenario; label: string; look: string }[] = [
  { id: 'oncall', label: 'A healthy 3-step run', look: 'Three model calls, two tool calls. Look at the “cached” numbers: from the second call on, almost the whole input is a prefix the provider has already read.' },
  { id: 'sloppy', label: 'The model gets an argument wrong', look: 'The model calls calculator with "expr" instead of "expression". The validator rejects it before anything runs and says exactly what was expected, so the next call is correct. Untick “Validate tool arguments”: the tool now has to cope with the bad call itself and returns a vaguer error. In the Python file, the unguarded loop crashes with a TypeError.' },
  { id: 'email', label: 'An irreversible tool', look: 'send_email is marked irreversible. The run pauses until a human decides. Decline, and the refusal goes back to the model as a tool result it can act on.' },
  { id: 'loop', label: 'A model stuck in a loop', look: 'The model reads the same 600-token log again and again. Loop detection stops it after three identical calls. Switch it off and let the token or cost budget do the job instead. Then switch compaction off and watch the input grow.' },
]
const TOKEN_LIMITS = [null, 500, 1000, 2000, 3000, 5000, 10000]
const COST_LIMITS = [null, 0.001, 0.002, 0.005, 0.01, 0.02]
const STOP_TEXT: Record<StopReason, string> = {
  answer: 'the model gave a final answer', max_steps: 'step budget exhausted', max_tokens: 'token budget exhausted', max_cost: 'cost budget exhausted',
  loop_detected: 'loop detected', awaiting_approval: 'paused: waiting for a human',
}
const usd = (n: number) => `$${n.toFixed(6)}`

function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  const id = useId()
  return <label htmlFor={id} style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}><input id={id} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} /> {label}</label>
}

export function TraceViewer() {
  const [scenario, setScenario] = useState<Scenario>('oncall')
  const [maxSteps, setMaxSteps] = useState(6)
  const [ti, setTi] = useState(0)
  const [ci, setCi] = useState(0)
  const [validate, setValidate] = useState(true)
  const [loopDetect, setLoopDetect] = useState(true)
  const [caching, setCaching] = useState(true)
  const [truncate, setTruncate] = useState(false)
  const [compaction, setCompaction] = useState(true)
  const [decision, setDecision] = useState<boolean | null>(null)
  const [open, setOpen] = useState<number | null>(null)

  const run = useMemo(() => runBudgeted(QUESTIONS_V2[scenario], {
    maxSteps, maxTokens: TOKEN_LIMITS[ti], maxCost: COST_LIMITS[ci], validate, maxRepeats: loopDetect ? 3 : 0, caching,
    maxResultTokens: truncate ? 100 : null, contextBudget: compaction ? 3500 : 1e9, approve: () => decision ?? 'pending',
  }), [scenario, maxSteps, ti, ci, validate, loopDetect, caching, truncate, compaction, decision])

  const totalMs = Math.max(1, run.totals.latencyMs)
  let clock = 0
  const placed = run.trace.map((s) => { const start = clock; clock += s.latencyMs; return { s, start } })
  const selected: Span | null = run.trace.find((s) => s.span === open) ?? null
  const good = run.stopped === 'answer'
  const pick = (id: Scenario) => { setScenario(id); setDecision(null); setOpen(null) }

  return (
    <Lab
      title="Read a trace, set a budget"
      goal={<>The agent loop from <a href="#/lesson/agents">lesson 9.3</a>, wrapped by a harness. Pick a scenario, then move the budgets until the run stops early and read <b>why</b>. Click any span to see what was sent or returned. Token counts are estimates (characters ÷ 4), prices are example numbers, and latencies are placeholders.</>}
    >
      <div className="steps" role="group" aria-label="Scenario">
        {SCENARIOS.map((s) => <button key={s.id} className="step-btn" aria-pressed={scenario === s.id} onClick={() => pick(s.id)}>{s.label}</button>)}
      </div>
      <p className="lab-note" style={{ marginTop: -6 }}><b>Question:</b> “{QUESTIONS_V2[scenario]}” {SCENARIOS.find((s) => s.id === scenario)!.look}</p>

      <div className="controls">
        <Slider label="Max steps (model calls)" value={maxSteps} min={1} max={8} step={1} onChange={setMaxSteps} />
        <Slider label="Max tokens for the whole run" value={ti} min={0} max={TOKEN_LIMITS.length - 1} step={1} onChange={setTi} format={(i) => (TOKEN_LIMITS[i] === null ? 'none' : TOKEN_LIMITS[i]!.toLocaleString('en-US'))} />
        <Slider label="Max cost for the whole run" value={ci} min={0} max={COST_LIMITS.length - 1} step={1} onChange={setCi} format={(i) => (COST_LIMITS[i] === null ? 'none' : `$${COST_LIMITS[i]}`)} />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 22px', marginBottom: 16 }}>
        <Check label="Validate tool arguments" checked={validate} onChange={setValidate} />
        <Check label="Loop detection" checked={loopDetect} onChange={setLoopDetect} />
        <Check label="Prompt caching" checked={caching} onChange={setCaching} />
        <Check label="Truncate tool results to 100 tokens" checked={truncate} onChange={setTruncate} />
        <Check label="Compaction at 3,500 characters" checked={compaction} onChange={setCompaction} />
      </div>

      <div aria-live="polite">
        <div style={{ display: 'grid', gap: 4 }}>
          {placed.map(({ s, start }) => {
            const isModel = s.kind === 'model'
            const bad = s.kind === 'tool' && s.status !== 'ok'
            const left = (start / totalMs) * 100
            const width = Math.max(0.8, (s.latencyMs / totalMs) * 100)
            return (
              <button key={s.span} onClick={() => setOpen(open === s.span ? null : s.span)} aria-pressed={open === s.span}
                style={{ display: 'block', textAlign: 'left', padding: '6px 10px', borderRadius: 7, border: `1px solid ${open === s.span ? 'var(--ink)' : 'var(--rule)'}`, background: 'var(--paper)', color: 'var(--ink)' }}>
                <span style={{ display: 'flex', flexWrap: 'wrap', gap: '2px 12px', alignItems: 'baseline', fontSize: 13 }} className="mono">
                  <span style={{ minWidth: 118, fontWeight: 700, paddingLeft: isModel ? 0 : 14 }}>{s.span}. {isModel ? `model call ${s.step}` : s.name}</span>
                  {s.kind === 'model'
                    ? <><span>{s.inputTokens} in{caching ? ` (${s.cachedTokens} cached)` : ''}</span><span>{s.outputTokens} out</span><span>{usd(s.cost)}</span></>
                    : <><span>result {s.resultTokens} tok{s.resultTokens < s.fullResultTokens ? ` (of ${s.fullResultTokens})` : ''}</span>{bad && <span style={{ color: 'var(--bad)', fontWeight: 700 }}>✗ {s.status.replace('_', ' ')}</span>}</>}
                  <span className="muted">{s.latencyMs} ms</span>
                </span>
                <span style={{ display: 'block', position: 'relative', height: 10, marginTop: 5, background: 'var(--paper-2)', borderRadius: 3 }} aria-hidden>
                  <span style={{ position: 'absolute', left: `${Math.min(left, 99.2)}%`, width: `${width}%`, top: 0, bottom: 0, borderRadius: 3, background: bad ? 'var(--bad)' : isModel ? 'var(--accent)' : 'var(--ink-3)' }} />
                </span>
              </button>
            )
          })}
        </div>

        {run.stopped === 'awaiting_approval' && run.pending && (
          <div className="card" style={{ marginTop: 10, padding: 14, borderColor: 'var(--k)' }} role="alertdialog" aria-label="Approval needed">
            <b>Approval needed.</b> The model wants to call <code>{run.pending.name}</code>, which is marked irreversible:
            <pre className="mono" style={{ fontSize: 13, whiteSpace: 'pre-wrap', margin: '8px 0' }}>{JSON.stringify(run.pending.args, null, 2)}</pre>
            <div className="btn-row">
              <button className="btn small primary" onClick={() => setDecision(true)}>Approve</button>
              <button className="btn small" onClick={() => setDecision(false)}>Decline</button>
            </div>
          </div>
        )}
        {decision !== null && scenario === 'email' && <p className="lab-note" style={{ marginTop: 8 }}>You {decision ? 'approved' : 'declined'}. Outbox: {run.outbox.length ? run.outbox.map((m) => `to ${m.to}: “${m.body}”`).join('; ') : 'empty'}. <button className="btn small" onClick={() => setDecision(null)}>Ask me again</button></p>}

        {selected && (
          <div className="card" style={{ marginTop: 10, padding: 14, fontSize: 14 }}>
            {selected.kind === 'model' ? (
              <>
                <div><b>Span {selected.span}: model call.</b> Input {selected.inputTokens} tokens, of which {selected.cachedTokens} were an unchanged prefix of the previous call. Without caching this call would cost {usd(selected.costWithoutCache)}.</div>
                <div style={{ marginTop: 6 }}>The model wrote:</div>
                <pre className="mono" style={{ fontSize: 13, whiteSpace: 'pre-wrap', margin: '6px 0 0' }}>{selected.text}</pre>
              </>
            ) : (
              <>
                <div><b>Span {selected.span}: tool call</b> <code>{selected.name}</code>, status <b>{selected.status.replace('_', ' ')}</b>. Arguments: <code>{JSON.stringify(selected.args)}</code></div>
                <div style={{ marginTop: 6 }}>Returned to the model ({selected.resultTokens} tokens):</div>
                <pre className="mono" style={{ fontSize: 12.5, whiteSpace: 'pre-wrap', margin: '6px 0 0', maxHeight: 180, overflow: 'auto' }}>{selected.result}</pre>
              </>
            )}
          </div>
        )}

        <div className="readout" style={{ marginTop: 12, borderLeft: `4px solid ${good ? 'var(--good)' : run.stopped === 'awaiting_approval' ? 'var(--k)' : 'var(--bad)'}` }}>
          <span>stopped: <b>{STOP_TEXT[run.stopped]}</b></span>
          {run.detail && <span>{run.detail}</span>}
          <span style={{ flexBasis: '100%' }}>answer: <b>{run.answer ?? 'none'}</b></span>
          <span>{run.totals.modelCalls} model calls, {run.totals.toolCalls} tool calls</span>
          <span>{run.totals.inputTokens.toLocaleString('en-US')} input tokens{caching ? ` (${run.totals.cachedTokens.toLocaleString('en-US')} cached)` : ''}</span>
          <span>{run.totals.outputTokens} output tokens</span>
          <span>cost <b>{usd(run.totals.cost)}</b>{caching ? ` (${usd(run.totals.costWithoutCache)} without caching)` : ''}</span>
          <span>{run.totals.latencyMs.toLocaleString('en-US')} ms</span>
        </div>
      </div>
      <p className="lab-note" style={{ marginTop: 12 }}><b>Honest note:</b> the “model” is a scripted stand-in, as in <code>mini_agent.py</code>, so every run is reproducible and matches <code>agent_budget.py</code> number for number. A real model would not repeat itself this neatly, which is exactly why production traces are sampled and read by people.</p>
    </Lab>
  )
}
