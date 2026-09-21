// The ReAct loop of mini_agent.py, one turn at a time. The run itself is a pure
// function (lib/agent.ts); this component only decides how much of it to reveal.
import { useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { Code } from '../components/Code'
import { QUESTIONS, TOOLS, runAgent, type Registry } from '../lib/agent'

type QKey = keyof typeof QUESTIONS
const Q_LABEL: Record<QKey, string> = { oncall: 'needs two tools (3 steps)', direct: 'no tool needed (1 step)', injection: 'prompt injection via a tool result' }

// the five things that happen in one loop iteration, and who does each
const PHASES = [
  { who: 'your code', label: 'builds the context and sends it' },
  { who: 'model', label: 'emits text' },
  { who: 'your code', label: 'parses the text' },
  { who: 'your code', label: 'runs the tool' },
  { who: 'your code', label: 'appends RESULT to the context' },
] as const

const pre = { whiteSpace: 'pre-wrap', fontSize: 13, padding: 12, margin: '6px 0 0', background: 'var(--paper-2)', border: '1px solid var(--rule)', borderRadius: 8, maxHeight: 300, overflow: 'auto' } as const
const Who = ({ who }: { who: string }) => <span className={`chip${who === 'model' ? ' acc' : ''}`} style={{ marginRight: 8 }}>{who === 'model' ? 'THE MODEL' : 'YOUR CODE'}</span>

export function AgentPlayground() {
  const [qKey, setQKey] = useState<QKey>('oncall')
  const [maxSteps, setMaxSteps] = useState(6)
  const [enabled, setEnabled] = useState<Record<string, boolean>>({ calculator: true, search_docs: true })
  const [sanitize, setSanitize] = useState(false)
  const [cursor, setCursor] = useState({ step: 0, phase: 0 })

  const tools: Registry = useMemo(() => Object.fromEntries(Object.entries(TOOLS).filter(([n]) => enabled[n])), [enabled])
  const run = useMemo(() => runAgent(QUESTIONS[qKey], { maxSteps, tools, sanitize }), [qKey, maxSteps, tools, sanitize])
  const restart = () => setCursor({ step: 0, phase: 0 })
  const change = (f: () => void) => { f(); restart() }

  const si = Math.min(cursor.step, run.steps.length - 1)
  const s = run.steps[si]
  const lastPhase = s.action.kind === 'answer' ? 2 : 4
  const phase = Math.min(cursor.phase, lastPhase)
  const atEnd = si === run.steps.length - 1 && phase === lastPhase
  const next = () => setCursor(phase < lastPhase ? { step: si, phase: phase + 1 } : { step: si + 1, phase: 0 })
  const prevCtx = si > 0 ? run.steps[si - 1].context : ''
  const grew = si > 0 && s.context.startsWith(prevCtx)

  return (
    <Lab
      title="An agent, one turn at a time"
      goal={<>Press <b>Next</b> and watch who does what. The model only ever receives text and emits text. Everything that <em>happens</em> is done by ordinary code. Then break it: set max steps to 2, switch a tool off, or run the injection question.</>}
    >
      <div className="steps" role="group" aria-label="Question">
        {(Object.keys(QUESTIONS) as QKey[]).map((k) => <button key={k} className="step-btn" aria-pressed={qKey === k} onClick={() => change(() => setQKey(k))}>{Q_LABEL[k]}</button>)}
      </div>
      <div className="readout" style={{ marginBottom: 12 }}><span>USER QUESTION: <b>{QUESTIONS[qKey]}</b></span></div>

      <div className="controls">
        <Slider label="max_steps (the step budget)" value={maxSteps} min={1} max={6} step={1} onChange={(v) => change(() => setMaxSteps(v))} />
        <fieldset style={{ border: 0, padding: 0, margin: 0, fontSize: 14.5 }}>
          <legend style={{ fontSize: 14, color: 'var(--ink-2)' }}>Tool registry</legend>
          {Object.keys(TOOLS).map((n) => (
            <label key={n} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input type="checkbox" checked={enabled[n]} onChange={(e) => change(() => setEnabled({ ...enabled, [n]: e.target.checked }))} /> <span className="mono">{n}</span>
            </label>
          ))}
        </fieldset>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}>
          <input type="checkbox" checked={sanitize} onChange={(e) => change(() => setSanitize(e.target.checked))} /> Sanitize tool results before they enter the context
        </label>
      </div>

      {/* timeline of finished iterations */}
      <ol style={{ listStyle: 'none', padding: 0, margin: '0 0 12px', display: 'grid', gap: 4 }} aria-label="Loop iterations so far">
        {run.steps.slice(0, si).map((t) => (
          <li key={t.step} className="mono" style={{ fontSize: 13, color: 'var(--ink-2)' }}>
            iteration {t.step}: model asked for <b>{t.action.kind === 'tool' ? t.action.name : 'nothing'}</b> → RESULT: {t.result}
          </li>
        ))}
      </ol>

      <div className="card" aria-live="polite">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center', marginBottom: 10 }}>
          <b style={{ marginRight: 6 }}>Iteration {s.step} of at most {maxSteps}</b>
          {PHASES.slice(0, lastPhase + 1).map((p, i) => (
            <button key={i} className="step-btn" style={{ fontSize: 12.5, marginBottom: 0 }} aria-pressed={i === phase} onClick={() => setCursor({ step: si, phase: i })}>{i + 1}</button>
          ))}
        </div>
        <p style={{ margin: 0 }}><Who who={PHASES[phase].who} /><b>{PHASES[phase].label}</b></p>

        {phase === 0 && (
          <>
            <p className="lab-note" style={{ marginTop: 8 }}>This is the <b>entire input</b> of this model call: {s.context.length} characters. The model remembers nothing from the previous call. {s.summarized && <b>The scratchpad went over budget, so older steps were collapsed into a summary line. </b>}{grew ? <>The <span className="acc">highlighted</span> part is new since the last iteration: your code put it there.</> : si === 0 ? <>System prompt (built from the tool registry) + the question. Nothing else yet.</> : null}</p>
            <pre className="mono" tabIndex={0} style={pre}>{grew ? <>{prevCtx}<span className="acc" style={{ fontWeight: 600 }}>{s.context.slice(prevCtx.length)}</span></> : s.context}</pre>
          </>
        )}
        {phase === 1 && (
          <>
            <p className="lab-note" style={{ marginTop: 8 }}>The model’s raw output. It is text and nothing but text. No function has been called. Nothing has happened in the world.</p>
            <pre className="mono" tabIndex={0} style={pre}>{s.modelText}</pre>
          </>
        )}
        {phase === 2 && (
          <>
            <p className="lab-note" style={{ marginTop: 8 }}><code>parse_action()</code> runs two regular expressions over that text: is there an <code>ANSWER:</code>? If not, is there a <code>TOOL:</code> + <code>ARGS:</code> pair?</p>
            {s.action.kind === 'tool' ? (
              <div className="readout"><span>kind: <b>tool</b></span><span>name: <b>{s.action.name}</b></span><span>args (JSON): <b>{s.action.rawArgs}</b></span></div>
            ) : (
              <>
                <div className="readout" style={{ borderColor: 'var(--good)' }}><span>kind: <b>answer</b>{s.action.malformed && ' (no recognised format: treated as final)'}</span><span>FINAL ANSWER: <b>{s.action.answer}</b></span></div>
                <p className="lab-note" style={{ marginTop: 8 }}><b>Stopping condition met:</b> the text contained <code>ANSWER:</code>, so the loop returns. {run.answer === 'BANANA' && <>The user asked about vacation and got “BANANA”. A sentence inside a <em>document</em> steered the run. Tick “Sanitize tool results” and run it again.</>}</p>
              </>
            )}
          </>
        )}
        {phase === 3 && s.action.kind === 'tool' && (
          <>
            <p className="lab-note" style={{ marginTop: 8 }}>Your code looks the name up in the registry and calls an ordinary function. This is the only place anything actually executes.</p>
            <pre className="mono" tabIndex={0} style={pre}>{s.action.name in tools ? `TOOLS["${s.action.name}"]["fn"](**${s.action.rawArgs})\n→ ${JSON.stringify(s.rawResult)}` : `"${s.action.name}" in TOOLS  → False\n→ ${JSON.stringify(s.rawResult)}`}</pre>
            {!(s.action.name in tools) && <p className="lab-note">The model asked for a tool that is not registered. Real models do this too (a hallucinated tool name). The loop does not crash: the error becomes the RESULT, so the model can read it and react.</p>}
          </>
        )}
        {phase === 4 && (
          <>
            <p className="lab-note" style={{ marginTop: 8 }}>The model’s text plus a <code>RESULT:</code> line is appended to the scratchpad. On the next iteration it is part of the context. That is the agent’s whole “memory”.</p>
            {s.removed && s.removed.length > 0 && <p className="lab-note"><b>Sanitizer:</b> removed “{s.removed.join(' ')}” because it looks like an instruction, not data. A pattern filter like this is easy to evade by rephrasing. It lowers the odds. It is not a security boundary.</p>}
            <pre className="mono" tabIndex={0} style={pre}>{s.modelText}{'\n'}<span className="acc" style={{ fontWeight: 600 }}>RESULT: {s.result}</span></pre>
            {si === run.steps.length - 1 && run.stopped === 'budget' && (
              <div className="readout" style={{ borderColor: 'var(--bad)', marginTop: 8 }}><span><b>Stopping condition met: step budget exhausted.</b> The loop returns “{run.answer}”. The task needed more iterations than max_steps allowed.</span></div>
            )}
          </>
        )}
      </div>

      <div className="btn-row" style={{ marginTop: 12 }}>
        <button className="btn small primary" disabled={atEnd} onClick={next}>Next</button>
        <button className="btn small" disabled={atEnd} onClick={() => setCursor({ step: run.steps.length - 1, phase: 4 })}>Run to the end</button>
        <button className="btn small" onClick={restart}>Restart</button>
        <span className="muted" style={{ fontSize: 13.5 }}>model calls so far: {si + (phase >= 1 ? 1 : 0)}</span>
      </div>

      <p className="lab-note" style={{ marginTop: 14 }}><b>The model here is a scripted stand-in</b>, exactly as in the Python file: a few <code>if</code> statements that follow the format the way a well-behaved LLM would (and that obey injected text the way a naive one might). That keeps the run offline and repeatable, so you can study the <em>loop</em>. The loop, registry, parser and memory code are the real thing. To use a real model, swap in these lines:</p>
      <Code source="phase5-agents/mini_agent.py" title="the 3-line swap to a real LLM">{`
def llm_model(context):
    r = client.messages.create(model=..., max_tokens=400,
            messages=[{"role": "user", "content": context}])
    return r.content[0].text

run_agent(question, model=llm_model)
`}</Code>
    </Lab>
  )
}
