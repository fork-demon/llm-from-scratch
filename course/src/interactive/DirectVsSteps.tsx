// The working-memory point, made concrete: the same puzzle answered in one token vs written out.
// Pure bookkeeping (what is on the page, how many forward passes ran). No model is being run here.
import { useState } from 'react'
import { Lab } from '../components/ui'
import { buildTrace, roughTokens, type Op } from '../lib/testTime'

const OPS: Op[] = [{ kind: 'times', by: 3 }, { kind: 'plus', by: 28 }, { kind: 'times', by: 2 }, { kind: 'minus', by: 45 }]
const WORDS = ['multiply by 3', 'add 28', 'double it', 'subtract 45']

export function DirectVsSteps() {
  const [start, setStart] = useState(17)
  const [mode, setMode] = useState<'direct' | 'steps'>('direct')
  const [shown, setShown] = useState(0)

  const trace = buildTrace(start, OPS)
  const final = trace[trace.length - 1].after
  const prompt = `Start with ${start}. ${WORDS.map((w) => w[0].toUpperCase() + w.slice(1)).join('. ')}. What do you get?`
  const lines = mode === 'direct' ? [`${final}`] : [...trace.map((s) => s.text), `Answer: ${final}`]
  const visible = lines.slice(0, shown)
  const passes = visible.reduce((s, l) => s + roughTokens(l), 0)
  const done = shown >= lines.length
  const reset = (m: 'direct' | 'steps') => { setMode(m); setShown(0) }

  return (
    <Lab
      title="One forward pass, or many?"
      goal={<>The same puzzle, answered two ways. Press <b>Generate next line</b> and keep an eye on two things: how many forward passes have run, and what the model can <em>read</em> when it produces the next token.</>}
    >
      <div className="controls">
        <div className="control">
          <label htmlFor="dvs-start"><span>starting number</span></label>
          <input id="dvs-start" className="input" type="number" min={2} max={99} value={start} onChange={(e) => { setStart(Math.min(99, Math.max(2, Math.round(Number(e.target.value)) || 2))); setShown(0) }} style={{ maxWidth: 110 }} />
        </div>
      </div>
      <div className="steps" role="group" aria-label="How the model answers">
        <button className="step-btn" aria-pressed={mode === 'direct'} onClick={() => reset('direct')}>Answer directly</button>
        <button className="step-btn" aria-pressed={mode === 'steps'} onClick={() => reset('steps')}>Write the steps first</button>
      </div>

      <div className="card" style={{ fontFamily: 'var(--mono)', fontSize: 14.5 }} aria-live="polite">
        <div className="muted">{prompt}</div>
        {visible.map((l, i) => <div key={i} style={{ marginTop: 4 }}><b>{l}</b></div>)}
        {!done && <div className="muted" style={{ marginTop: 4 }}>▍</div>}
      </div>

      <div className="readout" style={{ marginTop: 10 }} aria-live="polite">
        <span>forward passes so far: <b>{passes}</b></span>
        <span>intermediate results on the page: <b>{mode === 'steps' ? Math.min(shown, trace.length) : 0}</b> of {trace.length}</span>
      </div>

      <p className="lab-note">
        {mode === 'direct'
          ? done
            ? <>The answer token came out of <b>one</b> trip through the layers. All four operations, and the three intermediate numbers ({trace.slice(0, 3).map((s) => s.after).join(', ')}), had to be worked out and carried inside that single pass. None of them was ever written anywhere.</>
            : <>The very next token has to <em>be</em> the answer. Nothing has been worked out yet.</>
          : done
            ? <>Each line needed only <b>one</b> easy operation, and its input was sitting on the page, readable through attention. About {passes} forward passes instead of 1: more compute, spent in small, reliable steps.</>
            : shown === 0
              ? <>The next line only has to do the first operation.</>
              : <>To write the next line the model does not need to remember {trace[shown - 1].after}: it can <em>read</em> it, because its own output is now part of the context.</>}
      </p>
      <p className="lab-note">Token counts here are rough (one per number or symbol). This is bookkeeping, not a running model: whether a real model gets the direct answer right depends on the model and the numbers.</p>

      <div className="btn-row">
        <button className="btn small primary" disabled={done} onClick={() => setShown(shown + 1)}>Generate next line</button>
        <button className="btn small" disabled={shown === 0} onClick={() => setShown(0)}>Start over</button>
      </div>
    </Lab>
  )
}
