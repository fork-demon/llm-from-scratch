// Pick a symbol or a whole formula and see it as the line of code you already wrote, with shapes.
// Plus a quiz mode: given the formula, pick the code.
import { useState } from 'react'
import { Lab } from '../components/ui'
import { lessonById } from '../data/curriculum'
import { FORMULAS, QUIZ, scoreNotationQuiz, shuffledOptions, SYMBOLS } from '../lib/notation'
import { F } from './FormulaText'

type Mode = 'symbols' | 'formulas' | 'quiz'
const MODES: { id: Mode; label: string }[] = [{ id: 'symbols', label: 'One symbol' }, { id: 'formulas', label: 'A whole formula' }, { id: 'quiz', label: 'Quiz: pick the code' }]

const codeStyle = { display: 'block', fontFamily: 'var(--mono)', fontSize: 13, padding: '9px 12px', background: 'var(--paper-2)', borderRadius: 6, overflowX: 'auto', whiteSpace: 'pre' } as const
const formulaStyle = { fontFamily: 'var(--serif)', fontSize: 21, lineHeight: 1.5, overflowWrap: 'anywhere' } as const

function LessonLink({ id }: { id: string }) {
  const l = lessonById(id)
  return l ? <a href={`#/lesson/${l.id}`}>{l.code} {l.title}</a> : null
}

export function NotationDecoder() {
  const [mode, setMode] = useState<Mode>('symbols')
  const [sym, setSym] = useState(SYMBOLS[0].id)
  const [formula, setFormula] = useState(FORMULAS[0].id)
  const [step, setStep] = useState(0)
  const [picked, setPicked] = useState<Record<string, number>>({})
  const [checked, setChecked] = useState(false)

  const s = SYMBOLS.find((x) => x.id === sym)!
  const f = FORMULAS.find((x) => x.id === formula)!

  return (
    <Lab
      title="Notation decoder"
      goal={<>Every symbol here stands for a line of code you have already written. Pick one and read the code before the explanation. In “A whole formula”, step through a formula from the inside out. Then take the quiz.</>}
    >
      <div className="steps" role="group" aria-label="Mode">
        {MODES.map((m) => <button key={m.id} className="step-btn" aria-pressed={mode === m.id} onClick={() => setMode(m.id)}>{m.label}</button>)}
      </div>

      {mode === 'symbols' && (
        <>
          <div role="group" aria-label="Symbols" style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 16 }}>
            {SYMBOLS.map((x) => (
              <button key={x.id} aria-pressed={x.id === sym} onClick={() => setSym(x.id)} className="chip" style={{ fontFamily: 'var(--serif)', fontSize: 15, padding: '4px 10px', cursor: 'pointer', borderColor: x.id === sym ? 'var(--accent)' : undefined, background: x.id === sym ? 'var(--paper-2)' : undefined, color: 'var(--ink)' }}>
                <F>{x.symbol}</F>
              </button>
            ))}
          </div>
          <div aria-live="polite">
            <div style={formulaStyle}><F>{s.symbol}</F></div>
            <p className="muted" style={{ margin: '2px 0 12px', fontSize: 14.5 }}>Say it: {s.say}</p>
            <code style={codeStyle}>{s.code}</code>
            <p className="mono" style={{ fontSize: 12.5, color: 'var(--ink-2)', margin: '6px 0 12px' }}>shapes: <F>{s.shapes}</F></p>
            <p style={{ fontSize: 15.5 }}><F>{s.meaning}</F></p>
            <p className="lab-note">You wrote this in <LessonLink id={s.lesson} />.</p>
          </div>
        </>
      )}

      {mode === 'formulas' && (
        <>
          <div className="steps" role="group" aria-label="Formulas" style={{ marginBottom: 12 }}>
            {FORMULAS.map((x) => <button key={x.id} className="step-btn" aria-pressed={x.id === formula} onClick={() => { setFormula(x.id); setStep(0) }}>{x.name}</button>)}
          </div>
          <div style={formulaStyle}><F>{f.formula}</F></div>
          <p className="muted" style={{ margin: '2px 0 12px', fontSize: 14 }}>{f.from}</p>
          <div className="table-scroll" aria-live="polite">
            <table className="plain" style={{ fontSize: 14 }}>
              <thead><tr><th>piece of the formula</th><th>the line you wrote</th><th>shape</th></tr></thead>
              <tbody>
                {f.steps.map((st, i) => (
                  <tr key={i} style={{ opacity: i <= step ? 1 : 0.28 }}>
                    <td style={{ fontFamily: 'var(--serif)', fontSize: 16, whiteSpace: 'nowrap' }}>{i <= step ? <F>{st.piece}</F> : '…'}</td>
                    <td><code style={{ fontSize: 12.5, whiteSpace: 'pre' }}>{i <= step ? st.code : '…'}</code></td>
                    <td className="mono" style={{ fontSize: 12.5 }}>{i <= step ? st.shape : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="btn-row" style={{ margin: '10px 0' }}>
            <button className="btn small primary" disabled={step >= f.steps.length - 1} onClick={() => setStep(step + 1)}>Next piece ({Math.min(step + 2, f.steps.length)} of {f.steps.length})</button>
            <button className="btn small" disabled={step === 0} onClick={() => setStep(0)}>Start again</button>
          </div>
          {step >= f.steps.length - 1 && <p className="lab-note"><F>{f.note}</F> Taught in <LessonLink id={f.lesson} />.</p>}
        </>
      )}

      {mode === 'quiz' && (
        <>
          {QUIZ.map((q, qi) => {
            const sh = shuffledOptions(q)
            return (
              <fieldset className="quiz-q" key={q.id}>
                <legend>{qi + 1}. Which line computes <span style={{ fontFamily: 'var(--serif)', fontSize: 17 }}><F>{q.formula}</F></span> ?</legend>
                {sh.options.map((o, oi) => {
                  const isPicked = picked[q.id] === oi
                  const cls = checked ? (oi === sh.answer ? ' right' : isPicked ? ' wrong' : '') : ''
                  return (
                    <label key={o} className={`quiz-opt${cls}`}>
                      <input type="radio" name={`nd-${q.id}`} checked={isPicked} disabled={checked} onChange={() => setPicked({ ...picked, [q.id]: oi })} />
                      <span><code style={{ fontSize: 12.5, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{o}</code></span>
                    </label>
                  )
                })}
                {checked && <div className="quiz-explain">{picked[q.id] === sh.answer ? '✓ ' : '✗ '}<F>{q.explain}</F></div>}
              </fieldset>
            )
          })}
          <div className="btn-row">
            {!checked
              ? <button className="btn primary" disabled={QUIZ.some((q) => picked[q.id] === undefined)} onClick={() => setChecked(true)}>Check my answers</button>
              : <><strong role="status">{scoreNotationQuiz(picked)} / {QUIZ.length}</strong><button className="btn small" onClick={() => { setPicked({}); setChecked(false) }}>Try again</button></>}
          </div>
        </>
      )}
    </Lab>
  )
}
