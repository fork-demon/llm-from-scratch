// Matching exercise: real tensor names and shapes on one side, course concepts on the other.
import { useState } from 'react'
import { Lab } from '../components/ui'
import { grade, itemsFor, optionsFor, shapeText, type Round } from '../lib/tensorNames'
import { lessonById } from '../data/curriculum'

const ROUNDS: { id: Round; label: string; note: string }[] = [
  { id: 'gpt2', label: 'Round 1: GPT-2 small', note: 'Names and shapes exactly as inspect_hf_model.py prints them. D = 768, 12 heads, vocabulary 50,257, context 1,024.' },
  { id: 'llama', label: 'Round 2: Llama 3 8B', note: 'Names and shapes from the published checkpoint. D = 4,096, 32 query heads, 8 key/value heads, head size 128, MLP width 14,336, vocabulary 128,256.' },
]

export function TensorNameMatcher() {
  const [round, setRound] = useState<Round>('gpt2')
  const [answers, setAnswers] = useState<Record<Round, Record<string, string>>>({ gpt2: {}, llama: {} })
  const [checked, setChecked] = useState<Record<Round, boolean>>({ gpt2: false, llama: false })
  const items = itemsFor(round)
  const options = optionsFor(round)
  const mine = answers[round]
  const result = grade(round, mine)
  const isChecked = checked[round]
  const all = items.every((t) => mine[t.id])

  return (
    <Lab
      title="Name that tensor"
      goal={<>For each real tensor, pick what it is. Use the <b>name</b> and the <b>shape</b> together: 2304 = 3 × 768 is a clue, and so is 1024 = 8 × 128. Two tensors in round 2 have identical shapes, so only the name can separate them.</>}
    >
      <div className="steps" role="group" aria-label="Round">
        {ROUNDS.map((r) => <button key={r.id} className="step-btn" aria-pressed={round === r.id} onClick={() => setRound(r.id)}>{r.label}</button>)}
      </div>
      <p className="lab-note" style={{ marginTop: -6 }}>{ROUNDS.find((r) => r.id === round)!.note}</p>

      <div style={{ display: 'grid', gap: 10 }}>
        {items.map((t) => {
          const ok = isChecked && result.perItem[t.id]
          const wrong = isChecked && !result.perItem[t.id]
          const l = lessonById(t.lesson)
          return (
            <div key={t.id} className="card" style={{ margin: 0, padding: '10px 14px', borderColor: ok ? 'var(--good)' : wrong ? 'var(--bad)' : undefined }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 14px', alignItems: 'center' }}>
                <code style={{ fontSize: 13, overflowWrap: 'anywhere' }}>{t.name}</code>
                <span className="chip">{shapeText(t.shape)}</span>
                <label className="sr-only" htmlFor={`tnm-${t.id}`}>What is {t.name}?</label>
                <select
                  id={`tnm-${t.id}`}
                  className="input"
                  style={{ flex: '1 1 260px', width: 'auto', fontSize: 14, padding: '6px 8px' }}
                  value={mine[t.id] ?? ''}
                  onChange={(e) => { setAnswers({ ...answers, [round]: { ...mine, [t.id]: e.target.value } }); setChecked({ ...checked, [round]: false }) }}
                >
                  <option value="">choose what this is…</option>
                  {options.map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
                {isChecked && <b style={{ color: ok ? 'var(--good)' : 'var(--bad)', fontSize: 14 }}>{ok ? '✓ right' : '✗ not this one'}</b>}
              </div>
              {ok && <p style={{ margin: '8px 0 0', fontSize: 14.5, color: 'var(--ink-2)' }}>{t.explain} {l && <a href={`#/lesson/${l.id}`}>Lesson {l.code}</a>}</p>}
            </div>
          )
        })}
      </div>

      <div className="btn-row" style={{ marginTop: 14 }}>
        <button className="btn primary" disabled={!all} onClick={() => setChecked({ ...checked, [round]: true })}>Check my matches</button>
        <button className="btn small" onClick={() => { setAnswers({ ...answers, [round]: {} }); setChecked({ ...checked, [round]: false }) }}>Clear</button>
        <span role="status" aria-live="polite" className="mono" style={{ fontSize: 13.5 }}>
          {isChecked ? `${result.correct} of ${result.total} right.${result.correct < result.total ? ' Explanations appear for the right ones. Fix the rest and check again.' : round === 'gpt2' ? ' Now try round 2.' : ' You can orient yourself in a Llama-style checkpoint.'}` : !all ? 'Answer every row to check.' : ''}
        </span>
      </div>
    </Lab>
  )
}
