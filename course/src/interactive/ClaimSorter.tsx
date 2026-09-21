// A tiny checker: sort statements into Established / Mental model / Active research.
import { useState } from 'react'
import { LEVEL_LABEL, markClaims, scoreClaims, type Claim, type ClaimLevel } from '../lib/claims'
import { completeExercise, updateProgress, useProgress } from '../lib/progress'

const LEVELS: ClaimLevel[] = ['established', 'model', 'research']

export function ClaimSorter({ id, claims }: { id: string; claims: Claim[] }) {
  const [chosen, setChosen] = useState<(ClaimLevel | null)[]>(() => claims.map(() => null))
  const [checked, setChecked] = useState(false)
  const progress = useProgress()
  const marks = markClaims(chosen, claims)
  const score = scoreClaims(chosen, claims)
  const all = chosen.every((c) => c !== null)

  const check = () => {
    setChecked(true)
    if (score === claims.length) updateProgress((p) => completeExercise(p, id))
  }

  return (
    <div className="exercise" id={`ex-${id}`}>
      <div className="exercise-head">
        <span className="ex-type">experiment</span>
        <span className="ex-title">Sort the claims</span>
        {progress.exercises[id] && <span className="ex-done">✓ done</span>}
      </div>
      <div className="exercise-body">
        <p>For each statement decide: is it an <b>established</b> mechanism, a <b>simplified mental model</b> (useful, not literally true), or <b>active research</b> (debated or unknown)? Commit to all of them, then check.</p>
        {claims.map((c, i) => (
          <fieldset key={i} style={{ border: 0, padding: 0, margin: '0 0 16px' }}>
            <legend style={{ padding: 0, marginBottom: 6 }}>{i + 1}. “{c.text}”</legend>
            <div className="steps" style={{ marginBottom: 4 }}>
              {LEVELS.map((l) => (
                <button key={l} type="button" className="step-btn" aria-pressed={chosen[i] === l} disabled={checked} onClick={() => setChosen(chosen.map((v, k) => (k === i ? l : v)))}>
                  {LEVEL_LABEL[l]}
                </button>
              ))}
            </div>
            {checked && (
              <div className="quiz-explain">
                {marks[i] ? '✓ ' : `✗ Best label: ${LEVEL_LABEL[c.level]}. `}{c.why}
              </div>
            )}
          </fieldset>
        ))}
        <div className="btn-row">
          {!checked ? (
            <button className="btn small primary" disabled={!all} onClick={check}>Check my sorting</button>
          ) : (
            <>
              <strong role="status">{score} / {claims.length}</strong>
              <button className="btn small" onClick={() => { setChecked(false); setChosen(claims.map(() => null)) }}>Try again</button>
            </>
          )}
          {!all && !checked && <span className="muted" style={{ fontSize: 13.5 }}>Label every statement first.</span>}
        </div>
      </div>
    </div>
  )
}
