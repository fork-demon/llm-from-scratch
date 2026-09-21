// "Do I need this part?": a few questions at the top of a part's first lesson.
// Getting them all right suggests skimming; missing one names the lesson to read.
import { useState } from 'react'
import { lessonById } from '../data/curriculum'
import { completeLesson, updateProgress, useProgress } from '../lib/progress'

export interface DiagnosticQ { q: string; options: string[]; answer: number; lesson: string }

export function Diagnostic({ id, part, questions }: { id: string; part: string; questions: DiagnosticQ[] }) {
  const [open, setOpen] = useState(false)
  const [chosen, setChosen] = useState<(number | null)[]>(() => questions.map(() => null))
  const [checked, setChecked] = useState(false)
  const progress = useProgress()

  const score = questions.filter((q, i) => chosen[i] === q.answer).length
  const missed = questions.filter((q, i) => chosen[i] !== q.answer)
  const lessonIds = [...new Set(questions.map((q) => q.lesson))]
  const allDone = lessonIds.every((l) => progress.completed[l])

  if (allDone && !open) return null

  return (
    <div className="diagnostic">
      {!open ? (
        <div className="diag-head">
          <span><b>Already know {part}?</b> Take {questions.length} questions and find out what to skip.</span>
          <button className="btn small" onClick={() => setOpen(true)}>Check what I know</button>
        </div>
      ) : (
        <div className="diag-body">
          {questions.map((q, qi) => (
            <fieldset className="quiz-q" key={qi}>
              <legend>{qi + 1}. {q.q}</legend>
              {q.options.map((opt, oi) => {
                const cls = checked ? (oi === q.answer ? ' right' : chosen[qi] === oi ? ' wrong' : '') : ''
                return (
                  <label key={oi} className={`quiz-opt${cls}`}>
                    <input type="radio" name={`${id}-${qi}`} checked={chosen[qi] === oi} disabled={checked} onChange={() => setChosen(chosen.map((c, i) => (i === qi ? oi : c)))} />
                    <span>{opt}</span>
                  </label>
                )
              })}
            </fieldset>
          ))}
          {!checked ? (
            <div className="btn-row">
              <button className="btn primary" disabled={chosen.some((c) => c === null)} onClick={() => setChecked(true)}>See the result</button>
              <button className="btn small ghost" onClick={() => setOpen(false)}>Never mind, I will read it</button>
            </div>
          ) : (
            <div className="diag-result" role="status">
              <p><b>{score} of {questions.length} right.</b> {score === questions.length
                ? 'You know this part. Skim the diagrams and the “What you should remember” list, then move on.'
                : `Read ${missed.length === 1 ? 'this lesson' : 'these lessons'} carefully; the rest you can skim.`}</p>
              {missed.length > 0 && (
                <ul>
                  {[...new Set(missed.map((m) => m.lesson))].map((l) => {
                    const lesson = lessonById(l)
                    return lesson ? <li key={l}><a href={`#/lesson/${l}`}>{lesson.title}</a></li> : null
                  })}
                </ul>
              )}
              <div className="btn-row">
                {score === questions.length && (
                  <button className="btn small" onClick={() => updateProgress((p) => lessonIds.reduce((acc, l) => completeLesson(acc, l, true), p))}>
                    Mark this part as done
                  </button>
                )}
                <button className="btn small ghost" onClick={() => { setChecked(false); setChosen(questions.map(() => null)) }}>Try again</button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
