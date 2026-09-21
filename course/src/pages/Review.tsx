// #/review: the questions you got wrong, brought back when they are due.
// Questions are collected from the lessons themselves, so nothing is duplicated here.
import { useMemo, useState } from 'react'
import { LESSONS, lessonById } from '../data/curriculum'
import { dueItems, nextDue, recordAnswer, updateReview, useReview } from '../lib/review'
import { QUIZ_BANK, type BankQuestion } from '../data/quizBank'

const when = (ms: number): string => {
  const mins = Math.round((ms - Date.now()) / 60000)
  if (mins <= 0) return 'now'
  if (mins < 60) return `in ${mins} min`
  const hours = Math.round(mins / 60)
  if (hours < 24) return `in ${hours} hour${hours === 1 ? '' : 's'}`
  const days = Math.round(hours / 24)
  return `in ${days} day${days === 1 ? '' : 's'}`
}

export function ReviewPage() {
  const state = useReview()
  const [answers, setAnswers] = useState<Record<string, number>>({})
  const [checked, setChecked] = useState(false)

  const due = useMemo(() => dueItems(state), [state])
  const cards = useMemo(
    () => due.map((item) => ({ item, q: QUIZ_BANK[item.id] })).filter((c): c is { item: typeof due[number]; q: BankQuestion } => !!c.q),
    [due],
  )
  const total = Object.keys(state).length
  const upcoming = nextDue(state)

  const check = () => {
    setChecked(true)
    updateReview((r) => cards.reduce((acc, c) => recordAnswer(acc, c.item.id, c.item.lesson, answers[c.item.id] === c.q.answer, Date.now()), r))
  }

  return (
    <div>
      <h1 className="lesson-title">Review</h1>
      <p className="lesson-question">Questions you did not get right, brought back at growing intervals. Recalling something just as you are about to forget it is what makes it stick.</p>

      {total === 0 && (
        <div className="card">
          <p><b>Nothing to review yet.</b></p>
          <p className="muted" style={{ margin: 0 }}>Answer the “Can you explain this?” questions at the end of a lesson. Anything you get wrong appears here later.</p>
        </div>
      )}

      {total > 0 && cards.length === 0 && (
        <div className="card">
          <p><b>Nothing is due right now.</b></p>
          <p className="muted" style={{ margin: 0 }}>{total} question{total === 1 ? '' : 's'} waiting. The next one is due {upcoming ? when(upcoming) : 'later'}.</p>
        </div>
      )}

      {cards.length > 0 && (
        <>
          <p className="muted">{cards.length} question{cards.length === 1 ? '' : 's'} due. {total - cards.length > 0 && `${total - cards.length} more waiting for later.`}</p>
          {cards.map(({ item, q }, i) => {
            const lesson = lessonById(item.lesson)
            const picked = answers[item.id]
            return (
              <fieldset className="quiz-q" key={item.id}>
                <legend>{i + 1}. {q.q}</legend>
                {q.options.map((opt, oi) => {
                  const cls = checked ? (oi === q.answer ? ' right' : picked === oi ? ' wrong' : '') : ''
                  return (
                    <label key={oi} className={`quiz-opt${cls}`}>
                      <input type="radio" name={item.id} checked={picked === oi} disabled={checked} onChange={() => setAnswers({ ...answers, [item.id]: oi })} />
                      <span>{opt}</span>
                    </label>
                  )
                })}
                {checked && (
                  <div className="quiz-explain">
                    {picked === q.answer ? 'Right. ' : 'Not quite. '}{q.explain}
                    {lesson && <> <a href={`#/lesson/${lesson.id}`}>Re-read {lesson.title}</a>.</>}
                  </div>
                )}
              </fieldset>
            )
          })}
          <div className="btn-row">
            {!checked ? (
              <button className="btn primary" disabled={cards.some((c) => answers[c.item.id] === undefined)} onClick={check}>Check my answers</button>
            ) : (
              <>
                <strong role="status">{cards.filter((c) => answers[c.item.id] === c.q.answer).length} of {cards.length} right</strong>
                <button className="btn small" onClick={() => { setAnswers({}); setChecked(false) }}>Done</button>
              </>
            )}
          </div>
        </>
      )}

      {total > 0 && (
        <p className="muted" style={{ marginTop: 32, fontSize: 14 }}>
          Questions come from the lessons you have taken: {[...new Set(Object.values(state).map((i) => i.lesson))].map((id) => lessonById(id)?.title).filter(Boolean).slice(0, 6).join(', ')}
          {LESSONS.length > 0 && '.'}
        </p>
      )}
    </div>
  )
}
