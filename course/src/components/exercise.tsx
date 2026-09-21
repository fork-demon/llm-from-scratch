// The exercise system: try yourself -> hint 1 -> hint 2 -> hint 3 -> solution.
// Answers are never shown before the learner asks for them.
import { useState, type ReactNode } from 'react'
import { checkNumeric, checkOrder, checkText, countInPlace, optionOrder, scoreQuiz, type NumericAnswer } from '../lib/exercise'
import { completeExercise, recordQuiz, updateProgress, useProgress } from '../lib/progress'
import { recordAnswer, updateReview } from '../lib/review'
import { registerQuestions } from '../data/quizBank'

export type ExerciseType = 'predict' | 'calculate' | 'modify' | 'debug' | 'implement' | 'explain' | 'experiment' | 'trace'

interface ExerciseProps {
  /** Globally unique, e.g. "attention-1". Used for progress tracking. */
  id: string
  type: ExerciseType
  title: string
  children: ReactNode // the task
  hints?: ReactNode[] // progressively more revealing, up to 3
  solution: ReactNode // answer plus explanation
  /** Optional auto-checked answer box. */
  answer?: NumericAnswer | { text: string[] }
  answerLabel?: string
}

export function Exercise({ id, type, title, children, hints = [], solution, answer, answerLabel }: ExerciseProps) {
  const progress = useProgress()
  const done = !!progress.exercises[id]
  const [shown, setShown] = useState(0) // hints revealed
  const [solved, setSolved] = useState(false) // solution revealed
  const [raw, setRaw] = useState('')
  const [verdict, setVerdict] = useState<null | boolean>(null)

  const markDone = () => updateProgress((p) => completeExercise(p, id))
  const check = () => {
    if (!answer) return
    const ok = 'text' in answer ? checkText(raw, answer.text) : checkNumeric(raw, answer)
    setVerdict(ok)
    if (ok) markDone()
  }

  return (
    <div className="exercise" id={`ex-${id}`}>
      <div className="exercise-head">
        <span className="ex-type">{type}</span>
        <span className="ex-title">{title}</span>
        {done && <span className="ex-done">✓ done</span>}
      </div>
      <div className="exercise-body">
        {children}

        {answer && (
          <form className="answer-row" onSubmit={(e) => { e.preventDefault(); check() }}>
            <label className="sr-only" htmlFor={`ans-${id}`}>{answerLabel ?? 'Your answer'}</label>
            <input id={`ans-${id}`} className="input" placeholder={answerLabel ?? 'Your answer'} value={raw} onChange={(e) => { setRaw(e.target.value); setVerdict(null) }} autoComplete="off" />
            <button className="btn small" type="submit">Check</button>
            <span className={`verdict ${verdict ? 'ok' : 'no'}`} role="status">
              {verdict === true && 'Correct.'}
              {verdict === false && 'Not yet. Try again, or take a hint.'}
            </span>
          </form>
        )}

        {hints.slice(0, shown).map((h, i) => (
          <div className="hint" key={i}><b>Hint {i + 1}</b>{h}</div>
        ))}
        {solved && <div className="solution"><b>Solution</b>{solution}</div>}

        <div className="btn-row" style={{ marginTop: 12 }}>
          {shown < hints.length && !solved && (
            <button className="btn small" onClick={() => setShown(shown + 1)}>
              {shown === 0 ? 'I am stuck: give me a hint' : `Hint ${shown + 1} of ${hints.length}`}
            </button>
          )}
          {!solved && (shown >= hints.length || verdict === true) && (
            <button className="btn small" onClick={() => setSolved(true)}>{verdict === true ? 'Show the explanation' : 'Show the solution'}</button>
          )}
          {!answer && !done && (
            <button className="btn small" onClick={markDone}>I did this ✓</button>
          )}
        </div>
      </div>
    </div>
  )
}

/* ---------- "Explain it in your own words" ---------- */
export function ExplainBack({ id, prompt, modelAnswer }: { id: string; prompt: ReactNode; modelAnswer: ReactNode }) {
  const [text, setText] = useState('')
  const [revealed, setRevealed] = useState(false)
  const progress = useProgress()
  return (
    <div className="exercise">
      <div className="exercise-head">
        <span className="ex-type">explain</span>
        <span className="ex-title">In your own words</span>
        {progress.exercises[id] && <span className="ex-done">✓ done</span>}
      </div>
      <div className="exercise-body">
        <p>{prompt}</p>
        <label className="sr-only" htmlFor={`explain-${id}`}>Your explanation</label>
        <textarea id={`explain-${id}`} className="input" value={text} onChange={(e) => setText(e.target.value)} placeholder="Write two or three sentences. Nobody sees this but you: writing it is the point." />
        {revealed && <div className="solution"><b>One good answer</b>{modelAnswer}<p className="muted" style={{ marginTop: 8, marginBottom: 0 }}>Compare honestly: did yours cover the same ideas?</p></div>}
        <div className="btn-row" style={{ marginTop: 10 }}>
          {!revealed && (
            <button className="btn small" disabled={text.trim().length < 20} onClick={() => { setRevealed(true); updateProgress((p) => completeExercise(p, id)) }}>
              Compare with a model answer
            </button>
          )}
          {!revealed && text.trim().length < 20 && <span className="muted" style={{ fontSize: 13.5 }}>Write your own answer first.</span>}
        </div>
      </div>
    </div>
  )
}

/* ---------- put steps in order (used for "reconstruct the pipeline") ---------- */
export function OrderExercise({ id, title, prompt, correct, solutionNote }: { id: string; title: string; prompt: ReactNode; correct: string[]; solutionNote?: ReactNode }) {
  // deterministic shuffle so the starting order is never already correct
  const initial = () => {
    const a = correct.slice()
    for (let i = a.length - 1; i > 0; i--) {
      const j = (i * 7 + 3) % (i + 1)
      ;[a[i], a[j]] = [a[j], a[i]]
    }
    if (checkOrder(a, correct) && a.length > 1) [a[0], a[1]] = [a[1], a[0]]
    return a
  }
  const [order, setOrder] = useState(initial)
  const [checked, setChecked] = useState(false)
  const progress = useProgress()
  const ok = checkOrder(order, correct)
  const move = (i: number, d: -1 | 1) => {
    const j = i + d
    if (j < 0 || j >= order.length) return
    const next = order.slice()
    ;[next[i], next[j]] = [next[j], next[i]]
    setOrder(next)
    setChecked(false)
  }
  return (
    <div className="exercise">
      <div className="exercise-head">
        <span className="ex-type">trace</span>
        <span className="ex-title">{title}</span>
        {progress.exercises[id] && <span className="ex-done">✓ done</span>}
      </div>
      <div className="exercise-body">
        {prompt}
        <ol style={{ listStyle: 'none', padding: 0, display: 'grid', gap: 5, maxWidth: 460 }}>
          {order.map((item, i) => (
            <li key={item} style={{ display: 'grid', gridTemplateColumns: '28px 1fr auto auto', gap: 6, alignItems: 'center', margin: 0 }}>
              <span className="mono muted">{i + 1}</span>
              <span className="flow-node" style={{ textAlign: 'left', borderColor: checked ? (item === correct[i] ? 'var(--good)' : 'var(--bad)') : undefined }}>{item}</span>
              <button className="btn small" aria-label={`Move ${item} up`} disabled={i === 0} onClick={() => move(i, -1)}>↑</button>
              <button className="btn small" aria-label={`Move ${item} down`} disabled={i === order.length - 1} onClick={() => move(i, 1)}>↓</button>
            </li>
          ))}
        </ol>
        <div className="btn-row">
          <button className="btn small primary" onClick={() => { setChecked(true); if (ok) updateProgress((p) => completeExercise(p, id)) }}>Check order</button>
          <span className={`verdict ${ok ? 'ok' : 'no'}`} role="status">
            {checked && (ok ? 'Exactly right.' : `${countInPlace(order, correct)} of ${correct.length} in the right place. Keep going.`)}
          </span>
        </div>
        {checked && ok && solutionNote && <div className="solution"><b>Why this order</b>{solutionNote}</div>}
      </div>
    </div>
  )
}

/* ---------- multiple-choice quiz: tests understanding, not vocabulary ---------- */
export interface QuizQuestion {
  q: ReactNode
  options: ReactNode[]
  answer: number // index into options
  explain: ReactNode // shown after answering: why the right answer is right
}

export function Quiz({ id, questions, title, lesson }: { id: string; questions: QuizQuestion[]; title?: string; lesson?: string }) {
  const [chosen, setChosen] = useState<(number | null)[]>(() => questions.map(() => null))
  const [submitted, setSubmitted] = useState(false)
  const progress = useProgress()
  const best = progress.quiz[id]
  registerQuestions(id, questions)
  const all = chosen.every((c) => c !== null)
  const score = scoreQuiz(chosen, questions.map((q) => q.answer))

  const submit = () => {
    setSubmitted(true)
    updateProgress((p) => recordQuiz(p, id, score, questions.length))
    // Anything answered wrong comes back later in the review queue.
    updateReview((r) => questions.reduce((acc, q, i) => recordAnswer(acc, `${id}#${i}`, lesson ?? id, chosen[i] === q.answer, Date.now()), r))
  }
  const retry = () => {
    setChosen(questions.map(() => null))
    setSubmitted(false)
  }

  return (
    <div className="card" aria-label={title ?? 'Quiz'}>
      {title && <h3 style={{ marginTop: 0 }}>{title}</h3>}
      {questions.map((qu, qi) => (
        <fieldset className="quiz-q" key={qi}>
          <legend>{qi + 1}. {qu.q}</legend>
          {optionOrder(id, qi, qu.options.length).map((oi) => {
            const opt = qu.options[oi]
            const picked = chosen[qi] === oi
            const cls = submitted ? (oi === qu.answer ? ' right' : picked ? ' wrong' : '') : ''
            return (
              <label key={oi} className={`quiz-opt${cls}`}>
                <input type="radio" name={`${id}-${qi}`} checked={picked} disabled={submitted} onChange={() => setChosen(chosen.map((c, i) => (i === qi ? oi : c)))} />
                <span>{opt}</span>
              </label>
            )
          })}
          {submitted && <div className="quiz-explain">{chosen[qi] === qu.answer ? '✓ ' : '✗ '}{qu.explain}</div>}
        </fieldset>
      ))}
      <div className="btn-row">
        {!submitted ? (
          <button className="btn primary" disabled={!all} onClick={submit}>Check my answers</button>
        ) : (
          <>
            <strong role="status">{score} / {questions.length}</strong>
            <button className="btn small" onClick={retry}>Try again</button>
          </>
        )}
        {best && <span className="muted" style={{ fontSize: 13.5 }}>Best so far: {best.correct}/{best.total}</span>}
      </div>
    </div>
  )
}
