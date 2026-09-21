// The final challenge: rebuild the pipeline, explain every arrow, then say what
// RAG, fine-tuning and tools change. Optional 10-minute timer. Nothing is graded
// by the machine: the learner rates their own explanations against model answers.
import { useEffect, useState } from 'react'
import { Lab } from '../components/ui'
import { ExplainBack, OrderExercise } from '../components/exercise'
import { lessonById } from '../data/curriculum'
import { ARROWS, CHALLENGE_SECONDS, FULL_PIPELINE, formatTime, remainingSeconds, selfScore, verdictFor, type Rating } from '../lib/memoryChallenge'

const STAGES = ['1 · Rebuild the pipeline', '2 · Explain every arrow', '3 · What changes when…'] as const
const RATINGS: { id: Rating; label: string; mark: string }[] = [
  { id: 'got', label: 'Got it', mark: '✓' },
  { id: 'partly', label: 'Partly', mark: '~' },
  { id: 'missed', label: 'Missed it', mark: '✗' },
]

function Timer() {
  const [banked, setBanked] = useState(0) // ms used in earlier runs
  const [since, setSince] = useState<number | null>(null) // when the current run started
  const [now, setNow] = useState(() => Date.now())
  const running = since !== null
  const left = remainingSeconds(CHALLENGE_SECONDS, banked, since, now)

  useEffect(() => {
    if (!running) return
    const t = setInterval(() => setNow(Date.now()), 250)
    return () => clearInterval(t)
  }, [running])
  useEffect(() => {
    if (running && left === 0) { setBanked(CHALLENGE_SECONDS * 1000); setSince(null) }
  }, [running, left])

  const start = () => { const t = Date.now(); setNow(t); setSince(t) }
  const pause = () => { const t = Date.now(); setBanked(banked + (since === null ? 0 : t - since)); setSince(null) }
  const reset = () => { setBanked(0); setSince(null) }

  return (
    <div className="readout" style={{ alignItems: 'center', marginBottom: 14 }}>
      <span role="timer" aria-label={`Time remaining: ${formatTime(left)}`} style={{ fontSize: 22 }}><b>{formatTime(left)}</b></span>
      <span className="btn-row">
        {running
          ? <button className="btn small" onClick={pause}>Pause</button>
          : <button className="btn small primary" onClick={start} disabled={left === 0}>{banked > 0 ? 'Resume' : 'Start 10-minute timer'}</button>}
        <button className="btn small" onClick={reset} disabled={banked === 0 && !running}>Reset</button>
      </span>
      <span className="muted" role="status">{left === 0 ? 'Time. Finish your sentence, then carry on: the timer is a nudge, not a gate.' : 'Optional. A deadline stops you polishing and makes you retrieve.'}</span>
    </div>
  )
}

function Arrows() {
  const [i, setI] = useState(0)
  const [texts, setTexts] = useState<string[]>(() => ARROWS.map(() => ''))
  const [revealed, setRevealed] = useState<boolean[]>(() => ARROWS.map(() => false))
  const [ratings, setRatings] = useState<(Rating | null)[]>(() => ARROWS.map(() => null))
  const a = ARROWS[i]
  const score = selfScore(ratings)
  const set = <T,>(arr: T[], v: T) => arr.map((x, j) => (j === i ? v : x))
  const canReveal = texts[i].trim().length >= 15

  return (
    <>
      <p>For each arrow, write what happens there, in your own words, <b>before</b> revealing the model answer. Then rate yourself honestly. Nobody sees this but you.</p>
      <div className="steps" role="group" aria-label="Choose an arrow">
        {ARROWS.map((x, j) => {
          const r = RATINGS.find((k) => k.id === ratings[j])
          return <button key={j} className="step-btn" aria-pressed={j === i} aria-label={`Arrow ${j + 1}: ${x.from} to ${x.to}${r ? `, rated ${r.label}` : ''}`} onClick={() => setI(j)}>{j + 1}{r ? ` ${r.mark}` : ''}</button>
        })}
      </div>

      <div className="card">
        <h4 style={{ fontSize: 18, marginBottom: 8 }}><span className="muted mono" style={{ fontSize: 13 }}>arrow {i + 1} of {ARROWS.length}</span><br />{a.from} <span className="acc">→</span> {a.to}</h4>
        <label className="sr-only" htmlFor={`arrow-${i}`}>What happens between {a.from} and {a.to}?</label>
        <textarea id={`arrow-${i}`} className="input" value={texts[i]} onChange={(e) => setTexts(set(texts, e.target.value))} placeholder={`What happens between “${a.from}” and “${a.to}”? Two or three sentences.`} />
        {!revealed[i] ? (
          <div className="btn-row" style={{ marginTop: 10 }}>
            <button className="btn small" disabled={!canReveal} onClick={() => setRevealed(set(revealed, true))}>Reveal a model answer</button>
            {!canReveal && <span className="muted" style={{ fontSize: 13.5 }}>Write your own answer first.</span>}
          </div>
        ) : (
          <>
            <div className="solution"><b>One good answer</b><p style={{ marginBottom: 0 }}>{a.answer}</p></div>
            <div className="btn-row" style={{ marginTop: 10 }} role="group" aria-label="Rate your own answer">
              <span style={{ fontSize: 14.5 }}>Compared with that, I…</span>
              {RATINGS.map((r) => <button key={r.id} className="step-btn" aria-pressed={ratings[i] === r.id} onClick={() => setRatings(set(ratings, r.id))}>{r.mark} {r.label}</button>)}
            </div>
            <p className="lab-note" style={{ margin: '10px 0 0' }}>Covered in <a href={`#/lesson/${a.lesson}`}>{lessonById(a.lesson)?.title ?? a.lesson}</a>.</p>
          </>
        )}
      </div>

      <div className="btn-row">
        <button className="btn small" disabled={i === 0} onClick={() => setI(i - 1)}>← Previous arrow</button>
        <button className="btn small primary" disabled={i === ARROWS.length - 1} onClick={() => setI(i + 1)}>Next arrow</button>
      </div>

      <div className="readout" style={{ marginTop: 14 }} aria-live="polite">
        <span>rated <b>{score.rated} / {score.total}</b></span>
        <span>✓ got it <b>{score.got}</b></span>
        <span>~ partly <b>{score.partly}</b></span>
        <span>✗ missed <b>{score.missed}</b></span>
        <span>self-score <b>{score.points} / {score.maxPoints}</b>{score.complete ? ` (${score.percent}%)` : ''}</span>
      </div>
      {score.complete && (
        <div className="card" aria-live="polite">
          <h4 style={{ fontSize: 17, marginBottom: 6 }}>Your self-score: {score.percent}%</h4>
          <p>{verdictFor(score.percent)}</p>
          {score.revisit.length > 0 && (
            <>
              <p style={{ marginBottom: 4 }}>Arrows to revisit:</p>
              <ul style={{ marginTop: 0 }}>
                {score.revisit.map((j) => <li key={j}>{ARROWS[j].from} → {ARROWS[j].to}: <a href={`#/lesson/${ARROWS[j].lesson}`}>{lessonById(ARROWS[j].lesson)?.title ?? ARROWS[j].lesson}</a></li>)}
              </ul>
            </>
          )}
        </div>
      )}
    </>
  )
}

export function MemoryChallenge() {
  const [stage, setStage] = useState(0)
  return (
    <Lab
      title="Explain it from memory"
      goal={<>Close your notes and the other tabs. Three stages: <b>rebuild</b> the pipeline, <b>explain</b> each arrow, then say <b>what changes</b> when you add RAG, fine-tuning and tools. Getting things wrong here is useful: it shows you what to revisit.</>}
    >
      <Timer />
      <div className="steps" role="tablist" aria-label="Stages of the challenge">
        {STAGES.map((s, i) => <button key={s} role="tab" className="step-btn" aria-selected={i === stage} onClick={() => setStage(i)}>{s}</button>)}
      </div>

      {/* all three stay mounted so typed answers survive switching stages */}
      <div role="tabpanel" hidden={stage !== 0}>
        <OrderExercise
          id="from-memory-order"
          title="The full pipeline, in order"
          prompt={<p>Twelve stages, from the text you type to the loop. Do this one before stage 2, which gives the order away.</p>}
          correct={FULL_PIPELINE}
          solutionNote={<p>Text becomes numbers (tokens, ids, embeddings, position). The numbers are transformed (attention inside stacked Transformer blocks). The result becomes a choice (logits, probabilities, sampling, next token). The choice is fed back in (repeat).</p>}
        />
      </div>
      <div role="tabpanel" hidden={stage !== 1}><Arrows /></div>
      <div role="tabpanel" hidden={stage !== 2}>
        <p>Three things developers bolt onto an LLM. For each, the key is to say precisely <b>what changes and what does not</b>: the prompt, the weights, or the code around the model.</p>
        <ExplainBack
          id="from-memory-rag"
          prompt="What changes when we add RAG?"
          modelAnswer={<p><b>The prompt changes, not the weights.</b> Before the model runs, your code searches your documents (usually by embedding the question and finding the nearest chunk vectors) and pastes the best chunks into the prompt. The model is the same frozen function; it simply has the relevant text in its “text so far” and can copy and combine from it. That is why RAG can use information newer than the training cutoff, can cite sources, and can be updated by editing documents. It fails when retrieval brings back the wrong chunks, and the model may still ignore or misread the context.</p>}
        />
        <ExplainBack
          id="from-memory-finetune"
          prompt="What changes when we fine-tune?"
          modelAnswer={<p><b>The weights change, not the prompt.</b> Fine-tuning is more training: the same next-token loss and gradient descent, on a smaller, targeted dataset, starting from the pretrained parameters. It shifts the function itself, so the new behaviour (a tone, a format, a specialised skill) appears without being asked for in the prompt. LoRA freezes the original weights and trains small low-rank correction matrices instead. Costs: it needs data and compute, it can make the model forget other abilities, and it is a poor way to add facts that change, because every update means retraining.</p>}
        />
        <ExplainBack
          id="from-memory-tools"
          prompt="What changes when we add tools?"
          modelAnswer={<p><b>A loop is added around the model. Your code acts; the model only writes text.</b> The prompt describes the available tools and a format for requesting one. When the model’s output contains such a request, your code parses it, runs the real function, appends the result to the context and calls the model again, until it writes a final answer or a step limit is hit. The weights do not change. The model gains no abilities of its own: it gains a way to ask. Because tool results land in the same context as instructions, text inside them can steer the model (prompt injection), so the loop needs limits, validation and approval for irreversible actions.</p>}
        />
      </div>

      <div className="btn-row" style={{ marginTop: 14 }}>
        <button className="btn small" disabled={stage === 0} onClick={() => setStage(stage - 1)}>Previous stage</button>
        <button className="btn small primary" disabled={stage === STAGES.length - 1} onClick={() => setStage(stage + 1)}>Next stage</button>
      </div>
    </Lab>
  )
}
