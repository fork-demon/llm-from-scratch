// Lesson 0.2: one question, four kinds of system. The database, the search engine
// and the rule-based program really run (see lib/fourSystems.ts); the LLM is described.
import { useState, type ReactNode } from 'react'
import { Lab } from '../components/ui'
import { DB_ROWS, DOCS, QUESTIONS, askAll, type Outcome, type Verdict } from '../lib/fourSystems'

const VERDICT: Record<Verdict, { text: string; color: string }> = {
  works: { text: '✓ answers it', color: 'var(--good)' },
  partial: { text: '~ sort of', color: 'var(--k)' },
  fails: { text: '✗ fails', color: 'var(--bad)' },
}

function SystemCard({ name, stores, atQuery, outcome, children }: { name: string; stores: ReactNode; atQuery: ReactNode; outcome: Outcome | null; children?: ReactNode }) {
  return (
    <div className="card" style={{ margin: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <h4 style={{ fontSize: 17, margin: 0 }}>{name}</h4>
        {outcome && <span className="chip" style={{ color: VERDICT[outcome.verdict].color, borderColor: VERDICT[outcome.verdict].color }}>{VERDICT[outcome.verdict].text}</span>}
      </div>
      <p className="lab-note" style={{ margin: '8px 0 4px' }}><b>What it stores:</b> {stores}</p>
      <p className="lab-note" style={{ margin: '4px 0 8px' }}><b>At query time:</b> {atQuery}</p>
      {outcome ? (
        <>
          <div className="readout" style={{ display: 'block', whiteSpace: 'pre-wrap' }}>{outcome.output}</div>
          {children}
          <p style={{ fontSize: 15, marginTop: 8, marginBottom: 0 }}>{outcome.why}</p>
        </>
      ) : children}
    </div>
  )
}

export function FourSystems() {
  const [picked, setPicked] = useState(0) // index into QUESTIONS, or -1 for free text
  const [custom, setCustom] = useState('What is the capital of Italy?')
  const preset = picked >= 0 ? QUESTIONS[picked] : undefined
  const text = preset ? preset.text : custom
  const r = askAll(text, preset)

  return (
    <Lab
      title="One question, four systems"
      goal={<>Pick a question. Before you read the cards, <b>predict which systems will cope</b>. Work through all six: each system breaks on a different one, and <em>how</em> it breaks is the lesson.</>}
    >
      <div className="steps" role="group" aria-label="Choose a question">
        {QUESTIONS.map((q, i) => (
          <button key={q.id} className="step-btn" aria-pressed={picked === i} onClick={() => setPicked(i)}>{q.label}</button>
        ))}
        <button className="step-btn" aria-pressed={picked === -1} onClick={() => setPicked(-1)}>type your own</button>
      </div>

      {picked === -1 ? (
        <>
          <label className="sr-only" htmlFor="four-systems-q">Your question</label>
          <input id="four-systems-q" className="input" value={custom} onChange={(e) => setCustom(e.target.value)} />
        </>
      ) : (
        <div className="card center" style={{ fontFamily: 'var(--serif)', fontSize: 20 }}>“{text}”</div>
      )}

      <div className="grid-2" style={{ marginTop: 14 }} aria-live="polite">
        <SystemCard
          name="Database lookup"
          stores={<>{Object.keys(DB_ROWS).length} rows of exact question → answer.</>}
          atQuery={<>finds the row whose key equals your text. <span className="mono">SELECT answer WHERE question = ?</span></>}
          outcome={r.db}
        />
        <SystemCard
          name="Search engine"
          stores={<>an index over {DOCS.length} documents that people wrote.</>}
          atQuery="ranks the stored documents by how many keywords they share with your text."
          outcome={r.search}
        >
          {r.search.results.length > 0 && (
            <ol style={{ fontSize: 14, margin: '8px 0 0', paddingLeft: 20 }}>
              {r.search.results.map((x) => <li key={x.doc.title} style={{ margin: '2px 0' }}><b>{x.doc.title}</b> <span className="muted mono">({x.score} keyword{x.score === 1 ? '' : 's'})</span><br /><span className="muted">{x.doc.text}</span></li>)}
            </ol>
          )}
        </SystemCard>
        <SystemCard
          name="Traditional program"
          stores="rules a programmer wrote: an arithmetic rule, and a capital-city table with 3 countries."
          atQuery="matches your text against its patterns and runs the code for the first that fits."
          outcome={r.program}
        />
        <SystemCard
          name="LLM"
          stores="no questions, no answers, no documents. Only billions of numbers, tuned during training."
          atQuery="computes probabilities for the next token from the text so far, picks one, repeats."
          outcome={r.llm}
        >
          {!r.llm && <p style={{ fontSize: 15, marginBottom: 0 }}>We cannot run a real LLM inside this page, and we will not fake its answer. The other three cards are running for real on your text. Put the same question to a chat assistant and compare.</p>}
        </SystemCard>
      </div>

      {preset && <div className="readout" style={{ marginTop: 14 }}><span><b>Takeaway:</b> {preset.lesson}</span></div>}
      <p className="lab-note" style={{ marginTop: 10 }}>
        The first three systems here are tiny but real: the verdicts come from actually running them. The LLM card describes typical behaviour of a model answering from its own numbers, with no tools attached. The example wording of its answers is illustrative.
      </p>
    </Lab>
  )
}
