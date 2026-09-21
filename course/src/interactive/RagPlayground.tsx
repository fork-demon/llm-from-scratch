// The whole RAG pipeline of mini_rag.py, live in the browser: edit the documents,
// chunk them, embed them, search, read the exact prompt, see the answer or the refusal.
import { useMemo, useState } from 'react'
import { Bars, Lab, Slider } from '../components/ui'
import { Code } from '../components/Code'
import { fmt } from '../lib/math'
import { CORPUS, DEMO_QUERIES, ask, ingest, sentences, topTerms, type Doc } from '../lib/rag'

const STAGES = ['1 · Documents', '2 · Chunks', '3 · Embeddings', '4 · Search', '5 · Prompt', '6 · Answer'] as const
const DEFAULTS = { size: 2, overlap: 1, k: 3 }

interface Preset { label: string; question: string; size: number; overlap: number; k: number; stage: number; look: string }
const FAILURES: Preset[] = [
  {
    label: 'Different words, same meaning', question: 'can I deploy without the pipeline', ...DEFAULTS, stage: 3,
    look: 'This is the --show-failure query from mini_rag.py. The right chunk is found, but only just: top similarity 0.33, and the answer clears the 0.35 threshold by a hair (0.38). The question says “deploy”; this chunk says “Deployments” and “deploys”. To a bag-of-words embedder those are three unrelated strings, so only “pipeline” and “the” match. And “without”, the word that carries the whole point of the question, was never seen at ingest time, so it is silently dropped.',
  },
  {
    label: 'No shared words at all', question: 'am I allowed to push a release by hand', ...DEFAULTS, stage: 5,
    look: 'The answer IS in the corpus: “Manual deploys are forbidden except during a declared incident.” But the question shares no meaningful word with it, so the system refuses. Stage that failed: embedding. A Transformer encoder would place “by hand” near “manual”. This one cannot.',
  },
  {
    label: 'The answer is not in any document', question: 'what is the wifi password', ...DEFAULTS, stage: 5,
    look: 'Nothing in the corpus mentions wifi. Retrieval still returns k chunks (it always does: top-k has no “nothing found”). The scores are all low, and the answerer refuses. That refusal is the behaviour you want. Now set sentences per chunk to 1 and look again.',
  },
  {
    label: 'Chunking splits the answer', question: 'who writes the postmortem and when is it due', size: 1, overlap: 0, k: 1, stage: 4,
    look: 'One sentence per chunk, no overlap, k = 1. The prompt contains who writes the postmortem, but the deadline lives in the next sentence, which is now a different chunk that was not retrieved. No model can answer “when” from this prompt. Set sentences per chunk to 2 and overlap to 1: both facts arrive in one chunk.',
  },
  {
    label: 'k too small', question: 'how many approvals does a deploy need and what is the budget for learning materials', ...DEFAULTS, k: 1, stage: 4,
    look: 'A two-part question whose parts live in different documents. With k = 1 only expenses.md reaches the prompt, so half the question cannot be answered. Raise k to 2 and the deploy-policy chunk arrives. (Our stand-in still quotes only one sentence. A real LLM would use both.)',
  },
]

export function RagPlayground() {
  const [docs, setDocs] = useState<Doc[]>(CORPUS)
  const [size, setSize] = useState(DEFAULTS.size)
  const [overlap, setOverlap] = useState(DEFAULTS.overlap)
  const [k, setK] = useState(DEFAULTS.k)
  const [question, setQuestion] = useState(DEMO_QUERIES[0])
  const [stage, setStage] = useState(0)
  const [picked, setPicked] = useState(0) // chunk inspected in the embedding stage
  const [preset, setPreset] = useState<Preset | null>(null)

  const ov = Math.min(overlap, size - 1)
  const index = useMemo(() => ingest(docs, size, ov), [docs, size, ov])
  const res = useMemo(() => ask(index, question, k), [index, question, k])
  const qvec = useMemo(() => index.embedder.embed(question), [index, question])
  const unknown = index.embedder.unknown(question)
  const chunkIdx = Math.min(picked, Math.max(0, index.chunks.length - 1))
  const edited = docs !== CORPUS

  const apply = (p: Preset) => {
    setDocs(CORPUS); setSize(p.size); setOverlap(p.overlap); setK(p.k); setQuestion(p.question); setStage(p.stage); setPreset(p)
  }
  const setDoc = (i: number, patch: Partial<Doc>) => setDocs(docs.map((d, j) => (j === i ? { ...d, ...patch } : d)))
  const ask1 = (q: string) => { setQuestion(q); setPreset(null) }

  return (
    <Lab
      title="A complete RAG pipeline"
      goal={<>Walk the six stages left to right. Everything is computed live from the documents in stage 1, with the same algorithm and numbers as <code>mini_rag.py</code>. Then press the failure buttons and work out <b>which stage</b> let you down.</>}
    >
      <label htmlFor="rag-q" style={{ fontSize: 14, color: 'var(--ink-2)' }}>Your question</label>
      <input id="rag-q" className="input" value={question} onChange={(e) => ask1(e.target.value)} />
      <div className="steps" role="group" aria-label="Questions that work" style={{ marginTop: 8, marginBottom: 8 }}>
        {DEMO_QUERIES.map((q) => <button key={q} className="step-btn" aria-pressed={question === q && !preset} onClick={() => { ask1(q); setSize(DEFAULTS.size); setOverlap(DEFAULTS.overlap); setK(DEFAULTS.k) }}>{q}</button>)}
      </div>
      <div className="steps" role="group" aria-label="One-click failure cases">
        <span className="muted" style={{ fontSize: 13.5, alignSelf: 'center' }}>Break it:</span>
        {FAILURES.map((p) => <button key={p.label} className="step-btn" aria-pressed={preset === p} onClick={() => apply(p)}>{p.label}</button>)}
      </div>
      {preset && <p className="lab-note" role="status"><b>What to look for:</b> {preset.look}</p>}

      <div className="controls">
        <Slider label="Sentences per chunk" value={size} min={1} max={5} step={1} onChange={(v) => { setSize(v); setPicked(0) }} />
        <Slider label="Overlap (sentences shared with the next chunk)" value={ov} min={0} max={Math.max(0, size - 1)} step={1} onChange={setOverlap} />
        <Slider label="k (chunks to retrieve)" value={k} min={1} max={6} step={1} onChange={setK} />
      </div>

      <div className="steps" role="tablist" aria-label="Pipeline stages">
        {STAGES.map((s, i) => <button key={s} role="tab" className="step-btn" aria-selected={i === stage} onClick={() => setStage(i)}>{s}</button>)}
      </div>
      <p className="muted" style={{ fontSize: 13.5, marginTop: -8 }}>Stages 1 to 3 happen once, at <b>ingest time</b>. Stages 4 to 6 happen for every question, at <b>query time</b>.</p>

      <div role="tabpanel" aria-live="polite">
        {stage === 0 && (
          <>
            <p>The knowledge base. None of this is in any model’s weights. Edit it, delete a document, or add your own: everything downstream is recomputed.</p>
            {docs.map((d, i) => (
              <div key={i} className="card" style={{ padding: 12, marginBottom: 8 }}>
                <div style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
                  <input className="input mono" style={{ fontSize: 13.5 }} aria-label={`Name of document ${i + 1}`} value={d.name} onChange={(e) => setDoc(i, { name: e.target.value })} />
                  <button className="btn small" onClick={() => setDocs(docs.filter((_, j) => j !== i))} aria-label={`Remove ${d.name}`}>Remove</button>
                </div>
                <textarea className="input" rows={3} style={{ fontSize: 14.5 }} aria-label={`Text of ${d.name}`} value={d.text} onChange={(e) => setDoc(i, { text: e.target.value })} />
              </div>
            ))}
            <div className="btn-row">
              <button className="btn small" onClick={() => setDocs([...docs, { name: `notes-${docs.length + 1}.md`, text: 'The office wifi password is rotated every month. Ask the front desk for the current one.' }])}>+ Add a document</button>
              {edited && <button className="btn small" onClick={() => setDocs(CORPUS)}>Reset to the original 4 documents</button>}
            </div>
          </>
        )}

        {stage === 1 && (
          <>
            <p><b>{docs.length}</b> documents became <b>{index.chunks.length}</b> chunks. Each chunk is {size} sentence{size > 1 ? 's' : ''}; each window starts {Math.max(1, size - ov)} sentence{Math.max(1, size - ov) > 1 ? 's' : ''} after the previous one. {ov > 0 ? <>The <span className="acc"><b>highlighted</b></span> sentences also appear in the previous chunk: that is the overlap.</> : <>No overlap: a fact that straddles a boundary is cut in two.</>}</p>
            <div style={{ display: 'grid', gap: 6 }}>
              {index.chunks.map((c, i) => (
                <div key={i} className="card" style={{ padding: '8px 12px', fontSize: 14.5 }}>
                  <span className="chip" style={{ marginRight: 8 }}>{c.source} · sentence {c.pos + 1}</span>
                  {sentences(c.text).map((s, j) => {
                    const shared = c.pos > 0 && j < ov // the first `overlap` sentences repeat the end of the previous chunk
                    return <span key={j} className={shared ? 'acc' : undefined}>{s} </span>
                  })}
                </div>
              ))}
            </div>
          </>
        )}

        {stage === 2 && index.chunks.length > 0 && (
          <>
            <p>Every chunk becomes one vector with <b>{index.embedder.vocab.length}</b> numbers: one per distinct word in the corpus. Pick a chunk to see which words carry its weight.</p>
            <label htmlFor="rag-chunk" className="sr-only">Chunk to inspect</label>
            <select id="rag-chunk" className="input" value={chunkIdx} onChange={(e) => setPicked(Number(e.target.value))}>
              {index.chunks.map((c, i) => <option key={i} value={i}>{c.source}: {c.text.slice(0, 70)}…</option>)}
            </select>
            <p style={{ marginTop: 10 }}><em>{index.chunks[chunkIdx].text}</em></p>
            <Bars percent={false} items={topTerms(index.store.vecs[chunkIdx], index.embedder.vocab, 8).map((t) => ({ label: t.term, value: t.weight }))} />
            <p className="lab-note" style={{ marginTop: 10 }}><b>Honest warning:</b> this is a crude bag-of-words embedder (TF-IDF: rare words count more, “the” counts almost nothing, plus a little weight leaked onto words that co-occur). It knows nothing about meaning or word order: “deploy” and “deployments” are strangers. Real systems use a Transformer encoder that outputs a dense vector of roughly 400 to 3,000 numbers. The rest of the pipeline is identical.</p>
          </>
        )}

        {stage === 3 && (
          <>
            <p>The question is embedded <em>with the same embedder</em>, then compared with every chunk vector by <b>cosine similarity</b>. The top {k} are kept, however bad they are.</p>
            <p className="lab-note">Question’s heaviest words: {topTerms(qvec, index.embedder.vocab, 5).map((t) => <span key={t.term} className="token">{t.term} {fmt(t.weight)}</span>)}{unknown.length > 0 && <> · never seen, so ignored: {unknown.map((w) => <span key={w} className="token" style={{ opacity: 0.55 }}>{w}</span>)}</>}</p>
            <Bars max={1} percent={false} items={res.retrieved.map((r, i) => ({ label: `[${i + 1}] ${r.chunk.source}`, value: Math.max(0, r.sim) }))} />
            <div style={{ display: 'grid', gap: 6, marginTop: 10 }}>
              {res.retrieved.map((r, i) => (
                <div key={i} className="card" style={{ padding: '8px 12px', fontSize: 14.5 }}>
                  <span className="chip acc" style={{ marginRight: 8 }}>[{i + 1}] sim {fmt(r.sim)}</span>{r.chunk.text}
                </div>
              ))}
            </div>
            <p className="lab-note" style={{ marginTop: 10 }}>1.00 would mean “same direction”, 0 means “nothing in common”. With {index.chunks.length} chunks, exact search is {index.chunks.length} dot products.</p>
          </>
        )}

        {stage === 4 && (
          <>
            <p>This is the <b>exact text</b> that would be sent to the LLM. This string is the only thing RAG changes. The model and its weights are untouched.</p>
            <pre className="mono" tabIndex={0} style={{ whiteSpace: 'pre-wrap', fontSize: 13.5, padding: 14, background: 'var(--paper-2)', border: '1px solid var(--rule)', borderRadius: 8 }}>{res.prompt}</pre>
            <p className="lab-note">{res.prompt.length} characters. The instruction in the first two lines is what asks for grounding and refusal. The model can still ignore it.</p>
          </>
        )}

        {stage === 5 && (
          <>
            <div className="readout" style={{ borderColor: res.answer.found ? 'var(--good)' : 'var(--bad)' }}>
              {res.answer.found ? (
                <><span>ANSWER: <b>{res.answer.sentence}.</b></span><span>cited: [{res.answer.cite!.n}] {res.answer.cite!.source}</span><span>sentence similarity {fmt(res.answer.sim)} ≥ 0.35</span></>
              ) : (
                <><span>ANSWER: <b>Not found in the provided context.</b></span><span>best sentence similarity {fmt(Math.max(0, res.answer.sim))} &lt; 0.35, so it refuses</span></>
              )}
            </div>
            <p className="lab-note" style={{ marginTop: 10 }}><b>The “LLM” here is a stand-in</b>, exactly as in the Python file, so that everything runs offline. It is <em>extractive</em>: it scores every sentence in the retrieved chunks against the question, quotes the best one with its citation, and refuses if the best score is under 0.35. It cannot make anything up, and it cannot combine two sentences. A real model can do both.</p>
            <p>To use a real model, these three lines replace the stand-in. Nothing else in the pipeline changes:</p>
            <Code source="phase4-modern-llms/mini_rag.py" title="swap in any chat API">{`
response = client.messages.create(model=..., max_tokens=300,
    messages=[{"role": "user", "content": prompt}])
answer = response.content[0].text
`}</Code>
          </>
        )}
        {stage === 2 && index.chunks.length === 0 && <p>No chunks: add a document in stage 1.</p>}
      </div>

      <div className="btn-row" style={{ marginTop: 16 }}>
        <button className="btn small" disabled={stage === 0} onClick={() => setStage(stage - 1)}>Back</button>
        <button className="btn small primary" disabled={stage === STAGES.length - 1} onClick={() => setStage(stage + 1)}>Next stage</button>
      </div>
    </Lab>
  )
}
