// Train tiny word vectors live in the browser (port of tiny_word2vec.py) and
// watch similar words drift together.
import { useEffect, useMemo, useRef, useState } from 'react'
import { Bars, Lab } from '../components/ui'
import { fmt } from '../lib/math'
import { SENTENCES_BASE, buildDataset, initModel, nearest, similarity, trainStep, type W2VModel } from '../lib/word2vec'

const TOTAL = 2000
const WATCH: [string, string][] = [['cat', 'dog'], ['bread', 'cheese'], ['laptop', 'computer'], ['cat', 'computer'], ['bread', 'software']]
const PLOT = ['cat', 'dog', 'kitten', 'puppy', 'mouse', 'rice', 'bread', 'cheese', 'pasta', 'beans', 'computer', 'laptop', 'software', 'code', 'programmer']
const QUERIES = ['cat', 'bread', 'computer', 'dog', 'laptop', 'rice']
const SIZE = 420

export function Word2VecTrainer() {
  const data = useMemo(() => buildDataset(SENTENCES_BASE), [])
  const [dim, setDim] = useState(32)
  const [seed, setSeed] = useState(2)
  const [running, setRunning] = useState(false)
  const [query, setQuery] = useState('cat')
  const [, setTick] = useState(0)
  const modelRef = useRef<W2VModel | null>(null)
  if (!modelRef.current || modelRef.current.dim !== dim) modelRef.current = initModel(data.words.length, dim, seed)
  const model = modelRef.current

  // chunked training: a few milliseconds of SGD per frame, stops on unmount
  useEffect(() => {
    if (!running) return
    let raf = 0
    const frame = () => {
      const t0 = performance.now()
      while (model.step < TOTAL && performance.now() - t0 < 10) trainStep(model, data)
      setTick((t) => t + 1)
      if (model.step >= TOTAL) setRunning(false)
      else raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [running, model, data])

  const reset = (d = dim, s = seed) => {
    setRunning(false)
    modelRef.current = initModel(data.words.length, d, s)
    setDim(d)
    setSeed(s)
    setTick((t) => t + 1)
  }

  const id = (w: string) => data.stoi.get(w)!
  const pts = PLOT.map((w) => ({ w, x: model.E[id(w) * dim], y: model.E[id(w) * dim + 1] }))
  const extent = Math.max(0.3, ...pts.flatMap((p) => [Math.abs(p.x), Math.abs(p.y)])) * 1.25
  const px = (v: number) => SIZE / 2 + (v / extent) * (SIZE / 2)
  const py = (v: number) => SIZE / 2 - (v / extent) * (SIZE / 2)

  return (
    <Lab
      title="Watch word vectors learn"
      goal={<>Every word starts as random numbers. Press <b>Train</b>. The only task is “given a word, guess a word that appeared next to it”. Watch which similarities rise and which stay near zero. Nobody tells the model that cats and dogs are alike.</>}
    >
      <div className="steps" role="group" aria-label="Numbers per word">
        {[2, 8, 32].map((d) => <button key={d} className="step-btn" aria-pressed={dim === d} onClick={() => reset(d)}>{d} numbers per word</button>)}
      </div>
      <div className="btn-row" style={{ margin: '10px 0' }}>
        <button className="btn small primary" disabled={model.step >= TOTAL} onClick={() => setRunning(!running)}>{running ? 'Pause' : model.step === 0 ? 'Train' : 'Continue'}</button>
        <button className="btn small" disabled={running || model.step >= TOTAL} onClick={() => { for (let i = 0; i < 20 && model.step < TOTAL; i++) trainStep(model, data); setTick((t) => t + 1) }}>20 steps</button>
        <button className="btn small" onClick={() => reset()}>Reset to random</button>
        <button className="btn small" onClick={() => reset(dim, seed + 1)}>New random start</button>
      </div>
      <div className="readout" aria-live="polite">
        <span>step <b>{model.step}</b> / {TOTAL}</span>
        <span>loss on the guessing game: <b>{model.loss.toFixed(2)}</b> (pure guessing = ln 70 = 4.25)</span>
      </div>

      <div className="grid-2" style={{ marginTop: 12, alignItems: 'start' }}>
        <div>
          <div className="matrix-cap">Cosine similarity of word pairs</div>
          <Bars percent={false} max={1} items={WATCH.map(([a, b]) => ({ label: `${a} ~ ${b}`, value: similarity(model, id(a), id(b)), tone: 'accent' as const }))} />
          <p className="muted" style={{ fontSize: 13 }}>Bars only show positive values; the number on the right is exact. The top three pairs share contexts in the sentences. The bottom two do not.</p>
          <div className="matrix-cap" style={{ marginTop: 10 }}>Nearest neighbours of</div>
          <div className="steps" role="group" aria-label="Choose a word">
            {QUERIES.map((q) => <button key={q} className="step-btn" aria-pressed={q === query} onClick={() => setQuery(q)}>{q}</button>)}
          </div>
          <p className="mono" aria-live="polite" style={{ fontSize: 14 }}>{nearest(model, data, query, 4).map((n) => `${n.word} (${fmt(n.sim)})`).join(', ')}</p>
        </div>
        <div>
          {dim === 2 ? (
            <>
              <div className="matrix-cap">With 2 numbers per word, the plot <em>is</em> the embedding</div>
              <svg viewBox={`0 0 ${SIZE} ${SIZE}`} style={{ width: '100%', maxWidth: SIZE, border: '1px solid var(--rule)', borderRadius: 8, background: 'var(--card)' }} role="img" aria-label={`Scatter plot of the learned two-number vectors of ${PLOT.length} words.`}>
                <line className="axis" x1={SIZE / 2} y1={0} x2={SIZE / 2} y2={SIZE} />
                <line className="axis" x1={0} y1={SIZE / 2} x2={SIZE} y2={SIZE / 2} />
                {pts.map((p) => (
                  <g key={p.w}>
                    <circle cx={px(p.x)} cy={py(p.y)} r={4} fill="var(--accent)" />
                    <text x={px(p.x) + 6} y={py(p.y) + 4} fontSize={13}>{p.w}</text>
                  </g>
                ))}
              </svg>
              <p className="lab-note">Groups usually form, but look at the loss: it gets stuck around 3. Two numbers are too few to keep 70 words apart in all the ways the task needs. Switch to 32 and compare the loss.</p>
            </>
          ) : (
            <>
              <div className="matrix-cap">The vector of “{query}” right now ({dim} numbers)</div>
              <p className="mono" style={{ fontSize: 12.5, wordBreak: 'break-word', color: 'var(--ink-2)' }}>[{Array.from(model.E.subarray(id(query) * dim, (id(query) + 1) * dim)).map((v) => fmt(v, 2)).join(', ')}]</p>
              <p className="lab-note">There is no honest way to draw {dim} dimensions, so we do not try. Notice that no single number “means” anything you can name. The meaning is in how whole vectors compare, which is what the bars measure. Choose “2 numbers per word” to get a picture, at a price.</p>
            </>
          )}
        </div>
      </div>
      <p className="lab-note">
        Same model and corpus as tiny_word2vec.py: 28 short sentences about pets, food and computers, 70 different words. The random numbers differ from NumPy’s, so your values will not match the Python printout digit for digit, but the same pattern appears.
      </p>
    </Lab>
  )
}
