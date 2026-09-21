// Where is a fact? Train a one-matrix associative memory in the browser, then
// look for the fact, damage the weights, and ask about things it never saw.
import { useEffect, useMemo, useRef, useState } from 'react'
import { Bars, Lab, MatrixView, Slider } from '../components/ui'
import { argmax, fmt, type Mat } from '../lib/math'
import { makeRng } from '../lib/rng'
import {
  ATTRIBUTES, DIM, FACTS, SUBJECTS, accuracy, addNoise, contributions, damageCurve, initWeights, knockOut, logitsFor,
  meanCorrectProb, meanLoss, predict, randomUnitVector, subjectVectors, trainStep, trainWithout, unseenSubjects, weightRms, weightsChanged,
} from '../lib/assoc'

const TARGET_STEPS = 400
const STEPS_PER_FRAME = 8
const TABS = ['1 · Train and ask', '2 · Find the fact', '3 · Damage the weights', '4 · Ask about the unknown'] as const
const DIM_LABELS = Array.from({ length: DIM }, (_, d) => `x${d}`)
const KO_AMOUNTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

export function KnowledgeLab() {
  const X = useMemo(() => subjectVectors(), [])
  const [W, setW] = useState<Mat>(() => initWeights())
  const [steps, setSteps] = useState(0)
  const [history, setHistory] = useState<number[]>(() => [meanLoss(initWeights(), subjectVectors())])
  const [running, setRunning] = useState(false)
  const [tab, setTab] = useState(0)
  const [subject, setSubject] = useState(0)
  const [leftOut, setLeftOut] = useState<Mat | null>(null)
  const [knock, setKnock] = useState(0)
  const [noise, setNoise] = useState(0)
  const [pattern, setPattern] = useState(3)
  const [unseenIdx, setUnseenIdx] = useState(0)
  const [invented, setInvented] = useState(0)

  // training loop: a few steps per animation frame, stops at the target and on unmount
  const state = useRef({ W, steps })
  state.current = { W, steps }
  useEffect(() => {
    if (!running) return
    let raf = 0
    const tick = () => {
      let cur = state.current.W
      let n = state.current.steps
      const losses: number[] = []
      for (let i = 0; i < STEPS_PER_FRAME && n < TARGET_STEPS; i++) {
        const r = trainStep(cur, X)
        cur = r.W
        n++
        losses.push(meanLoss(cur, X))
      }
      state.current = { W: cur, steps: n }
      setW(cur)
      setSteps(n)
      setHistory((h) => [...h, ...losses])
      if (n >= TARGET_STEPS) setRunning(false)
      else raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [running, X])

  const reset = () => {
    setRunning(false)
    setW(initWeights())
    setSteps(0)
    setHistory([meanLoss(initWeights(), X)])
    setLeftOut(null)
    setKnock(0)
    setNoise(0)
  }

  const trained = steps >= TARGET_STEPS
  const probs = predict(W, X[subject])
  const rms = weightRms(W)

  const damaged = useMemo(() => addNoise(knockOut(W, knock, pattern), noise, pattern + 1000), [W, knock, noise, pattern])
  const curve = useMemo(() => (trained ? damageCurve(W, X, KO_AMOUNTS, 'knockout') : null), [trained, W, X])

  const unseen = useMemo(() => {
    const base = unseenSubjects(X)
    if (invented === 0) return base
    const rng = makeRng(500 + invented)
    return [...base, { name: `Invented #${invented}`, note: 'a brand-new random vector nobody trained on', truth: 'nothing', vector: randomUnitVector(rng) }]
  }, [X, invented])
  const u = unseen[Math.min(unseenIdx, unseen.length - 1)]
  const uProbs = predict(W, u.vector)
  const uBest = argmax(uProbs)

  const needTraining = !trained && (
    <p className="lab-note"><b>The model is {steps === 0 ? 'untrained' : 'only partly trained'}.</b> Go to tab 1 and press Train first, otherwise every answer here is close to a uniform guess.</p>
  )

  return (
    <Lab
      title="Where does a fact live?"
      goal={<>A model with <b>one weight matrix</b> and nothing else learns 8 country → capital facts. Train it, then go looking for the facts inside it. Work through the four tabs in order.</>}
    >
      <div className="steps" role="tablist" aria-label="Parts of the experiment">
        {TABS.map((t, i) => <button key={t} role="tab" className="step-btn" aria-selected={tab === i} onClick={() => setTab(i)}>{t}</button>)}
      </div>

      {tab === 0 && (
        <div role="tabpanel">
          <p>Each country is a fixed list of {DIM} numbers (a stand-in for an <em>embedding</em>). The model multiplies it by a {DIM}×{FACTS.length} matrix to get 8 logits, one per capital, then softmax. That matrix, 80 numbers, is <b>everything the model has</b>.</p>
          <div className="btn-row" style={{ marginBottom: 12 }}>
            <button className="btn small primary" disabled={running || trained} onClick={() => setRunning(true)}>{steps === 0 ? 'Train (400 steps of gradient descent)' : trained ? 'Trained' : 'Continue training'}</button>
            {running && <button className="btn small" onClick={() => setRunning(false)}>Pause</button>}
            <button className="btn small" onClick={reset}>Reset to random weights</button>
          </div>
          <div className="readout" aria-live="polite">
            <span>step <b>{steps}</b> / {TARGET_STEPS}</span>
            <span>loss <b>{fmt(history[history.length - 1], 3)}</b> {steps === 0 && <>(= ln 8, a pure guess)</>}</span>
            <span>facts recalled <b>{Math.round(accuracy(W, X) * 8)}</b> / 8</span>
            <span>avg. confidence in the right capital <b>{(meanCorrectProb(W, X) * 100).toFixed(0)}%</b></span>
          </div>
          <LossCurve history={history} />
          <p style={{ marginTop: 14 }}>Ask it something:</p>
          <div className="steps" role="group" aria-label="Choose a country to ask about">
            {SUBJECTS.map((s, i) => <button key={s} className="step-btn" aria-pressed={subject === i} onClick={() => setSubject(i)}>{s}</button>)}
          </div>
          <p className="mono" style={{ fontSize: 13.5 }}>“The capital of {SUBJECTS[subject]} is ___”</p>
          <div aria-live="polite">
            <Bars items={ATTRIBUTES.map((a, j) => ({ label: a, value: probs[j], tone: j === subject ? 'accent' : 'neutral' }))} max={1} />
          </div>
          <p className="lab-note">The training data (8 pairs) is thrown away after training. From tab 2 onwards, the only thing left is the matrix.</p>
        </div>
      )}

      {tab === 1 && (
        <div role="tabpanel">
          {needTraining}
          <p>Here is the whole model. Rows are the {DIM} input numbers, columns are the capitals. Try to find the cell that holds “{SUBJECTS[subject]} → {ATTRIBUTES[subject]}”.</p>
          <div className="steps" role="group" aria-label="Choose a fact to look for">
            {SUBJECTS.map((s, i) => <button key={s} className="step-btn" aria-pressed={subject === i} onClick={() => { setSubject(i); setLeftOut(null) }}>{s} → {ATTRIBUTES[i]}</button>)}
          </div>
          <MatrixView m={W} rows={DIM_LABELS} cols={ATTRIBUTES} heat tone="accent" digits={1} caption="W: all 80 weights. No row says “France”. No cell says “France → Paris”." highlight={(_, j) => j === subject} />
          <p>You cannot find it, because it is not in one place. The score for {ATTRIBUTES[subject]} is a dot product that uses <b>every</b> weight in the outlined column:</p>
          <div className="table-scroll">
            <table className="plain mono" style={{ fontSize: 13 }}>
              <thead><tr><th>input</th>{DIM_LABELS.map((d) => <th key={d}>{d}</th>)}<th>sum</th></tr></thead>
              <tbody>
                <tr><td>{SUBJECTS[subject]} vector</td>{X[subject].map((v, d) => <td key={d}>{fmt(v)}</td>)}<td /></tr>
                <tr><td>× W[·, {ATTRIBUTES[subject]}]</td>{W.map((r, d) => <td key={d}>{fmt(r[subject])}</td>)}<td /></tr>
                <tr><td>= contribution</td>{contributions(W, X[subject], subject).map((v, d) => <td key={d}>{fmt(v)}</td>)}<td><b>{fmt(logitsFor(W, X[subject])[subject])}</b></td></tr>
              </tbody>
            </table>
          </div>
          <p className="lab-note">And softmax compares that logit with the other seven, so the other seven columns matter too. Answering <em>any</em> question uses all 80 weights. Pick another fact: same 80 weights, different input.</p>
          <p><b>A sharper test.</b> If “{SUBJECTS[subject]} → {ATTRIBUTES[subject]}” owned particular weights, then training <em>without</em> that fact should leave all other weights the same.</p>
          <button className="btn small" disabled={!trained} onClick={() => setLeftOut(trainWithout(initWeights(), X, subject, TARGET_STEPS))}>Retrain from the same start, without this one fact</button>
          {leftOut && (
            <div aria-live="polite" style={{ marginTop: 12 }}>
              <MatrixView m={W.map((r, d) => r.map((w, j) => leftOut[d][j] - w))} rows={DIM_LABELS} cols={ATTRIBUTES} heat tone="k" digits={1} caption="difference: (weights trained without the fact) − (weights trained with it)" />
              <div className="readout">
                <span>weights that moved by more than 10% of a typical weight: <b>{weightsChanged(W, leftOut, 0.1 * rms)}</b> of 80</span>
                <span>P({ATTRIBUTES[subject]} | {SUBJECTS[subject]}) without the fact: <b>{(predict(leftOut, X[subject])[subject] * 100).toFixed(0)}%</b></span>
                <span>other 7 facts still recalled: <b>{X.filter((x, i) => i !== subject && argmax(predict(leftOut, x)) === i).length}</b> / 7</span>
              </div>
              <p className="lab-note">One fact fewer, and weights shift in many rows and several columns, most in the {ATTRIBUTES[subject]} column but far from only there. A fact is a small adjustment to many shared weights, not an entry.</p>
            </div>
          )}
        </div>
      )}

      {tab === 2 && (
        <div role="tabpanel">
          {needTraining}
          <p>In a database, deleting 30% of the rows deletes exactly 30% of the facts and leaves the rest perfect. Predict what happens here, then drag.</p>
          <div className="controls">
            <Slider label="Knock out (set to zero) this share of weights" value={knock} min={0} max={1} step={0.05} onChange={setKnock} format={(v) => `${Math.round(v * 100)}% = ${Math.round(v * 80)} weights`} />
            <Slider label="Add random noise to every weight" value={noise} min={0} max={3} step={0.25} onChange={setNoise} format={(v) => `${v.toFixed(2)} × typical weight`} />
          </div>
          <div className="btn-row" style={{ marginBottom: 10 }}>
            <button className="btn small" onClick={() => setPattern(pattern + 1)}>Different random damage pattern</button>
            <button className="btn small" onClick={() => { setKnock(0); setNoise(0) }}>Repair</button>
          </div>
          <div aria-live="polite">
            <div className="readout">
              <span>facts still recalled <b>{Math.round(accuracy(damaged, X) * 8)}</b> / 8</span>
              <span>avg. confidence in the right capital <b>{(meanCorrectProb(damaged, X) * 100).toFixed(0)}%</b> (undamaged: {(meanCorrectProb(W, X) * 100).toFixed(0)}%)</span>
            </div>
            <p className="muted" style={{ fontSize: 13.5, margin: '10px 0 2px' }}>Probability given to the correct capital, per fact. “wrong” means another capital now scores higher.</p>
            <Bars items={FACTS.map((f, i) => { const p = predict(damaged, X[i]); const ok = argmax(p) === i; return { label: `${f.subject}${ok ? '' : ' (wrong)'}`, value: p[i], dim: !ok, tone: 'accent' as const } })} max={1} />
          </div>
          <MatrixView m={damaged} rows={DIM_LABELS} cols={ATTRIBUTES} heat heatMax={Math.max(1e-9, ...W.flat().map(Math.abs))} tone="accent" digits={1} caption="the damaged matrix" highlight={(i, j) => knock > 0 && damaged[i][j] === 0} />
          {curve && <DamageChart curve={curve} at={knock} />}
          <p className="lab-note">Small damage: every fact survives, each a little less confident. Large damage: facts fail one after another, in an order that depends on the random pattern, not on which “record” you hit. This is what <b>distributed</b> storage looks like.</p>
        </div>
      )}

      {tab === 3 && (
        <div role="tabpanel">
          {needTraining}
          <p>Now ask about subjects that were <b>never in the training data</b>. A lookup table would say “key not found”. What can this model say?</p>
          <div className="steps" role="group" aria-label="Choose an unseen subject">
            {unseen.map((s, i) => <button key={s.name} className="step-btn" aria-pressed={unseenIdx === i} onClick={() => setUnseenIdx(i)}>{s.name}</button>)}
            <button className="btn small" onClick={() => { setInvented(invented + 1); setUnseenIdx(4) }}>Invent another</button>
          </div>
          <p className="mono" style={{ fontSize: 13.5 }}>“The capital of {u.name} is ___” <span className="muted">({u.note})</span></p>
          <div aria-live="polite">
            <Bars items={ATTRIBUTES.map((a, j) => ({ label: a, value: uProbs[j], tone: j === uBest ? 'accent' : 'neutral' }))} max={1} />
            <div className="readout">
              <span>model says <b>{ATTRIBUTES[uBest]}</b> with <b>{(uProbs[uBest] * 100).toFixed(0)}%</b></span>
              <span>correct answer: <b>{u.truth === 'nothing' ? 'there is none' : u.truth}</b></span>
              <span>probabilities sum to <b>{fmt(uProbs.reduce((a, b) => a + b, 0))}</b></span>
            </div>
          </div>
          <p className="lab-note">There is no “I do not know” output. Softmax must spread 100% over the 8 capitals it has, and a vector that resembles Germany gets Germany’s answer. The same arithmetic that recalled true facts in tab 1 produces this. Nothing inside the model marks the difference. (A toy: a real LLM can output the <em>words</em> “I am not sure”, but only if those words are the likely continuation.)</p>
        </div>
      )}
    </Lab>
  )
}

function LossCurve({ history }: { history: number[] }) {
  const W = 560, H = 120, L = 34, B = 18
  const max = Math.log(8) * 1.05
  const x = (i: number) => L + (i / TARGET_STEPS) * (W - L - 6)
  const y = (v: number) => 6 + (1 - v / max) * (H - B - 6)
  const pts = history.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  return (
    <div className="table-scroll"><svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', minWidth: 460, maxWidth: W, display: 'block', marginTop: 10 }} role="img" aria-label={`Training loss curve, currently ${history[history.length - 1].toFixed(3)} after ${history.length - 1} steps`}>
      <line className="axis" x1={L} y1={H - B} x2={W - 6} y2={H - B} stroke="var(--rule-strong)" />
      <line className="axis" x1={L} y1={6} x2={L} y2={H - B} stroke="var(--rule-strong)" />
      <line className="gridline" x1={L} y1={y(Math.log(8))} x2={W - 6} y2={y(Math.log(8))} stroke="var(--rule)" strokeDasharray="3 3" />
      <text x={L - 4} y={y(Math.log(8)) + 4} textAnchor="end" fontSize={10}>2.08</text>
      <text x={L - 4} y={H - B + 3} textAnchor="end" fontSize={10}>0</text>
      <text x={W - 6} y={H - 4} textAnchor="end" fontSize={10}>training step (0 to {TARGET_STEPS})</text>
      <text x={L + 6} y={16} fontSize={10}>loss</text>
      {history.length > 1 && <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={2} />}
    </svg></div>
  )
}

function DamageChart({ curve, at }: { curve: { amount: number; accuracy: number; correctProb: number }[]; at: number }) {
  const W = 560, H = 170, L = 40, B = 30
  const x = (a: number) => L + a * (W - L - 10)
  const y = (v: number) => 8 + (1 - v) * (H - B - 8)
  const line = (k: 'accuracy') => curve.map((c) => `${x(c.amount).toFixed(1)},${y(c[k]).toFixed(1)}`).join(' ')
  return (
    <figure style={{ margin: '14px 0 0' }}>
      <div className="table-scroll"><svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', minWidth: 460, maxWidth: W, display: 'block' }} role="img" aria-label={`Average over 40 random damage patterns. ${curve.map((c) => `${Math.round(c.amount * 100)} percent knocked out: ${Math.round(c.accuracy * 100)} percent of facts recalled`).join('; ')}`}>
        {[0, 0.5, 1].map((v) => <g key={v}><line className="gridline" x1={L} y1={y(v)} x2={W - 10} y2={y(v)} stroke="var(--rule)" /><text x={L - 5} y={y(v) + 4} textAnchor="end" fontSize={10}>{v * 100}%</text></g>)}
        {[0, 0.25, 0.5, 0.75, 1].map((a) => <text key={a} x={x(a)} y={H - B + 14} textAnchor="middle" fontSize={10}>{a * 100}%</text>)}
        <text x={(W + L) / 2} y={H - 2} textAnchor="middle" fontSize={10}>share of weights knocked out</text>
        <polyline points={line('accuracy')} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
        <polyline points={`${x(0)},${y(1)} ${x(1)},${y(0)}`} fill="none" stroke="var(--ink-3)" strokeWidth={1.5} strokeDasharray="4 4" />
        <line x1={x(at)} y1={8} x2={x(at)} y2={H - B} stroke="var(--ink)" strokeWidth={1} />
      </svg></div>
      <figcaption className="muted" style={{ fontSize: 13 }}>Solid line: share of the 8 facts this model still recalls, averaged over 40 random damage patterns. Dashed line: what a table of records would do (lose 30% of rows, lose 30% of facts). The vertical line is your slider.</figcaption>
    </figure>
  )
}
