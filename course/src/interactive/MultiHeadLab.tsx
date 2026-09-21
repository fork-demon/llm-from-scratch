// Multi-head attention. Part A: four hand-made (illustrative!) head patterns on one sentence.
// Part B: the shapes, (T,D) -> (h,T,D/h) -> (T,D), with real small numbers.
import { useMemo, useState } from 'react'
import { Bars, Lab, MatrixView } from '../components/ui'
import { averagePatterns, HEAD_SENTENCE, illustrativeHeads, multiHeadAttention, shapesDemo } from '../lib/multihead'

const SHAPE_STEPS = ['1 · Q, K, V', '2 · Split into heads', '3 · Each head attends', '4 · Each head’s output', '5 · Concatenate', '6 · Mix with Wo'] as const

export function MultiHeadLab() {
  const [part, setPart] = useState<'patterns' | 'shapes'>('patterns')
  return (
    <Lab
      title="Several heads, several questions"
      goal={part === 'patterns'
        ? <>Keep “it” selected and click through the four heads. Each one gives “it” a <b>different</b> row of weights over the same sentence. Then click “one head for everything”.</>
        : <>Change the number of heads and step through. Watch the shapes: the D numbers per token are <b>split</b> between the heads, never multiplied.</>}
    >
      <div className="steps" role="group" aria-label="Part of the lab">
        <button className="step-btn" aria-pressed={part === 'patterns'} onClick={() => setPart('patterns')}>A · Four heads, four patterns</button>
        <button className="step-btn" aria-pressed={part === 'shapes'} onClick={() => setPart('shapes')}>B · Follow the shapes</button>
      </div>
      {part === 'patterns' ? <Patterns /> : <Shapes />}
    </Lab>
  )
}

function Patterns() {
  const heads = useMemo(illustrativeHeads, [])
  const blended = useMemo(() => averagePatterns(heads.map((h) => h.weights)), [heads])
  const [head, setHead] = useState(1) // index into heads, or heads.length for the blend
  const [sel, setSel] = useState(HEAD_SENTENCE.indexOf('it'))
  const words = HEAD_SENTENCE
  const T = words.length
  const isBlend = head === heads.length
  const W = isBlend ? blended : heads[head].weights
  const row = W[sel]

  const width = 720
  const gap = width / T
  const cx = (i: number) => gap * i + gap / 2

  return (
    <>
      <p className="lab-note" style={{ borderLeft: '3px solid var(--accent)', paddingLeft: 10 }}>
        <b>These four patterns were written by hand as illustrations.</b> They are not read out of a trained model. Real heads are learned, and most of them do not have a tidy label.
      </p>
      <div className="steps" role="group" aria-label="Choose a head">
        {heads.map((h, i) => <button key={h.id} className="step-btn" aria-pressed={head === i} onClick={() => setHead(i)}>{h.name}</button>)}
        <button className="step-btn" aria-pressed={isBlend} onClick={() => setHead(heads.length)}>one head for everything</button>
      </div>

      <svg viewBox={`0 0 ${width} 120`} style={{ width: '100%', display: 'block' }} role="img" aria-label={`Weights from “${words[sel]}”: ${words.slice(0, sel + 1).map((w, j) => `${w} ${(row[j] * 100).toFixed(0)} percent`).join(', ')}`}>
        {words.map((_, j) => {
          const w = row[j]
          if (w < 0.01) return null
          if (j === sel) return <circle key={j} cx={cx(j)} cy={100} r={12 + w * 8} fill="none" stroke="var(--accent)" strokeWidth={1 + w * 7} opacity={0.25 + w * 0.6} />
          const mid = (cx(sel) + cx(j)) / 2
          const lift = 100 - Math.min(95, Math.abs(cx(sel) - cx(j)) * 0.4)
          return <path key={j} d={`M ${cx(sel)} 100 Q ${mid} ${lift - 20} ${cx(j)} 100`} fill="none" stroke="var(--accent)" strokeWidth={1 + w * 9} opacity={0.25 + w * 0.7} strokeLinecap="round" />
        })}
      </svg>
      <div role="group" aria-label="Choose the token that is looking" style={{ display: 'grid', gridTemplateColumns: `repeat(${T}, minmax(0, 1fr))`, gap: 3, marginTop: -22, position: 'relative' }}>
        {words.map((w, j) => (
          <button key={j} className="step-btn" aria-pressed={j === sel} onClick={() => setSel(j)} style={{ opacity: j > sel ? 0.4 : 1, padding: '6px 1px', fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', minWidth: 0 }}>
            {w}
          </button>
        ))}
      </div>

      <div className="readout" aria-live="polite" style={{ margin: '14px 0' }}>
        {isBlend ? (
          <span>One softmax row must serve all four needs at once. Largest weight for “{words[sel]}”: <b>{(Math.max(...row) * 100).toFixed(0)}%</b>. Every pattern is diluted.</span>
        ) : (
          <>
            <span>this head’s question: <b>{heads[head].asks}</b></span>
            <span>{heads[head].note}</span>
          </>
        )}
      </div>

      <Bars items={words.slice(0, sel + 1).map((w, j) => ({ label: w, value: row[j] }))} max={1} digits={0} />

      <details className="deep" style={{ marginTop: 12 }}>
        <summary>Show the full {T}×{T} weight grid for this head</summary>
        <div className="details-body table-scroll">
          <MatrixView m={W} rows={words} cols={words} heat heatMax={1} tone="accent" digits={2} highlight={(i) => i === sel} caption="rows: who is looking, columns: who is looked at (causal mask on)" />
        </div>
      </details>
      {isBlend && <p className="lab-note">This row is the average of the four heads’ rows. A single head has a single row of weights per token, so it cannot say “90% on the previous token” <em>and</em> “80% on animal”: the row has to sum to 1.</p>}
    </>
  )
}

function Shapes() {
  const demo = useMemo(() => shapesDemo(), [])
  const [h, setH] = useState(2)
  const [step, setStep] = useState(0)
  const r = useMemo(() => multiHeadAttention(demo.x, demo.Wq, demo.Wk, demo.Wv, demo.Wo, h), [demo, h])
  const T = demo.x.length
  const D = demo.x[0].length
  const hd = D / h
  const rows = demo.tokens
  const headLabel = (k: number) => `head ${k} (columns ${k * hd}${hd > 1 ? `–${(k + 1) * hd - 1}` : ''})`

  return (
    <>
      <p className="lab-note"><b>Untrained:</b> T = {T} tokens, D = {D} numbers per token, seeded random weights. The patterns mean nothing. The shapes and the arithmetic are exactly those of <code>multi_head_attention()</code> in the repo.</p>
      <div className="steps" role="group" aria-label="Number of heads">
        {[1, 2, 4].map((n) => <button key={n} className="step-btn" aria-pressed={h === n} onClick={() => setH(n)}>{n} head{n > 1 ? 's' : ''}</button>)}
      </div>
      <div className="readout" aria-live="polite" style={{ marginBottom: 14 }}>
        <span>per-head width D/h = {D}/{h} = <b>{hd}</b></span>
        <span>weights in Wq, Wk, Wv, Wo: 4 × {D}×{D} = <b>{4 * D * D}</b> (same for any h)</span>
        <span>score tables: <b>{h}</b> × ({T}×{T})</span>
      </div>
      <div className="steps" role="tablist" aria-label="Steps">
        {SHAPE_STEPS.map((s, i) => <button key={s} role="tab" className="step-btn" aria-selected={i === step} onClick={() => setStep(i)}>{s}</button>)}
      </div>

      <div role="tabpanel" aria-live="polite">
        {step === 0 && (
          <>
            <p>Exactly as before: three full-width projections. Shape <span className="mono">({T}, {D})</span> each.</p>
            <div className="matrix-row">
              <MatrixView m={r.Q} rows={rows} tone="q" heat caption={<span className="q">Q  ({T}, {D})</span>} />
              <MatrixView m={r.K} rows={rows} tone="k" heat caption={<span className="k">K  ({T}, {D})</span>} />
              <MatrixView m={r.V} rows={rows} tone="v" heat caption={<span className="v">V  ({T}, {D})</span>} />
            </div>
          </>
        )}
        {step === 1 && (
          <>
            <p>Cut every row into {h} slice{h > 1 ? 's' : ''} of {hd} number{hd > 1 ? 's' : ''}. No arithmetic happens: it is only a reshape. <span className="mono">({T}, {D}) → ({h}, {T}, {hd})</span>. Shown for <span className="q">Q</span>; K and V are cut the same way.</p>
            <div className="matrix-row">
              {r.Qh.map((m, k) => <MatrixView key={k} m={m} rows={rows} tone="q" heat caption={<span className="q">Q, {headLabel(k)}</span>} />)}
            </div>
            <p className="lab-note">Compare with step 1: the numbers are the same, only grouped. Each head will see <em>only its own slice</em> of the query and key.</p>
          </>
        )}
        {step === 2 && (
          <>
            <p>Each head runs the attention you already know on its own slice: score, scale by √{hd}, mask, softmax. Shape <span className="mono">({h}, {T}, {T})</span>: {h === 1 ? 'one table of weights' : `${h} different tables of weights for the same sentence`}.</p>
            <div className="matrix-row">
              {r.weights.map((m, k) => <MatrixView key={k} m={m} rows={rows} cols={rows} heat heatMax={1} tone="accent" caption={`weights, head ${k}`} />)}
            </div>
            <p className="lab-note">Every row of every table sums to 1. {h > 1 && 'Look at the last row: the heads disagree about who matters. That is the point.'}</p>
          </>
        )}
        {step === 3 && (
          <>
            <p>Each head blends <em>its slice</em> of the values with <em>its own</em> weights. Shape <span className="mono">({h}, {T}, {hd})</span>.</p>
            <div className="matrix-row">
              {r.headOut.map((m, k) => <MatrixView key={k} m={m} rows={rows} tone="v" heat caption={<span className="v">output of head {k}  ({T}, {hd})</span>} />)}
            </div>
          </>
        )}
        {step === 4 && (
          <>
            <p>Glue the head outputs back side by side. <span className="mono">({h}, {T}, {hd}) → ({T}, {D})</span>. Again no arithmetic: columns 0–{hd - 1} come from head 0, and so on.</p>
            <MatrixView m={r.concat} rows={rows} tone="v" heat caption={`concat  (${T}, ${D})`} />
          </>
        )}
        {step === 5 && (
          <>
            <p>After concatenation each head’s findings sit in separate columns. One last learned matrix, <b>Wo</b>, mixes them so that later layers see a single combined vector. <span className="mono">({T}, {D}) @ ({D}, {D}) → ({T}, {D})</span>.</p>
            <div className="matrix-row">
              <MatrixView m={r.concat} rows={rows} tone="v" heat caption={`concat  (${T}, ${D})`} />
              <MatrixView m={demo.Wo} tone="neutral" heat caption={`Wo  (${D}, ${D})`} />
              <MatrixView m={r.out} rows={rows} tone="accent" heat caption={`out = concat @ Wo  (${T}, ${D})`} />
            </div>
            <p className="lab-note">Same shape in as out: <span className="mono">({T}, {D})</span>. From outside, multi-head attention is a drop-in replacement for the single head.</p>
          </>
        )}
      </div>
      <div className="btn-row" style={{ marginTop: 14 }}>
        <button className="btn small" disabled={step === 0} onClick={() => setStep(step - 1)}>Back</button>
        <button className="btn small primary" disabled={step === SHAPE_STEPS.length - 1} onClick={() => setStep(step + 1)}>Next step</button>
      </div>
    </>
  )
}
