// The signature interactive of the course: single-head attention you can
// click through, step by step, with every number visible and editable.
import { useMemo, useState } from 'react'
import { Bars, Lab, MatrixView } from '../components/ui'
import { attentionFromQKV, fmt, matmul, type Mat } from '../lib/math'
import { DIMS, HAND_VOCAB, handWeights, randomEmbedding, randomWeights, splitWords } from '../lib/attentionDemo'

type Mode = 'river' | 'money' | 'custom'
const STEPS = ['1 · Make Q, K, V', '2 · Score: q · k', '3 · Scale by √d', '4 · Softmax', '5 · Blend the values'] as const

export function AttentionPlayground() {
  const [mode, setMode] = useState<Mode>('river')
  const [text, setText] = useState('the cat sat on the mat')
  const [selected, setSelected] = useState(2)
  const [step, setStep] = useState(0)
  const [causal, setCausal] = useState(true)
  const [scaled, setScaled] = useState(true)
  // learner edits to Q / K / V, keyed by which sentence they belong to
  const [edits, setEdits] = useState<{ key: string; Q?: Mat; K?: Mat; V?: Mat }>({ key: '' })

  const words = mode === 'custom' ? splitWords(text) : ['the', mode, 'bank']
  const key = `${mode}:${words.join(' ')}`

  const base = useMemo(() => {
    const x = words.map((w) => (mode === 'custom' ? randomEmbedding(w) : HAND_VOCAB[w]))
    const W = mode === 'custom' ? randomWeights(7) : handWeights()
    return { x, Q: matmul(x, W.Wq), K: matmul(x, W.Wk), V: matmul(x, W.Wv) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  const live = edits.key === key ? edits : { key }
  const Q = live.Q ?? base.Q
  const K = live.K ?? base.K
  const V = live.V ?? base.V
  const edited = !!(live.Q || live.K || live.V)
  const res = useMemo(() => attentionFromQKV(Q, K, V, { causal, scale: scaled }), [Q, K, V, causal, scaled])

  const T = words.length
  const sel = Math.min(selected, T - 1)
  const d = K[0]?.length ?? 4
  const editCell = (which: 'Q' | 'K' | 'V', src: Mat) => (i: number, j: number, value: number) => {
    const next = src.map((r) => r.slice())
    next[i][j] = value
    setEdits({ ...live, key, [which]: next })
  }

  if (T === 0) {
    return (
      <Lab title="Attention, one step at a time">
        <p>Type at least one word.</p>
        <input className="input" value={text} onChange={(e) => setText(e.target.value)} aria-label="Sentence" />
      </Lab>
    )
  }

  // geometry for the arc diagram
  const W = 640
  const gap = W / T
  const cx = (i: number) => gap * i + gap / 2
  const weights = res.weights[sel]

  return (
    <Lab
      title="Attention, one step at a time"
      goal={<>Pick a token. It becomes the one that is <b>looking</b>. Then walk through steps 1 to 5 and watch its row of numbers turn into a new vector. Edit any <span className="q">Q</span>, <span className="k">K</span> or <span className="v">V</span> number to see what changes.</>}
    >
      <div className="steps" role="group" aria-label="Choose a sentence">
        <button className="step-btn" aria-pressed={mode === 'river'} onClick={() => { setMode('river'); setSelected(2) }}>“the river bank”</button>
        <button className="step-btn" aria-pressed={mode === 'money'} onClick={() => { setMode('money'); setSelected(2) }}>“the money bank”</button>
        <button className="step-btn" aria-pressed={mode === 'custom'} onClick={() => setMode('custom')}>your own sentence</button>
      </div>

      {mode === 'custom' ? (
        <>
          <label className="sr-only" htmlFor="att-text">Your sentence (up to 8 words)</label>
          <input id="att-text" className="input" value={text} onChange={(e) => setText(e.target.value)} />
          <p className="lab-note" style={{ marginTop: 8 }}>
            <b>Honest warning:</b> for your own text we use random, <em>untrained</em> vectors and weights. The mechanics are exactly right, but the pattern of who-attends-to-whom is meaningless. Training is what makes the pattern useful.
          </p>
        </>
      ) : (
        <p className="lab-note">
          Here each token is a hand-made 4-number vector: <span className="mono">[{DIMS.join(', ')}]</span>. “bank” is ambiguous on purpose: <span className="mono">[0.5, 0.5, 0.9, 1]</span>. Switch between the two sentences and watch what happens to it.
        </p>
      )}

      {/* token row with attention arcs */}
      {step >= 3 && <svg viewBox={`0 0 ${W} 150`} style={{ width: '100%', maxWidth: W, display: 'block', margin: '8px auto 0' }} role="img" aria-label={`Attention from “${words[sel]}”: ${words.map((w, j) => `${w} ${(weights[j] * 100).toFixed(0)} percent`).join(', ')}`}>
        {step >= 3 && words.map((_, j) => {
          const w = weights[j]
          if (w < 0.005) return null
          if (j === sel) return <circle key={j} cx={cx(j)} cy={96} r={20 + w * 8} fill="none" stroke="var(--accent)" strokeWidth={1 + w * 7} opacity={0.25 + w * 0.6} />
          const mid = (cx(sel) + cx(j)) / 2
          const lift = 96 - Math.min(90, Math.abs(cx(sel) - cx(j)) * 0.45)
          return <path key={j} d={`M ${cx(sel)} 96 Q ${mid} ${lift - 20} ${cx(j)} 96`} fill="none" stroke="var(--accent)" strokeWidth={1 + w * 9} opacity={0.25 + w * 0.7} strokeLinecap="round" />
        })}
        {words.map((w, j) => (
          <g key={j}>
            <text x={cx(j)} y={142} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)', fontFamily: 'var(--mono)' }}>
              {step >= 3 ? `${(weights[j] * 100).toFixed(0)}%` : causal && j > sel ? 'hidden' : ''}
            </text>
          </g>
        ))}
      </svg>}
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${T}, 1fr)`, gap: 6, maxWidth: W, margin: step >= 3 ? '-64px auto 28px' : '12px auto 16px', position: 'relative' }} role="group" aria-label="Choose the token that is looking">
        {words.map((w, j) => (
          <button key={j} className="step-btn" aria-pressed={j === sel} onClick={() => setSelected(j)} style={{ opacity: causal && j > sel ? 0.45 : 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {w}
          </button>
        ))}
      </div>

      <div className="controls">
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}>
          <input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} /> Causal mask (cannot look at later tokens)
        </label>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}>
          <input type="checkbox" checked={scaled} onChange={(e) => setScaled(e.target.checked)} /> Divide scores by √d (√{d} = {fmt(Math.sqrt(d))})
        </label>
      </div>

      <div className="steps" role="tablist" aria-label="Steps of attention">
        {STEPS.map((s, i) => (
          <button key={s} role="tab" className="step-btn" aria-selected={i === step} onClick={() => setStep(i)}>{s}</button>
        ))}
      </div>

      <div role="tabpanel" aria-live="polite">
        {step === 0 && (
          <>
            <p>Every token gets three vectors, each made by multiplying its embedding by a learned matrix. One row per token. <b>These cells are editable.</b></p>
            <div className="matrix-row">
              <MatrixView m={Q} rows={words} tone="q" heat caption={<span className="q">Q: “what am I looking for?”</span>} onEdit={editCell('Q', Q)} highlight={(i) => i === sel} />
              <MatrixView m={K} rows={words} tone="k" heat caption={<span className="k">K: “what can I be matched on?”</span>} onEdit={editCell('K', K)} />
              <MatrixView m={V} rows={words} tone="v" heat caption={<span className="v">V: “what do I pass along?”</span>} onEdit={editCell('V', V)} />
            </div>
            <p className="lab-note">The outlined row is the query of “{words[sel]}”, the token doing the looking.</p>
          </>
        )}

        {step === 1 && (
          <>
            <p>Take the <span className="q">query</span> of “{words[sel]}” and dot it with the <span className="k">key</span> of every token. Big number = good match.</p>
            <div className="table-scroll">
              <table className="plain mono" style={{ fontSize: 13.5 }}>
                <thead><tr><th>token j</th><th><span className="q">q</span> of “{words[sel]}”</th><th><span className="k">k</span> of token j</th><th>q · k (multiply pairs, add up)</th></tr></thead>
                <tbody>
                  {words.map((w, j) => (
                    <tr key={j} style={{ opacity: causal && j > sel ? 0.4 : 1 }}>
                      <td>{w}</td>
                      <td className="q">[{Q[sel].map((n) => fmt(n, 1)).join(', ')}]</td>
                      <td className="k">[{K[j].map((n) => fmt(n, 1)).join(', ')}]</td>
                      <td>{Q[sel].map((q, i) => `${fmt(q, 1)}×${fmt(K[j][i], 1)}`).join(' + ')} = <b>{fmt(res.raw[sel][j])}</b></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="lab-note">Doing this for every token at once is the matrix product <span className="mono">Q × Kᵀ</span>: a {T}×{T} table of scores.</p>
            <MatrixView m={res.raw} rows={words} cols={words} heat tone="accent" caption="Q × Kᵀ (rows: who is looking, columns: who is looked at)" highlight={(i) => i === sel} />
          </>
        )}

        {step === 2 && (
          <>
            <p>{scaled ? <>Divide every score by √d = {fmt(Math.sqrt(d))}. This keeps scores in a gentle range so softmax does not become all-or-nothing.</> : <>Scaling is <b>off</b>. Compare the softmax step with it on and off: without scaling, the biggest score grabs almost everything.</>}{causal && <> Then the mask: every token to the right of the looker gets −∞, which softmax will turn into exactly 0.</>}</p>
            <MatrixView m={res.scores} rows={words} cols={words} heat tone="accent" caption={`scores${scaled ? ' ÷ √d' : ''}${causal ? ', masked' : ''}`} highlight={(i) => i === sel} />
          </>
        )}

        {step === 3 && (
          <>
            <p>Softmax turns the row of scores for “{words[sel]}” into weights that are positive and add up to 100%. This is how much it will listen to each token.</p>
            <Bars items={words.map((w, j) => ({ label: w, value: weights[j], dim: weights[j] === 0 }))} max={1} />
            <MatrixView m={res.weights} rows={words} cols={words} heat heatMax={1} tone="accent" caption="attention weights: every row sums to 1" highlight={(i) => i === sel} />
          </>
        )}

        {step === 4 && (
          <>
            <p>Finally, blend the <span className="v">values</span> using those weights. The result is the new vector for “{words[sel]}”: itself, plus what it gathered from the others.</p>
            <div className="table-scroll">
              <table className="plain mono" style={{ fontSize: 13.5 }}>
                <thead><tr><th>token</th><th>weight</th><th><span className="v">value</span></th><th>weight × value</th></tr></thead>
                <tbody>
                  {words.map((w, j) => (
                    <tr key={j} style={{ opacity: weights[j] < 0.005 ? 0.4 : 1 }}>
                      <td>{w}</td><td>{fmt(weights[j])}</td>
                      <td className="v">[{V[j].map((n) => fmt(n)).join(', ')}]</td>
                      <td>[{V[j].map((n) => fmt(n * weights[j])).join(', ')}]</td>
                    </tr>
                  ))}
                  <tr><td colSpan={3}><b>sum = new vector for “{words[sel]}”</b></td><td><b>[{res.out[sel].map((n) => fmt(n)).join(', ')}]</b></td></tr>
                </tbody>
              </table>
            </div>
            {mode !== 'custom' && !edited && sel === 2 && (
              <div className="readout" style={{ marginTop: 10 }}>
                <span>“bank” before: watery <b>0.50</b>, financial <b>0.50</b></span>
                <span>after: watery <b>{fmt(res.out[2][0])}</b>, financial <b>{fmt(res.out[2][1])}</b></span>
                <span>→ {mode === 'river' ? 'pulled toward the river meaning' : 'pulled toward the money meaning'}</span>
              </div>
            )}
          </>
        )}
      </div>

      <div className="btn-row" style={{ marginTop: 16 }}>
        <button className="btn small" disabled={step === 0} onClick={() => setStep(step - 1)}>Back</button>
        <button className="btn small primary" disabled={step === STEPS.length - 1} onClick={() => setStep(step + 1)}>Next step</button>
        {edited && <button className="btn small" onClick={() => setEdits({ key: '' })}>Reset my edits</button>}
      </div>
    </Lab>
  )
}
