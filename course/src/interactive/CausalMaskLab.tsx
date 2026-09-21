// The causal mask: scores -> masked scores -> weights, what leaks without it,
// and why it turns one sequence into T training examples.
import { useMemo, useState } from 'react'
import { Bars, Lab, MatrixView, Slider } from '../components/ui'
import { argmax, fmt } from '../lib/math'
import { cheatingQKV, futureWeight, leakOnAnswer, MASK_NEXT, MASK_TOKENS, randomQKV, runMask, trainingExamples } from '../lib/causalMask'

type Source = 'random' | 'cheat'
const VIEWS = ['1 · Scores', '2 · Apply the mask', '3 · Softmax → weights', 'What leaks?', 'One pass, T examples'] as const

export function CausalMaskLab() {
  const [source, setSource] = useState<Source>('random')
  const [strength, setStrength] = useState(4)
  const [masked, setMasked] = useState(true)
  const [sel, setSel] = useState(2)
  const [view, setView] = useState(0)

  const tokens = MASK_TOKENS
  const T = tokens.length
  const qkv = useMemo(() => (source === 'random' ? randomQKV() : cheatingQKV(strength)), [source, strength])
  const free = useMemo(() => runMask(qkv, false), [qkv])
  const causal = useMemo(() => runMask(qkv, true), [qkv])
  const live = masked ? causal : free
  const target = sel + 1 < T ? tokens[sel + 1] : MASK_NEXT
  const examples = trainingExamples(tokens, MASK_NEXT)
  const leakFree = leakOnAnswer(free.weights)
  const peeked = futureWeight(live.weights)[sel]

  return (
    <Lab
      title="The causal mask"
      goal={<>Click a token: it becomes the one that is <b>looking</b>, and its job is to predict the token after it. Switch the mask off and see where its attention goes. Then open “What leaks?”.</>}
    >
      <div className="steps" role="group" aria-label="Where the scores come from">
        <button className="step-btn" aria-pressed={source === 'random'} onClick={() => setSource('random')}>random, untrained scores</button>
        <button className="step-btn" aria-pressed={source === 'cheat'} onClick={() => setSource('cheat')}>a head that learned to cheat</button>
      </div>
      {source === 'random' ? (
        <p className="lab-note"><b>Untrained:</b> Q and K are seeded random numbers, so who-attends-to-whom is meaningless here. The masking arithmetic is exactly what a real model does.</p>
      ) : (
        <>
          <p className="lab-note">This is what gradient descent would find if nothing stopped it: a query at position t that matches the key of position t+1. It was built by hand to make the point. The slider is the score that training has pushed into the “answer” cell.</p>
          <div className="controls"><Slider label="how far training pushed the cheat (score)" value={strength} min={0} max={8} step={0.5} onChange={setStrength} /></div>
        </>
      )}

      <div role="group" aria-label="Choose the token that is looking" style={{ display: 'grid', gridTemplateColumns: `repeat(${T + 1}, 1fr)`, gap: 6, margin: '6px 0 10px' }}>
        {tokens.map((w, j) => (
          <button key={j} className="step-btn" aria-pressed={j === sel} onClick={() => setSel(j)} style={{ opacity: masked && j > sel ? 0.4 : 1 }}>
            {w}
          </button>
        ))}
        <span className="step-btn" aria-hidden style={{ opacity: 0.4, borderStyle: 'dashed', textAlign: 'center' }}>{MASK_NEXT}</span>
      </div>
      <div className="readout" aria-live="polite" style={{ marginBottom: 14 }}>
        <span>“{tokens[sel]}” must predict: <b>{target}</b></span>
        <span>it can see: <b>{tokens.filter((_, j) => !masked || j <= sel).join(' ')}</b></span>
        <span>weight on tokens it should not know yet: <b>{(peeked * 100).toFixed(0)}%</b></span>
      </div>

      <div className="controls">
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}>
          <input type="checkbox" checked={masked} onChange={(e) => setMasked(e.target.checked)} /> Causal mask on
        </label>
      </div>

      <div className="steps" role="tablist" aria-label="Views">
        {VIEWS.map((s, i) => <button key={s} role="tab" className="step-btn" aria-selected={i === view} onClick={() => setView(i)}>{s}</button>)}
      </div>

      <div role="tabpanel" aria-live="polite">
        {view === 0 && (
          <>
            <p>The T×T score table from the last lesson, already divided by √d. Row = who is looking. Column = who is looked at. Everything <b>right of the diagonal</b> is a token looking at its own future.</p>
            <MatrixView m={free.scores} rows={tokens} cols={tokens} heat tone="accent" caption="scores = Q Kᵀ / √d" highlight={(i, j) => i === sel && j > sel} />
            <p className="lab-note">Outlined: the cells where “{tokens[sel]}” scores tokens that come after it.</p>
          </>
        )}

        {view === 1 && (
          <>
            <p>{masked ? <>Every cell right of the diagonal is overwritten with −∞. Nothing else changes: compare with view 1.</> : <>The mask is <b>off</b>: scores pass through untouched. Tick “Causal mask on” to see the staircase appear.</>}</p>
            <MatrixView m={live.scores} rows={tokens} cols={tokens} heat tone="accent" caption={masked ? 'masked scores: a staircase of allowed cells' : 'scores (no mask)'} highlight={(i) => i === sel} />
            <p className="lab-note">Why −∞ and not 0? Because softmax computes e<sup>score</sup>. A score of 0 gives e⁰ = 1, which is a perfectly healthy weight. Only e<sup>−∞</sup> is exactly 0.</p>
          </>
        )}

        {view === 2 && (
          <>
            <p>Softmax, row by row. {masked ? 'Hidden tokens get weight exactly 0, and the visible ones share 100% between them.' : 'With no mask, later tokens get real weight.'}</p>
            <Bars items={tokens.map((w, j) => ({ label: w, value: live.weights[sel][j], dim: live.weights[sel][j] === 0 }))} max={1} />
            <MatrixView m={live.weights} rows={tokens} cols={tokens} heat heatMax={1} tone="accent" caption="attention weights: every row still sums to 1" highlight={(i) => i === sel} />
            <p className="lab-note">Row sums: <span className="mono">{live.weights.map((r) => fmt(r.reduce((a, b) => a + b, 0))).join(', ')}</span>. The first row is always <span className="mono">1, 0, 0, …</span> under the mask: the first token has only itself to look at.</p>
          </>
        )}

        {view === 3 && (
          <>
            <p>Each row is graded on predicting the <b>next</b> token. How much of its attention sits on that very token?</p>
            <div className="table-scroll">
              <table className="plain" style={{ fontSize: 14 }}>
                <thead><tr><th>looking token</th><th>must predict</th><th>weight on the answer, no mask</th><th>with mask</th>{source === 'cheat' && <th>token it mostly read, no mask</th>}</tr></thead>
                <tbody>
                  {leakFree.map((w, t) => (
                    <tr key={t} style={{ outline: t === sel ? '2px solid var(--accent)' : undefined }}>
                      <td className="mono">{tokens[t]}</td>
                      <td className="mono"><b>{tokens[t + 1]}</b></td>
                      <td className="mono" style={{ color: w > 0.5 ? 'var(--bad)' : undefined }}>{(w * 100).toFixed(1)}%{w > 0.5 ? ' (copying)' : ''}</td>
                      <td className="mono">{(causal.weights[t][t + 1] * 100).toFixed(1)}%</td>
                      {source === 'cheat' && <td className="mono">{tokens[argmax(free.out[t])]}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {source === 'random' ? (
              <p className="lab-note">These weights are random, so the leak is modest. But any weight above 0 is a channel through which the answer reaches the prediction, and training strengthens whatever lowers the loss. Switch to <b>“a head that learned to cheat”</b> to see where that ends.</p>
            ) : (
              <p className="lab-note">At score {fmt(strength, 1)} the unmasked model puts {(leakFree[0] * 100).toFixed(0)}% of each row on the answer and simply copies it. Its training loss would be near zero and it would have learned <em>nothing about language</em>. With the mask, that cell is −∞ whatever training does: the cheat is impossible, so the only way to lower the loss is to actually predict.</p>
            )}
          </>
        )}

        {view === 4 && (
          <>
            <p>With the mask on, row t has only seen tokens 0…t. So <b>every row is an honest prediction problem</b>, and all {T} are computed in the same forward pass:</p>
            <div className="table-scroll">
              <table className="plain" style={{ fontSize: 14 }}>
                <thead><tr><th>row</th><th>what this row can see</th><th>target</th></tr></thead>
                <tbody>
                  {examples.map((ex, t) => (
                    <tr key={t} style={{ outline: t === sel ? '2px solid var(--accent)' : undefined }}>
                      <td className="mono">{t}</td>
                      <td>{ex.context.map((w, j) => <span key={j} className="token">{w}</span>)}</td>
                      <td><span className="token" style={{ borderColor: 'var(--accent)' }}>{ex.target}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="readout" style={{ marginTop: 10 }}>
              <span>forward passes needed with the mask: <b>1</b></span>
              <span>without it (feed each prefix separately): <b>{T}</b></span>
              <span>for GPT-2’s 1024-token sequences: <b>1 vs 1024</b></span>
            </div>
            <p className="lab-note">{masked ? 'The mask is on: all of these examples are valid.' : 'The mask is off: every example except the last one is contaminated, because its row could see its own target.'}</p>
          </>
        )}
      </div>
    </Lab>
  )
}
