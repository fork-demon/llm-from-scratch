// Forward pass, backward pass and one update on a tiny network, with every number visible.
// 2 inputs -> 2 hidden ReLU -> 2 logits -> softmax -> cross-entropy. Maths: src/lib/tinyMlp.ts.
import { useMemo, useState, type ReactNode } from 'react'
import './BackpropFlow.css'
import { Lab, MatrixView, Slider } from '../components/ui'
import { fmt, type Mat } from '../lib/math'
import { applyUpdate, backward, DEMO_NET, DEMO_TARGET, DEMO_X, forward, getWeight, numericalGrad, type TinyNet, type WeightRef } from '../lib/tinyMlp'

const FWD = ['inputs only', 'hidden layer', 'logits', 'softmax', 'loss'] as const
const BWD = ['d_logits', 'd_W2 and d_h', 'ReLU gate', 'd_W1'] as const
type GradId = 'dLogits' | 'dW2' | 'dh' | 'dz1' | 'dW1'
const GRAD_STAGE: Record<GradId, number> = { dLogits: 1, dW2: 2, dh: 2, dz1: 3, dW1: 4 }
const GRAD_LABEL: Record<GradId, string> = { dLogits: 'd_logits', dW2: 'd_W2', dh: 'd_h', dz1: 'd_z (after the ReLU gate)', dW1: 'd_W1' }

const COL = { x: 60, h: 250, z: 440, p: 580 }
const ROW = [85, 215]
const halo = { paintOrder: 'stroke', stroke: 'var(--card)', strokeWidth: 4, strokeLinejoin: 'round' } as const
const accText = { ...halo, fill: 'var(--accent)', fontWeight: 600 } as const
const WEIGHTS: { ref: WeightRef; label: string }[] = [1, 2].flatMap((layer) => [0, 1].flatMap((i) => [0, 1].map((j) => ({
  ref: { layer: layer as 1 | 2, i, j },
  label: layer === 1 ? `W1: x${i + 1} → h${j + 1}` : `W2: h${i + 1} → logit ${j + 1}`,
}))))

export function BackpropFlow() {
  const [net, setNet] = useState<TinyNet>(DEMO_NET)
  const [x, setX] = useState<number[]>(DEMO_X)
  const [target, setTarget] = useState(DEMO_TARGET)
  const [fs, setFs] = useState(0) // forward stages revealed
  const [bs, setBs] = useState(0) // backward stages revealed
  const [sel, setSel] = useState<GradId>('dLogits')
  const [lr, setLr] = useState(0.1)
  const [bug, setBug] = useState(false)
  const [checkIdx, setCheckIdx] = useState(0)
  const [checked, setChecked] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<{ before: number; after: number; lr: number } | null>(null)

  const cache = useMemo(() => forward(net, x, target), [net, x, target])
  const g = useMemo(() => backward(net, cache, target, { reluGate: !bug }), [net, cache, target, bug])
  const edited = () => { setBs(0); setChecked(false); setLastUpdate(null) }

  const nextForward = () => setFs(Math.min(4, fs + 1))
  const nextBackward = () => {
    const n = Math.min(4, bs + 1)
    setBs(n)
    setSel(n === 1 ? 'dLogits' : n === 2 ? 'dW2' : n === 3 ? 'dz1' : 'dW1')
  }
  const apply = () => {
    const next = applyUpdate(net, g, lr)
    setLastUpdate({ before: cache.loss, after: forward(next, x, target).loss, lr })
    setNet(next)
    setBs(0)
    setChecked(false)
  }
  const resetAll = () => { setNet(DEMO_NET); setX(DEMO_X); setTarget(DEMO_TARGET); setFs(0); setBs(0); setBug(false); setChecked(false); setLastUpdate(null) }
  const editW = (key: 'W1' | 'W2') => (i: number, j: number, value: number) => {
    const W: Mat = net[key].map((r) => r.slice())
    W[i][j] = value
    setNet({ ...net, [key]: W })
    edited()
  }

  const oneHot = [0, 1].map((j) => (j === target ? 1 : 0))
  const check = WEIGHTS[checkIdx]
  const numeric = checked ? numericalGrad(net, x, target, check.ref) : null
  const analytic = (check.ref.layer === 1 ? g.dW1 : g.dW2)[check.ref.i][check.ref.j]

  /* ----- drawing helpers ----- */
  const edge = (layer: 1 | 2, i: number, j: number) => {
    const [x1, x2] = layer === 1 ? [COL.x, COL.h] : [COL.h, COL.z]
    const y1 = ROW[i], y2 = ROW[j]
    const t = i === j ? 0.5 : 0.33
    const lx = x1 + (x2 - x1) * t
    const ly = y1 + (y2 - y1) * t + (i === j ? -7 : 4)
    const w = (layer === 1 ? net.W1 : net.W2)[i][j]
    const showGrad = bs >= (layer === 2 ? 2 : 4)
    const dw = (layer === 1 ? g.dW1 : g.dW2)[i][j]
    const hot = showGrad && sel === (layer === 1 ? 'dW1' : 'dW2')
    return (
      <g key={`${layer}${i}${j}`}>
        <line x1={x1 + 28} y1={y1} x2={x2 - 28} y2={y2} stroke={hot ? 'var(--accent)' : 'var(--rule-strong)'} strokeWidth={hot ? 2 : 1.3} />
        <text x={lx} y={ly} fontSize={11.5} textAnchor="middle" style={halo}>{fmt(w)}</text>
        {showGrad && <text x={lx} y={ly + 13} fontSize={11} textAnchor="middle" style={accText}>d {fmt(dw)}</text>}
      </g>
    )
  }
  const node = (cx: number, cy: number, name: string, value: string | null, grad: string | null, hot: boolean, sub?: string) => (
    <g key={name}>
      <circle cx={cx} cy={cy} r={27} fill="var(--card)" stroke={hot ? 'var(--accent)' : 'var(--rule-strong)'} strokeWidth={hot ? 2.5 : 1.5} />
      <text x={cx} y={cy - 33} fontSize={11.5} textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{name}</text>
      <text x={cx} y={cy + (sub ? 1 : 5)} fontSize={14} textAnchor="middle" style={{ fontFamily: 'var(--mono)' }}>{value ?? '?'}</text>
      {sub && value && <text x={cx} y={cy + 15} fontSize={9.5} textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{sub}</text>}
      {grad && <text x={cx} y={cy + 43} fontSize={11.5} textAnchor="middle" style={accText}>d {grad}</text>}
    </g>
  )

  /* ----- explanation of the selected gradient ----- */
  const explain: Record<GradId, ReactNode> = {
    dLogits: (
      <>
        <p><b>Rule 3, softmax + cross-entropy:</b> <code>d_logits = probs − one_hot</code>. The gradient is the miss.</p>
        <p className="mono">[{fmt(cache.probs[0])}, {fmt(cache.probs[1])}] − [{oneHot[0]}, {oneHot[1]}] = [{fmt(g.dLogits[0])}, {fmt(g.dLogits[1])}]</p>
        <p>Negative means “raise this score”, positive means “lower it” (the update subtracts the gradient). The correct class is always the negative one.</p>
      </>
    ),
    dW2: (
      <>
        <p><b>Rule 1, linear layer, blame for the weights:</b> <code>d_W = X.T @ d_out</code>. For one weight: (the value that flowed in through it) × (the blame at the node it feeds).</p>
        {[0, 1].map((i) => [0, 1].map((j) => <div key={`${i}${j}`} className="mono">h{i + 1} → logit {j + 1}: {fmt(cache.h[i])} × {fmt(g.dLogits[j])} = {fmt(g.dW2[i][j])}</div>))}
        <p style={{ marginTop: 8 }}>A weight whose input was 0 carried nothing, so it gets no blame.</p>
      </>
    ),
    dh: (
      <>
        <p><b>Rule 1, linear layer, blame for the input:</b> <code>d_X = d_out @ W.T</code>. Each hidden output collects blame from every logit it fed, through the same weight it used on the way forward.</p>
        {[0, 1].map((i) => <div key={i} className="mono">d_h{i + 1} = {fmt(g.dLogits[0])}×{fmt(net.W2[i][0])} + {fmt(g.dLogits[1])}×{fmt(net.W2[i][1])} = {fmt(g.dh[i])}</div>)}
      </>
    ),
    dz1: (
      <>
        <p><b>Rule 2, ReLU:</b> <code>d_x = d_out * (x &gt; 0)</code>. Blame passes where the hinge was open and stops where it was shut.</p>
        {[0, 1].map((i) => <div key={i} className="mono">h{i + 1}: z = {fmt(cache.z1[i])}, hinge {cache.h[i] > 0 ? 'open' : 'shut'}: d_z = {bug ? `${fmt(g.dh[i])} (BUG: gate skipped)` : cache.h[i] > 0 ? `${fmt(g.dh[i])} × 1 = ${fmt(g.dz1[i])}` : `${fmt(g.dh[i])} × 0 = ${fmt(g.dz1[i])}`}</div>)}
        <p style={{ marginTop: 8 }}>A shut hinge output 0 whatever its inputs were (within a small nudge), so nothing upstream of it could have changed the loss.</p>
      </>
    ),
    dW1: (
      <>
        <p><b>Rule 1 again</b>, one layer further back: <code>d_W = X.T @ d_out</code>, where d_out is now d_z.</p>
        {[0, 1].map((i) => [0, 1].map((j) => <div key={`${i}${j}`} className="mono">x{i + 1} → h{j + 1}: {fmt(x[i])} × {fmt(g.dz1[j])} = {fmt(g.dW1[i][j])}</div>))}
        <p style={{ marginTop: 8 }}>Every weight now has its gradient. Cost: one backward sweep, reusing d_logits, d_h and d_z instead of recomputing them for each weight.</p>
      </>
    ),
  }
  const revealed = (Object.keys(GRAD_STAGE) as GradId[]).filter((k) => bs >= GRAD_STAGE[k])

  return (
    <Lab
      title="Forward, backward, update"
      goal={<>Press <b>Forward</b> until the loss appears. Then press <b>Backward</b> and watch blame (in <span className="acc">the accent colour</span>, marked “d”) flow from right to left over the same wires. Select any revealed gradient to see the rule that produced it.</>}
    >
      <div className="controls">
        <div className="control">
          <label htmlFor="bp-x1"><span>inputs x₁, x₂</span></label>
          <div style={{ display: 'flex', gap: 8 }}>
            {[0, 1].map((i) => <input key={i} id={`bp-x${i + 1}`} aria-label={`input x${i + 1}`} className="input" type="number" step={0.5} value={x[i]} onChange={(e) => { setX(x.map((v, k) => (k === i ? Number(e.target.value) || 0 : v))); edited() }} style={{ width: 90 }} />)}
          </div>
        </div>
        <div className="control">
          <label><span>correct class (the target)</span></label>
          <div className="steps" role="group" aria-label="Correct class" style={{ marginBottom: 0 }}>
            {[0, 1].map((t) => <button key={t} className="step-btn" aria-pressed={target === t} onClick={() => { setTarget(t); edited() }}>class {t + 1}</button>)}
          </div>
        </div>
      </div>

      <p className="BackpropFlow-scrollhint">The diagram is wider than your screen: swipe it sideways to reach the logits, the probabilities and the loss.</p>
      <div className="table-scroll">
        <svg viewBox="0 0 640 290" style={{ width: '100%', minWidth: 540, display: 'block' }} role="img"
          aria-label={`Network graph. Forward stage: ${FWD[fs]}. ${fs >= 4 ? `Loss ${fmt(cache.loss, 3)}.` : ''} ${bs > 0 ? `Backward stage: ${BWD[bs - 1]}.` : ''}`}>
          {[0, 1].map((i) => [0, 1].map((j) => edge(1, i, j)))}
          {[0, 1].map((i) => [0, 1].map((j) => edge(2, i, j)))}
          {[0, 1].map((j) => <line key={j} x1={COL.z + 28} y1={ROW[j]} x2={COL.p - 28} y2={ROW[j]} stroke="var(--rule-strong)" strokeDasharray="4 3" />)}
          <text x={(COL.z + COL.p) / 2} y={150} fontSize={11} textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>softmax</text>
          {[0, 1].map((i) => node(COL.x, ROW[i], `x${i + 1}`, fmt(x[i]), null, false))}
          {[0, 1].map((i) => node(COL.h, ROW[i], `h${i + 1} = relu(z)`, fs >= 1 ? fmt(cache.h[i]) : null, bs >= 3 ? fmt(g.dz1[i]) : bs >= 2 ? fmt(g.dh[i]) : null, bs >= 2 && (sel === 'dh' || sel === 'dz1'), `z = ${fmt(cache.z1[i])}${cache.h[i] > 0 ? '' : ', shut'}`))}
          {[0, 1].map((j) => node(COL.z, ROW[j], `logit ${j + 1}`, fs >= 2 ? fmt(cache.logits[j]) : null, bs >= 1 ? fmt(g.dLogits[j]) : null, bs >= 1 && sel === 'dLogits'))}
          {[0, 1].map((j) => node(COL.p, ROW[j], `P(class ${j + 1})${j === target ? ' ✓' : ''}`, fs >= 3 ? fmt(cache.probs[j]) : null, null, false))}
          <text x={COL.p} y={150} fontSize={12.5} textAnchor="middle" style={{ fontFamily: 'var(--mono)', fontWeight: 600 }}>{fs >= 4 ? `loss ${fmt(cache.loss, 3)}` : 'loss ?'}</text>
          <text x={8} y={282} fontSize={10.5} style={{ fill: 'var(--ink-3)' }}>numbers on wires: weights · ✓ marks the correct class · biases start at 0 and are updated too (not drawn)</text>
        </svg>
      </div>

      <div className="btn-row" style={{ marginTop: 8 }}>
        <button className="btn small primary" onClick={nextForward} disabled={fs >= 4}>Forward → {fs < 4 ? FWD[fs + 1] : 'done'}</button>
        <button className="btn small primary" onClick={nextBackward} disabled={fs < 4 || bs >= 4}>← Backward {bs < 4 ? BWD[bs] : 'done'}</button>
        <button className="btn small" onClick={resetAll}>Reset everything</button>
      </div>

      <div className="readout" aria-live="polite" style={{ marginTop: 12 }}>
        {fs === 0 && <span>Nothing computed yet. The “?” values fill in from left to right.</span>}
        {fs >= 1 && <span>z = x @ W1 + b1 = [{fmt(cache.z1[0])}, {fmt(cache.z1[1])}] → h = [{fmt(cache.h[0])}, {fmt(cache.h[1])}]</span>}
        {fs >= 2 && <span>logits = h @ W2 + b2 = [{fmt(cache.logits[0])}, {fmt(cache.logits[1])}]</span>}
        {fs >= 3 && <span>probs = [{fmt(cache.probs[0])}, {fmt(cache.probs[1])}]</span>}
        {fs >= 4 && <span>loss = −ln({fmt(cache.probs[target], 3)}) = <b>{fmt(cache.loss, 3)}</b></span>}
        {lastUpdate && <span>after the update with lr {lastUpdate.lr}: loss {fmt(lastUpdate.before, 3)} → <b>{fmt(lastUpdate.after, 3)}</b> {lastUpdate.after < lastUpdate.before ? '(lower: it learned)' : '(higher: the step was too big)'}</span>}
      </div>

      {bs > 0 && (
        <>
          <div className="steps" role="tablist" aria-label="Revealed gradients" style={{ marginTop: 14 }}>
            {revealed.map((k) => <button key={k} role="tab" className="step-btn" aria-selected={sel === k} onClick={() => setSel(k)}>{GRAD_LABEL[k]}</button>)}
          </div>
          <div className="card" role="tabpanel" aria-live="polite" style={{ fontSize: 14.5 }}>{explain[revealed.includes(sel) ? sel : 'dLogits']}</div>
        </>
      )}

      {bs >= 4 && (
        <div className="grid-2" style={{ marginTop: 14 }}>
          <div className="card">
            <h4 style={{ fontSize: 16, marginBottom: 6 }}>Apply the update</h4>
            <p style={{ fontSize: 14.5 }}>Every weight and bias: <code>W −= lr × d_W</code>. Then run forward again.</p>
            <Slider label="learning rate" value={lr} min={0.01} max={2} step={0.01} onChange={setLr} />
            <button className="btn small primary" onClick={apply}>Apply update</button>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 16, marginBottom: 6 }}>Check with nudges</h4>
            <label htmlFor="bp-check" className="sr-only">Weight to check</label>
            <select id="bp-check" className="input" value={checkIdx} onChange={(e) => { setCheckIdx(Number(e.target.value)); setChecked(false) }}>
              {WEIGHTS.map((w, i) => <option key={i} value={i}>{w.label} (now {fmt(getWeight(net, w.ref))})</option>)}
            </select>
            <div className="btn-row" style={{ marginTop: 8 }}>
              <button className="btn small" onClick={() => setChecked(true)}>Nudge it and re-measure the loss</button>
            </div>
            {numeric && (
              <div className="mono" style={{ fontSize: 13, marginTop: 8 }} aria-live="polite">
                loss(w + 0.00001) = {numeric.lossPlus.toFixed(8)}<br />
                loss(w − 0.00001) = {numeric.lossMinus.toFixed(8)}<br />
                difference ÷ 0.00002 = <b>{numeric.grad.toFixed(6)}</b><br />
                backprop said: <b>{analytic.toFixed(6)}</b><br />
                <span style={{ color: Math.abs(numeric.grad - analytic) < 1e-5 ? 'var(--good)' : 'var(--bad)' }}>
                  {Math.abs(numeric.grad - analytic) < 1e-5 ? 'PASS: they agree.' : 'FAIL: backprop is wrong for this weight.'}
                </span>
              </div>
            )}
            <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14, marginTop: 10 }}>
              <input type="checkbox" checked={bug} onChange={(e) => { setBug(e.target.checked); setChecked(false) }} /> Plant a bug: forget the ReLU gate
            </label>
          </div>
        </div>
      )}

      <details className="deep" style={{ marginTop: 14 }}>
        <summary>Edit the weights</summary>
        <div className="details-body">
          <div className="matrix-row">
            <MatrixView m={net.W1} rows={['x1', 'x2']} cols={['h1', 'h2']} caption="W1 (row: from, column: to)" onEdit={editW('W1')} />
            <MatrixView m={net.W2} rows={['h1', 'h2']} cols={['logit 1', 'logit 2']} caption="W2" onEdit={editW('W2')} />
          </div>
          <p className="lab-note">Try making W1 (x1 → h2) positive enough to open the second hinge, then run Backward again: blame now reaches that part of the network too.</p>
        </div>
      </details>
    </Lab>
  )
}
