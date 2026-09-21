// Gradient descent on a line, two ways: the learner turns the knobs by hand,
// then the machine runs the same loop. All maths lives in src/lib/gd.ts.
import { useEffect, useMemo, useRef, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import { makeRng } from '../lib/rng'
import { bestFit, describeGradient, gdStep, hasDiverged, lossAndGrads, makeLineData, mse, sampleBatch, type LineData, type LineParams, type StepRecord } from '../lib/gd'

const MAX_STEPS = 300
const START: LineParams = { w: 0, b: 0 }

/** Readable numbers even when training explodes. */
const num = (x: number, digits = 3): string => {
  if (!Number.isFinite(x)) return 'overflow'
  if (Math.abs(x) >= 1e4) return x.toExponential(2).replace('-', '−').replace('e+', 'e')
  return fmt(x, digits)
}

/** Negative numbers in brackets, so “1.4×(−0.9)” reads as arithmetic. */
const par = (s: string) => (s.startsWith('−') ? `(${s})` : s)

/* ---------- the scatter plot with the model's line and the error bars ---------- */
const PW = 560, PH = 330, X0 = -2.2, X1 = 2.2, Y0 = -9, Y1 = 6
const sx = (x: number) => 40 + ((x - X0) / (X1 - X0)) * (PW - 50)
const sy = (y: number) => 10 + ((Y1 - y) / (Y1 - Y0)) * (PH - 40)
const clampY = (y: number) => Math.max(Y0 - 40, Math.min(Y1 + 40, y))

function LinePlot({ data, p, batch, label }: { data: LineData; p: LineParams; batch?: number[] | null; label: string }) {
  const inBatch = batch ? new Set(batch) : null
  const ok = Number.isFinite(p.w) && Number.isFinite(p.b)
  return (
    <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', maxWidth: PW, display: 'block', margin: '0 auto' }} role="img" aria-label={label}>
      <defs><clipPath id="gd-plot-clip"><rect x={40} y={10} width={PW - 50} height={PH - 40} /></clipPath></defs>
      {[-8, -4, 0, 4].map((y) => (
        <g key={y}><line className={y === 0 ? 'axis' : 'gridline'} x1={40} x2={PW - 10} y1={sy(y)} y2={sy(y)} /><text x={34} y={sy(y) + 4} fontSize={11} textAnchor="end">{y}</text></g>
      ))}
      {[-2, -1, 0, 1, 2].map((x) => (
        <g key={x}><line className={x === 0 ? 'axis' : 'gridline'} x1={sx(x)} x2={sx(x)} y1={10} y2={PH - 30} /><text x={sx(x)} y={PH - 14} fontSize={11} textAnchor="middle">{x}</text></g>
      ))}
      <text x={PW - 12} y={PH - 2} fontSize={11} textAnchor="end" style={{ fill: 'var(--ink-3)' }}>input x</text>
      <g clipPath="url(#gd-plot-clip)">
        {ok && data.xs.map((x, i) => {
          const dim = inBatch && !inBatch.has(i)
          return <line key={i} x1={sx(x)} x2={sx(x)} y1={sy(data.ys[i])} y2={sy(clampY(p.w * x + p.b))} stroke="var(--bad)" strokeWidth={1.5} opacity={dim ? 0.12 : 0.6} />
        })}
        {ok && <line x1={sx(X0)} y1={sy(clampY(p.w * X0 + p.b))} x2={sx(X1)} y2={sy(clampY(p.w * X1 + p.b))} stroke="var(--accent)" strokeWidth={2.5} />}
        {data.xs.map((x, i) => (
          <circle key={i} cx={sx(x)} cy={sy(data.ys[i])} r={3.2} fill="var(--ink)" opacity={inBatch && !inBatch.has(i) ? 0.25 : 0.85} />
        ))}
      </g>
    </svg>
  )
}

/* ---------- loss over time ---------- */
function LossCurve({ history, showBatch }: { history: StepRecord[]; showBatch: boolean }) {
  const W = 280, H = 190
  const vals = history.map((r) => r.fullLoss).filter(Number.isFinite)
  const top = Math.max(1, ...vals, ...(showBatch ? history.map((r) => r.loss).filter(Number.isFinite) : []))
  const n = Math.max(20, history.length)
  const px = (i: number) => 36 + (i / (n - 1)) * (W - 46)
  const py = (v: number) => 12 + (1 - Math.min(v, top) / top) * (H - 40)
  const pts = history.filter((r) => Number.isFinite(r.fullLoss)).map((r) => `${px(r.step)},${py(r.fullLoss)}`).join(' ')
  const last = history[history.length - 1]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img" aria-label={last ? `Loss curve. Step ${last.step}, loss ${num(last.fullLoss)}.` : 'Loss curve, empty until you take a step.'}>
      <line className="axis" x1={36} x2={W - 10} y1={H - 28} y2={H - 28} />
      <line className="axis" x1={36} x2={36} y1={12} y2={H - 28} />
      <text x={32} y={18} fontSize={10} textAnchor="end">{num(top, 1)}</text>
      <text x={32} y={H - 26} fontSize={10} textAnchor="end">0</text>
      <text x={W - 10} y={H - 10} fontSize={10.5} textAnchor="end" style={{ fill: 'var(--ink-3)' }}>step →</text>
      <text x={40} y={H - 10} fontSize={10.5} style={{ fill: 'var(--ink-3)' }}>loss (all data)</text>
      {showBatch && history.map((r) => Number.isFinite(r.loss) && <circle key={r.step} cx={px(r.step)} cy={py(r.loss)} r={1.6} fill="var(--ink-3)" opacity={0.6} />)}
      {history.length > 1 && <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={2} />}
      {last && Number.isFinite(last.fullLoss) && <circle cx={px(last.step)} cy={py(last.fullLoss)} r={3.5} fill="var(--accent)" />}
    </svg>
  )
}

/* ---------- the path of (w, b) over a map of the loss ---------- */
const MW0 = -1, MW1 = 5, MB0 = -4, MB1 = 2, CELLS = 20
function PathMap({ data, trail, best }: { data: LineData; trail: LineParams[]; best: LineParams }) {
  const W = 280, H = 190
  const px = (w: number) => 36 + ((w - MW0) / (MW1 - MW0)) * (W - 46)
  const py = (b: number) => 12 + ((MB1 - b) / (MB1 - MB0)) * (H - 40)
  const cells = useMemo(() => {
    const out: { w: number; b: number; shade: number }[] = []
    const lo = Math.log(mse(best, data))
    const hi = Math.log(mse({ w: MW0, b: MB1 }, data))
    for (let i = 0; i < CELLS; i++) for (let j = 0; j < CELLS; j++) {
      const w = MW0 + ((i + 0.5) / CELLS) * (MW1 - MW0)
      const b = MB0 + ((j + 0.5) / CELLS) * (MB1 - MB0)
      out.push({ w, b, shade: Math.min(1, (Math.log(mse({ w, b }, data)) - lo) / (hi - lo)) })
    }
    return out
  }, [data, best])
  const cw = (W - 46) / CELLS, ch = (H - 40) / CELLS
  const clamp = (v: number, a: number, z: number) => Math.max(a - 3, Math.min(z + 3, v))
  const safe = trail.filter((p) => Number.isFinite(p.w) && Number.isFinite(p.b)).map((p) => ({ w: clamp(p.w, MW0, MW1), b: clamp(p.b, MB0, MB1) }))
  const here = trail[trail.length - 1]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img" aria-label={`Map of the loss for every w and b. Darker is higher loss. Current position w ${num(here.w, 2)}, b ${num(here.b, 2)}.`}>
      <defs><clipPath id="gd-map-clip"><rect x={36} y={12} width={W - 46} height={H - 40} /></clipPath></defs>
      {cells.map((c, k) => <rect key={k} x={px(c.w) - cw / 2} y={py(c.b) - ch / 2} width={cw + 0.5} height={ch + 0.5} fill="var(--ink)" opacity={0.04 + c.shade * 0.4} />)}
      <line className="axis" x1={36} x2={W - 10} y1={H - 28} y2={H - 28} />
      <line className="axis" x1={36} x2={36} y1={12} y2={H - 28} />
      {[0, 2, 4].map((w) => <text key={w} x={px(w)} y={H - 16} fontSize={10} textAnchor="middle">{w}</text>)}
      {[-3, -1, 1].map((b) => <text key={b} x={32} y={py(b) + 3} fontSize={10} textAnchor="end">{b}</text>)}
      <text x={W - 10} y={H - 3} fontSize={10.5} textAnchor="end" style={{ fill: 'var(--ink-3)' }}>w →</text>
      <text x={4} y={10} fontSize={10.5} style={{ fill: 'var(--ink-3)' }}>b ↑ (darker = higher loss)</text>
      <g clipPath="url(#gd-map-clip)">
        <polyline points={safe.map((p) => `${px(p.w)},${py(p.b)}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth={1.6} />
        {safe.map((p, i) => <circle key={i} cx={px(p.w)} cy={py(p.b)} r={i === safe.length - 1 ? 4 : 1.8} fill="var(--accent)" />)}
      </g>
      <path d={`M ${px(best.w) - 5} ${py(best.b) - 5} l 10 10 m -10 0 l 10 -10`} stroke="var(--good)" strokeWidth={2.2} fill="none" />
      <text x={px(best.w) + 8} y={py(best.b) - 6} fontSize={10.5} style={{ fill: 'var(--good)' }}>lowest loss</text>
    </svg>
  )
}

const knobAdvice = (name: string, grad: number) => {
  const d = describeGradient(grad)
  if (d.direction === 'leave') return <span><b>{name}</b>: about right (slope {num(grad, 2)})</span>
  return <span><b>{name}</b>: turn it <b>{d.direction === 'up' ? 'UP ↑' : 'DOWN ↓'}</b>, {d.strength} (slope {num(grad, 2)})</span>
}

export function GradientDescentPlayground() {
  const data = useMemo(() => makeLineData(), [])
  const best = useMemo(() => bestFit(data), [data])
  const floor = useMemo(() => mse(best, data), [best, data])
  const [mode, setMode] = useState<'manual' | 'auto'>('manual')

  /* ----- mode (a): you are the optimizer ----- */
  const [mw, setMw] = useState(0)
  const [mb, setMb] = useState(0)
  const [reveal, setReveal] = useState(false)
  const manual = lossAndGrads({ w: mw, b: mb }, data)
  const solved = manual.loss < floor * 1.1

  /* ----- mode (b): let the machine do it ----- */
  const [logLr, setLogLr] = useState(-1) // lr = 10^logLr
  const lr = Number((10 ** logLr).toPrecision(3))
  const [batchSize, setBatchSize] = useState(0) // 0 = all data
  const [sim, setSim] = useState<{ p: LineParams; history: StepRecord[] }>({ p: START, history: [] })
  const [running, setRunning] = useState(false)
  const rngRef = useRef(makeRng(7))
  const diverged = hasDiverged(sim.p)
  const finished = diverged || sim.history.length >= MAX_STEPS

  const stepOnce = () => {
    const idx = batchSize > 0 ? sampleBatch(rngRef.current, data.xs.length, batchSize) : null
    setSim((s) => {
      if (hasDiverged(s.p) || s.history.length >= MAX_STEPS) return s
      const rec = gdStep(s.p, data, lr, s.history.length, idx)
      return { p: rec.after, history: [...s.history, rec] }
    })
  }
  const stepRef = useRef(stepOnce)
  stepRef.current = stepOnce

  useEffect(() => {
    if (!running) return
    const t = setInterval(() => stepRef.current(), 70)
    return () => clearInterval(t) // also stops on unmount
  }, [running])
  useEffect(() => { if (running && finished) setRunning(false) }, [running, finished])

  const reset = () => { setRunning(false); rngRef.current = makeRng(7); setSim({ p: START, history: [] }) }
  const last = sim.history[sim.history.length - 1]
  const trail = [START, ...sim.history.map((r) => r.after)]
  const nowLoss = mse(sim.p, data)
  const shown = last ? (last.idx ?? data.xs.map((_, i) => i)).slice(0, 3) : []

  return (
    <Lab
      title="Gradient descent on a line"
      goal={<>40 data points were made from a hidden rule plus some noise. The model is a line, <span className="mono">y = w·x + b</span>, with two adjustable numbers. First find good values <b>yourself</b>. Then let the machine do exactly what you did.</>}
    >
      <div className="steps" role="group" aria-label="Choose a mode">
        <button className="step-btn" aria-pressed={mode === 'manual'} onClick={() => { setMode('manual'); setRunning(false) }}>A · You are the optimizer</button>
        <button className="step-btn" aria-pressed={mode === 'auto'} onClick={() => setMode('auto')}>B · Let the machine do it</button>
      </div>

      {mode === 'manual' && (
        <>
          <LinePlot data={data} p={{ w: mw, b: mb }} label={`Scatter plot of 40 points with your line y = ${fmt(mw)} x + ${fmt(mb)}. Loss ${fmt(manual.loss, 3)}.`} />
          <p className="lab-note">Dots: the data. <span className="acc">Line</span>: your model’s predictions. <span style={{ color: 'var(--bad)' }}>Vertical bars</span>: the error for each point. The loss is the average of those bars, squared.</p>
          <div className="controls">
            <Slider label="w (slope: how much y changes per unit of x)" value={mw} min={-1} max={5} step={0.05} onChange={setMw} format={(v) => fmt(v)} />
            <Slider label="b (offset: where the line crosses x = 0)" value={mb} min={-4} max={2} step={0.05} onChange={setMb} format={(v) => fmt(v)} />
          </div>
          <div className="readout" aria-live="polite">
            <span>loss (mean squared error) = <b>{fmt(manual.loss, 3)}</b></span>
            {knobAdvice('w', manual.gradW)}
            {knobAdvice('b', manual.gradB)}
          </div>
          <p className="lab-note" style={{ marginTop: 10 }}>
            <b>Your task:</b> make the line fit. Follow the hints: they are the <em>gradient</em>, one reading per knob. Move a knob a little, read the hints again, repeat. Notice that the hints for w change when you move b: each reading is only valid where you currently stand.
          </p>
          {solved && <p className="lab-note" role="status"><b>That is as good as a line gets.</b> The loss will not reach 0: the data is noisy, and no straight line passes through every point. You just performed gradient descent by hand.</p>}
          <div className="btn-row">
            <button className="btn small" onClick={() => { setMw(0); setMb(0); setReveal(false) }}>Reset</button>
            {!reveal && <button className="btn small" onClick={() => setReveal(true)}>I tried: reveal the best values</button>}
            <button className="btn small primary" onClick={() => setMode('auto')}>Now let the machine do it</button>
          </div>
          {reveal && (
            <div className="readout" style={{ marginTop: 10 }}>
              <span>best line for these 40 points: w = <b>{fmt(best.w)}</b>, b = <b>{fmt(best.b)}</b>, loss = <b>{fmt(floor, 3)}</b></span>
              <span>hidden rule that generated the data: y = 3x − 1.5 (plus noise)</span>
            </div>
          )}
        </>
      )}

      {mode === 'auto' && (
        <>
          <LinePlot data={data} p={sim.p} batch={last?.idx} label={`Scatter plot with the machine’s current line y = ${num(sim.p.w, 2)} x + ${num(sim.p.b, 2)}, after ${sim.history.length} steps.`} />
          <div className="controls">
            <Slider label="learning rate (step size)" value={logLr} min={-3} max={0.3} step={0.01} onChange={(v) => setLogLr(v)} format={() => String(lr)} />
            <div className="control">
              <label htmlFor="gd-batch"><span>data used per step</span><output>{batchSize === 0 ? 'all 40 points' : `random batch of ${batchSize}`}</output></label>
              <select id="gd-batch" className="input" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))}>
                <option value={0}>all 40 points (exact gradient)</option>
                <option value={16}>mini-batch of 16</option>
                <option value={4}>mini-batch of 4</option>
                <option value={1}>mini-batch of 1 (very noisy)</option>
              </select>
            </div>
          </div>
          <div className="btn-row">
            <button className="btn small primary" onClick={stepOnce} disabled={running || finished}>Step</button>
            {!running
              ? <button className="btn small" onClick={() => setRunning(true)} disabled={finished}>Run</button>
              : <button className="btn small" onClick={() => setRunning(false)}>Pause</button>}
            <button className="btn small" onClick={reset}>Reset (w = 0, b = 0)</button>
          </div>

          <div className="readout" aria-live="polite" style={{ marginTop: 12 }}>
            <span>step <b>{sim.history.length}</b></span>
            <span>w = <b>{num(sim.p.w)}</b></span>
            <span>b = <b>{num(sim.p.b)}</b></span>
            <span>loss = <b>{num(nowLoss)}</b></span>
            {diverged && <span style={{ color: 'var(--bad)' }}>Diverged: every step overshoots further than the last. Lower the learning rate and reset.</span>}
            {!diverged && sim.history.length >= MAX_STEPS && <span>Stopped after {MAX_STEPS} steps.</span>}
          </div>

          <div className="grid-2" style={{ marginTop: 12 }}>
            <div><LossCurve history={sim.history} showBatch={batchSize > 0} />{batchSize > 0 && <p className="lab-note">Grey dots: the loss on each small batch (what the machine actually sees). Line: the loss on all 40 points.</p>}</div>
            <div><PathMap data={data} trail={trail} best={best} /></div>
          </div>

          <h4 style={{ fontSize: 16, margin: '14px 0 6px' }}>{last ? `Inside step ${last.step}` : 'Inside one step'}</h4>
          {!last && <p className="lab-note">Press <b>Step</b> to run the loop once: predict, measure the loss, compute the gradient, update. Every number will be shown here.</p>}
          {last && (
            <div className="table-scroll">
              <table className="plain mono" style={{ fontSize: 13 }}>
                <tbody>
                  <tr><td>1 · predict</td><td>
                    {shown.map((i, k) => <div key={k}>x = {fmt(data.xs[i])}: pred = {num(last.before.w)}×{par(fmt(data.xs[i]))} + {par(num(last.before.b))} = {num(last.preds[k])}, true y = {fmt(data.ys[i])}, error = {num(last.errs[k])}</div>)}
                    <div className="muted">… and {last.preds.length - shown.length} more {last.idx ? 'in this batch' : 'points'}</div>
                  </td></tr>
                  <tr><td>2 · loss</td><td>mean(error²) = <b>{num(last.loss)}</b>{last.idx ? ` on this batch (all data: ${num(last.fullLoss)})` : ''}</td></tr>
                  <tr><td>3 · gradient</td><td>grad_w = mean(2·error·x) = <b>{num(last.gradW)}</b><br />grad_b = mean(2·error) = <b>{num(last.gradB)}</b></td></tr>
                  <tr><td>4 · update</td><td>
                    w −= lr·grad_w: {num(last.before.w)} − {last.lr}×{par(num(last.gradW))} = <b>{num(last.after.w)}</b><br />
                    b −= lr·grad_b: {num(last.before.b)} − {last.lr}×{par(num(last.gradB))} = <b>{num(last.after.b)}</b>
                  </td></tr>
                </tbody>
              </table>
            </div>
          )}
          <p className="lab-note" style={{ marginTop: 10 }}>Try learning rates 0.01, 0.1 and 1. Reset between runs. Changing the learning rate mid-run is allowed: real training schedules do that too.</p>
        </>
      )}
    </Lab>
  )
}
