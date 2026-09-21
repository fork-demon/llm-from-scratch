// Lesson 1.4: the nudge experiment. Bump x by h, see how far f moves, divide.
import { useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import { NUDGE_FUNCTIONS, convergence, nudgeReport, samplePoints } from '../lib/calculus'

const W = 420
const H = 300
const PAD = 14
const X_RANGE: [number, number] = [-3.5, 3.5]
const H_EXPONENTS = { min: -3, max: 0, step: 0.25 } // h = 10^e

const sig = (x: number, digits = 4) => (Math.abs(x) >= 1000 || (x !== 0 && Math.abs(x) < 0.001) ? x.toExponential(2) : fmt(x, digits))

export function NudgePlayground() {
  const [fnId, setFnId] = useState('square')
  const [x, setX] = useState(1.5)
  const [e, setE] = useState(0)
  const fn = NUDGE_FUNCTIONS.find((f) => f.id === fnId)!
  const h = 10 ** e
  const r = nudgeReport(fn, x, h)
  const table = convergence(fn.f, x)

  const [y0, y1] = fn.yRange
  const sx = (v: number) => PAD + ((v - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0])) * (W - 2 * PAD)
  const sy = (v: number) => H - PAD - ((v - y0) / (y1 - y0)) * (H - 2 * PAD)
  const curve = samplePoints(fn.f, X_RANGE[0], X_RANGE[1]).map(([px, py]) => `${sx(px).toFixed(1)},${sy(py).toFixed(1)}`).join(' ')
  // a straight line through (x, f(x)) with a given slope, drawn across the whole plot
  const lineAt = (slope: number) => ({
    x1: sx(X_RANGE[0]), y1: sy(r.fx + slope * (X_RANGE[0] - x)),
    x2: sx(X_RANGE[1]), y2: sy(r.fx + slope * (X_RANGE[1] - x)),
  })
  const atKink = Number.isNaN(r.exact)
  const direction = Math.abs(r.ratio) < 1e-9 ? 'flat here: nudging x does nothing to the output' : r.ratio > 0 ? 'positive: turn x up and the output goes up' : 'negative: turn x up and the output goes DOWN'

  return (
    <Lab
      title="The nudge experiment"
      goal={<>Pick a function and a point x. The nudge h starts big (1.0). <b>Shrink h</b> and watch the ratio settle on one number. That number is the derivative at x. Then move x: where is the ratio negative? Where is it zero?</>}
    >
      <div className="steps" role="group" aria-label="Choose a function">
        {NUDGE_FUNCTIONS.map((f) => (
          <button key={f.id} className="step-btn mono" aria-pressed={f.id === fnId} onClick={() => setFnId(f.id)}>f(x) = {f.label}</button>
        ))}
      </div>
      <div className="controls">
        <Slider label="x (where we stand)" value={x} min={-3} max={3} step={0.1} format={(v) => fmt(v, 1)} onChange={setX} />
        <Slider label="nudge size h" value={e} min={H_EXPONENTS.min} max={H_EXPONENTS.max} step={H_EXPONENTS.step} format={() => sig(h)} onChange={setE} />
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img" aria-label={`Plot of f(x) = ${fn.label}. At x = ${fmt(x, 1)} the line through the two measured points has slope ${fmt(r.ratio, 3)}.`}>
          <defs><clipPath id="np-clip"><rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} /></clipPath></defs>
          {[-3, -2, -1, 1, 2, 3].map((t) => <line key={t} className="gridline" x1={sx(t)} x2={sx(t)} y1={PAD} y2={H - PAD} />)}
          <line className="axis" x1={sx(0)} x2={sx(0)} y1={PAD} y2={H - PAD} />
          <line className="axis" x1={PAD} x2={W - PAD} y1={sy(0)} y2={sy(0)} />
          {[-3, -2, -1, 1, 2, 3].map((t) => <text key={t} x={sx(t)} y={sy(0) + 12} fontSize={9.5} textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{t}</text>)}
          <g clipPath="url(#np-clip)">
            <polyline points={curve} fill="none" stroke="var(--ink-2)" strokeWidth={2.5} />
            {!atKink && <line {...lineAt(r.exact)} stroke="var(--ink-3)" strokeWidth={1.5} strokeDasharray="5 4" />}
            <line {...lineAt(r.ratio)} stroke="var(--accent)" strokeWidth={2} />
            <line x1={sx(x)} x2={sx(x + h)} y1={sy(r.fx)} y2={sy(r.fx)} stroke="var(--good)" strokeWidth={2} />
            <line x1={sx(x + h)} x2={sx(x + h)} y1={sy(r.fx)} y2={sy(r.fxh)} stroke="var(--good)" strokeWidth={2} />
            <circle cx={sx(x + h)} cy={sy(r.fxh)} r={4.5} fill="var(--card)" stroke="var(--accent)" strokeWidth={2} />
            <circle cx={sx(x)} cy={sy(r.fx)} r={5.5} fill="var(--accent)" />
          </g>
          </svg>

        <div aria-live="polite" style={{ minWidth: 0 }}>
          <p className="lab-note" style={{ marginTop: 0 }}><span style={{ color: 'var(--accent)' }}>Solid line:</span> through both measured points. Dashed line: the true tangent at x.</p>
          <div className="readout" style={{ display: 'block', overflowWrap: 'anywhere' }}>
            <div>f(x) &nbsp;&nbsp;&nbsp;&nbsp;= f({fmt(x, 1)}) = <b>{sig(r.fx)}</b></div>
            <div>f(x + h) = f({sig(x + h)}) = <b>{sig(r.fxh, 6)}</b></div>
            <div>output moved by {sig(r.moved, 6)}</div>
            <div>input moved by &nbsp;{sig(h)}</div>
            <div style={{ marginTop: 6 }}>ratio = {sig(r.moved, 6)} ÷ {sig(h)} <span style={{ whiteSpace: 'nowrap' }}>= <b style={{ fontSize: 18 }}>{fmt(r.ratio, 4)}</b></span></div>
          </div>
          <p className="lab-note" style={{ marginTop: 8 }}>The ratio is {direction}.</p>
          <div className="table-scroll">
            <table className="plain mono" style={{ fontSize: 13.5 }}>
              <thead><tr><th>nudge h</th>{table.map((t) => <th key={t.h}>{t.h}</th>)}</tr></thead>
              <tbody><tr><td>ratio</td>{table.map((t) => <td key={t.h}>{fmt(t.ratio, 4)}</td>)}</tr></tbody>
            </table>
          </div>
          <p className="lab-note">
            {atKink
              ? <>You are standing exactly on the corner of max(0, x). Nudging right gives 1, nudging left would give 0: there is no single answer here. Real libraries just pick one (usually 0), and it causes no trouble in practice.</>
              : <>The school shortcut for {fn.label} is <span className="mono">{fn.rule}</span>, which gives <b className="mono">{fmt(r.exact, 4)}</b> here. The experiment gets there on its own as h shrinks.</>}
          </p>
        </div>
      </div>
    </Lab>
  )
}
