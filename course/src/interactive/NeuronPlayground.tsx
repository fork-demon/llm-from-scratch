// Build a curve out of ReLU hinges by hand. One input, four hidden units, one output.
// All maths lives in src/lib/hinges.ts.
import { useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import { bendAt, collapsedLine, curveError, grid, netOutput, SOLUTIONS, START_NET, TARGETS, unitOutput, X_MAX, X_MIN, type HingeNet, type HingeUnit, type TargetId } from '../lib/hinges'

const W = 560, H = 320, Y0 = -1.5, Y1 = 4.5
const sx = (x: number) => 36 + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - 46)
const sy = (y: number) => 10 + ((Y1 - Math.max(Y0 - 1, Math.min(Y1 + 1, y))) / (Y1 - Y0)) * (H - 36)
const XS = grid(81)
const path = (f: (x: number) => number) => XS.map((x, i) => `${i ? 'L' : 'M'} ${sx(x).toFixed(1)} ${sy(f(x)).toFixed(1)}`).join(' ')
const DASHES = ['6 3', '2 3', '9 3 2 3', '4 6']

export function NeuronPlayground() {
  const [net, setNet] = useState<HingeNet>(START_NET)
  const [target, setTarget] = useState<TargetId>('tent')
  const [activation, setActivation] = useState(true)
  const [sel, setSel] = useState(0)
  const [revealed, setRevealed] = useState(false)

  const f = TARGETS[target].f
  const err = curveError(net, f, activation)
  const line = collapsedLine(net)
  const u = net.units[sel]
  const setUnit = (patch: Partial<HingeUnit>) => setNet({ ...net, units: net.units.map((unit, i) => (i === sel ? { ...unit, ...patch } : unit)) })
  const bend = bendAt(u)
  const pickTarget = (t: TargetId) => { setTarget(t); setRevealed(false) }

  return (
    <Lab
      title="Build a curve out of hinges"
      goal={<>Each hidden unit computes <span className="mono">relu(w·x + b)</span>: a hinge. The output adds the hinges up, each multiplied by its output weight v. <b>Make the thick line match the grey target.</b> Then remove the activation and try again.</>}
    >
      <div className="steps" role="group" aria-label="Choose a target curve">
        {(Object.keys(TARGETS) as TargetId[]).map((t) => (
          <button key={t} className="step-btn" aria-pressed={t === target} onClick={() => pickTarget(t)}>target: {TARGETS[t].label}</button>
        ))}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, display: 'block', margin: '0 auto' }} role="img"
        aria-label={`Plot of the target curve, the ${net.units.length} hinges and their sum. Current error ${fmt(err, 3)}. ${activation ? '' : 'Activation removed: the sum is a straight line.'}`}>
        <defs><clipPath id="neuron-clip"><rect x={36} y={10} width={W - 46} height={H - 36} /></clipPath></defs>
        {[-1, 0, 1, 2, 3, 4].map((y) => <g key={y}><line className={y === 0 ? 'axis' : 'gridline'} x1={36} x2={W - 10} y1={sy(y)} y2={sy(y)} /><text x={30} y={sy(y) + 4} fontSize={11} textAnchor="end">{y}</text></g>)}
        {[-2, -1, 0, 1, 2].map((x) => <g key={x}><line className={x === 0 ? 'axis' : 'gridline'} x1={sx(x)} x2={sx(x)} y1={10} y2={H - 26} /><text x={sx(x)} y={H - 10} fontSize={11} textAnchor="middle">{x}</text></g>)}
        <g clipPath="url(#neuron-clip)">
          <path d={path(f)} fill="none" stroke="var(--ink-3)" strokeWidth={7} opacity={0.35} strokeLinecap="round" />
          {net.units.map((unit, i) => (
            <path key={i} d={path((x) => unitOutput(unit, x, activation))} fill="none" stroke={i === sel ? 'var(--accent)' : 'var(--ink-2)'} strokeWidth={i === sel ? 1.8 : 1.2} strokeDasharray={DASHES[i]} opacity={unit.v === 0 ? 0.25 : 0.9} />
          ))}
          <path d={path((x) => netOutput(net, x, activation))} fill="none" stroke="var(--accent)" strokeWidth={3} />
          {activation && bend !== null && bend >= X_MIN && bend <= X_MAX && <circle cx={sx(bend)} cy={sy(0)} r={5} fill="var(--card)" stroke="var(--accent)" strokeWidth={2} />}
        </g>
      </svg>
      <p className="lab-note">Thick grey: target. <span className="acc"><b>Thick line</b></span>: the network’s output (sum of all hinges). Dashed: each hinge times its output weight; the selected one is coloured and its bend is circled.</p>

      <div className="controls">
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}>
          <input type="checkbox" checked={!activation} onChange={(e) => setActivation(!e.target.checked)} /> Remove the activation (no ReLU: each unit is just w·x + b)
        </label>
      </div>

      <div className="steps" role="group" aria-label="Choose a hidden unit to adjust">
        {net.units.map((_, i) => <button key={i} className="step-btn" aria-pressed={i === sel} onClick={() => setSel(i)}>unit {i + 1}</button>)}
      </div>
      <div className="controls">
        <Slider label={`unit ${sel + 1} · weight w (steepness and direction)`} value={u.w} min={-2} max={2} step={0.25} onChange={(w) => setUnit({ w })} format={(v) => fmt(v)} />
        <Slider label={`unit ${sel + 1} · bias b (slides the bend sideways)`} value={u.b} min={-2} max={2} step={0.25} onChange={(b) => setUnit({ b })} format={(v) => fmt(v)} />
        <Slider label={`unit ${sel + 1} · output weight v (how much it counts)`} value={u.v} min={-3} max={3} step={0.25} onChange={(v) => setUnit({ v })} format={(v) => fmt(v)} />
      </div>

      <div className="readout" aria-live="polite">
        <span>error (mean squared gap to target) = <b>{fmt(err, 3)}</b></span>
        {activation
          ? <span>unit {sel + 1}: {fmt(u.v)} × relu({fmt(u.w)}·x + {fmt(u.b)}){bend !== null ? `, bends at x = ${fmt(bend)}` : ', never bends (w = 0)'}</span>
          : <span>without ReLU the whole network is the line y = <b>{fmt(line.slope)}</b>·x + <b>{fmt(line.intercept)}</b>, whatever you do</span>}
        {activation && err < 0.002 && <span style={{ color: 'var(--good)' }}>Matched.</span>}
      </div>
      {!activation && <p className="lab-note" style={{ marginTop: 10 }}>Move any slider. Twelve knobs, four units, and the output is still one straight line: only its slope (Σ v·w) and its offset (Σ v·b) change. A line cannot follow a bend.</p>}

      <div className="btn-row" style={{ marginTop: 12 }}>
        <button className="btn small" onClick={() => { setNet(START_NET); setRevealed(false) }}>Reset the units</button>
        {!revealed && <button className="btn small" onClick={() => { setNet(SOLUTIONS[target]); setActivation(true); setRevealed(true) }}>I tried: show one solution</button>}
      </div>
      {revealed && (
        <p className="lab-note" style={{ marginTop: 10 }}>
          {target === 'tent' && <>One solution: <span className="mono">relu(x + 1) − 2·relu(x) + relu(x − 1)</span>. Go up from x = −1, turn downward at 0, flatten out again at 1. Unit 4 is switched off (v = 0).</>}
          {target === 'abs' && <>One solution: <span className="mono">relu(x) + relu(−x)</span>. One hinge handles the right side, a mirrored one (w = −1) handles the left. Two units are unused.</>}
          {target === 'parabola' && <>A smooth curve cannot be matched exactly by 4 straight pieces, only approximated (error {fmt(curveError(SOLUTIONS.parabola, TARGETS.parabola.f), 3)}). With 40 hinges the pieces would be too short to see. More units, better fit.</>}
          {' '}Nobody sets these by hand in a real network: gradient descent finds them.
        </p>
      )}
    </Lab>
  )
}
