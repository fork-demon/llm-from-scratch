// Superposition in the smallest possible model: 5 features squeezed through 2 numbers.
// Train it with plain gradient descent and watch where the 5 feature directions end up.
import { useEffect, useRef, useState } from 'react'
import { Bars, Lab, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import { DEFAULT_STEPS, N_FEATURES, REPRESENTED, SHAPE_TEXT, analyse, createTrainer, evalLoss, importances, initModel, reconstruct, type ToyModel } from '../lib/superposition'

const STEPS_PER_FRAME = 150
const R = 70 // SVG units for an arrow of length 1
const CX = 120
const CY = 120

type Cfg = { sparsity: number; decay: number; seed: number }

export function SuperpositionLab() {
  const [cfg, setCfg] = useState<Cfg>({ sparsity: 0.95, decay: 0.9, seed: 1 })
  const [model, setModel] = useState<ToyModel>(() => initModel(1))
  const [steps, setSteps] = useState(0)
  const [loss, setLoss] = useState<number | null>(null)
  const [running, setRunning] = useState(false)
  const [probe, setProbe] = useState(0)
  const trainer = useRef(createTrainer(cfg))

  // any change of settings starts a fresh, untrained model
  const change = (next: Partial<Cfg>) => {
    const c = { ...cfg, ...next }
    setCfg(c)
    setRunning(false)
    trainer.current = createTrainer(c)
    setModel(initModel(c.seed))
    setSteps(0)
    setLoss(null)
  }

  useEffect(() => {
    if (!running) return
    let raf = 0
    const frame = () => {
      const t = trainer.current
      const k = Math.min(STEPS_PER_FRAME, DEFAULT_STEPS - t.steps)
      const m = t.step(k)
      setModel(m)
      setSteps(t.steps)
      if (t.steps >= DEFAULT_STEPS) {
        setRunning(false)
        setLoss(evalLoss(m, cfg.sparsity, cfg.decay))
      } else raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [running, cfg.sparsity, cfg.decay])

  const a = analyse(model)
  const I = importances(cfg.decay)
  const done = steps >= DEFAULT_STEPS
  const x = Array.from({ length: N_FEATURES }, (_, i) => (i === probe ? 1 : 0))
  const rec = reconstruct(model, x)
  const tip = (i: number) => [CX + model.W[0][i] * R, CY - model.W[1][i] * R]

  return (
    <Lab
      title="Five features, two dimensions"
      goal={<>Five input features must pass through only <b>2 numbers</b> and be rebuilt on the other side. Each arrow is the direction the model gives one feature. Train on <b>dense</b> data, then on <b>sparse</b> data, and compare where the arrows end up.</>}
    >
      <div className="steps" role="group" aria-label="Presets">
        <button className="step-btn" aria-pressed={cfg.sparsity === 0} onClick={() => change({ sparsity: 0 })}>Dense: every feature always on (S = 0)</button>
        <button className="step-btn" aria-pressed={cfg.sparsity === 0.95} onClick={() => change({ sparsity: 0.95 })}>Sparse: each feature on 5% of the time (S = 0.95)</button>
      </div>
      <div className="controls">
        <Slider label="sparsity S (chance a feature is 0)" value={cfg.sparsity} min={0} max={0.99} step={0.01} onChange={(v) => change({ sparsity: v })} format={(v) => v.toFixed(2)} />
        <Slider label="importance decay (feature i matters decayⁱ)" value={cfg.decay} min={0.5} max={1} step={0.05} onChange={(v) => change({ decay: v })} format={(v) => v.toFixed(2)} />
        <Slider label="random seed (starting weights and data)" value={cfg.seed} min={1} max={12} step={1} onChange={(v) => change({ seed: v })} />
      </div>
      <div className="btn-row">
        <button className="btn small primary" disabled={running || done} onClick={() => setRunning(true)}>{steps === 0 ? `Train ${DEFAULT_STEPS.toLocaleString('en-US')} steps` : 'Continue'}</button>
        <button className="btn small" disabled={!running} onClick={() => setRunning(false)}>Pause</button>
        <button className="btn small" onClick={() => change({})}>Reset</button>
        <span className="lab-note">plain gradient descent, learning rate 0.1, batches of 256 random examples</span>
      </div>

      <div className="grid-2" style={{ alignItems: 'start', marginTop: 12 }}>
        <svg viewBox="0 0 240 240" style={{ width: '100%', maxWidth: 360, display: 'block', margin: '0 auto' }} role="img"
          aria-label={`The five feature directions in the 2-D hidden space after ${steps} steps. ${a.represented.length} of 5 features have an arrow longer than ${REPRESENTED}. ${SHAPE_TEXT[a.shape]}.`}>
          <circle cx={CX} cy={CY} r={R} fill="none" className="gridline" strokeDasharray="3 3" />
          <line x1={CX - R - 20} y1={CY} x2={CX + R + 20} y2={CY} className="axis" />
          <line x1={CX} y1={CY - R - 20} x2={CX} y2={CY + R + 20} className="axis" />
          <text x={CX + R + 18} y={CY - 4} fontSize={10.5} textAnchor="end" style={{ fill: 'var(--ink-3)' }}>h₁</text>
          <text x={CX + 4} y={CY - R - 12} fontSize={10.5} style={{ fill: 'var(--ink-3)' }}>h₂</text>
          {model.W[0].map((_, i) => {
            const [tx, ty] = tip(i)
            const on = a.norms[i] > REPRESENTED
            const hot = i === probe
            const lx = CX + (tx - CX) * 1.18 + (a.norms[i] < 0.2 ? 10 : 0)
            const ly = CY + (ty - CY) * 1.18 + 4
            return (
              <g key={i}>
                <line x1={CX} y1={CY} x2={tx} y2={ty} stroke={hot ? 'var(--accent)' : 'var(--ink)'} strokeWidth={1 + 2.5 * I[i]} strokeDasharray={on ? undefined : '3 3'} opacity={on ? 1 : 0.55} strokeLinecap="round" />
                <circle cx={tx} cy={ty} r={3} fill={hot ? 'var(--accent)' : 'var(--ink)'} />
                <text x={lx} y={ly} fontSize={11} textAnchor="middle" style={{ fill: hot ? 'var(--accent)' : 'var(--ink-2)', fontFamily: 'var(--mono)' }}>f{i + 1}</text>
              </g>
            )
          })}
        </svg>
        <div>
          <div className="readout" aria-live="polite">
            <span>steps <b>{steps.toLocaleString('en-US')}</b></span>
            <span>represented <b>{a.represented.length} of 5</b></span>
            {loss !== null && <span>loss <b>{loss.toFixed(4)}</b></span>}
          </div>
          <p className="lab-note" style={{ marginTop: 8 }}>{steps === 0 ? 'Untrained: random short arrows. Press Train.' : done ? <><b>Result:</b> {SHAPE_TEXT[a.shape]}.</> : 'Training…'}</p>
          <p className="lab-note" style={{ marginBottom: 4 }}>Arrow length per feature (1 = fully kept, dashed below {REPRESENTED}):</p>
          <Bars items={a.norms.map((v, i) => ({ label: `f${i + 1} (imp. ${fmt(I[i])})`, value: v, tone: i === probe ? 'accent' : undefined }))} max={1.3} percent={false} />
          <p className="lab-note" style={{ margin: '8px 0 4px' }}>Bias per feature: <span className="mono">[{model.b.map((v) => fmt(v)).join(', ')}]</span></p>
        </div>
      </div>

      <h4 style={{ fontSize: 16, margin: '16px 0 6px' }}>Probe: switch on one feature alone</h4>
      <div className="steps" role="group" aria-label="Feature to switch on">
        {model.W[0].map((_, i) => (
          <button key={i} className="step-btn" aria-pressed={probe === i} onClick={() => setProbe(i)}>f{i + 1} = 1</button>
        ))}
      </div>
      <p className="lab-note">
        Input <span className="mono">[{x.join(', ')}]</span> → hidden <span className="mono">h = [{rec.h.map((v) => fmt(v)).join(', ')}]</span> → rebuilt output after ReLU. Anything non-zero on a feature that was <em>off</em> is <b>interference</b>.
      </p>
      <Bars items={rec.out.map((v, i) => ({ label: `f${i + 1}${i === probe ? ' (on)' : ''}`, value: v, tone: i === probe ? 'accent' : undefined }))} max={1.3} percent={false} />
    </Lab>
  )
}
