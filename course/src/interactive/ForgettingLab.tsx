// Catastrophic forgetting, live: fine-tune a tiny bigram LM on task B and watch task A.
import { useEffect, useMemo, useRef, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import { ADAPTER_RANK, V, ftStep, initFt, measureFt, pretrain, trainableCount, type FtState, type Losses, type Mode } from '../lib/forgetting'

const MODES: { id: Mode; label: string; blurb: string }[] = [
  { id: 'full', label: 'Full fine-tune', blurb: 'Every one of the 729 weights is updated, using task B only.' },
  { id: 'adapter', label: 'Frozen base + small adapter', blurb: `The 729 base weights are frozen. A rank-${ADAPTER_RANK} LoRA bypass (${2 * V * ADAPTER_RANK} numbers) is trained instead.` },
  { id: 'replay', label: 'Mix in 20% of task A (replay)', blurb: 'Every weight is updated, but 20% of each training signal still comes from task A.' },
]
const MAX_STEPS = 300
const PER_TICK = 5

export function ForgettingLab() {
  const base = useMemo(() => pretrain(), [])
  const [mode, setMode] = useState<Mode>('full')
  const [lr, setLr] = useState(10)
  const [state, setState] = useState<FtState>(() => initFt(base, 'full'))
  const [hist, setHist] = useState<Losses[]>(() => [measureFt(initFt(base, 'full'))])
  const [running, setRunning] = useState(false)
  const [finals, setFinals] = useState<Partial<Record<Mode, Losses>>>({})
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const reset = (m: Mode = mode) => {
    if (timer.current) clearTimeout(timer.current)
    const s = initFt(base, m)
    setRunning(false); setState(s); setHist([measureFt(s)])
  }
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current) }, [])
  useEffect(() => {
    if (!running) return
    if (state.step >= MAX_STEPS) { setRunning(false); setFinals((f) => ({ ...f, [mode]: measureFt(state) })); return }
    timer.current = setTimeout(() => {
      const next = ftStep(state, lr, PER_TICK)
      setState(next)
      setHist((h) => [...h, measureFt(next)])
    }, 25)
    return () => { if (timer.current) clearTimeout(timer.current) }
  }, [running, state, lr, mode])

  const now = hist[hist.length - 1]
  const start = hist[0]
  const W = 520, H = 190, lo = 1.9, hi = 2.9
  const x = (i: number) => (i / (MAX_STEPS / PER_TICK)) * W
  const y = (v: number) => H - ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * H
  const line = (f: (l: Losses) => number) => hist.map((l, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(f(l)).toFixed(1)}`).join(' ')

  return (
    <Lab
      title="Watch a model forget"
      goal={<>The model was pretrained on <b>task A</b> (Shakespeare-style text). Now fine-tune it on <b>task B</b> (plain modern text). Press <b>Fine-tune</b> and watch both losses. Then try the two mitigations, and the learning-rate slider.</>}
    >
      <div className="steps" role="group" aria-label="Fine-tuning method">
        {MODES.map((m) => <button key={m.id} className="step-btn" aria-pressed={mode === m.id} onClick={() => { setMode(m.id); reset(m.id) }}>{m.label}</button>)}
      </div>
      <p className="lab-note">{MODES.find((m) => m.id === mode)!.blurb} Trainable numbers: <b>{trainableCount(mode)}</b>.</p>
      <div className="controls">
        <Slider label="Learning rate" value={lr} min={1} max={15} step={1} onChange={setLr} />
        <div className="btn-row" style={{ alignSelf: 'end' }}>
          <button className="btn small primary" disabled={running || state.step >= MAX_STEPS} onClick={() => setRunning(true)}>{state.step === 0 ? 'Fine-tune on B' : 'Continue'}</button>
          <button className="btn small" disabled={!running} onClick={() => setRunning(false)}>Pause</button>
          <button className="btn small" onClick={() => reset()}>Reset to the pretrained model</button>
        </div>
      </div>

      <div className="table-scroll"><svg viewBox={`-40 -10 ${W + 140} ${H + 40}`} style={{ width: '100%', minWidth: 480, display: 'block' }} role="img" aria-label={`Loss curves after ${state.step} steps. Task A loss ${fmt(now.a)}, started at ${fmt(start.a)}. Task B loss ${fmt(now.b)}, started at ${fmt(start.b)}.`}>
        {[2.0, 2.2, 2.4, 2.6, 2.8].map((v) => <g key={v}><line className="gridline" x1={0} x2={W} y1={y(v)} y2={y(v)} /><text x={-6} y={y(v) + 4} textAnchor="end" fontSize={11}>{v.toFixed(1)}</text></g>)}
        <line className="axis" x1={0} y1={H} x2={W} y2={H} /><line className="axis" x1={0} y1={0} x2={0} y2={H} />
        <text x={W / 2} y={H + 24} textAnchor="middle" fontSize={11}>fine-tuning steps (0 to {MAX_STEPS})</text>
        <text x={-34} y={-2} fontSize={11}>loss</text>
        <path d={line((l) => l.b)} fill="none" stroke="var(--accent)" strokeWidth={2.4} />
        <path d={line((l) => l.a)} fill="none" stroke="var(--ink)" strokeWidth={2.4} strokeDasharray="7 4" />
        {mode === 'adapter' && <path d={line((l) => l.aBaseOnly)} fill="none" stroke="var(--good)" strokeWidth={2} strokeDasharray="2 3" />}
        <text x={x(hist.length - 1) + 6} y={y(now.b) + 4} fontSize={11.5} style={{ fill: 'var(--accent)' }}>B {fmt(now.b)}</text>
        <text x={x(hist.length - 1) + 6} y={y(now.a) + (Math.abs(y(now.a) - y(now.b)) < 12 ? -8 : 4)} fontSize={11.5}>A {fmt(now.a)}</text>
        {mode === 'adapter' && <text x={x(hist.length - 1) + 6} y={y(now.aBaseOnly) + 14} fontSize={11.5} style={{ fill: 'var(--good)' }}>A unplugged {fmt(now.aBaseOnly)}</text>}
      </svg></div>
      <p className="muted" style={{ fontSize: 13 }}>Solid line: loss on B, the new task. Dashed line: loss on A, the old skill. {mode === 'adapter' && 'Dotted line: loss on A with the adapter removed.'} Lower is better.</p>

      <div className="readout" aria-live="polite">
        <span>step <b>{state.step}</b></span>
        <span>B: {fmt(start.b)} → <b>{fmt(now.b)}</b></span>
        <span>A: {fmt(start.a)} → <b>{fmt(now.a)}</b> ({now.a - start.a >= 0 ? '+' : '−'}{fmt(Math.abs(now.a - start.a))} {now.a - start.a > 0.02 ? 'forgotten' : ''})</span>
        {mode === 'adapter' && <span>A with adapter unplugged: <b>{fmt(now.aBaseOnly)}</b> (unchanged)</span>}
      </div>

      {Object.keys(finals).length > 0 && (
        <div className="table-scroll" style={{ marginTop: 12 }}>
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>your finished runs (300 steps)</th><th>loss on A</th><th>loss on B</th></tr></thead>
            <tbody>
              <tr><td>pretrained, before fine-tuning</td><td>{fmt(start.a)}</td><td>{fmt(start.b)}</td></tr>
              {MODES.filter((m) => finals[m.id]).map((m) => <tr key={m.id}><td>{m.label}</td><td>{fmt(finals[m.id]!.a)}{m.id === 'adapter' && ` (unplugged: ${fmt(finals[m.id]!.aBaseOnly)})`}</td><td>{fmt(finals[m.id]!.b)}</td></tr>)}
            </tbody>
          </table>
        </div>
      )}
      <p className="lab-note" style={{ marginTop: 10 }}><b>What this is:</b> the character-level neural bigram from lesson 5.1 (a 27×27 weight matrix), trained with exact full-batch gradient descent on one copy of the two texts used by <code>finetune_tiny_gpt.py</code>. To keep it tiny, losses are measured on the training text itself; a real evaluation uses held-out text. The phenomenon is the same one the real script measures on a GPT.</p>
    </Lab>
  )
}
