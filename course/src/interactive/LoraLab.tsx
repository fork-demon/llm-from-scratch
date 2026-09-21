// LoRA: (1) count the parameters, (2) watch a low-rank patch try to reproduce a change.
import { useEffect, useMemo, useRef, useState } from 'react'
import { Bars, Lab, MatrixView, Slider, type Tone } from '../components/ui'
import { matmul } from '../lib/math'
import { D, errorByRank, fullParams, initLora, loraParams, loraPercent, relError, targetDelta, trainLora, type LoraState, type TargetKind } from '../lib/lora'

const PRESETS = [
  { label: 'our tiny GPT: 128 × 128, r = 4', dIn: 128, dOut: 128, r: 4 },
  { label: 'our tiny GPT qkv: 128 × 384, r = 4', dIn: 128, dOut: 384, r: 4 },
  { label: '7B-class layer: 4096 × 4096, r = 8', dIn: 4096, dOut: 4096, r: 8 },
  { label: '4096 × 4096, r = 64', dIn: 4096, dOut: 4096, r: 64 },
  { label: '70B-class layer: 8192 × 8192, r = 16', dIn: 8192, dOut: 8192, r: 16 },
]
const n = (x: number) => x.toLocaleString('en-US')

export function LoraCalculator() {
  const [dIn, setDIn] = useState(4096)
  const [dOut, setDOut] = useState(4096)
  const [r, setR] = useState(8)
  const rr = Math.min(r, dIn, dOut)
  const pct = loraPercent(dIn, dOut, rr)

  // drawing: W is a box scaled to its aspect ratio; A and B are strips whose thin side is r
  const big = 150
  const w = dOut >= dIn ? big : Math.max(30, (big * dOut) / dIn)
  const h = dIn >= dOut ? big : Math.max(30, (big * dIn) / dOut)
  const thin = Math.max(3, Math.min(big, (rr / Math.max(dIn, dOut)) * big))
  const num = (v: string, set: (x: number) => void) => set(Math.max(1, Math.min(65536, Math.round(Number(v)) || 1)))

  return (
    <Lab title="How small is a LoRA adapter?" goal={<>Pick a layer shape and a rank. Watch the two thin strips: that is everything that gets trained and saved. Then push <b>r</b> up until the adapter stops being small.</>}>
      <div className="steps" role="group" aria-label="Presets">
        {PRESETS.map((p) => <button key={p.label} className="step-btn" aria-pressed={p.dIn === dIn && p.dOut === dOut && p.r === r} onClick={() => { setDIn(p.dIn); setDOut(p.dOut); setR(p.r) }}>{p.label}</button>)}
      </div>
      <div className="controls">
        <div className="control"><label htmlFor="lora-din"><span>d_in (inputs to the layer)</span></label><input id="lora-din" className="input" type="number" min={1} value={dIn} onChange={(e) => num(e.target.value, setDIn)} /></div>
        <div className="control"><label htmlFor="lora-dout"><span>d_out (outputs of the layer)</span></label><input id="lora-dout" className="input" type="number" min={1} value={dOut} onChange={(e) => num(e.target.value, setDOut)} /></div>
        <Slider label="rank r" value={rr} min={1} max={Math.min(256, dIn, dOut)} step={1} onChange={setR} />
      </div>

      <div className="table-scroll"><svg viewBox="0 0 620 200" style={{ width: '100%', minWidth: 500, maxWidth: 620, display: 'block', margin: '0 auto' }} role="img" aria-label={`Frozen matrix W of ${dIn} by ${dOut}, plus trainable A of ${dIn} by ${rr} times B of ${rr} by ${dOut}`}>
        <rect x={20} y={100 - h / 2} width={w} height={h} fill="var(--paper-2)" stroke="var(--ink-3)" strokeWidth={1.5} strokeDasharray="5 3" />
        <text x={20 + w / 2} y={104} textAnchor="middle" fontSize={15}>W</text>
        <text x={20 + w / 2} y={100 + h / 2 + 16} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)' }}>{dIn} × {dOut} · frozen</text>
        <text x={215} y={106} textAnchor="middle" fontSize={22}>+</text>
        <rect x={260} y={100 - h / 2} width={thin} height={h} fill="var(--accent)" opacity={0.85} />
        <text x={260 + thin / 2} y={100 + h / 2 + 16} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)' }}>A: {dIn} × {rr}</text>
        <text x={260 + thin + 18} y={106} textAnchor="middle" fontSize={16}>·</text>
        <rect x={260 + thin + 36} y={100 - thin / 2} width={w} height={thin} fill="var(--accent)" opacity={0.85} />
        <text x={260 + thin + 36 + w / 2} y={100 + Math.max(thin / 2, 6) + 16} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)' }}>B: {rr} × {dOut} · starts at zero</text>
        <text x={260} y={18} fontSize={12} style={{ fill: 'var(--accent)' }}>trainable</text>
      </svg></div>

      <div className="readout" aria-live="polite" style={{ marginTop: 8 }}>
        <span>full fine-tune: {dIn} × {dOut} = <b>{n(fullParams(dIn, dOut))}</b></span>
        <span>LoRA: {rr} × ({dIn} + {dOut}) = <b>{n(loraParams(dIn, dOut, rr))}</b></span>
        <span>= <b>{pct < 10 ? pct.toFixed(2) : pct.toFixed(0)}%</b> of the layer{pct >= 100 ? ' (no saving left!)' : ''}</span>
      </div>
      <Bars max={1} digits={2} items={[{ label: 'full', value: 1, tone: 'neutral' }, { label: 'LoRA', value: pct / 100, tone: 'accent' }]} />
      <p className="lab-note">Naming follows the repo: the layer computes <span className="mono">x@W + (x@A)@B</span>. The LoRA paper multiplies from the other side and writes the same patch as <span className="mono">BA</span>. Same idea, same count.</p>
    </Lab>
  )
}

export function LoraTrainer() {
  const [kind, setKind] = useState<TargetKind>('low-rank')
  const [r, setR] = useState(1)
  const delta = useMemo(() => targetDelta(kind), [kind])
  const byRank = useMemo(() => errorByRank(kind), [kind])
  const [state, setState] = useState<LoraState>(() => initLora(1))
  const [hist, setHist] = useState<number[]>([1])
  const [running, setRunning] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const MAX = 150

  const reset = (rank = r) => { if (timer.current) clearTimeout(timer.current); setRunning(false); setState(initLora(rank)); setHist([1]) }
  useEffect(() => { reset(r) /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [kind, r])
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current) }, [])

  useEffect(() => {
    if (!running) return
    if (state.step >= MAX) { setRunning(false); return }
    timer.current = setTimeout(() => {
      const next = trainLora(state, delta, 2, 0.02)
      setState(next)
      setHist((h) => [...h, relError(next, delta)])
    }, 30)
    return () => { if (timer.current) clearTimeout(timer.current) }
  }, [running, state, delta])

  const err = relError(state, delta)
  const patch = r === 0 ? delta.map((row) => row.map(() => 0)) : matmul(state.A, state.B)
  const Wd = 300, Hd = 120
  const path = hist.map((e, i) => `${i === 0 ? 'M' : 'L'} ${(i / (MAX / 2)) * Wd} ${Hd - Math.min(1, e) * Hd}`).join(' ')

  return (
    <Lab title="Can a thin patch capture the change?" goal={<>A 6×6 “pretrained” matrix W needs to become W*. Only <span className="mono">A</span> (6×r) and <span className="mono">B</span> (r×6) are trained; W never moves. Choose a rank, press <b>Train</b>, and compare the two kinds of change.</>}>
      <div className="steps" role="group" aria-label="What kind of change does fine-tuning need?">
        <button className="step-btn" aria-pressed={kind === 'low-rank'} onClick={() => setKind('low-rank')}>the needed change is simple (rank 2)</button>
        <button className="step-btn" aria-pressed={kind === 'full-rank'} onClick={() => setKind('full-rank')}>the needed change is arbitrary (rank 6)</button>
      </div>
      <div className="controls">
        <Slider label={`rank r → ${2 * D * r} trainable numbers (full: ${D * D})`} value={r} min={0} max={6} step={1} onChange={setR} />
        <div className="btn-row" style={{ alignSelf: 'end' }}>
          <button className="btn small primary" disabled={running || r === 0 || state.step >= MAX} onClick={() => setRunning(true)}>Train A and B</button>
          <button className="btn small" onClick={() => reset()}>Reset (B = 0)</button>
        </div>
      </div>
      <div className="matrix-row">
        <MatrixView m={delta} heat tone="neutral" digits={1} caption="the change we want: W* − W" />
        <MatrixView m={patch} heat heatMax={Math.max(...delta.flat().map(Math.abs))} tone="accent" digits={1} caption={<>the patch so far: A·B (step {state.step})</>} />
      </div>
      <div className="grid-2" style={{ alignItems: 'start', marginTop: 8 }}>
        <div>
          <svg viewBox={`-30 -8 ${Wd + 40} ${Hd + 30}`} style={{ width: '100%', maxWidth: 380 }} role="img" aria-label={`Remaining error over training steps, now ${(err * 100).toFixed(0)} percent`}>
            <line className="axis" x1={0} y1={Hd} x2={Wd} y2={Hd} /><line className="axis" x1={0} y1={0} x2={0} y2={Hd} />
            <line className="gridline" x1={0} y1={Hd / 2} x2={Wd} y2={Hd / 2} />
            <text x={-6} y={4} textAnchor="end" fontSize={10}>100%</text><text x={-6} y={Hd + 3} textAnchor="end" fontSize={10}>0%</text>
            <text x={Wd / 2} y={Hd + 18} textAnchor="middle" fontSize={10}>training steps (0 to {MAX})</text>
            <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
          </svg>
          <div className="readout" aria-live="polite"><span>remaining error: <b>{(err * 100).toFixed(1)}%</b> of the wanted change</span></div>
          {state.step === 0 && <p className="lab-note">Step 0: B is all zeros, so A·B is all zeros and the layer behaves <b>exactly</b> like the pretrained one.</p>}
        </div>
        <div>
          <p className="lab-note" style={{ marginTop: 0 }}>Best error reachable at each rank for this target (each trained to convergence):</p>
          <Bars max={1} items={byRank.map((e, i) => ({ label: `r = ${i}`, value: e, tone: (i === r ? 'accent' : 'neutral') as Tone }))} />
        </div>
      </div>
      <p className="lab-note">Honest scope: the two targets here are made up. That real fine-tuning changes are close to low-rank is an <em>empirical finding</em> about large pretrained models, not a law.</p>
    </Lab>
  )
}

/** Both halves together. */
export function LoraLab() {
  return (<><LoraCalculator /><LoraTrainer /></>)
}
