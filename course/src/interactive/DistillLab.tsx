// Knowledge distillation on one next-token prediction: a teacher's distribution at a chosen
// temperature, and a six-number student learning it from hard labels, sampled labels or soft targets.
import { useMemo, useState } from 'react'
import { Bars, Lab, Slider } from '../components/ui'
import { fmt, softmax } from '../lib/math'
import { MODES, TEACHER_LOGITS, TOKENS, entropy, kl, stepsToReach, train, type Mode } from '../lib/distill'

const STEPS = 200
const Y_MAX = 1.0

const LABEL: Record<Mode, string> = {
  hard: 'hard label (top token only)',
  sampled: 'sampled label (one token the teacher wrote)',
  soft: 'soft targets (the whole distribution)',
}
const STYLE: Record<Mode, { stroke: string; dash?: string; width: number }> = {
  hard: { stroke: 'var(--ink-3)', dash: '6 4', width: 2 },
  sampled: { stroke: 'var(--ink-2)', dash: '2 3', width: 1.6 },
  soft: { stroke: 'var(--accent)', width: 2.6 },
}

export function DistillLab() {
  const [T, setT] = useState(4)
  const [step, setStep] = useState(20)
  const [mode, setMode] = useState<Mode>('soft')

  const teacher1 = softmax(TEACHER_LOGITS, 1)
  const teacherT = softmax(TEACHER_LOGITS, T)
  const traces = useMemo(() => ({
    hard: train('hard', STEPS),
    sampled: train('sampled', STEPS),
    soft: train('soft', STEPS, { T }),
  }), [T])

  const tr = traces[mode]
  const student = softmax(tr.logits[step])
  const lastTarget = step > 0 ? tr.targets[step - 1] : null
  const lastToken = lastTarget && mode !== 'soft' ? TOKENS[lastTarget.indexOf(1)] : null

  // chart geometry
  const W = 640
  const H = 220
  const L = 44
  const R = 12
  const Tp = 12
  const B = 30
  const x = (i: number) => L + (i / STEPS) * (W - L - R)
  const y = (v: number) => Tp + (1 - Math.min(v, Y_MAX) / Y_MAX) * (H - Tp - B)
  const path = (vals: number[]) => vals.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ')

  const reach = (m: Mode) => {
    const s = stepsToReach(traces[m].kl, 0.01)
    return s < 0 ? `not within ${STEPS} steps` : `step ${s}`
  }

  return (
    <Lab
      title="A teacher, a student and a temperature"
      goal={<>The teacher is a trained model finishing “The cat sat on the ___”. The student starts knowing nothing. Pick how the student is taught, drag the step slider, and watch the distance between them (the KL). Then change the temperature and look again at the <b>moon</b> row.</>}
    >
      <h4 style={{ fontSize: 16, margin: '0 0 6px' }}>1 · What the teacher says, at temperature T</h4>
      <Slider label="Temperature T applied to the teacher’s logits" value={T} min={0.5} max={8} step={0.5} onChange={setT} format={(v) => v.toFixed(1)} />
      <div className="grid-2">
        <div>
          <p className="muted" style={{ fontSize: 13.5, margin: '0 0 4px' }}>T = 1 (what the teacher really predicts)</p>
          <Bars items={TOKENS.map((t, i) => ({ label: t, value: teacher1[i] }))} max={1} />
        </div>
        <div>
          <p className="muted" style={{ fontSize: 13.5, margin: '0 0 4px' }}>T = {T.toFixed(1)} (the soft targets the student sees)</p>
          <Bars items={TOKENS.map((t, i) => ({ label: t, value: teacherT[i], tone: 'accent' as const }))} max={1} />
        </div>
      </div>
      <p className="lab-note" aria-live="polite">
        Teacher logits: <span className="mono">[{TEACHER_LOGITS.join(', ')}]</span>, divided by T before softmax. At T = {T.toFixed(1)}, “moon” gets <b>{(teacherT[5] * 100).toFixed(1)}%</b> (at T = 1: {(teacher1[5] * 100).toFixed(2)}%). The <em>order</em> never changes. What changes is how loudly the small numbers speak. Spread, measured as entropy: <b>{fmt(entropy(teacherT))}</b> nats (T = 1: {fmt(entropy(teacher1))}).
      </p>

      <h4 style={{ fontSize: 16, margin: '16px 0 6px' }}>2 · Train the student</h4>
      <div className="steps" role="group" aria-label="How the student is taught">
        {MODES.map((m) => (
          <button key={m} className="step-btn" aria-pressed={mode === m} onClick={() => setMode(m)}>{LABEL[m]}</button>
        ))}
      </div>
      <Slider label="Training steps (one example per step)" value={step} min={0} max={STEPS} step={1} onChange={setStep} format={(v) => String(v)} />

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img"
        aria-label={`Distance from teacher (KL) over ${STEPS} training steps. At step ${step}: hard label ${fmt(traces.hard.kl[step], 3)}, sampled label ${fmt(traces.sampled.kl[step], 3)}, soft targets at T ${T} ${fmt(traces.soft.kl[step], 3)}.`}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="gridline" />
            <text x={L - 6} y={y(v) + 4} textAnchor="end" fontSize={11}>{v.toFixed(2)}</text>
          </g>
        ))}
        {[0, 50, 100, 150, 200].map((s) => <text key={s} x={x(s)} y={H - 10} textAnchor="middle" fontSize={11}>{s}</text>)}
        <line x1={L} x2={L} y1={Tp} y2={H - B} className="axis" />
        <line x1={L} x2={W - R} y1={H - B} y2={H - B} className="axis" />
        <text x={L + 4} y={Tp + 10} fontSize={11}>KL(teacher ‖ student)</text>
        {MODES.map((m) => (
          <path key={m} d={path(traces[m].kl)} fill="none" stroke={STYLE[m].stroke} strokeWidth={m === mode ? STYLE[m].width + 0.8 : STYLE[m].width} strokeDasharray={STYLE[m].dash} opacity={m === mode ? 1 : 0.55} />
        ))}
        <line x1={x(step)} x2={x(step)} y1={Tp} y2={H - B} stroke="var(--accent)" strokeWidth={1} opacity={0.6} />
      </svg>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 18px', fontSize: 13.5, margin: '2px 0 8px' }} aria-hidden>
        {MODES.map((m) => (
          <span key={m} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <svg width="28" height="8"><line x1={0} x2={28} y1={4} y2={4} stroke={STYLE[m].stroke} strokeWidth={STYLE[m].width} strokeDasharray={STYLE[m].dash} /></svg>
            {m === 'soft' ? `soft, T = ${T.toFixed(1)}` : m}
          </span>
        ))}
      </div>

      <div className="readout" aria-live="polite">
        <span>step <b>{step}</b>, {LABEL[mode]}</span>
        <span>KL now: <b>{fmt(tr.kl[step], 3)}</b> nats</span>
        {lastToken && <span>label used at this step: <b>{lastToken}</b></span>}
      </div>

      <div className="table-scroll" style={{ marginTop: 10 }}>
        <table className="plain mono" style={{ fontSize: 13.5 }}>
          <thead><tr><th>token</th><th>teacher (T = 1)</th><th>student now</th><th>student ÷ teacher</th></tr></thead>
          <tbody>
            {TOKENS.map((t, i) => (
              <tr key={t}>
                <td>{t}</td>
                <td>{(teacher1[i] * 100).toFixed(2)}%</td>
                <td><b>{(student[i] * 100).toFixed(2)}%</b></td>
                <td>{fmt(student[i] / teacher1[i], 2)}×</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="readout" style={{ marginTop: 10 }}>
        <span>first step with KL below 0.01 and staying there:</span>
        <span>hard: <b>{reach('hard')}</b></span>
        <span>sampled: <b>{reach('sampled')}</b></span>
        <span>soft, T = {T.toFixed(1)}: <b>{reach('soft')}</b></span>
      </div>

      <p className="lab-note" style={{ marginTop: 10 }}>
        <b>What is real and what is toy:</b> the losses, gradients, the temperature and the T² scaling are the real ones from Hinton, Vinyals and Dean (2015). The student here is only six numbers for one context, so it can copy the teacher perfectly. A real student is a whole network that must fit millions of contexts with fewer weights than the teacher, so it never matches exactly. Learning rate 0.5; the sampled labels come from a fixed random seed so the curve is reproducible.
      </p>
    </Lab>
  )
}
