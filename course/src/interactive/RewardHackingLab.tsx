// Reward hacking: an optimiser pushes a policy towards a proxy grader's favourite replies.
// The proxy keeps rising; the true quality rises, peaks and collapses. A KL leash to the
// reference ("SFT") model limits how far the optimiser can go.
import { useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import { expectedCounts, mostLikely, proxyScore, renderReply, run, trueScore } from '../lib/rewardHacking'

const STEPS = 100
const Y_MIN = -3
const Y_MAX = 7

export function RewardHackingLab() {
  const [beta, setBeta] = useState(0)
  const [step, setStep] = useState(12)

  const free = useMemo(() => run(0, STEPS), [])
  const leashed = useMemo(() => run(beta, STEPS), [beta])
  const pt = leashed.points[step]
  const policy = leashed.policies[step]
  const top = mostLikely(policy)
  const counts = expectedCounts(policy)
  const peak = leashed.points.reduce((b, p) => (p.true > b.true ? p : b), leashed.points[0])

  const W = 640
  const H = 240
  const L = 40
  const R = 12
  const Tp = 12
  const B = 30
  const x = (i: number) => L + (i / STEPS) * (W - L - R)
  const y = (v: number) => Tp + ((Y_MAX - v) / (Y_MAX - Y_MIN)) * (H - Tp - B)
  const path = (vals: number[]) => vals.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ')

  return (
    <Lab
      title="Optimise a grader, watch the answers"
      goal={<>A support bot is tuned against a cheap automatic grader. Drag <b>optimisation steps</b> to the right with the leash at 0 and watch the two lines part ways. Then raise <b>β</b> and try again.</>}
    >
      <div className="grid-2">
        <div className="card">
          <h4 style={{ fontSize: 15, margin: '0 0 6px' }}>The grader (proxy reward)</h4>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 14.5 }}>
            <li>a sentence with a real step: <b>+1.0</b></li>
            <li>any other sentence (it likes length): <b>+0.3</b></li>
            <li>each extra “refund” (it likes the keyword): <b>+0.3</b></li>
          </ul>
        </div>
        <div className="card">
          <h4 style={{ fontSize: 15, margin: '0 0 6px' }}>What the customer needs (true quality)</h4>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 14.5 }}>
            <li>a sentence with a real step: <b>+1.0</b></li>
            <li>each filler sentence: <b>−0.4</b> (time wasted)</li>
            <li>each extra “refund”: <b>−0.5</b> (looks broken)</li>
          </ul>
        </div>
      </div>

      <div className="controls" style={{ marginTop: 10 }}>
        <Slider label="Optimisation steps (pressure on the grader)" value={step} min={0} max={STEPS} step={1} onChange={setStep} format={(v) => String(v)} />
        <Slider label="KL leash β (0 = no leash)" value={beta} min={0} max={3} step={0.05} onChange={setBeta} format={(v) => v.toFixed(2)} />
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img"
        aria-label={`Average grader score and average true quality over ${STEPS} optimisation steps with beta ${beta.toFixed(2)}. At step ${step}: grader ${fmt(pt.proxy)}, true quality ${fmt(pt.true)}.`}>
        {[-2, 0, 2, 4, 6].map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className={v === 0 ? 'axis' : 'gridline'} />
            <text x={L - 6} y={y(v) + 4} textAnchor="end" fontSize={11}>{v}</text>
          </g>
        ))}
        {[0, 25, 50, 75, 100].map((s) => <text key={s} x={x(s)} y={H - 10} textAnchor="middle" fontSize={11}>{s}</text>)}
        <line x1={L} x2={L} y1={Tp} y2={H - B} className="axis" />
        {beta > 0 && (
          <>
            <path d={path(free.points.map((p) => p.proxy))} fill="none" stroke="var(--ink-3)" strokeWidth={1.2} strokeDasharray="3 4" opacity={0.6} />
            <path d={path(free.points.map((p) => p.true))} fill="none" stroke="var(--ink-3)" strokeWidth={1.2} strokeDasharray="3 4" opacity={0.6} />
            <text x={x(STEPS) - 2} y={y(free.points[STEPS].true) - 6} textAnchor="end" fontSize={10.5} style={{ fill: 'var(--ink-3)' }}>no leash</text>
          </>
        )}
        <path d={path(leashed.points.map((p) => p.proxy))} fill="none" stroke="var(--accent)" strokeWidth={2.4} />
        <path d={path(leashed.points.map((p) => p.true))} fill="none" stroke="var(--ink)" strokeWidth={2.4} strokeDasharray="7 4" />
        <text x={x(4)} y={y(leashed.points[4].proxy) - 12} fontSize={11} style={{ fill: 'var(--accent)' }}>grader score</text>
        <text x={x(4)} y={y(leashed.points[4].true) + 18} fontSize={11}>true quality (dashed)</text>
        <line x1={x(step)} x2={x(step)} y1={Tp} y2={H - B} stroke="var(--accent)" strokeWidth={1} opacity={0.6} />
        <circle cx={x(step)} cy={y(pt.proxy)} r={4} fill="var(--accent)" />
        <circle cx={x(step)} cy={y(pt.true)} r={4} fill="var(--ink)" />
      </svg>

      <div className="readout" aria-live="polite">
        <span>step <b>{step}</b>, β = <b>{beta.toFixed(2)}</b></span>
        <span>average grader score: <b>{fmt(pt.proxy)}</b> (start 2.05)</span>
        <span>average true quality: <b>{fmt(pt.true)}</b> (start 1.20)</span>
        <span>KL from the SFT model: <b>{fmt(pt.kl)}</b></span>
      </div>
      <p className="lab-note">Best true quality on this curve: <b>{fmt(peak.true)}</b> at step {peak.step}. Average counts per reply now: {fmt(counts.f, 1)} real steps, {fmt(counts.p, 1)} filler sentences, {fmt(counts.k, 1)} extra “refund”s.</p>

      <div className="card" style={{ marginTop: 10 }}>
        <div className="muted" style={{ fontSize: 12.5, marginBottom: 4 }}>THE MODEL’S MOST LIKELY REPLY NOW · grader {fmt(proxyScore(top), 1)}, true {fmt(trueScore(top), 1)}</div>
        <div style={{ fontFamily: 'var(--serif)' }}>{renderReply(top)}</div>
      </div>

      <p className="lab-note" style={{ marginTop: 10 }}>
        <b>What is real and what is toy:</b> the objective, average grader score minus β × KL from the reference model, is the real RLHF objective. Here the “model” chooses among 196 replies described by three counts, and the optimiser takes exact gradient steps instead of noisy ones from sampled replies. Real reward hacking finds flaws nobody wrote down in advance. Here the flaw is printed at the top so you can watch it being found.
      </p>
    </Lab>
  )
}
