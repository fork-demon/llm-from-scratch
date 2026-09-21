// Test-time compute: what do N attempts buy you, and what do they cost?
// All numbers come from src/lib/testTime.ts (exact formulas + a seeded simulation of the literal procedure).
import { useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { makeRng } from '../lib/rng'
import {
  analytic, curves, distractorRate, drawAttempts, majorityCanWin, majorityPick, precisionWhenAccepted,
  simulate, tokensSpent, verifierPick, verify, type Accuracies,
} from '../lib/testTime'

const MAX_N = 32
const TRIALS = 4000
const LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
const pct = (x: number) => `${(x * 100).toFixed(1)}%`

const PRESETS = [
  { label: 'Right answer is the most common', p: 0.4, m: 4, a: 0.9 },
  { label: 'One popular wrong answer', p: 0.4, m: 1, a: 0.9 },
  { label: 'Hard problem, perfect checker', p: 0.05, m: 8, a: 1 },
] as const

const SERIES: { key: keyof Accuracies; label: string; stroke: string; dash?: string }[] = [
  { key: 'single', label: 'single attempt', stroke: 'var(--ink-3)', dash: '2 5' },
  { key: 'majority', label: 'majority vote', stroke: 'var(--accent)' },
  { key: 'bestOfN', label: 'best-of-N + verifier', stroke: 'var(--ink)', dash: '9 4' },
]

export function TestTimeCompute() {
  const [p, setP] = useState(0.4)
  const [m, setM] = useState(4)
  const [a, setA] = useState(0.9)
  const [N, setN] = useState(5)
  const [perAttempt, setPerAttempt] = useState(500)
  const [seed, setSeed] = useState(1)
  const [mc, setMc] = useState<null | { key: string; result: Accuracies }>(null)

  const params = useMemo(() => ({ p, m, a }), [p, m, a])
  const key = `${p}:${m}:${a}:${N}`
  const all = useMemo(() => curves(params, MAX_N), [params])
  const now = analytic(params, N)
  const rate = distractorRate(p, m)
  const modeOk = majorityCanWin(p, m)
  const ceiling = precisionWhenAccepted(p, a)

  // one concrete set of attempts, so the percentages are about something you can see
  const sample = useMemo(() => {
    const rng = makeRng(seed * 7919 + 13)
    const answers = drawAttempts(params, N, rng)
    const accepted = verify(answers, a, rng)
    return { answers, accepted, vote: majorityPick(answers, m, rng), picked: verifierPick(answers, accepted, rng) }
  }, [params, a, m, N, seed])

  // chart geometry
  const W = 640, H = 300, L = 46, R = 14, T = 14, B = 58
  const x = (n: number) => L + ((n - 1) / (MAX_N - 1)) * (W - L - R)
  const y = (v: number) => T + (1 - v) * (H - T - B)
  const path = (k: keyof Accuracies) => all.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${x(i + 1).toFixed(1)} ${y(pt[k]).toFixed(1)}`).join(' ')
  const ticks = [1, 4, 8, 16, 24, 32]

  return (
    <Lab
      title="What do N attempts buy?"
      goal={<>A pretend model gets a problem right with probability <b>p</b> per attempt. You may run it <b>N</b> times. Compare three ways of turning N attempts into one answer, and watch the token bill. Try each preset, then find settings where majority vote makes things <em>worse</em>.</>}
    >
      <p className="lab-note">
        <b>This is a simulation, not a language model.</b> Each attempt is right with probability p. Otherwise it is one of m different wrong answers, each equally likely. Attempts are independent. Real model errors are messier (they are often correlated), so treat the curves as the shape of the idea, not as a forecast.
      </p>

      <div className="steps" role="group" aria-label="Presets">
        {PRESETS.map((pr) => (
          <button key={pr.label} className="step-btn" aria-pressed={pr.p === p && pr.m === m && pr.a === a} onClick={() => { setP(pr.p); setM(pr.m); setA(pr.a) }}>{pr.label}</button>
        ))}
      </div>

      <div className="controls">
        <Slider label="p: chance one attempt is right" value={p} min={0.05} max={0.95} step={0.05} onChange={setP} format={pct} />
        <Slider label="m: number of different wrong answers" value={m} min={1} max={8} step={1} onChange={setM} />
        <Slider label="N: attempts per problem" value={N} min={1} max={MAX_N} step={1} onChange={setN} />
        <Slider label="verifier accuracy" value={a} min={0.5} max={1} step={0.05} onChange={setA} format={pct} />
        <Slider label="tokens per attempt" value={perAttempt} min={100} max={2000} step={100} onChange={setPerAttempt} format={(v) => v.toLocaleString('en-US')} />
      </div>

      <p className="lab-note">
        <b>The verifier, precisely:</b> it judges every attempt separately. It accepts a correct attempt with probability {pct(a)} and wrongly accepts an incorrect attempt with probability {pct(1 - a)}. We return a random accepted attempt, or a random attempt if none was accepted. 100% is a perfect checker, such as a full set of unit tests. 50% is a coin flip. <b>Majority vote</b> returns the most common answer, ties broken at random.
      </p>

      <div className="table-scroll">{/* on a phone the chart keeps a readable size and scrolls sideways inside this box */}
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', minWidth: 500, maxWidth: W, display: 'block', margin: '4px auto' }} role="img"
        aria-label={`Accuracy against number of attempts. At N = ${N}: single attempt ${pct(now.single)}, majority vote ${pct(now.majority)}, best of N with verifier ${pct(now.bestOfN)}.`}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line className="gridline" x1={L} x2={W - R} y1={y(v)} y2={y(v)} />
            <text x={L - 6} y={y(v) + 4} textAnchor="end" fontSize={11}>{v * 100}%</text>
          </g>
        ))}
        <line className="axis" x1={L} x2={W - R} y1={y(0)} y2={y(0)} />
        <line className="axis" x1={L} x2={L} y1={T} y2={y(0)} />
        {ticks.map((n) => (
          <g key={n}>
            <line className="axis" x1={x(n)} x2={x(n)} y1={y(0)} y2={y(0) + 4} />
            <text x={x(n)} y={y(0) + 17} textAnchor="middle" fontSize={11}>{n}</text>
            <text x={x(n)} y={y(0) + 32} textAnchor="middle" fontSize={10} style={{ fill: 'var(--ink-3)' }}>{(tokensSpent(n, perAttempt) / 1000).toFixed(1)}k</text>
          </g>
        ))}
        <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" fontSize={11}>N attempts (top row) · tokens spent (bottom row): cost is a straight line</text>
        <line x1={x(N)} x2={x(N)} y1={T} y2={y(0)} stroke="var(--rule-strong)" strokeDasharray="3 3" />
        {SERIES.map((s) => (
          <g key={s.key}>
            <path d={path(s.key)} fill="none" stroke={s.stroke} strokeWidth={2.4} strokeDasharray={s.dash} strokeLinejoin="round" />
            <circle cx={x(N)} cy={y(now[s.key])} r={4.5} fill="var(--card)" stroke={s.stroke} strokeWidth={2.2} />
          </g>
        ))}
      </svg>
      </div>
      <div className="readout" style={{ justifyContent: 'center' }} aria-hidden>
        {SERIES.map((s) => (
          <span key={s.key} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <svg width="34" height="8"><line x1="0" x2="34" y1="4" y2="4" stroke={s.stroke} strokeWidth="2.4" strokeDasharray={s.dash} /></svg>{s.label}
          </span>
        ))}
      </div>

      <div className="table-scroll" aria-live="polite">
        <table className="plain">
          <thead><tr><th>with N = {N}</th><th>accuracy (exact)</th><th>attempts</th><th>tokens</th></tr></thead>
          <tbody>
            <tr><td>single attempt</td><td className="mono"><b>{pct(now.single)}</b></td><td className="mono">1</td><td className="mono">{tokensSpent(1, perAttempt).toLocaleString('en-US')}</td></tr>
            <tr><td>majority vote</td><td className="mono"><b>{pct(now.majority)}</b> <span className="muted">({now.majority >= now.single - 1e-9 ? '+' : '−'}{(Math.abs(now.majority - now.single) * 100).toFixed(1)} pts)</span></td><td className="mono">{N}</td><td className="mono">{tokensSpent(N, perAttempt).toLocaleString('en-US')}</td></tr>
            <tr><td>best-of-N + verifier</td><td className="mono"><b>{pct(now.bestOfN)}</b> <span className="muted">(+{((now.bestOfN - now.single) * 100).toFixed(1)} pts)</span></td><td className="mono">{N}</td><td className="mono">{tokensSpent(N, perAttempt).toLocaleString('en-US')} <span className="muted">+ checking</span></td></tr>
          </tbody>
        </table>
      </div>

      <div className="grid-2" style={{ marginTop: 12 }}>
        <div className="card">
          <h4 style={{ fontSize: 16, marginBottom: 6 }}>Will voting help?</h4>
          <p style={{ marginBottom: 6 }}>Right answer: <b className="mono">{pct(p)}</b> of attempts. Each wrong answer: <b className="mono">{pct(rate)}</b>.</p>
          <p style={{ marginBottom: 0 }}>
            {modeOk
              ? <>The right answer is the <b>most common</b> one, so more votes make it win more often.</>
              : p === rate
                ? <>The right answer is <b>tied</b> with the wrong ones. Voting cannot separate them.</>
                : <>A wrong answer is <b>more common</b> than the right one. More votes make the wrong answer win more reliably: accuracy falls below a single attempt.</>}
          </p>
        </div>
        <div className="card">
          <h4 style={{ fontSize: 16, marginBottom: 6 }}>Where does the verifier top out?</h4>
          <p style={{ marginBottom: 6 }}>Of the attempts this verifier accepts, <b className="mono">{pct(ceiling)}</b> are actually right.</p>
          <p style={{ marginBottom: 0 }}>{a >= 1 ? <>A perfect checker never accepts a wrong answer, so accuracy climbs towards 100%: you only need <em>one</em> right attempt among N.</> : a <= 0.5 ? <>A coin-flip verifier carries no information. Best-of-N stays at p however large N gets.</> : <>That is the ceiling. However many attempts you buy, its false accepts keep accuracy at or below {pct(ceiling)}.</>}</p>
        </div>
      </div>

      <h4 style={{ fontSize: 16, margin: '18px 0 6px' }}>One concrete problem, {N} attempt{N === 1 ? '' : 's'}</h4>
      <p className="lab-note" style={{ marginTop: 0 }}>Answer <b>A</b> is the right one. {LETTERS.slice(1, m + 1).join(', ')} {m === 1 ? 'is the wrong answer' : 'are the wrong answers'}. “ok” means the verifier accepted that attempt.</p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }} role="list" aria-label="Sampled attempts">
        {sample.answers.map((ans, i) => (
          <span key={i} role="listitem" className="token" style={{ fontWeight: ans === 0 ? 700 : 400, borderColor: sample.accepted[i] ? 'var(--ink-2)' : undefined, opacity: sample.accepted[i] ? 1 : 0.6 }}>
            {LETTERS[ans]}{sample.accepted[i] ? ' ok' : ''}
          </span>
        ))}
      </div>
      <div className="readout" style={{ marginTop: 10 }} aria-live="polite">
        <span>first attempt: <b>{LETTERS[sample.answers[0]]}</b> {sample.answers[0] === 0 ? '(right)' : '(wrong)'}</span>
        <span>majority vote: <b>{LETTERS[sample.vote]}</b> {sample.vote === 0 ? '(right)' : '(wrong)'}</span>
        <span>verifier’s pick: <b>{LETTERS[sample.picked]}</b> {sample.picked === 0 ? '(right)' : '(wrong)'}</span>
      </div>
      <div className="btn-row" style={{ marginTop: 10 }}>
        <button className="btn small" onClick={() => setSeed(seed + 1)}>Draw another set of attempts</button>
        <button className="btn small primary" onClick={() => setMc({ key, result: simulate(params, N, TRIALS, makeRng(seed + 1000)) })}>Check the formulas: simulate {TRIALS.toLocaleString('en-US')} problems</button>
      </div>
      {mc && mc.key === key && (
        <div className="table-scroll" aria-live="polite" style={{ marginTop: 10 }}>
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>strategy</th><th>exact formula</th><th>measured over {TRIALS.toLocaleString('en-US')} simulated problems</th></tr></thead>
            <tbody>
              {SERIES.map((s) => <tr key={s.key}><td>{s.label}</td><td>{pct(now[s.key])}</td><td>{pct(mc.result[s.key])}</td></tr>)}
            </tbody>
          </table>
          <p className="lab-note">The measured column actually draws attempts, counts votes and runs the noisy verifier. It should land within a point or two of the exact column. The small gap is sampling noise.</p>
        </div>
      )}
    </Lab>
  )
}
