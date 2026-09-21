// How many different contexts are there, and how many of them has the model ever seen?
import { useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { CORPUS_BASE } from '../lib/bigram'
import { formatPow10, log10Contexts, sparsity } from '../lib/ngram'

const VOCABS = [27, 100, 1000, 10000, 50000]
const LANDMARKS: { log10: number; label: string }[] = [
  { log10: 9.9, label: 'people alive (8 × 10^9)' },
  { log10: 13.2, label: 'tokens used to train a large LLM (about 10^13)' },
  { log10: 80, label: 'atoms in the observable universe (about 10^80)' },
]
const AXIS_MAX = 100

export function ContextExplosion() {
  const [vi, setVi] = useState(0)
  const [n, setN] = useState(3)
  const V = VOCABS[vi]
  const lg = log10Contexts(V, n)
  const rows = useMemo(() => [1, 2, 3, 4, 5, 6, 8, 10].map((k) => sparsity(CORPUS_BASE, k, 27)), [])

  const W = 480
  const x = (l: number) => 20 + (Math.min(l, AXIS_MAX) / AXIS_MAX) * (W - 40)
  const beaten = LANDMARKS.filter((m) => lg > m.log10)

  return (
    <Lab
      title="The table that cannot be built"
      goal={<>A count table needs one row for every possible context. Drag the sliders and watch the number of rows. Then look at how few of those rows a real text ever fills in.</>}
    >
      <div className="controls">
        <Slider label="Vocabulary size V" value={vi} min={0} max={VOCABS.length - 1} step={1} onChange={setVi} format={(i) => `${VOCABS[i].toLocaleString('en-US')}${i === 0 ? ' (our characters)' : i === VOCABS.length - 1 ? ' (GPT-2 sized)' : ''}`} />
        <Slider label="Context length n (tokens the model can see)" value={n} min={1} max={20} step={1} onChange={setN} />
      </div>

      <div className="readout" aria-live="polite">
        <span>rows needed = V<sup>n</sup> = {V.toLocaleString('en-US')}<sup>{n}</sup> = <b>{formatPow10(lg)}</b></span>
        <span>{beaten.length === 0 ? 'still a table you could store' : `more than: ${beaten[beaten.length - 1].label}`}</span>
      </div>

      <svg viewBox={`0 0 ${W} 150`} style={{ width: '100%', maxWidth: W, display: 'block', margin: '12px auto 0' }} role="img" aria-label={`Logarithmic scale. Rows needed: ${formatPow10(lg)}. ${LANDMARKS.map((m) => m.label).join('; ')}.`}>
        <line className="axis" x1={x(0)} y1={92} x2={x(AXIS_MAX)} y2={92} />
        {[0, 20, 40, 60, 80, 100].map((t) => (
          <g key={t}>
            <line className="axis" x1={x(t)} y1={88} x2={x(t)} y2={96} />
            <text x={x(t)} y={112} textAnchor="middle" fontSize={14}>10^{t}</text>
          </g>
        ))}
        {LANDMARKS.map((m, i) => (
          <g key={i}>
            <line x1={x(m.log10)} y1={92} x2={x(m.log10)} y2={50 - i * 14} stroke="var(--ink-3)" strokeDasharray="3 3" />
            <text x={x(m.log10)} y={44 - i * 14} textAnchor="middle" fontSize={13} style={{ fontWeight: 600 }}>{'ABC'[i]}</text>
          </g>
        ))}
        <rect x={x(0)} y={84} width={Math.max(2, x(lg) - x(0))} height={16} fill="var(--accent)" opacity={0.75} rx={3} />
        <text x={x(lg) > W * 0.7 ? x(lg) : x(lg) + 2} y={136} textAnchor={x(lg) > W * 0.7 ? 'end' : 'start'} fontSize={13} style={{ fill: 'var(--accent-ink)', fontWeight: 600 }}>
          rows needed{lg > AXIS_MAX ? ' (off the chart)' : ''}
        </text>
      </svg>
      <p className="lab-note" style={{ textAlign: 'center' }}>
        {LANDMARKS.map((m, i) => <span key={i} style={{ marginRight: 14, display: 'inline-block' }}><b>{'ABC'[i]}</b> {m.label}</span>)}
        <br />The scale is logarithmic: each tick is 100 billion billion times the previous one.
      </p>

      <h4 style={{ marginTop: 18 }}>Now the other side: what does a real text fill in?</h4>
      <p style={{ marginTop: 0 }}>These are measured on our 493-character training text (V = 27). Read across one row.</p>
      <div className="table-scroll">
        <table className="plain mono" style={{ fontSize: 13.5, whiteSpace: 'nowrap' }}>
          <thead><tr><th>context length</th><th>possible contexts</th><th>contexts that occur</th><th>share of the table filled</th><th>seen exactly once</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.n} style={{ fontWeight: r.n === n && vi === 0 ? 700 : 400 }}>
                <td>{r.n}{r.n === n && vi === 0 ? ' ◀' : ''}</td>
                <td>{formatPow10(log10Contexts(27, r.n))}</td>
                <td>{r.distinct}</td>
                <td>{r.coverage >= 0.001 ? `${(r.coverage * 100).toFixed(1)}%` : `${(r.coverage * 100).toExponential(1)}%`}</td>
                <td>{r.once} of {r.distinct} = <b>{(r.onceFraction * 100).toFixed(0)}%</b></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="lab-note">
        Two things happen together. The table grows explosively, and the rows that do get filled are filled by a single example. A count of one is not a statistic, it is a memory.
      </p>
    </Lab>
  )
}
