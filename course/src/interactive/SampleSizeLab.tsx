// How much can you trust a score from n items, and how many items does it take to see a difference?
import { useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { independentDiscordance, nPaired, nUnpaired, normalInterval, simulateScores, standardError } from '../lib/evals'

const SIZES = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000]
const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`
const big = (n: number) => (Number.isFinite(n) ? n.toLocaleString('en-US') : '∞')

export function SampleSizeLab() {
  const [p, setP] = useState(0.8)
  const [ni, setNi] = useState(1)
  const [d, setD] = useState(0.05)
  const [psi, setPsi] = useState(0.1)
  const [seed, setSeed] = useState(1)

  const n = SIZES[ni]
  const pB = Math.min(0.995, p + d)
  const gap = pB - p
  const disagree = Math.max(psi, gap)
  const [loA, hiA] = normalInterval(p, n)
  const [loB, hiB] = normalInterval(pB, n)
  const overlap = hiA >= loB
  const scores = useMemo(() => simulateScores(p, n, 20, seed), [p, n, seed])
  const needU = nUnpaired(p, pB)
  const needP = nPaired(gap, disagree)
  const indep = independentDiscordance(p, pB)

  // chart 1: intervals on a 40% to 100% axis
  const W = 640
  const L = 86
  const R = 20
  const lo = 0.4
  const x = (v: number) => L + ((Math.max(lo, Math.min(1, v)) - lo) / (1 - lo)) * (W - L - R)
  // chart 2: needed items on a log axis from 10 to 100,000
  const lx = (v: number) => L + ((Math.log10(Math.max(10, Math.min(1e5, v))) - 1) / 4) * (W - L - R)

  const bar = (y: number, a: number, b: number, mid: number, tone: string, label: string) => (
    <g>
      <text x={L - 10} y={y + 4.5} textAnchor="end" fontSize={13} fontWeight={650}>{label}</text>
      <line x1={x(a)} y1={y} x2={x(b)} y2={y} stroke={tone} strokeWidth={3} strokeLinecap="round" opacity={0.55} />
      <line x1={x(a)} y1={y - 7} x2={x(a)} y2={y + 7} stroke={tone} strokeWidth={2} />
      <line x1={x(b)} y1={y - 7} x2={x(b)} y2={y + 7} stroke={tone} strokeWidth={2} />
      <circle cx={x(mid)} cy={y} r={6} fill={tone} stroke="var(--paper)" strokeWidth={2} />
    </g>
  )

  return (
    <Lab
      title="How many eval items is enough?"
      goal={<>Set the true accuracy and the size of your eval set. The bar is where a measured score can plausibly land. Then ask the question you actually care about: <b>can this eval tell system A from a slightly better system B?</b> Start with 20 items and try to tell 80% from 85%.</>}
    >
      <div className="controls">
        <Slider label="True accuracy of system A" value={p} min={0.5} max={0.95} step={0.01} onChange={setP} format={(v) => pct(v, 0)} />
        <Slider label="Items in the eval set (n)" value={ni} min={0} max={SIZES.length - 1} step={1} onChange={setNi} format={(i) => big(SIZES[i])} />
        <Slider label="B is truly better by" value={d} min={0.01} max={0.2} step={0.01} onChange={setD} format={(v) => `${(v * 100).toFixed(0)} points`} />
      </div>

      <svg viewBox={`0 0 ${W} 150`} style={{ width: '100%', maxWidth: W, display: 'block' }} role="img"
        aria-label={`System A: true accuracy ${pct(p, 0)}, 95 percent of measured scores fall between ${pct(loA)} and ${pct(hiA)}. System B: ${pct(pB, 0)}, between ${pct(loB)} and ${pct(hiB)}. The ranges ${overlap ? 'overlap' : 'do not overlap'}.`}>
        {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].map((t) => (
          <g key={t}>
            <line className="gridline" x1={x(t)} y1={10} x2={x(t)} y2={116} />
            <text x={x(t)} y={134} textAnchor="middle" fontSize={12} style={{ fill: 'var(--ink-3)' }}>{(t * 100).toFixed(0)}%</text>
          </g>
        ))}
        {overlap && <rect x={x(loB)} y={14} width={Math.max(0, x(hiA) - x(loB))} height={62} fill="var(--bad)" opacity={0.13} />}
        {bar(30, loA, hiA, p, 'var(--ink-3)', 'system A')}
        {bar(60, loB, hiB, pB, 'var(--accent)', 'system B')}
        <text x={L - 10} y={102} textAnchor="end" fontSize={12} style={{ fill: 'var(--ink-3)' }}>20 evals of A</text>
        {scores.map((s, i) => <circle key={i} cx={x(s)} cy={98 + ((i * 7) % 5) * 3 - 6} r={3.2} fill="var(--ink)" opacity={0.45} />)}
      </svg>
      <div className="readout" aria-live="polite">
        <span>standard error = √(p(1−p)/n) = <b>{(standardError(p, n) * 100).toFixed(1)} points</b></span>
        <span>95% interval = <b>± {((hiA - p) * 100).toFixed(1)} points</b></span>
        <span>{overlap ? <>the ranges of A and B <b>overlap</b>: one run cannot rank them</> : <>the ranges of A and B are <b>separate</b></>}</span>
      </div>
      <div className="btn-row" style={{ margin: '10px 0 0' }}>
        <button className="btn small" onClick={() => setSeed(seed + 1)}>Draw 20 new eval sets</button>
        <span className="muted" style={{ fontSize: 14 }}>Each grey dot is the score A would get on a fresh set of {big(n)} items. Nothing about A changes between dots. Lowest here: {pct(Math.min(...scores), 0)}, highest: {pct(Math.max(...scores), 0)}.</span>
      </div>

      <h4 style={{ marginTop: 22, marginBottom: 4 }}>How many items to detect the {(gap * 100).toFixed(0)}-point gap?</h4>
      <p className="lab-note" style={{ marginTop: 0 }}>Planning numbers for a two-sided 5% test that catches the difference 8 times out of 10.</p>
      <div className="controls">
        <Slider label="Paired: items on which A and B disagree" value={disagree} min={Math.ceil(gap * 100) / 100} max={0.5} step={0.01} onChange={setPsi} format={(v) => pct(v, 0)} />
      </div>
      <svg viewBox={`0 0 ${W} 124`} style={{ width: '100%', maxWidth: W, display: 'block' }} role="img"
        aria-label={`Items needed. Unpaired: ${big(needU)} per eval set. Paired: ${big(needP)}. You have ${big(n)}.`}>
        {[10, 100, 1000, 10000, 100000].map((t) => (
          <g key={t}>
            <line className="gridline" x1={lx(t)} y1={6} x2={lx(t)} y2={90} />
            <text x={lx(t)} y={108} textAnchor={t === 100000 ? 'end' : 'middle'} fontSize={12} style={{ fill: 'var(--ink-3)' }}>{big(t)}</text>
          </g>
        ))}
        <text x={L - 10} y={30} textAnchor="end" fontSize={13} fontWeight={650}>unpaired</text>
        <rect x={lx(10)} y={18} width={Math.max(2, lx(needU) - lx(10))} height={18} rx={3} fill="var(--ink-3)" opacity={0.7} />
        <text x={lx(needU) + 8} y={31.5} fontSize={13} style={{ fontFamily: 'var(--mono)' }}>{big(needU)}{needU > 1e5 ? ' (off the chart)' : ''}</text>
        <text x={L - 10} y={66} textAnchor="end" fontSize={13} fontWeight={650}>paired</text>
        <rect x={lx(10)} y={54} width={Math.max(2, lx(needP) - lx(10))} height={18} rx={3} fill="var(--accent)" />
        <text x={lx(needP) + 8} y={67.5} fontSize={13} style={{ fontFamily: 'var(--mono)' }}>{big(needP)}{needP > 1e5 ? ' (off the chart)' : ''}</text>
        <line x1={lx(n)} y1={4} x2={lx(n)} y2={92} stroke="var(--ink)" strokeWidth={1.5} strokeDasharray="4 3" />
        <text x={lx(n) + (lx(n) > W - 120 ? -6 : 6)} y={121} textAnchor={lx(n) > W - 120 ? 'end' : 'start'} fontSize={12} fontWeight={650}>you have {big(n)}</text>
      </svg>
      <div className="readout" aria-live="polite">
        <span>unpaired (a different set for each system): <b>{big(needU)}</b> items each</span>
        <span>paired (the same items through both): <b>{big(needP)}</b> items</span>
        <span>{n >= needP ? <>your {big(n)} items are <b>enough</b> for the paired test</> : <>your {big(n)} items are <b>{(needP / n).toFixed(needP / n < 10 ? 1 : 0)}× too few</b> even for the paired test</>}</span>
      </div>
      <div className="btn-row" style={{ margin: '10px 0 0' }}>
        <button className="btn small" onClick={() => setPsi(gap)}>Best case: B fixes items and breaks none ({pct(gap, 0)})</button>
        <button className="btn small" onClick={() => setPsi(Math.round(indep * 100) / 100)}>Worst case: unrelated errors ({pct(indep, 0)})</button>
      </div>
      <p className="lab-note" style={{ marginTop: 12 }}>
        Why pairing helps: items both systems get right, or both get wrong, say nothing about which is better, and on the same items they cancel out. Only the disagreements carry signal. A prompt tweak to the same model usually changes few answers, so disagreement is low and pairing wins big. Two unrelated systems disagree a lot, and pairing buys almost nothing. These are normal-approximation planning formulas: treat them as order-of-magnitude guidance.
      </p>
    </Lab>
  )
}
