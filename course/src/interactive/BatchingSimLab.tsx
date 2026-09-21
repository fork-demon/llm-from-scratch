// One stream of requests, three schedulers. The timeline makes the difference visible:
// rows are GPU slots, coloured runs are requests, hatched runs are padding.
import { useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { COST, defaultConfig, makeWorkload, simulate, type KvMode, type Policy, type SimResult } from '../lib/serving'

const POLICIES: { id: Policy; label: string }[] = [
  { id: 'none', label: 'No batching' },
  { id: 'static', label: 'Static batching' },
  { id: 'continuous', label: 'Continuous batching' },
]
const CURVE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
const WINDOWS = [3, 10, 30]
const MAX_ROWS = 16

const ms = (x: number) => (x >= 10000 ? `${(x / 1000).toFixed(1)} s` : x >= 100 ? `${x.toFixed(0)} ms` : `${x.toFixed(1)} ms`)

export function BatchingSimLab() {
  const [rate, setRate] = useState(8)
  const [logBatch, setLogBatch] = useState(4)
  const [sigma, setSigma] = useState(1.0)
  const [logKv, setLogKv] = useState(17)
  const [kvMode, setKvMode] = useState<KvMode>('paged')
  const [policy, setPolicy] = useState<Policy>('static')
  const [windowS, setWindowS] = useState(10)

  const maxBatch = 2 ** logBatch
  const kvBudget = 2 ** logKv

  const workload = useMemo(() => makeWorkload({ ratePerS: rate, outSigma: sigma }), [rate, sigma])
  const results = useMemo(() => {
    const out = {} as Record<Policy, SimResult>
    for (const p of POLICIES) out[p.id] = simulate(workload, defaultConfig({ policy: p.id, maxBatch, kvBudgetTokens: kvBudget, kvMode }))
    return out
  }, [workload, maxBatch, kvBudget, kvMode])
  // the curve uses a saturated server (40 arrivals per second), so it shows capacity, not demand
  const curve = useMemo(() => {
    const wl = makeWorkload({ n: 400, ratePerS: 40, outSigma: sigma })
    return CURVE_BATCHES.map((b) => {
      const m = simulate(wl, defaultConfig({ maxBatch: b, kvBudgetTokens: kvBudget, kvMode }))
      return { b, tput: m.throughputTokS, tpot: m.tpotP50, running: m.meanRunning }
    })
  }, [sigma, kvBudget, kvMode])

  const shown = results[policy]
  const rows = Math.min(MAX_ROWS, shown.maxBatch)
  const tEnd = windowS * 1000

  /* ---- timeline geometry ---- */
  const W = 760, LEFT = 50, RIGHT = 10, ROW = 15, TOP = 34
  const H = TOP + rows * ROW + 26
  const x = (t: number) => LEFT + (Math.min(t, tEnd) / tEnd) * (W - LEFT - RIGHT)
  const segs = shown.timeline.filter((s) => s.slot < rows && s.t0 < tEnd)
  const arrivals = workload.filter((r) => r.arrival < tEnd)
  const ticks = Array.from({ length: 6 }, (_, i) => (tEnd * i) / 5)

  /* ---- curve geometry ---- */
  const CW = 760, CH = 250, CL = 58, CB = 40, CT = 16, CR = 24
  const maxTput = Math.max(...curve.map((c) => c.tput)) * 1.1
  const maxTpot = Math.max(...curve.map((c) => c.tpot)) * 1.1
  const niceStep = (max: number) => { const raw = max / 5; const pow = 10 ** Math.floor(Math.log10(raw)); return ([1, 2, 2.5, 5, 10].find((m) => m * pow >= raw) ?? 10) * pow }
  const yStep = niceStep(maxTput / 1.1), xStep = niceStep(maxTpot / 1.1)
  const yTicks = Array.from({ length: Math.floor(maxTput / yStep) }, (_, i) => (i + 1) * yStep)
  const xTicks = Array.from({ length: Math.floor(maxTpot / xStep) }, (_, i) => (i + 1) * xStep)
  const cx = (tpot: number) => CL + (tpot / maxTpot) * (CW - CL - CR)
  const cy = (tput: number) => CH - CB - (tput / maxTput) * (CH - CB - CT)

  const offered = workload.reduce((s, r) => s + r.out, 0) / (workload[workload.length - 1].arrival / 1000)

  return (
    <Lab
      title="One stream of requests, three schedulers"
      goal={<>The same 200 requests are served three ways. Start with <b>Static batching</b> and find the hatched blocks: finished sequences holding a slot while the longest one runs. Then switch to <b>Continuous batching</b>. Raise the arrival rate until something breaks.</>}
    >
      <div className="controls">
        <Slider label="Arrivals per second" value={rate} min={1} max={40} step={1} onChange={setRate} />
        <Slider label="Slots (max batch)" value={logBatch} min={0} max={6} step={1} onChange={setLogBatch} format={(v) => String(2 ** v)} />
        <Slider label="Output length skew (σ of log-normal)" value={sigma} min={0} max={1.5} step={0.1} onChange={setSigma} format={(v) => v.toFixed(1)} />
        <Slider label="KV memory budget (tokens)" value={logKv} min={11} max={17} step={1} onChange={setLogKv} format={(v) => (2 ** v).toLocaleString('en-US')} />
      </div>
      <div className="steps" role="group" aria-label="How KV memory is handed out">
        <button className="step-btn" aria-pressed={kvMode === 'paged'} onClick={() => setKvMode('paged')}>KV: paged blocks, on demand</button>
        <button className="step-btn" aria-pressed={kvMode === 'reserved'} onClick={() => setKvMode('reserved')}>KV: reserve prompt + 512 up front</button>
      </div>
      <p className="lab-note">
        Offered load: about <b>{offered.toFixed(0)}</b> output tokens per second. Cost model: a decode step takes {COST.overheadMs} ms overhead plus the slower of memory traffic ({COST.weightReadMs} ms to read the weights, plus {COST.kvReadMsPerTok} ms per cached token) and arithmetic ({COST.computeMsPerTok} ms per sequence). It is an illustration of a 7B model in 16-bit on a 2 TB/s accelerator, not a benchmark.
      </p>

      {/* ---------- metrics, side by side ---------- */}
      <div className="table-scroll" aria-live="polite">
        <table className="plain mono" style={{ fontSize: 13.5 }}>
          <thead>
            <tr><th style={{ textAlign: 'left' }}>metric</th>{POLICIES.map((p) => <th key={p.id} style={{ textAlign: 'right' }}>{p.label}</th>)}</tr>
          </thead>
          <tbody>
            {([
              ['slots actually used', (m) => String(m.maxBatch)],
              ['throughput (tokens/s)', (m) => m.throughputTokS.toFixed(0)],
              ['time to first token, p50', (m) => ms(m.ttftP50)],
              ['time to first token, p99', (m) => ms(m.ttftP99)],
              ['time per output token, p50', (m) => ms(m.tpotP50)],
              ['time per output token, p99', (m) => ms(m.tpotP99)],
              ['end-to-end latency, p99', (m) => ms(m.e2eP99)],
              ['slot utilisation while busy', (m) => `${(m.slotUtil * 100).toFixed(0)}%`],
              ['decode steps wasted on padding', (m) => m.paddedSteps.toLocaleString('en-US')],
              ['requests inside the example SLO', (m) => `${(m.sloFraction * 100).toFixed(m.sloFraction > 0 && m.sloFraction < 0.1 ? 1 : 0)}%`],
              ['preemptions (KV ran out)', (m) => String(m.preemptions)],
            ] as [string, (m: SimResult) => string][]).map(([label, f]) => (
              <tr key={label}>
                <td style={{ fontFamily: 'var(--sans)' }}>{label}</td>
                {POLICIES.map((p) => <td key={p.id} style={{ textAlign: 'right', fontWeight: p.id === policy ? 700 : 400 }}>{f(results[p.id])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="lab-note">Example SLO: first token within 1,000 ms, then at most 50 ms per token. Static batching and no batching reserve worst-case KV memory per slot, so a small KV budget also cuts their slots.</p>

      {/* ---------- the timeline ---------- */}
      <div className="steps" role="group" aria-label="Which scheduler to draw" style={{ marginTop: 18, marginBottom: 8 }}>
        {POLICIES.map((p) => <button key={p.id} className="step-btn" aria-pressed={policy === p.id} onClick={() => setPolicy(p.id)}>{p.label}</button>)}
        <span style={{ flex: 1 }} />
        {WINDOWS.map((w) => <button key={w} className="step-btn" aria-pressed={windowS === w} onClick={() => setWindowS(w)}>first {w} s</button>)}
      </div>
      <div className="table-scroll">
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', minWidth: 620, display: 'block' }} role="img" aria-label={`Timeline of the first ${windowS} seconds under ${policy} batching: ${rows} slots, utilisation ${(shown.slotUtil * 100).toFixed(0)} percent, ${shown.paddedSteps} padded steps in the whole run`}>
          <defs>
            <pattern id="bsl-pad" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="6" height="6" fill="var(--bad)" opacity="0.14" />
              <line x1="0" y1="0" x2="0" y2="6" stroke="var(--bad)" strokeWidth="2.2" />
            </pattern>
            <pattern id="bsl-stall" width="5" height="5" patternUnits="userSpaceOnUse">
              <rect width="5" height="5" fill="var(--ink-3)" opacity="0.18" />
              <circle cx="2.5" cy="2.5" r="1" fill="var(--ink-3)" />
            </pattern>
          </defs>
          {/* arrivals */}
          <text x={LEFT - 6} y={20} textAnchor="end" fontSize={10.5} style={{ fill: 'var(--ink-3)' }}>arrivals</text>
          {arrivals.map((r) => <line key={r.id} x1={x(r.arrival)} x2={x(r.arrival)} y1={10} y2={24} stroke="var(--ink-3)" strokeWidth={1} />)}
          {/* slot tracks */}
          {Array.from({ length: rows }, (_, i) => (
            <g key={i}>
              <rect x={LEFT} y={TOP + i * ROW} width={W - LEFT - RIGHT} height={ROW - 3} fill="var(--paper-2)" rx={2} />
              {(i === 0 || i === rows - 1 || (i + 1) % 4 === 0) && <text x={LEFT - 6} y={TOP + i * ROW + 10} textAnchor="end" fontSize={10.5} style={{ fill: 'var(--ink-3)', fontFamily: 'var(--mono)' }}>slot {i + 1}</text>}
            </g>
          ))}
          {segs.map((s, i) => {
            const x0 = x(s.t0), w = Math.max(0.6, x(s.t1) - x0), y = TOP + s.slot * ROW
            if (s.kind === 'pad') return <rect key={i} x={x0} y={y} width={w} height={ROW - 3} fill="url(#bsl-pad)" />
            if (s.kind === 'stall') return <rect key={i} x={x0} y={y} width={w} height={ROW - 3} fill="url(#bsl-stall)" />
            if (s.kind === 'prefill') return <rect key={i} x={x0} y={y} width={w} height={ROW - 3} fill="var(--ink)" />
            return <rect key={i} x={x0} y={y} width={w} height={ROW - 3} fill="var(--accent)" opacity={0.45 + 0.18 * (s.req % 4)} />
          })}
          {/* time axis */}
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={TOP + rows * ROW} y2={TOP + rows * ROW + 4} className="axis" />
              <text x={x(t)} y={TOP + rows * ROW + 17} textAnchor={t === tEnd ? 'end' : t === 0 ? 'start' : 'middle'} fontSize={10.5} style={{ fill: 'var(--ink-3)', fontFamily: 'var(--mono)' }}>{(t / 1000).toFixed(windowS < 10 ? 1 : 0)} s</text>
            </g>
          ))}
        </svg>
      </div>
      <div className="readout" style={{ fontFamily: 'var(--sans)', fontSize: 13.5 }}>
        <span><Swatch fill="var(--ink)" /> prefill</span>
        <span><Swatch fill="var(--accent)" /> decoding (shade changes with the request)</span>
        <span><Swatch fill="url(#bsl-pad)" /> padding: finished, slot still held</span>
        <span><Swatch fill="url(#bsl-stall)" /> stalled by someone else’s prefill</span>
        <span><Swatch fill="var(--paper-2)" stroke /> idle</span>
      </div>
      {shown.maxBatch > MAX_ROWS && <p className="lab-note">Showing the first {MAX_ROWS} of {shown.maxBatch} slots.</p>}

      {/* ---------- throughput versus latency ---------- */}
      <h4 style={{ marginTop: 22, marginBottom: 4 }}>Throughput against per-user latency, as the batch grows</h4>
      <p className="lab-note">Continuous batching on a saturated server (40 arrivals per second), with your skew and KV settings. Each dot is one value of “slots”.</p>
      <div className="table-scroll">
        <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', minWidth: 560, display: 'block' }} role="img" aria-label={`Throughput against time per output token: ${curve.map((c) => `batch ${c.b}: ${c.tput.toFixed(0)} tokens per second at ${c.tpot.toFixed(1)} milliseconds per token`).join('; ')}`}>
          {yTicks.map((v) => (
            <g key={v}>
              <line x1={CL} x2={CW - CR} y1={cy(v)} y2={cy(v)} className="gridline" />
              <text x={CL - 6} y={cy(v) + 4} textAnchor="end" fontSize={10.5} style={{ fill: 'var(--ink-3)', fontFamily: 'var(--mono)' }}>{v.toLocaleString('en-US')}</text>
            </g>
          ))}
          <line x1={CL} x2={CW - CR} y1={CH - CB} y2={CH - CB} className="axis" />
          <line x1={CL} x2={CL} y1={CT} y2={CH - CB} className="axis" />
          {xTicks.map((v) => <text key={v} x={cx(v)} y={CH - CB + 15} textAnchor="middle" fontSize={10.5} style={{ fill: 'var(--ink-3)', fontFamily: 'var(--mono)' }}>{v} ms</text>)}
          <text x={(CL + CW - CR) / 2} y={CH - 6} textAnchor="middle" fontSize={11.5} style={{ fill: 'var(--ink-2)' }}>time per output token, p50 (what each user feels)</text>
          <text x={14} y={(CT + CH - CB) / 2} textAnchor="middle" fontSize={11.5} transform={`rotate(-90 14 ${(CT + CH - CB) / 2})`} style={{ fill: 'var(--ink-2)' }}>tokens/s (what you pay for)</text>
          <polyline points={curve.map((c) => `${cx(c.tpot)},${cy(c.tput)}`).join(' ')} fill="none" stroke="var(--ink-3)" strokeWidth={1.5} />
          {curve.map((c) => {
            const on = c.b === maxBatch
            return (
              <g key={c.b}>
                <circle cx={cx(c.tpot)} cy={cy(c.tput)} r={on ? 6 : 4} fill={on ? 'var(--accent)' : 'var(--paper)'} stroke={on ? 'var(--accent)' : 'var(--ink-2)'} strokeWidth={1.5} />
                <text x={c.b <= 4 ? cx(c.tpot) + 10 : cx(c.tpot)} y={c.b <= 4 ? cy(c.tput) + 4 : cy(c.tput) - 10} textAnchor={c.b <= 4 ? 'start' : 'middle'} fontSize={11} fontWeight={on ? 700 : 400} style={{ fill: on ? 'var(--accent)' : 'var(--ink-2)', fontFamily: 'var(--mono)' }}>{c.b}</text>
              </g>
            )
          })}
        </svg>
      </div>
      <p className="lab-note">If the dots bunch up at the top right, more slots no longer help: the KV budget, not the slot count, decides how many sequences run (mean running at 128 slots: {curve[curve.length - 1].running.toFixed(0)}).</p>
    </Lab>
  )
}

function Swatch({ fill, stroke }: { fill: string; stroke?: boolean }) {
  return (
    <svg width="22" height="11" style={{ verticalAlign: '-1px', marginRight: 5 }} aria-hidden>
      <rect width="22" height="11" rx="2" fill={fill} stroke={stroke ? 'var(--rule-strong)' : 'none'} />
    </svg>
  )
}
