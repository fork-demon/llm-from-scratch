// What fills an agent's context window at every step, when it overflows, and what the run costs.
// All arithmetic is in src/lib/contextBudget.ts.
import { useId, useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { DEFAULT_PARAMS, simulate, type BudgetParams, type CallRow, type Segments } from '../lib/contextBudget'

const WINDOWS = [8000, 16000, 32000, 64000, 128000, 200000]
const LAYERS: { key: keyof Segments; label: string; fill: string; opacity: number }[] = [
  { key: 'system', label: 'system prompt', fill: 'var(--ink)', opacity: 0.8 },
  { key: 'toolDefs', label: 'tool definitions', fill: 'var(--ink)', opacity: 0.58 },
  { key: 'docs', label: 'documents stuffed up front', fill: 'var(--ink)', opacity: 0.4 },
  { key: 'task', label: 'task', fill: 'var(--ink)', opacity: 0.95 },
  { key: 'summary', label: 'summary or sub-agent report', fill: 'url(#ctxlab-hatch)', opacity: 1 },
  { key: 'history', label: 'the model’s earlier turns', fill: 'var(--ink)', opacity: 0.22 },
  { key: 'toolResults', label: 'tool results', fill: 'var(--accent)', opacity: 0.9 },
]
const int = (n: number) => Math.round(n).toLocaleString('en-US')
const kTok = (n: number) => (n >= 1000 ? `${Math.round(n / 1000)}k` : String(n))
const usd = (n: number) => `$${n < 1 ? n.toFixed(3) : n.toFixed(2)}`

function Chart({ rows, window: win }: { rows: CallRow[]; window: number }) {
  const W = 680
  const H = 280
  const L = 46
  const R = 12
  const T = 14
  const B = 40
  const top = Math.max(win * 1.12, ...rows.map((r) => r.input + r.output))
  const y = (v: number) => T + (1 - v / top) * (H - T - B)
  const slot = (W - L - R) / rows.length
  const bw = Math.max(3, Math.min(34, slot - 3))
  const firstMain = rows.findIndex((r) => r.agent === 'main')
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * win)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, display: 'block' }} role="img"
      aria-label={`Stacked bars: tokens sent on each of ${rows.length} model calls, against a window of ${int(win)} tokens. ${rows.some((r) => r.overflow) ? `The window overflows at call ${rows.findIndex((r) => r.overflow) + 1}.` : 'Every call fits.'}`}>
      <defs>
        <pattern id="ctxlab-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" fill="var(--paper-2)" /><line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink)" strokeWidth="2.4" />
        </pattern>
      </defs>
      {ticks.map((t) => (
        <g key={t}>
          <line className="gridline" x1={L} y1={y(t)} x2={W - R} y2={y(t)} />
          <text x={L - 6} y={y(t) + 4} textAnchor="end" fontSize={11.5} style={{ fill: 'var(--ink-3)' }}>{kTok(t)}</text>
        </g>
      ))}
      {rows.map((r, i) => {
        const x = L + i * slot + (slot - bw) / 2
        let acc = 0
        const compaction = r.kind === 'compaction'
        return (
          <g key={i} opacity={compaction ? 0.55 : 1}>
            {LAYERS.map((l) => {
              const v = r.segments[l.key]
              if (v <= 0) return null
              const y1 = y(acc + v)
              const h = y(acc) - y1
              acc += v
              return <rect key={l.key} x={x} y={y1} width={bw} height={Math.max(0.5, h)} fill={l.fill} opacity={l.opacity} />
            })}
            {r.input > win && <rect x={x - 1} y={y(r.input)} width={bw + 2} height={y(win) - y(r.input)} fill="var(--bad)" opacity={0.85} />}
            {rows.length <= 26 && <text x={x + bw / 2} y={H - B + 15} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)' }}>{compaction ? 'C' : r.step}</text>}
          </g>
        )
      })}
      <line x1={L} y1={y(win)} x2={W - R} y2={y(win)} stroke="var(--ink)" strokeWidth={1.5} strokeDasharray="5 4" />
      <text x={L + 6} y={y(win) - 6} fontSize={12} fontWeight={700}>window: {int(win)} tokens</text>
      {firstMain > 0 && (
        <g>
          <line x1={L + firstMain * slot} y1={T} x2={L + firstMain * slot} y2={H - B + 22} stroke="var(--ink-3)" strokeDasharray="2 3" />
          <text x={L + (firstMain * slot) / 2} y={H - 6} textAnchor="middle" fontSize={11.5} fontWeight={650}>sub-agent’s own window</text>
          <text x={L + firstMain * slot + ((rows.length - firstMain) * slot) / 2} y={H - 6} textAnchor="middle" fontSize={11.5} fontWeight={650}>main agent</text>
        </g>
      )}
      {firstMain <= 0 && <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" fontSize={11.5} style={{ fill: 'var(--ink-3)' }}>model call (step){rows.some((r) => r.kind === 'compaction') ? ' · C = compaction call' : ''}</text>}
    </svg>
  )
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  const id = useId()
  return <label htmlFor={id} style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5, fontWeight: 600 }}><input id={id} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} /> {label}</label>
}

export function ContextBudgetLab() {
  const [p, setP] = useState<BudgetParams>(DEFAULT_PARAMS)
  const [trunc, setTrunc] = useState(false)
  const [truncTo, setTruncTo] = useState(500)
  const [summ, setSumm] = useState(false)
  const [every, setEvery] = useState(4)
  const [sub, setSub] = useState(false)
  const [subSteps, setSubSteps] = useState(6)
  const set = (patch: Partial<BudgetParams>) => setP({ ...p, ...patch })

  const active: BudgetParams = { ...p, truncateTo: trunc ? truncTo : null, summariseEvery: summ ? every : null, subAgentSteps: sub ? Math.min(subSteps, p.steps - 1) : 0 }
  const sim = useMemo(() => simulate(active), [JSON.stringify(active)]) // eslint-disable-line react-hooks/exhaustive-deps
  const naive = useMemo(() => simulate({ ...p, truncateTo: null, summariseEvery: null, subAgentSteps: 0 }), [p])
  const anyStrategy = trunc || summ || sub
  const delta = (a: number, b: number) => `${a <= b ? '−' : '+'}${Math.abs((1 - a / b) * 100).toFixed(0)}%`
  const of = sim.overflowAt

  return (
    <Lab
      title="Where do the tokens go?"
      goal={<>An agent re-sends its whole context on every step. Set up a run, find the step where the window overflows, then switch on strategies one at a time and watch the bars, the bill and the cache. Prices are <b>example numbers</b>: put in your own.</>}
    >
      <div className="controls">
        <Slider label="System prompt" value={p.system} min={0} max={8000} step={500} onChange={(v) => set({ system: v })} format={int} />
        <Slider label="Tool definitions" value={p.toolDefs} min={0} max={20000} step={1000} onChange={(v) => set({ toolDefs: v })} format={int} />
        <Slider label="Documents stuffed up front" value={p.docs} min={0} max={40000} step={2000} onChange={(v) => set({ docs: v })} format={int} />
        <Slider label="Average tool result" value={p.toolResult} min={100} max={10000} step={100} onChange={(v) => set({ toolResult: v })} format={int} />
        <Slider label="Steps in the run" value={p.steps} min={2} max={40} step={1} onChange={(v) => set({ steps: v })} />
        <Slider label="Context window" value={WINDOWS.indexOf(p.window)} min={0} max={WINDOWS.length - 1} step={1} onChange={(i) => set({ window: WINDOWS[i] })} format={(i) => int(WINDOWS[i])} />
      </div>

      <div className="grid-3" style={{ marginBottom: 14 }}>
        <div className="card" style={{ padding: 12 }}>
          <Toggle label="Truncate tool results" checked={trunc} onChange={setTrunc} />
          <Slider label="keep at most" value={truncTo} min={100} max={2000} step={100} onChange={setTruncTo} format={(v) => `${int(v)} tokens`} />
          <p className="lab-note" style={{ margin: 0, fontSize: 13.5 }}>The rest goes to a file. The model gets a reference it can ask for.</p>
        </div>
        <div className="card" style={{ padding: 12 }}>
          <Toggle label="Summarise the history" checked={summ} onChange={setSumm} />
          <Slider label="every" value={every} min={2} max={10} step={1} onChange={setEvery} format={(v) => `${v} steps`} />
          <p className="lab-note" style={{ margin: 0, fontSize: 13.5 }}>Costs one extra model call, and the cache after the prefix is lost.</p>
        </div>
        <div className="card" style={{ padding: 12 }}>
          <Toggle label="Sub-agent for the research phase" checked={sub} onChange={setSub} />
          <Slider label="first" value={Math.min(subSteps, p.steps - 1)} min={1} max={Math.max(1, p.steps - 1)} step={1} onChange={setSubSteps} format={(v) => `${v} steps`} />
          <p className="lab-note" style={{ margin: 0, fontSize: 13.5 }}>Its own window. Only a {int(p.summaryTokens)}-token report comes back.</p>
        </div>
      </div>

      <Chart rows={sim.rows} window={p.window} />
      <p className="lab-note" style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', marginTop: 6 }}>
        {LAYERS.map((l) => (
          <span key={l.key} style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <svg width="12" height="12" aria-hidden><rect width="12" height="12" fill={l.key === 'summary' ? 'var(--paper-2)' : l.fill} opacity={l.opacity} stroke={l.key === 'summary' ? 'var(--ink)' : 'none'} strokeDasharray={l.key === 'summary' ? '2 2' : undefined} /></svg>{l.label}{l.key === 'summary' ? ' (hatched)' : ''}
          </span>
        ))}
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}><svg width="12" height="12" aria-hidden><rect width="12" height="12" fill="var(--bad)" /></svg>does not fit</span>
      </p>

      <div className="readout" aria-live="polite" style={{ borderLeft: `4px solid ${of ? 'var(--bad)' : 'var(--good)'}` }}>
        <span>{of ? <>window overflows at <b>{of.agent === 'sub' ? 'sub-agent ' : ''}{of.kind === 'compaction' ? 'the compaction after ' : ''}step {of.step}</b></> : <>every call <b>fits</b></>}</span>
        <span>largest call: <b>{int(sim.peak)}</b> tokens</span>
        <span>model calls: <b>{sim.rows.length}</b></span>
      </div>
      <div className="table-scroll">
        <table className="plain mono" style={{ fontSize: 13.5 }}>
          <thead><tr><th /><th>input tokens billed</th><th>of which cache hits</th><th>output tokens</th><th>cost, no caching</th><th>cost, with caching</th></tr></thead>
          <tbody>
            <tr><td style={{ fontFamily: 'var(--sans)' }}>naive: resend everything</td><td>{int(naive.billedInput)}</td><td>{int(naive.cachedInput)}</td><td>{int(naive.billedOutput)}</td><td>{usd(naive.cost)}</td><td>{usd(naive.costCached)}</td></tr>
            {anyStrategy && <tr><td style={{ fontFamily: 'var(--sans)' }}><b>with your strategies</b></td><td><b>{int(sim.billedInput)}</b> ({delta(sim.billedInput, naive.billedInput)})</td><td>{int(sim.cachedInput)}</td><td>{int(sim.billedOutput)}</td><td>{usd(sim.cost)} ({delta(sim.cost, naive.cost)})</td><td>{usd(sim.costCached)} ({delta(sim.costCached, naive.costCached)})</td></tr>}
          </tbody>
        </table>
      </div>
      <p className="lab-note">A {p.steps}-step run sends the fixed prefix {p.steps} times and the appended material 1 + 2 + … + {p.steps - 1} = {(p.steps * (p.steps - 1)) / 2} times. Everything the model wrote is only {int(naive.billedOutput)} output tokens. Notice that caching lowers the bill and leaves the bars exactly where they were.</p>

      <div className="controls" style={{ marginTop: 14 }}>
        <Slider label="Example price: input, $ per million tokens" value={p.priceIn} min={0.1} max={20} step={0.1} onChange={(v) => set({ priceIn: v })} format={(v) => `$${v.toFixed(2)}`} />
        <Slider label="Example price: output, $ per million tokens" value={p.priceOut} min={0.1} max={100} step={0.1} onChange={(v) => set({ priceOut: v })} format={(v) => `$${v.toFixed(2)}`} />
        <Slider label="Cached input billed at (% of input price)" value={p.cacheRead} min={0.05} max={1} step={0.05} onChange={(v) => set({ cacheRead: v })} format={(v) => `${(v * 100).toFixed(0)}%`} />
        <Slider label="Cache writes billed at (% of input price)" value={p.cacheWrite} min={1} max={2} step={0.05} onChange={(v) => set({ cacheWrite: v })} format={(v) => `${(v * 100).toFixed(0)}%`} />
      </div>
      <p className="lab-note"><b>Simplifications, stated:</b> every step adds the same amounts, a summary is always {int(p.summaryTokens)} tokens and loses nothing that matters, the sub-agent needs no documents, and every cache lookup hits. Real runs are lumpier and real summaries do lose things. The shape of the curves is the lesson.</p>
    </Lab>
  )
}
