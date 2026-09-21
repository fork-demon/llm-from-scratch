// One context window, drawn to scale, at step 1 and at step 8 of the same agent run.
// The numbers come from src/lib/contextBudget.ts, so the picture and the lab always agree.
import { DEFAULT_PARAMS, simulate, type Segments } from '../lib/contextBudget'

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const ACC = 'var(--accent)'
const BAD = 'var(--bad)'

const PARAMS = { ...DEFAULT_PARAMS, window: 24000, steps: 8 }
const X0 = 20
const SCALE = 700 / 30000 // pixels per token: the 24k window is 560 px wide

type Key = 'system' | 'toolDefs' | 'docs' | 'task' | 'history' | 'toolResults'
const PARTS: { key: Key; label: string; opacity: number; accent?: boolean }[] = [
  { key: 'system', label: 'system prompt', opacity: 0.8 },
  { key: 'toolDefs', label: 'tool definitions', opacity: 0.58 },
  { key: 'docs', label: 'retrieved documents', opacity: 0.4 },
  { key: 'task', label: 'task', opacity: 0.95 },
  { key: 'history', label: 'the model’s own earlier turns', opacity: 0.25 },
  { key: 'toolResults', label: 'tool results', opacity: 1, accent: true },
]
const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}k` : String(n))

function Bar({ y, seg, step }: { y: number; seg: Segments; step: number }) {
  const windowEnd = X0 + PARAMS.window * SCALE
  let x = X0
  const used = PARTS.reduce((s, p) => s + seg[p.key], 0)
  return (
    <g>
      <text x={X0} y={y - 10} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>Step {step}: the model is sent {used.toLocaleString('en-US')} tokens</text>
      {/* the window itself */}
      <rect x={X0} y={y} width={PARAMS.window * SCALE} height={40} rx={4} fill="none" stroke={INK} strokeWidth={1.5} />
      {PARTS.map((p) => {
        const w = seg[p.key] * SCALE
        const px = x
        x += w
        if (w <= 0) return null
        const inside = Math.max(0, Math.min(w, windowEnd - px))
        return (
          <g key={p.key}>
            <rect x={px} y={y + 1} width={inside} height={38} fill={p.accent ? ACC : INK} opacity={p.opacity * (p.accent ? 0.85 : 1)} />
            {w - inside > 0 && <rect x={px + inside} y={y + 1} width={w - inside} height={38} fill={BAD} opacity={0.8} />}
            <line x1={px} y1={y + 1} x2={px} y2={y + 39} stroke="var(--paper)" strokeWidth={1.5} />
          </g>
        )
      })}
      {used < PARAMS.window && <text x={(x + windowEnd) / 2} y={y + 25} textAnchor="middle" fontSize={12} style={{ fill: SOFT }}>free: {k(PARAMS.window - used)} tokens</text>}
    </g>
  )
}

export function ContextWindowAnatomy() {
  const sim = simulate(PARAMS)
  const first = sim.rows[0].segments
  const last = sim.rows[7].segments
  const windowEnd = X0 + PARAMS.window * SCALE
  const over = sim.rows[7].input - PARAMS.window
  // label positions under the step-1 bar (fixed parts) and under the step-8 bar (what grew)
  let x = X0
  const starts = Object.fromEntries(PARTS.map((p) => { const s = x; x += last[p.key] * SCALE; return [p.key, s] })) as Record<Key, number>

  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 345" style={{ width: '100%', minWidth: 660, display: 'block' }} role="img" aria-labelledby="ctxanat-title">
          <title id="ctxanat-title">A 24,000 token context window drawn to scale. At step 1 it holds 9,200 tokens of system prompt, tool definitions, documents and task, and the rest is free. At step 8 tool results and earlier turns have been appended every step, the total is {sim.rows[7].input.toLocaleString('en-US')} tokens, and {over.toLocaleString('en-US')} of them no longer fit.</title>

          <Bar y={44} seg={first} step={1} />
          {/* labels for the fixed prefix: staggered on two rows so narrow segments do not collide */}
          {([['system', 'system · 2k', 104], ['toolDefs', 'tool definitions · 3k', 122], ['docs', 'retrieved documents · 4k', 104], ['task', 'task · 200', 122]] as [Key, string, number][]).map(([key, label, y]) => (
            <g key={key}>
              <line x1={starts[key] + 2} y1={86} x2={starts[key] + 2} y2={y - 10} stroke={SOFT} strokeWidth={1} />
              <text x={starts[key] + 2} y={y} fontSize={11.5} fontWeight={650} style={{ fill: INK }}>{label}</text>
            </g>
          ))}
          <text x={X0} y={146} fontSize={11.5} style={{ fill: SOFT }}>This prefix is identical on every call. It is re-sent, and re-billed, every step.</text>
          <text x={X0} y={161} fontSize={11.5} style={{ fill: SOFT }}>It is also the part a prompt cache can reuse.</text>

          <Bar y={206} seg={last} step={8} />
          <text x={starts.toolResults - 5} y={264} textAnchor="end" fontSize={11.5} fontWeight={650} style={{ fill: INK }}>earlier turns · 7 × 300</text>
          <text x={starts.toolResults + 3} y={264} fontSize={11.5} fontWeight={700} style={{ fill: ACC }}>tool results · 7 × 2,500 = {k(last.toolResults)}</text>
          <text x={starts.toolResults + 3} y={280} fontSize={11} style={{ fill: SOFT }}>most of the window is now tool output</text>

          {/* the edge of the window */}
          <line x1={windowEnd} y1={30} x2={windowEnd} y2={300} stroke={INK} strokeWidth={1.5} strokeDasharray="5 4" />
          <text x={windowEnd - 6} y={316} textAnchor="end" fontSize={12} fontWeight={700} style={{ fill: INK }}>context window: 24,000 tokens</text>
          <text x={windowEnd + 6} y={196} fontSize={12} fontWeight={700} style={{ fill: BAD }}>does not fit:</text>
          <text x={windowEnd + 6} y={264} fontSize={12} fontWeight={700} style={{ fill: BAD }}>{over.toLocaleString('en-US')} tokens</text>
          <line x1={X0} y1={336} x2={X0 + 5000 * SCALE} y2={336} stroke={LINE} strokeWidth={2} />
          <text x={X0 + 5000 * SCALE + 8} y={340} fontSize={11} style={{ fill: SOFT }}>5,000 tokens</text>
        </svg>
      </div>
    </figure>
  )
}
