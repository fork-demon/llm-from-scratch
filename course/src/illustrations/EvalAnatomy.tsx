// The anatomy of an eval as one picture. Every row is a real golden item and the real output of the
// mini-RAG pipeline under the default configuration, computed by src/lib/evals.ts when the page renders.
import { bootstrapInterval, evaluate, type EvalRow } from '../lib/evals'

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const FILL = 'var(--paper-2)'
const ACC = 'var(--accent)'
const GOOD = 'var(--good)'
const BAD = 'var(--bad)'

const SHOWN = ['d1', 'p7', 'u1', 'a3']
const clip = (s: string, n: number) => (s.length <= n ? s : '…' + s.slice(s.length - n + 1))
const head = (s: string, n: number) => (s.length <= n ? s : s.slice(0, n - 1) + '…')

function Mark({ x, y, ok, size = 7 }: { x: number; y: number; ok: boolean; size?: number }) {
  return ok
    ? <path d={`M${x - size} ${y} l${size * 0.7} ${size * 0.8} l${size * 1.3} ${-size * 1.6}`} fill="none" stroke={GOOD} strokeWidth={size > 5 ? 2.6 : 1.8} strokeLinecap="round" strokeLinejoin="round" />
    : <path d={`M${x - size * 0.8} ${y - size * 0.8} l${size * 1.6} ${size * 1.6} M${x + size * 0.8} ${y - size * 0.8} l${-size * 1.6} ${size * 1.6}`} fill="none" stroke={BAD} strokeWidth={size > 5 ? 2.6 : 1.8} strokeLinecap="round" />
}

function Caption({ x, y, n, title, line }: { x: number; y: number; n: number; title: string; line: string }) {
  return (
    <g>
      <text x={x} y={y} fontSize={12.5} fontWeight={700} style={{ fill: INK }}><tspan style={{ fill: ACC }}>{n} </tspan>{title}</text>
      <text x={x} y={y + 16} fontSize={11.5} style={{ fill: SOFT }}>{line}</text>
    </g>
  )
}

export function EvalAnatomy() {
  const rows = evaluate()
  const byId = new Map(rows.map((r) => [r.item.id, r]))
  const shown = SHOWN.map((id) => byId.get(id)).filter((r): r is EvalRow => !!r)
  const passed = rows.filter((r) => r.passed).length
  const acc = passed / rows.length
  const [lo, hi] = bootstrapInterval(rows.map((r) => Number(r.passed)))
  const ax = (p: number) => 476 + p * 250
  const pc = (v: number) => `${(v * 100).toFixed(0)}%`

  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 400" style={{ width: '100%', minWidth: 680, display: 'block' }} role="img" aria-labelledby="evalanat-title">
          <title id="evalanat-title">Golden items flow through the system under test into a scorer. Four real rows are shown with their pass or fail mark, then all 24 marks are aggregated into an accuracy of {pc(acc)} with a 95 percent interval from {pc(lo)} to {pc(hi)}.</title>
          <defs>
            <marker id="evalanat-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={SOFT} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
          </defs>

          <Caption x={14} y={16} n={1} title="Dataset: the golden set" line="question, and the outcome a person expects" />
          <Caption x={304} y={16} n={2} title="Task" line="called as a user would" />
          <Caption x={440} y={16} n={3} title="Scorer" line="compares what came back with what was expected" />

          {/* the system under test: one box that every row passes through */}
          <rect x={304} y={50} width={104} height={212} rx={10} fill={FILL} stroke={INK} strokeWidth={1.5} />
          <text x={356} y={140} textAnchor="middle" fontSize={12} fontWeight={700} style={{ fill: INK }}>mini-RAG</text>
          <text x={356} y={157} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>retrieve k = 3</text>
          <text x={356} y={172} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>answer or refuse</text>

          {shown.map((r, i) => {
            const y = 58 + i * 52
            return (
              <g key={r.item.id}>
                <rect x={14} y={y} width={262} height={44} rx={7} fill={FILL} stroke={LINE} />
                <text x={24} y={y + 18} fontSize={12} style={{ fill: INK }}>{r.item.q}</text>
                <text x={24} y={y + 35} fontSize={11.5} style={{ fill: SOFT }}>expect: {head(r.item.expected ?? 'a refusal', 34)}</text>
                <path d={`M280 ${y + 22} H298`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#evalanat-arrow)" />
                <path d={`M412 ${y + 22} H432`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#evalanat-arrow)" />
                <rect x={438} y={y} width={270} height={44} rx={7} fill="none" stroke={r.passed ? GOOD : BAD} strokeWidth={1.3} />
                <text x={448} y={y + 18} fontSize={11.5} style={{ fill: INK }}>{r.out.refused ? 'Not found in the provided context.' : clip(r.out.answer.sentence ?? '', 44)}</text>
                <text x={448} y={y + 35} fontSize={11} style={{ fill: SOFT }}>{r.passed ? 'pass' : `fail: ${r.diagnosis === 'should-refuse' ? 'should have refused' : r.diagnosis === 'refused-with-chunk' ? 'had the chunk, refused anyway' : r.reason}`}</text>
                <Mark x={732} y={y + 22} ok={r.passed} />
              </g>
            )
          })}
          <text x={145} y={282} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>… and 20 more items</text>

          {/* aggregate */}
          <Caption x={14} y={312} n={4} title="Aggregate: all 24 marks" line="grouped by category, because one number hides where it is weak" />
          {(() => {
            let x = 14
            let last = ''
            return rows.map((r) => {
              if (r.item.cat !== last) { x += last ? 12 : 0; last = r.item.cat }
              const cx = x + 7
              x += 16.5
              return (
                <g key={r.item.id}>
                  <rect x={cx - 7} y={342} width={14} height={14} rx={3} fill="none" stroke={r.passed ? GOOD : BAD} opacity={0.5} />
                  <Mark x={cx} y={349} ok={r.passed} size={3.6} />
                </g>
              )
            })
          })()}
          {[['direct', 14], ['paraphrase', 158], ['unanswerable', 302], ['adversarial', 380]].map(([t, x]) => <text key={t} x={x as number} y={374} fontSize={11} style={{ fill: SOFT }}>{t}</text>)}

          <text x={470} y={312} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>{passed}/24 = {(acc * 100).toFixed(1)}%</text>
          <text x={470} y={328} fontSize={11.5} style={{ fill: SOFT }}>95% interval {pc(lo)} to {pc(hi)}: a range, not a point</text>
          <line x1={ax(0)} y1={356} x2={ax(1)} y2={356} stroke={LINE} />
          {[0, 0.5, 1].map((t) => <g key={t}><line x1={ax(t)} y1={352} x2={ax(t)} y2={360} stroke={LINE} /><text x={ax(t)} y={376} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>{t * 100}%</text></g>)}
          <line x1={ax(lo)} y1={356} x2={ax(hi)} y2={356} stroke={ACC} strokeWidth={3.5} strokeLinecap="round" opacity={0.6} />
          <line x1={ax(lo)} y1={348} x2={ax(lo)} y2={364} stroke={ACC} strokeWidth={2} />
          <line x1={ax(hi)} y1={348} x2={ax(hi)} y2={364} stroke={ACC} strokeWidth={2} />
          <circle cx={ax(acc)} cy={356} r={6} fill={ACC} />
        </svg>
      </div>
      <ol className="sr-only">
        <li>Dataset: golden items, each a question with the expected outcome.</li>
        <li>Task: each question is sent through the system under test.</li>
        <li>Scorer: each output is compared with the expectation and marked pass or fail.</li>
        <li>Aggregate: the marks become an accuracy per category and overall, with a confidence interval.</li>
      </ol>
    </figure>
  )
}
