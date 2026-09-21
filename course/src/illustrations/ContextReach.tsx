// How far back each approach can see when it predicts the next token of the lesson's own sentence.
// Filled cells = tokens within reach. The accent IS the reach. The fourth row is the open question
// (deliberately unnamed here: the next lesson answers it).
import { ACC, Figure, INK, LINE, MONO, SOFT } from './diagramKit'

const TOKENS = ['The', 'animal', 'did', 'n’t', 'cross', 'the', 'road', 'because', 'it', 'was']
const TARGET = 'tired'
const N = TOKENS.length
const X0 = 186, CW = 52, CH = 26

/** reach(i) = how clearly token i is available, 0 to 1 */
const ROWS: { title: string; sub: string[]; reach: (i: number) => number; note: string }[] = [
  { title: 'Count tables (n-grams)', sub: ['standard into the 2010s'], reach: (i) => (i >= N - 2 ? 1 : 0), note: 'the last 2 tokens' },
  { title: 'Fixed window + MLP', sub: ['Model C; Bengio et al., 2003'], reach: (i) => (i >= N - 4 ? 1 : 0), note: 'the last 4, one slot each' },
  { title: 'Recurrent networks', sub: ['until 2017'], reach: (i) => 0.06 + 0.94 * Math.pow(0.68, N - 1 - i), note: 'everything, squeezed into one summary: the far tokens fade' },
  { title: '?', sub: ['next lesson'], reach: () => 1, note: 'every token equally within reach' },
]
const rowY = (r: number) => 66 + r * 62

export function ContextReach() {
  const uid = 'reach'
  return (
    <Figure
      id={uid}
      height={318}
      minWidth={700}
      title="Four ways to predict the word after “The animal didn’t cross the road because it was”. Count tables see the last 2 tokens, a fixed window the last 4, a recurrent network sees all of them but the distant ones only faintly, and the open question is how to keep every token equally within reach"
      after={<ol className="sr-only">{ROWS.map((r) => <li key={r.title}>{r.title === '?' ? 'Open question' : r.title} ({r.sub.join(', ')}): {r.note}</li>)}</ol>}
    >
      {/* the word that matters for this prediction */}
      <text x={X0 + CW * 1.5} y={16} textAnchor="middle" fontSize={11.5} style={{ fill: INK }}>the word that decides what comes next</text>
      <path d={`M${X0 + CW * 1.5} 24 V${rowY(0) - 12}`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow)`} />
      <text x={X0 + CW * (N + 1) - 4} y={rowY(0) - 22} textAnchor="end" fontSize={11.5} style={{ fill: INK }}>to predict</text>
      <path d={`M${X0 + CW * (N + 0.5)} ${rowY(0) - 17} V${rowY(0) - 8}`} fill="none" stroke={SOFT} strokeWidth={1.5} />

      {ROWS.map((row, r) => {
        const y = rowY(r)
        const open = row.title === '?'
        return (
          <g key={row.title}>
            <text x={12} y={y + 11} fontSize={open ? 20 : 12.5} fontWeight={700} style={{ fill: open ? ACC : INK }}>{row.title}</text>
            {row.sub.map((s, i) => <text key={s} x={open ? 32 : 12} y={open ? y + 10 : y + 27 + i * 14} fontSize={11.5} style={{ fill: SOFT }}>{s}</text>)}
            {TOKENS.map((t, i) => {
              const v = row.reach(i)
              const x = X0 + i * CW
              return (
                <g key={i}>
                  {v > 0 && <rect x={x} y={y - 4} width={CW - 4} height={CH} rx={5} fill={ACC} opacity={0.34 * v} />}
                  <rect x={x} y={y - 4} width={CW - 4} height={CH} rx={5} fill="none" stroke={v > 0 ? ACC : LINE} strokeOpacity={v > 0 ? Math.max(0.25, v) : 1} strokeWidth={1.2} strokeDasharray={v > 0 ? undefined : '3 3'} />
                  <text x={x + (CW - 4) / 2} y={y + 13} textAnchor="middle" fontSize={10.5} style={{ fill: v > 0 ? INK : SOFT, ...MONO }} opacity={v > 0 ? 0.45 + 0.55 * v : 0.7}>{t}</text>
                </g>
              )
            })}
            <rect x={X0 + N * CW} y={y - 4} width={CW - 4} height={CH} rx={5} fill="none" stroke={INK} strokeWidth={1.4} strokeDasharray="4 3" />
            <text x={X0 + N * CW + (CW - 4) / 2} y={y + 13} textAnchor="middle" fontSize={10.5} fontWeight={700} style={{ fill: INK, ...MONO }}>{TARGET}</text>
            <text x={X0} y={y + 38} fontSize={11.5} style={{ fill: SOFT }}>{row.note}</text>
          </g>
        )
      })}
    </Figure>
  )
}
