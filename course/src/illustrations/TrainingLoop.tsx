// The training loop as a ring, with the data drawn at every station:
// batch (x and the shifted answers y), forward (guesses as bars), loss (one number),
// backward (blame flowing right to left), step (a few weights nudged). In the middle, what it all buys: a falling loss.
// The accent marks the change: the nudged weights and the loss they bring down.
import { ACC, Caption, FILL, Figure, INK, LINE, MONO, SOFT } from './diagramKit'

const CX = 380, CY = 200, RX = 290, RY = 140
const at = (deg: number) => {
  const a = (deg * Math.PI) / 180
  return `${(CX + RX * Math.cos(a)).toFixed(1)} ${(CY + RY * Math.sin(a)).toFixed(1)}`
}
const arc = (from: number, to: number) => `M${at(from)} A${RX} ${RY} 0 0 1 ${at(to)}`

// the model's guesses for the character after "the ca". They sum to 1; the truth is "t" with 0.30,
// so the surprise at this position is -ln(0.30) = 1.20.
const GUESSES: [string, number][] = [['t', 0.3], ['r', 0.38], ['n', 0.16], ['p', 0.1], ['b', 0.06]]
const SNIPPETS: [string, string][] = [['the ca', 'he cat'], ['hello␣', 'ello␣w']]
const WEIGHTS = [0.5, 0.2, 0.7, 0.3, 0.6, 0.15, 0.4, 0.8, 0.25, 0.55, 0.3, 0.65, 0.2, 0.75, 0.45, 0.1, 0.6, 0.35, 0.7, 0.2, 0.45, 0.15, 0.55, 0.3, 0.8, 0.5, 0.25, 0.65, 0.4, 0.6]
const NUDGED = new Set([3, 8, 14, 21, 27])

function Chars({ x, y, s }: { x: number; y: number; s: string }) {
  return (
    <g>
      {[...s].map((ch, i) => (
        <g key={i}>
          <rect x={x + i * 15} y={y} width={14} height={17} rx={3} fill={FILL} stroke={LINE} />
          <text x={x + i * 15 + 7} y={y + 12.5} textAnchor="middle" fontSize={11} style={{ fill: INK, ...MONO }}>{ch === ' ' ? '␣' : ch}</text>
        </g>
      ))}
    </g>
  )
}

export function TrainingLoop() {
  const uid = 'trainloop'
  const curve: string[] = []
  for (let i = 0; i <= 28; i++) curve.push(`${318 + i * 4.5},${(236 - 62 * Math.exp(-i / 7) - 2 * Math.sin(i * 1.7) * Math.exp(-i / 14)).toFixed(1)}`)

  return (
    <Figure
      id={uid}
      height={364}
      minWidth={680}
      title="The training loop drawn as a ring: batch, forward, loss, backward, step, and round again, with the loss falling in the middle"
      after={<ol className="sr-only"><li>Batch: grab random stretches of text; the answer y is the input x shifted by one character</li><li>Forward: the model guesses every next character</li><li>Loss: the average surprise, one number</li><li>Backward: blame for every number, flowing back through the layers</li><li>Step: nudge every number a little, then go round again</li></ol>}
    >
      {/* ---- the ring ---- */}
      {[[216, 249], [-67, -37], [7, 24], [64, 116], [156, 167]].map(([a, b]) => (
        <path key={a} d={arc(a, b)} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow)`} />
      ))}

      {/* ---- 1 batch ---- */}
      <Caption x={14} y={120} n={1} title="batch" anchor="start" lines={['random stretches of text; the answer y', 'is the input x shifted by one character']} />
      {(['x', 'y'] as const).map((label, r) => (
        <g key={label}>
          <text x={18} y={177 + r * 30} fontSize={11.5} fontWeight={700} style={{ fill: SOFT, ...MONO }}>{label}</text>
          {SNIPPETS.map((sn, k) => <Chars key={k} x={34 + k * 104} y={164 + r * 30} s={sn[r]} />)}
        </g>
      ))}
      {/* the shift: each answer is the input's next character */}
      {SNIPPETS.map((_, k) => [0, 1, 2, 3, 4].map((i) => (
        <line key={`${k}-${i}`} x1={34 + k * 104 + (i + 1) * 15 + 7} y1={182.5} x2={34 + k * 104 + i * 15 + 7} y2={192.5} stroke={SOFT} strokeWidth={1.2} />
      )))}
      <text x={238} y={192} fontSize={13} style={{ fill: SOFT }}>…</text>

      {/* ---- 2 forward ---- */}
      <Caption x={CX} y={14} n={2} title="forward" lines={['the model guesses every next character']} />
      {GUESSES.map(([ch, p], i) => {
        const h = p * 140
        const truth = i === 0
        return (
          <g key={ch}>
            <rect x={318 + i * 26} y={104 - h} width={18} height={h} rx={3} fill={INK} opacity={truth ? 0.9 : 0.3} />
            <text x={327 + i * 26} y={118} textAnchor="middle" fontSize={11.5} fontWeight={truth ? 700 : 400} style={{ fill: truth ? INK : SOFT, ...MONO }}>{ch}</text>
          </g>
        )
      })}
      <text x={327} y={56} textAnchor="middle" fontSize={11} style={{ fill: INK }}>truth</text>
      <text x={CX} y={134} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>its guesses after “the ca”</text>

      {/* ---- 3 loss ---- */}
      <Caption x={604} y={134} n={3} title="loss" anchor="start" />
      <text x={604} y={166} fontSize={22} fontWeight={700} style={{ fill: INK, ...MONO }}>1.20</text>
      <text x={604} y={186} fontSize={11.5} style={{ fill: SOFT }}>surprise at “t”: −ln 0.30,</text>
      <text x={604} y={200} fontSize={11.5} style={{ fill: SOFT }}>averaged over the batch</text>

      {/* ---- 4 backward ---- */}
      <Caption x={536} y={282} n={4} title="backward" anchor="start" lines={['blame flows back through the layers']} />
      {[0, 1, 2].map((i) => (
        <g key={i}>
          <rect x={548 + i * 52} y={310} width={20} height={42} rx={4} fill={INK} opacity={0.2 + i * 0.12} />
          <path d={`M${548 + (i + 1) * 52 - 6} 331 H${548 + i * 52 + 27}`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow)`} />
        </g>
      ))}
      <text x={704} y={335} fontSize={11.5} style={{ fill: INK, ...MONO }}>loss</text>

      {/* ---- 5 step ---- */}
      <Caption x={30} y={282} n={5} title="step" anchor="start" lines={['nudge every number a little']} />
      {WEIGHTS.map((v, k) => {
        const hot = NUDGED.has(k)
        return <rect key={k} x={30 + (k % 10) * 19} y={310 + Math.floor(k / 10) * 15} width={16} height={12} rx={3} fill={hot ? ACC : INK} opacity={hot ? 1 : 0.12 + v * 0.5} />
      })}

      {/* ---- what the ring buys: a falling loss ---- */}
      <path d="M312 166 V240 H452" fill="none" stroke={LINE} strokeWidth={1.5} />
      <polyline points={curve.join(' ')} fill="none" stroke={ACC} strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" />
      <text x={382} y={258} textAnchor="middle" fontSize={11.5} fontWeight={700} style={{ fill: ACC }}>loss, step after step</text>
    </Figure>
  )
}
