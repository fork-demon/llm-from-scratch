// What survives training: pages of text are read and thrown away; each leaves only tiny nudges;
// the one thing that persists is a grid of numbers. No page is inside the grid.
import { ACC, Caption, FILL, Figure, INK, MONO, SOFT } from './diagramKit'

const PAGE_LINES = ['The cat sat on', 'the mat. It was', 'a warm day and', 'the door stood', 'open to the …']

// a fixed scatter of small nudges: [x, y, length, up or down]
const NUDGES: [number, number, number, number][] = []
for (let i = 0; i < 40; i++) {
  const col = i % 8, row = Math.floor(i / 8)
  NUDGES.push([268 + col * 25 + ((i * 7) % 5) * 2, 74 + row * 30 + ((i * 11) % 7) * 2, 5 + ((i * 5) % 4) * 2, (i * 3) % 2 === 0 ? 1 : -1])
}
const cellShade = (k: number) => 0.1 + (((k * 37) % 17) / 17) * 0.62

export function TextIntoWeights() {
  const uid = 'textweights'
  return (
    <Figure
      id={uid}
      height={272}
      title="Pages of training text are read and then discarded. Each leaves only small nudges to the weights. The grid of weights is the only thing that remains, and it contains numbers, not pages"
      after={<ol className="sr-only"><li>Training text: read, then discarded</li><li>Gradient descent: millions of small nudges</li><li>Parameters: the only thing that persists</li></ol>}
    >
      <Caption x={110} y={14} title="Training text" lines={['read once, then discarded']} />
      <Caption x={366} y={14} title="Gradient descent" lines={['millions of small nudges']} />
      <Caption x={620} y={14} title="Parameters" accent lines={['the only thing that persists']} />

      {/* a stack of pages, the older ones already fading; all of it is thrown away */}
      {[2, 1, 0].map((k) => (
        <g key={k} opacity={k === 0 ? 0.75 : k === 1 ? 0.4 : 0.2}>
          <rect x={44 + k * 16} y={64 + k * 10} width={116} height={146} rx={5} fill={FILL} stroke={SOFT} strokeWidth={1.2} strokeDasharray={k === 0 ? undefined : '4 3'} />
        </g>
      ))}
      {PAGE_LINES.map((l, i) => <text key={l} x={54} y={88 + i * 17} fontSize={11} opacity={0.75 - i * 0.12} style={{ fill: INK, fontFamily: 'var(--display)' }}>{l}</text>)}
      {[0, 1].map((i) => <rect key={i} x={54} y={166 + i * 14} width={i === 0 ? 92 : 60} height={5} rx={2.5} fill={INK} opacity={0.12 - i * 0.04} />)}
      <path d="M36 226 L200 58" stroke={INK} strokeWidth={1.5} strokeLinecap="round" />
      <text x={110} y={246} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>not kept anywhere</text>

      {/* the nudges: all that a page leaves behind */}
      <path d="M212 138 H246" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow)`} />
      {NUDGES.map(([x, y, len, dir], i) => (
        <path key={i} d={`M${x} ${y + (dir * len) / 2} V${y - (dir * len) / 2} m-2.5 ${dir * 3} l2.5 ${-dir * 3} l2.5 ${dir * 3}`} fill="none" stroke={SOFT} strokeWidth={1.3} strokeLinecap="round" strokeLinejoin="round" />
      ))}
      <text x={366} y={246} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>each one: a number, a little up or down</text>
      <path d="M484 138 H518" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow)`} />

      {/* the weights: a grid of numbers and nothing else */}
      <rect x={526} y={58} width={188} height={134} rx={8} fill="none" stroke={ACC} strokeWidth={2} />
      {Array.from({ length: 80 }, (_, k) => (
        <rect key={k} x={535 + (k % 10) * 17.2} y={67 + Math.floor(k / 10) * 14.8} width={14} height={11.5} rx={2.5} fill={INK} opacity={cellShade(k)} />
      ))}
      <text x={620} y={212} textAnchor="middle" fontSize={11} style={{ fill: INK, ...MONO }}>0.13 −0.72 0.05 1.31 −0.20 …</text>
      <text x={620} y={232} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>only numbers: no sentence, no page,</text>
      <text x={620} y={246} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>no index to look anything up in</text>
    </Figure>
  )
}
