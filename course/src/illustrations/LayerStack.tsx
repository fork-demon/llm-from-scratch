// A small network drawn in full: 2 → 4 → 4 → 3, every connection a weight.
// Hidden units carry a hinge (the bend); the output units do not, and end in three score bars.
// Under each set of connections: what happens there. Under each column: the shape of the data.
import { ACC, Caption, FILL, Figure, INK, LINE, SOFT } from './diagramKit'

const COLS = [
  { x: 70, n: 2, shape: '2 numbers', bend: false },
  { x: 250, n: 4, shape: '4 numbers', bend: true },
  { x: 430, n: 4, shape: '4 numbers', bend: true },
  { x: 610, n: 3, shape: '3 numbers', bend: false },
]
const MID = 132, GAP = 38, R = 12
const ys = (n: number) => Array.from({ length: n }, (_, i) => MID + (i - (n - 1) / 2) * GAP)
const SCORES = [34, 78, 52]

export function LayerStack() {
  const uid = 'layers'
  return (
    <Figure
      id={uid}
      height={300}
      title="A network with 2 inputs, two hidden layers of 4 units and 3 outputs. Each hidden layer multiplies, shifts and bends; the output layer only multiplies and shifts, giving three scores"
      after={<ol className="sr-only"><li>x: the input, 2 numbers</li><li>Layer 1: multiply, shift, bend; 4 numbers come out</li><li>Layer 2: multiply, shift, bend; 4 numbers come out</li><li>Output layer: multiply, shift, no bend; 3 numbers come out</li><li>Scores: one per class</li></ol>}
    >
      <Caption x={COLS[0].x} y={14} title="x" lines={['the input']} />
      <Caption x={COLS[1].x} y={14} title="layer 1" />
      <Caption x={COLS[2].x} y={14} title="layer 2" />
      <Caption x={COLS[3].x} y={14} title="output layer" />
      <Caption x={704} y={14} title="scores" lines={['one per class']} />

      {/* every connection is one weight */}
      {COLS.slice(0, -1).map((a, k) => ys(a.n).map((y1) => ys(COLS[k + 1].n).map((y2) => (
        <line key={`${k}-${y1}-${y2}`} x1={a.x + R} y1={y1} x2={COLS[k + 1].x - R} y2={y2} stroke={LINE} strokeWidth={1.2} />
      ))))}

      {COLS.map((c) => ys(c.n).map((y) => (
        <g key={`${c.x}-${y}`}>
          <circle cx={c.x} cy={y} r={R} fill={FILL} stroke={INK} strokeWidth={1.4} />
          {c.bend && <path d={`M${c.x - 7} ${y + 3} H${c.x - 1} L${c.x + 6} ${y - 5}`} fill="none" stroke={ACC} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />}
        </g>
      )))}

      {/* what happens between the columns */}
      {[0, 1, 2].map((k) => {
        const x = (COLS[k].x + COLS[k + 1].x) / 2
        const bend = COLS[k + 1].bend
        return (
          <g key={k} textAnchor="middle">
            <path d={`M${COLS[k].x + 42} 226 H${COLS[k + 1].x - 42}`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow)`} />
            <text x={x} y={246} fontSize={12} style={{ fill: INK }}>
              multiply, shift{bend && <>, <tspan fontWeight={700} style={{ fill: ACC }}>bend</tspan></>}
            </text>
            {!bend && <text x={x} y={262} fontSize={11.5} style={{ fill: SOFT }}>no bend at the end</text>}
          </g>
        )
      })}
      {COLS.map((c) => <text key={c.x} x={c.x} y={230} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>{c.shape}</text>)}
      <text x={40} y={288} fontSize={11.5} style={{ fill: SOFT }}>Every line is one weight. Drawn with 4 units per hidden layer; the network in this lesson has 64.</text>

      {/* the three outputs, as score bars */}
      {ys(3).map((y, i) => (
        <g key={y}>
          <line x1={COLS[3].x + R} y1={y} x2={650} y2={y} stroke={LINE} strokeWidth={1.2} />
          <rect x={654} y={y - 7} width={SCORES[i]} height={14} rx={3} fill={INK} opacity={i === 1 ? 0.85 : 0.35} />
        </g>
      ))}
    </Figure>
  )
}
