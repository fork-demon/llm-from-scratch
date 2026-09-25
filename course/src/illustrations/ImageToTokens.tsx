// An image becomes tokens: cut into patches -> flatten one patch -> project to D numbers ->
// the patch tokens sit in the same sequence as the text tokens. Used by the multimodal lesson.
import { ACC, Caption, FILL, Figure, INK, LINE, MONO, SOFT } from './diagramKit'

// a 4 × 4 "screenshot": light cells, a dark app bar on top, a red-ish centre, a blue button at the bottom
const CELLS = [
  [0.75, 0.75, 0.75, 0.75],
  [0.08, 0.3, 0.3, 0.08],
  [0.08, 0.2, 0.2, 0.08],
  [0.08, 0.45, 0.45, 0.08],
]
const PATCH = [0.1, 0.12, 0.5, 0.85, 0.3, 0.2, 0.6, 0.15, 0.4, 0.7, 0.25, 0.5] // opacity per drawn number
const TOKEN = [0.6, 0.2, 0.85, 0.35, 0.7, 0.45]

export function ImageToTokens() {
  const id = 'img2tok'
  const hot = { r: 1, c: 1 } // the patch we follow
  const cx = (c: number) => 30 + c * 30
  const cy = (r: number) => 64 + r * 30
  const MID = 118
  return (
    <Figure
      id={id}
      height={214}
      title="An image is cut into a grid of square patches. One patch is flattened into a long list of numbers, multiplied by a learned matrix to give D numbers, and placed in the model's input sequence between text tokens."
      after={<ol className="sr-only"><li>Image: cut into P by P pixel patches</li><li>Flatten: one 14 by 14 patch with 3 colours is 588 numbers</li><li>Project: multiply by a learned matrix and add a position vector, giving D numbers, one token</li><li>One sequence: the image tokens sit between the text tokens and the Transformer reads them all</li></ol>}
    >
      <Caption x={90} y={16} n={1} title="Image" lines={['cut into P × P patches']} />
      {CELLS.map((row, r) => row.map((v, c) => (
        <rect key={`${r}-${c}`} x={cx(c)} y={cy(r)} width={28} height={28} rx={2} fill={INK} opacity={0.1 + v * 0.6} />
      )))}
      <rect x={cx(hot.c) - 2} y={cy(hot.r) - 2} width={32} height={32} rx={3} fill="none" stroke={ACC} strokeWidth={2.2} />

      <path d={`M${cx(3) + 32} ${cy(hot.r) + 14} C 172 ${cy(hot.r) + 14}, 172 ${MID}, 196 ${MID}`} fill="none" stroke={ACC} strokeWidth={2} markerEnd={`url(#${id}-arrow-acc)`} />
      <Caption x={268} y={16} n={2} title="Flatten" lines={['one patch → 14 × 14 × 3', '= 588 numbers']} />
      {PATCH.map((v, i) => <rect key={i} x={204 + i * 11} y={MID - 7} width={9} height={14} rx={2} fill={INK} opacity={0.15 + v * 0.7} />)}
      <text x={268} y={MID + 28} textAnchor="middle" fontSize={11} style={{ fill: SOFT, ...MONO }}>[0.12, 0.23, 0.37, …]</text>

      <path d={`M340 ${MID} H372`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${id}-arrow)`} />
      <Caption x={434} y={16} n={3} title="Project" lines={['× learned matrix W_E', '+ position → D numbers']} />
      {TOKEN.map((v, i) => <rect key={i} x={382 + i * 18} y={MID - 8} width={15} height={16} rx={3} fill={ACC} opacity={0.2 + v * 0.8} />)}
      <text x={434} y={MID + 28} textAnchor="middle" fontSize={11} style={{ fill: SOFT, ...MONO }}>one token</text>

      <path d={`M496 ${MID} H528`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${id}-arrow)`} />
      <Caption x={648} y={16} n={4} title="One sequence" lines={['image tokens sit between', 'text tokens']} />
      <rect x={540} y={70} width={214} height={26} rx={6} fill={FILL} stroke={LINE} />
      <text x={647} y={87} textAnchor="middle" fontSize={11.5} style={{ fill: INK, ...MONO }}>text: “Why was I charged?”</text>
      {Array.from({ length: 7 }, (_, i) => (
        <rect key={i} x={540 + i * 22} y={MID - 12} width={18} height={18} rx={3} fill={ACC} opacity={i === 1 ? 1 : 0.45} />
      ))}
      <text x={700} y={MID + 1} textAnchor="middle" fontSize={11.5} style={{ fill: INK, ...MONO }}>… 256</text>
      <rect x={540} y={142} width={214} height={26} rx={6} fill={FILL} stroke={LINE} />
      <text x={647} y={159} textAnchor="middle" fontSize={11.5} style={{ fill: INK, ...MONO }}>text: the answer, token by token</text>
      <text x={647} y={196} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>the Transformer attends across all of them</text>
    </Figure>
  )
}
