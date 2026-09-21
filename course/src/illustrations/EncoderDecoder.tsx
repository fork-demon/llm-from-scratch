// The original Transformer (Vaswani et al., 2017) drawn with its data: a source sentence through
// the encoder (self-attention grid fully filled), the target so far through the decoder
// (lower-triangular grid), and cross-attention as a T_target x T_source rectangle whose queries
// come from the decoder and whose keys and values come from the encoder. The accent marks
// cross-attention, the one part the learner has not built. Weights are illustrative, not trained.
import { ACC, Caption, FILL, Figure, INK, LINE, MONO, ON_ACC, SOFT } from './diagramKit'

const Q = 'var(--q)'
const K = 'var(--k)'
const V = 'var(--v)'

const SRC = ['the', 'cat', 'is', 'small']
const TGT = ['<s>', 'die', 'Katze']
const ENC: number[][] = [[50, 30, 10, 10], [20, 40, 20, 20], [10, 30, 30, 30], [10, 40, 20, 30]]
const DEC: (number | null)[][] = [[100, null, null], [40, 60, null], [20, 30, 50]]
const CROSS: number[][] = [[70, 10, 10, 10], [15, 70, 10, 5], [5, 20, 65, 10]]

const CW = 32
const CH = 20

function Grid({ x, y, m }: { x: number; y: number; m: (number | null)[][] }) {
  return (
    <g>
      {m.map((row, i) => row.map((w, j) => (
        <g key={`${i}-${j}`}>
          {w === null
            ? <rect x={x + j * CW} y={y + i * CH} width={CW - 3} height={CH - 3} rx={3} fill="none" stroke={LINE} strokeDasharray="3 3" />
            : <rect x={x + j * CW} y={y + i * CH} width={CW - 3} height={CH - 3} rx={3} fill={INK} opacity={0.06 + (w / 100) * 0.3} />}
          {w !== null && <text x={x + j * CW + (CW - 3) / 2} y={y + i * CH + 12.5} textAnchor="middle" fontSize={10.5} style={{ fill: INK, ...MONO }}>{w}</text>}
        </g>
      )))}
    </g>
  )
}

function Chips({ x, y, words, w = 44 }: { x: number; y: number; words: string[]; w?: number }) {
  return (
    <g>
      {words.map((t, i) => (
        <g key={i}>
          <rect x={x + i * (w + 4)} y={y} width={w} height={22} rx={5} fill={FILL} stroke={LINE} />
          <text x={x + i * (w + 4) + w / 2} y={y + 15} textAnchor="middle" fontSize={11} style={{ fill: INK, ...MONO }}>{t}</text>
        </g>
      ))}
    </g>
  )
}

const Up = ({ x, y1, y2, colour = SOFT, marker = 'encdec-arrow' }: { x: number; y1: number; y2: number; colour?: string; marker?: string }) =>
  <path d={`M${x} ${y1} V${y2}`} fill="none" stroke={colour} strokeWidth={1.6} markerEnd={`url(#${marker})`} />

const SEQUENCE = [
  'Encoder self-attention: queries, keys and values all come from the source sentence. No mask: the 4 by 4 grid is completely filled, every source token sees the whole sentence.',
  'Decoder masked self-attention: queries, keys and values all come from the target written so far. The 3 by 3 grid is lower-triangular. This is the attention you built.',
  'Cross-attention: a 3 by 4 rectangle. One row per target position (queries from the decoder), one column per source token (keys and values from the encoder output). No mask.',
  'Feed-forward network and head, then the next target token: “ist”.',
  'Inset: a decoder-only model is the right half without cross-attention. Prompt and answer are one sequence.',
]

export function EncoderDecoder() {
  const gx = 196 // encoder grids
  const dx = 446 // decoder grids
  const cap = 590 // decoder captions, left-aligned in their own column
  return (
    <Figure
      id="encdec"
      height={584}
      minWidth={680}
      title="The encoder-decoder Transformer with its data: the source sentence passes through unmasked self-attention in the encoder; the target so far passes through masked self-attention in the decoder; cross-attention is a 3 by 4 grid with queries from the decoder and keys and values from the encoder; an inset shows that a decoder-only model is the right half without cross-attention"
      after={<ol className="sr-only">{SEQUENCE.map((s) => <li key={s}>{s}</li>)}</ol>}
    >
      <defs>
        {[['encdec-q', Q], ['encdec-k', K], ['encdec-v', V]].map(([id, c]) => (
          <marker key={id} id={id} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></marker>
        ))}
      </defs>

      <text x={14} y={16} fontSize={13} fontWeight={700} style={{ fill: INK }}>Encoder: reads the source, once</text>
      <text x={398} y={16} fontSize={13} fontWeight={700} style={{ fill: INK }}>Decoder: writes the target, token by token</text>
      <path d="M384 30 V84 M384 136 V446" stroke={LINE} strokeDasharray="2 5" />

      {/* ================= encoder, bottom to top ================= */}
      <Chips x={gx - 40} y={386} words={SRC} />
      <text x={gx + 54} y={426} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>source sentence</text>
      <Up x={gx + 62} y1={382} y2={372} />

      {SRC.map((t, j) => <text key={t} x={gx + j * CW + 14.5} y={282} textAnchor="middle" fontSize={10.5} style={{ fill: SOFT, ...MONO }}>{t}</text>)}
      {SRC.map((t, i) => <text key={t} x={gx - 6} y={288 + i * CH + 12.5} textAnchor="end" fontSize={10.5} style={{ fill: SOFT, ...MONO }}>{t}</text>)}
      <Grid x={gx} y={288} m={ENC} />
      <Caption x={14} y={300} n={1} title="Self-attention" anchor="start" lines={['Q, K, V: all from the', 'source. No mask: the', 'grid is full, every token', 'sees the whole sentence.']} />
      <Up x={gx + 62} y1={268} y2={234} />

      {/* encoder output: one vector per source token */}
      {SRC.map((t, i) => (
        <g key={t}>
          <text x={gx - 6} y={160 + i * 17 + 10} textAnchor="end" fontSize={10.5} style={{ fill: SOFT, ...MONO }}>{t}</text>
          {Array.from({ length: 12 }, (_, c) => <rect key={c} x={gx + c * 9} y={160 + i * 17} width={7.4} height={12} rx={2} fill={INK} opacity={0.14 + (((i + 2) * (c + 3) * 37) % 50) / 100} />)}
        </g>
      ))}
      <Caption x={14} y={170} title="Encoder output" anchor="start" lines={['one vector per source', 'token, computed once', 'and reused for every', 'target token']} />

      {/* K and V travel to the decoder and enter the cross-attention grid from above */}
      <path d={`M${gx + 112} 176 H362 V122 H${dx + 8} V130`} fill="none" stroke={K} strokeWidth={1.8} markerEnd="url(#encdec-k)" />
      <path d={`M${gx + 112} 200 H372 V112 H${dx + 24} V130`} fill="none" stroke={V} strokeWidth={1.8} markerEnd="url(#encdec-v)" />
      <text x={316} y={170} fontSize={12} fontWeight={700} style={{ fill: K }}>keys</text>
      <text x={316} y={216} fontSize={12} fontWeight={700} style={{ fill: V }}>values</text>

      {/* ================= decoder, bottom to top ================= */}
      <Chips x={dx - 16} y={386} words={TGT} />
      <text x={dx + 54} y={426} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>target so far</text>
      <Up x={dx + 40} y1={382} y2={364} />

      {TGT.map((t, j) => <text key={t} x={dx + j * CW + 14.5} y={294} textAnchor="middle" fontSize={10.5} style={{ fill: SOFT, ...MONO }}>{t}</text>)}
      {TGT.map((t, i) => <text key={t} x={dx - 6} y={300 + i * CH + 12.5} textAnchor="end" fontSize={10.5} style={{ fill: SOFT, ...MONO }}>{t}</text>)}
      <Grid x={dx} y={300} m={DEC} />
      <Caption x={cap} y={306} n={2} title="Masked self-attention" anchor="start" lines={['Q, K, V: all from the', 'target so far. Lower', 'triangle: what you built.']} />

      {/* Q goes up into cross-attention */}
      <Up x={dx + 112} y1={296} y2={226} colour={Q} marker="encdec-q" />
      <text x={dx + 120} y={262} fontSize={12} fontWeight={700} style={{ fill: Q }}>queries</text>

      {/* cross-attention: the point of the picture */}
      <rect x={dx - 50} y={132} width={50 + 4 * CW + 6} height={90} rx={8} fill="none" stroke={ACC} strokeWidth={1.8} />
      {SRC.map((t, j) => <text key={t} x={dx + j * CW + 14.5} y={150} textAnchor="middle" fontSize={10.5} style={{ fill: INK, ...MONO }}>{t}</text>)}
      {TGT.map((t, i) => <text key={t} x={dx - 6} y={156 + i * CH + 12.5} textAnchor="end" fontSize={10.5} style={{ fill: INK, ...MONO }}>{t}</text>)}
      <Grid x={dx} y={156} m={CROSS} />
      <Caption x={cap} y={150} n={3} title="Cross-attention" anchor="start" accent lines={['3 × 4: one row per target', 'token, one column per', 'source token. No mask.']} />

      {/* up to the next token */}
      <Up x={dx + 70} y1={130} y2={108} />
      <rect x={dx + 10} y={84} width={120} height={22} rx={11} fill="var(--paper)" stroke={LINE} />
      <text x={dx + 70} y={99} textAnchor="middle" fontSize={11} style={{ fill: INK }}>feed-forward, head</text>
      <Up x={dx + 70} y1={82} y2={72} />
      {[['ist', 71], ['sitzt', 9], ['war', 6]].map(([t, p], i) => (
        <g key={t as string}>
          <text x={dx + 28} y={30 + i * 13 + 9} textAnchor="end" fontSize={10.5} style={{ fill: INK, ...MONO }}>{t}</text>
          <rect x={dx + 34} y={30 + i * 13} width={(p as number) * 0.8} height={9} rx={3} fill={i === 0 ? ACC : INK} opacity={i === 0 ? 1 : 0.3} />
          <text x={dx + 39 + (p as number) * 0.8} y={30 + i * 13 + 9} fontSize={10.5} style={{ fill: SOFT, ...MONO }}>{p}%</text>
        </g>
      ))}
      <Caption x={cap} y={40} n={4} title="Next target token" anchor="start" lines={['it joins the target, and', 'the decoder runs again']} />

      <text x={14} y={446} fontSize={11} style={{ fill: SOFT }}>N = 6 layers: each repeats 1, then a feed-forward network</text>
      <text x={398} y={446} fontSize={11} style={{ fill: SOFT }}>N = 6 layers: each repeats 2, then 3, then a feed-forward network</text>

      {/* ================= inset ================= */}
      <rect x={14} y={460} width={732} height={114} rx={10} fill={FILL} stroke={LINE} />
      <text x={28} y={482} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>What you built: the right half, without cross-attention</text>
      <Chips x={28} y={498} words={['What', 'is', 'a', 'cat', '?']} w={38} />
      <text x={28} y={540} fontSize={11.5} style={{ fill: SOFT }}>one sequence: the prompt and the</text>
      <text x={28} y={555} fontSize={11.5} style={{ fill: SOFT }}>answer share the same masked grid</text>
      <path d="M246 509 H276" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#encdec-arrow)" />
      {Array.from({ length: 5 }, (_, i) => Array.from({ length: 5 }, (_, j) => (
        j <= i
          ? <rect key={`${i}-${j}`} x={288 + j * 15} y={490 + i * 15} width={12} height={12} rx={2.5} fill={INK} opacity={0.2 + ((i * 3 + j * 5) % 4) * 0.1} />
          : <rect key={`${i}-${j}`} x={288 + j * 15} y={490 + i * 15} width={12} height={12} rx={2.5} fill="none" stroke={LINE} strokeDasharray="2 2" />
      )))}
      <path d="M374 526 H404" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#encdec-arrow)" />
      <rect x={412} y={515} width={120} height={22} rx={11} fill="var(--paper)" stroke={LINE} />
      <text x={472} y={530} textAnchor="middle" fontSize={11} style={{ fill: INK }}>feed-forward, head</text>
      <path d="M538 526 H566" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#encdec-arrow)" />
      <rect x={574} y={513} width={40} height={26} rx={7} fill={ACC} />
      <text x={594} y={530.5} textAnchor="middle" fontSize={12.5} fontWeight={700} style={{ fill: ON_ACC, ...MONO }}>A</text>
      <text x={626} y={523} fontSize={11.5} style={{ fill: SOFT }}>no encoder, no</text>
      <text x={626} y={538} fontSize={11.5} style={{ fill: SOFT }}>cross-attention</text>
    </Figure>
  )
}
