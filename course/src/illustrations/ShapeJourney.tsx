// The whole GPT as one picture of SHAPES: the data is drawn as grids (one row per token),
// so you can see (T) become (T, D), stay (T, D) through every block, become (T, V), and then
// shrink to the one row that matters. The accent follows that last row to the next token.

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const FILL = 'var(--paper-2)'
const ACC = 'var(--accent)'
const MONO = 'var(--mono)'

const shade = (i: number, j: number, seed: number) => {
  const s = Math.sin((i + 1) * 12.9898 + (j + 1) * 78.233 + seed * 37.719) * 43758.5453
  return s - Math.floor(s)
}

const T = 6
const CHARS: [string, number][] = [['t', 58], ['h', 46], ['e', 43], ['␣', 1], ['c', 41], ['a', 39]]
const PROBS: [string, number][] = [['t', 58], ['n', 17], ['r', 12], ['l', 6], ['…', 7]]

/** A T-row grid; `cols` is drawn in proportion to the real width (D = 128 -> 12 columns, V = 65 -> 6). */
function Grid({ x, y, cols, seed, accentLast }: { x: number; y: number; cols: number; seed: number; accentLast?: boolean }) {
  return (
    <g>
      {Array.from({ length: T }, (_, r) => Array.from({ length: cols }, (_, c) => {
        const last = accentLast && r === T - 1
        return <rect key={`${r}-${c}`} x={x + c * 8} y={y + r * 7} width={6.6} height={5.6} rx={1.5} fill={last ? ACC : INK} opacity={last ? 0.45 + shade(r, c, seed) * 0.55 : (accentLast ? 0.1 : 0.16) + shade(r, c, seed) * (accentLast ? 0.25 : 0.55)} />
      }))}
    </g>
  )
}

function Caption({ x, y, n, title, shape, note }: { x: number; y: number; n?: number; title: string; shape?: string; note?: string }) {
  return (
    <g textAnchor="middle">
      <text x={x} y={y} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>
        {n !== undefined && <tspan style={{ fill: ACC }}>{n} </tspan>}{title}
      </text>
      {shape && <text x={x} y={y + 16} fontSize={11} style={{ fill: INK, fontFamily: MONO }}>{shape}</text>}
      {note && <text x={x} y={y + (shape ? 31 : 16)} fontSize={11.5} style={{ fill: SOFT }}>{note}</text>}
    </g>
  )
}

function Arrow({ d }: { d: string }) {
  return <path d={d} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#shape-arrow)" />
}

const STAGES = [
  'Input text: “the ca”',
  'Tokenizer: text becomes T = 6 token ids, shape (T)',
  'Embedding plus position: (T) becomes (T, D), one row of D = 128 numbers per token',
  'Transformer blocks × N: (T, D) in, (T, D) out, the shape does not change',
  'Final LayerNorm: still (T, D)',
  'Linear layer, the head: (T, D) becomes (T, V) logits, V = 65',
  'Softmax on the last row only: (V) probabilities',
  'Sampling: one token id',
  'Next token: append it to the text and go round again',
]

export function ShapeJourney() {
  const R1 = 84 // centre line of the top row
  const R2 = 234 // centre line of the bottom row
  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 342" style={{ width: '100%', minWidth: 680, display: 'block' }} role="img" aria-labelledby="shape-title">
          <title id="shape-title">The whole GPT drawn as data shapes: six characters become six ids, then a 6 by 128 grid that keeps its shape through the blocks and the final norm, then a 6 by 65 grid of logits whose last row becomes probabilities, one sampled id, and one appended character, with a loop back to the start</title>
          <defs>
            <marker id="shape-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={SOFT} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
            <marker id="shape-arrow-acc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={ACC} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></marker>
          </defs>

          {/* ---- 1 text ---- */}
          <Caption x={59} y={14} n={1} title="Input text" note="a string" />
          <rect x={14} y={R1 - 20} width={90} height={40} rx={8} fill={FILL} stroke={LINE} />
          <text x={24} y={R1 + 5} fontSize={13.5} style={{ fill: INK, fontFamily: MONO }}>the ca</text>
          <rect x={77} y={R1 - 11} width={19} height={22} rx={5} fill="none" stroke={ACC} strokeDasharray="4 3" />

          {/* ---- 2 tokenizer: characters, and the id under each ---- */}
          <Arrow d={`M108 ${R1} H128`} />
          <Caption x={193} y={14} n={2} title="Tokenizer" shape="text → (T), T = 6" />
          {CHARS.map(([ch, id], i) => (
            <g key={i}>
              <rect x={132 + i * 21} y={R1 - 24} width={18} height={22} rx={5} fill={FILL} stroke={LINE} />
              <text x={141 + i * 21} y={R1 - 8.5} textAnchor="middle" fontSize={11.5} style={{ fill: INK, fontFamily: MONO }}>{ch}</text>
              <path d={`M${141 + i * 21} ${R1 + 1} V${R1 + 7}`} stroke={LINE} strokeWidth={1.2} />
              <text x={141 + i * 21} y={R1 + 20} textAnchor="middle" fontSize={10.5} style={{ fill: INK, fontFamily: MONO }}>{id}</text>
            </g>
          ))}

          {/* ---- 3 embedding + position ---- */}
          <Arrow d={`M262 ${R1} H290`} />
          <Caption x={344} y={14} n={3} title="Embedding + position" shape="(T) → (T, D)" />
          <Grid x={296} y={R1 - 21} cols={12} seed={1} />
          <text x={344} y={R1 + 37} textAnchor="middle" fontSize={10.5} style={{ fill: SOFT, fontFamily: MONO }}>6 rows × 128</text>

          {/* ---- 4 blocks ---- */}
          <Arrow d={`M396 ${R1} H420`} />
          <Caption x={484} y={14} n={4} title="Blocks × N" shape="(T, D) → (T, D)" />
          <rect x={426} y={R1 - 34} width={116} height={68} rx={10} fill={FILL} stroke={INK} strokeWidth={1.5} />
          {[0, 1, 2, 3].map((i) => <rect key={i} x={438} y={R1 - 25 + i * 13.5} width={92} height={9} rx={4} fill={INK} opacity={0.2 + i * 0.1} />)}
          <Arrow d={`M546 ${R1} H574`} />
          <Caption x={628} y={14} title="Same shape out" shape="(T, D)" />
          <Grid x={580} y={R1 - 21} cols={12} seed={2} />
          <text x={628} y={R1 + 37} textAnchor="middle" fontSize={10.5} style={{ fill: SOFT, fontFamily: MONO }}>6 rows × 128</text>

          {/* ---- 5 final norm ---- */}
          <Arrow d={`M680 ${R1} H728 V${R2} H686`} />
          <rect x={584} y={R2 - 12} width={96} height={24} rx={12} fill="var(--paper)" stroke={LINE} strokeWidth={1.2} />
          <text x={632} y={R2 + 4} textAnchor="middle" fontSize={11} style={{ fill: INK }}>LayerNorm</text>
          <Caption x={632} y={288} n={5} title="Final LayerNorm" shape="(T, D)" />

          {/* ---- 6 head: logits, last row in accent ---- */}
          <Arrow d={`M578 ${R2} H540`} />
          <Grid x={486} y={R2 - 38} cols={6} seed={3} accentLast />
          <text x={510} y={R2 - 45} textAnchor="middle" fontSize={10.5} style={{ fill: SOFT, fontFamily: MONO }}>6 rows × 65</text>
          <Caption x={510} y={288} n={6} title="The head" shape="(T, D) → (T, V)" note="logits" />

          {/* ---- 7 softmax on the last row ---- */}
          <path d={`M480 ${R2} H452`} fill="none" stroke={ACC} strokeWidth={1.6} markerEnd="url(#shape-arrow-acc)" />
          {PROBS.map(([t, p], i) => (
            <g key={t}>
              <text x={338} y={R2 - 37 + i * 15 + 9} textAnchor="end" fontSize={11} style={{ fill: INK, fontFamily: MONO }}>{t}</text>
              <rect x={344} y={R2 - 37 + i * 15} width={p * 1.25} height={10} rx={3} fill={i === 0 ? ACC : INK} opacity={i === 0 ? 1 : 0.3} />
              <text x={349 + p * 1.25} y={R2 - 37 + i * 15 + 9} fontSize={10.5} style={{ fill: SOFT, fontFamily: MONO }}>{p}%</text>
            </g>
          ))}
          <Caption x={386} y={288} n={7} title="Softmax, last row" shape="(V)" note="probabilities" />

          {/* ---- 8 sampling ---- */}
          <Arrow d={`M322 ${R2} H292`} />
          <g transform={`translate(266 ${R2}) rotate(-10)`}>
            <rect x={-17} y={-17} width={34} height={34} rx={8} fill={FILL} stroke={INK} strokeWidth={1.5} />
            {[[-7, -7], [0, 0], [7, 7], [-7, 7], [7, -7]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r={2.5} fill={INK} />)}
          </g>
          <Arrow d={`M242 ${R2} H212`} />
          <rect x={170} y={R2 - 13} width={38} height={26} rx={6} fill="none" stroke={ACC} strokeWidth={1.6} />
          <text x={189} y={R2 + 4.5} textAnchor="middle" fontSize={12} fontWeight={700} style={{ fill: ACC, fontFamily: MONO }}>58</text>
          <Caption x={228} y={288} n={8} title="Sampling" note="one token id" />

          {/* ---- 9 next token ---- */}
          <Arrow d={`M164 ${R2} H110`} />
          <rect x={30} y={R2 - 16} width={58} height={32} rx={8} fill={ACC} />
          <text x={59} y={R2 + 5} textAnchor="middle" fontSize={14} fontWeight={700} style={{ fill: 'var(--on-accent)', fontFamily: MONO }}>t</text>
          <Caption x={59} y={288} n={9} title="Next token" note="id 58 is “t”" />

          {/* ---- the loop ---- */}
          <path d={`M59 ${R2 - 22} V${R1 + 28}`} fill="none" stroke={ACC} strokeWidth={2.2} strokeDasharray="6 5" markerEnd="url(#shape-arrow-acc)" />
          <text x={70} y={158} fontSize={12} fontWeight={700} style={{ fill: ACC }}>append it: T becomes 7,</text>
          <text x={70} y={173} fontSize={12} fontWeight={700} style={{ fill: ACC }}>then go round again</text>

          <text x={300} y={146} fontSize={11.5} style={{ fill: SOFT }}>Every grid has one row per token. Widths are in proportion</text>
          <text x={300} y={161} fontSize={11.5} style={{ fill: SOFT }}>(D = 128, V = 65), one drawn column for about ten real ones.</text>
        </svg>
      </div>
      <ol className="sr-only">{STAGES.map((s) => <li key={s}>{s}</li>)}</ol>
    </figure>
  )
}
