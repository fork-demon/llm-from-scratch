// Three silhouettes of the SAME architecture (embedding strip, a stack of blocks, output strip)
// at three sizes. Each silhouette draws its real number of blocks; the widths are only
// suggestive. The accent is on the numbers, because the numbers are the only thing that changes.

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const ACC = 'var(--accent)'

interface Model { cx: number; w: number; blocks: number; pitch: number; name: string; when: string; nBlocks: string; width: string; params: string }

const MODELS: Model[] = [
  { cx: 96, w: 46, blocks: 4, pitch: 9, name: 'This course', when: 'tiny_gpt.py', nBlocks: '4', width: '128', params: '0.8 million' },
  { cx: 310, w: 112, blocks: 48, pitch: 3, name: 'GPT-2 XL', when: '2019', nBlocks: '48', width: '1,600', params: '1.5 billion' },
  { cx: 560, w: 200, blocks: 80, pitch: 2.6, name: 'A modern 70B-class model', when: 'Llama 3 70B, 2024', nBlocks: '80', width: '8,192', params: '70 billion' },
]

const BASE = 250 // y of the bottom edge of every silhouette
const STRIP = 9 // height of the embedding and output strips

function Silhouette({ m }: { m: Model }) {
  const x = m.cx - m.w / 2
  const stackH = m.blocks * m.pitch
  const stackBottom = BASE - STRIP - 5
  const top = stackBottom - stackH - 5 - STRIP
  return (
    <g>
      {/* output strip */}
      <rect x={x} y={top} width={m.w} height={STRIP} rx={3} fill={INK} opacity={0.85} />
      {/* blocks: one bar each */}
      {Array.from({ length: m.blocks }, (_, i) => (
        <rect key={i} x={x} y={stackBottom - (i + 1) * m.pitch + m.pitch * 0.2} width={m.w} height={m.pitch * 0.62} rx={Math.min(3, m.pitch * 0.3)} fill={INK} opacity={0.42} />
      ))}
      {/* embedding strip */}
      <rect x={x} y={BASE - STRIP} width={m.w} height={STRIP} rx={3} fill={INK} opacity={0.85} />

      <g textAnchor="middle">
        <text x={m.cx} y={BASE + 24} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>{m.name}</text>
        <text x={m.cx} y={BASE + 39} fontSize={11.5} style={{ fill: SOFT }}>{m.when}</text>
        <text x={m.cx} y={BASE + 58} fontSize={11.5} style={{ fill: INK }}>
          <tspan fontWeight={700} style={{ fill: ACC }}>{m.nBlocks}</tspan> blocks, <tspan fontWeight={700} style={{ fill: ACC }}>{m.width}</tspan> wide
        </text>
        <text x={m.cx} y={BASE + 74} fontSize={11.5} style={{ fill: INK }}>
          about <tspan fontWeight={700} style={{ fill: ACC }}>{m.params}</tspan> parameters
        </text>
      </g>
    </g>
  )
}

export function SameShapeBiggerNumbers() {
  const big = MODELS[2]
  const bx = big.cx + big.w / 2
  const stackBottom = BASE - STRIP - 5
  const stackTop = stackBottom - big.blocks * big.pitch
  const top = stackTop - 5 - STRIP
  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 334" style={{ width: '100%', minWidth: 640, display: 'block' }} role="img" aria-labelledby="same-title">
          <title id="same-title">The same architecture at three sizes: this course’s tiny GPT with 4 blocks, 128 wide, about 0.8 million parameters; GPT-2 XL with 48 blocks, 1,600 wide, about 1.5 billion; and a modern 70B-class model with 80 blocks, 8,192 wide, about 70 billion. The drawing is the same; only the numbers change.</title>

          <line x1={20} y1={BASE + 0.5} x2={740} y2={BASE + 0.5} stroke={LINE} strokeWidth={1} />
          {MODELS.map((m) => <Silhouette key={m.name} m={m} />)}

          {/* the parts, named once on the largest silhouette */}
          <g fontSize={11.5} style={{ fill: SOFT }}>
            <path d={`M${bx + 4} ${top + 4.5} H${bx + 14}`} stroke={SOFT} strokeWidth={1.2} />
            <text x={bx + 18} y={top + 8.5}>output layer</text>
            <path d={`M${bx + 6} ${stackTop} H${bx + 10} V${stackBottom} H${bx + 6}`} fill="none" stroke={SOFT} strokeWidth={1.2} />
            <text x={bx + 18} y={(stackTop + stackBottom) / 2 - 3}>blocks,</text>
            <text x={bx + 18} y={(stackTop + stackBottom) / 2 + 11}>one bar each</text>
            <path d={`M${bx + 4} ${BASE - 4.5} H${bx + 14}`} stroke={SOFT} strokeWidth={1.2} />
            <text x={bx + 18} y={BASE - 0.5}>embeddings</text>
          </g>

          <text x={20} y={20} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>Same drawing. Only the numbers change.</text>
          <text x={20} y={36} fontSize={11.5} style={{ fill: SOFT }}>Block counts are drawn exactly; widths are not to scale.</text>
        </svg>
      </div>
      <ul className="sr-only">
        {MODELS.map((m) => <li key={m.name}>{m.name} ({m.when}): {m.nBlocks} blocks, {m.width} wide, about {m.params} parameters.</li>)}
      </ul>
    </figure>
  )
}
