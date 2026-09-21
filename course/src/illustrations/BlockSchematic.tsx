// The Transformer block as a schematic, read bottom to top. The residual stream (the T x D
// token vectors) is the one accented thing: both sub-layers leave it, read a normalised copy,
// and ADD their result back at a plus node. Nothing ever replaces the stream.

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const FILL = 'var(--paper-2)'
const ACC = 'var(--accent)'

const SX = 300 // x of the residual stream
const BX = 470 // x of the side branches

// deterministic "random" cell shade, so the grids look like numbers without being noise on every render
const shade = (i: number, j: number, seed: number) => {
  const s = Math.sin((i + 1) * 12.9898 + (j + 1) * 78.233 + seed * 37.719) * 43758.5453
  return s - Math.floor(s)
}

/** A T x D grid: 3 tokens (rows) of 4 numbers, as in the lesson's explorer. */
function Grid({ x, y, seed, color }: { x: number; y: number; seed: number; color: string }) {
  return (
    <g>
      {[0, 1, 2].map((r) => [0, 1, 2, 3].map((c) => (
        <rect key={`${r}-${c}`} x={x + c * 10} y={y + r * 10} width={8.5} height={8.5} rx={2} fill={color} opacity={0.22 + shade(r, c, seed) * 0.7} />
      )))}
    </g>
  )
}

function Plus({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <circle cx={x} cy={y} r={10} fill="var(--paper)" stroke={ACC} strokeWidth={2} />
      <path d={`M${x - 5} ${y} H${x + 5} M${x} ${y - 5} V${y + 5}`} stroke={ACC} strokeWidth={2} strokeLinecap="round" />
    </g>
  )
}

function Norm({ y }: { y: number }) {
  return (
    <g>
      <rect x={BX - 42} y={y} width={84} height={22} rx={11} fill="var(--paper)" stroke={LINE} strokeWidth={1.2} />
      <text x={BX} y={y + 15} textAnchor="middle" fontSize={11} style={{ fill: INK }}>LayerNorm</text>
    </g>
  )
}

/** One side branch: tap the stream at yTap, go up through LayerNorm and the sub-layer, re-join at yJoin. */
function BranchWires({ yTap, yJoin }: { yTap: number; yJoin: number }) {
  return (
    <g fill="none" stroke={SOFT} strokeWidth={1.5}>
      <path d={`M${SX} ${yTap} H${SX + 90}`} markerEnd="url(#blk-arrow)" />
      <path d={`M${SX + 88} ${yTap} H${BX} V${yJoin} H${SX + 13}`} markerEnd="url(#blk-arrow)" strokeLinejoin="round" />
      <circle cx={SX} cy={yTap} r={3.2} fill={ACC} stroke="none" />
    </g>
  )
}

export function BlockSchematic() {
  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 392" style={{ width: '100%', minWidth: 640, display: 'block' }} role="img" aria-labelledby="blk-title">
          <title id="blk-title">A Transformer block, bottom to top: token embeddings plus position vectors form the residual stream; a branch through LayerNorm and masked self-attention is added back to the stream; a second branch through LayerNorm and the MLP is added back; the block repeats N times</title>
          <defs>
            <marker id="blk-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={SOFT} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
            <marker id="blk-arrow-acc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={ACC} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></marker>
          </defs>

          {/* ---- the residual stream ---- */}
          <path d={`M${SX} 348 V12`} fill="none" stroke={ACC} strokeWidth={2.4} markerEnd="url(#blk-arrow-acc)" />

          {/* ---- bottom: token embeddings + position vectors ---- */}
          <Grid x={176} y={344} seed={1} color={INK} />
          <text x={166} y={356} textAnchor="end" fontSize={12.5} fontWeight={700} style={{ fill: INK }}>token embeddings</text>
          <text x={166} y={372} textAnchor="end" fontSize={11.5} style={{ fill: SOFT }}>one row per token</text>
          <path d={`M222 358 H${SX - 14}`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#blk-arrow)" />
          <Grid x={386} y={344} seed={2} color={INK} />
          <text x={436} y={356} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>position vectors</text>
          <text x={436} y={372} fontSize={11.5} style={{ fill: SOFT }}>slots 0, 1, 2 · added once, before the first block</text>
          <path d={`M380 358 H${SX + 14}`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#blk-arrow)" />
          <Plus x={SX} y={358} />

          {/* the stream's data, drawn once it exists */}
          <rect x={SX - 25} y={305} width={50} height={36} fill="var(--paper)" />
          <Grid x={SX - 19.5} y={308.5} seed={3} color={ACC} />

          {/* ---- the block bracket ---- */}
          <path d="M252 296 H240 V58 H252" fill="none" stroke={SOFT} strokeWidth={1.5} />
          <text x={228} y={172} textAnchor="end" fontSize={17} fontWeight={700} style={{ fill: INK }}>× N</text>
          <text x={228} y={190} textAnchor="end" fontSize={11.5} style={{ fill: SOFT }}>N blocks, each with</text>
          <text x={228} y={204} textAnchor="end" fontSize={11.5} style={{ fill: SOFT }}>its own weights</text>

          {/* ---- branch 1: attention ---- */}
          <BranchWires yTap={288} yJoin={184} />
          <Norm y={254} />
          <rect x={BX - 42} y={198} width={84} height={46} rx={8} fill={FILL} stroke={INK} strokeWidth={1.4} />
          {[0, 1, 2].map((r) => [0, 1, 2].map((c) => (
            c <= r
              ? <rect key={`${r}-${c}`} x={BX - 17 + c * 12} y={204 + r * 12} width={10} height={10} rx={2} fill={INK} opacity={0.3 + shade(r, c, 5) * 0.55} />
              : <rect key={`${r}-${c}`} x={BX - 16.5 + c * 12} y={204.5 + r * 12} width={9} height={9} rx={2} fill="none" stroke={LINE} />
          )))}
          <text x={530} y={214} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>masked self-attention</text>
          <text x={530} y={230} fontSize={11.5} style={{ fill: SOFT }}>communicate: tokens read earlier tokens</text>
          <text x={530} y={244} fontSize={11.5} style={{ fill: SOFT }}>(the T × T grid, upper half masked out)</text>
          <Plus x={SX} y={184} />

          {/* ---- branch 2: MLP ---- */}
          <BranchWires yTap={164} yJoin={70} />
          <Norm y={134} />
          <rect x={BX - 42} y={80} width={84} height={46} rx={8} fill={FILL} stroke={INK} strokeWidth={1.4} />
          <rect x={BX - 14} y={112} width={28} height={8} rx={3} fill={INK} opacity={0.45} />
          <rect x={BX - 34} y={99} width={68} height={8} rx={3} fill={INK} opacity={0.7} />
          <rect x={BX - 14} y={86} width={28} height={8} rx={3} fill={INK} opacity={0.45} />
          <text x={530} y={96} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>feed-forward MLP</text>
          <text x={530} y={112} fontSize={11.5} style={{ fill: SOFT }}>compute: each token alone,</text>
          <text x={530} y={126} fontSize={11.5} style={{ fill: SOFT }}>D numbers → 4D → back to D</text>
          <Plus x={SX} y={70} />

          {/* ---- labels on the stream ---- */}
          <text x={SX - 18} y={30} textAnchor="end" fontSize={12.5} fontWeight={700} style={{ fill: ACC }}>the residual stream x</text>
          <text x={SX - 18} y={46} textAnchor="end" fontSize={11.5} style={{ fill: SOFT }}>shape (T, D) all the way up</text>
          <text x={SX + 18} y={30} fontSize={11.5} style={{ fill: SOFT }}>to the next block, or to the output layer</text>
          <text x={SX - 16} y={74} textAnchor="end" fontSize={11.5} style={{ fill: SOFT }}>add</text>
          <text x={SX - 16} y={188} textAnchor="end" fontSize={11.5} style={{ fill: SOFT }}>add</text>
        </svg>
      </div>
      <ol className="sr-only">
        <li>Token embeddings plus position vectors, added once before the first block, start the residual stream x of shape (T, D).</li>
        <li>Communicate: a copy of x goes through LayerNorm and masked self-attention, and the result is added to x.</li>
        <li>Compute: a copy of x goes through LayerNorm and the feed-forward MLP, and the result is added to x.</li>
        <li>The last two steps repeat N times: N blocks, each with its own weights.</li>
      </ol>
    </figure>
  )
}
