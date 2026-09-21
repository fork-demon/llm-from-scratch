// One decode step with a KV cache. The two caches are drawn as tables with one row per token:
// the earlier rows are faint (already computed, only read), and exactly ONE new row is
// appended in full colour. The single q is compared with every K row to give one weight per
// row, and the weights blend the V rows. q/k/v colours are used only for q, k and v.

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const FILL = 'var(--paper-2)'
const ACC = 'var(--accent)'
const MONO = 'var(--mono)'
const Q = 'var(--q)'
const K = 'var(--k)'
const V = 'var(--v)'

const shade = (i: number, j: number, seed: number) => {
  const s = Math.sin((i + 1) * 12.9898 + (j + 1) * 78.233 + seed * 37.719) * 43758.5453
  return s - Math.floor(s)
}

const OLD = ['The', 'cat', 'sat', 'on', 'the'] // already in the cache
const NEW = 'mat' // the one token that goes in this step
const WEIGHTS = [0.06, 0.27, 0.22, 0.12, 0.05, 0.28]
const ROWS = OLD.length + 1
const RH = 18 // row pitch
const TOP = 112 // y of the first cache row
const KX = 300
const VX = 520
const rowY = (r: number) => TOP + r * RH
const NEWY = rowY(ROWS - 1) + 7.5 // centre line of the new row

/** One vector of 4 numbers, drawn as 4 cells. */
function Strip({ x, y, color, seed, strong = true }: { x: number; y: number; color: string; seed: number; strong?: boolean }) {
  return (
    <g>
      {[0, 1, 2, 3].map((c) => (
        <rect key={c} x={x + c * 18} y={y} width={16} height={15} rx={3} fill={color} opacity={strong ? 0.45 + shade(seed, c, 1) * 0.55 : 0.2 + shade(seed, c, 1) * 0.25} />
      ))}
    </g>
  )
}

function Cache({ x, color, seed, name }: { x: number; color: string; seed: number; name: string }) {
  return (
    <g>
      <text x={x + 35} y={TOP - 10} textAnchor="middle" fontSize={12.5} fontWeight={700} style={{ fill: color }}>{name}</text>
      {Array.from({ length: ROWS }, (_, r) => <Strip key={r} x={x} y={rowY(r)} color={color} seed={seed + r} strong={r === ROWS - 1} />)}
      <rect x={x - 4} y={rowY(ROWS - 1) - 3.5} width={78} height={22} rx={6} fill="none" stroke={ACC} strokeWidth={1.6} />
    </g>
  )
}

export function CachedStep() {
  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 318" style={{ width: '100%', minWidth: 680, display: 'block' }} role="img" aria-labelledby="cstep-title">
          <title id="cstep-title">One cached decode step: the single new token makes one query, one key and one value; the key and value are appended as one new row to the K and V caches, whose five earlier rows are reused; the query is compared with every K row to get weights, which blend the V rows into the output</title>
          <defs>
            <marker id="cstep-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={SOFT} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
            <marker id="cstep-arrow-q" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={Q} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
            <marker id="cstep-arrow-k" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={K} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
            <marker id="cstep-arrow-v" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={V} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
          </defs>

          {/* ---- the one new token and its vector ---- */}
          <text x={14} y={NEWY - 34} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>one token in</text>
          <text x={14} y={NEWY - 19} fontSize={11.5} style={{ fill: SOFT }}>only the newest</text>
          <rect x={14} y={NEWY - 12} width={42} height={24} rx={6} fill={FILL} stroke={INK} strokeWidth={1.4} />
          <text x={35} y={NEWY + 4.5} textAnchor="middle" fontSize={12} style={{ fill: INK, fontFamily: MONO }}>{NEW}</text>
          <path d={`M60 ${NEWY} H82`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#cstep-arrow)" />
          <Strip x={88} y={NEWY - 7.5} color={INK} seed={40} />

          {/* ---- it makes one q, one k, one v ---- */}
          <path d={`M123 ${NEWY - 12} V55 H372`} fill="none" stroke={Q} strokeWidth={1.5} markerEnd="url(#cstep-arrow-q)" />
          <path d={`M164 ${NEWY} H${KX - 10}`} fill="none" stroke={K} strokeWidth={1.5} markerEnd="url(#cstep-arrow-k)" />
          <path d={`M123 ${NEWY + 12} V262 H${VX + 35} V${NEWY + 17}`} fill="none" stroke={V} strokeWidth={1.5} markerEnd="url(#cstep-arrow-v)" />
          <text x={222} y={NEWY - 7} textAnchor="middle" fontSize={11.5} fontWeight={700} style={{ fill: K }}>k: append</text>
          <text x={300} y={279} textAnchor="middle" fontSize={11.5} fontWeight={700} style={{ fill: V }}>v: append</text>

          {/* legend, in the quiet corner */}
          <Strip x={150} y={76} color={INK} seed={50} strong={false} />
          <text x={228} y={88} fontSize={11.5} style={{ fill: SOFT }}>already computed,</text>
          <text x={228} y={102} fontSize={11.5} style={{ fill: SOFT }}>reused</text>

          {/* ---- q: used once, never stored ---- */}
          <Strip x={378} y={47.5} color={Q} seed={41} />
          <text x={456} y={52} fontSize={12.5} fontWeight={700} style={{ fill: Q }}>q</text>
          <text x={470} y={52} fontSize={11.5} style={{ fill: SOFT }}>one query, used once, not cached</text>
          <text x={456} y={67} fontSize={11.5} style={{ fill: SOFT }}>compared with every K row</text>

          {/* ---- K cache ---- */}
          {OLD.map((t, r) => <text key={r} x={KX - 10} y={rowY(r) + 12} textAnchor="end" fontSize={11} style={{ fill: SOFT, fontFamily: MONO }}>{t}</text>)}
          <Cache x={KX} color={K} seed={10} name="K cache" />

          {/* q sweeps down past every K row: one dot product per row gives one weight per row */}
          <path d={`M413 66 V${NEWY}`} fill="none" stroke={Q} strokeWidth={1.3} />
          {WEIGHTS.map((w, r) => (
            <g key={r}>
              <path d={`M${KX + 74} ${rowY(r) + 7.5} H436`} fill="none" stroke={SOFT} strokeWidth={1} />
              <circle cx={413} cy={rowY(r) + 7.5} r={2.6} fill={Q} />
              <path d={`M488 ${rowY(r) + 7.5} H${VX - 6}`} fill="none" stroke={SOFT} strokeWidth={1} />
              <rect x={440} y={rowY(r) + 2} width={Math.max(3, w * 150)} height={11} rx={3} fill={INK} opacity={0.75} />
            </g>
          ))}
          <text x={462} y={TOP - 10} textAnchor="middle" fontSize={12.5} fontWeight={700} style={{ fill: INK }}>weights</text>
          <text x={462} y={rowY(ROWS) + 12} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>softmax(q · k / √d)</text>

          {/* ---- V cache, blended by the weights ---- */}
          <Cache x={VX} color={V} seed={20} name="V cache" />
          {WEIGHTS.map((_, r) => <path key={r} d={`M${VX + 76} ${rowY(r) + 7.5} L652 ${rowY(2) + 16.5}`} fill="none" stroke={SOFT} strokeWidth={1} />)}
          <Strip x={656} y={rowY(2) + 9} color={INK} seed={42} />
          <text x={691} y={rowY(2) - 2} textAnchor="middle" fontSize={12.5} fontWeight={700} style={{ fill: INK }}>output</text>
          <text x={691} y={rowY(2) + 42} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>weighted blend</text>
          <text x={691} y={rowY(2) + 56} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>of the V rows</text>
          <text x={691} y={rowY(2) + 76} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>→ rest of this layer</text>

          {/* ---- the saving ---- */}
          <text x={380} y={306} textAnchor="middle" fontSize={13} style={{ fill: INK }}>
            <tspan fontWeight={700} style={{ fill: ACC }}>Computed this step: 1 row.</tspan>
            <tspan> Reused: T rows</tspan>
            <tspan style={{ fill: SOFT }}> (here T = 5), in every layer.</tspan>
          </text>
        </svg>
      </div>
      <ol className="sr-only">
        <li>One token in: only the newest.</li>
        <li>Make one query, one key and one value for that one token.</li>
        <li>Append the key and the value as one new row of this layer’s cache. The earlier rows are reused, not recomputed.</li>
        <li>Attend: compare the query with all cached keys, and blend the cached values with the resulting weights.</li>
        <li>The output goes on through the rest of this layer and the layers above. After the last layer come the logits, and a token is sampled.</li>
      </ol>
    </figure>
  )
}
