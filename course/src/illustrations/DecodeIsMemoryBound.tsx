// Why batching is nearly free: one decode step drawn twice. The same 14 GB of weights crosses the
// same memory bus once per step, whether the arithmetic units then serve 1 sequence or 32.
// The accent marks the arithmetic that is actually busy, and the tokens that come out.
// Numbers are the cost model of phase6-engineering/batching_sim.py (300 cached tokens per sequence).

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const FILL = 'var(--paper-2)'
const ACC = 'var(--accent)'
const MONO = 'var(--mono)'

const shade = (i: number, j: number) => {
  const s = Math.sin((i + 1) * 12.9898 + (j + 1) * 78.233) * 43758.5453
  return s - Math.floor(s)
}

function Panel({ x0, batch, bytesMs, mathMs, stepMs, tokS }: { x0: number; batch: number; bytesMs: number; mathMs: number; stepMs: number; tokS: string }) {
  const SCALE = 17 // px per millisecond in the two bars
  return (
    <g transform={`translate(${x0} 0)`}>
      <text x={185} y={18} textAnchor="middle" fontSize={13} fontWeight={700} style={{ fill: INK }}>{batch === 1 ? '1 sequence in the batch' : `${batch} sequences in the batch`}</text>

      {/* GPU memory: the weights */}
      <text x={62} y={44} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>GPU memory</text>
      <rect x={10} y={52} width={104} height={104} rx={8} fill={FILL} stroke={LINE} />
      {Array.from({ length: 7 }, (_, i) => Array.from({ length: 7 }, (_, j) => (
        <rect key={`${i}-${j}`} x={18 + j * 13} y={60 + i * 13} width={10.5} height={10.5} rx={2} fill={INK} opacity={0.18 + shade(i, j) * 0.5} />
      )))}
      <text x={62} y={172} textAnchor="middle" fontSize={11} style={{ fill: INK, fontFamily: MONO }}>14 GB of weights</text>

      {/* the bus: the same read in both panels */}
      <path d="M120 104 H176" fill="none" stroke={INK} strokeWidth={5} markerEnd="url(#dimb-arrow)" />
      <text x={148} y={90} textAnchor="middle" fontSize={11} style={{ fill: INK }}>all of it,</text>
      <text x={148} y={126} textAnchor="middle" fontSize={11} style={{ fill: INK }}>once</text>
      <text x={148} y={141} textAnchor="middle" fontSize={11} style={{ fill: SOFT, fontFamily: MONO }}>7 ms</text>

      {/* arithmetic units */}
      <text x={232} y={44} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>arithmetic units</text>
      <rect x={186} y={52} width={92} height={104} rx={8} fill={FILL} stroke={LINE} />
      {Array.from({ length: 8 }, (_, i) => Array.from({ length: 4 }, (_, j) => {
        const busy = i * 4 + j < batch
        return <rect key={`${i}-${j}`} x={194 + j * 20} y={59 + i * 11.6} width={16} height={8.6} rx={2} fill={busy ? ACC : INK} opacity={busy ? 1 : 0.13} />
      }))}
      <text x={232} y={172} textAnchor="middle" fontSize={11} style={{ fill: batch === 1 ? SOFT : ACC }}>{batch === 1 ? 'almost all idle' : 'busy'}</text>

      {/* tokens out */}
      <path d="M284 104 H306" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#dimb-arrow-soft)" />
      {Array.from({ length: batch }, (_, k) => {
        const cols = batch === 1 ? 1 : 4
        const rows = Math.ceil(batch / cols)
        const i = Math.floor(k / cols), j = k % cols
        return <rect key={k} x={314 + j * 12} y={104 - (rows * 11.6) / 2 + i * 11.6} width={9.5} height={8.6} rx={2} fill={ACC} />
      })}
      <text x={batch === 1 ? 319 : 337} y={172} textAnchor="middle" fontSize={11} fontWeight={700} style={{ fill: ACC }}>{batch === 1 ? '1 token' : '32 tokens'}</text>

      {/* the step: as long as the slower of the two */}
      <text x={10} y={204} fontSize={11} style={{ fill: SOFT }}>moving bytes</text>
      <rect x={96} y={195} width={bytesMs * SCALE} height={11} rx={3} fill={INK} />
      <text x={102 + bytesMs * SCALE} y={204.5} fontSize={11} style={{ fill: INK, fontFamily: MONO }}>{bytesMs.toFixed(1)} ms</text>
      <text x={10} y={224} fontSize={11} style={{ fill: SOFT }}>arithmetic</text>
      <rect x={96} y={215} width={Math.max(2, mathMs * SCALE)} height={11} rx={3} fill={ACC} />
      <text x={102 + Math.max(2, mathMs * SCALE)} y={224.5} fontSize={11} style={{ fill: INK, fontFamily: MONO }}>{mathMs.toFixed(2)} ms</text>
      <text x={10} y={252} fontSize={12} style={{ fill: INK }}>
        step = slower of the two + 3 ms overhead = <tspan fontWeight={700} style={{ fontFamily: MONO }}>{stepMs.toFixed(1)} ms</tspan>
      </text>
      <text x={10} y={270} fontSize={12} style={{ fill: INK }}>
        throughput: <tspan fontWeight={700} style={{ fill: ACC, fontFamily: MONO }}>{tokS} tokens/s</tspan>
      </text>
    </g>
  )
}

export function DecodeIsMemoryBound() {
  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 282" style={{ width: '100%', minWidth: 700, display: 'block' }} role="img" aria-labelledby="dimb-title">
          <title id="dimb-title">One decode step for a batch of 1 and a batch of 32. In both, all 14 gigabytes of weights move from GPU memory to the arithmetic units once, taking 7 milliseconds. With 1 sequence almost every arithmetic unit is idle and the step yields 1 token in 10.1 milliseconds. With 32 sequences the same read keeps the units busy and the step yields 32 tokens in 12.5 milliseconds.</title>
          <defs>
            <marker id="dimb-arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill={INK} /></marker>
            <marker id="dimb-arrow-soft" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1 L9 5 L1 9" fill="none" stroke={SOFT} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
          </defs>
          <Panel x0={0} batch={1} bytesMs={7.08} mathMs={0.09} stepMs={10.08} tokS="99" />
          <line x1={380} x2={380} y1={10} y2={272} stroke={LINE} strokeDasharray="3 5" />
          <Panel x0={392} batch={32} bytesMs={9.5} mathMs={2.88} stepMs={12.5} tokS="2,561" />
        </svg>
      </div>
      <figcaption className="muted" style={{ fontSize: 14, marginTop: 6 }}>
        The bytes bar grows a little on the right because each sequence also reads its own KV cache (here 300 tokens each). Illustrative numbers from the lesson’s cost model: a 7B model in 16-bit, 2 TB/s of memory bandwidth.
      </figcaption>
    </figure>
  )
}
