// Fine-tuning in one picture: the same grid of weights before and after. The example pairs feed
// the ordinary training loop; the grid keeps its shape and most of its values, and the accent
// marks the weights that moved most. Under each grid, what the model does with the same short
// prompt: the chatty paragraph the lesson complains about, then the fixed JSON, two sentences.
// (The LoRA version of this picture, W frozen plus A·B, is drawn by LoraLab further down.)

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const FILL = 'var(--paper-2)'
const ACC = 'var(--accent)'
const MONO = 'var(--mono)'

const COLS = 12
const ROWS_N = 8
// a fixed pseudo-random pattern, so both grids are the same picture
const weight = (r: number, c: number) => ((r * 37 + c * 61 + r * c * 13) % 17) / 16
const MOVED = new Set(['0-3', '1-9', '2-1', '2-6', '3-10', '4-4', '5-0', '5-7', '6-11', '6-2', '7-5', '3-3'])

function Grid({ x, y, after }: { x: number; y: number; after?: boolean }) {
  const cells = []
  for (let r = 0; r < ROWS_N; r++) for (let c = 0; c < COLS; c++) {
    const moved = after && MOVED.has(`${r}-${c}`)
    cells.push(<rect key={`${r}-${c}`} x={x + c * 14} y={y + r * 14} width={12} height={12} rx={2.5} fill={moved ? ACC : INK} opacity={moved ? 1 : 0.1 + weight(r, c) * 0.6} />)
  }
  return <>{cells}</>
}

export function WeightsShift() {
  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 318" style={{ width: '100%', minWidth: 660, display: 'block' }} role="img" aria-labelledby="weights-shift-title">
          <title id="weights-shift-title">The same grid of weights before and after fine-tuning on example pairs: a scattering of weights has changed, and the same prompt now gets a short JSON reply instead of a chatty paragraph</title>
          <defs>
            <marker id="ws-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={SOFT} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
          </defs>

          {/* before */}
          <text x={98} y={16} textAnchor="middle" fontSize={12.5} fontWeight={700} style={{ fill: INK }}>Pretrained weights</text>
          <text x={98} y={32} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>a general next-token predictor</text>
          <Grid x={15} y={44} />

          {/* after */}
          <text x={662} y={16} textAnchor="middle" fontSize={12.5} fontWeight={700} style={{ fill: INK }}>Fine-tuned weights</text>
          <text x={662} y={32} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>same shape, same size</text>
          <Grid x={579} y={44} after />
          <rect x={579} y={164} width={12} height={12} rx={2.5} fill={ACC} />
          <text x={597} y={174} fontSize={11} style={{ fill: SOFT }}>the weights that moved most</text>

          {/* the data: a stack of example pairs */}
          <text x={380} y={16} textAnchor="middle" fontSize={12.5} fontWeight={700} style={{ fill: INK }}>2,000 example pairs</text>
          <text x={380} y={32} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>customer message → ideal reply in your format</text>
          <rect x={244} y={56} width={272} height={50} rx={7} fill={FILL} stroke={LINE} />
          <rect x={238} y={50} width={272} height={50} rx={7} fill={FILL} stroke={LINE} />
          <rect x={232} y={44} width={272} height={50} rx={7} fill={FILL} stroke={LINE} />
          <text x={242} y={63} fontSize={11} style={{ fill: INK }}>“Can I change my delivery address?”</text>
          <text x={242} y={82} fontSize={10.5} style={{ fill: INK, fontFamily: MONO }}><tspan style={{ fill: SOFT }}>→ </tspan>{'{"reply": "Yes. Use the Orders…'}</text>

          {/* the loop that turns one grid into the other */}
          <path d="M380 110 V132" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#ws-arrow)" />
          <path d="M192 140 H566" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#ws-arrow)" />
          <text x={380} y={160} textAnchor="middle" fontSize={11.5} style={{ fill: INK }}>the same training loop, a few passes,</text>
          <text x={380} y={175} textAnchor="middle" fontSize={11.5} style={{ fill: INK }}>a small learning rate</text>

          {/* behaviour with the same short prompt */}
          <path d="M14 196 H746" stroke={LINE} strokeWidth={1} />
          <text x={380} y={217} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>The same short prompt to both: <tspan style={{ fill: INK }}>“Where is my order?”</tspan></text>

          <rect x={14} y={230} width={352} height={80} rx={8} fill={FILL} stroke={LINE} />
          <text x={26} y={249} fontSize={11} fontWeight={700} style={{ fill: INK }}>Before: a chatty paragraph</text>
          <text x={26} y={268} fontSize={11} style={{ fill: INK }}>Thanks so much for reaching out! There are a few</text>
          <text x={26} y={283} fontSize={11} style={{ fill: INK }}>reasons an order can be delayed, so let me walk you</text>
          <text x={26} y={298} fontSize={11} style={{ fill: INK }}>through them one at a time. First of all, …</text>

          <rect x={394} y={230} width={352} height={80} rx={8} fill={FILL} stroke={LINE} />
          <text x={406} y={249} fontSize={11} fontWeight={700} style={{ fill: INK }}>After: fixed JSON, two sentences</text>
          <text x={406} y={270} fontSize={10.5} style={{ fill: INK, fontFamily: MONO }}>{'{"reply": "It shipped today. It arrives Friday.",'}</text>
          <text x={406} y={287} fontSize={10.5} style={{ fill: INK, fontFamily: MONO }}>{' "status": "shipped"}'}</text>
        </svg>
      </div>
    </figure>
  )
}
