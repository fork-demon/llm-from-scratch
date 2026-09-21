// The life of ONE request on a time axis: wait in the queue, one prefill pass over the whole
// prompt, then a steady drip of single tokens. The metrics are drawn as brackets on that axis,
// and the row underneath shows how many tokens go through the model in each pass.
// Durations follow the lesson's cost model: a 500-token prompt prefills in about 48 ms,
// a decode step takes about 10 ms. The 40 ms queue wait is just an example.

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const FILL = 'var(--paper-2)'
const ACC = 'var(--accent)'
const MONO = 'var(--mono)'

const PX = 1.6 // pixels per millisecond
const X0 = 34
const ARRIVE = X0
const PREFILL0 = ARRIVE + 40 * PX
const FIRST = PREFILL0 + 48 * PX
const STEP = 10 * PX
const N_TOKENS = 30
const LAST = FIRST + (N_TOKENS - 1) * STEP

function Bracket({ x1, x2, y, up, label, sub, accent }: { x1: number; x2: number; y: number; up?: boolean; label: string; sub?: string; accent?: boolean }) {
  const d = up ? -6 : 6
  const c = accent ? ACC : INK
  return (
    <g>
      <path d={`M${x1} ${y + d} V${y} H${x2} V${y + d}`} fill="none" stroke={c} strokeWidth={1.4} />
      <text x={(x1 + x2) / 2} y={up ? y + 15 : y - 7} textAnchor="middle" fontSize={11.5} fontWeight={700} style={{ fill: c }}>{label}</text>
      {sub && <text x={(x1 + x2) / 2} y={up ? y + 29 : y - 21} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>{sub}</text>}
    </g>
  )
}

export function RequestTimeline() {
  const LANE = 96
  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 292" style={{ width: '100%', minWidth: 700, display: 'block' }} role="img" aria-labelledby="rtl-title">
          <title id="rtl-title">Timeline of one request. It waits in a queue, then one prefill pass processes all 500 prompt tokens, which produces the first token. Time to first token spans the queue wait and the prefill. After that, one token appears about every 10 milliseconds: the time per output token. End-to-end latency spans everything.</title>
          <defs>
            <pattern id="rtl-queue" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" stroke={SOFT} strokeWidth="1.6" />
            </pattern>
          </defs>

          {/* metrics above the lane */}
          <Bracket x1={ARRIVE} x2={FIRST} y={62} label="time to first token (TTFT)" sub="queue wait + prefill" />
          <Bracket x1={FIRST + 8 * STEP} x2={FIRST + 9 * STEP} y={62} label="time per output token (TPOT)" sub="one decode step, about 10 ms" accent />

          {/* the lane */}
          <line x1={ARRIVE} x2={ARRIVE} y1={LANE - 22} y2={LANE + 22} stroke={INK} strokeWidth={1.5} />
          <text x={ARRIVE - 4} y={LANE + 36} fontSize={11} style={{ fill: SOFT }}>request arrives</text>
          <rect x={ARRIVE} y={LANE - 13} width={PREFILL0 - ARRIVE} height={26} fill="url(#rtl-queue)" stroke={LINE} />
          <text x={(ARRIVE + PREFILL0) / 2} y={LANE - 19} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>queue</text>
          <rect x={PREFILL0} y={LANE - 13} width={FIRST - PREFILL0} height={26} rx={3} fill={INK} />
          <text x={(PREFILL0 + FIRST) / 2} y={LANE + 4} textAnchor="middle" fontSize={11.5} fontWeight={700} style={{ fill: 'var(--paper)' }}>prefill</text>
          {Array.from({ length: N_TOKENS }, (_, i) => (
            <rect key={i} x={FIRST + i * STEP - 3} y={LANE - 13} width={6} height={26} rx={2} fill={ACC} />
          ))}
          <text x={LAST + 12} y={LANE + 4} fontSize={13} style={{ fill: SOFT }}>…</text>
          <text x={FIRST} y={LANE + 36} textAnchor="middle" fontSize={11} style={{ fill: ACC }} fontWeight={700}>first token</text>

          {/* end to end */}
          <Bracket x1={ARRIVE} x2={LAST + 3} y={150} up label="end-to-end latency = TTFT + (output tokens − 1) × TPOT" />

          {/* what goes through the model in each pass */}
          <text x={X0} y={204} fontSize={11.5} fontWeight={700} style={{ fill: INK }}>tokens through the model per pass</text>
          {Array.from({ length: 5 }, (_, r) => Array.from({ length: 12 }, (_, c) => (
            <rect key={`${r}-${c}`} x={PREFILL0 + 2 + c * 6.1} y={214 + r * 6.1} width={4.6} height={4.6} rx={1} fill={INK} opacity={0.75} />
          )))}
          <text x={(PREFILL0 + FIRST) / 2} y={258} textAnchor="middle" fontSize={11} style={{ fill: INK, fontFamily: MONO }}>500 at once</text>
          <text x={(PREFILL0 + FIRST) / 2} y={273} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>limited by arithmetic</text>
          {Array.from({ length: N_TOKENS - 1 }, (_, i) => (
            <rect key={i} x={FIRST + (i + 1) * STEP - 2.3} y={226} width={4.6} height={4.6} rx={1} fill={ACC} />
          ))}
          <text x={(FIRST + LAST) / 2 + 8} y={258} textAnchor="middle" fontSize={11} style={{ fill: INK, fontFamily: MONO }}>1 per pass, and every pass reads all the weights</text>
          <text x={(FIRST + LAST) / 2 + 8} y={273} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>limited by memory traffic</text>
        </svg>
      </div>
      <ol className="sr-only">
        <li>The request arrives and waits in a queue.</li>
        <li>Prefill: all prompt tokens go through the model in one pass. This is limited by arithmetic.</li>
        <li>The first token appears. Arrival to here is the time to first token.</li>
        <li>Decode: one token per pass, each pass reading all the weights. The gap between tokens is the time per output token.</li>
        <li>End-to-end latency is the time to first token plus the remaining tokens times the time per output token.</li>
      </ol>
    </figure>
  )
}
