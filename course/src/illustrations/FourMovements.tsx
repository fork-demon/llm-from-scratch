// A deliberately EMPTY skeleton of the whole pipeline: twelve blank slots grouped into the
// four movements, plus the loop. It must not name any stage: the learner is about to
// reconstruct the names from memory. Only the grouping and the counts are given away.

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const ACC = 'var(--accent)'
const MONO = 'var(--mono)'

const W = 40 // slot width
const H = 34 // slot height
const GAP = 20 // gap between slots in one movement
const GROUP_GAP = 42 // gap between movements
const Y = 100 // top of the slot row
const X0 = 38

// 5 + 2 + 4 stages in a row; the twelfth stage is the loop itself
const GROUPS = [5, 2, 4]

export function FourMovements() {
  const slots: { x: number; n: number; g: number }[] = []
  const spans: { x1: number; x2: number }[] = []
  let x = X0
  let n = 1
  GROUPS.forEach((count, g) => {
    const x1 = x
    for (let i = 0; i < count; i++) {
      slots.push({ x, n: n++, g })
      x += W + (i < count - 1 ? GAP : 0)
    }
    spans.push({ x1, x2: x })
    x += GROUP_GAP
  })
  const mid = (s: { x1: number; x2: number }) => (s.x1 + s.x2) / 2
  const first = slots[0]
  const last = slots[slots.length - 1]
  const loopY = 204
  const cy = Y + H / 2

  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 262" style={{ width: '100%', minWidth: 640, display: 'block' }} role="img" aria-labelledby="four-title">
          <title id="four-title">A blank skeleton of the pipeline: five empty slots for “text becomes numbers”, two empty slots repeated N times for “numbers are transformed, using context”, four empty slots for “numbers become one choice”, and a loop arrow with one empty slot for “the choice is fed back in”</title>
          <defs>
            <marker id="four-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={SOFT} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></marker>
            <marker id="four-arrow-acc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={ACC} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></marker>
          </defs>

          {/* ---- movement brackets and names ---- */}
          {spans.map((s, g) => (
            <path key={g} d={`M${s.x1 - 6} ${Y - 8} V${Y - 16} H${s.x2 + 6} V${Y - 8}`} fill="none" stroke={SOFT} strokeWidth={1.5} />
          ))}
          <g textAnchor="middle">
            <text x={mid(spans[0])} y={50} fontSize={12.5} fontWeight={700} style={{ fill: INK }}><tspan style={{ fill: ACC }}>1 </tspan>Text becomes numbers</text>
            <text x={mid(spans[0])} y={67} fontSize={11.5} style={{ fill: SOFT }}>five stages</text>

            <text x={mid(spans[1])} y={18} fontSize={12.5} fontWeight={700} style={{ fill: INK }}><tspan style={{ fill: ACC }}>2 </tspan>Numbers are</text>
            <text x={mid(spans[1])} y={34} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>transformed,</text>
            <text x={mid(spans[1])} y={50} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>using context</text>
            <text x={mid(spans[1])} y={67} fontSize={11.5} style={{ fill: SOFT }}>two stages</text>

            <text x={mid(spans[2])} y={50} fontSize={12.5} fontWeight={700} style={{ fill: INK }}><tspan style={{ fill: ACC }}>3 </tspan>Numbers become one choice</text>
            <text x={mid(spans[2])} y={67} fontSize={11.5} style={{ fill: SOFT }}>four stages</text>
          </g>

          {/* ---- the empty slots ---- */}
          {slots.map((s, i) => (
            <g key={s.n}>
              <rect x={s.x} y={Y} width={W} height={H} rx={7} fill="none" stroke={SOFT} strokeWidth={1.4} strokeDasharray="5 4" />
              <text x={s.x + W / 2} y={Y + H / 2 + 4} textAnchor="middle" fontSize={11} style={{ fill: SOFT, fontFamily: MONO }}>{s.n}</text>
              {i < slots.length - 1 && (
                <path d={`M${s.x + W + 3} ${cy} H${slots[i + 1].x - 4}`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#four-arrow)" />
              )}
            </g>
          ))}

          {/* movement 2 is repeated N times */}
          <path d={`M${spans[1].x2 - W / 2} ${Y + H + 5} V${Y + H + 20} H${spans[1].x1 + W / 2} V${Y + H + 7}`} fill="none" stroke={SOFT} strokeWidth={1.5} strokeLinejoin="round" markerEnd="url(#four-arrow)" />
          <text x={mid(spans[1])} y={Y + H + 36} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>repeated N times</text>

          {/* ---- movement 4: the loop, with its one slot ---- */}
          <path
            d={`M${last.x + W / 2} ${Y + H + 5} V${loopY} H${first.x + W / 2} V${Y + H + 7}`}
            fill="none" stroke={ACC} strokeWidth={2.2} strokeDasharray="6 5" strokeLinejoin="round" markerEnd="url(#four-arrow-acc)"
          />
          <rect x={380 - W / 2 - 6} y={loopY - H / 2 - 2} width={W + 12} height={H + 4} fill="var(--paper)" />
          <rect x={380 - W / 2} y={loopY - H / 2} width={W} height={H} rx={7} fill="none" stroke={ACC} strokeWidth={1.4} strokeDasharray="5 4" />
          <text x={380} y={loopY + 4} textAnchor="middle" fontSize={11} style={{ fill: SOFT, fontFamily: MONO }}>12</text>
          <g textAnchor="middle">
            <text x={380} y={240} fontSize={12.5} fontWeight={700} style={{ fill: INK }}><tspan style={{ fill: ACC }}>4 </tspan>The choice is fed back in</text>
            <text x={380} y={256} fontSize={11.5} style={{ fill: SOFT }}>the loop: one stage</text>
          </g>
        </svg>
      </div>
      <ol className="sr-only">
        <li>Text becomes numbers: five stages.</li>
        <li>Numbers are transformed, using context: two stages, repeated N times.</li>
        <li>Numbers become one choice: four stages.</li>
        <li>The choice is fed back in: the loop.</li>
      </ol>
    </figure>
  )
}
