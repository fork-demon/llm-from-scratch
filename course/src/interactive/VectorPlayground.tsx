// Lesson 1.1: two arrows you can drag. The dot product, lengths and cosine
// similarity update live, with every pairwise product spelled out.
import { useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { Lab } from '../components/ui'
import { fmt, type Vec } from '../lib/math'
import { describeDot, fromScreen, projection, snap, toScreen, type Verdict } from '../lib/vectors2d'

const PLANE = { size: 360, limit: 5 }
const STEP = 0.5
type Which = 'A' | 'B'

const PRESETS: { label: string; a: Vec; b: Vec }[] = [
  { label: 'same direction', a: [3, 1], b: [3, 1] },
  { label: 'roughly agree', a: [3, 1], b: [2, 4] },
  { label: 'perpendicular', a: [3, 1], b: [-1, 3] },
  { label: 'opposite', a: [3, 1], b: [-3, -1] },
  { label: 'short vs long', a: [1, 0.5], b: [4, 2] },
]

const VERDICT: Record<Verdict, { word: string; text: string }> = {
  agree: { word: 'AGREE', text: 'The arrows point broadly the same way, so the products add up to a positive number.' },
  unrelated: { word: 'UNRELATED', text: 'The arrows are (nearly) at right angles. Neither says anything about the other: the products cancel out.' },
  oppose: { word: 'OPPOSE', text: 'The arrows point broadly opposite ways. Where one is positive the other is negative, so the sum is negative.' },
  zero: { word: 'NO DIRECTION', text: 'One arrow has length 0. It points nowhere, so there is nothing to agree with.' },
}

const n = (x: number) => fmt(x, Number.isInteger(x * 2) ? 1 : 2)
const paren = (x: number) => (x < 0 ? `(${n(x)})` : n(x))

/** A number input that lets you type "-" or clear the box without the value jumping back. */
function NumberField({ value, label, onChange }: { value: number; label: string; onChange: (x: number) => void }) {
  const [draft, setDraft] = useState<{ text: string; at: number } | null>(null)
  const shown = draft && draft.at === value ? draft.text : String(value)
  return (
    <input
      className="input mono"
      style={{ width: 76, padding: '4px 8px' }}
      type="number"
      step={STEP}
      min={-PLANE.limit}
      max={PLANE.limit}
      value={shown}
      aria-label={label}
      onBlur={() => setDraft(null)}
      onChange={(e) => {
        const text = e.target.value
        const x = Number(text)
        if (text === '' || !Number.isFinite(x)) return setDraft({ text, at: value })
        const snapped = snap(x, STEP, PLANE.limit)
        setDraft({ text: snapped === x ? text : String(snapped), at: snapped })
        onChange(snapped)
      }}
    />
  )
}

export function VectorPlayground() {
  const [a, setA] = useState<Vec>([3, 1])
  const [b, setB] = useState<Vec>([2, 4])
  const [shadow, setShadow] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)
  const dragging = useRef<Which | null>(null)

  const set = (which: Which, v: Vec) => (which === 'A' ? setA : setB)([snap(v[0], STEP, PLANE.limit), snap(v[1], STEP, PLANE.limit)])
  const r = describeDot(a, b)

  const pointerToPlane = (e: PointerEvent): Vec | null => {
    const box = svgRef.current?.getBoundingClientRect()
    if (!box || box.width === 0) return null
    const k = PLANE.size / box.width
    return fromScreen((e.clientX - box.left) * k, (e.clientY - box.top) * k, PLANE)
  }
  const onDown = (which: Which) => (e: PointerEvent<SVGGElement>) => {
    dragging.current = which
    e.currentTarget.setPointerCapture?.(e.pointerId)
  }
  const onMove = (e: PointerEvent<SVGGElement>) => {
    if (!dragging.current) return
    const p = pointerToPlane(e)
    if (p) set(dragging.current, p)
  }
  const onUp = () => { dragging.current = null }
  const onKey = (which: Which, v: Vec) => (e: KeyboardEvent) => {
    const d: Record<string, Vec> = { ArrowLeft: [-STEP, 0], ArrowRight: [STEP, 0], ArrowUp: [0, STEP], ArrowDown: [0, -STEP] }
    const move = d[e.key]
    if (!move) return
    e.preventDefault()
    set(which, [v[0] + move[0], v[1] + move[1]])
  }

  const [ox, oy] = toScreen([0, 0], PLANE)
  const ticks = Array.from({ length: PLANE.limit * 2 + 1 }, (_, i) => i - PLANE.limit)
  const proj = projection(a, b)
  const [px, py] = toScreen(proj, PLANE)

  const arrow = (which: Which, v: Vec, colour: string, dash?: string) => {
    const [x, y] = toScreen(v, PLANE)
    return (
      <g key={which}>
        <line x1={ox} y1={oy} x2={x} y2={y} stroke={colour} strokeWidth={3} strokeLinecap="round" strokeDasharray={dash} markerEnd={`url(#vp-head-${which})`} />
        <g
          tabIndex={0}
          role="slider"
          aria-label={`Tip of arrow ${which}. Arrow keys move it.`}
          aria-valuetext={`${which} is at ${n(v[0])}, ${n(v[1])}`}
          aria-valuenow={v[0]}
          aria-valuemin={-PLANE.limit}
          aria-valuemax={PLANE.limit}
          style={{ cursor: 'grab', touchAction: 'none', outlineOffset: 2 }}
          onPointerDown={onDown(which)}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onKeyDown={onKey(which, v)}
        >
          <circle cx={x} cy={y} r={20} fill="transparent" />
          <circle cx={x} cy={y} r={9} fill="var(--card)" stroke={colour} strokeWidth={3} />
        </g>
        <text x={x + 13} y={y - 11} fontSize={15} fontWeight={700} style={{ fill: colour }}>{which}</text>
      </g>
    )
  }

  const numberInput = (which: Which, v: Vec, i: 0 | 1) => (
    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 14 }}>
      <span className="mono">{which}{i === 0 ? 'x' : 'y'}</span>
      <NumberField
        value={v[i]}
        label={`${which}, ${i === 0 ? 'first number (left/right)' : 'second number (down/up)'}`}
        onChange={(x) => {
          const next = v.slice()
          next[i] = x
          set(which, next)
        }}
      />
    </label>
  )

  const isExactZero = r.dot === 0 && r.verdict !== 'zero'

  return (
    <Lab
      title="Two arrows, one question: do you agree?"
      goal={<>Drag the tips of <b>A</b> and <b>B</b> (or focus a tip and use the arrow keys, or type numbers). Before each move, <b>predict</b>: will the dot product go up, go down, or change sign?</>}
    >
      <div className="steps" role="group" aria-label="Starting positions">
        {PRESETS.map((p) => (
          <button key={p.label} className="step-btn" aria-pressed={p.a.join() === a.join() && p.b.join() === b.join()} onClick={() => { setA(p.a); setB(p.b) }}>{p.label}</button>
        ))}
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div>
          <svg
            ref={svgRef}
            viewBox={`0 0 ${PLANE.size} ${PLANE.size}`}
            style={{ width: '100%', maxWidth: 420, display: 'block', margin: '0 auto', userSelect: 'none', overflow: 'visible' }}
            role="group"
            aria-label={`A plane with two arrows from the origin. A points to ${n(a[0])}, ${n(a[1])}. B points to ${n(b[0])}, ${n(b[1])}. Their dot product is ${n(r.dot)}.`}
          >
            <defs>
              <marker id="vp-head-A" markerWidth="7" markerHeight="7" refX="9.5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--accent)" /></marker>
              <marker id="vp-head-B" markerWidth="7" markerHeight="7" refX="9.5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--ink-2)" /></marker>
            </defs>
            {ticks.map((t) => {
              const [x, y] = toScreen([t, t], PLANE)
              return (
                <g key={t}>
                  <line className={t === 0 ? 'axis' : 'gridline'} x1={x} y1={0} x2={x} y2={PLANE.size} />
                  <line className={t === 0 ? 'axis' : 'gridline'} x1={0} y1={y} x2={PLANE.size} y2={y} />
                  {t !== 0 && t % 2 === 0 && <text x={x} y={oy + 13} fontSize={9.5} textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{t}</text>}
                  {t !== 0 && t % 2 === 0 && <text x={ox - 5} y={y + 3} fontSize={9.5} textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{t}</text>}
                </g>
              )
            })}
            {shadow && r.verdict !== 'zero' && (
              <g>
                <line x1={toScreen(b, PLANE)[0]} y1={toScreen(b, PLANE)[1]} x2={px} y2={py} stroke="var(--ink-3)" strokeDasharray="3 4" />
                <line x1={ox} y1={oy} x2={px} y2={py} stroke={r.dot >= 0 ? 'var(--good)' : 'var(--bad)'} strokeWidth={7} opacity={0.45} strokeLinecap="round" />
              </g>
            )}
            {arrow('A', a, 'var(--accent)')}
            {arrow('B', b, 'var(--ink-2)', '7 4')}
          </svg>
          <p className="lab-note center" style={{ marginTop: 6 }}>A is the solid arrow, B is the dashed one. Tips snap to steps of {STEP}.</p>
        </div>

        <div>
          <div className="controls" style={{ gridTemplateColumns: '1fr 1fr', gap: '8px 12px' }}>
            {numberInput('A', a, 0)}{numberInput('A', a, 1)}
            {numberInput('B', b, 0)}{numberInput('B', b, 1)}
          </div>

          <div aria-live="polite">
            <div className="readout" style={{ display: 'block' }}>
              <div className="muted" style={{ fontSize: 12 }}>multiply matching slots, then add</div>
              <div>A · B = {n(a[0])}×{paren(b[0])} + {n(a[1])}×{paren(b[1])}</div>
              <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {n(r.terms[0])} + {paren(r.terms[1])} = <b style={{ fontSize: 18 }}>{n(r.dot)}</b></div>
            </div>
            <div className="card" style={{ margin: '10px 0', padding: '12px 16px', borderLeft: `4px solid ${r.verdict === 'agree' ? 'var(--good)' : r.verdict === 'oppose' ? 'var(--bad)' : 'var(--ink-3)'}` }}>
              <b className="mono" style={{ fontSize: 13, letterSpacing: '0.08em' }}>{r.dot > 0 ? '+ ' : r.dot < 0 ? '− ' : '0 '}{VERDICT[r.verdict].word}</b>
              <p style={{ fontSize: 14.5, marginTop: 4 }}>{VERDICT[r.verdict].text}</p>
              {isExactZero && <p style={{ fontSize: 14.5, marginTop: 4 }}><b>Exactly 0.</b> The arrows are perpendicular (90°).</p>}
            </div>
            <div className="readout">
              <span>length of A = <b>{fmt(r.lenA)}</b></span>
              <span>length of B = <b>{fmt(r.lenB)}</b></span>
              <span>angle = <b>{Number.isNaN(r.angleDeg) ? 'n/a' : `${fmt(r.angleDeg, 0)}°`}</b></span>
            </div>
            <div className="readout" style={{ display: 'block', marginTop: 10 }}>
              <div className="muted" style={{ fontSize: 12 }}>cosine similarity: the dot product with both lengths divided out</div>
              {r.verdict === 'zero'
                ? <div>not defined for an arrow of length 0 (we show 0)</div>
                : <div>{n(r.dot)} ÷ ({fmt(r.lenA)} × {fmt(r.lenB)}) = <b style={{ fontSize: 18 }}>{fmt(r.cosine)}</b> <span className="muted">(always between −1 and +1)</span></div>}
            </div>
          </div>

          <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5, marginTop: 12 }}>
            <input type="checkbox" checked={shadow} onChange={(e) => setShadow(e.target.checked)} /> Show B’s shadow on A (how much of B runs along A)
          </label>
          {shadow && <p className="lab-note" style={{ marginTop: 6 }}>The thick bar is the part of B that lies along A’s direction. The dot product is that bar’s length times A’s length, negative when the bar points backwards. At 90° the shadow vanishes, and so does the dot product.</p>}
        </div>
      </div>
    </Lab>
  )
}
