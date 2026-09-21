// A hand-placed 2D "embedding space": click two words to compare them, drag words
// to see every number change, and run the king − man + woman arrow demo.
import { useRef, useState } from 'react'
import { Lab } from '../components/ui'
import { fmt } from '../lib/math'
import { TOY_WORDS, analogy, neighbours, pairStats } from '../lib/embeddingToy'

const SIZE = 480 // viewBox is SIZE × SIZE, world coordinates run from -R to R
const R = 5.6
const sx = (x: number) => ((x + R) / (2 * R)) * SIZE
const sy = (y: number) => ((R - y) / (2 * R)) * SIZE
const clamp = (v: number) => Math.max(-5.2, Math.min(5.2, Math.round(v * 10) / 10))

export function EmbeddingExplorer() {
  const [words, setWords] = useState(() => TOY_WORDS.map((w) => ({ ...w, pos: w.pos.slice() })))
  const [sel, setSel] = useState<[number, number]>([5, 4]) // cat, dog
  const [showAnalogy, setShowAnalogy] = useState(false)
  const [focus, setFocus] = useState(-1)
  const svgRef = useRef<SVGSVGElement>(null)
  const drag = useRef<{ i: number; moved: boolean; x: number; y: number } | null>(null)

  const [a, b] = sel
  const A = words[a], B = words[b]
  const stats = pairStats(A.pos, B.pos)
  const near = neighbours(words, a).slice(0, 5)
  const ana = analogy(words, 'king', 'man', 'woman')
  const at = (w: string) => words.find((x) => x.word === w)!.pos
  const moved = words.some((w, i) => w.pos[0] !== TOY_WORDS[i].pos[0] || w.pos[1] !== TOY_WORDS[i].pos[1])

  const select = (i: number) => setSel(([pa, pb]) => (i === pa ? [pa, pb] : [i, pa]))
  const moveTo = (i: number, x: number, y: number) => setWords((ws) => ws.map((w, j) => (j === i ? { ...w, pos: [clamp(x), clamp(y)] } : w)))

  const toWorld = (e: React.PointerEvent) => {
    const box = svgRef.current!.getBoundingClientRect()
    return [((e.clientX - box.left) / box.width) * 2 * R - R, R - ((e.clientY - box.top) / box.height) * 2 * R]
  }
  const onMove = (e: React.PointerEvent) => {
    if (!drag.current) return
    // a few pixels of jitter still counts as a click, not a drag
    if (!drag.current.moved && Math.hypot(e.clientX - drag.current.x, e.clientY - drag.current.y) < 4) return
    drag.current.moved = true
    const [x, y] = toWorld(e)
    moveTo(drag.current.i, x, y)
  }
  const onUp = () => {
    if (drag.current && !drag.current.moved) select(drag.current.i)
    drag.current = null
  }
  const onKey = (i: number) => (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 1 : 0.2
    const d: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(i) }
    else if (d[e.key]) { e.preventDefault(); moveTo(i, words[i].pos[0] + d[e.key][0], words[i].pos[1] + d[e.key][1]) }
  }

  return (
    <Lab
      title="A toy embedding space you can rearrange"
      goal={<>Click two words to compare them. <b>Drag</b> a word (or focus it and use the arrow keys) and watch distance, dot product and cosine change. Then switch on the arrow demo.</>}
    >
      <p className="lab-note" style={{ marginTop: 0 }}>
        <b>We cannot draw the real space. This 2D picture is only a teaching model.</b> The positions were placed by hand so the picture is readable. Real embeddings have hundreds or thousands of coordinates, and nobody places them.
      </p>
      <div className="grid-2" style={{ alignItems: 'start' }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          style={{ width: '100%', maxWidth: SIZE, touchAction: 'none', userSelect: 'none', border: '1px solid var(--rule)', borderRadius: 8, background: 'var(--card)' }}
          role="group"
          aria-label={`Two-dimensional toy embedding space with ${words.length} draggable words. Selected: ${A.word} and ${B.word}.`}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerLeave={onUp}
        >
          <defs>
            <marker id="emb-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent)" /></marker>
          </defs>
          {[-4, -2, 2, 4].map((t) => (
            <g key={t}>
              <line className="gridline" x1={sx(t)} y1={0} x2={sx(t)} y2={SIZE} />
              <line className="gridline" x1={0} y1={sy(t)} x2={SIZE} y2={sy(t)} />
            </g>
          ))}
          <line className="axis" x1={sx(0)} y1={0} x2={sx(0)} y2={SIZE} />
          <line className="axis" x1={0} y1={sy(0)} x2={SIZE} y2={sy(0)} />
          <text x={sx(0) + 5} y={sy(0) + 14} fontSize={11}>(0, 0)</text>

          {/* the two selected words as arrows from the origin: cosine is about the angle between them */}
          {[A, B].map((w, i) => <line key={i} x1={sx(0)} y1={sy(0)} x2={sx(w.pos[0])} y2={sy(w.pos[1])} stroke="var(--ink-3)" strokeWidth={1.5} strokeDasharray={i === 0 ? undefined : '5 4'} />)}
          <line x1={sx(A.pos[0])} y1={sy(A.pos[1])} x2={sx(B.pos[0])} y2={sy(B.pos[1])} stroke="var(--ink-2)" strokeWidth={1} strokeDasharray="2 3" />

          {showAnalogy && (
            <g>
              <line x1={sx(at('man')[0])} y1={sy(at('man')[1])} x2={sx(at('woman')[0])} y2={sy(at('woman')[1])} stroke="var(--accent)" strokeWidth={2.5} markerEnd="url(#emb-arrow)" />
              <line x1={sx(at('king')[0])} y1={sy(at('king')[1])} x2={sx(ana.landing[0])} y2={sy(ana.landing[1])} stroke="var(--accent)" strokeWidth={2.5} strokeDasharray="6 4" markerEnd="url(#emb-arrow)" />
              <path d={`M ${sx(ana.landing[0]) - 7} ${sy(ana.landing[1]) - 7} l 14 14 m 0 -14 l -14 14`} stroke="var(--accent)" strokeWidth={2.5} />
              <text x={sx(ana.landing[0]) + 10} y={sy(ana.landing[1]) + 18} fontSize={11} style={{ fill: 'var(--accent-ink)' }}>king − man + woman</text>
            </g>
          )}

          {words.map((w, i) => {
            const isA = i === a, isB = i === b
            return (
              <g
                key={w.word}
                role="button"
                tabIndex={0}
                aria-pressed={isA || isB}
                aria-label={`${w.word}, at ${w.pos[0].toFixed(1)}, ${w.pos[1].toFixed(1)}${isA ? ', selected as A' : isB ? ', selected as B' : ''}. Enter selects, arrow keys move.`}
                style={{ cursor: 'grab', outline: 'none' }}
                onPointerDown={(e) => { drag.current = { i, moved: false, x: e.clientX, y: e.clientY }; (e.target as Element).setPointerCapture?.(e.pointerId) }}
                onKeyDown={onKey(i)}
                onFocus={() => setFocus(i)}
                onBlur={() => setFocus(-1)}
              >
                <circle cx={sx(w.pos[0])} cy={sy(w.pos[1])} r={isA || isB ? 9 : 7} fill={isA || isB ? 'var(--accent)' : 'var(--paper-2)'} stroke={isA || isB ? 'var(--accent-ink)' : 'var(--ink-3)'} strokeWidth={isA || isB ? 2.5 : 1} />
                <circle cx={sx(w.pos[0])} cy={sy(w.pos[1])} r={18} fill="transparent" stroke={focus === i ? 'var(--accent)' : 'none'} strokeWidth={2} strokeDasharray="3 3" />
                <text x={sx(w.pos[0]) + 12} y={sy(w.pos[1]) + 4} fontSize={14} style={{ fontWeight: isA || isB ? 700 : 400, pointerEvents: 'none' }}>
                  {w.word}{isA ? ' (A)' : isB ? ' (B)' : ''}
                </text>
              </g>
            )
          })}
        </svg>

        <div>
          <div className="readout" aria-live="polite" style={{ flexDirection: 'column', gap: 4 }}>
            <span>A = {A.word} <b>[{A.pos.map((v) => v.toFixed(1)).join(', ')}]</b></span>
            <span>B = {B.word} <b>[{B.pos.map((v) => v.toFixed(1)).join(', ')}]</b></span>
            <span>distance: <b>{fmt(stats.distance)}</b></span>
            <span>dot product: {fmt(A.pos[0], 1)}×{fmt(B.pos[0], 1)} + {fmt(A.pos[1], 1)}×{fmt(B.pos[1], 1)} = <b>{fmt(stats.dot)}</b></span>
            <span>cosine similarity: <b>{fmt(stats.cosine, 3)}</b></span>
          </div>
          <div className="matrix-cap" style={{ marginTop: 12 }}>Nearest neighbours of “{A.word}” (by cosine)</div>
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>word</th><th>cosine</th><th>distance</th></tr></thead>
            <tbody>
              {near.map((r) => <tr key={r.word}><td>{r.word}</td><td>{fmt(r.cosine, 3)}</td><td>{fmt(r.distance)}</td></tr>)}
            </tbody>
          </table>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5, marginTop: 12 }}>
            <input type="checkbox" checked={showAnalogy} onChange={(e) => setShowAnalogy(e.target.checked)} /> Show the arrow demo: king − man + woman
          </label>
          {showAnalogy && (
            <p className="lab-note" aria-live="polite">
              The solid arrow goes from “man” to “woman”. The dashed arrow is the <em>same</em> arrow, started at “king”. It lands at [{ana.landing.map((v) => v.toFixed(1)).join(', ')}] (the ✕). Closest word: <b>{ana.nearest.word}</b>, {fmt(ana.nearest.distance)} away. Drag “woman” and watch the ✕ move with it.
            </p>
          )}
          <div className="btn-row" style={{ marginTop: 10 }}>
            {moved && <button className="btn small" onClick={() => setWords(TOY_WORDS.map((w) => ({ ...w, pos: w.pos.slice() })))}>Reset positions</button>}
          </div>
          <p className="muted" style={{ fontSize: 13 }}>Arrows from (0, 0): solid to A, dashed to B. Cosine depends only on the angle between them. Distance is the dotted line.</p>
        </div>
      </div>
    </Lab>
  )
}
