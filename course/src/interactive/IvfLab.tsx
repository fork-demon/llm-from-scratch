// IVF in two dimensions: cluster once, then search only the nprobe nearest clusters.
import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { Lab, Slider } from '../components/ui'
import { buildIvf, makeClusteredPoints, measure, searchExact, searchIvf, type P2 } from '../lib/ivf'

const N_CLUSTERS = 8
const S = 400 // svg size
// shapes + hues so clusters are not told apart by colour alone
const HUES = [210, 35, 140, 280, 0, 175, 320, 90]

export function IvfLab() {
  const index = useMemo(() => buildIvf(makeClusteredPoints(), N_CLUSTERS), [])
  const table = useMemo(() => [1, 2, 3, 4, 8].map((n) => ({ n, ...measure(index, n) })), [index])
  const [q, setQ] = useState<P2>([0.5, 0.5])
  const [nprobe, setNprobe] = useState(1)
  const svg = useRef<SVGSVGElement>(null)
  const dragging = useRef(false)

  const truth = searchExact(index.points, q, 1)[0]
  const r = searchIvf(index, q, 1, nprobe)
  const found = r.result[0] === truth
  const probed = new Set(r.probed)

  const toPoint = (e: PointerEvent<SVGSVGElement>): P2 => {
    const box = svg.current!.getBoundingClientRect()
    const clamp = (x: number) => Math.min(1, Math.max(0, x))
    return [clamp((e.clientX - box.left) / box.width), clamp((e.clientY - box.top) / box.height)]
  }
  const onKey = (e: KeyboardEvent) => {
    const d = e.shiftKey ? 0.05 : 0.01
    const move: Record<string, P2> = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, -d], ArrowDown: [0, d] }
    const m = move[e.key]
    if (!m) return
    e.preventDefault()
    setQ([Math.min(1, Math.max(0, q[0] + m[0])), Math.min(1, Math.max(0, q[1] + m[1]))])
  }
  // a query where nprobe = 1 gets the wrong answer, found by scanning a grid
  const jumpToMiss = () => {
    for (let a = 0; a < 40; a++) for (let b = 0; b < 40; b++) {
      const p: P2 = [((a * 17) % 40 + 0.5) / 40, ((b * 23) % 40 + 0.5) / 40]
      if (searchIvf(index, p, 1, 1).result[0] !== searchExact(index.points, p, 1)[0]) { setQ(p); setNprobe(1); return }
    }
  }

  return (
    <Lab
      title="Search fewer vectors: the IVF index"
      goal={<>Click or drag anywhere to move the <b>query</b> (or focus the plot and use the arrow keys). Set <b>nprobe</b> to 1 and hunt for a spot where the index returns the wrong neighbour. Hint: stand near a border between two clusters.</>}
    >
      <div className="controls">
        <Slider label="nprobe (how many clusters to search)" value={nprobe} min={1} max={N_CLUSTERS} step={1} onChange={setNprobe} />
        <div className="btn-row" style={{ alignSelf: 'end' }}><button className="btn small" onClick={jumpToMiss}>Show me a miss at nprobe = 1</button></div>
      </div>
      <div className="grid-2" style={{ alignItems: 'start' }}>
        <svg
          ref={svg} viewBox={`0 0 ${S} ${S}`} tabIndex={0} onKeyDown={onKey}
          style={{ width: '100%', maxWidth: 440, touchAction: 'none', cursor: 'crosshair', border: '1px solid var(--rule)', borderRadius: 8, background: 'var(--paper-2)' }}
          role="img"
          aria-label={`Scatter plot of ${index.points.length} points in ${N_CLUSTERS} clusters. Query at ${q[0].toFixed(2)}, ${q[1].toFixed(2)}. Searching ${nprobe} clusters, ${r.candidates} points compared. True nearest neighbour ${found ? 'found' : 'missed'}.`}
          onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); setQ(toPoint(e)) }}
          onPointerMove={(e) => { if (dragging.current) setQ(toPoint(e)) }}
          onPointerUp={() => { dragging.current = false }}
        >
          {index.points.map((p, i) => {
            const c = index.assign[i]
            const on = probed.has(c)
            return <circle key={i} cx={p[0] * S} cy={p[1] * S} r={on ? 3.4 : 2.4} fill={`hsl(${HUES[c]} 60% 50%)`} opacity={on ? 0.95 : 0.22} />
          })}
          {index.centroids.map((c, j) => (
            <g key={j} opacity={probed.has(j) ? 1 : 0.45}>
              <rect x={c[0] * S - 6} y={c[1] * S - 6} width={12} height={12} transform={`rotate(45 ${c[0] * S} ${c[1] * S})`} fill="var(--card)" stroke="var(--ink)" strokeWidth={probed.has(j) ? 2.2 : 1} />
              <text x={c[0] * S} y={c[1] * S + 3.5} textAnchor="middle" fontSize={9}>{j + 1}</text>
            </g>
          ))}
          {/* what IVF returned, and the truth */}
          <line x1={q[0] * S} y1={q[1] * S} x2={index.points[r.result[0]][0] * S} y2={index.points[r.result[0]][1] * S} stroke={found ? 'var(--good)' : 'var(--bad)'} strokeWidth={2} />
          <circle cx={index.points[r.result[0]][0] * S} cy={index.points[r.result[0]][1] * S} r={7} fill="none" stroke={found ? 'var(--good)' : 'var(--bad)'} strokeWidth={2} />
          {!found && <>
            <line x1={q[0] * S} y1={q[1] * S} x2={index.points[truth][0] * S} y2={index.points[truth][1] * S} stroke="var(--good)" strokeWidth={1.5} strokeDasharray="4 3" />
            <circle cx={index.points[truth][0] * S} cy={index.points[truth][1] * S} r={7} fill="none" stroke="var(--good)" strokeWidth={2} strokeDasharray="3 2" />
          </>}
          <circle cx={q[0] * S} cy={q[1] * S} r={6} fill="var(--accent)" stroke="var(--card)" strokeWidth={2} />
          <text x={q[0] * S + 10} y={q[1] * S - 8} fontSize={12} fontWeight={600}>query</text>
        </svg>

        <div>
          <div className="readout" aria-live="polite" style={{ flexDirection: 'column', gap: 6 }}>
            <span>clusters searched: <b>{r.probed.map((c) => c + 1).join(', ')}</b> of {N_CLUSTERS}</span>
            <span>points compared: <b>{r.candidates}</b> of {index.points.length} (<b>{((100 * r.candidates) / index.points.length).toFixed(0)}%</b>) + {N_CLUSTERS} centroids</span>
            <span>true nearest neighbour: <b style={{ color: found ? 'var(--good)' : 'var(--bad)' }}>{found ? 'found ✓' : 'MISSED ✗'}</b></span>
          </div>
          <p className="lab-note" style={{ marginTop: 10 }}>◆ = cluster centre (found once, by k-means, at index-build time). Bright points are in the searched clusters; faded points are <em>never looked at</em>. {found ? '' : 'The dashed green circle is the true nearest neighbour. It sits in a cluster that was skipped, because the query is closer to another cluster’s centre.'}</p>
          <div className="table-scroll">
            <table className="plain mono" style={{ fontSize: 13.5 }}>
              <thead><tr><th>nprobe</th><th>points compared</th><th>finds the true nearest</th></tr></thead>
              <tbody>
                {table.map((t) => (
                  <tr key={t.n} style={{ fontWeight: t.n === nprobe ? 700 : 400 }}><td>{t.n}</td><td>{(t.fraction * 100).toFixed(0)}%</td><td>{(t.recall * 100).toFixed(0)}% of queries</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="lab-note">Measured live over a 15×15 grid of query positions on this data. 2-D is kind to IVF; in 64 dimensions the misses are far more common (see the real benchmark below).</p>
        </div>
      </div>
    </Lab>
  )
}
