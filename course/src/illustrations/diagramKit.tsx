// Shared bits for the quiet concept diagrams (see AUTHORING.md, "Concept diagrams").
// Theme variables only; the accent is reserved for the one thing a picture is about.
import type { CSSProperties, ReactNode } from 'react'

export const INK = 'var(--ink)'
export const SOFT = 'var(--ink-3)'
export const LINE = 'var(--rule-strong)'
export const FILL = 'var(--paper-2)'
export const ACC = 'var(--accent)'
export const ON_ACC = 'var(--on-accent)'
export const MONO: CSSProperties = { fontFamily: 'var(--mono)' }

/** Two open arrowheads: url(#id) in grey, url(#id-acc) in the accent. `id` must be unique per component. */
export function ArrowDefs({ id }: { id: string }) {
  const head = (mid: string, colour: string, w: number) => (
    <marker id={mid} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M1 1 L9 5 L1 9" fill="none" stroke={colour} strokeWidth={w} strokeLinecap="round" strokeLinejoin="round" />
    </marker>
  )
  return <defs>{head(id, SOFT, 1.6)}{head(`${id}-acc`, ACC, 1.8)}</defs>
}

/** Optional accent numeral + bold title + short grey lines. */
export function Caption({ x, y, n, title, lines = [], anchor = 'middle', accent = false }: { x: number; y: number; n?: number; title: string; lines?: string[]; anchor?: 'start' | 'middle' | 'end'; accent?: boolean }) {
  return (
    <g textAnchor={anchor}>
      <text x={x} y={y} fontSize={12.5} fontWeight={700} style={{ fill: accent ? ACC : INK }}>
        {n !== undefined && <tspan style={{ fill: ACC }}>{n} </tspan>}{title}
      </text>
      {lines.map((l, i) => <text key={i} x={x} y={y + 16 + i * 14} fontSize={11.5} style={{ fill: SOFT }}>{l}</text>)}
    </g>
  )
}

export function Figure({ id, title, height, minWidth = 640, children, after }: { id: string; title: string; height: number; minWidth?: number; children: ReactNode; after?: ReactNode }) {
  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox={`0 0 760 ${height}`} style={{ width: '100%', minWidth, display: 'block' }} role="img" aria-labelledby={`${id}-title`}>
          <title id={`${id}-title`}>{title}</title>
          <ArrowDefs id={`${id}-arrow`} />
          {children}
        </svg>
      </div>
      {after}
    </figure>
  )
}
