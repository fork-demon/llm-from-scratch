// Illustration kit: a flat, friendly cartoon style (thick outlines, rounded shapes,
// a small fixed palette) and "Tok", the robot who appears in every scene.
// Colours come from the --il-* CSS variables so scenes work in both themes.
import { useId, type ReactNode } from 'react'

export const C = {
  ink: 'var(--il-ink)',
  text: 'var(--il-text)',
  panel: 'var(--il-panel)',
  soft: 'var(--il-soft)',
  white: 'var(--il-w)',
  indigo: 'var(--il-a)',
  mint: 'var(--il-b)',
  amber: 'var(--il-c)',
  coral: 'var(--il-d)',
  sky: 'var(--il-e)',
}

const line = { stroke: C.ink, strokeWidth: 3, strokeLinejoin: 'round' as const, strokeLinecap: 'round' as const }
export const outlined = line

/** The frame every scene sits in: a soft rounded panel with a few decorative blobs. */
export function Scene({ label, children, blobs = true }: { label: string; children: ReactNode; blobs?: boolean }) {
  const clip = useId()
  return (
    <svg viewBox="0 0 480 300" role="img" aria-label={label} className="il-scene" preserveAspectRatio="xMidYMid meet">
      <defs><clipPath id={clip}><rect x="0" y="0" width="480" height="300" rx="28" /></clipPath></defs>
      <rect x="0" y="0" width="480" height="300" rx="28" fill={C.panel} />
      <g clipPath={`url(#${clip})`}>
      {blobs && (
        <g opacity="0.55" fill={C.soft}>
          <circle cx="52" cy="46" r="26" />
          <circle cx="438" cy="250" r="34" />
          <circle cx="420" cy="52" r="10" />
        </g>
      )}
      {children}
      </g>
    </svg>
  )
}

type Pt = [number, number]
type Mood = 'happy' | 'wow' | 'think' | 'wink'

/**
 * Tok the robot. Origin = between the feet. About 190 units tall at scale 1.
 * `look` shifts the pupils; `armL` / `armR` are hand positions relative to the origin.
 */
export function Tok({ x, y, s = 1, flip = false, look = [0, 0], mood = 'happy', armL = [-66, -48], armR = [66, -48], chest, hat }: {
  x: number; y: number; s?: number; flip?: boolean; look?: Pt; mood?: Mood; armL?: Pt; armR?: Pt; chest?: ReactNode; hat?: ReactNode
}) {
  const arm = (from: Pt, to: Pt, bend: number) => {
    const mx = (from[0] + to[0]) / 2 + bend
    const my = (from[1] + to[1]) / 2 + 14
    const d = `M${from[0]} ${from[1]} Q${mx} ${my} ${to[0]} ${to[1]}`
    return (
      <g>
        <path d={d} fill="none" stroke={C.ink} strokeWidth={13} strokeLinecap="round" />
        <path d={d} fill="none" stroke={C.indigo} strokeWidth={7} strokeLinecap="round" />
        <circle cx={to[0]} cy={to[1]} r={9} fill={C.white} {...line} />
      </g>
    )
  }
  const [lx, ly] = look
  return (
    <g transform={`translate(${x} ${y}) scale(${flip ? -s : s} ${s})`}>
      {/* shadow */}
      <ellipse cx="0" cy="2" rx="46" ry="7" fill={C.ink} opacity="0.12" />
      {/* legs + feet */}
      <rect x="-24" y="-30" width="14" height="28" rx="6" fill={C.indigo} {...line} />
      <rect x="10" y="-30" width="14" height="28" rx="6" fill={C.indigo} {...line} />
      <ellipse cx="-19" cy="-2" rx="15" ry="8" fill={C.white} {...line} />
      <ellipse cx="19" cy="-2" rx="15" ry="8" fill={C.white} {...line} />
      {/* arms go behind the body */}
      {arm([-38, -84], armL, -10)}
      {arm([38, -84], armR, 10)}
      {/* body */}
      <rect x="-42" y="-98" width="84" height="72" rx="20" fill={C.indigo} {...line} />
      <rect x="-24" y="-84" width="48" height="30" rx="9" fill={C.white} {...line} />
      <g>{chest ?? (<><rect x="-16" y="-66" width="7" height="8" rx="2" fill={C.mint} /><rect x="-4" y="-74" width="7" height="16" rx="2" fill={C.amber} /><rect x="8" y="-70" width="7" height="12" rx="2" fill={C.coral} /></>)}</g>
      {/* neck */}
      <rect x="-9" y="-108" width="18" height="12" rx="4" fill={C.soft} {...line} />
      {/* ears */}
      <rect x="-60" y="-152" width="14" height="26" rx="6" fill={C.mint} {...line} />
      <rect x="46" y="-152" width="14" height="26" rx="6" fill={C.mint} {...line} />
      {/* head */}
      <rect x="-50" y="-176" width="100" height="72" rx="28" fill={C.white} {...line} />
      {/* antenna */}
      {!hat && (<><path d="M0 -176 L0 -192" {...line} fill="none" /><circle cx="0" cy="-198" r="7" fill={C.coral} {...line} /></>)}
      {/* eyes */}
      {mood === 'wink' ? (
        <path d="M-29 -140 Q-20 -148 -11 -140" fill="none" {...line} />
      ) : (
        <><circle cx="-20" cy="-140" r="10" fill={C.ink} /><circle cx={-17 + lx} cy={-143 + ly} r="3.5" fill="#fff" /></>
      )}
      <circle cx="20" cy="-140" r="10" fill={C.ink} /><circle cx={23 + lx} cy={-143 + ly} r="3.5" fill="#fff" />
      {/* cheeks */}
      <circle cx="-36" cy="-124" r="6" fill={C.coral} opacity="0.55" />
      <circle cx="36" cy="-124" r="6" fill={C.coral} opacity="0.55" />
      {/* mouth */}
      {mood === 'wow' && <ellipse cx="0" cy="-120" rx="6" ry="7" fill={C.ink} />}
      {mood === 'think' && <path d="M-7 -119 L8 -121" fill="none" {...line} />}
      {(mood === 'happy' || mood === 'wink') && <path d="M-11 -124 Q0 -111 11 -124" fill="none" {...line} />}
      {hat}
    </g>
  )
}

/** A rounded label chip: the visual for "a token", "a number", "a document". Text is always dark on a light chip. */
export function Chip({ x, y, w, h = 30, text, fill = C.white, mono = true, size = 14, rotate = 0 }: { x: number; y: number; w: number; h?: number; text: string; fill?: string; mono?: boolean; size?: number; rotate?: number }) {
  return (
    <g transform={`translate(${x} ${y}) rotate(${rotate})`}>
      <rect x={-w / 2} y={-h / 2} width={w} height={h} rx={h / 2.6} fill={fill} {...line} />
      <text x="0" y={size * 0.36} textAnchor="middle" fontSize={size} fontWeight={600} style={{ fill: 'var(--il-chip-text)', fontFamily: mono ? 'var(--mono)' : 'var(--sans)' }}>{text}</text>
    </g>
  )
}

export function Sparkle({ x, y, s = 1, fill = C.amber }: { x: number; y: number; s?: number; fill?: string }) {
  return <path transform={`translate(${x} ${y}) scale(${s})`} d="M0 -10 Q2 -2 10 0 Q2 2 0 10 Q-2 2 -10 0 Q-2 -2 0 -10Z" fill={fill} {...line} strokeWidth={2} />
}

export function Cloud({ x, y, s = 1, opacity = 0.9 }: { x: number; y: number; s?: number; opacity?: number }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`} opacity={opacity}>
      <path d="M-34 10 Q-46 10 -44 -2 Q-44 -14 -30 -14 Q-26 -30 -8 -26 Q4 -38 20 -26 Q38 -28 38 -10 Q50 -6 44 6 Q42 12 32 10Z" fill={C.white} {...line} />
    </g>
  )
}

export function Bubble({ x, y, w, h, tail = 'left', children }: { x: number; y: number; w: number; h: number; tail?: 'left' | 'right'; children: ReactNode }) {
  const tx = tail === 'left' ? x + 26 : x + w - 26
  return (
    <g>
      <path d={`M${tx - 10} ${y + h - 2} L${tx + (tail === 'left' ? -16 : 16)} ${y + h + 18} L${tx + 10} ${y + h - 2}Z`} fill={C.white} {...line} />
      <rect x={x} y={y} width={w} height={h} rx={16} fill={C.white} {...line} />
      <rect x={tx - 9} y={y + h - 6} width={18} height={6} fill={C.white} />
      {children}
    </g>
  )
}

export const sceneText = (size = 14, weight = 600) => ({ fontSize: size, fontWeight: weight, style: { fill: 'var(--il-chip-text)', fontFamily: 'var(--sans)' } })
export const freeText = (size = 13, weight = 600) => ({ fontSize: size, fontWeight: weight, style: { fill: C.text, fontFamily: 'var(--sans)' } })
