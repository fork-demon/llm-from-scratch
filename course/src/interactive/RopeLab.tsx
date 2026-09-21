// RoPE in two dimensions: rotate the query and the key by (position x theta) and watch
// the score depend only on the DIFFERENCE between the two positions.
import { useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import { additiveScore, degToRad, rope2d, ropeScore, scoreByOffset } from '../lib/rope'

const Q0 = [2, 1]
const K0 = [1, 2]

export function RopeLab() {
  const [m, setM] = useState(3) // query position
  const [n, setN] = useState(7) // key position
  const [shift, setShift] = useState(0)
  const [deg, setDeg] = useState(20)

  const theta = degToRad(deg)
  const pm = m + shift
  const pn = n + shift
  const q = rope2d(Q0, pm, theta)
  const k = rope2d(K0, pn, theta)
  const score = ropeScore(Q0, K0, pm, pn, theta)
  const scoreUnshifted = ropeScore(Q0, K0, m, n, theta)
  const addScore = additiveScore(Q0, K0, pm, pn, theta)
  const addUnshifted = additiveScore(Q0, K0, m, n, theta)
  const offset = n - m
  const curve = scoreByOffset(Q0, K0, theta, -12, 12)

  // vector picture
  const S = 300
  const c = S / 2
  const unit = 46
  const px = (v: number[]) => [c + v[0] * unit, c - v[1] * unit]
  const arrow = (v: number[], colour: string, dashed?: boolean) => {
    const [x, y] = px(v)
    return <line x1={c} y1={c} x2={x} y2={y} stroke={colour} strokeWidth={dashed ? 1.5 : 3.5} strokeDasharray={dashed ? '4 4' : undefined} strokeLinecap="round" opacity={dashed ? 0.55 : 1} />
  }
  const [qx, qy] = px(q)
  const [kx, ky] = px(k)

  // curve picture
  const CW = 340
  const CH = 214
  const padL = 34
  const padB = 42
  const maxAbs = 5.2
  const cx = (o: number) => padL + ((o + 12) / 24) * (CW - padL - 8)
  const cy = (s: number) => 10 + ((maxAbs - s) / (2 * maxAbs)) * (CH - padB - 10)
  const path = curve.map((p, i) => `${i === 0 ? 'M' : 'L'} ${cx(p.offset).toFixed(1)} ${cy(p.score).toFixed(1)}`).join(' ')
  const inRange = offset >= -12 && offset <= 12

  return (
    <Lab
      title="RoPE: position as rotation"
      goal={<>The <span className="q">query</span> sits at one position and the <span className="k">key</span> at another. Each is rotated by <span className="mono">position × θ</span>. First change the two positions and watch the score move. Then drag <b>“shift both”</b>: both positions change, the distance between them does not. What happens to the score?</>}
    >
      <div className="controls">
        <Slider label="Query position m" value={m} min={0} max={20} step={1} onChange={setM} />
        <Slider label="Key position n" value={n} min={0} max={20} step={1} onChange={setN} />
        <Slider label="Shift both positions by" value={shift} min={0} max={100} step={1} onChange={setShift} format={(v) => `+${v}`} />
        <Slider label="θ (rotation per position)" value={deg} min={1} max={60} step={1} onChange={setDeg} format={(v) => `${v}°`} />
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div>
          <svg viewBox={`0 0 ${S} ${S}`} style={{ width: '100%', maxWidth: S, display: 'block', margin: '0 auto' }} role="img" aria-label={`Query rotated to ${fmt(q[0])}, ${fmt(q[1])}. Key rotated to ${fmt(k[0])}, ${fmt(k[1])}. Score ${fmt(score)}.`}>
            <line className="gridline" x1={0} y1={c} x2={S} y2={c} />
            <line className="gridline" x1={c} y1={0} x2={c} y2={S} />
            <circle cx={c} cy={c} r={Math.sqrt(5) * unit} fill="none" className="gridline" strokeDasharray="2 5" />
            {arrow(Q0, 'var(--q)', true)}
            {arrow(K0, 'var(--k)', true)}
            {arrow(q, 'var(--q)')}
            {arrow(k, 'var(--k)')}
            <circle cx={qx} cy={qy} r={5} fill="var(--q)" />
            <rect x={kx - 5} y={ky - 5} width={10} height={10} fill="var(--k)" />
            <text x={qx + (qx > c ? 8 : -8)} y={qy + (qy > c ? 16 : -8)} textAnchor={qx > c ? 'start' : 'end'} fontSize={12} style={{ fill: 'var(--q)' }}>q at {pm}</text>
            <text x={kx + (kx > c ? 8 : -8)} y={ky + (ky > c ? 16 : -8)} textAnchor={kx > c ? 'start' : 'end'} fontSize={12} style={{ fill: 'var(--k)' }}>k at {pn}</text>
          </svg>
          <p className="lab-note" style={{ textAlign: 'center' }}>Dashed: the vectors before rotation, <span className="q mono">q = [2, 1]</span> (circle) and <span className="k mono">k = [1, 2]</span> (square). Solid: after RoPE. Rotation never changes a vector’s length, so both stay on the dotted circle.</p>
        </div>

        <div>
          <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', maxWidth: CW, display: 'block', margin: '0 auto' }} role="img" aria-label={`Score as a function of the offset n minus m. Current offset ${offset}, score ${fmt(score)}.`}>
            <line className="axis" x1={padL} y1={cy(0)} x2={CW - 8} y2={cy(0)} />
            <line className="axis" x1={cx(0)} y1={10} x2={cx(0)} y2={CH - padB} />
            {[-4, 4].map((s) => <g key={s}><line className="gridline" x1={padL} y1={cy(s)} x2={CW - 8} y2={cy(s)} /><text x={padL - 4} y={cy(s) + 4} textAnchor="end" fontSize={10}>{s > 0 ? s : `−${-s}`}</text></g>)}
            <text x={padL - 4} y={cy(0) + 4} textAnchor="end" fontSize={10}>0</text>
            {[-12, -6, 6, 12].map((o) => <text key={o} x={cx(o)} y={CH - padB + 14} textAnchor="middle" fontSize={10}>{o > 0 ? `+${o}` : `−${-o}`}</text>)}
            <text x={(CW + padL) / 2} y={CH - 6} textAnchor="middle" fontSize={10.5}>offset = key position − query position</text>
            <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} />
            {curve.map((p) => <circle key={p.offset} cx={cx(p.offset)} cy={cy(p.score)} r={2} fill="var(--accent)" />)}
            {inRange && <circle cx={cx(offset)} cy={cy(score)} r={7} fill="none" stroke="var(--ink)" strokeWidth={2} />}
          </svg>
          <p className="lab-note" style={{ textAlign: 'center' }}>The whole behaviour of this query/key pair is one curve over the <em>offset</em>. The ring marks where you are. “Shift both” never moves the ring.</p>
        </div>
      </div>

      <div className="readout" aria-live="polite">
        <span>positions: <b className="q">m = {pm}</b>, <b className="k">n = {pn}</b></span>
        <span>offset n − m = <b>{offset}</b></span>
        <span>rotation between them = {offset} × {deg}° = <b>{offset * deg}°</b></span>
        <span>RoPE score q·k = <b>{fmt(score, 3)}</b></span>
      </div>

      <div className="table-scroll" style={{ marginTop: 10 }}>
        <table className="plain" style={{ fontSize: 14 }}>
          <thead><tr><th>scheme</th><th>score at ({m}, {n})</th><th>score at ({pm}, {pn})</th><th>changed by the shift?</th></tr></thead>
          <tbody>
            <tr>
              <td><b>RoPE</b>: rotate q and k</td>
              <td className="mono">{fmt(scoreUnshifted, 3)}</td>
              <td className="mono">{fmt(score, 3)}</td>
              <td>{Math.abs(score - scoreUnshifted) < 1e-6 ? 'no: identical' : 'yes'}</td>
            </tr>
            <tr>
              <td><b>Additive</b>: add a position vector to q and k</td>
              <td className="mono">{fmt(addUnshifted, 3)}</td>
              <td className="mono">{fmt(addScore, 3)}</td>
              <td>{Math.abs(addScore - addUnshifted) < 1e-6 ? (shift === 0 ? 'shift is 0' : 'no (coincidence at this shift)') : `yes: by ${fmt(addScore - addUnshifted, 3)}`}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="lab-note">
        <b>What is simplified:</b> real models have <span className="mono">head_dim</span> numbers per head (for example 128), split into 64 pairs. Every pair is rotated like this, each with its own θ, from fast to very slow. The additive row is a toy with a tidy circular position vector. The learned position table in our tiny GPT has no such structure, and has no row at all for positions it never trained on.
      </p>
    </Lab>
  )
}
