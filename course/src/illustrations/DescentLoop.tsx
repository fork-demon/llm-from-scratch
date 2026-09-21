// Gradient descent as a picture: a ball on a loss curve, the slope under it, one step downhill,
// and the trail of earlier positions. Beside it the four-step cycle, drawn as a cycle.
// The accent marks the update: the step on the curve and the formula that makes it.
import { ACC, Caption, Figure, INK, LINE, MONO, SOFT } from './diagramKit'

// the bowl in screen units (y grows downward): lowest point at (X0, Y0)
const X0 = 262, Y0 = 226, C = 0.0041
const yOf = (x: number) => Y0 - C * (x - X0) ** 2
const slopeAt = (x: number) => -2 * C * (x - X0)

// every step covers a quarter of the remaining distance: step = lr × slope, so steps shrink as the slope flattens
const TRAIL = [50, 103, 143]
const NOW = 173
const NEXT = 195
const R = 7

/** Centre of a ball of radius R resting on the curve at x (pushed out along the curve's normal). */
const ballAt = (x: number) => {
  const s = slopeAt(x)
  const n = Math.hypot(1, s)
  return { cx: x + (R * s) / n, cy: yOf(x) - R / n }
}

// the cycle on the right: an ellipse with the four labels sitting on it
const CX = 585, CY = 150, RX = 112, RY = 92
const onEllipse = (deg: number) => {
  const a = (deg * Math.PI) / 180
  return `${(CX + RX * Math.cos(a)).toFixed(1)} ${(CY + RY * Math.sin(a)).toFixed(1)}`
}
const arc = (from: number, to: number) => `M${onEllipse(from)} A${RX} ${RY} 0 0 1 ${onEllipse(to)}`

export function DescentLoop() {
  const uid = 'descent'
  const pts: string[] = []
  for (let x = 48; x <= 356; x += 4) pts.push(`${x},${yOf(x).toFixed(1)}`)
  const now = ballAt(NOW)
  const next = ballAt(NEXT)
  const s = slopeAt(NOW)

  return (
    <Figure
      id={uid}
      height={300}
      title="A ball rolling down a bowl-shaped loss curve in shrinking steps, next to the cycle predict, loss, gradient, update that produces each step"
      after={<ol className="sr-only"><li>Predict: run the model</li><li>Loss: one number, how wrong?</li><li>Gradient: a slope per parameter</li><li>Update: step against the slope, w = w − lr × slope</li><li>Repeat</li></ol>}
    >
      {/* ---- the loss curve ---- */}
      <path d="M40 30 V250 H372" fill="none" stroke={LINE} strokeWidth={1.5} />
      <text x={48} y={26} fontSize={11.5} style={{ fill: SOFT }}>loss</text>
      <text x={372} y={268} textAnchor="end" fontSize={11.5} style={{ fill: SOFT }}>w, one parameter</text>
      <polyline points={pts.join(' ')} fill="none" stroke={INK} strokeWidth={2} strokeLinecap="round" />
      <text x={X0} y={Y0 - 14} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>lowest loss</text>

      {/* earlier positions: the trail makes "repeat" visible */}
      {TRAIL.map((x, i) => {
        const b = ballAt(x)
        return <circle key={x} cx={b.cx} cy={b.cy} r={R} fill={INK} opacity={0.22 + i * 0.1} />
      })}
      <text x={128} y={62} fontSize={11.5} style={{ fill: SOFT }}>earlier steps: big where it is</text>
      <text x={128} y={76} fontSize={11.5} style={{ fill: SOFT }}>steep, smaller as it flattens</text>

      {/* the slope under the ball */}
      <path d={`M${NOW - 58} ${yOf(NOW) - 58 * s} L${NOW + 58} ${yOf(NOW) + 58 * s}`} stroke={SOFT} strokeWidth={1.5} strokeDasharray="5 4" />
      <text x={NOW - 56} y={yOf(NOW) - 58 * s + 30} textAnchor="end" fontSize={11.5} style={{ fill: SOFT }}>the slope</text>
      <text x={NOW - 56} y={yOf(NOW) - 58 * s + 44} textAnchor="end" fontSize={11.5} style={{ fill: SOFT }}>here</text>

      {/* the ball, and the step it is about to take */}
      <circle cx={next.cx} cy={next.cy} r={R} fill="none" stroke={ACC} strokeWidth={1.5} strokeDasharray="3 3" />
      <circle cx={now.cx} cy={now.cy} r={R} fill={INK} />
      <path d={`M${now.cx + 2} ${now.cy - 13} Q${(now.cx + next.cx) / 2 + 10} ${now.cy - 40} ${next.cx + 3} ${next.cy - 13}`} fill="none" stroke={ACC} strokeWidth={2.2} markerEnd={`url(#${uid}-arrow-acc)`} />
      <text x={now.cx + 26} y={now.cy - 30} fontSize={12} fontWeight={700} style={{ fill: ACC }}>one step downhill</text>

      {/* ---- the cycle ---- */}
      {[[-62, -24], [24, 62], [118, 152], [208, 242]].map(([a, b]) => (
        <path key={a} d={arc(a, b)} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow)`} />
      ))}
      <Caption x={CX} y={CY - RY + 2} n={1} title="predict" lines={['run the model']} />
      <Caption x={CX + RX} y={CY - 12} n={2} title="loss" lines={['one number:', 'how wrong?']} />
      <Caption x={CX} y={CY + RY - 8} n={3} title="gradient" lines={['a slope per parameter']} />
      <Caption x={CX - RX} y={CY - 18} n={4} title="update" lines={['step against the slope']} />
      <text x={CX - RX} y={CY + 17} textAnchor="middle" fontSize={11.5} fontWeight={700} style={{ fill: ACC, ...MONO }}>w = w − lr × slope</text>
      <text x={CX} y={CY - 2} textAnchor="middle" fontSize={12.5} fontWeight={700} style={{ fill: INK }}>repeat</text>
      <text x={CX} y={CY + 14} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>until the loss</text>
      <text x={CX} y={CY + 28} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>stops falling</text>
    </Figure>
  )
}
