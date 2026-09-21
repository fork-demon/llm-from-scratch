// One neuron, with the lesson's own numbers: inputs [1, 2], weights [1.0, 0.5], bias 0.
// Weighted sum 1×1.0 + 2×0.5 + 0 = 2.0, then the ReLU hinge: relu(2.0) = 2.0 (and relu(−0.5) = 0, shown faintly).
// The accent marks the new ingredient of the lesson: the bend.
import { ACC, Caption, FILL, Figure, INK, LINE, MONO, SOFT } from './diagramKit'

const X = [1, 2]
const W = [1.0, 0.5]
const B = 0
const Z = X.reduce((s, x, i) => s + x * W[i], B) // 2.0
const OUT = Math.max(0, Z)

// the little ReLU graph: origin and scale in screen units
const OX = 512, OY = 168, U = 24

export function NeuronAnatomy() {
  const uid = 'neuron'
  const inY = [84, 170]
  const node = { x: 286, y: 127, r: 30 }
  return (
    <Figure
      id={uid}
      height={262}
      title={`One neuron: inputs ${X.join(' and ')} are multiplied by weights ${W.join(' and ')} and added to the bias ${B}, giving ${Z.toFixed(1)}; the ReLU hinge turns that into the output ${OUT.toFixed(1)}`}
      after={<ol className="sr-only"><li>Inputs: x₁ = 1, x₂ = 2</li><li>Weighted sum: w·x + b = 1×1.0 + 2×0.5 + 0 = 2.0</li><li>Activation: relu(2.0) = 2.0; a negative sum such as −0.5 would become 0</li><li>Output: one number, 2.0</li></ol>}
    >
      <Caption x={62} y={14} n={1} title="Inputs" />
      <Caption x={node.x} y={14} n={2} title="Weighted sum" lines={['w · x + b']} />
      <Caption x={OX + 14} y={14} n={3} title="Activation" accent lines={['bend it: relu(z) = max(0, z)']} />
      <Caption x={706} y={14} n={4} title="Output" lines={['one number']} />

      {/* inputs, and the weights written on the wires */}
      {X.map((x, i) => (
        <g key={i}>
          <line x1={100} y1={inY[i]} x2={node.x - node.r * 0.92} y2={node.y + (i === 0 ? -12 : 12)} stroke={SOFT} strokeWidth={1.5} />
          <rect x={24} y={inY[i] - 17} width={76} height={34} rx={8} fill={FILL} stroke={LINE} />
          <text x={62} y={inY[i] + 5} textAnchor="middle" fontSize={13.5} style={{ fill: INK }}>x{i === 0 ? '₁' : '₂'} = {x}</text>
          <text x={176} y={i === 0 ? 86 : 180} textAnchor="middle" fontSize={12} style={{ fill: INK, ...MONO }}>× {W[i].toFixed(1)}</text>
          <text x={176} y={i === 0 ? 71 : 195} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>weight</text>
        </g>
      ))}

      {/* the bias comes in from below */}
      <line x1={node.x} y1={204} x2={node.x} y2={node.y + node.r} stroke={SOFT} strokeWidth={1.5} />
      <text x={node.x} y={220} textAnchor="middle" fontSize={12} style={{ fill: INK, ...MONO }}>+ {B}</text>
      <text x={node.x + 24} y={220} fontSize={11} style={{ fill: SOFT }}>bias</text>

      <circle cx={node.x} cy={node.y} r={node.r} fill={FILL} stroke={INK} strokeWidth={1.5} />
      <text x={node.x} y={node.y + 7} textAnchor="middle" fontSize={20} style={{ fill: INK }}>Σ</text>
      <text x={node.x} y={246} textAnchor="middle" fontSize={11.5} style={{ fill: INK, ...MONO }}>1×1.0 + 2×0.5 + {B} = {Z.toFixed(1)}</text>

      <path d={`M${node.x + node.r + 4} ${node.y} H428`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow)`} />
      <text x={374} y={node.y - 9} textAnchor="middle" fontSize={12} style={{ fill: INK, ...MONO }}>z = {Z.toFixed(1)}</text>

      {/* the hinge */}
      <path d={`M${OX - 72} ${OY} H${OX + 92} M${OX} ${OY + 22} V${OY - 96}`} fill="none" stroke={LINE} strokeWidth={1.5} />
      <text x={OX + 94} y={OY + 15} textAnchor="end" fontSize={11} style={{ fill: SOFT }}>z</text>
      <path d={`M${OX - 72} ${OY} H${OX} L${OX + 3.6 * U} ${OY - 3.6 * U}`} fill="none" stroke={ACC} strokeWidth={2.4} strokeLinejoin="round" strokeLinecap="round" />
      <path d={`M${OX + Z * U} ${OY} V${OY - OUT * U} H${OX}`} fill="none" stroke={SOFT} strokeWidth={1.2} strokeDasharray="3 3" />
      <circle cx={OX + Z * U} cy={OY - OUT * U} r={5} fill={ACC} />
      <text x={OX + Z * U} y={OY + 15} textAnchor="middle" fontSize={11} style={{ fill: INK, ...MONO }}>{Z.toFixed(1)}</text>
      <text x={OX - 6} y={OY - OUT * U + 4} textAnchor="end" fontSize={11} style={{ fill: INK, ...MONO }}>{OUT.toFixed(1)}</text>
      {/* the lesson's other example: a negative sum is flattened to 0 */}
      <circle cx={OX - 0.5 * U} cy={OY} r={4} fill="none" stroke={SOFT} strokeWidth={1.5} />
      <text x={OX - 0.5 * U - 4} y={OY + 15} textAnchor="middle" fontSize={11} style={{ fill: SOFT, ...MONO }}>−0.5</text>
      <text x={OX + 14} y={224} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>negative in → 0 out; positive passes unchanged</text>

      <path d={`M${OX + 100} ${node.y} H664`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow)`} />
      <text x={706} y={node.y + 8} textAnchor="middle" fontSize={22} fontWeight={700} style={{ fill: INK, ...MONO }}>{OUT.toFixed(1)}</text>
    </Figure>
  )
}
