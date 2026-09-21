// One road, driven twice. On top the forward pass: values, left to right (neutral).
// Underneath the backward pass: gradients, right to left (accent), dropping off d_W at each layer.
// All numbers are the lesson's demo network, computed with src/lib/tinyMlp.ts:
// W1 = [[1, −1], [0.5, 0.25]], W2 = [[0.5, −0.5], [0.5, 1]], biases 0, x = [1, 2], correct class = class 2.
import { ACC, FILL, Figure, INK, LINE, MONO, SOFT } from './diagramKit'

const VALUES: { x: number; name: string; v: string; d?: string; dName?: string }[] = [
  { x: 48, name: 'x', v: '[1, 2]', d: '[0.88, 0.44]', dName: 'd_x' },
  { x: 238, name: 'h', v: '[2, 0]', d: '[0.88, −0.44]', dName: 'd_h' },
  { x: 430, name: 'logits', v: '[1, −1]', d: '[0.88, −0.88]', dName: 'd_logits' },
  { x: 604, name: 'probs', v: '[0.88, 0.12]' },
  { x: 724, name: 'loss', v: '2.13' },
]
const OPS: { x: number; w: number; label: string; sub?: string; dW?: { name: string; m: string[][] } }[] = [
  { x: 143, w: 120, label: 'layer 1', sub: 'multiply, shift, bend', dW: { name: 'd_W1', m: [['0.88', '0'], ['1.76', '0']] } },
  { x: 334, w: 120, label: 'layer 2', sub: 'multiply, shift', dW: { name: 'd_W2', m: [['1.76', '−1.76'], ['0', '0']] } },
  { x: 517, w: 70, label: 'softmax' },
  { x: 672, w: 44, label: '−ln' },
]
const TOP = 104, BOTTOM = 190 // the two lanes

export function ForwardBackward() {
  const uid = 'fwdbwd'
  return (
    <Figure
      id={uid}
      height={312}
      minWidth={700}
      title="The network as one road. Values travel left to right along the top: x, h, logits, probs, loss. Gradients travel right to left along the bottom: d_logits, d_h, d_x, leaving d_W2 and d_W1 at the two layers"
      after={<ol className="sr-only"><li>Forward: x = [1, 2], h = [2, 0], logits = [1, −1], probs = [0.88, 0.12], loss = 2.13</li><li>Backward: d_logits = probs − one_hot = [0.88, −0.88], then d_h = [0.88, −0.44], then d_x = [0.88, 0.44]</li><li>On the way back, layer 2 gets d_W2 = [[1.76, −1.76], [0, 0]] and layer 1 gets d_W1 = [[0.88, 0], [1.76, 0]]</li></ol>}
    >
      {/* ---- forward lane ---- */}
      <text x={16} y={16} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>Forward pass <tspan fontWeight={400} style={{ fill: SOFT }}>values, left to right</tspan></text>
      <path d={`M16 ${TOP} H752`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow)`} />
      {VALUES.map((p) => (
        <g key={p.name} textAnchor="middle">
          <text x={p.x} y={62} fontSize={11.5} style={{ fill: SOFT }}>{p.name}</text>
          <text x={p.x} y={80} fontSize={12.5} fontWeight={700} style={{ fill: INK, ...MONO }}>{p.v}</text>
          <path d={`M${p.x} ${TOP - 5} V${TOP + 5}`} stroke={SOFT} strokeWidth={1.5} />
        </g>
      ))}
      <text x={604} y={44} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>correct class: the 2nd</text>

      {/* ---- the road itself: the same four operations serve both lanes ---- */}
      {OPS.map((o) => (
        <g key={o.label} textAnchor="middle">
          <path d={`M${o.x} ${TOP} V${BOTTOM}`} stroke={LINE} strokeWidth={1.2} />
          <rect x={o.x - o.w / 2} y={124} width={o.w} height={46} rx={8} fill={FILL} stroke={INK} strokeWidth={1.4} />
          <text x={o.x} y={o.sub ? 143 : 152} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>{o.label}</text>
          {o.sub && <text x={o.x} y={159} fontSize={10.5} style={{ fill: SOFT }}>{o.sub}</text>}
        </g>
      ))}

      {/* ---- backward lane ---- */}
      <path d={`M752 ${BOTTOM} H10`} fill="none" stroke={ACC} strokeWidth={2} markerEnd={`url(#${uid}-arrow-acc)`} />
      {VALUES.filter((p) => p.d).map((p) => (
        <g key={p.name} textAnchor="middle">
          <path d={`M${p.x} ${BOTTOM - 5} V${BOTTOM + 5}`} stroke={ACC} strokeWidth={2} />
          <text x={p.x} y={214} fontSize={12.5} fontWeight={700} style={{ fill: ACC, ...MONO }}>{p.d}</text>
          <text x={p.x} y={230} fontSize={11.5} style={{ fill: SOFT, ...MONO }}>{p.dName}</text>
        </g>
      ))}
      <text x={600} y={214} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>starts here:</text>
      <text x={600} y={230} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT, ...MONO }}>probs − one_hot</text>

      {/* each layer's own gradient is dropped off on the way past */}
      {OPS.filter((o) => o.dW).map((o) => (
        <g key={o.label}>
          <path d={`M${o.x} ${BOTTOM} V248`} fill="none" stroke={ACC} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow-acc)`} />
          {o.dW!.m.map((row, i) => row.map((v, j) => (
            <g key={`${i}-${j}`}>
              <rect x={o.x - 46 + j * 47} y={254 + i * 21} width={45} height={19} rx={3} fill={FILL} stroke={LINE} />
              <text x={o.x - 46 + j * 47 + 22.5} y={254 + i * 21 + 13.5} textAnchor="middle" fontSize={11} style={{ fill: INK, ...MONO }}>{v}</text>
            </g>
          )))}
          <text x={o.x - 54} y={279} textAnchor="end" fontSize={11.5} style={{ fill: ACC, ...MONO }}>{o.dW!.name}</text>
        </g>
      ))}
      <text x={744} y={288} textAnchor="end" fontSize={12.5} fontWeight={700} style={{ fill: ACC }}>Backward pass</text>
      <text x={744} y={304} textAnchor="end" fontSize={11.5} style={{ fill: SOFT }}>gradients, right to left, same road</text>
    </Figure>
  )
}
