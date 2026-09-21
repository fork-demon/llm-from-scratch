// The mental model of lesson 0.1 as one picture: the data is drawn at every stage
// (text, pieces, integers, vectors, percentages, one token) and the loop is a real arrow.
// Quiet on purpose: theme colours only, the accent is reserved for the loop.

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const FILL = 'var(--paper-2)'
const ACC = 'var(--accent)'

const STAGES = [
  ['Text', 'what you typed'],
  ['Tokens', 'the tokenizer cuts the text into pieces'],
  ['Token IDs', 'each piece becomes a whole number'],
  ['Embeddings', 'each number becomes a list of numbers'],
  ['Transformer', 'the neural network: one very large calculation'],
  ['Next-token probabilities', 'a percentage for every possible next piece'],
  ['Sampling', 'pick one, weighted by those percentages'],
  ['Next token', 'turned back into text and appended; then the longer text goes round again'],
]

function Caption({ x, y, n, title, lines }: { x: number; y: number; n: number; title: string; lines: string[] }) {
  return (
    <g textAnchor="middle">
      <text x={x} y={y} fontSize={12.5} fontWeight={700} style={{ fill: INK }}>
        <tspan style={{ fill: ACC }}>{n} </tspan>{title}
      </text>
      {lines.map((l, i) => <text key={i} x={x} y={y + 16 + i * 14} fontSize={11.5} style={{ fill: SOFT }}>{l}</text>)}
    </g>
  )
}

function Arrow({ d }: { d: string }) {
  return <path d={d} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#loop-arrow)" />
}

export function LoopDiagram() {
  const tokens: [string, number][] = [['What', 40], ['is', 22], ['a', 18], ['cat', 30], ['?', 16]]
  let tx = 214
  const probs: [string, number][] = [['A', 62], ['The', 21], ['Cats', 9], ['…', 4]]
  // five tiny vectors: opacity stands in for the size of each number
  const vectors = [[0.9, 0.3, 0.6, 0.2], [0.2, 0.8, 0.4, 0.7], [0.5, 0.5, 0.9, 0.1], [0.7, 0.2, 0.3, 0.9], [0.3, 0.6, 0.2, 0.5]]

  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 330" style={{ width: '100%', minWidth: 640, display: 'block' }} role="img" aria-labelledby="loop-title">
          <title id="loop-title">The pipeline from text to the next token, with an arrow looping from the chosen token back to the text</title>
          <defs>
            <marker id="loop-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={SOFT} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
            <marker id="loop-arrow-acc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={ACC} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></marker>
          </defs>

          {/* ---- row 1, left to right ---- */}
          <Caption x={95} y={14} n={1} title="Text" lines={['what you typed']} />
          <rect x={14} y={58} width={162} height={44} rx={8} fill={FILL} stroke={LINE} />
          <text x={26} y={85} fontSize={13} style={{ fill: INK }}>What is a cat?</text>
          <rect x={128} y={68} width={38} height={24} rx={6} fill="none" stroke={ACC} strokeDasharray="4 3" />
          <text x={147} y={85} textAnchor="middle" fontSize={12.5} fontWeight={700} style={{ fill: ACC, fontFamily: 'var(--mono)' }}>A</text>

          <Arrow d="M182 80 H206" />
          <Caption x={285} y={14} n={2} title="Tokens" lines={['the tokenizer cuts', 'the text into pieces']} />
          {tokens.map(([t, w]) => {
            const x = tx
            tx += w + 4
            return (
              <g key={t}>
                <rect x={x} y={67} width={w} height={26} rx={6} fill={FILL} stroke={LINE} />
                <text x={x + w / 2} y={84.5} textAnchor="middle" fontSize={11.5} style={{ fill: INK, fontFamily: 'var(--mono)' }}>{t}</text>
              </g>
            )
          })}

          <Arrow d="M366 80 H388" />
          <Caption x={475} y={14} n={3} title="Token IDs" lines={['each piece becomes', 'a whole number']} />
          <text x={475} y={84.5} textAnchor="middle" fontSize={11} style={{ fill: INK, fontFamily: 'var(--mono)' }}>[2061, 318, 257, 3797, 30]</text>

          <Arrow d="M562 80 H600" />
          <Caption x={665} y={14} n={4} title="Embeddings" lines={['each number becomes', 'a list of numbers']} />
          {vectors.map((col, i) => col.map((v, j) => (
            <rect key={`${i}-${j}`} x={612 + i * 22} y={56 + j * 15} width={18} height={12} rx={3} fill={INK} opacity={0.12 + v * 0.6} />
          )))}

          {/* ---- down into the Transformer ---- */}
          <Arrow d="M665 124 V178" />
          <rect x={590} y={184} width={150} height={74} rx={10} fill={FILL} stroke={INK} strokeWidth={1.5} />
          {[0, 1, 2].map((i) => <rect key={i} x={606} y={198 + i * 17} width={118} height={11} rx={4} fill={INK} opacity={0.16 + i * 0.12} />)}
          <Caption x={665} y={282} n={5} title="Transformer" lines={['the neural network:', 'one very large calculation']} />

          {/* ---- row 2, right to left ---- */}
          <Arrow d="M584 221 H516" />
          {probs.map(([t, p], i) => (
            <g key={t}>
              <text x={392} y={198 + i * 17} textAnchor="end" fontSize={11} style={{ fill: INK, fontFamily: 'var(--mono)' }}>{t}</text>
              <rect x={398} y={189 + i * 17} width={Math.max(4, p * 1.35)} height={11} rx={3} fill={i === 0 ? ACC : INK} opacity={i === 0 ? 1 : 0.3} />
              <text x={404 + Math.max(4, p * 1.35)} y={198 + i * 17} fontSize={10.5} style={{ fill: SOFT, fontFamily: 'var(--mono)' }}>{p}%</text>
            </g>
          ))}
          <Caption x={435} y={282} n={6} title="Next-token probabilities" lines={['a percentage for every', 'possible next piece']} />

          <Arrow d="M352 221 H294" />
          <g transform="translate(250 221) rotate(-10)">
            <rect x={-19} y={-19} width={38} height={38} rx={9} fill={FILL} stroke={INK} strokeWidth={1.5} />
            {[[-8, -8], [0, 0], [8, 8], [-8, 8], [8, -8]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r={2.8} fill={INK} />)}
          </g>
          <Caption x={250} y={282} n={7} title="Sampling" lines={['pick one, weighted', 'by those percentages']} />

          <Arrow d="M212 221 H132" />
          <rect x={66} y={205} width={58} height={32} rx={8} fill={ACC} />
          <text x={95} y={226} textAnchor="middle" fontSize={14} fontWeight={700} style={{ fill: 'var(--on-accent)', fontFamily: 'var(--mono)' }}>A</text>
          <Caption x={95} y={282} n={8} title="Next token" lines={['turned back into text']} />

          {/* ---- the loop ---- */}
          <path d="M95 199 V110" fill="none" stroke={ACC} strokeWidth={2.2} strokeDasharray="6 5" markerEnd="url(#loop-arrow-acc)" />
          <text x={106} y={150} fontSize={12} fontWeight={700} style={{ fill: ACC }}>append it,</text>
          <text x={106} y={165} fontSize={12} fontWeight={700} style={{ fill: ACC }}>then go round again</text>
        </svg>
      </div>
      <ol className="sr-only">{STAGES.map(([t, d]) => <li key={t}>{t}: {d}</li>)}</ol>
    </figure>
  )
}
