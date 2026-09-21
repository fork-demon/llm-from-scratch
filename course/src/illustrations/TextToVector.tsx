// "cat" -> token -> token ID -> one row of the embedding table -> that row as a vector.
// Shared by the tokenization lesson (focus="token") and the embeddings lesson (focus="lookup"):
// the accent sits on whichever half the lesson is about.
import { ACC, Caption, FILL, Figure, INK, LINE, MONO, ON_ACC, SOFT } from './diagramKit'

// opacity stands in for the size of each number; one list per drawn table row
const SHADES = [
  [0.5, 0.2, 0.7, 0.3, 0.6, 0.15],
  [0.25, 0.65, 0.35, 0.8, 0.2, 0.5],
  [0.7, 0.4, 0.15, 0.55, 0.3, 0.75],
  [0.3, 0.85, 0.6, 0.2, 0.7, 0.4],
  [0.6, 0.3, 0.45, 0.7, 0.15, 0.55],
]

export function TextToVector({ focus, word = 'cat', id, idNote, vector }: { focus: 'token' | 'lookup'; word?: string; id: number; idNote?: string; vector: string }) {
  const tok = focus === 'token'
  const uid = `ttv-${focus}`
  const rowY = (i: number) => 58 + i * 19
  const MID = rowY(4) + 7 // the highlighted row: everything on the main line is centred here
  const rows: (number | null)[] = [0, 1, null, id - 1, id, id + 1, null]
  let shade = 0

  return (
    <Figure
      id={uid}
      height={232}
      title={`The word ${word} becomes a token, the token becomes the ID ${id}, and the ID selects row ${id} of the embedding table, which is the vector ${vector}`}
      after={<ol className="sr-only"><li>Text: {word}</li><li>Token: a piece of text from a fixed list</li><li>Token ID: {id}, its position in the list</li><li>Embedding table: row {id} is looked up</li><li>Vector: {vector}</li></ol>}
    >
      <Caption x={58} y={14} title="Text" lines={['what you typed']} />
      <rect x={20} y={MID - 20} width={76} height={40} rx={8} fill={FILL} stroke={LINE} />
      <text x={58} y={MID + 5} textAnchor="middle" fontSize={14} style={{ fill: INK }}>“{word}”</text>

      <path d={`M102 ${MID} H134`} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd={`url(#${uid}-arrow)`} />
      <Caption x={170} y={14} title="Token" accent={tok} lines={['a piece from', 'a fixed list']} />
      <rect x={142} y={MID - 14} width={56} height={28} rx={7} fill={tok ? ACC : FILL} stroke={tok ? ACC : LINE} />
      <text x={170} y={MID + 4.5} textAnchor="middle" fontSize={13} fontWeight={tok ? 700 : 400} style={{ fill: tok ? ON_ACC : INK, ...MONO }}>{word}</text>

      <path d={`M204 ${MID} H240`} fill="none" stroke={tok ? ACC : SOFT} strokeWidth={tok ? 2 : 1.5} markerEnd={`url(#${uid}-arrow${tok ? '-acc' : ''})`} />
      <Caption x={282} y={14} title="Token ID" accent={tok} lines={['its position', 'in that list']} />
      <text x={282} y={MID + 7} textAnchor="middle" fontSize={20} fontWeight={700} style={{ fill: tok ? ACC : INK, ...MONO }}>{id}</text>
      {idNote && <text x={282} y={MID + 28} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>{idNote}</text>}

      {/* the ID is the row number */}
      <path d={`M322 ${MID} H372`} fill="none" stroke={tok ? SOFT : ACC} strokeWidth={tok ? 1.5 : 2} markerEnd={`url(#${uid}-arrow${tok ? '' : '-acc'})`} />
      <Caption x={489} y={14} title="Embedding table" accent={!tok} lines={tok ? ['one row of numbers per token', '(next lesson)'] : ['the ID is the row number']} />
      {rows.map((r, i) => {
        if (r === null) return <text key={i} x={489} y={rowY(i) + 12} textAnchor="middle" fontSize={13} style={{ fill: SOFT }}>⋮</text>
        const hot = r === id
        const cells = SHADES[shade++]
        return (
          <g key={i}>
            <text x={412} y={rowY(i) + 11} textAnchor="end" fontSize={11} fontWeight={hot ? 700 : 400} style={{ fill: hot ? (tok ? INK : ACC) : SOFT, ...MONO }}>{r}</text>
            {cells.map((v, j) => <rect key={j} x={420 + j * 23} y={rowY(i)} width={20} height={14} rx={3} fill={hot && !tok ? ACC : INK} opacity={0.14 + v * (hot ? 0.8 : 0.45)} />)}
            {hot && <rect x={416} y={rowY(i) - 3.5} width={146} height={21} rx={5} fill="none" stroke={tok ? INK : ACC} strokeWidth={tok ? 1.2 : 2} />}
          </g>
        )
      })}
      <text x={489} y={212} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>rows: every token in the vocabulary</text>

      {/* the row, pulled out */}
      <path d={`M568 ${MID} H600`} fill="none" stroke={tok ? SOFT : ACC} strokeWidth={tok ? 1.5 : 2} markerEnd={`url(#${uid}-arrow${tok ? '' : '-acc'})`} />
      <Caption x={678} y={14} title="Vector" accent={!tok} lines={tok ? ['what the model works with'] : ['that row, pulled out:', 'what the model works with']} />
      {SHADES[3].map((v, j) => <rect key={j} x={610 + j * 23} y={MID - 7} width={20} height={14} rx={3} fill={tok ? INK : ACC} opacity={0.14 + v * 0.8} />)}
      <text x={678} y={MID + 30} textAnchor="middle" fontSize={12} style={{ fill: INK, ...MONO }}>{vector}</text>
      {!tok && <text x={678} y={MID + 48} textAnchor="middle" fontSize={11} style={{ fill: SOFT, ...MONO }}>= E[{id}]</text>}
    </Figure>
  )
}
