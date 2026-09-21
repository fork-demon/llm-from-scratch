// RAG as two lanes that meet at one table. Top lane (once): oncall.md is cut into overlapping
// chunks, each chunk becomes a vector, each (vector, text) pair becomes a row. Bottom lane (per
// question): the question becomes a vector, is scored against every row, and the top 3 rows are
// pasted into the prompt. The accent follows those 3 chunks from the table into the prompt and
// out as the citation, because the lesson's thesis is "only the prompt changes".
// Texts and similarities are the real output of src/lib/rag.ts for the first demo question.

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const FILL = 'var(--paper-2)'
const ACC = 'var(--accent)'
const MONO = 'var(--mono)'

const SENTENCES = [
  'The oncall rotation changes every…',
  'Primary oncall must acknowledge…',
  'Secondary oncall is paged if the…',
  'After an incident the oncall…',
  'Postmortems are blameless and…',
]

// one row per chunk of oncall.md (2 sentences, overlap 1); sim is null outside the top 3
const ROWS: { text: string; vec: number[]; sim: string | null }[] = [
  { text: 'The oncall rotation changes… Primary oncall must acknowledge…', vec: [0.8, 0.2, 0.9, 0.3, 0.1], sim: '0.47' },
  { text: 'Primary oncall must acknowledge pages within five minutes. …', vec: [0.7, 0.1, 0.8, 0.5, 0.2], sim: '0.45' },
  { text: 'Secondary oncall is paged if the primary does not respond. …', vec: [0.3, 0.6, 0.4, 0.7, 0.2], sim: '0.09' },
  { text: 'After an incident the oncall engineer writes the postmortem. …', vec: [0.1, 0.9, 0.2, 0.4, 0.7], sim: null },
  { text: 'Postmortems are blameless and due within three business days.', vec: [0.2, 0.7, 0.1, 0.2, 0.9], sim: null },
]
const QUESTION_VEC = [0.9, 0.1, 0.8, 0.4, 0.1]

const PASTED = ['[1] (oncall.md) The oncall rotation…', '[2] (oncall.md) Primary oncall must…', '[3] (oncall.md) Secondary oncall is…']

const STEPS = [
  'Once, ahead of time: oncall.md is cut into overlapping two-sentence chunks.',
  'Each chunk is embedded into a vector, and the vector is stored in a table next to the chunk text. The whole wiki gives 19 rows.',
  'For every question: “how quickly must I acknowledge pages” is embedded with the same embedder.',
  'The question vector is scored against every row. The top 3 rows score 0.47, 0.45 and 0.09.',
  'Those 3 chunks are pasted into a prompt under “Context:”, followed by “Question:”.',
  'The LLM, unchanged, reads the prompt and answers “Primary oncall must acknowledge pages within five minutes” with the citation source 1: oncall.md.',
]

function Cells({ x, y, values, accent }: { x: number; y: number; values: number[]; accent?: boolean }) {
  return <>{values.map((v, j) => <rect key={j} x={x + j * 12} y={y} width={10} height={12} rx={2.5} fill={accent ? ACC : INK} opacity={0.14 + v * 0.62} />)}</>
}

export function RagTwoLanes() {
  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 404" style={{ width: '100%', minWidth: 680, display: 'block' }} role="img" aria-labelledby="rag-lanes-title">
          <title id="rag-lanes-title">Two lanes sharing one vector store: documents are chunked and embedded once; every question is embedded, matched to the top 3 rows, and those chunks are pasted into the prompt of an unchanged LLM</title>
          <defs>
            <marker id="rag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={SOFT} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
            <marker id="rag-arrow-acc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={ACC} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></marker>
          </defs>

          {/* ================= lane 1: once ================= */}
          <text x={14} y={16} fontSize={13} fontWeight={700} style={{ fill: INK }}>Once, ahead of time</text>
          <text x={258} y={36} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>chunk, then embed</text>
          <text x={746} y={36} textAnchor="end" fontSize={11} style={{ fill: SOFT }}>vector store: one row per chunk, 19 rows for the whole wiki</text>

          {/* the document, cut between sentences */}
          <rect x={14} y={44} width={200} height={156} rx={8} fill={FILL} stroke={LINE} />
          <text x={26} y={63} fontSize={11.5} fontWeight={700} style={{ fill: INK, fontFamily: MONO }}>oncall.md</text>
          {SENTENCES.map((s, i) => (
            <g key={i}>
              <text x={26} y={86 + i * 22} fontSize={10.5} style={{ fill: INK }}>{s}</text>
              {i < 4 && <path d={`M22 ${93 + i * 22} H206`} stroke={SOFT} strokeWidth={1} strokeDasharray="3 3" />}
            </g>
          ))}

          {/* overlapping chunks: each bracket covers two sentences and shares one with its neighbour */}
          {ROWS.map((_, i) => {
            const x = i % 2 === 0 ? 220 : 231
            const y1 = 76 + i * 22
            const y2 = i < 4 ? y1 + 36 : y1 + 14
            const yc = (y1 + y2) / 2
            return (
              <g key={i}>
                <path d={`M${x} ${y1} h6 V${y2} h-6`} fill="none" stroke={INK} strokeWidth={1.3} />
                <path d={`M${x + 8} ${yc} L294 ${75 + i * 22}`} fill="none" stroke={SOFT} strokeWidth={1.1} markerEnd="url(#rag-arrow)" />
              </g>
            )
          })}

          {/* the table */}
          <rect x={300} y={44} width={446} height={156} rx={8} fill={FILL} stroke={LINE} />
          <text x={308} y={58} fontSize={10.5} style={{ fill: SOFT }}>vector</text>
          <text x={376} y={58} fontSize={10.5} style={{ fill: SOFT }}>chunk text</text>
          <text x={738} y={58} textAnchor="end" fontSize={10.5} style={{ fill: SOFT }}>similarity</text>
          {ROWS.map((r, i) => {
            const y = 64 + i * 22
            const hit = r.sim !== null
            return (
              <g key={i}>
                {hit && <rect x={304} y={y + 1} width={438} height={20} rx={5} fill={ACC} fillOpacity={0.12} stroke={ACC} strokeWidth={1.2} />}
                <Cells x={308} y={y + 5} values={r.vec} />
                <text x={376} y={y + 15} fontSize={10.5} style={{ fill: INK }}>{r.text}</text>
                {hit && <text x={738} y={y + 15} textAnchor="end" fontSize={10.5} fontWeight={700} style={{ fill: ACC, fontFamily: MONO }}>{r.sim}</text>}
              </g>
            )
          })}
          <text x={376} y={191} fontSize={10.5} style={{ fill: SOFT }}>… 14 more rows from onboarding.md, deploy-policy.md, expenses.md</text>

          {/* ================= lane 2: every question ================= */}
          <text x={14} y={228} fontSize={13} fontWeight={700} style={{ fill: INK }}>For every question</text>

          <rect x={14} y={240} width={200} height={50} rx={8} fill={FILL} stroke={LINE} />
          <text x={26} y={261} fontSize={12.5} style={{ fill: INK }}>“how quickly must I</text>
          <text x={26} y={278} fontSize={12.5} style={{ fill: INK }}>acknowledge pages”</text>

          <path d="M50 290 V312" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#rag-arrow)" />
          <text x={60} y={306} fontSize={11} style={{ fill: SOFT }}>same embedder</text>
          <Cells x={22} y={318} values={QUESTION_VEC} />
          <text x={22} y={348} fontSize={11} style={{ fill: SOFT }}>the question’s vector</text>

          {/* question vector goes up to the vector column of the table */}
          <path d="M88 324 H336 V206" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#rag-arrow)" />
          <text x={328} y={228} textAnchor="end" fontSize={11} style={{ fill: SOFT }}>score every row</text>

          {/* the top 3 rows come down into the prompt */}
          <path d="M466 202 V236" fill="none" stroke={ACC} strokeWidth={2.2} strokeDasharray="6 5" markerEnd="url(#rag-arrow-acc)" />
          <text x={476} y={224} fontSize={12} fontWeight={700} style={{ fill: ACC }}>paste the top 3 into the prompt</text>

          {/* the prompt page */}
          <rect x={352} y={240} width={228} height={154} rx={8} fill={FILL} stroke={LINE} />
          <text x={362} y={258} fontSize={10.5} style={{ fill: SOFT }}>Answer using ONLY the context below.</text>
          <text x={362} y={277} fontSize={11} fontWeight={700} style={{ fill: INK }}>Context:</text>
          {PASTED.map((t, i) => (
            <g key={i}>
              <rect x={360} y={283 + i * 22} width={212} height={19} rx={5} fill={ACC} fillOpacity={0.12} stroke={ACC} strokeWidth={1.2} />
              <text x={366} y={296.5 + i * 22} fontSize={10.5} style={{ fill: INK }}>{t}</text>
            </g>
          ))}
          <text x={362} y={366} fontSize={11} style={{ fill: INK }}><tspan fontWeight={700}>Question:</tspan> how quickly must I</text>
          <text x={362} y={381} fontSize={11} style={{ fill: INK }}>acknowledge pages</text>

          {/* the model, untouched */}
          <path d="M580 270 H600" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#rag-arrow)" />
          <rect x={606} y={244} width={140} height={52} rx={8} fill={FILL} stroke={INK} strokeWidth={1.5} />
          <text x={676} y={266} textAnchor="middle" fontSize={12.5} fontWeight={700} style={{ fill: INK }}>LLM, unchanged</text>
          <text x={676} y={283} textAnchor="middle" fontSize={10.5} style={{ fill: SOFT }}>same weights as before</text>

          <path d="M676 296 V314" fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#rag-arrow)" />
          <rect x={606} y={320} width={140} height={74} rx={8} fill={FILL} stroke={LINE} />
          <text x={616} y={337} fontSize={10.5} style={{ fill: INK }}>Primary oncall must</text>
          <text x={616} y={351} fontSize={10.5} style={{ fill: INK }}>acknowledge pages</text>
          <text x={616} y={365} fontSize={10.5} style={{ fill: INK }}>within five minutes.</text>
          <rect x={611} y={371} width={130} height={17} rx={5} fill="none" stroke={ACC} strokeWidth={1.2} />
          <text x={676} y={383} textAnchor="middle" fontSize={10.5} style={{ fill: ACC, fontFamily: MONO }}>source 1: oncall.md</text>
        </svg>
      </div>
      <ol className="sr-only">{STEPS.map((s) => <li key={s}>{s}</li>)}</ol>
    </figure>
  )
}
