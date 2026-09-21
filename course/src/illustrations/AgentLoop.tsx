// The agent loop as a real cycle across a border. Left of the border: the model, which only ever
// writes text. Right of it: your code, which keeps the transcript, parses the text, runs the tool
// and appends the result. The accent follows the one thing that closes the loop: RESULT: 165
// being appended to the transcript. The exit leaves the cycle when the text starts with ANSWER:.
// Every string is the real one from src/lib/agent.ts (the oncall run, second iteration).

const INK = 'var(--ink)'
const SOFT = 'var(--ink-3)'
const LINE = 'var(--rule-strong)'
const FILL = 'var(--paper-2)'
const ACC = 'var(--accent)'
const MONO = 'var(--mono)'

const TRANSCRIPT: [string, boolean][] = [
  ['To use a tool reply EXACTLY: …', true],
  ['USER QUESTION: What is 23*7 plus…', false],
  ['TOOL: search_docs', false],
  ['RESULT: The oncall rotation has 4…', false],
  ['TOOL: calculator', false],
]

const STEPS = [
  'Your code sends the whole transcript to the model, on every call.',
  'The model only writes text: TOOL: calculator, ARGS: {"expression": "23*7 + 4"}.',
  'Your code parses that text.',
  'If it starts with TOOL:, your code runs the real function calculator("23*7 + 4"), which returns 165.',
  'Your code appends RESULT: 165 to the transcript and goes round again.',
  'If the text starts with ANSWER:, or the step limit is hit, the loop stops and the answer goes to the user.',
]

function Arrow({ d }: { d: string }) {
  return <path d={d} fill="none" stroke={SOFT} strokeWidth={1.5} markerEnd="url(#agent-arrow)" />
}

export function AgentLoop() {
  return (
    <figure style={{ margin: '24px 0' }}>
      <div className="table-scroll">
        <svg viewBox="0 0 760 378" style={{ width: '100%', minWidth: 680, display: 'block' }} role="img" aria-labelledby="agent-loop-title">
          <title id="agent-loop-title">The agent loop: your code sends the transcript to the model, the model writes a tool request as text, your code parses it, runs the calculator, appends RESULT: 165 to the transcript and calls the model again, until the text starts with ANSWER:</title>
          <defs>
            <marker id="agent-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={SOFT} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></marker>
            <marker id="agent-arrow-acc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke={ACC} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></marker>
          </defs>

          {/* ---- the border between the two territories ---- */}
          <path d="M276 8 V370" stroke={LINE} strokeWidth={1.2} strokeDasharray="2 5" />
          <text x={14} y={16} fontSize={13} fontWeight={700} style={{ fill: INK }}>The model</text>
          <text x={14} y={33} fontSize={11.5} style={{ fill: SOFT }}>only ever writes text</text>
          <text x={292} y={16} fontSize={13} fontWeight={700} style={{ fill: INK }}>Your code</text>
          <text x={292} y={33} fontSize={11.5} style={{ fill: SOFT }}>does things: keeps the transcript,</text>
          <text x={292} y={48} fontSize={11.5} style={{ fill: SOFT }}>parses, runs tools</text>

          {/* ---- the transcript (context), growing ---- */}
          <rect x={500} y={44} width={246} height={140} rx={8} fill={FILL} stroke={LINE} />
          <text x={512} y={62} fontSize={11.5} fontWeight={700} style={{ fill: INK }}>context: the transcript so far</text>
          {TRANSCRIPT.map(([t, soft], i) => (
            <text key={i} x={512} y={82 + i * 16} fontSize={10.5} style={{ fill: soft ? SOFT : INK, fontFamily: MONO }}>{t}</text>
          ))}
          <rect x={506} y={151} width={234} height={19} rx={5} fill={ACC} fillOpacity={0.12} stroke={ACC} strokeWidth={1.2} />
          <text x={512} y={164.5} fontSize={10.5} fontWeight={700} style={{ fill: ACC, fontFamily: MONO }}>RESULT: 165</text>

          {/* ---- transcript -> model ---- */}
          <Arrow d="M494 100 H246" />
          <text x={388} y={80} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>the whole transcript,</text>
          <text x={388} y={93} textAnchor="middle" fontSize={11} style={{ fill: SOFT }}>re-sent on every call</text>

          <rect x={40} y={66} width={200} height={68} rx={10} fill={FILL} stroke={INK} strokeWidth={1.5} />
          {[0, 1, 2].map((i) => <rect key={i} x={56} y={79 + i * 16} width={168} height={10} rx={4} fill={INK} opacity={0.16 + i * 0.12} />)}

          <text x={40} y={151} fontSize={10.5} style={{ fill: SOFT, fontFamily: MONO }}>model(context)</text>

          {/* ---- model -> text ---- */}
          <Arrow d="M140 134 V208" />
          <text x={150} y={176} fontSize={11} style={{ fill: SOFT }}>writes the next text</text>
          <rect x={14} y={214} width={244} height={72} rx={8} fill={FILL} stroke={LINE} />
          <text x={24} y={234} fontSize={10.5} style={{ fill: SOFT, fontFamily: MONO }}>Thought: … Now compute 23*7 + 4.</text>
          <text x={24} y={252} fontSize={10.5} fontWeight={700} style={{ fill: INK, fontFamily: MONO }}>TOOL: calculator</text>
          <text x={24} y={270} fontSize={10.5} fontWeight={700} style={{ fill: INK, fontFamily: MONO }}>{'ARGS: {"expression": "23*7 + 4"}'}</text>

          {/* ---- text -> parse -> tool ---- */}
          <Arrow d="M258 250 H292" />
          <rect x={298} y={230} width={84} height={40} rx={8} fill={FILL} stroke={INK} strokeWidth={1.5} />
          <text x={340} y={254.5} textAnchor="middle" fontSize={11.5} style={{ fill: INK, fontFamily: MONO }}>parse(text)</text>

          <Arrow d="M382 250 H454" />
          <text x={418} y={241} textAnchor="middle" fontSize={10.5} style={{ fill: SOFT, fontFamily: MONO }}>TOOL:</text>
          <rect x={460} y={226} width={160} height={48} rx={8} fill={FILL} stroke={INK} strokeWidth={1.5} />
          <text x={540} y={246} textAnchor="middle" fontSize={10.5} style={{ fill: INK, fontFamily: MONO }}>calculator("23*7 + 4")</text>
          <text x={540} y={263} textAnchor="middle" fontSize={10.5} style={{ fill: SOFT }}>runs for real, returns 165</text>

          <Arrow d="M620 250 H644" />
          <rect x={650} y={237} width={96} height={26} rx={6} fill={ACC} />
          <text x={698} y={254.5} textAnchor="middle" fontSize={11} fontWeight={700} style={{ fill: 'var(--on-accent)', fontFamily: MONO }}>RESULT: 165</text>

          {/* ---- the loop closes: append to the transcript ---- */}
          <path d="M698 237 V190" fill="none" stroke={ACC} strokeWidth={2.2} strokeDasharray="6 5" markerEnd="url(#agent-arrow-acc)" />
          <text x={688} y={205} textAnchor="end" fontSize={12} fontWeight={700} style={{ fill: ACC }}>append it, then call the model again</text>

          {/* ---- the exit ---- */}
          <Arrow d="M340 270 V322" />
          <text x={350} y={294} fontSize={10.5} style={{ fill: SOFT, fontFamily: MONO }}>ANSWER:</text>
          <text x={402} y={294} fontSize={11} style={{ fill: SOFT }}>(or the step limit is hit): the loop stops</text>
          <rect x={298} y={328} width={448} height={40} rx={8} fill={FILL} stroke={LINE} />
          <text x={308} y={345} fontSize={10.5} style={{ fill: INK, fontFamily: MONO }}>ANSWER: 165 -- that's 23*7 (161) plus the 4 oncall engineers.</text>
          <text x={308} y={360} fontSize={10.5} style={{ fill: SOFT }}>returned to the user; no more model calls</text>
        </svg>
      </div>
      <ol className="sr-only">{STEPS.map((s) => <li key={s}>{s}</li>)}</ol>
    </figure>
  )
}
