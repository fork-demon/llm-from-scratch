// The "one picture" of the course as an illustrated strip: a numbered tile per stage,
// each with a tiny drawing in the same cartoon style as the scenes.
import type { ReactNode } from 'react'
import { C, outlined } from './kit'

const LINE = 'var(--il-line)'
const o = { ...outlined, stroke: LINE, strokeWidth: 2.6 }
const STAGES: { label: string; note: string; lesson: string; tint: string; icon: ReactNode }[] = [
  { label: 'Text', note: '“What is a cat?”', lesson: 'prompt-to-answer', tint: C.white, icon: <><rect x="6" y="9" width="36" height="24" rx="8" fill={C.white} {...o} /><path d="M14 32 l-3 9 l11 -9" fill={C.white} {...o} /><path d="M14 18h20M14 25h13" {...o} /></> },
  { label: 'Tokens', note: 'What · is · a · cat · ?', lesson: 'tokenization', tint: C.amber, icon: <><rect x="4" y="16" width="17" height="16" rx="5" fill={C.amber} {...o} /><rect x="25" y="16" width="19" height="16" rx="5" fill={C.mint} {...o} /><path d="M23 8v32" {...o} strokeDasharray="3 4" stroke="var(--il-d)" /></> },
  { label: 'Token IDs', note: '2061, 318, 257 …', lesson: 'tokenization', tint: C.sky, icon: <><rect x="6" y="10" width="36" height="28" rx="8" fill={C.sky} {...o} /><path d="M19 17l-3 14M29 17l-3 14M14 21h18M13 27h18" {...o} /></> },
  { label: 'Embeddings', note: 'a vector per token', lesson: 'embeddings', tint: C.mint, icon: <><path d="M8 40V8M8 40h34" {...o} /><circle cx="19" cy="28" r="4" fill={C.coral} {...o} /><circle cx="25" cy="22" r="4" fill={C.coral} {...o} /><circle cx="36" cy="13" r="4" fill={C.sky} {...o} /><circle cx="33" cy="32" r="4" fill={C.amber} {...o} /></> },
  { label: 'Transformer blocks', note: 'attention + MLP, × N', lesson: 'transformer-block', tint: C.white, icon: <><rect x="8" y="28" width="32" height="10" rx="4" fill={C.white} {...o} /><rect x="8" y="17" width="32" height="10" rx="4" fill={C.amber} {...o} /><rect x="8" y="6" width="32" height="10" rx="4" fill={C.coral} {...o} /></> },
  { label: 'Logits', note: 'a raw score per token', lesson: 'softmax', tint: C.amber, icon: <><path d="M6 24h36" {...o} /><rect x="9" y="10" width="7" height="14" rx="2" fill={C.mint} {...o} /><rect x="20" y="16" width="7" height="8" rx="2" fill={C.amber} {...o} /><rect x="31" y="24" width="7" height="12" rx="2" fill={C.coral} {...o} /></> },
  { label: 'Probabilities', note: 'softmax: adds up to 100%', lesson: 'softmax', tint: C.mint, icon: <><circle cx="24" cy="24" r="17" fill={C.white} {...o} /><path d="M24 24V7a17 17 0 0 1 15 25z" fill={C.mint} {...o} /><path d="M24 24l15 8a17 17 0 0 1-22 7z" fill={C.amber} {...o} /></> },
  { label: 'Sample', note: 'roll the weighted die', lesson: 'inference', tint: C.coral, icon: <><rect x="9" y="9" width="30" height="30" rx="8" fill={C.white} {...o} transform="rotate(-10 24 24)" /><circle cx="18" cy="19" r="2.6" fill={LINE} /><circle cx="24" cy="24" r="2.6" fill={LINE} /><circle cx="30" cy="30" r="2.6" fill={LINE} /></> },
  { label: 'Repeat', note: 'append it, go again', lesson: 'next-token', tint: C.sky, icon: <><path d="M38 24a14 14 0 1 1-5-10.7" fill="none" {...o} /><path d="M35 5l-1 10 10-2" fill="none" {...o} /></> },
]

export function PipelineStrip() {
  return (
    <ol className="pipe" aria-label="The pipeline from text to the next token">
      {STAGES.map((s, i) => (
        <li key={s.label}>
          <a className="pipe-tile" href={`#/lesson/${s.lesson}`}>
            <span className="pipe-n">{i + 1}</span>
            <svg viewBox="0 0 48 48" aria-hidden>{s.icon}</svg>
            <span className="pipe-label">{s.label}</span>
            <span className="pipe-note">{s.note}</span>
          </a>
        </li>
      ))}
    </ol>
  )
}
