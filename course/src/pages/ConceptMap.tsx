import { useState } from 'react'
import { lessonById } from '../data/curriculum'
import { useProgress } from '../lib/progress'

// A hand-laid-out concept graph. Each node opens its lesson.
interface Node { id: string; label: string; lesson: string; x: number; y: number; tone?: 'q' | 'k' | 'v' }
const N: Node[] = [
  { id: 'llm', label: 'LLM', lesson: 'prompt-to-answer', x: 500, y: 40 },
  { id: 'tok', label: 'Tokenizer', lesson: 'tokenization', x: 130, y: 150 },
  { id: 'emb', label: 'Embeddings', lesson: 'embeddings', x: 310, y: 150 },
  { id: 'tf', label: 'Transformer', lesson: 'transformer-block', x: 500, y: 150 },
  { id: 'out', label: 'Logits → softmax', lesson: 'softmax', x: 690, y: 150 },
  { id: 'samp', label: 'Sampling', lesson: 'inference', x: 870, y: 150 },
  { id: 'att', label: 'Attention', lesson: 'attention', x: 380, y: 270 },
  { id: 'mlp', label: 'MLP', lesson: 'neurons', x: 560, y: 270 },
  { id: 'res', label: 'Residual + norm', lesson: 'transformer-block', x: 730, y: 270 },
  { id: 'q', label: 'Query', lesson: 'attention', x: 230, y: 380, tone: 'q' },
  { id: 'k', label: 'Key', lesson: 'attention', x: 350, y: 380, tone: 'k' },
  { id: 'v', label: 'Value', lesson: 'attention', x: 470, y: 380, tone: 'v' },
  { id: 'mask', label: 'Causal mask', lesson: 'masks-and-heads', x: 110, y: 270 },
  { id: 'heads', label: 'Multiple heads', lesson: 'masks-and-heads', x: 240, y: 270 },
  { id: 'dot', label: 'Dot product', lesson: 'vectors', x: 290, y: 480 },
  { id: 'mat', label: 'Matrix multiply', lesson: 'matrices', x: 470, y: 480 },
  { id: 'train', label: 'Training', lesson: 'training-gpt', x: 760, y: 380 },
  { id: 'gd', label: 'Gradient descent', lesson: 'gradient-descent', x: 670, y: 480 },
  { id: 'bp', label: 'Backpropagation', lesson: 'backprop', x: 860, y: 480 },
  { id: 'deriv', label: 'Derivatives', lesson: 'derivatives', x: 760, y: 570 },
  { id: 'lm', label: 'Next-token prediction', lesson: 'next-token', x: 600, y: 380 },
  { id: 'kv', label: 'KV cache', lesson: 'inference', x: 910, y: 270 },
  { id: 'rag', label: 'RAG', lesson: 'rag', x: 130, y: 570 },
  { id: 'ft', label: 'Fine-tuning', lesson: 'fine-tuning', x: 310, y: 570 },
  { id: 'agent', label: 'Agents', lesson: 'agents', x: 490, y: 570 },
  { id: 'gpt', label: 'GPT', lesson: 'build-gpt', x: 500, y: 95 },
]
const E: [string, string][] = [
  ['llm', 'gpt'], ['gpt', 'tok'], ['gpt', 'emb'], ['gpt', 'tf'], ['gpt', 'out'], ['out', 'samp'], ['tf', 'att'], ['tf', 'mlp'], ['tf', 'res'],
  ['att', 'q'], ['att', 'k'], ['att', 'v'], ['att', 'mask'], ['att', 'heads'], ['q', 'dot'], ['k', 'dot'], ['dot', 'mat'], ['v', 'mat'],
  ['out', 'lm'], ['lm', 'train'], ['train', 'gd'], ['train', 'bp'], ['gd', 'deriv'], ['bp', 'deriv'], ['samp', 'kv'],
  ['emb', 'dot'], ['rag', 'dot'], ['ft', 'gd'], ['agent', 'mat'],
]

export function ConceptMapPage() {
  const progress = useProgress()
  const [hover, setHover] = useState<string | null>(null)
  const byId = Object.fromEntries(N.map((n) => [n.id, n]))
  const linked = (id: string) => hover === id || E.some(([a, b]) => (a === hover && b === id) || (b === hover && a === id))
  // the three "around the model" nodes attach conceptually, not structurally
  const soft = new Set(['rag-dot', 'ft-gd', 'agent-mat'])

  return (
    <div>
      <h1 className="lesson-title">Concept map</h1>
      <p className="lesson-question">How the ideas connect. Select any node to open its lesson. Green means you have completed it.</p>
      <div className="card" style={{ overflowX: 'auto', padding: 8 }}>
        <svg viewBox="0 0 1000 620" style={{ width: '100%', minWidth: 760 }} role="group" aria-label="Concept map of the course">
          {E.map(([a, b]) => {
            const on = hover && (a === hover || b === hover)
            return <line key={`${a}-${b}`} x1={byId[a].x} y1={byId[a].y} x2={byId[b].x} y2={byId[b].y} stroke={on ? 'var(--accent)' : 'var(--rule-strong)'} strokeWidth={on ? 2.2 : 1.2} strokeDasharray={soft.has(`${a}-${b}`) ? '4 4' : undefined} />
          })}
          {N.map((n) => {
            const lesson = lessonById(n.lesson)
            const done = !!progress.completed[n.lesson]
            const w = Math.max(70, n.label.length * 8.2 + 22)
            const dim = hover && !linked(n.id)
            const stroke = n.tone ? `var(--${n.tone})` : done ? 'var(--good)' : 'var(--rule-strong)'
            return (
              <a key={n.id} href={`#/lesson/${n.lesson}`} aria-label={`${n.label}: opens lesson ${lesson?.code} ${lesson?.title}${done ? ' (completed)' : ''}`}
                onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(n.id)} onBlur={() => setHover(null)} style={{ opacity: dim ? 0.35 : 1, transition: 'opacity .15s' }}>
                <rect x={n.x - w / 2} y={n.y - 16} width={w} height={32} rx={16} fill={n.tone ? `var(--${n.tone}-soft)` : done ? 'var(--good-soft)' : 'var(--card)'} stroke={hover === n.id ? 'var(--accent)' : stroke} strokeWidth={hover === n.id ? 2.5 : 1.5} />
                <text x={n.x} y={n.y + 5} textAnchor="middle" fontSize={14} fontWeight={n.id === 'llm' || n.id === 'gpt' ? 700 : 500}>{n.label}</text>
              </a>
            )
          })}
          <text x={20} y={608} fontSize={12} style={{ fill: 'var(--ink-3)' }}>Dashed lines: RAG, fine-tuning and agents sit around the model and reuse these ideas.</text>
        </svg>
      </div>
      <p className="muted" style={{ fontSize: 15 }}>{hover ? `${byId[hover].label} → Lesson ${lessonById(byId[hover].lesson)?.code}: ${lessonById(byId[hover].lesson)?.title}` : 'Hover or tab through the nodes to see what connects to what.'}</p>
    </div>
  )
}
