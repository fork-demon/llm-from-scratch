// Small presentational building blocks used by every lesson.
import { useId, useState, type ReactNode } from 'react'
import { fmt, type Mat } from '../lib/math'
import { GLOSSARY } from '../data/glossary'
import { ErrorBoundary } from './ErrorBoundary'

/* ---------- callouts ---------- */
type CalloutKind = 'idea' | 'analogy' | 'model' | 'established' | 'research' | 'warn' | 'dev' | 'note'
const CALLOUT_LABEL: Record<CalloutKind, string> = {
  idea: 'Key idea',
  analogy: 'A useful mental model',
  model: 'Simplified mental model',
  established: 'Established',
  research: 'Active research',
  warn: 'Careful',
  dev: 'For developers',
  note: 'Note',
}

/**
 * kind="analogy"     -> an analogy; always labelled so it is never mistaken for the mechanism
 * kind="established" / "model" / "research" -> the three honesty levels (see the why-llms-know lesson)
 * kind="dev"         -> connection to everyday software engineering
 */
export function Callout({ kind = 'note', label, children }: { kind?: CalloutKind; label?: string; children: ReactNode }) {
  return (
    <aside className={`callout ${kind}`}>
      <span className="callout-label">{label ?? CALLOUT_LABEL[kind]}</span>
      {children}
    </aside>
  )
}

export function Card({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <div className="card">
      {title && <h4 style={{ fontSize: 17, marginBottom: 8 }}>{title}</h4>}
      {children}
    </div>
  )
}

/* ---------- jargon: term -> plain English -> tiny example -> formal ---------- */
export function Term({ name, plain, example, formal }: { name: string; plain: ReactNode; example: ReactNode; formal?: ReactNode }) {
  return (
    <div className="term">
      <div className="term-name">{name}</div>
      <div className="term-row"><span>Plain English</span><div>{plain}</div></div>
      <div className="term-row"><span>Tiny example</span><div>{example}</div></div>
      {formal && <div className="term-row"><span>Formally</span><div>{formal}</div></div>}
    </div>
  )
}

/** Inline glossary reference: dotted underline, tooltip, links to the glossary entry. */
export function G({ t, children }: { t: string; children?: ReactNode }) {
  const entry = GLOSSARY.find((g) => g.id === t)
  return (
    <a className="gloss" href={`#/glossary/${t}`} title={entry ? entry.short : undefined}>
      {children ?? entry?.term ?? t}
    </a>
  )
}

/* ---------- optional depth ---------- */
export function DeepDive({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className="deep">
      <summary><span className="deep-tag">Deep dive</span> {title}</summary>
      <div className="details-body">{children}</div>
    </details>
  )
}

export function Expand({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className="deep">
      <summary>{title}</summary>
      <div className="details-body">{children}</div>
    </details>
  )
}

/** Two depth levels for one topic: "Understand" (intuition) and "Go deeper" (maths / implementation). */
export function Depth({ understand, deeper }: { understand: ReactNode; deeper: ReactNode }) {
  return <Tabs tabs={[{ label: 'Understand', content: understand }, { label: 'Go deeper', content: deeper }]} />
}

export function Tabs({ tabs }: { tabs: { label: string; content: ReactNode }[] }) {
  const [active, setActive] = useState(0)
  const id = useId()
  return (
    <div className="tabs">
      <div className="tablist" role="tablist">
        {tabs.map((t, i) => (
          <button
            key={t.label}
            role="tab"
            id={`${id}-t${i}`}
            aria-selected={i === active}
            aria-controls={`${id}-p${i}`}
            tabIndex={i === active ? 0 : -1}
            className="tab"
            onClick={() => setActive(i)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowRight') setActive((active + 1) % tabs.length)
              if (e.key === 'ArrowLeft') setActive((active - 1 + tabs.length) % tabs.length)
            }}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="tabpanel" role="tabpanel" id={`${id}-p${active}`} aria-labelledby={`${id}-t${active}`}>
        {tabs[active].content}
      </div>
    </div>
  )
}

/* ---------- diagrams ---------- */
export interface FlowStep { label: ReactNode; sub?: ReactNode }

/** A simple top-to-bottom (or left-to-right) pipeline. Pass `active` to highlight one step. */
export function Flow({ steps, active, horizontal, onSelect }: { steps: (FlowStep | string)[]; active?: number; horizontal?: boolean; onSelect?: (i: number) => void }) {
  return (
    <div className={`flow${horizontal ? ' horizontal' : ''}`} role={onSelect ? 'group' : 'img'} aria-label={onSelect ? 'Pipeline steps' : `Pipeline: ${steps.map((s) => (typeof s === 'string' ? s : '')).join(' then ')}`}>
      {steps.map((s, i) => {
        const step = typeof s === 'string' ? { label: s } : s
        const cls = `flow-node${i === active ? ' active' : ''}`
        const inner = (<>{step.label}{step.sub && <small>{step.sub}</small>}</>)
        return (
          <div key={i}>
            {i > 0 && <div className="flow-arrow" aria-hidden>{horizontal ? '→' : '↓'}</div>}
            {onSelect ? (
              <button className={cls} aria-pressed={i === active} onClick={() => onSelect(i)}>{inner}</button>
            ) : (
              <div className={cls}>{inner}</div>
            )}
          </div>
        )
      })}
    </div>
  )
}

/** "Why does this exist?": problem -> naive solution -> why it fails -> new idea -> trade-off. */
export function WhyExists({ problem, naive, fails, idea, tradeoff }: { problem: ReactNode; naive: ReactNode; fails: ReactNode; idea: ReactNode; tradeoff: ReactNode }) {
  return (
    <div className="card">
      <h4 style={{ fontSize: 17, marginBottom: 4 }}>Why does this exist?</h4>
      <ol className="ladder">
        <li><b>Problem</b>{problem}</li>
        <li><b>Naive solution</b>{naive}</li>
        <li><b>Why it fails</b>{fails}</li>
        <li><b>New idea</b>{idea}</li>
        <li><b>Trade-off</b>{tradeoff}</li>
      </ol>
    </div>
  )
}

export function ToyVsReal({ toy, real }: { toy: ReactNode; real: ReactNode }) {
  return (
    <div className="toyreal">
      <div><h4>Our teaching version</h4>{toy}</div>
      <div><h4>Production LLMs</h4>{real}</div>
    </div>
  )
}

/* ---------- math ---------- */
/**
 * An equation never appears alone: it comes with a legend that says what each symbol means.
 * `children` is the formula as JSX (use <i>, <sub>, <sup>, and the .q/.k/.v colour classes).
 */
export function Equation({ children, symbols, label }: { children: ReactNode; symbols: [ReactNode, ReactNode][]; label?: string }) {
  return (
    <figure className="eq" aria-label={label ?? 'Equation'} style={{ marginInline: 0 }}>
      <div className="eq-formula">{children}</div>
      <div className="eq-legend">
        {symbols.map(([s, meaning], i) => (
          <div key={i}><b>{s}</b><span>{meaning}</span></div>
        ))}
      </div>
    </figure>
  )
}

/* ---------- numbers ---------- */
export type Tone = 'q' | 'k' | 'v' | 'accent' | 'neutral'
const TONE_VAR: Record<Tone, string> = { q: '--q', k: '--k', v: '--v', accent: '--accent', neutral: '--ink-3' }

/**
 * Show a matrix with optional row/column labels. `heat` tints cells by magnitude
 * (pass heatMax to fix the scale). `editable` turns cells into number inputs.
 */
export function MatrixView({ m, caption, rows, cols, tone = 'neutral', heat, heatMax, digits = 2, highlight, onEdit, onHover }: {
  m: Mat
  caption?: ReactNode
  rows?: string[]
  cols?: string[]
  tone?: Tone
  heat?: boolean
  heatMax?: number
  digits?: number
  highlight?: (i: number, j: number) => boolean
  onEdit?: (i: number, j: number, value: number) => void
  onHover?: (cell: [number, number] | null) => void
}) {
  const max = heatMax ?? Math.max(1e-9, ...m.flat().filter(Number.isFinite).map(Math.abs))
  return (
    <div className="matrix-wrap">
      {caption && <div className="matrix-cap">{caption}</div>}
      <table className="matrix">
        {cols && (
          <thead>
            <tr>{rows && <th />}{cols.map((c, j) => <th key={j} scope="col">{c}</th>)}</tr>
          </thead>
        )}
        <tbody>
          {m.map((row, i) => (
            <tr key={i}>
              {rows && <th scope="row">{rows[i]}</th>}
              {row.map((x, j) => {
                const a = heat && Number.isFinite(x) ? Math.min(1, Math.abs(x) / max) : 0
                const style = heat ? { background: `color-mix(in srgb, var(${TONE_VAR[tone]}) ${Math.round(a * 62)}%, var(--paper-2))` } : undefined
                return (
                  <td
                    key={j}
                    style={style}
                    className={highlight?.(i, j) ? 'hot' : undefined}
                    onMouseEnter={onHover ? () => onHover([i, j]) : undefined}
                    onMouseLeave={onHover ? () => onHover(null) : undefined}
                  >
                    {onEdit ? (
                      <NumCell value={Number.isFinite(x) ? x : 0} label={`${rows?.[i] ?? `row ${i + 1}`}, ${cols?.[j] ?? `column ${j + 1}`}`} onChange={(v) => onEdit(i, j, v)} />
                    ) : (
                      fmt(x, digits)
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** A numeric cell that lets you type "-", "." or an empty string without the value snapping back. */
function NumCell({ value, label, onChange }: { value: number; label: string; onChange: (v: number) => void }) {
  const [draft, setDraft] = useState<string | null>(null)
  return (
    <input
      type="text"
      inputMode="decimal"
      aria-label={label}
      value={draft ?? String(value)}
      onFocus={(e) => e.target.select()}
      onChange={(e) => {
        const t = e.target.value.replace('−', '-')
        setDraft(t)
        const n = Number(t)
        if (t.trim() !== '' && Number.isFinite(n)) onChange(n)
      }}
      onBlur={() => setDraft(null)}
    />
  )
}

/** Horizontal probability bars. `dim` marks entries that were cut off (e.g. by top-k). */
export function Bars({ items, max, percent = true, digits = 1 }: { items: { label: string; value: number; dim?: boolean; tone?: Tone }[]; max?: number; percent?: boolean; digits?: number }) {
  const top = max ?? Math.max(1e-9, ...items.map((i) => Math.abs(i.value)))
  return (
    <div className="bars" role="list">
      {items.map((it, i) => (
        <div key={i} className={`bar${it.dim ? ' dim' : ''}`} role="listitem">
          <span className="bar-label" title={it.label}>{it.label}</span>
          <span className="bar-track"><span className="bar-fill" style={{ width: `${Math.max(0, (it.value / top) * 100)}%`, background: it.tone && !it.dim ? `var(${TONE_VAR[it.tone]})` : undefined }} /></span>
          <span className="bar-val">{percent ? `${(it.value * 100).toFixed(digits)}%` : fmt(it.value, 2)}</span>
        </div>
      ))}
    </div>
  )
}

export function Slider({ label, value, min, max, step = 0.1, onChange, format }: { label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; format?: (v: number) => string }) {
  const id = useId()
  return (
    <div className="control">
      <label htmlFor={id}><span>{label}</span><output>{format ? format(value) : value}</output></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  )
}

/** The frame around every interactive. `goal` tells the learner what to try. */
export function Lab({ title, goal, children }: { title: string; goal?: ReactNode; children: ReactNode }) {
  return (
    <section className="lab" aria-label={`Interactive: ${title}`}>
      <div className="lab-head"><span className="lab-tag">Try it</span><span className="lab-title">{title}</span></div>
      {goal && <div className="lab-goal">{goal}</div>}
      <div className="lab-body"><ErrorBoundary what={title}>{children}</ErrorBoundary></div>
    </section>
  )
}
