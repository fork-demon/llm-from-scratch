// The fixed lesson rhythm. Every lesson is a <Lesson id="..."> containing these
// sections, in this order. Skip a section only when it truly does not apply.
//
//  <Why>          1  Why are we learning this?
//  <Problem>      2  The problem (what the previous idea could not do)
//  <MentalModel>  3  Mental model (labelled analogy)
//  <TryIt>        4  Interactive example
//  <Numbers>      5  Let's see the numbers
//  <TheMath>      6  The math
//  <CodeIt>       7  Let's code it
//  <BreakIt>      8  Break it
//  <Exercises>    9  Exercises
//  <CheckYourself>10 Check yourself
//  <Remember>     11 What you should remember
//  <RealLLM>      12 Where this appears in a real LLM
import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { LESSONS, LLM_TREE, lessonById, sourceUrl, type FlatLesson, type TreeNode } from '../data/curriculum'
import { completeLesson, updateProgress, useProgress } from '../lib/progress'
import { Quiz, type QuizQuestion } from './exercise'
import { ErrorBoundary } from './ErrorBoundary'
import { TokenTitle } from './TokenTitle'

const LessonCtx = createContext<FlatLesson | null>(null)
export const useLesson = () => useContext(LessonCtx)

/* ---------- "Where are we?" ---------- */
const pathTo = (node: TreeNode, id: string, trail: string[] = []): string[] | null => {
  const here = [...trail, node.id]
  if (node.id === id) return here
  for (const c of node.children ?? []) {
    const p = pathTo(c, id, here)
    if (p) return p
  }
  return null
}

export function WhereAreWe({ here }: { here: string }) {
  const [open, setOpen] = useState(false)
  const path = pathTo(LLM_TREE, here) ?? ['llm']
  const lines: ReactNode[] = []
  const walk = (node: TreeNode, prefix: string, isLast: boolean, depth: number) => {
    const branch = depth === 0 ? '' : isLast ? '└── ' : '├── '
    const isHere = node.id === here
    const onPath = path.includes(node.id)
    lines.push(
      <span key={node.id}>
        {prefix}{branch}
        <span className={isHere ? 'here' : onPath ? 'on-path' : undefined}>{node.label}{isHere ? '  ← you are here' : ''}</span>
        {'\n'}
      </span>,
    )
    const kids = node.children ?? []
    kids.forEach((k, i) => walk(k, depth === 0 ? '' : prefix + (isLast ? '    ' : '│   '), i === kids.length - 1, depth + 1))
  }
  walk(LLM_TREE, '', true, 0)
  const labels: Record<string, string> = {}
  const collect = (n: TreeNode) => { labels[n.id] = n.label; n.children?.forEach(collect) }
  collect(LLM_TREE)
  return (
    <>
      <div className="where" aria-label={`Where are we: ${path.map((id) => labels[id]).join(', then ')}`}>
        <span className="where-label">You are here:</span>
        {path.map((id, i) => (
          <span key={id} style={{ display: 'contents' }}>
            {i > 0 && <span className="where-sep" aria-hidden>›</span>}
            <span className={`where-step${id === here ? ' here' : ''}`}>{labels[id]}</span>
          </span>
        ))}
        <button className="where-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? 'Hide the map' : 'Show the whole map'}</button>
      </div>
      {open && (
        <div className="where-map">
          <pre className="tree">{lines}</pre>
        </div>
      )}
    </>
  )
}

/* ---------- one section at a time ----------
   A lesson's sections are grouped into five phases of the learning loop and shown one at a time.
   The rail at the top shows the phases, with one dot per section; the pager at the bottom moves on.
   Hidden sections stay in the page (hidden="until-found"), so their state survives and the
   browser's find-in-page can still reach them: a match opens its section. */
type PhaseKey = 'idea' | 'explore' | 'build' | 'practice' | 'recap'
// `tight` is the label on a phone, where all five must fit on one line
const PHASES: { key: PhaseKey; label: string; tight: string; ids: string[] }[] = [
  { key: 'idea', label: 'The idea', tight: 'Idea', ids: ['why', 'problem', 'mental-model'] },
  { key: 'explore', label: 'Explore', tight: 'Explore', ids: ['try-it', 'numbers'] },
  { key: 'build', label: 'Math and code', tight: 'Build', ids: ['math', 'code'] },
  { key: 'practice', label: 'Practice', tight: 'Practice', ids: ['break-it', 'exercises', 'check'] },
  { key: 'recap', label: 'Recap', tight: 'Recap', ids: ['remember', 'real-llm', 'before-moving-on'] },
]
const SHORT: Record<string, string> = {
  why: 'Why', problem: 'The problem', 'mental-model': 'Mental model', 'try-it': 'Try it', numbers: 'The numbers',
  math: 'The math', code: 'The code', 'break-it': 'Break it', exercises: 'Exercises', check: 'Check yourself',
  remember: 'Remember', 'real-llm': 'Real LLMs', 'before-moving-on': 'Checkpoint',
}
const phaseOfId = (id: string) => PHASES.find((p) => p.ids.includes(id))?.key

interface SectionInfo { id: string; title: string; short: string; phase: PhaseKey }

const lastKey = (id: string) => `llm-fp-section-${id}`
const seenKey = (id: string) => `llm-fp-seen-${id}`
const readSeen = (id: string): string[] => { try { return JSON.parse(localStorage.getItem(seenKey(id)) ?? '[]') as string[] } catch { return [] } }

/** `#/lesson/<id>/<section>` or `#/lesson/<id>#<element id>` */
const lessonHash = () => {
  const raw = window.location.hash.replace(/^#/, '')
  const cut = raw.indexOf('#')
  const path = cut >= 0 ? raw.slice(0, cut) : raw
  const [, page = '', lesson = '', sub = ''] = path.split('/')
  return { page, lesson: decodeURIComponent(lesson), sub: decodeURIComponent(sub), anchor: cut >= 0 ? decodeURIComponent(raw.slice(cut + 1)) : '' }
}

const reducedMotion = () => !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
/** Keys that belong to the focused control, not to the page. */
const ownsArrows = (el: EventTarget | null) => {
  const t = el as HTMLElement | null
  if (!t || !t.closest) return false
  if (/^(input|textarea|select)$/i.test(t.tagName) || t.isContentEditable) return true
  return !!t.closest('[role="tablist"],[role="radiogroup"],[role="slider"],[role="listbox"],[role="menu"],[role="grid"],.py-editor')
}

function useSections(lessonId: string, body: React.RefObject<HTMLDivElement | null>, anchor: React.RefObject<HTMLDivElement | null>) {
  const [sections, setSections] = useState<SectionInfo[]>([])
  const [active, setActive] = useState<string | null>(null)
  const [seen, setSeen] = useState<string[]>(() => readSeen(lessonId))
  const pending = useRef<{ scroll: boolean; focus: boolean; target?: string } | null>(null)

  // find the sections once the lesson has rendered, and pick the first one to show
  useLayoutEffect(() => {
    const els = [...(body.current?.querySelectorAll<HTMLElement>(':scope > section.section[id]') ?? [])]
    let prev: PhaseKey = 'idea'
    const found = els.map((el): SectionInfo => {
      const phase = phaseOfId(el.id) ?? (el.dataset.phase as PhaseKey | undefined) ?? prev
      prev = phase
      const title = el.querySelector('h2')?.textContent?.trim() || el.id
      return { id: el.id, title, short: SHORT[el.id] ?? title.replace(/^\d+\.\s*/, ''), phase }
    })
    setSections(found)
    if (!found.length) return
    const { sub, anchor: a } = lessonHash()
    const target = a ? document.getElementById(a) : null
    const holder = target?.closest<HTMLElement>('section.section[id]')
    const start = found.find((f) => f.id === sub)?.id ?? (holder && found.some((f) => f.id === holder.id) ? holder.id : found[0].id)
    if (holder && a) pending.current = { scroll: true, focus: false, target: a }
    setActive(start)
  }, [lessonId, body])

  // show the active section, hide the rest, and let find-in-page open a hidden one
  useLayoutEffect(() => {
    if (!active) return
    const els = [...(body.current?.querySelectorAll<HTMLElement>(':scope > section.section[id]') ?? [])]
    const offs = els.map((el) => {
      if (el.id === active) el.removeAttribute('hidden')
      else el.setAttribute('hidden', 'until-found')
      const on = () => setActive(el.id)
      el.addEventListener('beforematch', on)
      return () => el.removeEventListener('beforematch', on)
    })
    const p = pending.current
    pending.current = null
    if (p) {
      const el = p.target ? document.getElementById(p.target) : null
      const a = anchor.current
      if (el) el.scrollIntoView?.({ block: 'center' })
      else if (p.scroll && a && a.getBoundingClientRect().top < 0) a.scrollIntoView?.({ block: 'start', behavior: reducedMotion() ? 'auto' : 'smooth' })
      if (p.focus) document.getElementById(active)?.querySelector<HTMLElement>('h2')?.focus({ preventScroll: true })
    }
    return () => offs.forEach((off) => off())
  }, [active, sections, body, anchor])

  // remember it: the address says which section is open, and the rail shows which ones you have read
  useEffect(() => {
    if (!active || !sections.length) return
    const url = `#/lesson/${lessonId}/${active}`
    if (window.location.hash !== url) window.history.replaceState(window.history.state, '', url)
    setSeen((s) => {
      if (s.includes(active)) return s
      const next = [...s, active]
      try { localStorage.setItem(seenKey(lessonId), JSON.stringify(next)); localStorage.setItem(lastKey(lessonId), active) } catch { /* storage unavailable */ }
      return next
    })
    try { localStorage.setItem(lastKey(lessonId), active) } catch { /* storage unavailable */ }
  }, [active, sections, lessonId])

  const go = useCallback((id: string, how: { scroll?: boolean; focus?: boolean } = {}) => {
    pending.current = { scroll: how.scroll ?? true, focus: how.focus ?? false }
    setActive(id)
  }, [])

  // a link to another section of this lesson (or an element in it) while the lesson is open
  useEffect(() => {
    const on = () => {
      const h = lessonHash()
      if (h.page !== 'lesson' || h.lesson !== lessonId) return
      if (h.anchor) {
        const holder = document.getElementById(h.anchor)?.closest<HTMLElement>('section.section[id]')
        if (holder) { pending.current = { scroll: true, focus: false, target: h.anchor }; setActive(holder.id) }
      } else if (h.sub && sections.some((s) => s.id === h.sub)) go(h.sub)
    }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [lessonId, sections, go])

  // ← and → move between sections (unless the focused control uses them itself)
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') || e.metaKey || e.ctrlKey || e.altKey || e.shiftKey || ownsArrows(e.target)) return
      const i = sections.findIndex((s) => s.id === active)
      const n = sections[i + (e.key === 'ArrowRight' ? 1 : -1)]
      if (i < 0 || !n) return
      e.preventDefault()
      go(n.id, { focus: true })
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [sections, active, go])

  return { sections, active, seen, go }
}

function LessonRail({ sections, active, seen, go }: { sections: SectionInfo[]; active: string | null; seen: string[]; go: (id: string, how?: { scroll?: boolean; focus?: boolean }) => void }) {
  if (sections.length < 2) return null
  const current = sections.find((s) => s.id === active)
  const phases = PHASES.map((p) => {
    const list = sections.filter((s) => s.phase === p.key)
    const label = p.key === 'build' && list.some((s) => s.id !== 'math' && s.id !== 'code') ? 'Build' : p.label
    return { ...p, label, list }
  }).filter((p) => p.list.length)
  const n = sections.findIndex((s) => s.id === active) + 1
  return (
    <nav className="lesson-rail" aria-label="Sections of this lesson">
      {phases.map((p) => {
        const here = p.list.some((s) => s.id === active)
        return (
          <div key={p.key} className={`lesson-phase${here ? ' is-here' : ''}`}>
            <button className="lesson-phase-name" onClick={() => go(p.list[0].id)} aria-current={here ? 'true' : undefined} aria-label={p.label}>
              <span className="lesson-phase-wide" aria-hidden="true">{p.label}</span><span className="lesson-phase-tight" aria-hidden="true">{p.tight}</span>
            </button>
            <div className="lesson-dots">
              {p.list.map((s) => {
                const state = s.id === active ? ' is-here' : seen.includes(s.id) ? ' is-seen' : ''
                return (
                  <button key={s.id} className={`lesson-dot${state}`} onClick={() => go(s.id)} title={s.title}
                    aria-label={`${s.title}${s.id === active ? ' (open)' : seen.includes(s.id) ? ' (read)' : ''}`} aria-current={s.id === active ? 'step' : undefined} />
                )
              })}
            </div>
          </div>
        )
      })}
      <span className="sr-only" aria-live="polite">{current ? `Section ${n} of ${sections.length}: ${current.title}` : ''}</span>
    </nav>
  )
}

const Chevron = ({ dir }: { dir: 'left' | 'right' }) => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d={dir === 'right' ? 'M9 6l6 6-6 6' : 'M15 6l-6 6 6 6'} />
  </svg>
)

function SectionPager({ sections, active, go }: { sections: SectionInfo[]; active: string | null; go: (id: string, how?: { scroll?: boolean; focus?: boolean }) => void }) {
  const i = sections.findIndex((s) => s.id === active)
  if (i < 0 || sections.length < 2) return null
  const prev = sections[i - 1]
  const next = sections[i + 1]
  return (
    <nav className="lesson-pager" aria-label="Previous and next section">
      {prev
        ? <button className="lesson-pager-back" onClick={() => go(prev.id, { focus: true })} aria-keyshortcuts="ArrowLeft"><Chevron dir="left" />{prev.short}</button>
        : <span />}
      {next && (
        <button className="lesson-pager-next" onClick={() => go(next.id, { focus: true })} aria-keyshortcuts="ArrowRight">
          <span><small>Next</small>{next.title}</span><Chevron dir="right" />
        </button>
      )}
    </nav>
  )
}

function ResumeBar({ lessonId, sections, active, go }: { lessonId: string; sections: SectionInfo[]; active: string | null; go: (id: string) => void }) {
  const [last, setLast] = useState<string | null>(null)
  useEffect(() => {
    try { setLast(lessonHash().sub || lessonHash().anchor ? null : localStorage.getItem(lastKey(lessonId))) } catch { setLast(null) }
  }, [lessonId])
  const target = sections.find((s) => s.id === last)
  if (!target || target.id === sections[0]?.id || active !== sections[0]?.id) return null
  return (
    <div className="resume">
      <span>Last time you stopped at <b>{target.title}</b>.</span>
      <button className="btn small" onClick={() => { go(target.id); setLast(null) }}>Continue there</button>
      <button className="btn small ghost" onClick={() => setLast(null)}>Start from the beginning</button>
    </div>
  )
}

/* ---------- lesson wrapper ---------- */
export function Lesson({ id, children }: { id: string; children: ReactNode }) {
  const lesson = lessonById(id)
  const progress = useProgress()
  const body = useRef<HTMLDivElement>(null)
  const anchor = useRef<HTMLDivElement>(null)
  const { sections, active, seen, go } = useSections(id, body, anchor)
  useEffect(() => {
    if (lesson) updateProgress((p) => (p.last === id ? p : { ...p, last: id }))
  }, [id, lesson])
  if (!lesson) return <p>Unknown lesson: {id}</p>

  const prev = LESSONS[lesson.index - 1]
  const next = LESSONS[lesson.index + 1]
  const done = !!progress.completed[id]
  const atEnd = !sections.length || active === sections[sections.length - 1].id

  return (
    <LessonCtx.Provider value={lesson}>
      <article className="lesson">
        <nav className="crumbs" aria-label="Breadcrumb">
          <a href="#/">Course</a><span aria-hidden>/</span>
          <span>Part {lesson.part.number}: {lesson.part.title}</span><span aria-hidden>/</span>
          <span aria-current="page">{lesson.title}</span>
        </nav>
        <header>
          <h1 className="lesson-title"><TokenTitle text={lesson.title} /></h1>
          <p className="lesson-question">{lesson.question}</p>
          <div className="lesson-meta">
            <span>Lesson {lesson.code}</span>
            <span>About {lesson.minutes >= 90 ? `${Math.round(lesson.minutes / 60)} hours` : `${lesson.minutes} minutes`}</span>
            {lesson.sources?.length ? <span>Code: {lesson.sources.map((s) => s.path.split('/').pop()).join(', ')}</span> : null}
            {done && <span style={{ color: 'var(--good)', fontWeight: 600 }}>Completed</span>}
          </div>
        </header>
        <WhereAreWe here={lesson.here} />
        <ResumeBar lessonId={id} sections={sections} active={active} go={go} />
        <div ref={anchor} className="lesson-anchor" />
        <LessonRail sections={sections} active={active} seen={seen} go={go} />

        <div ref={body} className="lesson-body">
          <ErrorBoundary what={`the “${lesson.title}” lesson`}>{children}</ErrorBoundary>
        </div>
        <SectionPager sections={sections} active={active} go={go} />

        {atEnd && (
          <footer className="lesson-foot">
            <div className="complete-box">
              <div>
                <strong>{done ? 'Lesson complete.' : 'Finished this lesson?'}</strong>
                <div className="muted" style={{ fontSize: 14.5 }}>{done ? 'Nice. Come back any time to replay the experiments.' : 'Mark it complete only if you could explain the key ideas without looking.'}</div>
              </div>
              <button className={`btn${done ? '' : ' primary'}`} onClick={() => updateProgress((p) => completeLesson(p, id, !done))}>
                {done ? 'Mark as not done' : 'Mark complete'}
              </button>
            </div>
            <nav className="prevnext" aria-label="Previous and next lesson">
              {prev ? <a href={`#/lesson/${prev.id}`}><small>Previous lesson</small>{prev.title}</a> : <span />}
              {next ? <a className="next" href={`#/lesson/${next.id}`}><small>Next lesson</small>{next.title}</a> : <a className="next" href="#/"><small>Finished</small>Back to the course map</a>}
            </nav>
          </footer>
        )}
      </article>
    </LessonCtx.Provider>
  )
}

/* ---------- sections ---------- */
// The section's place in the lesson is shown by the rail above, so a section is just its heading and content.
// `n` and `kicker` are kept for readers of the source: they name the step of the lesson loop.
function Section({ title, children, id }: { n: number; kicker: string; title: ReactNode; children: ReactNode; id: string }) {
  return (
    <section className="section" id={id}>
      <h2 tabIndex={-1}>{title}</h2>
      {children}
    </section>
  )
}

type SP = { title?: ReactNode; children: ReactNode }
export const Why = ({ title = 'Why are we learning this?', children }: SP) => <Section n={1} id="why" kicker="Why" title={title}>{children}</Section>
export const Problem = ({ title = 'The problem', children }: SP) => <Section n={2} id="problem" kicker="Problem" title={title}>{children}</Section>
export const MentalModel = ({ title = 'A mental model', children }: SP) => <Section n={3} id="mental-model" kicker="Mental model" title={title}>{children}</Section>
export const TryIt = ({ title = 'Try it yourself', children }: SP) => <Section n={4} id="try-it" kicker="Interact" title={title}>{children}</Section>
export const Numbers = ({ title = "Let's see the numbers", children }: SP) => <Section n={5} id="numbers" kicker="Numbers" title={title}>{children}</Section>
export const TheMath = ({ title = 'The math', children }: SP) => <Section n={6} id="math" kicker="Math" title={title}>{children}</Section>
export const CodeIt = ({ title = "Let's code it", children }: SP) => <Section n={7} id="code" kicker="Code" title={title}>{children}</Section>
export const BreakIt = ({ title = 'Break it', children }: SP) => <Section n={8} id="break-it" kicker="Experiment" title={title}>{children}</Section>
export const Exercises = ({ title = 'Exercises', children }: SP) => <Section n={9} id="exercises" kicker="Exercise" title={title}>{children}</Section>

export function CheckYourself({ questions }: { questions: QuizQuestion[] }) {
  const lesson = useLesson()
  return (
    <Section n={10} id="check" kicker="Recall" title="Can you explain this?">
      <p className="muted">Answer from memory first. Going back to earlier sections is allowed, but only after you have committed to a guess.</p>
      <Quiz id={`check-${lesson?.id ?? 'x'}`} questions={questions} lesson={lesson?.id} />
    </Section>
  )
}

export function Remember({ items }: { items: ReactNode[] }) {
  return (
    <Section n={11} id="remember" kicker="Remember" title="What you should remember">
      <ol className="remember">{items.map((it, i) => <li key={i}><div>{it}</div></li>)}</ol>
    </Section>
  )
}

/** Section 12. Also renders the links to the real repository code for this lesson. */
export function RealLLM({ children }: { children: ReactNode }) {
  const lesson = useLesson()
  return (
    <Section n={12} id="real-llm" kicker="Real LLMs" title="Where this appears in a real LLM">
      {children}
      {lesson?.sources?.length ? (
        <>
          <h3>Read the real code</h3>
          <p>Everything in this lesson is implemented, runnable, in the repository. Open it, run it, change it.</p>
          <div className="source-list">
            {lesson.sources.map((s) => (
              <a key={s.path} className="source" href={sourceUrl(s.path)} target="_blank" rel="noreferrer">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden><path d="M8 6l-6 6 6 6M16 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" /></svg>
                <code>{s.path}</code>
                <span>{s.note}</span>
              </a>
            ))}
          </div>
          {lesson.notes && <p className="muted" style={{ fontSize: 15 }}>Prefer long-form reading? The original written notes for this topic are in <a href={sourceUrl(lesson.notes)} target="_blank" rel="noreferrer"><code>{lesson.notes}</code></a>.</p>}
        </>
      ) : null}
    </Section>
  )
}

/** Spaced recall at the end of a part: a mixed quiz that reaches back to earlier lessons. */
export function BeforeMovingOn({ id, intro, questions, children }: { id: string; intro?: ReactNode; questions: QuizQuestion[]; children?: ReactNode }) {
  return (
    <section className="section" id="before-moving-on">
      <h2 tabIndex={-1}>Before moving on…</h2>
      <p>{intro ?? 'These questions reach back to earlier lessons on purpose. Pulling an idea out of memory is what makes it stick.'}</p>
      {children}
      <Quiz id={`checkpoint-${id}`} questions={questions} />
    </section>
  )
}
