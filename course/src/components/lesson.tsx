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
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
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

/* ---------- outline: where you are in the lesson, and where you stopped ---------- */
const SECTION_LABELS: [string, string][] = [
  ['why', 'Why'], ['problem', 'The problem'], ['mental-model', 'Mental model'], ['try-it', 'Try it'],
  ['numbers', 'The numbers'], ['math', 'The math'], ['code', 'The code'], ['break-it', 'Break it'],
  ['exercises', 'Exercises'], ['check', 'Check yourself'], ['remember', 'Remember'], ['real-llm', 'Real LLMs'],
  ['before-moving-on', 'Checkpoint'],
]
const scrollKey = (id: string) => `llm-fp-place-${id}`

function LessonOutline({ lessonId }: { lessonId: string }) {
  const [present, setPresent] = useState<string[]>([])
  const [active, setActive] = useState<string>('')

  useEffect(() => {
    const found = SECTION_LABELS.filter(([sid]) => document.getElementById(sid)).map(([sid]) => sid)
    setPresent(found)
    // Not available in test environments or very old browsers: the outline still renders and links work.
    if (found.length === 0 || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).map((e) => e.target.id)
        if (visible.length) setActive(visible[0])
      },
      { rootMargin: '-80px 0px -65% 0px' },
    )
    found.forEach((sid) => { const el = document.getElementById(sid); if (el) io.observe(el) })
    return () => io.disconnect()
  }, [lessonId])

  // Remember the reading position, throttled, and restore it on a later visit.
  useEffect(() => {
    let raf = 0
    const save = () => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => {
        try {
          if (window.scrollY > 400) localStorage.setItem(scrollKey(lessonId), String(window.scrollY))
          else localStorage.removeItem(scrollKey(lessonId))
        } catch { /* storage unavailable */ }
      })
    }
    window.addEventListener('scroll', save, { passive: true })
    return () => { window.removeEventListener('scroll', save); cancelAnimationFrame(raf) }
  }, [lessonId])

  if (present.length < 3) return null
  return (
    <nav className="outline" aria-label="Sections of this lesson">
      {SECTION_LABELS.filter(([sid]) => present.includes(sid)).map(([sid, label]) => (
        <a key={sid} href={`#${sid}`} className={active === sid ? 'here' : undefined} aria-current={active === sid ? 'location' : undefined}
           onClick={(e) => { e.preventDefault(); document.getElementById(sid)?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }}>
          {label}
        </a>
      ))}
    </nav>
  )
}

function ResumeBar({ lessonId }: { lessonId: string }) {
  const [place, setPlace] = useState<number | null>(null)
  useEffect(() => {
    try {
      const raw = localStorage.getItem(scrollKey(lessonId))
      setPlace(raw ? Number(raw) : null)
    } catch { setPlace(null) }
  }, [lessonId])
  if (!place || place < 400) return null
  return (
    <div className="resume">
      <span>You stopped part way through this lesson.</span>
      <button className="btn small" onClick={() => { window.scrollTo({ top: place, behavior: 'smooth' }); setPlace(null) }}>Jump back to where you were</button>
      <button className="btn small ghost" onClick={() => { try { localStorage.removeItem(scrollKey(lessonId)) } catch { /* ignore */ } setPlace(null) }}>Start from the top</button>
    </div>
  )
}

/* ---------- lesson wrapper ---------- */
export function Lesson({ id, children }: { id: string; children: ReactNode }) {
  const lesson = lessonById(id)
  const progress = useProgress()
  useEffect(() => {
    if (lesson) updateProgress((p) => (p.last === id ? p : { ...p, last: id }))
  }, [id, lesson])
  if (!lesson) return <p>Unknown lesson: {id}</p>

  const prev = LESSONS[lesson.index - 1]
  const next = LESSONS[lesson.index + 1]
  const done = !!progress.completed[id]

  return (
    <LessonCtx.Provider value={lesson}>
      <article>
        <nav className="crumbs" aria-label="Breadcrumb">
          <a href="#/">Course</a><span aria-hidden>/</span>
          <span>Part {lesson.part.number}: {lesson.part.title}</span><span aria-hidden>/</span>
          <span aria-current="page">{lesson.title}</span>
        </nav>
        <header>
          <div className="lesson-code">Lesson {lesson.code}</div>
          <h1 className="lesson-title"><TokenTitle text={lesson.title} /></h1>
          <p className="lesson-question">{lesson.question}</p>
          <div className="lesson-meta">
            <span>About {lesson.minutes >= 90 ? `${Math.round(lesson.minutes / 60)} hours` : `${lesson.minutes} minutes`}</span>
            {lesson.sources?.length ? <span>Code: {lesson.sources.map((s) => s.path.split('/').pop()).join(', ')}</span> : null}
            {done && <span style={{ color: 'var(--good)', fontWeight: 600 }}>Completed</span>}
          </div>
        </header>
        <WhereAreWe here={lesson.here} />
        <ResumeBar lessonId={id} />
        <LessonOutline lessonId={id} />

        <ErrorBoundary what={`the “${lesson.title}” lesson`}>{children}</ErrorBoundary>

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
            {prev ? <a href={`#/lesson/${prev.id}`}><small>Previous</small>{prev.title}</a> : <span />}
            {next ? <a className="next" href={`#/lesson/${next.id}`}><small>Next</small>{next.title}</a> : <a className="next" href="#/"><small>Finished</small>Back to the course map</a>}
          </nav>
        </footer>
      </article>
    </LessonCtx.Provider>
  )
}

/* ---------- sections ---------- */
function Section({ n, kicker, title, children, id }: { n: number; kicker: string; title: ReactNode; children: ReactNode; id: string }) {
  return (
    <section className="section" id={id}>
      <div className="section-rail"><span className="section-n">{n}</span><span>{kicker}</span></div>
      <h2>{title}</h2>
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
      <p className="muted">Answer from memory first. Scrolling up is allowed, but only after you have committed to a guess.</p>
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
      <div className="section-rail"><span>Checkpoint</span></div>
      <h2>Before moving on…</h2>
      <p>{intro ?? 'These questions reach back to earlier lessons on purpose. Pulling an idea out of memory is what makes it stick.'}</p>
      {children}
      <Quiz id={`checkpoint-${id}`} questions={questions} />
    </section>
  )
}
