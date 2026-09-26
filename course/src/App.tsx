import { Suspense, lazy, useEffect, useMemo, useState, type ComponentType } from 'react'
import { LESSONS, PARTS, lessonById } from './data/curriculum'
import { percentComplete, useProgress } from './lib/progress'
import { dueItems, useReview } from './lib/review'
import { search } from './lib/search'
import { Home } from './pages/Home'
import { GlossaryPage } from './pages/Glossary'
import { ConceptMapPage } from './pages/ConceptMap'
import { SelfTestPage } from './pages/SelfTest'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ReviewPage } from './pages/Review'
import { ProjectPage } from './pages/Project'
// the GPT-2 Explainer page is code-split: its engine and worker load only when someone opens it
const Gpt2Page = lazy(() => import('./pages/Gpt2Page'))
const PlaygroundPage = lazy(() => import('./pages/Playground').then((m) => ({ default: m.PlaygroundPage })))

/* ---------- hash routing: static-host friendly, no dependency ---------- */
// #/lesson/<id>/<section> opens one section; #/lesson/<id>#<element> jumps to an element (see components/lesson.tsx)
const parseHash = () => {
  const path = window.location.hash.replace(/^#/, '').split('#')[0]
  const [, a = '', b = ''] = path.split('/')
  return { page: a || 'home', arg: decodeURIComponent(b) }
}
const useRoute = () => {
  const [route, setRoute] = useState(parseHash)
  useEffect(() => {
    const on = () => setRoute(parseHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return route
}

/* ---------- lessons are code-split and discovered by file name ---------- */
const modules = import.meta.glob<{ default: ComponentType }>(['./lessons/*.tsx', '!./lessons/*.test.tsx'])
const lazyLessons = new Map<string, ComponentType>()
const lessonComponent = (id: string) => {
  if (!lazyLessons.has(id)) {
    const loader = modules[`./lessons/${id}.tsx`]
    if (!loader) return null
    lazyLessons.set(id, lazy(loader))
  }
  return lazyLessons.get(id)!
}

/* ---------- theme ---------- */
type Theme = 'light' | 'dark'
const initialTheme = (): Theme => {
  try {
    const saved = localStorage.getItem('llm-fp-theme-v4')
    if (saved === 'light' || saved === 'dark') return saved
  } catch { /* ignore */ }
  return 'dark' // dark is the default; the toggle switches to light
}

function Sidebar({ open, current, onNavigate, theme, toggleTheme, onCollapse }: { open: boolean; current: string; onNavigate: () => void; theme: Theme; toggleTheme: () => void; onCollapse: () => void }) {
  const progress = useProgress()
  const review = useReview()
  const dueCount = dueItems(review).length
  const [query, setQuery] = useState('')
  const hits = useMemo(() => search(query), [query])
  const pct = percentComplete(progress)
  const route = parseHash()

  return (
    <aside className={`sidebar${open ? ' open' : ''}`} aria-label="Course navigation">
      <div className="brand-row">
      <a className="brand" href="#/" onClick={onNavigate}>
        <svg width="30" height="30" viewBox="0 0 40 40" aria-hidden>
          <path d="M20 9V4" stroke="var(--il-ink)" strokeWidth="2.5" strokeLinecap="round" /><circle cx="20" cy="4" r="3" fill="var(--il-d)" stroke="var(--il-ink)" strokeWidth="2" />
          <rect x="2" y="15" width="6" height="12" rx="3" fill="var(--il-b)" stroke="var(--il-ink)" strokeWidth="2" /><rect x="32" y="15" width="6" height="12" rx="3" fill="var(--il-b)" stroke="var(--il-ink)" strokeWidth="2" />
          <rect x="5" y="9" width="30" height="26" rx="10" fill="#fff" stroke="var(--il-ink)" strokeWidth="2.5" />
          <circle cx="14" cy="21" r="3.6" fill="var(--il-ink)" /><circle cx="26" cy="21" r="3.6" fill="var(--il-ink)" /><path d="M16 28q4 4 8 0" fill="none" stroke="var(--il-ink)" strokeWidth="2" strokeLinecap="round" />
        </svg>
        <div className="brand-name">LLM From<br />First Principles</div>
      </a>
      <button className="side-collapse" onClick={onCollapse} aria-label="Hide the lesson list" title="Hide the lesson list (press \\)">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M15 6l-6 6 6 6" /></svg>
      </button>
      </div>
      <div className="side-search">
        <label className="sr-only" htmlFor="search">Search lessons and glossary</label>
        <input id="search" className="input" type="search" placeholder="Search lessons and terms…" value={query} onChange={(e) => setQuery(e.target.value)} />
      </div>
      <div className="side-tools">
        <a href="#/map" aria-current={route.page === 'map' ? 'page' : undefined} onClick={onNavigate}>Concept map</a>
        <a href="#/glossary" aria-current={route.page === 'glossary' ? 'page' : undefined} onClick={onNavigate}>Glossary</a>
        <a href="#/review" aria-current={route.page === 'review' ? 'page' : undefined} onClick={onNavigate}>Review{dueCount > 0 ? ` (${dueCount})` : ''}</a>
        <a href="#/gpt2" aria-current={route.page === 'gpt2' ? 'page' : undefined} onClick={onNavigate}>GPT-2 Explainer</a>
        <a href="#/python" aria-current={route.page === 'python' ? 'page' : undefined} onClick={onNavigate}>Python playground</a>
        <a href="#/project" aria-current={route.page === 'project' ? 'page' : undefined} onClick={onNavigate}>Your support bot</a>
      </div>

      <nav className="nav" aria-label="Lessons">
        {query ? (
          <div role="region" aria-live="polite" aria-label="Search results">
            <div className="nav-part-title">{hits.length ? `${hits.length} result${hits.length > 1 ? 's' : ''}` : 'Nothing found'}</div>
            {hits.map((h) => (
              <a key={h.href} className="nav-link" href={h.href} onClick={() => { setQuery(''); onNavigate() }}>
                <span className="mono muted" style={{ fontSize: 11, marginTop: 3 }}>{h.kind === 'lesson' ? '▸' : 'Aa'}</span>
                <span>{h.title}<br /><small className="muted">{h.detail.length > 70 ? `${h.detail.slice(0, 70)}…` : h.detail}</small></span>
              </a>
            ))}
          </div>
        ) : (
          PARTS.map((part) => (
            <div className="nav-part" key={part.id}>
              <div className="nav-part-title"><span>{String(part.number).padStart(2, '0')}</span><b>{part.title}</b></div>
              {part.lessons.map((l) => {
                const done = !!progress.completed[l.id]
                return (
                  <a key={l.id} className="nav-link" href={`#/lesson/${l.id}`} aria-current={current === l.id ? 'page' : undefined} onClick={onNavigate}>
                    <span className={`nav-tick${done ? ' done' : ''}`} aria-hidden>✓</span>
                    <span>{l.title}{done && <span className="sr-only"> (completed)</span>}</span>
                  </a>
                )
              })}
            </div>
          ))
        )}
      </nav>

      <div className="side-foot">
        <div className="side-foot-top">
          <span>Progress</span>
          <button className="theme-toggle" onClick={toggleTheme} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              {theme === 'dark'
                ? <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>
                : <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />}
            </svg>
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>
        <div className="side-foot-count">
          <span>{Object.keys(progress.completed).length} of {LESSONS.length} lessons</span>
        </div>
        <div className="meter" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label="Course progress"><span style={{ width: `${pct}%` }} /></div>
      </div>
    </aside>
  )
}

const initialCollapsed = (): boolean => {
  try { return localStorage.getItem('llm-fp-sidebar') === 'collapsed' } catch { return false }
}

export function App() {
  const route = useRoute()
  const [open, setOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(initialCollapsed)
  const [theme, setTheme] = useState<Theme>(initialTheme)

  useEffect(() => {
    try { localStorage.setItem('llm-fp-sidebar', collapsed ? 'collapsed' : 'open') } catch { /* ignore */ }
  }, [collapsed])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('llm-fp-theme-v4', theme) } catch { /* ignore */ }
  }, [theme])

  // new page -> scroll to top and move focus to the content for keyboard / screen-reader users
  useEffect(() => {
    window.scrollTo(0, 0)
    document.getElementById('content')?.focus({ preventScroll: true })
    const l = route.page === 'lesson' ? lessonById(route.arg) : undefined
    document.title = l ? `${l.title} · LLM From First Principles` : route.page === 'gpt2' ? 'GPT-2 Explainer · LLM From First Principles' : route.page === 'python' ? 'Python playground · LLM From First Principles' : 'LLM From First Principles'
  }, [route.page, route.arg])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
      const typing = /input|textarea|select/i.test((e.target as HTMLElement).tagName)
      if (e.key === '\\' && !typing) { e.preventDefault(); setCollapsed((c) => !c) }
      if (e.key === '/' && !typing) {
        e.preventDefault()
        setOpen(true)
        document.getElementById('search')?.focus()
      }
      // j / k move between lessons, like a reader. Arrow keys are left alone for scrolling.
      if (!typing && !e.metaKey && !e.ctrlKey && !e.altKey && (e.key === 'j' || e.key === 'k')) {
        const here = LESSONS.findIndex((l) => l.id === parseHash().arg)
        if (here >= 0) {
          const next = LESSONS[here + (e.key === 'j' ? 1 : -1)]
          if (next) { e.preventDefault(); window.location.hash = `#/lesson/${next.id}` }
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const current = route.page === 'lesson' ? route.arg : ''
  const LessonBody = current ? lessonComponent(current) : null
  const meta = current ? lessonById(current) : undefined

  let body
  if (route.page === 'lesson') {
    body = LessonBody && meta ? (
      <Suspense fallback={<p className="muted">Loading lesson…</p>}><LessonBody /></Suspense>
    ) : (
      <div><h1 className="lesson-title">Lesson not found</h1><p><a href="#/">Back to the course</a></p></div>
    )
  } else if (route.page === 'glossary') body = <GlossaryPage focus={route.arg} />
  else if (route.page === 'map') body = <ConceptMapPage />
  else if (route.page === 'selftest') body = <SelfTestPage />
  else if (route.page === 'review') body = <ReviewPage />
  else if (route.page === 'project') body = <ProjectPage />
  else if (route.page === 'python') body = <Suspense fallback={<p className="muted">Loading the playground…</p>}><PlaygroundPage /></Suspense>
  else if (route.page === 'gpt2') body = <Suspense fallback={<p className="muted">Loading the GPT-2 Explainer…</p>}><Gpt2Page arg={route.arg} /></Suspense>
  else body = <Home />

  return (
    <div className={`shell${collapsed ? ' collapsed' : ''}`}>
      <a className="skip-link" href="#content">Skip to content</a>
      {open && <button className="scrim" aria-label="Close navigation" onClick={() => setOpen(false)} />}
      <Sidebar open={open} current={current} onNavigate={() => setOpen(false)} theme={theme} toggleTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')} onCollapse={() => setCollapsed(true)} />
      {collapsed && (
        <button className="sidebar-reopen" onClick={() => setCollapsed(false)} aria-label="Show the lesson list" title="Show the lesson list (press \\)">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><path d="M4 7h16M4 12h16M4 17h16" /></svg>
        </button>
      )}
      <div className="main">
        <div className="topbar">
          <button className="btn small" onClick={() => setOpen(true)} aria-label="Open navigation" aria-expanded={open} style={{ minWidth: 40, minHeight: 40, justifyContent: 'center' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><path d="M4 7h16M4 12h16M4 17h16" /></svg>
          </button>
          <span className="topbar-title">{meta ? meta.title : route.page === 'gpt2' ? 'GPT-2 Explainer' : 'LLM From First Principles'}</span>
        </div>
        <main id="content" tabIndex={-1} className={`page${route.page === 'map' || route.page === 'gpt2' ? ' wide' : ''}`} style={{ outline: 'none' }}>
          <ErrorBoundary what="this page">{body}</ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
