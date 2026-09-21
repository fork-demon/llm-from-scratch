import { useEffect, useMemo, useState } from 'react'
import { GLOSSARY } from '../data/glossary'
import { lessonById } from '../data/curriculum'

export function GlossaryPage({ focus }: { focus?: string }) {
  const [query, setQuery] = useState('')
  const entries = useMemo(() => {
    const q = query.trim().toLowerCase()
    const list = [...GLOSSARY].sort((a, b) => a.term.localeCompare(b.term))
    return q ? list.filter((g) => `${g.term} ${g.short} ${g.intuition}`.toLowerCase().includes(q)) : list
  }, [query])

  useEffect(() => {
    if (focus) document.getElementById(`g-${focus}`)?.scrollIntoView({ block: 'start' })
  }, [focus])

  return (
    <div>
      <h1 className="lesson-title">Glossary</h1>
      <p className="lesson-question">Every term: one sentence, an intuition, an example, and where the course teaches it.</p>
      <label className="sr-only" htmlFor="gq">Filter glossary</label>
      <input id="gq" className="input" type="search" placeholder="Filter terms…" value={query} onChange={(e) => setQuery(e.target.value)} style={{ maxWidth: 360, marginBottom: 12 }} />
      <p className="muted" role="status" style={{ fontSize: 14 }}>{entries.length} term{entries.length === 1 ? '' : 's'}</p>
      {entries.map((g) => (
        <div className="term" key={g.id} id={`g-${g.id}`} style={{ scrollMarginTop: 70, outline: focus === g.id ? '2px solid var(--accent)' : undefined }}>
          <div className="term-name">{g.term}</div>
          <div className="term-row"><span>Definition</span><div>{g.short}</div></div>
          <div className="term-row"><span>Intuition</span><div>{g.intuition}</div></div>
          <div className="term-row"><span>Example</span><div className="mono" style={{ fontSize: 14 }}>{g.example}</div></div>
          <div className="term-row"><span>Taught in</span><div>{g.lessons.map((id, i) => { const l = lessonById(id); return l ? <span key={id}>{i > 0 && ' · '}<a href={`#/lesson/${id}`}>{l.code} {l.title}</a></span> : null })}</div></div>
        </div>
      ))}
    </div>
  )
}
