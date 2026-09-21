import { LESSONS } from '../data/curriculum'
import { GLOSSARY } from '../data/glossary'
import { CODE_EXERCISES } from '../data/codeExercises'

// Lesson bodies as raw source, so search can look inside the prose. Stripping JSX tags and
// code-ish noise leaves readable sentences; good enough to find a phrase and jump to the lesson.
const sources = import.meta.glob('../lessons/*.tsx', { query: '?raw', import: 'default', eager: true }) as Record<string, string>

const cleanText = (src: string): string =>
  src
    .replace(/^import[\s\S]*?from '[^']+'\n/gm, ' ')
    .replace(/<Code[\s\S]*?<\/Code>/g, ' ')
    .replace(/\{`[\s\S]*?`\}/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/[{}()[\]`$]/g, ' ')
    .replace(/\s+/g, ' ')
    .toLowerCase()

const BODY: Record<string, string> = Object.fromEntries(
  Object.entries(sources)
    .filter(([path]) => !path.includes('.test.'))
    .map(([path, src]) => [path.split('/').pop()!.replace('.tsx', ''), cleanText(src)]),
)

/** A short snippet of the lesson text around the first match, for the result list. */
const snippet = (body: string, word: string): string => {
  const i = body.indexOf(word)
  if (i < 0) return ''
  const start = Math.max(0, i - 40)
  return (start > 0 ? '…' : '') + body.slice(start, i + word.length + 60).trim() + '…'
}

export interface SearchHit { kind: 'lesson' | 'term' | 'text' | 'code'; title: string; detail: string; href: string }

/** Tiny search over lesson titles/questions and glossary entries. All query words must match. */
export const search = (query: string, limit = 12): SearchHit[] => {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (words.length === 0) return []
  const matches = (hay: string) => words.every((w) => hay.includes(w))
  const hits: (SearchHit & { rank: number })[] = []
  for (const l of LESSONS) {
    const title = l.title.toLowerCase()
    const meta = `${title} ${l.question} ${l.part.title} ${(l.sources ?? []).map((s) => s.path).join(' ')}`.toLowerCase()
    const body = BODY[l.id] ?? ''
    if (matches(meta)) hits.push({ kind: 'lesson', title: l.title, detail: `Lesson ${l.code}`, href: `#/lesson/${l.id}`, rank: matches(title) ? 0 : 2 })
    else if (matches(body)) hits.push({ kind: 'text', title: l.title, detail: snippet(body, words[0]) || `Lesson ${l.code}`, href: `#/lesson/${l.id}`, rank: 4 })
  }
  for (const e of CODE_EXERCISES) {
    if (matches(`${e.title} ${e.prompt}`.toLowerCase())) hits.push({ kind: 'code', title: e.title, detail: 'Coding exercise', href: `#/lesson/${e.lesson}#ex-${e.id}`, rank: 1.5 })
  }
  for (const g of GLOSSARY) {
    const term = g.term.toLowerCase()
    const hay = `${term} ${g.short} ${g.intuition}`.toLowerCase()
    if (matches(hay)) hits.push({ kind: 'term', title: g.term, detail: g.short, href: `#/glossary/${g.id}`, rank: matches(term) ? 1 : 3 })
  }
  return hits.sort((a, b) => a.rank - b.rank).slice(0, limit).map(({ rank: _rank, ...h }) => h)
}
