import { describe, expect, it } from 'vitest'
import { search } from './search'
import { LESSONS, LLM_TREE, PARTS, type TreeNode } from '../data/curriculum'
import { GLOSSARY } from '../data/glossary'

describe('search', () => {
  it('finds lessons and glossary terms', () => {
    const hits = search('attention')
    expect(hits[0].kind).toBe('lesson')
    expect(hits.some((h) => h.kind === 'term')).toBe(true)
  })
  it('requires all words and handles empty queries', () => {
    expect(search('')).toEqual([])
    expect(search('kv cache').length).toBeGreaterThan(0)
    expect(search('zzzz qqqq')).toEqual([])
  })
})

describe('curriculum integrity', () => {
  const lessonModules = import.meta.glob('../lessons/*.tsx')
  const ids = LESSONS.map((l) => l.id)
  it('has unique lesson ids', () => {
    expect(new Set(ids).size).toBe(ids.length)
  })
  it('every lesson in the curriculum has a lesson file', () => {
    const files = Object.keys(lessonModules).map((p) => p.split('/').pop()!.replace('.tsx', ''))
    expect(ids.filter((id) => !files.includes(id))).toEqual([])
  })
  it('every lesson points at a real node of the where-are-we tree', () => {
    const nodes: string[] = []
    const walk = (n: TreeNode) => { nodes.push(n.id); n.children?.forEach(walk) }
    walk(LLM_TREE)
    expect(LESSONS.filter((l) => !nodes.includes(l.here)).map((l) => l.id)).toEqual([])
  })
  it('glossary entries are unique and link to real lessons', () => {
    const gids = GLOSSARY.map((g) => g.id)
    expect(new Set(gids).size).toBe(gids.length)
    const bad = GLOSSARY.flatMap((g) => g.lessons.filter((l) => !ids.includes(l)).map((l) => `${g.id}->${l}`))
    expect(bad).toEqual([])
  })
  it('parts are numbered in order', () => {
    expect(PARTS.map((p) => p.number)).toEqual(PARTS.map((_, i) => i))
  })
})

describe('full-text search', () => {
  it('finds a phrase that only appears in a lesson body', () => {
    const hits = search('river bank')
    expect(hits.length).toBeGreaterThan(0)
    expect(hits.some((h) => h.href.includes('attention'))).toBe(true)
  })
  it('finds coding exercises', () => {
    expect(search('dot product').some((h) => h.kind === 'code' || h.kind === 'lesson')).toBe(true)
  })
})
