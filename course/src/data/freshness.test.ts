import { describe, expect, it } from 'vitest'
import { LESSONS } from './curriculum'
import { FRESHNESS, dueForReview, lastChecked } from './freshness'

describe('the freshness register', () => {
  it('has unique ids, real lessons, sources and valid dates', () => {
    expect(new Set(FRESHNESS.map((c) => c.id)).size).toBe(FRESHNESS.length)
    const lessons = new Set(LESSONS.map((l) => l.id))
    for (const c of FRESHNESS) {
      expect(lessons.has(c.lesson), `${c.id}: unknown lesson ${c.lesson}`).toBe(true)
      expect(c.sources.length, `${c.id}: needs a source`).toBeGreaterThan(0)
      for (const u of c.sources) expect(u, c.id).toMatch(/^https:\/\//)
      expect(Number.isNaN(Date.parse(c.checked)), `${c.id}: bad date`).toBe(false)
    }
  })
  it('works out what is due and when a lesson was last checked', () => {
    expect(dueForReview('2026-09-28')).toEqual([])
    expect(dueForReview('2027-01-01').some((c) => c.every === 3)).toBe(true)
    expect(lastChecked('modern-architecture')).toBe('2026-09-29')
    expect(lastChecked('vectors')).toBeNull()
  })
})
