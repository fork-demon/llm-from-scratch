import { describe, expect, it } from 'vitest'
import { countMarks, loadMarks, MARKS_KEY, READING_PATH, sanitiseMarks, saveMarks, SECTIONS, sectionById, setMark } from './paperMap'
import { LESSONS } from '../data/curriculum'

const lessonIds = new Set(LESSONS.map((l) => l.id))

describe('the paper map', () => {
  it('uses the paper’s real section numbering, in order', () => {
    expect(SECTIONS.map((s) => s.number)).toEqual(['', '1', '2', '3', '3.1', '3.2', '3.2.1', '3.2.2', '3.2.3', '3.3', '3.4', '3.5', '4', '5', '6', '7'])
    expect(sectionById('s321')?.title).toBe('Scaled Dot-Product Attention')
    expect(sectionById('s323')?.title).toBe('Applications of Attention in our Model')
    expect(sectionById('s33')?.title).toBe('Position-wise Feed-Forward Networks')
    expect(sectionById('s4')?.title).toBe('Why Self-Attention')
  })
  it('depth follows the numbering', () => {
    for (const s of SECTIONS) expect(s.depth, s.id).toBe(s.number ? s.number.split('.').length - 1 : 0)
  })
  it('every referenced lesson exists in curriculum.ts', () => {
    for (const s of SECTIONS) {
      expect(s.built.length, s.id).toBeGreaterThan(0)
      for (const b of s.built) expect(lessonIds.has(b.lesson), `${s.id} -> ${b.lesson}`).toBe(true)
    }
    for (const p of READING_PATH) for (const l of p.prepared) expect(lessonIds.has(l), `${p.short} -> ${l}`).toBe(true)
  })
  it('every section has a summary of at most three sentences-worth of paragraphs, something to look at, and a difference', () => {
    for (const s of SECTIONS) {
      expect(s.summary.length).toBeGreaterThan(0)
      expect(s.summary.length).toBeLessThanOrEqual(3)
      expect(s.look.length).toBeGreaterThan(10)
      expect(s.differs.length).toBeGreaterThan(0)
    }
  })
  it('no em dashes anywhere in the text', () => {
    expect(JSON.stringify(SECTIONS) + JSON.stringify(READING_PATH)).not.toMatch(/—/)
  })
  it('the reading path has the 13 agreed papers with arXiv links where an id exists', () => {
    expect(READING_PATH).toHaveLength(13)
    for (const p of READING_PATH) {
      const m = p.id.match(/arXiv:(\d{4}\.\d{5})/g)
      if (m) expect(m.some((x) => p.url.endsWith(x.replace('arXiv:', ''))), p.short).toBe(true)
      expect(p.year).toBeGreaterThanOrEqual(2019)
    }
  })
})

describe('progress marks', () => {
  const fakeStorage = () => {
    const data: Record<string, string> = {}
    return { data, getItem: (k: string) => data[k] ?? null, setItem: (k: string, v: string) => { data[k] = v } }
  }
  it('round-trips through storage under its own key', () => {
    const st = fakeStorage()
    const marks = setMark(setMark({}, 's321', 're-derived'), 's1', 'read')
    expect(saveMarks(st, marks)).toBe(true)
    expect(Object.keys(st.data)).toEqual([MARKS_KEY])
    expect(loadMarks(st)).toEqual(marks)
  })
  it('setting “none” removes the mark', () => {
    expect(setMark({ s1: 'read' }, 's1', 'none')).toEqual({})
  })
  it('survives missing, throwing and corrupted storage', () => {
    expect(loadMarks(undefined)).toEqual({})
    expect(saveMarks(undefined, {})).toBe(false)
    const throwing = { getItem: () => { throw new Error('blocked') }, setItem: () => { throw new Error('blocked') } }
    expect(loadMarks(throwing)).toEqual({})
    expect(saveMarks(throwing, { s1: 'read' })).toBe(false)
    const st = fakeStorage()
    st.data[MARKS_KEY] = '{not json'
    expect(loadMarks(st)).toEqual({})
  })
  it('drops unknown sections and unknown marks', () => {
    expect(sanitiseMarks({ s1: 'read', nope: 'read', s2: 'mastered', s3: 'none', s4: 7 })).toEqual({ s1: 'read' })
    expect(sanitiseMarks(null)).toEqual({})
  })
  it('counts cumulatively: re-derived implies understood implies read', () => {
    expect(countMarks({ s1: 'read', s2: 'understood', s321: 're-derived' })).toEqual({ read: 3, understood: 2, 're-derived': 1 })
  })
})
