import { describe, expect, it } from 'vitest'
import { QUESTIONS, askAll, dbLookup, keywords, ruleProgram, searchDocs } from './fourSystems'

describe('database lookup', () => {
  it('hits only on an exact (normalised) key', () => {
    expect(dbLookup('  What is the capital of   France? ').output).toBe('Paris')
    expect(dbLookup("Which city is France's capital?").verdict).toBe('fails')
  })
})

describe('keyword search', () => {
  it('drops stop words', () => expect(keywords('What is the capital of France?')).toEqual(['capital', 'france']))
  it('ranks the France page first for both wordings', () => {
    expect(searchDocs('What is the capital of France?')[0].doc.title).toContain('France')
    expect(searchDocs("Which city is France's capital?")[0].doc.title).toContain('France')
  })
  it('returns related pages, not an answer, for the limerick', () => {
    const titles = searchDocs('Write a limerick about a Kubernetes pod').map((r) => r.doc.title).join(' ')
    expect(titles).toContain('Kubernetes')
    expect(titles).toContain('limerick')
  })
  it('finds the fresh match report and nothing for arithmetic', () => {
    expect(searchDocs("Who won yesterday's match?")[0].doc.title).toContain('Match report')
    expect(searchDocs('What is 48,193 × 7,206?')).toEqual([])
  })
})

describe('rule-based program', () => {
  it('multiplies exactly', () => expect(ruleProgram('What is 48,193 × 7,206?').output).toBe('347,278,758'))
  it('knows only the capitals it was given, and only one phrasing', () => {
    expect(ruleProgram('What is the capital of France?').output).toBe('Paris')
    expect(ruleProgram('What is the capital of Freedonia?').output).toContain('KeyError')
    expect(ruleProgram("Which city is France's capital?").verdict).toBe('fails')
  })
})

describe('preset questions', () => {
  it('every system fails on at least one question, and no two questions have the same pattern', () => {
    const patterns = QUESTIONS.map((q) => { const r = askAll(q.text, q); return [r.db, r.search, r.program, r.llm!].map((o) => o.verdict) })
    for (let s = 0; s < 4; s++) expect(patterns.some((p) => p[s] !== 'works')).toBe(true)
    for (let s = 0; s < 4; s++) expect(patterns.some((p) => p[s] === 'works')).toBe(true)
    expect(new Set(patterns.map((p) => p.join())).size).toBeGreaterThanOrEqual(5)
  })
  it('free text gets live results and no invented LLM answer', () => expect(askAll('hello there').llm).toBeNull())
})
