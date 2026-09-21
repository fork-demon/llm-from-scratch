import { describe, expect, it } from 'vitest'
import { completeExercise, completeLesson, emptyProgress, mergeProgress, nextUp, parseProgress, parseProgressFile, percentComplete, recordQuiz } from './progress'

describe('progress', () => {
  const ids = ['a', 'b', 'c', 'd']
  it('starts empty and survives corrupt storage', () => {
    expect(parseProgress(null)).toEqual(emptyProgress())
    expect(parseProgress('{not json')).toEqual(emptyProgress())
    expect(parseProgress('{"completed":{"a":true}}').quiz).toEqual({})
  })
  it('tracks completion and percent', () => {
    let p = completeLesson(emptyProgress(), 'a')
    p = completeLesson(p, 'b')
    expect(percentComplete(p, ids)).toBe(50)
    p = completeLesson(p, 'a', false)
    expect(percentComplete(p, ids)).toBe(25)
  })
  it('continues at the first unfinished lesson', () => {
    const p = completeLesson(completeLesson(emptyProgress(), 'a'), 'c')
    expect(nextUp(p, ids)).toBe('b')
  })
  it('keeps the best quiz score', () => {
    let p = recordQuiz(emptyProgress(), 'q', 3, 4)
    p = recordQuiz(p, 'q', 2, 4)
    expect(p.quiz.q.correct).toBe(3)
    p = recordQuiz(p, 'q', 4, 4)
    expect(p.quiz.q.correct).toBe(4)
  })
  it('does not mutate previous state', () => {
    const a = emptyProgress()
    const b = completeExercise(a, 'x')
    expect(a.exercises).toEqual({})
    expect(completeExercise(b, 'x')).toBe(b)
  })
})

describe('export and import', () => {
  it('rejects a file from somewhere else', () => {
    expect(() => parseProgressFile('{"app":"something-else"}')).toThrow(/not saved by this course/)
    expect(() => parseProgressFile('{"app":"llm-from-first-principles"}')).toThrow(/no progress/)
    expect(() => parseProgressFile('not json')).toThrow()
  })
  it('accepts its own file', () => {
    const f = parseProgressFile(JSON.stringify({ app: 'llm-from-first-principles', version: 1, progress: emptyProgress(), drafts: {} }))
    expect(f.progress).toEqual(emptyProgress())
  })
  it('merges without losing anything, keeping the better quiz score', () => {
    const mine = { ...emptyProgress(), completed: { a: true as const }, quiz: { q: { correct: 2, total: 4 } } }
    const theirs = { ...emptyProgress(), completed: { b: true as const }, quiz: { q: { correct: 4, total: 4 } }, last: 'b' }
    const m = mergeProgress(mine, theirs)
    expect(m.completed).toEqual({ a: true, b: true })
    expect(m.quiz.q.correct).toBe(4)
    expect(m.last).toBe('b')
  })
  it('does not downgrade a better local score', () => {
    const mine = { ...emptyProgress(), quiz: { q: { correct: 4, total: 4 } } }
    const theirs = { ...emptyProgress(), quiz: { q: { correct: 1, total: 4 } } }
    expect(mergeProgress(mine, theirs).quiz.q.correct).toBe(4)
  })
})
