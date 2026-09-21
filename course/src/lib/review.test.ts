import { describe, expect, it } from 'vitest'
import { GAPS, dueItems, gapFor, parseReview, recordAnswer } from './review'

const NOW = 1_700_000_000_000

describe('review queue', () => {
  it('survives missing or corrupt storage', () => {
    expect(parseReview(null)).toEqual({})
    expect(parseReview('not json')).toEqual({})
    expect(parseReview('[1,2]')).toEqual({})
    expect(parseReview('{"a":{"nope":1}}')).toEqual({})
  })
  it('ignores a question answered right the first time', () => {
    expect(recordAnswer({}, 'q#0', 'attention', true, NOW)).toEqual({})
  })
  it('queues a wrong answer to come back soon', () => {
    const s = recordAnswer({}, 'q#0', 'attention', false, NOW)
    expect(s['q#0'].wrongCount).toBe(1)
    expect(s['q#0'].dueAt).toBe(NOW + GAPS[0])
    expect(dueItems(s, NOW)).toHaveLength(0)
    expect(dueItems(s, NOW + GAPS[0])).toHaveLength(1)
  })
  it('spaces it out further after each correct answer', () => {
    let s = recordAnswer({}, 'q#0', 'attention', false, NOW)
    s = recordAnswer(s, 'q#0', 'attention', true, NOW)
    expect(s['q#0'].dueAt).toBe(NOW + GAPS[1])
    s = recordAnswer(s, 'q#0', 'attention', true, NOW)
    expect(s['q#0'].dueAt).toBe(NOW + GAPS[2])
  })
  it('retires a question answered right enough times', () => {
    let s = recordAnswer({}, 'q#0', 'attention', false, NOW)
    for (let i = 0; i < GAPS.length; i++) s = recordAnswer(s, 'q#0', 'attention', true, NOW)
    expect(s['q#0']).toBeUndefined()
  })
  it('a wrong answer resets the streak', () => {
    let s = recordAnswer({}, 'q#0', 'attention', false, NOW)
    s = recordAnswer(s, 'q#0', 'attention', true, NOW)
    s = recordAnswer(s, 'q#0', 'attention', false, NOW)
    expect(s['q#0'].rightStreak).toBe(0)
    expect(s['q#0'].wrongCount).toBe(2)
    expect(s['q#0'].dueAt).toBe(NOW + GAPS[0])
  })
  it('gapFor grows and then stops', () => {
    expect(gapFor(0)).toBe(GAPS[0])
    expect(gapFor(99)).toBe(GAPS[GAPS.length - 1])
  })
})
