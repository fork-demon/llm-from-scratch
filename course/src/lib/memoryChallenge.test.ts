import { describe, expect, it } from 'vitest'
import { ARROWS, CHALLENGE_SECONDS, FULL_PIPELINE, formatTime, remainingSeconds, selfScore, verdictFor } from './memoryChallenge'
import { LESSONS } from '../data/curriculum'

describe('pipeline and arrows', () => {
  it('has 12 stages and one arrow between each consecutive pair', () => {
    expect(FULL_PIPELINE.length).toBe(12)
    expect(ARROWS.length).toBe(11)
    ARROWS.forEach((a, i) => { expect(a.from).toBe(FULL_PIPELINE[i]); expect(a.to).toBe(FULL_PIPELINE[i + 1]) })
  })
  it('every arrow links to a real lesson and has a substantial model answer', () => {
    const ids = new Set(LESSONS.map((l) => l.id))
    for (const a of ARROWS) { expect(ids.has(a.lesson), a.lesson).toBe(true); expect(a.answer.length).toBeGreaterThan(100) }
  })
})

describe('timer', () => {
  it('formats mm:ss', () => {
    expect(formatTime(CHALLENGE_SECONDS)).toBe('10:00')
    expect(formatTime(605)).toBe('10:05')
    expect(formatTime(59.9)).toBe('00:59')
    expect(formatTime(-3)).toBe('00:00')
  })
  it('counts down from wall-clock time, survives pause, never goes negative', () => {
    expect(remainingSeconds(600, 0, null, 5_000)).toBe(600) // never started
    expect(remainingSeconds(600, 0, 1_000, 31_000)).toBe(570) // running for 30 s
    expect(remainingSeconds(600, 30_000, null, 999_000)).toBe(570) // paused: time banked, clock ignored
    expect(remainingSeconds(600, 30_000, 100_000, 110_500)).toBe(560) // resumed: 40.5 s used, rounds up
    expect(remainingSeconds(600, 0, 0, 10_000_000)).toBe(0)
  })
})

describe('self score', () => {
  it('gives 2 / 1 / 0 points and lists what to revisit', () => {
    const s = selfScore(['got', 'partly', 'missed', null], 4)
    expect([s.got, s.partly, s.missed, s.rated]).toEqual([1, 1, 1, 3])
    expect([s.points, s.maxPoints, s.percent]).toEqual([3, 8, 38])
    expect(s.complete).toBe(false)
    expect(s.revisit).toEqual([1, 2])
  })
  it('is complete at 100% when everything is "got"', () => {
    const s = selfScore(Array(11).fill('got'))
    expect([s.percent, s.complete, s.revisit.length]).toEqual([100, true, 0])
  })
  it('handles the empty case and picks a verdict per band', () => {
    expect(selfScore([]).percent).toBe(0)
    expect(new Set([verdictFor(90), verdictFor(70), verdictFor(20)]).size).toBe(3)
  })
})
