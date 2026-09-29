import { describe, expect, it } from 'vitest'
import { LESSONS } from './curriculum'
import { FAST_TRACK, fastTrackMinutes, nextOnFastTrack, skippedBefore } from './tracks'

describe('the fast track', () => {
  it('uses real lessons in course order', () => {
    const idx = FAST_TRACK.map((id) => LESSONS.findIndex((l) => l.id === id))
    expect(idx.every((i) => i >= 0)).toBe(true)
    expect([...idx].sort((a, b) => a - b)).toEqual(idx)
  })
  it('takes about 8 hours and ends at Build GPT', () => {
    expect(fastTrackMinutes()).toBeGreaterThanOrEqual(6 * 60)
    expect(fastTrackMinutes()).toBeLessThanOrEqual(9 * 60)
    expect(nextOnFastTrack('build-gpt')).toBeNull()
    expect(nextOnFastTrack('softmax')?.id).toBe('gradient-descent')
  })
  it('knows what it skipped', () => {
    expect(skippedBefore('softmax').map((l) => l.id)).toEqual(['matrices'])
  })
})
