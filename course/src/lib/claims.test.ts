import { describe, expect, it } from 'vitest'
import { markClaims, scoreClaims, type Claim } from './claims'

const claims: Claim[] = [
  { text: 'a', level: 'established', why: '' },
  { text: 'b', level: 'model', why: '' },
  { text: 'c', level: 'research', why: '' },
]

describe('claims', () => {
  it('marks each claim and counts the score', () => {
    expect(markClaims(['established', 'research', 'research'], claims)).toEqual([true, false, true])
    expect(scoreClaims(['established', 'model', 'research'], claims)).toBe(3)
  })
  it('unanswered claims are wrong', () => {
    expect(scoreClaims([null, null, 'research'], claims)).toBe(1)
  })
})
