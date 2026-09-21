import { describe, expect, it } from 'vitest'
import { makeRng } from './rng'
import {
  analytic, bestOfNAccuracy, buildTrace, curves, distractorRate, majorityCanWin, majorityVoteAccuracy,
  precisionWhenAccepted, roughTokens, simulate, tokensSpent, winGivenVotes,
} from './testTime'

/** Brute force: enumerate every one of the (m+1)^N answer sequences. */
const bruteMajority = (p: number, m: number, N: number): number => {
  const probs = [p, ...new Array(m).fill((1 - p) / m)]
  let acc = 0
  const counts = new Array(m + 1).fill(0)
  const rec = (i: number, pr: number) => {
    if (i === N) {
      const best = Math.max(...counts)
      if (counts[0] === best) acc += pr / counts.filter((c) => c === best).length
      return
    }
    for (let x = 0; x <= m; x++) {
      counts[x]++
      rec(i + 1, pr * probs[x])
      counts[x]--
    }
  }
  rec(0, 1)
  return acc
}

describe('N = 1: every strategy is just one attempt', () => {
  it('all equal p', () => {
    for (const p of [0.1, 0.4, 0.75]) {
      for (const a of [0.5, 0.8, 1]) {
        const r = analytic({ p, m: 4, a }, 1)
        expect(r.single).toBeCloseTo(p, 12)
        expect(r.majority).toBeCloseTo(p, 12)
        expect(r.bestOfN).toBeCloseTo(p, 12)
      }
    }
  })
})

describe('best-of-N with a verifier', () => {
  it('perfect verifier: 1 - (1 - p)^N', () => {
    for (const p of [0.05, 0.3, 0.6]) for (const N of [1, 2, 4, 8, 16]) expect(bestOfNAccuracy(p, 1, N)).toBeCloseTo(1 - (1 - p) ** N, 12)
    expect(bestOfNAccuracy(0.4, 1, 4)).toBeCloseTo(0.8704, 10)
  })
  it('coin-flip verifier adds nothing', () => {
    for (const N of [1, 3, 10]) expect(bestOfNAccuracy(0.3, 0.5, N)).toBeCloseTo(0.3, 12)
  })
  it('imperfect verifier saturates at P(correct | accepted), below 1', () => {
    const ceiling = precisionWhenAccepted(0.3, 0.9)
    expect(ceiling).toBeCloseTo(0.27 / 0.34, 12)
    expect(bestOfNAccuracy(0.3, 0.9, 200)).toBeCloseTo(ceiling, 6)
    expect(bestOfNAccuracy(0.3, 0.9, 200)).toBeLessThan(1)
  })
  it('grows with N when the verifier is better than chance', () => {
    let prev = 0
    for (let N = 1; N <= 20; N++) {
      const v = bestOfNAccuracy(0.3, 0.8, N)
      expect(v).toBeGreaterThan(prev)
      prev = v
    }
  })
})

describe('majority vote (exact DP)', () => {
  it('one distractor, N = 3: p^3 + 3 p^2 (1 - p)', () => {
    for (const p of [0.4, 0.5, 0.7]) expect(majorityVoteAccuracy(p, 1, 3)).toBeCloseTo(p ** 3 + 3 * p * p * (1 - p), 12)
    expect(majorityVoteAccuracy(0.4, 1, 3)).toBeCloseTo(0.352, 12)
  })
  it('N = 2 never helps: a 1-1 tie is a coin flip, so accuracy stays p', () => {
    for (const m of [1, 3, 6]) expect(majorityVoteAccuracy(0.35, m, 2)).toBeCloseTo(0.35, 12)
  })
  it('matches brute-force enumeration', () => {
    for (const [p, m, N] of [[0.4, 4, 3], [0.4, 4, 5], [0.3, 2, 6], [0.2, 3, 7], [0.55, 5, 4]] as const) {
      expect(majorityVoteAccuracy(p, m, N)).toBeCloseTo(bruteMajority(p, m, N), 10)
    }
  })
  it('helps when the right answer is the most common one, hurts when it is not', () => {
    // p = 0.4 against 4 distractors at 0.15 each: the right answer is the mode
    expect(majorityCanWin(0.4, 4)).toBe(true)
    expect(majorityVoteAccuracy(0.4, 4, 15)).toBeGreaterThan(0.4)
    expect(majorityVoteAccuracy(0.4, 4, 31)).toBeGreaterThan(majorityVoteAccuracy(0.4, 4, 15))
    // p = 0.4 against ONE distractor at 0.6: the wrong answer is the mode
    expect(majorityCanWin(0.4, 1)).toBe(false)
    expect(majorityVoteAccuracy(0.4, 1, 15)).toBeLessThan(0.4)
    expect(majorityVoteAccuracy(0.4, 1, 31)).toBeLessThan(majorityVoteAccuracy(0.4, 1, 15))
  })
  it('winGivenVotes edge cases', () => {
    expect(winGivenVotes(3, 0, 4)).toBe(1)
    expect(winGivenVotes(0, 2, 4)).toBe(0)
    expect(winGivenVotes(1, 1, 4)).toBeCloseTo(0.5, 12) // 1-1 tie
  })
  it('distractor rate', () => {
    expect(distractorRate(0.4, 4)).toBeCloseTo(0.15, 12)
  })
})

describe('Monte-Carlo agrees with the exact values', () => {
  it('within sampling noise, and is reproducible from the seed', () => {
    const params = { p: 0.4, m: 4, a: 0.85 }
    const N = 7
    const mc = simulate(params, N, 20000, makeRng(42))
    const ex = analytic(params, N)
    expect(mc.single).toBeCloseTo(ex.single, 1.7)
    expect(Math.abs(mc.majority - ex.majority)).toBeLessThan(0.015)
    expect(Math.abs(mc.bestOfN - ex.bestOfN)).toBeLessThan(0.015)
    expect(simulate(params, N, 500, makeRng(7))).toEqual(simulate(params, N, 500, makeRng(7)))
  })
})

describe('cost and curves', () => {
  it('tokens grow linearly with N', () => {
    expect(tokensSpent(1, 400)).toBe(400)
    expect(tokensSpent(16, 400)).toBe(6400)
  })
  it('curves has one entry per N', () => {
    const c = curves({ p: 0.4, m: 4, a: 0.9 }, 12)
    expect(c).toHaveLength(12)
    expect(c[0].majority).toBeCloseTo(0.4, 12)
  })
})

describe('the step-by-step puzzle', () => {
  it('builds the trace used in the lesson', () => {
    const t = buildTrace(17, [{ kind: 'times', by: 3 }, { kind: 'plus', by: 28 }, { kind: 'times', by: 2 }, { kind: 'minus', by: 45 }])
    expect(t.map((s) => s.after)).toEqual([51, 79, 158, 113])
    expect(t[1].text).toBe('51 + 28 = 79')
    expect(roughTokens(t[1].text)).toBe(5)
  })
})
