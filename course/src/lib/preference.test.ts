import { describe, expect, it } from 'vitest'
import { argmax } from './math'
import { CANDIDATES, CANDIDATE_PROMPT, COMPARISONS, comparisonLoss, featurise, klDivergence, personaChoice, preferProb, reward, trainOnChoices, tunedPolicy, updateWeights } from './preference'

const choicesOf = (persona: 'longer' | 'correct' | 'polite') => COMPARISONS.map((c) => personaChoice(c, persona))
const candidateRewards = (w: number[]) => CANDIDATES.map((c) => reward(w, featurise(c.text, CANDIDATE_PROMPT.key)))

describe('features', () => {
  it('length, polite marker, contains-the-answer', () => {
    expect(featurise('Paris.', 'paris')).toEqual([1 / 30, 0, 1])
    expect(featurise('Happy to help! 2 + 2 = 4.', '4')).toEqual([8 / 30, 1, 1])
    expect(featurise('Ag.', 'au')[2]).toBe(0)
    expect(featurise('It is 1440.', '144')[2]).toBe(0) // whole-token match only
    expect(featurise(Array(50).fill('word').join(' '), 'x')[0]).toBe(1) // capped
  })
  it('every comparison differs in at least one feature', () => {
    for (const c of COMPARISONS) expect(featurise(c.a, c.key)).not.toEqual(featurise(c.b, c.key))
  })
})

describe('Bradley-Terry', () => {
  it('matches the numbers printed in the lesson', () => {
    expect(preferProb(2.0, 0.5)).toBeCloseTo(0.8176, 4)
    expect(-Math.log(preferProb(2.0, 0.5))).toBeCloseTo(0.2014, 4)
    expect(-Math.log(preferProb(0.5, 2.0))).toBeCloseTo(1.7014, 4)
    expect(preferProb(1.2, 0.2)).toBeCloseTo(0.731, 3)
    expect(preferProb(1, 1)).toBe(0.5)
  })
  it('only the difference of rewards matters', () => {
    expect(preferProb(3, 1)).toBeCloseTo(preferProb(103, 101), 12)
  })
  it('a gradient step lowers the loss on that comparison and agrees with the numerical gradient', () => {
    const w = [0.3, -0.2, 0.1]
    const ch = [0.1, 0, 1]
    const rej = [1, 1, 0]
    expect(comparisonLoss(updateWeights(w, ch, rej, 0.5), ch, rej)).toBeLessThan(comparisonLoss(w, ch, rej))
    const eps = 1e-5
    const stepped = updateWeights(w, ch, rej, 1)
    w.forEach((_, i) => {
      const up = w.slice(); up[i] += eps
      const dn = w.slice(); dn[i] -= eps
      const g = (comparisonLoss(up, ch, rej) - comparisonLoss(dn, ch, rej)) / (2 * eps)
      expect(stepped[i] - w[i]).toBeCloseTo(-g, 5)
    })
  })
})

describe('learning from consistent preferences', () => {
  it('always preferring the answer that contains the answer makes that weight the largest', () => {
    const w = trainOnChoices(COMPARISONS, choicesOf('correct'))
    expect(w[2]).toBeGreaterThan(1)
    expect(argmax(w)).toBe(2)
    // numbers quoted in the lesson
    expect(w[0]).toBeCloseTo(-0.88, 2)
    expect(w[1]).toBeCloseTo(-0.70, 2)
    expect(w[2]).toBeCloseTo(1.68, 2)
  })
  it('always preferring the longer answer makes the reward model love length, not correctness', () => {
    const w = trainOnChoices(COMPARISONS, choicesOf('longer'))
    expect(w[0]).toBeGreaterThan(1)
    expect(Math.abs(w[2])).toBeLessThan(0.2)
    expect(w[0]).toBeCloseTo(1.57, 2)
    expect(w[1]).toBeCloseTo(1.51, 2)
    expect(w[2]).toBeCloseTo(-0.09, 2)
    // optimising against it picks a long answer that never says "Paris"
    const best = argmax(candidateRewards(w))
    expect(featurise(CANDIDATES[best].text, 'paris')[2]).toBe(0)
  })
  it('more passes over the same choices push the weights further the same way', () => {
    const w1 = trainOnChoices(COMPARISONS, choicesOf('correct'), 1, 1)
    const w5 = trainOnChoices(COMPARISONS, choicesOf('correct'), 1, 5)
    expect(w5[2]).toBeGreaterThan(w1[2])
  })
})

describe('the lazy-rater exercise in the lesson', () => {
  const pParis = (choices: string) => {
    const w = trainOnChoices(COMPARISONS, choices.split('') as ('a' | 'b')[])
    const pol = tunedPolicy(CANDIDATES.map((c) => c.refProb), candidateRewards(w), 0.5)
    return { w, p: pol.reduce((s, x, j) => s + x * featurise(CANDIDATES[j].text, 'paris')[2], 0), top: argmax(pol) }
  }
  it('careful rater (correct first, thorough on ties): 99% Paris', () => {
    const r = pParis('aabbbabb')
    expect(r.w[2]).toBeCloseTo(1.74, 2)
    expect(r.p).toBeCloseTo(0.99, 2)
  })
  it('two lazy choices (5 and 7) are enough: 41% Paris, long non-answer on top', () => {
    const r = pParis('aabbaaab')
    expect(r.p).toBeCloseTo(0.41, 2)
    expect(r.top).toBe(3)
  })
  it('three lazy choices equal the always-longer rater: 20% Paris', () => {
    expect('babbaaab').toBe(choicesOf('longer').join(''))
    expect(pParis('babbaaab').p).toBeCloseTo(0.2, 2)
  })
})

describe('KL-leashed policy', () => {
  const ref = CANDIDATES.map((c) => c.refProb)
  it('reference probabilities sum to 1', () => expect(ref.reduce((a, b) => a + b, 0)).toBeCloseTo(1))
  it('is a distribution, stays at the SFT model for a tight leash, chases reward for a loose one', () => {
    const r = candidateRewards(trainOnChoices(COMPARISONS, choicesOf('longer')))
    const tight = tunedPolicy(ref, r, 1000)
    const loose = tunedPolicy(ref, r, 0.1)
    expect(tight.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
    tight.forEach((p, i) => expect(p).toBeCloseTo(ref[i], 2))
    expect(argmax(loose)).toBe(argmax(r))
    expect(loose[argmax(r)]).toBeGreaterThan(0.99)
    expect(klDivergence(loose, ref)).toBeGreaterThan(klDivergence(tight, ref))
    expect(klDivergence(ref, ref)).toBeCloseTo(0)
  })
  it('expected reward never decreases as the leash loosens', () => {
    const r = candidateRewards(trainOnChoices(COMPARISONS, choicesOf('longer')))
    const exp = (beta: number) => tunedPolicy(ref, r, beta).reduce((s, p, i) => s + p * r[i], 0)
    expect(exp(0.5)).toBeGreaterThan(exp(2))
    expect(exp(2)).toBeGreaterThan(exp(50))
  })
})
