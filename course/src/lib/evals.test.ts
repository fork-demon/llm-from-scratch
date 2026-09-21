import { describe, expect, it } from 'vitest'
import { makeRng } from './rng'
import {
  CATEGORIES, DEFAULT_CONFIG, GOLDEN_SET, HUMAN_LABELS_DEFAULT, accuracy, bootstrapInterval, byCategory, cohensKappa, compare,
  evaluate, independentDiscordance, simulateScores, nPaired, nUnpaired, normalInterval, passAtK, recallAtK, signTest, simulatePairedPower,
  simulateUnpairedPower, standardError, wilsonInterval, type ScorerName,
} from './evals'
import { CORPUS } from './rag'

describe('golden set', () => {
  it('has 24 unique items in four categories, and every gold sentence exists in its document', () => {
    expect(GOLDEN_SET).toHaveLength(24)
    expect(new Set(GOLDEN_SET.map((i) => i.id)).size).toBe(24)
    expect(new Set(GOLDEN_SET.map((i) => i.cat))).toEqual(new Set(CATEGORIES))
    for (const it of GOLDEN_SET) {
      if (it.gold) expect(CORPUS.find((d) => d.name === it.source)!.text).toContain(it.gold)
    }
  })
})

describe('reproduces eval_harness.py', () => {
  it('same pass counts per scorer as the Python (4, 12, 16 of 24)', () => {
    const passed = (s: ScorerName) => evaluate(DEFAULT_CONFIG, s).filter((r) => r.passed).length
    expect([passed('exact'), passed('substring'), passed('rubric')]).toEqual([4, 12, 16])
  })
  it('same failing items, categories, recall and diagnoses', () => {
    const rows = evaluate()
    expect(rows.filter((r) => !r.passed).map((r) => r.item.id)).toEqual(['p5', 'p6', 'p7', 'p8', 'u2', 'a1', 'a3', 'a4'])
    expect(byCategory(rows)).toEqual({ direct: { passed: 8, total: 8 }, paraphrase: { passed: 4, total: 8 }, unanswerable: { passed: 3, total: 4 }, adversarial: { passed: 1, total: 4 } })
    expect(recallAtK(rows)).toEqual({ hits: 16, total: 17 })
    const d = Object.fromEntries(rows.map((r) => [r.item.id, r.diagnosis]))
    expect([d.p5, d.p6, d.p7, d.u2, d.a4, d.d1]).toEqual(['wrong-sentence', 'retrieval-miss', 'refused-with-chunk', 'should-refuse', 'scorer', 'ok'])
  })
  it('same bootstrap interval as the Python: 45.8% to 83.3% (same generator, same seed)', () => {
    const [lo, hi] = bootstrapInterval(evaluate().map((r) => Number(r.passed)))
    expect(lo).toBeCloseTo(11 / 24, 10)
    expect(hi).toBeCloseTo(20 / 24, 10)
  })
  it('same scorer-versus-human agreement: kappa 0.15, 0.58, 0.90', () => {
    const human = GOLDEN_SET.map((i) => HUMAN_LABELS_DEFAULT[i.id])
    const kappa = (s: ScorerName) => cohensKappa(evaluate(DEFAULT_CONFIG, s).map((r) => r.passed), human)
    expect(kappa('exact')).toBeCloseTo(0.15, 2)
    expect(kappa('substring')).toBeCloseTo(0.58, 2)
    expect(kappa('rubric')).toBeCloseTo(0.9, 2)
  })
  it('same regression verdicts', () => {
    const base = evaluate()
    const small = compare(base, evaluate({ ...DEFAULT_CONFIG, sentencesPerChunk: 1, overlap: 0 }))
    expect([small.onlyA, small.onlyB, small.p, small.verdict]).toEqual([2, 0, 0.5, 'cannot tell'])
    expect(small.ci[0]).toBeCloseTo(-5 / 24, 10)
    expect(small.ci[1]).toBe(0)
    const big = compare(base, evaluate({ ...DEFAULT_CONFIG, threshold: 0.9 }))
    expect([big.onlyA, big.onlyB, big.verdict]).toEqual([12, 2, 'A is better'])
    expect(big.p).toBeCloseTo(0.013, 3)
    expect(big.ci[0]).toBeCloseTo(-16 / 24, 10)
    expect(big.ci[1]).toBeCloseTo(-4 / 24, 10)
    const k1 = compare(base, evaluate({ ...DEFAULT_CONFIG, k: 1 }))
    expect([k1.onlyA, k1.onlyB, k1.p]).toEqual([0, 0, 1])
  })
})

describe('statistics against closed forms', () => {
  it('standard error and the ±11 points of the lesson', () => {
    expect(standardError(0.8, 50)).toBeCloseTo(Math.sqrt(0.16 / 50), 12)
    const [lo, hi] = normalInterval(0.8, 50)
    expect((hi - 0.8) * 100).toBeCloseTo(11.09, 2)
    expect((0.8 - lo) * 100).toBeCloseTo(11.09, 2)
    expect((normalInterval(0.8, 1000)[1] - 0.8) * 100).toBeCloseTo(2.48, 2)
  })
  it('quadrupling n halves the interval', () => {
    const w = (n: number) => normalInterval(0.7, n)[1] - 0.7
    expect(w(400) / w(100)).toBeCloseTo(0.5, 10)
  })
  it('Wilson interval: textbook value, and it never collapses at p = 1', () => {
    const [lo, hi] = wilsonInterval(0.8, 50)
    expect(lo).toBeCloseTo(0.67, 3)
    expect(hi).toBeCloseTo(0.888, 3)
    const edge = wilsonInterval(1, 10)
    expect(edge[1]).toBeCloseTo(1, 10)
    expect(edge[0]).toBeGreaterThan(0.7)
    expect(normalInterval(1, 10)).toEqual([1, 1]) // the textbook formula claims certainty from 10 items
  })
  it('bootstrap agrees with the analytic interval on a known case (p = 0.8, n = 200)', () => {
    const values = [...Array(160).fill(1), ...Array(40).fill(0)]
    const [lo, hi] = bootstrapInterval(values, 4000, 1)
    const [aLo, aHi] = normalInterval(0.8, 200)
    expect(Math.abs(lo - aLo)).toBeLessThan(0.015)
    expect(Math.abs(hi - aHi)).toBeLessThan(0.015)
    expect(bootstrapInterval(values, 500, 3)).toEqual(bootstrapInterval(values, 500, 3))
  })
  it('the generator matches the reference values asserted in the Python test', () => {
    const rng = makeRng(42)
    expect([rng.next(), rng.next(), rng.next()].map((v) => Number(v.toFixed(10)))).toEqual([0.6011037519, 0.448290559, 0.8524657935])
  })
  it('sign test exact values', () => {
    expect(signTest(0, 0)).toBe(1)
    expect(signTest(2, 0)).toBe(0.5)
    expect(signTest(3, 3)).toBe(1)
    expect(signTest(12, 2)).toBeCloseTo((2 * (1 + 14 + 91)) / 2 ** 14, 12)
    expect(signTest(0, 6)).toBeCloseTo(2 / 64, 12)
    expect(signTest(5, 0)).toBeGreaterThan(0.05) // five wins out of five is still not enough
  })
  it('Cohen kappa closed form', () => {
    const x = [...Array(25).fill(true), ...Array(25).fill(false)]
    const y = [...Array(20).fill(true), ...Array(5).fill(false), ...Array(10).fill(true), ...Array(15).fill(false)]
    expect(cohensKappa(x, y)).toBeCloseTo(0.4, 12)
    expect(cohensKappa(x, x)).toBe(1)
  })
  it('pass@k equals 1 - C(n-c,k)/C(n,k)', () => {
    expect(passAtK(10, 3, 1)).toBeCloseTo(0.3, 12)
    expect(passAtK(10, 3, 5)).toBeCloseTo(1 - 21 / 252, 12)
    expect(passAtK(200, 20, 10)).toBeCloseTo(0.6602, 4)
    expect(passAtK(5, 0, 3)).toBe(0)
    expect(passAtK(5, 4, 3)).toBe(1)
    expect(accuracy([{ passed: true }, { passed: false }])).toBe(0.5)
  })
})

describe('sample-size planning', () => {
  it('simulated evals scatter around the truth by about one standard error', () => {
    const scores = simulateScores(0.8, 50, 4000, 3)
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length
    const sd = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / scores.length)
    expect(mean).toBeCloseTo(0.8, 2)
    expect(sd).toBeCloseTo(standardError(0.8, 50), 2)
    const inside = scores.filter((v) => Math.abs(v - 0.8) <= 1.959964 * standardError(0.8, 50)).length / scores.length
    expect(inside).toBeGreaterThan(0.92)
    expect(inside).toBeLessThan(0.98)
  })
  it('80% versus 85%: about 900 items per set unpaired', () => {
    expect(nUnpaired(0.8, 0.85)).toBe(906)
    expect(nUnpaired(0.8, 0.8)).toBe(Infinity)
  })
  it('paired needs far fewer when the systems mostly agree, and no fewer when they are unrelated', () => {
    expect(nPaired(0.05, 0.05)).toBe(155)
    expect(nPaired(0.05, 0.1)).toBe(312)
    const psi = independentDiscordance(0.8, 0.85)
    expect(psi).toBeCloseTo(0.29, 12)
    expect(Math.abs(nPaired(0.05, psi) - nUnpaired(0.8, 0.85))).toBeLessThan(10)
    expect(nPaired(0.05, 0.01)).toBe(nPaired(0.05, 0.05)) // disagreement can never be below the gap
  })
  it('the formulas deliver roughly the promised 80% power in simulation', () => {
    expect(simulatePairedPower(nPaired(0.05, 0.1), 0.05, 0.1)).toBeGreaterThan(0.75)
    expect(simulatePairedPower(nPaired(0.05, 0.1), 0.05, 0.1)).toBeLessThan(0.86)
    expect(simulateUnpairedPower(nUnpaired(0.8, 0.85), 0.8, 0.85, 1000)).toBeGreaterThan(0.75)
    expect(simulateUnpairedPower(nUnpaired(0.8, 0.85), 0.8, 0.85, 1000)).toBeLessThan(0.86)
    // and 20 items cannot tell 80% from 85%
    expect(simulateUnpairedPower(20, 0.8, 0.85, 2000)).toBeLessThan(0.1)
    expect(simulatePairedPower(20, 0.05, 0.1)).toBeLessThan(0.15)
  })
})
