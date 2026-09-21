import { describe, expect, it } from 'vitest'
import { argmax } from './math'
import {
  ATTRIBUTES, DIM, FACTS, accuracy, addNoise, contributions, damageCurve, initWeights, knockOut, logitsFor,
  meanCorrectProb, meanLoss, predict, subjectVectors, train, trainStep, trainWithout, unseenSubjects, weightRms, weightsChanged,
} from './assoc'

const X = subjectVectors()
const trained = train(initWeights(), X, 400).W
const sum = (v: number[]) => v.reduce((a, b) => a + b, 0)

describe('assoc: a tiny associative memory in one weight matrix', () => {
  it('starts out knowing nothing: uniform answers, loss = ln(8)', () => {
    expect(meanLoss(initWeights(), X)).toBeCloseTo(Math.log(8), 2)
  })

  it('is deterministic for a seed', () => {
    expect(subjectVectors()).toEqual(X)
    expect(train(initWeights(), X, 20).W).toEqual(train(initWeights(), X, 20).W)
  })

  it('gradient descent lowers the loss at every step', () => {
    let W = initWeights()
    let prev = Infinity
    for (let s = 0; s < 50; s++) {
      const r = trainStep(W, X)
      expect(r.loss).toBeLessThan(prev)
      prev = r.loss
      W = r.W
    }
  })

  it('analytic gradient matches a numerical nudge', () => {
    const W = train(initWeights(), X, 5).W
    const eps = 1e-5
    const bumped = W.map((r) => r.slice())
    bumped[3][2] += eps
    const numeric = (meanLoss(bumped, X) - meanLoss(W, X)) / eps
    const lr = 1
    const analytic = (W[3][2] - trainStep(W, X, lr).W[3][2]) / lr
    expect(analytic).toBeCloseTo(numeric, 4)
  })

  it('the trained model recalls every pair', () => {
    expect(accuracy(trained, X)).toBe(1)
    FACTS.forEach((f, i) => {
      const p = predict(trained, X[i])
      expect(ATTRIBUTES[argmax(p)]).toBe(f.attribute)
      expect(p[i]).toBeGreaterThan(0.85)
    })
    // the numbers quoted in the lesson
    expect(meanCorrectProb(trained, X)).toBeCloseTo(0.94, 2)
    expect(meanLoss(trained, X)).toBeCloseTo(0.06, 2)
  })

  it('has no record per fact: 8 facts share 80 weights and every weight is used by every answer', () => {
    expect(trained.length * trained[0].length).toBe(DIM * FACTS.length)
    X.forEach((x, i) => {
      const c = contributions(trained, x, i)
      expect(c.every((v) => v !== 0)).toBe(true)
      expect(sum(c)).toBeCloseTo(logitsFor(trained, x)[i], 10)
    })
  })

  it('leaving one fact out of training changes weights all over the matrix, not one row or column', () => {
    const without = trainWithout(initWeights(), X, 0, 400) // never sees France -> Paris
    const changed = weightsChanged(trained, without, 0.1 * weightRms(trained))
    console.log('weights changed when France is left out:', changed, 'of 80; P(Paris | France) =', predict(without, X[0])[0].toFixed(3))
    expect(changed).toBeGreaterThan(40)
    // the other seven facts are still recalled
    expect(X.slice(1).every((x, i) => argmax(predict(without, x)) === i + 1)).toBe(true)
  })

  it('an unseen subject still gets a full probability distribution, often a confident one', () => {
    const unseen = unseenSubjects(X)
    for (const u of unseen) {
      const p = predict(trained, u.vector)
      expect(sum(p)).toBeCloseTo(1, 10)
      expect(Math.max(...p)).toBeGreaterThan(3 / 8) // far above the 1/8 of "no idea"
    }
    const austria = predict(trained, unseen[0].vector)
    expect(ATTRIBUTES[argmax(austria)]).toBe('Berlin') // fluent, confident, wrong
    expect(Math.max(...austria)).toBeGreaterThan(0.9)
  })

  it('knocking out weights degrades all facts gradually, not one fact at a time', () => {
    const curve = damageCurve(trained, X, [0, 0.2, 0.4, 0.6, 0.8, 1], 'knockout')
    for (let i = 1; i < curve.length; i++) {
      expect(curve[i].correctProb).toBeLessThan(curve[i - 1].correctProb)
      expect(curve[i].accuracy).toBeLessThanOrEqual(curve[i - 1].accuracy + 1e-9)
    }
    expect(curve[0].accuracy).toBe(1)
    expect(curve[1].accuracy).toBeGreaterThan(0.95) // 20% of the weights gone, facts mostly intact
    expect(curve[5].correctProb).toBeCloseTo(1 / 8, 10) // everything zero: uniform
    // at 30% damage with one fixed pattern, EVERY fact has lost confidence or stayed close; none is simply deleted
    const D = knockOut(trained, 0.3, 3)
    const drops = X.map((x, i) => predict(trained, x)[i] - predict(D, x)[i])
    expect(drops.filter((d) => d > 0.01).length).toBeGreaterThanOrEqual(6)
    expect(accuracy(D, X)).toBe(1)
  })

  it('a larger knockout fraction removes a superset of the weights', () => {
    const a = knockOut(trained, 0.2, 9)
    const b = knockOut(trained, 0.5, 9)
    a.forEach((row, i) => row.forEach((w, j) => { if (w === 0) expect(b[i][j]).toBe(0) }))
    expect(b.flat().filter((w) => w === 0).length).toBe(40)
  })

  it('noise degrades the memory monotonically in expectation', () => {
    const curve = damageCurve(trained, X, [0, 0.5, 1, 2, 3], 'noise')
    for (let i = 1; i < curve.length; i++) expect(curve[i].correctProb).toBeLessThan(curve[i - 1].correctProb)
    expect(addNoise(trained, 0, 1)).toEqual(trained)
  })

  it('reproduces the numbers printed in the why-llms-know lesson', () => {
    const u = unseenSubjects(X)
    const france = logitsFor(trained, X[0])
    expect(france[0]).toBeCloseTo(4.47, 2)
    expect(predict(trained, X[0])[0]).toBeCloseTo(0.921, 3)
    expect(france.reduce((s, z) => s + Math.exp(z), 0)).toBeCloseTo(94.81, 1)
    const austria = logitsFor(trained, u[0].vector)
    expect(austria[6]).toBeCloseTo(4.52, 2)
    expect(predict(trained, u[0].vector)[6]).toBeCloseTo(0.928, 3)
    expect(austria.reduce((s, z) => s + Math.exp(z), 0)).toBeCloseTo(98.91, 1)
    const atlantis = predict(trained, u[2].vector)
    expect(ATTRIBUTES[argmax(atlantis)]).toBe('Cairo')
    expect(atlantis[2]).toBeCloseTo(0.54, 2)
    // Break it: 30% knocked out with the first damage pattern (seed 3)
    const D = knockOut(trained, 0.3, 3)
    expect(accuracy(D, X)).toBe(1)
    expect(meanCorrectProb(D, X)).toBeCloseTo(0.81, 2)
  })
})
