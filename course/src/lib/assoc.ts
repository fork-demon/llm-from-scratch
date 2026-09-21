// A tiny associative memory: ONE weight matrix that maps a "subject" vector
// (a country) to a probability distribution over "attributes" (capitals).
//   logits = x @ W        (D numbers in, C logits out)
//   probs  = softmax(logits)
// Trained by full-batch gradient descent on cross-entropy, exactly the rule from
// the backprop lesson: d_logits = probs - one_hot, d_W = X.T @ d_logits.
//
// The point of the toy: there is no row or record per fact. Every subject vector
// is dense, so every weight takes part in every answer, damage degrades all facts
// together, and an unseen subject still gets a confident answer (softmax always answers).
// Pure functions, seeded, no DOM.
import { argmax, crossEntropy, dot, norm, softmax, type Mat, type Vec } from './math'
import { makeRng } from './rng'

export const FACTS: { subject: string; attribute: string }[] = [
  { subject: 'France', attribute: 'Paris' },
  { subject: 'Japan', attribute: 'Tokyo' },
  { subject: 'Egypt', attribute: 'Cairo' },
  { subject: 'Peru', attribute: 'Lima' },
  { subject: 'Kenya', attribute: 'Nairobi' },
  { subject: 'Norway', attribute: 'Oslo' },
  { subject: 'Germany', attribute: 'Berlin' },
  { subject: 'India', attribute: 'Delhi' },
]
export const SUBJECTS = FACTS.map((f) => f.subject)
export const ATTRIBUTES = FACTS.map((f) => f.attribute)

/** Length of a subject vector. 8 facts share a 10 × 8 matrix: 80 weights, no slot per fact. */
export const DIM = 10
export const DATA_SEED = 11
export const INIT_SEED = 5
export const LEARNING_RATE = 0.5

/** A dense random vector scaled to length 1: a stand-in for a learned embedding. */
export const randomUnitVector = (rng: ReturnType<typeof makeRng>, dim = DIM): Vec => {
  const v = Array.from({ length: dim }, () => rng.normal())
  const n = norm(v) || 1
  return v.map((x) => x / n)
}

/** One fixed vector per trained subject. Same seed, same vectors, every time. */
export const subjectVectors = (seed = DATA_SEED, dim = DIM): Mat => {
  const rng = makeRng(seed)
  return FACTS.map(() => randomUnitVector(rng, dim))
}

/** Small random starting weights, shape (dim, number of attributes). */
export const initWeights = (seed = INIT_SEED, dim = DIM, nOut = FACTS.length): Mat => {
  const rng = makeRng(seed)
  return Array.from({ length: dim }, () => Array.from({ length: nOut }, () => rng.normal() * 0.01))
}

export const logitsFor = (W: Mat, x: Vec): Vec => W[0].map((_, j) => dot(x, W.map((row) => row[j])))

/** The model's whole "answer": a probability for every attribute. Always sums to 1. */
export const predict = (W: Mat, x: Vec): Vec => softmax(logitsFor(W, x))

export interface StepResult { W: Mat; loss: number }

/** One step of full-batch gradient descent. Target of row i of X is attribute i. */
export const trainStep = (W: Mat, X: Mat, lr = LEARNING_RATE, targets: number[] = X.map((_, i) => i)): StepResult => {
  const n = X.length
  const grad: Mat = W.map((row) => row.map(() => 0))
  let loss = 0
  X.forEach((x, i) => {
    const p = predict(W, x)
    loss += crossEntropy(p, targets[i])
    p.forEach((pj, j) => {
      const dLogit = pj - (j === targets[i] ? 1 : 0) // probs - one_hot
      for (let d = 0; d < x.length; d++) grad[d][j] += (x[d] * dLogit) / n
    })
  })
  return { W: W.map((row, d) => row.map((w, j) => w - lr * grad[d][j])), loss: loss / n }
}

export const train = (W: Mat, X: Mat, steps: number, lr = LEARNING_RATE): StepResult => {
  let cur: StepResult = { W, loss: meanLoss(W, X) }
  for (let s = 0; s < steps; s++) cur = trainStep(cur.W, X, lr)
  return { W: cur.W, loss: meanLoss(cur.W, X) }
}

/** Train from the same start on every fact EXCEPT one. Used to ask: which weights did that fact own? */
export const trainWithout = (W: Mat, X: Mat, leaveOut: number, steps: number, lr = LEARNING_RATE): Mat => {
  const keep = X.map((_, i) => i).filter((i) => i !== leaveOut)
  let cur = W
  for (let s = 0; s < steps; s++) cur = trainStep(cur, keep.map((i) => X[i]), lr, keep).W
  return cur
}

/** How many weights differ between two matrices by more than `threshold`. */
export const weightsChanged = (A: Mat, B: Mat, threshold: number): number =>
  A.flat().filter((a, i) => Math.abs(a - B.flat()[i]) > threshold).length

export const meanLoss = (W: Mat, X: Mat): number =>
  X.reduce((s, x, i) => s + crossEntropy(predict(W, x), i), 0) / X.length

/** Fraction of trained facts whose most likely attribute is the right one. */
export const accuracy = (W: Mat, X: Mat): number =>
  X.filter((x, i) => argmax(predict(W, x)) === i).length / X.length

/** Average probability given to the right attribute: a softer health measure than accuracy. */
export const meanCorrectProb = (W: Mat, X: Mat): number =>
  X.reduce((s, x, i) => s + predict(W, x)[i], 0) / X.length

/**
 * A fixed random order of all weight positions for a seed. Knocking out a fraction f
 * zeroes the first round(f × count) positions, so a bigger f always removes a superset.
 */
export const knockoutOrder = (rows: number, cols: number, seed: number): [number, number][] => {
  const cells: [number, number][] = []
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) cells.push([i, j])
  const rng = makeRng(seed)
  for (let i = cells.length - 1; i > 0; i--) {
    const j = rng.int(i + 1)
    ;[cells[i], cells[j]] = [cells[j], cells[i]]
  }
  return cells
}

/** Set a fraction of the weights to zero (chosen at random, but reproducibly). */
export const knockOut = (W: Mat, fraction: number, seed: number): Mat => {
  const out = W.map((r) => r.slice())
  const order = knockoutOrder(W.length, W[0].length, seed)
  const count = Math.round(Math.min(1, Math.max(0, fraction)) * order.length)
  for (let c = 0; c < count; c++) out[order[c][0]][order[c][1]] = 0
  return out
}

/** Root-mean-square size of the weights: the natural unit for "how much noise is a lot". */
export const weightRms = (W: Mat): number => {
  const flat = W.flat()
  return Math.sqrt(flat.reduce((s, w) => s + w * w, 0) / flat.length)
}

/** Add Gaussian noise to every weight. `level` is relative: 1 means noise as large as a typical weight. */
export const addNoise = (W: Mat, level: number, seed: number): Mat => {
  const rng = makeRng(seed)
  const sd = level * weightRms(W)
  return W.map((row) => row.map((w) => w + rng.normal() * sd))
}

export interface DamagePoint { amount: number; accuracy: number; correctProb: number }

/** Average health of the memory over several random damage patterns, for each damage amount. */
export const damageCurve = (W: Mat, X: Mat, amounts: number[], kind: 'knockout' | 'noise', trials = 40, seed = 1): DamagePoint[] =>
  amounts.map((amount) => {
    let acc = 0
    let prob = 0
    for (let t = 0; t < trials; t++) {
      const D = kind === 'knockout' ? knockOut(W, amount, seed + t) : addNoise(W, amount, seed + t)
      acc += accuracy(D, X)
      prob += meanCorrectProb(D, X)
    }
    return { amount, accuracy: acc / trials, correctProb: prob / trials }
  })

/**
 * Subjects the model was NEVER trained on.
 *  - "blend": mostly the vector of a trained subject plus noise (a neighbour in embedding space)
 *  - "random": an unrelated random direction
 * Both are scaled to length 1 like the trained subjects.
 */
export interface UnseenSubject { name: string; note: string; truth: string; vector: Vec }

export const blendVector = (base: Vec, other: Vec, mix: number): Vec => {
  const v = base.map((b, i) => mix * b + (1 - mix) * other[i])
  const n = norm(v) || 1
  return v.map((x) => x / n)
}

export const unseenSubjects = (X: Mat = subjectVectors(), seed = 77): UnseenSubject[] => {
  const rng = makeRng(seed)
  const idx = (name: string) => SUBJECTS.indexOf(name)
  const near = (name: string) => blendVector(X[idx(name)], randomUnitVector(rng, X[0].length), 0.8)
  return [
    { name: 'Austria', note: 'never trained; its vector sits close to Germany', truth: 'Vienna', vector: near('Germany') },
    { name: 'Sweden', note: 'never trained; its vector sits close to Norway', truth: 'Stockholm', vector: near('Norway') },
    { name: 'Atlantis', note: 'does not exist; an unrelated random vector', truth: 'nothing', vector: randomUnitVector(rng, X[0].length) },
    { name: 'Narnia', note: 'does not exist; an unrelated random vector', truth: 'nothing', vector: randomUnitVector(rng, X[0].length) },
  ]
}

/** How each input dimension contributes to one logit: x_d × W[d][j]. They sum to the logit. */
export const contributions = (W: Mat, x: Vec, j: number): Vec => x.map((xd, d) => xd * W[d][j])
