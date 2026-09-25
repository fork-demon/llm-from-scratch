// Knowledge distillation on one toy next-token prediction.
//
// A fixed "teacher" has logits for the six tokens that could finish "The cat sat on the ___".
// A "student" starts knowing nothing (all logits 0, so every token gets 1/6) and learns from
// one training example per step, in one of three ways:
//
//   hard    : the target is the teacher's single top token ("mat"), as a one-hot label.
//             loss = cross-entropy = -log q(mat).           gradient on student logits: q - onehot
//   sampled : the target is one token drawn from the teacher's distribution (seeded), one-hot.
//             This is what "train the student on text the teacher wrote" does, one token at a time.
//   soft    : the target is the teacher's whole distribution at temperature T, compared with the
//             student's distribution at the same T (Hinton et al., 2015).
//             loss = T^2 * cross-entropy(p_T, q_T).         gradient on student logits: T * (q_T - p_T)
//
// The student here is just six numbers (one logit per token), so it can match the teacher exactly.
// Real students are whole networks that must share their weights across millions of contexts.
import { softmax, type Vec } from './math'
import { makeRng } from './rng'

export const TOKENS = ['mat', 'sofa', 'floor', 'bed', 'roof', 'moon'] as const
export const TEACHER_LOGITS: Vec = [4.0, 2.2, 2.0, 1.6, 0.5, -2.0]

export type Mode = 'hard' | 'sampled' | 'soft'
export const MODES: Mode[] = ['hard', 'sampled', 'soft']

/** KL(p || q) = sum p log(p / q), in nats. Terms with p = 0 contribute 0. */
export const kl = (p: Vec, q: Vec): number =>
  p.reduce((s, pi, i) => (pi > 0 ? s + pi * Math.log(pi / Math.max(q[i], 1e-300)) : s), 0)

/** Cross-entropy H(p, q) = -sum p log q, in nats. With a one-hot p this is -log q[target]. */
export const crossEntropySoft = (p: Vec, q: Vec): number =>
  p.reduce((s, pi, i) => (pi > 0 ? s - pi * Math.log(Math.max(q[i], 1e-300)) : s), 0)

/** Entropy H(p) = -sum p log p, in nats. */
export const entropy = (p: Vec): number => crossEntropySoft(p, p)

export const oneHot = (n: number, i: number): Vec => Array.from({ length: n }, (_, j) => (j === i ? 1 : 0))

/** Index drawn from a distribution with a uniform number u in [0, 1). */
export const drawIndex = (probs: Vec, u: number): number => {
  let c = 0
  for (let i = 0; i < probs.length; i++) {
    c += probs[i]
    if (u < c) return i
  }
  return probs.length - 1
}

/** Gradient of the loss with respect to the student's logits, for one example. */
export const gradient = (student: Vec, target: Vec, T = 1): Vec => {
  const q = softmax(student, T)
  // d/dz [ T^2 * H(target, softmax(z/T)) ] = T^2 * (1/T) * (q - target) = T * (q - target)
  return q.map((qi, i) => T * (qi - target[i]))
}

/** The loss the student is trained on, for one example (T^2 scaling included for soft targets). */
export const loss = (student: Vec, target: Vec, T = 1): number => T * T * crossEntropySoft(target, softmax(student, T))

export interface Trace {
  /** KL(teacher at T=1 || student at T=1) after each step; index 0 = before training. */
  kl: number[]
  /** student logits after each step; index 0 = before training. */
  logits: Vec[]
  /** the one-hot or soft target used at each step (index i = target for step i+1). */
  targets: Vec[]
}

/**
 * Train the six-logit student for `steps` steps of plain gradient descent.
 * `T` is the distillation temperature (used only in soft mode). `seed` fixes the sampled labels.
 */
export const train = (mode: Mode, steps: number, opts: { T?: number; lr?: number; seed?: number; teacher?: Vec } = {}): Trace => {
  const { T = 1, lr = 0.5, seed = 7, teacher = TEACHER_LOGITS } = opts
  const n = teacher.length
  const p1 = softmax(teacher, 1)
  const pT = softmax(teacher, T)
  const top = p1.indexOf(Math.max(...p1))
  const rng = makeRng(seed)
  let z: Vec = new Array(n).fill(0)
  const trace: Trace = { kl: [kl(p1, softmax(z))], logits: [z.slice()], targets: [] }
  for (let s = 0; s < steps; s++) {
    let target: Vec
    let temp = 1
    if (mode === 'hard') target = oneHot(n, top)
    else if (mode === 'sampled') target = oneHot(n, drawIndex(p1, rng.next()))
    else { target = pT; temp = T }
    const g = gradient(z, target, temp)
    z = z.map((zi, i) => zi - lr * g[i])
    trace.kl.push(kl(p1, softmax(z)))
    trace.logits.push(z.slice())
    trace.targets.push(target)
  }
  return trace
}

/** First step at which KL drops below `threshold` and stays below for the rest of the trace, or -1. */
export const stepsToReach = (klTrace: number[], threshold: number): number => {
  let last = -1
  for (let i = klTrace.length - 1; i >= 0; i--) {
    if (klTrace[i] < threshold) last = i
    else break
  }
  return last
}

/**
 * Model collapse in one line: a token with probability p, a dataset of n samples.
 * Chance the token never appears (so a model fitted only to those samples gives it probability 0).
 */
export const chanceMissing = (p: number, n: number): number => Math.pow(1 - p, n)
