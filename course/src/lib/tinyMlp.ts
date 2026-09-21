// A tiny classifier: 2 inputs -> 2 hidden ReLU units -> 2 logits -> softmax -> cross-entropy.
// Forward, backward and the numerical gradient check, for ONE example.
// Mirrors phase1-foundations/mlp_numpy.py (MLP.forward, MLP.backward, gradient_check):
// same row-vector convention (h = x @ W + b, W has shape (n_in, n_out)) and the same three rules:
//   linear:   d_W = X.T @ d_out ;  d_X = d_out @ W.T ;  d_b = sum(d_out)
//   relu:     d_x = d_out * (x > 0)
//   softmax+cross-entropy at the logits:  d_logits = probs - one_hot
import { crossEntropy, softmax, type Mat, type Vec } from './math'

export interface TinyNet { W1: Mat; b1: Vec; W2: Mat; b2: Vec }

export interface ForwardCache {
  x: Vec
  z1: Vec // hidden pre-activation: x @ W1 + b1
  h: Vec // after ReLU (in the Python file the two are one line; the mask h > 0 equals z1 > 0)
  logits: Vec
  probs: Vec
  loss: number
}

export interface Grads {
  dLogits: Vec
  dW2: Mat
  db2: Vec
  dh: Vec // blame arriving at the hidden outputs
  dz1: Vec // after the ReLU gate
  dW1: Mat
  db1: Vec
}

/** x @ W + b for one row vector x. W is (n_in, n_out). */
export const linear = (x: Vec, W: Mat, b: Vec): Vec => b.map((bj, j) => x.reduce((s, xi, i) => s + xi * W[i][j], bj))

export const forward = (net: TinyNet, x: Vec, target: number, opts: { relu?: boolean } = {}): ForwardCache => {
  const useRelu = opts.relu ?? true
  const z1 = linear(x, net.W1, net.b1)
  const h = useRelu ? z1.map((z) => Math.max(0, z)) : z1.slice() // linear + ReLU hinge
  const logits = linear(h, net.W2, net.b2) // last layer: no ReLU
  const probs = softmax(logits)
  return { x, z1, h, logits, probs, loss: crossEntropy(probs, target) }
}

/** outer(a, b)[i][j] = a[i] * b[j]. For one example this is X.T @ d_out. */
const outer = (a: Vec, b: Vec): Mat => a.map((ai) => b.map((bj) => ai * bj))

/**
 * Walk the assembly line in reverse. `reluGate: false` plants the classic bug
 * (forgetting rule 2) so the gradient check can catch it.
 */
export const backward = (net: TinyNet, cache: ForwardCache, target: number, opts: { reluGate?: boolean } = {}): Grads => {
  const gate = opts.reluGate ?? true
  const dLogits = cache.probs.map((p, j) => p - (j === target ? 1 : 0)) // rule 3: probs - one_hot
  const dW2 = outer(cache.h, dLogits) // rule 1: blame -> weights
  const db2 = dLogits.slice()
  const dh = net.W2.map((row) => row.reduce((s, w, j) => s + w * dLogits[j], 0)) // rule 1: d @ W2.T
  const dz1 = dh.map((d, i) => (gate ? (cache.h[i] > 0 ? d : 0) : d)) // rule 2: ReLU gate
  const dW1 = outer(cache.x, dz1)
  const db1 = dz1.slice()
  return { dLogits, dW2, db2, dh, dz1, dW1, db1 }
}

export const lossOf = (net: TinyNet, x: Vec, target: number): number => forward(net, x, target).loss

export type WeightRef = { layer: 1 | 2; i: number; j: number }

const withWeight = (net: TinyNet, ref: WeightRef, value: number): TinyNet => {
  const key = ref.layer === 1 ? 'W1' : 'W2'
  const W = net[key].map((r) => r.slice())
  W[ref.i][ref.j] = value
  return { ...net, [key]: W }
}

export const getWeight = (net: TinyNet, ref: WeightRef): number => (ref.layer === 1 ? net.W1 : net.W2)[ref.i][ref.j]

/** The brute-force oracle from gradient_check(): nudge up, nudge down, centered difference. */
export const numericalGrad = (net: TinyNet, x: Vec, target: number, ref: WeightRef, h = 1e-5): { lossPlus: number; lossMinus: number; grad: number } => {
  const orig = getWeight(net, ref)
  const lossPlus = lossOf(withWeight(net, ref, orig + h), x, target)
  const lossMinus = lossOf(withWeight(net, ref, orig - h), x, target)
  return { lossPlus, lossMinus, grad: (lossPlus - lossMinus) / (2 * h) }
}

/** MLP.step: every parameter moves against its gradient. */
export const applyUpdate = (net: TinyNet, g: Grads, lr: number): TinyNet => ({
  W1: net.W1.map((r, i) => r.map((w, j) => w - lr * g.dW1[i][j])),
  b1: net.b1.map((b, i) => b - lr * g.db1[i]),
  W2: net.W2.map((r, i) => r.map((w, j) => w - lr * g.dW2[i][j])),
  b2: net.b2.map((b, i) => b - lr * g.db2[i]),
})

/** The starting network used in the Backpropagation lesson. Hidden unit 2 starts with its hinge shut.
 * Chosen so that hidden unit 1 stays open through repeated updates at lr = 0.1 (a larger step, e.g. 0.5, shuts it). */
export const DEMO_NET: TinyNet = {
  W1: [[1.0, -1.0], [0.5, 0.25]],
  b1: [0, 0],
  W2: [[0.5, -0.5], [0.5, 1.0]],
  b2: [0, 0],
}
export const DEMO_X: Vec = [1, 2]
export const DEMO_TARGET = 1
