// Catastrophic forgetting you can watch, in the smallest language model of the
// course: the neural bigram from lesson 5.1 (logits for the next character =
// one row of a V x V weight matrix, picked by the current character).
//
// Task A = the Shakespeare-style text, task B = the plain modern text: the same
// two corpora as phase4-modern-llms/finetune_tiny_gpt.py (one copy each,
// lower-cased, letters and spaces only). "Pretrain" on A, then fine-tune on B:
//   full     every weight in W moves
//   adapter  W is frozen; a low-rank bypass A @ B (LoRA) is trained instead
//   replay   every weight moves, but 20% of each gradient still comes from task A
// Because a bigram's loss depends only on bigram counts, the full-batch gradient
// is exact and cheap: no sampling noise, fully deterministic.
import { matmul, softmax, transpose, type Mat } from './math'
import { makeRng } from './rng'

const SHAKESPEARE =
  'O for a Muse of fire, that would ascend the brightest heaven of invention! Once more unto the breach, dear friends, once more; or close the wall up with our English dead. What light through yonder window breaks? It is the east, and Juliet is the sun. To be, or not to be, that is the question: whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles. Now is the winter of our discontent made glorious summer by this son of York. Friends, Romans, countrymen, lend me your ears; I come to bury Caesar, not to praise him. The evil that men do lives after them. Cowards die many times before their deaths; the valiant never taste of death but once. All the world is a stage, and all the men and women merely players. They have their exits and their entrances. '
const MODERN =
  'The meeting starts at nine tomorrow morning. Please bring your laptop and the quarterly report. We need to fix the login bug before the release on friday. The coffee machine on the third floor is broken again. Can you review my pull request when you get a chance? The deploy pipeline failed twice last night. Let us schedule a quick call to talk about the roadmap. The new feature works fine on staging but not in production. Remember to submit your expense report by the end of the month. The team lunch is moved to thursday because of the demo. I will send the notes after the standup. The database migration finished without any errors. Please update the documentation when the api changes. '

export const CHARS = ' abcdefghijklmnopqrstuvwxyz'.split('')
export const V = CHARS.length

const clean = (t: string) => t.toLowerCase().replace(/[^a-z ]/g, '').replace(/ +/g, ' ')

/** A task = the table of how often character j follows character i, as fractions of all pairs. */
export interface Task { name: string; P: Mat }
const makeTask = (name: string, text: string): Task => {
  const ids = clean(text).split('').map((c) => CHARS.indexOf(c))
  const P: Mat = Array.from({ length: V }, () => new Array<number>(V).fill(0))
  for (let t = 0; t + 1 < ids.length; t++) P[ids[t]][ids[t + 1]] += 1 / (ids.length - 1)
  return { name, P }
}
export const TASK_A = makeTask('Shakespeare-style', SHAKESPEARE)
export const TASK_B = makeTask('plain modern', MODERN)

const zeros = (n: number, m: number): Mat => Array.from({ length: n }, () => new Array<number>(m).fill(0))
const addM = (a: Mat, b: Mat, k = 1): Mat => a.map((row, i) => row.map((v, j) => v + k * b[i][j]))

/** Average cross-entropy per character (the usual LM loss) of logits on a task. */
export const lossOn = (logits: Mat, task: Task): number => {
  let L = 0
  for (let i = 0; i < V; i++) {
    const p = softmax(logits[i])
    for (let j = 0; j < V; j++) if (task.P[i][j] > 0) L -= task.P[i][j] * Math.log(p[j] + 1e-12)
  }
  return L
}

/** dLoss/dLogits: for row i, (how often i occurs) * (predicted distribution - observed distribution). */
export const gradLogits = (logits: Mat, task: Task): Mat =>
  logits.map((row, i) => {
    const n = task.P[i].reduce((a, b) => a + b, 0)
    const p = softmax(row)
    return p.map((pj, j) => n * pj - task.P[i][j])
  })

/** "Pretraining": fit W to task A from zeros with plain gradient descent. */
export const pretrain = (steps = 400, lr = 20): Mat => {
  let W = zeros(V, V)
  for (let t = 0; t < steps; t++) W = addM(W, gradLogits(W, TASK_A), -lr)
  return W
}

export type Mode = 'full' | 'adapter' | 'replay'
export const ADAPTER_RANK = 2
export const REPLAY_FRACTION = 0.2

export interface FtState {
  mode: Mode
  W: Mat // the base weights (move only in 'full' and 'replay')
  A: Mat // adapter (V x r), random small
  B: Mat // adapter (r x V), starts at zero
  step: number
}

export const initFt = (base: Mat, mode: Mode, seed = 9): FtState => {
  const rng = makeRng(seed)
  return {
    mode,
    W: base.map((r) => r.slice()),
    A: Array.from({ length: V }, () => Array.from({ length: ADAPTER_RANK }, () => 0.3 * rng.normal())),
    B: zeros(ADAPTER_RANK, V),
    step: 0,
  }
}

/** The logits the model actually uses. withAdapter = false is "unplug the adapter". */
export const logitsOf = (s: FtState, withAdapter = true): Mat => (s.mode === 'adapter' && withAdapter ? addM(s.W, matmul(s.A, s.B)) : s.W)

export const ftStep = (s: FtState, lr: number, steps = 1): FtState => {
  let { W, A, B } = s
  for (let t = 0; t < steps; t++) {
    if (s.mode === 'adapter') {
      const G = gradLogits(addM(W, matmul(A, B)), TASK_B)
      const gA = matmul(G, transpose(B))
      const gB = matmul(transpose(A), G)
      A = addM(A, gA, -lr)
      B = addM(B, gB, -lr)
    } else {
      let G = gradLogits(W, TASK_B)
      if (s.mode === 'replay') G = addM(G.map((r) => r.map((v) => v * (1 - REPLAY_FRACTION))), gradLogits(W, TASK_A), REPLAY_FRACTION)
      W = addM(W, G, -lr)
    }
  }
  return { ...s, W, A, B, step: s.step + steps }
}

export const trainableCount = (mode: Mode) => (mode === 'adapter' ? 2 * V * ADAPTER_RANK : V * V)

export interface Losses { a: number; b: number; aBaseOnly: number }
export const measureFt = (s: FtState): Losses => {
  const l = logitsOf(s)
  return { a: lossOn(l, TASK_A), b: lossOn(l, TASK_B), aBaseOnly: lossOn(s.W, TASK_A) }
}
