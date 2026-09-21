// Skip-gram word2vec, ported from phase2-language/tiny_word2vec.py.
// Same corpus, same model (lookup -> linear -> softmax -> cross-entropy), same
// hand-written gradients, same update order. The random numbers differ from
// NumPy's, so exact values differ, but the geometry that emerges is the same.
import { makeRng, type Rng } from './rng'

export const SENTENCES_BASE = [
  'the cat chased the mouse', 'the dog chased the cat',
  'my pet cat sleeps all day', 'my pet dog barks loudly',
  'feed the cat some fish', 'feed the dog some meat',
  'the kitten is a small cat', 'the puppy is a small dog',
  'the cat and the dog play together', 'a hungry dog eats meat',
  'a hungry cat eats fish', 'the mouse ran from the cat',
  'i cooked rice and beans for dinner', 'we ate bread and cheese for lunch',
  'she cooked pasta with tomato sauce', 'he ate rice with fish for dinner',
  'fresh bread smells wonderful at breakfast', 'we had cheese and bread for breakfast',
  'dinner was pasta and tomato salad', 'lunch was rice beans and meat',
  'the programmer wrote code on the computer', 'the computer runs the software',
  'she debugged the software on her laptop', 'the laptop compiled the code quickly',
  'new software update for the computer', 'the programmer fixed the laptop',
  'code review improved the software', 'the laptop runs code and software',
]

export interface Dataset {
  words: string[]
  stoi: Map<string, number>
  /** (center, one context word) pairs, flattened: [c0, x0, c1, x1, ...] */
  pairs: Int32Array
}

/** build_dataset(): every word paired with each neighbour up to `window` words away. */
export const buildDataset = (sentences: string[], window = 2): Dataset => {
  const words = Array.from(new Set(sentences.flatMap((s) => s.split(' ')))).sort()
  const stoi = new Map(words.map((w, i) => [w, i]))
  const pairs: number[] = []
  for (const s of sentences) {
    const toks = s.split(' ').map((w) => stoi.get(w)!)
    toks.forEach((center, i) => {
      for (let j = Math.max(0, i - window); j < Math.min(toks.length, i + window + 1); j++) if (j !== i) pairs.push(center, toks[j])
    })
  }
  return { words, stoi, pairs: Int32Array.from(pairs) }
}

export interface W2VModel {
  V: number
  dim: number
  E: Float64Array // (V, dim) the embedding matrix: THE product
  W: Float64Array // (dim, V) output layer: scaffolding, discarded later
  step: number
  loss: number // loss of the most recent batch
  rng: Rng
}

export const initModel = (V: number, dim: number, seed = 1): W2VModel => {
  const rng = makeRng(seed)
  const E = Float64Array.from({ length: V * dim }, () => 0.1 * rng.normal())
  const W = Float64Array.from({ length: dim * V }, () => 0.1 * rng.normal())
  return { V, dim, E, W, step: 0, loss: Math.log(V), rng }
}

/** One mini-batch of SGD, mutating the model in place (as the NumPy code does). Returns the batch loss. */
export const trainStep = (m: W2VModel, data: Dataset, lr = 0.5, batch = 256): number => {
  const { V, dim, E, W } = m
  const nPairs = data.pairs.length / 2
  const centers = new Int32Array(batch)
  const dLogits = new Float64Array(batch * V)
  let loss = 0

  // forward: lookup -> linear -> softmax; then d_logits = (probs - one_hot) / batch
  for (let b = 0; b < batch; b++) {
    const p = m.rng.int(nPairs)
    const c = (centers[b] = data.pairs[2 * p])
    const target = data.pairs[2 * p + 1]
    let max = -Infinity
    for (let v = 0; v < V; v++) {
      let z = 0
      for (let d = 0; d < dim; d++) z += E[c * dim + d] * W[d * V + v]
      dLogits[b * V + v] = z
      if (z > max) max = z
    }
    let sum = 0
    for (let v = 0; v < V; v++) sum += dLogits[b * V + v] = Math.exp(dLogits[b * V + v] - max)
    for (let v = 0; v < V; v++) dLogits[b * V + v] /= sum
    loss += -Math.log(dLogits[b * V + target] + 1e-12)
    dLogits[b * V + target] -= 1
    for (let v = 0; v < V; v++) dLogits[b * V + v] /= batch
  }

  // backward, both gradients computed with the OLD weights, as in the Python
  const dW = new Float64Array(dim * V)
  const dEmb = new Float64Array(batch * dim)
  for (let b = 0; b < batch; b++) {
    const c = centers[b]
    for (let d = 0; d < dim; d++) {
      const e = E[c * dim + d]
      let g = 0
      for (let v = 0; v < V; v++) {
        const dl = dLogits[b * V + v]
        dW[d * V + v] += e * dl // d_W = emb.T @ d_logits
        g += dl * W[d * V + v] // d_emb = d_logits @ W.T
      }
      dEmb[b * dim + d] = g
    }
  }
  for (let i = 0; i < W.length; i++) W[i] -= lr * dW[i]
  // only the looked-up rows of E change (np.add.at: repeated rows accumulate)
  for (let b = 0; b < batch; b++) for (let d = 0; d < dim; d++) E[centers[b] * dim + d] -= lr * dEmb[b * dim + d]

  m.step += 1
  m.loss = loss / batch
  return m.loss
}

export const vectorOf = (m: W2VModel, id: number): number[] => Array.from(m.E.subarray(id * m.dim, (id + 1) * m.dim))

export const similarity = (m: W2VModel, a: number, b: number): number => {
  let ab = 0, aa = 0, bb = 0
  for (let d = 0; d < m.dim; d++) {
    const x = m.E[a * m.dim + d], y = m.E[b * m.dim + d]
    ab += x * y; aa += x * x; bb += y * y
  }
  return ab / (Math.sqrt(aa) * Math.sqrt(bb) + 1e-9)
}

/** nearest(): the k most cosine-similar words, skipping the query itself. */
export const nearest = (m: W2VModel, data: Dataset, query: string, k = 4): { word: string; sim: number }[] => {
  const q = data.stoi.get(query)
  if (q === undefined) return []
  return data.words
    .map((word, i) => ({ word, sim: similarity(m, q, i), i }))
    .filter((r) => r.i !== q)
    .sort((a, b) => b.sim - a.sim)
    .slice(0, k)
    .map(({ word, sim }) => ({ word, sim }))
}
