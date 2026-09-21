// Model A of phase2-language/bigram_lm.py: the count-table bigram language model,
// at the character level, with the same corpus and the same smoothing.
import { sampleIndex, type Mat, type Vec } from './math'

/** One copy of the corpus text. bigram_lm.py trains on this repeated 3 times. */
export const CORPUS_BASE =
  'the quick brown fox jumps over the lazy dog and the cat sleeps ' +
  'in the warm sun while the dog barks at the mailman who walks ' +
  'down the street every morning with letters for the people in ' +
  'the town where the children play in the park near the river ' +
  'that flows past the old mill and under the stone bridge to the ' +
  'sea where the fishermen cast their nets in the early light of ' +
  'dawn and sing the old songs of the water and the wind and the ' +
  'long summer days that fade into the quiet evenings of autumn '

export const CORPUS_REPEAT = 3
export const CORPUS = CORPUS_BASE.repeat(CORPUS_REPEAT)
export const SMOOTHING = 0.01

export interface Vocab {
  chars: string[]
  stoi: Map<string, number>
}

export const buildVocab = (text: string): Vocab => {
  const chars = Array.from(new Set(Array.from(text))).sort((a, b) => a.codePointAt(0)! - b.codePointAt(0)!)
  return { chars, stoi: new Map(chars.map((c, i) => [c, i])) }
}

export const toIds = (text: string, vocab: Vocab): number[] => Array.from(text).map((c) => vocab.stoi.get(c)!)

/** Raw tallies: counts[a][b] = how often character b came right after character a. */
export const countPairs = (ids: number[], V: number): Mat => {
  const counts: Mat = Array.from({ length: V }, () => new Array<number>(V).fill(0))
  for (let i = 0; i + 1 < ids.length; i++) counts[ids[i]][ids[i + 1]] += 1
  return counts
}

/** count_model(): add `smoothing` to every cell, then normalise each row into probabilities. */
export const countModel = (ids: number[], V: number, smoothing = SMOOTHING): Mat =>
  countPairs(ids, V).map((row) => {
    const smoothed = row.map((c) => c + smoothing)
    const total = smoothed.reduce((s, c) => s + c, 0)
    return smoothed.map((c) => c / total)
  })

/** cross_entropy_of_table(): the average surprise, −log P(next | current), over the text. */
export const crossEntropyOfTable = (table: Mat, ids: number[]): number => {
  let sum = 0
  for (let i = 0; i + 1 < ids.length; i++) sum += -Math.log(table[ids[i]][ids[i + 1]])
  return sum / (ids.length - 1)
}

export const perplexity = (crossEntropy: number): number => Math.exp(crossEntropy)

export interface Roll {
  next: number // sampled character id
  u: number // the uniform random number that decided it
  probs: Vec // the row that was used
}

/** One turn of generate(): look up the row of `cur`, roll the weighted die (or take the max if greedy). */
export const rollNext = (table: Mat, cur: number, u: number, greedy = false): Roll => {
  const probs = table[cur]
  const next = greedy ? probs.indexOf(Math.max(...probs)) : sampleIndex(probs, u)
  return { next, u, probs }
}

/** generate(): repeat rollNext, feeding each output back in as the next input. */
export const generate = (table: Mat, start: number, n: number, rand: () => number, greedy = false): number[] => {
  const out: number[] = []
  let cur = start
  for (let i = 0; i < n; i++) {
    cur = rollNext(table, cur, rand(), greedy).next
    out.push(cur)
  }
  return out
}

/** The training examples hiding in a text: every position gives (input, target) = (this char, next char). */
export const shiftedPairs = (text: string): [string, string][] => {
  const cs = Array.from(text)
  return cs.slice(0, -1).map((c, i) => [c, cs[i + 1]])
}
