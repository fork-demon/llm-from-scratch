// Byte Pair Encoding, ported from phase2-language/bpe_tokenizer.py.
// Same greedy algorithm, same tie-breaking: Python's Counter.most_common(1)
// returns the FIRST pair (in order of first appearance in the corpus) among
// those with the highest count, and so do we.
// Pure functions, no DOM. "Characters" are Unicode code points, as in Python.

/** The sample text from bpe_tokenizer.py. The Python file trains on this repeated 4 times. */
export const BPE_SAMPLE =
  'the quick brown fox jumps over the lazy dog. ' +
  'the dog barked at the fox. the fox ran into the forest. ' +
  'learning about the internals of the tokenizer teaches the ' +
  'engineer the fundamentals of the language model. ' +
  'the tokens in the text represent the meaning of the words. '

export const BPE_REPEAT = 4

export interface Merge {
  pair: [number, number]
  newId: number
  /** How often the pair occurred in the working corpus when it was merged. */
  count: number
}

export interface Tokenizer {
  /** id -> the string it stands for. ids 0..base-1 are single characters, sorted. */
  vocab: string[]
  /** The learned artefact: merge rules, in the order they were learned. */
  merges: Merge[]
}

export interface TrainState extends Tokenizer {
  /** The training corpus as token ids, after all merges so far. */
  ids: number[]
  /** True when no pair occurs at least twice: nothing left worth merging. */
  done: boolean
}

const chars = (text: string): string[] => Array.from(text)

/** Step 1 of train(): every unique character gets an id, in sorted (code point) order. */
export const initTraining = (text: string): TrainState => {
  const cs = chars(text)
  const vocab = Array.from(new Set(cs)).sort((a, b) => a.codePointAt(0)! - b.codePointAt(0)!)
  const stoi = new Map(vocab.map((c, i) => [c, i]))
  const ids = cs.map((c) => stoi.get(c)!)
  return { vocab, merges: [], ids, done: bestPair(ids) === null }
}

/**
 * The most frequent adjacent pair, or null if no pair occurs at least twice.
 * Ties go to the pair that appears first in the sequence (Counter.most_common order).
 */
export const bestPair = (ids: number[]): { pair: [number, number]; count: number } | null => {
  const counts = new Map<string, { pair: [number, number]; count: number }>()
  for (let i = 0; i + 1 < ids.length; i++) {
    const key = `${ids[i]},${ids[i + 1]}`
    const hit = counts.get(key)
    if (hit) hit.count++
    else counts.set(key, { pair: [ids[i], ids[i + 1]], count: 1 })
  }
  let best: { pair: [number, number]; count: number } | null = null
  for (const entry of counts.values()) if (!best || entry.count > best.count) best = entry
  return best && best.count >= 2 ? best : null
}

/** Replace every (left-to-right, non-overlapping) occurrence of `pair` with `newId`. */
export const applyMerge = (ids: number[], pair: [number, number], newId: number): number[] => {
  const out: number[] = []
  let i = 0
  while (i < ids.length) {
    if (i < ids.length - 1 && ids[i] === pair[0] && ids[i + 1] === pair[1]) {
      out.push(newId)
      i += 2
    } else {
      out.push(ids[i])
      i += 1
    }
  }
  return out
}

/** One turn of the training loop: find the winning pair, mint a token, rewrite the corpus. */
export const mergeStep = (s: TrainState): TrainState => {
  const best = bestPair(s.ids)
  if (!best) return { ...s, done: true }
  const newId = s.vocab.length
  const [a, b] = best.pair
  const vocab = [...s.vocab, s.vocab[a] + s.vocab[b]]
  const merges = [...s.merges, { pair: best.pair, newId, count: best.count }]
  const ids = applyMerge(s.ids, best.pair, newId)
  return { vocab, merges, ids, done: bestPair(ids) === null }
}

/** train(text, vocab_size): merge until the vocabulary is big enough or nothing repeats. */
export const train = (text: string, vocabSize: number): TrainState => {
  let s = initTraining(text)
  while (s.vocab.length < vocabSize && !s.done) s = mergeStep(s)
  return s
}

/** Thrown by encode() for a character the tokenizer never saw, like Python's KeyError. */
export class UnknownCharError extends Error {
  constructor(public char: string) {
    super(`unknown character ${JSON.stringify(char)}: it was not in the training text`)
  }
}

export const UNK = -1

const baseIndex = (tok: Tokenizer) => {
  const stoi = new Map<string, number>()
  tok.vocab.forEach((s, i) => { if (chars(s).length === 1 && !stoi.has(s)) stoi.set(s, i) })
  return stoi
}

const replay = (ids: number[], merges: Merge[]) => {
  for (const m of merges) ids = applyMerge(ids, m.pair, m.newId) // ORDER MATTERS
  return ids
}

/** Faithful encode(): characters to ids, then replay every merge in training order. Throws on unseen characters. */
export const encode = (tok: Tokenizer, text: string): number[] => {
  const stoi = baseIndex(tok)
  const ids = chars(text).map((c) => {
    const id = stoi.get(c)
    if (id === undefined) throw new UnknownCharError(c)
    return id
  })
  return replay(ids, tok.merges)
}

export interface Piece {
  id: number // UNK (-1) for a character outside the base vocabulary
  text: string
}

/**
 * encode() that never crashes: an unseen character becomes an <unk> piece that
 * remembers its text. No merge rule mentions UNK, so merges simply stop at it.
 */
export const encodeSafe = (tok: Tokenizer, text: string): Piece[] => {
  const stoi = baseIndex(tok)
  const cs = chars(text)
  const out: Piece[] = []
  let run: number[] = []
  const flush = () => {
    for (const id of replay(run, tok.merges)) out.push({ id, text: tok.vocab[id] })
    run = []
  }
  for (const c of cs) {
    const id = stoi.get(c)
    if (id === undefined) {
      flush()
      out.push({ id: UNK, text: c })
    } else run.push(id)
  }
  flush()
  return out
}

export const decode = (tok: Tokenizer, ids: number[]): string => ids.map((i) => tok.vocab[i]).join('')

/** The UTF-8 bytes of a character: what a byte-level tokenizer falls back to. */
export const utf8Bytes = (c: string): number[] => Array.from(new TextEncoder().encode(c))
