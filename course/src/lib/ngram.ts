// Counting with longer contexts, and why it stops working.
// Used by the "context wall" lesson: context explosion, data sparsity, and an
// n-gram generator that slides from gibberish to reciting its training text.
import { sampleIndex } from './math'

/** log10 of the number of possible contexts: V^n = 10^(n · log10 V). Avoids overflow. */
export const log10Contexts = (vocabSize: number, n: number): number => n * Math.log10(vocabSize)

/** "3.9 × 10^14" style formatting from a log10 value; plain digits while the number is small. */
export const formatPow10 = (log10: number): string => {
  if (log10 < 6) return Math.round(10 ** log10).toLocaleString('en-US')
  const exp = Math.floor(log10 + 1e-9)
  const mant = 10 ** (log10 - exp)
  return `${mant.toFixed(1)} × 10^${exp}`
}

/** How often each length-n window of characters occurs in the text. */
export const contextCounts = (text: string, n: number): Map<string, number> => {
  const counts = new Map<string, number>()
  for (let i = 0; i + n <= text.length; i++) {
    const c = text.slice(i, i + n)
    counts.set(c, (counts.get(c) ?? 0) + 1)
  }
  return counts
}

export interface Sparsity {
  n: number
  distinct: number // different length-n contexts that occur at all
  once: number // of those, how many occur exactly once
  onceFraction: number
  possible: number // V^n (as a float; may be huge)
  coverage: number // distinct / possible
}

export const sparsity = (text: string, n: number, vocabSize?: number): Sparsity => {
  const counts = contextCounts(text, n)
  let once = 0
  for (const c of counts.values()) if (c === 1) once++
  const V = vocabSize ?? new Set(text).size
  const possible = V ** n
  const distinct = counts.size
  return { n, distinct, once, onceFraction: distinct ? once / distinct : 0, possible, coverage: distinct / possible }
}

/** context (k characters) -> next character -> count. No smoothing: unseen means impossible. */
export type NgramTable = Map<string, Map<string, number>>

export const buildNgram = (text: string, k: number): NgramTable => {
  const table: NgramTable = new Map()
  for (let i = 0; i + k < text.length; i++) {
    const ctx = text.slice(i, i + k)
    const next = text[i + k]
    let row = table.get(ctx)
    if (!row) table.set(ctx, (row = new Map()))
    row.set(next, (row.get(next) ?? 0) + 1)
  }
  return table
}

export interface NgramSample {
  text: string // seed + generated characters
  forced: number // steps where the context had exactly one possible continuation
  steps: number
}

/**
 * Generate `length` characters using k characters of context. Starts from the first
 * k characters of the corpus. If a context was never seen (only possible at the very
 * end of the corpus) it backs off to a shorter context.
 */
export const generateNgram = (text: string, k: number, length: number, rand: () => number): NgramSample => {
  const tables = Array.from({ length: k + 1 }, (_, j) => buildNgram(text, j))
  let out = text.slice(0, k)
  let forced = 0
  for (let s = 0; s < length; s++) {
    let row: Map<string, number> | undefined
    for (let j = k; j >= 0 && !row; j--) row = tables[j].get(j === 0 ? '' : out.slice(-j))
    if (!row) break
    const options = Array.from(row.entries())
    const total = options.reduce((t, [, c]) => t + c, 0)
    if (options.length === 1) forced++
    out += options[sampleIndex(options.map(([, c]) => c / total), rand())][0]
  }
  return { text: out, forced, steps: length }
}

/** Length of the longest piece of `generated` that appears word for word in `corpus`. */
export const longestCopiedRun = (generated: string, corpus: string): { start: number; length: number } => {
  let best = { start: 0, length: 0 }
  for (let i = 0; i < generated.length; i++) {
    let len = best.length + 1 // only look for improvements
    while (i + len <= generated.length && corpus.includes(generated.slice(i, i + len))) len++
    if (len - 1 > best.length) best = { start: i, length: len - 1 }
  }
  return best
}
