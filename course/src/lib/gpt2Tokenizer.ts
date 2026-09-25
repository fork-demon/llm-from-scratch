// GPT-2's byte-level BPE tokenizer, exactly (checked against tiktoken's "gpt2" encoding in gpt2Tokenizer.test.ts).
//
// The vocabulary is not shipped as a table: it follows from the merges.
//   ids 0..255     the 256 possible bytes, in GPT-2's bytes_to_unicode order
//   id 256 + i     merge i: the concatenation of the two tokens it joins
//   id 50256       <|endoftext|>
// public/models/gpt2/merges.bin holds the 50,000 merges as little-endian uint16 pairs (left id, right id),
// written by phase6-engineering/export_gpt2.py.
//
// Encoding = (1) split the text into pieces with GPT-2's regex (words with their leading space, numbers,
// punctuation runs, whitespace), (2) turn each piece into UTF-8 bytes, (3) repeatedly merge the adjacent pair
// with the lowest merge rank, the same greedy rule as the BPE lesson.

export const EOT_ID = 50256

/** GPT-2's pre-tokenisation pattern (same as openai/gpt-2 encoder.py and tiktoken "gpt2"). */
export const GPT2_SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

/** The byte order GPT-2 uses for ids 0..255 (bytes_to_unicode in encoder.py): printable ranges first. */
export const byteOrder = (): number[] => {
  const bs: number[] = []
  for (let b = 33; b <= 126; b++) bs.push(b)
  for (let b = 161; b <= 172; b++) bs.push(b)
  for (let b = 174; b <= 255; b++) bs.push(b)
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) bs.push(b)
  return bs
}

export interface Gpt2Tokenizer {
  vocabSize: number
  encode: (text: string) => number[]
  /** Pieces the regex produced, each with its ids: handy for showing why a word became several tokens. */
  encodePieces: (text: string) => { piece: string; ids: number[] }[]
  decode: (ids: number[]) => string
  /** The raw bytes of one token. */
  bytesOf: (id: number) => Uint8Array
  /** A human-readable label for one token: text if it is whole UTF-8, otherwise its bytes in hex. */
  label: (id: number) => string
}

const utf8 = new TextEncoder()

export const createGpt2Tokenizer = (merges: Uint16Array): Gpt2Tokenizer => {
  if (merges.length % 2) throw new Error('merges must hold pairs of ids')
  const nMerges = merges.length / 2
  const order = byteOrder()
  const byteToId = new Int32Array(256)
  order.forEach((b, id) => { byteToId[b] = id })

  const tokBytes: Uint8Array[] = order.map((b) => Uint8Array.of(b))
  const rank = new Map<number, number>() // (left << 16 | right) -> merge index
  for (let i = 0; i < nMerges; i++) {
    const a = merges[2 * i]
    const b = merges[2 * i + 1]
    if (a >= 256 + i || b >= 256 + i) throw new Error(`merge ${i} uses a token that does not exist yet`)
    const t = new Uint8Array(tokBytes[a].length + tokBytes[b].length)
    t.set(tokBytes[a])
    t.set(tokBytes[b], tokBytes[a].length)
    tokBytes.push(t)
    rank.set(a * 65536 + b, i)
  }
  const eot = utf8.encode('<|endoftext|>')
  tokBytes.push(eot)
  const vocabSize = tokBytes.length

  const cache = new Map<string, number[]>()
  const bpe = (piece: string): number[] => {
    const hit = cache.get(piece)
    if (hit) return hit
    let ids = Array.from(utf8.encode(piece), (b) => byteToId[b])
    while (ids.length > 1) {
      let best = Infinity
      for (let i = 0; i + 1 < ids.length; i++) {
        const r = rank.get(ids[i] * 65536 + ids[i + 1])
        if (r !== undefined && r < best) best = r
      }
      if (best === Infinity) break
      const a = merges[2 * best]
      const b = merges[2 * best + 1]
      const next: number[] = []
      for (let i = 0; i < ids.length; i++) {
        if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) { next.push(256 + best); i++ }
        else next.push(ids[i])
      }
      ids = next
    }
    if (cache.size > 20000) cache.clear()
    cache.set(piece, ids)
    return ids
  }

  const encodePieces = (text: string) => Array.from(text.matchAll(GPT2_SPLIT), (m) => ({ piece: m[0], ids: bpe(m[0]) }))
  const encode = (text: string) => encodePieces(text).flatMap((p) => p.ids)

  const bytesOf = (id: number): Uint8Array => {
    if (!(id >= 0 && id < vocabSize)) throw new Error(`token id ${id} is outside the vocabulary`)
    return tokBytes[id]
  }
  const decode = (ids: number[]): string => {
    const parts = ids.map(bytesOf)
    const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0))
    let o = 0
    for (const p of parts) { all.set(p, o); o += p.length }
    return new TextDecoder().decode(all)
  }
  const strict = new TextDecoder('utf-8', { fatal: true })
  const label = (id: number): string => {
    const b = bytesOf(id)
    try {
      return strict.decode(b)
    } catch {
      // part of a multi-byte character (Hindi, emoji...): show the bytes
      return Array.from(b, (x) => `<${x.toString(16).toUpperCase().padStart(2, '0')}>`).join('')
    }
  }
  return { vocabSize, encode, encodePieces, decode, bytesOf, label }
}

/** Make whitespace visible in a token chip: leading space as ␣, newline as ↵, tab as ⇥. */
export const showToken = (s: string): string => s.replace(/ /g, '␣').replace(/\n/g, '↵').replace(/\r/g, '⏎').replace(/\t/g, '⇥')
