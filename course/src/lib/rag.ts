// A faithful TypeScript port of phase4-modern-llms/mini_rag.py.
// Same algorithm, same corpus, same numbers: chunk with overlap, a crude
// TF-IDF + co-occurrence embedder, exact cosine top-k, prompt assembly and an
// extractive stand-in for the LLM that can only quote, cite or refuse.
import { dot, norm, type Vec } from './math'

export interface Doc { name: string; text: string }
export interface Chunk { text: string; source: string; pos: number }
export type Retrieved = { chunk: Chunk; sim: number }[]

/** The tiny internal "wiki" from mini_rag.py: knowledge that is in no model's weights. */
export const CORPUS: Doc[] = [
  {
    name: 'onboarding.md',
    text: 'New engineers get laptop access on day one. The onboarding buddy is assigned by the team lead. All new hires must complete security training within two weeks. Production access requires completing the incident response course. The engineering handbook lives in the internal wiki.',
  },
  {
    name: 'deploy-policy.md',
    text: 'Deployments to production happen through the CI pipeline only. Manual deploys are forbidden except during a declared incident. Every deploy requires two approvals on the pull request. Rollbacks are triggered from the deploy dashboard. Deploy freezes apply during the last week of each quarter.',
  },
  {
    name: 'oncall.md',
    text: 'The oncall rotation changes every Monday at 10am. Primary oncall must acknowledge pages within five minutes. Secondary oncall is paged if the primary does not respond. After an incident the oncall engineer writes the postmortem. Postmortems are blameless and due within three business days.',
  },
  {
    name: 'expenses.md',
    text: 'Engineers may expense up to 500 dollars per year for learning materials. Conference travel requires manager approval in advance. Receipts must be submitted within thirty days of purchase. Home office equipment is budgeted separately at 1000 dollars.',
  },
]

export const DEMO_QUERIES = [
  'how quickly must I acknowledge pages',
  'what is the budget for learning materials',
  'who writes the postmortem after an incident',
  'when do deploy freezes apply',
]

/** Split a text into sentences the way the Python does: on full stops. */
export const sentences = (text: string): string[] =>
  text.split('.').map((s) => s.trim()).filter(Boolean).map((s) => s + '.')

/** 1. CHUNKING: windows of `sentencesPerChunk` sentences that move forward by (size - overlap). */
export const chunk = (text: string, source: string, sentencesPerChunk = 2, overlap = 1): Chunk[] => {
  const sents = sentences(text)
  const step = Math.max(1, sentencesPerChunk - overlap)
  const chunks: Chunk[] = []
  for (let i = 0; i < sents.length; i += step) {
    chunks.push({ text: sents.slice(i, i + sentencesPerChunk).join(' '), source, pos: i })
  }
  return chunks
}

export const chunkAll = (docs: Doc[], sentencesPerChunk = 2, overlap = 1): Chunk[] =>
  docs.flatMap((d) => chunk(d.text, d.name, sentencesPerChunk, overlap))

export const tokenize = (text: string): string[] =>
  text.toLowerCase().replace(/[.,]/g, '').split(/\s+/).filter(Boolean)

export interface Embedder {
  vocab: string[]
  idf: Vec
  embed: (text: string) => Vec
  /** Words of the text that the embedder has never seen (they contribute nothing). */
  unknown: (text: string) => string[]
}

/**
 * 2. EMBEDDING: TF-IDF bag of words (rare words matter, "the" does not), lightly
 * smoothed by a row-normalised co-occurrence matrix so some weight bleeds onto
 * words that appear together in the corpus. Output is unit length, so dot = cosine.
 */
export const buildEmbedder = (texts: string[], smooth = 0.3): Embedder => {
  const vocab = [...new Set(texts.flatMap(tokenize))].sort()
  const stoi = new Map(vocab.map((w, i) => [w, i]))
  const Vn = vocab.length

  const df = new Array<number>(Vn).fill(0)
  for (const t of texts) for (const w of new Set(tokenize(t))) df[stoi.get(w)!] += 1
  const idf = df.map((d) => Math.log(texts.length / (d + 1e-9)))

  const C: number[][] = Array.from({ length: Vn }, () => new Array<number>(Vn).fill(0))
  for (const t of texts) {
    const ws = tokenize(t).map((w) => stoi.get(w)!)
    for (const a of ws) for (const b of ws) if (a !== b) C[a][b] += 1
  }
  for (const row of C) {
    const s = row.reduce((x, y) => x + y, 0) + 1e-9
    for (let j = 0; j < Vn; j++) row[j] /= s
  }

  const embed = (text: string): Vec => {
    const v = new Array<number>(Vn).fill(0)
    for (const w of tokenize(text)) {
      const i = stoi.get(w)
      if (i !== undefined) v[i] += idf[i] // the tf-idf part
    }
    const out = v.slice()
    for (let a = 0; a < Vn; a++) {
      if (v[a] === 0) continue
      const row = C[a]
      for (let b = 0; b < Vn; b++) out[b] += smooth * v[a] * row[b] // bleed onto related words
    }
    const n = norm(out) + 1e-9
    return out.map((x) => x / n)
  }
  const unknown = (text: string) => [...new Set(tokenize(text).filter((w) => !stoi.has(w)))]
  return { vocab, idf, embed, unknown }
}

/** The heaviest dimensions of an embedding, as words: what this vector is "about". */
export const topTerms = (vec: Vec, vocab: string[], n = 6): { term: string; weight: number }[] =>
  vec.map((weight, i) => ({ term: vocab[i], weight }))
    .filter((t) => t.weight > 1e-9)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, n)

/** 3. VECTOR STORE: exact search. One dot product per stored chunk, sort, keep k. */
export class Store {
  vecs: Vec[] = []
  chunks: Chunk[] = []
  add(vec: Vec, c: Chunk) {
    this.vecs.push(vec)
    this.chunks.push(c)
  }
  search(qvec: Vec, k = 3): Retrieved {
    return this.vecs
      .map((v, i) => ({ chunk: this.chunks[i], sim: dot(v, qvec) }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, k)
  }
}

/** 4. PROMPT ASSEMBLY: exactly what the LLM would receive. */
export const assemblePrompt = (question: string, retrieved: Retrieved): string => {
  const ctx = retrieved.map(({ chunk: c }, i) => `[${i + 1}] (${c.source}) ${c.text}`).join('\n')
  return (
    'Answer using ONLY the context below. If the answer is not in the\n' +
    "context, say 'Not found in the provided context.'\n\n" +
    `Context:\n${ctx}\n\nQuestion: ${question}\nAnswer:`
  )
}

export interface Answer {
  found: boolean
  sentence: string | null
  cite: { n: number; source: string } | null
  sim: number
  /** The same string mini_rag.py prints after "ANSWER:". */
  text: string
}

/**
 * 5. THE "LLM": an honest extractive stand-in. It picks the single retrieved
 * sentence most similar to the question, cites it, or refuses below a threshold.
 */
export const extractiveAnswer = (question: string, retrieved: Retrieved, embed: (t: string) => Vec, threshold = 0.35): Answer => {
  const qv = embed(question)
  let best: string | null = null
  let bestSim = -1
  let cite: Answer['cite'] = null
  retrieved.forEach(({ chunk: c }, i) => {
    for (const sent of c.text.split('.')) {
      if (!sent.trim()) continue
      const sim = dot(embed(sent), qv)
      if (sim > bestSim) {
        best = sent.trim()
        bestSim = sim
        cite = { n: i + 1, source: c.source }
      }
    }
  })
  if (bestSim < threshold || best === null || cite === null) {
    return { found: false, sentence: null, cite: null, sim: bestSim, text: `Not found in the provided context. (best match ${bestSim.toFixed(2)})` }
  }
  const c = cite as { n: number; source: string }
  return { found: true, sentence: best, cite: c, sim: bestSim, text: `${best}. [source ${c.n}: ${c.source}] (sim ${bestSim.toFixed(2)})` }
}

/* ---------- wired together ---------- */
export interface Index { chunks: Chunk[]; embedder: Embedder; store: Store }

/** INGEST: docs -> chunks -> embedder fitted on the chunks -> store. */
export const ingest = (docs: Doc[], sentencesPerChunk = 2, overlap = 1): Index => {
  const chunks = chunkAll(docs, sentencesPerChunk, overlap)
  const embedder = buildEmbedder(chunks.map((c) => c.text))
  const store = new Store()
  for (const c of chunks) store.add(embedder.embed(c.text), c)
  return { chunks, embedder, store }
}

/** QUERY: question -> embed -> top-k -> prompt -> answer from context only. */
export const ask = (index: Index, question: string, k = 3, threshold = 0.35) => {
  const retrieved = index.store.search(index.embedder.embed(question), k)
  return { retrieved, prompt: assemblePrompt(question, retrieved), answer: extractiveAnswer(question, retrieved, index.embedder.embed, threshold) }
}
