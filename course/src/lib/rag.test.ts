import { describe, expect, it } from 'vitest'
import { CORPUS, DEMO_QUERIES, ask, assemblePrompt, chunk, chunkAll, ingest, sentences, tokenize, topTerms } from './rag'
import { norm } from './math'

// Reference values come from running phase4-modern-llms/mini_rag.py (and --show-failure).
describe('chunking behaves like mini_rag.py', () => {
  it('the default corpus gives 19 chunks', () => {
    expect(chunkAll(CORPUS).length).toBe(19)
  })
  it('size 2, overlap 1: every sentence starts a chunk and neighbours share a sentence', () => {
    const cs = chunk(CORPUS[2].text, 'oncall.md', 2, 1)
    expect(cs.map((c) => c.pos)).toEqual([0, 1, 2, 3, 4])
    expect(cs[0].text).toBe('The oncall rotation changes every Monday at 10am. Primary oncall must acknowledge pages within five minutes.')
    expect(sentences(cs[0].text)[1]).toBe(sentences(cs[1].text)[0])
    expect(sentences(cs[4].text).length).toBe(1) // the last window is short
  })
  it('no overlap: the step equals the chunk size', () => {
    expect(chunk(CORPUS[2].text, 'x', 2, 0).map((c) => c.pos)).toEqual([0, 2, 4])
    expect(chunk(CORPUS[2].text, 'x', 1, 0).length).toBe(5)
  })
  it('overlap >= size still moves forward by one', () => {
    expect(chunk(CORPUS[2].text, 'x', 2, 5).map((c) => c.pos)).toEqual([0, 1, 2, 3, 4])
  })
})

describe('embedder', () => {
  const index = ingest(CORPUS)
  it('tokenize lowercases and drops . and ,', () => {
    expect(tokenize('Friends, Romans. OK')).toEqual(['friends', 'romans', 'ok'])
  })
  it('embeddings are unit length', () => {
    expect(norm(index.embedder.embed('deploy freezes apply'))).toBeCloseTo(1, 6)
  })
  it('a text of unknown words embeds to the zero vector', () => {
    expect(norm(index.embedder.embed('zebra xylophone'))).toBe(0)
    expect(index.embedder.unknown('the wifi password')).toEqual(['wifi', 'password'])
  })
  it('rare words outweigh common ones', () => {
    const terms = topTerms(index.embedder.embed(index.chunks[0].text), index.embedder.vocab, 40).map((t) => t.term)
    expect(terms.indexOf('laptop')).toBeLessThan(terms.indexOf('the'))
  })
})

describe('the four demo queries reproduce the Python output', () => {
  const index = ingest(CORPUS)
  const expected: [string, number[], string][] = [
    ['oncall.md', [0.47, 0.45, 0.09], 'Primary oncall must acknowledge pages within five minutes. [source 1: oncall.md] (sim 0.68)'],
    ['expenses.md', [0.53, 0.08, 0.06], 'Engineers may expense up to 500 dollars per year for learning materials. [source 1: expenses.md] (sim 0.59)'],
    ['oncall.md', [0.63, 0.61, 0.11], 'After an incident the oncall engineer writes the postmortem. [source 1: oncall.md] (sim 0.89)'],
    ['deploy-policy.md', [0.65, 0.6, 0.26], 'Deploy freezes apply during the last week of each quarter. [source 1: deploy-policy.md] (sim 0.65)'],
  ]
  DEMO_QUERIES.forEach((q, i) => {
    it(q, () => {
      const [source, sims, answer] = expected[i]
      const r = ask(index, q)
      expect(r.retrieved[0].chunk.source).toBe(source)
      r.retrieved.forEach((x, j) => expect(x.sim).toBeCloseTo(sims[j], 2))
      expect(r.answer.text).toBe(answer)
    })
  })
})

describe('failure cases from --show-failure', () => {
  const index = ingest(CORPUS)
  it('the paraphrase query only just clears the threshold', () => {
    const r = ask(index, 'can I deploy without the pipeline')
    expect(r.retrieved.map((x) => Number(x.sim.toFixed(2)))).toEqual([0.33, 0.24, 0.23])
    expect(r.answer.text).toBe('Deployments to production happen through the CI pipeline only. [source 1: deploy-policy.md] (sim 0.38)')
  })
  it('an absent answer is refused', () => {
    const r = ask(index, 'what is the wifi password')
    expect(r.answer.found).toBe(false)
    expect(r.answer.text).toBe('Not found in the provided context. (best match 0.28)')
  })
  it('an empty index refuses instead of crashing', () => {
    expect(ask(ingest([]), 'anything').answer.found).toBe(false)
  })
})

describe('prompt assembly', () => {
  it('matches the Python format exactly', () => {
    const p = assemblePrompt('q?', [{ chunk: { text: 'A. B.', source: 's.md', pos: 0 }, sim: 1 }])
    expect(p).toBe("Answer using ONLY the context below. If the answer is not in the\ncontext, say 'Not found in the provided context.'\n\nContext:\n[1] (s.md) A. B.\n\nQuestion: q?\nAnswer:")
  })
})
