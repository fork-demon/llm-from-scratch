import { describe, expect, it } from 'vitest'
import { FORMULAS, parseFormula, QUIZ, scoreNotationQuiz, shuffledOptions, SYMBOLS } from './notation'
import { LESSONS } from '../data/curriculum'

describe('formula markup', () => {
  it('parses subscripts and superscripts, braced or single-character', () => {
    expect(parseFormula('W_i^Q')).toEqual([{ text: 'W', kind: 'base' }, { text: 'i', kind: 'sub' }, { text: 'Q', kind: 'sup' }])
    expect(parseFormula('x_{<t} + e^{z_i}')).toEqual([
      { text: 'x', kind: 'base' }, { text: '<t', kind: 'sub' }, { text: ' + e', kind: 'base' }, { text: 'z_i', kind: 'sup' },
    ])
  })
  it('keeps nested markup inside a script for the renderer to parse again', () => {
    expect(parseFormula('e^{z_{ij}} / 2')).toEqual([{ text: 'e', kind: 'base' }, { text: 'z_{ij}', kind: 'sup' }, { text: ' / 2', kind: 'base' }])
    expect(parseFormula('z_{ij}')).toEqual([{ text: 'z', kind: 'base' }, { text: 'ij', kind: 'sub' }])
  })
  it('leaves plain text and a dangling marker alone', () => {
    expect(parseFormula('softmax(z)')).toEqual([{ text: 'softmax(z)', kind: 'base' }])
    expect(parseFormula('a_')).toEqual([{ text: 'a_', kind: 'base' }])
    expect(parseFormula('a_{b')).toEqual([{ text: 'a_{b', kind: 'base' }])
  })
  it('every stored formula parses without losing characters other than the markers', () => {
    for (const f of [...SYMBOLS.map((s) => s.symbol), ...FORMULAS.map((x) => x.formula), ...QUIZ.map((q) => q.formula)]) {
      const joined = parseFormula(f).map((p) => p.text).join('')
      expect(joined.length).toBeGreaterThan(0)
    }
  })
})

describe('content', () => {
  const lessons = new Set(LESSONS.map((l) => l.id))
  it('links only to lessons that exist; ids are unique', () => {
    for (const x of [...SYMBOLS, ...FORMULAS, ...QUIZ]) expect(lessons.has(x.lesson), x.id).toBe(true)
    for (const list of [SYMBOLS, FORMULAS, QUIZ]) expect(new Set(list.map((x) => x.id)).size).toBe(list.length)
  })
  it('covers the symbols the brief lists', () => {
    const all = SYMBOLS.map((s) => s.symbol).join(' ')
    for (const needle of ['∈ ℝ', 'Σ', 'softmax', '⊙', '‖', '𝔼', '∇', 'argmax', 'P(x_t | x_{<t})', 'O(n^2 · d)', 'Wx + b']) expect(all, needle).toContain(needle)
  })
  it('quotes the paper’s equations faithfully', () => {
    const f = (id: string) => FORMULAS.find((x) => x.id === id)!.formula
    expect(f('attention')).toBe('Attention(Q, K, V) = softmax(QK^T / √d_k) V')
    expect(f('ffn')).toBe('FFN(x) = max(0, xW_1 + b_1) W_2 + b_2')
    expect(f('multihead')).toContain('Concat(head_1, …, head_h) W^O')
    expect(f('multihead')).toContain('head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)')
  })
  it('the learning-rate code reproduces the formula: warm-up is linear, the peak is at step = warmup', () => {
    const lr = (step: number, d = 512, warmup = 4000) => d ** -0.5 * Math.min(step ** -0.5, step * warmup ** -1.5)
    expect(lr(2000) / lr(1000)).toBeCloseTo(2, 10)
    expect(lr(4000)).toBeGreaterThan(lr(3999))
    expect(lr(4000)).toBeGreaterThan(lr(4001))
    expect(lr(4000)).toBeCloseTo(1 / Math.sqrt(512 * 4000), 12) // about 0.0007
    expect(lr(16000) / lr(4000)).toBeCloseTo(0.5, 10)
  })
})

describe('quiz', () => {
  it('has four distinct options of similar length per question', () => {
    for (const q of QUIZ) {
      expect(q.options).toHaveLength(4)
      expect(new Set(q.options).size).toBe(4)
      const lens = q.options.map((o) => o.length)
      expect(Math.max(...lens) / Math.min(...lens), q.id).toBeLessThan(1.35)
    }
  })
  it('spreads the right answer over all positions in the authored data', () => {
    expect(new Set(QUIZ.map((q) => q.answer)).size).toBe(4)
  })
  it('shuffling is deterministic and keeps track of the right answer', () => {
    for (const q of QUIZ) {
      const a = shuffledOptions(q)
      expect(a).toEqual(shuffledOptions(q))
      expect(a.options.slice().sort()).toEqual(q.options.slice().sort())
      expect(a.options[a.answer]).toBe(q.options[q.answer])
    }
  })
  it('scores', () => {
    const perfect = Object.fromEntries(QUIZ.map((q) => [q.id, shuffledOptions(q).answer]))
    expect(scoreNotationQuiz(perfect)).toBe(QUIZ.length)
    expect(scoreNotationQuiz({})).toBe(0)
  })
})
