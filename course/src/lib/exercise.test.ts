import { describe, expect, it } from 'vitest'
import { checkNumeric, checkOrder, checkText, countInPlace, optionOrder, parseNumber, scoreQuiz } from './exercise'
import { highlight } from './highlight'

describe('exercise validation', () => {
  it('parses the ways people type numbers', () => {
    expect(parseNumber(' .5 ')).toBe(0.5)
    expect(parseNumber('1/2')).toBe(0.5)
    expect(parseNumber('50%')).toBe(0.5)
    expect(parseNumber('−3')).toBe(-3)
    expect(parseNumber('abc')).toBeNull()
    expect(parseNumber('1/0')).toBeNull()
    expect(parseNumber('')).toBeNull()
  })
  it('checks numbers with tolerance', () => {
    expect(checkNumeric('0.66', { value: 2 / 3, tolerance: 0.01 })).toBe(true)
    expect(checkNumeric('0.6', { value: 2 / 3, tolerance: 0.01 })).toBe(false)
    expect(checkNumeric('32', { value: 32 })).toBe(true)
  })
  it('checks text loosely', () => {
    expect(checkText(' (5, 3) ', ['(5,3)'])).toBe(true)
    expect(checkText('Softmax.', ['softmax'])).toBe(true)
    expect(checkText('relu', ['softmax'])).toBe(false)
  })
  it('checks orderings', () => {
    expect(checkOrder(['a', 'b'], ['a', 'b'])).toBe(true)
    expect(checkOrder(['b', 'a'], ['a', 'b'])).toBe(false)
    expect(countInPlace(['a', 'c', 'b'], ['a', 'b', 'c'])).toBe(1)
  })
  it('scores quizzes', () => {
    expect(scoreQuiz([0, 2, null], [0, 1, 2])).toBe(1)
  })
})

describe('quiz option order', () => {
  it('is a stable permutation', () => {
    const a = optionOrder('check-attention', 2, 4)
    expect(a).toEqual(optionOrder('check-attention', 2, 4))
    expect([...a].sort()).toEqual([0, 1, 2, 3])
  })
  it('spreads the first option across positions', () => {
    const positions = new Set<number>()
    for (let q = 0; q < 40; q++) positions.add(optionOrder('quiz', q, 4).indexOf(0))
    expect(positions.size).toBe(4)
  })
})

describe('highlighter', () => {
  it('round-trips the source text exactly', () => {
    const code = 'def f(x):  # comment\n    return "s" + str(3.5)\n'
    expect(highlight(code).map((t) => t.text).join('')).toBe(code)
  })
  it('classifies tokens', () => {
    const toks = highlight('for i in range(10): # hi')
    expect(toks.find((t) => t.text === 'for')?.kind).toBe('k')
    expect(toks.find((t) => t.text === 'range')?.kind).toBe('f')
    expect(toks.find((t) => t.text === '10')?.kind).toBe('n')
    expect(toks.find((t) => t.text === '# hi')?.kind).toBe('c')
  })
})
