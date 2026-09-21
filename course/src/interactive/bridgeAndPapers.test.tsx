// Interaction smoke tests for the four interactives of lessons 10.3 and 10.4:
// every state must render without throwing, and the key numbers must appear.
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { ConfigReader } from './ConfigReader'
import { TensorNameMatcher } from './TensorNameMatcher'
import { PaperMap } from './PaperMap'
import { NotationDecoder } from './NotationDecoder'
import { SECTIONS, MARKS_KEY } from '../lib/paperMap'
import { FORMULAS, QUIZ, shuffledOptions, SYMBOLS } from '../lib/notation'
import { itemsFor } from '../lib/tensorNames'

afterEach(cleanup)

describe('ConfigReader', () => {
  it('sizes every preset and reacts to edits and to bad input', () => {
    const { container } = render(<ConfigReader />)
    expect(container.textContent).toContain('124,439,808')
    fireEvent.click(screen.getByRole('button', { name: 'Llama 3 8B' }))
    expect(container.textContent).toContain('8,030,261,248')
    expect(container.textContent).toContain('GQA')
    const box = screen.getByLabelText('config.json') as HTMLTextAreaElement
    fireEvent.change(box, { target: { value: box.value.replace('"num_key_value_heads": 8', '"num_key_value_heads": 32') } })
    expect(container.textContent).toContain('8,835,567,616') // plain multi-head attention: K and V grow fourfold, + 805,306,368
    expect(container.textContent).toContain('524 KB') // and so does the KV cache per token
    fireEvent.click(screen.getByRole('button', { name: 'Mistral 7B v0.1' }))
    expect(container.textContent).toContain('7,241,732,096')
    expect(container.textContent).toContain('capped by the sliding window')
    fireEvent.click(screen.getByRole('button', { name: 'Qwen2.5 0.5B' }))
    expect(container.textContent).toContain('494,032,768')
    fireEvent.change(box, { target: { value: '{ oops' } })
    expect(screen.getByRole('alert').textContent).toMatch(/not valid JSON/)
  })
})

describe('TensorNameMatcher', () => {
  it('checks answers and explains the right ones, in both rounds', () => {
    render(<TensorNameMatcher />)
    for (const round of ['gpt2', 'llama'] as const) {
      if (round === 'llama') fireEvent.click(screen.getByRole('button', { name: /Round 2/ }))
      const items = itemsFor(round)
      items.forEach((t, i) => fireEvent.change(screen.getByLabelText(`What is ${t.name}?`), { target: { value: i === 0 ? items[1].concept : t.concept } }))
      fireEvent.click(screen.getByRole('button', { name: 'Check my matches' }))
      expect(screen.getByRole('status').textContent).toContain(`${items.length - 1} of ${items.length} right`)
      fireEvent.change(screen.getByLabelText(`What is ${items[0].name}?`), { target: { value: items[0].concept } })
      fireEvent.click(screen.getByRole('button', { name: 'Check my matches' }))
      expect(screen.getByRole('status').textContent).toContain(`${items.length} of ${items.length} right`)
    }
  })
})

describe('PaperMap', () => {
  it('renders every section and stores marks under its own key', () => {
    window.localStorage.clear()
    const { container } = render(<PaperMap />)
    const nav = screen.getByRole('navigation', { name: 'Sections of the paper' })
    for (const s of SECTIONS) {
      fireEvent.click(within(nav).getAllByRole('button')[SECTIONS.indexOf(s)])
      expect(container.querySelector('h4')?.textContent).toBe(s.title)
    }
    fireEvent.click(screen.getByRole('button', { name: 'understood' }))
    expect(JSON.parse(window.localStorage.getItem(MARKS_KEY)!)).toEqual({ s7: 'understood' })
    expect(container.textContent).toContain('read: 1 of 16')
  })
})

describe('NotationDecoder', () => {
  it('renders every symbol, every formula step, and scores the quiz', () => {
    const { container } = render(<NotationDecoder />)
    for (const s of SYMBOLS) {
      fireEvent.click(within(screen.getByRole('group', { name: 'Symbols' })).getAllByRole('button')[SYMBOLS.indexOf(s)])
      expect(container.textContent).toContain(s.code)
    }
    fireEvent.click(screen.getByRole('button', { name: 'A whole formula' }))
    for (const f of FORMULAS) {
      fireEvent.click(screen.getByRole('button', { name: f.name }))
      for (let i = 1; i < f.steps.length; i++) fireEvent.click(screen.getByRole('button', { name: /Next piece/ }))
      expect(container.textContent).toContain(f.steps[f.steps.length - 1].code)
    }
    fireEvent.click(screen.getByRole('button', { name: 'Quiz: pick the code' }))
    for (const q of QUIZ) {
      const sh = shuffledOptions(q)
      const radios = container.querySelectorAll<HTMLInputElement>(`input[name="nd-${q.id}"]`)
      fireEvent.click(radios[sh.answer])
    }
    fireEvent.click(screen.getByRole('button', { name: 'Check my answers' }))
    expect(screen.getByRole('status').textContent).toContain(`${QUIZ.length} / ${QUIZ.length}`)
  })
})
