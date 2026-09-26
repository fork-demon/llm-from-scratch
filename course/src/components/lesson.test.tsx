// A lesson shows one section at a time: the rail and the pager move between them, the address names the
// open section, and hidden sections stay in the page (hidden="until-found") so find-in-page can reach them.
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import Softmax from '../lessons/softmax'

const open = (c: HTMLElement) => [...c.querySelectorAll('.lesson-body > section.section')].filter((s) => !s.hasAttribute('hidden')).map((s) => s.id)

beforeEach(() => { localStorage.clear(); window.history.replaceState(null, '', '#/lesson/softmax') })
afterEach(() => cleanup())

describe('Lesson: one section at a time', () => {
  it('opens on the first section and keeps the others hidden but searchable', () => {
    const { container } = render(<Softmax />)
    expect(open(container)).toEqual(['why'])
    expect(container.querySelector('#remember')?.getAttribute('hidden')).toBe('until-found')
    expect(window.location.hash).toBe('#/lesson/softmax/why')
    // the lesson footer waits for the last section
    expect(container.querySelector('.lesson-foot')).toBeNull()
  })

  it('moves with the pager, the arrow keys and the rail, and remembers what was read', () => {
    const { container } = render(<Softmax />)
    fireEvent.click(screen.getByRole('button', { name: /^Next/ }))
    expect(open(container)).toEqual(['problem'])
    expect(window.location.hash).toBe('#/lesson/softmax/problem')

    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(open(container)).toEqual(['mental-model'])
    fireEvent.keyDown(window, { key: 'ArrowLeft' })
    expect(open(container)).toEqual(['problem'])

    // the rail: a phase name opens its first section; a dot opens its own section
    fireEvent.click(screen.getByRole('button', { name: 'Recap' }))
    expect(open(container)).toEqual(['remember'])
    expect(container.querySelector('.lesson-dot[aria-label^="Why are we learning this?"]')?.className).toContain('is-seen')
    expect(JSON.parse(localStorage.getItem('llm-fp-seen-softmax') ?? '[]')).toEqual(['why', 'problem', 'mental-model', 'remember'])
  })

  it('opens the section a link names, and the one find-in-page reaches', () => {
    window.history.replaceState(null, '', '#/lesson/softmax/math')
    const { container } = render(<Softmax />)
    expect(open(container)).toEqual(['math'])
    act(() => { container.querySelector('#check')!.dispatchEvent(new Event('beforematch')) })
    expect(open(container)).toEqual(['check'])
  })

  it('shows the lesson footer on the last section', () => {
    window.history.replaceState(null, '', '#/lesson/softmax/real-llm')
    const { container } = render(<Softmax />)
    expect(container.querySelector('.lesson-foot')).not.toBeNull()
    expect(screen.queryByRole('button', { name: /^Next/ })).toBeNull()
  })
})
