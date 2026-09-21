// Smoke test: every lesson file renders without throwing and follows the lesson format.
import { describe, expect, it } from 'vitest'
import { render } from '@testing-library/react'
import type { ComponentType } from 'react'
import { LESSONS } from '../data/curriculum'

const modules = import.meta.glob<{ default: ComponentType }>('./*.tsx', { eager: true })

describe('lessons render', () => {
  for (const [path, mod] of Object.entries(modules)) {
    if (path.includes('.test.')) continue
    const id = path.replace('./', '').replace('.tsx', '')
    it(`${id}`, () => {
      expect(LESSONS.some((l) => l.id === id), `${id} is not in curriculum.ts`).toBe(true)
      const Body = mod.default
      const { container, unmount } = render(<Body />)
      expect(container.querySelector('h1')?.textContent).toBeTruthy()
      // the golden rule: why first, recall and real-LLM connection at the end
      expect(container.querySelector('#why'), 'missing <Why>').not.toBeNull()
      expect(container.querySelector('#remember'), 'missing <Remember>').not.toBeNull()
      expect(container.textContent!.length).toBeGreaterThan(1500)
      unmount()
    })
  }
})
