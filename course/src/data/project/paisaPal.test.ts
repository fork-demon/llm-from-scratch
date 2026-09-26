// The project holds together: every planned piece is a real exercise, in its lesson, built on paisa_pal.py.
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { CODE_EXERCISES } from '../codeExercises'
import { LESSONS } from '../curriculum'
import { PAISA_PAL, PROJECT_PIECES } from './paisaPal'

describe('the Paisa Pal project', () => {
  it('lists each lesson once, in course order', () => {
    const order = PROJECT_PIECES.map((p) => LESSONS.findIndex((l) => l.id === p.lesson))
    expect(order.every((i) => i >= 0)).toBe(true)
    expect([...order].sort((a, b) => a - b)).toEqual(order)
  })
  for (const p of PROJECT_PIECES) {
    it(`${p.piece}: exists, starts from paisa_pal.py, and is in its lesson`, () => {
      const e = CODE_EXERCISES.find((x) => x.id === p.exercise)
      expect(e, `missing code exercise ${p.exercise}`).toBeTruthy()
      expect(e!.lesson).toBe(p.lesson)
      expect(e!.project?.piece).toBe(p.piece)
      expect(e!.prelude?.startsWith(PAISA_PAL), 'the prelude must start with the shared project file').toBe(true)
      const lesson = readFileSync(resolve(__dirname, `../../lessons/${p.lesson}.tsx`), 'utf8')
      expect(lesson.includes(`<CodeExercise id="${p.exercise}"`), `${p.lesson}.tsx must show <CodeExercise id="${p.exercise}" />`).toBe(true)
    })
  }
})
