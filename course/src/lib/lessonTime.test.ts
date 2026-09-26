import { readFileSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { LESSONS } from '../data/curriculum'
import { HANDS_ON, coreOf, estimateLesson, proseWords } from './lessonTime'

const dir = resolve(__dirname, '../lessons')
const source = (id: string) => readFileSync(resolve(dir, `${id}.tsx`), 'utf8')

describe('lesson time estimate', () => {
  it('counts prose, leaves out code and optional material', () => {
    expect(proseWords('<p>one two three four</p><Code>{`x = 1 and more code words`}</Code>')).toBe(4)
    expect(coreOf('<p>kept</p><DeepDive title="x"><p>gone</p></DeepDive>')).toBe('<p>kept</p>')
    expect(coreOf('<p>kept</p><details className="deep"><summary>More</summary><Exercise /></details>')).toBe('<p>kept</p>')
  })

  it('every lesson file exists and is estimated', () => {
    const files = readdirSync(dir).filter((f) => f.endsWith('.tsx') && !f.includes('.test.'))
    expect(files.length).toBe(LESSONS.length)
  })

  // The minutes shown to learners must stay close to what the content takes. When you add or cut
  // content, update the lesson's minutes in curriculum.ts (npx vite-node scripts/lessonTimes.ts prints them).
  for (const l of LESSONS) {
    if (HANDS_ON[l.id]) continue
    it(`${l.id}: stated minutes match the content`, () => {
      const est = estimateLesson(source(l.id)).minutes
      expect(Math.abs(l.minutes - est), `${l.id} says ${l.minutes} min, content suggests about ${est}`).toBeLessThanOrEqual(Math.max(5, est * 0.2))
    })
  }
})
