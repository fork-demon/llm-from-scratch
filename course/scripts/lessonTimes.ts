// Print the estimated minutes for every lesson; with --write, update curriculum.ts.
// Run: npx vite-node scripts/lessonTimes.ts [--write]
import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { LESSONS } from '../src/data/curriculum'
import { HANDS_ON, estimateLesson } from '../src/lib/lessonTime'

const root = resolve(__dirname, '..')
const curPath = resolve(root, 'src/data/curriculum.ts')
let cur = readFileSync(curPath, 'utf8')
let total = 0
for (const l of LESSONS) {
  const e = estimateLesson(readFileSync(resolve(root, `src/lessons/${l.id}.tsx`), 'utf8'))
  const mins = HANDS_ON[l.id] ? l.minutes : e.minutes
  if (!HANDS_ON[l.id]) total += mins
  console.log(`${l.id.padEnd(24)} ${String(l.minutes).padStart(4)} -> ${String(mins).padStart(4)}   words ${e.words}, exercises ${e.exercises}, code ${e.codeExercises}, quiz ${e.quiz}, labs ${e.labs}`)
  if (process.argv.includes('--write') && mins !== l.minutes) {
    const re = new RegExp(`(id: '${l.id}'[^\\n]*?minutes: )\\d+`)
    if (!re.test(cur)) throw new Error(`could not find minutes for ${l.id}`)
    cur = cur.replace(re, `$1${mins}`)
  }
}
console.log(`total without the capstone: ${(total / 60).toFixed(1)} hours`)
if (process.argv.includes('--write')) writeFileSync(curPath, cur)
