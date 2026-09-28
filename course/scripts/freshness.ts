// Print the freshness register: what is due for review. Run: npx vite-node scripts/freshness.ts [YYYY-MM-DD]
import { FRESHNESS, dueForReview } from '../src/data/freshness'

const today = process.argv[2] ?? new Date().toISOString().slice(0, 10)
const due = dueForReview(today)
console.log(`${FRESHNESS.length} dated claims; ${due.length} due for review on ${today}`)
for (const c of due) console.log(`\n[${c.id}] ${c.lesson} (checked ${c.checked}, every ${c.every} months)\n  ${c.claim}\n  ${c.sources.join('\n  ')}`)
