// Lesson 0.2: the same question put to four kinds of system.
// Three of them are small enough to run for real in the browser: an exact-match
// key-value store, a keyword search over a handful of documents, and a rule-based
// program. The LLM cannot run here, so its behaviour is described per question.

export type Verdict = 'works' | 'partial' | 'fails'
export interface Outcome {
  verdict: Verdict
  output: string // what comes back
  why: string // what happened at query time
}

/* ---------- 1. database: exact key -> stored value ---------- */
export const normaliseKey = (q: string): string => q.trim().toLowerCase().replace(/\s+/g, ' ')

export const DB_ROWS: Record<string, string> = {
  'what is the capital of france?': 'Paris',
  'what is the capital of japan?': 'Tokyo',
  'what is a cat?': 'A small domesticated carnivorous mammal.',
}

export const dbLookup = (q: string): Outcome => {
  const key = normaliseKey(q)
  const hit = DB_ROWS[key]
  return hit !== undefined
    ? { verdict: 'works', output: hit, why: `The key "${key}" matches a stored row exactly, so the stored value comes back.` }
    : { verdict: 'fails', output: 'NULL (0 rows)', why: `No row has exactly the key "${key}". A lookup cannot answer what nobody stored, but at least it says so.` }
}

/* ---------- 2. search engine: rank stored documents by shared keywords ---------- */
export interface Doc { title: string; text: string }
export const DOCS: Doc[] = [
  { title: 'France (encyclopedia)', text: 'Paris is the capital and largest city of France.' },
  { title: 'Japan (encyclopedia)', text: 'Tokyo is the capital of Japan.' },
  { title: 'Kubernetes docs: Pods', text: 'A pod is the smallest deployable unit in Kubernetes. A pod can crash and restart.' },
  { title: 'Poetry guide: the limerick', text: 'A limerick is a humorous poem of five lines with the rhyme scheme AABBA.' },
  { title: 'Match report (published yesterday)', text: 'Yesterday the Rovers won the match against United, 2 to 1.' },
  { title: 'Cats (pet care site)', text: 'A cat is a small furry animal often kept as a pet.' },
]

const STOP = new Set(['a', 'an', 'the', 'is', 'of', 'what', 'who', 'which', 'about', 'and', 'in', 's', 'to', 'me', 'write', 'can', 'with'])
export const keywords = (text: string): string[] => (text.toLowerCase().match(/[a-z]+/g) ?? []).filter((w) => !STOP.has(w))

/** Score = how many distinct query keywords appear in the document. */
export const searchDocs = (q: string): { doc: Doc; score: number }[] => {
  const qs = [...new Set(keywords(q))]
  return DOCS.map((doc) => {
    const words = new Set(keywords(`${doc.title} ${doc.text}`))
    return { doc, score: qs.filter((w) => words.has(w)).length }
  }).filter((r) => r.score > 0).sort((a, b) => b.score - a.score)
}

/* ---------- 3. traditional program: hand-written rules ---------- */
const CAPITALS: Record<string, string> = { france: 'Paris', japan: 'Tokyo', italy: 'Rome' }

export const ruleProgram = (q: string): Outcome => {
  const math = q.replace(/,/g, '').match(/(\d+)\s*([+\-*x×/])\s*(\d+)/)
  if (math) {
    const a = Number(math[1])
    const b = Number(math[3])
    const op = math[2]
    const r = op === '+' ? a + b : op === '-' ? a - b : op === '/' ? a / b : a * b
    return { verdict: 'works', output: r.toLocaleString('en-US'), why: 'The arithmetic rule matched. The CPU multiplies exactly, every time, for any numbers.' }
  }
  const cap = q.toLowerCase().match(/^what is the capital of ([a-z]+)\??$/)
  if (cap) {
    const c = CAPITALS[cap[1]]
    return c
      ? { verdict: 'works', output: c, why: 'The pattern "what is the capital of X?" matched, and X is in the table a programmer typed in.' }
      : { verdict: 'fails', output: `KeyError: '${cap[1]}'`, why: 'The pattern matched, but the programmer never added this country. It fails loudly.' }
  }
  return { verdict: 'fails', output: 'Error: no rule matches this input', why: 'A program only handles the cases somebody anticipated and wrote code for.' }
}

/* ---------- the preset questions ---------- */
export interface Question {
  id: string
  label: string // what kind of question this is
  text: string
  /** The LLM cannot run in the page: this is what typically happens, and why. */
  llm: Outcome
  /** A human judgement of the search result (the ranking itself is computed live). */
  searchVerdict: Verdict
  searchNote: string
  lesson: string // one-line takeaway
}

export const QUESTIONS: Question[] = [
  {
    id: 'fact',
    label: 'A famous fact',
    text: 'What is the capital of France?',
    searchVerdict: 'works',
    searchNote: 'The top page contains the answer. You still have to read it: a search engine returns documents, not answers.',
    llm: { verdict: 'works', output: 'The capital of France is Paris.', why: 'After “The capital of France is”, the token “ Paris” gets a very high probability, because training text continued that way thousands of times. No row was fetched.' },
    lesson: 'Everyone gets this right. You cannot tell the four systems apart from an easy question.',
  },
  {
    id: 'reworded',
    label: 'The same fact, reworded',
    text: "Which city is France's capital?",
    searchVerdict: 'works',
    searchNote: 'Keywords still overlap (“france”, “capital”, “city”), so the right page is still on top.',
    llm: { verdict: 'works', output: "France's capital is Paris.", why: 'Different tokens in, but the calculation lands on the same high-probability continuation. It never depended on exact wording.' },
    lesson: 'Exact-match systems break when the wording changes. The LLM does not care.',
  },
  {
    id: 'novel',
    label: 'Something nobody has written',
    text: 'Write a limerick about a Kubernetes pod',
    searchVerdict: 'fails',
    searchNote: 'It finds a page about pods and a page about limericks. Neither is a limerick about a pod. It can only return what already exists.',
    llm: { verdict: 'works', output: 'A pod that kept crashing at night / gave the on-call team quite a fright…', why: 'It has no such limerick stored. It produces one token by token: the patterns of limericks and the vocabulary of Kubernetes both shape the probabilities.' },
    lesson: 'Only a system that generates can answer a question that has no stored answer.',
  },
  {
    id: 'arithmetic',
    label: 'Exact arithmetic',
    text: 'What is 48,193 × 7,206?',
    searchVerdict: 'fails',
    searchNote: 'No keywords match. Nobody has published a page for this particular product.',
    llm: { verdict: 'partial', output: '347,218,758  (looks right; the true answer is 347,278,758)', why: 'Predicting digits token by token is not the same as carrying out a multiplication. Without help, models often get the first and last digits right and slip in the middle. Modern assistants do better by writing out the steps or by calling a calculator tool (see the Agents lesson).' },
    lesson: 'A four-line program beats a billion-parameter model at arithmetic. Fluent is not the same as exact.',
  },
  {
    id: 'recent',
    label: 'Something that happened yesterday',
    text: "Who won yesterday's match?",
    searchVerdict: 'works',
    searchNote: 'Search engines re-crawl the web all the time, so yesterday’s report is already in the index.',
    llm: { verdict: 'fails', output: 'I do not have information about that. (Or worse: a confident, invented score.)', why: 'Its numbers were frozen when training ended, months ago. Yesterday is not in them. The fix is to paste fresh text into the prompt (see the RAG lesson), not to change the model.' },
    lesson: 'An LLM’s knowledge has a cutoff date. Retrieval systems stay current.',
  },
  {
    id: 'nonexistent',
    label: 'A question with no true answer',
    text: 'What is the capital of Freedonia?',
    searchVerdict: 'partial',
    searchNote: 'Pages match the word “capital”, but none mention Freedonia. You can see for yourself that nothing relevant was found.',
    llm: { verdict: 'fails', output: 'The capital of Freedonia is Fredville.', why: 'The shape “The capital of X is …” strongly predicts a city-like name, so one may be produced. There is no “row not found”: a fluent answer comes out either way. Well-trained assistants often catch this one; with obscure real topics they catch it far less often.' },
    lesson: 'The database and the program fail loudly. The LLM can fail silently, in perfect English.',
  },
]

export const SYSTEMS = ['Database lookup', 'Search engine', 'Traditional program', 'LLM'] as const

/** All four outcomes for one question text. `preset` supplies the parts that cannot be computed. */
export const askAll = (text: string, preset?: Question): { db: Outcome; search: Outcome & { results: { doc: Doc; score: number }[] }; program: Outcome; llm: Outcome | null } => {
  const results = searchDocs(text).slice(0, 3)
  const search = {
    results,
    verdict: preset?.searchVerdict ?? (results.length ? 'partial' as const : 'fails' as const),
    output: results.length ? results.map((r) => r.doc.title).join(' | ') : 'No results',
    why: preset?.searchNote ?? (results.length ? 'These documents share keywords with your question. Whether any of them answers it is for you to judge.' : 'No stored document shares a keyword with your question.'),
  }
  return { db: dbLookup(text), search, program: ruleProgram(text), llm: preset?.llm ?? null }
}
