// A coding exercise that runs in the browser. Everything is plain strings so the same data
// can be checked by real Python in src/data/codeExercises/codeExercises.test.ts.
export interface CodeExerciseDef {
  id: string // globally unique: "<lesson-id>-code-<slug>"
  lesson: string // lesson id it appears in
  title: string
  /** What to do. Plain text; blank lines separate paragraphs; `backticks` become inline code. */
  prompt: string
  /** What the learner starts with. Must NOT pass all the tests. */
  starter: string
  /** A reference answer. Must pass all the tests. Shown only after the hints are used. */
  solution: string
  /** Each test runs after the learner's code, in the same namespace. Use assert with a helpful message. */
  tests: { name: string; code: string }[]
  /** 2 to 3 hints, each giving away more. */
  hints: string[]
  /** Why the solution is right, and what to notice. */
  explanation: string
  /** Repo file this function lives in, e.g. "phase3-transformers/attention_numpy.py". */
  source?: string
}
