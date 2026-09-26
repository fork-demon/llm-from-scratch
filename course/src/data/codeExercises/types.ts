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
  /**
   * Code that runs before the learner's code and before the tests, shown read-only above the editor.
   * Project exercises use it for the shared Paisa Pal data (data/project/paisaPal.ts) plus any pieces
   * "already written" earlier in the project. The learner's starter must not repeat it.
   */
  prelude?: string
  /** Part of the Paisa Pal support-bot project: which piece of the bot this exercise builds. */
  project?: { piece: string }
  /** Repo file this function lives in, e.g. "phase3-transformers/attention_numpy.py". */
  source?: string
}

/**
 * The program actually run for an exercise: the prelude executes as its own file (paisa_pal.py), so
 * tracebacks from the learner's code keep their own line numbers (off by one line only).
 */
export const withPrelude = (prelude: string | undefined, code: string) =>
  prelude ? `exec(compile(${JSON.stringify(prelude)}, "paisa_pal.py", "exec"))\n${code}` : code
