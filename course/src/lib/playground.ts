// Hand a snippet from a lesson to the Python playground page (#/python).
const KEY = 'llm-fp-playground'

export interface PlaygroundSeed { code: string; from?: string }

/**
 * A lesson excerpt made runnable: `setup` defines the made-up inputs it needs, the excerpt itself follows
 * unchanged, and `show` prints the result. <Code> offers "Try it" only when a block has these, and
 * tryIt.test.tsx runs every such program in real Python, so a learner never meets a NameError.
 */
export const assembleTryIt = ({ code, setup, show }: { code: string; setup?: string; show?: string }): string => {
  const trim = (s?: string) => (s ?? '').replace(/^\n+/, '').replace(/\s+$/, '')
  const parts = [
    trim(setup) && `# Setup: small made-up inputs so this excerpt runs on its own\n${trim(setup)}`,
    `# The code from the lesson\n${trim(code)}`,
    trim(show) && `# See the result\n${trim(show)}`,
  ].filter(Boolean)
  return `${parts.join('\n\n')}\n`
}

export const seedFromSnippet = (program: string, from?: string): string =>
  `${from ? `# From the lesson: ${from}\n\n` : ''}${program}`

export function openInPlayground(code: string, from?: string) {
  try { sessionStorage.setItem(KEY, JSON.stringify({ code: seedFromSnippet(code, from), from } satisfies PlaygroundSeed)) } catch { /* storage unavailable */ }
  window.location.hash = '#/python'
}

/** Read (and clear) a snippet handed over by a lesson. */
export function takeSeed(): PlaygroundSeed | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    if (!raw) return null
    sessionStorage.removeItem(KEY)
    return JSON.parse(raw) as PlaygroundSeed
  } catch { return null }
}
