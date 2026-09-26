// Hand a snippet from a lesson to the Python playground page (#/python).
const KEY = 'llm-fp-playground'

export interface PlaygroundSeed { code: string; from?: string }

/** Lesson snippets are excerpts: add the NumPy import they assume, and say where the code came from. */
export const seedFromSnippet = (code: string, from?: string): string => {
  const needsNp = /\bnp\./.test(code) && !/^\s*import numpy as np/m.test(code)
  const head = `# ${from ? `From the lesson: ${from}. ` : ''}This is an excerpt, so it may use names defined elsewhere in the lesson.\n`
  return `${head}${needsNp ? 'import numpy as np\n' : ''}\n${code}\n`
}

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
