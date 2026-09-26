import { useId, useState } from 'react'
import { RunPython } from '../components/python'
import { REPO_FILES, REPO_SCRIPTS, type RepoPath } from '../data/repoFiles'
import { takeSeed } from '../lib/playground'

const EXAMPLE = `import numpy as np

# Attention in six lines: "bank" looks at "the river bank"
tokens = ["the", "river", "bank"]
Q = np.array([[0, 0], [0, 0], [2, 2]])   # queries: only "bank" is looking
K = np.array([[0, 0], [2, 0], [1, 1]])   # keys: what each token offers
V = np.array([[0.0, 0.1], [1.0, 0.8], [0.5, 0.9]])

scores = Q @ K.T / np.sqrt(K.shape[1])
weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
print("bank attends:", {t: round(float(w), 2) for t, w in zip(tokens, weights[2])})
print("new vector for bank:", (weights @ V)[2].round(2))
`

// scripts that import a sibling file need the repo runner in their lesson; the playground runs one file
const SCRIPTS = (Object.keys(REPO_SCRIPTS) as RepoPath[]).filter((p) => REPO_SCRIPTS[p].deps.length === 0)

/** #/python: an editor with Python 3 and NumPy, running in the browser. Lessons can hand it a snippet. */
export function PlaygroundPage() {
  const [seed] = useState(takeSeed)
  const [start, setStart] = useState<string>(seed ? 'lesson' : 'example')
  const [n, setN] = useState(0) // bumps to reload the editor
  const id = useId()
  const code = start === 'lesson' && seed ? seed.code : start === 'example' ? EXAMPLE : REPO_FILES[start as RepoPath] ?? EXAMPLE

  return (
    <div>
      <h1 className="lesson-title">Python playground</h1>
      <p className="lesson-question">Python 3 with NumPy, running in your browser. Change anything and press Run, or Ctrl/Cmd+Enter.</p>
      <div className="playground-bar">
        <label htmlFor={id}>Start from</label>
        <select id={id} className="input" value={start} onChange={(e) => { setStart(e.target.value); setN(n + 1) }}>
          {seed && <option value="lesson">The snippet from {seed.from ?? 'the lesson'}</option>}
          <option value="example">A small attention example</option>
          <optgroup label="Scripts from the repository">
            {SCRIPTS.map((p) => <option key={p} value={p}>{p}</option>)}
          </optgroup>
        </select>
      </div>
      <RunPython key={`${start}-${n}`} code={code} title={start.includes('/') ? start.split('/').pop()! : 'playground.py'} source={start.includes('/') ? start : undefined} minRows={16} />
      <p className="muted" style={{ fontSize: 14 }}>
        The first run downloads Python (about 10 MB); after that it starts at once. Only the standard library and NumPy are available, so PyTorch code
        from the lessons runs on your own machine instead. Nothing you write is sent anywhere.
      </p>
    </div>
  )
}
