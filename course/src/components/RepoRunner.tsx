// "Run this file in your browser": runs one of the repository's NumPy-only scripts, unchanged, in
// the same Pyodide worker as the coding exercises. Sibling modules it imports are written next to it
// in Pyodide's filesystem first (see buildRepoProgram). The learner can edit a copy and run that.
//
//   <RepoRunner path="phase1-foundations/mlp_numpy.py" />
//   <RepoRunner path="phase6-engineering/eval_harness.py" title="Run the eval harness">
//     <p>Optional lead-in shown above the buttons.</p>
//   </RepoRunner>
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { CodeEditor } from './python'
import { pythonIsWarm, runPython, type PyPhase, type PyRunResult } from '../lib/pyRunner'
import { sourceUrl } from '../data/curriculum'
import { REPO_FILES, REPO_SCRIPTS, buildRepoProgram, isRepoPath, type RepoPath } from '../data/repoFiles'

const fileName = (p: string) => p.split('/').pop() ?? p

export function RepoRunner({ path, title, children }: { path: RepoPath; title?: string; children?: ReactNode }) {
  const known = isRepoPath(path)
  const original = known ? REPO_FILES[path] : ''
  const [value, setValue] = useState(original)
  const [editing, setEditing] = useState(false)
  const [phase, setPhase] = useState<PyPhase | null>(null)
  const [result, setResult] = useState<PyRunResult | null>(null)
  const [ranEdited, setRanEdited] = useState(false)
  // StrictMode mounts twice: set alive back to true on every mount (same pattern as python.tsx).
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => { alive.current = false }
  }, [])

  if (!known) return <p className="py-error-note">Missing repo script: {path}</p>
  const { deps, seconds } = REPO_SCRIPTS[path]
  const edited = value !== original

  const run = async () => {
    setResult(null)
    setRanEdited(edited)
    setPhase(pythonIsWarm() ? 'running' : 'loading')
    const r = await runPython(buildRepoProgram(path, edited ? value : undefined), [], (p) => alive.current && setPhase(p))
    if (!alive.current) return
    setPhase(null)
    setResult(r)
  }

  return (
    <div className="py-cell">
      <div className="code-bar">
        <span>{title ?? `Run ${fileName(path)} in your browser`}</span>
        <span className="spacer" />
        <a href={sourceUrl(path)} target="_blank" rel="noreferrer">{fileName(path)} ↗</a>
        <span className="py-badge">runs in your browser</span>
      </div>
      <div style={{ padding: '12px 18px 0', fontSize: 14, lineHeight: 1.6 }}>
        {children}
        <p style={{ margin: 0 }}>
          This runs <code>{path}</code> from the repository, unchanged, with Python and NumPy inside this page
          {deps.length > 0 && <> (plus {deps.map((d, i) => <span key={d}>{i > 0 && ', '}<code>{fileName(d)}</code></span>)}, which it imports)</>}.
          {' '}It takes about {seconds} {seconds === 1 ? 'second' : 'seconds'} once Python has loaded. Nothing is sent to a server.
        </p>
      </div>
      {editing && (
        <div style={{ maxHeight: 460, overflow: 'auto', marginTop: 12 }}>
          <CodeEditor value={value} onChange={setValue} label={`${fileName(path)}: editable copy`} minRows={12} />
        </div>
      )}
      <div className="py-actions">
        <button className="btn small primary" onClick={run} disabled={!!phase}>{edited ? 'Run your edited copy' : `Run ${fileName(path)}`}</button>
        <button className="btn small" onClick={() => setEditing(!editing)} aria-expanded={editing}>{editing ? 'Hide the code' : 'Edit a copy'}</button>
        {edited && <button className="btn small" onClick={() => { setValue(original); setResult(null) }}>Back to the original</button>}
        {phase && (
          <span className="py-status" role="status">
            {phase === 'loading' ? 'Starting Python in your browser. The first run downloads about 10 MB, later runs are instant.' : `Running ${fileName(path)}…`}
          </span>
        )}
      </div>
      {result && (
        <div className="py-output" aria-live="polite">
          {ranEdited && <p className="muted" style={{ margin: 0, fontSize: 13 }}>Output of your edited copy.</p>}
          {result.fatal && <p className="py-error-note">{result.fatal}</p>}
          {result.timedOut && <p className="py-error-note">Stopped after 20 seconds. {ranEdited ? 'Your edit may have made it much slower, or added a loop that never ends.' : 'The script took longer than the in-browser limit.'}</p>}
          {result.stdout && <pre className="py-stdout" style={{ maxHeight: 520, overflowY: 'auto' }}>{result.stdout}</pre>}
          {result.error && <pre className="py-traceback">{result.error}</pre>}
          {!result.stdout && !result.error && !result.fatal && !result.timedOut && <p className="muted" style={{ margin: 0, fontSize: 14 }}>Ran without printing anything.</p>}
        </div>
      )}
    </div>
  )
}
