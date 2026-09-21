// Python that runs in the browser: a small editor, a scratch runner, and coding exercises
// whose hidden tests check the learner's function. The runtime is Pyodide in a Web Worker.
import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { highlight } from '../lib/highlight'
import { pythonIsWarm, runPython, summarise, type PyPhase, type PyRunResult, type PyTest } from '../lib/pyRunner'
import { completeExercise, updateProgress, useProgress } from '../lib/progress'
import { codeExerciseById } from '../data/codeExercises'
import { sourceUrl } from '../data/curriculum'

/* ---------- editor: a textarea over a highlighted copy of itself ---------- */
export function CodeEditor({ value, onChange, label, minRows = 6 }: { value: string; onChange: (v: string) => void; label: string; minRows?: number }) {
  const pre = useRef<HTMLPreElement>(null)
  const rows = Math.max(minRows, value.split('\n').length + 1)

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const ta = e.currentTarget
    const { selectionStart: a, selectionEnd: b } = ta
    const set = (next: string, caret: number) => {
      onChange(next)
      requestAnimationFrame(() => { ta.selectionStart = ta.selectionEnd = caret })
    }
    if (e.key === 'Tab' && !e.shiftKey && !e.altKey && !e.ctrlKey && !e.metaKey) {
      // Tab indents. Escape then Tab leaves the editor, so keyboard users are never trapped.
      if (ta.dataset.escape === '1') { ta.dataset.escape = ''; return }
      e.preventDefault()
      set(value.slice(0, a) + '    ' + value.slice(b), a + 4)
    } else if (e.key === 'Escape') {
      ta.dataset.escape = '1'
    } else if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
      const lineStart = value.lastIndexOf('\n', a - 1) + 1
      const line = value.slice(lineStart, a)
      const indent = (line.match(/^\s*/) ?? [''])[0] + (line.trimEnd().endsWith(':') ? '    ' : '')
      e.preventDefault()
      set(value.slice(0, a) + '\n' + indent + value.slice(b), a + 1 + indent.length)
    } else if (e.key === 'Backspace' && a === b) {
      const lineStart = value.lastIndexOf('\n', a - 1) + 1
      const before = value.slice(lineStart, a)
      if (before.length >= 4 && /^ +$/.test(before) && before.length % 4 === 0) {
        e.preventDefault()
        set(value.slice(0, a - 4) + value.slice(a), a - 4)
      }
    }
  }

  return (
    <div className="py-editor">
      <pre ref={pre} aria-hidden className="py-editor-view"><code>{highlight(value).map((t, i) => (t.kind ? <span key={i} className={`tok-${t.kind}`}>{t.text}</span> : t.text))}{'\n'}</code></pre>
      <textarea
        className="py-editor-input"
        aria-label={label}
        aria-description="Tab indents. Press Escape and then Tab to leave the editor."
        value={value}
        rows={rows}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        onScroll={(e) => { if (pre.current) { pre.current.scrollLeft = e.currentTarget.scrollLeft; pre.current.scrollTop = e.currentTarget.scrollTop } }}
      />
    </div>
  )
}

/* ---------- shared run state ---------- */
function useRunner(tests: PyTest[]) {
  const [phase, setPhase] = useState<PyPhase | null>(null)
  const [result, setResult] = useState<PyRunResult | null>(null)
  // StrictMode mounts, unmounts and remounts: set alive back to true on every mount,
  // or the cleanup from the first pass would silently discard every later result.
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => { alive.current = false }
  }, [])
  const run = async (code: string) => {
    setResult(null)
    setPhase(pythonIsWarm() ? 'running' : 'loading')
    const r = await runPython(code, tests, (p) => alive.current && setPhase(p))
    if (!alive.current) return r
    setPhase(null)
    setResult(r)
    return r
  }
  return { phase, result, run, clear: () => setResult(null) }
}

function RunStatus({ phase }: { phase: PyPhase | null }) {
  if (!phase) return null
  return (
    <span className="py-status" role="status">
      {phase === 'loading' ? 'Starting Python in your browser. The first run downloads about 10 MB, later runs are instant.' : 'Running…'}
    </span>
  )
}

function Output({ result, showTests }: { result: PyRunResult; showTests: boolean }) {
  return (
    <div className="py-output" aria-live="polite">
      {result.fatal && <p className="py-error-note">{result.fatal}</p>}
      {result.timedOut && <p className="py-error-note">Stopped after 20 seconds. That usually means a loop that never ends. Check the condition of your <code>while</code> or the range of your <code>for</code>.</p>}
      {result.stdout && <pre className="py-stdout">{result.stdout}</pre>}
      {result.error && <pre className="py-traceback">{result.error}</pre>}
      {!result.stdout && !result.error && !result.fatal && !result.timedOut && !showTests && <p className="muted" style={{ margin: 0, fontSize: 14 }}>Ran without printing anything. Add a <code>print(...)</code> to see a value.</p>}
      {showTests && result.results.length > 0 && (
        <ul className="py-tests">
          {result.results.map((t) => (
            <li key={t.name} className={t.ok ? 'ok' : 'no'}>
              <span className="py-test-mark" aria-hidden>{t.ok ? '✓' : '✗'}</span>
              <span><span className="sr-only">{t.ok ? 'Passed: ' : 'Failed: '}</span>{t.name}{!t.ok && t.message && <span className="py-test-msg">{t.message}</span>}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/* ---------- a scratch cell: edit, run, see the output ---------- */
export function RunPython({ code, title = 'Python', source }: { code: string; title?: string; source?: string }) {
  const initial = code.replace(/^\n/, '').replace(/\s+$/, '') + '\n'
  const [value, setValue] = useState(initial)
  const { phase, result, run, clear } = useRunner([])
  return (
    <div className="py-cell">
      <div className="code-bar">
        <span>{title}</span>
        <span className="spacer" />
        {source && <a href={sourceUrl(source)} target="_blank" rel="noreferrer">{source.split('/').pop()} ↗</a>}
        <span className="py-badge">runs in your browser</span>
      </div>
      <CodeEditor value={value} onChange={setValue} label={`${title}: editable Python`} />
      <div className="py-actions">
        <button className="btn small primary" onClick={() => run(value)} disabled={!!phase}>Run</button>
        {value !== initial && <button className="btn small" onClick={() => { setValue(initial); clear() }}>Reset</button>}
        <RunStatus phase={phase} />
      </div>
      {result && <Output result={result} showTests={false} />}
    </div>
  )
}

/* ---------- a checked coding exercise ---------- */
const draftKey = (id: string) => `llm-fp-code-${id}`
const Prose = ({ text }: { text: string }) => (
  <>
    {text.trim().split(/\n\s*\n/).map((para, i) => (
      <p key={i}>{para.split(/(`[^`]+`)/).map((part, j) => (part.startsWith('`') && part.endsWith('`') ? <code key={j}>{part.slice(1, -1)}</code> : part))}</p>
    ))}
  </>
)

export function CodeExercise({ id, children }: { id: string; children?: ReactNode }) {
  const def = codeExerciseById(id)
  const progress = useProgress()
  const uid = useId()
  const [value, setValue] = useState(() => {
    if (!def) return ''
    try { return localStorage.getItem(draftKey(id)) ?? def.starter } catch { return def.starter }
  })
  const [hints, setHints] = useState(0)
  const [showSolution, setShowSolution] = useState(false)
  const { phase, result, run, clear } = useRunner(def?.tests ?? [])

  useEffect(() => {
    if (!def) return
    try {
      if (value === def.starter) localStorage.removeItem(draftKey(id))
      else localStorage.setItem(draftKey(id), value)
    } catch { /* storage unavailable: the draft lives for this visit only */ }
  }, [value, id, def])

  if (!def) return <p className="py-error-note">Missing coding exercise: {id}</p>
  const done = !!progress.exercises[id]
  const summary = result ? summarise(result) : null

  const check = async () => {
    const r = await run(value)
    if (summarise(r).allPassed) updateProgress((p) => completeExercise(p, id))
  }

  return (
    <div className="exercise" id={`ex-${id}`}>
      <div className="exercise-head">
        <span className="ex-type">code</span>
        <span className="ex-title">{def.title}</span>
        {done && <span className="ex-done">Done</span>}
      </div>
      <div className="exercise-body">
        <Prose text={def.prompt} />
        {children}
        <div className="py-cell">
          <div className="code-bar">
            <span>your code</span>
            <span className="spacer" />
            {def.source && <a href={sourceUrl(def.source)} target="_blank" rel="noreferrer">{def.source.split('/').pop()} ↗</a>}
            <span className="py-badge">runs in your browser</span>
          </div>
          <CodeEditor value={value} onChange={setValue} label={`${def.title}: your Python code`} minRows={8} />
          <div className="py-actions">
            <button className="btn small primary" onClick={check} disabled={!!phase}>Run the tests</button>
            {value !== def.starter && <button className="btn small" onClick={() => { setValue(def.starter); clear() }}>Start over</button>}
            <RunStatus phase={phase} />
            {summary && !phase && (
              <span className={`verdict ${summary.allPassed ? 'ok' : 'no'}`} role="status">
                {summary.allPassed ? `All ${summary.total} tests pass.` : result?.error ? 'Your code raised an error before the tests could run.' : summary.total ? `${summary.passed} of ${summary.total} tests pass.` : ''}
              </span>
            )}
          </div>
          {result && <Output result={result} showTests />}
        </div>

        {def.hints.slice(0, hints).map((h, i) => <div className="hint" key={i}><b>Hint {i + 1}</b><Prose text={h} /></div>)}
        {(showSolution || summary?.allPassed) && (
          <div className="solution">
            <b>{summary?.allPassed && !showSolution ? 'Why it works' : 'One solution'}</b>
            {showSolution && <pre className="py-solution"><code>{highlight(def.solution.trim()).map((t, i) => (t.kind ? <span key={i} className={`tok-${t.kind}`}>{t.text}</span> : t.text))}</code></pre>}
            <Prose text={def.explanation} />
          </div>
        )}
        <div className="btn-row" style={{ marginTop: 12 }} aria-describedby={uid}>
          {hints < def.hints.length && !showSolution && (
            <button className="btn small" onClick={() => setHints(hints + 1)}>{hints === 0 ? 'I am stuck: give me a hint' : `Hint ${hints + 1} of ${def.hints.length}`}</button>
          )}
          {!showSolution && (hints >= def.hints.length || summary?.allPassed) && (
            <button className="btn small" onClick={() => setShowSolution(true)}>{summary?.allPassed ? 'Compare with a reference solution' : 'Show a solution'}</button>
          )}
        </div>
      </div>
    </div>
  )
}
