// #/selftest: runs every coding exercise's reference solution (and its starter) through the
// in-browser Python runtime. If this page is all green, the browser runner agrees with real Python.
import { useEffect, useState } from 'react'
import { CODE_EXERCISES } from '../data/codeExercises'
import { runPython, summarise } from '../lib/pyRunner'

interface Row { id: string; solution: 'pending' | 'pass' | 'FAIL'; starter: 'pending' | 'ok' | 'ALREADY-PASSES'; detail: string }

export function SelfTestPage() {
  const [rows, setRows] = useState<Row[]>(() => CODE_EXERCISES.map((e) => ({ id: e.id, solution: 'pending', starter: 'pending', detail: '' })))
  const [state, setState] = useState<'idle' | 'running' | 'done'>('idle')

  const start = async () => {
    setState('running')
    for (const e of CODE_EXERCISES) {
      const sol = await runPython(e.solution, e.tests)
      const s = summarise(sol)
      const st = await runPython(e.starter, e.tests)
      const detail = s.allPassed ? '' : sol.fatal ?? sol.error ?? sol.results.filter((t) => !t.ok).map((t) => `${t.name}: ${t.message}`).join(' | ')
      setRows((prev) => prev.map((r) => (r.id === e.id ? { id: e.id, solution: s.allPassed ? 'pass' : 'FAIL', starter: summarise(st).allPassed ? 'ALREADY-PASSES' : 'ok', detail } : r)))
    }
    setState('done')
  }
  useEffect(() => { if (window.location.hash.includes('auto')) void start() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const bad = rows.filter((r) => r.solution === 'FAIL' || r.starter === 'ALREADY-PASSES').length
  return (
    <div>
      <h1 className="lesson-title">Python self-test</h1>
      <p className="lesson-question">Runs all {CODE_EXERCISES.length} coding exercises in your browser’s Python runtime.</p>
      <div className="btn-row"><button className="btn primary" onClick={start} disabled={state === 'running'}>Run the self-test</button>
        <span id="selftest-state" role="status">{state === 'done' ? `SELFTEST-DONE bad=${bad} total=${rows.length}` : state === 'running' ? 'Running…' : ''}</span></div>
      <div className="table-scroll"><table className="plain mono" style={{ fontSize: 13 }}>
        <thead><tr><th>exercise</th><th>solution</th><th>starter</th><th>detail</th></tr></thead>
        <tbody>{rows.map((r) => <tr key={r.id}><td>{r.id}</td><td style={{ color: r.solution === 'FAIL' ? 'var(--bad)' : r.solution === 'pass' ? 'var(--good)' : undefined }}>{r.solution}</td><td style={{ color: r.starter === 'ALREADY-PASSES' ? 'var(--bad)' : undefined }}>{r.starter}</td><td>{r.detail}</td></tr>)}</tbody>
      </table></div>
    </div>
  )
}
