// Main-thread side of the in-browser Python runner. One shared worker; a run that exceeds the
// time limit kills the worker (that is the only way to stop an infinite loop) and a new one is made.
export interface PyTest { name: string; code: string }
export interface PyTestResult { name: string; ok: boolean; message: string }
export interface PyRunResult {
  stdout: string
  error: string | null
  results: PyTestResult[]
  timedOut?: boolean
  fatal?: string
}
export type PyPhase = 'loading' | 'running'

const TIME_LIMIT_MS = 20000
let worker: Worker | null = null
let seq = 0
let warm = false

// BASE_URL keeps this correct under GitHub Pages (base './') and in dev. Resolving against
// document.baseURI would break on hash routes, where the browser would fetch index.html instead.
const workerUrl = () => new URL(`${import.meta.env.BASE_URL}py-worker.js`, window.location.href).href

const getWorker = () => {
  if (!worker) worker = new Worker(workerUrl())
  return worker
}

export const pythonIsWarm = () => warm

export function runPython(code: string, tests: PyTest[] = [], onPhase?: (p: PyPhase) => void): Promise<PyRunResult> {
  const id = ++seq
  const w = getWorker()
  return new Promise((resolve) => {
    let timer: ReturnType<typeof setTimeout> | undefined
    const finish = (r: PyRunResult) => {
      if (timer) clearTimeout(timer)
      w.removeEventListener('message', onMessage)
      w.removeEventListener('error', onError)
      resolve(r)
    }
    // One absolute deadline for the actual execution. It is set when the worker says it is
    // running (so the one-off runtime download is not counted) and never extended after that,
    // otherwise a program that never yields could keep pushing it back.
    let deadlineSet = false
    const startClock = () => {
      if (deadlineSet) return
      deadlineSet = true
      timer = setTimeout(() => {
        worker?.terminate()
        worker = null
        warm = false
        finish({ stdout: '', error: null, results: [], timedOut: true })
      }, TIME_LIMIT_MS)
    }
    const onMessage = (ev: MessageEvent) => {
      const m = ev.data
      if (m.id !== id) return
      if (m.phase === 'loading') onPhase?.('loading')
      else if (m.phase === 'running') { warm = true; onPhase?.('running'); startClock() }
      else if (m.phase === 'done') {
        if (!m.fatal) warm = true
        finish({ stdout: m.stdout ?? '', error: m.error ?? null, results: m.results ?? [], fatal: m.fatal })
      }
    }
    const onError = (e: ErrorEvent) => {
      worker?.terminate()
      worker = null
      finish({ stdout: '', error: null, results: [], fatal: `The Python runtime stopped unexpectedly. ${e.message ?? ''}` })
    }
    w.addEventListener('message', onMessage)
    w.addEventListener('error', onError)
    w.postMessage({ id, code, tests })
    // Safety net: if the runtime is already warm the 'running' message can be missed, and a
    // program that never yields would otherwise run forever. Give the download a grace period.
    // Safety net for the case where the 'running' message never arrives at all
    // (a worker that dies during boot); the normal path starts the clock on 'running'.
    setTimeout(startClock, 90000)
  })
}

/** Summarise a run for screen readers and the verdict line. */
export const summarise = (r: PyRunResult): { passed: number; total: number; allPassed: boolean } => {
  const total = r.results.length
  const passed = r.results.filter((t) => t.ok).length
  return { passed, total, allPassed: total > 0 && passed === total && !r.error && !r.timedOut && !r.fatal }
}
