// Runs learner Python in a Web Worker with Pyodide, so a slow or infinite loop never
// freezes the page (the main thread can terminate this worker). Classic worker, no bundling.
/* global importScripts, loadPyodide */
const PYODIDE_VERSION = '0.28.3'
const BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`
let ready = null

function boot() {
  if (!ready) {
    ready = (async () => {
      importScripts(`${BASE}pyodide.js`)
      return loadPyodide({ indexURL: BASE })
    })()
    ready.catch(() => { ready = null }) // allow a retry after a network failure
  }
  return ready
}

// Each test runs in the namespace the learner's code created. One failure does not stop the others.
const HARNESS = `
def _run_tests(_ns, _tests):
    _out = []
    for _name, _code in _tests:
        try:
            exec(_code, _ns)
            _out.append((_name, True, ''))
        except AssertionError as e:
            _out.append((_name, False, str(e) or 'assertion failed'))
        except Exception as e:
            _out.append((_name, False, type(e).__name__ + ': ' + str(e)))
    return _out
`

// Pyodide tracebacks start with frames from its own internals: keep the part about the learner's code.
function cleanError(msg) {
  const lines = msg.split('\n')
  const i = lines.findIndex((l) => l.includes('File "<exec>"'))
  const kept = i >= 0 ? ['Traceback (most recent call last):', ...lines.slice(i)] : lines
  return kept.join('\n').trim()
}

self.onmessage = async (ev) => {
  const { id, code, tests } = ev.data
  let py
  try {
    self.postMessage({ id, phase: 'loading' })
    py = await boot()
  } catch (e) {
    self.postMessage({ id, phase: 'done', fatal: 'Could not load the Python runtime. Check your internet connection and try again. (' + e + ')' })
    return
  }
  const out = []
  py.setStdout({ batched: (s) => out.push(s) })
  py.setStderr({ batched: (s) => out.push(s) })
  try {
    await py.loadPackagesFromImports(code + '\n' + (tests || []).map((t) => t.code).join('\n'))
    self.postMessage({ id, phase: 'running' })
    const ns = py.globals.get('dict')()
    let error = null
    try {
      py.runPython(code, { globals: ns })
    } catch (e) {
      error = cleanError(String(e.message || e))
    }
    let results = []
    if (!error && tests && tests.length) {
      py.runPython(HARNESS, { globals: ns })
      const run = ns.get('_run_tests')
      const pyTests = py.toPy(tests.map((t) => [t.name, t.code]))
      const res = run(ns, pyTests)
      results = res.toJs().map(([name, ok, message]) => ({ name, ok, message }))
      res.destroy(); pyTests.destroy(); run.destroy()
    }
    ns.destroy()
    self.postMessage({ id, phase: 'done', stdout: out.join('\n'), error, results })
  } catch (e) {
    self.postMessage({ id, phase: 'done', stdout: out.join('\n'), error: cleanError(String(e.message || e)), results: [] })
  }
}
