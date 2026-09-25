// The repository's NumPy-only scripts, bundled as text so <RepoRunner> can run them unchanged in
// Pyodide. The map is explicit (no glob) so only files verified to run in the browser are shipped.
// `seconds` is the typical run time in Pyodide 0.28 (Node on an M-series Mac; mlp_numpy measured 8 to 16 s
// depending on machine load, the closest to the limit). The in-browser runner stops anything after
// 20 s, so slower scripts are left out:
//   phase2-language/bigram_lm.py          ~24 s in Pyodide (trains two neural models)
//   phase6-engineering/speculative_demo.py ~34 s in Pyodide (100k-sample acceptance check)
// repoFiles.test.ts checks every file exists, matches the disk, and imports only numpy, the
// standard library, or a sibling listed in `deps`; it also runs each assembled program in real Python.
import mathPrimer from '../../../phase1-foundations/math_primer.py?raw'
import gradientDescent from '../../../phase1-foundations/gradient_descent.py?raw'
import mlpNumpy from '../../../phase1-foundations/mlp_numpy.py?raw'
import bpeTokenizer from '../../../phase2-language/bpe_tokenizer.py?raw'
import tinyWord2vec from '../../../phase2-language/tiny_word2vec.py?raw'
import attentionNumpy from '../../../phase3-transformers/attention_numpy.py?raw'
import kvCacheDemo from '../../../phase3-transformers/kv_cache_demo.py?raw'
import miniRag from '../../../phase4-modern-llms/mini_rag.py?raw'
import vectorDb from '../../../phase4-modern-llms/vector_db.py?raw'
import miniAgent from '../../../phase5-agents/mini_agent.py?raw'
import agentBudget from '../../../phase6-engineering/agent_budget.py?raw'
import batchingSim from '../../../phase6-engineering/batching_sim.py?raw'
import evalHarness from '../../../phase6-engineering/eval_harness.py?raw'
import quantizeDemo from '../../../phase6-engineering/quantize_demo.py?raw'

/** Repo path -> file text. */
export const REPO_FILES = {
  'phase1-foundations/math_primer.py': mathPrimer,
  'phase1-foundations/gradient_descent.py': gradientDescent,
  'phase1-foundations/mlp_numpy.py': mlpNumpy,
  'phase2-language/bpe_tokenizer.py': bpeTokenizer,
  'phase2-language/tiny_word2vec.py': tinyWord2vec,
  'phase3-transformers/attention_numpy.py': attentionNumpy,
  'phase3-transformers/kv_cache_demo.py': kvCacheDemo,
  'phase4-modern-llms/mini_rag.py': miniRag,
  'phase4-modern-llms/vector_db.py': vectorDb,
  'phase5-agents/mini_agent.py': miniAgent,
  'phase6-engineering/agent_budget.py': agentBudget,
  'phase6-engineering/batching_sim.py': batchingSim,
  'phase6-engineering/eval_harness.py': evalHarness,
  'phase6-engineering/quantize_demo.py': quantizeDemo,
} as const satisfies Record<string, string>

export type RepoPath = keyof typeof REPO_FILES

export interface RepoScript {
  /** Other repo files this one imports (via its own sys.path trick); written next to it before the run. */
  deps: RepoPath[]
  /** Measured Pyodide run time, rounded up. */
  seconds: number
}

/** Every mapped file can be run as a script. */
export const REPO_SCRIPTS: Record<RepoPath, RepoScript> = {
  'phase1-foundations/math_primer.py': { deps: [], seconds: 1 },
  'phase1-foundations/gradient_descent.py': { deps: [], seconds: 2 },
  'phase1-foundations/mlp_numpy.py': { deps: [], seconds: 12 },
  'phase2-language/bpe_tokenizer.py': { deps: [], seconds: 1 },
  'phase2-language/tiny_word2vec.py': { deps: [], seconds: 10 },
  'phase3-transformers/attention_numpy.py': { deps: [], seconds: 1 },
  'phase3-transformers/kv_cache_demo.py': { deps: [], seconds: 3 },
  'phase4-modern-llms/mini_rag.py': { deps: [], seconds: 1 },
  'phase4-modern-llms/vector_db.py': { deps: [], seconds: 6 },
  'phase5-agents/mini_agent.py': { deps: [], seconds: 1 },
  'phase6-engineering/agent_budget.py': { deps: ['phase5-agents/mini_agent.py'], seconds: 1 },
  'phase6-engineering/batching_sim.py': { deps: [], seconds: 2 },
  'phase6-engineering/eval_harness.py': { deps: ['phase4-modern-llms/mini_rag.py'], seconds: 4 },
  'phase6-engineering/quantize_demo.py': { deps: [], seconds: 1 },
}

export const isRepoPath = (p: string): p is RepoPath => Object.prototype.hasOwnProperty.call(REPO_FILES, p)

const moduleName = (path: string) => path.split('/').pop()!.replace(/\.py$/, '')
const usesNumpy = (text: string) => /^\s*(import numpy|from numpy)/m.test(text)

/**
 * One Python program that recreates the repo layout in a temp folder, then runs `path` as
 * `__main__` from its own folder, exactly like `cd phaseN && python file.py`. Sibling imports work
 * because the scripts' own `sys.path.insert(0, .../../phaseM)` lines find the files written here.
 * `mainText` replaces the script's text (the learner's edited copy). Global state (cwd, sys.path,
 * sys.argv, the modules this run loaded) is restored afterwards, because the worker is reused.
 */
export function buildRepoProgram(path: RepoPath, mainText?: string): string {
  const deps = REPO_SCRIPTS[path].deps
  const files: Record<string, string> = {}
  for (const d of deps) files[d] = REPO_FILES[d]
  files[path] = mainText ?? REPO_FILES[path]
  const numpy = Object.values(files).some(usesNumpy)
  const pyFiles = Object.entries(files).map(([p, t]) => `    ${JSON.stringify(p)}: ${JSON.stringify(t)},`).join('\n')
  const modules = JSON.stringify([...deps, path].map(moduleName))
  return `${numpy ? 'import numpy  # listed here so the runtime loads it before the script starts\n' : ''}import os as _os, sys as _sys, runpy as _runpy, tempfile as _tempfile
_ROOT = _os.path.join(_tempfile.gettempdir(), "llm-from-scratch")
_FILES = {
${pyFiles}
}
for _p, _text in _FILES.items():
    _full = _os.path.join(_ROOT, _p)
    _os.makedirs(_os.path.dirname(_full), exist_ok=True)
    with open(_full, "w", encoding="utf-8") as _f:
        _f.write(_text)
_main = _os.path.join(_ROOT, ${JSON.stringify(path)})
_saved = (_os.getcwd(), list(_sys.path), list(_sys.argv))
for _m in ${modules}:
    _sys.modules.pop(_m, None)
try:
    _os.chdir(_os.path.dirname(_main))
    _sys.path.insert(0, _os.path.dirname(_main))
    _sys.argv = [_main]
    _runpy.run_path(_main, run_name="__main__")
finally:
    _os.chdir(_saved[0])
    _sys.path[:] = _saved[1]
    _sys.argv[:] = _saved[2]
    for _m in ${modules}:
        _sys.modules.pop(_m, None)
`
}
