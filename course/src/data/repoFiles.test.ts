// The repo scripts that <RepoRunner> runs in the browser: every mapped file must exist, match the
// repository byte for byte, and import only numpy, the standard library, or a sibling it declares.
// The assembled program (the exact string the browser runs) is also executed with real Python.
import { describe, expect, it } from 'vitest'
import { execFileSync, spawnSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { REPO_FILES, REPO_SCRIPTS, buildRepoProgram, type RepoPath } from './repoFiles'

const ROOT = resolve(__dirname, '../../..')
const paths = Object.keys(REPO_FILES) as RepoPath[]

const candidates = [process.env.PYTHON, '/Users/arvind/miniconda3/bin/python3', 'python3', 'python'].filter(Boolean) as string[]
const python = candidates.find((p) => {
  try { return spawnSync(p, ['-c', 'import numpy'], { stdio: 'ignore' }).status === 0 } catch { return false }
})

/** Top-level module names a file imports (import a, b / from a.b import c). Relative imports excluded. */
const importsOf = (text: string): string[] => {
  const mods = new Set<string>()
  for (const line of text.split('\n')) {
    const m = line.match(/^\s*import\s+(.+)$/)
    if (m) m[1].split('#')[0].split(',').forEach((part) => mods.add(part.trim().split(/\s+/)[0].split('.')[0]))
    const f = line.match(/^\s*from\s+([\w.]+)\s+import\s/)
    if (f && !f[1].startsWith('.')) mods.add(f[1].split('.')[0])
  }
  mods.delete('')
  return [...mods]
}

const stdlib: Set<string> = python
  ? new Set(JSON.parse(execFileSync(python, ['-c', 'import sys, json; print(json.dumps(sorted(sys.stdlib_module_names)))'], { encoding: 'utf8' })))
  : new Set()

describe('repo files for the in-browser runner', () => {
  it.each(paths)('%s exists and matches the repository', (p) => {
    const disk = resolve(ROOT, p)
    expect(existsSync(disk)).toBe(true)
    expect(REPO_FILES[p]).toBe(readFileSync(disk, 'utf8'))
  })

  it('every script has an entry and its deps are in the map', () => {
    expect(Object.keys(REPO_SCRIPTS).sort()).toEqual([...paths].sort())
    for (const p of paths) for (const d of REPO_SCRIPTS[p].deps) expect(paths, `${p} -> ${d}`).toContain(d)
  })

  it('every script stays within the 20 s in-browser limit', () => {
    for (const p of paths) expect(REPO_SCRIPTS[p].seconds, p).toBeLessThanOrEqual(15)
  })

  it.skipIf(!python).each(paths)('%s imports only numpy, the standard library, or a declared sibling', (p) => {
    const siblings = new Set(REPO_SCRIPTS[p].deps.map((d) => d.split('/').pop()!.replace(/\.py$/, '')))
    const unmet = importsOf(REPO_FILES[p]).filter((m) => m !== 'numpy' && m !== '__future__' && !stdlib.has(m) && !siblings.has(m))
    expect(unmet).toEqual([])
  })
})

describe.skipIf(!python)('the assembled program runs in real Python', () => {
  it.each(paths)('%s', (p) => {
    const r = spawnSync(python!, ['-c', buildRepoProgram(p)], { encoding: 'utf8', timeout: 60000, cwd: '/' })
    expect(r.stderr).toBe('')
    expect(r.status).toBe(0)
    expect(r.stdout.length).toBeGreaterThan(100)
  }, 60000)

  it('runs an edited copy of the main file, and the sibling import still works', () => {
    const edited = REPO_FILES['phase6-engineering/eval_harness.py'].replace(/if __name__ == "__main__":[\s\S]*$/, 'print("EDITED", rag.__name__)\n')
    const r = spawnSync(python!, ['-c', buildRepoProgram('phase6-engineering/eval_harness.py', edited)], { encoding: 'utf8', timeout: 60000 })
    expect(r.stderr).toBe('')
    expect(r.stdout.trim()).toBe('EDITED mini_rag')
  }, 60000)
})
