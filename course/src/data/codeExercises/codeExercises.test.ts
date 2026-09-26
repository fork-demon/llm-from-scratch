// Every coding exercise is checked with REAL Python: the reference solution must pass all of its
// tests, and the starter must not (otherwise the exercise is already solved, or the tests are empty).
// Needs a python3 with numpy on PATH, or set PYTHON=/path/to/python. Skips if none is found.
import { describe, expect, it } from 'vitest'
import { execFileSync, spawnSync } from 'node:child_process'
import { CODE_EXERCISES, withPrelude } from './index'
import { LESSONS } from '../curriculum'

const candidates = [process.env.PYTHON, '/Users/arvind/miniconda3/bin/python3', 'python3', 'python'].filter(Boolean) as string[]
const python = candidates.find((p) => {
  try { return spawnSync(p, ['-c', 'import numpy'], { stdio: 'ignore' }).status === 0 } catch { return false }
})

const HARNESS = `
import json, sys
spec = json.load(sys.stdin)
ns = {}
out = {"error": None, "results": []}
try:
    exec(spec["code"], ns)
except BaseException as e:
    out["error"] = type(e).__name__ + ": " + str(e)
if out["error"] is None:
    for t in spec["tests"]:
        try:
            exec(t["code"], ns)
            out["results"].append([t["name"], True, ""])
        except BaseException as e:
            out["results"].append([t["name"], False, type(e).__name__ + ": " + str(e)])
sys.stdout = sys.__stdout__
print("\\n@@RESULT@@" + json.dumps(out))
`
const run = (code: string, tests: { name: string; code: string }[]) => {
  const stdout = execFileSync(python!, ['-c', HARNESS], { input: JSON.stringify({ code, tests }), encoding: 'utf8', timeout: 60000 })
  return JSON.parse(stdout.slice(stdout.lastIndexOf('@@RESULT@@') + 10)) as { error: string | null; results: [string, boolean, string][] }
}

describe('coding exercises: structure', () => {
  it('ids are unique and point at real lessons', () => {
    const ids = CODE_EXERCISES.map((e) => e.id)
    expect(new Set(ids).size).toBe(ids.length)
    const lessons = new Set(LESSONS.map((l) => l.id))
    expect(CODE_EXERCISES.filter((e) => !lessons.has(e.lesson)).map((e) => e.id)).toEqual([])
  })
  it('each has tests, 2 to 3 hints, a solution that differs from the starter, and an explanation', () => {
    for (const e of CODE_EXERCISES) {
      expect(e.tests.length, e.id).toBeGreaterThanOrEqual(3)
      expect(e.hints.length, e.id).toBeGreaterThanOrEqual(2)
      expect(e.hints.length, e.id).toBeLessThanOrEqual(3)
      expect(e.solution.trim(), e.id).not.toBe(e.starter.trim())
      expect(e.explanation.length, e.id).toBeGreaterThan(40)
    }
  })
})

describe.skipIf(!python)('coding exercises: checked with real Python', () => {
  for (const e of CODE_EXERCISES) {
    it(`${e.id}: the solution passes every test`, () => {
      const r = run(withPrelude(e.prelude, e.solution), e.tests)
      expect(r.error).toBeNull()
      expect(r.results.filter(([, ok]) => !ok)).toEqual([])
    })
    it(`${e.id}: the starter does not already pass`, () => {
      const r = run(withPrelude(e.prelude, e.starter), e.tests)
      const allPass = r.error === null && r.results.every(([, ok]) => ok)
      expect(allPass).toBe(false)
    })
  }
})
