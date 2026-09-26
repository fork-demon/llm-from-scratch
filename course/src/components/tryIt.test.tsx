// Every lesson code block that offers "Try it" must run in real Python (NumPy + standard library, as in
// the browser) and print something. Needs a python3 with numpy; skips if none is found.
import { describe, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { render } from '@testing-library/react'
import type { ComponentType } from 'react'
import { codeBlockSink, type CodeBlockInfo } from './Code'
import { assembleTryIt } from '../lib/playground'

const candidates = [process.env.PYTHON, '/Users/arvind/miniconda3/bin/python3', 'python3', 'python'].filter(Boolean) as string[]
const python = candidates.find((p) => {
  try { return spawnSync(p, ['-c', 'import numpy'], { stdio: 'ignore' }).status === 0 } catch { return false }
})
const BANNED = /^\s*(import|from)\s+(torch|transformers|datasets|peft|trl|vllm|openai|anthropic|requests|scipy|pandas|sklearn)\b/m

const modules = import.meta.glob<{ default: ComponentType }>(['../lessons/*.tsx', '!../lessons/*.test.tsx'], { eager: true })

describe('Try it: assembly', () => {
  it('puts setup, the excerpt and the result in that order', () => {
    const p = assembleTryIt({ setup: 'x = 2', code: 'y = x * 3', show: 'print(y)' })
    expect(p.indexOf('x = 2')).toBeLessThan(p.indexOf('y = x * 3'))
    expect(p.indexOf('y = x * 3')).toBeLessThan(p.indexOf('print(y)'))
  })
})

describe.skipIf(!python)('Try it: every offered lesson snippet runs', () => {
  for (const [path, mod] of Object.entries(modules)) {
    const lesson = path.split('/').pop()!.replace('.tsx', '')
    const blocks: CodeBlockInfo[] = []
    codeBlockSink.current = (b) => blocks.push(b)
    const { unmount } = render(<mod.default />)
    unmount()
    codeBlockSink.current = null
    if (!blocks.length) continue
    const seen = new Set<string>()
    for (const b of blocks) {
      const program = assembleTryIt(b)
      if (seen.has(program)) continue // StrictMode-free render, but a block can repeat
      seen.add(program)
      it(`${lesson}: ${b.title ?? b.code.split('\n')[0].slice(0, 50)}`, () => {
        expect(BANNED.test(program), 'uses a package the browser does not have').toBe(false)
        const r = spawnSync(python!, ['-c', program], { encoding: 'utf8', timeout: 20000, cwd: '/tmp' })
        expect(r.status, `${r.stderr}\n--- program ---\n${program}`).toBe(0)
        expect(r.stdout.trim().length, `prints nothing\n--- program ---\n${program}`).toBeGreaterThan(0)
      })
    }
  }
})
