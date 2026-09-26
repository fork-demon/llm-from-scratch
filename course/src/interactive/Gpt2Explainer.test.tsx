// The explainer end to end, with the real weights: a fake Worker runs the real worker logic (gpt2WorkerCore.ts)
// in-process, and fetch serves the files from public/models/gpt2.
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { createGpt2Handler } from '../lib/gpt2WorkerCore'
import type { Gpt2Request } from '../lib/gpt2Protocol'

const dir = resolve(__dirname, '../../public/models/gpt2')
const file = (name: string) => { const b = readFileSync(resolve(dir, name)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer }

let failFetch = true
const fakeFetch = vi.fn(async (url: string) => {
  const name = String(url).split('?')[0].split('/').pop()!
  if (failFetch && name.endsWith('.bin')) return { ok: false, status: 503, body: null, json: async () => ({}), arrayBuffer: async () => new ArrayBuffer(0) }
  return { ok: true, status: 200, body: null, json: async () => JSON.parse(readFileSync(resolve(dir, name), 'utf8')), arrayBuffer: async () => file(name) }
})

class FakeWorker {
  onmessage: ((e: { data: unknown }) => void) | null = null
  onerror: ((e: unknown) => void) | null = null
  private handle = createGpt2Handler((m) => setTimeout(() => this.onmessage?.({ data: m }), 0))
  postMessage(m: Gpt2Request) { setTimeout(() => void this.handle(m), 0) }
  terminate() {}
}

afterEach(() => cleanup())

describe('Gpt2Explainer', () => {
  it('fails gracefully, retries, then runs GPT-2 and lets you open a block and generate', async () => {
    vi.stubGlobal('fetch', fakeFetch)
    vi.stubGlobal('Worker', FakeWorker)
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
    const { Gpt2Explainer } = await import('./Gpt2Explainer')
    const { container } = render(<Gpt2Explainer />)

    fireEvent.click(await screen.findByRole('button', { name: /Load GPT-2/ }))
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('HTTP 503'), { timeout: 5000 })

    failFetch = false
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    await waitFor(() => expect(container.textContent).toContain('124,439,808 parameters'), { timeout: 30000 })
    // the default prompt runs once: "The cat sat on the" -> GPT-2's top guess is " floor"
    await waitFor(() => expect(container.textContent).toContain('Forward pass over 5 tokens'), { timeout: 30000 })
    expect(screen.getByRole('button', { name: /position 2: token " cat", id 3797/ })).toBeTruthy()
    expect(container.querySelector('.Gpt2Explainer-guess')?.textContent).toContain('␣floor')
    const tab = (name: RegExp) => screen.getByRole('tab', { name })

    // one stage shows at a time, and each names the lesson that explains it
    expect(tab(/^Tokens/).getAttribute('aria-selected')).toBe('true')
    expect((container.querySelector('a[href="#/lesson/tokenization"]') as HTMLAnchorElement).textContent).toBe('Tokenization')
    expect(container.textContent).toContain('18 characters became 5 tokens')

    // the blocks: block 1 opens by default; pick block 3, then its Q, K, V and its MLP
    fireEvent.click(tab(/^12 blocks/))
    fireEvent.click(screen.getByRole('button', { name: /Block 3/ }))
    await waitFor(() => expect(container.textContent).toContain('Inside block 3'), { timeout: 30000 })
    fireEvent.click(tab(/^Q, K, V$/))
    await waitFor(() => expect(container.textContent).toContain('Queries, keys, values'), { timeout: 30000 })
    fireEvent.click(screen.getAllByRole('button', { name: 'head 5' })[0])
    expect(container.textContent).toContain('Head 5 uses numbers 256 to 319')
    fireEvent.click(tab(/^MLP$/))
    expect(container.textContent).toContain('of 3,072 are positive')

    // the scores, and the logit lens
    fireEvent.click(screen.getByRole('button', { name: /Next: Scores/ }))
    expect(container.querySelector('[aria-label^="The 10 highest logits"]')?.textContent).toContain('␣floor')
    fireEvent.click(tab(/Guess after each block/))
    expect(container.textContent).toContain('final answer')

    // the next token: temperature 0 -> the top token gets 100%
    fireEvent.click(tab(/^Next token/))
    expect(container.querySelector('[aria-label^="Top 10 next tokens"]')?.textContent).toContain('␣floor')
    const temp = screen.getByRole('slider', { name: /^temperature/ }) as HTMLInputElement
    fireEvent.change(temp, { target: { value: '0' } })
    expect(container.querySelector('[aria-label^="Top 10 next tokens"] .Gpt2Explainer-probval')?.textContent).toBe('100%')

    fireEvent.click(screen.getByRole('button', { name: 'Generate next token' }))
    await waitFor(() => expect(container.querySelector('.Gpt2Explainer-gentext')?.textContent).toBe(' floor'), { timeout: 30000 })
    expect(container.textContent).toContain('With the KV cache only the new position was computed')
    // the sentence at the top grows by the generated token
    expect(screen.getByRole('button', { name: /position 6: token " floor", id \d+, generated/ })).toBeTruthy()
  }, 90000)
})
