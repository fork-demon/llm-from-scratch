// Smoke test: the explorer loads the real checkpoint through fetch, generates, and survives a failed download.
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { TrainedGptExplorer } from './TrainedGptExplorer'

const dir = resolve(__dirname, '../../public/models')
const manifestText = readFileSync(resolve(dir, 'tiny-gpt.json'), 'utf8')
const bin = readFileSync(resolve(dir, 'tiny-gpt.bin'))

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('TrainedGptExplorer', () => {
  it('shows a clear message and a retry button when the checkpoint cannot be fetched', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 404, json: async () => ({}), arrayBuffer: async () => new ArrayBuffer(0) })))
    render(<TrainedGptExplorer />)
    expect(screen.getByText(/Loading the trained weights/)).toBeTruthy()
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('HTTP 404'))
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
  })

  it('loads, describes the model, generates one character and shows its top 5', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
    vi.stubGlobal('fetch', vi.fn(async (url: string) => ({
      ok: true, status: 200,
      json: async () => JSON.parse(manifestText),
      arrayBuffer: async () => (url.endsWith('.bin') ? bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) : new ArrayBuffer(0)),
    })))
    const { container } = render(<TrainedGptExplorer />)
    await waitFor(() => expect(container.textContent).toContain('809,856 parameters'), { timeout: 5000 })
    fireEvent.click(screen.getByRole('button', { name: 'One character' }))
    await waitFor(() => expect(container.textContent).toContain('from these top 5'), { timeout: 5000 })
    expect(screen.getAllByRole('button', { name: /^Layer \d, head \d/ }).length).toBe(16)
    fireEvent.click(screen.getByRole('button', { name: /^Layer 2, head 3/ }))
    expect(container.textContent).toContain('Layer 2, head 3')
    await waitFor(() => expect(container.textContent).toContain('What this model shows'), { timeout: 10000 })
  }, 20000)
})
