// Smoke test: the trainer renders, trains a few steps through the in-page fallback engine (jsdom has no
// Worker), shows its parameter count, and the attention controls respond.
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { GptTrainer } from './GptTrainer'
import { countParams, makeCharData } from '../lib/gptTrain'
import { CORPORA } from '../lib/gptCorpora'

afterEach(cleanup)

describe('GptTrainer', () => {
  it('renders, trains, and shows attention for the prompt', async () => {
    const { container } = render(<GptTrainer corpus="tickets" size="tiny" />)
    const V = makeCharData(CORPORA.find((c) => c.id === 'tickets')!.text).chars.length
    const n = countParams({ nLayer: 1, nHead: 2, nEmbd: 16, contextLen: 16, vocabSize: V })
    expect(container.textContent).toContain(`${n.toLocaleString()} parameters`)
    const train = await screen.findByRole('button', { name: 'Train' })
    await waitFor(() => expect((train as HTMLButtonElement).disabled).toBe(false), { timeout: 10000 })
    fireEvent.click(train)
    await waitFor(() => expect(Number(container.querySelector('.readout b')!.textContent!.replace(/,/g, ''))).toBeGreaterThan(2), { timeout: 10000 })
    fireEvent.click(screen.getByRole('button', { name: 'Pause' }))
    await screen.findByRole('button', { name: 'Continue' }, { timeout: 5000 })
    // head 2 of layer 1, then click the first prompt character
    fireEvent.click(screen.getByRole('button', { name: '2' }))
    const tok = container.querySelectorAll('.GptTrainer-tok')[0] as HTMLButtonElement
    fireEvent.click(tok)
    expect(tok.getAttribute('aria-pressed')).toBe('true')
    expect(container.textContent).toContain('looks mostly at')
    expect(container.querySelector('svg.GptTrainer-heatmap')).not.toBeNull()
  }, 30000)
})
