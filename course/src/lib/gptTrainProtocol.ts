// Messages between GptTrainer and its Web Worker (gptTrain.worker.ts), plus an in-page fallback
// that behaves identically when Workers are unavailable (tests, very old browsers).
import { createSession, sampleText, trainFor, type LossPoint, type Sample, type Session, type TrainOptions } from './gptTrain'

/** `gen` tags every message with the run it belongs to, so late replies from a reset run are ignored. */
export type WorkerRequest =
  | { type: 'init'; gen: number; text: string; opts: TrainOptions }
  | { type: 'train'; gen: number; budgetMs: number; temperature: number }
  | { type: 'generate'; gen: number; id: number; prompt: string; len: number; temperature: number }

export interface Progress { type: 'progress'; gen: number; step: number; seconds: number; lastLoss: number; history: LossPoint[]; samples: Sample[]; params: Float32Array }
export type WorkerReply =
  | Progress
  | { type: 'generated'; gen: number; id: number; text: string }
  | { type: 'error'; gen: number; message: string }

export interface Engine { send: (m: WorkerRequest) => void; dispose: () => void; inWorker: boolean }

/** Prefer a real Worker; fall back to running the same code on the page (in short chunks). */
export const makeEngine = (onReply: (r: WorkerReply) => void): Engine => {
  if (typeof Worker !== 'undefined') {
    try {
      const w = new Worker(new URL('./gptTrain.worker.ts', import.meta.url), { type: 'module' })
      w.onmessage = (e: MessageEvent<WorkerReply>) => onReply(e.data)
      w.onerror = (e) => onReply({ type: 'error', gen: -1, message: e.message || 'The training worker failed to start.' })
      return { send: (m) => w.postMessage(m), dispose: () => w.terminate(), inWorker: true }
    } catch {
      /* fall through */
    }
  }
  let session: Session | null = null
  let alive = true
  const handle = (m: WorkerRequest) => {
    if (!alive) return
    try {
      if (m.type === 'init') session = createSession(m.text, m.opts)
      else if (!session) return
      else if (m.type === 'train') { session.opts.temperature = m.temperature; trainFor(session, Math.min(m.budgetMs, 30), 1) }
      else { onReply({ type: 'generated', gen: m.gen, id: m.id, text: sampleText(session, m.len, m.temperature, m.prompt) }); return }
      const s = session!
      onReply({ type: 'progress', gen: m.gen, step: s.step, seconds: s.seconds, lastLoss: s.lastLoss, history: s.history.slice(), samples: s.samples.slice(), params: new Float32Array(s.model.params) })
    } catch (err) {
      onReply({ type: 'error', gen: m.gen, message: err instanceof Error ? err.message : String(err) })
    }
  }
  return { send: (m) => setTimeout(() => handle(m), 0), dispose: () => { alive = false }, inWorker: false }
}
