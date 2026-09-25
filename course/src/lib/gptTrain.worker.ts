// Web Worker for GptTrainer: runs the training loop of gptTrain.ts off the main thread so the page never janks.
// The page drives it: every 'train' message runs one time-boxed chunk and answers with one 'progress' message
// (so Pause is just "stop asking"). All the maths is in gptTrain.ts; this file only moves messages.
import { createSession, sampleText, trainFor, type Session } from './gptTrain'
import type { WorkerRequest, WorkerReply } from './gptTrainProtocol'

const ctx = self as unknown as { postMessage: (m: WorkerReply, transfer?: Transferable[]) => void; onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null }
let session: Session | null = null

ctx.onmessage = (e) => {
  const msg = e.data
  try {
    if (msg.type === 'init') {
      session = createSession(msg.text, msg.opts)
      reply(msg.gen)
    } else if (msg.type === 'train') {
      if (!session) return
      session.opts.temperature = msg.temperature
      trainFor(session, msg.budgetMs)
      reply(msg.gen)
    } else if (msg.type === 'generate') {
      if (!session) return
      ctx.postMessage({ type: 'generated', gen: msg.gen, id: msg.id, text: sampleText(session, msg.len, msg.temperature, msg.prompt) })
    }
  } catch (err) {
    ctx.postMessage({ type: 'error', gen: msg.gen, message: err instanceof Error ? err.message : String(err) })
  }
}

const reply = (gen: number) => {
  const s = session!
  const params = new Float32Array(s.model.params)
  ctx.postMessage({ type: 'progress', gen, step: s.step, seconds: s.seconds, lastLoss: s.lastLoss, history: s.history, samples: s.samples, params }, [params.buffer])
}
