// Web Worker for the GPT-2 Explainer. The logic lives in gpt2WorkerCore.ts (so tests can run it without a Worker).
import { createGpt2Handler } from './gpt2WorkerCore'
import type { Gpt2Reply, Gpt2Request } from './gpt2Protocol'

const ctx = self as unknown as { postMessage: (m: Gpt2Reply, transfer?: Transferable[]) => void; onmessage: ((e: MessageEvent<Gpt2Request>) => void) | null }
const handle = createGpt2Handler((m, transfer) => ctx.postMessage(m, transfer))
ctx.onmessage = (e) => { void handle(e.data) }
