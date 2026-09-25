// Messages between the GPT-2 Explainer page and its Web Worker (gpt2.worker.ts), plus the page-side client.
// The worker owns the 127 MB of weights and the running state (KV cache + every intermediate);
// the page asks for a summary after each run and for one layer's full detail when a block is opened.
import type { LensRow } from './gpt2'
import type { LoadProgress } from './gpt2Load'

/** The explainer keeps every intermediate, so it holds at most this many tokens (GPT-2 itself reads 1024). */
export const EXPLAINER_MAX_TOKENS = 96

export interface ModelInfo {
  nParams: number
  totalBytes: number
  chunkBytes: number[]
  nLayer: number
  nHead: number
  nEmbd: number
  vocabSize: number
  nCtx: number
  outlierCols: number[]
  quantization: string
  source: string
}

/** Everything the overview needs after a run, trimmed to T positions. */
export interface TraceSummary {
  ids: number[]
  T: number
  /** How many positions this call computed (T for a fresh run, 1 when a token was appended via the KV cache). */
  computed: number
  ms: number
  tokEmb: Float32Array // (T, D)
  posEmb: Float32Array // (T, D)
  /** attn[l] = (H, T, T) attention weights of layer l. */
  attn: Float32Array[]
  /** Per layer, per position: length of the residual stream after the block, and of what attention / MLP added. */
  residNorm: Float32Array[] // (T) each
  attnNorm: Float32Array[]
  mlpNorm: Float32Array[]
  lnFLast: Float32Array // (D): final LayerNorm output at the last position
  logits: Float32Array // (V): at the last position
  lens: LensRow[] | null
}

/** One layer in full, trimmed to T positions. `xIn` is the residual stream entering the block. */
export interface LayerDetail {
  l: number
  T: number
  xIn: Float32Array
  ln1: Float32Array
  q: Float32Array
  k: Float32Array
  v: Float32Array
  scores: Float32Array // (H, T, T); entries above the diagonal are masked (not computed) and set to NaN
  attn: Float32Array // (H, T, T)
  heads: Float32Array
  attnOut: Float32Array
  resid1: Float32Array
  ln2: Float32Array
  act: Float32Array // (T, 4D)
  mlpOut: Float32Array
  resid2: Float32Array
}

export type Gpt2Request =
  | { type: 'load'; req: number; baseUrl: string }
  | { type: 'run'; req: number; ids: number[]; lens: boolean }
  | { type: 'extend'; req: number; ids: number[]; lens: boolean }
  | { type: 'layer'; req: number; l: number }

export type Gpt2Reply =
  | { type: 'progress'; req: number; progress: LoadProgress }
  | { type: 'ready'; req: number; info: ModelInfo; merges: ArrayBuffer; fromCache: boolean; ms: number }
  | { type: 'trace'; req: number; trace: TraceSummary }
  | { type: 'layer'; req: number; detail: LayerDetail }
  | { type: 'error'; req: number; message: string }

/* ---------------- page-side client: one worker per page, shared by every explainer ---------------- */

export interface Gpt2Client {
  load: (baseUrl: string, onProgress: (p: LoadProgress) => void) => Promise<{ info: ModelInfo; merges: ArrayBuffer; fromCache: boolean; ms: number }>
  run: (ids: number[], lens: boolean) => Promise<TraceSummary>
  extend: (ids: number[], lens: boolean) => Promise<TraceSummary>
  layer: (l: number) => Promise<LayerDetail>
  /** Set once the model has loaded in this page (so a second explainer on the page starts ready). */
  ready: { info: ModelInfo; merges: ArrayBuffer; fromCache: boolean; ms: number } | null
  dispose: () => void
}

type DistOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never
type Pending = { resolve: (r: Gpt2Reply) => void; reject: (e: Error) => void; onProgress?: (p: LoadProgress) => void }

export const makeGpt2Client = (makeWorker: () => Worker): Gpt2Client => {
  let worker: Worker | null = null
  let next = 1
  const pending = new Map<number, Pending>()
  const failAll = (msg: string) => {
    for (const p of pending.values()) p.reject(new Error(msg))
    pending.clear()
    worker?.terminate()
    worker = null
  }
  const ensure = () => {
    if (worker) return worker
    const w = makeWorker()
    w.onmessage = (e: MessageEvent<Gpt2Reply>) => {
      const r = e.data
      const p = pending.get(r.req)
      if (!p) return
      if (r.type === 'progress') { p.onProgress?.(r.progress); return }
      pending.delete(r.req)
      if (r.type === 'error') p.reject(new Error(r.message))
      else p.resolve(r)
    }
    w.onerror = (e) => failAll(e.message || 'The GPT-2 worker stopped unexpectedly (possibly out of memory).')
    worker = w
    return w
  }
  const call = (msg: DistOmit<Gpt2Request, 'req'>, onProgress?: (p: LoadProgress) => void) =>
    new Promise<Gpt2Reply>((resolve, reject) => {
      const req = next++
      pending.set(req, { resolve, reject, onProgress })
      ensure().postMessage({ ...msg, req } as Gpt2Request)
    })
  const client: Gpt2Client = {
    ready: null,
    load: async (baseUrl, onProgress) => {
      try {
        const r = await call({ type: 'load', baseUrl }, onProgress)
        if (r.type !== 'ready') throw new Error('unexpected reply')
        client.ready = { info: r.info, merges: r.merges, fromCache: r.fromCache, ms: r.ms }
        return client.ready
      } catch (e) {
        failAll('load failed') // start from a fresh worker on retry
        throw e
      }
    },
    run: async (ids, lens) => { const r = await call({ type: 'run', ids, lens }); if (r.type !== 'trace') throw new Error('unexpected reply'); return r.trace },
    extend: async (ids, lens) => { const r = await call({ type: 'extend', ids, lens }); if (r.type !== 'trace') throw new Error('unexpected reply'); return r.trace },
    layer: async (l) => { const r = await call({ type: 'layer', l }); if (r.type !== 'layer') throw new Error('unexpected reply'); return r.detail },
    dispose: () => failAll('disposed'),
  }
  return client
}
