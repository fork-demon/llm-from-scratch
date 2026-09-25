// The GPT-2 Explainer's worker logic: download (or read from the Cache API) the int8 weights, keep them,
// run GPT-2 and answer the page. All the maths is in gpt2.ts; this file only moves messages.
// gpt2.worker.ts wires it to the real Worker; tests drive it directly with a fake Worker.
import { buildGpt2, extend, logitLens, newState, norm2, type Gpt2Model, type Gpt2State } from './gpt2'
import { downloadGpt2 } from './gpt2Load'
import { EXPLAINER_MAX_TOKENS, type Gpt2Reply, type Gpt2Request, type LayerDetail, type TraceSummary } from './gpt2Protocol'


const summary = (s: Gpt2State, computed: number, ms: number, logits: Float32Array, lens: boolean): TraceSummary => {
  const { nEmbd: D, nHead: H } = s.m.cfg
  const T = s.ids.length
  const cap = s.cap
  const attn = s.layers.map((L) => {
    const a = new Float32Array(H * T * T)
    for (let h = 0; h < H; h++) for (let t = 0; t < T; t++) a.set(L.attn.subarray((h * cap + t) * cap, (h * cap + t) * cap + T), (h * T + t) * T)
    return a
  })
  const perPos = (f: (t: number) => number) => Float32Array.from({ length: T }, (_, t) => f(t))
  return {
    ids: s.ids.slice(),
    T,
    computed,
    ms,
    tokEmb: s.tokEmb.slice(0, T * D),
    posEmb: s.posEmb.slice(0, T * D),
    attn,
    residNorm: s.layers.map((L) => perPos((t) => norm2(L.resid2, t * D, D))),
    attnNorm: s.layers.map((L) => perPos((t) => norm2(L.attnOut, t * D, D))),
    mlpNorm: s.layers.map((L) => perPos((t) => norm2(L.mlpOut, t * D, D))),
    lnFLast: s.lnF.slice((T - 1) * D, T * D),
    logits,
    lens: lens ? logitLens(s) : null,
  }
}

const detail = (s: Gpt2State, l: number): LayerDetail => {
  const { nEmbd: D, nHead: H } = s.m.cfg
  const T = s.ids.length
  const cap = s.cap
  const L = s.layers[l]
  const sq = (a: Float32Array, masked: number) => {
    const o = new Float32Array(H * T * T)
    for (let h = 0; h < H; h++) for (let t = 0; t < T; t++) for (let u = 0; u < T; u++) o[(h * T + t) * T + u] = u <= t ? a[(h * cap + t) * cap + u] : masked
    return o
  }
  const cut = (a: Float32Array, w = D) => a.slice(0, T * w)
  return {
    l, T,
    xIn: cut(l === 0 ? s.x0 : s.layers[l - 1].resid2),
    ln1: cut(L.ln1), q: cut(L.q), k: cut(L.k), v: cut(L.v),
    scores: sq(L.scores, NaN), attn: sq(L.attn, 0),
    heads: cut(L.heads), attnOut: cut(L.attnOut), resid1: cut(L.resid1), ln2: cut(L.ln2),
    act: cut(L.act, 4 * D), mlpOut: cut(L.mlpOut), resid2: cut(L.resid2),
  }
}

const buffers = (t: TraceSummary) => [t.tokEmb, t.posEmb, t.lnFLast, t.logits, ...t.attn, ...t.residNorm, ...t.attnNorm, ...t.mlpNorm].map((a) => a.buffer)

export type Post = (m: Gpt2Reply, transfer?: Transferable[]) => void

/** One handler = one model + one running state. */
export const createGpt2Handler = (post: Post) => {
  let model: Gpt2Model | null = null
  let state: Gpt2State | null = null
  const ctx = { postMessage: post }
  return async (msg: Gpt2Request): Promise<void> => {
    try {
      if (msg.type === 'load') {
        const t0 = performance.now()
        const d = await downloadGpt2(msg.baseUrl, (progress) => ctx.postMessage({ type: 'progress', req: msg.req, progress }))
        model = buildGpt2(d.man, d.chunks)
        const merges = d.merges.slice(0)
        ctx.postMessage({
          type: 'ready', req: msg.req, fromCache: d.fromCache, ms: performance.now() - t0, merges,
          info: {
            nParams: d.man.n_params, totalBytes: d.man.total_bytes, chunkBytes: d.man.chunks.map((c) => c.bytes),
            nLayer: model.cfg.nLayer, nHead: model.cfg.nHead, nEmbd: model.cfg.nEmbd, vocabSize: model.cfg.vocabSize, nCtx: model.cfg.nCtx,
            outlierCols: Array.from(model.wte.outCols ?? []), quantization: d.man.quantization, source: d.man.source,
          },
        }, [merges])
        return
      }
      if (!model) throw new Error('GPT-2 is not loaded yet')
      if (msg.type === 'run' || msg.type === 'extend') {
        if (msg.type === 'run' || !state) state = newState(model, true, EXPLAINER_MAX_TOKENS)
        if (state.ids.length + msg.ids.length > EXPLAINER_MAX_TOKENS) throw new Error(`the explainer holds at most ${EXPLAINER_MAX_TOKENS} tokens`)
        const t0 = performance.now()
        const logits = extend(state, msg.ids)
        const t = summary(state, msg.ids.length, performance.now() - t0, logits, msg.lens) // ms = the forward pass alone
        ctx.postMessage({ type: 'trace', req: msg.req, trace: t }, buffers(t))
      } else if (msg.type === 'layer') {
        if (!state || !state.ids.length) throw new Error('run a prompt first')
        const d = detail(state, msg.l)
        ctx.postMessage({ type: 'layer', req: msg.req, detail: d }, Object.values(d).filter((v): v is Float32Array => v instanceof Float32Array).map((a) => a.buffer))
      }
    } catch (err) {
      ctx.postMessage({ type: 'error', req: msg.req, message: err instanceof Error ? err.message : String(err) })
    }
  }
}
