// GPT-2 Explainer: the REAL GPT-2 small (124M parameters, int8) running in the learner's browser.
// It shows WHAT happens at every stage of one forward pass, and each stage links to the lesson that explains WHY.
// Weights: public/models/gpt2/ (phase6-engineering/export_gpt2.py). Engine: lib/gpt2.ts in a Web Worker.
import { useCallback, useEffect, useId, useMemo, useRef, useState, type ReactNode } from 'react'
import { Lab, Slider } from '../components/ui'
import { lessonById } from '../data/curriculum'
import { distribution, geluNew, orderByLogit, sample, type Distribution } from '../lib/gpt2'
import { cachedFiles, fetchManifest, type LoadProgress } from '../lib/gpt2Load'
import { EXPLAINER_MAX_TOKENS, makeGpt2Client, type Gpt2Client, type LayerDetail, type ModelInfo, type TraceSummary } from '../lib/gpt2Protocol'
import { createGpt2Tokenizer, showToken, type Gpt2Tokenizer } from '../lib/gpt2Tokenizer'
import { makeRng } from '../lib/rng'
import './Gpt2Explainer.css'

/* ---------------- one worker per page, shared by every explainer on it ---------------- */
let client: Gpt2Client | null = null
const getClient = () =>
  (client ??= makeGpt2Client(() => new Worker(new URL('../lib/gpt2.worker.ts', import.meta.url), { type: 'module' })))
const baseUrl = () => new URL(`${import.meta.env.BASE_URL}models/gpt2/`, window.location.href).href
let tokenizer: Gpt2Tokenizer | null = null

const MAX_PROMPT = 64
const PRESETS = ['The cat sat on the', 'What is a cat? A cat is a', 'The capital of France is', 'Riya opened her laptop and', 'One, two, three, four,']
const MB = (b: number) => `${(b / 1e6).toFixed(1)} MB`
const pct = (p: number) => (p >= 0.9995 ? '100%' : p < 0.0005 ? '<0.1%' : `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`)
const fmt = (x: number, d = 2) => (Math.abs(x) >= 1000 ? x.toFixed(0) : x.toFixed(d))

/* ---------------- "Why?" links ---------------- */
type WhyText = { text: ReactNode; lessons: string[] }
const WHY: Record<string, WhyText> = {
  tokens: { text: <>GPT-2 never sees letters. Its tokenizer cuts text into pieces from a fixed list of 50,257 and passes on their ids. A common word is one token; a rare word, a Hindi word or an emoji can be several.</>, lessons: ['tokenization'] },
  embed: { text: <>An id is only a row number. Row <i>id</i> of a 50,257 × 768 table is the token’s starting vector. Row <i>position</i> of a second 1,024 × 768 table is added, because attention on its own cannot tell where a token sits.</>, lessons: ['embeddings', 'transformer-block'] },
  blocks: { text: <>The same block, 12 times, each with its own weights. Each block reads the running vector of every token (the residual stream) and adds to it: first attention (tokens exchange information), then the MLP (each token works alone).</>, lessons: ['transformer-block', 'build-gpt'] },
  ln: { text: <>LayerNorm rescales each token’s vector to mean 0 and spread 1, then applies learned gains. The numbers entering attention stay in a steady range, however large the stream grows.</>, lessons: ['transformer-block'] },
  qkv: { text: <>Each token asks (query), advertises (key) and offers content (value). The score q·k is divided by √64 = 8 so the softmax does not get too sharp as vectors get longer.</>, lessons: ['attention'] },
  mask: { text: <>A token may only look at itself and earlier tokens: GPT-2 learned to predict the next token, so looking ahead would be cheating. Each of the 12 heads does its own attention with 64 numbers, so each can look for something different.</>, lessons: ['masks-and-heads', 'attention'] },
  proj: { text: <>The 12 head outputs (12 × 64) are put side by side into 768 numbers, then mixed by one more matrix, so the heads’ findings can be combined.</>, lessons: ['masks-and-heads'] },
  resid: { text: <>The block adds its result to the stream instead of replacing it. Earlier information survives, and during training the gradient has a direct path back.</>, lessons: ['transformer-block'] },
  mlp: { text: <>After the meeting, everyone returns to their desk: each token alone goes 768 → 3,072 → 768 with GELU in between. About two thirds of a block’s weights live here.</>, lessons: ['transformer-block'] },
  logits: { text: <>One last LayerNorm, then a dot product with every row of the token table (the same table as the input: tied weights) gives 50,257 scores, one per possible next token.</>, lessons: ['build-gpt', 'softmax'] },
  probs: { text: <>Softmax turns scores into probabilities. Temperature divides the scores first (below 1 sharpens, above 1 flattens). Top-k and top-p cut the long tail before the dice are rolled.</>, lessons: ['softmax', 'inference'] },
  sample: { text: <>The chosen token is appended and the pipeline runs again for one new position only. Keys and values of earlier tokens never change, so they are kept (the KV cache) instead of recomputed.</>, lessons: ['inference'] },
  lens: { text: <>Apply the final LayerNorm and the output head to the stream after each block, as if the model stopped there. A rough probe: early layers were never trained to be read this way.</>, lessons: ['interpretability'] },
  int8: { text: <>Each weight is stored as one byte, plus one scale per row: 4 times smaller than float32, a little less exact. 8 columns of the token table stay float32 because GPT-2’s final vector is enormous in those dimensions.</>, lessons: ['making-models-cheaper'] },
}

function WhyButton({ k, open, setOpen, controls }: { k: string; open: boolean; setOpen: (o: boolean) => void; controls: string }) {
  return (
    <button className="Gpt2Explainer-why" aria-expanded={open} aria-controls={controls} onClick={() => setOpen(!open)} aria-label={`Why? (${k})`}>
      Why?
    </button>
  )
}
function WhyPanel({ k, id }: { k: string; id: string }) {
  const w = WHY[k]
  return (
    <div className="Gpt2Explainer-whytext" id={id}>
      {w.text}{' '}
      {w.lessons.map((l, i) => {
        const meta = lessonById(l)
        return <span key={l}>{i > 0 && ' · '}<a href={`#/lesson/${l}`}>Lesson: {meta?.title ?? l} →</a></span>
      })}
    </div>
  )
}

/** One stage of the pipeline: number, title, shape, a Why? toggle, then the content. */
function Stage({ n, title, shape, why, active, children, sub }: { n: string; title: string; shape?: string; why: string; active?: boolean; children: ReactNode; sub?: boolean }) {
  const [open, setOpen] = useState(false)
  const id = useId()
  return (
    <section className={`Gpt2Explainer-stage${active ? ' is-active' : ''}${sub ? ' is-sub' : ''}`} aria-labelledby={`${id}-h`}>
      <div className="Gpt2Explainer-stagehead">
        <span className="Gpt2Explainer-num" aria-hidden="true">{n}</span>
        <h4 id={`${id}-h`} className="Gpt2Explainer-title">{title}</h4>
        {shape && <span className="Gpt2Explainer-shape">{shape}</span>}
        <WhyButton k={title} open={open} setOpen={setOpen} controls={`${id}-why`} />
      </div>
      {open && <WhyPanel k={why} id={`${id}-why`} />}
      <div className="Gpt2Explainer-stagebody">{children}</div>
    </section>
  )
}

/* ---------------- drawing helpers (canvas + theme colours) ---------------- */

function useThemeKey() {
  const [k, setK] = useState(0)
  useEffect(() => {
    const mo = new MutationObserver(() => setK((x) => x + 1))
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)')
    const on = () => setK((x) => x + 1)
    mq?.addEventListener?.('change', on)
    return () => { mo.disconnect(); mq?.removeEventListener?.('change', on) }
  }, [])
  return k
}

type RGB = [number, number, number]
const rgbCache = new Map<string, RGB>()
/** Resolve any CSS colour (including a var) to RGB by painting one pixel. */
const rgbOf = (el: Element, cssVar: string): RGB => {
  const val = getComputedStyle(el).getPropertyValue(cssVar).trim() || '#888'
  const hit = rgbCache.get(val)
  if (hit) return hit
  const c = document.createElement('canvas')
  c.width = c.height = 1
  const x = c.getContext('2d')
  if (!x) return [128, 128, 128]
  x.fillStyle = val
  x.fillRect(0, 0, 1, 1)
  const d = x.getImageData(0, 0, 1, 1).data
  const rgb: RGB = [d[0], d[1], d[2]]
  rgbCache.set(val, rgb)
  return rgb
}

/** The value at the 99th percentile of |v|: colours are clipped there so a few huge dimensions do not wash out the rest. */
const clipOf = (v: ArrayLike<number>, off: number, len: number): number => {
  const a = new Float32Array(len)
  for (let i = 0; i < len; i++) a[i] = Math.abs(v[off + i])
  a.sort()
  return Math.max(a[Math.min(len - 1, Math.floor(len * 0.99))], 1e-6)
}

interface HeatProps {
  rows: number
  cols: number
  /** Value at (r, c); NaN = masked. */
  get: (r: number, c: number) => number
  clip: number
  pos?: string
  neg?: string
  label: string
  cell?: [number, number] // pixel size of one cell on the backing canvas
  outlineRow?: number
  className?: string
  style?: React.CSSProperties
  onPick?: (row: number, col: number) => void
  themeKey: number
  deps: unknown[]
}
/** A heatmap drawn with ImageData: positive values in `pos`, negative in `neg`, masked cells hatched grey. */
function Heat({ rows, cols, get, clip, pos = '--accent', neg = '--il-a', label, cell = [1, 1], outlineRow, className, style, onPick, themeKey, deps }: HeatProps) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    const ctx = c?.getContext?.('2d')
    if (!c || !ctx || !rows || !cols) return
    const [cw, ch] = cell
    c.width = cols * cw
    c.height = rows * ch
    const P = rgbOf(c, pos), N = rgbOf(c, neg), B = rgbOf(c, '--paper-2'), M = rgbOf(c, '--rule-strong')
    const img = ctx.createImageData(c.width, c.height)
    const d = img.data
    for (let r = 0; r < rows; r++) for (let col = 0; col < cols; col++) {
      const v = get(r, col)
      let rgb: RGB
      if (Number.isNaN(v)) rgb = M
      else {
        const a = Math.min(1, Math.abs(v) / clip)
        const T = v >= 0 ? P : N
        rgb = [B[0] + (T[0] - B[0]) * a, B[1] + (T[1] - B[1]) * a, B[2] + (T[2] - B[2]) * a]
      }
      for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
        const masked = Number.isNaN(v) && (x + y) % 4 === 0
        const o = ((r * ch + y) * c.width + col * cw + x) * 4
        d[o] = masked ? B[0] : rgb[0]; d[o + 1] = masked ? B[1] : rgb[1]; d[o + 2] = masked ? B[2] : rgb[2]; d[o + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)
    if (outlineRow !== undefined && outlineRow < rows) {
      const I = getComputedStyle(c).getPropertyValue('--ink').trim() || '#000'
      ctx.strokeStyle = I
      ctx.lineWidth = Math.max(1, Math.min(2, Math.min(cw, ch) / 8))
      ctx.strokeRect(ctx.lineWidth / 2, outlineRow * ch + ctx.lineWidth / 2, c.width - ctx.lineWidth, ch - ctx.lineWidth)
    }
  }, [rows, cols, clip, pos, neg, outlineRow, themeKey, cell[0], cell[1], ...deps]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <canvas
      ref={ref} role="img" aria-label={label} className={`Gpt2Explainer-canvas ${className ?? ''}`} style={style}
      onClick={onPick ? (e) => {
        const r = e.currentTarget.getBoundingClientRect()
        onPick(Math.min(rows - 1, Math.floor(((e.clientY - r.top) / r.height) * rows)), Math.min(cols - 1, Math.floor(((e.clientX - r.left) / r.width) * cols)))
      } : undefined}
    />
  )
}

/** One vector as a thin strip: every number is one column. */
function Strip({ v, off = 0, len, label, themeKey, pos, neg, marks }: { v: Float32Array; off?: number; len: number; label: string; themeKey: number; pos?: string; neg?: string; marks?: number }) {
  const clip = useMemo(() => clipOf(v, off, len), [v, off, len])
  let n2 = 0
  let big = 0
  for (let i = 0; i < len; i++) { n2 += v[off + i] ** 2; if (Math.abs(v[off + i]) > Math.abs(v[off + big])) big = i }
  return (
    <div className="Gpt2Explainer-strip">
      <div className="Gpt2Explainer-striplabel"><span>{label}</span><span className="mono">length {fmt(Math.sqrt(n2), 1)} · largest {fmt(v[off + big], 1)} (#{big})</span></div>
      <div className="Gpt2Explainer-stripwrap">
        <Heat rows={1} cols={len} get={(_, c) => v[off + c]} clip={clip} pos={pos} neg={neg} cell={[1, 1]} themeKey={themeKey} deps={[v, off]}
          label={`${label}: ${len} numbers, length ${fmt(Math.sqrt(n2), 1)}, largest ${fmt(v[off + big], 1)} at dimension ${big}`} className="Gpt2Explainer-stripcanvas" />
        {marks ? <div className="Gpt2Explainer-marks" aria-hidden="true">{Array.from({ length: marks - 1 }, (_, i) => <span key={i} style={{ left: `${((i + 1) / marks) * 100}%` }} />)}</div> : null}
      </div>
    </div>
  )
}

/* ---------------- the component ---------------- */

type Status = { kind: 'idle' } | { kind: 'loading'; p: LoadProgress | null } | { kind: 'error'; message: string } | { kind: 'ready' }

export function Gpt2Explainer({ autoload = false }: { autoload?: boolean }) {
  const [status, setStatus] = useState<Status>(() => (client?.ready ? { kind: 'ready' } : { kind: 'idle' }))
  const [cached, setCached] = useState<'unknown' | 'yes' | 'partly' | 'no'>('unknown')
  const [size, setSize] = useState<number | null>(null)
  const alive = useRef(true)
  useEffect(() => () => { alive.current = false }, [])

  useEffect(() => {
    if (status.kind !== 'idle') return
    let ok = true
    fetchManifest(baseUrl())
      .then(async (man) => {
        if (!ok) return
        setSize(man.total_bytes)
        const n = await cachedFiles(baseUrl(), man)
        if (ok) setCached(n === man.chunks.length ? 'yes' : n > 0 ? 'partly' : 'no')
      })
      .catch(() => undefined)
    return () => { ok = false }
  }, [status.kind])

  const load = useCallback(() => {
    setStatus({ kind: 'loading', p: null })
    getClient()
      .load(baseUrl(), (p) => alive.current && setStatus({ kind: 'loading', p }))
      .then((r) => {
        tokenizer ??= createGpt2Tokenizer(new Uint16Array(r.merges))
        if (alive.current) setStatus({ kind: 'ready' })
      })
      .catch((e: unknown) => alive.current && setStatus({ kind: 'error', message: e instanceof Error ? e.message : String(e) }))
  }, [])

  useEffect(() => { if (autoload && status.kind === 'idle') load() }, [autoload]) // eslint-disable-line react-hooks/exhaustive-deps

  const goal = <>This is the <b>real GPT-2 small</b> (OpenAI, 2019): 124 million learned numbers, running on your device. Type a prompt, follow it through every stage, open any of the 12 blocks, and press <b>Why?</b> on a stage to jump to the lesson that explains it.</>
  const ready = status.kind === 'ready' && client?.ready && tokenizer
  return (
    <Lab title="GPT-2 Explainer: one forward pass, stage by stage" goal={goal}>
      {ready ? (
        <Explorer info={client!.ready!.info} tok={tokenizer!} fromCache={client!.ready!.fromCache} loadMs={client!.ready!.ms} />
      ) : (
        <div className="Gpt2Explainer-load">
          <p className="lab-note" style={{ marginTop: 0 }}>
            The weights are {size ? MB(size) : 'about 127 MB'}: every one of GPT-2’s 124,439,808 parameters stored as a single byte (int8).
            They download once; your browser then keeps them, so the next visit starts in a second or two. Everything runs on your device and nothing you type is sent anywhere.
            On a phone, use Wi-Fi.
          </p>
          <div className="btn-row">
            <button className="btn primary" onClick={load} disabled={status.kind === 'loading'}>
              {cached === 'yes' ? 'Load GPT-2 (already in your browser)' : 'Load GPT-2 (≈130 MB, cached after first time)'}
            </button>
          </div>
          {status.kind === 'loading' && (
            <div className="Gpt2Explainer-progress" aria-live="polite">
              <div className="meter" role="progressbar" aria-label="GPT-2 download" aria-valuemin={0} aria-valuemax={100}
                aria-valuenow={status.p ? Math.round((status.p.loaded / status.p.total) * 100) : 0}>
                <span style={{ width: `${status.p ? (status.p.loaded / status.p.total) * 100 : 1}%` }} />
              </div>
              <p className="lab-note">
                {status.p
                  ? status.p.loaded >= status.p.total
                    ? 'Downloaded. Setting up the model…'
                    : `${MB(status.p.loaded)} of ${MB(status.p.total)} ${status.p.fromCache ? '(from your browser’s cache)' : '(downloading)'}`
                  : 'Starting…'}
              </p>
            </div>
          )}
          {status.kind === 'error' && (
            <div role="alert" className="Gpt2Explainer-error">
              <p className="lab-note"><b>GPT-2 could not be loaded.</b> {status.message}. Check your connection (and free memory: the model needs about 200 MB), then try again. The rest of the page works without it.</p>
              <button className="btn small" onClick={load}>Try again</button>
            </div>
          )}
        </div>
      )}
    </Lab>
  )
}

/* ---------------- the explorer, once the model is loaded ---------------- */

interface GenTok { id: number; p: number; rank: number; kept: number }

function Explorer({ info, tok, fromCache, loadMs }: { info: ModelInfo; tok: Gpt2Tokenizer; fromCache: boolean; loadMs: number }) {
  const { nLayer: L, nHead: H, nEmbd: D } = info
  const themeKey = useThemeKey()
  const c = getClient()

  const [prompt, setPrompt] = useState(PRESETS[0])
  const [trace, setTrace] = useState<TraceSummary | null>(null)
  const [promptLen, setPromptLen] = useState(0)
  const [ranPrompt, setRanPrompt] = useState('')
  const [gen, setGen] = useState<GenTok[]>([])
  const [firstMs, setFirstMs] = useState<number | null>(null)
  const [sel, setSel] = useState<number | null>(null)
  const [open, setOpen] = useState<number | null>(null)
  const [head, setHead] = useState(0)
  const [temperature, setTemperature] = useState(1)
  const [topKv, setTopK] = useState(40)
  const [topP, setTopP] = useState(1)
  const [seed, setSeed] = useState(1)
  const [busy, setBusy] = useState<null | 'run' | 'gen'>(null)
  const [error, setError] = useState<string | null>(null)
  const [animate, setAnimate] = useState(false)
  const [stage, setStage] = useState<number | null>(null)
  const [detail, setDetail] = useState<LayerDetail | null>(null)
  const stopRef = useRef(false)
  const alive = useRef(true)
  useEffect(() => () => { alive.current = false; stopRef.current = true }, [])

  const liveIds = useMemo(() => tok.encode(prompt), [tok, prompt])
  const tooLong = liveIds.length > MAX_PROMPT

  const run = useCallback(async (text: string) => {
    const ids = tok.encode(text)
    if (!ids.length || ids.length > MAX_PROMPT) return
    stopRef.current = true
    setBusy('run'); setError(null)
    try {
      const t = await c.run(ids, true)
      if (!alive.current) return
      setTrace(t); setPromptLen(ids.length); setRanPrompt(text); setGen([]); setSel(null); setDetail(null); setFirstMs(t.ms)
    } catch (e) {
      if (alive.current) setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (alive.current) setBusy(null)
    }
  }, [c, tok])

  // run the default prompt once when the model is ready (a single forward pass, not a loop)
  useEffect(() => { void run(PRESETS[0]) }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const order = useMemo(() => (trace ? orderByLogit(trace.logits) : null), [trace])
  const opts = { temperature, topK: topKv, topP }
  const dist = useMemo(() => (trace && order ? distribution(trace.logits, opts, order) : null), [trace, order, temperature, topKv, topP]) // eslint-disable-line react-hooks/exhaustive-deps

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
  const generate = useCallback(async (n: number) => {
    if (!trace || !order) return
    stopRef.current = false
    setBusy('gen'); setError(null)
    const rng = makeRng(seed * 7919 + trace.T)
    let t = trace
    let ord = order
    let out = gen.slice()
    try {
      for (let i = 0; i < n && !stopRef.current; i++) {
        if (t.T >= EXPLAINER_MAX_TOKENS) break
        const d = distribution(t.logits, { temperature, topK: topKv, topP }, ord)
        const pick = sample(d, rng.next())
        const last = i === n - 1 || t.T + 1 >= EXPLAINER_MAX_TOKENS
        t = await c.extend([pick.id], animate || last)
        if (!alive.current) return
        ord = orderByLogit(t.logits)
        out = [...out, { id: pick.id, p: pick.p, rank: pick.rank, kept: d.kept }]
        setTrace(t); setGen(out); setSel(null)
        if (animate) {
          for (let s = 0; s <= L + 5 && !stopRef.current; s++) { setStage(s); await sleep(s >= 2 && s < L + 2 ? 45 : 140) }
          setStage(null)
        }
      }
    } catch (e) {
      if (alive.current) setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (alive.current) { setBusy(null); setStage(null) }
    }
  }, [trace, order, gen, seed, temperature, topKv, topP, animate, c, L])

  // fetch the open block's full detail (not while generating: it changes every step)
  useEffect(() => {
    if (open === null || !trace || busy) return
    let ok = true
    c.layer(open).then((d) => ok && setDetail(d), (e: unknown) => ok && setError(e instanceof Error ? e.message : String(e)))
    return () => { ok = false }
  }, [open, trace, busy, c])

  if (!trace || !dist) {
    return (
      <div aria-live="polite">
        <p className="lab-note">{error ? <>Something went wrong: {error}</> : 'Running GPT-2 on the first prompt…'}</p>
      </div>
    )
  }

  const T = trace.T
  const q = sel !== null && sel < T ? sel : T - 1
  const labels = trace.ids.map((id) => showToken(tok.label(id)))
  const isGen = (t: number) => t >= promptLen
  const lensLast = trace.lens
  const stale = prompt !== ranPrompt

  return (
    <div className="Gpt2Explainer">
      <p className="lab-note Gpt2Explainer-card">
        <b>GPT-2 small</b>: {info.nParams.toLocaleString('en-US')} parameters, {L} blocks × {H} heads, vectors of {D} numbers, {info.vocabSize.toLocaleString('en-US')} tokens, a context of {info.nCtx.toLocaleString('en-US')}.
        {' '}Stored as int8 ({MB(info.totalBytes)}, {fromCache ? `read from your browser’s cache in ${(loadMs / 1000).toFixed(1)} s` : `downloaded in ${(loadMs / 1000).toFixed(1)} s and now cached`}).
        {' '}This is a close approximation of GPT-2, not bit-identical: see stage 8.
      </p>

      {/* ---------- the prompt ---------- */}
      <div className="Gpt2Explainer-prompt">
        <label htmlFor="g2e-prompt" className="Gpt2Explainer-label">Prompt ({liveIds.length} token{liveIds.length === 1 ? '' : 's'}; at most {MAX_PROMPT})</label>
        <div className="Gpt2Explainer-promptrow">
          <input id="g2e-prompt" className="input" value={prompt} spellCheck={false} onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !busy) void run(prompt) }} />
          <button className="btn small primary" disabled={!!busy || !liveIds.length || tooLong} onClick={() => void run(prompt)}>Run GPT-2</button>
        </div>
        <div className="btn-row" role="group" aria-label="Example prompts" style={{ marginTop: 6 }}>
          {PRESETS.map((p) => <button key={p} className="btn small" aria-pressed={ranPrompt === p} disabled={!!busy} onClick={() => { setPrompt(p); void run(p) }}>{p}</button>)}
        </div>
        {tooLong && <p className="lab-note" role="alert">That is {liveIds.length} tokens; the explainer keeps every intermediate, so it takes at most {MAX_PROMPT}.</p>}
        <p className="lab-note" aria-live="polite">
          {busy === 'run' ? 'Running…' : stale ? 'The prompt changed: press Run GPT-2 (or Enter) to send it through the model.' : firstMs !== null ? <>Forward pass over {promptLen} tokens: <b>{(firstMs / 1000).toFixed(2)} s</b> in your browser.</> : null}
          {error && <> <span className="Gpt2Explainer-err">Error: {error}</span></>}
        </p>
      </div>

      {/* ---------- 1. tokens ---------- */}
      <Stage n="1" title="Text → tokens" shape={`${T} ids`} why="tokens" active={stage === 0}>
        <div className="Gpt2Explainer-chips" role="group" aria-label="Tokens: pick one to follow it through the model">
          {trace.ids.map((id, t) => (
            <button key={t} className={`Gpt2Explainer-chip${isGen(t) ? ' is-gen' : ''}`} aria-pressed={t === q} onClick={() => setSel(t)}
              aria-label={`position ${t + 1}: token ${JSON.stringify(tok.label(id))}, id ${id}${isGen(t) ? ', generated' : ''}`}>
              <span className="Gpt2Explainer-chiptext">{labels[t]}</span>
              <span className="Gpt2Explainer-chipid">{id}</span>
            </button>
          ))}
        </div>
        <p className="lab-note">␣ marks a space: GPT-2 glues the space to the word after it. Click a token to follow it through every stage below (now: <b>{labels[q]}</b> at position {q + 1}).{gen.length > 0 && ' Generated tokens have a dashed border.'}</p>
      </Stage>

      {/* ---------- 2. embeddings ---------- */}
      <Stage n="2" title="Embeddings + position" shape={`${T} × ${D}`} why="embed" active={stage === 1}>
        <EmbeddingView trace={trace} q={q} D={D} labels={labels} themeKey={themeKey} />
      </Stage>

      {/* ---------- 3. the 12 blocks ---------- */}
      <Stage n="3" title={`${L} Transformer blocks`} shape={`${T} × ${D} → ${T} × ${D}`} why="blocks" active={stage !== null && stage >= 2 && stage < L + 2}>
        <p className="lab-note" style={{ marginTop: 0 }}>
          Each row is one block. The picture is its attention (all heads averaged; row = the token looking, column = the token looked at).
          The numbers are for <b>{labels[q]}</b>: the length of its vector after the block, and how much attention and the MLP added. The last column is the logit lens: the block’s best guess for the token after “{labels[T - 1]}”.
          Click a block to open it.
        </p>
        <div className="Gpt2Explainer-blocks">
          {Array.from({ length: L }, (_, l) => (
            <BlockRow key={l} l={l} trace={trace} q={q} tok={tok} themeKey={themeKey} open={open === l} active={stage === l + 2}
              onToggle={() => { setOpen(open === l ? null : l); setDetail(null) }}>
              {open === l && (detail && detail.l === l && detail.T === T
                ? <BlockDetail d={detail} q={q} setSel={setSel} head={head} setHead={setHead} labels={labels} H={H} D={D} themeKey={themeKey} />
                : <p className="lab-note" aria-busy="true">{busy === 'gen' ? 'Opens when generation stops…' : 'Reading block ' + (l + 1) + '…'}</p>)}
            </BlockRow>
          ))}
        </div>
      </Stage>

      {/* ---------- 4. final LayerNorm and logits ---------- */}
      <Stage n="4" title="Final LayerNorm → logits" shape={`${D} → ${info.vocabSize.toLocaleString('en-US')}`} why="logits" active={stage === L + 2 || stage === L + 3}>
        <p className="lab-note" style={{ marginTop: 0 }}>Only the <b>last</b> position’s vector (“{labels[T - 1]}”) is used to predict what comes next. After one more LayerNorm it is compared, by dot product, with all {info.vocabSize.toLocaleString('en-US')} rows of the token table.</p>
        <Strip v={trace.lnFLast} len={D} label={`ln_f output at “${labels[T - 1]}”`} themeKey={themeKey} />
        <LogitList dist={dist} logits={trace.logits} tok={tok} />
      </Stage>

      {/* ---------- 5. probabilities ---------- */}
      <Stage n="5" title="Probabilities: softmax, temperature, top-k, top-p" shape="top 10 of 50,257" why="probs" active={stage === L + 4}>
        <div className="controls">
          <Slider label="temperature (0 = always the top token)" value={temperature} min={0} max={2} step={0.05} onChange={setTemperature} />
          <Slider label="top-k (0 = off)" value={topKv} min={0} max={200} step={1} onChange={setTopK} />
          <Slider label="top-p (1 = off)" value={topP} min={0.05} max={1} step={0.05} onChange={setTopP} />
        </div>
        <ProbBars dist={dist} tok={tok} />
      </Stage>

      {/* ---------- 6. sample and append ---------- */}
      <Stage n="6" title="Sample → append → run again (KV cache)" shape="1 token" why="sample" active={stage === L + 5}>
        <div className="btn-row">
          {busy === 'gen'
            ? <button className="btn small" onClick={() => { stopRef.current = true }}>Stop</button>
            : <>
                <button className="btn small primary" disabled={!!busy || T >= EXPLAINER_MAX_TOKENS} onClick={() => void generate(1)}>Generate next token</button>
                <button className="btn small" disabled={!!busy || T >= EXPLAINER_MAX_TOKENS} onClick={() => void generate(10)}>Generate 10</button>
                <button className="btn small" disabled={!!busy || !gen.length} onClick={() => void run(ranPrompt)}>Back to the prompt</button>
                <button className="btn small" disabled={!!busy} onClick={() => setSeed(seed + 1)}>New dice (seed {seed})</button>
              </>}
          <label className="Gpt2Explainer-check"><input type="checkbox" checked={animate} onChange={(e) => setAnimate(e.target.checked)} disabled={busy === 'gen'} /> Show each step (slower)</label>
        </div>
        <div className="Gpt2Explainer-text" aria-label="Prompt and generated text" aria-live="polite">
          <span className="Gpt2Explainer-prompttext">{tok.decode(trace.ids.slice(0, promptLen))}</span>
          <span className="Gpt2Explainer-gentext">{tok.decode(trace.ids.slice(promptLen))}</span>
        </div>
        {gen.length > 0 && (
          <p className="lab-note" aria-live="polite">
            Last pick: <b>{showToken(tok.label(gen[gen.length - 1].id))}</b>, probability {pct(gen[gen.length - 1].p)} (rank {gen[gen.length - 1].rank} of the {gen[gen.length - 1].kept.toLocaleString('en-US')} tokens left after the cuts).
            {' '}With the KV cache only the new position was computed: <b>{(trace.ms / 1000).toFixed(2)} s</b> for 1 token, against {firstMs !== null ? `${(firstMs / 1000).toFixed(2)} s` : '…'} for the {promptLen}-token prompt.
          </p>
        )}
        {T >= EXPLAINER_MAX_TOKENS && <p className="lab-note">The explainer keeps every intermediate of every token, so it stops at {EXPLAINER_MAX_TOKENS} tokens (GPT-2 itself reads up to {info.nCtx}).</p>}
      </Stage>

      {/* ---------- 7. logit lens ---------- */}
      <Stage n="7" title="Logit lens: how the prediction forms" shape={`${L} layers`} why="lens">
        {lensLast ? <LensTable lens={lensLast} tok={tok} after={labels[T - 1]} /> : <p className="lab-note">Updating after generation…</p>}
      </Stage>

      {/* ---------- 8. the weights ---------- */}
      <Stage n="8" title="About these weights: int8" shape={MB(info.totalBytes)} why="int8">
        <p className="lab-note" style={{ marginTop: 0 }}>
          Every weight matrix is stored as int8 (whole numbers from −127 to 127) with one float32 scale per row; LayerNorm parameters and biases stay float32.
          In float32 the same model would be {MB(info.nParams * 4)}. Split into {info.chunkBytes.length} files of {info.chunkBytes.map((b) => MB(b)).join(', ')}.
        </p>
        <p className="lab-note">
          One exception, found by measuring: GPT-2’s final vector is huge in a few dimensions (on average, #496 is about 500 times the size of a typical dimension, and #430 and #36 about 200 times; look at “largest” in stage 4).
          Multiplied by those huge values, the small rounding errors of int8 in the output table changed GPT-2’s top prediction on 3 of 9 test prompts.
          So {info.outlierCols.length} columns of the token table (#{info.outlierCols.join(', #')}) stay float32 (1.6 MB): the “outlier features” idea of LLM.int8(). With them, the top prediction matched float32 GPT-2 on all 9; the probabilities still differ slightly.
        </p>
      </Stage>
    </div>
  )
}

/* ---------------- stage views ---------------- */

function EmbeddingView({ trace, q, D, labels, themeKey }: { trace: TraceSummary; q: number; D: number; labels: string[]; themeKey: number }) {
  const T = trace.T
  const x0 = useMemo(() => {
    const x = new Float32Array(T * D)
    for (let i = 0; i < x.length; i++) x[i] = trace.tokEmb[i] + trace.posEmb[i]
    return x
  }, [trace, T, D])
  const clip = useMemo(() => clipOf(x0, 0, x0.length), [x0])
  return (
    <>
      <Strip v={trace.tokEmb} off={q * D} len={D} label={`token embedding: row ${trace.ids[q]} of wte (“${labels[q]}”)`} themeKey={themeKey} />
      <Strip v={trace.posEmb} off={q * D} len={D} label={`+ position embedding: row ${q} of wpe`} themeKey={themeKey} />
      <Strip v={x0} off={q * D} len={D} label="= the vector that enters block 1" themeKey={themeKey} />
      <p className="lab-note">All {T} input vectors, one row per token (the selected one outlined):</p>
      <Heat rows={T} cols={D} get={(r, cc) => x0[r * D + cc]} clip={clip} cell={[1, 6]} outlineRow={q} themeKey={themeKey} deps={[x0]}
        label={`The ${T} by ${D} input matrix: one row per token`} className="Gpt2Explainer-matrix" />
      <p className="Gpt2Explainer-legend"><span className="Gpt2Explainer-sw pos" /> positive <span className="Gpt2Explainer-sw neg" /> negative; stronger colour = larger. Colours are clipped at the 99th percentile, because a few dimensions are far larger than the rest.</p>
    </>
  )
}

function BlockRow({ l, trace, q, tok, themeKey, open, active, onToggle, children }: {
  l: number; trace: TraceSummary; q: number; tok: Gpt2Tokenizer; themeKey: number; open: boolean; active: boolean; onToggle: () => void; children: ReactNode
}) {
  const T = trace.T
  const H = trace.attn[l].length / (T * T)
  const avg = useMemo(() => {
    const a = new Float32Array(T * T)
    const A = trace.attn[l]
    for (let h = 0; h < H; h++) for (let i = 0; i < T * T; i++) a[i] += A[h * T * T + i] / H
    return a
  }, [trace, l, T, H])
  const lens = trace.lens?.[l]
  const id = useId()
  return (
    <div className={`Gpt2Explainer-block${open ? ' is-open' : ''}${active ? ' is-active' : ''}`}>
      <button className="Gpt2Explainer-blockbtn" aria-expanded={open} aria-controls={id} onClick={onToggle}>
        <span className="Gpt2Explainer-blockname">Block {l + 1}</span>
        <Heat rows={T} cols={T} get={(r, cc) => (cc > r ? NaN : avg[r * T + cc])} clip={0.5} cell={[2, 2]} themeKey={themeKey} deps={[avg]}
          label={`average attention pattern of block ${l + 1}`} className="Gpt2Explainer-thumb" />
        <span className="Gpt2Explainer-blocknums mono">
          <span title="length of the residual stream after this block">‖x‖ {fmt(trace.residNorm[l][q], 0)}</span>
          <span title="length of what attention added">+attn {fmt(trace.attnNorm[l][q], 1)}</span>
          <span title="length of what the MLP added">+mlp {fmt(trace.mlpNorm[l][q], 1)}</span>
        </span>
        <span className="Gpt2Explainer-blocklens mono">{lens ? <>→ {showToken(tok.label(lens.top[0].id))} {pct(lens.top[0].p)}</> : ''}</span>
        <span className="Gpt2Explainer-caret" aria-hidden="true">{open ? '▾' : '▸'}</span>
      </button>
      {open && <div id={id} className="Gpt2Explainer-detail">{children}</div>}
    </div>
  )
}

function HeadPicker({ H, head, setHead }: { H: number; head: number; setHead: (h: number) => void }) {
  return (
    <div className="Gpt2Explainer-heads" role="group" aria-label="Attention head">
      <span className="Gpt2Explainer-label" style={{ margin: 0 }}>head</span>
      {Array.from({ length: H }, (_, h) => <button key={h} className="Gpt2Explainer-head" aria-pressed={h === head} onClick={() => setHead(h)} aria-label={`head ${h + 1}`}>{h + 1}</button>)}
    </div>
  )
}

function BlockDetail({ d, q, setSel, head, setHead, labels, H, D, themeKey }: {
  d: LayerDetail; q: number; setSel: (t: number) => void; head: number; setHead: (h: number) => void; labels: string[]; H: number; D: number; themeKey: number
}) {
  const T = d.T
  const hd = D / H
  const ho = head * hd
  const at = (a: Float32Array, t: number, u: number) => a[(head * T + t) * T + u]
  const row = Array.from({ length: q + 1 }, (_, u) => ({ u, w: at(d.attn, q, u), s: at(d.scores, q, u) }))
  const top = row.slice().sort((a, b) => b.w - a.w).slice(0, 3)
  const qkvClip = useMemo(() => {
    const m = (a: Float32Array) => { const x = new Float32Array(T * hd); for (let t = 0; t < T; t++) x.set(a.subarray(t * D + ho, t * D + ho + hd), t * hd); return clipOf(x, 0, x.length) }
    return [m(d.q), m(d.k), m(d.v)]
  }, [d, T, D, ho, hd])
  const cellQ: [number, number] = [4, Math.max(4, Math.min(12, Math.floor(160 / T)))]
  const cellA = Math.max(8, Math.min(40, Math.floor(340 / T)))
  const act = d.act.subarray(q * 4 * D, (q + 1) * 4 * D)
  const actStats = useMemo(() => {
    let posN = 0, big = 0
    for (let i = 0; i < act.length; i++) { if (act[i] > 0) posN++; if (act[i] > act[big]) big = i }
    const tops = Array.from(act, (v, i) => ({ i, v })).sort((a, b) => b.v - a.v).slice(0, 5)
    return { posN, big, tops, clip: clipOf(act, 0, act.length) }
  }, [act])
  const lab = labels[q]
  return (
    <div className="Gpt2Explainer-inner">
      <Stage n="a" title="LayerNorm 1" shape={`${D}`} why="ln" sub>
        <Strip v={d.xIn} off={q * D} len={D} label={`stream entering block ${d.l + 1} at “${lab}”`} themeKey={themeKey} />
        <Strip v={d.ln1} off={q * D} len={D} label="after LayerNorm 1" themeKey={themeKey} />
      </Stage>

      <Stage n="b" title="Queries, keys, values" shape={`${T} × 64 per head`} why="qkv" sub>
        <HeadPicker H={H} head={head} setHead={setHead} />
        <p className="lab-note">One matrix (768 → 2,304) makes all three for all 12 heads at once; head {head + 1} uses numbers {ho}–{ho + hd - 1} of each. One row per token, the selected one outlined.</p>
        <div className="Gpt2Explainer-qkv">
          {([['q', 'Q (query)', d.q], ['k', 'K (key)', d.k], ['v', 'V (value)', d.v]] as const).map(([tone, name, a], i) => (
            <figure key={tone} className="Gpt2Explainer-qkvfig">
              <figcaption className={`chip ${tone}`}>{name}</figcaption>
              <Heat rows={T} cols={hd} get={(r, cc) => a[r * D + ho + cc]} clip={qkvClip[i]} pos={`--${tone}`} neg="--ink-3" cell={cellQ} outlineRow={q} themeKey={themeKey} deps={[d, head]}
                label={`${name} of head ${head + 1}: ${T} rows of 64 numbers`} className="Gpt2Explainer-qkvcanvas" />
            </figure>
          ))}
        </div>
        <p className="Gpt2Explainer-legend">Colour = positive, grey = negative.</p>
      </Stage>

      <Stage n="c" title="Scores → mask → softmax" shape={`${T} × ${T}`} why="mask" sub>
        <HeadPicker H={H} head={head} setHead={setHead} />
        <p className="lab-note">score = q·k / √64. Row = the token looking, column = the token looked at. Hatched cells are the future: masked before the softmax. Click a row (or a token below) to choose who is looking.</p>
        <div className="table-scroll">
          <div className="Gpt2Explainer-attn" style={{ gridTemplateColumns: `auto ${T * cellA}px` }}>
            <div className="Gpt2Explainer-attnlabels" aria-hidden="true">
              {labels.map((s, t) => <span key={t} style={{ height: cellA, lineHeight: `${cellA}px`, fontSize: Math.min(12, cellA - 1) }} className={t === q ? 'is-q' : ''}>{cellA >= 9 ? s : ''}</span>)}
            </div>
            <Heat rows={T} cols={T} get={(r, cc) => (cc > r ? NaN : at(d.attn, r, cc))} clip={1} cell={[cellA, cellA]} outlineRow={q} onPick={(r) => setSel(r)} themeKey={themeKey} deps={[d, head]}
              label={`Attention weights of block ${d.l + 1}, head ${head + 1}`} style={{ width: T * cellA, height: T * cellA, cursor: 'pointer' }} />
          </div>
        </div>
        <div className="Gpt2Explainer-chips" role="group" aria-label="Token that is looking">
          {labels.map((s, t) => <button key={t} className="Gpt2Explainer-chip small" aria-pressed={t === q} onClick={() => setSel(t)}>{s}</button>)}
        </div>
        <p className="lab-note" style={{ marginBottom: 4 }}>Where <b>{lab}</b> looks in head {head + 1} (shade = weight):</p>
        <div className="Gpt2Explainer-chips" aria-hidden="true">
          {row.map(({ u, w }) => <span key={u} className="Gpt2Explainer-heatchip" style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(w * 85)}%, var(--paper-2))` }}>{labels[u]}</span>)}
        </div>
        <p className="readout" aria-live="polite">
          {top.map(({ u, w, s }) => <span key={u}>{labels[u]}{u === q ? ' (itself)' : ''}: score {fmt(s)} → <b>{pct(w)}</b></span>)}
        </p>
      </Stage>

      <Stage n="d" title="Heads side by side → output projection" shape={`12 × 64 → ${D}`} why="proj" sub>
        <Strip v={d.heads} off={q * D} len={D} marks={H} label={`the ${H} head outputs, side by side (head ${head + 1} = part ${head + 1})`} themeKey={themeKey} />
        <Strip v={d.attnOut} off={q * D} len={D} label="after the output projection (768 × 768): what attention adds" themeKey={themeKey} />
      </Stage>

      <Stage n="e" title="Residual add" shape={`${D} + ${D}`} why="resid" sub>
        <Strip v={d.xIn} off={q * D} len={D} label="stream in" themeKey={themeKey} />
        <Strip v={d.attnOut} off={q * D} len={D} label="+ attention output" themeKey={themeKey} />
        <Strip v={d.resid1} off={q * D} len={D} label="= stream after attention" themeKey={themeKey} />
      </Stage>

      <Stage n="f" title="MLP: 768 → 3,072 → GELU → 768" shape={`${D} → ${4 * D} → ${D}`} why="mlp" sub>
        <Strip v={d.ln2} off={q * D} len={D} label="after LayerNorm 2" themeKey={themeKey} />
        <div className="Gpt2Explainer-mlp">
          <figure className="Gpt2Explainer-mlpgrid">
            <Heat rows={48} cols={64} get={(r, cc) => act[r * 64 + cc]} clip={actStats.clip} cell={[3, 3]} themeKey={themeKey} deps={[act]}
              label={`The 3,072 hidden numbers after GELU for “${lab}”, as a 48 by 64 grid`} className="Gpt2Explainer-grid" />
            <figcaption className="Gpt2Explainer-legend">The 3,072 hidden numbers after GELU, 64 per row.</figcaption>
          </figure>
          <div>
            <GeluPlot tops={actStats.tops.map((t) => t.v)} />
            <p className="lab-note" aria-live="polite">
              {actStats.posN.toLocaleString('en-US')} of 3,072 are positive ({pct(actStats.posN / 3072)}). GELU lets positive inputs through and squashes negative ones to small values (never below −0.17), so a few neurons dominate: the largest are
              {' '}{actStats.tops.map((t) => `#${t.i} (${fmt(t.v)})`).join(', ')}.
            </p>
          </div>
        </div>
        <Strip v={d.mlpOut} off={q * D} len={D} label="MLP output (3,072 → 768): what the MLP adds" themeKey={themeKey} />
        <Strip v={d.resid2} off={q * D} len={D} label={`= stream after block ${d.l + 1} (attention stream + MLP output)`} themeKey={themeKey} />
      </Stage>
    </div>
  )
}

function GeluPlot({ tops }: { tops: number[] }) {
  // x in [-4, 4] -> [30, 230]; y in [-0.5, 4] -> [110, 10]
  const X = (x: number) => 30 + ((x + 4) / 8) * 200
  const Y = (y: number) => 110 - ((y + 0.5) / 4.5) * 100
  const pts = Array.from({ length: 81 }, (_, i) => { const x = -4 + i * 0.1; return `${X(x).toFixed(1)},${Y(geluNew(x)).toFixed(1)}` }).join(' ')
  // invert GELU on the positive branch (monotonic for x > -0.75) to place each top neuron on the curve
  const inv = (y: number) => { let lo = -0.75, hi = 20; for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; if (geluNew(m) < y) lo = m; else hi = m } return lo }
  return (
    <svg viewBox="0 0 240 125" className="Gpt2Explainer-gelu" role="img" aria-label="The GELU curve: close to zero for negative inputs, close to the input itself for positive ones. Dots mark the five largest neurons of the selected token (values above 4 are drawn at the edge).">
      <line className="axis" x1={30} x2={230} y1={Y(0)} y2={Y(0)} />
      <line className="axis" x1={X(0)} x2={X(0)} y1={10} y2={115} />
      <polyline points={pts} fill="none" stroke="var(--ink)" strokeWidth={1.8} />
      {tops.map((v, i) => { const x = Math.min(4, inv(v)); return <circle key={i} cx={X(x)} cy={Y(Math.min(4, v))} r={3.5} fill="var(--accent)" /> })}
      <text x={232} y={Y(0) - 4} textAnchor="end" fontSize={11}>input</text>
      <text x={X(0) + 4} y={20} fontSize={11}>GELU(input)</text>
      <text x={X(-4)} y={Y(0) + 14} fontSize={11}>−4</text>
      <text x={X(4) - 8} y={Y(0) + 14} fontSize={11}>4</text>
    </svg>
  )
}

function LogitList({ dist, logits, tok }: { dist: Distribution; logits: Float32Array; tok: Gpt2Tokenizer }) {
  const top = Array.from(dist.order.slice(0, 10))
  const hi = logits[top[0]]
  const lo = logits[top[9]]
  return (
    <>
      <p className="lab-note" style={{ marginBottom: 4 }}>The 10 highest of the {logits.length.toLocaleString('en-US')} logits (raw scores):</p>
      <div className="Gpt2Explainer-probs" role="list" aria-label="The 10 highest logits at the last position">
        {top.map((id) => (
          <div key={id} role="listitem" className="Gpt2Explainer-prob">
            <span className="Gpt2Explainer-problabel mono">{showToken(tok.label(id))}</span>
            <span className="Gpt2Explainer-probtrack"><span className="Gpt2Explainer-logitbar" style={{ width: `${15 + ((logits[id] - lo) / Math.max(1e-6, hi - lo)) * 85}%` }} /></span>
            <span className="Gpt2Explainer-probval mono">{logits[id].toFixed(2)}</span>
          </div>
        ))}
      </div>
      <p className="lab-note">Logits are unbounded scores and only their differences matter: GPT-2’s are all large negative numbers, and softmax does not care. (Bars start at 15% so the 10th is visible.)</p>
    </>
  )
}

function ProbBars({ dist, tok }: { dist: Distribution; tok: Gpt2Tokenizer }) {
  const n = 10
  const max = Math.max(dist.probs[0], dist.before[0], 1e-9)
  let shown = 0
  for (let r = 0; r < n; r++) shown += dist.probs[r]
  return (
    <>
      <div className="Gpt2Explainer-probs" role="list" aria-label="Top 10 next tokens with their probability before and after the cuts">
        {Array.from({ length: n }, (_, r) => {
          const id = dist.order[r]
          const cut = dist.probs[r] === 0
          return (
            <div key={id} role="listitem" className={`Gpt2Explainer-prob${cut ? ' is-cut' : ''}`}>
              <span className="Gpt2Explainer-problabel mono">{showToken(tok.label(id))}</span>
              <span className="Gpt2Explainer-probtrack">
                <span className="Gpt2Explainer-probbefore" style={{ width: `${(dist.before[r] / max) * 100}%` }} />
                <span className="Gpt2Explainer-probafter" style={{ width: `${(dist.probs[r] / max) * 100}%` }} />
              </span>
              <span className="Gpt2Explainer-probval mono">{cut ? 'cut' : pct(dist.probs[r])}</span>
            </div>
          )
        })}
      </div>
      <p className="lab-note" aria-live="polite">
        Solid bar: probability the dice will use. Outline: after temperature only, before top-k/top-p.
        {' '}<b>{dist.kept.toLocaleString('en-US')}</b> of 50,257 tokens survive the cuts; these 10 hold {pct(shown)} of the probability.
      </p>
    </>
  )
}

function LensTable({ lens, tok, after }: { lens: NonNullable<TraceSummary['lens']>; tok: Gpt2Tokenizer; after: string }) {
  const final = lens[lens.length - 1].top[0].id
  return (
    <>
      <p className="lab-note" style={{ marginTop: 0 }}>
        The guess for the token after “{after}”, read out after every block. The model’s final answer is <b>{showToken(tok.label(final))}</b>; the last column shows its probability and rank at each layer.
      </p>
      <div className="Gpt2Explainer-lens" role="table" aria-label="Top 3 predictions after each block, and the final answer’s probability and rank">
        <div role="row" className="Gpt2Explainer-lensrow is-head">
          <span role="columnheader">after</span><span role="columnheader">top 3 guesses</span><span role="columnheader">final answer: p, rank</span>
        </div>
        {lens.map((r) => (
          <div role="row" key={r.layer} className="Gpt2Explainer-lensrow">
            <span role="rowheader">block {r.layer + 1}</span>
            <span role="cell" className="Gpt2Explainer-lenschips">{r.top.slice(0, 3).map((t) => <span key={t.id} className={`chip${t.id === final ? ' acc' : ''}`}>{showToken(tok.label(t.id))} {pct(t.p)}</span>)}</span>
            <span role="cell" className="mono Gpt2Explainer-lensfinal">
              <span className="Gpt2Explainer-lensbar" style={{ width: `${Math.max(2, r.pFinal * 60)}px` }} /> {pct(r.pFinal)} <span className="muted">#{r.rankFinal.toLocaleString('en-US')}</span>
            </span>
          </div>
        ))}
      </div>
      <p className="lab-note">Early rows often guess common filler tokens; the answer usually appears only in the last few blocks. Treat early rows as hints, not as what the model “thinks”.</p>
    </>
  )
}
