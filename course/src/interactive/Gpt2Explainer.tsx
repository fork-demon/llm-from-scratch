// GPT-2 Explainer: the REAL GPT-2 small (124M parameters, int8) running in the learner's browser.
// The prompt sits on top as a row of tokens ending in GPT-2's guess. Below it, one forward pass is split into
// five chapters on a rail (tokens, embeddings, blocks, scores, next token); one chapter shows at a time, and
// each names the lesson that explains it.
// Weights: public/models/gpt2/ (phase6-engineering/export_gpt2.py). Engine: lib/gpt2.ts in a Web Worker.
import { useCallback, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
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
const n0 = (x: number) => x.toLocaleString('en-US')
const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/* ---------------- the five chapters of one forward pass ---------------- */
type ChapterKey = 'tokens' | 'embed' | 'blocks' | 'scores' | 'next'
interface Chapter { key: ChapterKey; name: string; title: string; body: ReactNode; lessons: string[] }
const CHAPTERS: Chapter[] = [
  {
    key: 'tokens', name: 'Tokens', title: 'Text becomes token ids',
    body: <>GPT-2 never sees letters. Its tokenizer cuts text into pieces from a fixed list of 50,257 and passes on their ids. Common words are one piece; rare words split into several. <span className="mono">␣</span> marks a space, which GPT-2 attaches to the word after it.</>,
    lessons: ['tokenization'],
  },
  {
    key: 'embed', name: 'Embeddings', title: 'Each id becomes 768 numbers',
    body: <>The id picks one row of a 50,257 × 768 table: the token’s starting vector. A second table adds a row for the position, because attention alone cannot tell word order. Their sum enters the first block.</>,
    lessons: ['embeddings', 'transformer-block'],
  },
  {
    key: 'blocks', name: '12 blocks', title: 'Twelve blocks refine the vectors',
    body: <>Each block reads every token’s running vector and adds to it. Attention lets tokens share information; the MLP then works on each token alone. The design repeats twelve times, each with its own weights. Under each block is its best guess so far.</>,
    lessons: ['transformer-block', 'build-gpt'],
  },
  {
    key: 'scores', name: 'Scores', title: 'The last vector scores every token',
    body: <>Only the last position predicts what comes next. After one more LayerNorm, its vector is compared by dot product with all 50,257 rows of the token table, the same table the input used. That gives one score, a logit, per possible next token.</>,
    lessons: ['build-gpt', 'softmax'],
  },
  {
    key: 'next', name: 'Next token', title: 'Scores become a choice',
    body: <>Softmax turns the scores into probabilities. Temperature sharpens or flattens them; top-k and top-p cut the long tail. One token is drawn and appended, and the model runs again for that one position, reusing the keys and values it already has: the KV cache.</>,
    lessons: ['softmax', 'inference'],
  },
]

/* the parts of one block, in the order the data flows through them */
type PartKey = 'ln' | 'qkv' | 'attn' | 'proj' | 'resid' | 'mlp'
const PARTS: { key: PartKey; name: string; text: ReactNode; lessons: string[] }[] = [
  { key: 'ln', name: 'LayerNorm', text: <>LayerNorm rescales each vector to mean 0 and spread 1, then applies learned gains, so attention always sees numbers in a steady range.</>, lessons: ['transformer-block'] },
  { key: 'qkv', name: 'Q, K, V', text: <>One matrix (768 → 2,304) makes a query, a key and a value for all 12 heads at once. Each head uses its own slice of 64 numbers.</>, lessons: ['attention'] },
  { key: 'attn', name: 'Attention', text: <>Each score is q·k / √64. A token may look only at itself and earlier tokens, so the future is masked. Softmax turns each row into weights that add up to 1.</>, lessons: ['attention', 'masks-and-heads'] },
  { key: 'proj', name: 'Combine heads', text: <>The 12 head outputs sit side by side (12 × 64 = 768), then one more matrix mixes them, so the heads’ findings can be combined.</>, lessons: ['masks-and-heads'] },
  { key: 'resid', name: 'Residual', text: <>The block adds its result to the stream instead of replacing it. Earlier information survives, and in training the gradient has a direct path back.</>, lessons: ['transformer-block'] },
  { key: 'mlp', name: 'MLP', text: <>Each token alone goes 768 → 3,072 → 768, with GELU in between. About two thirds of a block’s weights live here.</>, lessons: ['transformer-block'] },
]

function LessonLinks({ ids }: { ids: string[] }) {
  return (
    <span className="Gpt2Explainer-lessons">
      Explained in{' '}
      {ids.map((l, i) => (
        <span key={l}>{i > 0 && (i === ids.length - 1 ? ' and ' : ', ')}<a href={`#/lesson/${l}`}>{lessonById(l)?.title ?? l}</a></span>
      ))}
    </span>
  )
}

const Chevron = ({ dir }: { dir: 'left' | 'right' }) => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d={dir === 'right' ? 'M9 6l6 6-6 6' : 'M15 6l-6 6 6 6'} />
  </svg>
)

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
  /** Hatch masked cells (default). Off for thumbnails, where hatching turns to noise. */
  hatch?: boolean
  themeKey: number
  deps: unknown[]
}
/** A heatmap drawn with ImageData: positive values in `pos`, negative in `neg`, masked cells hatched grey. */
function Heat({ rows, cols, get, clip, pos = '--accent', neg = '--il-a', label, cell = [1, 1], outlineRow, className, style, onPick, hatch = true, themeKey, deps }: HeatProps) {
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
        const masked = hatch && Number.isNaN(v) && (x + y) % 4 === 0
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
  }, [rows, cols, clip, pos, neg, outlineRow, hatch, themeKey, cell[0], cell[1], ...deps]) // eslint-disable-line react-hooks/exhaustive-deps
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

/** One vector as a thin strip: every number is one column. `op` (+ or =) makes a stack of strips read as a sum. */
function Strip({ v, off = 0, len, label, themeKey, pos, neg, marks, op, peak }: {
  v: Float32Array; off?: number; len: number; label: ReactNode; themeKey: number; pos?: string; neg?: string; marks?: number; op?: '+' | '='; peak?: boolean
}) {
  const clip = useMemo(() => clipOf(v, off, len), [v, off, len])
  let n2 = 0
  let big = 0
  for (let i = 0; i < len; i++) { n2 += v[off + i] ** 2; if (Math.abs(v[off + i]) > Math.abs(v[off + big])) big = i }
  const length = fmt(Math.sqrt(n2), 1)
  return (
    <div className={`Gpt2Explainer-strip${op ? ' has-op' : ''}`}>
      {op && <span className="Gpt2Explainer-op" aria-hidden="true">{op}</span>}
      <div className="Gpt2Explainer-striphead">
        <span>{label}</span>
        <span className="Gpt2Explainer-stat">length {length}{peak && <>, peak {fmt(v[off + big], 1)} at #{big}</>}</span>
      </div>
      <div className="Gpt2Explainer-stripwrap">
        <Heat rows={1} cols={len} get={(_, c) => v[off + c]} clip={clip} pos={pos} neg={neg} cell={[1, 1]} themeKey={themeKey} deps={[v, off]}
          label={`${len} numbers, length ${length}, largest ${fmt(v[off + big], 1)} at dimension ${big}`} className="Gpt2Explainer-stripcanvas" />
        {marks ? <div className="Gpt2Explainer-marks" aria-hidden="true">{Array.from({ length: marks - 1 }, (_, i) => <span key={i} style={{ left: `${((i + 1) / marks) * 100}%` }} />)}</div> : null}
      </div>
    </div>
  )
}

const Legend = ({ children }: { children?: ReactNode }) => (
  <p className="Gpt2Explainer-legend">
    <span className="Gpt2Explainer-sw pos" /> positive <span className="Gpt2Explainer-sw neg" /> negative{children}
  </p>
)

/* ---------------- the rail: five chapters, one forward pass ---------------- */

function Rail({ active, onPick, meta, flow, disabled, sticky, idBase }: {
  active: number; onPick: (i: number) => void; meta?: ReactNode[]; flow?: number | null; disabled?: boolean; sticky?: boolean; idBase: string
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const track = useRef<HTMLDivElement>(null)
  // on a narrow screen the rail scrolls sideways: keep the current stage in view
  useEffect(() => {
    const t = track.current, b = refs.current[active]
    if (t && b && t.scrollWidth > t.clientWidth) t.scrollLeft = Math.max(0, b.offsetLeft - 24)
  }, [active])
  const onKey = (e: KeyboardEvent, i: number) => {
    const n = e.key === 'ArrowRight' ? Math.min(CHAPTERS.length - 1, i + 1) : e.key === 'ArrowLeft' ? Math.max(0, i - 1) : e.key === 'Home' ? 0 : e.key === 'End' ? CHAPTERS.length - 1 : -1
    if (n < 0) return
    e.preventDefault()
    onPick(n)
    refs.current[n]?.focus()
  }
  return (
    <div className={`Gpt2Explainer-rail${sticky ? ' is-sticky' : ''}`}>
      <div ref={track} className="Gpt2Explainer-railtrack" role="tablist" aria-label="Stages of one forward pass">
        {CHAPTERS.map((c, i) => (
          <button
            key={c.key} ref={(el) => { refs.current[i] = el }} role="tab" id={`${idBase}-tab-${c.key}`} aria-controls={`${idBase}-panel`}
            aria-selected={i === active} tabIndex={i === active ? 0 : -1} disabled={disabled}
            className={`Gpt2Explainer-step${i < active ? ' is-past' : ''}${flow === i ? ' is-flow' : ''}`}
            onClick={() => onPick(i)} onKeyDown={(e) => onKey(e, i)}
          >
            <span className="Gpt2Explainer-stepdot" aria-hidden="true" />
            <span className="Gpt2Explainer-stepname">{c.name}</span>
            <span className="Gpt2Explainer-stepmeta">{meta?.[i] ?? ' '}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

/* ---------------- the component ---------------- */

type Status = { kind: 'idle' } | { kind: 'loading'; p: LoadProgress | null } | { kind: 'error'; message: string } | { kind: 'ready' }

/**
 * `framed` (the default) puts the explainer in a lab frame, for use inside a lesson.
 * The #/gpt2 page passes framed={false}: the page is the explainer, so it needs no frame.
 */
export function Gpt2Explainer({ autoload = false, framed = true }: { autoload?: boolean; framed?: boolean }) {
  const [status, setStatus] = useState<Status>(() => (client?.ready ? { kind: 'ready' } : { kind: 'idle' }))
  const [cached, setCached] = useState<'unknown' | 'yes' | 'partly' | 'no'>('unknown')
  const [size, setSize] = useState<number | null>(null)
  const alive = useRef(true)
  // set on every mount, not only the first: React's StrictMode unmounts and remounts once in development
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])

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

  const ready = status.kind === 'ready' && client?.ready && tokenizer
  const body = ready
    ? <Explorer info={client!.ready!.info} tok={tokenizer!} fromCache={client!.ready!.fromCache} loadMs={client!.ready!.ms} sticky={!framed} />
    : <Loader status={status} cached={cached} size={size} onLoad={load} />

  return framed
    ? <Lab title="GPT-2, running in your browser" goal={<>Type a sentence and follow it through the real GPT-2 small, one stage at a time.</>}>{body}</Lab>
    : <div className="Gpt2Explainer-page">{body}</div>
}

function Loader({ status, cached, size, onLoad }: { status: Status; cached: 'unknown' | 'yes' | 'partly' | 'no'; size: number | null; onLoad: () => void }) {
  const idBase = useId()
  const p = status.kind === 'loading' ? status.p : null
  const frac = p ? p.loaded / p.total : 0
  return (
    <div className="Gpt2Explainer-loader">
      <Rail active={-1} onPick={() => undefined} disabled idBase={idBase} />
      <div className="Gpt2Explainer-loadcard">
        <h3>{cached === 'yes' ? 'GPT-2 is already on this device' : 'Load the model to begin'}</h3>
        <p>
          {cached === 'yes'
            ? <>Your browser kept the weights from last time, so it starts in a second or two.</>
            : <>The weights are {size ? MB(size) : 'about 127 MB'}: all 124,439,808 of GPT-2’s numbers, one byte each. They download once and your browser keeps them. On a phone, use Wi-Fi.</>}
          {' '}Everything runs on your device; nothing you type is sent anywhere.
        </p>
        <button className="btn primary" onClick={onLoad} disabled={status.kind === 'loading'}>
          {cached === 'yes' ? 'Load GPT-2' : `Load GPT-2 (${size ? MB(size) : '127 MB'})`}
        </button>
        {status.kind === 'loading' && (
          <div className="Gpt2Explainer-progress" aria-live="polite">
            <div className="meter" role="progressbar" aria-label="GPT-2 download" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(frac * 100)}>
              <span style={{ width: `${Math.max(1, frac * 100)}%` }} />
            </div>
            <p className="Gpt2Explainer-note">
              {p
                ? p.loaded >= p.total
                  ? 'Downloaded. Setting up the model…'
                  : `${MB(p.loaded)} of ${MB(p.total)}${p.fromCache ? ', from your browser’s cache' : ''}`
                : 'Starting…'}
            </p>
          </div>
        )}
        {status.kind === 'error' && (
          <div role="alert" className="Gpt2Explainer-error">
            <p><b>GPT-2 could not be loaded.</b> {status.message}. Check your connection and free memory (the model needs about 200 MB), then try again. The rest of the page works without it.</p>
            <button className="btn small" onClick={onLoad}>Try again</button>
          </div>
        )}
      </div>
    </div>
  )
}

/* ---------------- the explorer, once the model is loaded ---------------- */

interface GenTok { id: number; p: number; rank: number; kept: number }

/** The chapter the running generation step belongs to (for the "Show each step" animation). */
const chapterOfStage = (s: number, L: number) => (s <= 1 ? s : s < L + 2 ? 2 : s < L + 4 ? 3 : 4)

function Explorer({ info, tok, fromCache, loadMs, sticky }: { info: ModelInfo; tok: Gpt2Tokenizer; fromCache: boolean; loadMs: number; sticky: boolean }) {
  const { nLayer: L, nHead: H, nEmbd: D } = info
  const themeKey = useThemeKey()
  const c = getClient()
  const idBase = useId()

  const [prompt, setPrompt] = useState(PRESETS[0])
  const [trace, setTrace] = useState<TraceSummary | null>(null)
  const [promptLen, setPromptLen] = useState(0)
  const [ranPrompt, setRanPrompt] = useState('')
  const [gen, setGen] = useState<GenTok[]>([])
  const [firstMs, setFirstMs] = useState<number | null>(null)
  const [sel, setSel] = useState<number | null>(null)
  const [chapter, setChapter] = useState(0)
  const [open, setOpen] = useState(0)
  const [part, setPart] = useState<PartKey>('attn')
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
  const anchor = useRef<HTMLDivElement>(null)
  useEffect(() => { alive.current = true; return () => { alive.current = false; stopRef.current = true } }, [])

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

  // fetch the open block's full detail, only while its chapter is showing (and not while generating)
  const blocksShown = CHAPTERS[chapter].key === 'blocks'
  useEffect(() => {
    if (!blocksShown || !trace || busy) return
    let ok = true
    c.layer(open).then((d) => ok && setDetail(d), (e: unknown) => ok && setError(e instanceof Error ? e.message : String(e)))
    return () => { ok = false }
  }, [blocksShown, open, trace, busy, c])

  const goTo = (i: number, scroll = false) => {
    setChapter(i)
    // from the pager at the bottom of a long chapter: bring the top of the explainer back into view
    const a = anchor.current
    if (scroll && a && a.getBoundingClientRect().top < 0) a.scrollIntoView({ block: 'start', behavior: reducedMotion() ? 'auto' : 'smooth' })
  }

  if (!trace || !dist) {
    return (
      <div aria-live="polite" className="Gpt2Explainer-loader">
        <p className="Gpt2Explainer-note">{error ? <>Something went wrong: {error}</> : 'Running GPT-2 on the first prompt…'}</p>
      </div>
    )
  }

  const T = trace.T
  const q = sel !== null && sel < T ? sel : T - 1
  const labels = trace.ids.map((id) => showToken(tok.label(id)))
  const stale = prompt !== ranPrompt
  const guess = dist.order[0]
  const flow = stage === null ? null : chapterOfStage(stage, L)
  const meta = [
    `${T} token${T === 1 ? '' : 's'}`,
    `${T} × ${D}`,
    'attention + MLP',
    `${n0(info.vocabSize)} logits`,
    <><span className="mono">{showToken(tok.label(guess))}</span> {pct(dist.probs[0])}</>,
  ]
  const ch = CHAPTERS[chapter]
  const prev = CHAPTERS[chapter - 1]
  const next = CHAPTERS[chapter + 1]

  return (
    <div className="Gpt2Explainer">
      {/* ---------- the prompt ---------- */}
      <div className="Gpt2Explainer-prompt">
        <label htmlFor={`${idBase}-prompt`} className="sr-only">Prompt</label>
        <div className="Gpt2Explainer-field">
          <input id={`${idBase}-prompt`} value={prompt} spellCheck={false} autoComplete="off" placeholder="Type a sentence…"
            onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !busy) void run(prompt) }} />
          <button className="btn primary" disabled={!!busy || !liveIds.length || tooLong} onClick={() => void run(prompt)}>Run</button>
        </div>
        <div className="Gpt2Explainer-presets" role="group" aria-label="Example prompts">
          <span className="Gpt2Explainer-presetlabel">Try</span>
          {PRESETS.map((p) => (
            <button key={p} className="Gpt2Explainer-preset" aria-pressed={ranPrompt === p} disabled={!!busy} onClick={() => { setPrompt(p); void run(p) }}>{p}</button>
          ))}
        </div>
        <p className="Gpt2Explainer-status" aria-live="polite">
          {tooLong
            ? <span role="alert">That is {liveIds.length} tokens. The explainer keeps every intermediate, so it takes at most {MAX_PROMPT}.</span>
            : busy === 'run' ? 'Running…'
            : stale ? 'Press Run (or Enter) to send the new prompt through the model.'
            : firstMs !== null ? <>Forward pass over {promptLen} tokens: {(firstMs / 1000).toFixed(2)} s on your device.</> : null}
          {error && <> <span className="Gpt2Explainer-err">Error: {error}</span></>}
        </p>
      </div>

      {/* ---------- the sentence, as GPT-2 sees it, ending in its guess ---------- */}
      <div className="Gpt2Explainer-sentence" role="group" aria-label="Tokens: pick one to follow it through the model">
        {trace.ids.map((id, t) => (
          <button key={t} className={`Gpt2Explainer-tok${t >= promptLen ? ' is-gen' : ''}`} data-tone={t % 5} aria-pressed={t === q} onClick={() => setSel(t)}
            aria-label={`position ${t + 1}: token ${JSON.stringify(tok.label(id))}, id ${id}${t >= promptLen ? ', generated' : ''}`}>
            {labels[t]}
          </button>
        ))}
        <span className="Gpt2Explainer-guess" title="GPT-2’s most likely next token">
          <svg width="18" height="12" viewBox="0 0 18 12" aria-hidden="true"><path d="M1 6h14M11 2l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
          <span className="mono">{showToken(tok.label(guess))}</span>
          <span className="Gpt2Explainer-guessp">{pct(dist.probs[0])}</span>
        </span>
      </div>
      <p className="Gpt2Explainer-hint">
        Following <b className="mono">{labels[q]}</b>{q === T - 1 ? ', the last token, whose vector makes the prediction' : `, token ${q + 1} of ${T}`}. Click any token to follow it instead.
      </p>

      {/* ---------- the rail and one chapter ---------- */}
      <div ref={anchor} className="Gpt2Explainer-anchor" />
      <Rail active={chapter} onPick={(i) => goTo(i)} meta={meta} flow={flow} sticky={sticky} idBase={idBase} />

      <div className="Gpt2Explainer-panel" role="tabpanel" id={`${idBase}-panel`} aria-labelledby={`${idBase}-tab-${ch.key}`}>
        <div className="Gpt2Explainer-aside">
          <h3>{ch.title}</h3>
          <p>{ch.body}</p>
          <p><LessonLinks ids={ch.lessons} /></p>
        </div>
        <div className="Gpt2Explainer-main">
          {ch.key === 'tokens' && <TokensView trace={trace} labels={labels} q={q} promptLen={promptLen} text={ranPrompt} tok={tok} />}
          {ch.key === 'embed' && <EmbeddingView trace={trace} q={q} D={D} labels={labels} themeKey={themeKey} />}
          {ch.key === 'blocks' && (
            <BlocksView trace={trace} q={q} tok={tok} labels={labels} themeKey={themeKey} open={open} setOpen={(l) => { setOpen(l); setDetail(null) }}
              flowBlock={stage !== null && stage >= 2 && stage < L + 2 ? stage - 2 : null} busy={busy}
              detail={detail && detail.l === open && detail.T === T ? detail : null}
              part={part} setPart={setPart} head={head} setHead={setHead} setSel={setSel} H={H} D={D} />
          )}
          {ch.key === 'scores' && <ScoresView trace={trace} dist={dist} tok={tok} labels={labels} themeKey={themeKey} D={D} />}
          {ch.key === 'next' && (
            <NextView dist={dist} tok={tok} trace={trace} promptLen={promptLen} gen={gen} firstMs={firstMs} nCtx={info.nCtx}
              temperature={temperature} setTemperature={setTemperature} topK={topKv} setTopK={setTopK} topP={topP} setTopP={setTopP}
              busy={busy} animate={animate} setAnimate={setAnimate} seed={seed} setSeed={setSeed}
              onGenerate={(n) => void generate(n)} onStop={() => { stopRef.current = true }} onReset={() => void run(ranPrompt)} />
          )}
        </div>
      </div>

      <nav className="Gpt2Explainer-pager" aria-label="Previous and next stage">
        {prev ? <button className="btn ghost small" onClick={() => goTo(chapter - 1, true)}><Chevron dir="left" />{prev.name}</button> : <span />}
        {next && <button className="btn small" onClick={() => goTo(chapter + 1, true)}>Next: {next.name}<Chevron dir="right" /></button>}
      </nav>

      <About info={info} fromCache={fromCache} loadMs={loadMs} />
    </div>
  )
}

/* ---------------- chapter views ---------------- */

function TokensView({ trace, labels, q, promptLen, text, tok }: { trace: TraceSummary; labels: string[]; q: number; promptLen: number; text: string; tok: Gpt2Tokenizer }) {
  const T = trace.T
  const [word, setWord] = useState('Srirangapatna')
  const wordIds = useMemo(() => tok.encode(word).slice(0, 40), [tok, word])
  const id = useId()
  return (
    <>
      <div className="Gpt2Explainer-ids" role="list" aria-label="Each token and its id">
        {trace.ids.map((tid, t) => (
          <div role="listitem" key={t} className={`Gpt2Explainer-idcol${t === q ? ' is-sel' : ''}${t >= promptLen ? ' is-gen' : ''}`} data-tone={t % 5}>
            <span className="Gpt2Explainer-idtok">{labels[t]}</span>
            <span className="Gpt2Explainer-idnum">{tid}</span>
          </div>
        ))}
      </div>
      <p className="Gpt2Explainer-note">
        {text.length} characters became {promptLen} token{promptLen === 1 ? '' : 's'}{T > promptLen && <>, and GPT-2 has added {T - promptLen} more</>}.
        {' '}The model only ever receives the ids: each is a row number in its list of 50,257 pieces.
      </p>

      <div className="Gpt2Explainer-try">
        <label htmlFor={id}>Try any word</label>
        <input id={id} className="input" value={word} spellCheck={false} autoComplete="off" onChange={(e) => setWord(e.target.value)} />
        <div className="Gpt2Explainer-ids is-small" aria-live="polite" aria-label={`${wordIds.length} tokens`}>
          {wordIds.map((tid, i) => (
            <div key={i} className="Gpt2Explainer-idcol" data-tone={i % 5}>
              <span className="Gpt2Explainer-idtok">{showToken(tok.label(tid))}</span>
              <span className="Gpt2Explainer-idnum">{tid}</span>
            </div>
          ))}
        </div>
        <p className="Gpt2Explainer-note">{wordIds.length === 0 ? 'Type something to see how it is cut.' : wordIds.length === 1 ? 'One token: GPT-2 knows this one whole.' : `${wordIds.length} tokens. Rare words, names and other scripts split into several pieces.`}</p>
      </div>
    </>
  )
}

function EmbeddingView({ trace, q, D, labels, themeKey }: { trace: TraceSummary; q: number; D: number; labels: string[]; themeKey: number }) {
  const T = trace.T
  const x0 = useMemo(() => {
    const x = new Float32Array(T * D)
    for (let i = 0; i < x.length; i++) x[i] = trace.tokEmb[i] + trace.posEmb[i]
    return x
  }, [trace, T, D])
  const clip = useMemo(() => clipOf(x0, 0, x0.length), [x0])
  const rowH = Math.max(4, Math.min(18, Math.floor(200 / T)))
  return (
    <>
      <div className="Gpt2Explainer-sum">
        <Strip v={trace.tokEmb} off={q * D} len={D} label={<>Row {trace.ids[q]} of the token table, for <b className="mono">{labels[q]}</b></>} themeKey={themeKey} />
        <Strip op="+" v={trace.posEmb} off={q * D} len={D} label={<>Row {q} of the position table</>} themeKey={themeKey} />
        <Strip op="=" v={x0} off={q * D} len={D} label="Their sum: the vector that enters block 1" themeKey={themeKey} />
      </div>
      <h4 className="Gpt2Explainer-subhead">All {T} input vectors</h4>
      <div className="Gpt2Explainer-matrix">
        <div className="Gpt2Explainer-rowlabels" aria-hidden="true">
          {labels.map((s, t) => <span key={t} style={{ height: rowH, lineHeight: `${rowH}px`, fontSize: Math.min(12, rowH - 1) }} className={t === q ? 'is-q' : ''}>{rowH >= 9 ? s : ''}</span>)}
        </div>
        <Heat rows={T} cols={D} get={(r, cc) => x0[r * D + cc]} clip={clip} cell={[1, rowH]} outlineRow={q} themeKey={themeKey} deps={[x0]}
          label={`The ${T} by ${D} input matrix: one row per token`} className="Gpt2Explainer-matrixcanvas" style={{ height: T * rowH }} />
      </div>
      <Legend>. One row per token, one column per number; the outlined row is the token you are following. Colours are clipped at the 99th percentile, because a few dimensions are far larger than the rest.</Legend>
    </>
  )
}

function BlocksView({ trace, q, tok, labels, themeKey, open, setOpen, flowBlock, busy, detail, part, setPart, head, setHead, setSel, H, D }: {
  trace: TraceSummary; q: number; tok: Gpt2Tokenizer; labels: string[]; themeKey: number; open: number; setOpen: (l: number) => void; flowBlock: number | null
  busy: null | 'run' | 'gen'; detail: LayerDetail | null; part: PartKey; setPart: (p: PartKey) => void; head: number; setHead: (h: number) => void; setSel: (t: number) => void; H: number; D: number
}) {
  const L = trace.attn.length
  const T = trace.T
  return (
    <>
      <div className="Gpt2Explainer-tiles" role="group" aria-label="The 12 blocks: pick one to look inside">
        {Array.from({ length: L }, (_, l) => (
          <BlockTile key={l} l={l} trace={trace} tok={tok} themeKey={themeKey} open={open === l} flowing={flowBlock === l} onPick={() => setOpen(l)} />
        ))}
      </div>
      <p className="Gpt2Explainer-legend">Each picture is the block’s attention, all 12 heads averaged: row = the token looking, column = the token looked at. Under it, the block’s guess for the token after “{labels[T - 1]}”.</p>

      <section className="Gpt2Explainer-detail" aria-label={`Inside block ${open + 1}`}>
        <div className="Gpt2Explainer-detailhead">
          <h4>Inside block {open + 1}</h4>
          <p>
            For <b className="mono">{labels[q]}</b>, attention added a vector of length {fmt(trace.attnNorm[open][q], 1)} and the MLP one of length {fmt(trace.mlpNorm[open][q], 1)}.
            {' '}Its running vector now has length {fmt(trace.residNorm[open][q], 0)}.
          </p>
        </div>
        <div className="Gpt2Explainer-seg" role="tablist" aria-label="Part of the block">
          {PARTS.map((p) => (
            <button key={p.key} role="tab" aria-selected={part === p.key} className="Gpt2Explainer-segbtn" onClick={() => setPart(p.key)}>{p.name}</button>
          ))}
        </div>
        {detail
          ? <BlockPart d={detail} part={part} q={q} setSel={setSel} head={head} setHead={setHead} labels={labels} H={H} D={D} themeKey={themeKey} />
          : <p className="Gpt2Explainer-note" aria-busy="true">{busy === 'gen' ? 'Opens when generation stops…' : `Reading block ${open + 1}…`}</p>}
      </section>
    </>
  )
}

function BlockTile({ l, trace, tok, themeKey, open, flowing, onPick }: { l: number; trace: TraceSummary; tok: Gpt2Tokenizer; themeKey: number; open: boolean; flowing: boolean; onPick: () => void }) {
  const T = trace.T
  const H = trace.attn[l].length / (T * T)
  const avg = useMemo(() => {
    const a = new Float32Array(T * T)
    const A = trace.attn[l]
    for (let h = 0; h < H; h++) for (let i = 0; i < T * T; i++) a[i] += A[h * T * T + i] / H
    return a
  }, [trace, l, T, H])
  const lens = trace.lens?.[l]
  const cell = Math.max(1, Math.floor(48 / T))
  return (
    <button className={`Gpt2Explainer-tile${flowing ? ' is-flow' : ''}`} aria-pressed={open} onClick={onPick}>
      <span className="Gpt2Explainer-tilename">Block {l + 1}</span>
      <Heat rows={T} cols={T} get={(r, cc) => (cc > r ? NaN : avg[r * T + cc])} clip={0.5} cell={[cell, cell]} hatch={false} themeKey={themeKey} deps={[avg]}
        label={`average attention pattern of block ${l + 1}`} className="Gpt2Explainer-thumb" />
      <span className="Gpt2Explainer-tileguess">
        {lens ? <><span className="mono">{showToken(tok.label(lens.top[0].id))}</span> {pct(lens.top[0].p)}</> : ' '}
      </span>
    </button>
  )
}

function HeadPicker({ H, head, setHead }: { H: number; head: number; setHead: (h: number) => void }) {
  return (
    <div className="Gpt2Explainer-heads" role="group" aria-label="Attention head">
      <span className="Gpt2Explainer-headlabel">Head</span>
      {Array.from({ length: H }, (_, h) => <button key={h} className="Gpt2Explainer-head" aria-pressed={h === head} onClick={() => setHead(h)} aria-label={`head ${h + 1}`}>{h + 1}</button>)}
    </div>
  )
}

function BlockPart({ d, part, q, setSel, head, setHead, labels, H, D, themeKey }: {
  d: LayerDetail; part: PartKey; q: number; setSel: (t: number) => void; head: number; setHead: (h: number) => void; labels: string[]; H: number; D: number; themeKey: number
}) {
  const T = d.T
  const hd = D / H
  const ho = head * hd
  const at = (a: Float32Array, t: number, u: number) => a[(head * T + t) * T + u]
  const lab = labels[q]
  const info = PARTS.find((p) => p.key === part)!
  const caption = <p className="Gpt2Explainer-partcap">{info.text} <LessonLinks ids={info.lessons} /></p>

  const qkvClip = useMemo(() => {
    const m = (a: Float32Array) => { const x = new Float32Array(T * hd); for (let t = 0; t < T; t++) x.set(a.subarray(t * D + ho, t * D + ho + hd), t * hd); return clipOf(x, 0, x.length) }
    return [m(d.q), m(d.k), m(d.v)]
  }, [d, T, D, ho, hd])
  const act = d.act.subarray(q * 4 * D, (q + 1) * 4 * D)
  const actStats = useMemo(() => {
    let posN = 0
    for (let i = 0; i < act.length; i++) if (act[i] > 0) posN++
    const tops = Array.from(act, (v, i) => ({ i, v })).sort((a, b) => b.v - a.v).slice(0, 5)
    return { posN, tops, clip: clipOf(act, 0, act.length) }
  }, [act])

  if (part === 'ln') {
    return (
      <>{caption}
        <Strip v={d.xIn} off={q * D} len={D} label={<>Stream entering block {d.l + 1} at <b className="mono">{lab}</b></>} themeKey={themeKey} />
        <Strip v={d.ln1} off={q * D} len={D} label="After LayerNorm 1" themeKey={themeKey} />
        <Legend />
      </>
    )
  }

  if (part === 'qkv') {
    const cellQ: [number, number] = [4, Math.max(4, Math.min(12, Math.floor(160 / T)))]
    return (
      <>{caption}
        <h5 className="Gpt2Explainer-parttitle">Queries, keys, values</h5>
        <HeadPicker H={H} head={head} setHead={setHead} />
        <div className="Gpt2Explainer-qkv">
          {([['q', 'Query', d.q], ['k', 'Key', d.k], ['v', 'Value', d.v]] as const).map(([tone, name, a], i) => (
            <figure key={tone} className="Gpt2Explainer-qkvfig">
              <figcaption><span className={`chip ${tone}`}>{name}</span></figcaption>
              <Heat rows={T} cols={hd} get={(r, cc) => a[r * D + ho + cc]} clip={qkvClip[i]} pos={`--${tone}`} neg="--ink-3" cell={cellQ} outlineRow={q} themeKey={themeKey} deps={[d, head]}
                label={`${name} of head ${head + 1}: ${T} rows of 64 numbers`} className="Gpt2Explainer-qkvcanvas" />
            </figure>
          ))}
        </div>
        <p className="Gpt2Explainer-legend">Head {head + 1} uses numbers {ho} to {ho + hd - 1} of each. One row per token; the outlined row is <span className="mono">{lab}</span>. Colour is positive, grey is negative.</p>
      </>
    )
  }

  if (part === 'attn') {
    const row = Array.from({ length: q + 1 }, (_, u) => ({ u, w: at(d.attn, q, u), s: at(d.scores, q, u) }))
    const top = row.slice().sort((a, b) => b.w - a.w).slice(0, 3)
    const cellA = Math.max(8, Math.min(40, Math.floor(300 / T)))
    return (
      <>{caption}
        <HeadPicker H={H} head={head} setHead={setHead} />
        <div className="Gpt2Explainer-attnwrap">
          <div className="table-scroll">
            <div className="Gpt2Explainer-attn" style={{ gridTemplateColumns: `auto ${T * cellA}px` }}>
              <div className="Gpt2Explainer-rowlabels" aria-hidden="true">
                {labels.map((s, t) => <span key={t} style={{ height: cellA, lineHeight: `${cellA}px`, fontSize: Math.min(12, cellA - 1) }} className={t === q ? 'is-q' : ''}>{cellA >= 9 ? s : ''}</span>)}
              </div>
              <Heat rows={T} cols={T} get={(r, cc) => (cc > r ? NaN : at(d.attn, r, cc))} clip={1} cell={[cellA, cellA]} outlineRow={q} onPick={(r) => setSel(r)} themeKey={themeKey} deps={[d, head]}
                label={`Attention weights of block ${d.l + 1}, head ${head + 1}`} style={{ width: T * cellA, height: T * cellA, cursor: 'pointer' }} />
            </div>
          </div>
          <div className="Gpt2Explainer-looks">
            <h5 className="Gpt2Explainer-parttitle">Where <span className="mono">{lab}</span> looks</h5>
            <div className="Gpt2Explainer-heatchips" aria-hidden="true">
              {row.map(({ u, w }) => <span key={u} className="Gpt2Explainer-heatchip" style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(w * 85)}%, var(--paper-2))` }}>{labels[u]}</span>)}
            </div>
            <ul className="Gpt2Explainer-toplist" aria-live="polite">
              {top.map(({ u, w, s }) => (
                <li key={u}><span><span className="mono">{labels[u]}</span>{u === q ? ' (itself)' : ''}</span><span className="Gpt2Explainer-stat">score {fmt(s)}</span><b>{pct(w)}</b></li>
              ))}
            </ul>
          </div>
        </div>
        <p className="Gpt2Explainer-legend">Hatched cells are the future, masked before the softmax. Click a row, or a token at the top, to change who is looking.</p>
      </>
    )
  }

  if (part === 'proj') {
    return (
      <>{caption}
        <Strip v={d.heads} off={q * D} len={D} marks={H} label={<>The {H} head outputs side by side (head {head + 1} is part {head + 1})</>} themeKey={themeKey} />
        <Strip v={d.attnOut} off={q * D} len={D} label="After the output projection (768 × 768): what attention adds" themeKey={themeKey} />
        <Legend />
      </>
    )
  }

  if (part === 'resid') {
    return (
      <>{caption}
        <div className="Gpt2Explainer-sum">
          <Strip v={d.xIn} off={q * D} len={D} label="Stream in" themeKey={themeKey} />
          <Strip op="+" v={d.attnOut} off={q * D} len={D} label="Attention output" themeKey={themeKey} />
          <Strip op="=" v={d.resid1} off={q * D} len={D} label="Stream after attention" themeKey={themeKey} />
          <Strip op="+" v={d.mlpOut} off={q * D} len={D} label="MLP output" themeKey={themeKey} />
          <Strip op="=" v={d.resid2} off={q * D} len={D} label={`Stream after block ${d.l + 1}`} themeKey={themeKey} />
        </div>
        <Legend />
      </>
    )
  }

  return (
    <>{caption}
      <Strip v={d.ln2} off={q * D} len={D} label="After LayerNorm 2: the MLP’s input" themeKey={themeKey} />
      <div className="Gpt2Explainer-mlp">
        <figure className="Gpt2Explainer-mlpgrid">
          <Heat rows={48} cols={64} get={(r, cc) => act[r * 64 + cc]} clip={actStats.clip} cell={[3, 3]} themeKey={themeKey} deps={[act]}
            label={`The 3,072 hidden numbers after GELU for “${lab}”, as a 48 by 64 grid`} className="Gpt2Explainer-grid" />
          <figcaption className="Gpt2Explainer-legend">The 3,072 hidden numbers after GELU, 64 per row.</figcaption>
        </figure>
        <div>
          <GeluPlot tops={actStats.tops.map((t) => t.v)} />
          <p className="Gpt2Explainer-note" aria-live="polite">
            {n0(actStats.posN)} of 3,072 are positive ({pct(actStats.posN / 3072)}). GELU passes positive inputs and squashes negative ones to small values (never below −0.17), so a few neurons dominate. The largest:
            {' '}{actStats.tops.map((t) => `#${t.i} (${fmt(t.v)})`).join(', ')}.
          </p>
        </div>
      </div>
      <Strip v={d.mlpOut} off={q * D} len={D} label="MLP output (3,072 → 768): what the MLP adds" themeKey={themeKey} />
    </>
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

function ScoresView({ trace, dist, tok, labels, themeKey, D }: { trace: TraceSummary; dist: Distribution; tok: Gpt2Tokenizer; labels: string[]; themeKey: number; D: number }) {
  const [view, setView] = useState<'top' | 'lens'>('top')
  const T = trace.T
  return (
    <>
      <Strip v={trace.lnFLast} len={D} label={<>The vector at <b className="mono">{labels[T - 1]}</b> after the final LayerNorm</>} themeKey={themeKey} peak />
      <div className="Gpt2Explainer-seg" role="tablist" aria-label="Scores view">
        <button role="tab" aria-selected={view === 'top'} className="Gpt2Explainer-segbtn" onClick={() => setView('top')}>Highest scores</button>
        <button role="tab" aria-selected={view === 'lens'} className="Gpt2Explainer-segbtn" onClick={() => setView('lens')}>Guess after each block</button>
      </div>
      {view === 'top'
        ? <LogitList dist={dist} logits={trace.logits} tok={tok} />
        : trace.lens ? <LensTable lens={trace.lens} tok={tok} after={labels[T - 1]} /> : <p className="Gpt2Explainer-note">Updating after generation…</p>}
    </>
  )
}

function LogitList({ dist, logits, tok }: { dist: Distribution; logits: Float32Array; tok: Gpt2Tokenizer }) {
  const top = Array.from(dist.order.slice(0, 10))
  const hi = logits[top[0]]
  const lo = logits[top[9]]
  return (
    <>
      <div className="Gpt2Explainer-bars" role="list" aria-label="The 10 highest logits at the last position">
        {top.map((id) => (
          <div key={id} role="listitem" className="Gpt2Explainer-bar">
            <span className="Gpt2Explainer-barlabel mono">{showToken(tok.label(id))}</span>
            <span className="Gpt2Explainer-bartrack"><span className="Gpt2Explainer-logitbar" style={{ width: `${15 + ((logits[id] - lo) / Math.max(1e-6, hi - lo)) * 85}%` }} /></span>
            <span className="Gpt2Explainer-barval">{logits[id].toFixed(2)}</span>
          </div>
        ))}
      </div>
      <p className="Gpt2Explainer-legend">The 10 highest of {n0(logits.length)} scores. Only their differences matter, which is why GPT-2’s can all be large negative numbers. Bars start at 15% so the tenth stays visible.</p>
    </>
  )
}

function LensTable({ lens, tok, after }: { lens: NonNullable<TraceSummary['lens']>; tok: Gpt2Tokenizer; after: string }) {
  const final = lens[lens.length - 1].top[0].id
  return (
    <>
      <p className="Gpt2Explainer-note">
        The logit lens: apply the final LayerNorm and the scoring to the stream after every block, as if the model stopped there. The model’s final answer after “{after}” is <b className="mono">{showToken(tok.label(final))}</b>.
      </p>
      <div className="Gpt2Explainer-lens" role="table" aria-label="Top 3 predictions after each block, and the final answer’s probability and rank">
        <div role="row" className="Gpt2Explainer-lensrow is-head">
          <span role="columnheader">After</span><span role="columnheader">Top 3 guesses</span><span role="columnheader">Final answer</span>
        </div>
        {lens.map((r) => (
          <div role="row" key={r.layer} className="Gpt2Explainer-lensrow">
            <span role="rowheader">Block {r.layer + 1}</span>
            <span role="cell" className="Gpt2Explainer-lenschips">{r.top.slice(0, 3).map((t) => <span key={t.id} className={`Gpt2Explainer-lenschip${t.id === final ? ' is-final' : ''}`}><span className="mono">{showToken(tok.label(t.id))}</span> {pct(t.p)}</span>)}</span>
            <span role="cell" className="Gpt2Explainer-lensfinal">
              <span className="Gpt2Explainer-lensbar"><span style={{ width: `${Math.max(2, r.pFinal * 100)}%` }} /></span>
              <span>{pct(r.pFinal)}</span><span className="Gpt2Explainer-stat">rank {n0(r.rankFinal)}</span>
            </span>
          </div>
        ))}
      </div>
      <p className="Gpt2Explainer-legend">Early blocks were never trained to be read this way, so treat their guesses as hints, not as what the model “thinks”. The answer usually appears only in the last few blocks.</p>
    </>
  )
}

function NextView({ dist, tok, trace, promptLen, gen, firstMs, nCtx, temperature, setTemperature, topK, setTopK, topP, setTopP, busy, animate, setAnimate, seed, setSeed, onGenerate, onStop, onReset }: {
  dist: Distribution; tok: Gpt2Tokenizer; trace: TraceSummary; promptLen: number; gen: GenTok[]; firstMs: number | null; nCtx: number
  temperature: number; setTemperature: (v: number) => void; topK: number; setTopK: (v: number) => void; topP: number; setTopP: (v: number) => void
  busy: null | 'run' | 'gen'; animate: boolean; setAnimate: (v: boolean) => void; seed: number; setSeed: (v: number) => void
  onGenerate: (n: number) => void; onStop: () => void; onReset: () => void
}) {
  const T = trace.T
  const full = T >= EXPLAINER_MAX_TOKENS
  const lastPick = gen[gen.length - 1]
  return (
    <>
      <div className="Gpt2Explainer-nextgrid">
        <ProbBars dist={dist} tok={tok} />
        <div className="Gpt2Explainer-knobs">
          <Slider label="temperature" value={temperature} min={0} max={2} step={0.05} onChange={setTemperature} format={(v) => (v === 0 ? '0, always the top' : v.toFixed(2))} />
          <Slider label="top-k" value={topK} min={0} max={200} step={1} onChange={setTopK} format={(v) => (v === 0 ? 'off' : String(v))} />
          <Slider label="top-p" value={topP} min={0.05} max={1} step={0.05} onChange={setTopP} format={(v) => (v >= 1 ? 'off' : v.toFixed(2))} />
        </div>
      </div>

      <div className="Gpt2Explainer-genbar">
        <div className="btn-row">
          {busy === 'gen'
            ? <button className="btn" onClick={onStop}>Stop</button>
            : <>
                <button className="btn primary" disabled={!!busy || full} onClick={() => onGenerate(1)}>Generate next token</button>
                <button className="btn" disabled={!!busy || full} onClick={() => onGenerate(10)}>Generate 10</button>
                <button className="btn ghost" disabled={!!busy || !gen.length} onClick={onReset}>Reset to the prompt</button>
              </>}
        </div>
        <div className="Gpt2Explainer-genopts">
          <label className="Gpt2Explainer-check"><input type="checkbox" checked={animate} onChange={(e) => setAnimate(e.target.checked)} disabled={busy === 'gen'} /> Show each step on the rail</label>
          <span>
            Dice seed {seed}{' '}
            <button className="Gpt2Explainer-link" disabled={!!busy} onClick={() => setSeed(seed + 1)}>Change</button>
            <span className="sr-only">. The dice are seeded, so the same settings give the same text.</span>
          </span>
        </div>
      </div>

      <div className="Gpt2Explainer-text" aria-label="Prompt and generated text" aria-live="polite">
        <span className="Gpt2Explainer-prompttext">{tok.decode(trace.ids.slice(0, promptLen))}</span>
        <span className="Gpt2Explainer-gentext">{tok.decode(trace.ids.slice(promptLen))}</span>
      </div>
      {lastPick && (
        <p className="Gpt2Explainer-note" aria-live="polite">
          Last pick: <b className="mono">{showToken(tok.label(lastPick.id))}</b>, probability {pct(lastPick.p)}, rank {lastPick.rank} of the {n0(lastPick.kept)} tokens left after the cuts.
          {' '}With the KV cache only the new position was computed: {(trace.ms / 1000).toFixed(2)} s for 1 token, against {firstMs !== null ? `${(firstMs / 1000).toFixed(2)} s` : '…'} for the {promptLen}-token prompt.
        </p>
      )}
      {full && <p className="Gpt2Explainer-note">The explainer keeps every intermediate of every token, so it stops at {EXPLAINER_MAX_TOKENS} tokens. GPT-2 itself reads up to {n0(nCtx)}.</p>}
    </>
  )
}

function ProbBars({ dist, tok }: { dist: Distribution; tok: Gpt2Tokenizer }) {
  const n = 10
  const max = Math.max(dist.probs[0], dist.before[0], 1e-9)
  let shown = 0
  for (let r = 0; r < n; r++) shown += dist.probs[r]
  return (
    <div>
      <div className="Gpt2Explainer-bars" role="list" aria-label="Top 10 next tokens with their probability before and after the cuts">
        {Array.from({ length: n }, (_, r) => {
          const id = dist.order[r]
          const cut = dist.probs[r] === 0
          return (
            <div key={id} role="listitem" className={`Gpt2Explainer-bar${cut ? ' is-cut' : ''}`}>
              <span className="Gpt2Explainer-barlabel mono">{showToken(tok.label(id))}</span>
              <span className="Gpt2Explainer-bartrack">
                <span className="Gpt2Explainer-probbefore" style={{ width: `${(dist.before[r] / max) * 100}%` }} />
                <span className="Gpt2Explainer-probafter" style={{ width: `${(dist.probs[r] / max) * 100}%` }} />
              </span>
              <span className="Gpt2Explainer-probval">{cut ? 'cut' : pct(dist.probs[r])}</span>
            </div>
          )
        })}
      </div>
      <p className="Gpt2Explainer-legend" aria-live="polite">
        Filled: the probability the dice use. Outline: after temperature only, before top-k and top-p.
        {' '}{n0(dist.kept)} of 50,257 tokens survive the cuts; these 10 hold {pct(shown)}.
      </p>
    </div>
  )
}

function About({ info, fromCache, loadMs }: { info: ModelInfo; fromCache: boolean; loadMs: number }) {
  return (
    <details className="Gpt2Explainer-about">
      <summary>
        <span>About this model</span>
        <span className="Gpt2Explainer-stat">GPT-2 small, {n0(info.nParams)} parameters, stored as int8 ({MB(info.totalBytes)}, {fromCache ? `read from your browser’s cache in ${(loadMs / 1000).toFixed(1)} s` : `downloaded in ${(loadMs / 1000).toFixed(1)} s and now cached`})</span>
      </summary>
      <div className="Gpt2Explainer-aboutbody">
        <dl className="Gpt2Explainer-spec">
          <div><dt>Blocks</dt><dd>{info.nLayer}</dd></div>
          <div><dt>Heads per block</dt><dd>{info.nHead}</dd></div>
          <div><dt>Vector size</dt><dd>{info.nEmbd}</dd></div>
          <div><dt>Vocabulary</dt><dd>{n0(info.vocabSize)}</dd></div>
          <div><dt>Context</dt><dd>{n0(info.nCtx)} tokens</dd></div>
        </dl>
        <p>
          Every weight matrix is stored as int8, whole numbers from −127 to 127, with one float32 scale per row. LayerNorm parameters and biases stay float32.
          In float32 the same model would be {MB(info.nParams * 4)}. That makes this a close approximation of GPT-2, not a bit-identical copy.
        </p>
        <p>
          One exception, found by measuring: GPT-2’s final vector is huge in a few dimensions (#496 averages about 500 times a typical dimension, #430 and #36 about 200 times; see the peak in the Scores stage).
          Multiplied by those, the small rounding errors of int8 in the output table changed GPT-2’s top prediction on 3 of 9 test prompts.
          So {info.outlierCols.length} columns of the token table (#{info.outlierCols.join(', #')}) stay float32, the “outlier features” idea of LLM.int8(). With them, the top prediction matched float32 GPT-2 on all 9.
          {' '}<LessonLinks ids={['making-models-cheaper']} />
        </p>
      </div>
    </details>
  )
}
