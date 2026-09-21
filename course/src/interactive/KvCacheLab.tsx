// Step through generation token by token: naive (recompute everything) next to
// cached (compute one row, read the rest). Counts, a scaling chart and a memory calculator.
// All maths lives in src/lib/kvcache.ts.
import { useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { fmt, type Mat } from '../lib/math'
import {
  KV_PRESETS, emptyCache, formatBytes, forwardFull, forwardStep, kvBytesPerToken, kvCacheBytes,
  kvRowsForToken, kvRowsTotal, makeToyParams, maxAbsDiff, type KvConfig,
} from '../lib/kvcache'

const PROMPT = ['The', 'cat', 'sat']
const SCRIPT = ['on', 'the', 'mat', 'and', 'purred', '.']
const WORDS = [...PROMPT, ...SCRIPT]
const P = PROMPT.length
const GPU_GB = 24

const row4 = (r: number[]) => `[${r.slice(0, 4).map((x) => fmt(x, 2)).join(', ')}, …]`

export function KvCacheLab() {
  const [g, setG] = useState(1) // we are producing new token number g
  const [chartN, setChartN] = useState(200)
  const [cfg, setCfg] = useState<KvConfig>(KV_PRESETS[2])
  const [presetId, setPresetId] = useState('7b')
  const [tokens, setTokens] = useState(4096)
  const [users, setUsers] = useState(1)

  // A real (random-weight) 2-layer Transformer. For every step: K rows recomputed from scratch, and K rows held in a cache.
  const sim = useMemo(() => {
    const vocab = Array.from(new Set(WORDS))
    const ids = WORDS.map((w) => vocab.indexOf(w))
    const p = makeToyParams({ vocab: vocab.length, dim: 16, heads: 4, layers: 2, maxLen: WORDS.length }, 9)
    const cache = emptyCache(p)
    const steps: { naiveK: Mat; cachedK: Mat; diff: number }[] = []
    for (let step = 1; step <= SCRIPT.length; step++) {
      const len = P + step - 1
      for (let i = cache[0].K.length; i < len; i++) forwardStep(p, ids[i], i, cache) // only the rows that are new
      const naiveK = forwardFull(p, ids.slice(0, len)).kv[0].K // everything, again
      const cachedK = cache[0].K.map((r) => r.slice())
      steps.push({ naiveK, cachedK, diff: maxAbsDiff(naiveK, cachedK) })
    }
    return steps
  }, [])

  const len = P + g - 1 // tokens the model must have keys and values for, to produce token g
  const cur = sim[g - 1]
  const isNew = (i: number) => (g === 1 ? true : i === len - 1)
  const naiveNow = kvRowsForToken(P, g, false)
  const cachedNow = kvRowsForToken(P, g, true)
  const naiveTotal = kvRowsTotal(P, g, false)
  const cachedTotal = kvRowsTotal(P, g, true)

  // scaling chart
  const W = 640, H = 220, L = 56, R = 12, T = 14, B = 32
  const nMax = kvRowsTotal(P, chartN, false)
  const pts = (cached: boolean) => Array.from({ length: 41 }, (_, i) => {
    const n = Math.max(1, Math.round((i / 40) * chartN))
    return `${i ? 'L' : 'M'}${(L + (n / chartN) * (W - L - R)).toFixed(1)},${(T + (1 - kvRowsTotal(P, n, cached) / nMax) * (H - T - B)).toFixed(1)}`
  }).join(' ')
  const cMax = kvRowsTotal(P, chartN, true)

  const perToken = kvBytesPerToken(cfg)
  const totalBytes = kvCacheBytes(cfg, tokens) * users
  const gpuShare = totalBytes / (GPU_GB * 1024 ** 3)
  const setField = (k: keyof KvConfig, v: number) => { setCfg({ ...cfg, [k]: Math.max(1, Math.round(v) || 1) }); setPresetId('') }

  const tokenRow = (cached: boolean) => (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, margin: '8px 0' }}>
      {WORDS.slice(0, len).map((w, i) => {
        const computed = !cached || isNew(i)
        return (
          <span key={i} className="token" style={{ textAlign: 'center', margin: 0, borderColor: computed ? 'var(--accent)' : undefined, borderWidth: computed ? 2 : 1, borderStyle: computed ? 'solid' : 'dashed', opacity: computed ? 1 : 0.75 }}>
            {w}
            <br />
            <small style={{ fontSize: 10.5, color: computed ? 'var(--accent-ink)' : 'var(--ink-3)' }}>{computed ? 'compute k,v' : 'from cache'}</small>
          </span>
        )
      })}
      <span className="token" style={{ margin: 0, borderStyle: 'dotted', textAlign: 'center' }}>{WORDS[len]}<br /><small style={{ fontSize: 10.5 }} className="muted">← next</small></span>
    </div>
  )

  return (
    <Lab
      title="The KV cache, one token at a time"
      goal={<>Click <b>Generate next token</b> and compare the two columns. Solid outline = <span className="k">key</span> and <span className="v">value</span> vectors computed in this step. Dashed = read from the cache. Watch the two counters drift apart.</>}
    >
      <p className="lab-note">
        <b>Honest note:</b> the text is scripted (the prompt is “The cat sat”, the continuation is fixed). The numbers are real: a 2-layer Transformer with random weights runs underneath, once the naive way and once with a cache, and we compare their key vectors.
      </p>

      <div className="btn-row" style={{ marginBottom: 8 }}>
        <button className="btn small" disabled={g === 1} onClick={() => setG(g - 1)}>Back</button>
        <button className="btn small primary" disabled={g === SCRIPT.length} onClick={() => setG(g + 1)}>Generate next token</button>
        <button className="btn small" disabled={g === 1} onClick={() => setG(1)}>Restart</button>
        <span className="muted" style={{ fontSize: 13.5 }}>producing new token {g} of {SCRIPT.length}{g === 1 ? ' (the prompt is processed now: prefill)' : ' (decode)'}</span>
      </div>

      <div className="grid-2">
        <div className="card">
          <h4 style={{ fontSize: 16, marginBottom: 2 }}>Naive: run the whole sequence again</h4>
          {tokenRow(false)}
          <div className="readout"><span>this step: <b>{naiveNow}</b> rows</span><span>so far: <b>{naiveTotal}</b></span></div>
        </div>
        <div className="card">
          <h4 style={{ fontSize: 16, marginBottom: 2 }}>With a KV cache: run only what is new</h4>
          {tokenRow(true)}
          <div className="readout"><span>this step: <b>{cachedNow}</b> {cachedNow === 1 ? 'row' : 'rows'}</span><span>so far: <b>{cachedTotal}</b></span></div>
        </div>
      </div>
      <p className="lab-note" aria-live="polite">
        “Rows” = key rows computed per layer (the same number of value rows comes with them). Cache now holds <b>{len}</b> tokens. Naive has done <b>{(naiveTotal / cachedTotal).toFixed(1)}×</b> the work of cached so far.
      </p>

      <div className="table-scroll">
        <table className="plain mono" style={{ fontSize: 13 }}>
          <thead><tr><th>token</th><th><span className="k">key</span> row, recomputed now (naive)</th><th><span className="k">key</span> row in the cache</th><th>cached row was computed</th></tr></thead>
          <tbody>
            {WORDS.slice(0, len).map((w, i) => (
              <tr key={i}>
                <td>{w}</td>
                <td className="k">{row4(cur.naiveK[i])}</td>
                <td className="k">{row4(cur.cachedK[i])}</td>
                <td>{isNew(i) ? <b>this step</b> : i < P ? 'at prefill' : `${len - 1 - i} step${len - 1 - i === 1 ? '' : 's'} ago`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="readout" aria-live="polite">
        <span>largest difference between recomputed and cached keys (layer 1, all 16 numbers): <b>{cur.diff === 0 ? '0 (bit for bit identical)' : cur.diff.toExponential(1)}</b></span>
      </div>

      <h4 style={{ fontSize: 16, margin: '20px 0 4px' }}>How the work grows</h4>
      <div className="controls"><Slider label="new tokens to generate" value={chartN} min={10} max={1000} step={10} onChange={setChartN} /></div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img" aria-label={`Key and value rows computed to generate ${chartN} tokens: naive ${nMax.toLocaleString()}, cached ${cMax.toLocaleString()}.`}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line className="gridline" x1={L} x2={W - R} y1={T + (1 - f) * (H - T - B)} y2={T + (1 - f) * (H - T - B)} />
            <text x={L - 6} y={T + (1 - f) * (H - T - B) + 4} textAnchor="end" fontSize={11} style={{ fill: 'var(--ink-3)' }}>{Math.round(f * nMax).toLocaleString()}</text>
          </g>
        ))}
        <line className="axis" x1={L} x2={L} y1={T} y2={H - B} />
        <line className="axis" x1={L} x2={W - R} y1={H - B} y2={H - B} />
        {[0, 0.5, 1].map((f) => <text key={f} x={L + f * (W - L - R)} y={H - 14} textAnchor={f === 1 ? 'end' : f === 0 ? 'start' : 'middle'} fontSize={11} style={{ fill: 'var(--ink-3)' }}>{Math.round(f * chartN)}</text>)}
        <text x={(L + W - R) / 2} y={H - 1} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)' }}>new tokens generated</text>
        <path d={pts(false)} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
        <path d={pts(true)} fill="none" stroke="var(--ink)" strokeWidth={2} strokeDasharray="7 4" />
        <text x={W - R - 6} y={T + 14} textAnchor="end" fontSize={12} style={{ fill: 'var(--accent-ink)' }}>naive: {nMax.toLocaleString()} rows (curves upward, like T²)</text>
        <text x={W - R - 6} y={H - B - 8} textAnchor="end" fontSize={12}>cached: {cMax.toLocaleString()} rows (a straight line, like T)</text>
      </svg>
      <p className="lab-note">At {chartN} tokens the naive loop computes <b>{(nMax / cMax).toFixed(0)}×</b> more key/value rows. (Rows are only part of the cost: see the deep dive on what the cache does not save.)</p>

      <h4 style={{ fontSize: 16, margin: '20px 0 4px' }}>What the speed costs: memory</h4>
      <p className="lab-note mono" style={{ fontSize: 13.5 }}>cache bytes = 2 (K and V) × layers × kv_heads × head_dim × tokens × bytes_per_number</p>
      <div className="steps" role="group" aria-label="Model presets">
        {KV_PRESETS.map((p) => (
          <button key={p.id} className="step-btn" aria-pressed={presetId === p.id} onClick={() => { setCfg(p); setPresetId(p.id); setTokens(p.context) }}>{p.name}</button>
        ))}
      </div>
      <div className="controls">
        {([['layers', 'layers'], ['kvHeads', 'kv_heads'], ['headDim', 'head_dim']] as const).map(([k, label]) => (
          <label key={k} style={{ fontSize: 14 }}>{label}<input className="input" type="number" min={1} value={cfg[k]} onChange={(e) => setField(k, Number(e.target.value))} /></label>
        ))}
        <label style={{ fontSize: 14 }}>bytes per number
          <select className="input" value={cfg.bytesPerNumber} onChange={(e) => setField('bytesPerNumber', Number(e.target.value))}>
            <option value={4}>4 (32-bit float)</option><option value={2}>2 (16-bit float)</option><option value={1}>1 (8-bit)</option>
          </select>
        </label>
        <label style={{ fontSize: 14 }}>tokens in the conversation<input className="input" type="number" min={1} value={tokens} onChange={(e) => setTokens(Math.max(1, Math.round(Number(e.target.value)) || 1))} /></label>
        <label style={{ fontSize: 14 }}>conversations served at once<input className="input" type="number" min={1} value={users} onChange={(e) => setUsers(Math.max(1, Math.round(Number(e.target.value)) || 1))} /></label>
      </div>
      <div className="readout" aria-live="polite">
        <span>per token: <b>{formatBytes(perToken)}</b></span>
        <span>2 × {cfg.layers} × {cfg.kvHeads} × {cfg.headDim} × {tokens.toLocaleString()} × {cfg.bytesPerNumber}{users > 1 ? ` × ${users}` : ''} = <b>{formatBytes(totalBytes)}</b></span>
      </div>
      <div style={{ margin: '10px 0 4px' }} role="img" aria-label={`The cache would fill ${(gpuShare * 100).toFixed(1)} percent of a ${GPU_GB} GB GPU`}>
        <div className="bar-track" style={{ height: 18 }}><div className="bar-fill" style={{ width: `${Math.min(100, gpuShare * 100)}%`, background: gpuShare > 1 ? 'var(--bad)' : undefined }} /></div>
      </div>
      <p className="lab-note">
        {gpuShare > 1
          ? <><b>Does not fit:</b> {(gpuShare * 100).toFixed(0)}% of a {GPU_GB} GB GPU, before counting the model’s own weights.</>
          : <>{gpuShare < 0.001 ? 'Less than 0.1' : (gpuShare * 100).toFixed(1)}% of a {GPU_GB} GB GPU, before counting the model’s own weights.</>}
        {' '}{presetId && KV_PRESETS.find((p) => p.id === presetId)?.note}
      </p>
    </Lab>
  )
}
