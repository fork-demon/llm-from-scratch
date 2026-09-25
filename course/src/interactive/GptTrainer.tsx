// Train a real (tiny) GPT in the browser: pick a text and a size, press Train, watch train/validation loss fall,
// read samples as they improve, and look inside every attention head for a prompt you type.
// All maths lives in src/lib/gptTrain.ts; training runs in a Web Worker (src/lib/gptTrain.worker.ts).
import { useEffect, useMemo, useRef, useState } from 'react'
import { Bars, Lab, Slider } from '../components/ui'
import { countParams, createGpt, DEFAULT_TRAIN, encodeChars, inspect, makeCharData, minTextLength, type GptSizes, type LossPoint, type Sample, type TrainOptions } from '../lib/gptTrain'
import { CORPORA } from '../lib/gptCorpora'
import { makeEngine, type Engine, type WorkerReply } from '../lib/gptTrainProtocol'
import './GptTrainer.css'

export type GptTrainerCorpus = 'shakespeare' | 'tickets' | 'sentences'
export type GptTrainerSize = 'tiny' | 'small' | 'bigger'

const SIZES: { id: GptTrainerSize; label: string; sizes: GptSizes; note: string }[] = [
  { id: 'tiny', label: 'Tiny', sizes: { nLayer: 1, nHead: 2, nEmbd: 16, contextLen: 16 }, note: 'fastest, learns least' },
  { id: 'small', label: 'Small', sizes: { nLayer: 2, nHead: 4, nEmbd: 32, contextLen: 32 }, note: 'the default' },
  { id: 'bigger', label: 'Bigger', sizes: { nLayer: 2, nHead: 4, nEmbd: 48, contextLen: 32 }, note: 'about 2× slower per step' },
]
const DEFAULT_PROMPTS: Record<string, string> = { shakespeare: 'To be, or not to be, that is', tickets: 'Paisa Pal: Sorry for the', sentences: 'the cat sat on the mat. a', custom: '' }
const PAUSE_EVERY = 1000 // steps: never burn the learner's battery forever
const CHUNK_MS = 150
const MAX_PASTE = 100_000

const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : '–')
const showChar = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '↵' : c === '\t' ? '⇥' : c)

interface Run { step: number; seconds: number; lastLoss: number; history: LossPoint[]; samples: Sample[] }
const EMPTY_RUN: Run = { step: 0, seconds: 0, lastLoss: NaN, history: [], samples: [] }

export function GptTrainer({ corpus = 'shakespeare', size = 'small' }: { corpus?: GptTrainerCorpus; size?: GptTrainerSize }) {
  const [corpusId, setCorpusId] = useState<string>(corpus)
  const [sizeId, setSizeId] = useState<GptTrainerSize>(size)
  const [pasted, setPasted] = useState('')
  const [customText, setCustomText] = useState('')
  const [running, setRunning] = useState(false)
  const [run, setRun] = useState<Run>(EMPTY_RUN)
  const [error, setError] = useState<string | null>(null)
  const [temperature, setTemperature] = useState(0.8)
  const [prompt, setPrompt] = useState(DEFAULT_PROMPTS[corpus])
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [query, setQuery] = useState<number | null>(null)
  const [continuation, setContinuation] = useState<string | null>(null)
  const [paramsVersion, setParamsVersion] = useState(0)

  const preset = SIZES.find((s) => s.id === sizeId)!
  const text = corpusId === 'custom' ? customText : CORPORA.find((c) => c.id === corpusId)!.text
  const data = useMemo(() => (text ? makeCharData(text) : null), [text])
  const opts: TrainOptions = useMemo(() => ({ ...DEFAULT_TRAIN, sizes: preset.sizes, temperature }), [preset]) // eslint-disable-line react-hooks/exhaustive-deps
  const cfg = data ? { ...preset.sizes, vocabSize: data.chars.length } : null
  // a copy of the weights on the page, refreshed after every chunk, for the attention maps
  const model = useMemo(() => (cfg ? createGpt(cfg, opts.seed) : null), [data, preset]) // eslint-disable-line react-hooks/exhaustive-deps

  const engineRef = useRef<Engine | null>(null)
  const genRef = useRef(0)
  const live = useRef({ running, temperature, model })
  live.current = { running, temperature, model }

  const onReply = (r: WorkerReply) => {
    if (r.type === 'error') { setError(r.message); setRunning(false); return }
    if (r.gen !== genRef.current) return // a reply from a run that was reset
    if (r.type === 'generated') { setContinuation(r.text); return }
    const m = live.current.model
    if (m && m.params.length === r.params.length) { m.params.set(r.params); setParamsVersion((v) => v + 1) }
    setRun({ step: r.step, seconds: r.seconds, lastLoss: r.lastLoss, history: r.history, samples: r.samples })
    if (!Number.isFinite(r.lastLoss)) { setRunning(false); return }
    if (live.current.running) {
      if (r.step > 0 && r.step % PAUSE_EVERY === 0) setRunning(false)
      else engineRef.current?.send({ type: 'train', gen: genRef.current, budgetMs: CHUNK_MS, temperature: live.current.temperature })
    }
  }
  const replyRef = useRef(onReply)
  replyRef.current = onReply

  const init = () => {
    genRef.current++
    setRunning(false)
    setRun(EMPTY_RUN)
    setContinuation(null)
    setError(null)
    if (text) engineRef.current?.send({ type: 'init', gen: genRef.current, text, opts })
  }

  // one engine (Web Worker) per mounted trainer; terminated on unmount
  useEffect(() => {
    engineRef.current = makeEngine((r) => replyRef.current(r))
    init()
    return () => { engineRef.current?.dispose(); engineRef.current = null }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
  // a new text or size is a new model: start again from random weights
  const firstRender = useRef(true)
  useEffect(() => {
    if (firstRender.current) { firstRender.current = false; return }
    init()
  }, [text, preset]) // eslint-disable-line react-hooks/exhaustive-deps

  const start = () => {
    if (!run.history.length) return
    setRunning(true)
    live.current.running = true
    engineRef.current?.send({ type: 'train', gen: genRef.current, budgetMs: CHUNK_MS, temperature })
  }

  const pickCorpus = (id: string) => {
    setCorpusId(id)
    setPrompt(DEFAULT_PROMPTS[id] ?? '')
    setQuery(null)
  }
  const pickSize = (id: GptTrainerSize) => { setSizeId(id); setLayer(0); setHead(0) }
  const minLen = minTextLength(preset.sizes.contextLen)
  const usePasted = () => {
    const t = pasted.slice(0, MAX_PASTE)
    if (t.length < minLen) { setError(`Paste at least ${minLen.toLocaleString()} characters (you have ${t.length.toLocaleString()}).`); return }
    setCustomText(t)
    setPrompt(t.slice(0, 20))
  }

  /* ---------- derived numbers ---------- */
  const V = data?.chars.length ?? 0
  const nParams = cfg ? countParams(cfg) : 0
  const base = V ? Math.log(V) : 0
  const hist = run.history
  const first = hist[0]
  const last = hist[hist.length - 1]
  const best = hist.length ? hist.reduce((b, h) => (h.val < b.val ? h : b), hist[0]) : undefined
  const overfitting = !!(last && best && last.val > best.val + 0.1 && last.train < last.val - 0.2)
  const stepsPerSec = run.seconds > 0 ? run.step / run.seconds : 0

  /* ---------- attention for the prompt ---------- */
  const promptIds = data ? encodeChars(data, prompt).slice(-preset.sizes.contextLen) : []
  const dropped = data ? [...prompt].length - encodeChars(data, prompt).length : 0
  const cropped = data ? encodeChars(data, prompt).length > preset.sizes.contextLen : false
  const view = useMemo(() => (model && promptIds.length ? inspect(model, promptIds, temperature) : null), [model, paramsVersion, promptIds.join(','), temperature]) // eslint-disable-line react-hooks/exhaustive-deps
  const T = promptIds.length
  const L = Math.min(layer, preset.sizes.nLayer - 1)
  const H = Math.min(head, preset.sizes.nHead - 1)
  const q = query !== null && query < T ? query : T - 1
  const map = view?.attn[L]?.[H]
  const row = map?.[q] ?? []
  const tokChars = promptIds.map((i) => data!.chars[i])
  const topLooks = row.map((p, j) => ({ p, j })).filter((x) => x.j <= q).sort((a, b) => b.p - a.p).slice(0, 3)
  const nextTop = view ? view.probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 5) : []

  // chart geometry
  const W = 640, CH = 250, PL = 44, PR = 12, PT = 14, PB = 34
  const xMax = Math.max(200, run.step)
  const yMax = Math.max(1, Math.ceil(Math.max(base * 1.08, ...hist.flatMap((h) => [h.train, h.val]).filter(Number.isFinite))))
  const px = (s: number) => PL + (s / xMax) * (W - PL - PR)
  const py = (v: number) => PT + (1 - Math.min(v, yMax) / yMax) * (CH - PT - PB)
  const line = (key: 'train' | 'val') => hist.map((h, i) => `${i ? 'L' : 'M'}${px(h.step).toFixed(1)},${py(h[key]).toFixed(1)}`).join(' ')
  const yTicks = Array.from({ length: yMax + 1 }, (_, i) => i)
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round((f * xMax) / 50) * 50)

  // heatmap geometry
  const cell = 12, HL = 18, HT = 18
  const hw = HL + T * cell + 2

  const shownSamples = run.samples.length <= 4 ? run.samples : [run.samples[0], ...run.samples.slice(-3)]
  const statusText = running ? 'training…' : run.step > 0 && run.step % PAUSE_EVERY === 0 ? `paused after ${run.step.toLocaleString()} steps; Continue for more` : run.step > 0 ? 'paused' : ''

  return (
    <Lab
      title="Train a real GPT in your browser"
      goal={<>Press <b>Train</b>. Within half a minute the loss falls and the samples turn from noise into words. Then type a prompt below and click its characters to see where each attention head looks.</>}
    >
      <p className="lab-note">
        <b>This is a real Transformer</b>, not an animation: the same architecture as <code>tiny_gpt.py</code> (token + position embeddings, {preset.sizes.nLayer === 1 ? 'one block' : `${preset.sizes.nLayer} blocks`} of pre-LayerNorm causal multi-head attention and a GELU MLP, a final LayerNorm, and an output head tied to the embedding table). It has <span className="GptTrainer-badge">{nParams.toLocaleString()} parameters</span>, starts from random numbers, and learns by backpropagation and AdamW, all in your browser{engineRef.current && !engineRef.current.inWorker ? '' : ' (in a background thread)'}. Nothing is sent to a server. It reads characters, not words, and it is small: GPT-2 small has 124 million parameters, about {nParams ? Math.round(124e6 / nParams).toLocaleString() : '–'} times more.
      </p>

      <fieldset className="GptTrainer-group">
        <legend>Training text (changing it starts over)</legend>
        <div className="GptTrainer-pick">
          {CORPORA.map((c) => <button key={c.id} className="btn small" aria-pressed={corpusId === c.id} onClick={() => pickCorpus(c.id)}>{c.label}</button>)}
          <button className="btn small" aria-pressed={corpusId === 'custom'} onClick={() => pickCorpus('custom')}>Paste your own</button>
        </div>
        {corpusId !== 'custom' ? (
          <p className="lab-note" style={{ margin: 0 }}>{CORPORA.find((c) => c.id === corpusId)!.blurb} {text.length.toLocaleString()} characters, {V} different ones.</p>
        ) : (
          <div>
            <label htmlFor="GptTrainer-paste" className="lab-note" style={{ display: 'block', marginBottom: 4 }}>Paste any text of at least {minLen.toLocaleString()} characters (up to {MAX_PASTE.toLocaleString()} are used). Try song lyrics, code, or your own writing.</label>
            <textarea id="GptTrainer-paste" className="input GptTrainer-paste" value={pasted} onChange={(e) => setPasted(e.target.value)} rows={5} />
            <div className="btn-row" style={{ marginTop: 6 }}>
              <button className="btn small" onClick={usePasted}>Train on this text</button>
              <span className="muted" style={{ fontSize: 13.5 }}>{customText ? `Using ${customText.length.toLocaleString()} characters, ${V} different ones.` : 'No text yet.'}</span>
            </div>
          </div>
        )}
      </fieldset>

      <fieldset className="GptTrainer-group">
        <legend>Model size (changing it starts over)</legend>
        <div className="GptTrainer-pick">
          {SIZES.map((s) => (
            <button key={s.id} className="btn small" aria-pressed={sizeId === s.id} onClick={() => pickSize(s.id)}>
              {s.label}{V ? ` · ${countParams({ ...s.sizes, vocabSize: V }).toLocaleString()} params` : ''}
            </button>
          ))}
        </div>
        <p className="lab-note" style={{ margin: 0 }}>
          {preset.sizes.nLayer} layer{preset.sizes.nLayer > 1 ? 's' : ''} × {preset.sizes.nHead} heads, {preset.sizes.nEmbd}-number embeddings, a context window of {preset.sizes.contextLen} characters ({preset.note}). Batch of {opts.batchSize} windows per step, AdamW learning rate {opts.optim.lr}, gradient clipping at {opts.optim.clip}.
        </p>
      </fieldset>

      <div className="btn-row" style={{ marginBottom: 12 }}>
        <button className="btn small primary" disabled={running || !hist.length} onClick={start}>{run.step === 0 ? 'Train' : 'Continue'}</button>
        <button className="btn small" disabled={!running} onClick={() => setRunning(false)}>Pause</button>
        <button className="btn small" disabled={!text} onClick={init}>Reset</button>
        <span className="muted" style={{ fontSize: 13.5 }}>{!hist.length && text ? 'setting up…' : statusText}</span>
      </div>
      {error && <p className="lab-note" role="alert" style={{ borderLeft: '3px solid var(--bad)', paddingLeft: 12 }}>{error}</p>}

      <div className="readout" aria-live="off">
        <span>step <b>{run.step.toLocaleString()}</b></span>
        <span>time <b>{run.seconds.toFixed(0)} s</b></span>
        <span>speed <b>{stepsPerSec ? stepsPerSec.toFixed(1) : '–'}</b> steps/s</span>
        <span>train loss <b>{f2(last?.train ?? NaN)}</b></span>
        <span>val loss <b>{f2(last?.val ?? NaN)}</b></span>
        <span>guessing = ln({V}) = <b>{base.toFixed(2)}</b></span>
      </div>

      <svg viewBox={`0 0 ${W} ${CH}`} style={{ width: '100%', display: 'block', marginTop: 8 }} role="img"
        aria-label={last ? `Loss curves after ${run.step} steps: train loss ${f2(last.train)}, validation loss ${f2(last.val)}, starting from ${f2(first.train)}. Guessing baseline ${base.toFixed(2)}.` : 'Loss curves: no training yet.'}>
        {yTicks.map((v) => (
          <g key={v}>
            <line className="gridline" x1={PL} x2={W - PR} y1={py(v)} y2={py(v)} />
            <text x={PL - 6} y={py(v) + 4} textAnchor="end" fontSize={11} style={{ fill: 'var(--ink-3)' }}>{v}</text>
          </g>
        ))}
        {xTicks.map((s, i) => <text key={i} x={px(Math.min(s, xMax))} y={CH - 16} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)' }}>{s}</text>)}
        <text x={(PL + W - PR) / 2} y={CH - 2} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)' }}>training step</text>
        <text x={12} y={(PT + CH - PB) / 2} textAnchor="middle" fontSize={11} transform={`rotate(-90 12 ${(PT + CH - PB) / 2})`} style={{ fill: 'var(--ink-3)' }}>loss</text>
        <line className="axis" x1={PL} x2={PL} y1={PT} y2={CH - PB} />
        <line className="axis" x1={PL} x2={W - PR} y1={CH - PB} y2={CH - PB} />
        {base > 0 && <>
          <line x1={PL} x2={W - PR} y1={py(base)} y2={py(base)} stroke="var(--ink-3)" strokeWidth={1.5} strokeDasharray="2 4" />
          <text x={W - PR - 4} y={py(base) - 5} textAnchor="end" fontSize={11} style={{ fill: 'var(--ink-3)' }}>pure guessing: ln({V}) = {base.toFixed(2)}</text>
        </>}
        {hist.length > 0 && <>
          <path d={line('val')} fill="none" stroke="var(--ink)" strokeWidth={2} strokeDasharray="7 4" />
          <path d={line('train')} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
        </>}
        {overfitting && best && (
          <g>
            <circle cx={px(best.step)} cy={py(best.val)} r={4.5} fill="var(--card)" stroke="var(--ink)" strokeWidth={2} />
            <text x={px(best.step)} y={py(best.val) + 18} textAnchor="middle" fontSize={11}>best val {f2(best.val)}</text>
          </g>
        )}
      </svg>
      <div className="readout" style={{ border: 0, background: 'transparent', padding: '0 0 8px' }}>
        <span><svg width="34" height="8" aria-hidden><line x1="0" x2="34" y1="4" y2="4" stroke="var(--accent)" strokeWidth="2.2" /></svg> train loss (solid)</span>
        <span><svg width="34" height="8" aria-hidden><line x1="0" x2="34" y1="4" y2="4" stroke="var(--ink)" strokeWidth="2" strokeDasharray="7 4" /></svg> validation loss on text it never trains on (dashed)</span>
      </div>
      <div aria-live="polite">
        {overfitting && best && (
          <p className="lab-note" style={{ borderLeft: '3px solid var(--accent)', paddingLeft: 12 }}>
            <b>Overfitting.</b> Train loss keeps falling ({f2(last!.train)}) but validation loss bottomed out at step {best.step} ({f2(best.val)}) and is now {f2(last!.val)}. With only {text.length.toLocaleString()} characters, the model has started memorising its training text.
          </p>
        )}
        {!Number.isFinite(run.lastLoss) && run.step > 0 && <p className="lab-note" role="alert">The loss became NaN (not a number), so training stopped. Press Reset.</p>}
      </div>

      <h4 style={{ fontSize: 16, margin: '14px 0 6px' }}>What the model writes (a fresh sample every {opts.sampleEvery} steps)</h4>
      <Slider label="Sampling temperature" value={temperature} min={0.1} max={1.5} step={0.1} onChange={setTemperature} format={(v) => v.toFixed(1)} />
      <div className="table-scroll">
        <table className="plain mono" style={{ fontSize: 13.5, margin: 0 }}>
          <thead><tr><th style={{ width: 60 }}>step</th><th>{opts.sampleLen} characters generated by the weights at that step</th></tr></thead>
          <tbody>
            {shownSamples.map((s) => <tr key={s.step}><td>{s.step}</td><td className="GptTrainer-sample">{s.text}</td></tr>)}
          </tbody>
        </table>
      </div>

      <h4 style={{ fontSize: 16, margin: '18px 0 6px' }}>Look inside: where does each head look?</h4>
      <label htmlFor="GptTrainer-prompt" className="lab-note" style={{ display: 'block', marginBottom: 4 }}>Type a prompt (the last {preset.sizes.contextLen} characters are used):</label>
      <input id="GptTrainer-prompt" className="input" style={{ width: '100%', fontFamily: 'var(--mono)' }} value={prompt} onChange={(e) => { setPrompt(e.target.value); setQuery(null) }} />
      {(dropped > 0 || cropped) && (
        <p className="lab-note" style={{ margin: '4px 0 0' }}>
          {dropped > 0 && `${dropped} character${dropped > 1 ? 's are' : ' is'} not in this text's vocabulary and ${dropped > 1 ? 'were' : 'was'} dropped. `}
          {cropped && `Only the last ${preset.sizes.contextLen} characters fit in the context window.`}
        </p>
      )}

      {T > 0 && view && map && (
        <>
          <div className="btn-row" style={{ margin: '10px 0 4px' }} role="group" aria-label="Choose a layer">
            <span className="muted" style={{ fontSize: 13.5 }}>Layer</span>
            {Array.from({ length: preset.sizes.nLayer }, (_, l) => <button key={l} className="btn small" aria-pressed={L === l} onClick={() => setLayer(l)}>{l + 1}</button>)}
            <span className="muted" style={{ fontSize: 13.5, marginLeft: 8 }}>Head</span>
            {Array.from({ length: preset.sizes.nHead }, (_, h) => <button key={h} className="btn small" aria-pressed={H === h} onClick={() => setHead(h)}>{h + 1}</button>)}
          </div>

          <p className="lab-note" style={{ margin: '6px 0 0' }}>Click a character to see where it looks (layer {L + 1}, head {H + 1}). The shading and the percentage under each earlier character show how much attention it gets. Later characters are dashed: the causal mask hides the future.</p>
          <div className="GptTrainer-tokens" role="group" aria-label="Prompt characters">
            {tokChars.map((c, j) => {
              const p = j <= q ? row[j] : 0
              return (
                <button key={j} className={`GptTrainer-tok${j > q ? ' future' : ''}`} aria-pressed={j === q} onClick={() => setQuery(j)}
                  aria-label={`${c === ' ' ? 'space' : c === '\n' ? 'new line' : c} at position ${j + 1}${j <= q ? `, gets ${(p * 100).toFixed(0)}% of the attention` : ', in the future'}`}>
                  <span className="GptTrainer-heat" style={{ opacity: j <= q ? Math.min(0.85, p) : 0 }} />
                  <span>{showChar(c)}</span>
                  <small>{j <= q ? `${(p * 100).toFixed(0)}%` : '·'}</small>
                </button>
              )
            })}
          </div>
          <p className="readout" aria-live="polite" style={{ margin: '4px 0 12px' }}>
            <span>“{showChar(tokChars[q])}” (position {q + 1}) looks mostly at: {topLooks.map((x) => `“${showChar(tokChars[x.j])}” pos ${x.j + 1} (${(x.p * 100).toFixed(0)}%)`).join(', ')}</span>
          </p>

          <div className="GptTrainer-maps">
            <div>
              <svg className="GptTrainer-heatmap" viewBox={`0 0 ${hw} ${HT + T * cell + 2}`} role="img"
                aria-label={`Attention map of layer ${L + 1}, head ${H + 1} for ${T} characters. Row = the character that is looking, column = the character it looks at. Darker means more attention. Everything above the diagonal is zero because of the causal mask.`}>
                {tokChars.map((c, j) => (
                  <g key={j}>
                    <text x={HL + j * cell + cell / 2} y={HT - 5} textAnchor="middle" fontSize={9} style={{ fill: 'var(--ink-2)', fontFamily: 'var(--mono)' }}>{showChar(c)}</text>
                    <text x={HL - 4} y={HT + j * cell + cell / 2 + 3} textAnchor="end" fontSize={9} style={{ fill: j === q ? 'var(--ink)' : 'var(--ink-2)', fontFamily: 'var(--mono)', fontWeight: j === q ? 700 : 400 }}>{showChar(c)}</text>
                  </g>
                ))}
                {map.map((r, t) => r.map((p, t2) => (
                  <g key={`${t}-${t2}`}>
                    <rect className="grid-cell" x={HL + t2 * cell} y={HT + t * cell} width={cell} height={cell} />
                    {t2 <= t && <rect className="cell" x={HL + t2 * cell + 0.5} y={HT + t * cell + 0.5} width={cell - 1} height={cell - 1} fill="var(--accent)" fillOpacity={Math.min(1, p)} onClick={() => setQuery(t)} />}
                  </g>
                )))}
                <rect x={HL} y={HT + q * cell} width={T * cell} height={cell} fill="none" stroke="var(--ink)" strokeWidth={1.5} />
              </svg>
              <p className="lab-note" style={{ margin: '4px 0 0', fontSize: 13 }}>Rows: the character doing the looking. Columns: the character it looks at. Darker = more attention; each row adds up to 100%.</p>
            </div>
            <div>
              <p className="lab-note" style={{ margin: '0 0 6px' }}>What the whole model predicts after “{showChar(tokChars[T - 1])}” (temperature {temperature.toFixed(1)}):</p>
              <Bars items={nextTop.map((x) => ({ label: `“${showChar(data!.chars[x.i])}”`, value: x.p }))} max={1} />
              <div className="btn-row" style={{ marginTop: 10 }}>
                <button className="btn small" disabled={!hist.length} onClick={() => engineRef.current?.send({ type: 'generate', gen: genRef.current, id: Date.now(), prompt, len: 150, temperature })}>Continue this prompt</button>
              </div>
              {continuation && <p className="mono GptTrainer-sample" aria-live="polite" style={{ fontSize: 13.5, background: 'var(--paper-2)', padding: '8px 10px', borderRadius: 6, marginTop: 8 }}>{continuation}</p>}
            </div>
          </div>
          <p className="lab-note" style={{ marginTop: 10 }}>
            {run.step === 0
              ? 'At step 0 the weights are random, so every head spreads its attention almost evenly over the past. Train for a minute and come back: heads start to specialise, for example looking at the previous character, or back to the start of the word.'
              : `These are the weights after ${run.step.toLocaleString()} steps; the map updates live while training. Compare heads: they usually learn different jobs.`}
          </p>
        </>
      )}
    </Lab>
  )
}
