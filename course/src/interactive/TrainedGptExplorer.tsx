// Look inside a REAL trained tiny GPT, in the browser: generation with top-5 alternatives,
// every attention head, a logit lens, and heads that look like previous-token or induction heads.
// Weights: public/models/tiny-gpt.{json,bin}, written by phase3-transformers/export_checkpoint.py.
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Bars, Lab, Slider } from '../components/ui'
import {
  encode, forward, generateStep, headScoresAsync, INDUCTION_THRESHOLD, labelHead, loadTrainedGpt, PREV_THRESHOLD, showTok,
  softmaxRow, topN, type GenStep, type HeadLabel, type HeadScores, type TrainedGpt,
} from '../lib/trainedGpt'
import { makeRng } from '../lib/rng'
import './TrainedGptExplorer.css'

/* ---------- one shared download per page ---------- */
let modelPromise: Promise<TrainedGpt> | null = null
const getModel = () => {
  if (!modelPromise) {
    modelPromise = loadTrainedGpt(`${import.meta.env.BASE_URL}models/`)
    modelPromise.catch(() => { modelPromise = null }) // allow a retry
  }
  return modelPromise
}

const PRESETS: { label: string; text: string }[] = [
  { label: 'ROMEO:', text: 'ROMEO:\n' },
  { label: 'First Citizen', text: 'First Citizen:\nWe are accounted poor citizens, the patricians good.\n' },
  { label: 'KING RICHAR…', text: 'KING RICHAR' },
  { label: 'repeated nonsense', text: 'Zqvex blorp tiwh. Zqvex blorp tiwh. Zqvex bl' },
]

const fmtPct = (p: number) => (p >= 0.995 ? '100%' : p < 0.001 ? '<0.1%' : `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`)
const tokName = (ch: string) => (ch === ' ' ? 'space' : ch === '\n' ? 'newline' : `“${ch}”`)

/** Track the colour theme so canvases can redraw with the right colours. */
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

/** A T x T attention pattern drawn on a canvas (cheap enough for 16 thumbnails of 64 x 64). */
function AttnCanvas({ att, size, themeKey, label, focusRow }: { att: Float32Array[]; size: number; themeKey: number; label: string; focusRow?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    const ctx = c?.getContext?.('2d')
    if (!c || !ctx) return
    const T = att.length
    const css = getComputedStyle(c)
    const accent = css.getPropertyValue('--accent').trim() || '#0f9d63'
    const paper = css.getPropertyValue('--paper-2').trim() || '#f5f5fa'
    const px = Math.max(1, Math.floor(size / T))
    c.width = px * T
    c.height = px * T
    ctx.fillStyle = paper
    ctx.fillRect(0, 0, c.width, c.height)
    ctx.fillStyle = accent
    for (let t = 0; t < T; t++) for (let s = 0; s <= t; s++) {
      const w = att[t][s]
      if (w < 0.01) continue
      ctx.globalAlpha = Math.min(1, w)
      ctx.fillRect(s * px, t * px, px, px)
    }
    ctx.globalAlpha = 1
    if (focusRow !== undefined && focusRow < T) {
      ctx.strokeStyle = css.getPropertyValue('--ink').trim() || '#000'
      ctx.lineWidth = Math.max(1, px / 3)
      ctx.strokeRect(0.5, focusRow * px + 0.5, (focusRow + 1) * px - 1, px - 1)
    }
  }, [att, size, themeKey, focusRow])
  return <canvas ref={ref} className="TrainedGptExplorer-canvas" role="img" aria-label={label} />
}

const badge = (l: HeadLabel) => (l === 'induction' ? 'looks like induction' : l === 'previous-token' ? 'looks like previous-token' : null)

export function TrainedGptExplorer() {
  const [model, setModel] = useState<TrainedGpt | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let alive = true
    setError(null)
    getModel().then((m) => alive && setModel(m), (e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)))
    return () => { alive = false }
  }, [attempt])

  const goal = <>This is <b>not</b> a toy with random weights: it is a small GPT that was actually trained. Type a prompt, generate text, then open the heads to see <em>what it learned to look at</em> and watch the prediction form layer by layer.</>

  if (!model) {
    return (
      <Lab title="Inside a trained tiny GPT" goal={goal}>
        {error ? (
          <div role="alert">
            <p className="lab-note"><b>The trained model could not be loaded.</b> {error}. The rest of the page works without it.</p>
            <button className="btn small" onClick={() => setAttempt(attempt + 1)}>Try again</button>
          </div>
        ) : (
          <p className="lab-note" aria-live="polite" aria-busy="true">Loading the trained weights (about 1.6 MB)…</p>
        )}
      </Lab>
    )
  }
  return (
    <Lab title="Inside a trained tiny GPT" goal={goal}>
      <Explorer m={model} />
    </Lab>
  )
}

function Explorer({ m }: { m: TrainedGpt }) {
  const { nLayer: L, nHead: H, contextLen: C, nEmbd: D, vocabSize: V } = m.cfg
  const tr = m.training
  const themeKey = useThemeKey()

  const [prompt, setPrompt] = useState('ROMEO:\n')
  const [gen, setGen] = useState<GenStep[]>([])
  const [running, setRunning] = useState(false)
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(10)
  const [length, setLength] = useState(80)
  const [seed, setSeed] = useState(1)
  const [shownStep, setShownStep] = useState<number | null>(null)
  const [sel, setSel] = useState<{ l: number; h: number }>({ l: 0, h: 0 })
  const [pos, setPos] = useState<number | null>(null)
  const [scores, setScores] = useState<HeadScores | null>(null)
  const timer = useRef<number | undefined>(undefined)

  const { ids: promptIds, dropped } = useMemo(() => encode(m, prompt), [m, prompt])

  // head scores: ~10 forward passes, spread over several frames
  useEffect(() => {
    let alive = true
    headScoresAsync(m).then((s) => alive && setScores(s), () => undefined)
    return () => { alive = false }
  }, [m])

  // stop generating on unmount
  useEffect(() => () => window.clearTimeout(timer.current), [])

  const stop = useCallback(() => { window.clearTimeout(timer.current); setRunning(false) }, [])

  const run = useCallback((n: number) => {
    if (!promptIds.length) return
    window.clearTimeout(timer.current)
    const rng = makeRng(seed * 7919 + gen.length)
    let steps = gen.slice()
    let left = n
    setRunning(true)
    const tick = () => {
      const ids = [...promptIds, ...steps.map((s) => s.id)]
      const s = generateStep(m, ids, temperature, topK, rng.next())
      steps = [...steps, s]
      left--
      setGen(steps)
      setShownStep(steps.length - 1)
      if (left > 0) timer.current = window.setTimeout(tick, 0)
      else setRunning(false)
    }
    timer.current = window.setTimeout(tick, 0)
  }, [m, promptIds, gen, seed, temperature, topK])

  const clearGen = () => { stop(); setGen([]); setShownStep(null); setPos(null) }

  // the text being analysed: prompt + generated, cropped to the context window
  const allIds = useMemo(() => [...promptIds, ...gen.map((s) => s.id)], [promptIds, gen])
  const ids = allIds.slice(-C)
  const cropped = allIds.length - ids.length
  const analysisKey = running ? 'frozen' : ids.join(',')
  const lastIds = useRef<number[]>(ids)
  if (!running) lastIds.current = ids
  const aIds = lastIds.current
  const trace = useMemo(() => (aIds.length ? forward(m, aIds) : null), [m, analysisKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const T = aIds.length
  const q = pos !== null && pos < T ? pos : T - 1
  const chars = aIds.map((i) => m.vocab[i])

  const shown = shownStep !== null && shownStep < gen.length ? gen[shownStep] : null

  return (
    <>
      <p className="lab-note TrainedGptExplorer-card">
        <b>The model.</b> The <code>GPT</code> class from <code>tiny_gpt.py</code>, trained by <code>export_checkpoint.py</code>: {m.nParams.toLocaleString('en-US')} parameters,
        {' '}{L} layers × {H} heads, {D}-number vectors, a context window of {C} characters, {V} possible characters (one token = one character).
        Trained on tiny Shakespeare ({(tr.data_chars / 1e6).toFixed(1)} M characters; the last 10% held out) for {tr.steps.toLocaleString('en-US')} steps of {tr.batch_size} × {C} characters.
        Loss on held-out text: <b>{tr.val_loss.toFixed(2)}</b> (training text: {tr.train_loss.toFixed(2)}); guessing uniformly would score ln {V} = {Math.log(V).toFixed(2)}.
        Weights stored as 16-bit floats; the maths below runs in your browser.
      </p>

      {/* ---------------- 1. prompt and generation ---------------- */}
      <h4 className="TrainedGptExplorer-h">1. Generate</h4>
      <label htmlFor="tge-prompt" className="TrainedGptExplorer-label">Prompt (case matters; Enter makes a new line, which the model also reads as a character)</label>
      <textarea id="tge-prompt" className="input TrainedGptExplorer-prompt" value={prompt} rows={2} onChange={(e) => { setPrompt(e.target.value); clearGen() }} spellCheck={false} />
      <div className="btn-row" style={{ marginTop: 6 }} role="group" aria-label="Example prompts">
        {PRESETS.map((p) => <button key={p.label} className="btn small" aria-pressed={prompt === p.text} onClick={() => { setPrompt(p.text); clearGen() }}>{p.label}</button>)}
      </div>
      {dropped.length > 0 && <p className="lab-note">Ignored {dropped.length} character{dropped.length > 1 ? 's' : ''} the model never saw in training: {[...new Set(dropped)].map((c) => `“${c}”`).join(' ')}.</p>}
      {!promptIds.length && <p className="lab-note">Type at least one character to start.</p>}

      <div className="controls">
        <Slider label="temperature" value={temperature} min={0.1} max={2} step={0.1} onChange={setTemperature} />
        <Slider label="top-k (0 = off)" value={topK} min={0} max={V} step={1} onChange={setTopK} />
        <Slider label="characters to generate" value={length} min={10} max={300} step={10} onChange={setLength} />
      </div>
      <div className="btn-row">
        {running
          ? <button className="btn small" onClick={stop}>Stop</button>
          : <button className="btn small primary" disabled={!promptIds.length} onClick={() => run(length)}>{gen.length ? `Generate ${length} more` : `Generate ${length}`}</button>}
        <button className="btn small" disabled={running || !promptIds.length} onClick={() => run(1)}>One character</button>
        <button className="btn small" disabled={running || !gen.length} onClick={clearGen}>Clear</button>
        <button className="btn small" disabled={running} onClick={() => { clearGen(); setSeed(seed + 1) }}>New dice (seed {seed})</button>
      </div>

      <div className="TrainedGptExplorer-output" aria-label="Prompt and generated text">
        <span className="TrainedGptExplorer-prompttext">{prompt}</span>
        {gen.map((s, i) => (
          <button key={i} className="TrainedGptExplorer-gen" aria-pressed={shownStep === i} title={`p = ${fmtPct(s.p)}`}
            aria-label={`generated ${tokName(m.vocab[s.id])}, probability ${fmtPct(s.p)}`}
            onClick={() => setShownStep(i)}>{m.vocab[s.id] === '\n' ? '↵\n' : m.vocab[s.id]}</button>
        ))}
      </div>
      <div aria-live="polite">
        {shown && (
          <div className="TrainedGptExplorer-top5">
            <p className="lab-note" style={{ margin: '4px 0' }}>Character {shownStep! + 1} of the continuation: the model picked <b>{tokName(m.vocab[shown.id])}</b> (p = {fmtPct(shown.p)}) from these top 5 (after temperature {temperature}{topK ? ` and top-${topK}` : ''}). Click any generated character to see its alternatives.</p>
            <Bars items={shown.top.map((r) => ({ label: showTok(m.vocab[r.id]), value: r.p, tone: r.id === shown.id ? 'accent' as const : undefined }))} max={1} />
          </div>
        )}
      </div>

      {trace && (
        <>
          {/* ---------------- 2. attention ---------------- */}
          <h4 className="TrainedGptExplorer-h">2. Where does each head look?</h4>
          <p className="lab-note">
            Analysing the last {T} characters{running ? ' (paused while generating)' : ''}{cropped > 0 && !running ? ` (the first ${cropped} fall outside the ${C}-character window)` : ''}.
            Each thumbnail is one head: row = the character doing the looking, column = the earlier character it looks at, darker = more attention. Click one to enlarge. A dashed border marks a head that passes one of the tests in part 4 (“prev?” = looks like previous-token, “ind?” = looks like induction).
          </p>
          <div className="TrainedGptExplorer-grid" role="group" aria-label={`${L} layers by ${H} heads`} style={{ gridTemplateColumns: `auto repeat(${H}, minmax(0, 1fr))` }}>
            <span />
            {Array.from({ length: H }, (_, h) => <span key={h} className="TrainedGptExplorer-axis">head {h + 1}</span>)}
            {trace.attention.map((layer, l) => (
              <FragmentRow key={l} label={`layer ${l + 1}`}>
                {layer.map((att, h) => {
                  const lab = scores ? labelHead(scores, l, h) : null
                  return (
                    <button key={h} className={`TrainedGptExplorer-thumb${lab ? ' flagged' : ''}`} aria-pressed={sel.l === l && sel.h === h} onClick={() => setSel({ l, h })}
                      aria-label={`Layer ${l + 1}, head ${h + 1}${lab ? `, ${badge(lab)}` : ''}`}>
                      <AttnCanvas att={att} size={96} themeKey={themeKey} label={`attention pattern of layer ${l + 1} head ${h + 1}`} />
                      {lab && <span className="TrainedGptExplorer-badge">{lab === 'induction' ? 'ind?' : 'prev?'}</span>}
                    </button>
                  )
                })}
              </FragmentRow>
            ))}
          </div>

          <HeadDetail att={trace.attention[sel.l][sel.h]} l={sel.l} h={sel.h} q={q} chars={chars} setPos={setPos} themeKey={themeKey} scores={scores} />

          {/* ---------------- 3. logit lens ---------------- */}
          <h4 className="TrainedGptExplorer-h">3. How the prediction forms, layer by layer (logit lens)</h4>
          <p className="lab-note">
            After each block, take the running vector for the selected character and push it straight through the final LayerNorm and the output head, as if the model stopped there.
            This “logit lens” is a rough probe: early layers were never trained to be read this way, so treat their rows as hints, not as what the model “thinks”.
          </p>
          <TokenPicker chars={chars} q={q} setPos={setPos} label="Character whose next-character prediction to inspect" />
          <LensTable m={m} lens={trace.lens} q={q} next={q < T - 1 ? aIds[q + 1] : null} />
        </>
      )}

      {/* ---------------- 4. head scores ---------------- */}
      <h4 className="TrainedGptExplorer-h">4. Heads with a recognisable job</h4>
      <HeadScoreTable scores={scores} L={L} H={H} C={C} sel={sel} setSel={setSel} />
    </>
  )
}

function FragmentRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <span className="TrainedGptExplorer-axis TrainedGptExplorer-rowlabel">{label}</span>
      {children}
    </>
  )
}

function TokenPicker({ chars, q, setPos, label }: { chars: string[]; q: number; setPos: (p: number) => void; label: string }) {
  return (
    <div className="TrainedGptExplorer-tokens" role="group" aria-label={label}>
      {chars.map((c, i) => (
        <button key={i} className="TrainedGptExplorer-tok" aria-pressed={i === q} onClick={() => setPos(i)} aria-label={`position ${i + 1}: ${tokName(c)}`}>{showTok(c)}</button>
      ))}
    </div>
  )
}

function HeadDetail({ att, l, h, q, chars, setPos, themeKey, scores }: {
  att: Float32Array[]; l: number; h: number; q: number; chars: string[]; setPos: (p: number) => void; themeKey: number; scores: HeadScores | null
}) {
  const row = att[q]
  const top = Array.from(row.slice(0, q + 1), (w, s) => ({ s, w })).sort((a, b) => b.w - a.w).slice(0, 3)
  const lab = scores ? labelHead(scores, l, h) : null
  const max = Math.max(...Array.from(row))
  return (
    <div className="TrainedGptExplorer-detail">
      <div className="TrainedGptExplorer-big">
        <AttnCanvas att={att} size={256} themeKey={themeKey} focusRow={q} label={`Enlarged attention pattern of layer ${l + 1} head ${h + 1}; the outlined row is the selected character`} />
      </div>
      <div className="TrainedGptExplorer-detailtext">
        <p style={{ marginTop: 0 }}><b>Layer {l + 1}, head {h + 1}</b>{lab && <> · <span className="chip acc">{badge(lab)}</span></>}</p>
        {scores && <p className="lab-note" style={{ marginTop: 0 }}>previous-token score {scores.previousToken[l][h].toFixed(2)}, induction score {scores.induction[l][h].toFixed(2)} (explained in part 4).</p>}
        <TokenPicker chars={chars} q={q} setPos={setPos} label="Pick the character that is looking" />
        <p className="lab-note" style={{ marginBottom: 4 }}>Where {tokName(chars[q])} at position {q + 1} looks (shading = attention):</p>
        <div className="TrainedGptExplorer-tokens" aria-hidden="true">
          {chars.slice(0, q + 1).map((c, s) => (
            <span key={s} className="TrainedGptExplorer-heat" style={{ background: `color-mix(in srgb, var(--accent) ${Math.round((row[s] / Math.max(max, 1e-9)) * 70)}%, var(--paper-2))` }}>{showTok(c)}</span>
          ))}
        </div>
        <p className="readout" aria-live="polite" style={{ marginTop: 8 }}>
          {top.map(({ s, w }) => <span key={s}>{tokName(chars[s])} at {s + 1}{s === q ? ' (itself)' : s === q - 1 ? ' (the one before)' : ''}: <b>{fmtPct(w)}</b></span>)}
        </p>
      </div>
    </div>
  )
}

function LensTable({ m, lens, q, next }: { m: TrainedGpt; lens: Float32Array[][]; q: number; next: number | null }) {
  return (
    <div className="table-scroll" aria-live="polite">
      <table className="plain TrainedGptExplorer-lens">
        <caption className="sr-only">Top 5 next-character predictions read out after each layer</caption>
        <thead>
          <tr><th scope="col">after</th><th scope="col">top 5 guesses for the next character</th>{next !== null && <th scope="col">actual next {tokName(m.vocab[next])}</th>}</tr>
        </thead>
        <tbody>
          {lens.map((rows, l) => {
            const p = softmaxRow(rows[q])
            return (
              <tr key={l}>
                <th scope="row">layer {l + 1}{l === lens.length - 1 ? ' (output)' : ''}</th>
                <td>{topN(p, 5).map((r) => <span key={r.id} className={`chip${r.id === next ? ' acc' : ''}`} style={{ marginRight: 4 }}>{showTok(m.vocab[r.id])} {fmtPct(r.p)}</span>)}</td>
                {next !== null && <td className="mono">{fmtPct(p[next])}</td>}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function HeadScoreTable({ scores, L, H, C, sel, setSel }: { scores: HeadScores | null; L: number; H: number; C: number; sel: { l: number; h: number }; setSel: (s: { l: number; h: number }) => void }) {
  if (!scores) return <p className="lab-note" aria-busy="true">Measuring every head…</p>
  const named = (kind: HeadLabel) => Array.from({ length: L * H }, (_, i) => [Math.floor(i / H), i % H]).filter(([l, h]) => labelHead(scores, l, h) === kind).map(([l, h]) => `layer ${l + 1} head ${h + 1}`)
  const prevHeads = named('previous-token')
  const indHeads = named('induction')
  const cell = (l: number, h: number) => {
    const lab = labelHead(scores, l, h)
    return (
      <td key={h}>
        <button className="TrainedGptExplorer-score" aria-pressed={sel.l === l && sel.h === h} onClick={() => setSel({ l, h })} aria-label={`layer ${l + 1} head ${h + 1}: previous-token ${scores.previousToken[l][h].toFixed(2)}, induction ${scores.induction[l][h].toFixed(2)}${lab ? `, ${badge(lab)}` : ''}`}>
          <span className={scores.previousToken[l][h] >= PREV_THRESHOLD ? 'hit' : ''}>{scores.previousToken[l][h].toFixed(2)}</span>
          {' / '}
          <span className={scores.induction[l][h] >= INDUCTION_THRESHOLD ? 'hit' : ''}>{scores.induction[l][h].toFixed(2)}</span>
        </button>
      </td>
    )
  }
  return (
    <>
      <p className="lab-note">Two simple measurements, computed in your browser when the model loaded:</p>
      <ul className="lab-note">
        <li><b>Previous-token score</b>: on four lines of held-out Shakespeare, the average attention each character gives to the character <em>just before it</em>. A head that spread attention evenly would score about {scores.uniformPrevious.toFixed(2)}. We say “looks like a previous-token head” at {PREV_THRESHOLD} or more.</li>
        <li><b>Induction score</b>: feed {Math.min(48, Math.floor(C / 2))} random lower-case letters, then the same letters again. In the second copy, how much attention does each character give to the character that came <em>after</em> its earlier copy? That is the move “last time I saw this, what came next?”. Random text means the model cannot have memorised it. Even spreading would give about {scores.uniformInduction.toFixed(2)}; we say “looks like an induction head” at {INDUCTION_THRESHOLD} or more.</li>
      </ul>
      <div className="table-scroll">
        <table className="plain TrainedGptExplorer-scores">
          <caption className="sr-only">Previous-token score / induction score for every head</caption>
          <thead><tr><th scope="col">prev / induction</th>{Array.from({ length: H }, (_, h) => <th key={h} scope="col">head {h + 1}</th>)}</tr></thead>
          <tbody>{Array.from({ length: L }, (_, l) => <tr key={l}><th scope="row">layer {l + 1}</th>{Array.from({ length: H }, (_, h) => cell(l, h))}</tr>)}</tbody>
        </table>
      </div>
      <p className="lab-note">Scores at or above the threshold are bold and underlined. These are behavioural tests, not proof of a mechanism: a head can pass one on this data and do something else as well.</p>
      <p className="lab-note" aria-live="polite">
        <b>What this model shows.</b>{' '}
        {prevHeads.length
          ? <>{prevHeads.length === 1 ? 'One head looks' : `${prevHeads.length} heads look`} like a previous-token head ({prevHeads.join(', ')}). </>
          : <>No head passes the previous-token test; several split their attention between the last few characters. </>}
        {indHeads.length
          ? <>{indHeads.length === 1 ? 'One head looks' : `${indHeads.length} heads look`} like an induction head ({indHeads.join(', ')}).</>
          : <>No head passes the induction test, and the model does not use the repeat: its loss on the second copy of the random letters is {scores.repeatLoss.second.toFixed(2)} versus {scores.repeatLoss.first.toFixed(2)} on the first (a model that copied would get close to 0). That is a real result, not a bug: this model is small, trained for a few minutes, and sees only {C} characters, so repeats inside its window are rare in training. Induction heads were found in larger models trained for longer (Olsson et al., 2022). Try the “repeated nonsense” prompt and see that the logit lens does not predict the repeat either.</>}
      </p>
    </>
  )
}
