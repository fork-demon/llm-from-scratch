// Train a real (tiny) character-level language model in the browser and watch
// train loss, validation loss and text samples change live.
// All maths lives in src/lib/tinyLm.ts; this file only drives it in small chunks.
import { useEffect, useRef, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { TINY_DATA_CHARS, baselineLoss, countParams, createTrainer, exampleCount, trainSteps, type Trainer, type TrainerOptions } from '../lib/tinyLm'

const LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
const BATCHES = [4, 8, 16, 32, 64, 128]
const DIMS = [4, 8, 16, 32]
const HIDDENS = [8, 32, 64, 128, 256]
const PAUSE_EVERY = 2000 // never burn the learner's battery forever
const FRAME_BUDGET_MS = 12
const DEFAULTS: TrainerOptions = { ctx: 4, dim: 16, hidden: 64, seed: 7, tinyData: false }

const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : 'NaN')

export function TrainingPlayground() {
  const [opts, setOpts] = useState<TrainerOptions>(DEFAULTS)
  const [lrIdx, setLrIdx] = useState(5)
  const [batchIdx, setBatchIdx] = useState(3)
  const [running, setRunning] = useState(false)
  const [, setTick] = useState(0)
  const trainerRef = useRef<Trainer | null>(null)
  if (trainerRef.current === null) trainerRef.current = createTrainer(opts)
  const t = trainerRef.current
  // the loop reads the latest slider values without restarting
  const live = useRef({ lr: LRS[lrIdx], batch: BATCHES[batchIdx] })
  live.current = { lr: LRS[lrIdx], batch: BATCHES[batchIdx] }

  const reset = (next: TrainerOptions = opts) => {
    setRunning(false)
    trainerRef.current = createTrainer(next)
    setOpts(next)
    setTick((n) => n + 1)
  }

  // The training loop: a few steps per animation frame, stopped on pause and on unmount.
  useEffect(() => {
    if (!running) return
    let raf = 0
    let alive = true
    const frame = () => {
      if (!alive) return
      const tr = trainerRef.current!
      const start = performance.now()
      do {
        trainSteps(tr, 1, live.current.lr, live.current.batch)
      } while (!tr.diverged && tr.step % PAUSE_EVERY !== 0 && performance.now() - start < FRAME_BUDGET_MS)
      setTick((n) => n + 1)
      if (tr.diverged || tr.step % PAUSE_EVERY === 0) { setRunning(false); return }
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => { alive = false; cancelAnimationFrame(raf) }
  }, [running])

  const V = t.ds.chars.length
  const base = baselineLoss(V)
  const hist = t.history
  const last = hist[hist.length - 1]
  const best = hist.reduce((b, h) => (h.val < b.val ? h : b), hist[0])
  const nTrain = exampleCount(t.trainIds, opts.ctx)
  const overfitting = !t.diverged && last.val > best.val + 0.2 && last.train < last.val - 0.5
  const aboveBaseline = !t.diverged && t.step >= 50 && last.train > base + 0.3

  // chart geometry
  const W = 640, H = 270, L = 44, R = 12, T = 14, B = 34
  const xMax = Math.max(200, t.step)
  const finite = hist.flatMap((h) => [h.train, h.val]).filter(Number.isFinite)
  const yMax = Math.min(8, Math.max(base * 1.15, ...finite))
  const px = (s: number) => L + (s / xMax) * (W - L - R)
  const py = (v: number) => T + (1 - Math.min(v, yMax) / yMax) * (H - T - B)
  const line = (key: 'train' | 'val') => hist.filter((h) => Number.isFinite(h[key])).map((h, i) => `${i ? 'L' : 'M'}${px(h.step).toFixed(1)},${py(h[key]).toFixed(1)}`).join(' ')
  const yTicks = [0, 1, 2, 3, 4, 5, 6, 7, 8].filter((v) => v <= yMax)
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round((f * xMax) / 50) * 50)

  const shownSamples = t.samples.length <= 4 ? t.samples : [t.samples[0], ...t.samples.slice(-3)]

  return (
    <Lab
      title="Train a language model, right here"
      goal={<>Press <b>Start</b> and watch three things at once: the two loss curves, the step counter, and the text samples. Then break it: a huge learning rate, a tiny dataset, a bigger model.</>}
    >
      <p className="lab-note">
        <b>What this is, honestly:</b> a real neural language model that predicts the next <em>character</em> from the previous {opts.ctx}, trained from random weights in your browser with hand-written backpropagation. It is the small MLP model from <code>bigram_lm.py</code> (Model C), <b>not a Transformer</b>: a Transformer in plain JavaScript would be too slow to be fun. It uses plain gradient descent, not AdamW. The training <em>loop</em> is exactly the same one. The real thing is <code>python tiny_gpt.py --quick</code>.
      </p>

      <div className="controls">
        <Slider label="Learning rate (live)" value={lrIdx} min={0} max={LRS.length - 1} step={1} onChange={setLrIdx} format={(i) => String(LRS[i])} />
        <Slider label="Batch size (live)" value={batchIdx} min={0} max={BATCHES.length - 1} step={1} onChange={setBatchIdx} format={(i) => String(BATCHES[i])} />
        <Slider label="Context length (resets)" value={opts.ctx} min={1} max={8} step={1} onChange={(ctx) => reset({ ...opts, ctx })} format={(v) => `${v} chars`} />
        <Slider label="Embedding size (resets)" value={DIMS.indexOf(opts.dim)} min={0} max={DIMS.length - 1} step={1} onChange={(i) => reset({ ...opts, dim: DIMS[i] })} format={(i) => String(DIMS[i])} />
        <Slider label="Hidden units (resets)" value={HIDDENS.indexOf(opts.hidden)} min={0} max={HIDDENS.length - 1} step={1} onChange={(i) => reset({ ...opts, hidden: HIDDENS[i] })} format={(i) => String(HIDDENS[i])} />
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}>
          <input type="checkbox" checked={opts.tinyData} onChange={(e) => reset({ ...opts, tinyData: e.target.checked })} />
          Tiny dataset: train on only {TINY_DATA_CHARS} characters (resets)
        </label>
      </div>

      <div className="btn-row" style={{ marginBottom: 12 }}>
        <button className="btn small primary" disabled={running || t.diverged} onClick={() => setRunning(true)}>{t.step === 0 ? 'Start' : 'Continue'}</button>
        <button className="btn small" disabled={!running} onClick={() => setRunning(false)}>Pause</button>
        <button className="btn small" onClick={() => reset()}>Reset</button>
        <span className="muted" style={{ fontSize: 13.5 }}>{running ? 'training…' : t.step > 0 && t.step % PAUSE_EVERY === 0 && !t.diverged ? `paused after ${t.step.toLocaleString()} steps; Continue for more` : ''}</span>
      </div>

      <div className="readout" aria-live="off">
        <span>step <b>{t.step.toLocaleString()}</b></span>
        <span>tokens seen <b>{t.tokensSeen.toLocaleString()}</b></span>
        <span>epochs <b>{(t.tokensSeen / nTrain).toFixed(1)}</b></span>
        <span>train loss <b>{f2(last.train)}</b></span>
        <span>val loss <b>{f2(last.val)}</b></span>
        <span>guessing = ln({V}) = <b>{base.toFixed(2)}</b></span>
      </div>
      <p className="lab-note" style={{ marginTop: 6 }}>
        {countParams(t.model.cfg).toLocaleString()} parameters. Training text: {t.trainIds.length.toLocaleString()} characters ({nTrain.toLocaleString()} examples = 1 epoch). Validation text: {t.ds.val.length.toLocaleString()} characters the model never trains on. Vocabulary: {V} characters.
      </p>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img"
        aria-label={`Loss curves after ${t.step} steps. Train loss ${f2(last.train)}, validation loss ${f2(last.val)}, guessing baseline ${base.toFixed(2)}.`}>
        {yTicks.map((v) => (
          <g key={v}>
            <line className="gridline" x1={L} x2={W - R} y1={py(v)} y2={py(v)} />
            <text x={L - 6} y={py(v) + 4} textAnchor="end" fontSize={11} style={{ fill: 'var(--ink-3)' }}>{v}</text>
          </g>
        ))}
        {xTicks.map((s, i) => <text key={i} x={px(Math.min(s, xMax))} y={H - 16} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)' }}>{s}</text>)}
        <text x={(L + W - R) / 2} y={H - 2} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)' }}>training step</text>
        <text x={12} y={(T + H - B) / 2} textAnchor="middle" fontSize={11} transform={`rotate(-90 12 ${(T + H - B) / 2})`} style={{ fill: 'var(--ink-3)' }}>loss</text>
        <line className="axis" x1={L} x2={L} y1={T} y2={H - B} />
        <line className="axis" x1={L} x2={W - R} y1={H - B} y2={H - B} />
        {/* the ln(V) sanity line */}
        <line x1={L} x2={W - R} y1={py(base)} y2={py(base)} stroke="var(--ink-3)" strokeWidth={1.5} strokeDasharray="2 4" />
        <text x={W - R - 4} y={py(base) - 5} textAnchor="end" fontSize={11} style={{ fill: 'var(--ink-3)' }}>pure guessing: ln({V}) = {base.toFixed(2)}</text>
        <path d={line('val')} fill="none" stroke="var(--ink)" strokeWidth={2} strokeDasharray="7 4" />
        <path d={line('train')} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
        {hist.length > 1 && best.step !== last.step && last.val > best.val + 0.1 && (
          <g>
            <circle cx={px(best.step)} cy={py(best.val)} r={4.5} fill="var(--card)" stroke="var(--ink)" strokeWidth={2} />
            <text x={px(best.step) + 8} y={py(best.val) - 8} fontSize={11}>best val {f2(best.val)} at step {best.step}</text>
          </g>
        )}
      </svg>
      <div className="readout" style={{ border: 0, background: 'transparent', padding: '0 0 8px' }}>
        <span><svg width="34" height="8" aria-hidden><line x1="0" x2="34" y1="4" y2="4" stroke="var(--accent)" strokeWidth="2.2" /></svg> train loss (solid)</span>
        <span><svg width="34" height="8" aria-hidden><line x1="0" x2="34" y1="4" y2="4" stroke="var(--ink)" strokeWidth="2" strokeDasharray="7 4" /></svg> validation loss (dashed)</span>
      </div>

      <div aria-live="polite">
        {t.diverged && (
          <p className="lab-note" style={{ borderLeft: '3px solid var(--bad)', paddingLeft: 12 }}>
            <b>The loss blew up at step {t.step}.</b> Nothing is broken: this is what “diverging” looks like. With learning rate {LRS[lrIdx]}, each update overshot the valley so far that the weights grew without limit, until the numbers overflowed into <code>NaN</code> (“not a number”). We stopped training for you. Lower the learning rate and press Reset.
          </p>
        )}
        {aboveBaseline && (
          <p className="lab-note" style={{ borderLeft: '3px solid var(--bad)', paddingLeft: 12 }}>
            <b>The loss is above the guessing line.</b> The model is now worse than knowing nothing. The steps are too big: try a smaller learning rate.
          </p>
        )}
        {overfitting && (
          <p className="lab-note" style={{ borderLeft: '3px solid var(--accent)', paddingLeft: 12 }}>
            <b>Overfitting.</b> Train loss keeps falling ({f2(last.train)}) but validation loss turned around at step {best.step} and is climbing ({f2(best.val)} → {f2(last.val)}). The model is memorising its training text instead of learning patterns that carry over.
          </p>
        )}
      </div>

      <h4 style={{ fontSize: 16, margin: '14px 0 6px' }}>What the model writes (a fresh sample every 100 steps)</h4>
      <div className="table-scroll">
        <table className="plain mono" style={{ fontSize: 13.5, margin: 0 }}>
          <thead><tr><th style={{ width: 70 }}>step</th><th>70 characters generated by the current weights</th></tr></thead>
          <tbody>
            {shownSamples.map((s) => <tr key={s.step}><td>{s.step}</td><td style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{s.text}</td></tr>)}
          </tbody>
        </table>
      </div>
      <p className="lab-note" style={{ marginTop: 8 }}>
        The training text is simple generated sentences such as “{'the cat sat on the mat. a bird ran past the river at dawn.'}” Same seed, same settings, same curve: every run is reproducible.
      </p>
    </Lab>
  )
}
