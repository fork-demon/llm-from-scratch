// From logits to one chosen token: temperature, top-k, top-p, then the dice roll.
// All maths lives in src/lib/toyLm.ts and src/lib/math.ts.
import { useMemo, useRef, useState } from 'react'
import { Bars, Lab, Slider } from '../components/ui'
import { fmt, sampleIndex } from '../lib/math'
import { makeRng } from '../lib/rng'
import { CAT_JUNK_FROM, CAT_LOGITS, CAT_PROMPT, CAT_TOKENS, TOY_JUNK, detokenize, samplingStages, toyGenerate, type SamplingSettings } from '../lib/toyLm'

const PRESETS: { label: string; s: SamplingSettings }[] = [
  { label: 'Greedy (T = 0)', s: { temperature: 0, topK: 0, topP: 1 } },
  { label: 'Focused (T 0.7, top-p 0.9)', s: { temperature: 0.7, topK: 0, topP: 0.9 } },
  { label: 'Plain (T = 1)', s: { temperature: 1, topK: 0, topP: 1 } },
  { label: 'Too hot (T = 3)', s: { temperature: 3, topK: 0, topP: 1 } },
  { label: 'Hot, but top-k 4', s: { temperature: 3, topK: 4, topP: 1 } },
]
const STAGES = ['3 · softmax', '4 · after top-k', '5 · after top-p: the die'] as const
const pct = (p: number) => (p === 0 ? '0' : p < 0.0005 ? '<0.1%' : `${(p * 100).toFixed(1)}%`)
const same = (a: SamplingSettings, b: SamplingSettings) => a.temperature === b.temperature && a.topK === b.topK && a.topP === b.topP

export function SamplingPlayground() {
  const [s, setS] = useState<SamplingSettings>({ temperature: 1, topK: 0, topP: 1 })
  const [stage, setStage] = useState(2)
  const [counts, setCounts] = useState<number[]>(() => CAT_TOKENS.map(() => 0))
  const [recent, setRecent] = useState<number[]>([])
  const [maxTokens, setMaxTokens] = useState(16)
  const [seed, setSeed] = useState(1)
  const rng = useRef(makeRng(42))

  const st = useMemo(() => samplingStages(CAT_LOGITS, s), [s])
  const total = counts.reduce((a, b) => a + b, 0)
  const junkMass = st.final.slice(CAT_JUNK_FROM).reduce((a, b) => a + b, 0)
  const kept = st.final.filter((p) => p > 0).length

  const clearTally = () => { setCounts(CAT_TOKENS.map(() => 0)); setRecent([]); rng.current = makeRng(42) }
  const change = (next: SamplingSettings) => { setS(next); clearTally() }
  const draw = (n: number) => {
    const c = counts.slice()
    const r = recent.slice()
    for (let i = 0; i < n; i++) { const ix = sampleIndex(st.final, rng.current.next()); c[ix]++; r.push(ix) }
    setCounts(c)
    setRecent(r.slice(-14))
  }

  const runs = useMemo(() => [0, 1, 2].map((i) => toyGenerate(['the', 'cat'], s, maxTokens, seed + i)), [s, maxTokens, seed])
  const stageProbs = [st.probs, st.afterTopK, st.final][stage]

  return (
    <Lab
      title="From scores to one chosen token"
      goal={<>The model has done its job: it produced one score (logit) per token for “{CAT_PROMPT} ___”. Everything below happens <b>outside</b> the network. Move the sliders, watch each column change, then roll the die.</>}
    >
      <div className="steps" role="group" aria-label="Presets">
        {PRESETS.map((p) => <button key={p.label} className="step-btn" aria-pressed={same(p.s, s)} onClick={() => change(p.s)}>{p.label}</button>)}
      </div>
      <div className="controls">
        <Slider label="Temperature T" value={s.temperature} min={0} max={3} step={0.1} onChange={(temperature) => change({ ...s, temperature })} format={(v) => (v === 0 ? '0 (greedy)' : v.toFixed(1))} />
        <Slider label="top-k" value={s.topK} min={0} max={CAT_TOKENS.length} step={1} onChange={(topK) => change({ ...s, topK })} format={(v) => (v === 0 ? 'off' : `keep ${v} best`)} />
        <Slider label="top-p" value={s.topP} min={0.1} max={1} step={0.05} onChange={(topP) => change({ ...s, topP })} format={(v) => (v >= 1 ? 'off (1.0)' : v.toFixed(2))} />
      </div>

      <div className="table-scroll">
        <table className="plain mono" style={{ fontSize: 13.5, marginTop: 0 }}>
          <thead>
            <tr><th>token</th><th>1 · logit</th><th>2 · ÷ T</th><th>3 · softmax</th><th>4 · top-k</th><th>5 · top-p</th></tr>
          </thead>
          <tbody>
            {CAT_TOKENS.map((tok, i) => {
              const cutK = st.afterTopK[i] === 0 && st.probs[i] > 0
              const cutP = st.final[i] === 0 && st.afterTopK[i] > 0
              return (
                <tr key={tok} style={{ opacity: st.final[i] === 0 ? 0.5 : 1 }}>
                  <td>{tok}{i >= CAT_JUNK_FROM ? ' (junk)' : ''}</td>
                  <td>{fmt(st.logits[i], 1)}</td>
                  <td>{s.temperature === 0 ? 'n/a' : fmt(st.scaled[i], 2)}</td>
                  <td>{pct(st.probs[i])}</td>
                  <td>{cutK ? 'cut' : pct(st.afterTopK[i])}</td>
                  <td>{cutP ? 'cut' : st.afterTopK[i] === 0 ? '' : <b>{pct(st.final[i])}</b>}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="lab-note">
        {s.temperature === 0
          ? <>T = 0 cannot be divided by, so it is defined as its limit: all probability on the single biggest logit. That is <b>greedy decoding</b>.</>
          : <>Column 2 divides every logit by T = {s.temperature.toFixed(1)}. The gap between “mat” and “floor” goes from {fmt(st.logits[0] - st.logits[1], 1)} to {fmt(st.scaled[0] - st.scaled[1], 2)}: {s.temperature < 1 ? 'bigger gaps, so softmax favours the leader more.' : s.temperature > 1 ? 'smaller gaps, so softmax spreads the probability out.' : 'unchanged at T = 1.'}</>}
        {' '}After each cut, the survivors are rescaled to sum to 100% again.
      </p>

      <div className="steps" role="tablist" aria-label="Which distribution to draw as bars">
        {STAGES.map((label, i) => <button key={label} role="tab" className="step-btn" aria-selected={i === stage} onClick={() => setStage(i)}>{label}</button>)}
      </div>
      <Bars items={CAT_TOKENS.map((tok, i) => ({ label: i >= CAT_JUNK_FROM ? `${tok} (junk)` : tok, value: stageProbs[i], dim: stageProbs[i] === 0 }))} max={1} />
      <div className="readout" aria-live="polite">
        <span>tokens still in the game: <b>{kept}</b> of {CAT_TOKENS.length}</span>
        <span>chance of drawing junk: <b>{pct(junkMass)}</b></span>
      </div>

      <h4 style={{ fontSize: 16, margin: '18px 0 6px' }}>Roll the die</h4>
      <div className="btn-row">
        <button className="btn small primary" onClick={() => draw(1)}>Sample 1 token</button>
        <button className="btn small" onClick={() => draw(100)}>Sample 100</button>
        <button className="btn small" disabled={total === 0} onClick={clearTally}>Clear</button>
      </div>
      <p className="mono" style={{ fontSize: 14, minHeight: 24 }} aria-live="polite">
        {recent.length === 0 ? <span className="muted">no draws yet</span> : <>last draws: {recent.map((ix, j) => <span key={j} className="token">{CAT_TOKENS[ix]}</span>)}</>}
      </p>
      {total > 0 && (
        <>
          <Bars items={CAT_TOKENS.map((tok, i) => ({ label: `${tok} ×${counts[i]}`, value: counts[i] / total, dim: counts[i] === 0 }))} max={1} />
          <p className="lab-note">{total} draws, {counts.filter((c) => c > 0).length} different tokens seen, {counts.slice(CAT_JUNK_FROM).reduce((a, b) => a + b, 0)} junk. The draws use a seeded random generator, so Clear and resample gives the same sequence.</p>
        </>
      )}

      <h4 style={{ fontSize: 16, margin: '18px 0 6px' }}>Same settings, whole sentences</h4>
      <p className="lab-note">
        Now the loop: pick a token, append it, ask for new logits, repeat. The “model” here is a <b>hand-made toy table</b> (the next word depends only on the previous word), not a Transformer, so the text stays readable. Generation stops when the table emits the end-of-sequence token <span className="token">&lt;eos&gt;</span>, or when it hits <b>max tokens</b>, whichever comes first.
      </p>
      <div className="controls">
        <Slider label="max tokens" value={maxTokens} min={1} max={40} step={1} onChange={setMaxTokens} />
        <div className="btn-row"><button className="btn small" onClick={() => setSeed(seed + 3)}>Generate three more</button></div>
      </div>
      <div aria-live="polite">
        {runs.map((g, i) => (
          <div key={i} className="card" style={{ padding: '10px 14px', marginBottom: 8 }}>
            <span className="muted">the cat </span>
            {g.tokens.some((w) => TOY_JUNK.has(w))
              ? g.tokens.map((w, j) => (TOY_JUNK.has(w) ? <span key={j}> <span className="token" title="junk token">{w}</span></span> : <span key={j}>{w === '.' || j === 0 ? w : ` ${w}`}</span>))
              : detokenize(g.tokens)}
            <div className="mono muted" style={{ fontSize: 12.5, marginTop: 4 }}>
              {g.tokens.length} tokens · {g.stopped === 'eos' ? 'stopped by itself: <eos>' : 'cut off by max tokens'}
            </div>
          </div>
        ))}
      </div>
    </Lab>
  )
}
