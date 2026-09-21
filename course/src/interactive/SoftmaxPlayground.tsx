// Lesson 1.3: scores -> probabilities (SoftmaxPlayground), and "surprise"
// for the cross-entropy loss (SurprisePlayground).
import { useRef, useState } from 'react'
import { Bars, Lab, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import { makeRng } from '../lib/rng'
import { fmtBig, rollCounts, softmaxSteps, surprise } from '../lib/softmaxPlay'

const TOKENS = ['mat', 'sofa', 'floor', 'moon', 'banana']
const START = [4.2, 3.1, 2.5, 0.3, -1.5]
const SEED = 42

export function SoftmaxPlayground() {
  const [logits, setLogits] = useState(START)
  const [temperature, setTemperature] = useState(1)
  const [plus100, setPlus100] = useState(false)
  const [tally, setTally] = useState<{ key: string; counts: number[]; last: number | null }>({ key: '', counts: [], last: null })
  const rng = useRef(makeRng(SEED))

  const s = softmaxSteps(logits, temperature, plus100 ? 100 : 0)
  const key = s.probs.map((p) => p.toFixed(6)).join()
  const counts = tally.key === key ? tally.counts : TOKENS.map(() => 0)
  const last = tally.key === key ? tally.last : null
  const rolls = counts.reduce((a, b) => a + b, 0)

  const roll = (n: number) => {
    const next = rollCounts(s.probs, n, rng.current, counts)
    const drawn = n === 1 ? next.findIndex((c, i) => c !== counts[i]) : null
    setTally({ key, counts: next, last: drawn })
  }
  const resetRolls = () => {
    rng.current = makeRng(SEED)
    setTally({ key: '', counts: [], last: null })
  }

  return (
    <Lab
      title="From scores to probabilities"
      goal={<>The prompt is <b>“The cat sat on the ___”</b>. Each slider is the model’s raw score for one candidate word. Move one slider and watch <em>all</em> the probabilities change. Then try the temperature, and the “add 100” switch. Predict before you click.</>}
    >
      <div className="controls">
        {TOKENS.map((t, i) => (
          <Slider key={t} label={`score for “${t}”`} value={logits[i]} min={-5} max={8} step={0.1} format={(v) => fmt(v, 1)} onChange={(v) => setLogits(logits.map((z, k) => (k === i ? v : z)))} />
        ))}
        <Slider label="temperature T (divide every score by T)" value={temperature} min={0.2} max={3} step={0.1} format={(v) => fmt(v, 1)} onChange={setTemperature} />
      </div>
      <div className="btn-row" style={{ marginBottom: 14 }}>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}>
          <input type="checkbox" checked={plus100} onChange={(e) => setPlus100(e.target.checked)} /> Add 100 to every score
        </label>
        <button className="btn small" onClick={() => { setLogits(START); setTemperature(1); setPlus100(false) }}>Reset scores</button>
      </div>

      <div aria-live="polite">
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead>
              <tr><th>token</th><th>score z</th><th>z ÷ T</th><th>e<sup>z ÷ T</sup></th><th>÷ total = probability</th></tr>
            </thead>
            <tbody>
              {TOKENS.map((t, i) => (
                <tr key={t}>
                  <td>{t}</td>
                  <td>{fmt(s.shifted[i], 1)}</td>
                  <td>{fmt(s.scaled[i])}</td>
                  <td>{fmtBig(s.exps[i])}</td>
                  <td><b>{(s.probs[i] * 100).toFixed(1)}%</b></td>
                </tr>
              ))}
              <tr><td colSpan={3}>total</td><td>{fmtBig(s.expSum)}</td><td>100.0%</td></tr>
            </tbody>
          </table>
        </div>
        <Bars items={TOKENS.map((t, i) => ({ label: t, value: s.probs[i] }))} max={1} />
        {plus100 && <p className="lab-note">Look at the e<sup>z</sup> column: the numbers became astronomically large, yet every probability is identical to before. Softmax only cares about the <b>gaps</b> between scores, not about the scores themselves.</p>}
        {!Number.isFinite(s.expSum) && <p className="lab-note"><b>Overflow.</b> e<sup>z</sup> no longer fits in a floating-point number. The probabilities shown are still right because this page uses the stable trick from the deep dive below.</p>}
      </div>

      <h4 style={{ fontSize: 17, margin: '22px 0 6px' }}>Sampling: roll the weighted die</h4>
      <p className="lab-note">The top word is not always picked. The sampler, a separate step after the model, rolls a die with {TOKENS.length} faces, where each face is as wide as that token’s probability. Roll a few times, then many times.</p>
      <div className="btn-row" style={{ margin: '8px 0' }}>
        <button className="btn small primary" onClick={() => roll(1)}>Roll once</button>
        <button className="btn small" onClick={() => roll(100)}>Roll 100 times</button>
        <button className="btn small" onClick={resetRolls} disabled={rolls === 0}>Clear tally</button>
        <span className="mono" aria-live="polite" style={{ fontSize: 14 }}>
          {rolls === 0 ? 'no rolls yet' : <>{rolls} roll{rolls === 1 ? '' : 's'}{last !== null && last >= 0 && <>, last draw: <b>“{TOKENS[last]}”</b></>}</>}
        </span>
      </div>
      <div className="table-scroll">
        <table className="plain mono" style={{ fontSize: 13.5 }}>
          <thead><tr><th>token</th><th>probability</th><th>times drawn</th><th>share of rolls</th></tr></thead>
          <tbody>
            {TOKENS.map((t, i) => (
              <tr key={t} style={{ fontWeight: last === i ? 700 : 400 }}>
                <td>{t}</td>
                <td>{(s.probs[i] * 100).toFixed(1)}%</td>
                <td>{counts[i]}</td>
                <td>{rolls ? `${((counts[i] / rolls) * 100).toFixed(1)}%` : 'n/a'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="lab-note">The tally clears when the probabilities change. The random numbers come from a seeded generator, so “Clear tally” replays the same sequence of rolls.</p>
    </Lab>
  )
}

/* ---------- cross-entropy as surprise ---------- */
const W = 360
const H = 200
const PAD = { l: 38, r: 10, t: 12, b: 30 }
const Y_MAX = 5
const sx = (p: number) => PAD.l + p * (W - PAD.l - PAD.r)
const sy = (loss: number) => PAD.t + (1 - Math.min(loss, Y_MAX) / Y_MAX) * (H - PAD.t - PAD.b)
const CURVE = Array.from({ length: 100 }, (_, i) => 0.007 + (i / 99) * 0.993).map((p) => `${sx(p).toFixed(1)},${sy(surprise(p)).toFixed(1)}`).join(' ')

export function SurprisePlayground() {
  const [p, setP] = useState(0.5)
  const loss = surprise(p)
  const mood = p >= 0.8 ? 'Barely surprised. Almost nothing to fix.' : p >= 0.3 ? 'Somewhat surprised.' : p >= 0.05 ? 'Very surprised. A large loss.' : 'Shocked. The model was confidently wrong, and the loss explodes.'
  return (
    <Lab
      title="Loss as surprise"
      goal={<>The correct next word was “mat”. Slide the probability the model gave to “mat” and watch the loss. Where does the loss change fastest: near 1, or near 0?</>}
    >
      <div className="grid-2" style={{ alignItems: 'center' }}>
        <div>
          <Slider label="probability given to the correct word" value={p} min={0.01} max={0.99} step={0.01} format={(v) => `${(v * 100).toFixed(0)}%`} onChange={setP} />
          <div className="readout" aria-live="polite" style={{ display: 'block', marginTop: 10 }}>
            <div>loss = −ln({fmt(p)}) = <b style={{ fontSize: 18 }}>{fmt(loss, 3)}</b></div>
            <div style={{ fontFamily: 'var(--sans)', fontSize: 14.5, marginTop: 4 }}>{mood}</div>
          </div>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: 440 }} role="img" aria-label={`Curve of loss against probability. At probability ${fmt(p)} the loss is ${fmt(loss, 3)}. The curve is near zero at probability 1 and shoots up as the probability approaches 0.`}>
          {[0, 1, 2, 3, 4, 5].map((v) => (
            <g key={v}>
              <line className={v === 0 ? 'axis' : 'gridline'} x1={PAD.l} x2={W - PAD.r} y1={sy(v)} y2={sy(v)} />
              <text x={PAD.l - 6} y={sy(v) + 3.5} fontSize={10} textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{v}</text>
            </g>
          ))}
          {[0, 0.25, 0.5, 0.75, 1].map((v) => <text key={v} x={sx(v)} y={H - 14} fontSize={10} textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{v}</text>)}
          <line className="axis" x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={sy(0)} />
          <text x={(W + PAD.l) / 2} y={H - 2} fontSize={10.5} textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>probability given to the correct word</text>
          <text x={PAD.l + 22} y={PAD.t + 9} fontSize={10.5} style={{ fill: 'var(--ink-2)' }}>loss = −ln(p)</text>
          <polyline points={CURVE} fill="none" stroke="var(--accent)" strokeWidth={2.5} />
          <line x1={sx(p)} x2={sx(p)} y1={sy(0)} y2={sy(loss)} stroke="var(--ink-3)" strokeDasharray="3 3" />
          <circle cx={sx(p)} cy={sy(loss)} r={6} fill="var(--card)" stroke="var(--accent)" strokeWidth={3} />
        </svg>
      </div>
    </Lab>
  )
}
