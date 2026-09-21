// Lesson 1.4, second interactive: nudges travelling through a pipeline.
// x -> square -> +1 -> x3. Local amplifications multiply: that is the chain rule.
import { useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import { chainReport, wiggleTrace } from '../lib/calculus'

const NUDGE = 0.001
const n = (x: number, digits = 2) => fmt(x, Number.isInteger(x) ? 0 : digits)

export function ChainRulePipeline() {
  const [x, setX] = useState(2)
  const r = chainReport(x)
  const wiggle = wiggleTrace(x, NUDGE)
  const agrees = Math.abs(r.product - r.measured) < 1e-3

  return (
    <Lab
      title="A nudge travels through a pipeline"
      goal={<>Three tiny stages in a row. Each stage has its own amplification: how much it stretches a nudge passing through. Move x and watch which amplification changes, and which never do. Can you find the x where a nudge has <b>no effect at all</b> on the output?</>}
    >
      <Slider label="input x" value={x} min={-3} max={3} step={0.1} format={(v) => fmt(v, 1)} onChange={setX} />

      <div aria-live="polite">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10, margin: '14px 0' }}>
          <div className="card" style={{ margin: 0, padding: '12px 14px' }}>
            <div className="muted mono" style={{ fontSize: 11.5 }}>INPUT</div>
            <div className="mono" style={{ fontSize: 17 }}>x = {n(x, 1)}</div>
            <div className="lab-note" style={{ marginTop: 6 }}>nudge: +{NUDGE}</div>
          </div>
          {r.stages.map((s, i) => (
            <div key={s.name} className="card" style={{ margin: 0, padding: '12px 14px', borderTop: '3px solid var(--accent)' }}>
              <div className="muted mono" style={{ fontSize: 11.5 }}>STAGE {i + 1}: {s.name.toUpperCase()}</div>
              <div className="mono" style={{ fontSize: 15 }}>{s.expr}</div>
              <div className="mono" style={{ fontSize: 14, marginTop: 4 }}>{n(s.input)} → <b>{n(s.output)}</b></div>
              <div style={{ marginTop: 6, fontSize: 14 }}>amplifies nudges <b className="mono">×{n(s.local)}</b></div>
              <div className="muted" style={{ fontSize: 12.5 }}>({s.localRule})</div>
              <div className="lab-note" style={{ marginTop: 6 }}>nudge is now {fmt(wiggle[i + 1], 6)}</div>
            </div>
          ))}
        </div>

        <div className="readout" style={{ display: 'block' }}>
          <div className="muted" style={{ fontSize: 12 }}>chain rule: multiply the local amplifications</div>
          <div>{r.stages.map((s) => (s.local < 0 ? `(${n(s.local)})` : n(s.local))).join(' × ')} = <b style={{ fontSize: 18 }}>{n(r.product)}</b></div>
          <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>honest check: nudge the whole pipeline by 0.000001 and divide</div>
          <div>measured = <b>{fmt(r.measured, 4)}</b> {agrees ? <span style={{ color: 'var(--good)' }}>✓ they agree</span> : <span style={{ color: 'var(--bad)' }}>✗ they differ</span>}</div>
        </div>
        <p className="lab-note" style={{ marginTop: 8 }}>
          Output y = {n(r.output)}. {r.product === 0
            ? <>At x = 0 the first stage amplifies by 0, so the whole product is 0. One dead stage silences everything upstream of it. Remember this when you meet “vanishing gradients”.</>
            : <>A nudge of +{NUDGE} at the input arrives at the output as about {fmt(wiggle[3], 4)}: the nudge multiplied by {n(r.product)}{r.product < 0 ? ', so the output moves the other way' : ''}.</>}
        </p>
      </div>
    </Lab>
  )
}
