// Speculative decoding, one round at a time, with the real probabilities, and the exact
// tokens-per-pass curve against draft quality. Logic: src/lib/speculative.ts.
import { useMemo, useRef, useState } from 'react'
import { Bars, Lab, Slider } from '../components/ui'
import { acceptanceRate, expectedTokensPerPass, iidTokensPerPass, makeModels, speculativeRound, type SpecRound } from '../lib/speculative'
import { makeRng } from '../lib/rng'

const WORDS = ['the', 'cat', 'sat', 'on', 'mat', '.']
const EPS_GRID = Array.from({ length: 21 }, (_, i) => i / 20)

export function SpeculativeLab() {
  const [eps, setEps] = useState(0.4)
  const [gamma, setGamma] = useState(4)
  const rng = useRef(makeRng(23))
  // start with one round already proposed, so the table is never empty
  const [round, setRound] = useState<SpecRound | null>(() => { const m = makeModels(0.4); return speculativeRound(m.P, m.Q, 0, makeRng(16), 4) })
  const [last, setLast] = useState(0)
  const [revealed, setRevealed] = useState(0)
  const [tally, setTally] = useState({ passes: 0, tokens: 0 })
  const [text, setText] = useState<number[]>([0])

  const { P, Q } = useMemo(() => makeModels(eps), [eps])
  const alpha = acceptanceRate(P, Q)
  const curve = useMemo(() => EPS_GRID.map((e) => { const m = makeModels(e); return { e, exact: expectedTokensPerPass(m.P, m.Q, gamma), iid: iidTokensPerPass(acceptanceRate(m.P, m.Q), gamma) } }), [gamma])

  const reset = () => { setRound(null); setRevealed(0); setTally({ passes: 0, tokens: 0 }); setText([0]); setLast(0); rng.current = makeRng(23) }
  const newRound = () => {
    const ctx = text[text.length - 1]
    setLast(ctx)
    setRound(speculativeRound(P, Q, ctx, rng.current, gamma))
    setRevealed(0)
  }
  const totalReveals = round ? round.steps.length + 1 : 0 // verdicts, then the resample or bonus
  const finished = round !== null && revealed >= totalReveals
  const reveal = () => {
    if (!round) return
    const next = revealed + 1
    setRevealed(next)
    if (next >= totalReveals) {
      setText((t) => [...t, ...round.out])
      setTally((t) => ({ passes: t.passes + 1, tokens: t.tokens + round.out.length }))
    }
  }

  // which context's distributions to draw: the step being judged
  const focus = round ? round.steps[Math.min(revealed, round.steps.length - 1)] : null
  const focusCtx = focus ? focus.ctx : last
  const showResidual = round && round.residual && revealed >= round.steps.length

  /* curve geometry */
  const CW = 760, CH = 230, CL = 46, CB = 38, CT = 14, CR = 16
  const cx = (e: number) => CL + e * (CW - CL - CR)
  const cy = (v: number) => CH - CB - ((v - 1) / gamma) * (CH - CB - CT)

  return (
    <Lab
      title="Draft, verify, accept or replace"
      goal={<>A small draft model guesses {gamma} tokens. The big target model scores all of them in one pass. Press <b>New round</b>, then reveal the verdicts one by one. Make the draft worse and watch how many tokens one target pass still yields.</>}
    >
      <div className="controls">
        <Slider label="How wrong the draft is (0 = identical to the target)" value={eps} min={0} max={1} step={0.05} onChange={(v) => { setEps(v); reset() }} format={(v) => v.toFixed(2)} />
        <Slider label="Tokens proposed per round (γ)" value={gamma} min={1} max={8} step={1} onChange={(v) => { setGamma(v); reset() }} />
      </div>
      <p className="lab-note">Both “models” are toy Markov chains over six tokens: the next-token probabilities depend only on the last token, and they are random, not trained. The words are just labels. The accept and replace rule is the real one.</p>

      <div className="readout" style={{ fontFamily: 'var(--sans)', fontSize: 14 }}>
        <span>text so far: {text.slice(-14).map((t, i) => <span key={i} className="chip" style={{ marginRight: 3 }}>{WORDS[t]}</span>)}</span>
      </div>

      <div className="btn-row" style={{ margin: '12px 0' }}>
        <button className="btn small primary" disabled={round !== null && !finished} onClick={newRound}>New round</button>
        <button className="btn small" disabled={!round || finished} onClick={reveal}>{round && revealed >= round.steps.length ? (round.bonus !== null ? 'Take the bonus token' : 'Draw the replacement') : 'Reveal next verdict'}</button>
        <button className="btn small" onClick={reset}>Reset</button>
      </div>

      {round && (
        <div className="table-scroll" aria-live="polite">
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>after</th><th>draft proposes</th><th>draft q</th><th>target p</th><th>accept with min(1, p/q)</th><th>random draw</th><th>verdict</th></tr></thead>
            <tbody>
              {round.proposals.map((tok, i) => {
                const s = round.steps[i]
                const seen = s && i < revealed
                return (
                  <tr key={i} style={{ opacity: s ? 1 : 0.4 }}>
                    <td>{s ? WORDS[s.ctx] : '…'}</td>
                    <td><span className="chip">{WORDS[tok]}</span></td>
                    <td>{seen ? s.q.toFixed(3) : ''}</td>
                    <td>{seen ? s.p.toFixed(3) : ''}</td>
                    <td>{seen ? s.acceptProb.toFixed(3) : ''}</td>
                    <td>{seen ? s.u.toFixed(3) : ''}</td>
                    <td style={{ color: seen ? (s.accepted ? 'var(--good)' : 'var(--bad)') : undefined, fontWeight: 700 }}>{seen ? (s.accepted ? 'accept' : 'reject') : s ? '?' : 'never checked'}</td>
                  </tr>
                )
              })}
              {finished && (
                <tr>
                  <td colSpan={7}>
                    {round.bonus !== null
                      ? <>All {gamma} accepted. The target pass also scored the next position, so one more token comes free, drawn from p: <span className="chip acc">{WORDS[round.bonus]}</span></>
                      : <>Rejected. The replacement is drawn from the residual, max(0, p − q) rescaled: <span className="chip acc">{WORDS[round.resampled!]}</span>. Later proposals are thrown away.</>}
                    {' '}This pass produced <b>{round.out.length}</b> token{round.out.length > 1 ? 's' : ''}.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div>
          <div className="matrix-cap">target p( · | “{WORDS[focusCtx]}”)</div>
          <Bars items={P[focusCtx].map((v, i) => ({ label: WORDS[i], value: v }))} max={1} />
        </div>
        <div>
          <div className="matrix-cap">{showResidual ? 'residual: max(0, p − q), rescaled to sum to 1' : `draft q( · | “${WORDS[focusCtx]}”)`}</div>
          <Bars items={(showResidual ? round!.residual! : Q[focusCtx]).map((v, i) => ({ label: WORDS[i], value: v, tone: 'neutral' as const }))} max={1} />
        </div>
      </div>

      <div className="readout" aria-live="polite">
        <span>acceptance rate α = Σ min(p, q), averaged over contexts: <b>{alpha.toFixed(3)}</b></span>
        <span>your rounds: <b>{tally.tokens}</b> tokens in <b>{tally.passes}</b> target passes{tally.passes ? <> = <b>{(tally.tokens / tally.passes).toFixed(2)}</b> per pass</> : null}</span>
        <span>exact expectation: <b>{expectedTokensPerPass(P, Q, gamma).toFixed(2)}</b></span>
      </div>

      <h4 style={{ marginTop: 20, marginBottom: 4 }}>Tokens per target pass against draft quality</h4>
      <div className="table-scroll">
        <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', minWidth: 540, display: 'block' }} role="img" aria-label={`Expected tokens per target pass falls from ${gamma + 1} with a perfect draft to ${curve[curve.length - 1].exact.toFixed(2)} with an unrelated draft`}>
          {Array.from({ length: gamma + 1 }, (_, i) => i + 1).filter((v) => gamma <= 4 || v % 2 === 1).map((v) => (
            <g key={v}>
              <line x1={CL} x2={CW - CR} y1={cy(v)} y2={cy(v)} className="gridline" />
              <text x={CL - 6} y={cy(v) + 4} textAnchor="end" fontSize={10.5} style={{ fill: 'var(--ink-3)', fontFamily: 'var(--mono)' }}>{v}</text>
            </g>
          ))}
          <line x1={CL} x2={CW - CR} y1={CH - CB} y2={CH - CB} className="axis" />
          {[0, 0.25, 0.5, 0.75, 1].map((e) => <text key={e} x={cx(e)} y={CH - CB + 15} textAnchor="middle" fontSize={10.5} style={{ fill: 'var(--ink-3)', fontFamily: 'var(--mono)' }}>{e}</text>)}
          <text x={(CL + CW - CR) / 2} y={CH - 5} textAnchor="middle" fontSize={11.5} style={{ fill: 'var(--ink-2)' }}>how wrong the draft is</text>
          <polyline points={curve.map((c) => `${cx(c.e)},${cy(c.iid)}`).join(' ')} fill="none" stroke="var(--ink-3)" strokeWidth={1.3} strokeDasharray="5 4" />
          <polyline points={curve.map((c) => `${cx(c.e)},${cy(c.exact)}`).join(' ')} fill="none" stroke="var(--ink)" strokeWidth={1.8} />
          <circle cx={cx(eps)} cy={cy(expectedTokensPerPass(P, Q, gamma))} r={6} fill="var(--accent)" />
          <text x={CW - CR - 4} y={CT + 12} textAnchor="end" fontSize={11} style={{ fill: 'var(--ink-2)' }}>solid: exact for these toy models</text>
          <text x={CW - CR - 4} y={CT + 27} textAnchor="end" fontSize={11} style={{ fill: 'var(--ink-3)' }}>dashed: (1 − α^(γ+1)) / (1 − α), the paper’s i.i.d. estimate</text>
        </svg>
      </div>
      <p className="lab-note">Even an unrelated draft yields more than one token per pass, and the text is distributed exactly as the target would have written it. What a bad draft costs is speed: every round also pays for {gamma} draft steps.</p>
    </Lab>
  )
}
