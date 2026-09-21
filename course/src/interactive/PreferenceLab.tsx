// You are the human labeller. Your clicks train a tiny Bradley-Terry reward model live,
// then a policy is optimised against it (closed form, with a KL leash to the SFT model).
import { useMemo, useState } from 'react'
import { Bars, Lab, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import { CANDIDATES, CANDIDATE_PROMPT, COMPARISONS, FEATURES, featurise, klDivergence, personaChoice, preferProb, reward, trainOnChoices, tunedPolicy, type Choice, type Persona } from '../lib/preference'

const short = (s: string, n = 58) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)

export function PreferenceLab() {
  const [choices, setChoices] = useState<Choice[]>([])
  const [beta, setBeta] = useState(0.5)

  const w = useMemo(() => trainOnChoices(COMPARISONS, choices), [choices])
  const i = choices.length
  const done = i >= COMPARISONS.length
  const cmp = COMPARISONS[Math.min(i, COMPARISONS.length - 1)]
  const fa = featurise(cmp.a, cmp.key)
  const fb = featurise(cmp.b, cmp.key)
  const pA = preferProb(reward(w, fa), reward(w, fb))

  const refProbs = CANDIDATES.map((c) => c.refProb)
  const rewards = CANDIDATES.map((c) => reward(w, featurise(c.text, CANDIDATE_PROMPT.key)))
  const policy = tunedPolicy(refProbs, rewards, beta)
  const kl = klDivergence(policy, refProbs)
  const pCorrect = policy.reduce((s, p, j) => s + p * featurise(CANDIDATES[j].text, CANDIDATE_PROMPT.key)[2], 0)
  const refCorrect = refProbs.reduce((s, p, j) => s + p * featurise(CANDIDATES[j].text, CANDIDATE_PROMPT.key)[2], 0)

  const runPersona = (p: Persona) => setChoices(COMPARISONS.map((c) => personaChoice(c, p)))
  const maxW = Math.max(1, ...w.map(Math.abs))

  return (
    <Lab
      title="Train a reward model with your own preferences"
      goal={<>You are the human rater. For each prompt, click the answer you prefer. Every click is one gradient step on three reward weights. First answer honestly. Then reset and <b>always pick the longer answer</b>, and look at what the reward model learned.</>}
    >
      <h4 style={{ fontSize: 16, margin: '0 0 6px' }}>Step 1 · Compare ({Math.min(i + 1, COMPARISONS.length)} of {COMPARISONS.length})</h4>
      {!done ? (
        <>
          <p style={{ marginBottom: 8 }}><span className="muted">Prompt:</span> <b>{cmp.prompt}</b></p>
          <div className="grid-2">
            {(['a', 'b'] as const).map((side) => {
              const f = side === 'a' ? fa : fb
              return (
                <button key={side} className="card" onClick={() => setChoices([...choices, side])} style={{ textAlign: 'left', cursor: 'pointer', font: 'inherit', color: 'inherit' }} aria-label={`Prefer answer ${side.toUpperCase()}: ${side === 'a' ? cmp.a : cmp.b}`}>
                  <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>ANSWER {side.toUpperCase()} · click to prefer</div>
                  <div>{side === 'a' ? cmp.a : cmp.b}</div>
                  <div className="mono muted" style={{ fontSize: 12, marginTop: 8 }}>features: length {fmt(f[0])} · polite {f[1]} · has answer {f[2]}</div>
                </button>
              )
            })}
          </div>
          <p className="lab-note" style={{ marginTop: 8 }} aria-live="polite">
            Before your click, the reward model predicts: P(A preferred) = <b>{fmt(pA)}</b>. {i === 0 ? 'All weights are 0, so it has no opinion yet.' : pA > 0.65 ? 'It already expects you to pick A.' : pA < 0.35 ? 'It already expects you to pick B.' : 'It is unsure.'} A surprising click moves the weights a lot, an expected click barely at all.
          </p>
        </>
      ) : (
        <p aria-live="polite"><b>All {COMPARISONS.length} comparisons done.</b> Look at the weights below, then at step 3.</p>
      )}

      <div className="btn-row" style={{ margin: '6px 0 16px' }}>
        <button className="btn small" onClick={() => setChoices(choices.slice(0, -1))} disabled={i === 0}>Undo</button>
        <button className="btn small" onClick={() => setChoices([])} disabled={i === 0}>Reset</button>
        <span className="muted" style={{ fontSize: 13.5, alignSelf: 'center' }}>or let a simulated rater answer all 8:</span>
        <button className="btn small" onClick={() => runPersona('correct')}>always prefers the correct answer</button>
        <button className="btn small" onClick={() => runPersona('longer')}>always prefers the longer answer</button>
        <button className="btn small" onClick={() => runPersona('polite')}>always prefers the polite answer</button>
      </div>

      <h4 style={{ fontSize: 16, margin: '0 0 6px' }}>Step 2 · What the reward model has learned</h4>
      <p className="muted" style={{ fontSize: 14.5, marginBottom: 6 }}>reward(answer) = w · features. Positive weight: “raters like this”. Negative: “raters dislike this”.</p>
      <div className="table-scroll" aria-live="polite">
        <table className="plain" style={{ fontSize: 14.5 }}>
          <thead><tr><th>feature</th><th>weight</th><th style={{ width: '50%' }}>−  0  +</th></tr></thead>
          <tbody>
            {FEATURES.map((name, j) => (
              <tr key={name}>
                <td>{name}</td>
                <td className="mono"><b>{fmt(w[j])}</b></td>
                <td>
                  <svg viewBox="0 0 200 14" style={{ width: '100%', height: 14, display: 'block' }} role="img" aria-label={`${name} weight ${fmt(w[j])}`}>
                    <line x1={100} y1={0} x2={100} y2={14} className="axis" />
                    <rect x={w[j] >= 0 ? 100 : 100 + (w[j] / maxW) * 96} y={3} width={Math.abs(w[j] / maxW) * 96} height={8} fill={w[j] >= 0 ? 'var(--good)' : 'var(--bad)'} />
                  </svg>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4 style={{ fontSize: 16, margin: '18px 0 6px' }}>Step 3 · Optimise the model against that reward</h4>
      <p style={{ marginBottom: 6 }}>Prompt: <b>{CANDIDATE_PROMPT.prompt}</b> Five answers the SFT model might give, how your reward model scores them, and what the tuned model does.</p>
      <Slider label="KL leash β (how strongly the tuned model is held near the SFT model)" value={beta} min={0.1} max={5} step={0.1} onChange={setBeta} format={(v) => v.toFixed(1)} />
      <div className="table-scroll">
        <table className="plain" style={{ fontSize: 14 }}>
          <thead><tr><th>candidate answer</th><th>reward</th><th>SFT model</th><th>tuned model</th></tr></thead>
          <tbody>
            {CANDIDATES.map((c, j) => (
              <tr key={j}>
                <td title={c.text}>{short(c.text)} <span className="muted">({featurise(c.text, CANDIDATE_PROMPT.key)[2] ? 'says Paris' : 'never says Paris'})</span></td>
                <td className="mono">{fmt(rewards[j])}</td>
                <td className="mono">{(c.refProb * 100).toFixed(0)}%</td>
                <td className="mono"><b>{(policy[j] * 100).toFixed(0)}%</b></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Bars items={CANDIDATES.map((c, j) => ({ label: `${featurise(c.text, CANDIDATE_PROMPT.key)[2] ? 'Paris' : 'no Paris'}: ${short(c.text, 16)}`, value: policy[j], tone: featurise(c.text, CANDIDATE_PROMPT.key)[2] ? 'accent' as const : 'neutral' as const }))} max={1} digits={0} />
      <div className="readout" aria-live="polite">
        <span>chance the answer contains “Paris”: SFT <b>{(refCorrect * 100).toFixed(0)}%</b> → tuned <b>{(pCorrect * 100).toFixed(0)}%</b></span>
        <span>distance from SFT model (KL): <b>{fmt(kl)}</b></span>
      </div>
      {i > 0 && pCorrect < refCorrect - 0.05 && (
        <p className="lab-note" style={{ marginTop: 8 }}><b>Reward hacking, in miniature.</b> The reward went up and the answers got worse. The optimiser did its job perfectly: it found what the reward model likes. The reward model is only a stand-in for what people actually want.</p>
      )}
      <p className="lab-note" style={{ marginTop: 10 }}>
        <b>What is real and what is toy:</b> the comparison loss and its gradient step are the real Bradley-Terry ones. The “tuned model” column is the exact mathematical optimum of “reward minus β × KL”, computed over just five candidate answers. A real reward model is a full Transformer, not three hand-made features, and a real policy is tuned by many small gradient steps over all possible texts. The SFT percentages are hand-made.
      </p>
    </Lab>
  )
}
