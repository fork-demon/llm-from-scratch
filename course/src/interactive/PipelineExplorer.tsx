// Lesson 0.1: the whole prompt -> answer pipeline, every stage clickable,
// with the actual (toy) data at that stage and a real generation loop.
import { useState } from 'react'
import { Bars, Flow, Lab, MatrixView } from '../components/ui'
import { fmt } from '../lib/math'
import { EMBED_DIMS, END, MAX_NEW_TOKENS, PROMPT, VOCAB, decode, embed, encode, generateStep, toyLogits, uniformAt, type PickMode } from '../lib/pipeline'

const STAGES = ['Text', 'Tokenizer', 'Tokens', 'Token IDs', 'Embeddings', 'Transformer', 'Next-token probabilities', 'Sampling', 'Next token', 'Repeat'] as const
const LATER: { id: string; label: string }[][] = [
  [],
  [{ id: 'tokenization', label: 'Tokenization' }],
  [{ id: 'tokenization', label: 'Tokenization' }],
  [{ id: 'tokenization', label: 'Tokenization' }],
  [{ id: 'embeddings', label: 'Embeddings' }],
  [{ id: 'attention', label: 'Attention' }, { id: 'transformer-block', label: 'The Transformer block' }, { id: 'build-gpt', label: 'Build GPT' }],
  [{ id: 'softmax', label: 'Softmax' }, { id: 'next-token', label: 'Predicting the next token' }],
  [{ id: 'inference', label: 'Inference: sampling and the KV cache' }],
  [{ id: 'next-token', label: 'Predicting the next token' }],
  [{ id: 'inference', label: 'Inference: sampling and the KV cache' }],
]

const show = (tok: string) => tok.replace(/ /g, '·')
const PROMPT_IDS = encode(PROMPT)

export function PipelineExplorer() {
  const [stage, setStage] = useState(0)
  const [generated, setGenerated] = useState<number[]>([])
  const [seed, setSeed] = useState(1)
  const [mode, setMode] = useState<PickMode>('sample')

  const ids = [...PROMPT_IDS, ...generated]
  const finished = generated[generated.length - 1] === END || generated.length >= MAX_NEW_TOKENS
  // what the *next* turn of the loop will do, given everything so far
  // (once <end> has been picked we keep showing that last turn)
  const ended = generated[generated.length - 1] === END
  const turn = ended ? generated.length - 1 : generated.length
  const context = ended ? ids.slice(0, -1) : ids
  const step = generateStep(context, mode, uniformAt(seed, turn))
  const tokens = context.map((i) => VOCAB[i])
  const text = decode(context)
  const logits = toyLogits(step.probs)
  const candidates = step.probs.map((p, i) => ({ i, p })).filter((c) => c.p > 0).sort((a, b) => b.p - a.p)

  const reset = () => setGenerated([])
  const generate = () => { if (!finished) setGenerated([...generated, step.next]) }
  const runAll = () => {
    const out = generated.slice()
    while (out[out.length - 1] !== END && out.length < MAX_NEW_TOKENS) {
      out.push(generateStep([...PROMPT_IDS, ...out], mode, uniformAt(seed, out.length)).next)
    }
    setGenerated(out)
  }

  // cumulative ranges in vocabulary order: this is exactly how sampleIndex walks the list
  let cum = 0
  const ranges = step.probs.map((p, i) => ({ i, p, from: cum, to: (cum += p) })).filter((r) => r.p > 0)

  return (
    <Lab
      title="From prompt to answer, one stage at a time"
      goal={<>Click any stage to see the <b>actual data</b> at that point for the text so far. Then press <b>Generate next token</b> and watch every stage update. Press it again. And again. That loop is how every answer you have ever seen from an LLM was written.</>}
    >
      <p className="lab-note">
        <b>Honest labels:</b> the vocabulary has {VOCAB.length} entries, embeddings have 4 numbers, and the “Transformer” is a small hand-written probability table. A real model has about 100,000 tokens, thousands of numbers per token, and computes the probabilities with billions of learned numbers. The <em>stages and the loop</em> are the real ones.
      </p>

      <div className="readout" aria-live="polite" style={{ marginBottom: 12, alignItems: 'center' }}>
        <span style={{ whiteSpace: 'pre-wrap' }}>
          {PROMPT}
          <b>{decode(generated)}</b>
          {!finished && <span className="muted"> ▌</span>}
        </span>
        <span className="muted">{generated.length === 0 ? 'nothing generated yet' : `${generated.length} token${generated.length === 1 ? '' : 's'} generated${generated[generated.length - 1] === END ? ', ended with <end>' : finished ? ', hit the length limit' : ''}`}</span>
      </div>

      <div className="btn-row" style={{ marginBottom: 14 }}>
        <button className="btn small primary" onClick={generate} disabled={finished}>Generate next token</button>
        <button className="btn small" onClick={runAll} disabled={finished}>Run to the end</button>
        <button className="btn small" onClick={reset} disabled={generated.length === 0}>Reset</button>
      </div>

      <div className="controls">
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}>
          Random seed
          <input className="input mono" type="number" min={1} max={999} value={seed} style={{ width: 90 }} onChange={(e) => { setSeed(Math.max(1, Math.floor(Number(e.target.value)) || 1)); reset() }} />
        </label>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}>
          <input type="checkbox" checked={mode === 'greedy'} onChange={(e) => { setMode(e.target.checked ? 'greedy' : 'sample'); reset() }} /> Always take the most likely token (no randomness)
        </label>
      </div>

      <Flow horizontal steps={STAGES.map((s) => ({ label: s }))} active={stage} onSelect={setStage} />

      <div className="card" aria-live="polite" style={{ marginTop: 8 }}>
        <h4 style={{ fontSize: 17, marginBottom: 8 }}>{stage + 1}. {STAGES[stage]}</h4>

        {stage === 0 && (
          <>
            <p>This is all the model is given: a string. Your prompt, plus everything it has written so far.</p>
            <div className="readout"><span>"{text}"</span><span className="muted">{text.length} characters</span></div>
            <p className="lab-note" style={{ marginTop: 10 }}>A neural network can only do arithmetic. You cannot multiply a string, so the next few stages turn it into numbers.</p>
          </>
        )}

        {stage === 1 && (
          <>
            <p>The tokenizer is an ordinary program (no neural network) that cuts text into pieces from a fixed list. Our toy rule: a word keeps the space in front of it, punctuation stands alone.</p>
            <div className="readout"><span>"{PROMPT}"</span><span>→</span><span>{encode(PROMPT).length} pieces</span></div>
            <p className="lab-note" style={{ marginTop: 10 }}>It runs once, on your prompt. Everything the model generates afterwards is already a token. Real tokenizers cut rarer words into several sub-word pieces.</p>
          </>
        )}

        {stage === 2 && (
          <>
            <p>The pieces are called <b>tokens</b>. The dot “·” marks a space that belongs to the token.</p>
            <div>{tokens.map((t, i) => <span key={i} className="token" style={i >= PROMPT_IDS.length ? { borderColor: 'var(--accent)' } : undefined}>{show(t)}</span>)}</div>
            <p className="lab-note" style={{ marginTop: 10 }}>{tokens.length} tokens so far. Outlined tokens were written by the model, not by you. To the model there is no difference.</p>
          </>
        )}

        {stage === 3 && (
          <>
            <p>Each token is replaced by its row number in the vocabulary. From here on, the model never sees letters again.</p>
            <div className="table-scroll">
              <table className="plain mono" style={{ fontSize: 13.5 }}>
                <tbody>
                  <tr><td>token</td>{tokens.map((t, i) => <td key={i}>{show(t)}</td>)}</tr>
                  <tr><td>id</td>{context.map((id, i) => <td key={i}><b>{id}</b></td>)}</tr>
                </tbody>
              </table>
            </div>
            <details className="deep"><summary>The whole toy vocabulary ({VOCAB.length} entries)</summary>
              <div className="details-body">{VOCAB.map((t, i) => <span key={i} className="token">{i}: {show(t)}</span>)}</div>
            </details>
          </>
        )}

        {stage === 4 && (
          <>
            <p>An id is just a label: 4 is not “twice” 2. So each id is swapped for a row of numbers from a big table. That row is the token’s <b>embedding</b>.</p>
            <MatrixView m={embed(context)} rows={tokens.map(show)} cols={[...EMBED_DIMS]} heat tone="accent" digits={1} caption={`${context.length} tokens × 4 numbers`} />
            <p className="lab-note" style={{ marginTop: 10 }}>We hand-picked these numbers and gave the columns names so you can read them. In a real model the numbers are learned, there are thousands per token, and no column has a clean human meaning.</p>
          </>
        )}

        {stage === 5 && (
          <>
            <p>The Transformer takes that whole table of numbers ({context.length} rows × 4) and produces one <b>score for every entry in the vocabulary</b>: how well would this token fit next?</p>
            <div className="table-scroll">
              <table className="plain mono" style={{ fontSize: 13.5 }}>
                <thead><tr><th>candidate next token</th><th>score</th></tr></thead>
                <tbody>
                  {candidates.map((c) => <tr key={c.i}><td>{show(VOCAB[c.i])}</td><td>{fmt(logits[c.i])}</td></tr>)}
                  <tr className="muted"><td>the other {VOCAB.length - candidates.length} tokens</td><td>extremely low</td></tr>
                </tbody>
              </table>
            </div>
            <p className="lab-note"><b>This is the stage we fake.</b> Here the scores come from a hand-written table that looks at the last one or two tokens. In a real LLM this box is where the billions of learned numbers live, and it reads the <em>entire</em> text so far. Most of this course is about opening this box.</p>
          </>
        )}

        {stage === 6 && (
          <>
            <p>Scores become probabilities: all positive, adding up to 100%. This is the model’s real output. Not a sentence. A probability for every possible next token.</p>
            <Bars items={candidates.map((c) => ({ label: show(VOCAB[c.i]), value: c.p }))} max={1} />
            <p className="lab-note" style={{ marginTop: 10 }}>After “{show(tokens[tokens.length - 1])}”. The other {VOCAB.length - candidates.length} tokens get 0% here. A real model gives every token at least a sliver.</p>
          </>
        )}

        {stage === 7 && (
          <>
            {mode === 'greedy' ? (
              <p>Randomness is off, so the rule is simple: take the token with the highest probability. Same prompt, same answer, every time.</p>
            ) : (
              <>
                <p>Now one token is picked <b>at random, weighted by probability</b>. Lay the probabilities end to end from 0 to 1, draw a random number, and see where it lands.</p>
                <div className="table-scroll">
                  <table className="plain mono" style={{ fontSize: 13.5 }}>
                    <thead><tr><th>token</th><th>probability</th><th>owns the range</th><th /></tr></thead>
                    <tbody>
                      {ranges.map((r) => {
                        const hit = r.i === step.next
                        return <tr key={r.i} style={hit ? { background: 'var(--accent-soft)' } : undefined}><td>{show(VOCAB[r.i])}</td><td>{(r.p * 100).toFixed(0)}%</td><td>{fmt(r.from)} to {fmt(r.to)}</td><td>{hit ? '← the random number lands here' : ''}</td></tr>
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            )}
            <div className="readout">
              {step.u !== null && <span>random number: <b>{fmt(step.u, 3)}</b> (seed {seed}, draw {turn + 1})</span>}
              <span>picked: <b>{show(VOCAB[step.next])}</b></span>
            </div>
            <p className="lab-note" style={{ marginTop: 10 }}>The sampler is a few lines of ordinary code <em>outside</em> the neural network. The seed only makes this demo repeatable.</p>
          </>
        )}

        {stage === 8 && (
          <>
            <p>The chosen id is looked up in the vocabulary to get text back, and appended.</p>
            <div className="readout"><span>id <b>{step.next}</b></span><span>→</span><span>token <b>{show(VOCAB[step.next])}</b></span></div>
            <p style={{ marginTop: 10 }}>{finished ? 'The answer is finished. Press Reset to run it again.' : step.next === END ? <>This one is special: <span className="mono">&lt;end&gt;</span> means “stop”. It is a token like any other, and the model predicts it like any other.</> : <>Press <b>Generate next token</b> to append it. One forward pass through the whole model bought us exactly one token.</>}</p>
          </>
        )}

        {stage === 9 && (
          <>
            <p>The new token joins the input, and the <em>whole pipeline runs again</em> for the token after that. The model’s own output becomes its input.</p>
            <ol style={{ fontFamily: 'var(--mono)', fontSize: 13.5 }}>
              {generated.length === 0 && <li className="muted" style={{ listStyle: 'none' }}>No turns of the loop yet. Press Generate next token.</li>}
              {generated.map((g, n) => <li key={n}>"{decode([...PROMPT_IDS, ...generated.slice(0, n)])}" → <b>{show(VOCAB[g])}</b></li>)}
            </ol>
            <p className="lab-note">It stops when the model picks <span className="mono">&lt;end&gt;</span> or a length limit is hit (here {MAX_NEW_TOKENS} tokens).</p>
          </>
        )}

        {LATER[stage].length > 0 && (
          <p className="lab-note" style={{ marginTop: 12, marginBottom: 0 }}>
            Gets its own lesson: {LATER[stage].map((l, i) => <span key={l.id}>{i > 0 && ', '}<a href={`#/lesson/${l.id}`}>{l.label}</a></span>)}.
          </p>
        )}
      </div>

      <div className="btn-row" style={{ marginTop: 12 }}>
        <button className="btn small" disabled={stage === 0} onClick={() => setStage(stage - 1)}>Previous stage</button>
        <button className="btn small" disabled={stage === STAGES.length - 1} onClick={() => setStage(stage + 1)}>Next stage</button>
      </div>
    </Lab>
  )
}
