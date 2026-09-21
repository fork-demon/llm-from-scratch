// Trace one prompt through a complete (tiny, untrained) GPT: shape and real numbers at every stage.
import { useMemo, useState } from 'react'
import { Bars, Lab, MatrixView, Slider } from '../components/ui'
import { fmt, sampleIndex, softmax } from '../lib/math'
import { makeRng } from '../lib/rng'
import { CHARS, cropToContext, encode, gptForward, makeTinyGpt, showChar, TRACER_CONFIG } from '../lib/tinyGpt'

const cfg = TRACER_CONFIG
const STAGES = ['Text', 'Tokenizer', 'Token embeddings', '+ positions', ...Array.from({ length: cfg.nLayer }, (_, i) => `Block ${i + 1}`), 'Final LayerNorm', 'Head → logits', 'Last row', 'Softmax', 'Sample']
const B0 = 4 // index of the first block stage
const AFTER = B0 + cfg.nLayer // index of "Final LayerNorm"

export function GptTracer() {
  const model = useMemo(() => makeTinyGpt(), [])
  const [text, setText] = useState('the ca')
  const [stage, setStage] = useState(0)
  const [temperature, setTemperature] = useState(1)
  const [roll, setRoll] = useState(1)

  const allIds = encode(text)
  const ids = cropToContext(allIds, cfg.contextLen)
  const cropped = allIds.length - ids.length
  const trace = useMemo(() => (ids.length ? gptForward(model, ids) : null), [model, ids.join(',')]) // eslint-disable-line react-hooks/exhaustive-deps

  const promptBox = (
    <>
      <label htmlFor="gpt-tracer-text" style={{ fontSize: 14, color: 'var(--ink-3)' }}>Prompt (lower-case letters, space and ' , . ? only; the model sees the last {cfg.contextLen} characters)</label>
      <input id="gpt-tracer-text" className="input" value={text} onChange={(e) => setText(e.target.value)} style={{ fontFamily: 'var(--mono)' }} />
    </>
  )
  const goal = <>Type a short prompt and press “Next stage” until a new character comes out. At every stage, read the <b>shape</b> first, then peek at the numbers. The weights are random: this model has never been trained.</>

  if (!trace) {
    return (
      <Lab title="Trace one token through a GPT" goal={goal}>
        {promptBox}
        <p className="lab-note" style={{ marginTop: 8 }}>Type at least one letter (a to z, space, or one of <span className="mono">' , . ?</span>).</p>
      </Lab>
    )
  }

  const T = ids.length
  const D = cfg.nEmbd
  const V = cfg.vocabSize
  const rows = ids.map((i) => showChar(CHARS[i]))
  const last = trace.logits[T - 1]
  const probs = softmax(last, temperature)
  const u = makeRng(roll * 7919 + T).next()
  const picked = sampleIndex(probs, u)
  const ranked = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 8)

  const shapes = ['a string', `(${T})`, `(${T}, ${D})`, `(${T}, ${D})`, ...Array.from({ length: cfg.nLayer }, () => `(${T}, ${D})`), `(${T}, ${D})`, `(${T}, ${V})`, `(${V})`, `(${V})`, 'one id']
  const blockIdx = stage - B0

  return (
    <Lab
      title="Trace one token through a GPT"
      goal={goal}
    >
      {promptBox}
      <p className="lab-note" style={{ marginTop: 8 }}>
        This GPT: vocab_size {V}, context_len {cfg.contextLen}, n_embd {D}, n_head {cfg.nHead}, n_layer {cfg.nLayer}. Same architecture as <code>tiny_gpt.py</code>, shrunk so the numbers fit on screen (and ReLU instead of GELU).
      </p>

      <div className="steps" role="tablist" aria-label="Pipeline stages">
        {STAGES.map((s, i) => <button key={s} role="tab" className="step-btn" aria-selected={i === stage} onClick={() => setStage(i)}>{i}. {s}</button>)}
      </div>

      <div className="readout" aria-live="polite" style={{ marginBottom: 12 }}>
        <span>stage: <b>{STAGES[stage]}</b></span>
        <span>shape now: <b>{shapes[stage]}</b></span>
        <span>T = {T} tokens, D = {D}, V = {V}</span>
      </div>

      <div role="tabpanel" aria-live="polite">
        {stage === 0 && (
          <>
            <p>It starts as text. The model cannot multiply letters.</p>
            <div>{[...text].map((c, i) => <span key={i} className="token">{showChar(c)}</span>)}</div>
            {cropped > 0 && <p className="lab-note"><b>Cropped.</b> Your prompt has {allIds.length} tokens but the context window is {cfg.contextLen}. The first {cropped} are dropped and the model never sees them: <code>idx[:, -context_len:]</code>.</p>}
          </>
        )}
        {stage === 1 && (
          <>
            <p>Character-level tokenizer: every character is looked up in a fixed list of {V} characters and replaced by its index. Shape <span className="mono">({T})</span>: a list of {T} integers.</p>
            <div>{ids.map((id, i) => <span key={i} className="token">{rows[i]} → <b>{id}</b></span>)}</div>
            <p className="lab-note">Real models use <a href="#/lesson/tokenization">BPE tokens</a> instead of characters. Everything after this stage is identical.</p>
          </>
        )}
        {stage === 2 && (
          <>
            <p>Each id selects one row of the embedding table <span className="mono">({V}, {D})</span>. No arithmetic: a lookup. Identical characters get identical rows.</p>
            <div className="table-scroll"><MatrixView m={trace.tok} rows={rows} heat digits={2} caption={`tok_emb(idx)  (${T}, ${D})`} /></div>
          </>
        )}
        {stage === 3 && (
          <>
            <p>Add the vector for slot 0 to the first token, slot 1 to the second, and so on. Now identical characters in different places differ.</p>
            <div className="table-scroll"><MatrixView m={trace.x0} rows={rows} heat tone="accent" digits={2} caption={`x = tok_emb(idx) + pos_emb(pos)  (${T}, ${D})`} /></div>
          </>
        )}
        {stage >= B0 && stage < AFTER && (
          <>
            <p>Block {blockIdx + 1}: <span className="mono">x = x + attn(ln1(x))</span>, then <span className="mono">x = x + ffn(ln2(x))</span>. Tokens look at earlier tokens, then each one is processed alone. Shape in = shape out.</p>
            <div className="table-scroll"><MatrixView m={trace.blockOuts[blockIdx]} rows={rows} heat tone="accent" digits={2} caption={`x after block ${blockIdx + 1}  (${T}, ${D})`} /></div>
            <p className="lab-note">Compare with the previous stage: every row has moved a little. Each block adds a correction to what was there.</p>
          </>
        )}
        {stage === AFTER && (
          <>
            <p>One last LayerNorm so that the head receives vectors of a standard size.</p>
            <div className="table-scroll"><MatrixView m={trace.final} rows={rows} heat tone="accent" digits={2} caption={`ln_f(x)  (${T}, ${D})`} /></div>
          </>
        )}
        {stage === AFTER + 1 && (
          <>
            <p>The head is one matrix multiply: <span className="mono">({T}, {D}) @ ({D}, {V}) → ({T}, {V})</span>. Every row now holds {V} <b>logits</b>: one raw score for each character that could come next <em>after that position</em>.</p>
            <div className="table-scroll"><MatrixView m={trace.logits} rows={rows} cols={CHARS.map(showChar)} heat tone="accent" digits={1} highlight={(i) => i === T - 1} caption={`logits  (${T}, ${V})`} /></div>
            <p className="lab-note">All {T} rows are predictions. During training every row is graded. To generate, only the outlined last row matters.</p>
          </>
        )}
        {stage === AFTER + 2 && (
          <>
            <p>To continue the text we only need the prediction after the <em>last</em> token, “{rows[T - 1]}”: <code>logits[:, -1, :]</code>. Shape <span className="mono">({V})</span>.</p>
            <Bars items={ranked.map(({ i }) => ({ label: showChar(CHARS[i]), value: last[i] }))} percent={false} />
            <p className="lab-note">The 8 largest of the {V} logits. They are scores, not probabilities: they can be negative and do not sum to anything.</p>
          </>
        )}
        {stage >= AFTER + 3 && (
          <>
            <p>{stage === AFTER + 3 ? <>Softmax turns the {V} logits into {V} probabilities. Divide the logits by the temperature first.</> : <>Roll a weighted die over these probabilities.</>}</p>
            <div className="controls"><Slider label="temperature" value={temperature} min={0.2} max={2} step={0.1} onChange={setTemperature} /></div>
            <Bars items={ranked.map(({ p, i }) => ({ label: showChar(CHARS[i]), value: p, tone: stage === AFTER + 4 && i === picked ? 'accent' : undefined }))} max={Math.max(0.2, ranked[0].p)} />
            <p className="lab-note">Top 8 shown. All {V} probabilities sum to <span className="mono">{fmt(probs.reduce((a, b) => a + b, 0))}</span>. A uniform guess would be {(100 / V).toFixed(1)}% each.</p>
            {stage === AFTER + 4 && (
              <>
                <div className="readout">
                  <span>random number: <b>{fmt(u, 3)}</b></span>
                  <span>sampled id: <b>{picked}</b></span>
                  <span>next token: <b>“{showChar(CHARS[picked])}”</b> (p = {(probs[picked] * 100).toFixed(1)}%)</span>
                </div>
                <div className="btn-row" style={{ marginTop: 10 }}>
                  <button className="btn small" onClick={() => setRoll(roll + 1)}>Roll again</button>
                  <button className="btn small primary" onClick={() => { setText(text + CHARS[picked]); setRoll(roll + 1); setStage(0) }}>Append “{showChar(CHARS[picked])}” and go round again</button>
                </div>
                <p className="lab-note"><b>The prediction is noise.</b> These weights are random, so the model has no reason to prefer “t” after “the ca”. The pipeline is complete and correct; what is missing is training, which is the next lesson.</p>
              </>
            )}
          </>
        )}
      </div>

      <div className="btn-row" style={{ marginTop: 16 }}>
        <button className="btn small" disabled={stage === 0} onClick={() => setStage(stage - 1)}>Back</button>
        <button className="btn small primary" disabled={stage === STAGES.length - 1} onClick={() => setStage(stage + 1)}>Next stage</button>
      </div>
    </Lab>
  )
}
