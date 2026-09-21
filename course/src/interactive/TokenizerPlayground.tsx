// Train a BPE tokenizer one merge at a time, then use the tokenizer you trained.
import { useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { BPE_REPEAT, BPE_SAMPLE, UNK, encodeSafe, initTraining, mergeStep, utf8Bytes, type TrainState } from '../lib/bpe'

const show = (s: string) => s.replace(/ /g, '␣').replace(/\n/g, '⏎')
const PRESETS = ['the fox jumps over the lazy tokenizer', 'foxes', 'unbelievable', 'teh quikc borwn fox', 'getTokenById(x)', 'der schnelle Fuchs', 'नमस्ते']
const MAX_CORPUS = 1500

function TokenBox({ text, id, unk, alt }: { text: string; id?: number; unk?: boolean; alt: boolean }) {
  return (
    <span
      className="token"
      style={{
        background: unk ? 'transparent' : alt ? 'color-mix(in srgb, var(--accent) 16%, var(--paper-2))' : undefined,
        borderStyle: unk ? 'dashed' : undefined,
        borderColor: unk ? 'var(--bad)' : undefined,
        textAlign: 'center',
        lineHeight: 1.25,
      }}
    >
      {show(text)}
      {id !== undefined && <small style={{ display: 'block', fontSize: 10.5, color: unk ? 'var(--bad)' : 'var(--ink-3)' }}>{unk ? '<unk>' : id}</small>}
    </span>
  )
}

export function TokenizerPlayground() {
  const [corpus, setCorpus] = useState(BPE_SAMPLE.trim())
  const [repeat, setRepeat] = useState(true)
  const [target, setTarget] = useState(80)
  const [nMerges, setNMerges] = useState(0)
  const [text, setText] = useState(PRESETS[0])

  const trainingText = useMemo(() => {
    const one = corpus.slice(0, MAX_CORPUS)
    return repeat ? (one + ' ').repeat(BPE_REPEAT) : one
  }, [corpus, repeat])

  // every state from 0 merges up to the requested number (cheap: the corpus is tiny)
  const history = useMemo(() => {
    const states: TrainState[] = [initTraining(trainingText)]
    while (states.length <= nMerges && !states[states.length - 1].done) states.push(mergeStep(states[states.length - 1]))
    return states
  }, [trainingText, nMerges])

  const state = history[history.length - 1]
  const base = history[0].vocab.length
  const applied = state.merges.length
  const maxMerges = Math.max(0, target - base)
  const canMerge = !state.done && applied < maxMerges
  const last = state.merges[applied - 1]
  const startLen = history[0].ids.length

  const pieces = useMemo(() => encodeSafe(state, text), [state, text])
  const nChars = Array.from(text).length
  const unknown = Array.from(new Set(pieces.filter((p) => p.id === UNK).map((p) => p.text)))

  const reset = () => setNMerges(0)
  const preview = state.ids.slice(0, 60)

  return (
    <Lab
      title="Train a tokenizer, then use it"
      goal={<>Press <b>Merge one step</b> a few times and watch the most frequent pair become a new token. Then type your own text below and see how <em>your</em> tokenizer cuts it up.</>}
    >
      <h4 style={{ marginTop: 0 }}>1 · Train</h4>
      <label htmlFor="tok-corpus" className="muted" style={{ fontSize: 14 }}>Training text (edit it: training restarts)</label>
      <textarea id="tok-corpus" className="input" style={{ minHeight: 84, fontFamily: 'var(--mono)', fontSize: 13 }} value={corpus} onChange={(e) => { setCorpus(e.target.value); reset() }} />
      <div className="controls" style={{ marginTop: 10 }}>
        <Slider label="Target vocabulary size" value={Math.max(target, base)} min={base} max={base + 120} step={1} onChange={(v) => { setTarget(v); setNMerges(Math.min(nMerges, Math.max(0, v - base))) }} />
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}>
          <input type="checkbox" checked={repeat} onChange={(e) => { setRepeat(e.target.checked); reset() }} /> Repeat the text {BPE_REPEAT}× (as bpe_tokenizer.py does)
        </label>
      </div>
      <div className="btn-row">
        <button className="btn small primary" disabled={!canMerge} onClick={() => setNMerges(applied + 1)}>Merge one step</button>
        <button className="btn small" disabled={!canMerge} onClick={() => setNMerges(maxMerges)}>Run to target</button>
        <button className="btn small" disabled={applied === 0} onClick={() => setNMerges(applied - 1)}>Undo one</button>
        <button className="btn small" disabled={applied === 0} onClick={reset}>Reset</button>
      </div>

      <div className="readout" aria-live="polite" style={{ marginTop: 12 }}>
        <span>vocabulary: <b>{base}</b> characters + <b>{applied}</b> merges = <b>{state.vocab.length}</b></span>
        <span>training text: <b>{startLen}</b> → <b>{state.ids.length}</b> tokens</span>
        {last && <span>last merge: <b>'{show(state.vocab[last.pair[0]])}' + '{show(state.vocab[last.pair[1]])}' → '{show(state.vocab[last.newId])}'</b> (seen {last.count}×)</span>}
      </div>
      {state.done && <p className="lab-note">Training stopped by itself: no adjacent pair occurs twice any more, so there is nothing left worth merging.</p>}
      {!state.done && applied >= maxMerges && applied > 0 && <p className="lab-note">Target vocabulary size reached. Raise the slider to allow more merges.</p>}

      <div className="grid-2" style={{ marginTop: 12 }}>
        <div>
          <div className="matrix-cap">The start of the training text, as the tokenizer now sees it</div>
          <div aria-label="Training text split into current tokens" style={{ lineHeight: 2.1 }}>
            {preview.map((id, i) => <TokenBox key={i} text={state.vocab[id]} alt={i % 2 === 1} />)}
            {state.ids.length > preview.length && <span className="muted"> …</span>}
          </div>
        </div>
        <div>
          <div className="matrix-cap">The merge list: this ordered list <em>is</em> the trained tokenizer</div>
          {applied === 0 ? <p className="muted" style={{ fontSize: 14 }}>No merges yet. Every token is a single character.</p> : (
            <ol className="mono" style={{ fontSize: 13, maxHeight: 190, overflowY: 'auto', margin: 0, paddingLeft: 34 }} tabIndex={0} aria-label="Learned merges in order">
              {state.merges.map((m, i) => (
                <li key={i} style={{ margin: 0, fontWeight: i === applied - 1 ? 700 : 400 }}>
                  '{show(state.vocab[m.pair[0]])}' + '{show(state.vocab[m.pair[1]])}' → '{show(state.vocab[m.newId])}' <span className="muted">id {m.newId}, {m.count}×</span>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>

      <h4 style={{ marginTop: 22 }}>2 · Use the tokenizer you just trained</h4>
      <label className="sr-only" htmlFor="tok-text">Text to tokenize</label>
      <input id="tok-text" className="input" value={text} onChange={(e) => setText(e.target.value)} autoComplete="off" spellCheck={false} />
      <div className="steps" role="group" aria-label="Example texts" style={{ marginTop: 8 }}>
        {PRESETS.map((p) => <button key={p} className="step-btn" aria-pressed={text === p} onClick={() => setText(p)}>{p}</button>)}
      </div>
      <div aria-label="Your text split into tokens, with token IDs underneath" style={{ lineHeight: 2.6, marginTop: 6 }}>
        {pieces.map((p, i) => <TokenBox key={i} text={p.text} id={p.id} unk={p.id === UNK} alt={i % 2 === 1} />)}
      </div>
      <div className="readout" aria-live="polite">
        <span><b>{nChars}</b> characters</span>
        <span><b>{pieces.length}</b> tokens</span>
        <span><b>{pieces.length ? (nChars / pieces.length).toFixed(2) : '0.00'}</b> characters per token</span>
        <span>IDs: [{pieces.map((p) => (p.id === UNK ? '?' : p.id)).join(', ')}]</span>
      </div>
      {unknown.length > 0 && (
        <p className="lab-note" style={{ marginTop: 10 }}>
          <b>{unknown.map((c) => `“${c}”`).join(' ')}</b> never appeared in the training text, so {unknown.length === 1 ? 'it has' : 'they have'} no ID at all. The Python file crashes here with a <code>KeyError</code>; we show <code>&lt;unk&gt;</code> instead. Real tokenizers avoid the problem by starting from the 256 possible <em>bytes</em> instead of characters, so nothing is ever unknown: “{unknown[0]}” would become the {utf8Bytes(unknown[0]).length} byte{utf8Bytes(unknown[0]).length === 1 ? '' : 's'} <span className="mono">{utf8Bytes(unknown[0]).map((b) => '0x' + b.toString(16).toUpperCase().padStart(2, '0')).join(' ')}</span>, one token each.
        </p>
      )}
      <p className="lab-note">
        <b>Honest warning:</b> this tokenizer learned from a few hundred characters. GPT-style tokenizers run the same algorithm over bytes, on huge corpora, for 50,000 to 200,000 merges, so they split the same text very differently (usually into far fewer tokens).
      </p>
    </Lab>
  )
}
