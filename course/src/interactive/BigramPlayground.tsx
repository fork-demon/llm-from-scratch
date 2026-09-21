// The count-table bigram model (Model A of bigram_lm.py): look at one row,
// roll the weighted die, and watch text grow one character at a time.
import { useMemo, useState } from 'react'
import { Bars, Lab } from '../components/ui'
import { CORPUS_BASE, CORPUS_REPEAT, SMOOTHING, buildVocab, countModel, countPairs, crossEntropyOfTable, perplexity, rollNext, toIds } from '../lib/bigram'
import { makeRng } from '../lib/rng'

const show = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '⏎' : c)
const MAX_CORPUS = 3000

interface Gen { seed: number; start: string; text: string; lastU: number | null; rolls: number }

export function BigramPlayground() {
  const [corpus, setCorpus] = useState(CORPUS_BASE.trim())
  const [greedy, setGreedy] = useState(false)
  const [gen, setGen] = useState<Gen>({ seed: 7, start: 't', text: 't', lastU: null, rolls: 0 })

  const model = useMemo(() => {
    const text = (corpus.slice(0, MAX_CORPUS) + ' ').repeat(CORPUS_REPEAT)
    const vocab = buildVocab(text)
    const ids = toIds(text, vocab)
    const V = vocab.chars.length
    const table = countModel(ids, V, SMOOTHING)
    const ce = ids.length > 1 ? crossEntropyOfTable(table, ids) : 0
    return { vocab, V, table, counts: countPairs(ids, V), ce, length: text.length }
  }, [corpus])

  const { vocab, table, counts } = model
  const lastChar = Array.from(gen.text).pop() ?? ''
  const cur = vocab.stoi.get(lastChar) ?? 0
  const curChar = vocab.chars[cur]
  const prevChar = Array.from(gen.text).slice(-2, -1)[0]
  const prev = gen.rolls > 0 && prevChar !== undefined ? vocab.stoi.get(prevChar) : undefined

  const restart = (start: string, seed = gen.seed) => setGen({ seed, start, text: start, lastU: null, rolls: 0 })

  const roll = (n: number) => {
    // replay the seeded generator up to where we are, so the run is reproducible
    const rng = makeRng(gen.seed)
    for (let i = 0; i < gen.rolls; i++) rng.next()
    let text = gen.text
    let c = cur
    let u = 0
    for (let i = 0; i < n; i++) {
      u = rng.next()
      c = rollNext(table, c, u, greedy).next
      text += vocab.chars[c]
    }
    setGen({ ...gen, text, lastU: u, rolls: gen.rolls + n })
  }

  // the row that produced the last character (so the sampled bar can be pointed at), else the current row
  const shownRow = prev !== undefined ? prev : cur
  const shownChar = vocab.chars[shownRow]
  const sampled = prev !== undefined ? cur : -1
  const order = table[shownRow].map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p)
  const top = order.slice(0, 8)
  if (sampled >= 0 && !top.some((t) => t.i === sampled)) top.push(order.find((t) => t.i === sampled)!)
  const restP = 1 - top.reduce((s, t) => s + t.p, 0)
  const rowTotal = counts[shownRow].reduce((s, c) => s + c, 0)

  return (
    <Lab
      title="A language model made of counts"
      goal={<>Pick a character to see what usually follows it in the training text. Then press <b>Roll once</b> repeatedly: each press is one prediction plus one roll of a weighted die. That loop is text generation.</>}
    >
      <div className="matrix-cap">Current character (the only thing this model can see)</div>
      <div className="steps" role="group" aria-label="Choose the current character" style={{ flexWrap: 'wrap' }}>
        {vocab.chars.map((c) => (
          <button key={c} className="step-btn" style={{ minWidth: 34, padding: '4px 8px', fontFamily: 'var(--mono)' }} aria-pressed={gen.rolls === 0 && c === curChar} aria-label={c === ' ' ? 'space' : c} onClick={() => restart(c)}>{show(c)}</button>
        ))}
      </div>

      <div className="grid-2" style={{ marginTop: 12 }}>
        <div>
          <div className="matrix-cap">
            {sampled >= 0
              ? <>Row of ‘{show(shownChar)}’: the die that was just rolled</>
              : <>Row of ‘{show(shownChar)}’: P(next character | ‘{show(shownChar)}’)</>}
          </div>
          <Bars
            items={[
              ...top.map(({ p, i }) => ({ label: `${show(vocab.chars[i])}${i === sampled ? ' ◀' : ''}  (${counts[shownRow][i]}×)`, value: p, tone: i === sampled ? ('accent' as const) : ('neutral' as const) })),
              ...(restP > 0.0005 ? [{ label: 'all others', value: restP, dim: true }] : []),
            ]}
            max={1}
          />
          <p className="lab-note" style={{ marginTop: 6 }}>
            ‘{show(shownChar)}’ was followed by something {rowTotal} times. Each probability is (count + {SMOOTHING}) ÷ ({rowTotal} + {model.V} × {SMOOTHING}).
            {sampled >= 0 && gen.lastU !== null && !greedy && <> The die gave u = {gen.lastU.toFixed(3)}, which fell in the slice of ‘{show(vocab.chars[sampled])}’ (marked ◀).</>}
            {sampled >= 0 && greedy && <> No die: the biggest bar always wins.</>}
          </p>
        </div>
        <div>
          <div className="matrix-cap">Generated text (seed {gen.seed})</div>
          <div className="card mono" aria-live="polite" style={{ minHeight: 96, fontSize: 15, wordBreak: 'break-all', whiteSpace: 'pre-wrap' }}>
            {gen.text.slice(0, -1)}<b className="acc" style={{ textDecoration: 'underline' }}>{show(gen.text.slice(-1))}</b>
          </div>
          <div className="btn-row" style={{ marginTop: 8 }}>
            <button className="btn small primary" onClick={() => roll(1)}>Roll once</button>
            <button className="btn small" onClick={() => roll(40)}>Roll 40×</button>
            <button className="btn small" onClick={() => restart(gen.start)}>Start over</button>
            <button className="btn small" onClick={() => restart(gen.start, gen.seed + 1)}>New seed</button>
          </div>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5, marginTop: 8 }}>
            <input type="checkbox" checked={greedy} onChange={(e) => setGreedy(e.target.checked)} /> No die: always take the most likely character
          </label>
        </div>
      </div>

      <div className="readout" aria-live="polite" style={{ marginTop: 14 }}>
        <span>training text: <b>{model.length}</b> characters, <b>{model.V}</b> different ones</span>
        <span>table: <b>{model.V} × {model.V}</b> = {model.V * model.V} numbers</span>
        <span>average surprise (cross-entropy): <b>{model.ce.toFixed(4)}</b></span>
        <span>perplexity: <b>{perplexity(model.ce).toFixed(2)}</b></span>
      </div>

      <details className="deep" style={{ marginTop: 12 }}>
        <summary>Edit the training text</summary>
        <div className="details-body">
          <label className="sr-only" htmlFor="bigram-corpus">Training text</label>
          <textarea id="bigram-corpus" className="input" style={{ minHeight: 120, fontFamily: 'var(--mono)', fontSize: 13 }} value={corpus} onChange={(e) => { setCorpus(e.target.value); setGen({ ...gen, text: gen.start, lastU: null, rolls: 0 }) }} />
          <p className="lab-note">As in bigram_lm.py, the text is used {CORPUS_REPEAT} times over. With the original text you get exactly the numbers the Python file prints: 1.7518 and 5.76. Paste in some code or another language and watch the rows and the gibberish change character.</p>
        </div>
      </details>
    </Lab>
  )
}
