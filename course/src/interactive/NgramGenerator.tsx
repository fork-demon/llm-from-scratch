// An n-gram text generator with a context-length dial: gibberish at one end, recitation at the other.
import { useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { CORPUS_BASE } from '../lib/bigram'
import { buildNgram, generateNgram, longestCopiedRun } from '../lib/ngram'
import { makeRng } from '../lib/rng'

const LENGTH = 220

export function NgramGenerator() {
  const [k, setK] = useState(1)
  const [seed, setSeed] = useState(11)

  const res = useMemo(() => {
    const g = generateNgram(CORPUS_BASE, k, LENGTH, makeRng(seed).next)
    return { ...g, copied: longestCopiedRun(g.text, CORPUS_BASE), contexts: buildNgram(CORPUS_BASE, k).size }
  }, [k, seed])

  const { start, length } = res.copied
  return (
    <Lab
      title="Turn up the context of a count model"
      goal={<>Same counting idea as the bigram, but the model now sees the last <b>n</b> characters. Slide from 1 to 6 and read the output each time. When does it start to look like English? When does it stop being <em>new</em>?</>}
    >
      <div className="controls">
        <Slider label="Context length n (characters)" value={k} min={1} max={6} step={1} onChange={setK} />
        <div className="btn-row"><button className="btn small" onClick={() => setSeed(seed + 1)}>Generate again (new seed)</button></div>
      </div>
      <div className="card mono" aria-live="polite" style={{ fontSize: 14.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
        {res.text.slice(0, start)}
        <span style={{ background: 'color-mix(in srgb, var(--accent) 18%, transparent)', textDecoration: 'underline' }}>{res.text.slice(start, start + length)}</span>
        {res.text.slice(start + length)}
      </div>
      <div className="readout" aria-live="polite" style={{ marginTop: 10 }}>
        <span>longest stretch copied word for word from the training text (underlined): <b>{length}</b> characters</span>
        <span>steps where only one continuation was ever seen: <b>{((res.forced / res.steps) * 100).toFixed(0)}%</b></span>
        <span>rows in the table: <b>{res.contexts}</b></span>
      </div>
      <p className="lab-note">
        When “only one continuation was ever seen”, the model is not choosing anything. It is reading its training text back to you. No smoothing is used here, so an unseen context simply has no row.
      </p>
    </Lab>
  )
}
