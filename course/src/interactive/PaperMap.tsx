// "Attention Is All You Need" as a clickable outline: plain summary, the equation to look at,
// where you built it, what differs from your build, and a progress mark per section.
import { useEffect, useState, type ReactNode } from 'react'
import { Equation, Lab } from '../components/ui'
import { lessonById } from '../data/curriculum'
import { countMarks, loadMarks, MARKS, PAPER_URL, saveMarks, SECTIONS, setMark, type Mark, type Marks, type PaperSection } from '../lib/paperMap'
import { F } from './FormulaText'

const getStorage = (): Storage | undefined => {
  try { return window.localStorage } catch { return undefined }
}

const MARK_LABEL: Record<Mark, string> = { none: 'not yet', read: 'read', understood: 'understood', 're-derived': 're-derived' }

const EQUATIONS: Record<NonNullable<PaperSection['equation']>, ReactNode> = {
  attention: (
    <Equation label="Attention of Q, K, V equals softmax of Q K transpose over root d k, times V" symbols={[
      [<span className="q">Q</span>, 'queries, one row per position that is looking'],
      [<><span className="k">K</span>, <span className="v">V</span></>, 'keys and values, one row per position that is looked at. They need not come from the same sequence as Q'],
      [<>d<sub>k</sub></>, 'length of one key vector: 64 in the paper’s base model'],
      ['softmax', 'applied to each row, so every query’s weights sum to 1'],
    ]}>
      Attention(<span className="q">Q</span>, <span className="k">K</span>, <span className="v">V</span>) = softmax( <span className="q">Q</span><span className="k">K</span><sup>T</sup> / √d<sub>k</sub> ) <span className="v">V</span>
    </Equation>
  ),
  multihead: (
    <Equation label="MultiHead equals the concatenation of h heads times W O, where head i is attention on projected Q, K, V" symbols={[
      [<>head<sub>i</sub></>, 'one attention computation on its own small projection'],
      [<>W<sub>i</sub><sup>Q</sup>, W<sub>i</sub><sup>K</sup></>, <>projections for head i, each in ℝ<sup>d<sub>model</sub> × d<sub>k</sub></sup> = 512 × 64. The superscript is a label, not a power</>],
      [<>W<sub>i</sub><sup>V</sup></>, <>in ℝ<sup>d<sub>model</sub> × d<sub>v</sub></sup>, also 512 × 64</>],
      ['h', 'number of heads: 8'],
      ['Concat', 'glue the 8 outputs of 64 numbers back into 512'],
      [<>W<sup>O</sup></>, <>in ℝ<sup>h·d<sub>v</sub> × d<sub>model</sub></sup> = 512 × 512: lets the heads mix. Your <code>attn.proj</code></>],
    ]}>
      MultiHead(Q, K, V) = Concat(head<sub>1</sub>, …, head<sub>h</sub>) W<sup>O</sup><br />
      head<sub>i</sub> = Attention(QW<sub>i</sub><sup>Q</sup>, KW<sub>i</sub><sup>K</sup>, VW<sub>i</sub><sup>V</sup>)
    </Equation>
  ),
  ffn: (
    <Equation label="FFN of x equals max of zero and x W 1 plus b 1, times W 2, plus b 2" symbols={[
      ['x', 'the vector of one position: 512 numbers. Row convention, x on the left, as in this course'],
      [<>W<sub>1</sub>, b<sub>1</sub></>, 'up to 2048 numbers'],
      ['max(0, ·)', 'ReLU, written out'],
      [<>W<sub>2</sub>, b<sub>2</sub></>, 'back down to 512'],
    ]}>
      FFN(x) = max(0, xW<sub>1</sub> + b<sub>1</sub>) W<sub>2</sub> + b<sub>2</sub>
    </Equation>
  ),
  pe: (
    <Equation label="Positional encoding: sine for even dimensions, cosine for odd dimensions, of position over ten thousand to the power two i over d model" symbols={[
      ['pos', 'the position in the sequence: 0, 1, 2, …'],
      ['i', 'which pair of dimensions: 0 to 255 for a 512-number vector'],
      ['2i, 2i + 1', 'the even dimension of the pair gets the sine, the odd one the cosine'],
      [<>10000<sup>2i/d<sub>model</sub></sup></>, 'sets the wavelength of that pair: from 2π for the first pair up to 10000 · 2π for the last, in a geometric progression'],
    ]}>
      PE<sub>(pos, 2i)</sub> = sin( pos / 10000<sup>2i/d<sub>model</sub></sup> )<br />
      PE<sub>(pos, 2i+1)</sub> = cos( pos / 10000<sup>2i/d<sub>model</sub></sup> )
    </Equation>
  ),
  lrate: (
    <Equation label="Learning rate equals d model to the minus one half, times the minimum of step number to the minus one half and step number times warmup steps to the minus one point five" symbols={[
      ['step_num', 'the training step'],
      ['warmup_steps', '4000 in the paper'],
      [<>step_num · warmup_steps<sup>−1.5</sup></>, 'the smaller of the two while step_num < 4000: a straight line upwards'],
      [<>step_num<sup>−0.5</sup></>, 'the smaller of the two afterwards: decay like 1/√step'],
      [<>d<sub>model</sub><sup>−0.5</sup></>, <>wider models get a smaller rate. For d<sub>model</sub> = 512 the peak, at step 4000, is about 0.0007</>],
    ]}>
      lrate = d<sub>model</sub><sup>−0.5</sup> · min( step_num<sup>−0.5</sup>, step_num · warmup_steps<sup>−1.5</sup> )
    </Equation>
  ),
  postnorm: (
    <Equation label="The paper: LayerNorm of x plus Sublayer of x. Your block: x plus Sublayer of LayerNorm of x" symbols={[
      ['Sublayer', 'attention or the feed-forward network'],
      ['x + …', 'the residual connection'],
      ['paper', 'post-norm: add first, then normalise. The normalisation sits on the residual path'],
      ['your Block', 'pre-norm: normalise the input of the sub-layer only. The residual path stays a clean sum'],
    ]}>
      paper: LayerNorm( x + Sublayer(x) )<br />
      your Block: x + Sublayer( LayerNorm(x) )
    </Equation>
  ),
}

export function PaperMap() {
  const [active, setActive] = useState('abstract')
  const [marks, setMarks] = useState<Marks>({})
  const [saved, setSaved] = useState(true)
  useEffect(() => { setMarks(loadMarks(getStorage())) }, [])

  const section = SECTIONS.find((s) => s.id === active)!
  const counts = countMarks(marks)
  const mark = (m: Mark) => {
    const next = setMark(marks, section.id, m)
    setMarks(next)
    setSaved(saveMarks(getStorage(), next))
  }
  const idx = SECTIONS.indexOf(section)

  return (
    <Lab
      title="Paper map: Attention Is All You Need"
      goal={<>Open <a href={PAPER_URL} target="_blank" rel="noreferrer">the paper</a> in another window. Go through it section by section. For each one, read the paper’s text first, then check yourself against this map, and mark how far you got. Marks are kept in this browser.</>}
    >
      <div className="readout" aria-live="polite" style={{ marginBottom: 14 }}>
        <span>read: <b>{counts.read}</b> of {SECTIONS.length}</span>
        <span>understood: <b>{counts.understood}</b></span>
        <span>re-derived: <b>{counts['re-derived']}</b></span>
        {!saved && <span style={{ color: 'var(--bad)' }}>this browser blocks storage, so marks last only until you leave the page</span>}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, alignItems: 'flex-start' }}>
        <nav aria-label="Sections of the paper" style={{ flex: '1 1 230px', maxWidth: 320, display: 'grid', gap: 3 }}>
          {SECTIONS.map((s) => {
            const m = marks[s.id] ?? 'none'
            const on = s.id === active
            return (
              <button
                key={s.id}
                aria-pressed={on}
                onClick={() => setActive(s.id)}
                style={{
                  display: 'flex', gap: 8, alignItems: 'baseline', textAlign: 'left', padding: '5px 9px', marginLeft: s.depth * 14, borderRadius: 7, fontSize: 14,
                  border: `1px solid ${on ? 'var(--accent)' : 'transparent'}`, background: on ? 'var(--paper-2)' : 'transparent', color: 'var(--ink)', fontWeight: on ? 650 : 450,
                }}
              >
                <span className="mono" style={{ color: 'var(--ink-3)', fontSize: 12.5, minWidth: s.number.length > 3 ? 34 : s.number.length > 1 ? 24 : 12 }}>{s.number}</span>
                <span style={{ flex: 1 }}>{s.title}</span>
                {m !== 'none' && <span className="mono" style={{ fontSize: 11, color: m === 're-derived' ? 'var(--accent-ink)' : 'var(--ink-3)', whiteSpace: 'nowrap' }}>{m === 're-derived' ? '✓✓✓' : m === 'understood' ? '✓✓' : '✓'}<span className="sr-only"> {MARK_LABEL[m]}</span></span>}
              </button>
            )
          })}
        </nav>

        <div style={{ flex: '3 1 380px', minWidth: 0 }} aria-live="polite">
          <div className="mono" style={{ fontSize: 12.5, color: 'var(--ink-3)' }}>{section.number ? `Section ${section.number}` : 'Before section 1'}</div>
          <h4 style={{ fontSize: 19, margin: '2px 0 10px' }}>{section.title}</h4>

          <h5 style={{ fontSize: 13, margin: '0 0 4px', color: 'var(--ink-3)', fontWeight: 650 }}>What it says</h5>
          {section.summary.map((p, i) => <p key={i} style={{ marginTop: 0, fontSize: 15.5 }}><F>{p}</F></p>)}

          <h5 style={{ fontSize: 13, margin: '14px 0 4px', color: 'var(--ink-3)', fontWeight: 650 }}>Where to look</h5>
          <p style={{ marginTop: 0, fontSize: 15.5 }}><F>{section.look}</F></p>
          {section.equation && EQUATIONS[section.equation]}

          <h5 style={{ fontSize: 13, margin: '14px 0 4px', color: 'var(--ink-3)', fontWeight: 650 }}>You built this in</h5>
          <ul style={{ marginTop: 0, fontSize: 15 }}>
            {section.built.map((b) => {
              const l = lessonById(b.lesson)
              return <li key={b.lesson}>{l ? <a href={`#/lesson/${l.id}`}>{l.code} {l.title}</a> : b.lesson}: {b.note}</li>
            })}
          </ul>

          <h5 style={{ fontSize: 13, margin: '14px 0 4px', color: 'var(--ink-3)', fontWeight: 650 }}>What differs from your build</h5>
          <ul style={{ marginTop: 0, fontSize: 15 }}>{section.differs.map((d, i) => <li key={i}><F>{d}</F></li>)}</ul>

          <div className="steps" role="group" aria-label={`How far did you get with ${section.title}?`} style={{ marginTop: 16, marginBottom: 8 }}>
            {MARKS.map((m) => <button key={m} className="step-btn" aria-pressed={(marks[section.id] ?? 'none') === m} onClick={() => mark(m)}>{MARK_LABEL[m]}</button>)}
          </div>
          <p className="lab-note" style={{ marginTop: 0 }}>“Re-derived” means you closed the paper and rebuilt the section’s formula or argument on paper or in code.</p>
          <div className="btn-row">
            <button className="btn small" disabled={idx === 0} onClick={() => setActive(SECTIONS[idx - 1].id)}>← Previous section</button>
            <button className="btn small" disabled={idx === SECTIONS.length - 1} onClick={() => setActive(SECTIONS[idx + 1].id)}>Next section →</button>
          </div>
        </div>
      </div>
    </Lab>
  )
}
