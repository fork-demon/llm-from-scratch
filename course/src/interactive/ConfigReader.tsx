// Paste any config.json: every field explained, then the model sized from it
// (parameters by component, weight memory, KV cache, training memory).
import { useId, useState } from 'react'
import { Lab } from '../components/ui'
import { humanCount } from '../lib/params'
import { BYTES_PER_PARAM, CONFIG_PRESETS, countSpec, formatGB, kvTokenCap, parseConfig, specKvBytesPerToken, trainingBytes, weightBytes, type Precision } from '../lib/hfConfig'
import { lessonById } from '../data/curriculum'

const n = (x: number) => x.toLocaleString('en-US')
const PRECISIONS: { id: Precision; label: string }[] = [
  { id: 'fp32', label: 'fp32' }, { id: 'bf16', label: 'bf16 / fp16' }, { id: 'int8', label: 'int8' }, { id: 'int4', label: 'int4' },
]

export function ConfigReader() {
  const [text, setText] = useState(CONFIG_PRESETS[0].json)
  const [sizingOnly, setSizingOnly] = useState(false)
  const id = useId()
  const result = parseConfig(text)
  const preset = CONFIG_PRESETS.find((p) => p.json === text)

  return (
    <Lab
      title="Config reader"
      goal={<>Load a preset, read the explanation of each field, then <b>edit the JSON</b>. Predict first: in the Llama 3 file, change <code>num_key_value_heads</code> from 8 to 32. Which numbers below move, and by how much? You can also paste the <code>config.json</code> of any dense decoder-only model from the Hub.</>}
    >
      <div className="steps" role="group" aria-label="Preset config files">
        {CONFIG_PRESETS.map((p) => <button key={p.id} className="step-btn" aria-pressed={preset?.id === p.id} onClick={() => setText(p.json)}>{p.label}</button>)}
      </div>
      <p className="lab-note" style={{ marginTop: -6 }}>
        {preset ? <>The real file, as published: <a href={preset.source} target="_blank" rel="noreferrer">{preset.source.replace('https://huggingface.co/', '').replace('/blob/main/config.json', '')}</a> ({preset.note}).</> : 'Edited or pasted config.'}
      </p>

      <label htmlFor={`${id}-json`} className="sr-only">config.json</label>
      <textarea
        id={`${id}-json`}
        className="input"
        spellCheck={false}
        value={text}
        onChange={(e) => setText(e.target.value)}
        style={{ fontFamily: 'var(--mono)', fontSize: 12.5, lineHeight: 1.45, minHeight: 210, whiteSpace: 'pre', overflowWrap: 'normal', overflowX: 'auto' }}
      />

      {!result.ok ? (
        <p className="lab-note" role="alert" style={{ color: 'var(--bad)' }}>{result.error}</p>
      ) : (
        <ConfigReport result={result} sizingOnly={sizingOnly} setSizingOnly={setSizingOnly} />
      )}
    </Lab>
  )
}

function ConfigReport({ result, sizingOnly, setSizingOnly }: { result: Extract<ReturnType<typeof parseConfig>, { ok: true }>; sizingOnly: boolean; setSizingOnly: (b: boolean) => void }) {
  const { spec, fields, warnings } = result
  const b = countSpec(spec)
  const gqa = spec.kvHeads < spec.heads
  const kvOut = spec.kvHeads * spec.headDim
  const qOut = spec.heads * spec.headDim
  const kvTok = specKvBytesPerToken(spec)
  const cap = kvTokenCap(spec)
  const shown = sizingOnly ? fields.filter((f) => f.sizing) : fields

  const parts = [
    { label: 'Token embeddings', formula: `V × D = ${n(spec.vocab)} × ${n(spec.hidden)}`, value: b.tokenEmb, color: 'var(--ink-3)' },
    { label: 'Position table', formula: spec.positions === 'rope' ? 'RoPE rotates Q and K: no table' : `context × D = ${n(spec.context)} × ${n(spec.hidden)}`, value: b.posEmb, color: 'var(--rule-strong)' },
    {
      label: 'Attention, all blocks',
      formula: `${spec.layers} × (Q ${n(spec.hidden)}×${n(qOut)} + K, V ${n(spec.hidden)}×${n(kvOut)} each + O ${n(qOut)}×${n(spec.hidden)}${spec.qkvBias ? ' + biases' : ''})`,
      value: b.attn, color: 'color-mix(in srgb, var(--accent) 50%, var(--paper-2))',
    },
    {
      label: 'MLPs, all blocks',
      formula: spec.mlp === 'swiglu' ? `${spec.layers} × 3 × ${n(spec.hidden)} × ${n(spec.intermediate)} (gate, up, down)` : `${spec.layers} × (2 × ${n(spec.hidden)} × ${n(spec.intermediate)} + biases)`,
      value: b.mlp, color: 'var(--accent)',
    },
    { label: spec.norm === 'rmsnorm' ? 'RMSNorms (gain only)' : 'LayerNorms (gain + bias)', formula: spec.norm === 'rmsnorm' ? `${spec.layers} × 2D + D` : `${spec.layers} × 4D + 2D`, value: b.norms, color: 'var(--good)' },
    { label: spec.tied ? 'Head (tied to the token table)' : 'Head (its own matrix)', formula: spec.tied ? '0 extra' : `D × V = ${n(spec.hidden)} × ${n(spec.vocab)}`, value: b.head, color: 'var(--ink)' },
  ]

  return (
    <div aria-live="polite">
      {warnings.map((w) => <p key={w} className="lab-note" style={{ color: 'var(--bad)' }}>Careful: {w}</p>)}

      <h4 style={{ margin: '18px 0 6px', fontSize: 16 }}>1. What each field means</h4>
      <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14, marginBottom: 6 }}>
        <input type="checkbox" checked={sizingOnly} onChange={(e) => setSizingOnly(e.target.checked)} /> Only the fields that change the size of the model
      </label>
      <div className="table-scroll" style={{ maxHeight: 330, overflowY: 'auto' }}>
        <table className="plain" style={{ fontSize: 13.5 }}>
          <thead><tr><th>field</th><th>value</th><th>in the words of this course</th><th>taught in</th></tr></thead>
          <tbody>
            {shown.map((f) => {
              const l = f.lesson ? lessonById(f.lesson) : undefined
              return (
                <tr key={f.key}>
                  <td className="mono" style={{ fontSize: 12.5, fontWeight: f.sizing ? 700 : 400 }}>{f.key}</td>
                  <td className="mono" style={{ fontSize: 12.5 }}>{f.value}</td>
                  <td>{f.meaning}</td>
                  <td style={{ whiteSpace: 'nowrap' }}>{l ? <a href={`#/lesson/${l.id}`}>{l.code} {l.title.length > 22 ? `${l.title.slice(0, 20)}…` : l.title}</a> : ''}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <h4 style={{ margin: '22px 0 6px', fontSize: 16 }}>2. The shape of the model</h4>
      <div className="readout">
        <span>family: <b>{spec.family === 'gpt2' ? 'GPT-2 style' : 'Llama style'}</b></span>
        <span>blocks: <b>{spec.layers}</b></span>
        <span>D: <b>{n(spec.hidden)}</b></span>
        <span>heads: <b>{spec.heads}</b> query, <b>{spec.kvHeads}</b> key/value{gqa ? ` (GQA: ${spec.heads / spec.kvHeads} query heads share each)` : ''}</span>
        <span>head size: <b>{spec.headDim}</b></span>
        <span>positions: <b>{spec.positions === 'rope' ? `RoPE${spec.ropeTheta ? `, base ${n(spec.ropeTheta)}` : ''}` : 'learned table'}</b></span>
        <span>context: <b>{n(spec.context)}</b>{spec.slidingWindow ? <>, window <b>{n(spec.slidingWindow)}</b></> : null}</span>
      </div>

      <h4 style={{ margin: '22px 0 6px', fontSize: 16 }}>3. Parameters: {n(b.total)} ≈ {humanCount(b.total)}</h4>
      <div role="img" aria-label={`Share of parameters: ${parts.map((p) => `${p.label} ${((p.value / b.total) * 100).toFixed(1)} percent`).join(', ')}`} style={{ display: 'flex', height: 24, borderRadius: 6, overflow: 'hidden', border: '1px solid var(--rule-strong)', margin: '8px 0 10px' }}>
        {parts.map((p) => p.value > 0 && <div key={p.label} title={p.label} style={{ width: `${(p.value / b.total) * 100}%`, background: p.color }} />)}
      </div>
      <div className="table-scroll">
        <table className="plain" style={{ fontSize: 13.5 }}>
          <thead><tr><th>component</th><th>how it is counted</th><th style={{ textAlign: 'right' }}>parameters</th><th style={{ textAlign: 'right' }}>share</th></tr></thead>
          <tbody>
            {parts.map((p) => (
              <tr key={p.label}>
                <td><span aria-hidden style={{ display: 'inline-block', width: 11, height: 11, borderRadius: 2, background: p.color, marginRight: 7, border: '1px solid var(--rule-strong)' }} />{p.label}</td>
                <td className="mono" style={{ fontSize: 12 }}>{p.formula}</td>
                <td className="mono" style={{ textAlign: 'right' }}>{n(p.value)}</td>
                <td className="mono" style={{ textAlign: 'right' }}>{((p.value / b.total) * 100).toFixed(1)}%</td>
              </tr>
            ))}
            <tr><td colSpan={2}><b>total</b></td><td className="mono" style={{ textAlign: 'right' }}><b>{n(b.total)}</b></td><td className="mono" style={{ textAlign: 'right' }}>100%</td></tr>
          </tbody>
        </table>
      </div>
      {gqa && <p className="lab-note">GQA at work: each K and V matrix is {n(spec.hidden)} × {n(kvOut)}, which is {spec.heads / spec.kvHeads}× smaller than Q ({n(spec.hidden)} × {n(qOut)}).</p>}

      <h4 style={{ margin: '22px 0 6px', fontSize: 16 }}>4. Memory</h4>
      <div className="table-scroll">
        <table className="plain" style={{ fontSize: 13.5 }}>
          <thead><tr><th>what</th><th>how it is counted</th><th style={{ textAlign: 'right' }}>size</th></tr></thead>
          <tbody>
            {PRECISIONS.map((p) => (
              <tr key={p.id}>
                <td>Weights in {p.label}{spec.dtype && ((p.id === 'bf16' && /16/.test(spec.dtype)) || (p.id === 'fp32' && /32/.test(spec.dtype))) ? ' (as published)' : ''}</td>
                <td className="mono" style={{ fontSize: 12 }}>{humanCount(b.total)} × {BYTES_PER_PARAM[p.id]} byte{BYTES_PER_PARAM[p.id] === 1 ? '' : 's'}</td>
                <td className="mono" style={{ textAlign: 'right' }}>{formatGB(weightBytes(b.total, p.id))}</td>
              </tr>
            ))}
            <tr>
              <td>KV cache, one token</td>
              <td className="mono" style={{ fontSize: 12 }}>2 × {spec.layers} layers × {spec.kvHeads} KV heads × {spec.headDim} × 2 bytes</td>
              <td className="mono" style={{ textAlign: 'right' }}>{formatGB(kvTok)}</td>
            </tr>
            <tr>
              <td>KV cache, one full sequence</td>
              <td className="mono" style={{ fontSize: 12 }}>× {n(cap)} tokens{spec.slidingWindow && cap < spec.context ? ' (capped by the sliding window)' : ' (the whole context)'}</td>
              <td className="mono" style={{ textAlign: 'right' }}>{formatGB(kvTok * cap)}</td>
            </tr>
            <tr>
              <td>Full fine-tuning with Adam, before activations</td>
              <td className="mono" style={{ fontSize: 12 }}>{humanCount(b.total)} × 16 bytes (2 + 2 + 4 + 4 + 4)</td>
              <td className="mono" style={{ textAlign: 'right' }}><b>{formatGB(trainingBytes(b.total))}</b></td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="lab-note">
        Sizes are in decimal units (1 GB = 10⁹ bytes), the units the Hub uses for file sizes. A “24 GB” GPU has 24 × 2³⁰ bytes, about 7% more. The int8 and int4 rows ignore the small extra cost of the scales (see <a href="#/lesson/inference-systems">Inference systems</a>). The KV cache is counted at 16 bits per number. The 16 bytes per parameter is the standard estimate for mixed-precision Adam from the ZeRO paper; activations come on top and grow with batch size and sequence length.
      </p>
    </div>
  )
}
