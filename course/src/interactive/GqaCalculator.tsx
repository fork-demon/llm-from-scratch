// KV-cache size for multi-head (MHA), grouped-query (GQA) and multi-query (MQA) attention.
import { useId, useState } from 'react'
import { Bars, Lab, Slider } from '../components/ui'
import { GIB, KV_PRESETS, compareKv, divisors, formatBytes, kvBytesPerToken } from '../lib/modernArch'

const PRECISIONS = [
  { label: '32-bit float (4 bytes)', bytes: 4 },
  { label: '16-bit float (2 bytes)', bytes: 2 },
  { label: '8-bit (1 byte)', bytes: 1 },
  { label: '4-bit (0.5 byte)', bytes: 0.5 },
]
const HEAD_OPTIONS = [4, 8, 12, 16, 32, 40, 64, 96]
const HEAD_DIMS = [32, 64, 96, 128]

export function GqaCalculator() {
  const uid = useId()
  const [preset, setPreset] = useState('llama-70b')
  const [layers, setLayers] = useState(80)
  const [heads, setHeads] = useState(64)
  const [kvHeads, setKvHeads] = useState(8)
  const [headDim, setHeadDim] = useState(128)
  const [logTokens, setLogTokens] = useState(12) // tokens = 2^logTokens
  const [bytes, setBytes] = useState(2)

  const tokens = 2 ** logTokens
  const validKv = divisors(heads)
  const kv = validKv.includes(kvHeads) ? kvHeads : heads
  const cmp = compareKv({ layers, queryHeads: heads, kvHeads: kv, headDim, tokens, bytesPerValue: bytes })
  const perToken = kvBytesPerToken({ layers, kvHeads: kv, headDim, bytesPerValue: bytes })
  const kind = kv === heads ? 'plain multi-head attention (MHA)' : kv === 1 ? 'multi-query attention (MQA)' : 'grouped-query attention (GQA)'

  const unit = cmp.mha >= GIB ? { name: 'GiB', size: GIB } : cmp.mha >= 1024 ** 2 ? { name: 'MiB', size: 1024 ** 2 } : { name: 'KiB', size: 1024 }

  const load = (id: string) => {
    const p = KV_PRESETS.find((x) => x.id === id)
    if (!p) return
    setPreset(id)
    setLayers(p.layers)
    setHeads(p.queryHeads)
    setKvHeads(p.kvHeads)
    setHeadDim(p.headDim)
    setLogTokens(Math.round(Math.log2(p.context)))
    setBytes(id === 'tiny' ? 4 : 2)
  }
  const custom = () => setPreset('')
  const note = KV_PRESETS.find((p) => p.id === preset)?.note

  return (
    <Lab
      title="How big is the KV cache?"
      goal={<>Load a model, then change the number of <span className="k">K</span>/<span className="v">V</span> heads. The query heads stay the same. Watch what happens to the memory needed to hold one conversation, then stretch the context length.</>}
    >
      <div className="steps" role="group" aria-label="Load a publicly documented configuration">
        {KV_PRESETS.map((p) => (
          <button key={p.id} className="step-btn" aria-pressed={preset === p.id} onClick={() => load(p.id)}>{p.label}</button>
        ))}
      </div>
      <p className="lab-note">{note ?? 'Custom configuration.'}</p>

      <div className="controls">
        <Slider label="Layers" value={layers} min={1} max={128} step={1} onChange={(v) => { setLayers(v); custom() }} />
        <Slider label="Context length (tokens in the cache)" value={logTokens} min={6} max={20} step={1} onChange={(v) => { setLogTokens(v); custom() }} format={(v) => (2 ** v).toLocaleString('en-US')} />
        <div className="control">
          <label htmlFor={`${uid}-h`}><span>Query heads</span></label>
          <select id={`${uid}-h`} className="input" value={heads} onChange={(e) => { const h = Number(e.target.value); setHeads(h); if (!divisors(h).includes(kvHeads)) setKvHeads(h); custom() }}>
            {HEAD_OPTIONS.map((h) => <option key={h} value={h}>{h}</option>)}
          </select>
        </div>
        <div className="control">
          <label htmlFor={`${uid}-kv`}><span><span className="k">K</span>/<span className="v">V</span> heads (must divide the query heads)</span></label>
          <select id={`${uid}-kv`} className="input" value={kv} onChange={(e) => { setKvHeads(Number(e.target.value)); custom() }}>
            {validKv.map((h) => <option key={h} value={h}>{h}{h === heads ? ' (MHA: one per query head)' : h === 1 ? ' (MQA: one for all)' : ` (GQA: ${heads / h} query heads share each)`}</option>)}
          </select>
        </div>
        <div className="control">
          <label htmlFor={`${uid}-hd`}><span>Numbers per head (head_dim)</span></label>
          <select id={`${uid}-hd`} className="input" value={headDim} onChange={(e) => { setHeadDim(Number(e.target.value)); custom() }}>
            {HEAD_DIMS.map((h) => <option key={h} value={h}>{h}</option>)}
          </select>
        </div>
        <div className="control">
          <label htmlFor={`${uid}-p`}><span>Precision of each cached number</span></label>
          <select id={`${uid}-p`} className="input" value={bytes} onChange={(e) => { setBytes(Number(e.target.value)); custom() }}>
            {PRECISIONS.map((p) => <option key={p.bytes} value={p.bytes}>{p.label}</option>)}
          </select>
        </div>
      </div>

      <p className="mono" style={{ fontSize: 13.5, overflowWrap: 'anywhere' }}>
        2 × {layers} layers × <b>{kv} KV heads</b> × {headDim} × {tokens.toLocaleString('en-US')} tokens × {bytes} bytes = <b>{formatBytes(cmp.gqa)}</b>
      </p>

      <div aria-live="polite">
        <Bars
          percent={false}
          max={cmp.mha / unit.size}
          items={[
            { label: `MHA: ${heads} KV`, value: cmp.mha / unit.size, tone: 'neutral' },
            { label: `Yours: ${kv} KV`, value: cmp.gqa / unit.size, tone: 'accent' },
            { label: 'MQA: 1 KV', value: cmp.mqa / unit.size, tone: 'neutral' },
          ]}
        />
        <p className="lab-note">MHA = one K/V head per query head, “Yours” = the K/V heads chosen above, MQA = a single K/V head. Bar values are in {unit.name} (binary units: 1 GiB = 1,024 MiB = 1,073,741,824 bytes), for one sequence.</p>
        <div className="readout">
          <span>this config is <b>{kind}</b></span>
          <span>cache per token: <b>{formatBytes(perToken)}</b></span>
          <span>cache for {tokens.toLocaleString('en-US')} tokens: <b>{formatBytes(cmp.gqa)}</b></span>
          <span>with MHA it would be <b>{formatBytes(cmp.mha)}</b></span>
          <span>saving vs MHA: <b>{cmp.savingVsMha}×</b></span>
        </div>
      </div>
      <p className="lab-note">
        This counts only the cache for <em>one</em> conversation. A server holding 50 conversations needs 50 of these, on top of the weights. Presets use each model’s published layer and head counts; the context slider is yours to stretch beyond what that model was trained for, to see the memory bill only.
      </p>
    </Lab>
  )
}
