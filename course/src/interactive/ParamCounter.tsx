// "A GPT is these numbers": change the config, see where the parameters live.
import { useId, useState } from 'react'
import { Lab } from '../components/ui'
import { countParams, headDim, humanCount, PRESETS, type GptConfig } from '../lib/params'

const FIELDS: { key: keyof GptConfig; label: string; hint: string; min: number; max: number }[] = [
  { key: 'vocabSize', label: 'vocab_size', hint: 'different tokens', min: 2, max: 300000 },
  { key: 'contextLen', label: 'context_len', hint: 'max tokens it can see', min: 1, max: 200000 },
  { key: 'nEmbd', label: 'n_embd (D)', hint: 'numbers per token', min: 2, max: 20000 },
  { key: 'nHead', label: 'n_head', hint: 'attention heads', min: 1, max: 256 },
  { key: 'nLayer', label: 'n_layer', hint: 'blocks stacked', min: 1, max: 200 },
]

const n = (x: number) => x.toLocaleString('en-US')

export function ParamCounter() {
  const [cfg, setCfg] = useState<GptConfig>(PRESETS[0].config)
  const [tied, setTied] = useState(true)
  const id = useId()
  const b = countParams(cfg, { tied })
  const hd = headDim(cfg)
  const D = cfg.nEmbd
  const activePreset = PRESETS.find((p) => (Object.keys(p.config) as (keyof GptConfig)[]).every((k) => p.config[k] === cfg[k]))

  const parts = [
    { label: 'Token embeddings', formula: `V × D = ${n(cfg.vocabSize)} × ${n(D)}`, value: b.tokenEmb, color: 'var(--ink-3)' },
    { label: 'Position embeddings', formula: `context × D = ${n(cfg.contextLen)} × ${n(D)}`, value: b.posEmb, color: 'var(--rule-strong)' },
    { label: 'Attention, all blocks', formula: `n_layer × (4D² + 4D) = ${cfg.nLayer} × ${n(b.attnPerBlock)}`, value: b.attn, color: 'color-mix(in srgb, var(--accent) 50%, var(--paper-2))' },
    { label: 'MLPs, all blocks', formula: `n_layer × (8D² + 5D) = ${cfg.nLayer} × ${n(b.mlpPerBlock)}`, value: b.mlp, color: 'var(--accent)' },
    { label: 'LayerNorms', formula: `n_layer × 4D + 2D`, value: b.norms, color: 'var(--good)' },
    { label: tied ? 'Output head (tied: reuses the token embeddings)' : 'Output head (separate matrix)', formula: tied ? '0 extra' : `D × V = ${n(D)} × ${n(cfg.vocabSize)}`, value: b.head, color: 'var(--ink)' },
  ]

  return (
    <Lab
      title="Where do the parameters live?"
      goal={<>Load a preset, then change one number at a time. Before each change, predict: which part of the bar will grow? Try doubling n_embd, then doubling n_layer, then changing n_head.</>}
    >
      <div className="steps" role="group" aria-label="Presets">
        {PRESETS.map((p) => <button key={p.id} className="step-btn" aria-pressed={activePreset?.id === p.id} onClick={() => setCfg(p.config)}>{p.label}</button>)}
      </div>
      {activePreset && <p className="lab-note">{activePreset.note}</p>}

      <div className="controls">
        {FIELDS.map((f) => (
          <div className="control" key={f.key}>
            <label htmlFor={`${id}-${f.key}`}><span className="mono">{f.label}</span><output style={{ color: 'var(--ink-3)' }}>{f.hint}</output></label>
            <input
              id={`${id}-${f.key}`}
              className="input"
              type="number"
              min={f.min}
              max={f.max}
              value={cfg[f.key]}
              onChange={(e) => setCfg({ ...cfg, [f.key]: Math.min(f.max, Math.max(f.min, Math.round(Number(e.target.value) || f.min))) })}
              style={{ fontFamily: 'var(--mono)' }}
            />
          </div>
        ))}
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}>
          <input type="checkbox" checked={tied} onChange={(e) => setTied(e.target.checked)} /> Weight tying (head shares the embedding table)
        </label>
      </div>

      <div className="readout" aria-live="polite">
        <span>total parameters: <b>{n(b.total)}</b> ≈ <b>{humanCount(b.total)}</b></span>
        <span>per-head width D / n_head: <b>{hd ?? 'not a whole number'}</b></span>
        <span>one block: <b>{n(b.attnPerBlock + b.mlpPerBlock + b.normPerBlock)}</b></span>
      </div>
      {hd === null && <p className="lab-note" style={{ color: 'var(--bad)' }}>n_embd must be divisible by n_head, otherwise the D numbers cannot be cut into equal heads. The count below is unaffected: heads never add parameters.</p>}

      <div role="img" aria-label={`Share of parameters: ${parts.map((p) => `${p.label} ${((p.value / b.total) * 100).toFixed(1)} percent`).join(', ')}`} style={{ display: 'flex', height: 26, borderRadius: 6, overflow: 'hidden', border: '1px solid var(--rule-strong)', margin: '14px 0 10px' }}>
        {parts.map((p) => p.value > 0 && <div key={p.label} title={p.label} style={{ width: `${(p.value / b.total) * 100}%`, background: p.color }} />)}
      </div>

      <div className="table-scroll">
        <table className="plain" style={{ fontSize: 14 }}>
          <thead><tr><th>component</th><th>how it is counted</th><th style={{ textAlign: 'right' }}>parameters</th><th style={{ textAlign: 'right' }}>share</th></tr></thead>
          <tbody>
            {parts.map((p) => (
              <tr key={p.label}>
                <td><span aria-hidden style={{ display: 'inline-block', width: 11, height: 11, borderRadius: 2, background: p.color, marginRight: 7, border: '1px solid var(--rule-strong)' }} />{p.label}</td>
                <td className="mono" style={{ fontSize: 12.5 }}>{p.formula}</td>
                <td className="mono" style={{ textAlign: 'right' }}>{n(p.value)}</td>
                <td className="mono" style={{ textAlign: 'right' }}>{((p.value / b.total) * 100).toFixed(1)}%</td>
              </tr>
            ))}
            <tr><td colSpan={2}><b>total</b></td><td className="mono" style={{ textAlign: 'right' }}><b>{n(b.total)}</b></td><td className="mono" style={{ textAlign: 'right' }}>100%</td></tr>
          </tbody>
        </table>
      </div>
      <p className="lab-note">
        4D² + 4D: four D×D matrices (Wq, Wk, Wv, Wo) and their four bias vectors. 8D² + 5D: a D→4D layer (4D² weights + 4D biases) and a 4D→D layer (4D² weights + D biases). This is exact for <code>tiny_gpt.py</code> and GPT-2; modern models drop the biases and change the MLP shape, so their formulas differ slightly.
      </p>
    </Lab>
  )
}
