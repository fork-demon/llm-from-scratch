// A small weight matrix before and after quantization, with the error drawn, plus the memory
// arithmetic for a whole model. All numbers come from src/lib/quantize.ts.
import { useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { bitsPerWeight, decodeTokPerSBound, demoMatrices, fakeQuantize, modelGB, outputError, plantOutlier, relErr, type Granularity } from '../lib/quantize'
import type { Mat } from '../lib/math'

const ROWS = 8, COLS = 32, GROUP = 8
const OUTLIER: [number, number, number] = [2, 5, 0.5]
const W_CLIP = 0.06 // colour scale for weights: 3 standard deviations
const E_CLIP = 0.03 // colour scale for errors

function HeatMap({ m, clip, kind, title, groupLines, mark, onPick, picked }: { m: Mat; clip: number; kind: 'signed' | 'error'; title: string; groupLines: number; mark?: [number, number]; onPick: (c: [number, number]) => void; picked: [number, number] | null }) {
  const CW = 22, CH = 15, LEFT = 4, W = LEFT * 2 + COLS * CW, H = ROWS * CH + 8
  return (
    <div style={{ margin: '10px 0' }}>
      <div className="matrix-cap">{title}</div>
      <div className="table-scroll">
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', minWidth: 520, display: 'block' }} role="img" aria-label={title}>
          {m.map((row, i) => row.map((v, j) => {
            const a = Math.min(1, Math.abs(v) / clip)
            const fill = kind === 'error' ? 'var(--bad)' : v >= 0 ? 'var(--accent)' : 'var(--ink)'
            const on = picked && picked[0] === i && picked[1] === j
            return (
              <g key={`${i}-${j}`} onMouseEnter={() => onPick([i, j])} onClick={() => onPick([i, j])}>
                <rect x={LEFT + j * CW} y={4 + i * CH} width={CW - 2} height={CH - 2} rx={2} fill="var(--paper-2)" />
                <rect x={LEFT + j * CW} y={4 + i * CH} width={CW - 2} height={CH - 2} rx={2} fill={fill} opacity={a * (kind === 'signed' && v < 0 ? 0.75 : 0.95)} stroke={on ? 'var(--ink)' : 'none'} strokeWidth={on ? 1.5 : 0} />
              </g>
            )
          }))}
          {groupLines > 0 && Array.from({ length: COLS / groupLines - 1 }, (_, g) => (
            <line key={g} x1={LEFT + (g + 1) * groupLines * CW - 1} x2={LEFT + (g + 1) * groupLines * CW - 1} y1={2} y2={H - 2} stroke="var(--ink-3)" strokeWidth={1} strokeDasharray="3 3" />
          ))}
          {mark && <rect x={LEFT + mark[1] * CW - 2} y={4 + mark[0] * CH - 2} width={CW + 2} height={CH + 2} rx={3} fill="none" stroke="var(--bad)" strokeWidth={2} />}
        </svg>
      </div>
    </div>
  )
}

export function QuantizeLab() {
  const [bits, setBits] = useState(4)
  const [gran, setGran] = useState<Granularity>('tensor')
  const [outlier, setOutlier] = useState(false)
  const [picked, setPicked] = useState<[number, number] | null>(null)
  // the model calculator
  const [params, setParams] = useState(7)
  const [fmt, setFmt] = useState<'fp16' | 'int8' | 'int4'>('int4')
  const [group, setGroup] = useState(128)
  const [bandwidth, setBandwidth] = useState(2)

  const base = useMemo(() => demoMatrices(ROWS, COLS, 16, 17), [])
  const W = useMemo(() => (outlier ? plantOutlier(base.W, ...OUTLIER) : base.W), [base, outlier])
  const qz = useMemo(() => fakeQuantize(W, bits, gran, GROUP), [W, bits, gran])
  const err = useMemo(() => W.map((row, i) => row.map((v, j) => v - qz.What[i][j])), [W, qz])
  const wErr = relErr(W, qz.What)
  const oErr = outputError(W, qz.What, base.X)
  const levels = new Set(qz.Q.flat()).size
  const zeros = qz.Q.flat().filter((q) => q === 0).length
  const bytesBefore = ROWS * COLS * 2
  const bytesAfter = (ROWS * COLS * bits) / 8 + qz.nScales * 2
  const scaleOf = (i: number, j: number) => (gran === 'tensor' ? qz.scales[0] : gran === 'row' ? qz.scales[i] : qz.scales[i * (COLS / GROUP) + Math.floor(j / GROUP)])

  const bpw = fmt === 'fp16' ? 16 : fmt === 'int8' ? bitsPerWeight(8) : bitsPerWeight(4, group)
  const gb = modelGB(params * 1e9, bpw)
  const fp16gb = modelGB(params * 1e9, 16)

  return (
    <Lab
      title="Quantize a weight matrix and measure the damage"
      goal={<>This is one small layer: 8 output rows of 32 weights, drawn as colours. Pick the bits and how many scales to use. Then press <b>Plant an outlier</b> and watch what one large weight does to “one scale for the tensor”.</>}
    >
      <div className="steps" role="group" aria-label="Bits per weight">
        {[8, 4].map((b) => <button key={b} className="step-btn" aria-pressed={bits === b} onClick={() => setBits(b)}>int{b} ({2 ** b - 1} levels)</button>)}
      </div>
      <div className="steps" role="group" aria-label="How many scales">
        {([['tensor', 'one scale for the tensor'], ['row', 'one scale per row'], ['group', `one scale per group of ${GROUP}`]] as [Granularity, string][]).map(([g, label]) => (
          <button key={g} className="step-btn" aria-pressed={gran === g} onClick={() => setGran(g)}>{label}</button>
        ))}
      </div>
      <div className="steps" role="group" aria-label="Outlier">
        <button className="step-btn" aria-pressed={outlier} onClick={() => setOutlier(!outlier)}>{outlier ? 'Outlier planted: one weight = 0.5 (25× typical)' : 'Plant an outlier'}</button>
      </div>

      <HeatMap m={W} clip={W_CLIP} kind="signed" title="original weights, 16-bit (coloured = positive, neutral = negative, stronger = larger)" groupLines={gran === 'group' ? GROUP : 0} mark={outlier ? [OUTLIER[0], OUTLIER[1]] : undefined} onPick={setPicked} picked={picked} />
      <HeatMap m={qz.What} clip={W_CLIP} kind="signed" title={`after int${bits}: integer × scale`} groupLines={gran === 'group' ? GROUP : 0} mark={outlier ? [OUTLIER[0], OUTLIER[1]] : undefined} onPick={setPicked} picked={picked} />
      <HeatMap m={err} clip={E_CLIP} kind="error" title="error = original − restored (stronger = worse)" groupLines={gran === 'group' ? GROUP : 0} onPick={setPicked} picked={picked} />

      <div className="readout" aria-live="polite">
        <span>weight error <b>{(wErr * 100).toFixed(1)}%</b></span>
        <span>output error of y = x·Wᵀ <b>{(oErr * 100).toFixed(1)}%</b></span>
        <span>levels in use <b>{levels}</b> of {2 ** bits - 1}</span>
        <span>weights rounded to 0: <b>{zeros}</b> / {ROWS * COLS}</span>
        <span>scales <b>{qz.nScales}</b></span>
        <span>bytes <b>{bytesBefore}</b> → <b>{bytesAfter}</b> ({(bytesBefore / bytesAfter).toFixed(2)}× smaller)</span>
      </div>
      <p className="lab-note mono" style={{ fontSize: 13, minHeight: 20 }}>
        {picked
          ? `row ${picked[0]}, column ${picked[1]}: weight ${W[picked[0]][picked[1]].toFixed(4)} ÷ scale ${scaleOf(...picked).toFixed(5)} → stored integer ${qz.Q[picked[0]][picked[1]]} → restored ${qz.What[picked[0]][picked[1]].toFixed(4)} (error ${err[picked[0]][picked[1]].toFixed(4)})`
          : 'Point at a cell to see its integer and its scale.'}
      </p>
      <p className="lab-note">Output error is measured on 16 random input vectors. The weights are random, not from a trained model, and this is plain round-to-nearest: the baseline that GPTQ and AWQ improve on. In this tiny layer the group size is 8; real int4 formats use 32 to 128.</p>

      <h4 style={{ marginTop: 22, marginBottom: 4 }}>What it buys for a whole model</h4>
      <div className="steps" role="group" aria-label="Model size">
        {[7, 13, 70].map((p) => <button key={p} className="step-btn" aria-pressed={params === p} onClick={() => setParams(p)}>{p}B parameters</button>)}
      </div>
      <div className="steps" role="group" aria-label="Weight format">
        {(['fp16', 'int8', 'int4'] as const).map((f) => <button key={f} className="step-btn" aria-pressed={fmt === f} onClick={() => setFmt(f)}>{f}</button>)}
        {fmt === 'int4' && [32, 64, 128].map((g) => <button key={g} className="step-btn" aria-pressed={group === g} onClick={() => setGroup(g)}>group of {g}</button>)}
      </div>
      <div className="controls">
        <Slider label="Memory bandwidth (TB/s)" value={bandwidth} min={0.5} max={5} step={0.1} onChange={setBandwidth} format={(v) => v.toFixed(1)} />
      </div>
      <div className="readout" aria-live="polite">
        <span>bits per weight, scales included <b>{bpw.toFixed(3)}</b></span>
        <span>weights <b>{gb.toFixed(2)} GB</b> (fp16: {fp16gb.toFixed(1)} GB, {(fp16gb / gb).toFixed(2)}× smaller)</span>
        <span>one stream, upper bound <b>{decodeTokPerSBound(bandwidth * 1e12, gb).toFixed(0)} tokens/s</b> (fp16: {decodeTokPerSBound(bandwidth * 1e12, fp16gb).toFixed(0)})</span>
      </div>
      <p className="lab-note">The speed line is bandwidth ÷ bytes of weights, an upper bound for one stream. Real kernels lose some of it to unpacking the integers, and embeddings or norms often stay in 16-bit.</p>
    </Lab>
  )
}
