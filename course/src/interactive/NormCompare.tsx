// LayerNorm vs RMSNorm on one editable 4-number vector.
import { useState } from 'react'
import { Lab, MatrixView } from '../components/ui'
import { fmt } from '../lib/math'
import { centre, layerNorm, rmsNorm } from '../lib/modernArch'

const START = [2, 4, 6, 8]
const COLS = ['x₁', 'x₂', 'x₃', 'x₄']

export function NormCompare() {
  const [x, setX] = useState<number[]>(START)
  const ln = layerNorm(x)
  const rms = rmsNorm(x)
  const maxDiff = Math.max(...ln.out.map((v, i) => Math.abs(v - rms.out[i])))
  const edit = (_i: number, j: number, value: number) => setX(x.map((v, idx) => (idx === j ? value : v)))

  return (
    <Lab
      title="LayerNorm vs RMSNorm on four numbers"
      goal={<>This is one token’s vector (a real one has thousands of numbers). Edit it and compare the two outputs. Then press <b>“Subtract the mean first”</b> and compare again.</>}
    >
      <MatrixView m={[x]} rows={['x']} cols={COLS} onEdit={edit} caption="the input vector (editable)" />
      <div className="btn-row" style={{ margin: '10px 0' }}>
        <button className="btn small" onClick={() => setX(centre(x))}>Subtract the mean first</button>
        <button className="btn small" onClick={() => setX(x.map((v) => v * 10))}>Multiply everything by 10</button>
        <button className="btn small" onClick={() => setX(x.map((v) => v + 5))}>Add 5 to everything</button>
        <button className="btn small" onClick={() => setX(START)}>Reset</button>
      </div>

      <div className="grid-2" aria-live="polite">
        <div className="card" style={{ minWidth: 0 }}>
          <h4 style={{ fontSize: 16, marginBottom: 6 }}>LayerNorm (our tiny GPT)</h4>
          <ol style={{ paddingLeft: 18, fontSize: 14.5 }}>
            <li>mean = <b className="mono">{fmt(ln.mean, 3)}</b></li>
            <li>subtract it: <span className="mono">[{x.map((v) => fmt(v - ln.mean)).join(', ')}]</span></li>
            <li>spread = √(mean of squares of those) = <b className="mono">{fmt(Math.sqrt(ln.variance), 3)}</b></li>
            <li>divide by the spread</li>
          </ol>
          <MatrixView m={[ln.out]} rows={['out']} cols={COLS} heat tone="accent" digits={3} />
          <p className="lab-note">Two statistics (mean, spread). Output always has mean 0.</p>
        </div>
        <div className="card" style={{ minWidth: 0 }}>
          <h4 style={{ fontSize: 16, marginBottom: 6 }}>RMSNorm (most 2020s open models)</h4>
          <ol style={{ paddingLeft: 18, fontSize: 14.5 }}>
            <li>mean of x² = <b className="mono">{fmt(rms.meanSquare, 3)}</b></li>
            <li>size = √ of that = <b className="mono">{fmt(Math.sqrt(rms.meanSquare), 3)}</b></li>
            <li>divide by the size</li>
          </ol>
          <MatrixView m={[rms.out]} rows={['out']} cols={COLS} heat tone="accent" digits={3} />
          <p className="lab-note">One statistic. The vector keeps its direction; only its length is standardised.</p>
        </div>
      </div>

      <div className="readout" aria-live="polite">
        <span>mean of x = <b>{fmt(ln.mean, 3)}</b></span>
        <span>largest difference between the two outputs = <b>{fmt(maxDiff, 3)}</b></span>
        <span>{maxDiff < 1e-3 ? 'The mean is 0, so the two are the same computation.' : 'They differ only because the mean is not 0.'}</span>
      </div>
      <p className="lab-note">Both are shown without their learned parameters. LayerNorm then multiplies by a learned gain and adds a learned bias per dimension; RMSNorm only multiplies by a learned gain. Both add a tiny constant (1e-5 here) under the square root so an all-zero vector does not divide by zero.</p>
    </Lab>
  )
}
