// Lesson 1.2: matrix multiply as a grid of dot products. Pick a result cell
// and see exactly which row of A and which column of B produced it.
import { useState } from 'react'
import { Lab, MatrixView } from '../components/ui'
import { fmt, matmul, shape, transpose, type Mat } from '../lib/math'
import { SHAPES, canMultiply, explainCell, resize, shapeLabel, transposeFix, type Shape } from '../lib/matrixPlay'

// the worked example from phase1-foundations/math_primer.py, section 5
const A0: Mat = [[1, 2], [3, 0]]
const B0: Mat = [[1, 0, 1], [2, 1, 0]]

const n = (x: number) => fmt(x, Number.isInteger(x) ? 0 : 2)
const paren = (x: number) => (x < 0 ? `(${n(x)})` : n(x))
const sameShape = (a: Shape, b: Shape) => a[0] === b[0] && a[1] === b[1]

function ShapePicker({ name, current, onPick }: { name: string; current: Shape; onPick: (s: Shape) => void }) {
  return (
    <div className="steps" role="group" aria-label={`Shape of ${name}`} style={{ marginBottom: 6 }}>
      <span className="mono" style={{ alignSelf: 'center', fontSize: 13 }}>{name} is</span>
      {SHAPES.map((s) => (
        <button key={s.join('x')} className="step-btn" aria-pressed={sameShape(s, current)} onClick={() => onPick(s)}>{s[0]} × {s[1]}</button>
      ))}
    </div>
  )
}

export function MatrixPlayground() {
  const [A, setA] = useState<Mat>(A0)
  const [B, setB] = useState<Mat>(B0)
  const [cell, setCell] = useState<[number, number]>([0, 0])

  const sa = shape(A) as Shape
  const sb = shape(B) as Shape
  const ok = canMultiply(sa, sb)
  const out = ok ? matmul(A, B) : null
  const i = out ? Math.min(cell[0], out.length - 1) : 0
  const j = out ? Math.min(cell[1], out[0].length - 1) : 0
  const ex = out ? explainCell(A, B, i, j) : null
  const fixes = ok ? [] : transposeFix(sa, sb)

  const edit = (m: Mat, set: (m: Mat) => void) => (r: number, c: number, value: number) => {
    const next = m.map((row) => row.slice())
    next[r][c] = value
    set(next)
  }

  return (
    <Lab
      title="Every result cell is one dot product"
      goal={<>Hover, click or tab onto any cell of the <b>result</b>. The row of A and the column of B that made it light up, and the sum is spelled out. Then edit numbers, change the shapes, and try to make a multiply that does <em>not</em> fit.</>}
    >
      <div className="grid-2" style={{ gap: 8 }}>
        <ShapePicker name="A" current={sa} onPick={(s) => setA(resize(A, s))} />
        <ShapePicker name="B" current={sb} onPick={(s) => setB(resize(B, s))} />
      </div>

      <div className="readout" aria-live="polite" style={{ marginBottom: 12 }}>
        <span>
          {shapeLabel(sa)} @ {shapeLabel(sb)} →{' '}
          {ok
            ? <b>({sa[0]}, {sb[1]})</b>
            : <b style={{ color: 'var(--bad)' }}>does not fit</b>}
        </span>
        <span>inner numbers: <b style={{ color: ok ? 'var(--good)' : 'var(--bad)' }}>{sa[1]} and {sb[0]} {ok ? 'match ✓' : 'differ ✗'}</b></span>
        {ok && <span>outer numbers: {sa[0]} and {sb[1]} = shape of the answer</span>}
      </div>

      <div className="matrix-row">
        <MatrixView m={A} caption={`A ${shapeLabel(sa)}: rows are used`} tone="accent" onEdit={edit(A, setA)} highlight={(r) => !!out && r === i} />
        <span className="op" aria-hidden>@</span>
        <MatrixView m={B} caption={`B ${shapeLabel(sb)}: columns are used`} tone="accent" onEdit={edit(B, setB)} highlight={(_, c) => !!out && c === j} />
        <span className="op" aria-hidden>=</span>
        {out ? (
          <div className="matrix-wrap">
            <div className="matrix-cap">A @ B ({sa[0]}, {sb[1]}): pick a cell</div>
            <table className="matrix">
              <tbody>
                {out.map((row, r) => (
                  <tr key={r}>
                    {row.map((x, c) => (
                      <td key={c} className={r === i && c === j ? 'hot' : undefined} style={{ padding: 0 }}>
                        <button
                          className="mono"
                          style={{ all: 'unset', display: 'block', boxSizing: 'border-box', width: '100%', minWidth: 50, padding: '4px 8px', textAlign: 'right', cursor: 'pointer', fontWeight: r === i && c === j ? 700 : 400 }}
                          aria-pressed={r === i && c === j}
                          aria-label={`Result row ${r}, column ${c}: ${n(x)}. Show how it was computed.`}
                          onMouseEnter={() => setCell([r, c])}
                          onFocus={() => setCell([r, c])}
                          onClick={() => setCell([r, c])}
                        >
                          {n(x)}
                        </button>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="card" style={{ margin: 0, maxWidth: 330, borderLeft: '4px solid var(--bad)' }} role="status">
            <b>These shapes do not fit. Nothing is broken, it is just impossible.</b>
            <p style={{ fontSize: 14.5, marginTop: 6 }}>
              Each row of A has <b>{sa[1]}</b> numbers. Each column of B has <b>{sb[0]}</b>. A dot product needs two lists of the same length, so there is no way to pair them up.
            </p>
            <p style={{ fontSize: 14.5 }}>It is a type error, like passing a <code>String</code> where an <code>int</code> is expected. A common fix is to flip one matrix:</p>
            <div className="btn-row">
              {fixes.includes('A') && <button className="btn small" onClick={() => setA(transpose(A))}>Transpose A → ({sa[1]}, {sa[0]})</button>}
              {fixes.includes('B') && <button className="btn small" onClick={() => setB(transpose(B))}>Transpose B → ({sb[1]}, {sb[0]})</button>}
            </div>
          </div>
        )}
      </div>

      {ex && (
        <div className="readout" aria-live="polite" style={{ display: 'block' }}>
          <div className="muted" style={{ fontSize: 12 }}>result cell (row {i}, column {j}) = row {i} of A · column {j} of B</div>
          <div>[{ex.row.map(n).join(', ')}] · [{ex.col.map(n).join(', ')}]</div>
          <div>= {ex.row.map((v, t) => `${n(v)}×${paren(ex.col[t])}`).join(' + ')}</div>
          <div>= {ex.terms.map((v, t) => (t === 0 ? n(v) : paren(v))).join(' + ')} = <b style={{ fontSize: 18 }}>{n(ex.sum)}</b></div>
        </div>
      )}

      <div className="btn-row" style={{ marginTop: 14 }}>
        <button className="btn small" onClick={() => setA(transpose(A))}>Transpose A</button>
        <button className="btn small" onClick={() => setB(transpose(B))}>Transpose B</button>
        <button className="btn small" onClick={() => { setA(B); setB(A) }}>Swap A and B</button>
        <button className="btn small" onClick={() => { setA(A0); setB(B0); setCell([0, 0]) }}>Reset</button>
      </div>
      <p className="lab-note" style={{ marginTop: 10 }}>Rows and columns are counted from 0, as in Python. {out && <>This result needed {out.length} × {out[0].length} = {out.length * out[0].length} dot products of {sa[1]} multiplications each.</>}</p>
    </Lab>
  )
}
