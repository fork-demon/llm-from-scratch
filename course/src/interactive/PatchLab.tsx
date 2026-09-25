// How a model "reads" a photo: cut it into square patches, flatten each patch into a list of
// numbers, project that list to D numbers, add a position. One patch = one token.
import { useEffect, useId, useMemo, useRef, useState, type MouseEvent } from 'react'
import { Lab, MatrixView, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import {
  IMAGES, IMG, PATCH_SIZES, contextShare, extractPatch, imageTokens, makeImage, patchGrid, patchLength,
  positionVector, project, projectionMatrix, tiledTokens, type ImageName, type Img,
} from '../lib/patches'

const SCALE = 3 // canvas pixels per image pixel
const REAL_SIDES = [224, 336, 448, 896, 1344]
const WINDOWS = [8192, 32768, 131072]

/** Draw an RGB image (optionally a sub-square of it) onto a canvas, scaled up with hard pixel edges. */
function paint(canvas: HTMLCanvasElement | null, img: Img, x0: number, y0: number, size: number, scale: number) {
  const ctx = canvas?.getContext?.('2d')
  if (!canvas || !ctx) return // jsdom and very old browsers: nothing to draw on
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const k = ((y0 + y) * img.w + (x0 + x)) * 3
      ctx.fillStyle = `rgb(${img.data[k]},${img.data[k + 1]},${img.data[k + 2]})`
      ctx.fillRect(x * scale, y * scale, scale, scale)
    }
}

export function PatchLab() {
  const uid = useId()
  const [name, setName] = useState<ImageName>('screenshot')
  const [p, setP] = useState<number>(14)
  const [row, setRow] = useState(1)
  const [col, setCol] = useState(3)
  const [D, setD] = useState(8)
  const [side, setSide] = useState(448)
  const [realPatch, setRealPatch] = useState(14)
  const [merge, setMerge] = useState(2)
  const [win, setWin] = useState(32768)

  const img = useMemo(() => makeImage(name), [name])
  const grid = patchGrid(IMG, IMG, p)
  const r = Math.min(row, grid.rows - 1)
  const c = Math.min(col, grid.cols - 1)
  const index = r * grid.cols + c
  const vec = useMemo(() => extractPatch(img, p, r, c), [img, p, r, c])
  const W = useMemo(() => projectionMatrix(patchLength(p), D), [p, D])
  const emb = useMemo(() => project(vec, W), [vec, W])
  const pos = useMemo(() => positionVector(index, D), [index, D])
  const token = emb.map((v, i) => v + pos[i])

  // canvases: the whole image with its grid, and the chosen patch blown up
  const big = useRef<HTMLCanvasElement>(null)
  const zoom = useRef<HTMLCanvasElement>(null)
  const strip = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    paint(big.current, img, 0, 0, IMG, SCALE)
    const ctx = big.current?.getContext?.('2d')
    if (!ctx) return
    ctx.strokeStyle = 'rgba(255,255,255,0.85)'
    ctx.lineWidth = 1
    for (let k = 1; k < grid.cols; k++) {
      ctx.beginPath(); ctx.moveTo(k * p * SCALE + 0.5, 0); ctx.lineTo(k * p * SCALE + 0.5, IMG * SCALE); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(0, k * p * SCALE + 0.5); ctx.lineTo(IMG * SCALE, k * p * SCALE + 0.5); ctx.stroke()
    }
    ctx.strokeStyle = '#000'
    ctx.lineWidth = 3
    ctx.strokeRect(c * p * SCALE + 1.5, r * p * SCALE + 1.5, p * SCALE - 3, p * SCALE - 3)
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 1
    ctx.strokeRect(c * p * SCALE + 1.5, r * p * SCALE + 1.5, p * SCALE - 3, p * SCALE - 3)
  }, [img, p, r, c, grid.cols])
  useEffect(() => {
    paint(zoom.current, img, c * p, r * p, p, Math.floor(112 / p))
  }, [img, p, r, c])
  useEffect(() => {
    // the flattened patch: one thin bar per number, tinted by its channel, darker = smaller
    const cv = strip.current
    const ctx = cv?.getContext?.('2d')
    if (!cv || !ctx) return
    ctx.clearRect(0, 0, cv.width, cv.height)
    const w = cv.width / vec.length
    vec.forEach((v, i) => {
      const ch = i % 3
      const lv = Math.round(v * 255)
      ctx.fillStyle = ch === 0 ? `rgb(${lv},0,0)` : ch === 1 ? `rgb(0,${lv},0)` : `rgb(0,0,${lv})`
      ctx.fillRect(i * w, 0, Math.max(1, w), cv.height)
    })
  }, [vec])

  const pickFromCanvas = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * IMG
    const y = ((e.clientY - rect.top) / rect.height) * IMG
    setCol(Math.min(grid.cols - 1, Math.floor(x / p)))
    setRow(Math.min(grid.rows - 1, Math.floor(y / p)))
  }

  const real = imageTokens({ width: side, height: side, patch: realPatch, merge })
  const realLen = patchLength(realPatch)
  const tiles = tiledTokens(side, side)
  const sideOk = side % realPatch === 0

  return (
    <Lab
      title="Cut a picture into tokens"
      goal={<>Pick a picture and a patch size. Click any square (or use the sliders) to see that patch as the list of numbers the model receives, and what one token costs. Then scale it up to real image sizes.</>}
    >
      <div className="steps" role="group" aria-label="Choose a picture">
        {(Object.keys(IMAGES) as ImageName[]).map((k) => (
          <button key={k} className="step-btn" aria-pressed={name === k} onClick={() => setName(k)}>{IMAGES[k]}</button>
        ))}
      </div>
      <div className="steps" role="group" aria-label="Patch size in pixels">
        {PATCH_SIZES.map((s) => (
          <button key={s} className="step-btn" aria-pressed={p === s} onClick={() => setP(s)}>{s} × {s} px</button>
        ))}
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div>
          <canvas
            ref={big}
            width={IMG * SCALE}
            height={IMG * SCALE}
            onClick={pickFromCanvas}
            role="img"
            aria-label={`${IMAGES[name]}, ${IMG} by ${IMG} pixels, cut into a ${grid.rows} by ${grid.cols} grid of ${p}-pixel patches. Selected patch: row ${r + 1}, column ${c + 1}.`}
            style={{ width: '100%', maxWidth: IMG * SCALE, imageRendering: 'pixelated', cursor: 'crosshair', borderRadius: 6, display: 'block' }}
          />
          <div className="controls" style={{ marginTop: 10 }}>
            <Slider label="patch row" value={r + 1} min={1} max={grid.rows} step={1} onChange={(v) => setRow(v - 1)} />
            <Slider label="patch column" value={c + 1} min={1} max={grid.cols} step={1} onChange={(v) => setCol(v - 1)} />
          </div>
        </div>
        <div>
          <p className="lab-note" style={{ marginTop: 0 }}>Patch {index + 1} of {grid.count} (row {r + 1}, column {c + 1}), enlarged:</p>
          <canvas ref={zoom} width={p * Math.floor(112 / p)} height={p * Math.floor(112 / p)} role="img" aria-label={`Patch ${index + 1} enlarged`} style={{ width: 112, height: 112, imageRendering: 'pixelated', borderRadius: 4, border: '1px solid var(--rule-strong)' }} />
          <p className="lab-note">Flattened: {p} × {p} pixels × 3 colours = <b>{vec.length}</b> numbers, one bar each (red, green, blue in turn; brighter = bigger).</p>
          <canvas ref={strip} width={588} height={22} role="img" aria-label={`The ${vec.length} numbers of this patch as a strip`} style={{ width: '100%', height: 22, borderRadius: 3, border: '1px solid var(--rule-strong)' }} />
          <p className="mono" style={{ fontSize: 12.5, wordBreak: 'break-all' }}>[{vec.slice(0, 9).map((v) => fmt(v)).join(', ')}, … {vec.length - 9} more]</p>
        </div>
      </div>

      <div className="readout" aria-live="polite" style={{ marginTop: 8 }}>
        <span>grid <b>{grid.rows} × {grid.cols}</b></span>
        <span>patches = tokens <b>{grid.count}</b></span>
        <span>numbers per patch <b>{vec.length}</b></span>
        <span>numbers in the whole image <b>{(IMG * IMG * 3).toLocaleString('en-US')}</b></span>
      </div>

      <h4 style={{ fontSize: 16, margin: '18px 0 6px' }}>From {vec.length} numbers to one token of D numbers</h4>
      <Slider label="D (the model's width; real models: roughly 1,000 to 16,000)" value={D} min={2} max={12} step={1} onChange={setD} />
      <div className="matrix-row">
        <MatrixView m={[emb]} caption={<>patch × W<sub>E</sub> ({vec.length} → {D})</>} heat tone="accent" />
        <MatrixView m={[pos]} caption={<>+ position vector for patch {index + 1}</>} heat tone="neutral" />
        <MatrixView m={[token]} caption="= the token the Transformer sees" heat tone="accent" />
      </div>
      <p className="lab-note">
        <b>Honest warning:</b> W<sub>E</sub> ({vec.length} × {D}) and the position vectors here are seeded random numbers, not trained ones. The arithmetic is exactly what a Vision Transformer does; the <em>meaning</em> of the {D} numbers only appears after training.
      </p>

      <h4 style={{ fontSize: 16, margin: '18px 0 6px' }}>The same arithmetic at real sizes</h4>
      <div className="controls">
        <div className="control">
          <label htmlFor={`${uid}-side`}><span>image side (square)</span></label>
          <select id={`${uid}-side`} className="input" value={side} onChange={(e) => setSide(Number(e.target.value))}>
            {REAL_SIDES.map((s) => <option key={s} value={s}>{s} × {s}</option>)}
          </select>
        </div>
        <div className="control">
          <label htmlFor={`${uid}-rp`}><span>patch size</span></label>
          <select id={`${uid}-rp`} className="input" value={realPatch} onChange={(e) => setRealPatch(Number(e.target.value))}>
            <option value={14}>14 px (CLIP ViT-L/14, SigLIP, InternViT)</option>
            <option value={16}>16 px (original ViT-B/16)</option>
          </select>
        </div>
        <div className="control">
          <label htmlFor={`${uid}-m`}><span>merge neighbours</span></label>
          <select id={`${uid}-m`} className="input" value={merge} onChange={(e) => setMerge(Number(e.target.value))}>
            <option value={1}>no merging</option>
            <option value={2}>2 × 2 patches → 1 token</option>
          </select>
        </div>
        <div className="control">
          <label htmlFor={`${uid}-w`}><span>context window</span></label>
          <select id={`${uid}-w`} className="input" value={win} onChange={(e) => setWin(Number(e.target.value))}>
            {WINDOWS.map((w) => <option key={w} value={w}>{w.toLocaleString('en-US')} tokens</option>)}
          </select>
        </div>
      </div>
      <div className="readout" aria-live="polite">
        <span>{side} / {realPatch} = <b>{fmt(side / realPatch, sideOk ? 0 : 2)}</b>{!sideOk && ' (not whole: real models resize so it divides)'}</span>
        <span>patches <b>{real.rows} × {real.cols} = {real.patches.toLocaleString('en-US')}</b></span>
        <span>each {realLen} numbers</span>
        <span>tokens after merging <b>{real.tokens.toLocaleString('en-US')}</b></span>
        <span>share of context <b>{(contextShare(real.tokens, win) * 100).toFixed(1)}%</b></span>
        <span>images that fit <b>{Math.floor(win / Math.max(1, real.tokens))}</b></span>
      </div>
      {side > 448 && (
        <p className="lab-note">
          Most encoders are trained at one small size, so a {side} × {side} image is usually <b>cut into tiles</b> instead of fed whole. With 448-pixel tiles at 256 tokens each (the InternVL recipe): {tiles.tiles} tiles + {tiles.thumbnail} thumbnail = <b>{tiles.tokens.toLocaleString('en-US')}</b> tokens. Each model family has its own rule; always check its documentation.
        </p>
      )}
    </Lab>
  )
}
