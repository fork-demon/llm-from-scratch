// GPU KV memory drawn as a grid of blocks: worst-case contiguous reservation versus paged
// allocation, for the same queue of requests, step by step.
import { useEffect, useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { gridRequests, simulateKvGrid, type GridResult, type GridSnapshot } from '../lib/serving'

const N_BLOCKS = 64
const BLOCK = 16
const COLS = 16
const letter = (id: number) => String.fromCharCode(65 + (id % 26))

function Grid({ snap, selected, patternId }: { snap: GridSnapshot; selected: number | null; patternId: string }) {
  const CW = 44, CH = 34, W = COLS * CW + 8, H = (N_BLOCKS / COLS) * CH + 8
  // tokens really stored in physical block b
  const storedIn = (b: number): number => {
    const id = snap.owner[b]
    if (id < 0) return 0
    const k = snap.blockLists[id].indexOf(b)
    return Math.max(0, Math.min(BLOCK, snap.tokens[id] - k * BLOCK))
  }
  return (
    <div className="table-scroll">
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', minWidth: 560, display: 'block' }} role="img" aria-label={`KV memory map: ${snap.running.length} sequences running, ${snap.stored} tokens stored in ${snap.held} tokens of held memory`}>
        {snap.owner.map((id, b) => {
          const x = 4 + (b % COLS) * CW, y = 4 + Math.floor(b / COLS) * CH
          const stored = storedIn(b)
          const on = id >= 0 && id === selected
          return (
            <g key={b}>
              <rect x={x + 1} y={y + 1} width={CW - 3} height={CH - 3} rx={4} fill={id < 0 ? 'var(--paper-2)' : `url(#${patternId})`} stroke={on ? 'var(--ink)' : 'var(--rule-strong)'} strokeWidth={on ? 2 : 1} />
              {id >= 0 && stored > 0 && <rect x={x + 1} y={y + 1} width={((CW - 3) * stored) / BLOCK} height={CH - 3} rx={4} fill={`color-mix(in srgb, var(--accent) ${55 + 20 * (id % 3)}%, var(--paper-2))`} />}
              {id >= 0 && <text x={x + CW / 2} y={y + CH / 2 + 4} textAnchor="middle" fontSize={12} fontWeight={700} style={{ fill: 'var(--ink)', fontFamily: 'var(--mono)', paintOrder: 'stroke', stroke: 'var(--paper)', strokeWidth: 3, strokeLinejoin: 'round' }}>{letter(id)}</text>}
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function Panel({ title, note, result, step, total, selected, onSelect, patternId }: { title: string; note: string; result: GridResult; step: number; total: number; selected: number | null; onSelect: (id: number | null) => void; patternId: string }) {
  const finishedAll = step >= result.snaps.length
  const snap = result.snaps[Math.min(step, result.snaps.length - 1)]
  const live: GridSnapshot = finishedAll ? { ...snap, owner: snap.owner.map(() => -1), running: [], tokens: {}, blockLists: {}, stored: 0, held: 0, waiting: 0 } : snap
  const finished = total - live.running.length - live.waiting
  const waste = live.held ? 1 - live.stored / live.held : 0
  const sel = selected !== null && live.running.includes(selected) ? selected : null
  return (
    <div className="card" style={{ margin: '14px 0' }}>
      <h4 style={{ fontSize: 16.5, marginBottom: 2 }}>{title}</h4>
      <p className="lab-note" style={{ marginTop: 0 }}>{note}</p>
      <Grid snap={live} selected={sel} patternId={patternId} />
      <div className="readout" aria-live="polite" style={{ marginTop: 8 }}>
        <span>running <b>{live.running.length}</b></span>
        <span>waiting <b>{live.waiting}</b></span>
        <span>finished <b>{finished}</b> / {total}</span>
        <span>tokens stored <b>{live.stored}</b> of {live.held} held</span>
        <span>held but empty <b>{(waste * 100).toFixed(0)}%</b></span>
        <span>all done after <b>{result.steps}</b> steps</span>
      </div>
      <div className="steps" role="group" aria-label={`${title}: show the block table of a running sequence`} style={{ marginTop: 10, marginBottom: 6 }}>
        {live.running.map((id) => <button key={id} className="step-btn" aria-pressed={sel === id} onClick={() => onSelect(sel === id ? null : id)}>{letter(id)}</button>)}
      </div>
      {sel !== null && (
        <p className="lab-note mono" style={{ fontSize: 13 }}>
          block table of {letter(sel)} ({live.tokens[sel]} tokens): {live.blockLists[sel].map((b, k) => `${k}→${b}`).join('  ')}
        </p>
      )}
    </div>
  )
}

export function PagedKvLab() {
  const [step, setStep] = useState(20)
  const [maxNew, setMaxNew] = useState(128)
  const [playing, setPlaying] = useState(false)
  const [selected, setSelected] = useState<number | null>(null)

  const requests = useMemo(() => gridRequests(), [])
  const contiguous = useMemo(() => simulateKvGrid(requests, { mode: 'contiguous', maxNew, nBlocks: N_BLOCKS, blockSize: BLOCK }), [requests, maxNew])
  const paged = useMemo(() => simulateKvGrid(requests, { mode: 'paged', maxNew, nBlocks: N_BLOCKS, blockSize: BLOCK }), [requests, maxNew])
  const last = Math.max(contiguous.steps, paged.steps)

  useEffect(() => {
    if (!playing) return
    const t = setInterval(() => setStep((s) => { if (s >= last) { setPlaying(false); return s } return s + 1 }), 120)
    return () => clearInterval(t)
  }, [playing, last])

  return (
    <Lab
      title="KV memory, block by block"
      goal={<>The same 24 requests queue for the same 1,024 tokens of KV memory (64 blocks of 16). Step through time. In each cell, the solid part is tokens really stored; the hatched part is memory that is held but empty. Pick a letter to see that sequence’s block table.</>}
    >
      <svg width="0" height="0" style={{ position: 'absolute' }} aria-hidden>
        <defs>
          <pattern id="pkv-waste" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill="var(--bad)" opacity="0.12" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--bad)" strokeWidth="1.8" opacity="0.75" />
          </pattern>
        </defs>
      </svg>
      <div className="controls">
        <Slider label="Time (decode steps)" value={Math.min(step, last)} min={0} max={last} step={1} onChange={(v) => { setPlaying(false); setStep(v) }} />
        <Slider label="max_tokens cap the server must reserve for" value={maxNew} min={128} max={256} step={64} onChange={setMaxNew} />
      </div>
      <div className="btn-row" style={{ marginBottom: 6 }}>
        <button className="btn small" onClick={() => { setPlaying(false); setStep(Math.max(0, Math.min(step, last) - 1)) }}>Back</button>
        <button className="btn small primary" onClick={() => { setPlaying(false); setStep(Math.min(last, step + 1)) }}>Step</button>
        <button className="btn small" onClick={() => { if (step >= last) setStep(0); setPlaying(!playing) }}>{playing ? 'Pause' : 'Run'}</button>
        <button className="btn small" onClick={() => { setPlaying(false); setStep(0) }}>Reset</button>
      </div>
      <p className="lab-note">Requests: prompts of {Math.min(...requests.map((r) => r.prompt))} to {Math.max(...requests.map((r) => r.prompt))} tokens, answers of {Math.min(...requests.map((r) => r.out))} to {Math.max(...requests.map((r) => r.out))} tokens (most are short). The server does not know the answer length in advance.</p>

      <Panel
        title="Contiguous reservation"
        note={`Each request gets one unbroken run of blocks sized for prompt + ${maxNew} tokens, first fit. On ${contiguous.fragmentationRefusals} steps the next request was refused although enough free blocks existed in total: they were not next to each other.`}
        result={contiguous} step={step} total={requests.length} selected={selected} onSelect={setSelected} patternId="pkv-waste"
      />
      <Panel
        title="Paged allocation"
        note="Each sequence holds only the blocks its tokens need, and takes any free block when the last one fills up. The block table records where each logical block lives."
        result={paged} step={step} total={requests.length} selected={selected} onSelect={setSelected} patternId="pkv-waste"
      />
      <div className="readout" aria-live="polite">
        <span>whole run, contiguous: mean <b>{contiguous.meanRunning.toFixed(1)}</b> running, <b>{(contiguous.waste * 100).toFixed(0)}%</b> of held memory empty</span>
        <span>paged: mean <b>{paged.meanRunning.toFixed(1)}</b> running, <b>{(paged.waste * 100).toFixed(0)}%</b> empty{paged.preemptions ? `, ${paged.preemptions} preemptions` : ''}</span>
      </div>
    </Lab>
  )
}
