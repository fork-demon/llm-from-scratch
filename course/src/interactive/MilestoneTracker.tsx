// Capstone progress: one checklist per milestone, saved in this browser only.
// <MilestoneTracker /> shows the overview; <MilestoneTracker only="rag" /> shows one checklist.
// All instances on the page share one small store, so ticking a box updates the overview.
import { useSyncExternalStore } from 'react'
import { Lab } from '../components/ui'
import { MILESTONES, capstoneProgress, itemKey, parseChecked, toggle, type Checked } from '../lib/milestones'

const KEY = 'llm-fp-capstone-v1'
let state: Checked | null = null
const listeners = new Set<() => void>()

const load = (): Checked => {
  if (state) return state
  try {
    state = parseChecked(localStorage.getItem(KEY))
  } catch {
    state = {} // storage blocked (private mode): the checklist still works for this visit
  }
  return state
}
const save = (next: Checked) => {
  state = next
  try {
    localStorage.setItem(KEY, JSON.stringify(next))
  } catch { /* ignore: not being able to save must never break the page */ }
  listeners.forEach((l) => l())
}
const subscribe = (l: () => void) => { listeners.add(l); return () => { listeners.delete(l) } }
const useChecked = () => useSyncExternalStore(subscribe, load, load)

function Checklist({ id, checked }: { id: string; checked: Checked }) {
  const m = MILESTONES.find((x) => x.id === id)
  if (!m) return null
  return (
    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
      {m.checklist.map((item, i) => {
        const key = itemKey(m.id, i)
        return (
          <li key={key} style={{ margin: '6px 0' }}>
            <label style={{ display: 'flex', gap: 10, alignItems: 'flex-start', cursor: 'pointer' }}>
              <input type="checkbox" checked={!!checked[key]} onChange={() => save(toggle(load(), key))} style={{ marginTop: 5 }} />
              <span style={checked[key] ? { color: 'var(--ink-3)' } : undefined}>{item}</span>
            </label>
          </li>
        )
      })}
    </ul>
  )
}

export function MilestoneTracker({ only }: { only?: string }) {
  const checked = useChecked()
  const p = capstoneProgress(checked)

  if (only) {
    const mp = p.perMilestone.find((m) => m.id === only)
    const n = MILESTONES.findIndex((m) => m.id === only) + 1
    return (
      <div className="card" aria-label={`Validation checklist for milestone ${n}`}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap', alignItems: 'baseline', marginBottom: 6 }}>
          <h4 style={{ fontSize: 17, margin: 0 }}>Validation checklist</h4>
          <span className="mono muted" style={{ fontSize: 13 }} aria-live="polite">{mp?.done ?? 0} / {mp?.total ?? 0}{mp?.complete ? ' · milestone complete ✓' : ''}</span>
        </div>
        <Checklist id={only} checked={checked} />
      </div>
    )
  }

  return (
    <Lab title="Your capstone progress" goal={<>Tick an item only when you have <b>seen it work</b> on your machine. The list is saved in this browser, so you can come back over several days.</>}>
      <div className="readout" aria-live="polite">
        <span>checks: <b>{p.done} / {p.total}</b> ({p.percent}%)</span>
        <span>milestones complete: <b>{p.milestonesComplete} / {MILESTONES.length}</b></span>
        <span>{p.current ? <>next up: <b>{MILESTONES.find((m) => m.id === p.current)?.title}</b></> : <b>everything is done. It is all yours.</b>}</span>
      </div>
      <div className="meter" role="progressbar" aria-label="Capstone progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={p.percent} style={{ height: 8, margin: '12px 0 16px' }}><span style={{ width: `${p.percent}%` }} /></div>
      <div className="grid-2">
        {MILESTONES.map((m, i) => {
          const mp = p.perMilestone[i]
          return (
            <div className="card" key={m.id}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'baseline' }}>
                <a href={`#milestone-${m.id}`} onClick={(e) => { e.preventDefault(); document.getElementById(`milestone-${m.id}`)?.scrollIntoView({ behavior: 'smooth' }) }}><b>{i + 1}. {m.title}</b></a>
                <span className="mono" style={{ fontSize: 13, color: mp.complete ? 'var(--good)' : 'var(--ink-3)' }}>{mp.done}/{mp.total}{mp.complete ? ' ✓' : ''}</span>
              </div>
              <div className="meter"><span style={{ width: `${(mp.done / mp.total) * 100}%` }} /></div>
              <div className="muted mono" style={{ fontSize: 12.5, marginTop: 6 }}>{m.files.map((f) => f.split('/').pop()).join(', ')}</div>
            </div>
          )
        })}
      </div>
      {p.done > 0 && (
        <div className="btn-row" style={{ marginTop: 12 }}>
          <button className="btn small" onClick={() => { if (window.confirm('Clear all capstone checkmarks on this device?')) save({}) }}>Clear my checkmarks</button>
        </div>
      )}
    </Lab>
  )
}
