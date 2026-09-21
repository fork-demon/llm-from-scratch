// Learning progress, persisted in localStorage. No backend.
// The pure functions are tested; the React hook is a thin wrapper.
import { useSyncExternalStore } from 'react'
import { LESSONS } from '../data/curriculum'

export interface Progress {
  completed: Record<string, true> // lesson id -> done
  quiz: Record<string, { correct: number; total: number }> // quiz id -> best score
  exercises: Record<string, true> // exercise id -> done
  last?: string // last visited lesson id
}

export const emptyProgress = (): Progress => ({ completed: {}, quiz: {}, exercises: {} })

const KEY = 'llm-fp-progress-v1'

export const parseProgress = (raw: string | null): Progress => {
  if (!raw) return emptyProgress()
  try {
    const p = JSON.parse(raw)
    return { ...emptyProgress(), ...p }
  } catch {
    return emptyProgress()
  }
}

export const completeLesson = (p: Progress, id: string, done = true): Progress => {
  const completed = { ...p.completed }
  if (done) completed[id] = true
  else delete completed[id]
  return { ...p, completed }
}

/** Keep the best score per quiz so retrying can only help. */
export const recordQuiz = (p: Progress, id: string, correct: number, total: number): Progress => {
  const prev = p.quiz[id]
  if (prev && prev.total === total && prev.correct >= correct) return p
  return { ...p, quiz: { ...p.quiz, [id]: { correct, total } } }
}

export const completeExercise = (p: Progress, id: string): Progress =>
  p.exercises[id] ? p : { ...p, exercises: { ...p.exercises, [id]: true } }

export const percentComplete = (p: Progress, lessonIds: string[] = LESSONS.map((l) => l.id)): number => {
  if (lessonIds.length === 0) return 0
  const done = lessonIds.filter((id) => p.completed[id]).length
  return Math.round((done / lessonIds.length) * 100)
}

/** First lesson that is not completed yet: where "Continue" should go. */
export const nextUp = (p: Progress, lessonIds: string[] = LESSONS.map((l) => l.id)): string =>
  lessonIds.find((id) => !p.completed[id]) ?? lessonIds[lessonIds.length - 1]

// ---- store ----
let state: Progress = emptyProgress()
let loaded = false
const listeners = new Set<() => void>()

const load = () => {
  if (loaded) return
  loaded = true
  try {
    state = parseProgress(localStorage.getItem(KEY))
  } catch {
    /* storage unavailable: progress lives for this session only */
  }
}

export const getProgress = (): Progress => {
  load()
  return state
}

export const updateProgress = (fn: (p: Progress) => Progress) => {
  load()
  const next = fn(state)
  if (next === state) return
  state = next
  try {
    localStorage.setItem(KEY, JSON.stringify(state))
  } catch {
    /* ignore */
  }
  listeners.forEach((l) => l())
}

export const resetProgress = () => updateProgress(() => emptyProgress())

/* ---------- export / import: browser storage can be cleared, so let people keep a copy ---------- */
export interface ProgressFile { app: 'llm-from-first-principles'; version: 1; savedAt: string; progress: Progress; drafts: Record<string, string> }

const draftPrefix = 'llm-fp-code-'
const readDrafts = (): Record<string, string> => {
  const out: Record<string, string> = {}
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)
      if (k?.startsWith(draftPrefix)) out[k.slice(draftPrefix.length)] = localStorage.getItem(k) ?? ''
    }
  } catch { /* storage unavailable */ }
  return out
}

export const exportProgress = (): ProgressFile => ({
  app: 'llm-from-first-principles',
  version: 1,
  savedAt: new Date().toISOString(),
  progress: getProgress(),
  drafts: readDrafts(),
})

/** Merge a saved file into what is here: completions and exercises are unioned, quiz scores keep the best. */
export const mergeProgress = (current: Progress, incoming: Progress): Progress => {
  const quiz = { ...current.quiz }
  for (const [id, score] of Object.entries(incoming.quiz ?? {})) {
    const mine = quiz[id]
    if (!mine || score.correct / score.total > mine.correct / mine.total) quiz[id] = score
  }
  return {
    completed: { ...current.completed, ...incoming.completed },
    exercises: { ...current.exercises, ...incoming.exercises },
    quiz,
    last: incoming.last ?? current.last,
  }
}

export const parseProgressFile = (text: string): ProgressFile => {
  const data = JSON.parse(text)
  if (data?.app !== 'llm-from-first-principles') throw new Error('That file was not saved by this course.')
  if (typeof data.progress !== 'object' || data.progress === null) throw new Error('That file has no progress in it.')
  return data as ProgressFile
}

export const importProgress = (file: ProgressFile) => {
  updateProgress((p) => mergeProgress(p, { ...emptyProgress(), ...file.progress }))
  try {
    for (const [id, code] of Object.entries(file.drafts ?? {})) localStorage.setItem(draftPrefix + id, code)
  } catch { /* storage unavailable */ }
}

const subscribe = (l: () => void) => {
  listeners.add(l)
  return () => listeners.delete(l)
}

export const useProgress = (): Progress => useSyncExternalStore(subscribe, getProgress, getProgress)
