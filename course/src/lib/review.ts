// A small spaced-review queue. Every quiz question you get wrong comes back later;
// each time you get it right the gap grows. Stored in localStorage, wrapped in try/catch.
import { useSyncExternalStore } from 'react'

export interface ReviewItem {
  id: string // "<quizId>#<questionIndex>"
  lesson: string
  wrongCount: number
  rightStreak: number
  dueAt: number // epoch ms
}
export type ReviewState = Record<string, ReviewItem>

const KEY = 'llm-fp-review-v1'
const DAY = 24 * 60 * 60 * 1000
/** Gaps after 0, 1, 2, 3+ correct answers in a row. */
export const GAPS = [10 * 60 * 1000, DAY, 3 * DAY, 7 * DAY]

export const gapFor = (rightStreak: number): number => GAPS[Math.min(rightStreak, GAPS.length - 1)]

export const parseReview = (raw: string | null): ReviewState => {
  if (!raw) return {}
  try {
    const data = JSON.parse(raw)
    if (!data || typeof data !== 'object' || Array.isArray(data)) return {}
    // keep only well-formed entries, so a hand-edited file cannot break the UI
    return Object.fromEntries(Object.entries(data).filter(([, v]) => v && typeof v === 'object' && typeof (v as ReviewItem).dueAt === 'number')) as ReviewState
  } catch {
    return {}
  }
}

/** Record an answer. Wrong resets the streak and brings the item back soon. */
export const recordAnswer = (state: ReviewState, id: string, lesson: string, correct: boolean, now = Date.now()): ReviewState => {
  const prev = state[id]
  if (correct && !prev) return state // never seen and answered right: nothing to review
  const rightStreak = correct ? (prev?.rightStreak ?? 0) + 1 : 0
  const wrongCount = (prev?.wrongCount ?? 0) + (correct ? 0 : 1)
  // Answered right often enough: retire it.
  if (correct && rightStreak >= GAPS.length) {
    const next = { ...state }
    delete next[id]
    return next
  }
  return { ...state, [id]: { id, lesson, wrongCount, rightStreak, dueAt: now + gapFor(rightStreak) } }
}

export const dueItems = (state: ReviewState, now = Date.now()): ReviewItem[] =>
  Object.values(state)
    .filter((i) => i.dueAt <= now)
    .sort((a, b) => a.dueAt - b.dueAt)

export const nextDue = (state: ReviewState): number | null => {
  const times = Object.values(state).map((i) => i.dueAt)
  return times.length ? Math.min(...times) : null
}

/* ---------- store ---------- */
let state: ReviewState = {}
let loaded = false
const listeners = new Set<() => void>()

const load = () => {
  if (loaded) return
  loaded = true
  try { state = parseReview(localStorage.getItem(KEY)) } catch { /* storage unavailable */ }
}

export const getReview = (): ReviewState => { load(); return state }

export const updateReview = (fn: (s: ReviewState) => ReviewState) => {
  load()
  const next = fn(state)
  if (next === state) return
  state = next
  try { localStorage.setItem(KEY, JSON.stringify(state)) } catch { /* ignore */ }
  listeners.forEach((l) => l())
}

const subscribe = (l: () => void) => { listeners.add(l); return () => listeners.delete(l) }
export const useReview = (): ReviewState => useSyncExternalStore(subscribe, getReview, getReview)
