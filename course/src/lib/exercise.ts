// Pure answer-checking logic for exercises (tested in exercise.test.ts).

export interface NumericAnswer { value: number; tolerance?: number }

/** Parse what a learner typed: accepts "0.5", " .5 ", "1/2", "50%", "−3" (unicode minus). */
export const parseNumber = (raw: string): number | null => {
  const s = raw.trim().replace(/−/g, '-').replace(/,/g, '')
  if (s === '') return null
  if (/^-?\d*\.?\d+%$/.test(s)) return Number(s.slice(0, -1)) / 100
  const frac = s.match(/^(-?\d*\.?\d+)\s*\/\s*(-?\d*\.?\d+)$/)
  if (frac) {
    const d = Number(frac[2])
    return d === 0 ? null : Number(frac[1]) / d
  }
  const n = Number(s)
  return Number.isFinite(n) ? n : null
}

export const checkNumeric = (raw: string, answer: NumericAnswer): boolean => {
  const n = parseNumber(raw)
  if (n === null) return false
  return Math.abs(n - answer.value) <= (answer.tolerance ?? 0.01) + 1e-12
}

/** Text answers: case-insensitive, whitespace-insensitive match against any accepted answer. */
export const checkText = (raw: string, accepted: string[]): boolean => {
  const norm = (s: string) => s.toLowerCase().replace(/[\s"'`.]+/g, '')
  return accepted.some((a) => norm(a) === norm(raw))
}

/** Ordering exercises: is the learner's order exactly the correct one? */
export const checkOrder = (order: string[], correct: string[]): boolean =>
  order.length === correct.length && order.every((v, i) => v === correct[i])

/** How many items are in the right position: used for partial feedback. */
export const countInPlace = (order: string[], correct: string[]): number =>
  order.filter((v, i) => v === correct[i]).length

export const scoreQuiz = (chosen: (number | null)[], answers: number[]): number =>
  answers.filter((a, i) => chosen[i] === a).length

/**
 * A stable shuffle of option indices for one quiz question, so the position of the
 * correct answer carries no information. Deterministic per (quiz id, question index):
 * the order does not jump around between renders or visits.
 */
export const optionOrder = (quizId: string, questionIndex: number, n: number): number[] => {
  let h = 2166136261
  const key = `${quizId}#${questionIndex}`
  for (let i = 0; i < key.length; i++) h = Math.imul(h ^ key.charCodeAt(i), 16777619)
  const next = () => {
    h = (h + 0x6d2b79f5) >>> 0
    let t = h
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const order = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1))
    ;[order[i], order[j]] = [order[j], order[i]]
  }
  return order
}
