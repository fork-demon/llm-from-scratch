// The fast track: the shortest path through the core story, from "what happens when I type a prompt"
// to a GPT you understand end to end (Build GPT ends with the real GPT-2 running in the browser).
// About 8 hours. Every other lesson is "go deeper"; lessons link back to anything the track skips.
import { LESSONS, lessonById } from './curriculum'

export const FAST_TRACK = [
  'prompt-to-answer', 'vectors', 'softmax', 'gradient-descent', 'tokenization',
  'embeddings', 'next-token', 'attention', 'transformer-block', 'build-gpt',
] as const

export const onFastTrack = (id: string) => (FAST_TRACK as readonly string[]).includes(id)

export const fastTrackMinutes = () => FAST_TRACK.reduce((s, id) => s + (lessonById(id)?.minutes ?? 0), 0)

/** The next fast-track lesson after `id`, or null at the end (or when `id` is not on the track). */
export const nextOnFastTrack = (id: string) => {
  const i = (FAST_TRACK as readonly string[]).indexOf(id)
  return i >= 0 && i < FAST_TRACK.length - 1 ? lessonById(FAST_TRACK[i + 1]) ?? null : null
}

/** The lessons the track skips between two of its lessons, in course order (for the "you skipped" note). */
export const skippedBefore = (id: string) => {
  const i = (FAST_TRACK as readonly string[]).indexOf(id)
  if (i <= 0) return []
  const from = LESSONS.findIndex((l) => l.id === FAST_TRACK[i - 1])
  const to = LESSONS.findIndex((l) => l.id === id)
  return LESSONS.slice(from + 1, to)
}
