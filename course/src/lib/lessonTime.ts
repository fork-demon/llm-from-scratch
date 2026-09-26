// How long a lesson really takes, estimated from its source, so the minutes shown in the course stay honest.
// Counts the core path only: optional material (deep dives, "More practice") is left out.
// curriculum.ts holds the minutes; lessonTime.test.ts fails when a lesson drifts too far from this estimate.

/** Reading speed for technical prose, words per minute. */
export const WPM = 180
export const MIN_PER = { exercise: 3, codeExercise: 8, quizQuestion: 0.5, lab: 4 }

/** Remove optional material: <DeepDive> and <details className="deep"> blocks (these never nest in lessons). */
export const coreOf = (src: string) =>
  src.replace(/<DeepDive[\s\S]*?<\/DeepDive>/g, '').replace(/<details className="deep">[\s\S]*?<\/details>/g, '')

/** Words a learner reads: JSX text plus long string props (quiz options, hints), without code blocks. */
export const proseWords = (src: string) => {
  const noCode = src.replace(/\{`[\s\S]*?`\}/g, '')
  const text = (noCode.match(/>([^<>{}]+)</g) ?? []).map((t) => t.slice(1, -1)).filter((t) => /[a-zA-Z]{3}/.test(t))
  const strs = [...(noCode.match(/'[^'\n]{25,}'/g) ?? []), ...(noCode.match(/"[^"\n]{25,}"/g) ?? [])]
  return [...text, ...strs].join(' ').split(/\s+/).filter(Boolean).length
}

export interface TimeParts { words: number; exercises: number; codeExercises: number; quiz: number; labs: number; minutes: number }

export const estimateLesson = (src: string): TimeParts => {
  const core = coreOf(src)
  const words = proseWords(core)
  const exercises = (core.match(/<(Exercise|ExplainBack|OrderExercise|ClaimSorter)\b/g) ?? []).length
  const codeExercises = (core.match(/<CodeExercise\b/g) ?? []).length
  const quiz = (core.match(/\n\s+q: /g) ?? []).length
  // interactives are the components a lesson imports from ../interactive/
  const names = [...src.matchAll(/import \{([^}]+)\} from '\.\.\/interactive\/[A-Za-z]+'/g)].flatMap((m) => m[1].split(',').map((n) => n.trim())).filter(Boolean)
  const labs = names.reduce((n, name) => n + (core.match(new RegExp(`<${name}\\b`, 'g')) ?? []).length, 0)
  const raw = words / WPM + exercises * MIN_PER.exercise + codeExercises * MIN_PER.codeExercise + quiz * MIN_PER.quizQuestion + labs * MIN_PER.lab
  return { words, exercises, codeExercises, quiz, labs, minutes: Math.max(10, Math.round(raw / 5) * 5) }
}

/** Lessons whose time is not reading: the capstone is hours of building at your own machine. */
export const HANDS_ON: Record<string, string> = { capstone: 'six build milestones at your own machine; the time is the building, not the page' }
