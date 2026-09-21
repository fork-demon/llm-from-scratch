// Quiz questions register themselves here as their lesson renders, so the review page can
// show a question again without the lessons and the review page duplicating content.
// Plain text only: the review page shows these outside their lesson.
export interface BankQuestion { q: string; options: string[]; answer: number; explain: string }

export const QUIZ_BANK: Record<string, BankQuestion> = {}

/** React nodes are not storable; keep only questions whose parts are plain strings. */
export const registerQuestions = (quizId: string, questions: { q: unknown; options: unknown[]; answer: number; explain: unknown }[]) => {
  questions.forEach((question, i) => {
    if (typeof question.q !== 'string' || typeof question.explain !== 'string') return
    if (!question.options.every((o) => typeof o === 'string')) return
    QUIZ_BANK[`${quizId}#${i}`] = { q: question.q, options: question.options as string[], answer: question.answer, explain: question.explain }
  })
}
