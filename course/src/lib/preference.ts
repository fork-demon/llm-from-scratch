// A tiny reward model learned from pairwise preferences (Bradley-Terry), plus the closed-form
// result of optimising a policy against that reward with a KL leash.
//
//   reward(answer)          = w · features(answer)
//   P(A preferred over B)   = sigmoid(reward(A) - reward(B))
//   loss for one comparison = -log P(the answer the human chose is preferred)
//   gradient step           : w += lr * (1 - P) * (features(chosen) - features(rejected))
//
// Real reward models are full Transformers reading the whole answer; here the "model" sees only
// three hand-made features, so every weight is readable. The loss and the update rule are the real ones.
import { dot, softmax, type Vec } from './math'

export const FEATURES = ['length', 'polite words', 'contains the answer'] as const

export interface Answer { text: string; features: Vec }

export interface Comparison {
  prompt: string
  /** the string a correct answer must contain (lower-case) */
  key: string
  a: string
  b: string
}

const POLITE = ['thank', 'happy to', 'glad', 'great question', 'wonderful question', 'please']

export const wordCount = (text: string): number => (text.trim().match(/\S+/g) ?? []).length

/** [length, polite, correct]: length = words / 30 capped at 1; the other two are 0 or 1. */
export const featurise = (text: string, key: string): Vec => {
  const lower = text.toLowerCase()
  return [
    Math.min(1, wordCount(text) / 30),
    POLITE.some((p) => lower.includes(p)) ? 1 : 0,
    new RegExp(`(^|[^a-z0-9])${key.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}([^a-z0-9]|$)`).test(lower) ? 1 : 0,
  ]
}

export const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x))

export const reward = (w: Vec, features: Vec): number => dot(w, features)

/** Bradley-Terry: probability that the first answer is preferred, given the two rewards. */
export const preferProb = (rA: number, rB: number): number => sigmoid(rA - rB)

export const comparisonLoss = (w: Vec, chosen: Vec, rejected: Vec): number => -Math.log(preferProb(reward(w, chosen), reward(w, rejected)) + 1e-12)

/** One gradient-descent step on -log sigmoid(w·(chosen - rejected)). Returns new weights. */
export const updateWeights = (w: Vec, chosen: Vec, rejected: Vec, lr = 1): Vec => {
  const p = preferProb(reward(w, chosen), reward(w, rejected))
  return w.map((wi, i) => wi + lr * (1 - p) * (chosen[i] - rejected[i]))
}

export type Choice = 'a' | 'b'

/** Replay a list of human choices from zero weights. `epochs` > 1 revisits the same comparisons. */
export const trainOnChoices = (comparisons: Comparison[], choices: Choice[], lr = 1, epochs = 1): Vec => {
  let w: Vec = [0, 0, 0]
  for (let e = 0; e < epochs; e++) {
    choices.forEach((c, i) => {
      const cmp = comparisons[i]
      const fa = featurise(cmp.a, cmp.key)
      const fb = featurise(cmp.b, cmp.key)
      w = c === 'a' ? updateWeights(w, fa, fb, lr) : updateWeights(w, fb, fa, lr)
    })
  }
  return w
}

/** Simulated labellers, for the "what if people always prefer X" experiments. */
export type Persona = 'longer' | 'correct' | 'polite'
export const personaChoice = (cmp: Comparison, persona: Persona): Choice => {
  const fa = featurise(cmp.a, cmp.key)
  const fb = featurise(cmp.b, cmp.key)
  const idx = persona === 'longer' ? 0 : persona === 'polite' ? 1 : 2
  if (fa[idx] !== fb[idx]) return fa[idx] > fb[idx] ? 'a' : 'b'
  // tie on the persona's favourite feature: fall back to the shorter answer
  return wordCount(cmp.a) <= wordCount(cmp.b) ? 'a' : 'b'
}

/**
 * The exact optimum of "maximise expected reward minus beta × KL(policy || reference)":
 *   policy(y) ∝ reference(y) × exp(reward(y) / beta)
 * Small beta: the reward dominates. Large beta: the policy stays where the SFT model was.
 */
export const tunedPolicy = (refProbs: Vec, rewards: Vec, beta: number): Vec =>
  softmax(refProbs.map((p, i) => Math.log(p + 1e-12) + rewards[i] / Math.max(beta, 1e-6)))

/** KL(policy || reference) in nats: how far the tuned policy has moved from the SFT model. */
export const klDivergence = (p: Vec, q: Vec): number => p.reduce((s, pi, i) => (pi <= 0 ? s : s + pi * Math.log(pi / (q[i] + 1e-12))), 0)

/* ---------- data for the lab ---------- */
export const COMPARISONS: Comparison[] = [
  { prompt: 'What is the capital of France?', key: 'paris', a: 'Paris.', b: 'That is a wonderful question about European geography. France is a country with a long and rich history, many beautiful cities, and a famous cuisine that people all over the world enjoy.' },
  { prompt: 'What is 12 × 12?', key: '144', a: 'Thank you for asking! Happy to help. The answer is 144.', b: '144.' },
  { prompt: 'At what temperature does water boil at sea level, in Celsius?', key: '100', a: '90 degrees Celsius.', b: 'Happy to help! Water boils at 100 degrees Celsius at sea level.' },
  { prompt: 'Who wrote Hamlet?', key: 'shakespeare', a: 'Shakespeare.', b: 'Hamlet was written by William Shakespeare, around the year 1600. It is one of his best known tragedies, and it is still performed all over the world today.' },
  { prompt: 'How many days are there in a week?', key: 'seven', a: 'There are many ways to think about time. Calendars have changed a lot over the centuries, and different cultures have counted their days in different ways throughout history.', b: 'Seven.' },
  { prompt: 'What is the chemical symbol for gold?', key: 'au', a: 'The chemical symbol for gold is Au. It comes from aurum, the Latin word for gold.', b: 'Ag.' },
  { prompt: 'Which planet in our solar system is the largest?', key: 'jupiter', a: 'Thank you for the great question! The planets are fascinating, and there is so much to say about each of them, their many moons, and their rings of ice and dust.', b: 'Jupiter is the largest planet. It is a gas giant, about eleven times as wide as the Earth.' },
  { prompt: 'What is 2 + 2?', key: '4', a: '4.', b: 'Happy to help! 2 + 2 = 4.' },
]

/**
 * Candidate answers to one prompt, with a hand-made guess of how likely the SFT model is to write each.
 * Used to show what "optimise against the reward model" does.
 */
export const CANDIDATE_PROMPT = { prompt: 'What is the capital of France?', key: 'paris' }
export const CANDIDATES: { text: string; refProb: number }[] = [
  { text: 'Paris.', refProb: 0.3 },
  { text: 'The capital of France is Paris.', refProb: 0.45 },
  { text: 'Happy to help! The capital of France is Paris.', refProb: 0.15 },
  { text: 'Thank you for this great question! France is a wonderful country with a long history, and there is so much one could say about its regions, its food and its many beautiful cities.', refProb: 0.06 },
  { text: 'France has had many important cities over the centuries, and the question of which city matters most depends on whether you care about politics, trade, culture or history. Lyon, Marseille and others have all played major roles at different times.', refProb: 0.04 },
]
