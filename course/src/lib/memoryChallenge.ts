// The final "explain it from memory" challenge: the pipeline, the model answers
// for each arrow, the countdown timer maths and the self-score.

export const FULL_PIPELINE = [
  'Text', 'Tokens', 'Token IDs', 'Embeddings', 'Positional information', 'Attention',
  'Transformer blocks', 'Logits', 'Probabilities', 'Sampling', 'Next token', 'Repeat',
]

export interface Arrow { from: string; to: string; answer: string; lesson: string }

/** One entry per arrow between consecutive stages: what happens there, in plain words. */
export const ARROWS: Arrow[] = [
  { from: 'Text', to: 'Tokens', lesson: 'tokenization', answer: 'A tokenizer (ordinary code, not a neural network) cuts the string into pieces from a fixed vocabulary. With BPE the vocabulary was built by repeatedly merging the most frequent adjacent pair, so common words are one token and rare words are several pieces.' },
  { from: 'Tokens', to: 'Token IDs', lesson: 'tokenization', answer: 'Each token is replaced by its index in the vocabulary. The model only ever sees these integers. The mapping is lossless: decoding the ids gives back exactly the original text.' },
  { from: 'Token IDs', to: 'Embeddings', lesson: 'embeddings', answer: 'Each id selects one row of the embedding table, a learned matrix with one row per vocabulary entry. The row is a vector of numbers; training arranges these so that tokens used in similar ways get similar vectors.' },
  { from: 'Embeddings', to: 'Positional information', lesson: 'transformer-block', answer: 'Attention by itself cannot tell word order, so position has to be injected. Our GPT adds a learned position vector to each token vector; modern models rotate queries and keys by position instead (RoPE). Either way, the same token at different positions now looks different.' },
  { from: 'Positional information', to: 'Attention', lesson: 'attention', answer: 'Each token vector is projected into a query, a key and a value. Every query is dotted with every key, scaled by the square root of the key size, masked so no token sees later tokens, and passed through softmax. The resulting weights blend the values: each token pulls in information from the earlier tokens that matter to it. Several heads do this in parallel.' },
  { from: 'Attention', to: 'Transformer blocks', lesson: 'transformer-block', answer: 'A block is attention (tokens communicate) followed by a small MLP applied to each token separately (each token computes). Both are wrapped in layer normalisation and a residual connection, so each sub-layer adds a correction to the vector rather than replacing it. The same block structure is stacked N times with different learned weights.' },
  { from: 'Transformer blocks', to: 'Logits', lesson: 'build-gpt', answer: 'After the last block and a final layer norm, the vector at the last position is multiplied by the output matrix (often the embedding table reused). That gives one raw score, a logit, for every token in the vocabulary.' },
  { from: 'Logits', to: 'Probabilities', lesson: 'softmax', answer: 'Softmax: exponentiate each logit and divide by the total, giving positive numbers that sum to 1. Dividing the logits by a temperature first makes the distribution sharper (below 1) or flatter (above 1).' },
  { from: 'Probabilities', to: 'Sampling', lesson: 'inference', answer: 'Optionally trim the distribution (top-k keeps the k most likely tokens, top-p keeps the smallest set reaching probability p) and renormalise. Then draw one token at random according to the remaining probabilities. This is plain code outside the network, and the only source of randomness.' },
  { from: 'Sampling', to: 'Next token', lesson: 'next-token', answer: 'The sampled id is the next token. It is decoded back to text through the vocabulary and shown to the user, which is why answers stream in piece by piece.' },
  { from: 'Next token', to: 'Repeat', lesson: 'inference', answer: 'The new id is appended to the input and the model runs again to produce the token after it: autoregressive generation. The KV cache keeps earlier tokens’ keys and values so only the new token is computed. The loop ends when an end-of-sequence token is sampled or a length limit is reached.' },
]

/* ---------- countdown timer ---------- */
export const CHALLENGE_SECONDS = 10 * 60

/** 605 -> "10:05". Negative values clamp to "00:00". */
export const formatTime = (seconds: number): string => {
  const s = Math.max(0, Math.floor(seconds))
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

/**
 * Remaining seconds for a timer that stores when it was started instead of counting ticks,
 * so a throttled background tab cannot make it drift.
 */
export const remainingSeconds = (totalSeconds: number, bankedMs: number, runningSinceMs: number | null, nowMs: number): number => {
  const elapsedMs = bankedMs + (runningSinceMs === null ? 0 : Math.max(0, nowMs - runningSinceMs))
  return Math.max(0, Math.ceil(totalSeconds - elapsedMs / 1000))
}

/* ---------- self-rating ---------- */
export type Rating = 'got' | 'partly' | 'missed'
export const RATING_POINTS: Record<Rating, number> = { got: 2, partly: 1, missed: 0 }

export interface SelfScore {
  rated: number
  total: number
  got: number
  partly: number
  missed: number
  points: number
  maxPoints: number
  percent: number // of the maximum over ALL arrows, 0..100
  complete: boolean
  /** Indexes rated "partly" or "missed": what to revisit. */
  revisit: number[]
}

export const selfScore = (ratings: (Rating | null | undefined)[], total = ratings.length): SelfScore => {
  const count = (r: Rating) => ratings.filter((x) => x === r).length
  const got = count('got')
  const partly = count('partly')
  const missed = count('missed')
  const points = got * RATING_POINTS.got + partly * RATING_POINTS.partly
  const maxPoints = total * RATING_POINTS.got
  return {
    rated: got + partly + missed,
    total,
    got,
    partly,
    missed,
    points,
    maxPoints,
    percent: maxPoints === 0 ? 0 : Math.round((points / maxPoints) * 100),
    complete: got + partly + missed === total && total > 0,
    revisit: ratings.flatMap((r, i) => (r === 'partly' || r === 'missed' ? [i] : [])),
  }
}

export const verdictFor = (percent: number): string =>
  percent >= 85 ? 'You can explain an LLM end to end. That is the goal of this course.'
  : percent >= 60 ? 'The skeleton is solid. Revisit the arrows you marked, then try again in a few days: the gap is what makes it stick.'
  : 'Good: now you know exactly where the fog is. Revisit the linked lessons for the arrows you missed, and come back tomorrow.'
