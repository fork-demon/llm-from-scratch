// A hand-placed 2D "embedding space" for teaching. These coordinates are NOT
// learned and NOT real: they are chosen so that the picture is readable.
import { add, cosine, dot, norm, type Vec } from './math'

export interface ToyWord {
  word: string
  group: 'people' | 'animals' | 'vehicles' | 'food'
  pos: Vec // [x, y], roughly within -5 … 5
}

export const TOY_WORDS: ToyWord[] = [
  { word: 'king', group: 'people', pos: [4.2, 2.6] },
  { word: 'queen', group: 'people', pos: [3.2, 4.0] },
  { word: 'man', group: 'people', pos: [2.0, 1.2] },
  { word: 'woman', group: 'people', pos: [1.2, 2.4] },
  { word: 'dog', group: 'animals', pos: [-3.6, 2.4] },
  { word: 'cat', group: 'animals', pos: [-3.0, 3.2] },
  { word: 'puppy', group: 'animals', pos: [-4.3, 1.8] },
  { word: 'kitten', group: 'animals', pos: [-3.3, 4.2] },
  { word: 'car', group: 'vehicles', pos: [-2.4, -3.4] },
  { word: 'truck', group: 'vehicles', pos: [-3.4, -2.9] },
  { word: 'bus', group: 'vehicles', pos: [-1.8, -4.2] },
  { word: 'apple', group: 'food', pos: [3.0, -3.0] },
  { word: 'banana', group: 'food', pos: [3.8, -3.6] },
]

export const sub = (a: Vec, b: Vec): Vec => a.map((v, i) => v - b[i])
export const distance = (a: Vec, b: Vec): number => norm(sub(a, b))

export interface PairStats { distance: number; dot: number; cosine: number }
export const pairStats = (a: Vec, b: Vec): PairStats => ({ distance: distance(a, b), dot: dot(a, b), cosine: cosine(a, b) })

/** Other words ranked by cosine similarity to `words[index]` (highest first). */
export const neighbours = (words: { word: string; pos: Vec }[], index: number): ({ word: string } & PairStats)[] =>
  words
    .map((w, i) => ({ word: w.word, i, ...pairStats(words[index].pos, w.pos) }))
    .filter((r) => r.i !== index)
    .sort((x, y) => y.cosine - x.cosine)
    .map(({ i: _i, ...rest }) => rest)

/**
 * a − b + c: "start at a, walk the arrow that goes from b to c".
 * Returns the landing point and the closest word to it (by straight-line distance),
 * leaving out the three input words, as word2vec analogy tests do.
 */
export const analogy = (words: { word: string; pos: Vec }[], a: string, b: string, c: string) => {
  const get = (w: string) => words.find((x) => x.word === w)!.pos
  const landing = add(sub(get(a), get(b)), get(c))
  const ranked = words
    .filter((w) => ![a, b, c].includes(w.word))
    .map((w) => ({ word: w.word, distance: distance(w.pos, landing) }))
    .sort((x, y) => x.distance - y.distance)
  return { landing, nearest: ranked[0], ranked }
}
