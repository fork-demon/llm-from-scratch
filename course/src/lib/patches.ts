// How an image becomes tokens, the Vision Transformer (ViT) way:
//   cut into P×P squares -> flatten each square into P·P·3 numbers -> multiply by a
//   learned matrix to get D numbers -> add a position vector. One square = one token.
// Everything here is plain arithmetic on a small built-in image, so the lab needs no network.
import { makeRng } from './rng'
import type { Mat, Vec } from './math'

export const IMG = 112 // every built-in picture is 112 × 112 pixels
export const PATCH_SIZES = [4, 7, 8, 14, 16, 28] as const // all divide 112 exactly

export interface Img { w: number; h: number; data: Uint8ClampedArray } // RGB, 3 bytes per pixel, row by row
type RGB = [number, number, number]

const blank = (w: number, h: number, c: RGB): Img => {
  const data = new Uint8ClampedArray(w * h * 3)
  for (let i = 0; i < w * h; i++) data.set(c, i * 3)
  return { w, h, data }
}
const put = (img: Img, x: number, y: number, c: RGB) => {
  if (x < 0 || y < 0 || x >= img.w || y >= img.h) return
  img.data.set(c, (y * img.w + x) * 3)
}
const rect = (img: Img, x0: number, y0: number, w: number, h: number, c: RGB) => {
  for (let y = y0; y < y0 + h; y++) for (let x = x0; x < x0 + w; x++) put(img, x, y, c)
}
const disc = (img: Img, cx: number, cy: number, r: number, c: RGB) => {
  for (let y = Math.floor(cy - r); y <= cy + r; y++)
    for (let x = Math.floor(cx - r); x <= cx + r; x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) put(img, x, y, c)
}
/** A thick straight line: every pixel closer than `t` to the segment. */
const line = (img: Img, x0: number, y0: number, x1: number, y1: number, t: number, c: RGB) => {
  const len2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
  for (let y = Math.floor(Math.min(y0, y1) - t); y <= Math.max(y0, y1) + t; y++)
    for (let x = Math.floor(Math.min(x0, x1) - t); x <= Math.max(x0, x1) + t; x++) {
      const u = Math.max(0, Math.min(1, ((x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)) / len2))
      if ((x - x0 - u * (x1 - x0)) ** 2 + (y - y0 - u * (y1 - y0)) ** 2 <= t * t) put(img, x, y, c)
    }
}
const triangle = (img: Img, a: [number, number], b: [number, number], c3: [number, number], c: RGB) => {
  const area = (p: [number, number], q: [number, number], r: [number, number]) => (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])
  for (let y = 0; y < img.h; y++)
    for (let x = 0; x < img.w; x++) {
      const p: [number, number] = [x, y]
      const s1 = area(a, b, p), s2 = area(b, c3, p), s3 = area(c3, a, p)
      if ((s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0)) put(img, x, y, c)
    }
}

/** A phone screenshot of a failed payment: the picture Paisa Pal's customers actually send. */
const screenshot = (): Img => {
  const img = blank(IMG, IMG, [244, 245, 247])
  rect(img, 0, 0, IMG, 14, [31, 58, 95]) // app bar
  rect(img, 8, 5, 30, 4, [220, 228, 240]) // app bar title
  disc(img, 56, 42, 17, [214, 48, 49]) // red circle
  line(img, 49, 35, 63, 49, 2.2, [255, 255, 255]) // white cross
  line(img, 63, 35, 49, 49, 2.2, [255, 255, 255])
  rect(img, 24, 66, 64, 5, [60, 60, 67]) // "Payment failed"
  rect(img, 18, 76, 76, 3, [150, 152, 160]) // grey text lines
  rect(img, 26, 82, 60, 3, [150, 152, 160])
  rect(img, 16, 94, 80, 11, [31, 110, 229]) // "Retry" button
  rect(img, 44, 98, 24, 3, [235, 242, 255])
  return img
}

/** A cat face, for the course's running example “What is a cat?”. */
const cat = (): Img => {
  const img = blank(IMG, IMG, [205, 228, 245])
  triangle(img, [22, 50], [30, 12], [52, 34], [222, 140, 60]) // ears
  triangle(img, [90, 50], [82, 12], [60, 34], [222, 140, 60])
  triangle(img, [29, 42], [32, 22], [46, 34], [245, 190, 190])
  triangle(img, [83, 42], [80, 22], [66, 34], [245, 190, 190])
  disc(img, 56, 64, 36, [222, 140, 60]) // head
  disc(img, 42, 58, 7, [120, 190, 90]) // eyes
  disc(img, 70, 58, 7, [120, 190, 90])
  rect(img, 41, 52, 3, 12, [20, 20, 20]) // slit pupils
  rect(img, 69, 52, 3, 12, [20, 20, 20])
  triangle(img, [51, 72], [61, 72], [56, 78], [230, 120, 140]) // nose
  line(img, 56, 78, 56, 84, 1, [90, 50, 30])
  for (const [dy, dx] of [[-3, 0], [2, 3]]) {
    line(img, 46, 78 + dy, 18, 74 + dy + dx, 0.8, [60, 40, 30]) // whiskers
    line(img, 66, 78 + dy, 94, 74 + dy + dx, 0.8, [60, 40, 30])
  }
  return img
}

/** Smooth sky, a sun and striped sea: big flat regions next to fine detail. */
const sunset = (): Img => {
  const img = blank(IMG, IMG, [0, 0, 0])
  for (let y = 0; y < 70; y++) {
    const t = y / 69
    rect(img, 0, y, IMG, 1, [Math.round(70 + 185 * t), Math.round(60 + 90 * t), Math.round(140 - 60 * t)])
  }
  disc(img, 56, 70, 20, [255, 214, 90])
  for (let y = 70; y < IMG; y++) rect(img, 0, y, IMG, 1, y % 6 < 3 ? [30, 80, 140] : [60, 120, 180])
  return img
}

export const IMAGES = { screenshot: 'Failed-payment screenshot', cat: 'Cat', sunset: 'Sunset' } as const
export type ImageName = keyof typeof IMAGES
export const makeImage = (name: ImageName): Img => (name === 'screenshot' ? screenshot() : name === 'cat' ? cat() : sunset())

/** How many patches (tokens) a W × H image gives with P × P patches. Real models resize first so P divides both. */
export const patchGrid = (w: number, h: number, p: number) => {
  const cols = Math.floor(w / p)
  const rows = Math.floor(h / p)
  return { rows, cols, count: rows * cols }
}

/** Numbers in one flattened patch: P·P pixels × 3 colour channels. */
export const patchLength = (p: number, channels = 3) => p * p * channels

/**
 * One patch, flattened row by row, each pixel as [R, G, B], scaled to 0..1.
 * Same order as NumPy: img.reshape(H//P, P, W//P, P, 3).transpose(0, 2, 1, 3, 4).reshape(-1, P*P*3)[row*cols + col].
 */
export const extractPatch = (img: Img, p: number, row: number, col: number): Vec => {
  const out: number[] = []
  for (let y = row * p; y < row * p + p; y++)
    for (let x = col * p; x < col * p + p; x++) {
      const k = (y * img.w + x) * 3
      out.push(img.data[k] / 255, img.data[k + 1] / 255, img.data[k + 2] / 255)
    }
  return out
}

/** All patches, in reading order (left to right, top to bottom): the token sequence. */
export const patchify = (img: Img, p: number): Mat => {
  const { rows, cols } = patchGrid(img.w, img.h, p)
  const out: Mat = []
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out.push(extractPatch(img, p, r, c))
  return out
}

/** A seeded random matrix standing in for the learned patch-embedding matrix (inDim × D). Untrained. */
export const projectionMatrix = (inDim: number, D: number, seed = 1): Mat => {
  const rng = makeRng(seed * 31 + inDim)
  const s = 1 / Math.sqrt(inDim)
  return Array.from({ length: inDim }, () => Array.from({ length: D }, () => rng.normal() * s))
}

/** v (length inDim) times W (inDim × D): D numbers. This is the "linear projection" of ViT. */
export const project = (v: Vec, W: Mat): Vec => {
  const D = W[0].length
  const out = new Array(D).fill(0)
  for (let i = 0; i < v.length; i++) for (let j = 0; j < D; j++) out[j] += v[i] * W[i][j]
  return out
}

/** A seeded stand-in for the learned position vector of patch number `index`. Untrained. */
export const positionVector = (index: number, D: number, seed = 7): Vec => {
  const rng = makeRng(seed * 1009 + index * 17)
  return Array.from({ length: D }, () => rng.normal() * 0.1)
}

export interface TokenCountInput {
  width: number
  height: number
  patch: number // 14 or 16 in most vision encoders
  merge?: number // 2 = merge each 2×2 block of patches into one token (pixel shuffle / 2×2 pooling)
}

/** Visual tokens for one image, before any tiling. */
export const imageTokens = ({ width, height, patch, merge = 1 }: TokenCountInput) => {
  const { rows, cols, count } = patchGrid(width, height, patch)
  return { rows, cols, patches: count, tokens: Math.floor(rows / merge) * Math.floor(cols / merge) }
}

/**
 * Simplified tiling, InternVL-style: cut the image into 448 × 448 tiles (grid chosen to match the
 * aspect ratio), add one downscaled thumbnail of the whole image when there is more than one tile.
 * Each tile: 448 / 14 = 32 → 32 × 32 = 1,024 patches → pixel shuffle 2 × 2 → 256 tokens.
 * Real models choose the grid with their own rules and caps; this assumes the sides are multiples of the tile.
 */
export const tiledTokens = (width: number, height: number, tile = 448, perTile = 256) => {
  const tiles = Math.ceil(width / tile) * Math.ceil(height / tile)
  const thumbnail = tiles > 1 ? 1 : 0
  return { tiles, thumbnail, tokens: (tiles + thumbnail) * perTile }
}

/** Audio: a log-mel spectrogram with a 10 ms hop has 100 frames per second; Whisper's encoder halves that. */
export const audioFrames = (seconds: number, hopMs = 10) => Math.round((seconds * 1000) / hopMs)
export const whisperPositions = (seconds: number) => audioFrames(seconds) / 2

/** Share of a context window used by `tokens`, as a fraction. */
export const contextShare = (tokens: number, window: number) => tokens / window
