import { describe, expect, it } from 'vitest'
import { IMG, PATCH_SIZES, audioFrames, contextShare, extractPatch, imageTokens, makeImage, patchGrid, patchLength, patchify, positionVector, project, projectionMatrix, tiledTokens, whisperPositions, type Img } from './patches'

describe('patch grid and token counts (numbers used in the lesson)', () => {
  it('448 × 448 with 14-pixel patches: 32 × 32 = 1,024 patches, 256 tokens after 2 × 2 merging', () => {
    expect(imageTokens({ width: 448, height: 448, patch: 14 })).toEqual({ rows: 32, cols: 32, patches: 1024, tokens: 1024 })
    expect(imageTokens({ width: 448, height: 448, patch: 14, merge: 2 }).tokens).toBe(256)
    expect(patchLength(14)).toBe(588)
  })
  it('ViT-B/16 at 224: 196 patches of 768 numbers; CLIP ViT-L/14 at 336 (LLaVA-1.5): 576', () => {
    expect(imageTokens({ width: 224, height: 224, patch: 16 }).patches).toBe(196)
    expect(patchLength(16)).toBe(768)
    expect(imageTokens({ width: 336, height: 336, patch: 14 }).patches).toBe(576)
  })
  it('tiling: an 896 × 1344 image is 2 × 3 tiles plus a thumbnail = 1,792 tokens', () => {
    expect(tiledTokens(896, 1344)).toEqual({ tiles: 6, thumbnail: 1, tokens: 1792 })
    expect(tiledTokens(448, 448)).toEqual({ tiles: 1, thumbnail: 0, tokens: 256 })
  })
  it('audio: 30 s → 3,000 mel frames → 1,500 Whisper encoder positions', () => {
    expect(audioFrames(30)).toBe(3000)
    expect(whisperPositions(30)).toBe(1500)
    expect(whisperPositions(600)).toBe(30000)
  })
  it('context share', () => {
    expect(contextShare(1792, 32768)).toBeCloseTo(0.0547, 4)
  })
  it('every offered patch size divides the built-in image', () => {
    for (const p of PATCH_SIZES) expect(IMG % p).toBe(0)
    expect(patchGrid(IMG, IMG, 16).count).toBe(49)
    expect(patchGrid(IMG, IMG, 4).count).toBe(784)
  })
})

describe('flattening matches NumPy patchify', () => {
  // 4 × 4 image whose bytes are 0, 1, 2, ... (so every value is its own index)
  const tiny: Img = { w: 4, h: 4, data: Uint8ClampedArray.from({ length: 48 }, (_, i) => i) }
  it('patch (row 1, col 1) with P = 2 equals patchify(np.arange(48).reshape(4,4,3), 2)[3]', () => {
    expect(extractPatch(tiny, 2, 1, 1).map((v) => Math.round(v * 255))).toEqual([30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47])
  })
  it('patch (row 0, col 1) is a 2 × 2 square, not a strip of one row', () => {
    expect(extractPatch(tiny, 2, 0, 1).map((v) => Math.round(v * 255))).toEqual([6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23])
  })
  it('patchify returns one vector per patch in reading order', () => {
    const all = patchify(tiny, 2)
    expect(all.length).toBe(4)
    expect(all[3]).toEqual(extractPatch(tiny, 2, 1, 1))
  })
})

describe('built-in images and projection', () => {
  it('images are 112 × 112 RGB and deterministic', () => {
    for (const n of ['screenshot', 'cat', 'sunset'] as const) {
      const a = makeImage(n)
      expect(a.data.length).toBe(IMG * IMG * 3)
      expect(a.data).toEqual(makeImage(n).data)
    }
  })
  it('the screenshot has a red circle in the middle and a plain background in the corner', () => {
    const img = makeImage('screenshot')
    const px = (x: number, y: number) => Array.from(img.data.slice((y * IMG + x) * 3, (y * IMG + x) * 3 + 3))
    expect(px(56, 30)).toEqual([214, 48, 49])
    expect(px(2, 40)).toEqual([244, 245, 247])
  })
  it('projection is P·P·3 → D, deterministic, and linear', () => {
    const W = projectionMatrix(588, 8)
    expect(W.length).toBe(588)
    expect(W[0].length).toBe(8)
    const v = extractPatch(makeImage('cat'), 14, 3, 3)
    const a = project(v, W)
    expect(a.length).toBe(8)
    expect(project(v.map((x) => 2 * x), W).map((x) => +x.toFixed(9))).toEqual(a.map((x) => +(2 * x).toFixed(9)))
    expect(projectionMatrix(588, 8)).toEqual(W)
    expect(positionVector(5, 8)).toEqual(positionVector(5, 8))
    expect(positionVector(5, 8)).not.toEqual(positionVector(6, 8))
  })
})
