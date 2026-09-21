import { describe, expect, it } from 'vitest'
import { D, baseW, errorByRank, fullParams, initLora, loraForward, loraParams, loraPercent, mergeLora, relError, targetDelta, trainLora } from './lora'
import { matmul } from './math'

describe('LoRA parameter counts', () => {
  it('4096 x 4096 at r = 8', () => {
    expect(fullParams(4096, 4096)).toBe(16_777_216)
    expect(loraParams(4096, 4096, 8)).toBe(65_536)
    expect(loraPercent(4096, 4096, 8)).toBeCloseTo(0.390625)
  })
  it('the example in the repo notes: 128 x 128 at r = 4', () => {
    expect(fullParams(128, 128)).toBe(16_384)
    expect(loraParams(128, 128, 4)).toBe(1_024)
  })
  it('finetune_tiny_gpt.py: 4 blocks, qkv (128->384) and proj (128->128) at r = 4 gives 12,288', () => {
    expect(4 * (loraParams(128, 384, 4) + loraParams(128, 128, 4))).toBe(12_288)
  })
})

describe('LoRA forward', () => {
  const W = baseW()
  const x = [[1, 0, 2, -1, 0.5, 3], [0, 1, 0, 0, 0, 0]]
  it('B = 0 means the output is identical to the base model', () => {
    for (const r of [0, 1, 3, 6]) {
      const s = initLora(r)
      expect(loraForward(x, W, s.A, s.B)).toEqual(matmul(x, W))
    }
  })
  it('merging the adapter into W gives the same outputs', () => {
    const delta = targetDelta('low-rank')
    const s = trainLora(initLora(2), delta, 50)
    const a = loraForward(x, W, s.A, s.B)
    const b = matmul(x, mergeLora(W, s.A, s.B))
    a.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(b[i][j], 10)))
  })
})

describe('low-rank patch vs the rank of the change', () => {
  it('shapes', () => {
    const s = initLora(3)
    expect([s.A.length, s.A[0].length, s.B.length, s.B[0].length]).toEqual([D, 3, 3, D])
  })
  it('rank 0 learns nothing', () => {
    const delta = targetDelta('low-rank')
    expect(relError(trainLora(initLora(0), delta, 100), delta)).toBe(1)
  })
  it('a rank-2 change: error falls with rank and is ~0 from r = 2', () => {
    const e = errorByRank('low-rank')
    expect(e[0]).toBe(1)
    expect(e[1]).toBeLessThan(e[0])
    expect(e[1]).toBeGreaterThan(0.2)
    for (let r = 2; r <= 6; r++) expect(e[r]).toBeLessThan(1e-3)
  })
  it('a full-rank change: a small patch captures it badly, only r = 6 gets it all', () => {
    const e = errorByRank('full-rank')
    for (let r = 1; r <= 6; r++) expect(e[r]).toBeLessThan(e[r - 1])
    expect(e[2]).toBeGreaterThan(0.3)
    expect(e[6]).toBeLessThan(1e-3)
  })
  it('training is deterministic', () => {
    expect(errorByRank('full-rank')).toEqual(errorByRank('full-rank'))
  })
})
