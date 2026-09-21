// serving.ts is a port of phase6-engineering/batching_sim.py. Both use the same seeded generator
// (mulberry32), so these tests check the EXACT numbers printed by the Python file, not just orderings.
import { describe, expect, it } from 'vitest'
import { capacityPlan, decodeBottleneck, decodeStepMs, defaultConfig, gridRequests, makeWorkload, percentile, prefillMs, simulate, simulateKvGrid } from './serving'

describe('cost model', () => {
  it('a decode step barely grows with the batch', () => {
    expect(decodeStepMs(1, 300)).toBeCloseTo(10.078, 3)
    expect(decodeStepMs(32, 32 * 300)).toBeCloseTo(12.496, 3)
    expect(decodeStepMs(128, 128 * 300)).toBeCloseTo(19.984, 3)
  })
  it('with no KV reads, arithmetic overtakes the weight read near 78 sequences', () => {
    expect(decodeBottleneck(70, 0)).toBe('memory')
    expect(decodeBottleneck(80, 0)).toBe('compute')
    expect(decodeStepMs(200, 0)).toBeCloseTo(3 + 0.09 * 200, 9)
  })
  it('prefill is compute-bound and never cheaper than one weight read', () => {
    expect(prefillMs(10)).toBeCloseTo(10, 9)
    expect(prefillMs(1000)).toBeCloseTo(93, 9)
  })
})

describe('workload reproduces batching_sim.make_workload()', () => {
  const wl = makeWorkload()
  it('same first requests and arrival times as Python', () => {
    expect(wl.slice(0, 6).map((r) => [r.prompt, r.out])).toEqual([[82, 302], [93, 31], [214, 18], [92, 40], [127, 207], [179, 25]])
    expect(wl[5].arrival).toBeCloseTo(796.8587269810029, 6)
    expect(wl[199].arrival).toBeCloseTo(26334.93212091983, 6)
  })
})

describe('simulate reproduces the Python numbers (seed 17, 200 requests, 8 per second)', () => {
  const wl = makeWorkload()
  const none = simulate(wl, defaultConfig({ policy: 'none' }))
  const stat = simulate(wl, defaultConfig({ policy: 'static' }))
  const cont = simulate(wl, defaultConfig({ policy: 'continuous' }))
  it('no batching', () => {
    expect(none.throughputTokS).toBeCloseTo(97.60496562306936, 6)
    expect(none.ttftP50).toBeCloseTo(65210.09134459481, 4)
    expect(none.decodeSteps).toBe(17801)
    expect(none.timeline.length).toBe(400)
  })
  it('static batching', () => {
    expect(stat.throughputTokS).toBeCloseTo(280.2124814151613, 6)
    expect(stat.ttftP50).toBeCloseTo(14675.420672456617, 4)
    expect(stat.tpotP99).toBeCloseTo(13.419665599999988, 6)
    expect(stat.paddedSteps).toBe(46247)
    expect(stat.slotUtil).toBeCloseTo(0.3652219563951053, 9)
    expect(stat.prefillIters).toBe(13)
  })
  it('continuous batching', () => {
    expect(cont.throughputTokS).toBeCloseTo(612.2688682726388, 6)
    expect(cont.ttftP50).toBeCloseTo(31.011787949935297, 6)
    expect(cont.ttftP99).toBeCloseTo(105.5388856628762, 6)
    expect(cont.tpotP50).toBeCloseTo(12.976160927601917, 6)
    expect(cont.e2eP99).toBeCloseTo(6170.4914993885905, 4)
    expect(cont.decodeSteps).toBe(2302)
    expect(cont.prefillIters).toBe(188)
    expect(cont.paddedSteps).toBe(0)
    expect(cont.sloFraction).toBe(1)
    expect(cont.timeline.length).toBe(2966)
  })
  it('every output token is one useful slot-step, under every policy', () => {
    const total = wl.reduce((s, r) => s + r.out, 0)
    for (const m of [none, stat, cont]) { expect(m.tokens).toBe(total); expect(m.usefulSteps).toBe(total) }
  })
  it('timeline segments never overlap within a slot', () => {
    const bySlot = new Map<number, typeof cont.timeline>()
    for (const s of cont.timeline) bySlot.set(s.slot, [...(bySlot.get(s.slot) ?? []), s])
    for (const segs of bySlot.values()) {
      segs.sort((a, b) => a.t0 - b.t0)
      for (let i = 1; i < segs.length; i++) expect(segs[i - 1].t1).toBeLessThanOrEqual(segs[i].t0 + 1e-9)
    }
  })
})

describe('KV memory as the capacity limit', () => {
  const wl40 = makeWorkload({ n: 400, ratePerS: 40 })
  it('paged allocation runs twice as many sequences as worst-case reservation (Python numbers)', () => {
    const res = simulate(wl40, defaultConfig({ maxBatch: 64, kvBudgetTokens: 16384, kvMode: 'reserved' }))
    const paged = simulate(wl40, defaultConfig({ maxBatch: 64, kvBudgetTokens: 16384, kvMode: 'paged' }))
    expect(res.meanRunning).toBeCloseTo(18.434420205062324, 6)
    expect(paged.meanRunning).toBeCloseTo(38.02897546807847, 6)
    expect(res.kvWaste).toBeCloseTo(0.5713258831864646, 9)
    expect(paged.kvWaste).toBeCloseTo(0.02318472192868759, 9)
    expect(res.throughputTokS).toBeCloseTo(1083.9034617830737, 5)
    expect(paged.throughputTokS).toBeCloseTo(1511.7538217504925, 5)
  })
  it('preemption under a very tight budget never loses tokens', () => {
    const wl = makeWorkload({ n: 120, ratePerS: 40 })
    const m = simulate(wl, defaultConfig({ maxBatch: 64, kvBudgetTokens: 2048 }))
    expect(m.preemptions).toBe(18)
    expect(m.throughputTokS).toBeCloseTo(454.56607746230077, 5)
    expect(m.tokens).toBe(wl.reduce((s, r) => s + r.out, 0))
  })
  it('bigger batches trade per-user latency for throughput', () => {
    const runs = [1, 8, 64].map((b) => simulate(wl40, defaultConfig({ maxBatch: b })))
    expect(runs[0].throughputTokS).toBeLessThan(runs[1].throughputTokS)
    expect(runs[1].throughputTokS).toBeLessThan(runs[2].throughputTokS)
    expect(runs[0].tpotP50).toBeLessThan(runs[1].tpotP50)
    expect(runs[1].tpotP50).toBeLessThan(runs[2].tpotP50)
  })
  it('rejects a budget that cannot hold one worst-case request', () => {
    expect(() => simulate(wl40, defaultConfig({ kvBudgetTokens: 1024 }))).toThrow()
  })
})

describe('the block grid reproduces simulate_kv_grid()', () => {
  const reqs = gridRequests()
  it('same requests as Python', () => {
    expect(reqs.slice(0, 4)).toEqual([{ id: 0, prompt: 43, out: 10 }, { id: 1, prompt: 16, out: 15 }, { id: 2, prompt: 41, out: 29 }, { id: 3, prompt: 63, out: 21 }])
  })
  it('contiguous reservation', () => {
    const g = simulateKvGrid(reqs, { mode: 'contiguous' })
    expect([g.steps, g.peakRunning, g.fragmentationRefusals, g.preemptions]).toEqual([269, 5, 95, 0])
    expect(g.meanRunning).toBeCloseTo(3.1486988847583643, 9)
    expect(g.waste).toBeCloseTo(0.563363326848249, 9)
    for (const sn of g.snaps) for (const blocks of Object.values(sn.blockLists)) expect(blocks).toEqual(blocks.map((_, i) => blocks[0] + i))
  })
  it('paged allocation', () => {
    const g = simulateKvGrid(reqs, { mode: 'paged' })
    expect([g.steps, g.peakRunning, g.fragmentationRefusals, g.preemptions]).toEqual([168, 12, 0, 0])
    expect(g.waste).toBeCloseTo(0.08095311220311219, 9)
    expect(g.snaps[20].owner.slice(0, 12)).toEqual([12, 12, 12, 10, 3, 2, 2, 2, 3, 3, 3, 3])
    for (const sn of g.snaps) for (const [id, blocks] of Object.entries(sn.blockLists)) {
      for (const b of blocks) expect(sn.owner[b]).toBe(Number(id))
      expect(blocks.length * 16).toBeGreaterThanOrEqual(sn.tokens[Number(id)])
    }
  })
})

describe('percentile', () => {
  it('interpolates like numpy.percentile', () => {
    const v = [3, 1, 4, 1, 5, 9, 2, 6]
    expect(percentile(v, 50)).toBeCloseTo(3.5, 9)
    expect(percentile(v, 90)).toBeCloseTo(6.9, 9)
    expect(percentile(v, 100)).toBe(9)
  })
})

describe('capacity planning (the worked example in the lesson)', () => {
  const plan = capacityPlan({
    params: 8e9, bytesPerWeight: 2, gpuBytes: 80e9, reserveFraction: 0.1,
    layers: 32, kvHeads: 8, headDim: 128, kvBytesPerNumber: 2,
    promptTokens: 1000, outputTokens: 300,
    bandwidthBytesPerS: 2e12, usableFlopsPerS: 150e12, overheadMs: 3, gpuHourUsd: 2,
  })
  it('memory', () => {
    expect(plan.weightBytes).toBe(16e9)
    expect(plan.kvBytesPerToken).toBe(131072)
    expect(plan.kvBudgetBytes).toBeCloseTo(56e9, -3)
    expect(plan.maxConcurrent).toBe(328)
  })
  it('speed and cost', () => {
    expect(plan.singleStreamTokPerS).toBeCloseTo(125, 9)
    expect(plan.stepMemoryMs).toBeCloseTo(32.72, 2)
    expect(plan.stepComputeMs).toBeCloseTo(34.99, 2)
    expect(plan.stepMs).toBeCloseTo(37.99, 2)
    expect(plan.prefillMs).toBeCloseTo(109.67, 2)
    expect(plan.gpuMsPerRequest).toBeCloseTo(144.41, 2)
    expect(plan.requestsPerS).toBeCloseTo(6.92, 2)
    expect(plan.outputTokPerS).toBeCloseTo(2077.4, 1)
    expect(plan.usdPerMillionOutputTokens).toBeCloseTo(0.267, 3)
  })
  it('the two exercise variants: int4 weights, and no GQA', () => {
    const base = { params: 8e9, bytesPerWeight: 2, gpuBytes: 80e9, reserveFraction: 0.1, layers: 32, kvHeads: 8, headDim: 128, kvBytesPerNumber: 2, promptTokens: 1000, outputTokens: 300, bandwidthBytesPerS: 2e12, usableFlopsPerS: 150e12, overheadMs: 3, gpuHourUsd: 2 }
    expect(capacityPlan({ ...base, bytesPerWeight: 4.125 / 8 }).maxConcurrent).toBe(398)
    expect(capacityPlan({ ...base, kvHeads: 32 }).maxConcurrent).toBe(82)
  })
})
