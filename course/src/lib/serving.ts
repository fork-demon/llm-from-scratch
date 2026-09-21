// A TypeScript port of phase6-engineering/batching_sim.py: a simulator of the SCHEDULER around
// an LLM (no neural network runs here). Same cost model, same seeded workload, same scheduling
// rules, so the labs show exactly the numbers the Python file prints. serving.test.ts checks that.
import { makeRng } from './rng'

/* ------------------------------------------------------------------ */
/* 1. The cost model: the only physics in this file                     */
/* ------------------------------------------------------------------ */

/** An illustrative setup, not a benchmark: a 7B model in 16-bit (14 GB of weights, 0.5 MiB of KV
 *  cache per token) on an accelerator with about 2 TB/s of memory bandwidth and 150 TFLOP/s usable. */
export interface CostModel {
  weightReadMs: number // 14 GB / 2 TB/s: every decode step reads all the weights once
  kvReadMsPerTok: number // 0.5 MiB / 2 TB/s: each sequence also reads its own cache
  computeMsPerTok: number // 2 x 7e9 FLOP / 150 TFLOP/s: arithmetic for one token
  overheadMs: number // scheduling, kernel launches, sampling
}
export const COST: CostModel = { weightReadMs: 7.0, kvReadMsPerTok: 0.00026, computeMsPerTok: 0.09, overheadMs: 3.0 }

/** One decode step: every sequence gets one token. The step lasts as long as the SLOWER of
 *  "move the bytes" and "do the arithmetic" (the roofline idea), plus a fixed overhead. */
export const decodeStepMs = (nSeqs: number, ctxTokens: number, cost: CostModel = COST): number => {
  const memory = cost.weightReadMs + cost.kvReadMsPerTok * ctxTokens
  const compute = cost.computeMsPerTok * nSeqs
  return cost.overheadMs + Math.max(memory, compute)
}

/** Which side of the roofline a decode step is on. */
export const decodeBottleneck = (nSeqs: number, ctxTokens: number, cost: CostModel = COST): 'memory' | 'compute' =>
  cost.weightReadMs + cost.kvReadMsPerTok * ctxTokens >= cost.computeMsPerTok * nSeqs ? 'memory' : 'compute'

/** Prefill: all prompt tokens in parallel, so arithmetic dominates (but never less than one weight read). */
export const prefillMs = (promptTokens: number, cost: CostModel = COST): number =>
  cost.overheadMs + Math.max(cost.weightReadMs, cost.computeMsPerTok * promptTokens)

/* ------------------------------------------------------------------ */
/* 2. The workload                                                      */
/* ------------------------------------------------------------------ */
export const MAX_PROMPT = 1024
export const MAX_NEW = 512 // the max_tokens cap every request carries

export interface Request { id: number; arrival: number; prompt: number; out: number }
export interface WorkloadOptions { n?: number; ratePerS?: number; seed?: number; promptMedian?: number; promptSigma?: number; outMedian?: number; outSigma?: number }

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x))

/** Poisson arrivals (exponential gaps) and log-normal lengths. outSigma is the skew knob. */
export const makeWorkload = (o: WorkloadOptions = {}): Request[] => {
  const { n = 200, ratePerS = 8, seed = 17, promptMedian = 200, promptSigma = 0.5, outMedian = 60, outSigma = 1.0 } = o
  const rng = makeRng(seed)
  let t = 0
  const reqs: Request[] = []
  for (let i = 0; i < n; i++) {
    t += (-Math.log(1 - rng.next()) / ratePerS) * 1000
    const prompt = Math.floor(promptMedian * Math.exp(promptSigma * rng.normal()) + 0.5)
    const out = Math.floor(outMedian * Math.exp(outSigma * rng.normal()) + 0.5)
    reqs.push({ id: i, arrival: t, prompt: clamp(prompt, 8, MAX_PROMPT), out: clamp(out, 1, MAX_NEW) })
  }
  return reqs
}

/* ------------------------------------------------------------------ */
/* 3. The simulator                                                     */
/* ------------------------------------------------------------------ */
export type Policy = 'none' | 'static' | 'continuous'
export type KvMode = 'paged' | 'reserved'
export interface SimConfig {
  policy: Policy
  maxBatch: number // GPU slots: sequences decoded together
  kvBudgetTokens: number // KV memory, counted in tokens
  kvMode: KvMode
  blockSize: number // tokens per KV block
  staticMaxWaitMs: number // static batching gives up waiting for a full batch after this
  sloTtftMs: number // an EXAMPLE service-level objective
  sloTpotMs: number
}
export const defaultConfig = (o: Partial<SimConfig> = {}): SimConfig => ({
  policy: 'continuous', maxBatch: 16, kvBudgetTokens: 131072, kvMode: 'paged', blockSize: 16,
  staticMaxWaitMs: 2000, sloTtftMs: 1000, sloTpotMs: 50, ...o,
})

export type SegKind = 'prefill' | 'decode' | 'pad' | 'stall'
export interface Segment { slot: number; req: number; kind: SegKind; t0: number; t1: number }

export interface SimResult {
  policy: Policy
  maxBatch: number
  tokens: number
  spanS: number
  throughputTokS: number
  ttftP50: number; ttftP99: number
  tpotP50: number; tpotP99: number
  e2eP50: number; e2eP99: number
  slotUtil: number
  paddedSteps: number; usefulSteps: number; decodeSteps: number; prefillIters: number
  preemptions: number
  sloFraction: number; goodputReqS: number
  meanRunning: number; peakRunning: number
  kvWaste: number
  timeline: Segment[]
}

interface Live extends Request { generated: number; first: number; finish: number; blocks: number; slot: number }

const blocksFor = (tokens: number, bs: number) => Math.floor((tokens + bs - 1) / bs)

/** Linear interpolation between closest ranks (NumPy's default). */
export const percentile = (values: number[], p: number): number => {
  if (!values.length) return 0
  const v = values.slice().sort((a, b) => a - b)
  const k = ((v.length - 1) * p) / 100
  const lo = Math.floor(k)
  const hi = Math.min(lo + 1, v.length - 1)
  return v[lo] + (v[hi] - v[lo]) * (k - lo)
}

export const simulate = (workload: Request[], cfg: SimConfig, cost: CostModel = COST): SimResult => {
  const reqs: Live[] = workload.map((r) => ({ ...r, generated: 0, first: -1, finish: -1, blocks: 0, slot: -1 }))
  const bs = cfg.blockSize
  const totalBlocks = Math.floor(cfg.kvBudgetTokens / bs)
  if (totalBlocks * bs < MAX_PROMPT + MAX_NEW) throw new Error('KV budget smaller than one worst-case request')
  let maxBatch = cfg.policy === 'none' ? 1 : cfg.maxBatch
  // A static batch is one rectangular tensor: every slot is sized for the worst case.
  if (cfg.policy !== 'continuous') maxBatch = Math.max(1, Math.min(maxBatch, Math.floor((totalBlocks * bs) / (MAX_PROMPT + MAX_NEW))))

  const segs: Segment[] = []
  const lastSeg = new Map<number, Segment>()
  const mark = (slot: number, req: number, kind: SegKind, t0: number, t1: number) => {
    const prev = lastSeg.get(slot)
    if (prev && prev.req === req && prev.kind === kind && Math.abs(prev.t1 - t0) < 1e-9) { prev.t1 = t1; return }
    const seg = { slot, req, kind, t0, t1 }
    segs.push(seg)
    lastSeg.set(slot, seg)
  }

  let now = 0
  let busyMs = 0, usefulSlotMs = 0, decodeSteps = 0, prefillIters = 0, paddedSteps = 0, usefulSteps = 0, preemptions = 0
  let free = totalBlocks
  let kvStoredMs = 0, kvHeldMs = 0, runningMs = 0, peakRunning = 0
  let next = 0 // index of the first request that has not arrived yet
  const waiting: Live[] = []
  let running: Live[] = []
  let done = 0

  const arrive = () => { while (next < reqs.length && reqs[next].arrival <= now) waiting.push(reqs[next++]) }
  const account = (dur: number, usefulSlots: number) => {
    busyMs += dur
    usefulSlotMs += dur * usefulSlots
    let stored = 0
    for (const r of running) stored += r.prompt + r.generated
    kvStoredMs += dur * stored
    kvHeldMs += dur * (totalBlocks - free) * bs
    runningMs += dur * running.length
    peakRunning = Math.max(peakRunning, running.length)
  }

  if (cfg.policy !== 'continuous') {
    while (done < reqs.length) {
      arrive()
      const pending = next < reqs.length
      const ready = waiting.length > 0 && (waiting.length >= maxBatch || !pending || now - waiting[0].arrival >= cfg.staticMaxWaitMs)
      if (!ready) {
        const nxt: number[] = []
        if (pending) nxt.push(reqs[next].arrival)
        if (waiting.length) nxt.push(waiting[0].arrival + cfg.staticMaxWaitMs)
        now = Math.max(now, Math.min(...nxt))
        continue
      }
      const batch = waiting.splice(0, Math.min(maxBatch, waiting.length))
      const n = batch.length
      const padPrompt = Math.max(...batch.map((r) => r.prompt)) // prompts are padded to the longest
      const steps = Math.max(...batch.map((r) => r.out)) // and everyone waits for the longest answer
      running = batch
      free = totalBlocks - n * blocksFor(MAX_PROMPT + MAX_NEW, bs)
      let dur = prefillMs(n * padPrompt, cost)
      let t0 = now
      now += dur
      prefillIters++
      batch.forEach((r, i) => {
        r.slot = i; r.generated = 1; r.first = now
        mark(i, r.id, 'prefill', t0, now)
        if (r.out === 1) r.finish = now
      })
      account(dur, n)
      usefulSteps += n
      for (let k = 2; k <= steps; k++) {
        dur = decodeStepMs(n, n * (padPrompt + k - 1), cost)
        t0 = now
        now += dur
        decodeSteps++
        let live = 0
        batch.forEach((r, i) => {
          if (r.generated < r.out) {
            r.generated++
            live++
            mark(i, r.id, 'decode', t0, now)
            if (r.generated === r.out) r.finish = now
          } else mark(i, r.id, 'pad', t0, now) // finished, still occupying the slot
        })
        paddedSteps += n - live
        usefulSteps += live
        account(dur, live)
      }
      done += n
      running = []
      free = totalBlocks
    }
  } else {
    const slots: (number | null)[] = Array(maxBatch).fill(null)
    while (done < reqs.length) {
      arrive()
      if (!running.length && !waiting.length) { now = reqs[next].arrival; continue }
      // admission: first come, first served, while a slot AND KV memory are free
      const admitted: Live[] = []
      while (waiting.length && running.length + admitted.length < maxBatch) {
        const r = waiting[0]
        let need: number, headroom: number
        if (cfg.kvMode === 'reserved') { need = blocksFor(r.prompt + MAX_NEW, bs); headroom = 0 } // worst case, up front
        else { need = blocksFor(r.prompt + r.generated + 1, bs); headroom = running.length + admitted.length } // one spare block per running sequence
        if (free - need < headroom) break
        free -= need
        r.blocks = need
        r.slot = slots.indexOf(null)
        slots[r.slot] = r.id
        admitted.push(waiting.shift()!)
      }
      if (admitted.length) {
        // a prefill iteration: running sequences STALL while it happens
        let tokens = 0
        for (const r of admitted) tokens += r.prompt + r.generated // (+generated: recompute after preemption)
        const dur = prefillMs(tokens, cost)
        const t0 = now
        now += dur
        prefillIters++
        for (const r of running) mark(r.slot, r.id, 'stall', t0, now)
        for (const r of admitted) {
          mark(r.slot, r.id, 'prefill', t0, now)
          r.generated++
          if (r.first < 0) r.first = now
        }
        running.push(...admitted)
        usefulSteps += admitted.length
        account(dur, admitted.length)
      } else {
        if (cfg.kvMode === 'paged') {
          let i = 0
          while (i < running.length) {
            const r = running[i]
            if (blocksFor(r.prompt + r.generated + 1, bs) > r.blocks) {
              if (free === 0) {
                // out of blocks: preempt the most recently admitted sequence; it is re-prefilled later
                const victim = running.pop()!
                free += victim.blocks
                victim.blocks = 0
                slots[victim.slot] = null
                preemptions++
                waiting.unshift(victim)
                if (victim === r) break
                continue
              }
              free--
              r.blocks++
            }
            i++
          }
        }
        if (!running.length) continue
        let ctx = 0
        for (const r of running) ctx += r.prompt + r.generated
        const dur = decodeStepMs(running.length, ctx, cost)
        const t0 = now
        now += dur
        decodeSteps++
        usefulSteps += running.length
        for (const r of running) { r.generated++; mark(r.slot, r.id, 'decode', t0, now) }
        account(dur, running.length)
      }
      // finished sequences leave immediately: their slot and blocks are free next iteration
      for (const r of running.filter((x) => x.generated >= x.out)) {
        r.finish = now
        free += r.blocks
        r.blocks = 0
        slots[r.slot] = null
        done++
      }
      running = running.filter((x) => x.generated < x.out)
    }
  }

  const ttft = reqs.map((r) => r.first - r.arrival)
  const e2e = reqs.map((r) => r.finish - r.arrival)
  const tpotOf = (r: Live) => (r.finish - r.first) / (r.out - 1)
  const tpot = reqs.filter((r) => r.out > 1).map(tpotOf)
  const tokens = reqs.reduce((s, r) => s + r.out, 0)
  const spanS = (Math.max(...reqs.map((r) => r.finish)) - reqs[0].arrival) / 1000
  const ok = reqs.filter((r) => r.first - r.arrival <= cfg.sloTtftMs && (r.out === 1 || tpotOf(r) <= cfg.sloTpotMs)).length
  return {
    policy: cfg.policy, maxBatch, tokens, spanS, throughputTokS: tokens / spanS,
    ttftP50: percentile(ttft, 50), ttftP99: percentile(ttft, 99),
    tpotP50: percentile(tpot, 50), tpotP99: percentile(tpot, 99),
    e2eP50: percentile(e2e, 50), e2eP99: percentile(e2e, 99),
    slotUtil: usefulSlotMs / (maxBatch * busyMs),
    paddedSteps, usefulSteps, decodeSteps, prefillIters, preemptions,
    sloFraction: ok / reqs.length, goodputReqS: ok / spanS,
    meanRunning: runningMs / busyMs, peakRunning,
    kvWaste: kvHeldMs ? 1 - kvStoredMs / kvHeldMs : 0,
    timeline: segs,
  }
}

/* ------------------------------------------------------------------ */
/* 4. KV memory, block by block                                         */
/* ------------------------------------------------------------------ */
export type GridMode = 'contiguous' | 'paged'
export interface GridRequest { id: number; prompt: number; out: number }
export interface GridSnapshot {
  step: number
  owner: number[] // owner[b] = id of the sequence holding block b, or -1
  running: number[]
  tokens: Record<number, number> // tokens really stored, per running sequence
  blockLists: Record<number, number[]> // the block table: logical order -> physical block
  stored: number
  held: number
  waiting: number
}
export interface GridResult { snaps: GridSnapshot[]; steps: number; meanRunning: number; peakRunning: number; waste: number; fragmentationRefusals: number; preemptions: number }

export const gridRequests = (n = 24, seed = 5, promptMedian = 40, outMedian = 30, outSigma = 0.9, maxNew = 128): GridRequest[] => {
  const rng = makeRng(seed)
  return Array.from({ length: n }, (_, id) => {
    const prompt = clamp(Math.floor(promptMedian * Math.exp(0.5 * rng.normal()) + 0.5), 4, 96)
    const out = clamp(Math.floor(outMedian * Math.exp(outSigma * rng.normal()) + 0.5), 1, maxNew)
    return { id, prompt, out }
  })
}

/** All requests wait at step 0. Each step: admit in order while memory allows, then every running
 *  sequence grows by one token; a finished sequence releases its memory.
 *  contiguous: reserve ceil((prompt + maxNew) / blockSize) blocks in ONE run, first fit.
 *  paged: hold only the blocks the tokens so far need, take ANY free block when the last one fills. */
export const simulateKvGrid = (requests: GridRequest[], o: { nBlocks?: number; blockSize?: number; mode?: GridMode; maxNew?: number; maxSteps?: number } = {}): GridResult => {
  const { nBlocks = 64, blockSize = 16, mode = 'paged', maxNew = 128, maxSteps = 5000 } = o
  const owner: number[] = Array(nBlocks).fill(-1)
  interface Seq extends GridRequest { gen: number; blocks: number[] }
  const waiting: Seq[] = requests.map((r) => ({ ...r, gen: 0, blocks: [] }))
  let running: Seq[] = []
  const snaps: GridSnapshot[] = []
  let fragmentationRefusals = 0
  let preemptions = 0
  const freeCount = () => owner.reduce((s, x) => s + (x < 0 ? 1 : 0), 0)
  const firstFit = (need: number) => {
    let run = 0
    for (let i = 0; i < nBlocks; i++) {
      run = owner[i] < 0 ? run + 1 : 0
      if (run === need) return i - need + 1
    }
    return -1
  }
  const release = (s: Seq) => { for (const b of s.blocks) owner[b] = -1; s.blocks = [] }

  let step = 0
  while ((waiting.length || running.length) && step < maxSteps) {
    while (waiting.length) {
      const s = waiting[0]
      if (mode === 'contiguous') {
        const need = blocksFor(s.prompt + maxNew, blockSize)
        const start = firstFit(need)
        if (start < 0) { if (freeCount() >= need) fragmentationRefusals++; break }
        s.blocks = Array.from({ length: need }, (_, i) => start + i)
      } else {
        const need = blocksFor(s.prompt + s.gen + 1, blockSize)
        if (freeCount() - need < running.length) break
        const freeIdx: number[] = []
        for (let i = 0; i < nBlocks && freeIdx.length < need; i++) if (owner[i] < 0) freeIdx.push(i)
        s.blocks = freeIdx
      }
      for (const b of s.blocks) owner[b] = s.id
      running.push(waiting.shift()!)
    }
    let i = 0
    while (i < running.length) {
      const s = running[i]
      if (mode === 'paged' && blocksFor(s.prompt + s.gen + 1, blockSize) > s.blocks.length) {
        if (freeCount() === 0) {
          const victim = running.pop()!
          release(victim)
          preemptions++
          waiting.unshift(victim)
          if (victim === s) break
          continue
        }
        const b = owner.indexOf(-1)
        owner[b] = s.id
        s.blocks.push(b)
      }
      s.gen++
      i++
    }
    const tokens: Record<number, number> = {}
    const blockLists: Record<number, number[]> = {}
    let stored = 0, held = 0
    for (const s of running) { tokens[s.id] = s.prompt + s.gen; blockLists[s.id] = s.blocks.slice(); stored += s.prompt + s.gen; held += s.blocks.length * blockSize }
    snaps.push({ step, owner: owner.slice(), running: running.map((s) => s.id), tokens, blockLists, stored, held, waiting: waiting.length })
    for (const s of running.filter((x) => x.gen >= x.out)) release(s)
    running = running.filter((x) => x.gen < x.out)
    step++
  }
  const busy = snaps.filter((s) => s.held > 0)
  return {
    snaps, steps: step,
    meanRunning: snaps.reduce((s, x) => s + x.running.length, 0) / snaps.length,
    peakRunning: Math.max(...snaps.map((x) => x.running.length)),
    waste: 1 - busy.reduce((s, x) => s + x.stored, 0) / busy.reduce((s, x) => s + x.held, 0),
    fragmentationRefusals, preemptions,
  }
}

/* ------------------------------------------------------------------ */
/* 5. Capacity planning by hand (the worked example in the lesson)      */
/* ------------------------------------------------------------------ */
export interface CapacityInput {
  params: number // number of weights
  bytesPerWeight: number // 2 for 16-bit, about 0.52 for int4 with group scales
  gpuBytes: number
  reserveFraction: number // memory kept back for activations and safety
  layers: number; kvHeads: number; headDim: number; kvBytesPerNumber: number
  promptTokens: number; outputTokens: number
  bandwidthBytesPerS: number
  usableFlopsPerS: number
  overheadMs: number
  gpuHourUsd: number // an EXAMPLE price
}
export interface CapacityPlan {
  weightBytes: number; kvBudgetBytes: number; kvBytesPerToken: number
  maxConcurrent: number
  stepMemoryMs: number; stepComputeMs: number; stepMs: number
  prefillMs: number; gpuMsPerRequest: number
  requestsPerS: number; outputTokPerS: number
  usdPerMillionOutputTokens: number
  singleStreamTokPerS: number
}

/** Three ceilings: KV memory (how many sequences fit), the decode step at that batch (memory
 *  traffic or arithmetic, whichever is slower), and the GPU time each request needs in total. */
export const capacityPlan = (c: CapacityInput): CapacityPlan => {
  const weightBytes = c.params * c.bytesPerWeight
  const kvBudgetBytes = c.gpuBytes * (1 - c.reserveFraction) - weightBytes
  const kvBytesPerToken = 2 * c.layers * c.kvHeads * c.headDim * c.kvBytesPerNumber
  const peakTokens = c.promptTokens + c.outputTokens
  const maxConcurrent = Math.max(0, Math.floor(kvBudgetBytes / (kvBytesPerToken * peakTokens)))
  const avgTokens = c.promptTokens + c.outputTokens / 2
  const flopsPerToken = 2 * c.params
  const stepMemoryMs = ((weightBytes + maxConcurrent * avgTokens * kvBytesPerToken) / c.bandwidthBytesPerS) * 1000
  const stepComputeMs = ((maxConcurrent * flopsPerToken) / c.usableFlopsPerS) * 1000
  const stepMs = c.overheadMs + Math.max(stepMemoryMs, stepComputeMs)
  const pre = c.overheadMs + Math.max((weightBytes / c.bandwidthBytesPerS) * 1000, ((c.promptTokens * flopsPerToken) / c.usableFlopsPerS) * 1000)
  const gpuMsPerRequest = maxConcurrent > 0 ? pre + (c.outputTokens * stepMs) / maxConcurrent : Infinity
  const requestsPerS = 1000 / gpuMsPerRequest
  const outputTokPerS = requestsPerS * c.outputTokens
  return {
    weightBytes, kvBudgetBytes, kvBytesPerToken, maxConcurrent, stepMemoryMs, stepComputeMs, stepMs,
    prefillMs: pre, gpuMsPerRequest, requestsPerS, outputTokPerS,
    usdPerMillionOutputTokens: c.gpuHourUsd / ((outputTokPerS * 3600) / 1e6),
    singleStreamTokPerS: c.bandwidthBytesPerS / weightBytes,
  }
}
