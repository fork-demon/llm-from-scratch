// Pure helpers for lesson 1.4: the nudge experiment and the chain rule pipeline.
// `nudgeDerivative` is the same three-step recipe as `nudge_derivative` in
// phase1-foundations/math_primer.py: nudge, re-measure, divide.

export type Fn = (x: number) => number

/** (f(x + h) - f(x)) / h */
export const nudgeDerivative = (f: Fn, x: number, h = 1e-6): number => (f(x + h) - f(x)) / h

export interface NudgeFunction {
  id: string
  label: string // how we write it in the UI
  f: Fn
  /** The exact slope from the school formula. NaN where it does not exist (the kink of max(0, x)). */
  exact: Fn
  rule: string // the shortcut formula, in words a developer can read
  yRange: [number, number] // plot window
}

export const NUDGE_FUNCTIONS: NudgeFunction[] = [
  { id: 'square', label: 'x²', f: (x) => x * x, exact: (x) => 2 * x, rule: '2x', yRange: [-2, 10] },
  { id: 'cube', label: 'x³', f: (x) => x ** 3, exact: (x) => 3 * x * x, rule: '3x²', yRange: [-12, 12] },
  { id: 'line', label: '3x + 1', f: (x) => 3 * x + 1, exact: () => 3, rule: '3 (the same everywhere)', yRange: [-10, 12] },
  { id: 'relu', label: 'max(0, x)', f: (x) => Math.max(0, x), exact: (x) => (x > 0 ? 1 : x < 0 ? 0 : NaN), rule: '1 if x > 0, 0 if x < 0', yRange: [-1, 4] },
]

export interface NudgeReport {
  fx: number
  fxh: number
  moved: number // f(x+h) - f(x)
  ratio: number // moved / h
  exact: number
}

export const nudgeReport = (fn: NudgeFunction, x: number, h: number): NudgeReport => {
  const fx = fn.f(x)
  const fxh = fn.f(x + h)
  return { fx, fxh, moved: fxh - fx, ratio: (fxh - fx) / h, exact: fn.exact(x) }
}

/** The ratio for a list of shrinking nudges: watch it settle. */
export const convergence = (f: Fn, x: number, hs: number[] = [1, 0.1, 0.01, 0.001]): { h: number; ratio: number }[] =>
  hs.map((h) => ({ h, ratio: nudgeDerivative(f, x, h) }))

/* ---------- chain rule: x -> square -> +1 -> x3 ---------- */
export interface Stage {
  name: string
  expr: string
  apply: Fn
  /** local amplification: derivative of this stage at its own input */
  local: Fn
  localRule: string
}

export const PIPELINE: Stage[] = [
  { name: 'square', expr: 'u = x²', apply: (x) => x * x, local: (x) => 2 * x, localRule: '2 × input' },
  { name: 'add 1', expr: 'v = u + 1', apply: (u) => u + 1, local: () => 1, localRule: 'always 1' },
  { name: 'times 3', expr: 'y = 3v', apply: (v) => 3 * v, local: () => 3, localRule: 'always 3' },
]

export const runPipeline = (x: number, stages: Stage[] = PIPELINE): number => stages.reduce((v, s) => s.apply(v), x)

export interface StageTrace { name: string; expr: string; localRule: string; input: number; output: number; local: number }

export interface ChainReport {
  stages: StageTrace[]
  output: number
  product: number // local amplifications multiplied together: the chain rule
  measured: number // nudge experiment on the whole pipeline
}

export const chainReport = (x: number, stages: Stage[] = PIPELINE, h = 1e-6): ChainReport => {
  const trace: StageTrace[] = []
  let v = x
  for (const s of stages) {
    const out = s.apply(v)
    trace.push({ name: s.name, expr: s.expr, localRule: s.localRule, input: v, output: out, local: s.local(v) })
    v = out
  }
  return {
    stages: trace,
    output: v,
    product: trace.reduce((p, s) => p * s.local, 1),
    measured: nudgeDerivative((t) => runPipeline(t, stages), x, h),
  }
}

/** Follow an actual nudge through the pipeline: how big is the wiggle after each stage? */
export const wiggleTrace = (x: number, h: number, stages: Stage[] = PIPELINE): number[] => {
  const out = [h]
  let a = x
  let b = x + h
  for (const s of stages) {
    a = s.apply(a)
    b = s.apply(b)
    out.push(b - a)
  }
  return out
}

/* ---------- plotting ---------- */
export const samplePoints = (f: Fn, x0: number, x1: number, n = 140): [number, number][] =>
  Array.from({ length: n + 1 }, (_, i) => {
    const x = x0 + ((x1 - x0) * i) / n
    return [x, f(x)] as [number, number]
  })
