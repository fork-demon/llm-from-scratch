// A one-input network with a few hidden ReLU units ("hinges"):
//   hidden_i = relu(w_i * x + b_i)        output = sum_i v_i * hidden_i + c
// Same layer maths as MLP.forward in phase1-foundations/mlp_numpy.py, with 1 input and 1 output.
// Pure functions, no DOM.

export interface HingeUnit {
  w: number // input weight
  b: number // bias: slides the bend left or right
  v: number // output weight: how much (and in which direction) this hinge counts
}

export interface HingeNet { units: HingeUnit[]; c: number }

export const relu = (z: number): number => Math.max(0, z)
export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z))

/** What one unit contributes to the output at x. With `activation` off, the hinge never bends. */
export const unitOutput = (u: HingeUnit, x: number, activation = true): number => {
  const z = u.w * x + u.b
  return u.v * (activation ? relu(z) : z)
}

/** Forward pass for one input. */
export const netOutput = (net: HingeNet, x: number, activation = true): number =>
  net.units.reduce((s, u) => s + unitOutput(u, x, activation), net.c)

/** Where a unit's hinge bends: w*x + b = 0. null if w = 0 (no bend anywhere). */
export const bendAt = (u: HingeUnit): number | null => (u.w === 0 ? null : -u.b / u.w)

/**
 * Without the activation the whole network is one straight line, slope*x + intercept,
 * however many units it has: sum v_i (w_i x + b_i) + c = (sum v_i w_i) x + (sum v_i b_i + c).
 */
export const collapsedLine = (net: HingeNet): { slope: number; intercept: number } => ({
  slope: net.units.reduce((s, u) => s + u.v * u.w, 0),
  intercept: net.units.reduce((s, u) => s + u.v * u.b, net.c),
})

export type TargetId = 'tent' | 'abs' | 'parabola'
export const TARGETS: Record<TargetId, { label: string; f: (x: number) => number }> = {
  tent: { label: 'a bump', f: (x) => Math.max(0, 1 - Math.abs(x)) },
  abs: { label: 'a V shape, |x|', f: (x) => Math.abs(x) },
  parabola: { label: 'a smooth bowl, x²', f: (x) => x * x },
}

export const X_MIN = -2
export const X_MAX = 2
export const grid = (n = 81): number[] => Array.from({ length: n }, (_, i) => X_MIN + ((X_MAX - X_MIN) * i) / (n - 1))

/** Mean squared error between the network and the target curve over the visible range. */
export const curveError = (net: HingeNet, target: (x: number) => number, activation = true, xs: number[] = grid()): number =>
  xs.reduce((s, x) => s + (netOutput(net, x, activation) - target(x)) ** 2, 0) / xs.length

/** One hand-made solution per target (revealed only on request). */
export const SOLUTIONS: Record<TargetId, HingeNet> = {
  // relu(x+1) - 2 relu(x) + relu(x-1): up, then down twice as fast, then flat again
  tent: { c: 0, units: [{ w: 1, b: 1, v: 1 }, { w: 1, b: 0, v: -2 }, { w: 1, b: -1, v: 1 }, { w: 1, b: 0, v: 0 }] },
  // relu(x) + relu(-x)
  abs: { c: 0, units: [{ w: 1, b: 0, v: 1 }, { w: -1, b: 0, v: 1 }, { w: 1, b: 0, v: 0 }, { w: 1, b: 0, v: 0 }] },
  // straight pieces through (0,0), (±1,1), (±2,4)
  parabola: { c: 0, units: [{ w: 1, b: 0, v: 1 }, { w: -1, b: 0, v: 1 }, { w: 1, b: -1, v: 2 }, { w: -1, b: -1, v: 2 }] },
}

export const START_NET: HingeNet = {
  c: 0,
  units: [{ w: 1, b: 1.5, v: 0.5 }, { w: 1, b: 0.5, v: -0.5 }, { w: 1, b: -0.5, v: 0.5 }, { w: -1, b: -1, v: 0.5 }],
}
