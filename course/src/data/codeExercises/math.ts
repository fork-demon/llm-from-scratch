import type { CodeExerciseDef } from './types'

const exercises: CodeExerciseDef[] = [
  {
    id: 'vectors-code-dot',
    lesson: 'vectors',
    title: 'Write the dot product',
    prompt: `Write \`dot(a, b)\` for two Python lists of equal length: multiply the pairs, add up the results.

Use a plain loop or \`sum\` with \`zip\`. No NumPy yet: the point is to see that there is nothing hidden inside it.`,
    starter: `def dot(a, b):
    # multiply matching entries, then add everything up
    ...

print(dot([1, 2, 3], [4, 5, 6]))   # should print 32
`,
    solution: `def dot(a, b):
    total = 0
    for x, y in zip(a, b):
        total += x * y
    return total

print(dot([1, 2, 3], [4, 5, 6]))   # 32
`,
    tests: [
      { name: 'dot([1, 2, 3], [4, 5, 6]) is 32', code: `assert dot([1, 2, 3], [4, 5, 6]) == 32, f"got {dot([1, 2, 3], [4, 5, 6])}"` },
      { name: 'perpendicular vectors give 0', code: `assert dot([1, 0], [0, 1]) == 0, f"got {dot([1, 0], [0, 1])}"` },
      { name: 'opposite directions give a negative number', code: `assert dot([2, 1], [-2, -1]) == -5, f"got {dot([2, 1], [-2, -1])}"` },
      { name: 'works in 4 dimensions', code: `assert abs(dot([0.5, -1, 2, 0], [2, 1, 0.5, 9]) - 1.0) < 1e-9, f"got {dot([0.5, -1, 2, 0], [2, 1, 0.5, 9])}"` },
    ],
    hints: [
      '`zip(a, b)` gives you the pairs `(a[0], b[0])`, `(a[1], b[1])`, and so on.',
      'Start a running total at 0. For each pair `x, y`, add `x * y` to it. Return the total after the loop.',
      'In one line: `return sum(x * y for x, y in zip(a, b))`.',
    ],
    explanation: `Multiply matching entries and add them up: that is the whole operation. Large and positive means the two lists point the same way, zero means unrelated, negative means opposed.

Every attention score and every next-token score in an LLM is this loop, run millions of times on a GPU.`,
    source: 'phase1-foundations/math_primer.py',
  },
  {
    id: 'vectors-code-cosine',
    lesson: 'vectors',
    title: 'Cosine similarity from scratch',
    prompt: `Write \`cosine(a, b)\`: the dot product with both lengths divided out. A vector’s length is the square root of its dot product with itself.

If either vector is all zeros, return 0.0 instead of dividing by zero.`,
    starter: `import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def cosine(a, b):
    ...

print(cosine([1, 1], [5, 5]))   # same direction: 1.0
`,
    solution: `import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def cosine(a, b):
    length_a = math.sqrt(dot(a, a))
    length_b = math.sqrt(dot(b, b))
    if length_a == 0 or length_b == 0:
        return 0.0
    return dot(a, b) / (length_a * length_b)

print(cosine([1, 1], [5, 5]))   # 1.0
`,
    tests: [
      { name: 'same direction, different lengths: 1.0', code: `assert abs(cosine([1, 1], [5, 5]) - 1.0) < 1e-9, f"got {cosine([1, 1], [5, 5])}"` },
      { name: 'perpendicular: 0.0', code: `assert abs(cosine([1, 0], [0, 3])) < 1e-9, f"got {cosine([1, 0], [0, 3])}"` },
      { name: 'opposite: -1.0', code: `assert abs(cosine([2, 1], [-4, -2]) + 1.0) < 1e-9, f"got {cosine([2, 1], [-4, -2])}"` },
      { name: '45 degrees: about 0.707', code: `assert abs(cosine([1, 0], [1, 1]) - 0.70710678) < 1e-6, f"got {cosine([1, 0], [1, 1])}"` },
      { name: 'a zero vector returns 0.0 and does not crash', code: `assert cosine([0, 0], [1, 2]) == 0.0, f"got {cosine([0, 0], [1, 2])}"` },
    ],
    hints: [
      'Length of `a` is `math.sqrt(dot(a, a))`: Pythagoras in any number of dimensions.',
      'The formula is `dot(a, b) / (length_a * length_b)`.',
      'Handle the zero vector first: if either length is 0, return `0.0` before dividing.',
    ],
    explanation: `Dividing by both lengths removes size from the comparison, so only direction is left. A long document no longer wins a search just by being long.

Remember from the lesson: attention uses the raw dot product, not the cosine, so there length still counts.`,
    source: 'phase1-foundations/math_primer.py',
  },
  {
    id: 'softmax-code-softmax',
    lesson: 'softmax',
    title: 'Write a softmax that cannot overflow',
    prompt: `Write \`softmax(logits)\` for a list of numbers: exponentiate each one, then divide by the total.

Make it numerically stable: subtract the largest logit from every logit first. The answer does not change, because softmax only cares about differences, but \`exp(1000)\` no longer overflows.`,
    starter: `import math

def softmax(logits):
    ...

print(softmax([4.2, 2.1, -0.7]))   # about [0.885, 0.108, 0.007]
`,
    solution: `import math

def softmax(logits):
    biggest = max(logits)
    exps = [math.exp(z - biggest) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

print(softmax([4.2, 2.1, -0.7]))
`,
    tests: [
      { name: 'the probabilities add up to 1', code: `p = softmax([4.2, 2.1, -0.7]); assert abs(sum(p) - 1) < 1e-9, f"sum is {sum(p)}"` },
      { name: 'matches the lesson: about 0.885, 0.108, 0.007', code: `p = softmax([4.2, 2.1, -0.7]); assert abs(p[0] - 0.8853) < 1e-3 and abs(p[1] - 0.1084) < 1e-3 and abs(p[2] - 0.0066) < 1e-3, f"got {p}"` },
      { name: 'adding 100 to every logit changes nothing', code: `a = softmax([1, 2, 3]); b = softmax([101, 102, 103]); assert all(abs(x - y) < 1e-9 for x, y in zip(a, b)), f"{a} vs {b}"` },
      { name: 'huge logits do not overflow', code: `p = softmax([1000, 999]); assert abs(p[0] - 0.7310586) < 1e-6, f"got {p}"` },
      { name: 'equal logits give equal probabilities', code: `p = softmax([3, 3, 3, 3]); assert all(abs(x - 0.25) < 1e-9 for x in p), f"got {p}"` },
    ],
    hints: [
      'Three steps: `math.exp` of each logit, the sum of those, then each one divided by the sum.',
      'If the "huge logits" test fails with OverflowError, subtract `max(logits)` from every logit before calling `math.exp`.',
      '`exps = [math.exp(z - max(logits)) for z in logits]`, then `return [e / sum(exps) for e in exps]`.',
    ],
    explanation: `Exponentiating makes everything positive and turns gaps between logits into ratios. Dividing by the total makes the numbers add up to 1.

Subtracting the maximum is free, because shifting all logits by the same amount cancels out in the division. Every real implementation does it, including the \`softmax\` in the repo.`,
    source: 'phase1-foundations/mlp_numpy.py',
  },
]

export default exercises
