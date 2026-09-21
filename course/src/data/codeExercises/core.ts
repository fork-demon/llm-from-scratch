import type { CodeExerciseDef } from './types'

// One exercise per idea the course says you should be able to implement.
// Each one is the real algorithm from the repo's Python, reduced to the part that teaches it.
const exercises: CodeExerciseDef[] = [
  /* ---------- Part 1: matrices ---------- */
  {
    id: 'matrices-code-matmul',
    lesson: 'matrices',
    title: 'Multiply two matrices',
    prompt: `Write \`matmul(A, B)\` for matrices held as lists of rows. Cell (i, j) of the result is row i of A dotted with column j of B.

Return the result as a list of rows. Do not use NumPy: this is the loop NumPy runs for you.`,
    starter: `def matmul(A, B):
    # result[i][j] = row i of A, dotted with column j of B
    ...

print(matmul([[2, 0], [1, 3]], [[1], [2]]))   # should print [[2], [7]]
`,
    solution: `def matmul(A, B):
    n = len(A)          # rows of A
    k = len(B)          # rows of B, which must equal the columns of A
    m = len(B[0])       # columns of B
    out = []
    for i in range(n):
        row = []
        for j in range(m):
            total = 0
            for t in range(k):
                total += A[i][t] * B[t][j]
            row.append(total)
        out.append(row)
    return out

print(matmul([[2, 0], [1, 3]], [[1], [2]]))
`,
    tests: [
      { name: 'the lesson example gives [[2], [7]]', code: `assert matmul([[2, 0], [1, 3]], [[1], [2]]) == [[2], [7]], f"got {matmul([[2, 0], [1, 3]], [[1], [2]])}"` },
      { name: 'the identity matrix changes nothing', code: `I = [[1, 0], [0, 1]]\nassert matmul([[5, 6], [7, 8]], I) == [[5, 6], [7, 8]], f"got {matmul([[5, 6], [7, 8]], I)}"` },
      { name: 'shapes follow the rule (2,3) @ (3,2) gives (2,2)', code: `r = matmul([[1, 2, 3], [4, 5, 6]], [[1, 0], [0, 1], [1, 1]])\nassert len(r) == 2 and len(r[0]) == 2, f"got shape {len(r)}x{len(r[0])}"\nassert r == [[4, 5], [10, 11]], f"got {r}"` },
      { name: 'order matters: A @ B is not B @ A', code: `A = [[1, 2], [3, 4]]; B = [[0, 1], [1, 0]]\nassert matmul(A, B) != matmul(B, A), "these two products should differ"` },
    ],
    hints: [
      'Three nested loops: one for the row of A, one for the column of B, one to walk along the shared dimension.',
      'The shared dimension is the number of columns of A, which is also the number of rows of B: `len(B)`.',
      'Innermost line: `total += A[i][t] * B[t][j]`, summed over `t`.',
    ],
    explanation: `A matrix multiply is a dot product for every (row, column) pair. That is why the inner dimensions must match: the dot product needs two lists of the same length.

Every layer of a Transformer is this operation. A GPU runs thousands of these dot products at the same time, which is the whole reason LLMs are trained on GPUs.`,
    source: 'phase1-foundations/math_primer.py',
  },

  /* ---------- Part 2: gradient descent ---------- */
  {
    id: 'gradient-descent-code-step',
    lesson: 'gradient-descent',
    title: 'One step of gradient descent',
    prompt: `The model is a line: \`pred = w * x + b\`. The loss is the mean squared error over all the points.

Write \`step(w, b, xs, ys, lr)\` that does one update and returns the new \`(w, b)\`.

The gradients, from the chain rule: for each point the error is \`pred - y\`; then
\`grad_w\` is the mean of \`2 * error * x\` and \`grad_b\` is the mean of \`2 * error\`.
Move each parameter against its gradient.`,
    starter: `def step(w, b, xs, ys, lr):
    # 1. predictions  2. errors  3. gradients  4. update
    ...

xs = [1.0, 2.0, 3.0]
ys = [3.0, 5.0, 7.0]        # the true rule is y = 2x + 1
w, b = 0.0, 0.0
for _ in range(200):
    w, b = step(w, b, xs, ys, 0.05)
print(round(w, 3), round(b, 3))   # should approach 2.0 and 1.0
`,
    solution: `def step(w, b, xs, ys, lr):
    n = len(xs)
    errors = [(w * x + b) - y for x, y in zip(xs, ys)]
    grad_w = sum(2 * e * x for e, x in zip(errors, xs)) / n
    grad_b = sum(2 * e for e in errors) / n
    return w - lr * grad_w, b - lr * grad_b

xs = [1.0, 2.0, 3.0]
ys = [3.0, 5.0, 7.0]
w, b = 0.0, 0.0
for _ in range(200):
    w, b = step(w, b, xs, ys, 0.05)
print(round(w, 3), round(b, 3))
`,
    tests: [
      { name: 'returns two numbers', code: `r = step(0.0, 0.0, [1.0, 2.0], [2.0, 4.0], 0.1)\nassert len(r) == 2, f"expected (w, b), got {r}"` },
      { name: 'one step from zero moves w up for y = 2x', code: `w, b = step(0.0, 0.0, [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 0.01)\nassert w > 0, f"w should increase, got {w}"` },
      { name: 'the first step matches the hand calculation', code: `w, b = step(0.0, 0.0, [1.0, 2.0, 3.0], [3.0, 5.0, 7.0], 0.05)\nassert abs(w - 1.1333333333) < 1e-6 and abs(b - 0.5) < 1e-9, f"expected about (1.133, 0.5), got ({w}, {b})"` },
      { name: '200 steps find the rule y = 2x + 1', code: `w, b = 0.0, 0.0\nfor _ in range(200):\n    w, b = step(w, b, [1.0, 2.0, 3.0], [3.0, 5.0, 7.0], 0.05)\nassert abs(w - 2) < 0.05 and abs(b - 1) < 0.1, f"got w={w}, b={b}"` },
      { name: 'a perfect fit does not move', code: `w, b = step(2.0, 1.0, [1.0, 2.0, 3.0], [3.0, 5.0, 7.0], 0.5)\nassert abs(w - 2) < 1e-9 and abs(b - 1) < 1e-9, f"got ({w}, {b})"` },
    ],
    hints: [
      'First build the list of errors: `(w * x + b) - y` for every point.',
      'grad_w is the mean of `2 * error * x`. grad_b is the mean of `2 * error`. Mean means divide by the number of points.',
      'The update is `w - lr * grad_w` and `b - lr * grad_b`. Subtract, because the gradient points uphill.',
    ],
    explanation: `This is the loop that trains every neural network, including GPT. Only the model between the prediction and the loss gets more complicated.

Notice the last test: when the fit is already perfect the errors are zero, so the gradients are zero and nothing moves. A model at the bottom of the valley stays there.`,
    source: 'phase1-foundations/gradient_descent.py',
  },

  /* ---------- Part 3: neurons ---------- */
  {
    id: 'neurons-code-relu-layer',
    lesson: 'neurons',
    title: 'A layer: weighted sums, then a bend',
    prompt: `Write \`layer(x, W, b)\`: multiply the input vector by the weight matrix, add the bias, then apply ReLU to every entry.

\`W\` has one row per input and one column per neuron, so \`x @ W\` is the course's row-vector convention. ReLU is \`max(0, value)\`.`,
    starter: `def relu(v):
    ...

def layer(x, W, b):
    # for each neuron j: sum over i of x[i] * W[i][j], plus b[j], then relu
    ...

print(layer([1.0, 2.0], [[1.0, -1.0], [0.5, 0.25]], [0.0, 0.0]))   # [2.0, 0.0]
`,
    solution: `def relu(v):
    return v if v > 0 else 0.0

def layer(x, W, b):
    n_out = len(b)
    out = []
    for j in range(n_out):
        total = b[j]
        for i in range(len(x)):
            total += x[i] * W[i][j]
        out.append(relu(total))
    return out

print(layer([1.0, 2.0], [[1.0, -1.0], [0.5, 0.25]], [0.0, 0.0]))
`,
    tests: [
      { name: 'the lesson example gives [2.0, 0.0]', code: `r = layer([1.0, 2.0], [[1.0, -1.0], [0.5, 0.25]], [0.0, 0.0])\nassert abs(r[0] - 2.0) < 1e-9 and abs(r[1]) < 1e-9, f"got {r}"` },
      { name: 'negatives are clipped to zero', code: `r = layer([1.0], [[-5.0]], [0.0])\nassert r == [0.0] or abs(r[0]) < 1e-12, f"got {r}"` },
      { name: 'the bias shifts before the bend', code: `r = layer([1.0], [[-5.0]], [6.0])\nassert abs(r[0] - 1.0) < 1e-9, f"got {r}"` },
      { name: 'the output has one entry per neuron', code: `r = layer([1.0, 2.0, 3.0], [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], [0.0, 0.0, 0.0, 0.0])\nassert len(r) == 4, f"expected 4 outputs, got {len(r)}"\nassert r == [1.0, 2.0, 3.0, 6.0], f"got {r}"` },
    ],
    hints: [
      'One neuron at a time: for neuron `j`, start the total at `b[j]`, then add `x[i] * W[i][j]` for every input `i`.',
      'Apply `relu` to the total after the bias, not before.',
      'The number of neurons is `len(b)`, which is also the number of columns of W.',
    ],
    explanation: `A layer is a weighted sum per neuron, then a bend. Without the bend, stacking layers would collapse into a single straight-line function, however many you stack.

The feed-forward network inside every Transformer block is exactly this, twice: wide, bend, back down.`,
    source: 'phase1-foundations/mlp_numpy.py',
  },

  /* ---------- Part 4: tokenization ---------- */
  {
    id: 'tokenization-code-merge',
    lesson: 'tokenization',
    title: 'One BPE merge',
    prompt: `Byte pair encoding repeats one move: find the most frequent adjacent pair, then replace every occurrence of it with a new symbol.

Write \`most_common_pair(ids)\` returning the most frequent adjacent pair as a tuple, and \`merge(ids, pair, new_id)\` returning a new list with every occurrence of that pair replaced.

On a tie, return the pair that appears first.`,
    starter: `def most_common_pair(ids):
    ...

def merge(ids, pair, new_id):
    ...

ids = [1, 2, 1, 2, 3]
p = most_common_pair(ids)
print(p, merge(ids, p, 99))   # (1, 2) [99, 99, 3]
`,
    solution: `def most_common_pair(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    best, best_count = None, 0
    for pair in zip(ids, ids[1:]):          # walk in order, so ties keep the first pair
        if counts[pair] > best_count:
            best, best_count = pair, counts[pair]
    return best

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

ids = [1, 2, 1, 2, 3]
p = most_common_pair(ids)
print(p, merge(ids, p, 99))
`,
    tests: [
      { name: 'finds the most frequent pair', code: `assert most_common_pair([1, 2, 1, 2, 3]) == (1, 2), f"got {most_common_pair([1, 2, 1, 2, 3])}"` },
      { name: 'merge replaces every occurrence', code: `assert merge([1, 2, 1, 2, 3], (1, 2), 99) == [99, 99, 3], f"got {merge([1, 2, 1, 2, 3], (1, 2), 99)}"` },
      { name: 'merging shortens the sequence', code: `ids = [5, 6, 5, 6, 5, 6]\nassert len(merge(ids, (5, 6), 7)) == 3, f"got {merge(ids, (5, 6), 7)}"` },
      { name: 'overlapping pairs are not merged twice', code: `assert merge([1, 1, 1], (1, 1), 9) == [9, 1], f"got {merge([1, 1, 1], (1, 1), 9)}"` },
      { name: 'a pair that is not there leaves the list unchanged', code: `assert merge([1, 2, 3], (7, 8), 9) == [1, 2, 3], f"got {merge([1, 2, 3], (7, 8), 9)}"` },
    ],
    hints: [
      '`zip(ids, ids[1:])` gives every adjacent pair. Count them in a dictionary.',
      'For merge, walk with a `while` loop and an index. When you match the pair, append the new id and jump the index forward by 2.',
      'Jumping by 2 is what stops `[1, 1, 1]` from merging the middle 1 twice.',
    ],
    explanation: `Run these two functions a few thousand times over a large corpus and you have trained a real tokenizer. The ordered list of merges IS the tokenizer.

The overlap test matters: after merging a pair, the merged symbol cannot also be part of the next pair in the same pass. That is why encoding replays the merges in the order they were learned.`,
    source: 'phase2-language/bpe_tokenizer.py',
  },

  /* ---------- Part 6: attention ---------- */
  {
    id: 'attention-code-attention',
    lesson: 'attention',
    title: 'Single-head attention, end to end',
    prompt: `Write \`attention(Q, K, V, causal)\` with NumPy. Q, K and V are arrays of shape (T, d): one row per token.

The steps are the ones from the lesson: scores are \`Q @ K.T\` divided by the square root of d; if \`causal\` is true, every position above the diagonal becomes \`-inf\`; softmax each row; then multiply by V.

Return \`(output, weights)\`.`,
    starter: `import numpy as np

def attention(Q, K, V, causal=False):
    # scores -> scale -> mask -> softmax -> blend
    ...

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
out, w = attention(Q, Q, Q, causal=True)
print(w)   # [[1, 0], [~0.38, ~0.62]]
`,
    solution: `import numpy as np

def attention(Q, K, V, causal=False):
    d = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    if causal:
        T = scores.shape[0]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    weights = e / e.sum(axis=-1, keepdims=True)
    return weights @ V, weights

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
out, w = attention(Q, Q, Q, causal=True)
print(w)
`,
    tests: [
      { name: 'every row of the weights sums to 1', code: `import numpy as np\nQ = np.random.default_rng(0).normal(size=(4, 8))\n_, w = attention(Q, Q, Q)\nassert np.allclose(w.sum(axis=-1), 1), f"row sums are {w.sum(axis=-1)}"` },
      { name: 'the causal mask blocks the future exactly', code: `import numpy as np\nQ = np.random.default_rng(1).normal(size=(5, 4))\n_, w = attention(Q, Q, Q, causal=True)\nassert np.allclose(np.triu(w, k=1), 0), "weights above the diagonal must be 0"\nassert abs(w[0, 0] - 1) < 1e-9, "the first token can only see itself"` },
      { name: 'the scores are scaled by sqrt(d)', code: `import numpy as np\nQ = np.array([[2.0, 2.0, 0.0, 0.0]])\nK = np.array([[2.0, 2.0, 0.0, 0.0]])\n_, w = attention(Q, K, K)\nassert w.shape == (1, 1) and abs(w[0, 0] - 1) < 1e-9, f"got {w}"\nQ2 = np.array([[1.0, 0.0], [0.0, 1.0]])\n_, w2 = attention(Q2, Q2, Q2)\nassert abs(w2[0, 0] - 0.6706) < 1e-3, f"without the sqrt(d) scaling this is wrong: got {w2[0, 0]}"` },
      { name: 'the output is a blend, so it stays within the range of V', code: `import numpy as np\nQ = np.random.default_rng(2).normal(size=(4, 3))\nV = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])\nout, _ = attention(Q, Q, V)\nassert out.min() >= V.min() - 1e-9 and out.max() <= V.max() + 1e-9, f"output range {out.min()}..{out.max()} is outside V"` },
      { name: 'it matches a hand-worked example', code: `import numpy as np\nQ = np.array([[1.0, 0.0]])\nK = np.array([[1.0, 0.0], [0.0, 1.0]])\nV = np.array([[1.0, 0.0], [0.0, 1.0]])\nout, w = attention(Q, K, V)\nimport math\nexpected = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)\nassert abs(w[0, 0] - expected) < 1e-6, f"expected {expected}, got {w[0, 0]}"` },
    ],
    hints: [
      '`Q @ K.T` gives the T by T table of scores. `K.shape[-1]` is d.',
      'For the mask, `np.triu(np.ones((T, T), dtype=bool), k=1)` is True strictly above the diagonal. Use `np.where(mask, -np.inf, scores)`.',
      'For a stable softmax, subtract the row maximum before `np.exp`, then divide by the row sum. Use `keepdims=True` so the shapes line up.',
    ],
    explanation: `This is the whole mechanism. Four lines: score, scale, softmax, blend.

The masked entries become \`-inf\`, and \`exp(-inf)\` is exactly 0, so a masked token receives no weight at all while the remaining weights still sum to 1.

The function you just wrote is the same one in the repository, and the same maths that runs inside GPT and Llama. Production systems change how it is executed, not what it computes.`,
    source: 'phase3-transformers/attention_numpy.py',
  },

  /* ---------- Part 7: sampling ---------- */
  {
    id: 'inference-code-topp',
    lesson: 'inference',
    title: 'Temperature and top-p',
    prompt: `Write two functions that reshape a probability distribution before a token is drawn.

\`apply_temperature(logits, t)\` divides the logits by \`t\` and returns probabilities via a stable softmax.

\`top_p(probs, p)\` keeps the smallest set of tokens whose probabilities reach \`p\`, sets the rest to 0, and renormalises so the kept ones sum to 1.

Sort by probability, highest first. Always keep at least one token.`,
    starter: `import numpy as np

def apply_temperature(logits, t):
    ...

def top_p(probs, p):
    ...

probs = np.array([0.5, 0.3, 0.15, 0.05])
print(top_p(probs, 0.8))   # keeps the top two: [0.625, 0.375, 0, 0]
`,
    solution: `import numpy as np

def apply_temperature(logits, t):
    z = np.asarray(logits, dtype=float) / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def top_p(probs, p):
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs)                 # highest first
    cumulative = np.cumsum(probs[order])
    # keep every token up to and including the first one that reaches p
    keep_count = int(np.searchsorted(cumulative, p) + 1)
    keep_count = max(1, min(keep_count, len(probs)))
    out = np.zeros_like(probs)
    kept = order[:keep_count]
    out[kept] = probs[kept]
    return out / out.sum()

probs = np.array([0.5, 0.3, 0.15, 0.05])
print(top_p(probs, 0.8))
`,
    tests: [
      { name: 'temperature 1 leaves the distribution alone', code: `import numpy as np\np = apply_temperature([2.0, 1.0, 0.0], 1.0)\nassert abs(p[0] - 0.6652) < 1e-3, f"got {p}"\nassert abs(p.sum() - 1) < 1e-9` },
      { name: 'low temperature sharpens, high temperature flattens', code: `import numpy as np\nsharp = apply_temperature([2.0, 1.0, 0.0], 0.5)[0]\nbase = apply_temperature([2.0, 1.0, 0.0], 1.0)[0]\nflat = apply_temperature([2.0, 1.0, 0.0], 2.0)[0]\nassert sharp > base > flat, f"got {sharp}, {base}, {flat}"` },
      { name: 'top-p keeps the smallest set that reaches p', code: `import numpy as np\nr = top_p(np.array([0.5, 0.3, 0.15, 0.05]), 0.8)\nassert int((r > 0).sum()) == 2, f"expected 2 tokens kept, got {int((r > 0).sum())}: {r}"\nassert abs(r[0] - 0.625) < 1e-6 and abs(r[1] - 0.375) < 1e-6, f"got {r}"` },
      { name: 'a larger p keeps more tokens', code: `import numpy as np\nr = top_p(np.array([0.5, 0.3, 0.15, 0.05]), 0.81)\nassert int((r > 0).sum()) == 3, f"expected 3, got {int((r > 0).sum())}: {r}"` },
      { name: 'a tiny p still keeps one token, and the result sums to 1', code: `import numpy as np\nr = top_p(np.array([0.5, 0.3, 0.15, 0.05]), 0.01)\nassert int((r > 0).sum()) == 1 and abs(r.sum() - 1) < 1e-9, f"got {r}"` },
    ],
    hints: [
      'For temperature: divide the logits by t, then run the same stable softmax as before (subtract the max, exponentiate, divide by the sum).',
      'For top-p: `np.argsort(-probs)` sorts highest first. `np.cumsum` of the sorted probabilities tells you where the running total reaches p.',
      '`np.searchsorted(cumulative, p)` gives the index where p fits; keep that many tokens plus one, and never fewer than one.',
    ],
    explanation: `Both of these happen outside the network. The model's job ends when it produces logits; these functions decide what to do with them.

Temperature rescales the gaps between logits, so it changes how confident the distribution looks. Top-p cuts the long tail of junk tokens: it keeps 2 tokens when the model is certain and many more when it is unsure, which is why it adapts better than a fixed top-k.`,
    source: 'phase3-transformers/kv_cache_demo.py',
  },

  /* ---------- Part 9: RAG ---------- */
  {
    id: 'rag-code-retrieve',
    lesson: 'rag',
    title: 'Retrieve the top-k chunks',
    prompt: `Write \`top_k(query_vec, chunk_vecs, k)\`: score every chunk by cosine similarity with the query, then return the indices of the k best, highest first.

\`chunk_vecs\` is a list of vectors. Use NumPy.`,
    starter: `import numpy as np

def top_k(query_vec, chunk_vecs, k):
    ...

q = np.array([1.0, 0.0])
chunks = [np.array([1.0, 0.1]), np.array([0.0, 1.0]), np.array([0.9, 0.0])]
print(top_k(q, chunks, 2))   # [0, 2] or [2, 0] by similarity: the two pointing along x
`,
    solution: `import numpy as np

def top_k(query_vec, chunk_vecs, k):
    q = np.asarray(query_vec, dtype=float)
    q_norm = np.linalg.norm(q)
    scores = []
    for v in chunk_vecs:
        v = np.asarray(v, dtype=float)
        denom = q_norm * np.linalg.norm(v)
        scores.append(0.0 if denom == 0 else float(q @ v / denom))
    order = np.argsort(-np.array(scores))
    return [int(i) for i in order[:k]]

q = np.array([1.0, 0.0])
chunks = [np.array([1.0, 0.1]), np.array([0.0, 1.0]), np.array([0.9, 0.0])]
print(top_k(q, chunks, 2))
`,
    tests: [
      { name: 'returns k indices', code: `import numpy as np\nr = top_k(np.array([1.0, 0.0]), [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])], 2)\nassert len(r) == 2, f"got {r}"` },
      { name: 'the best match comes first', code: `import numpy as np\nr = top_k(np.array([1.0, 0.0]), [np.array([0.0, 1.0]), np.array([1.0, 0.0])], 2)\nassert r[0] == 1, f"expected index 1 first, got {r}"` },
      { name: 'length does not decide the ranking, direction does', code: `import numpy as np\nr = top_k(np.array([1.0, 0.0]), [np.array([100.0, 100.0]), np.array([0.2, 0.0])], 2)\nassert r[0] == 1, f"the short vector points the right way, so it should win: got {r}"` },
      { name: 'a zero vector does not crash the search', code: `import numpy as np\nr = top_k(np.array([1.0, 0.0]), [np.array([0.0, 0.0]), np.array([1.0, 0.0])], 2)\nassert r[0] == 1, f"got {r}"` },
      { name: 'asking for more than there are returns them all', code: `import numpy as np\nr = top_k(np.array([1.0, 0.0]), [np.array([1.0, 0.0])], 5)\nassert len(r) == 1, f"got {r}"` },
    ],
    hints: [
      'Cosine similarity is `q @ v / (norm(q) * norm(v))`. `np.linalg.norm` gives the length.',
      'Collect the scores in a list, then `np.argsort(-scores)` puts the highest first.',
      'Guard against a zero-length vector before dividing, and slice the sorted order to k.',
    ],
    explanation: `This is the whole retrieval step of RAG. Everything else in a vector database is about doing this faster than comparing against every chunk.

The third test is the reason cosine is used instead of the raw dot product here: a long vector should not win a search just by being long.`,
    source: 'phase4-modern-llms/mini_rag.py',
  },
]

export default exercises
