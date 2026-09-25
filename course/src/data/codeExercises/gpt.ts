import type { CodeExerciseDef } from './types'

// The core of the course, as code you write yourself: derivatives, backprop, embeddings,
// the first language model, multi-head attention, the block, the GPT forward pass and training.
// Each is the NumPy version of the real function in the repo, cut down to the part that teaches it.
const exercises: CodeExerciseDef[] = [
  /* ---------- derivatives ---------- */
  {
    id: 'derivatives-code-chain-rule',
    lesson: 'derivatives',
    title: 'Measure a derivative, then predict it with the chain rule',
    prompt: `Two functions.

\`derivative(f, x, h=1e-5)\` measures the slope of \`f\` at \`x\` by nudging. Use the centred nudge: \`(f(x + h) - f(x - h)) / (2 * h)\`. It is far more accurate than nudging in one direction only.

\`chain_slope(x)\` returns the exact slope of \`y = (3x + 1) ** 2\` without nudging. Think of it as a pipeline: \`u = 3x + 1\`, then \`y = u ** 2\`. The chain rule says: multiply the local slopes, \`dy/du\` times \`du/dx\`.`,
    starter: `def derivative(f, x, h=1e-5):
    # nudge both ways, measure the change, divide by the size of the nudge
    ...

def chain_slope(x):
    # u = 3x + 1, y = u ** 2. Multiply the two local slopes.
    ...

print(derivative(lambda x: x ** 2, 3.0))   # about 6
print(chain_slope(2.0))                    # 42
`,
    solution: `def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def chain_slope(x):
    u = 3 * x + 1
    dy_du = 2 * u        # slope of u ** 2
    du_dx = 3            # slope of 3x + 1
    return dy_du * du_dx

print(derivative(lambda x: x ** 2, 3.0))   # 6.000000000
print(chain_slope(2.0))                    # 42.0
`,
    tests: [
      { name: 'the slope of x squared at 3 is 6', code: `d = derivative(lambda x: x ** 2, 3.0)\nassert abs(d - 6) < 1e-6, f"got {d}"` },
      { name: 'the centred nudge is accurate: slope of x cubed at 1 is 3 to within 1e-6', code: `d = derivative(lambda x: x ** 3, 1.0)\nassert abs(d - 3) < 1e-6, f"got {d}; a one-sided nudge is off by about 3e-5 here, use (f(x+h) - f(x-h)) / (2h)"` },
      { name: 'chain_slope(2) is 42', code: `s = chain_slope(2.0)\nassert abs(s - 42) < 1e-9, f"got {s}; at x = 2, u = 7, dy/du = 14 and du/dx = 3"` },
      { name: 'the chain rule agrees with the measured slope everywhere', code: `for x in [-2.0, -0.5, 0.0, 1.3, 4.0]:\n    measured = derivative(lambda t: (3 * t + 1) ** 2, x)\n    exact = chain_slope(x)\n    assert abs(measured - exact) < 1e-4, f"at x = {x}: measured {measured}, chain rule {exact}"` },
    ],
    hints: [
      '`derivative` is one line: `return (f(x + h) - f(x - h)) / (2 * h)`.',
      'For `chain_slope`, the slope of `u ** 2` with respect to `u` is `2 * u`, and the slope of `3 * x + 1` is `3`.',
      'Compute `u = 3 * x + 1` first, then `return 2 * u * 3`.',
    ],
    explanation: `The nudge measures a slope; the chain rule predicts it. They agree, which is the whole reason backpropagation can be trusted: it is the chain rule applied mechanically, and a nudge test (a "gradient check") catches any mistake.

The centred nudge cancels the first error term, so its error shrinks with h squared instead of h. That is why gradient checks in the repo nudge both ways.`,
    source: 'phase1-foundations/math_primer.py',
  },

  /* ---------- backprop ---------- */
  {
    id: 'backprop-code-linear-backward',
    lesson: 'backprop',
    title: 'Backward through one linear layer',
    prompt: `A linear layer computes \`Y = X @ W + b\`. X is (n, d_in), W is (d_in, d_out), b is (d_out,).

During backprop you are handed \`dY\`: how much the loss changes per unit change of each entry of Y. Write \`linear_backward(X, W, dY)\` that returns \`(dX, dW, db)\`, each with the same shape as the thing it is the gradient of.

The rules from the lesson: blame flows to the weights as \`X.T @ dY\`, to the input as \`dY @ W.T\`, and the bias was added to every row, so its gradient is the sum of dY over the rows.`,
    starter: `import numpy as np

def linear_backward(X, W, dY):
    # return dX (like X), dW (like W), db (like b)
    ...

X = np.array([[1.0, 2.0], [3.0, 4.0]])
W = np.array([[1.0, 0.0, -1.0], [0.5, 2.0, 0.0]])
dY = np.ones((2, 3))
print(linear_backward(X, W, dY))
`,
    solution: `import numpy as np

def linear_backward(X, W, dY):
    dX = dY @ W.T            # (n, d_out) @ (d_out, d_in) -> (n, d_in)
    dW = X.T @ dY            # (d_in, n) @ (n, d_out)     -> (d_in, d_out)
    db = dY.sum(axis=0)      # b was added to every row
    return dX, dW, db

X = np.array([[1.0, 2.0], [3.0, 4.0]])
W = np.array([[1.0, 0.0, -1.0], [0.5, 2.0, 0.0]])
dY = np.ones((2, 3))
print(linear_backward(X, W, dY))
`,
    tests: [
      { name: 'the shapes match X, W and b', code: `import numpy as np\nrng = np.random.default_rng(0)\nX = rng.normal(size=(5, 4)); W = rng.normal(size=(4, 3)); dY = rng.normal(size=(5, 3))\ndX, dW, db = linear_backward(X, W, dY)\nassert dX.shape == (5, 4) and dW.shape == (4, 3) and db.shape == (3,), f"got {dX.shape}, {dW.shape}, {db.shape}; expected (5, 4), (4, 3), (3,)"` },
      { name: 'the worked example: db is [2, 2, 2] and dW row 0 is [4, 4, 4]', code: `import numpy as np\nX = np.array([[1.0, 2.0], [3.0, 4.0]])\nW = np.array([[1.0, 0.0, -1.0], [0.5, 2.0, 0.0]])\ndX, dW, db = linear_backward(X, W, np.ones((2, 3)))\nassert np.allclose(db, [2, 2, 2]), f"db = {db}"\nassert np.allclose(dW, [[4, 4, 4], [6, 6, 6]]), f"dW = {dW}"\nassert np.allclose(dX, [[0, 2.5], [0, 2.5]]), f"dX = {dX}"` },
      { name: 'all three gradients pass a nudge test', code: `import numpy as np\nrng = np.random.default_rng(1)\nX = rng.normal(size=(3, 4)); W = rng.normal(size=(4, 2)); b = rng.normal(size=2); G = rng.normal(size=(3, 2))\nloss = lambda X, W, b: float(((X @ W + b) * G).sum())   # so dY is exactly G\ndX, dW, db = linear_backward(X, W, G)\nh = 1e-6\nfor name, arr, grad in [("dX", X, dX), ("dW", W, dW), ("db", b, db)]:\n    for idx in np.ndindex(arr.shape):\n        old = arr[idx]; arr[idx] = old + h; up = loss(X, W, b); arr[idx] = old - h; down = loss(X, W, b); arr[idx] = old\n        num = (up - down) / (2 * h)\n        assert abs(num - grad[idx]) < 1e-5, f"{name}{list(idx)}: nudge says {num:.6f}, you have {grad[idx]:.6f}"` },
    ],
    hints: [
      'Match shapes. dW must be (d_in, d_out): the only product of X and dY that gives that is `X.T @ dY`.',
      'dX must be (n, d_in): `dY @ W.T`. db must be (d_out,): add dY up over the rows with `dY.sum(axis=0)`.',
      '`return dY @ W.T, X.T @ dY, dY.sum(axis=0)`.',
    ],
    explanation: `Every linear layer in every network, GPT included, runs these three lines on the way back. You can always recover them from shapes alone: there is only one way to multiply X and dY into something shaped like W.

\`dW = X.T @ dY\` says a weight is blamed in proportion to how active its input was. \`dX = dY @ W.T\` passes the blame one layer further down, which is what makes the chain rule run through a whole stack.`,
    source: 'phase1-foundations/mlp_numpy.py',
  },
  {
    id: 'backprop-code-two-layer',
    lesson: 'backprop',
    title: 'Backprop through a whole two-layer network',
    prompt: `The network is \`h = relu(X @ W1 + b1)\`, \`logits = h @ W2 + b2\`, and the loss is the average cross-entropy of \`softmax(logits)\` against integer labels \`y\`. \`forward\` and \`loss\` are written for you.

Write \`backward(X, y, W1, b1, W2, b2)\` that returns a dict with the gradients \`"W1"\`, \`"b1"\`, \`"W2"\`, \`"b2"\`. Walk the network in reverse:

1. \`d_logits = (probs - one_hot(y)) / n\`
2. the last layer: the same rules as a linear layer
3. pass the blame to \`h\`, then through the ReLU gate: zero it wherever the pre-activation was not positive
4. the first layer: the same rules again`,
    starter: `import numpy as np

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h = np.maximum(0, z1)
    return z1, h, h @ W2 + b2

def loss(X, y, W1, b1, W2, b2):
    _, _, logits = forward(X, W1, b1, W2, b2)
    p = softmax(logits)
    return -np.log(p[np.arange(len(y)), y]).mean()

def backward(X, y, W1, b1, W2, b2):
    z1, h, logits = forward(X, W1, b1, W2, b2)
    n = len(y)
    # 1. d_logits  2. last layer  3. back through the ReLU  4. first layer
    ...
`,
    solution: `import numpy as np

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h = np.maximum(0, z1)
    return z1, h, h @ W2 + b2

def loss(X, y, W1, b1, W2, b2):
    _, _, logits = forward(X, W1, b1, W2, b2)
    p = softmax(logits)
    return -np.log(p[np.arange(len(y)), y]).mean()

def backward(X, y, W1, b1, W2, b2):
    z1, h, logits = forward(X, W1, b1, W2, b2)
    n = len(y)
    d = softmax(logits)
    d[np.arange(n), y] -= 1          # probs - one_hot
    d /= n                           # the loss is an average
    grads = {"W2": h.T @ d, "b2": d.sum(axis=0)}
    d = d @ W2.T                     # blame on h
    d = d * (z1 > 0)                 # the ReLU gate
    grads["W1"] = X.T @ d
    grads["b1"] = d.sum(axis=0)
    return grads
`,
    tests: [
      { name: 'returns all four gradients with the right shapes', code: `import numpy as np\nrng = np.random.default_rng(0)\nX = rng.normal(size=(6, 2)); y = np.array([0, 1, 2, 0, 1, 2])\nW1 = rng.normal(size=(2, 5)); b1 = rng.normal(size=5) * 0.1; W2 = rng.normal(size=(5, 3)); b2 = np.zeros(3)\ng = backward(X, y, W1, b1, W2, b2)\nfor k, ref in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:\n    assert k in g, f"missing gradient {k}"\n    assert g[k].shape == ref.shape, f"{k}: got shape {g[k].shape}, expected {ref.shape}"` },
      { name: 'the gradient check passes for every weight', code: `import numpy as np\nrng = np.random.default_rng(1)\nX = rng.normal(size=(8, 2)); y = rng.integers(0, 3, size=8)\nP = {"W1": rng.normal(size=(2, 6)), "b1": rng.normal(size=6) * 0.1, "W2": rng.normal(size=(6, 3)), "b2": rng.normal(size=3) * 0.1}\ng = backward(X, y, P["W1"], P["b1"], P["W2"], P["b2"])\nh = 1e-5\nworst = 0.0\nfor k, arr in P.items():\n    for idx in np.ndindex(arr.shape):\n        old = arr[idx]; arr[idx] = old + h; up = loss(X, y, **P); arr[idx] = old - h; down = loss(X, y, **P); arr[idx] = old\n        num = (up - down) / (2 * h)\n        assert abs(num - g[k][idx]) < 1e-6, f"{k}{list(idx)}: the nudge says {num:.7f}, backprop says {g[k][idx]:.7f}"` },
      { name: 'a neuron that never fires gets no gradient (the ReLU gate)', code: `import numpy as np\nX = np.array([[1.0, 2.0], [0.5, 1.0]]); y = np.array([0, 1])\nW1 = np.array([[1.0, -1.0], [1.0, -1.0]]); b1 = np.zeros(2)   # neuron 1 is always negative\nW2 = np.array([[1.0, 0.0], [0.0, 1.0]]); b2 = np.zeros(2)\ng = backward(X, y, W1, b1, W2, b2)\nassert np.allclose(g["W1"][:, 1], 0) and abs(g["b1"][1]) < 1e-12, f"W1 column 1 should be 0, got {g['W1'][:, 1]}"\nassert not np.allclose(g["W1"][:, 0], 0), "neuron 0 fires, so it should get a gradient"` },
      { name: 'one step against the gradient lowers the loss', code: `import numpy as np\nrng = np.random.default_rng(2)\nX = rng.normal(size=(20, 2)); y = rng.integers(0, 3, size=20)\nW1 = rng.normal(size=(2, 8)); b1 = np.zeros(8); W2 = rng.normal(size=(8, 3)); b2 = np.zeros(3)\nbefore = loss(X, y, W1, b1, W2, b2)\ng = backward(X, y, W1, b1, W2, b2)\nafter = loss(X, y, W1 - 0.1 * g["W1"], b1 - 0.1 * g["b1"], W2 - 0.1 * g["W2"], b2 - 0.1 * g["b2"])\nassert after < before, f"loss went from {before:.4f} to {after:.4f}"` },
    ],
    hints: [
      'Start with `d = softmax(logits)`, then `d[np.arange(n), y] -= 1` and `d /= n`. That is the gradient of the loss with respect to the logits.',
      'Last layer: `W2` gets `h.T @ d`, `b2` gets `d.sum(axis=0)`. Then `d = d @ W2.T` is the blame on `h`.',
      'The ReLU gate is `d = d * (z1 > 0)`. After that the first layer is the same two lines with `X` in place of `h`.',
    ],
    explanation: `This is \`MLP.backward\` from the repo, unrolled for two layers. Nothing new appears: two linear-layer backward passes and one gate in between.

The \`probs - one_hot\` line is where softmax and cross-entropy meet. Their combined derivative is that simple, which is why every framework fuses them. The ReLU gate explains "dead neurons": a unit whose input is never positive passes back zero, so it never learns its way out.

The gradient check is how the repo proves the whole thing right, and you just passed it.`,
    source: 'phase1-foundations/mlp_numpy.py',
  },

  /* ---------- embeddings ---------- */
  {
    id: 'embeddings-code-skipgram-step',
    lesson: 'embeddings',
    title: 'One skip-gram training step',
    prompt: `Skip-gram learns word vectors by asking each word to predict its neighbours. For one (center, context) pair:

- look up the center word's row of the embedding matrix: \`emb = E[center]\` (shape D)
- score every word in the vocabulary: \`logits = emb @ W\` (shape V)
- \`probs = softmax(logits)\`; the loss is \`-log(probs[context])\`

Write \`sgd_step(E, W, center, context, lr)\` that does one gradient step and returns the updated \`(E, W)\`. Work on copies. The gradient on the logits is \`probs\` with 1 subtracted at the context word. Only the center row of E is touched.`,
    starter: `import numpy as np

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def sgd_step(E, W, center, context, lr):
    E, W = E.copy(), W.copy()
    # forward: lookup, logits, probs. backward: d_logits, dW, d_emb. update.
    ...
    return E, W
`,
    solution: `import numpy as np

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def sgd_step(E, W, center, context, lr):
    E, W = E.copy(), W.copy()
    emb = E[center]                  # (D,)
    probs = softmax(emb @ W)         # (V,)
    d_logits = probs.copy()
    d_logits[context] -= 1           # probs - one_hot
    dW = np.outer(emb, d_logits)     # (D, V)
    d_emb = W @ d_logits             # (D,), using W before it is updated
    W -= lr * dW
    E[center] -= lr * d_emb          # only the looked-up row moves
    return E, W
`,
    tests: [
      { name: 'only the center word’s row of E changes', code: `import numpy as np\nrng = np.random.default_rng(0)\nE = rng.normal(size=(6, 4)) * 0.1; W = rng.normal(size=(4, 6)) * 0.1\nE2, W2 = sgd_step(E.copy(), W.copy(), 2, 5, 0.5)\nchanged = [i for i in range(6) if not np.allclose(E2[i], E[i])]\nassert changed == [2], f"rows that changed: {changed}; only row 2 was looked up"` },
      { name: 'the loss on that pair goes down', code: `import numpy as np\nrng = np.random.default_rng(1)\nE = rng.normal(size=(8, 5)) * 0.1; W = rng.normal(size=(5, 8)) * 0.1\nL = lambda E, W: -np.log(softmax(E[3] @ W)[6])\nE2, W2 = sgd_step(E.copy(), W.copy(), 3, 6, 0.5)\nassert L(E2, W2) < L(E, W), f"loss went from {L(E, W):.4f} to {L(E2, W2):.4f}"` },
      { name: 'the update matches the gradient exactly', code: `import numpy as np\nrng = np.random.default_rng(2)\nE = rng.normal(size=(5, 3)); W = rng.normal(size=(3, 5))\nlr = 0.1\nE2, W2 = sgd_step(E.copy(), W.copy(), 1, 4, lr)\np = softmax(E[1] @ W); d = p.copy(); d[4] -= 1\nassert np.allclose(W2, W - lr * np.outer(E[1], d)), "W was not updated by lr * outer(emb, d_logits)"\nassert np.allclose(E2[1], E[1] - lr * (W @ d)), "E[center] should move by lr * (W @ d_logits), using W from BEFORE its update"` },
      { name: 'repeating the step makes the context word the favourite', code: `import numpy as np\nrng = np.random.default_rng(3)\nE = rng.normal(size=(7, 4)) * 0.1; W = rng.normal(size=(4, 7)) * 0.1\nfor _ in range(50):\n    E, W = sgd_step(E, W, 0, 3, 0.5)\ntop = int(np.argmax(E[0] @ W))\nassert top == 3, f"after 50 steps the top prediction for word 0 is {top}, expected 3"` },
    ],
    hints: [
      'Forward: `emb = E[center]`, `probs = softmax(emb @ W)`. Backward: `d_logits = probs.copy(); d_logits[context] -= 1`.',
      '`dW = np.outer(emb, d_logits)` has shape (D, V). `d_emb = W @ d_logits` has shape (D,). Compute both before changing W.',
      'Update with `W -= lr * dW` and `E[center] -= lr * d_emb`.',
    ],
    explanation: `This is the inner loop of \`tiny_word2vec.py\`, for one pair instead of a batch. The embedding row is just a weight that gets a gradient only when its word appears, which is why the repo uses \`np.add.at\` to scatter the updates.

Words that share neighbours get pushed toward the same predictions, so their rows drift toward each other. Nothing in the code says "cat is like dog": the geometry is a side effect of predicting context.`,
    source: 'phase2-language/tiny_word2vec.py',
  },

  /* ---------- next-token ---------- */
  {
    id: 'next-token-code-bigram',
    lesson: 'next-token',
    title: 'A count-table language model and its loss',
    prompt: `Model A from the lesson: count how often each token follows each other token, then turn every row into probabilities.

\`bigram_table(ids, V, smoothing=0.01)\` returns a (V, V) array where row \`a\` is P(next | current = a). Start every cell at \`smoothing\` (so unseen pairs are unlikely, not impossible), add 1 for each consecutive pair \`(ids[i], ids[i+1])\`, then divide each row by its sum.

\`avg_cross_entropy(table, ids)\` is the average of \`-log P(ids[i+1] | ids[i])\` over every consecutive pair: the model's average surprise.`,
    starter: `import numpy as np

def bigram_table(ids, V, smoothing=0.01):
    ...

def avg_cross_entropy(table, ids):
    ...

ids = [0, 1, 0, 1, 0, 2]
print(bigram_table(ids, 3))
`,
    solution: `import numpy as np

def bigram_table(ids, V, smoothing=0.01):
    counts = np.full((V, V), smoothing)
    for a, b in zip(ids, ids[1:]):
        counts[a, b] += 1
    return counts / counts.sum(axis=1, keepdims=True)

def avg_cross_entropy(table, ids):
    ids = np.asarray(ids)
    p = table[ids[:-1], ids[1:]]      # probability of each actual next token
    return float(-np.log(p).mean())

ids = [0, 1, 0, 1, 0, 2]
print(bigram_table(ids, 3).round(3))
`,
    tests: [
      { name: 'every row is a probability distribution', code: `import numpy as np\nt = bigram_table([0, 1, 0, 1, 0, 2, 2, 1], 3)\nassert t.shape == (3, 3), f"shape {t.shape}"\nassert np.allclose(t.sum(axis=1), 1), f"row sums {t.sum(axis=1)}"` },
      { name: 'the counts are right: after 0 comes 1 twice and 2 once', code: `import numpy as np\nt = bigram_table([0, 1, 0, 1, 0, 2], 3, smoothing=1e-12)\nassert abs(t[0, 1] - 2 / 3) < 1e-6 and abs(t[0, 2] - 1 / 3) < 1e-6, f"row 0 is {t[0]}"\nassert abs(t[1, 0] - 1) < 1e-6, f"row 1 is {t[1]}; 1 is always followed by 0"` },
      { name: 'smoothing keeps unseen pairs above zero', code: `import numpy as np\nt = bigram_table([0, 1, 0, 1], 3)\nassert (t > 0).all(), f"found a zero probability: {t}"\nce = avg_cross_entropy(t, [0, 2])\nassert np.isfinite(ce), f"an unseen pair gave loss {ce}"` },
      { name: 'a uniform model scores exactly log(V)', code: `import numpy as np\nV = 5\nce = avg_cross_entropy(np.full((V, V), 1 / V), [0, 3, 1, 4, 2, 2])\nassert abs(ce - np.log(V)) < 1e-9, f"got {ce}, expected log(5) = {np.log(V):.4f}"` },
      { name: 'the counted model beats guessing on its own text', code: `import numpy as np\ntext = "the cat sat on the mat and the cat ran"\nchars = sorted(set(text)); ids = [chars.index(c) for c in text]\nV = len(chars)\nce = avg_cross_entropy(bigram_table(ids, V), ids)\nassert ce < np.log(V) - 0.5, f"loss {ce:.3f} should be well below log(V) = {np.log(V):.3f}"` },
    ],
    hints: [
      '`counts = np.full((V, V), smoothing)`, then loop over `zip(ids, ids[1:])` and add 1 to `counts[a, b]`.',
      'Normalise rows with `counts / counts.sum(axis=1, keepdims=True)`.',
      'For the loss, fancy indexing picks every needed probability at once: `table[ids[:-1], ids[1:]]`. Then `-np.log(...).mean()`.',
    ],
    explanation: `Row a of the table is the model: a probability for every possible next token, given only the current one. Generating text is just rolling that weighted die over and over.

Cross-entropy is the scoreboard every language model uses, GPT included. log(V) is the score of pure guessing, so any real learning shows up as a number below it. The smoothing matters because one zero probability on a pair that does occur makes the loss infinite.`,
    source: 'phase2-language/bigram_lm.py',
  },

  /* ---------- masks-and-heads ---------- */
  {
    id: 'masks-and-heads-code-multihead',
    lesson: 'masks-and-heads',
    title: 'Multi-head causal attention',
    prompt: `Write three functions, all in NumPy.

\`split_heads(M, n_heads)\` turns a (T, D) array into (n_heads, T, D // n_heads): head h gets columns \`h*hd\` to \`(h+1)*hd\`.

\`merge_heads(M)\` is the inverse: (H, T, hd) back to (T, H*hd).

\`multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads)\`: project x to Q, K and V, split each into heads, run causal attention in every head at once (scale by the square root of the head size, mask the future, softmax, blend), merge the heads, and multiply by Wo. Return the (T, D) output.`,
    starter: `import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def split_heads(M, n_heads):
    ...

def merge_heads(M):
    ...

def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads):
    ...
`,
    solution: `import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def split_heads(M, n_heads):
    T, D = M.shape
    return M.reshape(T, n_heads, D // n_heads).transpose(1, 0, 2)   # (H, T, hd)

def merge_heads(M):
    H, T, hd = M.shape
    return M.transpose(1, 0, 2).reshape(T, H * hd)                   # (T, D)

def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads):
    T, D = x.shape
    hd = D // n_heads
    Q = split_heads(x @ Wq, n_heads)
    K = split_heads(x @ Wk, n_heads)
    V = split_heads(x @ Wv, n_heads)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(hd)                  # (H, T, T)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)
    out = softmax(scores) @ V                                        # (H, T, hd)
    return merge_heads(out) @ Wo
`,
    tests: [
      { name: 'split_heads gives each head its own slice of columns', code: `import numpy as np\nM = np.arange(24.0).reshape(3, 8)\nS = split_heads(M, 4)\nassert S.shape == (4, 3, 2), f"shape {None if S is None else S.shape}, expected (4, 3, 2)"\nassert np.array_equal(S[1], M[:, 2:4]), f"head 1 should be columns 2:4, got {S[1]}"` },
      { name: 'merge_heads undoes split_heads', code: `import numpy as np\nM = np.random.default_rng(0).normal(size=(5, 12))\nassert np.array_equal(merge_heads(split_heads(M, 3)), M), "merge_heads(split_heads(M)) should give M back"` },
      { name: 'with one head it is ordinary causal attention', code: `import numpy as np\nrng = np.random.default_rng(1)\nT, D = 4, 6\nx = rng.normal(size=(T, D)); Wq, Wk, Wv = (rng.normal(size=(D, D)) for _ in range(3)); Wo = np.eye(D)\nQ, K, V = x @ Wq, x @ Wk, x @ Wv\ns = Q @ K.T / np.sqrt(D); s[np.triu_indices(T, 1)] = -np.inf\nref = softmax(s) @ V\nout = multi_head_attention(x, Wq, Wk, Wv, Wo, 1)\nassert np.allclose(out, ref), "with n_heads=1 and Wo = identity this should equal single-head causal attention"` },
      { name: 'the future cannot leak: changing token 3 leaves tokens 0 to 2 unchanged', code: `import numpy as np\nrng = np.random.default_rng(2)\nT, D = 5, 8\nx = rng.normal(size=(T, D)); W = [rng.normal(size=(D, D)) for _ in range(4)]\na = multi_head_attention(x, *W, 2)\nx2 = x.copy(); x2[3:] += 5.0\nb = multi_head_attention(x2, *W, 2)\nassert a.shape == (T, D), f"output shape {a.shape}"\nassert np.allclose(a[:3], b[:3]), "outputs for positions 0-2 changed when only later tokens changed: check the causal mask"\nassert not np.allclose(a[3:], b[3:]), "positions 3 and 4 should change"` },
      { name: 'heads really are separate: 2 heads differ from 1 head', code: `import numpy as np\nrng = np.random.default_rng(3)\nx = rng.normal(size=(4, 8)); W = [rng.normal(size=(8, 8)) for _ in range(4)]\nassert not np.allclose(multi_head_attention(x, *W, 1), multi_head_attention(x, *W, 2)), "splitting into heads should change the result: each head has its own softmax"` },
    ],
    hints: [
      '`split_heads`: `M.reshape(T, n_heads, hd)` puts each head’s columns together, then `.transpose(1, 0, 2)` moves the head axis to the front.',
      'NumPy matmul works on the last two axes, so `Q @ K.transpose(0, 2, 1)` gives (H, T, T) scores: every head at once. Divide by `np.sqrt(hd)`, not `np.sqrt(D)`.',
      'Mask with `np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, scores)`; it broadcasts over the heads. Then `merge_heads(softmax(scores) @ V) @ Wo`.',
    ],
    explanation: `The heads cost nothing extra: the same D columns are just sliced into groups, and each group gets its own softmax. That is why one head can track the previous word while another tracks the subject of the sentence.

The reshape and transpose are the only new parts, and they are exactly the \`split\` inside \`multi_head_attention\` in the repo. \`Wo\` then mixes what the heads found back into one vector per token.`,
    source: 'phase3-transformers/attention_numpy.py',
  },

  /* ---------- transformer-block ---------- */
  {
    id: 'transformer-block-code-prenorm',
    lesson: 'transformer-block',
    title: 'LayerNorm and a pre-norm block',
    prompt: `\`layernorm(x, gamma, beta, eps=1e-5)\` normalises each row of x (each token) on its own: subtract the row mean, divide by \`sqrt(row variance + eps)\`, then scale by \`gamma\` and shift by \`beta\`.

\`block(x, attn, mlp, ln1, ln2)\` is the two-line Transformer block from \`tiny_gpt.py\`. \`attn\` and \`mlp\` are functions from (T, D) to (T, D); \`ln1\` and \`ln2\` are \`(gamma, beta)\` pairs. Normalise, apply the sub-layer, and ADD the result to x. First with attention, then with the MLP.`,
    starter: `import numpy as np

def layernorm(x, gamma, beta, eps=1e-5):
    ...

def block(x, attn, mlp, ln1, ln2):
    # x = x + attn(LN1(x));  x = x + mlp(LN2(x))
    ...
`,
    solution: `import numpy as np

def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

def block(x, attn, mlp, ln1, ln2):
    x = x + attn(layernorm(x, *ln1))   # communicate, as a correction
    x = x + mlp(layernorm(x, *ln2))    # compute, as a correction
    return x
`,
    tests: [
      { name: 'layernorm gives every row mean 0 and variance 1', code: `import numpy as np\nx = np.random.default_rng(0).normal(3, 7, size=(4, 16))\ny = layernorm(x, np.ones(16), np.zeros(16))\nassert np.allclose(y.mean(axis=1), 0, atol=1e-7), f"row means {y.mean(axis=1)}"\nassert np.allclose(y.var(axis=1), 1, atol=1e-3), f"row variances {y.var(axis=1)}"` },
      { name: 'layernorm applies gamma and beta after normalising', code: `import numpy as np\nx = np.array([[1.0, 2.0, 3.0]])\ny = layernorm(x, np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]), eps=0.0)\nexpected = np.array([[1 - 2 * 1.2247449, 1.0, 1 + 2 * 1.2247449]])\nassert np.allclose(y, expected, atol=1e-6), f"got {y}, expected {expected}"` },
      { name: 'if both sub-layers output zeros, the block passes x through unchanged', code: `import numpy as np\nx = np.random.default_rng(1).normal(size=(3, 4))\nzero = lambda h: np.zeros_like(h)\nln = (np.ones(4), np.zeros(4))\nout = block(x, zero, zero, ln, ln)\nassert np.allclose(out, x), "the residual connection should carry x straight through"` },
      { name: 'the MLP sees the output of the attention step', code: `import numpy as np\nrng = np.random.default_rng(2)\nx = rng.normal(size=(3, 4)); A = rng.normal(size=(4, 4)); M = rng.normal(size=(4, 4))\nln1 = (rng.normal(size=4), rng.normal(size=4)); ln2 = (rng.normal(size=4), rng.normal(size=4))\nattn = lambda h: h @ A\nmlp = lambda h: np.maximum(0, h @ M)\ndef LN(v, g, b):\n    return (v - v.mean(-1, keepdims=True)) / np.sqrt(v.var(-1, keepdims=True) + 1e-5) * g + b\nx1 = x + attn(LN(x, *ln1))\nref = x1 + mlp(LN(x1, *ln2))\nassert np.allclose(block(x, attn, mlp, ln1, ln2), ref), "expected x1 = x + attn(LN1(x)), then x1 + mlp(LN2(x1))"` },
    ],
    hints: [
      'Use `axis=-1, keepdims=True` for the mean and the variance so they broadcast back over each row. NumPy has `x.var(...)`.',
      'Unpack the pairs with `layernorm(x, *ln1)`.',
      'Two lines: `x = x + attn(layernorm(x, *ln1))`, then `x = x + mlp(layernorm(x, *ln2))`, then return x.',
    ],
    explanation: `The whole Transformer is this block repeated. Attention lets tokens exchange information; the MLP processes each token on its own; LayerNorm keeps the numbers in a steady range; and the \`+\` makes every sub-layer a correction to x rather than a replacement.

That \`+\` is also a gradient highway: during backprop the gradient flows straight through the addition, which is why dozens of these blocks can be stacked and still train. "Pre-norm" means the LayerNorm sits inside the branch, so the main path is never normalised away.`,
    source: 'phase3-transformers/tiny_gpt.py',
  },

  /* ---------- build-gpt ---------- */
  {
    id: 'build-gpt-code-forward',
    lesson: 'build-gpt',
    title: 'The whole GPT forward pass',
    prompt: `Write \`gpt_forward(ids, params)\` for a list of T token ids. \`params\` holds:

- \`"tok_emb"\`: (V, D), one row per token in the vocabulary
- \`"pos_emb"\`: (T_max, D), one row per position
- \`"blocks"\`: a list of functions, each mapping (T, D) to (T, D)
- \`"ln_f"\`: a \`(gamma, beta)\` pair for the final LayerNorm

The steps, exactly as in \`GPT.forward\`: token embedding plus position embedding, every block in order, the final LayerNorm, then the output head. The head is tied to the token embedding: logits are \`x @ tok_emb.T\`. Return the (T, V) logits. \`layernorm\` is given.`,
    starter: `import numpy as np

def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

def gpt_forward(ids, params):
    # embed + position -> blocks -> final layernorm -> tied head
    ...

rng = np.random.default_rng(0)
V, D = 10, 8
params = {
    "tok_emb": rng.normal(size=(V, D)) * 0.1,
    "pos_emb": rng.normal(size=(16, D)) * 0.1,
    "blocks": [lambda x: x + np.tanh(x)],
    "ln_f": (np.ones(D), np.zeros(D)),
}
print(gpt_forward([1, 4, 2], params))
`,
    solution: `import numpy as np

def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

def gpt_forward(ids, params):
    ids = np.asarray(ids)
    T = len(ids)
    x = params["tok_emb"][ids] + params["pos_emb"][:T]   # meaning + position
    for block in params["blocks"]:
        x = block(x)                                      # communicate / compute
    x = layernorm(x, *params["ln_f"])
    return x @ params["tok_emb"].T                        # tied head: (T, V)

rng = np.random.default_rng(0)
V, D = 10, 8
params = {
    "tok_emb": rng.normal(size=(V, D)) * 0.1,
    "pos_emb": rng.normal(size=(16, D)) * 0.1,
    "blocks": [lambda x: x + np.tanh(x)],
    "ln_f": (np.ones(D), np.zeros(D)),
}
print(gpt_forward([1, 4, 2], params).shape)   # (3, 10)
`,
    tests: [
      { name: 'the logits have shape (T, V)', code: `import numpy as np\nrng = np.random.default_rng(0)\nV, D = 11, 6\nP = {"tok_emb": rng.normal(size=(V, D)), "pos_emb": rng.normal(size=(20, D)), "blocks": [lambda x: x], "ln_f": (np.ones(D), np.zeros(D))}\nout = gpt_forward([3, 1, 4, 1, 5], P)\nassert out is not None and out.shape == (5, V), f"got shape {None if out is None else out.shape}, expected (5, 11)"` },
      { name: 'matches the reference forward pass, with two blocks', code: `import numpy as np\nrng = np.random.default_rng(1)\nV, D = 7, 4\nA = rng.normal(size=(D, D)); B = rng.normal(size=(D, D))\nP = {"tok_emb": rng.normal(size=(V, D)), "pos_emb": rng.normal(size=(9, D)), "blocks": [lambda x: x + np.tanh(x @ A), lambda x: x + np.tanh(x @ B)], "ln_f": (rng.normal(size=D), rng.normal(size=D))}\nids = [6, 0, 2]\nx = P["tok_emb"][ids] + P["pos_emb"][:3]\nx = x + np.tanh(x @ A); x = x + np.tanh(x @ B)\nref = layernorm(x, *P["ln_f"]) @ P["tok_emb"].T\nassert np.allclose(gpt_forward(ids, P), ref), "expected embed + pos, block 1, block 2, final layernorm, then @ tok_emb.T"` },
      { name: 'position matters: the same token at two positions gives different logits', code: `import numpy as np\nrng = np.random.default_rng(2)\nV, D = 5, 4\nP = {"tok_emb": rng.normal(size=(V, D)), "pos_emb": rng.normal(size=(8, D)), "blocks": [], "ln_f": (np.ones(D), np.zeros(D))}\nout = gpt_forward([2, 2], P)\nassert not np.allclose(out[0], out[1]), "token 2 at position 0 and position 1 should differ: did you add pos_emb[:T]?"` },
      { name: 'the head is tied to the token embedding', code: `import numpy as np\nD = 4\ntok = np.eye(D)   # 4 tokens, each its own direction\nP = {"tok_emb": tok * 3.0, "pos_emb": np.zeros((4, D)), "blocks": [], "ln_f": (np.ones(D), np.zeros(D))}\nout = gpt_forward([0, 1, 2, 3], P)\nassert (out.argmax(axis=1) == np.arange(4)).all(), f"with no blocks each token should predict itself through the tied head; argmax is {out.argmax(axis=1)}"` },
    ],
    hints: [
      'Embedding lookup is indexing: `params["tok_emb"][ids]` is (T, D). Add `params["pos_emb"][:T]`.',
      'Loop `for block in params["blocks"]: x = block(x)`, then `x = layernorm(x, *params["ln_f"])`.',
      'Tied head: `return x @ params["tok_emb"].T`, which is (T, D) @ (D, V).',
    ],
    explanation: `That is a GPT, start to finish. Everything the model knows lives in the embedding tables and the blocks; the rest is plumbing you have now written yourself.

Weight tying means the same matrix reads tokens in and writes predictions out: the logit for a token is a dot product with that token's embedding. It saves V times D parameters, which for a real vocabulary is a large share of the model. The repo's \`GPT\` class does the same with \`self.head.weight = self.tok_emb.weight\`.`,
    source: 'phase3-transformers/tiny_gpt.py',
  },

  /* ---------- training-gpt ---------- */
  {
    id: 'training-gpt-code-get-batch',
    lesson: 'training-gpt',
    title: 'Cut a training batch out of the text',
    prompt: `Training reads random windows of the token stream. Write \`get_batch(data, context_len, batch_size, rng)\`:

- \`data\` is a 1-D NumPy array of token ids
- pick \`batch_size\` random start positions with \`rng.integers(0, len(data) - context_len, size=batch_size)\`
- \`x\` rows are \`data[i : i + context_len]\`; \`y\` rows are the same windows shifted one to the right, \`data[i + 1 : i + context_len + 1]\`

Return \`(x, y)\`, both (batch_size, context_len). Every position of x is a training example whose answer sits at the same position in y.`,
    starter: `import numpy as np

def get_batch(data, context_len, batch_size, rng):
    ...

data = np.arange(20)
print(get_batch(data, 4, 2, np.random.default_rng(0)))
`,
    solution: `import numpy as np

def get_batch(data, context_len, batch_size, rng):
    ix = rng.integers(0, len(data) - context_len, size=batch_size)
    x = np.stack([data[i:i + context_len] for i in ix])
    y = np.stack([data[i + 1:i + context_len + 1] for i in ix])
    return x, y

data = np.arange(20)
x, y = get_batch(data, 4, 2, np.random.default_rng(0))
print(x)
print(y)
`,
    tests: [
      { name: 'x and y have shape (batch_size, context_len)', code: `import numpy as np\nx, y = get_batch(np.arange(100), 8, 5, np.random.default_rng(0))\nassert x.shape == (5, 8) and y.shape == (5, 8), f"got {x.shape} and {y.shape}"` },
      { name: 'y is x shifted by one token', code: `import numpy as np\ndata = np.random.default_rng(1).integers(0, 50, size=200)\nx, y = get_batch(data, 10, 6, np.random.default_rng(2))\nassert np.array_equal(x[:, 1:], y[:, :-1]), "y[:, :-1] should equal x[:, 1:]"` },
      { name: 'each row is one contiguous window of the text', code: `import numpy as np\ndata = np.arange(1000)\nx, y = get_batch(data, 6, 8, np.random.default_rng(3))\nfor row_x, row_y in zip(x, y):\n    s = row_x[0]\n    assert np.array_equal(row_x, np.arange(s, s + 6)) and np.array_equal(row_y, np.arange(s + 1, s + 7)), f"row {row_x} / {row_y} is not a window of data"` },
      { name: 'the last possible window is used and nothing runs off the end', code: `import numpy as np\ndata = np.arange(5)\nx, y = get_batch(data, 4, 3, np.random.default_rng(4))\nassert np.array_equal(x, np.tile(np.arange(4), (3, 1))) and np.array_equal(y, np.tile(np.arange(1, 5), (3, 1))), f"with 5 tokens and context 4 only one window fits; got x={x}, y={y}"\nx, y = get_batch(np.arange(300), 16, 400, np.random.default_rng(5))\nassert y.max() <= 299, "a window ran past the end of the data"` },
    ],
    hints: [
      'Draw the starts: `ix = rng.integers(0, len(data) - context_len, size=batch_size)`. The upper bound is exclusive, so the last window ends exactly at the final token.',
      'Build rows with a list comprehension and stack them: `np.stack([data[i:i + context_len] for i in ix])`.',
      'y is the same with `i + 1` in both places.',
    ],
    explanation: `One window of length T holds T training examples, not one: position 0 predicts position 1, positions 0 to 1 predict position 2, and so on. The causal mask is what makes all of those predictions honest at the same time.

Random starts mean every step sees a different slice of the text, which is the "stochastic" in stochastic gradient descent. The repo's \`get_batch\` does the same with \`torch.randint\` and \`torch.stack\`.`,
    source: 'phase3-transformers/tiny_gpt.py',
  },
  {
    id: 'training-gpt-code-cross-entropy',
    lesson: 'training-gpt',
    title: 'The training loss, from logits',
    prompt: `The model returns logits of shape (B, T, V); the targets are integer token ids of shape (B, T). Write \`cross_entropy(logits, targets)\`: the average, over all B times T positions, of \`-log softmax(logits)[target]\`.

Do it stably with log-sum-exp: \`log softmax(z) = z - max(z) - log(sum(exp(z - max(z))))\`. Never compute a probability first and take its log, or large logits overflow.`,
    starter: `import numpy as np

def cross_entropy(logits, targets):
    # flatten to (B*T, V) and (B*T,), log-softmax each row, pick the targets, average
    ...

logits = np.zeros((2, 3, 5))
targets = np.array([[0, 1, 2], [3, 4, 0]])
print(cross_entropy(logits, targets))   # log(5) = 1.609
`,
    solution: `import numpy as np

def cross_entropy(logits, targets):
    V = logits.shape[-1]
    z = logits.reshape(-1, V)
    t = targets.reshape(-1)
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(t)), t].mean())

logits = np.zeros((2, 3, 5))
targets = np.array([[0, 1, 2], [3, 4, 0]])
print(cross_entropy(logits, targets))   # 1.6094
`,
    tests: [
      { name: 'all-zero logits over 65 tokens give log(65), the untrained loss', code: `import numpy as np\nce = cross_entropy(np.zeros((4, 8, 65)), np.random.default_rng(0).integers(0, 65, size=(4, 8)))\nassert abs(ce - np.log(65)) < 1e-9, f"got {ce}, expected log(65) = {np.log(65):.4f}"` },
      { name: 'a hand-worked example', code: `import numpy as np\nlogits = np.array([[[2.0, 1.0, 0.0], [0.0, 0.0, 3.0]]])\ntargets = np.array([[0, 2]])\nl1 = -(2 - np.log(np.exp(2) + np.exp(1) + 1)); l2 = -(3 - np.log(2 + np.exp(3)))\nce = cross_entropy(logits, targets)\nassert abs(ce - (l1 + l2) / 2) < 1e-9, f"got {ce}, expected {(l1 + l2) / 2:.6f}"` },
      { name: 'huge logits do not overflow', code: `import numpy as np\nlogits = np.array([[[1000.0, 0.0], [0.0, 1000.0]]])\nce = cross_entropy(logits, np.array([[1, 1]]))\nassert np.isfinite(ce) and abs(ce - 500.0) < 1e-6, f"got {ce}, expected 500"` },
      { name: 'confident and right is almost 0; confident and wrong is large', code: `import numpy as np\nz = np.full((1, 1, 4), -20.0); z[0, 0, 2] = 20.0\nright = cross_entropy(z, np.array([[2]])); wrong = cross_entropy(z, np.array([[0]]))\nassert right < 1e-6, f"right answer loss {right}"\nassert wrong > 39, f"wrong answer loss {wrong}"` },
    ],
    hints: [
      'Flatten first: `z = logits.reshape(-1, V)` and `t = targets.reshape(-1)`. Now it is the same as a batch of ordinary classifications.',
      'Subtract the row max, then `log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))`.',
      'Pick each row’s target with `log_probs[np.arange(len(t)), t]`, negate, and take the mean.',
    ],
    explanation: `This is \`F.cross_entropy(logits.view(-1, V), targets.view(-1))\` from \`GPT.forward\`. Flattening works because each of the B times T positions is its own next-token question.

The first number a training run prints should be close to log(V): an untrained model spreads its bets evenly. If it is much higher, the initialisation is too confident; if the loss later goes below what the data allows, you are memorising.`,
    source: 'phase3-transformers/tiny_gpt.py',
  },
]

export default exercises
