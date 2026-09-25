import type { CodeExerciseDef } from './types'

// Around the model: the KV cache, LoRA, quantization, RAG chunking and eval statistics.
// Each is the NumPy core of a real repo file, reduced to the part the lesson teaches.
const exercises: CodeExerciseDef[] = [
  /* ---------- inference ---------- */
  {
    id: 'inference-code-kv-cache',
    lesson: 'inference',
    title: 'A KV cache that gives the same answer',
    prompt: `\`full_attention(X, Wq, Wk, Wv)\` is given: causal single-head attention over the whole sequence, recomputed from scratch every time.

Write \`cached_step(x, cache, Wq, Wk, Wv)\` for ONE new token \`x\` (shape D). Compute its q, k and v, append k and v as new rows of \`cache["K"]\` and \`cache["V"]\`, then attend from q over everything in the cache. Return the output vector for this token (shape D).

No mask is needed: the cache only contains the past and the token itself. Feeding tokens one at a time must give exactly the rows of \`full_attention\`.`,
    starter: `import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def full_attention(X, Wq, Wk, Wv):
    T, D = X.shape
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    s = Q @ K.T / np.sqrt(D)
    s = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf, s)
    return softmax(s) @ V

def cached_step(x, cache, Wq, Wk, Wv):
    # q, k, v for this token; grow the cache by one row; attend over the cache
    ...

D = 4
cache = {"K": np.zeros((0, D)), "V": np.zeros((0, D))}
`,
    solution: `import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def full_attention(X, Wq, Wk, Wv):
    T, D = X.shape
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    s = Q @ K.T / np.sqrt(D)
    s = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf, s)
    return softmax(s) @ V

def cached_step(x, cache, Wq, Wk, Wv):
    D = x.shape[0]
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    cache["K"] = np.vstack([cache["K"], k])    # old rows never change
    cache["V"] = np.vstack([cache["V"], v])
    scores = cache["K"] @ q / np.sqrt(D)       # one score per cached token
    return softmax(scores) @ cache["V"]

D = 4
cache = {"K": np.zeros((0, D)), "V": np.zeros((0, D))}
`,
    tests: [
      { name: 'the first token’s output is its own v', code: `import numpy as np\nrng = np.random.default_rng(0)\nD = 4; Wq, Wk, Wv = (rng.normal(size=(D, D)) for _ in range(3))\nx = rng.normal(size=D)\nc = {"K": np.zeros((0, D)), "V": np.zeros((0, D))}\nout = cached_step(x, c, Wq, Wk, Wv)\nassert np.allclose(out, x @ Wv), f"with only one token in the cache, attention must return its value vector; got {out}"` },
      { name: 'the cache grows by one row per token', code: `import numpy as np\nrng = np.random.default_rng(1)\nD = 3; W = [rng.normal(size=(D, D)) for _ in range(3)]\nc = {"K": np.zeros((0, D)), "V": np.zeros((0, D))}\nfor t in range(5):\n    cached_step(rng.normal(size=D), c, *W)\n    assert c["K"].shape == (t + 1, D) and c["V"].shape == (t + 1, D), f"after {t + 1} tokens the cache is {c['K'].shape}"` },
      { name: 'token by token equals full attention, row for row', code: `import numpy as np\nrng = np.random.default_rng(2)\nT, D = 7, 6\nX = rng.normal(size=(T, D)); W = [rng.normal(size=(D, D)) for _ in range(3)]\nfull = full_attention(X, *W)\nc = {"K": np.zeros((0, D)), "V": np.zeros((0, D))}\nfor t in range(T):\n    out = cached_step(X[t], c, *W)\n    assert np.allclose(out, full[t]), f"position {t}: cached {out} vs full {full[t]}"` },
      { name: 'old cache rows are never rewritten', code: `import numpy as np\nrng = np.random.default_rng(3)\nD = 4; W = [rng.normal(size=(D, D)) for _ in range(3)]\nc = {"K": np.zeros((0, D)), "V": np.zeros((0, D))}\ncached_step(rng.normal(size=D), c, *W); cached_step(rng.normal(size=D), c, *W)\nsnapshot = c["K"].copy()\ncached_step(rng.normal(size=D), c, *W)\nassert np.array_equal(c["K"][:2], snapshot), "the first two K rows changed when a third token arrived"` },
    ],
    hints: [
      'Project the one token: `q, k, v = x @ Wq, x @ Wk, x @ Wv`. Each is shape (D,).',
      'Append with `cache["K"] = np.vstack([cache["K"], k])`, and the same for V. Assign back into the dict so the caller sees the bigger cache.',
      'Scores are `cache["K"] @ q / np.sqrt(D)`: one per cached token. Then `softmax(scores) @ cache["V"]`.',
    ],
    explanation: `Because of the causal mask, a token's k and v never depend on anything after it, so once computed they never change. The cache just remembers them, and each new token costs one row of work instead of redoing the whole sequence.

That turns generation from quadratic to linear in the number of projections, at the price of memory: K and V for every layer, head and past token. That memory is what serving systems spend most of their effort managing. \`kv_cache_demo.py\` proves the two paths identical in the same way your test does.`,
    source: 'phase3-transformers/kv_cache_demo.py',
  },

  /* ---------- fine-tuning ---------- */
  {
    id: 'fine-tuning-code-lora',
    lesson: 'fine-tuning',
    title: 'A LoRA layer and what it saves',
    prompt: `LoRA freezes a weight matrix W (d_in, d_out) and learns a small correction \`A @ B\` beside it, with A (d_in, r) and B (r, d_out).

- \`lora_init(d_in, d_out, r, rng)\` returns \`(A, B)\`: A small random (\`rng.normal(size=(d_in, r)) * 0.01\`), B all zeros.
- \`lora_forward(x, W, A, B, alpha, r)\` returns \`x @ W + (x @ A) @ B * (alpha / r)\`.
- \`trainable_params(d_in, d_out, r)\` returns how many numbers LoRA trains: the entries of A and B.`,
    starter: `import numpy as np

def lora_init(d_in, d_out, r, rng):
    ...

def lora_forward(x, W, A, B, alpha, r):
    ...

def trainable_params(d_in, d_out, r):
    ...

print(trainable_params(128, 384, 4), "trained, instead of", 128 * 384)
`,
    solution: `import numpy as np

def lora_init(d_in, d_out, r, rng):
    A = rng.normal(size=(d_in, r)) * 0.01
    B = np.zeros((r, d_out))          # zeros: the model starts out unchanged
    return A, B

def lora_forward(x, W, A, B, alpha, r):
    return x @ W + (x @ A) @ B * (alpha / r)

def trainable_params(d_in, d_out, r):
    return d_in * r + r * d_out

print(trainable_params(128, 384, 4), "trained, instead of", 128 * 384)
`,
    tests: [
      { name: 'at initialisation the layer is exactly the frozen layer', code: `import numpy as np\nrng = np.random.default_rng(0)\nW = rng.normal(size=(16, 8)); x = rng.normal(size=(3, 16))\nA, B = lora_init(16, 8, 4, rng)\nassert A.shape == (16, 4) and B.shape == (4, 8), f"A {A.shape}, B {B.shape}"\nassert np.allclose(lora_forward(x, W, A, B, 8, 4), x @ W), "with B = 0 the output must equal x @ W"` },
      { name: 'the correction is scaled by alpha / r', code: `import numpy as np\nx = np.array([[1.0, 2.0]]); W = np.zeros((2, 2))\nA = np.array([[1.0], [1.0]]); B = np.array([[1.0, -1.0]])\nout = lora_forward(x, W, A, B, alpha=8, r=1)\nassert np.allclose(out, [[24.0, -24.0]]), f"got {out}; (x @ A) @ B is [[3, -3]] and alpha / r is 8"` },
      { name: 'the parameter count: 2,048 instead of 49,152 for a 128 by 384 layer at r = 4', code: `n = trainable_params(128, 384, 4)\nassert n == 2048, f"got {n}; A has 128 * 4 entries and B has 4 * 384"` },
      { name: 'a full-rank update could not be expressed, but a rank-r one can', code: `import numpy as np\nrng = np.random.default_rng(1)\nW = rng.normal(size=(6, 5)); x = np.eye(6)\nA = rng.normal(size=(6, 2)); B = rng.normal(size=(2, 5))\ndelta = lora_forward(x, W, A, B, 2, 2) - x @ W\nassert np.linalg.matrix_rank(delta) == 2, f"the change to the layer should have rank r = 2, got rank {np.linalg.matrix_rank(delta)}"` },
    ],
    hints: [
      '`lora_init`: `return rng.normal(size=(d_in, r)) * 0.01, np.zeros((r, d_out))`.',
      '`lora_forward`: compute `x @ A` first (shape (n, r)), then `@ B`, then multiply by `alpha / r`, and add `x @ W`.',
      'A has `d_in * r` entries and B has `r * d_out`.',
    ],
    explanation: `B starts at zero, so on step 0 the fine-tuned model is exactly the pretrained one; training can only move away from it gradually. A cannot also be zero, or neither matrix would ever receive a gradient.

The count is the point: r times (d_in + d_out) grows with the sum of the sides, full fine-tuning with their product. The rank test shows the trade: LoRA can only change the layer in r directions, which is plenty for teaching a style or format and is also why it forgets less. \`LoRALinear\` in the repo is these same lines in PyTorch.`,
    source: 'phase4-modern-llms/finetune_tiny_gpt.py',
  },

  /* ---------- making-models-cheaper ---------- */
  {
    id: 'making-models-cheaper-code-int8',
    lesson: 'making-models-cheaper',
    title: 'Per-row int8 quantization',
    prompt: `Store each row of a weight matrix as 8-bit integers plus one float scale.

\`quantize_rows(W)\` returns \`(q, scale)\`: \`scale\` has shape (rows, 1) and is each row's largest absolute value divided by 127; \`q = round(W / scale)\` clipped to [-127, 127] and stored as \`np.int8\`. A row of all zeros gets scale 1.0 so nothing divides by zero.

\`dequantize(q, scale)\` returns the float approximation \`q * scale\`.`,
    starter: `import numpy as np

def quantize_rows(W):
    ...

def dequantize(q, scale):
    ...

W = np.array([[0.5, -1.0, 0.25], [10.0, 3.0, -7.0]])
print(quantize_rows(W))
`,
    solution: `import numpy as np

def quantize_rows(W):
    scale = np.abs(W).max(axis=1, keepdims=True) / 127
    scale[scale == 0] = 1.0
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q, scale

def dequantize(q, scale):
    return q.astype(np.float64) * scale

W = np.array([[0.5, -1.0, 0.25], [10.0, 3.0, -7.0]])
q, s = quantize_rows(W)
print(q)
print(np.abs(dequantize(q, s) - W).max())
`,
    tests: [
      { name: 'q is int8 and each row’s largest weight maps to ±127', code: `import numpy as np\nW = np.array([[0.5, -1.0, 0.25], [10.0, 3.0, -7.0]])\nq, s = quantize_rows(W)\nassert q.dtype == np.int8, f"dtype is {q.dtype}"\nassert s.shape == (2, 1), f"scale shape {s.shape}, expected (2, 1)"\nassert q[0, 1] == -127 and q[1, 0] == 127, f"got {q}"` },
      { name: 'the round-trip error is at most half a step in every row', code: `import numpy as np\nW = np.random.default_rng(0).normal(size=(8, 64)) * np.array([[0.01], [0.1], [1], [10], [0.5], [2], [0.02], [5]])\nq, s = quantize_rows(W)\nerr = np.abs(dequantize(q, s) - W)\nassert (err <= s / 2 + 1e-12).all(), f"worst error {err.max():.3g} is more than half a step"` },
      { name: 'an all-zero row does not produce NaN', code: `import numpy as np\nW = np.array([[0.0, 0.0], [1.0, -2.0]])\nq, s = quantize_rows(W)\nback = dequantize(q, s)\nassert np.isfinite(back).all() and np.allclose(back[0], 0), f"got {back}"` },
      { name: 'per-row scales beat one scale for the whole matrix', code: `import numpy as np\nrng = np.random.default_rng(1)\nW = rng.normal(size=(4, 32)) * np.array([[0.01], [0.01], [0.01], [50.0]])   # one loud row\nq, s = quantize_rows(W)\nt = np.abs(W).max() / 127\ntensor_err_small = np.abs(np.round(W[:3] / t) * t - W[:3]).mean()\nrow_err_small = np.abs(dequantize(q, s)[:3] - W[:3]).mean()\nassert row_err_small < tensor_err_small / 100, f"quiet rows: per-row error {row_err_small:.2e}, per-tensor {tensor_err_small:.2e}"` },
    ],
    hints: [
      '`np.abs(W).max(axis=1, keepdims=True)` gives each row’s largest magnitude with shape (rows, 1). Divide by 127.',
      'Fix zero rows with `scale[scale == 0] = 1.0` before dividing.',
      '`q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)`; `dequantize` is `q * scale`.',
    ],
    explanation: `Eight bits instead of thirty-two is a 4x smaller model, and a smaller model is a faster one, because generating a token is mostly waiting for weights to arrive from memory.

Rounding to the nearest step means no weight is off by more than half a step, and the step is set by the largest weight in its group. One loud row with a shared scale would flatten every quiet row to zero; giving each row its own scale is the cheapest fix. Smaller groups (int4 with a scale per 64 weights) push the same idea further, as \`quantize_demo.py\` measures.`,
    source: 'phase6-engineering/quantize_demo.py',
  },

  /* ---------- rag ---------- */
  {
    id: 'rag-code-chunk',
    lesson: 'rag',
    title: 'Chunk a document with overlap',
    prompt: `Write \`chunk(text, size=2, overlap=1)\`. Split the text into sentences on \`"."\` (strip each one, drop empty ones, and put the full stop back). Then group them into chunks of \`size\` sentences, where each chunk starts \`size - overlap\` sentences after the previous one (at least 1).

Stop as soon as a chunk reaches the last sentence, so the final chunk is never just a tail that the previous chunk already contains. Return the chunks as strings, sentences joined with a single space.`,
    starter: `def chunk(text, size=2, overlap=1):
    ...

doc = "Cats sleep a lot. Dogs bark. Birds sing. Fish swim."
print(chunk(doc))
`,
    solution: `def chunk(text, size=2, overlap=1):
    sents = [s.strip() + "." for s in text.split(".") if s.strip()]
    step = max(1, size - overlap)
    chunks = []
    i = 0
    while True:
        chunks.append(" ".join(sents[i:i + size]))
        if i + size >= len(sents):
            break                      # this chunk already reaches the end
        i += step
    return chunks

doc = "Cats sleep a lot. Dogs bark. Birds sing. Fish swim."
for c in chunk(doc):
    print(c)
`,
    tests: [
      { name: 'the lesson example: 4 sentences, size 2, overlap 1 gives 3 chunks', code: `r = chunk("Cats sleep a lot. Dogs bark. Birds sing. Fish swim.")\nassert r == ["Cats sleep a lot. Dogs bark.", "Dogs bark. Birds sing.", "Birds sing. Fish swim."], f"got {r}"` },
      { name: 'overlap 0 splits cleanly with no repeats', code: `r = chunk("A. B. C. D. E.", size=2, overlap=0)\nassert r == ["A. B.", "C. D.", "E."], f"got {r}"` },
      { name: 'no redundant tail chunk', code: `r = chunk("A. B. C.", size=2, overlap=1)\nassert r == ["A. B.", "B. C."], f"got {r}; a third chunk 'C.' would repeat what 'B. C.' already holds"` },
      { name: 'overlap as big as the chunk still moves forward, and every sentence is covered', code: `text = " ".join(f"S{i}." for i in range(9))\nr = chunk(text, size=3, overlap=3)\nassert len(r) < 20, f"{len(r)} chunks: the step must be at least 1"\nfor i in range(9):\n    assert any(f"S{i}." in c for c in r), f"sentence S{i} is in no chunk"` },
      { name: 'a document shorter than one chunk is a single chunk', code: `r = chunk("Only one sentence here.", size=3, overlap=1)\nassert r == ["Only one sentence here."], f"got {r}"` },
    ],
    hints: [
      'Sentences: `[s.strip() + "." for s in text.split(".") if s.strip()]`. The step is `max(1, size - overlap)`.',
      'Loop with an index `i`: take `sents[i:i + size]`, join with spaces, append.',
      'Break right after appending if `i + size >= len(sents)`; otherwise `i += step`.',
    ],
    explanation: `Chunks are what the retriever returns, so their boundaries decide what the model gets to read. Overlap means a fact that straddles a boundary still appears whole in at least one chunk.

The stopping rule is a small improvement on \`chunk\` in \`mini_rag.py\`, which keeps going while \`i\` is inside the document and so can emit a last chunk that is already inside the one before it. Duplicate chunks waste slots in the top-k and crowd out useful context.`,
    source: 'phase4-modern-llms/mini_rag.py',
  },

  /* ---------- evals ---------- */
  {
    id: 'evals-code-bootstrap',
    lesson: 'evals',
    title: 'A bootstrap confidence interval',
    prompt: `Your eval scored each golden-set item 1 (pass) or 0 (fail). Accuracy is the mean, but how sure are you?

Write \`bootstrap_ci(values, resamples=4000, seed=0)\`: with \`rng = np.random.default_rng(seed)\`, draw \`resamples\` new samples of the same size, WITH replacement, from \`values\`; compute each sample's mean; return the 2.5th and 97.5th percentiles of those means as a tuple \`(low, high)\`.

Tip: \`rng.integers(0, n, size=(resamples, n))\` draws every resample's item indices at once.`,
    starter: `import numpy as np

def bootstrap_ci(values, resamples=4000, seed=0):
    ...

scores = [1] * 14 + [0] * 6       # 70% on 20 items
print(bootstrap_ci(scores))
`,
    solution: `import numpy as np

def bootstrap_ci(values, resamples=4000, seed=0):
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))   # which items each resample picks
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)

scores = [1] * 14 + [0] * 6       # 70% on 20 items
print(bootstrap_ci(scores))       # roughly (0.5, 0.9)
`,
    tests: [
      { name: 'the same seed gives the same interval', code: `v = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]\na = bootstrap_ci(v, seed=3); b = bootstrap_ci(v, seed=3)\nassert a == b, f"{a} vs {b}: use np.random.default_rng(seed)"` },
      { name: 'the interval contains the observed accuracy', code: `v = [1] * 14 + [0] * 6\nlo, hi = bootstrap_ci(v)\nassert lo < 0.7 < hi, f"interval ({lo:.3f}, {hi:.3f}) does not contain 0.7"` },
      { name: 'if every item passes there is no spread to resample', code: `lo, hi = bootstrap_ci([1] * 12)\nassert lo == 1.0 and hi == 1.0, f"got ({lo}, {hi})"` },
      { name: '20 items give a wide interval: about ±0.2', code: `lo, hi = bootstrap_ci([1] * 14 + [0] * 6)\nassert 0.3 < hi - lo < 0.5, f"width {hi - lo:.3f}; with 20 items the interval should be roughly 0.4 wide"` },
      { name: 'five times the data makes it about half as wide', code: `small = bootstrap_ci([1] * 14 + [0] * 6)\nbig = bootstrap_ci([1] * 70 + [0] * 30)\nratio = (big[1] - big[0]) / (small[1] - small[0])\nassert 0.35 < ratio < 0.6, f"width ratio {ratio:.2f}; width shrinks like 1 / sqrt(n), so expect about 0.45"` },
    ],
    hints: [
      'Convert first: `values = np.asarray(values, dtype=float)` so fancy indexing works, and make the generator with `np.random.default_rng(seed)`.',
      '`idx = rng.integers(0, n, size=(resamples, n))`, then `values[idx]` has shape (resamples, n). Its `.mean(axis=1)` is one mean per resample.',
      '`np.percentile(means, [2.5, 97.5])` gives the two ends.',
    ],
    explanation: `Resampling the golden set with replacement simulates "what if I had drawn a different 20 questions?" The spread of the resampled accuracies is how much your number would wobble, with no formula needed.

The width shrinks like one over the square root of the number of items, so going from 20 to 100 items only halves it. A 5-point improvement on a 20-item set is noise. \`eval_harness.py\` uses the same idea, with its own seeded generator so the browser and the terminal print identical numbers.`,
    source: 'phase6-engineering/eval_harness.py',
  },
]

export default exercises
