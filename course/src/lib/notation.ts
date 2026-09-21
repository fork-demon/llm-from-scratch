// Notation decoder: the symbols that make papers look hard, each mapped to a line of code the
// learner has already written in this course. Pure data plus two small pure helpers:
// a formula markup parser (so formulas can be stored as strings and tested) and the quiz logic.
import { makeRng } from './rng'

/* ------------------------------------------------------------------ */
/* Formula markup: "W_i^Q" or "x_{<t}" -> parts with sub/superscripts   */
/* ------------------------------------------------------------------ */

export interface FormulaPart { text: string; kind: 'base' | 'sub' | 'sup' }

/**
 * `_x`, `_{...}`, `^x`, `^{...}` become sub/superscripts. Everything else is literal.
 * The text of a sub/superscript may itself contain markup (e^{z_i}); renderers parse it again.
 */
export const parseFormula = (src: string): FormulaPart[] => {
  const parts: FormulaPart[] = []
  const push = (text: string, kind: FormulaPart['kind']) => {
    if (!text) return
    const last = parts[parts.length - 1]
    if (last && last.kind === kind && kind === 'base') last.text += text
    else parts.push({ text, kind })
  }
  let i = 0
  while (i < src.length) {
    const ch = src[i]
    if ((ch === '_' || ch === '^') && i + 1 < src.length) {
      const kind = ch === '_' ? 'sub' : 'sup'
      if (src[i + 1] === '{') {
        // find the matching brace, so that e^{z_{ij}} keeps its inner braces
        let depth = 1
        let end = i + 2
        while (end < src.length && depth > 0) {
          if (src[end] === '{') depth++
          else if (src[end] === '}') depth--
          if (depth > 0) end++
        }
        if (depth > 0) { push(src.slice(i), 'base'); break }
        push(src.slice(i + 2, end), kind)
        i = end + 1
      } else {
        push(src[i + 1], kind)
        i += 2
      }
    } else {
      push(ch, 'base')
      i++
    }
  }
  return parts
}

/* ------------------------------------------------------------------ */
/* Symbols                                                              */
/* ------------------------------------------------------------------ */

export interface SymbolEntry {
  id: string
  symbol: string // formula markup
  say: string // how to read it aloud
  meaning: string
  code: string
  shapes: string
  lesson: string
}

export const SYMBOLS: SymbolEntry[] = [
  { id: 'in-r', symbol: 'X ∈ ℝ^{n×d}', say: '“X is in R n by d”', meaning: 'X is an array of real numbers (floats) with n rows and d columns. ∈ means “is a member of”; ℝ is the set of real numbers. It is a type annotation, nothing more.',
    code: 'X = np.zeros((n, d))        # X.shape == (n, d), dtype float', shapes: 'ℝ^d is a vector of d floats; ℝ^{n×d} is a matrix; ℝ^{b×n×d} adds a batch axis', lesson: 'matrices' },
  { id: 'sum', symbol: 'Σ_{i=1}^{n} a_i b_i', say: '“the sum over i from 1 to n of a i times b i”', meaning: 'A for-loop that adds. The subscript under Σ is the loop variable. This particular sum is the dot product. Papers count from 1, code from 0.',
    code: 'total = sum(a[i] * b[i] for i in range(n))    # == a @ b', shapes: 'a, b: (n,) → a single number', lesson: 'vectors' },
  { id: 'prod', symbol: 'P(x) = ∏_{t=1}^{T} P(x_t | x_{<t})', say: '“the product over t of…”', meaning: 'A for-loop that multiplies: the probability of a whole text is the product of each token’s probability given the tokens before it. In code you add logs instead, because a product of thousands of small numbers underflows to 0.',
    code: 'log_p_text = logp[range(T), tokens].sum()', shapes: 'logp: (T, V), tokens: (T,) → a single number', lesson: 'next-token' },
  { id: 'cond', symbol: 'P(x_t | x_{<t})', say: '“the probability of x t given everything before t”', meaning: 'The bar | means “given”. x_{<t} is every token before position t. This is one row of your model’s output after softmax: the next-token distribution.',
    code: 'probs = F.softmax(model(idx[:, :t])[0][:, -1, :], dim=-1)', shapes: 'idx[:, :t]: (B, t) → probs: (B, V)', lesson: 'next-token' },
  { id: 'softmax', symbol: 'softmax(z)_i = e^{z_i} / Σ_j e^{z_j}', say: '“softmax of z, component i”', meaning: 'Exponentiate every score, divide by the total. The subscript i on the left says the formula gives one output component; j is a separate loop variable for the sum.',
    code: 'e = np.exp(z - z.max()); p = e / e.sum()', shapes: 'z: (V,) → p: (V,), positive, sums to 1', lesson: 'softmax' },
  { id: 'transpose', symbol: 'QK^T', say: '“Q K transpose”', meaning: 'Two matrices written side by side are matrix-multiplied. The superscript T swaps rows and columns (it is not a power, and not the sequence length).',
    code: 'scores = Q @ K.T', shapes: 'Q: (T, d), K.T: (d, T) → (T, T)', lesson: 'matrices' },
  { id: 'hadamard', symbol: 'a ⊙ b', say: '“a element-wise times b” (Hadamard product)', meaning: 'Multiply matching entries. Same shapes in, same shape out. It is the plain * of NumPy, and it is not the matrix product. You met it in the SwiGLU gate.',
    code: 'out = silu(gate) * up          # NOT gate @ up', shapes: '(T, d) ⊙ (T, d) → (T, d)', lesson: 'modern-architecture' },
  { id: 'norm', symbol: '‖x‖', say: '“the norm of x”', meaning: 'The length of a vector: square every entry, add, take the root. With a subscript 2 it is the same thing; with a subscript 1 it is the sum of absolute values.',
    code: 'length = np.sqrt((x ** 2).sum())   # == np.linalg.norm(x)', shapes: 'x: (d,) → a single number ≥ 0', lesson: 'vectors' },
  { id: 'expect', symbol: '𝔼_{x∼D}[ℓ(x)]', say: '“the expectation over x drawn from D of the loss”', meaning: 'An average. x ∼ D means “x sampled from the dataset D”. In practice you cannot average over all data, so you average over a batch.',
    code: 'loss = losses.mean()           # losses: one per example in the batch', shapes: 'losses: (B,) → a single number', lesson: 'training-gpt' },
  { id: 'grad', symbol: '∇_θ L', say: '“the gradient of L with respect to theta”', meaning: 'For every parameter, how much the loss L changes if you nudge that parameter. It has exactly the shape of the parameters. θ (theta) is shorthand for “all the parameters”.',
    code: 'loss.backward()                # now p.grad holds ∇ for each p', shapes: 'p.grad.shape == p.shape for every parameter p', lesson: 'backprop' },
  { id: 'update', symbol: 'θ ← θ − η ∇_θ L', say: '“theta gets theta minus eta times the gradient”', meaning: 'One step of gradient descent. The arrow is assignment. η (eta) is the learning rate. Some papers write θ_{t+1} = θ_t − … instead.',
    code: 'for p in params: p -= lr * p.grad', shapes: 'every p keeps its shape', lesson: 'gradient-descent' },
  { id: 'argmax', symbol: 'argmax_i z_i', say: '“the arg max over i of z i”', meaning: 'Not the largest value: the index where it occurs. Greedy decoding is an argmax over the vocabulary.',
    code: 'next_id = logits.argmax()', shapes: 'logits: (V,) → one integer in [0, V)', lesson: 'inference' },
  { id: 'bigo', symbol: 'O(n^2 · d)', say: '“order n squared d”', meaning: 'How the work grows, ignoring constant factors. n is the sequence length, d the vector width. Attention compares every token with every token (n²) and each comparison is a dot product of d numbers.',
    code: 'scores = Q @ K.T               # n * n dot products of length d', shapes: 'Q, K: (n, d) → (n, n): doubling n quadruples the work', lesson: 'attention' },
  { id: 'indices', symbol: 'W_i^Q,  h^{(l)}', say: '“W Q for head i”, “h at layer l”', meaning: 'Sub- and superscripts are usually labels, not powers. W_i^Q is the query matrix of head i. A parenthesised superscript such as (l) is almost always a layer number.',
    code: 'Wq_i = Wq[:, i*hd:(i+1)*hd]; h = blocks[l](h)', shapes: 'W_i^Q: (d_{model}, d_k); h^{(l)}: (T, d_{model}) for every l', lesson: 'masks-and-heads' },
  { id: 'column', symbol: 'y = Wx + b', say: '“y equals W x plus b”', meaning: 'The column-vector convention used by most papers: x is a column, W has shape (out, in), and W stands on the left. This course and the Transformer paper use rows: y = xW + b with W of shape (in, out). The two are transposes of each other. nn.Linear stores the paper’s W and computes x @ W.T.',
    code: 'y = x @ W.T + b                # W.shape == (out, in), as in nn.Linear', shapes: 'x: (in,), W: (out, in), b: (out,) → y: (out,)', lesson: 'matrices' },
]

/* ------------------------------------------------------------------ */
/* Whole formulas, decoded line by line                                 */
/* ------------------------------------------------------------------ */

export interface FormulaEntry {
  id: string
  name: string
  formula: string
  from: string
  steps: { piece: string; code: string; shape: string }[]
  note: string
  lesson: string
}

export const FORMULAS: FormulaEntry[] = [
  { id: 'attention', name: 'Scaled dot-product attention', formula: 'Attention(Q, K, V) = softmax(QK^T / √d_k) V', from: 'Vaswani et al. 2017, equation 1', lesson: 'attention',
    steps: [
      { piece: 'QK^T', code: 'scores = Q @ K.T', shape: '(T, d_k) @ (d_k, T) → (T, T)' },
      { piece: '/ √d_k', code: 'scores = scores / np.sqrt(d_k)', shape: '(T, T)' },
      { piece: 'softmax( · )', code: 'weights = softmax(scores)      # along each row', shape: '(T, T), rows sum to 1' },
      { piece: '· V', code: 'out = weights @ V', shape: '(T, T) @ (T, d_v) → (T, d_v)' },
    ],
    note: 'Read a formula from the inside out. The innermost expression is the first line of code.' },
  { id: 'multihead', name: 'Multi-head attention', formula: 'MultiHead(Q, K, V) = Concat(head_1, …, head_h) W^O,  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)', from: 'Vaswani et al. 2017, section 3.2.2', lesson: 'masks-and-heads',
    steps: [
      { piece: 'QW_i^Q, KW_i^K, VW_i^V for all i', code: 'q, k, v = self.qkv(x).split(D, dim=2)', shape: '(B, T, D) each: all heads’ projections at once' },
      { piece: 'one slice per head', code: 'q = q.view(B, T, n_head, hd).transpose(1, 2)', shape: '(B, n_head, T, hd)' },
      { piece: 'head_i = Attention(…)', code: 'y = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(hd), dim=-1) @ v', shape: '(B, n_head, T, hd)' },
      { piece: 'Concat(head_1, …, head_h)', code: 'y = y.transpose(1, 2).contiguous().view(B, T, D)', shape: '(B, T, D)' },
      { piece: '· W^O', code: 'return self.proj(y)', shape: '(B, T, D)' },
    ],
    note: 'The paper’s h separate matrices W_i^Q, side by side, are your one fused matrix. (The causal mask is left out here, as it is in the paper’s formula.)' },
  { id: 'ffn', name: 'Position-wise feed-forward network', formula: 'FFN(x) = max(0, xW_1 + b_1) W_2 + b_2', from: 'Vaswani et al. 2017, equation 2', lesson: 'transformer-block',
    steps: [
      { piece: 'xW_1 + b_1', code: 'h = x @ W1 + b1', shape: '(T, 512) @ (512, 2048) → (T, 2048)' },
      { piece: 'max(0, · )', code: 'h = np.maximum(0, h)           # ReLU', shape: '(T, 2048)' },
      { piece: '· W_2 + b_2', code: 'out = h @ W2 + b2', shape: '(T, 2048) @ (2048, 512) → (T, 512)' },
    ],
    note: 'This paper uses the row convention, x on the left, the same as this course. max(0, ·) is ReLU written out.' },
  { id: 'loss', name: 'The language-modelling loss', formula: 'L(θ) = − (1/T) Σ_{t=1}^{T} log P_θ(x_t | x_{<t})', from: 'the standard form in GPT-style papers', lesson: 'next-token',
    steps: [
      { piece: 'P_θ(x_t | x_{<t}) for every t', code: 'logits, _ = model(x)           # one forward pass gives all T rows', shape: '(B, T, V)' },
      { piece: '− (1/T) Σ_t log …', code: 'loss = F.cross_entropy(logits.view(-1, V), y.view(-1))', shape: 'a single number' },
    ],
    note: 'θ as a subscript means “computed with the current parameters”. The sum over t and the 1/T are the mean that the cross-entropy function takes. y is x shifted by one position.' },
  { id: 'lora', name: 'LoRA', formula: 'h = W_0 x + ΔW x = W_0 x + BAx,   B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}', from: 'Hu et al. 2021, equation 3', lesson: 'fine-tuning',
    steps: [
      { piece: 'W_0 x', code: 'self.base(x)', shape: '(B, T, d): frozen' },
      { piece: 'Ax, then B(Ax)', code: '(x @ self.A) @ self.B', shape: '(B, T, k) → (B, T, r) → (B, T, d)' },
      { piece: 'the paper’s α / r scaling', code: '* self.scale', shape: 'a constant' },
    ],
    note: 'Column convention: the paper’s A is (r × k) and acts first because it stands next to x. Your A is (k × r) and also acts first, because in row convention x stands on the left. Same names, transposed shapes.' },
  { id: 'adam-lr', name: 'The Transformer’s learning-rate schedule', formula: 'lrate = d_{model}^{−0.5} · min(step^{−0.5}, step · warmup^{−1.5})', from: 'Vaswani et al. 2017, equation 3, with warmup_steps = 4000', lesson: 'training-gpt',
    steps: [
      { piece: 'step · warmup^{−1.5}', code: 'a = step * warmup ** -1.5      # grows linearly: the warm-up', shape: 'a number' },
      { piece: 'step^{−0.5}', code: 'b = step ** -0.5               # shrinks like 1/√step', shape: 'a number' },
      { piece: 'd_{model}^{−0.5} · min( · , · )', code: 'lr = d_model ** -0.5 * min(a, b)', shape: 'peaks at step == warmup' },
    ],
    note: 'A negative exponent is a division: x^{−0.5} = 1/√x. The two curves cross exactly at step = warmup.' },
]

/* ------------------------------------------------------------------ */
/* Quiz: given the formula, pick the code                               */
/* ------------------------------------------------------------------ */

export interface NotationQuestion { id: string; formula: string; options: string[]; answer: number; explain: string; lesson: string }

export const QUIZ: NotationQuestion[] = [
  { id: 'q-hadamard', formula: 'y = σ(xW) ⊙ (xV)', lesson: 'modern-architecture', answer: 0,
    options: ['y = sigmoid(x @ W) * (x @ V)', 'y = sigmoid(x @ W) @ (x @ V)', 'y = sigmoid(x @ W) + (x @ V)', 'y = sigmoid(x * W) @ (x * V)'],
    explain: '⊙ is the element-wise product, NumPy’s *. Letters side by side (xW) are a matrix product, @.' },
  { id: 'q-softmax-axis', formula: 'A_{ij} = exp(S_{ij}) / Σ_k exp(S_{ik})', lesson: 'attention', answer: 1,
    options: ['A = np.exp(S) / np.exp(S).sum(axis=0, keepdims=True)', 'A = np.exp(S) / np.exp(S).sum(axis=1, keepdims=True)', 'A = np.exp(S) / np.exp(S).sum(axis=None, keepdims=True)', 'A = np.exp(S) / np.exp(S).max(axis=1, keepdims=True)'],
    explain: 'The sum runs over k, which replaces j: the second index, the columns. So each row i is normalised: axis=1.' },
  { id: 'q-argmax', formula: 'x̂ = argmax_v P(v | x_{<t})', lesson: 'inference', answer: 2,
    options: ['next_id = probs.max()', 'next_id = probs.sum()', 'next_id = probs.argmax()', 'next_id = probs.argmin()'],
    explain: 'argmax returns the index of the largest entry, here a token id. max would return the probability itself.' },
  { id: 'q-column', formula: 'y = Wx + b,  W ∈ ℝ^{m×n}, x ∈ ℝ^n', lesson: 'matrices', answer: 3,
    options: ['y = x @ W + b      # x: (n,)  W: (m, n)', 'y = W.T @ x + b    # x: (n,)  W: (m, n)', 'y = x * W + b      # x: (n,)  W: (m, n)', 'y = x @ W.T + b    # x: (n,)  W: (m, n)'],
    explain: 'W is (m, n), so with x on the left you need W.T: (n,) @ (n, m) → (m,). This is what nn.Linear computes.' },
  { id: 'q-expect', formula: 'L = 𝔼_{(x,y)∼D} [ − log p_θ(y | x) ]', lesson: 'training-gpt', answer: 0,
    options: ['loss = -log_p_correct.mean()', 'loss = -log_p_correct.max()', 'loss = -log_p_correct.prod()', 'loss = -log_p_correct[0]'],
    explain: 'An expectation over the data becomes a mean over the batch.' },
  { id: 'q-norm', formula: 'x̄ = x / √( (1/d) Σ_i x_i^2 + ε )', lesson: 'modern-architecture', answer: 1,
    options: ['x_bar = x / np.sqrt((x ** 2).sum() + eps)', 'x_bar = x / np.sqrt((x ** 2).mean() + eps)', 'x_bar = x / np.sqrt((x.mean()) ** 2 + eps)', 'x_bar = x / ((x ** 2).mean() + eps)'],
    explain: '(1/d) Σ is a mean, taken of the squares, and the root covers the mean plus ε. This is RMSNorm without its gain.' },
  { id: 'q-update', formula: 'θ_{t+1} = θ_t − η ∇_θ L(θ_t)', lesson: 'gradient-descent', answer: 2,
    options: ['p += lr * p.grad', 'p -= lr / p.grad', 'p -= lr * p.grad', 'p = lr * p.grad'],
    explain: 'Step against the gradient, scaled by the learning rate η.' },
  { id: 'q-cond', formula: 'P(x_{1:T}) = ∏_{t=1}^{T} P(x_t | x_{<t})', lesson: 'next-token', answer: 3,
    options: ['log_p = np.log(p_each_token).mean()', 'log_p = np.log(p_each_token).max()', 'log_p = np.log(p_each_token.sum())', 'log_p = np.log(p_each_token).sum()'],
    explain: 'The log of a product is the sum of the logs. (The mean of the logs, negated, is the training loss: a different quantity.)' },
]

/** A fixed, per-question shuffle, so options never appear in authoring order and tests are deterministic. */
export const shuffledOptions = (q: NotationQuestion, seed = 1): { options: string[]; answer: number } => {
  let h = seed
  for (const ch of q.id) h = (h * 31 + ch.charCodeAt(0)) % 2147483647
  const rng = makeRng(h)
  const order = q.options.map((_, i) => i)
  for (let i = order.length - 1; i > 0; i--) {
    const j = rng.int(i + 1)
    ;[order[i], order[j]] = [order[j], order[i]]
  }
  return { options: order.map((i) => q.options[i]), answer: order.indexOf(q.answer) }
}

export const scoreNotationQuiz = (picked: Record<string, number>, seed = 1): number =>
  QUIZ.filter((q) => picked[q.id] === shuffledOptions(q, seed).answer).length
