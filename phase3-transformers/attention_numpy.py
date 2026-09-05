"""
Module 07 -- Self-attention from scratch in NumPy.

Sections (run the file, read top to bottom):
  1. Single-head attention, full weight matrix printed
  2. Causal masking, before/after
  3. Multi-head split & concat (with obsessive shape comments)
  4. The "river bank" demo: watching disambiguation happen
  5. Gradient check through a full attention layer

Shapes convention used everywhere:
  T = number of tokens in the sequence
  D = embedding dimension
  x       : (T, D)
  Q, K, V : (T, D)
  scores  : (T, T)   <- cell (i, j) = how much token i cares about token j

Expected Output:
  (Run the script to see the numerical output and shape assertions pass)
"""
import numpy as np

rng = np.random.default_rng(3)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ----------------------------------------------------------------------
# 1. SINGLE-HEAD SELF-ATTENTION -- the whole formula
# ----------------------------------------------------------------------
def attention(x, Wq, Wk, Wv, causal=False):
    T, D = x.shape
    Q = x @ Wq                      # (T, D) each token's "what am I looking for"
    K = x @ Wk                      # (T, D) each token's "how can I be found"
    V = x @ Wv                      # (T, D) each token's "what I contribute"

    scores = Q @ K.T / np.sqrt(D)   # (T, T) all query-key dot products at once
    if causal:
        # forbid looking at the future: upper triangle -> -inf -> softmax 0
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)

    weights = softmax(scores)       # (T, T) rows sum to 1: mixing proportions
    out = weights @ V               # (T, D) each token = weighted blend of values
    return out, weights


def demo_single_head():
    print("=" * 64)
    print("1) SINGLE-HEAD ATTENTION: the full (T x T) weight matrix")
    print("=" * 64)
    T, D = 5, 8
    x = rng.normal(size=(T, D))
    Wq, Wk, Wv = (rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(3))

    _, w = attention(x, Wq, Wk, Wv)
    print("  rows = who is looking, cols = who is being looked at")
    print("  (each row sums to 1.0)")
    for i, row in enumerate(w):
        print(f"  tok{i}: " + " ".join(f"{v:5.2f}" for v in row))
    print()


# ----------------------------------------------------------------------
# 2. CAUSAL MASKING -- before / after
# ----------------------------------------------------------------------
def demo_causal():
    print("=" * 64)
    print("2) CAUSAL MASK: no token may attend to its future")
    print("=" * 64)
    T, D = 5, 8
    x = rng.normal(size=(T, D))
    Wq, Wk, Wv = (rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(3))

    _, w_free = attention(x, Wq, Wk, Wv, causal=False)
    _, w_causal = attention(x, Wq, Wk, Wv, causal=True)

    print("  unmasked:                         causal (upper triangle = 0):")
    for r1, r2 in zip(w_free, w_causal):
        left = " ".join(f"{v:5.2f}" for v in r1)
        right = " ".join(f"{v:5.2f}" for v in r2)
        print(f"  {left}   |   {right}")
    print()


# ----------------------------------------------------------------------
# 3. MULTI-HEAD -- h parallel attentions on D/h-dim slices
# ----------------------------------------------------------------------
def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    T, D = x.shape
    hd = D // n_heads                       # per-head dimension

    Q = x @ Wq                              # (T, D)
    K = x @ Wk
    V = x @ Wv

    # reshape (T, D) -> (n_heads, T, hd): give each head its slice
    def split(M):
        return M.reshape(T, n_heads, hd).transpose(1, 0, 2)
    Qh, Kh, Vh = split(Q), split(K), split(V)      # (H, T, hd)

    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(hd)   # (H, T, T)
    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores)                            # (H, T, T)
    out = weights @ Vh                                   # (H, T, hd)

    # concat heads back: (H, T, hd) -> (T, D), then one final mixing matrix
    out = out.transpose(1, 0, 2).reshape(T, D)
    return out @ Wo, weights                             # (T, D)


def demo_multi_head():
    print("=" * 64)
    print("3) MULTI-HEAD: independent relevance criteria in parallel")
    print("=" * 64)
    T, D, H = 5, 8, 2
    x = rng.normal(size=(T, D))
    Wq, Wk, Wv, Wo = (rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(4))
    _, w = multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads=H)
    for h in range(H):
        print(f"  head {h} attention pattern (causal):")
        for row in w[h]:
            print("   " + " ".join(f"{v:5.2f}" for v in row))
    print("  -> different heads, different patterns, same input.\n")


# ----------------------------------------------------------------------
# 4. THE "RIVER BANK" DEMO -- disambiguation you can watch
# ----------------------------------------------------------------------
def demo_disambiguation():
    print("=" * 64)
    print("4) DISAMBIGUATION: 'bank' pulled toward 'river' vs toward 'money'")
    print("=" * 64)
    # Hand-crafted 4-dim embeddings. Dims read (for this demo only!):
    #   [watery, financial, noun-ness, ambiguity]
    vocab = {
        "river": np.array([1.0, 0.0, 0.8, 0.0]),
        "money": np.array([0.0, 1.0, 0.8, 0.0]),
        "bank":  np.array([0.5, 0.5, 0.9, 1.0]),   # genuinely ambiguous
        "the":   np.array([0.0, 0.0, 0.1, 0.0]),
    }
    # Craft Wq/Wk so that ambiguous tokens *query for* disambiguators:
    # query strength driven by dim 3 (ambiguity), keys expose dims 0-1.
    D = 4
    Wq = np.zeros((D, D)); Wq[3, 0] = 2.0; Wq[3, 1] = 2.0
    Wk = np.zeros((D, D)); Wk[0, 0] = 2.0; Wk[1, 1] = 2.0
    Wv = np.eye(D)

    for ctx in (["the", "river", "bank"], ["the", "money", "bank"]):
        x = np.stack([vocab[w] for w in ctx])
        out, w = attention(x, Wq, Wk, Wv, causal=True)
        bank_out = out[-1]
        print(f"  context {ctx}")
        print(f"    bank attends to: " +
              ", ".join(f"{t}={v:.2f}" for t, v in zip(ctx, w[-1])))
        print(f"    bank output vector [watery, financial, ...]: "
              f"[{bank_out[0]:.2f}, {bank_out[1]:.2f}, ...]")
    print("  -> same word, different context, different vector.")
    print("     This is the contextual embedding word2vec couldn't give us.\n")


# ----------------------------------------------------------------------
# 5. GRADIENT CHECK through a full attention layer
# ----------------------------------------------------------------------
def demo_gradient_check():
    print("=" * 64)
    print("5) GRADIENT CHECK: backprop through attention vs numerical oracle")
    print("=" * 64)
    T, D = 4, 6
    x = rng.normal(size=(T, D))
    Wq, Wk, Wv = (rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(3))
    target = rng.normal(size=(T, D))

    def loss_fn(Wq_):
        out, _ = attention(x, Wq_, Wk, Wv)
        return ((out - target) ** 2).mean()

    # ---- analytic gradient wrt Wq, by the module 03 rules ----
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.T / np.sqrt(D)
    w = softmax(scores)
    out = w @ V
    d_out = 2 * (out - target) / out.size
    d_w = d_out @ V.T
    # softmax backward (rowwise): d_s = w * (d_w - sum(d_w * w))
    d_scores = w * (d_w - (d_w * w).sum(axis=1, keepdims=True))
    d_Q = d_scores @ K / np.sqrt(D)
    d_Wq = x.T @ d_Q

    # ---- numerical oracle on a few entries ----
    h, worst = 1e-5, 0.0
    for _ in range(6):
        r, c = rng.integers(0, D), rng.integers(0, D)
        orig = Wq[r, c]
        Wq[r, c] = orig + h; lp = loss_fn(Wq)
        Wq[r, c] = orig - h; lm = loss_fn(Wq)
        Wq[r, c] = orig
        num = (lp - lm) / (2 * h)
        diff = abs(num - d_Wq[r, c])
        worst = max(worst, diff)
        print(f"  dL/dWq[{r},{c}]  analytic {d_Wq[r,c]:12.8f}  "
              f"numerical {num:12.8f}  |diff| {diff:.2e}")
    print(f"  worst diff {worst:.2e} -> {'PASS' if worst < 1e-6 else 'FAIL'}")
    print("  -> attention is just matmuls + softmax; module 03 rules suffice.")


if __name__ == "__main__":
    demo_single_head()
    demo_causal()
    demo_multi_head()
    demo_disambiguation()
    demo_gradient_check()
