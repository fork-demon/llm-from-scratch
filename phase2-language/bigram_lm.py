"""
Module 06 -- Your first language model(s).

Three models of "predict the next character", in increasing power:
  A. Count table (classical bigram statistics -- optimal for 1 char of context)
  B. Neural bigram trained by gradient descent -- converges to A. Watch it.
  C. Neural model with 3 characters of context -- beats A. More context wins.

Plus: autoregressive generation (the ChatGPT loop) and perplexity.
"""
import numpy as np

rng = np.random.default_rng(7)

CORPUS = (
    "the quick brown fox jumps over the lazy dog and the cat sleeps "
    "in the warm sun while the dog barks at the mailman who walks "
    "down the street every morning with letters for the people in "
    "the town where the children play in the park near the river "
    "that flows past the old mill and under the stone bridge to the "
    "sea where the fishermen cast their nets in the early light of "
    "dawn and sing the old songs of the water and the wind and the "
    "long summer days that fade into the quiet evenings of autumn "
) * 3


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return chars, stoi, itos


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ----------------------------------------------------------------------
# MODEL A: the count table
# ----------------------------------------------------------------------
def count_model(ids, V):
    counts = np.full((V, V), 0.01)    # tiny "smoothing" instead of 0:
                                      # unseen pairs get small prob, not zero,
                                      # so cross-entropy can't blow up to inf.
                                      # (Try 1.0 here: heavier smoothing makes
                                      # the table WORSE on seen pairs -- the
                                      # neural model will then beat it!)
    for a, b in zip(ids, ids[1:]):
        counts[a, b] += 1
    return counts / counts.sum(axis=1, keepdims=True)   # rows -> probabilities


def cross_entropy_of_table(probs_table, ids):
    """Average surprise of the corpus under a table of P(next|current)."""
    p = probs_table[ids[:-1], ids[1:]]
    return -np.log(p).mean()


# ----------------------------------------------------------------------
# MODEL B: neural bigram -- same task, learned by gradient descent
# ----------------------------------------------------------------------
def train_neural_bigram(ids, V, steps=3000, lr=0.5, batch=256, dim=24):
    E = 0.1 * rng.normal(size=(V, dim))     # embedding table
    W = 0.1 * rng.normal(size=(dim, V))     # output projection
    xs, ys = np.array(ids[:-1]), np.array(ids[1:])

    for step in range(steps + 1):
        idx = rng.integers(0, len(xs), size=batch)
        xb, yb = xs[idx], ys[idx]

        emb = E[xb]                          # forward
        logits = emb @ W
        probs = softmax(logits)
        loss = -np.log(probs[np.arange(batch), yb] + 1e-12).mean()

        d_logits = probs                     # backward (the usual rules)
        d_logits[np.arange(batch), yb] -= 1
        d_logits /= batch
        d_W = emb.T @ d_logits
        d_E = d_logits @ W.T
        W -= lr * d_W
        np.add.at(E, xb, -lr * d_E)

        if step % 500 == 0:
            yield step, loss, (E, W)


def neural_probs_table(E, W, V):
    """The neural model, evaluated at every input char -> a table like Model A's."""
    return softmax(E[np.arange(V)] @ W)


# ----------------------------------------------------------------------
# MODEL C: 3 characters of context (concatenated embeddings -> MLP head)
# ----------------------------------------------------------------------
def train_context3(ids, V, ctx=3, steps=4000, lr=0.3, batch=256, dim=16, hidden=64):
    E = 0.1 * rng.normal(size=(V, dim))
    W1 = 0.1 * rng.normal(size=(ctx * dim, hidden)); b1 = np.zeros(hidden)
    W2 = 0.1 * rng.normal(size=(hidden, V));         b2 = np.zeros(V)

    ids = np.array(ids)
    # windows: each row = ctx consecutive char ids; target = the next one
    starts = np.arange(len(ids) - ctx)
    final_loss = None

    for step in range(steps):
        idx = rng.integers(0, len(starts), size=batch)
        xb = np.stack([ids[s:s + ctx] for s in idx])    # (batch, ctx)
        yb = ids[idx + ctx]

        emb = E[xb].reshape(batch, -1)                  # concat ctx embeddings
        h = np.maximum(0, emb @ W1 + b1)                # module 03's MLP
        logits = h @ W2 + b2
        probs = softmax(logits)
        final_loss = -np.log(probs[np.arange(batch), yb] + 1e-12).mean()

        d_logits = probs
        d_logits[np.arange(batch), yb] -= 1
        d_logits /= batch
        d_W2 = h.T @ d_logits; d_b2 = d_logits.sum(0)
        d_h = (d_logits @ W2.T) * (h > 0)
        d_W1 = emb.T @ d_h;     d_b1 = d_h.sum(0)
        d_emb = (d_h @ W1.T).reshape(batch, ctx, dim)

        W2 -= lr * d_W2; b2 -= lr * d_b2
        W1 -= lr * d_W1; b1 -= lr * d_b1
        np.add.at(E, xb, -lr * d_emb)

    return final_loss


# ----------------------------------------------------------------------
# GENERATION: the autoregressive loop -- this IS how ChatGPT emits text
# ----------------------------------------------------------------------
def generate(probs_table, itos, start_id, n=200):
    out, cur = [], start_id
    for _ in range(n):
        cur = rng.choice(len(itos), p=probs_table[cur])   # roll the weighted die
        out.append(itos[cur])                             # feed output back as input
    return "".join(out)


if __name__ == "__main__":
    chars, stoi, itos = build_vocab(CORPUS)
    V = len(chars)
    ids = [stoi[c] for c in CORPUS]
    print(f"corpus: {len(CORPUS)} chars, vocab {V}\n")

    print("=" * 64)
    print("MODEL A: count-table bigram (classical statistics)")
    print("=" * 64)
    table_A = count_model(ids, V)
    loss_A = cross_entropy_of_table(table_A, ids)
    print(f"  cross-entropy: {loss_A:.4f}   perplexity: {np.exp(loss_A):.2f}")
    print(f"  sample: {generate(table_A, itos, stoi['t'])!r}\n")

    print("=" * 64)
    print("MODEL B: neural bigram -- watch it CONVERGE TO MODEL A's loss")
    print("=" * 64)
    print(f"  (target: {loss_A:.4f})")
    for step, loss, (E, W) in train_neural_bigram(ids, V):
        print(f"  step {step:5d}  loss {loss:.4f}")
    table_B = neural_probs_table(E, W, V)
    loss_B = cross_entropy_of_table(table_B, ids)
    print(f"  final cross-entropy {loss_B:.4f} vs count-table {loss_A:.4f}")
    print("  -> gradient descent rediscovered the count statistics.")
    print(f"  sample: {generate(table_B, itos, stoi['t'])!r}\n")

    print("=" * 64)
    print("MODEL C: 3 chars of context -- breaking the bigram ceiling")
    print("=" * 64)
    loss_C = train_context3(ids, V)
    print(f"  context-3 loss {loss_C:.4f}  vs bigram optimum {loss_A:.4f}")
    print("  -> more context beats any cleverness with less context.")
    print("     (Attention, module 07, is 'more context' done right.)")
