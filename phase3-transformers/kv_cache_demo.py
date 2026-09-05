"""
Module 09 -- KV cache and sampling, from scratch in NumPy.

  1. A minimal causal transformer (random weights -- we measure mechanics)
  2. generate_naive():  recompute the whole prefix every token
  3. generate_cached(): run only the new token; read old K/V from cache
  4. Proof of equivalence + timing table (watch the speedup grow with T)
  5. Sampling: temperature / top-k / top-p, visualized on a toy distribution

Expected Output:
  (Run the script to see the numerical output and shape assertions pass)
"""
import time

import numpy as np

rng = np.random.default_rng(9)

# ---- tiny model config ----
V, D, H, LAYERS = 50, 64, 4, 2
HD = D // H


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def make_params():
    p = {"emb": 0.1 * rng.normal(size=(V, D)),
         "pos": 0.1 * rng.normal(size=(512, D)),
         "out": 0.1 * rng.normal(size=(D, V))}
    for l in range(LAYERS):
        for name in ("q", "k", "v", "o"):
            p[f"W{name}{l}"] = rng.normal(size=(D, D)) / np.sqrt(D)
        p[f"ff1_{l}"] = rng.normal(size=(D, 4 * D)) / np.sqrt(D)
        p[f"ff2_{l}"] = rng.normal(size=(4 * D, D)) / np.sqrt(4 * D)
    return p


def heads(M, T):        # (T, D) -> (H, T, hd)
    return M.reshape(T, H, HD).transpose(1, 0, 2)


def unheads(M, T):      # (H, T, hd) -> (T, D)
    return M.transpose(1, 0, 2).reshape(T, D)


# ----------------------------------------------------------------------
# Forward pass over a whole sequence (used by the naive loop)
# ----------------------------------------------------------------------
def forward_full(p, ids):
    T = len(ids)
    x = p["emb"][ids] + p["pos"][:T]
    for l in range(LAYERS):
        Q = heads(x @ p[f"Wq{l}"], T)
        K = heads(x @ p[f"Wk{l}"], T)
        Vv = heads(x @ p[f"Wv{l}"], T)
        s = Q @ K.transpose(0, 2, 1) / np.sqrt(HD)
        s = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -1e9, s)  # causal
        x = x + unheads(softmax(s) @ Vv, T) @ p[f"Wo{l}"]
        x = x + np.maximum(0, x @ p[f"ff1_{l}"]) @ p[f"ff2_{l}"]
    return x[-1] @ p["out"]          # logits for the LAST position only


# ----------------------------------------------------------------------
# Forward pass for ONE new token, reading/updating a KV cache
# ----------------------------------------------------------------------
def forward_step(p, token_id, pos_idx, cache):
    x = (p["emb"][token_id] + p["pos"][pos_idx])[None, :]     # (1, D)
    for l in range(LAYERS):
        q = heads(x @ p[f"Wq{l}"], 1)                          # (H, 1, hd)
        k = heads(x @ p[f"Wk{l}"], 1)
        v = heads(x @ p[f"Wv{l}"], 1)

        # append this token's K,V to the layer cache. Old K,V never change
        # (causal mask means nothing attends forward) -- that's the whole trick.
        cache[l]["K"] = np.concatenate([cache[l]["K"], k], axis=1)
        cache[l]["V"] = np.concatenate([cache[l]["V"], v], axis=1)
        Kc, Vc = cache[l]["K"], cache[l]["V"]                  # (H, T_so_far, hd)

        s = q @ Kc.transpose(0, 2, 1) / np.sqrt(HD)            # (H, 1, T_so_far)
        # no mask needed: the cache only CONTAINS the past.
        x = x + unheads(softmax(s) @ Vc, 1) @ p[f"Wo{l}"]
        x = x + np.maximum(0, x @ p[f"ff1_{l}"]) @ p[f"ff2_{l}"]
    return (x[0] @ p["out"])         # logits, (V,)


def empty_cache():
    return [{"K": np.zeros((H, 0, HD)), "V": np.zeros((H, 0, HD))}
            for _ in range(LAYERS)]


# ----------------------------------------------------------------------
# The two generation loops
# ----------------------------------------------------------------------
def generate_naive(p, prompt, n):
    ids = list(prompt)
    for _ in range(n):
        logits = forward_full(p, ids)          # reprocess EVERYTHING. O(T^2) work/token
        ids.append(int(np.argmax(logits)))     # greedy, for determinism
    return ids


def generate_cached(p, prompt, n):
    cache = empty_cache()
    logits = None
    for i, t in enumerate(prompt):             # "prefill": build cache from prompt
        logits = forward_step(p, t, i, cache)
    ids = list(prompt)
    for _ in range(n):                         # "decode": one token per step
        ids.append(int(np.argmax(logits)))
        logits = forward_step(p, ids[-1], len(ids) - 1, cache)
    return ids


# ----------------------------------------------------------------------
# Sampling policies (decoding is OUTSIDE the network)
# ----------------------------------------------------------------------
def sample_demo():
    print("=" * 64)
    print("SAMPLING: same logits, different policies")
    print("=" * 64)
    labels = ["the", "a", "cat", "dog", "runs", "zx@!", "qq9", "###"]
    logits = np.array([3.0, 2.5, 2.0, 1.8, 1.0, -1.0, -1.5, -2.0])

    def show(name, probs):
        bars = "  ".join(f"{l}:{p:.2f}" for l, p in zip(labels, probs))
        print(f"  {name:<22} {bars}")

    show("temperature 1.0", softmax(logits))
    show("temperature 0.3 (sharp)", softmax(logits / 0.3))
    show("temperature 3.0 (flat)", softmax(logits / 3.0))

    # top-k: keep k best, renormalize -- the garbage tail gets exactly 0
    k = 4
    p = softmax(logits).copy()
    cutoff = np.sort(p)[-k]
    p[p < cutoff] = 0
    show(f"top-k (k={k})", p / p.sum())

    # top-p: keep smallest set covering 90% of the mass
    probs = softmax(logits)
    order = np.argsort(-probs)
    csum = np.cumsum(probs[order])
    keep = order[:np.searchsorted(csum, 0.9) + 1]
    p = np.zeros_like(probs)
    p[keep] = probs[keep]
    show("top-p (p=0.9)", p / p.sum())
    print("  note the junk tokens: tiny probability each, but 50k of them")
    print("  adds up -- truncation is defensive programming.\n")


if __name__ == "__main__":
    p = make_params()
    prompt = [1, 7, 3]

    print("=" * 64)
    print("KV CACHE: correctness")
    print("=" * 64)
    a = generate_naive(p, prompt, 20)
    b = generate_cached(p, prompt, 20)
    assert a == b, "cache changed the output -- that's a bug!"
    print(f"  naive and cached outputs identical: {a[:10]}... OK\n")

    print("=" * 64)
    print("KV CACHE: the speedup grows with sequence length")
    print("=" * 64)
    print(f"  {'new tokens':>10} {'naive (s)':>10} {'cached (s)':>10} {'speedup':>8}")
    for n in (20, 60, 120):
        t0 = time.perf_counter(); generate_naive(p, prompt, n)
        t1 = time.perf_counter(); generate_cached(p, prompt, n)
        t2 = time.perf_counter()
        naive, cached = t1 - t0, t2 - t1
        print(f"  {n:>10} {naive:>10.3f} {cached:>10.3f} {naive/cached:>7.1f}x")

    # cache memory: layers x 2 (K and V) x T x D floats
    T = 123
    floats = LAYERS * 2 * T * D
    print(f"\n  cache size at T={T}: {floats*4/1024:.1f} KB for this toy.")
    print("  Same formula, GPT-2-XL numbers, context 1024: ~ hundreds of MB.")
    print("  That formula is why long context costs what it costs.\n")

    sample_demo()
