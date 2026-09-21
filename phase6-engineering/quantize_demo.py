"""
Module 17b -- Weight quantization from scratch in NumPy.

A weight stored as a 16-bit float costs 2 bytes. Stored as an 8-bit integer it
costs 1 byte, as a 4-bit integer half a byte. Decoding is limited by how fast the
weights can be READ from memory, so fewer bytes means less memory AND more speed.
The price is rounding error. This file measures that price.

What you will see when you run it:

  1. One row of weights quantized by hand, number by number.
  2. int8 and int4 on a well-behaved matrix: per-tensor, per-row and group-wise
     scales. We report the error in the WEIGHTS and, more importantly, the error
     in the OUTPUT  y = x @ W.T  on random activations.
  3. The outlier problem, in weights: plant 8 large weights and watch per-tensor
     scaling collapse while per-row and group-wise scaling survive.
  4. The outlier problem, in activations: a few input channels 30x larger than
     the rest. Now a handful of weight columns decide almost the whole output,
     and an AWQ-style rescaling of just those columns cuts the output error.
  5. 8-bit ACTIVATIONS with outlier channels: per-tensor and per-token scales
     lose several times more accuracy, keeping the outlier channels in 16-bit
     restores it (the LLM.int8() idea).
  6. Memory for a 7B-parameter model at fp16 / int8 / int4, scales included,
     and the single-stream decode speed each one allows on 2 TB/s of bandwidth.

What this file is NOT: a real quantizer. Round-to-nearest with absmax scales is
the baseline every real method starts from. GPTQ and AWQ choose the rounding or
the scales more cleverly; NF4 (QLoRA) uses 16 non-uniform levels. See the notes
at the end of each section.
"""
import numpy as np

rng = np.random.default_rng(17)


# ----------------------------------------------------------------------
# The whole algorithm: symmetric, absmax, round to nearest
# ----------------------------------------------------------------------
def qmax(bits):
    return 2 ** (bits - 1) - 1          # int8 -> 127, int4 -> 7


def quantize_block(w, bits):
    """One scale for the whole array `w`. Returns (integers, scale)."""
    scale = np.abs(w).max() / qmax(bits)
    if scale == 0:
        return np.zeros_like(w, dtype=np.int8), 1.0
    q = np.clip(np.round(w / scale), -qmax(bits), qmax(bits)).astype(np.int8)
    return q, scale


def fake_quantize(W, bits, granularity="tensor", group=64):
    """Quantize then dequantize, so the result can be compared with W directly.

    granularity: "tensor"  one scale for the whole matrix
                 "row"     one scale per output channel (per row of W)
                 "group"   one scale per `group` consecutive weights inside a row
    Returns (W_hat, number_of_scales).
    """
    if granularity == "tensor":
        q, s = quantize_block(W, bits)
        return q * s, 1
    if granularity == "row":
        scale = np.abs(W).max(axis=1, keepdims=True) / qmax(bits)
        scale[scale == 0] = 1.0
        q = np.clip(np.round(W / scale), -qmax(bits), qmax(bits))
        return q * scale, W.shape[0]
    if granularity == "group":
        rows, cols = W.shape
        assert cols % group == 0, "row length must be a multiple of the group size"
        G = W.reshape(rows, cols // group, group)
        scale = np.abs(G).max(axis=2, keepdims=True) / qmax(bits)
        scale[scale == 0] = 1.0
        q = np.clip(np.round(G / scale), -qmax(bits), qmax(bits))
        return (q * scale).reshape(rows, cols), rows * (cols // group)
    raise ValueError(granularity)


def rel_err(a, b):
    """Relative error: size of the difference divided by the size of the original."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def report(W, X, configs):
    Y = X @ W.T
    rows = []
    for bits, gran, group in configs:
        W_hat, n_scales = fake_quantize(W, bits, gran, group)
        rows.append((bits, gran, group, rel_err(W, W_hat), rel_err(Y, X @ W_hat.T), n_scales))
    return rows


def print_rows(rows):
    print("  bits  scales        weight error   output error   number of scales")
    for bits, gran, group, we, oe, n in rows:
        name = gran if gran != "group" else f"group of {group}"
        print(f"  {bits:>4}  {name:<13}{we * 100:>11.2f}%{oe * 100:>14.2f}%{n:>19,}")


CONFIGS = [(8, "tensor", 0), (8, "row", 0), (4, "tensor", 0), (4, "row", 0),
           (4, "group", 128), (4, "group", 64), (4, "group", 32)]


# ----------------------------------------------------------------------
# Memory arithmetic
# ----------------------------------------------------------------------
def bits_per_weight(bits, group=None, scale_bits=16, row_len=4096):
    """Storage per weight including its share of the scales.
    group=None means one scale per row of `row_len` weights."""
    per = group if group else row_len
    return bits + scale_bits / per


def model_gb(params, bpw):
    return params * bpw / 8 / 1e9


def banner(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main():
    banner("1. Eight weights, quantized by hand to int4 (levels -7 ... 7)")
    w = np.array([0.021, -0.013, 0.004, 0.035, -0.028, 0.009, -0.002, 0.017])
    q, s = quantize_block(w, 4)
    print(f"  weights   {np.array2string(w, precision=3, floatmode='fixed')}")
    print(f"  scale   = largest |w| / 7 = {np.abs(w).max():.3f} / 7 = {s:.4f}")
    print(f"  integers  {q}          <- this is what gets stored: 4 bits each")
    print(f"  restored  {np.array2string(q * s, precision=3, floatmode='fixed')}")
    print(f"  error     {np.array2string(w - q * s, precision=3, floatmode='fixed')}"
          f"   never more than scale / 2 = {s / 2:.4f}")

    banner("2. A well-behaved matrix: W is 256 x 1024, entries ~ N(0, 0.02)")
    W = rng.normal(0, 0.02, size=(256, 1024))
    X = rng.normal(0, 1.0, size=(64, 1024))
    print_rows(report(W, X, CONFIGS))
    print("  8 bits: about 1% error whatever the granularity. 4 bits needs small groups.")

    banner("3. Outliers in the WEIGHTS: 8 of the 262,144 weights set to +-1.0 (50x the typical size)")
    Wo = W.copy()
    idx = rng.choice(W.size, size=8, replace=False)
    Wo.flat[idx] = rng.choice([-1.0, 1.0], size=8)
    print_rows(report(Wo, X, CONFIGS))
    print("  One scale for the whole tensor must stretch to reach 1.0, so the int8 step becomes\n"
          "  1/127 = 0.008 and ordinary weights (about 0.02) land on only a handful of levels.\n"
          "  At int4 the step is 0.14: every ordinary weight rounds to ZERO. A scale per row or\n"
          "  per group confines the damage to the few rows or groups that contain an outlier.")

    banner("4. Outliers in the ACTIVATIONS: 6 of 1024 input channels are 30x larger")
    hot = rng.choice(1024, size=6, replace=False)
    Xo = X.copy()
    Xo[:, hot] *= 30.0
    print_rows(report(W, Xo, [(4, "group", 128), (4, "group", 32)]))
    share = np.linalg.norm(Xo[:, hot] @ W[:, hot].T) ** 2 / np.linalg.norm(Xo @ W.T) ** 2
    print(f"  The relative errors look like section 2, but the output now depends on very few\n"
          f"  weights: {share * 100:.0f}% of the output's energy flows through those 6 columns of W.\n"
          f"  The rounding of the other 1018 columns hardly matters. Precision should go where\n"
          f"  the activations are large.")
    # AWQ-style fix: make the important columns larger before rounding (so their relative
    # rounding error shrinks), and divide the matching activations by the same factor.
    # Mathematically  (x / s) @ (W * s).T == x @ W.T,  so only the ROUNDING changes.
    s = np.ones(1024)
    s[hot] = 4.0
    for group in (128, 32):
        W_hat, _ = fake_quantize(W * s, 4, "group", group)
        err = rel_err(Xo @ W.T, (Xo / s) @ W_hat.T)
        print(f"  with the 6 hot columns scaled 4x before rounding, group of {group:<4} output error {err * 100:>6.2f}%")
    print("  That is the idea of AWQ (Lin et al., MLSys 2024): choose per-channel scales from\n"
          "  activation statistics to protect the small fraction of weights that matter most.\n"
          "  GPTQ (Frantar et al., ICLR 2023) attacks the same output error differently: it\n"
          "  rounds the weights in order and adjusts the not-yet-rounded ones to compensate.")

    banner("5. 8-bit ACTIVATIONS too (W8A8), with those outlier channels")
    Y = Xo @ W.T
    W8, _ = fake_quantize(W, 8, "row")
    xq, xs = quantize_block(X, 8)
    print(f"  (no outliers, per-tensor)     output error {rel_err(X @ W.T, (xq * xs) @ W8.T) * 100:>6.2f}%   <- the baseline")
    xq, xs = quantize_block(Xo, 8)
    print(f"  activations per-tensor        output error {rel_err(Y, (xq * xs) @ W8.T) * 100:>6.2f}%")
    X_tok, _ = fake_quantize(Xo, 8, "row")      # one scale per token (per row of X)
    print(f"  activations per-token         output error {rel_err(Y, X_tok @ W8.T) * 100:>6.2f}%")
    cold = np.setdiff1d(np.arange(1024), hot)
    X_mix = Xo.copy()
    X_mix[:, cold], _ = fake_quantize(Xo[:, cold], 8, "row")   # int8 for the ordinary channels ...
    W_mix = W.copy()                                           # ... against int8 weights,
    W_mix[:, cold] = W8[:, cold]                               # the 6 outlier channels stay 16-bit
    print(f"  outlier channels kept 16-bit  output error {rel_err(Y, X_mix @ W_mix.T) * 100:>6.2f}%")
    print("  Every token contains the outlier channels, so a scale per token does not help.\n"
          "  LLM.int8() (Dettmers et al., NeurIPS 2022) splits the matrix multiply: the few\n"
          "  outlier feature dimensions run in 16-bit, everything else in int8.")

    banner("6. Memory for a 7B-parameter model (7.0e9 weights, 16-bit scales included)")
    N = 7.0e9
    BANDWIDTH = 2.0e12   # bytes per second: an illustrative 2 TB/s accelerator
    table = [("fp16", 16.0), ("int8, scale per row of 4096", bits_per_weight(8)),
             ("int4, group of 128", bits_per_weight(4, 128)), ("int4, group of 64", bits_per_weight(4, 64)),
             ("int4, group of 32", bits_per_weight(4, 32))]
    print("  format                         bits/weight      GB    vs fp16   decode tokens/s, one stream")
    for name, bpw in table:
        gb = model_gb(N, bpw)
        print(f"  {name:<30}{bpw:>12.3f}{gb:>8.2f}{16 / bpw:>9.2f}x{BANDWIDTH / (gb * 1e9):>14.0f}  (upper bound)")
    print("  tokens/s = memory bandwidth / bytes of weights: every generated token reads every\n"
          "  weight once. Real kernels lose some of this to dequantization work and overheads.\n"
          "  In practice embeddings and norms often stay in 16-bit, so real files are a little\n"
          "  larger. NF4 (QLoRA, Dettmers et al., NeurIPS 2023) spends its 4 bits on 16 levels\n"
          "  spaced for bell-shaped weights instead of evenly, with one scale per 64 weights.")


if __name__ == "__main__":
    main()
