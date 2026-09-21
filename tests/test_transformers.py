"""Part 6 (attention) and lesson 7.3 (KV cache)."""
import numpy as np

import attention_numpy as att
import kv_cache_demo as kv

rng = np.random.default_rng(0)


def _weights(D):
    return [rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(3)]


def test_attention_rows_sum_to_one():
    x = rng.normal(size=(5, 8))
    _, w = att.attention(x, *_weights(8))
    assert np.allclose(w.sum(axis=1), 1.0)


def test_causal_mask_blocks_the_future():
    x = rng.normal(size=(5, 8))
    _, w = att.attention(x, *_weights(8), causal=True)
    assert np.allclose(np.triu(w, k=1), 0.0)
    assert np.isclose(w[0, 0], 1.0)


def test_causal_attention_ignores_later_tokens():
    x = rng.normal(size=(5, 8))
    W = _weights(8)
    out1, _ = att.attention(x, *W, causal=True)
    x2 = x.copy()
    x2[4] += 10.0                      # change only the last token
    out2, _ = att.attention(x2, *W, causal=True)
    assert np.allclose(out1[:4], out2[:4])


def test_attention_is_order_blind_without_positions():
    x = rng.normal(size=(4, 8))
    W = _weights(8)
    out, _ = att.attention(x, *W)
    perm = [2, 0, 3, 1]
    out_p, _ = att.attention(x[perm], *W)
    assert np.allclose(out[perm], out_p)


def test_multi_head_shapes():
    T, D, H = 5, 8, 2
    x = rng.normal(size=(T, D))
    Wq, Wk, Wv = _weights(D)
    out, w = att.multi_head_attention(x, Wq, Wk, Wv, np.eye(D), n_heads=H)
    assert out.shape == (T, D) and w.shape == (H, T, T)


def test_kv_cache_gives_identical_output():
    p = kv.make_params()
    assert kv.generate_naive(p, [1, 7, 3], 12) == kv.generate_cached(p, [1, 7, 3], 12)
