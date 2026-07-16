# When Things Break: The Debugging Guide

From-scratch ML code fails in a small number of characteristic ways. This page maps each symptom to its usual causes, so a broken run costs you minutes instead of an evening. Keep it open whenever you're running course code — or your own.

**The 30-minute rule (from the README, worth repeating):** if you're stuck longer than 30 minutes, stop pushing. Check this guide, then re-read the *intuition* section of the relevant module. Confusion almost always means a "why" got skipped, not that you lack ability.

## The Symptom Table

| Symptom | Most likely causes, in order | Where to look |
|---|---|---|
| **Loss becomes NaN or explodes** (2.1 → 47.3 → 8812 → NaN) | 1. Learning rate too high — the classic. Cut it 10× and retry. 2. Missing `/ √d` scaling in attention. 3. `log(0)` somewhere — add the `1e-12` guard inside every `log()`. 4. Numbers overflowing in `exp()` — you forgot the subtract-the-max trick in softmax. | Module 02 (lr), 07 (scaling), 03 (softmax stability) |
| **Loss frozen, never moves** | 1. You're not actually updating — check the minus sign and that the update line runs. 2. Learning rate absurdly small. 3. Weights initialized to all zeros — every neuron identical, gradients identical, nothing can differentiate ("symmetry breaking" needs randomness). 4. Gradients not flowing: a detached tensor in PyTorch, or you zeroed the wrong thing. | Module 02, mlp_numpy.py init comment |
| **Loss falls, then plateaus far above where it should** | 1. Model class can't represent the answer — are you asking a linear model to draw a curve? 2. Not enough capacity: widen or deepen. 3. Learning rate too high to settle into the valley — decay it or lower it. 4. You've hit the actual floor for this context length (a bigram can't beat the bigram optimum — module 06 proved it). | Modules 02 (ex. 2), 03, 06 |
| **Shape error / matmul dimension mismatch** | Not really a bug — a type error. Write the shape next to every array in the failing expression, as comments. Check the *inner* numbers match: `(m, n) @ (n, p)`. 90% of the time you needed a transpose; 9% you batched wrong. | Module 00 §4–6 |
| **Gradient check FAILS** | 1. Dropped or misplaced transpose in a backward rule (THE classic — `X.T @ d` vs `X @ d.T` vs `d @ X.T`). 2. Forgot the ReLU gate `* (x > 0)` in the backward pass. 3. Forgot to divide by batch size exactly once (not zero, not twice). 4. Your numerical check itself: `h` too small (float precision — use ~1e-5) or you modified a weight and forgot to restore it. | Module 03's table + mlp_numpy.py |
| **Gradient check passes, training still bad** | The math is right; the *setup* is wrong. 1. Data bug: inputs and targets misaligned by one (off-by-one in next-token pairs is the classic). 2. Train/test leakage or wrong split. 3. Labels shuffled independently of inputs. Print 3 (input, target) pairs and read them with your eyes. | Module 06's dataset builders |
| **Generated text loops forever** ("the the the...") | You're doing greedy argmax. Sample from the distribution instead, or raise temperature. This is expected behavior, not a bug — module 06 explains why the most likely step ≠ likely text. | Modules 06, 09 |
| **Generated text is pure noise after lots of training** | 1. Feeding the model's *input* back instead of its *output* in the generation loop. 2. Encoding/decoding mismatch — different vocab or tokenizer between train and generate. 3. Positions: you cropped the context but not the position indices. | Modules 06, 08 |
| **Accuracy stuck at exactly chance** (33% on 3 classes, 50% on 2) | The model is learning nothing: see "loss frozen" row. If loss *is* falling while accuracy sits at chance — you're measuring accuracy on the wrong thing (untrained model, wrong split, argmax over wrong axis). | Module 03 |
| **KV-cached output ≠ naive output** | 1. Forgot to append K or V for one layer (they silently desync). 2. Wrong position index for the new token. 3. Applied the causal mask inside the cached path — the cache only *contains* the past, masking it again shifts everything. | Module 09, kv_cache_demo.py |
| **Vector search returns garbage / all-zero scores** | 1. Query words not in the embedder's vocabulary (crude embedders can't handle unseen words — real ones use subword tokenization, module 04). 2. Forgot to normalize vectors, so long documents dominate. 3. Different embedder for ingest vs query — hashes from different hash functions. | Modules 11, 12 |
| **Training works, val loss rises** | Not a bug! That's overfitting/memorization, and now you can *see* it. More data, smaller model, or stop earlier. | Modules 08 (ex. 4), 10 |

## The Universal Four-Step Fallback

When the table doesn't cover it:

1. **Shrink until it works.** 10 data points, 2 dimensions, 1 layer, 5 steps. Bugs that hide in big runs are naked in tiny ones — and a tiny run takes 2 seconds, so you can bisect fast.
2. **Print shapes first, values second.** Shape bugs are cheap to find and cause most "mysterious" behavior.
3. **Trust the gradient check.** If it passes, stop suspecting the math and start suspecting the data. If you haven't run one, run one.
4. **Diff against the course code.** Every build has a known-good reference in this repo. `diff` your version against it before rewriting anything.

One encouragement to close: every symptom in this table is one *I'd expect* you to hit at some point — several of the exercises deliberately cause them. Hitting them is the curriculum working, not you failing.
