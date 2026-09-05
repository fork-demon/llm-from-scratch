# Module 09 — Training & Inference as Systems Problems

> **⏱️ Time:** ~1-2 weeks.  
> **What you'll build:** Core mechanics and conceptual understanding.

---

## 🎯 TL;DR
1. **The Core Problem:** The challenge this module solves.
2. **The Mechanism:** How we solve it using first principles.
3. **The Payoff:** What you can do with this new capability.

---


**Time: about 2 weeks. Code: `kv_cache_demo.py`. This module speaks your native language — caching, latency, memory budgets — so expect it to feel like home turf.**

## Part 1 — What "Training an LLM" Actually Involves

You've now trained a GPT, but only the first act of the real pipeline. Modern assistants like ChatGPT and Claude go through three stages, and knowing which stage does what will sharpen your intuition about everything these systems do:

**Stage 1 — Pretraining.** Your module 08 loop, scaled: next-token prediction over *trillions* of tokens of internet text, months of GPU time. This is 99%+ of the total compute, and it produces something worth understanding precisely: a **base model** — a phenomenally good autocomplete with no concept of being helpful. Prompt a base model with "What is the capital of France?" and it may well continue with *"What is the capital of Spain? What is the capital of Italy?"* — because a list of quiz questions is a perfectly plausible way for that text to continue. Nothing is wrong with it. It's doing exactly what it was trained to do. It just wasn't trained to be an assistant.

**Stage 2 — Supervised fine-tuning (SFT).** Keep training — same loss, same gradient descent, nothing new mechanically — but on a small, curated dataset of conversations: prompt, then a high-quality response written by humans. A few thousand to a few hundred thousand examples. This teaches the *format* of assistance: when text looks like a question, what follows is an answer.

**Stage 3 — Preference tuning (RLHF and relatives).** Humans compare pairs of model outputs — "this response is better than that one" — and those preferences are distilled into a training signal nudging the model toward helpful, honest, harmless behavior. Different objective, same underlying gradient machinery.

The engineering insight worth keeping: stages 2 and 3 are *cheap* — a sliver of stage 1's cost — and they mostly change **behavior**, not knowledge. What the model *knows* is overwhelmingly baked in during pretraining; what it *does with what it knows* is shaped afterward. That distinction — knowledge vs. behavior — becomes a load-bearing idea in module 10 when we ask what fine-tuning can and can't do for you.

## Part 2 — Inference: The Generation Loop, Instrumented

Module 06 gave you the autoregressive loop; module 08 gave it a real model to drive. Now let's look at it the way you'd look at any production service: what are the knobs, and where does the time go?

### The Sampling Knobs (what the API parameters actually do)

Here's a framing that instantly demystifies the API docs: the model outputs scores; **how you pick a token from those scores is a policy decision made entirely outside the network.** Temperature, top-k, top-p — none of them touch the model. They're all post-processing on one array of numbers.

- **Greedy** — always take the highest-scoring token. Deterministic, and prone to the repetition loops you met in module 06.
- **Temperature** — divide all the scores by a constant T before softmax. T below 1 stretches the gaps between scores → the favorite dominates → safe, predictable output. T above 1 compresses the gaps → underdogs get real chances → creative, riskier output. It's a contrast knob on the probability distribution, nothing more. And there is no "correct" setting — code generation wants low T (there's one right answer), brainstorming wants high T (there isn't).
- **Top-k and top-p** — chop off the tail of the distribution before sampling (keep the k best, or the smallest set covering p of the probability). Why does the tail need chopping? A subtle numbers game: with 50,000 tokens in the vocabulary, even if each garbage token has tiny probability, *thousands of garbage tokens* add up to a real chance of drawing one per step. And one bad draw compounds — every later token conditions on it. Tail truncation is defensive programming for the sampling loop.

The code includes a small sampler where you can feel each knob reshape a distribution you can see.

### The KV Cache — the Part Everyone Hand-Waves, Which You Will Build

Here's a performance bug hiding in plain sight in module 08's `generate()`. Each new token, we feed the *entire* sequence through the model again. Generate token 500 → process 500 tokens. Token 501 → process 501. The work per token grows with the sequence, and total work grows quadratically. Ouch.

Now the observation that fixes it — and notice it's *your* observation, from module 07: **under the causal mask, no token ever attends forward.** So when we append token 501, what changes about tokens 1–500? Their keys? No. Their values? No — nothing behind them changed, and nothing ever attends ahead. **Every K and V we computed last step gets recomputed, identical, this step.** Recomputing identical results every iteration — you have a word for this shape of problem, and you have the standard fix:

> **Cache it.** After each step, keep every layer's K and V for all positions. Next step, run *only the new token* through the model; its attention reads old K/V from the cache and appends its own.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 440" font-family="sans-serif">
  <rect width="760" height="440" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">KV cache: old tokens' K,V never change — stop recomputing them</text>

  <!-- naive panel -->
  <text x="60" y="62" font-size="14" fill="#E8734A" font-weight="bold">NAIVE: every step reprocesses everything</text>
  <g font-size="12" text-anchor="middle">
    <text x="45" y="100" fill="#666" text-anchor="start">step 3:</text>
    <rect x="110" y="82" width="50" height="28" rx="5" fill="#fdf3ee" stroke="#E8734A"/><text x="135" y="101" fill="#333">the</text>
    <rect x="166" y="82" width="50" height="28" rx="5" fill="#fdf3ee" stroke="#E8734A"/><text x="191" y="101" fill="#333">cat</text>
    <rect x="222" y="82" width="50" height="28" rx="5" fill="#fdf3ee" stroke="#E8734A" stroke-width="2"/><text x="247" y="101" fill="#333">sat</text>

    <text x="45" y="145" fill="#666" text-anchor="start">step 4:</text>
    <rect x="110" y="127" width="50" height="28" rx="5" fill="#fdf3ee" stroke="#E8734A"/><text x="135" y="146" fill="#333">the</text>
    <rect x="166" y="127" width="50" height="28" rx="5" fill="#fdf3ee" stroke="#E8734A"/><text x="191" y="146" fill="#333">cat</text>
    <rect x="222" y="127" width="50" height="28" rx="5" fill="#fdf3ee" stroke="#E8734A"/><text x="247" y="146" fill="#333">sat</text>
    <rect x="278" y="127" width="50" height="28" rx="5" fill="#fdf3ee" stroke="#E8734A" stroke-width="2"/><text x="303" y="146" fill="#333">on</text>
  </g>
  <text x="360" y="101" font-size="11" fill="#E8734A">← ALL recomputed</text>
  <text x="360" y="146" font-size="11" fill="#E8734A">← ALL recomputed again (identical results!)</text>
  <text x="60" y="185" font-size="12" fill="#888">work per token grows with T → O(T²)-ish generation</text>

  <!-- cached panel -->
  <text x="60" y="235" font-size="14" fill="#55A868" font-weight="bold">CACHED: run only the new token; read the past from the cache</text>
  <g font-size="12" text-anchor="middle">
    <text x="45" y="285" fill="#666" text-anchor="start">step 4:</text>
    <rect x="110" y="262" width="162" height="34" rx="5" fill="#eefaf1" stroke="#55A868" stroke-dasharray="5,3"/>
    <text x="191" y="283" fill="#55A868">K,V cache: the, cat, sat</text>
    <rect x="278" y="262" width="50" height="34" rx="5" fill="#eefaf1" stroke="#55A868" stroke-width="2.5"/>
    <text x="303" y="283" fill="#333">on</text>
  </g>
  <defs>
    <marker id="ka" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#55A868"/>
    </marker>
  </defs>
  <line x1="290" y1="262" x2="230" y2="245" stroke="#55A868" stroke-width="2" marker-end="url(#ka)"/>
  <text x="340" y="252" font-size="11" fill="#55A868">new token's Q attends to cached K,V,</text>
  <text x="340" y="266" font-size="11" fill="#55A868">then appends its own K,V to the cache</text>
  <text x="60" y="325" font-size="12" fill="#888">work per token ~constant in model size → O(T) attention reads; Q is never cached (used once)</text>

  <!-- memory formula -->
  <rect x="60" y="350" width="640" height="66" rx="10" fill="#f6f8fa" stroke="#ddd"/>
  <text x="380" y="378" text-anchor="middle" font-size="14" fill="#333" font-family="monospace">cache bytes = layers × 2 × context_len × dim × bytes_per_float</text>
  <text x="380" y="402" text-anchor="middle" font-size="12" fill="#888">grows linearly with context — THE GPU-memory constraint of long-context serving, and why prefill vs decode are different regimes</text>
</svg>

*The most consequential cache in modern computing. One dashed box replaces a mountain of redundant work.*

(Why isn't Q cached too? Nice interview question — answer it from the mechanics: a token's query is used exactly once, at the step where that token asks "who matters to me?" Nobody ever looks anything up *by* an old query again. Keys and values get *read* repeatedly by every future token; queries are fire-and-forget.)

Once you hold the cache in your head, a series of previously-opaque industry facts snap into focus as *derivations* rather than trivia:

- **Why "time to first token" and "time per token after that" are different numbers** in every benchmark. Processing your prompt (**prefill**) runs all tokens through at once, in parallel — GPU-friendly, compute-bound, brief. Generating the reply (**decode**) is one token at a time, each step limited mostly by how fast K/V can be streamed from memory — bandwidth-bound. Two regimes, two line items.
- **Why long context costs real money.** Look at the formula in the diagram: cache size grows linearly with context length, *per request being served*. A model with 128k context holding many concurrent conversations is, to a first approximation, a GPU-memory bill. When a provider prices long context steeply, you're looking at that formula.
- **What "prompt caching" on API pricing pages is.** KV caching *across* requests: if your prompt starts with the same long system preamble as your last request, the provider can reuse the stored K/V for that prefix instead of re-prefilling it. It's precisely the mechanism you're about to build, promoted to a billed product feature.

### Context Windows, Honestly

Worth saying plainly, because marketing blurs it: the context window is **not** a memory, a database, or something the model "keeps." It's the maximum T — position embeddings exist up to there, attention cost is O(T²) up to there. Between API calls, *nothing persists*. The "memory" of your chat is the application re-sending the whole conversation history each turn — and paying prefill on it each time. What feels like a model that remembers you is a stateless function being handed a longer and longer argument.


> **🛑 CHECKPOINT:** Run the code and modify it before proceeding. Reading without running is an illusion of knowledge.

## The Code

`kv_cache_demo.py` — NumPy, so the mechanism sits in plain sight with no framework wrapping:

1. A minimal 2-layer causal transformer with random weights (we're measuring *mechanics*, not text quality).
2. `generate_naive()` — module 08's recompute-everything loop.
3. `generate_cached()` — the real thing: per-layer K/V append, new-token-only forward pass.
4. **The proof and the payoff:** an assertion that both produce *identical* output (a cache that changes results isn't a cache, it's a bug), then a timing table showing the speedup growing with sequence length — 3× at 20 tokens, 10× at 120, and climbing.
5. The sampler: temperature, top-k, top-p applied to one visible distribution, so each knob's effect is something you've *seen* rather than read.

## Modify-It Exercises

1. Compute your toy model's cache size from the formula. Then plug in GPT-2-XL's numbers (48 layers, dim 1600, context 1024) — now hundreds of megabytes. Then a modern 128k-context model. Somewhere in that arithmetic you'll involuntarily say "oh, *that's* why" — that moment is the exercise.
2. Generate from your module 08 model at temperature 0.01, then 5.0. Read both outputs. You now have a calibrated feel for the knob most people set by superstition.
3. **The real exercise of the module:** retrofit a KV cache into `tiny_gpt.py`'s `generate()`, and measure the wall-clock speedup at 500 tokens. This is a genuine optimization PR against your own model — resume-grade, and more instructive than anything I could add to the demo.
4. Sabotage the cache: append K but "forget" to append V for one layer. Watch the output turn to garbage *silently* — no crash, just wrongness. Now re-read the assertion in step 4 of the demo and appreciate it as the regression test you'd insist on in production.

## Best External Resources

- Karpathy, *"State of GPT"* (Microsoft Build talk) — the full pretraining → SFT → RLHF pipeline explained by someone who ran it at scale. Perfect right after Part 1.
- Jay Alammar, *"The Illustrated GPT-2"* — its KV-cache diagrams will read as documentation of the demo you just built.

Next: Phase 4 — we stop asking *how it works* and start asking *what it knows*: **10 — Knowledge, Scaling, and Hallucination.**
