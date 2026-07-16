# Appendix — Frontier Topics, One Honest Page Each

**Read these after finishing the course (any time after module 10 works). No code — the goal is that when these topics come up in papers, launches, or interviews, you have a correct mental model built from things you personally made. Each section says what's established and what's still open.**

---

## 1. Reasoning Models (o1, R1, "thinking" models)

**What you notice as a user:** some models now visibly "think" before answering — producing a long internal monologue, then a final response. They're slower, cost more per query, and are markedly better at math, code, and multi-step problems. What changed?

**Start from what you built.** Module 14 showed that a written thought becomes context that steers later tokens — the model's own words are its only working memory. "Chain-of-thought" prompting (just asking a model to reason step by step) exploits this: intermediate steps written down make the *next* steps conditioned on them, and accuracy jumps. That much you can explain with module 06 alone.

Reasoning models take the next step: **instead of merely prompting for reasoning, train for it.** The recipe (as published for DeepSeek-R1; OpenAI's is undisclosed but likely similar in spirit): generate many candidate reasoning chains for problems with *checkable answers* — math with a known result, code with unit tests — then use reinforcement learning to reward chains that reached correct answers. The model learns not just to produce reasoning-shaped text (module 10's worry) but reasoning that *wins*. Strikingly, behaviors nobody scripted emerge under this training: the models learn to backtrack ("wait, that's wrong, let me reconsider"), to check their own work, to try a second approach — because chains that do these things get more answers right, and rewarded behavior grows.

**The economic reframe is the part worth remembering:** this creates a second scaling dial. Modules 08–10 taught you *training-time* compute (bigger model, more data → smoothly better). Reasoning models add **test-time compute**: the same model, allowed to think longer at inference, performs better — and you (or the API's "effort" knob) choose per-query how much thinking to buy. Easy question, short thought, cheap; hard question, long thought, expensive. Compute is shifting from a fixed cost of training to a variable cost of answering.

**What's open:** whether the visible "thoughts" faithfully reflect the actual computation (evidence says: not always — models sometimes reach answers their stated reasoning doesn't support), how far checkable-domain training transfers to fuzzy domains like strategy or writing, and whether RL-trained reasoning is composition or very good interpolation — module 10's unresolved question, now with more money riding on it.

---

## 2. Multimodality (how a text model "sees" an image)

**The one-sentence answer, which you're equipped to fully parse:** an image is turned into a sequence of embedding vectors, and those vectors enter the transformer *exactly as if they were token embeddings* — the model beyond the input layer neither knows nor cares that they came from pixels.

Unpack it with course pieces. Module 05 taught that a token embedding is just a row of learned numbers — a vector that *stands for* the token. Nothing in modules 07–08 ever used the fact that these vectors came from text: attention mixes vectors, FFNs transform vectors, the machinery is modality-blind. So the only question is: how do you turn an image into a sequence of good vectors?

The standard recipe (Vision Transformer, "ViT"): **chop the image into a grid of patches** — say 16×16-pixel squares, so a 512×512 image becomes ~1,024 patches. Flatten each patch's pixels into a vector, pass it through a learned linear projection (module 00: one matrix multiply) to get it into the model's embedding dimension, add a position embedding (module 07 — now 2-D positions), and you have a sequence of "image tokens." Patches are the tokenizer of vision (module 04's job, done with scissors instead of BPE).

Then train with a fake task, module 05 style. CLIP's contrastive recipe: take millions of (image, caption) pairs from the web, pull each image's vectors toward its caption's vectors, push mismatched pairs apart — the same push-pull that trained module 11's embedding models, now welding *pictures* and *words* into one shared geometry. That shared geometry is why "a photo of a dog" and an actual photo of a dog land near each other, which is why you can search photos with text, and why an LLM given image tokens can discuss what's in them.

Generation direction (making images) runs on a different engine — diffusion — which is genuinely outside this course's toolkit; its language-side cousin (autoregressive image generation, predicting patches like tokens) you *can* already reason about: it's module 06's loop over patch-tokens.

**What's open:** the best way to fuse modalities deeply (adapters vs. native joint training), video's brutal token economics (module 09's O(T²) meets 30 frames per second), and whether spatial reasoning failures ("how many legs does the horse have?") are a data problem or an architecture problem.

---

## 3. Distributed Training (how a model that fits on no single GPU gets trained)

Your module 08 GPT fit in megabytes. A 405B-parameter model needs ~800GB *just for the weights* in fp16 — an order of magnitude more than any single GPU's memory — before you count gradients, optimizer state (Adam keeps two extra numbers per parameter — roughly triple the memory of the weights alone), and activations. Training it is therefore a distributed-systems problem, and — like module 11 — this is your home field wearing ML clothes. There are exactly three ways to split the work, and real training runs use all three at once:

**Data parallelism — shard the *batch*.** Every GPU holds a full copy of the model; each processes a different slice of the batch; gradients get averaged across GPUs every step (an all-reduce), so all copies stay identical. It's read-replica scaling. Limit: each GPU must still hold the whole model — this alone can't train anything big. (The refinement, ZeRO/FSDP: shard the *optimizer state and weights* across the data-parallel GPUs too, gathering pieces just-in-time — trading network chatter for memory. Straightforwardly a distributed cache design.)

**Tensor parallelism — shard the *matrices*.** Module 00 said each row of a weight matrix is an independent dot product — so split the matrix's rows across 8 GPUs, let each compute its slice, and concatenate. The catch: results must be exchanged *within every layer's forward pass*, so this only works across GPUs connected by very fast links (within one server chassis). It's why "8×H100 nodes" is the standard unit of AI infrastructure — 8 is the tensor-parallel group size that the intra-node interconnect can feed.

**Pipeline parallelism — shard the *layers*.** GPUs 1–8 hold layers 1–12, GPUs 9–16 hold layers 13–24, and so on: an assembly line (module 03's metaphor, now literal hardware). Naively this leaves most stages idle (each waits for the previous); the fix is streaming many micro-batches through so all stages stay busy — with an unavoidable "bubble" of idle time at the start and end of every step. Pure pipelining tradeoffs, as familiar to you as CPU design or stream processing.

A real frontier run composes them: tensor-parallel within a node, pipeline-parallel across racks, data-parallel across the whole fleet — thousands of GPUs, months of wall-clock, engineered around one merciless constraint: **a training step advances at the pace of the slowest participant**, so stragglers, network hiccups, and hardware failures (at 10,000 GPUs, something is *always* failing) dominate the engineering. Checkpointing strategy, failure recovery, and interconnect topology decide these runs' fates as much as any ML idea does. If you ever move into AI infrastructure, you will feel suspiciously at home.

**What's open (or at least, unpublished):** exact frontier-lab recipes are trade secrets; async approaches that relax the lockstep constraint keep being proposed and mostly keep losing to synchronous training; and the hardware itself (interconnects, memory bandwidth — remember module 09's decode being bandwidth-bound?) is evolving fast enough that today's best split is next year's anti-pattern.

---

*Each of these topics has a natural next step if it hooks you: reasoning → the R1 paper (it's unusually readable); multimodality → the CLIP and ViT papers plus Jay Alammar's illustrated guides; distributed training → the Megatron-LM paper and Hugging Face's "Ultra-Scale Playbook." All of them will read as engineering docs now. That was the whole point of the course.*
