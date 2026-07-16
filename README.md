# LLMs From First Principles — A Curriculum for Engineers

A learning path from "I understand gradient descent intuitively" to "I can read the GPT-2 paper and build a working miniature of it." Written for an experienced software engineer with practical (not theoretical) math.

## The Core Idea

Every module follows the same loop:

1. **Why does this exist?** — the problem that forced someone to invent it
2. **Intuition first** — analogies and pictures before any notation
3. **Math after intuition** — only what you need, explained in engineering terms
4. **Full worked code** — small, annotated, runnable implementations you read, run, and modify
5. **Modify-it exercises** — active learning without a blank page

The math never goes deeper than matrix multiplication and derivatives — things you already know. When a formula appears, it appears *after* you already know what it's supposed to do, so it reads like documentation rather than a puzzle.

## Pacing (3–5 hrs/week)

| Phase | Modules | Weeks | You will build |
|-------|---------|-------|----------------|
| 1. Foundations | 00–03 | 1–10 | The complete math toolkit, gradient descent, a full neural net + backprop in NumPy |
| 2. Language | 04–06 | 11–17 | A BPE tokenizer, trained word embeddings, your first language model |
| 3. Transformers | 07–09 | 18–28 | Self-attention in NumPy, a tiny GPT, sampling + KV cache |
| 4. Modern LLMs | 10–12 | 29–36 | A vector database, semantic search, a RAG pipeline |
| 5. Capstones (optional) | 13–14 | 37–40 | Fine-tune your GPT (LoRA, forgetting); build an agent from scratch (ReAct, tools, memory) |

Roughly 8–9 months at a sustainable pace. Each module says how long to spend on it. **Do not rush Phase 1** — backprop (module 03) is the single concept everything else stands on.

## Module Map

```
phase1-foundations/
  00-math-primer.md                ALL the math the course needs (8 ideas),
                                   worked by hand + math_primer.py verifying it
  01-meaning-as-geometry.md        Why vectors? What embeddings really are
  02-gradient-descent.md           + gradient_descent.py
  03-neural-net-backprop.md        + mlp_numpy.py (the heart of the course)

phase2-language/
  04-tokenization.md               + bpe_tokenizer.py
  05-embeddings.md                 + tiny_word2vec.py
  06-first-language-model.md       + bigram_lm.py

phase3-transformers/
  07-attention.md                  + attention_numpy.py
  08-tiny-gpt.md                   + tiny_gpt.py (PyTorch — you've earned it)
  09-training-and-inference.md     sampling, temperature, KV cache + kv_cache_demo.py

phase4-modern-llms/
  10-knowledge-scaling-hallucination.md   weights vs. embeddings, fine-tuning, MoE
  11-vector-search.md              + vector_db.py (semantic search engine)
  12-rag.md                        + mini_rag.py
  13-fine-tuning.md                + finetune_tiny_gpt.py (optional capstone:
                                   SFT, LoRA, catastrophic forgetting — measured)

phase5-agents/
  14-agents.md                     + mini_agent.py (optional capstone: the ReAct
                                   loop, tool calling, memory, prompt injection —
                                   runs offline, 3-line swap to a real LLM)

appendix-frontier-topics.md        Reasoning models, multimodality, distributed
                                   training — one honest page each, read anytime
                                   after module 10

hints.md                           Graduated hints for every exercise —
                                   read one level at a time when stuck
debugging-guide.md                 Symptom → cause → fix table; keep open
                                   whenever code is running

diagrams/                          SVG diagrams embedded in the modules
                                   (view the .md files in VS Code preview,
                                   GitHub, or any markdown viewer to see them)
resources.md                       Curated external videos/articles per module
```

## Rules of Engagement

- **Run every piece of code.** Reading code convinces you that you understand; running and breaking it proves whether you do.
- **Do the modify-it exercises.** They are calibrated to take 30–60 min each and are where the real learning happens.
- **NumPy until module 08.** You'll feel every matrix multiply. PyTorch enters only once you could, in principle, write what it automates.
- **When stuck for more than 30 minutes**: broken code → `debugging-guide.md`; stuck exercise → `hints.md`, one hint level at a time. Then re-read the module's intuition section — confusion almost always means a *why* was skipped, not that you lack math.

## Prerequisites

Python 3, NumPy (`pip install numpy`), and from module 08 onward PyTorch (`pip install torch`). Nothing else. All models train in minutes on a laptop CPU.

**Math prerequisites: none.** Module 00 rebuilds everything the course uses — even if school math is 20 years behind you. If you're already fluent in matrix multiplication and derivatives, skim module 00 in an evening and keep its notation table handy; otherwise give it its full two weeks. Whenever any later module's math feels slippery, the bug is almost always a module 00 concept that needs one more rep — go back, rerun `math_primer.py`, come forward again. That loop is normal and expected.

## Where You'll Be at the End

You will have built, with your own hands: a tokenizer, an autodiff-free neural net trained by backprop you wrote yourself, word embeddings whose geometry you can inspect, a working GPT that generates text, and a RAG system. At that point papers like "Attention Is All You Need" read as design docs for systems you've already implemented.
