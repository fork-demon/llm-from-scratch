# LLM From First Principles

> Understand how LLMs work by building one yourself.

An interactive, browser-based course for software developers. It takes you from a single dot product to
attention, a working GPT, RAG, fine-tuning and agents. Every idea is something you can see, change and break,
and every toy is connected to runnable Python code in this repository.

**[Start learning →](https://fork-demon.github.io/llm-from-scratch/)** &nbsp;·&nbsp; or run it locally: `cd course && npm install && npm run dev`

## Who is this for?

A developer who knows Python or Java, has seen APIs and data structures, and wants to *really* understand LLMs.
You do not need machine-learning experience, calculus, linear algebra or PyTorch. The little maths an LLM needs
is introduced only at the moment it becomes necessary, as a picture first and a formula second.

## What will I learn?

By the end you can, from memory:

- trace a token from raw text through tokenization, embeddings, Transformer blocks, logits and sampling
- explain why attention exists, what Query / Key / Value are, why the scores are scaled, why softmax, why the causal mask
- explain what training actually changes, and what temperature, top-p and the KV cache actually do
- say precisely what changes when you add **RAG** (the prompt), **fine-tuning** (the weights) or **tools** (a loop of your code)

The goal is one sentence: *"I don't use LLMs as magic anymore. I understand what is happening underneath."*

## How it teaches

Every lesson runs the same loop, in the same order:

```text
QUESTION → WHY → INTUITION → VISUAL → EXPERIMENT → NUMBERS → MATH → CODE → EXERCISE → RECALL → REAL LLM
```

You never meet an equation before you know what problem it solves. Exercises give graduated hints before any
solution. Each lesson ends with recall questions, each part ends with a mixed "Before moving on…" quiz that
reaches back to earlier parts, and the course ends with an explain-it-from-memory challenge.
Claims are labelled **Established**, **Simplified mental model** or **Active research**.

## How long does it take?

About 16 hours of lessons, plus a capstone project that takes as long as you want to give it.
Progress, quiz scores and completed exercises are stored in your browser. There is no sign-up and no backend.

## Curriculum map

```text
Text → Tokenization → Embeddings → Neural network → Attention → Transformer
     → Training → Inference → LLM → RAG → Fine-tuning → Agents
```

| Part | Lessons | Interactive experiments | Repository code |
|------|---------|-------------------------|-----------------|
| 0. What is an LLM? | What happens when I type a prompt · The surprising idea · Your map | clickable prompt-to-answer pipeline, four systems compared | |
| 1. The tiny bit of math an LLM needs | Vectors and the dot product · Matrices · Softmax · Derivatives and the chain rule | vector, matrix, softmax and nudge playgrounds | `phase1-foundations/math_primer.py` |
| 2. How machines learn | Gradient descent | fit a line by hand, then watch the machine do it | `phase1-foundations/gradient_descent.py` |
| 3. Neural networks | Neurons and layers · Backpropagation | build a curve from ReLU hinges, watch gradients flow backward | `phase1-foundations/mlp_numpy.py` |
| 4. How text becomes numbers | Tokenization · Embeddings | train a BPE tokenizer, explore an embedding space | `phase2-language/bpe_tokenizer.py`, `tiny_word2vec.py` |
| 5. Your first language model | Predicting the next token · Why simple models break | bigram generator, context explosion | `phase2-language/bigram_lm.py` |
| 6. Attention and the Transformer | Attention · Causal masks and multiple heads · The Transformer block | step-by-step Q/K/V playground, mask lab, clickable block | `phase3-transformers/attention_numpy.py` |
| 7. Build, train and run a GPT | Build GPT · Training GPT · Inference: sampling and the KV cache | token tracer, **a real GPT you train in your browser**, **a trained GPT you can look inside** (attention heads, logit lens), parameter counter, sampling and KV-cache labs | `phase3-transformers/tiny_gpt.py`, `kv_cache_demo.py` |
| 8. From GPT to modern LLMs | Why LLMs know things · Modern architecture · From raw text to assistant · Small models from big ones · Alignment and safety · Models that see and hear · Looking inside the model · Reasoning models | knowledge-in-weights lab, RoPE, GQA calculator, distillation lab, reward-hacking lab, image-patch lab, superposition lab, test-time compute simulator | |
| 9. Building with LLMs | RAG · Fine-tuning · Agents | mini-RAG system, LoRA and forgetting labs, tool-calling agent stepper | `phase4-modern-llms/*.py`, `phase5-agents/mini_agent.py` |
| 10. From understanding to engineering | Evals · Inference systems · The PyTorch and Hugging Face bridge · How to read an LLM paper · Context engineering and production agents | eval lab with confidence intervals and A/B tests, batching and paged-KV simulators, quantization lab, config reader, guided reading of “Attention Is All You Need”, context-budget lab and trace viewer | `phase6-engineering/*.py` |
| 11. Capstone | Build your own mini LLM system · Explain it from memory | milestone tracker, timed memory challenge | everything above |

The lessons follow one story: Riya, a backend developer at a Bengaluru fintech, learns how LLMs work from the
bottom up, with help from her mentor Kabir, her mother Amma and her sceptical flatmate Dev (see `course/STORY.md`).

Also in the app: **25 coding exercises that run in your browser** (write `softmax`, attention, a backprop step, a GPT
forward pass or a KV cache yourself and have hidden tests check it, powered by Pyodide, no install), a **"Run this file"**
button that runs 14 of the repository's NumPy scripts in the browser, a spaced **review queue** for questions
you got wrong, **skip-ahead diagnostics** at the start of a part, progress **export and import**, a searchable **glossary** (definition, intuition, example, where it is taught), a clickable
**concept map**, lesson search (press `/`), dark and light themes, keyboard navigation and a mobile layout.

## Repository layout

```text
course/                  the interactive course (React + TypeScript + Vite, static site)
  src/lessons/           one file per lesson, all following the same 12-section format
  src/interactive/       the playgrounds (one concept each)
  src/lib/               pure, unit-tested logic behind every interactive (softmax, attention, BPE, RAG, ...)
  src/components/        lesson layout, exercise system (hints → solution), quizzes, UI kit
  src/data/              curriculum map, glossary
  AUTHORING.md           how lessons are written (read this before contributing)
phase1-foundations/ … phase6-engineering/
                         the runnable Python implementations every lesson links to,
                         plus the original long-form written notes (*.md) as companion reading
tests/                   pytest tests for the Python implementations
hints.md, debugging-guide.md, resources.md, appendix-frontier-topics.md
                         companion material for the Python exercises
```

The course and the code stay connected: each lesson goes concept → interactive demo → exercise → the actual
file in this repository, and the TypeScript behind the demos is tested against the behaviour of the Python.

## Running the Python code

```bash
pip install -r requirements.txt        # numpy (+ torch from "Build GPT" onward; transformers and peft only for Part 10)
python phase3-transformers/attention_numpy.py
python phase3-transformers/tiny_gpt.py --quick
pytest tests
```

All models train in minutes on a laptop CPU. No API keys are needed anywhere.

## Developing the course

```bash
cd course
npm install
npm run dev          # local dev server
npm test             # unit tests: maths, tokenizer, attention, exercises, progress, every lesson renders
npm run build        # type-check + static build into course/dist
```

The site is fully static (hash routing, relative asset paths), so `course/dist` can be hosted anywhere.
`.github/workflows/course.yml` runs all tests and deploys to GitHub Pages on every push to `main`
(enable Pages with "GitHub Actions" as the source in the repository settings).
