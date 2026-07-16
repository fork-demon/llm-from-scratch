# Curated External Resources, by Module

Rule of thumb baked into the whole course: **build first, then watch/read.** Videos consolidate; they don't substitute. Each module doc says *when* to use its resources; this file collects them plus a few extras.

## Phase 1 — Foundations

| Module | Resource | Why / when |
|---|---|---|
| 00 | 3Blue1Brown, *Essence of Linear Algebra* ch. 1–4 only | Rewatch AFTER module 00 — the animations attach to mechanics you now own. Skip determinants/eigenvectors; the course never needs them |
| 00 | 3Blue1Brown, *Essence of Calculus* ch. 1–3 only | Same rule: after the nudge-experiment sections |
| 01 | 3Blue1Brown, *"But what is a GPT?"* (Deep Learning ch. 5) | Rewatch the embedding section after reading module 01 |
| 01, 05 | Jay Alammar, *"The Illustrated Word2vec"* | Best embeddings visuals in existence; read twice (before 01, after 05) |
| 02 | 3Blue1Brown, Deep Learning ch. 2 (gradient descent) | After running Stage C |
| 02–03 | Karpathy, *"building micrograd"* | First 40 min with module 02; the rest after your mlp_numpy gradient check passes |
| 03 | 3Blue1Brown, Deep Learning ch. 3–4 (backprop) | After your gradient check passes |
| 03 | CS231n notes: "Backpropagation, Intuitions" | Written companion to module 03 |

## Phase 2 — Language

| Module | Resource | Why / when |
|---|---|---|
| 04 | Karpathy, *"Let's build the GPT Tokenizer"* | After building your BPE; goes further (regex splits, special tokens) |
| 04 | tiktokenizer.vercel.app | Poke real GPT tokenizers; 10 minutes, permanent intuition |
| 05 | projector.tensorflow.org | Fly through real embeddings in 3D |
| 06 | Karpathy, *"makemore"* part 1 | Deliberately compatible with module 06; watch after |
| 06 | Shannon 1948, §3 only | The n-gram idea in the founding document of information theory |

## Phase 3 — Transformers

| Module | Resource | Why / when |
|---|---|---|
| 07 | Jay Alammar, *"The Illustrated Transformer"* | The canonical visuals, after your attention code runs |
| 07 | 3Blue1Brown, ch. 6 (attention) | Best QKV animation ever made; after the code |
| 08 | **Karpathy, *"Let's build GPT from scratch"*** | The single best video on the subject; you're now its ideal viewer |
| 08 | nanoGPT repo (`model.py`) | Production-grade version of your tiny_gpt.py; read all ~300 lines |
| 08 | Vaswani et al. 2017, *Attention Is All You Need* | Read §3.2 for real; skip encoder-decoder plumbing |
| 09 | Karpathy, *"State of GPT"* | The pretrain→SFT→RLHF pipeline from the source |
| 09 | Jay Alammar, *"The Illustrated GPT-2"* | Its KV-cache diagrams document your own demo |

## Phase 4 — Modern LLMs

| Module | Resource | Why / when |
|---|---|---|
| 10 | Karpathy, *"Intro to Large Language Models"* (1 hr) | Ecosystem view, perfectly timed here |
| 10 | 3Blue1Brown, ch. 7 (how LLMs store facts) | Animated companion to module 10's first half |
| 10 | Meng et al., ROME paper (abstract + fig. 1) | Fact-editing evidence for FFN-as-KV-store |
| 10 | Hoffmann et al. 2022 (Chinchilla), §1 + tables | The scaling recipe that reorganized the industry |
| 10 | Anthropic: "Toy Models of Superposition", "Scaling Monosemanticity" | Skim for pictures; you now have the prerequisites |
| 11 | Pinecone, *"Faiss: The Missing Manual"* (IVF & HNSW chapters) | Best applied ANN writing, after building your IVF |
| 12 | Lewis et al. 2020 (RAG paper), §1–2 | The origin; the rest is dated |
| 12 | Anthropic, "Contextual Retrieval" post | Practical upgrade path from your mini_rag.py |

## Phase 5 — Capstone

| Module | Resource | Why / when |
|---|---|---|
| 13 | Hu et al. 2021, the LoRA paper (§1 and §4) | After running your own LoRA — Figure 1 will look like your experiment |
| 14 | Yao et al. 2022, the ReAct paper (§1–2) | After building the loop; reads like your own design notes |
| 14 | Anthropic, *"Building effective agents"* post | Every diagram is a variation of your mini_agent.py |
| 14 | Simon Willison on prompt injection | The clearest ongoing coverage of the failure you reproduced |
| App. | DeepSeek-R1 paper | Unusually readable; the reasoning-training recipe |
| App. | CLIP + ViT papers; HF *Ultra-Scale Playbook* | Multimodality and distributed training, after the appendix |

## After the Course

- transformer-circuits.pub — interpretability (Anthropic). You now meet the prerequisites.
- vLLM PagedAttention paper — virtual memory for your KV cache; delightful if you liked module 09.
- Llama papers (1–3) — read as engineering docs: every architecture delta from your tiny GPT is one paragraph each (RoPE, GQA, SwiGLU, RMSNorm).
- Karpathy, llm.c — the whole training stack in C, for when frameworks feel too magical again.
- fast.ai part 2 ("from the foundations") — you bounced off fast.ai before; part 2 rebuilds everything from scratch and lands very differently after this course.
