# Module 13 — Fine-Tuning Your Own GPT (Optional Capstone)

> **⏱️ Time:** ~1-2 weeks.  
> **What you'll build:** Core mechanics and conceptual understanding.

---

## 🎯 TL;DR
1. **The Core Problem:** The challenge this module solves.
2. **The Mechanism:** How we solve it using first principles.
3. **The Payoff:** What you can do with this new capability.

---


**Time: about 2 weeks. Code: `finetune_tiny_gpt.py` (imports your module 08 model). This module closes the course's one theory-only gap: modules 09 and 10 *told* you about fine-tuning, catastrophic forgetting, and LoRA. Now you'll cause all three on your own machine.**

## What You're About to Do

Take your module 08 GPT — trained on Shakespeare — and retrain it to write in a completely different style: plain modern English. Along the way you'll watch, with real measurements, the three phenomena module 10 could only describe:

1. **Fine-tuning changes behavior fast.** A few hundred steps and the sampled text audibly changes voice.
2. **Catastrophic forgetting is real and measurable.** As the model learns the new style, you'll watch its Shakespeare loss climb — skill actively draining away, on a chart, in front of you.
3. **LoRA sidesteps the trade.** Freeze the whole model, train two skinny matrices per layer (under 2% of the parameters), get most of the new style — and because the base weights never moved, the original skill is *perfectly preserved*: delete the LoRA matrices and your Shakespeare model is back, byte for byte.

## Why Forgetting Happens — One Paragraph, No Hand-Waving

Recall where capability lives: distributed across shared weights (module 10), every weight serving many patterns at once (superposition). When gradient descent chases the new objective, it adjusts whatever weights reduce the *new* loss — with zero regard for what else those weights were doing. There's no isolation, no namespace, no transaction log. Each update to serve "plain English" quietly bulldozes a little of the structure that served "Elizabethan verse." Forgetting isn't a malfunction; it's the absence of any mechanism that could prevent it.

## LoRA — The Idea in Plain Terms

Full fine-tuning rewrites the book. LoRA (**Lo**w-**R**ank **A**daptation) writes sticky notes.

Mechanically: for a weight matrix W (say 128×128 = 16,384 numbers), freeze it. Add a bypass: two small matrices A (128×r) and B (r×128), where r — the "rank" — is tiny, like 4. The layer now computes `x @ W + x @ A @ B`. Only A and B train. At r=4 that's 1,024 trainable numbers standing in for 16,384 — and since A@B starts at zero (B is initialized to zeros), training begins from *exactly* the original model and learns only a correction. Sound familiar? It's module 08's residual idea — "learn a correction, not a replacement" — applied to weights instead of activations.

Why does a rank-4 correction suffice when the weight matrix is rank 128? The empirical finding that launched a thousand fine-tunes: *the change* needed to adapt a pretrained model is usually simple — low-rank — even though the model itself is not. You're not teaching it English from scratch; you're nudging a competent system's style. Small nudge, small matrix.

The operational wins fall out immediately, and you'll verify each: tiny checkpoint files (kilobytes, not megabytes — you only save A and B), perfect base-model preservation (unplug the bypass, original restored), and swappable personalities (one LoRA per style, hot-swapped over a shared base — this is how one GPU serves a hundred fine-tuned "models").


> **🛑 CHECKPOINT:** Run the code and modify it before proceeding. Reading without running is an illusion of knowledge.

## The Code

`finetune_tiny_gpt.py` runs a complete four-act experiment (add `--quick` for a fast smoke run):

1. **Pretrain** the module 08 GPT on Shakespeare-style text, briefly. Baseline losses recorded on *both* corpora.
2. **Full fine-tune** on plain modern English. Every 50 steps it logs both losses — new-style loss falling, Shakespeare loss rising. That widening scissors pattern is catastrophic forgetting, in your terminal.
3. **Reset, then LoRA fine-tune** — same data, same steps, but only A/B matrices train (the script prints the trainable-parameter count: expect ~1–2%). New-style loss falls nearly as far; Shakespeare loss barely moves — and the script proves base-weight preservation with a checksum.
4. **Samples from all three models** — base, full-FT, LoRA-FT — side by side, so your eyes confirm what the numbers said.

## Modify-It Exercises

(Hints in `hints.md`, as always.)

1. Plot the forgetting curve properly: Shakespeare val loss vs fine-tuning steps. Find the step where new-style learning has plateaued — every step past it is pure damage. You've discovered why "how long do I fine-tune?" is a real hyperparameter.
2. Sweep the LoRA rank: r = 1, 4, 16. For each, record new-style loss and trainable-parameter count. Find your model's knee — the r where more rank stops buying quality.
3. Implement the oldest anti-forgetting trick, **replay**: mix 20% original Shakespeare into the fine-tuning batches. Measure how much forgetting it prevents and what it costs in new-style quality.

## Best External Resources

- The LoRA paper (Hu et al., 2021) — after this module, read §1 and §4 and enjoy recognizing your own experiment in Figure 1.
- Karpathy's *"State of GPT"* (again) — the SFT section will now map onto something you've run.

## The Actual End

Module 12 said "look at what's on your disk." Add today's item: a base model you fine-tuned two different ways, a forgetting curve you plotted, and a LoRA implementation you wrote — the exact workflow (minus a few zeros of scale) that companies run to customize frontier models. The gap between you and applied AI engineering is now experience, not concepts. Go build something.
