"""
Module 13 -- Fine-tune your module 08 GPT, twice, and measure everything.

  python finetune_tiny_gpt.py --quick    # ~3 min smoke run on CPU
  python finetune_tiny_gpt.py            # real run, ~15-25 min on CPU

Four acts:
  1. Pretrain on Shakespeare-style text (the "base model")
  2. FULL fine-tune on plain modern English
        -> watch new-style loss fall AND Shakespeare loss rise
           (catastrophic forgetting, live)
  3. LoRA fine-tune (base frozen, tiny A/B correction matrices train)
        -> new style learned with <2% of params trained; the BASE WEIGHTS are
           untouched, so removing the adapter restores the original model exactly
           (with the adapter attached, behaviour on Shakespeare still shifts)
  4. Side-by-side samples from base / full-FT / LoRA-FT

Imports the GPT from module 08 -- same model, no changes needed.
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the module 08 model (this file lives in phase4-modern-llms/)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "phase3-transformers"))
from tiny_gpt import GPT, Config  # noqa: E402

torch.manual_seed(1337)

# ----------------------------------------------------------------------
# Two corpora with very different voices.
# (Small and synthetic on purpose -- we're measuring phenomena, not art.)
# ----------------------------------------------------------------------
SHAKESPEARE = (
    "O for a Muse of fire, that would ascend the brightest heaven of "
    "invention! Once more unto the breach, dear friends, once more; or "
    "close the wall up with our English dead. What light through yonder "
    "window breaks? It is the east, and Juliet is the sun. To be, or "
    "not to be, that is the question: whether tis nobler in the mind to "
    "suffer the slings and arrows of outrageous fortune, or to take "
    "arms against a sea of troubles. Now is the winter of our "
    "discontent made glorious summer by this son of York. Friends, "
    "Romans, countrymen, lend me your ears; I come to bury Caesar, not "
    "to praise him. The evil that men do lives after them. Cowards die "
    "many times before their deaths; the valiant never taste of death "
    "but once. All the world is a stage, and all the men and women "
    "merely players. They have their exits and their entrances. "
) * 30

MODERN = (
    "The meeting starts at nine tomorrow morning. Please bring your "
    "laptop and the quarterly report. We need to fix the login bug "
    "before the release on friday. The coffee machine on the third "
    "floor is broken again. Can you review my pull request when you "
    "get a chance? The deploy pipeline failed twice last night. Let us "
    "schedule a quick call to talk about the roadmap. The new feature "
    "works fine on staging but not in production. Remember to submit "
    "your expense report by the end of the month. The team lunch is "
    "moved to thursday because of the demo. I will send the notes "
    "after the standup. The database migration finished without any "
    "errors. Please update the documentation when the api changes. "
) * 30


# ----------------------------------------------------------------------
# LoRA: wrap a frozen Linear with a trainable low-rank bypass.
#   forward:  x @ W  (frozen)  +  (x @ A) @ B * scale   (trainable)
# B starts at zero -> the wrapped layer starts EXACTLY equal to the original.
# (Module 08's residual idea, applied to weights: learn a correction.)
# ----------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                     # freeze the book
        self.A = nn.Parameter(torch.randn(base.in_features, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, base.out_features))  # zeros!
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A) @ self.B * self.scale


def apply_lora(model, r=4):
    """Wrap every attention Linear (qkv + proj) with LoRA; freeze the rest."""
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, r=r)
        block.attn.proj = LoRALinear(block.attn.proj, r=r)
    return model


# ----------------------------------------------------------------------
# Shared plumbing
# ----------------------------------------------------------------------
def make_data(text, stoi, ctx):
    ids = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:]


def get_batch(data, ctx, batch=32):
    ix = torch.randint(len(data) - ctx - 1, (batch,))
    x = torch.stack([data[i:i + ctx] for i in ix])
    y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
    return x, y


@torch.no_grad()
def eval_loss(model, data, ctx, iters=25):
    model.eval()
    losses = [model(*get_batch(data, ctx))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model, itos, n=150, temperature=0.8):
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, n, temperature=temperature)[0].tolist()
    return "".join(itos[i] for i in out)


def train(model, data, ctx, steps, lr, eval_sets=None, tag=""):
    """eval_sets: dict name -> val data; both losses logged every 50 steps."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for step in range(steps + 1):
        x, y = get_batch(data, ctx)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if eval_sets and step % 50 == 0:
            readout = "  ".join(f"{name} {eval_loss(model, d, ctx):.3f}"
                                for name, d in eval_sets.items())
            print(f"  {tag} step {step:4d}:  {readout}")
    return model


def checksum(model):
    """Sum of all BASE weights -- proves LoRA never touched them."""
    return sum(p.detach().sum().item()
               for n, p in model.named_parameters()
               if "A" not in n.split(".")[-1] and "B" not in n.split(".")[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    pre_steps = 400 if args.quick else 2500
    ft_steps = 200 if args.quick else 600

    # shared vocab across both corpora (so the model can express both)
    chars = sorted(set(SHAKESPEARE + MODERN))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    cfg = Config()
    cfg.vocab_size = len(chars)
    ctx = cfg.context_len

    shk_train, shk_val = make_data(SHAKESPEARE, stoi, ctx)
    mod_train, mod_val = make_data(MODERN, stoi, ctx)
    evals = {"shakespeare": shk_val, "modern": mod_val}

    # ---------------- Act 1: pretrain the base model ----------------
    print("=" * 66)
    print("ACT 1: pretrain on Shakespeare (the base model)")
    print("=" * 66)
    base = GPT(cfg)
    n_all = sum(p.numel() for p in base.parameters())
    print(f"  parameters: {n_all/1e6:.2f}M")
    train(base, shk_train, ctx, pre_steps, lr=3e-4, eval_sets=evals, tag="pre")
    base_shk = eval_loss(base, shk_val, ctx)
    base_mod = eval_loss(base, mod_val, ctx)
    print(f"\n  BASELINE  shakespeare {base_shk:.3f}   modern {base_mod:.3f}")
    print("  (good at its training style, bad at the other -- as expected)\n")

    # ---------------- Act 2: FULL fine-tune -> forgetting ----------------
    print("=" * 66)
    print("ACT 2: full fine-tune on modern English -- watch the scissors:")
    print("       modern loss falls while shakespeare loss RISES (forgetting)")
    print("=" * 66)
    full_ft = copy.deepcopy(base)
    train(full_ft, mod_train, ctx, ft_steps, lr=3e-4, eval_sets=evals, tag="full")
    full_shk = eval_loss(full_ft, shk_val, ctx)
    full_mod = eval_loss(full_ft, mod_val, ctx)
    print(f"\n  FULL-FT   shakespeare {full_shk:.3f} (was {base_shk:.3f}"
          f" -> forgot {full_shk-base_shk:+.3f})   modern {full_mod:.3f}\n")

    # ---------------- Act 3: LoRA fine-tune -> no forgetting ----------------
    print("=" * 66)
    print("ACT 3: LoRA fine-tune (base frozen, only A/B bypass matrices train)")
    print("=" * 66)
    lora_ft = apply_lora(copy.deepcopy(base), r=4)
    before = checksum(lora_ft)
    n_train = sum(p.numel() for p in lora_ft.parameters() if p.requires_grad)
    print(f"  trainable: {n_train:,} of {n_all:,} params "
          f"({100*n_train/n_all:.2f}%)")
    train(lora_ft, mod_train, ctx, ft_steps, lr=1e-3, eval_sets=evals, tag="lora")
    lora_shk = eval_loss(lora_ft, shk_val, ctx)
    lora_mod = eval_loss(lora_ft, mod_val, ctx)
    assert abs(checksum(lora_ft) - before) < 1e-3, "base weights moved!"
    print(f"\n  LoRA-FT   shakespeare {lora_shk:.3f}   modern {lora_mod:.3f}")
    print("  base-weight checksum unchanged: original model fully preserved")
    print("  (delete A/B and you have your Shakespeare model back, byte for byte)\n")

    # ---------------- Act 4: hear the difference ----------------
    print("=" * 66)
    print("ACT 4: samples (temperature 0.8)")
    print("=" * 66)
    print(f"\nBASE (shakespeare-flavored):\n{sample(base, itos)}\n")
    print(f"FULL-FT (modern-flavored, shakespeare damaged):\n{sample(full_ft, itos)}\n")
    print(f"LoRA-FT (modern-flavored, shakespeare intact underneath):\n{sample(lora_ft, itos)}\n")

    print("=" * 66)
    print("SCOREBOARD (val loss; lower = better)")
    print("=" * 66)
    print(f"  {'model':<10} {'shakespeare':>12} {'modern':>9}")
    print(f"  {'base':<10} {base_shk:>12.3f} {base_mod:>9.3f}")
    print(f"  {'full-FT':<10} {full_shk:>12.3f} {full_mod:>9.3f}   <- forgot the old")
    print(f"  {'LoRA-FT':<10} {lora_shk:>12.3f} {lora_mod:>9.3f}   <- base weights intact: detach adapter = original")
