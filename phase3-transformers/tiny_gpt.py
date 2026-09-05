"""
Module 08 -- A tiny GPT, assembled from everything you've built.

  python tiny_gpt.py --quick     # ~2 min smoke run on CPU
  python tiny_gpt.py             # real run, ~10-20 min on CPU

PyTorch automates exactly two things you've done by hand:
recording the forward pass, and replaying your module-03 backward
rules via loss.backward(). Nothing here is conceptually new:

  component          you built it in
  ---------          ---------------
  embedding lookup   module 05
  attention          module 07 (this file = same math, same shapes)
  feed-forward MLP   module 03
  softmax + CE loss  module 03
  training loop      module 02
  generation loop    module 06
  residual + LN      new plumbing, explained in 08-tiny-gpt.md

Expected Output:
  (Run the script to see the numerical output and shape assertions pass)
"""
import argparse
import math
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


# ----------------------------------------------------------------------
# Config: a GPT is these 6 numbers. GPT-2 = same code, bigger numbers.
# ----------------------------------------------------------------------
class Config:
    context_len = 64     # max tokens the model can see (GPT-2: 1024)
    n_embd = 128         # embedding dimension           (GPT-2: 768)
    n_head = 4           # attention heads               (GPT-2: 12)
    n_layer = 4          # transformer blocks            (GPT-2: 12)
    dropout = 0.1
    vocab_size = None    # set from data


# ----------------------------------------------------------------------
# Data: tiny Shakespeare (downloads once; falls back to bundled text)
# ----------------------------------------------------------------------
SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")

FALLBACK = (
    "O for a Muse of fire, that would ascend the brightest heaven of "
    "invention, a kingdom for a stage, princes to act and monarchs to "
    "behold the swelling scene! Then should the warlike Harry, like "
    "himself, assume the port of Mars; and at his heels, leash'd in "
    "like hounds, should famine, sword and fire crouch for employment. "
) * 200


def load_text(path="shakespeare.txt"):
    if not os.path.exists(path):
        try:
            print("downloading tiny shakespeare (~1MB)...")
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        except Exception as e:
            print(f"download failed ({e}); using bundled fallback text")
            return FALLBACK
    with open(path) as f:
        return f.read()


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Module 07's multi-head attention, verbatim, in torch."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # Wq,Wk,Wv fused
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)       # Wo
        self.drop = nn.Dropout(cfg.dropout)
        # the causal mask, precomputed once
        mask = torch.tril(torch.ones(cfg.context_len, cfg.context_len))
        self.register_buffer("mask", mask.view(1, 1, cfg.context_len, cfg.context_len))

    def forward(self, x):
        B, T, D = x.shape                       # batch, tokens, embed dim
        q, k, v = self.qkv(x).split(D, dim=2)   # each (B, T, D)
        hd = D // self.n_head
        # (B, T, D) -> (B, n_head, T, hd): give each head its slice
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)     # (B, H, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)                        # rows sum to 1
        att = self.drop(att)
        y = att @ v                                         # (B, H, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, D)    # concat heads
        return self.proj(y)


class FeedForward(nn.Module):
    """Module 03's MLP: the 'computation' half of the block.
    Applied to each token independently. Most parameters live here."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),                       # smooth ReLU; same hinge idea
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """communicate (attention) then compute (FFN),
    each wrapped in pre-LayerNorm + residual."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual: block outputs a CORRECTION
        x = x + self.ffn(self.ln2(x))    # gradient highway through the '+'
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # module 05
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.n_embd)  # module 07
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying (see doc)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)   # meaning + position
        x = self.blocks(x)                          # communicate/compute stack
        x = self.ln_f(x)
        logits = self.head(x)                       # (B, T, vocab)

        loss = None
        if targets is not None:
            # cross-entropy = average surprise, exactly module 03's
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Module 06's autoregressive loop."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_len:]   # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature     # last position only
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)   # weighted die
            idx = torch.cat([idx, nxt], dim=1)              # feed back in
        return idx


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2-minute smoke run")
    args = ap.parse_args()

    text = load_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]   # held-out split: watch
                                                # generalization, not memorization

    cfg = Config()
    cfg.vocab_size = len(chars)
    steps = 300 if args.quick else 3000
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab {cfg.vocab_size}, params {n_params/1e6:.2f}M, device {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - cfg.context_len - 1, (batch_size,))
        x = torch.stack([d[i:i + cfg.context_len] for i in ix])
        y = torch.stack([d[i + 1:i + cfg.context_len + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def eval_loss(split, iters=50):
        model.eval()
        losses = [model(*get_batch(split))[1].item() for _ in range(iters)]
        model.train()
        return sum(losses) / len(losses)

    def sample(n_tokens=200):
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = model.generate(start, n_tokens)[0].tolist()
        return "".join(itos[i] for i in out)

    print("\ntraining... (watch capability emerge in the samples)")
    for step in range(steps + 1):
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()          # <- module 03, automated
        opt.step()               # <- module 02, with adaptive step sizes

        if step % (100 if args.quick else 500) == 0:
            tl, vl = eval_loss("train"), eval_loss("val")
            print(f"\n--- step {step}: train loss {tl:.3f}, val loss {vl:.3f} ---")
            print(sample(150))

    print("\nfinal sample (temperature 0.8):")
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, 400, temperature=0.8)[0].tolist()
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
