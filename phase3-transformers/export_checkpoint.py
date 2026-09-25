"""
Train tiny_gpt.py's model on tiny Shakespeare and export it for the browser.

  python export_checkpoint.py              # ~3 min on an Apple-silicon GPU (mps); ~20 min on CPU
  python export_checkpoint.py --steps 300  # smoke run

Reuses the model code from tiny_gpt.py unchanged (import, not copy). Two things
differ from `python tiny_gpt.py`, both standard GPT-2 practice for a better
model in the same few minutes:
  * GPT-2 style init: weights ~ N(0, 0.02), biases 0, and the two residual
    output projections (attn.proj, ffn.net[2]) scaled by 1/sqrt(2 * n_layer)
  * AdamW at lr 1e-3 with 100 warmup steps and cosine decay, batch 64, dropout 0

Writes (paths relative to the repo root):
  course/public/models/tiny-gpt.json   manifest: config, vocab, losses, tensor offsets
  course/public/models/tiny-gpt.bin    all weights, float16, little-endian
  course/src/lib/__fixtures__/trainedGpt.fixture.json
      logits / attention / logit lens that PyTorch computes for fixed prompts,
      using the float16-rounded weights, so the TypeScript forward pass can be
      tested against it.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_gpt import GPT, Config, load_text

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "course", "public", "models")
FIXTURE = os.path.join(ROOT, "course", "src", "lib", "__fixtures__", "trainedGpt.fixture.json")


def gpt2_init(model, n_layer):
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            std = 0.02
            if name.endswith("attn.proj.weight") or name.endswith("ffn.net.2.weight"):
                std = 0.02 / math.sqrt(2 * n_layer)
            nn.init.normal_(p, mean=0.0, std=std)
        elif "ln" in name:
            continue  # LayerNorm keeps gain 1, bias 0
        else:
            nn.init.zeros_(p)


def tensor_list(model):
    """(name, tensor) in the order the browser reads them. The head is tied, so not exported."""
    out = [("tok_emb", model.tok_emb.weight), ("pos_emb", model.pos_emb.weight)]
    for i, b in enumerate(model.blocks):
        p = f"blocks.{i}."
        out += [
            (p + "ln1.weight", b.ln1.weight), (p + "ln1.bias", b.ln1.bias),
            (p + "attn.qkv.weight", b.attn.qkv.weight), (p + "attn.qkv.bias", b.attn.qkv.bias),
            (p + "attn.proj.weight", b.attn.proj.weight), (p + "attn.proj.bias", b.attn.proj.bias),
            (p + "ln2.weight", b.ln2.weight), (p + "ln2.bias", b.ln2.bias),
            (p + "ffn.fc.weight", b.ffn.net[0].weight), (p + "ffn.fc.bias", b.ffn.net[0].bias),
            (p + "ffn.proj.weight", b.ffn.net[2].weight), (p + "ffn.proj.bias", b.ffn.net[2].bias),
        ]
    out += [("ln_f.weight", model.ln_f.weight), ("ln_f.bias", model.ln_f.bias)]
    return out


@torch.no_grad()
def trace(model, idx):
    """Forward pass that also returns attention weights and the logit lens (batch 1)."""
    cfg = model.cfg
    T = idx.shape[1]
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T))
    atts, lens = [], []
    for b in model.blocks:
        a = b.attn
        h = b.ln1(x)
        q, k, v = a.qkv(h).split(cfg.n_embd, dim=2)
        hd = cfg.n_embd // cfg.n_head
        q = q.view(1, T, cfg.n_head, hd).transpose(1, 2)
        k = k.view(1, T, cfg.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(a.mask[:, :, :T, :T] == 0, float("-inf"))
        atts.append(F.softmax(att, dim=-1)[0])  # (H, T, T)
        x = b(x)  # the real block (eval mode), so this matches model(idx) exactly
        lens.append(model.head(model.ln_f(x))[0])
    logits, _ = model(idx)
    return logits[0], atts, lens


@torch.no_grad()
def head_scores(model, stoi, text, n_seqs=20, half=None, seed=0):
    """Same scores the browser computes (see trainedGpt.ts::headScores)."""
    cfg = model.cfg
    g = torch.Generator().manual_seed(seed)
    # previous-token score: mean attention from position t to t-1 on real text
    L, H = cfg.n_layer, cfg.n_head
    half = half or min(48, cfg.context_len // 2)
    prev = torch.zeros(L, H)
    ind = torch.zeros(L, H)
    for s in range(n_seqs):
        start = int(torch.randint(len(text) - cfg.context_len, (1,), generator=g))
        ids = torch.tensor([[stoi[c] for c in text[start:start + cfg.context_len]]])
        _, atts, _ = trace(model, ids)
        for l in range(L):
            a = atts[l]
            prev[l] += torch.stack([a[:, t, t - 1] for t in range(1, ids.shape[1])], 1).mean(1)
        # induction score: random lower-case letters repeated twice; from position t in the
        # second copy, attention to the token AFTER the earlier copy of token t
        lower = torch.tensor([stoi[c] for c in "abcdefghijklmnopqrstuvwxyz"])
        letters = lower[torch.randint(26, (half,), generator=g)]
        ids = torch.cat([letters, letters]).view(1, -1)
        _, atts, _ = trace(model, ids)
        for l in range(L):
            a = atts[l]
            ind[l] += torch.stack([a[:, t, t - half + 1] for t in range(half, 2 * half)], 1).mean(1)
    return (prev / n_seqs).tolist(), (ind / n_seqs).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=OUT_DIR, help="folder for tiny-gpt.json / tiny-gpt.bin")
    ap.add_argument("--fixture", default=FIXTURE)
    ap.add_argument("--context", type=int, default=64, help="context_len (tiny_gpt.py default: 64)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto = Apple-silicon GPU (mps) or cuda if present, else cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)

    text = load_text(os.path.join(HERE, "shakespeare.txt"))
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = Config()          # n_embd 128, n_head 4, n_layer 4 (tiny_gpt.py defaults)
    cfg.dropout = 0.0
    cfg.context_len = args.context
    cfg.vocab_size = len(chars)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = GPT(cfg)
    gpt2_init(model, cfg.n_layer)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab {cfg.vocab_size}, params {n_params} ({n_params/1e6:.3f}M)")

    def get_batch(split, bs=args.batch):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - cfg.context_len - 1, (bs,))
        x = torch.stack([d[i:i + cfg.context_len] for i in ix])
        y = torch.stack([d[i + 1:i + cfg.context_len + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def eval_loss(split, iters=100):
        model.eval()
        g_state = torch.get_rng_state()
        torch.manual_seed(42)  # the same eval batches every time
        losses = [model(*get_batch(split))[1].item() for _ in range(iters)]
        torch.set_rng_state(g_state)
        model.train()
        return sum(losses) / len(losses)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = 100

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        p = (step - warm) / max(1, args.steps - warm)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    t0 = time.time()
    for step in range(args.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"step {step:5d}  train {eval_loss('train', 20):.3f}  val {eval_loss('val', 20):.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    train_secs = time.time() - t0
    train_device = device

    # round every weight to float16, exactly what the browser will load; evaluate on the CPU
    device = "cpu"
    model.to(device)
    model.eval()
    with torch.no_grad():
        for _, p in tensor_list(model):
            p.copy_(p.half().float())
    train_loss, val_loss = eval_loss("train"), eval_loss("val")
    print(f"final (fp16 weights): train {train_loss:.4f}  val {val_loss:.4f}  in {train_secs:.0f}s")

    # ---- weights ----
    os.makedirs(args.out, exist_ok=True)
    tensors, chunks, offset = [], [], 0
    for name, p in tensor_list(model):
        arr = p.detach().numpy().astype("<f2")
        tensors.append({"name": name, "shape": list(arr.shape), "offset": offset, "length": int(arr.size)})
        chunks.append(arr.tobytes())
        offset += arr.size
    with open(os.path.join(args.out, "tiny-gpt.bin"), "wb") as f:
        f.write(b"".join(chunks))

    prev, ind = head_scores(model, stoi, text[n:])
    with torch.no_grad():
        start = torch.zeros((1, 1), dtype=torch.long)
        torch.manual_seed(7)
        sample = "".join(chars[i] for i in model.generate(start, 300, temperature=0.8)[0].tolist())
    print("\nsample:\n" + sample)
    print("\nprevious-token score (layer x head):")
    for row in prev:
        print("  " + "  ".join(f"{v:.2f}" for v in row))
    print("induction score (layer x head):")
    for row in ind:
        print("  " + "  ".join(f"{v:.2f}" for v in row))

    manifest = {
        "format": "tiny-gpt/1",
        "dtype": "float16",
        "weights": "tiny-gpt.bin",
        "config": {"vocab_size": cfg.vocab_size, "context_len": cfg.context_len, "n_embd": cfg.n_embd,
                   "n_head": cfg.n_head, "n_layer": cfg.n_layer, "tied_head": True, "gelu": "erf"},
        "vocab": chars,
        "n_params": n_params,
        "training": {
            "data": "tiny Shakespeare (karpathy/char-rnn), first 90% of characters for training, last 10% for validation",
            "data_chars": len(text), "steps": args.steps, "batch_size": args.batch, "lr": args.lr,
            "schedule": "100 warmup steps, cosine decay to 10%", "optimizer": "AdamW, weight_decay 0.01, grad clip 1.0",
            "init": "GPT-2 style: N(0, 0.02), residual projections scaled by 1/sqrt(2*n_layer)",
            "dropout": 0.0, "device": train_device, "train_seconds": round(train_secs), "torch": torch.__version__,
            "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
            "loss_note": "mean cross-entropy (nats per character) over 100 random batches of 64 x 64 characters, float16 weights",
        },
        "python_head_scores": {"previous_token": prev, "induction": ind},
        "tensors": tensors,
    }
    with open(os.path.join(args.out, "tiny-gpt.json"), "w") as f:
        json.dump(manifest, f)

    # ---- fixture for the TypeScript test ----
    cases = []
    for prompt in ["ROMEO:\nWhat is a cat?", "First Citizen:\nBefore we proceed any further, hear me speak."]:
        ids = torch.tensor([[stoi[c] for c in prompt]])
        logits, atts, lens = trace(model, ids)
        cases.append({
            "prompt": prompt,
            "logits": [[round(v, 5) for v in row] for row in logits.tolist()],
            "attn_l1_h2": [[round(v, 6) for v in row] for row in atts[1][2].tolist()],
            "lens_last": [[round(v, 5) for v in lens[l][-1].tolist()] for l in range(cfg.n_layer)],
        })
    os.makedirs(os.path.dirname(args.fixture), exist_ok=True)
    with open(args.fixture, "w") as f:
        json.dump({"note": "written by phase3-transformers/export_checkpoint.py", "cases": cases}, f)
    size = os.path.getsize(os.path.join(args.out, "tiny-gpt.bin")) + os.path.getsize(os.path.join(args.out, "tiny-gpt.json"))
    print(f"\nwrote {args.out} ({size/1e6:.2f} MB) and {args.fixture}")


if __name__ == "__main__":
    main()
