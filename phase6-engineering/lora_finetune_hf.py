"""
Part 10 -- A real LoRA fine-tune, with the libraries people actually use.

  pip install transformers peft
  python lora_finetune_hf.py --quick     # ~2-4 min on a laptop CPU
  python lora_finetune_hf.py             # 300 steps, a few minutes more

In phase4-modern-llms/finetune_tiny_gpt.py you wrote LoRALinear by hand and
wrapped the attention layers of your own GPT. Here the SAME idea runs on the
real GPT-2 small (124M parameters, MIT licence, no login needed), using
Hugging Face `transformers` for the model and `peft` for the adapter.

What you will see:
  1. a tiny instruction dataset with a strange output format, written below
  2. prompt/response LOSS MASKING: the model is graded only on the response
  3. LoraConfig: r, lora_alpha, target_modules, and GPT-2's fan_in_fan_out trap
  4. trainable versus total parameters, checked against your own arithmetic
  5. before/after generations, and the loss on tickets the model never saw
  6. the adapter on disk (about 1.8 MB) next to the base model (about 500 MB)
  7. reload the adapter onto a fresh base model, then merge it into the weights

Options you will meet in every real training script, available here as flags:
  --grad-accum N   add up gradients over N small batches before each optimizer step
  --bf16           bfloat16 autocast for the forward pass (CUDA GPUs only; see main())

No API key. If `transformers` or `peft` is missing, or the model cannot be
downloaded, the script says what to do and exits cleanly.
"""
import argparse
import os
import re
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

# The fast tokenizer uses threads; saving the adapter can fork a process, and the
# library then prints a long warning. We tokenize a few dozen lines: no threads needed.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "gpt2"
IGNORE = -100   # the label value that cross_entropy skips (PyTorch's ignore_index default)

# ----------------------------------------------------------------------
# 1. The data. A support-ticket triage task with a DISTINCTIVE output format:
#        <<team:payments|priority:P1>>
#    Base GPT-2 has never seen this format, so the change is easy to see.
#    Small and synthetic on purpose: we are measuring the mechanism, not
#    building a product. 40 training tickets, 8 held out.
# ----------------------------------------------------------------------
TRAIN = [
    ("Customers are being charged twice for one order.", "payments", "P1"),
    ("The refund I asked for last week has not arrived.", "payments", "P2"),
    ("Please add my company VAT number to the invoice.", "payments", "P3"),
    ("Card payments fail with error 402 for everyone.", "payments", "P1"),
    ("My invoice shows the wrong billing address.", "payments", "P3"),
    ("I was billed after cancelling my subscription.", "payments", "P2"),
    ("Checkout rejects every credit card since this morning.", "payments", "P1"),
    ("Can I switch from monthly to yearly billing?", "payments", "P3"),
    ("The discount code was not applied to my payment.", "payments", "P2"),
    ("All payouts to sellers are stuck since yesterday.", "payments", "P1"),
    ("Nobody can log in, the login page returns an error.", "auth", "P1"),
    ("I did not get the password reset email.", "auth", "P2"),
    ("Please change the email address on my account.", "auth", "P3"),
    ("Two factor codes are rejected for all users.", "auth", "P1"),
    ("My account got locked after three wrong passwords.", "auth", "P2"),
    ("How do I turn on single sign on for my team?", "auth", "P3"),
    ("Users are logged out every few minutes.", "auth", "P1"),
    ("The reset password link says it has expired.", "auth", "P2"),
    ("I want to delete an old login from my profile.", "auth", "P3"),
    ("Login with Google stopped working for the whole company.", "auth", "P1"),
    ("The API returns 500 errors for every request.", "infra", "P1"),
    ("The nightly backup job failed twice this week.", "infra", "P2"),
    ("Please increase the disk quota on our staging server.", "infra", "P3"),
    ("The whole site is down in the Europe region.", "infra", "P1"),
    ("Response times doubled after the last deploy.", "infra", "P2"),
    ("Can we get a second staging environment?", "infra", "P3"),
    ("The database is out of connections and requests time out.", "infra", "P1"),
    ("The deploy pipeline is slow on Monday mornings.", "infra", "P2"),
    ("Rotate the old TLS certificate before next month.", "infra", "P3"),
    ("Every server in the cluster is out of memory.", "infra", "P1"),
    ("The save button does nothing on the settings page.", "ui", "P2"),
    ("The logo looks blurry on large screens.", "ui", "P3"),
    ("The checkout page is blank for all mobile users.", "ui", "P1"),
    ("Dark mode makes the menu text unreadable.", "ui", "P2"),
    ("There is a typo in the footer of the home page.", "ui", "P3"),
    ("The dashboard crashes the browser for every customer.", "ui", "P1"),
    ("The date picker opens off screen on small phones.", "ui", "P2"),
    ("Please make the table columns sortable.", "ui", "P3"),
    ("The sign up form cannot be submitted by anyone.", "ui", "P1"),
    ("Tooltips overlap the chart on the reports page.", "ui", "P2"),
]

HELD_OUT = [
    ("Customers were charged but their orders show as unpaid.", "payments", "P1"),
    ("I need a copy of my invoice from March.", "payments", "P3"),
    ("The password reset email never arrives.", "auth", "P2"),
    ("No user can log in since the update.", "auth", "P1"),
    ("The API is down and returns errors for every customer.", "infra", "P1"),
    ("The backup job was slow last night.", "infra", "P2"),
    ("The submit button is hidden on mobile screens.", "ui", "P2"),
    ("There is a spelling mistake on the pricing page.", "ui", "P3"),
]

FORMAT = re.compile(r"^\s*<<team:(payments|auth|infra|ui)\|priority:(P[123])>>")


def make_prompt(ticket):
    """The prompt template. It must be IDENTICAL at training time and at
    inference time, character for character (see the training-pipeline lesson:
    a chat template is exactly this, agreed once and never varied)."""
    return f"### Ticket\n{ticket}\n### Triage\n"


def make_response(team, priority):
    return f"<<team:{team}|priority:{priority}>>"


# ----------------------------------------------------------------------
# 2. Loss masking. Pure functions: tests/test_engineering_bridge.py uses them.
# ----------------------------------------------------------------------
def build_example(prompt_ids, response_ids, eos_id, max_len=128):
    """One training example: input_ids = prompt + response + EOS.

    labels are a copy of input_ids with every PROMPT position set to -100, so
    the loss only grades the response (and the EOS, so the model learns to stop).
    Why: the ticket text is given. Spending gradient on predicting the user's
    own words teaches nothing about how to answer.
    """
    input_ids = (list(prompt_ids) + list(response_ids) + [eos_id])[:max_len]
    labels = ([IGNORE] * len(prompt_ids) + list(response_ids) + [eos_id])[:max_len]
    return input_ids, labels


def collate(examples, pad_id):
    """Pad a list of (input_ids, labels) on the RIGHT to one length.

    Padding gets attention_mask 0 (nobody attends to it) and label -100 (never
    graded). GPT-2 has no pad token, so pad_id is its EOS id; that is why we
    mask padding by POSITION here, not by comparing ids: the one real EOS at the
    end of each response must stay graded.
    """
    width = max(len(ids) for ids, _ in examples)
    input_ids, attention_mask, labels = [], [], []
    for ids, lab in examples:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [IGNORE] * pad)
    return (torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels))


def masked_next_token_loss(logits, labels):
    """Your loss from tiny_gpt.py, plus two details.

    The shift: position t predicts token t+1, so logits[:, :-1] are compared
    with labels[:, 1:]. (tiny_gpt.py did the same shift when it built y from
    x moved one step.) The mask: ignore_index skips every -100 label, and the
    mean is taken over graded tokens only.
    """
    V = logits.size(-1)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
                           ignore_index=IGNORE)


def expected_lora_params(n_layer, n_embd, r):
    """Your arithmetic from the fine-tuning lesson: an adapter on a d_in -> d_out
    layer trains r * (d_in + d_out) numbers. Per block we adapt c_attn
    (D -> 3D) and attn.c_proj (D -> D)."""
    D = n_embd
    return n_layer * (r * (D + 3 * D) + r * (D + D))


# ----------------------------------------------------------------------
# Plumbing
# ----------------------------------------------------------------------
def banner(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"    # Apple-silicon GPU
    return "cpu"


def encode_set(tok, rows):
    out = []
    for ticket, team, priority in rows:
        p = tok(make_prompt(ticket)).input_ids
        r = tok(make_response(team, priority)).input_ids
        out.append(build_example(p, r, tok.eos_token_id))
    return out


@torch.no_grad()
def eval_loss(model, examples, pad_id, device):
    model.eval()
    ids, mask, labels = (t.to(device) for t in collate(examples, pad_id))
    loss = masked_next_token_loss(model(input_ids=ids, attention_mask=mask).logits, labels)
    model.train()
    return loss.item()


@torch.no_grad()
def answer(model, tok, ticket, device, max_new_tokens=16):
    """Greedy decoding, one ticket at a time, returning only the new text."""
    model.eval()
    enc = tok(make_prompt(ticket), return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    model.train()
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


def score(model, tok, rows, device):
    """How many held-out answers have the exact format, and how many name the right team?"""
    fmt = team_ok = 0
    for ticket, team, _ in rows:
        m = FORMAT.match(answer(model, tok, ticket, device))
        fmt += bool(m)
        team_ok += bool(m and m.group(1) == team)
    return fmt, team_ok


def dir_size(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="100 steps instead of 300")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="sum gradients over N small batches before each optimizer step")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast for the forward pass (CUDA GPUs only)")
    ap.add_argument("--out", default=None, help="where to save the adapter (default: a temp dir)")
    args = ap.parse_args()

    try:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"Missing library: {e.name}\n"
              "This script needs Hugging Face transformers and peft:\n"
              "  pip install transformers peft\n"
              "Then run:  python phase6-engineering/lora_finetune_hf.py --quick")
        sys.exit(0)
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Could not download or load '{MODEL_ID}': {type(e).__name__}: {e}\n"
              "The first run needs internet access to huggingface.co (about 550 MB, then cached).")
        sys.exit(0)

    torch.manual_seed(1337)
    device = pick_device(args.device)
    steps = 100 if args.quick else 300
    batch_size = 8
    base.to(device)
    base_bytes = sum(p.numel() * p.element_size() for p in base.parameters())   # before any adapter
    train_set, held_set = encode_set(tok, TRAIN), encode_set(tok, HELD_OUT)
    pad_id = tok.eos_token_id   # GPT-2 has no pad token; see collate()

    # ------------------------------------------------------------------
    banner("1. One training example, and what the loss will grade")
    # ------------------------------------------------------------------
    ids, labels = train_set[0]
    print(f"  device: {device}    train examples: {len(train_set)}    held out: {len(held_set)}")
    print("  text:")
    for line in (make_prompt(TRAIN[0][0]) + make_response(*TRAIN[0][1:])).split("\n"):
        print(f"    {line}")
    graded = sum(l != IGNORE for l in labels)
    print(f"  {len(ids)} tokens; labels = -100 on the first {len(ids) - graded} (the prompt),"
          f" real ids on the last {graded} (response + EOS)")
    print("  token            label")
    for i, l in list(zip(ids, labels))[-graded - 2:]:
        print(f"    {tok.decode([i])!r:<17}{l}")

    # ------------------------------------------------------------------
    banner("2. Before: the base model on held-out tickets")
    # ------------------------------------------------------------------
    base_loss = eval_loss(base, held_set, pad_id, device)
    print(f"  held-out loss on response tokens: {base_loss:.3f}"
          f"   (perplexity {torch.exp(torch.tensor(base_loss)).item():.1f})")
    before = [answer(base, tok, t, device) for t, _, _ in HELD_OUT[:3]]
    for (t, _, _), a in zip(HELD_OUT[:3], before):
        print(f"  ticket: {t}\n    base model says: {a!r}")

    # ------------------------------------------------------------------
    banner("3. Attach LoRA with peft")
    # ------------------------------------------------------------------
    config = LoraConfig(
        r=args.rank,                  # the rank: width of the thin path, your `r`
        lora_alpha=2 * args.rank,     # scale = lora_alpha / r = 2, exactly your alpha / r
        target_modules=["c_attn", "attn.c_proj"],   # names from named_modules(): fused QKV + Wo,
                                                    # the same two layers apply_lora() wrapped
        fan_in_fan_out=True,          # GPT-2's Conv1D stores weight as (in, out); see inspect_hf_model.py
        lora_dropout=0.0,
        bias="none",                  # train no biases: detaching the adapter restores the base exactly
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, config)   # freezes every base weight, wraps the targets
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    mine = expected_lora_params(base.config.n_layer, base.config.n_embd, args.rank)
    print(f"  your arithmetic: {base.config.n_layer} blocks x (r x (D + 3D) + r x (D + D)) = {mine:,}")
    assert trainable == mine, (trainable, mine)
    wrapped = model.base_model.model.transformer.h[0].attn.c_attn
    print(f"  h.0.attn.c_attn is now a peft {type(wrapped).__name__} holding:")
    print(f"    base_layer.weight {tuple(wrapped.base_layer.weight.shape)}  frozen")
    print(f"    lora_A.default    {tuple(wrapped.lora_A['default'].weight.shape)}  trainable, random start")
    print(f"    lora_B.default    {tuple(wrapped.lora_B['default'].weight.shape)}  trainable, ZERO start")
    assert wrapped.lora_B["default"].weight.detach().abs().sum().item() == 0.0
    print(f"  held-out loss with the fresh adapter attached: {eval_loss(model, held_set, pad_id, device):.3f}"
          "  (unchanged: B = 0)")

    # ------------------------------------------------------------------
    banner(f"4. Train: {steps} steps of AdamW on {trainable:,} numbers")
    # ------------------------------------------------------------------
    # Only the adapter goes to the optimizer, so Adam's two moment buffers exist
    # for 0.4% of the model. That is where LoRA's memory saving comes from.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    # Gradient accumulation: call backward() on N small batches before one
    # opt.step(); the gradients add up, so it acts like a batch N times larger
    # while only one small batch of activations is in memory at a time.
    # Mixed precision (--bf16): autocast runs matrix multiplies in bfloat16, which
    # has float32's exponent range in half the bytes, while the weights being
    # trained and the optimizer state stay in float32.
    use_bf16 = args.bf16 and device == "cuda" and torch.cuda.is_bf16_supported()
    if args.bf16 and not use_bf16:
        print("  --bf16 ignored: it needs a CUDA GPU with bfloat16 support. CPUs without native")
        print("  bf16 emulate it, which is many times SLOWER than float32.")
    g = torch.Generator().manual_seed(1337)
    model.train()
    t0 = time.time()
    for step in range(steps + 1):
        for _ in range(args.grad_accum):
            pick = torch.randint(len(train_set), (batch_size,), generator=g).tolist()
            ids, mask, labels = (t.to(device) for t in collate([train_set[i] for i in pick], pad_id))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = masked_next_token_loss(logits.float(), labels) / args.grad_accum
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % (25 if args.quick else 50) == 0:
            print(f"  step {step:4d}   train batch loss {loss.item() * args.grad_accum:.3f}"
                  f"   held-out loss {eval_loss(model, held_set, pad_id, device):.3f}"
                  f"   ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------
    banner("5. After: same tickets, adapter attached")
    # ------------------------------------------------------------------
    tuned_loss = eval_loss(model, held_set, pad_id, device)
    print(f"  held-out loss on response tokens: {base_loss:.3f} -> {tuned_loss:.3f}")
    for (t, team, pr), b in zip(HELD_OUT[:3], before):
        print(f"  ticket: {t}\n    before: {b!r}\n    after:  {answer(model, tok, t, device)!r}"
              f"    (reference: {make_response(team, pr)})")
    fmt, team_ok = score(model, tok, HELD_OUT, device)
    print(f"  on all {len(HELD_OUT)} held-out tickets: exact format {fmt}/{len(HELD_OUT)},"
          f" right team {team_ok}/{len(HELD_OUT)}")
    print("  (8 tickets is a smoke test, not an eval: see the Evals lesson for how to size one.)")
    with model.disable_adapter():   # the thin path switched off: the base model, untouched
        off = eval_loss(model, held_set, pad_id, device)
    print(f"  adapter switched off: held-out loss {off:.3f} (the base model's {base_loss:.3f}: no base weight moved)")
    assert abs(off - base_loss) < 1e-3

    # ------------------------------------------------------------------
    banner("6. Save ONLY the adapter")
    # ------------------------------------------------------------------
    out_dir = args.out or tempfile.mkdtemp(prefix="ticket-lora-")
    model.save_pretrained(out_dir)   # writes adapter_config.json + adapter_model.safetensors
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f:<28}{os.path.getsize(os.path.join(out_dir, f)):>12,} bytes")
    print(f"  adapter on disk: {dir_size(out_dir) / 1e6:.2f} MB    base weights in fp32: {base_bytes / 1e6:.0f} MB"
          f"    ratio 1 : {base_bytes / dir_size(out_dir):.0f}")
    print(f"  trainable {trainable:,} of {total:,} parameters = {100 * trainable / total:.2f}%")
    print(f"  saved to {out_dir}")

    # ------------------------------------------------------------------
    banner("7. Reload onto a fresh base model, then merge")
    # ------------------------------------------------------------------
    probe = tok(make_prompt(HELD_OUT[0][0]), return_tensors="pt").to(device)
    with torch.no_grad():
        model.eval()
        want = model(**probe).logits
        fresh = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
        reloaded = PeftModel.from_pretrained(fresh, out_dir).eval()
        same = torch.allclose(reloaded(**probe).logits, want, atol=1e-4)
        print(f"  fresh base + saved adapter reproduces the tuned logits: {same}")
        assert same
        # merge_and_unload() adds scale * (the low-rank product) INTO each frozen
        # weight and removes the wrappers: a plain GPT-2 again, no extra latency,
        # but the adapter can no longer be detached or swapped.
        merged = reloaded.merge_and_unload()
        kind = type(merged.transformer.h[0].attn.c_attn).__name__
        diff = (merged(**probe).logits - want).abs().max().item()
        print(f"  after merge_and_unload(): c_attn is a plain {kind} again;"
              f" largest logit difference {diff:.1e}")
        assert diff < 1e-2
    print("\n  Same maths as your LoRALinear. The library adds bookkeeping: finding layers by")
    print("  name, handling Conv1D, saving, loading, merging, and switching adapters.")


if __name__ == "__main__":
    main()
