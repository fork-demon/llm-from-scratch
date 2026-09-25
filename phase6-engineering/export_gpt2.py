"""
Export GPT-2 small (124M, openai-community/gpt2) for the course's in-browser GPT-2 Explainer.

What it writes to course/public/models/gpt2/:
  gpt2-int8-<n>.bin   the weights, split into chunks of at most 40 MB (GitHub-friendly, no git LFS)
  merges.bin          the tokenizer: 50,000 BPE merges as pairs of uint16 token ids (little-endian)
  manifest.json       config, tensor table (chunk, byte offset, shape, dtype) and chunk hashes
  README.md           where the weights come from and their licence

The quantization is the one taught in the "Making models cheaper" lesson:
  every weight MATRIX is stored as int8, one scale per output row (symmetric: scale = max|row| / 127,
  q = round(w / scale)), scales in float32. LayerNorm parameters and biases stay float32.
  One exception, measured rather than assumed: 8 "outlier" columns of the tied embedding/head wte stay
  float32 (see outlier_columns); without them int8 flips GPT-2's top-1 prediction on 3 of 9 test prompts.
  Linear weights are stored as (out, in), i.e. PyTorch nn.Linear layout. GPT-2's Hugging Face
  checkpoint uses Conv1D with (in, out), so those are transposed first.

With --fixtures it also writes the test fixtures the TypeScript engine and tokenizer are checked against:
  course/src/lib/__fixtures__/gpt2Tokens.fixture.json   tiktoken ids for varied strings
  course/src/lib/__fixtures__/gpt2Forward.fixture.json  PyTorch activations on the SAME int8-dequantised
                                                        weights, plus fp32 GPT-2's top-1 predictions

Run:  python phase6-engineering/export_gpt2.py --fixtures
Needs: torch, transformers (and tiktoken for --fixtures).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'course', 'public', 'models', 'gpt2')
FIX = os.path.join(REPO, 'course', 'src', 'lib', '__fixtures__')
CHUNK_LIMIT = 40_000_000  # bytes; every chunk stays at or under this

LICENSE = """Modified MIT License

Software Copyright (c) 2019 OpenAI

We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
The above copyright notice and this permission notice need not be included
with content created by the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
"""


def load_gpt2():
    from transformers import GPT2LMHeadModel
    for name in ('openai-community/gpt2', 'gpt2'):
        try:
            return GPT2LMHeadModel.from_pretrained(name, attn_implementation='eager').eval(), name
        except Exception as e:  # noqa: BLE001  (try the short alias next)
            last = e
    raise last


def quantize_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8, one scale per row. Returns (int8 matrix, float32 scales)."""
    w = w.astype(np.float32)
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale[:, None]


def collect_tensors(model):
    """(name, kind, array): kind 'q' = int8 matrix (out, in), 'f' = float32 vector."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    # wte is both the input embedding and the output head (tied). It gets the outlier treatment: see outlier_columns.
    out = [('wte', 'qo', sd['transformer.wte.weight']), ('wpe', 'q', sd['transformer.wpe.weight'])]
    n_layer = model.config.n_layer
    for i in range(n_layer):
        p = f'transformer.h.{i}.'
        b = f'h.{i}.'
        out += [
            (b + 'ln_1.weight', 'f', sd[p + 'ln_1.weight']), (b + 'ln_1.bias', 'f', sd[p + 'ln_1.bias']),
            (b + 'attn.c_attn.weight', 'q', sd[p + 'attn.c_attn.weight'].T), (b + 'attn.c_attn.bias', 'f', sd[p + 'attn.c_attn.bias']),
            (b + 'attn.c_proj.weight', 'q', sd[p + 'attn.c_proj.weight'].T), (b + 'attn.c_proj.bias', 'f', sd[p + 'attn.c_proj.bias']),
            (b + 'ln_2.weight', 'f', sd[p + 'ln_2.weight']), (b + 'ln_2.bias', 'f', sd[p + 'ln_2.bias']),
            (b + 'mlp.c_fc.weight', 'q', sd[p + 'mlp.c_fc.weight'].T), (b + 'mlp.c_fc.bias', 'f', sd[p + 'mlp.c_fc.bias']),
            (b + 'mlp.c_proj.weight', 'q', sd[p + 'mlp.c_proj.weight'].T), (b + 'mlp.c_proj.bias', 'f', sd[p + 'mlp.c_proj.bias']),
        ]
    out += [('ln_f.weight', 'f', sd['transformer.ln_f.weight']), ('ln_f.bias', 'f', sd['transformer.ln_f.bias'])]
    return out


N_OUTLIERS = 8
CALIBRATION = [
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "import numpy as np\nx = np.zeros((3, 4))\nprint(x.shape)",
    "The committee will meet on Tuesday to discuss the budget for next year.",
]


@torch.no_grad()
def outlier_columns(model, k: int = N_OUTLIERS) -> list[int]:
    """The k dimensions where GPT-2's final hidden state (after ln_f) is largest on some calibration text.

    Why: the head computes logit[v] = h . wte[v]. A handful of h's 768 dimensions are 100 to 600 times
    larger than the median one (a known quirk of GPT-2). Per-row int8 rounding error in wte, multiplied by
    those huge values, is enough to swap the top-1 token on some prompts. So those k columns of wte are kept
    in float32 (the "outlier features" idea of LLM.int8(), Dettmers et al. 2022) and the rest is int8.
    Cost: 50257 x 8 x 4 bytes = 1.6 MB. The calibration text is NOT the text the tests check.
    """
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    mags = []
    for text in CALIBRATION:
        out = model(torch.tensor([tok.encode(text)]), output_hidden_states=True)
        mags.append(out.hidden_states[-1][0].abs())  # the last hidden state already has ln_f applied
    mean = torch.cat(mags).mean(0)
    cols = torch.argsort(-mean)[:k].tolist()
    print('wte outlier columns (float32):', cols, 'mean |h|:', [round(float(mean[c]), 1) for c in cols], 'median:', round(float(mean.median()), 2))
    return sorted(cols)


def pad4(n: int) -> int:
    return (n + 3) & ~3


def export(model, source: str) -> dict:
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.startswith('gpt2-int8-') and f.endswith('.bin'):
            os.remove(os.path.join(OUT, f))
    cfg = model.config
    chunks: list[bytearray] = [bytearray()]
    table = []
    dequant = {}  # name -> float32 array the browser will effectively use (for the fixtures)
    cols = outlier_columns(model)

    def put(blob: bytes) -> tuple[int, int]:
        """Append bytes (4-byte aligned) to the current chunk, opening a new one if it would not fit."""
        if len(chunks[-1]) and pad4(len(chunks[-1])) + len(blob) > CHUNK_LIMIT:
            chunks.append(bytearray())
        c = chunks[-1]
        c.extend(b'\0' * (pad4(len(c)) - len(c)))
        off = len(c)
        c.extend(blob)
        return len(chunks) - 1, off

    for name, kind, arr in collect_tensors(model):
        if kind in ('q', 'qo'):
            keep = None
            if kind == 'qo':
                arr = arr.astype(np.float32).copy()
                keep = np.ascontiguousarray(arr[:, cols])
                arr[:, cols] = 0  # the int8 part no longer has to cover the outlier columns
            q, s = quantize_rows(arr)
            # keep a tensor's int8 rows and its scales in the same chunk
            need = pad4(q.nbytes) + s.nbytes
            if len(chunks[-1]) and pad4(len(chunks[-1])) + need > CHUNK_LIMIT:
                chunks.append(bytearray())
            ci, off = put(q.tobytes())
            _, soff = put(s.tobytes())
            table.append({'name': name, 'dtype': 'int8', 'shape': list(q.shape), 'chunk': ci, 'offset': off, 'scale_offset': soff})
            dequant[name] = dequantize(q, s)
            if keep is not None:
                ci, off = put(keep.tobytes())
                table.append({'name': name + '.outliers', 'dtype': 'float32', 'shape': list(keep.shape), 'chunk': ci, 'offset': off, 'columns': cols})
                dequant[name][:, cols] = keep
        else:
            v = np.ascontiguousarray(arr, dtype=np.float32)
            ci, off = put(v.tobytes())
            table.append({'name': name, 'dtype': 'float32', 'shape': list(v.shape), 'chunk': ci, 'offset': off})
            dequant[name] = v

    files = []
    for i, c in enumerate(chunks):
        fname = f'gpt2-int8-{i}.bin'
        with open(os.path.join(OUT, fname), 'wb') as f:
            f.write(c)
        files.append({'file': fname, 'bytes': len(c), 'sha256': hashlib.sha256(c).hexdigest()})
        assert len(c) <= CHUNK_LIMIT

    merges_bytes = write_merges()
    n_params = sum(p.numel() for p in model.parameters())
    man = {
        'format': 'llm-fp-gpt2-int8-v1',
        'source': f'{source} (Hugging Face), originally released by OpenAI',
        'license': 'Modified MIT License, github.com/openai/gpt-2 (see README.md in this folder)',
        'quantization': 'int8 symmetric, one float32 scale per output row (w = q * scale); LayerNorm and biases float32; '
                        f'wte keeps {len(cols)} outlier columns in float32 (wte.outliers)',
        'config': {'vocab_size': cfg.vocab_size, 'n_ctx': cfg.n_positions, 'n_embd': cfg.n_embd, 'n_head': cfg.n_head,
                   'n_layer': cfg.n_layer, 'layer_norm_epsilon': cfg.layer_norm_epsilon, 'activation': cfg.activation_function,
                   'tied_head': True},
        'n_params': n_params,
        'chunks': files,
        'total_bytes': sum(f['bytes'] for f in files) + merges_bytes,
        'tokenizer': {'file': 'merges.bin', 'bytes': merges_bytes, 'n_merges': 50000, 'eot_id': 50256,
                      'note': 'ids 0..255 are the 256 bytes in GPT-2 bytes_to_unicode order; id 256+i is merge i; 50256 is <|endoftext|>'},
        'tensors': table,
    }
    with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
        json.dump(man, f, indent=1)
    with open(os.path.join(OUT, 'README.md'), 'w') as f:
        f.write(README.format(total=man['total_bytes'] / 1e6, n=len(files)) + '\n```\n' + LICENSE + '```\n')
    print(f'wrote {len(files)} chunks: ' + ', '.join(f"{x['bytes'] / 1e6:.1f} MB" for x in files))
    print(f"total {man['total_bytes'] / 1e6:.1f} MB (weights + tokenizer), {n_params:,} parameters")
    return dequant


README = """# GPT-2 small, quantized for this course

These files are **GPT-2 small (124M parameters) by OpenAI**, released under the modified MIT license
from https://github.com/openai/gpt-2 (reproduced below). They were downloaded from the Hugging Face copy
`openai-community/gpt2` and **quantized for this course** by `phase6-engineering/export_gpt2.py`:
every weight matrix is stored as int8 with one float32 scale per output row; LayerNorm parameters and
biases stay float32. The token embedding (which is also the output head) keeps 8 "outlier" columns in
float32, because GPT-2's final hidden state is huge in those few dimensions. That changes the weights slightly (see the "Making models cheaper" lesson), so the
model here is a close approximation of GPT-2, not bit-identical to it.

- `gpt2-int8-*.bin`: {n} weight chunks (each at most 40 MB), described by `manifest.json`
- `merges.bin`: the GPT-2 byte-level BPE merges as uint16 id pairs (the vocabulary follows from them)
- total about {total:.0f} MB

Content generated with these weights is GPT-2 output. GPT-2 was trained on web text (WebText, 2019) and
can produce text that is wrong, biased or offensive.

## License of the original weights
"""


def write_merges() -> int:
    """50,000 merges as (left id, right id) uint16 pairs, in rank order. Merge i creates token 256 + i."""
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    vocab = tok.get_vocab()
    import transformers.models.gpt2.tokenization_gpt2 as t2
    slow = t2.GPT2Tokenizer.from_pretrained('gpt2')
    ranks = sorted(slow.bpe_ranks.items(), key=lambda kv: kv[1])
    buf = bytearray()
    for i, ((a, b), r) in enumerate(ranks):
        assert r == i and vocab[a + b] == 256 + i, 'GPT-2 vocab is not bytes-then-merges'
        buf += struct.pack('<HH', vocab[a], vocab[b])
    with open(os.path.join(OUT, 'merges.bin'), 'wb') as f:
        f.write(buf)
    return len(buf)


# ---------------------------------------------------------------- fixtures

TOKEN_STRINGS = [
    'Hello world',
    'The cat sat on the mat.',
    "  leading spaces,   runs of   spaces and trailing  ",
    "It's, we'll, they've, I'm, you'd, she's; don't!",
    'Numbers: 3.14159, 2026, 1,000,000 and -42%',
    'def add(a, b):\n    return a + b\n\n',
    'tabs\tand\nnewlines\r\n\r\nand nbsp',
    'naïve café résumé Zürich',
    'नमस्ते, आप कैसे हैं?',
    'ಕನ್ನಡ ಭಾಷೆ',
    '日本語のテキスト',
    'Emoji: 😀👍🏽 🇮🇳 ❤️',
    'URL https://marqa.tech/#/lesson/tokenization?x=1&y=2',
    '<|im_start|> angle <brackets> and {braces}',
    'What is a cat?',
    'ALL CAPS and MiXeD cAsE',
    '!!!???... ---- ____ ====',
    "'''quotes''' \"double\" `back`",
    ' ',
    '',
]

PROMPTS = [
    'The cat sat on the',
    'What is a cat? A cat is a small',
    'The capital of France is',
    'Riya opened her laptop and',
]


def token_fixture():
    import tiktoken
    from transformers import GPT2TokenizerFast
    enc = tiktoken.get_encoding('gpt2')
    hf = GPT2TokenizerFast.from_pretrained('gpt2')
    cases = []
    for s in TOKEN_STRINGS:
        ids = enc.encode(s, disallowed_special=())
        assert ids == hf.encode(s), f'tiktoken and transformers disagree on {s!r}'
        cases.append({'text': s, 'ids': ids})
    with open(os.path.join(FIX, 'gpt2Tokens.fixture.json'), 'w') as f:
        json.dump({'source': f'tiktoken {tiktoken.__version__} "gpt2" (checked equal to transformers GPT2TokenizerFast)', 'cases': cases}, f, ensure_ascii=False, indent=0)
    print(f'wrote {len(cases)} tokenizer cases')


def r(a, n=6):
    return [round(float(x), n) for x in np.asarray(a).ravel()]


@torch.no_grad()
def forward_fixture(model, dequant):
    import copy
    enc = __import__('tiktoken').get_encoding('gpt2')
    fp32 = model
    qm = copy.deepcopy(model)
    sd = qm.state_dict()
    # load the int8-dequantised weights into a copy of the model (Conv1D stores (in, out): transpose back)
    sd['transformer.wte.weight'].copy_(torch.from_numpy(dequant['wte']))
    sd['transformer.wpe.weight'].copy_(torch.from_numpy(dequant['wpe']))
    for i in range(qm.config.n_layer):
        for part in ('attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj'):
            sd[f'transformer.h.{i}.{part}.weight'].copy_(torch.from_numpy(dequant[f'h.{i}.{part}.weight'].T.copy()))
    assert torch.equal(qm.lm_head.weight, qm.transformer.wte.weight), 'head must stay tied'

    cases = []
    for prompt in PROMPTS:
        ids = enc.encode(prompt)
        x = torch.tensor([ids])
        cap = {}
        hooks = []
        for i, blk in enumerate(qm.transformer.h):
            hooks.append(blk.ln_1.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(f'ln1.{i}', o[0])))
            hooks.append(blk.attn.c_attn.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(f'qkv.{i}', o[0])))
            hooks.append(blk.attn.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(f'attn.{i}', o[0][0])))
            hooks.append(blk.mlp.act.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(f'act.{i}', o[0])))
            hooks.append(blk.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(f'block.{i}', o[0][0] if isinstance(o, tuple) else o[0])))
        out = qm(x, output_attentions=True)
        for h in hooks:
            h.remove()
        T = len(ids)
        logits = out.logits[0, -1].numpy()
        top = np.argsort(-logits)[:20]
        fixed = np.array([0, 11, 13, 262, 290, 1000, 3797, 25000, 50255, 50256])
        sel = np.concatenate([top, fixed])
        # logit lens: ln_f + tied head on each block's output at the last position
        W = qm.transformer.wte.weight
        lens = []
        for i in range(qm.config.n_layer):
            h = qm.transformer.ln_f(cap[f'block.{i}'][-1])
            p = torch.softmax(h @ W.T, -1)
            v, k = torch.topk(p, 5)
            lens.append({'ids': k.tolist(), 'p': r(v)})
        fp_logits = fp32(x).logits[0, -1]
        # greedy continuation with the quantised model (checks the KV-cache path in the engine)
        g = qm.generate(x, max_new_tokens=8, do_sample=False, pad_token_id=50256)[0, T:].tolist()
        D = 768
        cases.append({
            'prompt': prompt,
            'ids': ids,
            'logits': {'index': sel.tolist(), 'value': r(logits[sel], 5), 'mean': float(logits.mean()), 'std': float(logits.std()),
                       'argmax': int(logits.argmax())},
            'fp32_top1': int(fp_logits.argmax()),
            'fp32_top5': torch.topk(fp_logits, 5).indices.tolist(),
            'ln1_l0_last16': r(cap['ln1.0'][-1, :16]),
            'q_l3_h5_last': r(cap['qkv.3'][-1, 5 * 64:6 * 64]),
            'k_l3_h5_first': r(cap['qkv.3'][0, D + 5 * 64:D + 6 * 64]),
            'v_l3_h5_last': r(cap['qkv.3'][-1, 2 * D + 5 * 64:2 * D + 6 * 64]),
            'attn_l0_h0_last': r(out.attentions[0][0, 0, -1]),
            'attn_l5_h1_last': r(out.attentions[5][0, 1, -1]),
            'attn_l11_h7_all': r(out.attentions[11][0, 7]),
            'attnout_l7_last16': r(cap['attn.7'][-1, :16]),
            'act_l2_last16': r(cap['act.2'][-1, :16]),
            'act_l2_last_stats': {'mean': float(cap['act.2'][-1].mean()), 'max': float(cap['act.2'][-1].max()),
                                  'frac_pos': float((cap['act.2'][-1] > 0).float().mean())},
            'block_last16': [r(cap[f'block.{i}'][-1, :16], 5) for i in range(qm.config.n_layer)],
            'lnf_last16': r(qm.transformer.ln_f(cap[f'block.{qm.config.n_layer - 1}'][-1])[:16]),
            'lens_top5': lens,
            'greedy8': g,
        })
        print(f'{prompt!r}: int8 top1 {enc.decode([int(logits.argmax())])!r}, fp32 top1 {enc.decode([int(fp_logits.argmax())])!r}, greedy {enc.decode(g)!r}')
    with open(os.path.join(FIX, 'gpt2Forward.fixture.json'), 'w') as f:
        json.dump({'note': 'PyTorch (transformers GPT2LMHeadModel, eager attention, fp32) on the int8-dequantised weights of manifest.json',
                   'torch': torch.__version__, 'cases': cases}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixtures', action='store_true', help='also write the TypeScript test fixtures')
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    model, source = load_gpt2()
    dequant = export(model, source)
    if args.fixtures:
        os.makedirs(FIX, exist_ok=True)
        token_fixture()
        forward_fixture(model, dequant)


if __name__ == '__main__':
    main()
