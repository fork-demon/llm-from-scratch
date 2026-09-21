"""Part 10, the PyTorch and Hugging Face bridge. No test here needs the network:
the model-based tests build a tiny random GPT-2 from a config, and skip if the
optional libraries are not installed."""
import pytest

torch = pytest.importorskip("torch")

import inspect_hf_model as ins  # noqa: E402
import lora_finetune_hf as lf  # noqa: E402


# ---------- the config -> parameter count formula ----------
@pytest.mark.parametrize("name, n_embd, n_layer, expected", [
    ("gpt2", 768, 12, 124_439_808),
    ("gpt2-medium", 1024, 24, 354_823_168),
    ("gpt2-large", 1280, 36, 774_030_080),
    ("gpt2-xl", 1600, 48, 1_557_611_200),
])
def test_gpt2_family_param_counts(name, n_embd, n_layer, expected):
    assert ins.gpt2_param_count(50257, 1024, n_embd, n_layer) == expected


def test_untied_head_adds_one_more_table():
    tied = ins.gpt2_param_count(50257, 1024, 768, 12, tied=True)
    untied = ins.gpt2_param_count(50257, 1024, 768, 12, tied=False)
    assert untied - tied == 50257 * 768


def test_formula_matches_a_real_gpt2_module_without_downloading():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(vocab_size=101, n_positions=32, n_embd=48, n_layer=3, n_head=4)
    model = transformers.GPT2LMHeadModel(cfg)
    real = sum(p.numel() for p in model.parameters())
    assert ins.gpt2_param_count(101, 32, 48, 3) == real


# ---------- the forward pass by hand ----------
def test_manual_forward_reproduces_hf_logits():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=101, n_positions=32, n_embd=48, n_layer=3, n_head=4)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    ids = torch.randint(0, 101, (1, 11))
    with torch.no_grad():
        theirs = model(ids).logits[0]
        ours = ins.manual_forward(model.state_dict(), ids[0], cfg.n_layer, cfg.n_head, cfg.layer_norm_epsilon)
    assert ours.shape == (11, 101)
    assert torch.allclose(theirs, ours, atol=1e-4)


def test_conv1d_stores_weights_transposed():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(vocab_size=101, n_positions=32, n_embd=48, n_layer=1, n_head=4)
    c_attn = transformers.GPT2LMHeadModel(cfg).transformer.h[0].attn.c_attn
    assert tuple(c_attn.weight.shape) == (48, 144)                     # (in, out)
    assert tuple(torch.nn.Linear(48, 144).weight.shape) == (144, 48)   # (out, in)
    x = torch.randn(5, 48)
    assert torch.allclose(c_attn(x), x @ c_attn.weight + c_attn.bias, atol=1e-6)


def test_gelu_new_is_close_to_exact_gelu():
    x = torch.linspace(-6, 6, 241)
    assert (ins.gelu_new(x) - torch.nn.functional.gelu(x)).abs().max() < 1e-3


def test_layer_norm_matches_torch():
    torch.manual_seed(1)
    x, w, b = torch.randn(4, 16), torch.randn(16), torch.randn(16)
    assert torch.allclose(ins.layer_norm(x, w, b, 1e-5), torch.nn.functional.layer_norm(x, (16,), w, b, 1e-5), atol=1e-6)


# ---------- sampling ----------
def test_sample_next_top_k_1_is_argmax_and_top_k_limits_support():
    logits = torch.tensor([0.1, 2.0, -1.0, 1.9, 0.5])
    g = torch.Generator().manual_seed(0)
    assert all(ins.sample_next(logits, 1.0, top_k=1, generator=g) == 1 for _ in range(20))
    assert {ins.sample_next(logits, 5.0, top_k=2, generator=g) for _ in range(200)} == {1, 3}


# ---------- loss masking ----------
def test_build_example_masks_the_prompt_only():
    ids, labels = lf.build_example([5, 6, 7], [8, 9], eos_id=0)
    assert ids == [5, 6, 7, 8, 9, 0]
    assert labels == [-100, -100, -100, 8, 9, 0]      # the EOS is graded: the model must learn to stop


def test_build_example_truncates_both_the_same_way():
    ids, labels = lf.build_example([1] * 10, [2] * 10, eos_id=0, max_len=12)
    assert len(ids) == len(labels) == 12


def test_collate_masks_padding_by_position_not_by_id():
    a = lf.build_example([5, 6, 7], [8], eos_id=0)        # length 5
    b = lf.build_example([5], [8], eos_id=0)              # length 3, padded with the EOS id
    ids, mask, labels = lf.collate([a, b], pad_id=0)
    assert ids.tolist() == [[5, 6, 7, 8, 0], [5, 8, 0, 0, 0]]
    assert mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert labels.tolist() == [[-100, -100, -100, 8, 0], [-100, 8, 0, -100, -100]]


def test_masked_loss_ignores_prompt_and_padding():
    torch.manual_seed(2)
    logits = torch.randn(1, 5, 11)
    labels = torch.tensor([[-100, -100, 3, 4, -100]])
    # only positions 1 and 2 are graded: they predict tokens 3 and 4
    by_hand = -(torch.log_softmax(logits[0, 1], -1)[3] + torch.log_softmax(logits[0, 2], -1)[4]) / 2
    assert torch.isclose(lf.masked_next_token_loss(logits, labels), by_hand, atol=1e-6)
    # changing logits at ungraded positions must not change the loss
    other = logits.clone()
    other[0, 0] += 5.0
    other[0, 3:] -= 5.0
    assert torch.isclose(lf.masked_next_token_loss(other, labels), by_hand, atol=1e-6)


def test_masked_loss_equals_the_library_loss():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(3)
    cfg = transformers.GPT2Config(vocab_size=101, n_positions=32, n_embd=48, n_layer=2, n_head=4)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    ids, mask, labels = lf.collate([lf.build_example([5, 6, 7, 9], [8, 3], 0), lf.build_example([5], [8], 0)], pad_id=0)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
    assert torch.isclose(out.loss, lf.masked_next_token_loss(out.logits, labels), atol=1e-5)


def test_training_tickets_follow_the_format_and_do_not_leak_into_held_out():
    for _, team, pr in lf.TRAIN + lf.HELD_OUT:
        assert lf.FORMAT.match(lf.make_response(team, pr))
    assert not {t for t, _, _ in lf.TRAIN} & {t for t, _, _ in lf.HELD_OUT}


# ---------- peft smoke test (skipped when peft is not installed) ----------
def test_peft_lora_on_a_tiny_gpt2_trains_only_the_adapter():
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    torch.manual_seed(4)
    # dropout off, so that two forward passes of the same weights give the same logits
    cfg = transformers.GPT2Config(vocab_size=101, n_positions=32, n_embd=48, n_layer=2, n_head=4,
                                  resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    base = transformers.GPT2LMHeadModel(cfg)
    ids = torch.randint(0, 101, (2, 9))
    with torch.no_grad():
        before = base(ids).logits.clone()
    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn", "attn.c_proj"],
                             fan_in_fan_out=True, bias="none", task_type="CAUSAL_LM")
    model = peft.get_peft_model(base, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == lf.expected_lora_params(n_layer=2, n_embd=48, r=4)
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, before, atol=1e-6)   # B starts at zero
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        loss = lf.masked_next_token_loss(model(ids).logits, ids)
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        assert not torch.allclose(model(ids).logits, before, atol=1e-6)   # the adapter moved
        with model.disable_adapter():
            assert torch.allclose(model(ids).logits, before, atol=1e-6)   # the base did not
