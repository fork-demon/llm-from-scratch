import { PAISA_PAL } from '../project/paisaPal'
import type { CodeExerciseDef } from './types'

// The Paisa Pal project, first four pieces (Part 8): a confidence check, rotary positions,
// the chat template with its loss mask, and the distillation loss. Each starts from paisa_pal.py
// plus a small toy model given in the prelude, so none depends on another being finished.
const exercises: CodeExerciseDef[] = [
  /* ---------- why-llms-know: Confidence check ---------- */
  {
    id: 'why-llms-know-code-bot-confidence',
    lesson: 'why-llms-know',
    title: 'The bot’s confidence check',
    project: { piece: 'Confidence check' },
    prelude: PAISA_PAL + '\n' + `
# ---- given: a toy bot ----
# For a few questions, its logits over six candidate answers: made-up numbers of the kind a trained
# model produces. "14 days" is the refund policy the vendor bot invented in this lesson's story.
ANSWERS = ["3 working days", "48 hours", "100 rupees", "30 minutes", "7 working days", "14 days"]
BOT_LOGITS = {
    QUESTIONS[0][0]: [4.5, 0.9, 0.2, 0.3, 1.1, 0.4],    # failed UPI payment: when is the refund?
    QUESTIONS[2][0]: [0.8, 3.9, 1.2, 0.5, 0.6, 0.3],    # how long does cashback take?
    QUESTIONS[10][0]: [0.5, 0.4, 0.7, 0.3, 0.5, 0.6],   # the biryani question
    QUESTIONS[11][0]: [0.9, 0.3, 1.4, 0.2, 0.6, 0.8],   # the share price
    "what is the paisa pal refund policy": [0.7, 0.2, 0.3, 0.1, 0.5, 4.6],
}
NOT_SURE = "I am not sure. Let me connect you to a human."
`,
    prompt: `The vendor bot always answers, because softmax always produces a winner. Riya adds a check before the bot speaks.

- \`confidence(logits)\` turns the logits into probabilities (\`softmax\` is in \`paisa_pal.py\`) and returns \`(best_index, top_probability)\`.
- \`reply(question, threshold=0.6)\` looks up \`BOT_LOGITS[question]\`. If the top probability is at least \`threshold\`, it returns that entry of \`ANSWERS\`. Otherwise it returns \`NOT_SURE\`.

Then read the last test carefully: it is the point of the lesson.`,
    starter: `def confidence(logits):
    """Return (best_index, top_probability) for one list of logits."""
    ...

def reply(question, threshold=0.6):
    """The answer from ANSWERS if the bot is confident enough, else NOT_SURE."""
    ...

for q in BOT_LOGITS:
    print(q, "->", reply(q))
`,
    solution: `def confidence(logits):
    """Return (best_index, top_probability) for one list of logits."""
    p = softmax(logits)               # always sums to 1, whatever the question
    best = int(np.argmax(p))
    return best, float(p[best])

def reply(question, threshold=0.6):
    """The answer from ANSWERS if the bot is confident enough, else NOT_SURE."""
    best, top = confidence(BOT_LOGITS[question])
    if top >= threshold:
        return ANSWERS[best]
    return NOT_SURE

for q in BOT_LOGITS:
    print(q, "->", reply(q))
`,
    tests: [
      { name: 'the refund question: “3 working days” at 90.4%', code: `best, top = confidence(BOT_LOGITS[QUESTIONS[0][0]])\nassert best == 0, f"best index should be 0 (3 working days), got {best}"\nassert abs(top - 0.9043) < 1e-3, f"top probability should be about 0.9043, got {top}"` },
      { name: 'confident questions get their answer', code: `r1 = reply(QUESTIONS[0][0]); r2 = reply(QUESTIONS[2][0])\nassert r1 == "3 working days", f"refund question: got {r1!r}"\nassert r2 == "48 hours", f"cashback question: got {r2!r}"` },
      { name: 'the biryani question and the share price: the bot says it is not sure', code: `b = confidence(BOT_LOGITS[QUESTIONS[10][0]])[1]\nassert b < 0.25, f"the biryani logits are nearly flat, top probability should be about 0.20, got {b}"\nfor q in (QUESTIONS[10][0], QUESTIONS[11][0]):\n    r = reply(q)\n    assert r == NOT_SURE, f"{q!r}: got {r!r}, expected NOT_SURE"` },
      { name: 'the threshold is a setting: at 0.95 even the refund answer is held back', code: `r = reply(QUESTIONS[0][0], threshold=0.95)\nassert r == NOT_SURE, f"top probability 0.904 is below 0.95, so expected NOT_SURE, got {r!r}"` },
      { name: 'the point: the invented policy passes the check, because confidence is not truth', code: `best, top = confidence(BOT_LOGITS["what is the paisa pal refund policy"])\nr = reply("what is the paisa pal refund policy")\nassert abs(top - 0.9313) < 1e-3, f"top probability should be about 0.9313, got {top}"\nassert r == "14 days", f"got {r!r}. The invention (93.1%) is MORE confident than the true refund answer (90.4%): no threshold can separate them"` },
    ],
    hints: [
      '`p = softmax(logits)`, then `best = int(np.argmax(p))` and the top probability is `float(p[best])`.',
      'In `reply`, compare the top probability with `threshold` and index `ANSWERS` with `best`.',
      'Any threshold that lets “3 working days” (0.904) through also lets “14 days” (0.931) through. That is not a bug in your code.',
    ],
    explanation: `Softmax always sums to 1, so the bot always has a winner. The size of that winner is a real signal: on the biryani question the logits are nearly flat and the top answer gets only 20%, so the check catches it and hands over to a human.

But the last test is the lesson in miniature. The invented “14 days” policy comes out at 93.1%, more confident than the true refund answer at 90.4%, just like Austria and Berlin beat France and Paris in the lesson. A threshold on probability catches flat guesses, not confident inventions. That is why the bot will need retrieval from the help pages in Part 9.

Real systems read the same signal from a model’s token probabilities (sometimes averaged over the answer, or checked by sampling several answers and seeing whether they agree), and train the model to say “I am not sure” when it is likely. All of these reduce hallucination; none removes it.`,
  },

  /* ---------- modern-architecture: Rotary positions ---------- */
  {
    id: 'modern-architecture-code-bot-rope',
    lesson: 'modern-architecture',
    title: 'Rotary positions for the bot’s attention',
    project: { piece: 'Rotary positions' },
    prelude: PAISA_PAL + '\n' + `
# ---- given: one attention head of the bot (random, untrained weights: the point is the positions) ----
rng = np.random.default_rng(8)
HEAD_DIM = 8
VOCAB = sorted({w for q, _, _ in QUESTIONS for w in tokenize(q)} | set(tokenize("hi paisa pal team")))
EMB = {w: rng.normal(size=HEAD_DIM) for w in VOCAB}
Wq = rng.normal(size=(HEAD_DIM, HEAD_DIM)) / np.sqrt(HEAD_DIM)
Wk = rng.normal(size=(HEAD_DIM, HEAD_DIM)) / np.sqrt(HEAD_DIM)

def queries_and_keys(text):
    """One query row and one key row per token, with NO position information yet."""
    X = np.array([EMB[w] for w in tokenize(text)])
    return X @ Wq, X @ Wk
`,
    prompt: `The bot’s attention head has no idea where each word sits. Give it RoPE, the NumPy version of the lesson’s \`rope\`.

- \`rope(x, pos, base=10000.0)\`: \`x\` has shape (T, hd), \`pos\` has T positions. Treat each row as hd/2 pairs: (x[0], x[1]), (x[2], x[3]), ... Pair i turns at speed \`base ** (-2i / hd)\` per position, so its angle is \`pos × speed\`. Rotate each pair by its angle: [a, b] becomes [a·cos − b·sin, a·sin + b·cos]. Return the same shape, pairs in the same places.
- \`rope_scores(text, start=0)\`: get Q and K from \`queries_and_keys(text)\`, give the tokens positions start, start + 1, ..., rotate both Q and K, and return the (T, T) scores \`Q_rot @ K_rot.T / sqrt(hd)\`.

A customer who opens with “hi paisa pal team” pushes every word 4 places later. The scores between the words of the question must not change.`,
    starter: `def rope(x, pos, base=10000.0):
    """Rotate each pair (x[..., 2i], x[..., 2i+1]) by pos * base**(-2i/hd). Same shape out."""
    ...

def rope_scores(text, start=0):
    """(T, T) attention scores with RoPE applied to the queries and keys."""
    ...

# When both work, uncomment:
# q = QUESTIONS[0][0]
# print(rope_scores(q)[8, 3])                              # "deducted" looking at "failed"
# print(rope_scores("hi paisa pal team " + q)[12, 7])      # the same two words, 4 places later
`,
    solution: `def rope(x, pos, base=10000.0):
    """Rotate each pair (x[..., 2i], x[..., 2i+1]) by pos * base**(-2i/hd). Same shape out."""
    hd = x.shape[-1]
    i = np.arange(0, hd, 2)                             # 0, 2, 4, ...: one entry per pair
    speeds = base ** (-i / hd)                          # (hd/2,) one theta per pair
    ang = np.asarray(pos, float)[:, None] * speeds      # (T, hd/2) position x theta
    a, b = x[..., 0::2], x[..., 1::2]                   # the two halves of each pair
    out = np.stack([a * np.cos(ang) - b * np.sin(ang),
                    a * np.sin(ang) + b * np.cos(ang)], axis=-1)
    return out.reshape(x.shape)                         # pairs back in place

def rope_scores(text, start=0):
    """(T, T) attention scores with RoPE applied to the queries and keys."""
    Q, K = queries_and_keys(text)
    pos = np.arange(start, start + len(Q))
    Qr, Kr = rope(Q, pos), rope(K, pos)                 # v would NOT be rotated
    return Qr @ Kr.T / np.sqrt(Q.shape[-1])

q = QUESTIONS[0][0]
print(rope_scores(q)[8, 3])                              # "deducted" looking at "failed"
print(rope_scores("hi paisa pal team " + q)[12, 7])      # the same two words, 4 places later
`,
    tests: [
      { name: 'the lesson’s hand example: q = k = [1, 0], 30° per step, positions (5, 3) and (105, 103) both score 0.5', code: `import numpy as np\nx = np.array([[1.0, 0.0], [1.0, 0.0]])\nfor p in ([5, 3], [105, 103]):\n    r = rope(x, np.array(p) * np.pi / 6)   # hd = 2: one pair, speed 1 radian, so pos * pi/6 means 30 degrees a step\n    s = float(r[0] @ r[1])\n    assert abs(s - 0.5) < 1e-9, f"positions {p}: score {s}, expected cos(60 degrees) = 0.5"` },
      { name: 'each pair turns at its own speed, and pairs stay in place', code: `import numpy as np\nout = rope(np.array([[1.0, 0.0, 1.0, 0.0]]), np.array([1]))\nexp = np.array([[np.cos(1), np.sin(1), np.cos(0.01), np.sin(0.01)]])\nassert out.shape == (1, 4), f"shape {out.shape}"\nassert np.allclose(out, exp), f"got {out.round(4)}; pair 0 turns 1 radian, pair 1 turns 10000**(-2/4) = 0.01"` },
      { name: 'position 0 changes nothing, and rotation keeps every vector’s length', code: `import numpy as np\nQ, _ = queries_and_keys(QUESTIONS[3][0])\nassert np.allclose(rope(Q[:1], np.array([0])), Q[:1]), "at position 0 the angle is 0, so the vector must be unchanged"\nR = rope(Q, np.arange(len(Q)))\nassert np.allclose(np.linalg.norm(R, axis=1), np.linalg.norm(Q, axis=1)), "a rotation must not change lengths"` },
      { name: 'rope_scores: shape, and positions really change the scores', code: `import numpy as np\nq = QUESTIONS[0][0]\nS = rope_scores(q)\nQ, K = queries_and_keys(q)\nassert S.shape == (15, 15), f"15 tokens, so (15, 15); got {S.shape}"\nplain = Q @ K.T / np.sqrt(8)\nassert not np.allclose(S, plain), "the scores equal the no-position scores: were Q and K rotated?"\nassert np.allclose(np.diag(S), np.diag(plain)), "a token looking at itself sees no rotation (offset 0)"` },
      { name: 'the point: “hi paisa pal team” shifts the question by 4, and no score between its words changes', code: `import numpy as np\nq = QUESTIONS[0][0]\nS = rope_scores(q)\nS2 = rope_scores("hi paisa pal team " + q)\nassert np.allclose(S2[4:, 4:], S), "with the greeting in front, the question-to-question scores changed"\nassert np.allclose(rope_scores(q, start=1000), S), "starting 1,000 positions later changed the scores"\nassert abs(S2[12, 7] - S[8, 3]) < 1e-9, f"deducted -> failed: {S[8, 3]:.4f} vs {S2[12, 7]:.4f}"` },
    ],
    hints: [
      'Split the pairs with `a, b = x[..., 0::2], x[..., 1::2]`. The speeds are `base ** (-np.arange(0, hd, 2) / hd)`.',
      'The angles form a (T, hd/2) table: `pos[:, None] * speeds`. Then `np.stack([a*cos - b*sin, a*sin + b*cos], axis=-1).reshape(x.shape)` puts each rotated pair back where it was.',
      'In `rope_scores`, `pos = np.arange(start, start + len(Q))`, rotate both Q and K with the same positions, then `Qr @ Kr.T / np.sqrt(hd)`.',
    ],
    explanation: `Rotating the query by mθ and the key by nθ leaves them an angle of (n − m)θ apart, pair by pair, so the dot product depends only on the offset. That is why the greeting in front, or starting at position 1,000, leaves every score between the question’s own words untouched, and why a token looking at itself (offset 0) gets its plain score.

The fast pairs (1 radian per step) tell neighbours apart; the slow ones (0.01 per step here, far slower in a 128-number head) tell distant tokens apart. No table, no parameters.

A real model does exactly this in every layer and every head, on the GPU, for q and k only (v is never rotated), and long-context models slow the rotations down (a larger base, or YaRN) to reach 128 thousand tokens. With a KV cache, remember the lesson’s bug: the new token must be rotated by its true position, not by 0.`,
  },

  /* ---------- training-pipeline: Chat template and loss mask ---------- */
  {
    id: 'training-pipeline-code-bot-chat-template',
    lesson: 'training-pipeline',
    title: 'The bot’s chat template and loss mask',
    project: { piece: 'Chat template and loss mask' },
    prelude: PAISA_PAL + '\n' + `
# ---- given: the lesson's toy tokenizer and two conversations ----
import re
IM_START, IM_END = "<|im_start|>", "<|im_end|>"

def split_content(text):
    """Words keep their leading space; punctuation and newlines are their own tokens."""
    return re.findall(r"\\n| ?[A-Za-z0-9']+| ?[^\\sA-Za-z0-9']| +|\\s", text)

CAT_CHAT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a cat?"},
    {"role": "assistant", "content": "A cat is a small furry animal."},
]
SUPPORT_CHAT = [
    {"role": "system", "content": "You are Paisa Pal's support assistant. Answer only from the help pages."},
    {"role": "user", "content": QUESTIONS[0][0]},
    {"role": "assistant", "content": "The refund reaches your bank account within 3 working days."},
]
`,
    prompt: `Turn Paisa Pal support conversations into SFT training data, ChatML style. Each message becomes

\`<|im_start|>\`, the role, \`"\\n"\`, the tokens of \`split_content(content)\`, \`<|im_end|>\`, \`"\\n"\`

- \`render_chat(messages)\` returns \`(tokens, mask)\`: one flat token list, and one bool per token that is True only for an assistant message’s content tokens and its closing \`<|im_end|>\` (the bot must learn to stop). Headers, newlines, system and user tokens are False.
- \`generation_prompt(messages)\` returns the tokens for inference: the rendered chat plus an open assistant header \`<|im_start|>\`, \`"assistant"\`, \`"\\n"\`.
- \`masked_mean(token_losses, mask)\` averages the per-token losses over the True positions only.`,
    starter: `def render_chat(messages):
    """(tokens, mask): the flattened conversation, and True where the loss counts."""
    ...

def generation_prompt(messages):
    """The tokens the bot continues from at inference time."""
    ...

def masked_mean(token_losses, mask):
    """Mean loss over the positions where mask is True."""
    ...

# When render_chat works, uncomment:
# tokens, mask = render_chat(SUPPORT_CHAT)
# print(len(tokens), "tokens,", sum(mask), "graded:", "".join(t for t, m in zip(tokens, mask) if m))
`,
    solution: `def render_chat(messages):
    """(tokens, mask): the flattened conversation, and True where the loss counts."""
    tokens, mask = [], []
    for m in messages:
        body = split_content(m["content"])
        graded = m["role"] == "assistant"
        tokens += [IM_START, m["role"], "\\n"] + body + [IM_END, "\\n"]
        mask += [False] * 3 + [graded] * len(body) + [graded, False]   # <|im_end|> counts: learn to stop
    return tokens, mask

def generation_prompt(messages):
    """The tokens the bot continues from at inference time."""
    tokens, _ = render_chat(messages)
    return tokens + [IM_START, "assistant", "\\n"]    # open the bot's turn

def masked_mean(token_losses, mask):
    """Mean loss over the positions where mask is True."""
    m = np.asarray(mask, dtype=bool)
    return float(np.asarray(token_losses, dtype=float)[m].mean())

tokens, mask = render_chat(SUPPORT_CHAT)
print(len(tokens), "tokens,", sum(mask), "graded:", "".join(t for t, m in zip(tokens, mask) if m))
`,
    tests: [
      { name: 'the lesson’s cat conversation: 34 tokens, 9 graded', code: `tokens, mask = render_chat(CAT_CHAT)\nassert len(tokens) == len(mask), f"{len(tokens)} tokens but {len(mask)} mask entries"\nassert len(tokens) == 34, f"expected 34 tokens as in the lesson, got {len(tokens)}"\nassert sum(mask) == 9, f"expected 9 graded tokens (8 of the answer + <|im_end|>), got {sum(mask)}"\nassert tokens[:3] == [IM_START, "system", "\\n"], f"starts with {tokens[:3]}"` },
      { name: 'Paisa Pal: only the answer and its <|im_end|> are graded', code: `tokens, mask = render_chat(SUPPORT_CHAT)\ngraded = [t for t, m in zip(tokens, mask) if m]\nassert graded[-1] == IM_END, f"the last graded token should be <|im_end|>, got {graded[-1]!r}"\nassert "".join(graded[:-1]) == SUPPORT_CHAT[2]["content"], f"graded text: {''.join(graded[:-1])!r}"\nassert not mask[-1], "the final newline is not part of the answer"` },
      { name: 'two turns: both answers graded, the follow-up question not', code: `chat = SUPPORT_CHAT + [{"role": "user", "content": "and if it has not come after 3 days?"}, {"role": "assistant", "content": "Raise a dispute from the transaction screen."}]\ntokens, mask = render_chat(chat)\ngraded = "".join(t for t, m in zip(tokens, mask) if m)\nassert graded.count(IM_END) == 2, f"expected two graded <|im_end|>, got {graded!r}"\nassert "dispute" in graded and "refund" in graded, f"both answers must be graded, got {graded!r}"\nassert "come after" not in graded and "deducted" not in graded, f"a user message was graded: {graded!r}"` },
      { name: 'generation prompt: ends with an open assistant turn', code: `p = generation_prompt(SUPPORT_CHAT[:2])\nassert p[-3:] == [IM_START, "assistant", "\\n"], f"the prompt must end with the assistant header, it ends with {p[-3:]}"\nassert " refund" not in p, "the answer must not be in the prompt"\nassert p[:-3] == render_chat(SUPPORT_CHAT[:2])[0], "before the header, the prompt must be exactly the rendered chat (same template as training)"` },
      { name: 'the point: the loss ignores how badly the model would write the customer’s question', code: `import numpy as np\ntokens, mask = render_chat(SUPPORT_CHAT)\nlosses = np.where(mask, 0.5, 9.0)    # the model is bad at imitating users, good at answering\nm = masked_mean(losses, mask)\nassert abs(m - 0.5) < 1e-9, f"masked mean should be 0.5 (only answer tokens), got {m}; the plain mean would be {losses.mean():.2f}"` },
    ],
    hints: [
      'Build both lists in one loop over the messages, adding the same number of mask entries as tokens each time.',
      'Per message: `[IM_START, role, "\\n"] + split_content(content) + [IM_END, "\\n"]`, with mask `[False]*3 + [g]*len(body) + [g, False]` where `g` is whether the role is "assistant".',
      'For `masked_mean`, turn the mask into a bool array and use it to index the losses: `losses[m].mean()`.',
    ],
    explanation: `A chat is serialised into one token sequence, and SFT is the same next-token loss as pretraining with a mask on it. The assistant’s tokens and its \`<|im_end|>\` are graded, so the bot learns to answer and then stop. Everything else is still read as context but produces no gradient, so the model spends none of its learning on imitating customers. The generation prompt must reproduce the training template exactly and open the assistant turn, or the bot may start writing the customer’s side (the lesson’s debug exercise).

The real version does the same with a real tokenizer (each marker is one special token id), with the template shipped alongside the model, and with \`ignore_index=-100\` in PyTorch’s cross-entropy doing what your \`masked_mean\` does, across batches of thousands of conversations.`,
  },

  /* ---------- distillation: Distillation loss ---------- */
  {
    id: 'distillation-code-bot-soft-labels',
    lesson: 'distillation',
    title: 'A small bot learns from a big one',
    project: { piece: 'Distillation loss' },
    prelude: PAISA_PAL + '\n' + `
# ---- given: the big bot's logits for one next token ----
# Context: "If a UPI payment fails, the refund reaches your bank account within 3 working ..."
# (the same numbers as the teacher in the lesson's lab)
NEXT = ["days", "hours", "weeks", "minutes", "months", "biryani"]
BIG_BOT = np.array([4.0, 2.2, 2.0, 1.6, 0.5, -2.0])
SMALL_BOT = np.zeros(6)      # the laptop-sized student, untrained: every token equally likely
`,
    prompt: `The support team’s laptop bot should learn from the big bot’s probabilities, not only from its top answer.

- \`distill_loss(student_logits, teacher_logits, T)\` returns T² × KL(p ‖ q), where p = softmax(teacher_logits / T) and q = softmax(student_logits / T). KL(p ‖ q) = Σ p × ln(p ÷ q). (\`softmax\` is in \`paisa_pal.py\`.)
- \`distill_step(student_logits, teacher_logits, T, lr=0.5)\` returns new student logits after one gradient step. The gradient of that loss with respect to the student’s logits is T × (q − p).

Both models must be softened by the same T.`,
    starter: `def distill_loss(student_logits, teacher_logits, T):
    """T^2 * KL(p || q), both softened by T."""
    ...

def distill_step(student_logits, teacher_logits, T, lr=0.5):
    """One gradient step on distill_loss. Returns the new student logits."""
    ...

# When both work, uncomment:
# z = SMALL_BOT.copy()
# for _ in range(20):
#     z = distill_step(z, BIG_BOT, T=4)
# print(dict(zip(NEXT, softmax(z).round(3))))
`,
    solution: `def distill_loss(student_logits, teacher_logits, T):
    """T^2 * KL(p || q), both softened by T."""
    p = softmax(np.asarray(teacher_logits, float) / T)   # soft targets
    q = softmax(np.asarray(student_logits, float) / T)   # student at the SAME T
    return float(T * T * np.sum(p * np.log(p / q)))

def distill_step(student_logits, teacher_logits, T, lr=0.5):
    """One gradient step on distill_loss. Returns the new student logits."""
    p = softmax(np.asarray(teacher_logits, float) / T)
    q = softmax(np.asarray(student_logits, float) / T)
    grad = T * (q - p)                                    # gradient of T^2 * KL
    return student_logits - lr * grad

z = SMALL_BOT.copy()
for _ in range(20):
    z = distill_step(z, BIG_BOT, T=4)
print(dict(zip(NEXT, softmax(z).round(3))))
`,
    tests: [
      { name: 'the lesson’s worked example: KL([0.7, 0.2, 0.1] ‖ [0.5, 0.3, 0.2]) = 0.0851', code: `import numpy as np\nk = distill_loss(np.log([0.5, 0.3, 0.2]), np.log([0.7, 0.2, 0.1]), 1)\nassert abs(k - 0.0851) < 1e-4, f"got {k}"` },
      { name: 'zero when the small bot matches, and multiplied by T² at T = 2', code: `import numpy as np\nassert abs(distill_loss(BIG_BOT, BIG_BOT, 2)) < 1e-12, "a student equal to the teacher must have loss 0"\np = softmax(BIG_BOT / 2); q = softmax(SMALL_BOT / 2)\nexp = 4 * np.sum(p * np.log(p / q))\ngot = distill_loss(SMALL_BOT, BIG_BOT, 2)\nassert abs(got - exp) < 1e-9, f"expected T^2 * KL = {exp:.4f}, got {got}; soften BOTH by T and multiply by T*T"` },
      { name: 'the step follows the true gradient of the loss, and lowers it', code: `import numpy as np\nz = SMALL_BOT.copy(); eps = 1e-6\nnum = np.array([(distill_loss(z + eps * e, BIG_BOT, 2) - distill_loss(z - eps * e, BIG_BOT, 2)) / (2 * eps) for e in np.eye(6)])\ngot = z - distill_step(z, BIG_BOT, 2, lr=1.0)\nassert np.allclose(got, num, atol=1e-5), f"step direction {got.round(4)} vs numerical gradient {num.round(4)}"\nassert distill_loss(distill_step(z, BIG_BOT, 2), BIG_BOT, 2) < distill_loss(z, BIG_BOT, 2), "one step must lower the loss"` },
      { name: 'the point: at T = 4 the small bot learns the small probabilities too (the lesson’s 0.0487 vs 0.0060)', code: `import numpy as np\ndef train(T):\n    z = SMALL_BOT.copy()\n    for _ in range(20):\n        z = distill_step(z, BIG_BOT, T)\n    return softmax(z)\np = softmax(BIG_BOT)\nkl = lambda q: float(np.sum(p * np.log(p / q)))\nq1, q4 = train(1), train(4)\nassert abs(kl(q1) - 0.0487) < 1e-3 and abs(kl(q4) - 0.0060) < 1e-3, f"KL after 20 steps: T=1 {kl(q1):.4f}, T=4 {kl(q4):.4f}"\nassert q4[5] < 0.01 < q1[5], f"'biryani' after 'within 3 working': teacher {p[5]:.4f}, T=4 student {q4[5]:.4f}, T=1 student {q1[5]:.4f}"\nassert abs(q4[0] - p[0]) < 0.05, f"at normal temperature the student should say 'days' about as often as the teacher ({p[0]:.3f}), got {q4[0]:.3f}"` },
    ],
    hints: [
      'Both use `p = softmax(teacher_logits / T)` and `q = softmax(student_logits / T)`. Convert to float arrays first.',
      'The loss is `T * T * np.sum(p * np.log(p / q))`. The step is `student_logits - lr * T * (q - p)`.',
      'If the T² test fails, check that the student is divided by T as well. That is the lesson’s debug exercise.',
    ],
    explanation: `The big bot’s soft targets say more than “days”: hours and weeks are plausible, biryani is absurd. Softening both models by T makes those small probabilities loud enough to matter, and T² keeps the gradient from shrinking as T grows. So in 20 steps the T = 4 student gets within 0.006 of the teacher and learns that biryani is near impossible, while the T = 1 student is still at 0.049 and gives biryani 4%.

Because the student is compared at the same T, its logits learn to match the teacher’s, so at normal temperature it answers like the teacher. A real distillation run does this at every position of every sequence, over billions of tokens, often mixed with the ordinary hard-label loss. The student also inherits the teacher’s mistakes: it cannot be more right than the big bot it copies.`,
  },
]

export default exercises
