import type { CodeExerciseDef } from './types'
import { PAISA_PAL } from '../project/paisaPal'

// The Paisa Pal support-bot project, Part 8 (second half): preference loss, screenshot tokens,
// a refund detector inside the model, and checked answers. Each starts from paisa_pal.py plus a small
// given toy, so every exercise stands alone.

/* ---------- alignment-safety: DPO ---------- */
const DPO_GIVEN = `
# ---- given: three replies the SFT bot might write to the refund question ----
# To keep it small, the "policy" is one logit per whole reply. A real model scores a reply
# by adding up the log-probabilities of its tokens; the loss below does not change.
REPLIES = [
    "If a UPI payment fails but money is deducted, the refund reaches your bank account within 3 working days. (From the refunds help page.)",
    "Paisa Pal refunds every failed payment instantly, and adds 50 rupees for the trouble.",
    "Sorry, I cannot help with payments.",
]
CHOSEN, REJECTED = 0, 1        # a rater preferred the honest, cited reply over the invented policy
REF_LOGITS = np.array([1.0, 1.5, 0.5])   # the frozen SFT model: it likes the confident invention best

def log_probs(logits):
    return np.log(softmax(logits))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
`

/* ---------- multimodal: patch tokens ---------- */
const SHOT_GIVEN = `
# ---- given: a toy failed-payment screenshot and the (untrained) weights that read it ----
def failed_payment_screenshot():
    """16 x 16 pixels, 3 colours, values 0..1: a white screen, a red icon with a white cross, a grey Retry button."""
    img = np.ones((16, 16, 3))
    img[1:9, 4:12] = [0.85, 0.1, 0.1]       # the red "Payment failed" icon
    for i in range(8):                      # its white cross
        img[1 + i, 4 + i] = 1.0
        img[1 + i, 11 - i] = 1.0
    img[12:14, 2:14] = 0.6                  # the grey Retry button
    return img

P, D = 4, 8                                 # patch side in pixels, model width
_rng = np.random.default_rng(0)
W_E = _rng.normal(size=(P * P * 3, D)) * 0.1    # patch embedding: 48 numbers in, D out (learned in a real model)
POS = _rng.normal(size=(16, D)) * 0.02          # one position vector per patch slot (learned in a real model)
VOCAB = sorted({w for q, _, _ in QUESTIONS for w in tokenize(q)})
E = _rng.normal(size=(len(VOCAB), D)) * 0.1     # the text embedding table

def embed(text):
    """Text tokens: one row of E per word, as in the GPT you built."""
    return E[[VOCAB.index(w) for w in tokenize(text)]]
`

/* ---------- interpretability: a linear probe ---------- */
const PROBE_GIVEN = `
# ---- given: a toy model's middle layer. You may read it, but a real model hands you no such code ----
import zlib
D = 16
_rng = np.random.default_rng(7)
_REFUND = _rng.normal(size=D)
_REFUND /= np.linalg.norm(_REFUND)          # the hidden "money should come back" feature direction
_MONEY_BACK = {"refund", "refunded", "deducted", "debited", "reversed", "back", "return"}

def _word_vector(w):
    return np.random.default_rng(zlib.crc32(w.encode())).normal(size=D)

def hidden_state(question):
    """The residual stream at a middle layer, for the question's last token: D numbers."""
    words = tokenize(question)
    h = np.mean([_word_vector(w) for w in words], axis=0)
    if any(w in _MONEY_BACK for w in words):
        h = h + 2.0 * _REFUND               # the model has noticed "this is about getting money back"
    return h

# labelled questions: 1 = about a refund, 0 = not
TRAIN = [
    ("where is my refund", 1), ("i want a refund for a failed payment", 1),
    ("refund not received yet", 1), ("the recharge failed and i need a refund", 1),
    ("has my refund been processed", 1), ("my refund is still pending", 1),
    ("how do i change my upi pin", 0), ("what is the wallet limit", 0),
    ("cashback not credited for my bill payment", 0), ("how do i complete kyc", 0),
    ("can i send money from a new phone", 0), ("how do i raise a dispute", 0),
]
HELD_OUT = [
    ("amount debited but the recharge failed", 1), ("when will the money come back", 1),
    ("i was charged twice please return one payment", 1),
    ("what documents are needed for kyc", 0), ("my account is locked", 0),
    ("is there cashback on wallet top ups", 0),
]
`

/* ---------- reasoning-models: best-of-N ---------- */
const BON_GIVEN = `
# ---- given: a toy bot that answers a help question correctly only some of the time ----
import random
from collections import Counter

GENERIC_WRONG = [
    "Please wait 24 hours and try again.",
    "Please contact your bank, they handle this.",
    "This is usually sorted out within a week.",
]
POPULAR_WRONG = {   # two questions where the bot has one favourite wrong answer
    1: "Cashback is credited within 48 hours of an eligible payment, so please wait.",
    8: "You can send up to 1,00,000 rupees a day by UPI.",
}

def bot_answer(i, rng):
    """One sampled answer to QUESTIONS[i] (temperature above 0, so answers vary)."""
    question, page, fact = QUESTIONS[i]
    r = rng.random()
    if i in POPULAR_WRONG:                  # right 30%, the favourite wrong answer 50%
        if r < 0.3:
            return HELP_PAGES[page]
        if r < 0.8:
            return POPULAR_WRONG[i]
        return rng.choice(GENERIC_WRONG)
    if r < 0.4:                             # right 40%, each generic wrong answer 20%
        return HELP_PAGES[page]
    return rng.choice(GENERIC_WRONG)

ANSWERABLE = [i for i, (_, page, _) in enumerate(QUESTIONS) if page is not None]
`

const exercises: CodeExerciseDef[] = [
  {
    id: 'alignment-safety-code-bot-dpo',
    lesson: 'alignment-safety',
    title: 'Preference loss: teach the bot to prefer the honest reply',
    project: { piece: 'Preference loss' },
    prompt: `A rater compared two replies to “my upi payment failed but the money was deducted”. Reply 0 quotes the refunds help page. Reply 1 invents a policy (instant refunds plus 50 rupees). The frozen SFT model \`REF_LOGITS\` actually likes the invention more. Teach the bot the rater’s preference with DPO.

\`dpo_loss(logp, ref_logp, chosen, rejected, beta=0.1)\` takes log-probabilities from the policy and from the reference. Compute the margin \`m = beta * ((logp[chosen] - ref_logp[chosen]) - (logp[rejected] - ref_logp[rejected]))\` and return \`-log(sigmoid(m))\` as a float.

\`dpo_step(logits, ref_logits, chosen, rejected, beta=0.1, lr=1.0)\` does one gradient step on the policy’s logits and returns new logits (do not change the input). The gradient of the loss is \`-sigmoid(-m) * beta\` for the chosen logit, \`+sigmoid(-m) * beta\` for the rejected one, and 0 for every other reply. Use \`log_probs\` and \`sigmoid\` from the given code.`,
    prelude: PAISA_PAL + '\n' + DPO_GIVEN,
    starter: `def dpo_loss(logp, ref_logp, chosen, rejected, beta=0.1):
    """-log sigmoid(beta * (how much more the policy likes chosen than the reference does,
    minus the same for rejected))."""
    ...

def dpo_step(logits, ref_logits, chosen, rejected, beta=0.1, lr=1.0):
    """One gradient-descent step on dpo_loss. Returns new logits."""
    ...

logits = REF_LOGITS.copy()
for step in range(100):
    logits = dpo_step(logits, REF_LOGITS, CHOSEN, REJECTED)
print(softmax(logits).round(3))
`,
    solution: `def dpo_loss(logp, ref_logp, chosen, rejected, beta=0.1):
    """-log sigmoid(beta * (how much more the policy likes chosen than the reference does,
    minus the same for rejected))."""
    m = beta * ((logp[chosen] - ref_logp[chosen]) - (logp[rejected] - ref_logp[rejected]))
    return float(-np.log(sigmoid(m)))

def dpo_step(logits, ref_logits, chosen, rejected, beta=0.1, lr=1.0):
    """One gradient-descent step on dpo_loss. Returns new logits."""
    logp, ref_logp = log_probs(logits), log_probs(ref_logits)
    m = beta * ((logp[chosen] - ref_logp[chosen]) - (logp[rejected] - ref_logp[rejected]))
    push = sigmoid(-m) * beta              # big while the pair is still ranked wrongly
    new = np.array(logits, dtype=float)    # a copy: never edit the caller's array
    new[chosen] += lr * push               # minus the gradient
    new[rejected] -= lr * push
    return new

logits = REF_LOGITS.copy()
for step in range(100):
    logits = dpo_step(logits, REF_LOGITS, CHOSEN, REJECTED)
print(softmax(logits).round(3))           # the honest reply now wins
`,
    tests: [
      {
        name: 'before training the policy is the reference, so the loss is log 2',
        code: `lp = log_probs(REF_LOGITS)
v = dpo_loss(lp, lp, CHOSEN, REJECTED)
assert abs(v - np.log(2)) < 1e-9, f"got {v}; with no difference from the reference the margin is 0 and -log(sigmoid(0)) = 0.693"`,
      },
      {
        name: 'a worked example: policy -0.5 and -2.0, reference -1.0 and -1.0, beta 0.1 gives 0.621',
        code: `v = dpo_loss(np.array([-0.5, -2.0, -3.0]), np.array([-1.0, -1.0, -2.0]), 0, 1, beta=0.1)
assert abs(v - 0.62096) < 1e-4, f"got {v}; margin = 0.1 * ((-0.5 + 1.0) - (-2.0 + 1.0)) = 0.15, and -log(sigmoid(0.15)) = 0.621"`,
      },
      {
        name: 'it is relative: a policy that still prefers the invention, but less than the reference does, already scores better than log 2',
        code: `p = np.array([1.3, 1.5, 0.5])            # reply 1 is still the most likely
v = dpo_loss(log_probs(p), log_probs(REF_LOGITS), CHOSEN, REJECTED)
assert softmax(p)[1] > softmax(p)[0] and v < np.log(2), f"loss {v}: DPO rewards moving towards the chosen reply relative to the reference, not absolute probability"`,
      },
      {
        name: 'the step is minus the true gradient, and the refusal reply is left alone',
        code: `L = np.array([0.3, 1.2, -0.4])
def f(z):
    return dpo_loss(log_probs(z), log_probs(REF_LOGITS), CHOSEN, REJECTED, beta=0.5)
g = np.array([(f(L + h) - f(L - h)) / 2e-6 for h in np.eye(3) * 1e-6])
new = dpo_step(L, REF_LOGITS, CHOSEN, REJECTED, beta=0.5, lr=1.0)
assert np.allclose(new, L - g, atol=1e-5), f"step {new - L}, expected {-g}"
assert new[2] == L[2], "reply 2 is in no preference pair, so DPO must not touch its logit"
assert np.array_equal(L, [0.3, 1.2, -0.4]), "dpo_step changed the input array: return a copy"`,
      },
      {
        name: 'after 100 steps the loss has fallen and the honest reply is the most likely',
        code: `L = REF_LOGITS.copy(); ref = log_probs(REF_LOGITS); losses = []
for _ in range(100):
    losses.append(dpo_loss(log_probs(L), ref, CHOSEN, REJECTED))
    L = dpo_step(L, REF_LOGITS, CHOSEN, REJECTED)
p = softmax(L)
assert all(b < a for a, b in zip(losses, losses[1:])), "the loss should fall on every step"
assert losses[-1] < 0.4 and p[0] > 0.95, f"final loss {losses[-1]:.3f}, P(honest reply) {p[0]:.3f}"`,
      },
    ],
    hints: [
      '`dpo_loss`: compute `m` exactly as written, then `float(-np.log(sigmoid(m)))`.',
      '`dpo_step`: get `logp = log_probs(logits)` and `ref_logp = log_probs(ref_logits)`, compute the same `m`, then `push = sigmoid(-m) * beta`.',
      'Gradient descent subtracts the gradient: `new = np.array(logits, dtype=float)`, then `new[chosen] += lr * push` and `new[rejected] -= lr * push`.',
    ],
    explanation: `DPO turns “the rater preferred A to B” into an ordinary loss. The margin measures how much more the policy likes the chosen reply than the reference does, minus the same for the rejected reply. So the model is rewarded for moving away from the SFT model in the rater’s direction, not for absolute probability. That is the reward r = β log(π / π_ref) from the lesson’s π* ∝ π_ref e^(r/β), and β is still the leash: once the pair is ranked well, sigmoid(-m) shrinks and the updates fade.

Notice what the gradient touches: only the two replies in the pair. The over-refusal reply was in no pair, so nothing taught the bot about it. That is why preference data has to cover the failures you care about, including needless refusals.

The real version differs in scale, not in the loss. The policy is the whole Transformer, a reply’s log-probability is the sum over its tokens, the gradient flows through backpropagation into billions of weights, and there are tens of thousands of pairs written by raters (or by an AI judge against written principles). Here one logit per reply stands in for all of that.`,
  },

  {
    id: 'multimodal-code-bot-screenshot',
    lesson: 'multimodal',
    title: 'Screenshot tokens: turn a failed-payment photo into the bot’s input',
    project: { piece: 'Screenshot tokens' },
    prompt: `A customer sends a screenshot and one line of text. Turn both into one sequence of vectors the language model can read, exactly as the lesson does: z = x W_E + e.

\`patchify(img, p)\`: cut an (H, W, 3) image into p × p squares, left to right and top to bottom, and flatten each square into one row of p · p · 3 numbers. Return shape (number of patches, p · p · 3). A plain reshape gives stripes, not squares.

\`screenshot_tokens(img)\`: patchify with the given \`P\` (4), multiply by \`W_E\`, add \`POS\`. Returns (16, D) for the 16 × 16 screenshot.

\`ticket_sequence(img, message)\`: the screenshot tokens followed by \`embed(message)\`, as one (T, D) array.`,
    prelude: PAISA_PAL + '\n' + SHOT_GIVEN,
    starter: `def patchify(img, p):
    """(H, W, 3) image -> (num_patches, p * p * 3), one square patch per row, in reading order."""
    ...

def screenshot_tokens(img):
    """One D-wide token per patch: flattened patch @ W_E + its position vector."""
    ...

def ticket_sequence(img, message):
    """Image tokens, then the message's word tokens: one (T, D) sequence."""
    ...

img = failed_payment_screenshot()
print(ticket_sequence(img, QUESTIONS[0][0]).shape)
`,
    solution: `def patchify(img, p):
    """(H, W, 3) image -> (num_patches, p * p * 3), one square patch per row, in reading order."""
    H, W, C = img.shape
    x = img.reshape(H // p, p, W // p, p, C)   # cut rows and columns into blocks
    x = x.transpose(0, 2, 1, 3, 4)             # (rows, cols, p, p, C): one block per patch
    return x.reshape(-1, p * p * C)

def screenshot_tokens(img):
    """One D-wide token per patch: flattened patch @ W_E + its position vector."""
    patches = patchify(img, P)                 # (16, 48)
    return patches @ W_E + POS[:len(patches)]  # (16, D)

def ticket_sequence(img, message):
    """Image tokens, then the message's word tokens: one (T, D) sequence."""
    return np.concatenate([screenshot_tokens(img), embed(message)])

img = failed_payment_screenshot()
print(ticket_sequence(img, QUESTIONS[0][0]).shape)   # 16 image tokens + 15 words
`,
    tests: [
      {
        name: 'patches are squares, not stripes: on a 4 × 4 image with p = 2, patch 1 is the top-right square',
        code: `tiny = np.array([[[r, c, 0] for c in range(4)] for r in range(4)])   # pixel (r, c) holds [r, c, 0]
out = patchify(tiny, 2)
assert out.shape == (4, 12), f"shape {out.shape}, expected (4, 12)"
where = out[1].reshape(-1, 3)[:, :2].tolist()
assert where == [[0, 2], [0, 3], [1, 2], [1, 3]], f"patch 1 holds pixels {where}; it should be rows 0-1, columns 2-3"`,
      },
      {
        name: 'the screenshot becomes 16 tokens of width D: (16 / 4) × (16 / 4)',
        code: `img = failed_payment_screenshot()
t = screenshot_tokens(img)
assert t.shape == (16, D), f"got {t.shape}"
assert np.allclose(t, patchify(img, P) @ W_E + POS), "each token should be its flattened patch @ W_E plus its position vector"`,
      },
      {
        name: 'two blank corners look identical, and only the position vector tells them apart',
        code: `img = failed_payment_screenshot()
x = patchify(img, P)
assert (x[0] == 1).all() and (x[3] == 1).all(), "patches 0 and 3 are the plain white top corners"
t = screenshot_tokens(img)
assert not np.allclose(t[0], t[3]), "the two corner tokens are identical: did you add POS?"
assert np.allclose(t[0] - t[3], POS[0] - POS[3]), "the corner tokens should differ by exactly their position vectors"`,
      },
      {
        name: 'the ticket is one sequence: 16 image tokens then 15 word tokens, all the same width',
        code: `img = failed_payment_screenshot()
msg = QUESTIONS[0][0]
seq = ticket_sequence(img, msg)
assert seq.shape == (16 + len(tokenize(msg)), D), f"got {seq.shape}, expected {(16 + len(tokenize(msg)), D)}"
assert np.allclose(seq[:16], screenshot_tokens(img)) and np.allclose(seq[16:], embed(msg)), "image tokens first, then the message"`,
      },
    ],
    hints: [
      'Reshape to `(H // p, p, W // p, p, C)`: block row, row inside the block, block column, column inside the block, colour.',
      'Then `.transpose(0, 2, 1, 3, 4)` brings the two block indices to the front, and `.reshape(-1, p * p * C)` flattens each block.',
      '`screenshot_tokens` is `patchify(img, P) @ W_E + POS`; `ticket_sequence` is `np.concatenate([screenshot_tokens(img), embed(message)])`.',
    ],
    explanation: `A patch is its own input row: its pixels go straight into a matrix multiply, so there is no vocabulary and no unknown token. The position vector is the only thing that separates two identical white corners, which is why shuffling patches without positions would lose the layout (the red icon above the Retry button).

After that, the image is just more rows in the (T, D) array. The language model never learns it is looking at a picture.

The real version adds a lot between these steps: 14-pixel patches on images of 448 × 448 or more (1,024 patches per tile), a vision encoder (a ViT trained with CLIP- or SigLIP-style contrastive learning) that lets the patches attend to each other, 2 × 2 merging, and a projector MLP to the language model’s width. All those weights are learned; here W_E, POS and E are random, so the shapes and the order are right but the tokens mean nothing yet.`,
  },

  {
    id: 'interpretability-code-bot-probe',
    lesson: 'interpretability',
    title: 'Refund detector: find the direction that means “refund”',
    project: { piece: 'Refund detector' },
    prompt: `Riya wants to know, from inside the model, when a customer is asking for money back, so the bot can route it to the refunds page. \`hidden_state(question)\` gives the toy model’s middle-layer vector. \`TRAIN\` and \`HELD_OUT\` are labelled questions.

Build a linear probe the simplest way, by difference of means (the same recipe as a steering vector):

\`fit_probe(H, y)\`: \`H\` is (n, D) hidden states, \`y\` the 0/1 labels. The direction \`d\` is the mean of the refund rows minus the mean of the other rows, scaled to length 1. The threshold \`t\` is halfway between the two class means projected onto \`d\`. Return \`(d, t)\`.

\`is_refund(question, d, t)\`: True if \`hidden_state(question) @ d > t\`.`,
    prelude: PAISA_PAL + '\n' + PROBE_GIVEN,
    starter: `def fit_probe(H, y):
    """Difference-of-means probe. Returns (unit direction d, threshold t)."""
    ...

def is_refund(question, d, t):
    """Does the model's middle layer say this question is about a refund?"""
    ...

H = np.array([hidden_state(q) for q, _ in TRAIN])
y = np.array([label for _, label in TRAIN])
d, t = fit_probe(H, y)
for q, label in HELD_OUT:
    print(label, is_refund(q, d, t), q)
`,
    solution: `def fit_probe(H, y):
    """Difference-of-means probe. Returns (unit direction d, threshold t)."""
    H, y = np.asarray(H, dtype=float), np.asarray(y)
    mean_yes = H[y == 1].mean(axis=0)
    mean_no = H[y == 0].mean(axis=0)
    d = mean_yes - mean_no
    d = d / np.linalg.norm(d)                  # only the direction matters
    t = (mean_yes @ d + mean_no @ d) / 2       # halfway between the two groups
    return d, t

def is_refund(question, d, t):
    """Does the model's middle layer say this question is about a refund?"""
    return bool(hidden_state(question) @ d > t)

H = np.array([hidden_state(q) for q, _ in TRAIN])
y = np.array([label for _, label in TRAIN])
d, t = fit_probe(H, y)
for q, label in HELD_OUT:
    print(label, is_refund(q, d, t), q)
`,
    tests: [
      {
        name: 'the direction has length 1 and the threshold sits halfway between the groups',
        code: `H = np.array([[2.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 2.0]]); y = np.array([1, 1, 0, 0])
d, t = fit_probe(H, y)
assert np.isclose(np.linalg.norm(d), 1.0), f"|d| = {np.linalg.norm(d):.3f}, expected 1"
assert np.allclose(d, np.array([3.0, -1.0]) / np.sqrt(10)), f"d = {d}; means are [3, 0] and [0, 1], so d points along [3, -1]"
assert np.isclose(t, (9 / np.sqrt(10) - 1 / np.sqrt(10)) / 2), f"t = {t}; the projections of the means are 2.846 and -0.316"`,
      },
      {
        name: 'the probe finds the model’s own refund direction (cosine above 0.8)',
        code: `H = np.array([hidden_state(q) for q, _ in TRAIN]); y = np.array([l for _, l in TRAIN])
d, t = fit_probe(H, y)
cos = float(d @ _REFUND)
assert cos > 0.8, f"cosine with the hidden feature direction is {cos:.2f}"`,
      },
      {
        name: 'it works on held-out questions it never saw',
        code: `H = np.array([hidden_state(q) for q, _ in TRAIN]); y = np.array([l for _, l in TRAIN])
d, t = fit_probe(H, y)
wrong = [q for q, l in HELD_OUT if is_refund(q, d, t) != bool(l)]
assert wrong == [], f"misclassified: {wrong}"`,
      },
      {
        name: 'it reads the idea, not the word: the first real question never says “refund”, and only it is flagged',
        code: `H = np.array([hidden_state(q) for q, _ in TRAIN]); y = np.array([l for _, l in TRAIN])
d, t = fit_probe(H, y)
first = QUESTIONS[0][0]
assert "refund" not in tokenize(first), "sanity check"
flags = [is_refund(q, d, t) for q, _, _ in QUESTIONS]
assert flags[0] is True, f"missed: {first!r}; a keyword search for 'refund' misses it too, the probe should not"
assert not any(flags[1:]), f"false alarms: {[q for (q, _, _), f in zip(QUESTIONS, flags) if f][1:]}"`,
      },
    ],
    hints: [
      'Convert first: `H, y = np.asarray(H, dtype=float), np.asarray(y)`. Then `H[y == 1].mean(axis=0)` is the average refund vector.',
      '`d = mean_yes - mean_no`, then `d = d / np.linalg.norm(d)`. Project each mean with `@ d`.',
      '`t = (mean_yes @ d + mean_no @ d) / 2`, and `is_refund` returns `bool(hidden_state(question) @ d > t)`.',
    ],
    explanation: `Averaging over many refund questions cancels out the words they do not share and keeps what they do share: the feature direction the model adds when a question is about money coming back. That is why the probe catches “the money was deducted, when do i get it back”, which a keyword search for “refund” misses.

Be honest about what this shows. A probe that works proves the information is present and linearly readable at this layer. It does not prove the model uses that direction to decide its answer. For that you would intervene: add or remove the direction during generation (steering) and check the reply changes as predicted.

In a real model, the hidden states come from hooks on a middle layer of a Transformer with thousands of dimensions, the labelled set has hundreds or thousands of examples, and the probe is often a logistic regression rather than a difference of means. Here the model is a made-up formula with one feature planted in it, so the probe finds it cleanly. Real features overlap (superposition), and the direction you find is messier.`,
  },

  {
    id: 'reasoning-models-code-bot-best-of-n',
    lesson: 'reasoning-models',
    title: 'Checked answers: best-of-N against majority vote',
    project: { piece: 'Checked answers' },
    prompt: `The toy bot \`bot_answer(i, rng)\` answers \`QUESTIONS[i]\` right 40% of the time. On two questions it has a favourite wrong answer (50%) and is right only 30%. Spend test-time compute on it two ways and measure both.

\`has_fact(answer, fact)\`: the checker. True if the required fact appears in the answer, ignoring case.

\`majority_vote(answers)\`: the most common answer (on a tie, the one seen first; \`Counter.most_common\` does this).

\`best_of_n(answers, fact)\`: the first answer that passes the checker; if none does, \`answers[0]\`.

\`evaluate(N, question_ids, trials=200, seed=0)\`: with \`rng = random.Random(seed)\`, for each trial and each question id, sample N answers, then score both methods with \`has_fact\` against that question’s fact. Return \`(vote_accuracy, checked_accuracy)\`.`,
    prelude: PAISA_PAL + '\n' + BON_GIVEN,
    starter: `def has_fact(answer, fact):
    ...

def majority_vote(answers):
    ...

def best_of_n(answers, fact):
    ...

def evaluate(N, question_ids, trials=200, seed=0):
    """Returns (vote_accuracy, checked_accuracy) over trials x questions."""
    ...

for N in [1, 3, 5, 15]:
    print(N, evaluate(N, ANSWERABLE))
`,
    solution: `def has_fact(answer, fact):
    return fact.lower() in answer.lower()

def majority_vote(answers):
    return Counter(answers).most_common(1)[0][0]

def best_of_n(answers, fact):
    for a in answers:
        if has_fact(a, fact):          # the verifier
            return a
    return answers[0]

def evaluate(N, question_ids, trials=200, seed=0):
    """Returns (vote_accuracy, checked_accuracy) over trials x questions."""
    rng = random.Random(seed)
    vote = checked = total = 0
    for _ in range(trials):
        for i in question_ids:
            fact = QUESTIONS[i][2]
            answers = [bot_answer(i, rng) for _ in range(N)]
            vote += has_fact(majority_vote(answers), fact)
            checked += has_fact(best_of_n(answers, fact), fact)
            total += 1
    return vote / total, checked / total

for N in [1, 3, 5, 15]:
    print(N, evaluate(N, ANSWERABLE))
`,
    tests: [
      {
        name: 'the checker and the vote',
        code: `assert has_fact(HELP_PAGES["security"], "never ask") and has_fact("full kyc needs a pan card", "PAN"), "the check should ignore case"
assert not has_fact(GENERIC_WRONG[0], "3 working days"), "got True for an answer without the fact"
r = majority_vote(["wait", "3 working days", "wait"])
assert r == "wait", f"got {r!r}"
r = majority_vote(["a", "b", "c"])
assert r == "a", f"got {r!r}; on a tie return the first one seen"`,
      },
      {
        name: 'best-of-N returns the first answer that passes, or the first answer if none does',
        code: `ans = [GENERIC_WRONG[0], HELP_PAGES["refunds"], GENERIC_WRONG[1]]
r = best_of_n(ans, "3 working days")
assert r == HELP_PAGES["refunds"], f"got {r!r}"
r = best_of_n(GENERIC_WRONG, "3 working days")
assert r == GENERIC_WRONG[0], f"got {r!r}; with nothing accepted, fall back to answers[0]"`,
      },
      {
        name: 'with N = 1 there is nothing to choose: both methods score the same',
        code: `v, c = evaluate(1, ANSWERABLE, trials=100)
assert v == c, f"vote {v}, checked {c}"
assert 0.3 < v < 0.45, f"single-answer accuracy {v:.3f}; expected about 0.38"`,
      },
      {
        name: 'with N = 5 the checker matches 1 - (1 - p)^N: about 0.90 over the ten questions',
        code: `v, c = evaluate(5, ANSWERABLE, trials=300)
expected = (8 * (1 - 0.6 ** 5) + 2 * (1 - 0.7 ** 5)) / 10
assert abs(c - expected) < 0.04, f"checked accuracy {c:.3f}, the formula gives {expected:.3f}"
assert c > v + 0.3, f"checked {c:.3f} should beat the vote {v:.3f} by a wide margin"`,
      },
      {
        name: 'the point: on “i paid 50 rupees, why no cashback”, more votes make it worse, more checked tries make it better',
        code: `v1, c1 = evaluate(1, [1], trials=400)
v15, c15 = evaluate(15, [1], trials=400)
assert v15 < v1, f"vote accuracy went {v1:.3f} -> {v15:.3f}; voting should amplify the popular wrong answer"
assert c15 > 0.95, f"checked accuracy at N = 15 is {c15:.3f}; 1 - 0.7**15 = 0.995"`,
      },
    ],
    hints: [
      '`has_fact` is one line with `.lower()` on both sides. `majority_vote` is `Counter(answers).most_common(1)[0][0]`.',
      '`best_of_n`: loop over the answers and return the first one where `has_fact(a, fact)`; after the loop, `return answers[0]`.',
      'In `evaluate`, make the generator once, before the loops. For each trial and each `i`, `answers = [bot_answer(i, rng) for _ in range(N)]`, and add `has_fact(...)` of each method’s pick to its count (True counts as 1).',
    ],
    explanation: `Best-of-N with a checker only needs one right answer among the N, so it follows 1 - (1 - p)^N. Majority vote returns the mode of the bot’s answers, so it helps only when the right answer is the most common one. On the cashback question the bot’s favourite answer (“48 hours”) is wrong, and voting makes that mistake more reliable as N grows.

The honest catch: this checker knows the required fact because these questions are labelled. That is like unit tests for code: a near-perfect verifier that exists only for some tasks. For a live customer question, the real options are weaker checks (does the answer quote the retrieved help page? does a scoring model approve?), and the lesson showed that a sloppy verifier puts a ceiling on accuracy that more samples cannot lift.

Real reasoning models add long written traces before each answer, sample at scale, and are trained with reinforcement learning on checkable problems so that p itself goes up. The measuring you just did, same questions, same seed, both methods, is what their evaluations do too.`,
  },
]

export default exercises
