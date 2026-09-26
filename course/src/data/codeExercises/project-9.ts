import type { CodeExerciseDef } from './types'
import { PAISA_PAL } from '../project/paisaPal'

// Paisa Pal project, Part 9 and the eval harness: the retriever, the LoRA adapter, the tool loop and the eval.
// Each prelude is paisa_pal.py plus only what that piece needs, so every exercise stands alone.

/* ---------- rag: the retriever ---------- */
const RETRIEVER_GIVEN = `# ---- given: the words the retriever works with ----
# Stop words say nothing about which page you want. The company's own name is one of them:
# customers say "paisa pal" whatever they are asking about.
STOP = set("a an the is are was be has have to of in on for by and or but my i me you your it its "
           "this that do does can how what why when who not no only get paisa pal".split())

def words(text):
    """tokenize, drop stop words, and strip a plural s, so "disputes" matches "dispute"."""
    return [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in tokenize(text) if w not in STOP]

PAGE_IDS = list(HELP_PAGES)                                          # refunds, cashback, kyc, ...
VOCAB = sorted({w for text in HELP_PAGES.values() for w in words(text)})   # one slot per help-page word
`

/* ---------- fine-tuning: the LoRA adapter ---------- */
const LORA_GIVEN = `# ---- given: a frozen "tone" layer. Question features in, one score per reply style out. ----
STYLES = ["formal", "warm", "safety first"]
# formal:       "Dear Valued Customer, we deeply regret any inconvenience..." (the base model's habit)
# warm:         two friendly sentences with the answer (the Paisa Pal brand rule)
# safety first: warm, but opening with "We will never ask for your PIN or OTP" (security questions)
FEATURE_WORDS = sorted({w for q, _, _ in QUESTIONS for w in tokenize(q)})

def features(question):
    """Bag of words over FEATURE_WORDS, scaled to length 1."""
    v = np.zeros(len(FEATURE_WORDS))
    for w in tokenize(question):
        if w in FEATURE_WORDS:
            v[FEATURE_WORDS.index(w)] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v

X_TRAIN = np.array([features(q) for q, _, _ in QUESTIONS])                             # (12, 71)
Y_TRAIN = np.array([2 if page == "security" else 1 for _, page, _ in QUESTIONS])   # the brand team's labels

_rng = np.random.default_rng(7)
W_BASE = _rng.normal(size=(len(FEATURE_WORDS), 3)) * 0.3
W_BASE[:, 0] += 1.0          # the pretrained habit: every word pushes towards "formal"

def lora_logits(X, W, A, B, scale):
    """The LoRA layer: x W + (x A) B * (alpha / r)."""
    return X @ W + (X @ A) @ B * scale

def cross_entropy(logits, y):
    p = softmax(logits)
    return float(-np.log(p[np.arange(len(y)), y]).mean())
`

/* ---------- agents: the tool loop ---------- */
const AGENT_GIVEN = `# ---- given: the convention, the tools the bot may call, and a stand-in model ----
import re, json

SYSTEM_PROMPT = (
    "You are the Paisa Pal support bot. You may use tools.\\n"
    "To use a tool reply EXACTLY:\\nThought: <why>\\nTOOL: <name>\\nARGS: <json>\\n"
    "When you have the answer reply:\\nThought: <why>\\nANSWER: <final answer>\\n\\n"
    "Available tools:\\n"
    "- get_balance: a customer's wallet balance. args: {\\"customer_id\\": str}\\n"
    "- refund_status: where a refund has got to. args: {\\"transaction_id\\": str}"
)
BOT_TOOLS = {"get_balance": get_balance, "refund_status": refund_status}   # send_money is not wired in

def scripted_model(context):
    """Stands in for the LLM: a few if statements that write text in the agreed format."""
    question = re.search(r"USER QUESTION: (.+)", context).group(1)
    results = re.findall(r"RESULT: (.+)", context)
    if results and results[-1].startswith("ERROR"):
        return "Thought: The lookup failed.\\nANSWER: Sorry, I could not look that up. Please check the id."
    if results and "TOOL: get_balance" in context:
        return f"Thought: I have the balance.\\nANSWER: Your balance is {results[-1]} rupees."
    if results:
        return f"Thought: I have the refund status.\\nANSWER: Your refund is {results[-1]}."
    cust = re.search(r"\\bC\\d+", question)
    txn = re.search(r"\\bT\\d+", question)
    if "balance" in question.lower() and cust:
        return f'Thought: I need the balance.\\nTOOL: get_balance\\nARGS: {{"customer_id": "{cust.group()}"}}'
    if txn:
        return f'Thought: I need the refund status.\\nTOOL: refund_status\\nARGS: {{"transaction_id": "{txn.group()}"}}'
    return "Thought: No tool can help here.\\nANSWER: Sorry, I can only help with your Paisa Pal account."
`

/* ---------- evals: the eval harness ---------- */
const EVAL_GIVEN = `# ---- given: the bot under test (a keyword bot, deliberately imperfect) and the bootstrap ----
KEYWORDS = {"refunds": ["refund", "deducted", "failed"], "cashback": ["cashback"],
            "kyc": ["kyc", "documents", "wallet"], "security": ["pin", "otp", "locked"],
            "limits": ["send"], "disputes": ["dispute"]}

def toy_bot(question):
    """Returns {"page": page id, or None when it refuses, "answer": text}. It answers with the page's first sentence."""
    ws = tokenize(question)
    for page, keys in KEYWORDS.items():
        if any(k in ws for k in keys):
            return {"page": page, "answer": HELP_PAGES[page].split(". ")[0] + "."}
    return {"page": None, "answer": "Sorry, I can only help with Paisa Pal payments."}

def bootstrap_interval(marks, resamples=2000, seed=0):
    """The middle 95% of resampled pass rates (the bootstrap from the evals lesson)."""
    marks = np.asarray(marks, dtype=float)
    rng = np.random.default_rng(seed)
    means = marks[rng.integers(0, len(marks), size=(resamples, len(marks)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
`

const exercises: CodeExerciseDef[] = [
  /* ---------- rag ---------- */
  {
    id: 'rag-code-bot-retriever',
    lesson: 'rag',
    project: { piece: 'Retriever' },
    title: 'Paisa Pal’s retriever: the right help page, or none',
    prelude: `${PAISA_PAL}\n${RETRIEVER_GIVEN}`,
    prompt: `The bot must answer from Paisa Pal’s six help pages, not from memory. Build the retriever with the lesson’s crude embedder: rarity weights, cosine, and a threshold for “no page answers this”.

- \`idf_weights()\` returns a NumPy array with one weight per word in \`VOCAB\`: \`ln(number of pages / number of pages containing the word)\`. Use \`words(text)\` (given) to split a page.
- \`embed(text, idf)\` returns a vector of length \`len(VOCAB)\`: for each word of \`words(text)\` that is in \`VOCAB\`, add that word’s idf to its slot. Then scale to length 1, so the dot product is the cosine. A text with no known words stays all zeros.
- \`retrieve(question, threshold=0.15)\` embeds every help page and the question, and returns the id of the page with the highest cosine, or \`None\` when even the best score is below \`threshold\`.

The bot must find the refunds page for the failed UPI payment, and must refuse the biryani and share-price questions.`,
    starter: `def idf_weights():
    """One weight per VOCAB word: ln(pages / pages that contain the word)."""
    ...

def embed(text, idf):
    """Sum of idf weights for the text's known words, scaled to length 1 (all zeros stay zeros)."""
    ...

def retrieve(question, threshold=0.15):
    """The id of the most similar help page by cosine, or None if the best score is below threshold."""
    ...

print(retrieve("my upi payment failed but the money was deducted"))
`,
    solution: `def idf_weights():
    """One weight per VOCAB word: ln(pages / pages that contain the word)."""
    page_words = [set(words(text)) for text in HELP_PAGES.values()]
    n = len(page_words)
    return np.array([np.log(n / sum(w in pw for pw in page_words)) for w in VOCAB])

def embed(text, idf):
    """Sum of idf weights for the text's known words, scaled to length 1 (all zeros stay zeros)."""
    v = np.zeros(len(VOCAB))
    for w in words(text):
        if w in VOCAB:
            v[VOCAB.index(w)] += idf[VOCAB.index(w)]    # rare words count more
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def retrieve(question, threshold=0.15):
    """The id of the most similar help page by cosine, or None if the best score is below threshold."""
    idf = idf_weights()
    pages = np.array([embed(HELP_PAGES[p], idf) for p in PAGE_IDS])
    scores = pages @ embed(question, idf)              # every cosine at once
    best = int(np.argmax(scores))
    return PAGE_IDS[best] if scores[best] >= threshold else None

for q, page, _ in QUESTIONS:
    print(retrieve(q), "<-", q)
`,
    tests: [
      { name: 'rare words weigh more: “kyc” (1 page) gets ln 6, “rupee” (3 pages) gets ln 2', code: `import numpy as np
idf = idf_weights()
assert len(idf) == len(VOCAB), f"expected {len(VOCAB)} weights, one per VOCAB word, got {len(idf)}"
k, r = idf[VOCAB.index("kyc")], idf[VOCAB.index("rupee")]
assert abs(k - np.log(6)) < 1e-9 and abs(r - np.log(2)) < 1e-9, f"kyc {k:.3f} (want 1.792), rupee {r:.3f} (want 0.693)"` },
      { name: 'vectors have length 1, and a question with no known words is all zeros, not NaN', code: `import numpy as np
idf = idf_weights()
v = embed("full kyc kyc documents", idf)
assert abs(np.linalg.norm(v) - 1) < 1e-9, f"length {np.linalg.norm(v):.4f}; divide by np.linalg.norm(v)"
z = embed("biryani bengaluru", idf)
assert np.isfinite(z).all() and not z.any(), f"no help-page words, so the vector must be all zeros; got {z[z != 0]}"` },
      { name: 'all 10 answerable customer questions find the right help page', code: `wrong = [(q, page, retrieve(q)) for q, page, _ in QUESTIONS if page is not None and retrieve(q) != page]
assert not wrong, f"(question, right page, got): {wrong}"` },
      { name: 'the threshold is what lets the bot say no: biryani and share price are refused', code: `for q, page, _ in QUESTIONS:
    if page is None:
        got = retrieve(q)
        assert got is None, f"{q!r} should be refused, got page {got!r}"
        assert retrieve(q, threshold=0.0) is not None, "with no threshold, a nearest page always exists, even for nonsense"` },
    ],
    hints: [
      'For `idf_weights`: make `page_words = [set(words(t)) for t in HELP_PAGES.values()]`; a word’s count is `sum(w in pw for pw in page_words)`. `np.log` is the natural log.',
      '`embed`: start from `np.zeros(len(VOCAB))`, add `idf[VOCAB.index(w)]` at `VOCAB.index(w)`, then divide by `np.linalg.norm(v)` only if it is above 0.',
      '`retrieve`: stack the six page vectors into a matrix, `scores = pages @ embed(question, idf)`, take `np.argmax`, and compare the best score with the threshold.',
    ],
    explanation: `Nothing in this touches a model. The bot’s “knowledge” is now six strings, and fixing a policy means editing a page, not retraining. Rarity weighting is why the question about a failed payment lands on refunds: “deducted” is on one page only, while “rupee” is on three and tells you less.

The threshold is the piece a bare top-k search does not have. A nearest page always exists, so without it the bot would answer the biryani question from whichever page happened to be least far away. A real retriever swaps \`embed\` for a trained Transformer encoder (so “money back” finds “refund” with no shared word), chunks long policy PDFs instead of using whole pages, stores millions of vectors in an index, and tunes the threshold on an eval set rather than by eye.`,
    source: 'phase4-modern-llms/mini_rag.py',
  },

  /* ---------- fine-tuning ---------- */
  {
    id: 'fine-tuning-code-bot-lora',
    lesson: 'fine-tuning',
    project: { piece: 'LoRA adapter' },
    title: 'Teach the frozen tone layer the Paisa Pal habit',
    prelude: `${PAISA_PAL}\n${LORA_GIVEN}`,
    prompt: `The given layer \`W_BASE\` (71 words in, 3 reply styles out) has a habit: it picks "formal" for every one of the 12 customer questions. The brand team wants "warm", and "safety first" for the two security questions (\`Y_TRAIN\`). You may not touch \`W_BASE\`. Train a LoRA adapter beside it.

Write \`train_lora(X, y, W, r=2, alpha=4, steps=200, lr=1.0, seed=0)\` that returns \`(A, B, losses)\`:

- Start the way the repo does: \`A = rng.normal(size=(d_in, r)) * 0.01\` with \`rng = np.random.default_rng(seed)\`, and \`B = zeros((r, d_out))\`. \`scale = alpha / r\`.
- Each step: \`logits = lora_logits(X, W, A, B, scale)\`, append \`cross_entropy(logits, y)\` to \`losses\`, then the gradients of the loss:
  \`G = (softmax(logits) - Y) / n\` where \`Y\` is the one-hot of \`y\` and n the number of questions,
  \`grad_B = scale * (X @ A).T @ G\`,  \`grad_A = scale * X.T @ (G @ B.T)\`.
- Update only A and B: \`A -= lr * grad_A\`, \`B -= lr * grad_B\`. W is never written to.`,
    starter: `def train_lora(X, y, W, r=2, alpha=4, steps=200, lr=1.0, seed=0):
    """Train A (d_in, r) and B (r, d_out) beside the frozen W. Returns (A, B, losses)."""
    ...

print("base model:", [STYLES[i] for i in (X_TRAIN @ W_BASE).argmax(axis=1)][:3])
`,
    solution: `def train_lora(X, y, W, r=2, alpha=4, steps=200, lr=1.0, seed=0):
    """Train A (d_in, r) and B (r, d_out) beside the frozen W. Returns (A, B, losses)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(X.shape[1], r)) * 0.01    # small random
    B = np.zeros((r, W.shape[1]))                  # zeros: step 0 is exactly the base model
    scale = alpha / r
    Y = np.eye(W.shape[1])[y]                      # one-hot targets
    losses = []
    for _ in range(steps):
        logits = lora_logits(X, W, A, B, scale)
        losses.append(cross_entropy(logits, y))
        G = (softmax(logits) - Y) / len(y)
        grad_B = scale * (X @ A).T @ G
        grad_A = scale * X.T @ (G @ B.T)
        A -= lr * grad_A                           # only the adapter moves
        B -= lr * grad_B
    return A, B, losses

A, B, losses = train_lora(X_TRAIN, Y_TRAIN, W_BASE)
print(f"loss {losses[0]:.3f} -> {losses[-1]:.4f}")
print("with adapter:", [STYLES[i] for i in lora_logits(X_TRAIN, W_BASE, A, B, 2).argmax(axis=1)][:3])
`,
    tests: [
      { name: 'W_BASE is untouched: only A and B were trained', code: `import numpy as np
W_copy = W_BASE.copy()
A, B, losses = train_lora(X_TRAIN, Y_TRAIN, W_BASE)
assert np.array_equal(W_BASE, W_copy), "W_BASE changed: that is a full fine-tune, not LoRA"
assert A.shape == (71, 2) and B.shape == (2, 3), f"A {A.shape} (want (71, 2)), B {B.shape} (want (2, 3))"` },
      { name: 'step 0 is exactly the base model, then the loss falls', code: `import numpy as np
A, B, losses = train_lora(X_TRAIN, Y_TRAIN, W_BASE)
base = cross_entropy(X_TRAIN @ W_BASE, Y_TRAIN)
assert abs(losses[0] - base) < 1e-9, f"first loss {losses[0]:.4f} but the base model’s loss is {base:.4f}: B must start at zero"
assert losses[-1] < 0.05, f"final loss {losses[-1]:.4f}; it should fall well below 0.05"` },
      { name: 'the habit changed: warm for all, safety first for the PIN and OTP questions', code: `import numpy as np
A, B, _ = train_lora(X_TRAIN, Y_TRAIN, W_BASE)
before = [STYLES[i] for i in (X_TRAIN @ W_BASE).argmax(axis=1)]
after = [STYLES[i] for i in lora_logits(X_TRAIN, W_BASE, A, B, 2).argmax(axis=1)]
assert set(before) == {"formal"}, f"the base model should be formal everywhere: {before}"
assert after == [STYLES[i] for i in Y_TRAIN], f"got {after}"` },
      { name: 'it carries over to a question it never saw, and unplugging the adapter gives the base model back', code: `import numpy as np
A, B, _ = train_lora(X_TRAIN, Y_TRAIN, W_BASE)
x = features("a caller wants my otp")[None]
got = STYLES[int(lora_logits(x, W_BASE, A, B, 2).argmax())]
assert got == "safety first", f"new OTP question styled {got!r}"
assert np.allclose(lora_logits(X_TRAIN, W_BASE, A * 0, B * 0, 2), X_TRAIN @ W_BASE), "removing the adapter must restore the base layer"
merged = W_BASE + A @ B * 2
assert np.allclose(X_TRAIN @ merged, lora_logits(X_TRAIN, W_BASE, A, B, 2)), "merging W + A B (alpha / r) must give the same outputs"` },
    ],
    hints: [
      'Set up: `rng = np.random.default_rng(seed)`, `A = rng.normal(size=(X.shape[1], r)) * 0.01`, `B = np.zeros((r, W.shape[1]))`, `Y = np.eye(W.shape[1])[y]`.',
      'Inside the loop compute the logits and the loss first, then `G = (softmax(logits) - Y) / len(y)`. Both gradients come from G, and both must be computed before either matrix is updated.',
      'The gradient for B goes through A (`(X @ A).T @ G`) and the gradient for A goes through B (`G @ B.T`). That is why B = 0 is safe on step 0: B still learns, and A starts learning one step later.',
    ],
    explanation: `The frozen layer never moves, so the “Paisa Pal tone” lives entirely in 148 numbers of A and B, and you can peel it off to get the original layer back exactly, or merge it into W for free at serving time. Starting B at zero means training begins at the pretrained behaviour and moves away gradually. A rank-2 correction is enough here because the wanted change is simple: push “formal” down everywhere, and push “safety first” up where PIN and OTP words appear. That is the kind of low-rank nudge LoRA is good at.

The toy is one tiny layer, so the saving is small (148 trained numbers against 213). In a real model LoRA wraps the attention layers of every block, W is 4096 by 4096, the gradients come from backpropagation through the whole network (PyTorch computes the two lines you wrote), and the data is a few hundred reviewed example replies rather than 12 labels.`,
    source: 'phase4-modern-llms/finetune_tiny_gpt.py',
  },

  /* ---------- agents ---------- */
  {
    id: 'agents-code-bot-tool-loop',
    lesson: 'agents',
    project: { piece: 'Tool loop' },
    title: 'The tool loop: check a balance, check a refund, then answer',
    prelude: `${PAISA_PAL}\n${AGENT_GIVEN}`,
    prompt: `Riya’s customers ask about their own money. The answers are in \`get_balance\` and \`refund_status\`, not in any page or weight. Put the model inside a loop.

\`parse_action(text)\` reads ONE model reply, as in \`mini_agent.py\`:
- if it contains \`ANSWER:\`, return \`("answer", everything after it, stripped)\`;
- else if it contains \`TOOL: <name>\` followed by \`ARGS: {json}\`, return \`("tool", (name, args_dict))\`;
- else return \`("answer", text.strip())\`.

\`run_bot(question, model=scripted_model, max_steps=4)\` runs the loop. At each step, build the context exactly as
\`f"{SYSTEM_PROMPT}\\n\\nUSER QUESTION: {question}\\n\\n" + "\\n".join(scratchpad)\`,
call \`model(context)\`, and parse ONLY the model’s reply. On an answer, return it. On a tool call, run it from \`BOT_TOOLS\` and append \`f"{text}\\nRESULT: {result}"\` to the scratchpad. An unknown tool gives the result \`"ERROR: unknown tool <name>"\`, and a tool that raises gives a result starting with \`"ERROR"\`: errors go back to the model as text, the loop does not crash. After \`max_steps\` model calls with no answer, return \`"(step budget exhausted)"\`.`,
    starter: `def parse_action(text):
    """("answer", final text) or ("tool", (name, args dict)), from one model reply."""
    ...

def run_bot(question, model=scripted_model, max_steps=4):
    """Call the model in a loop, running the tools it asks for, until it answers or the budget runs out."""
    ...

print(run_bot("What is my balance? I am customer C101."))
`,
    solution: `def parse_action(text):
    """("answer", final text) or ("tool", (name, args dict)), from one model reply."""
    ans = re.search(r"ANSWER:\\s*(.+)", text, re.S)
    if ans:
        return ("answer", ans.group(1).strip())
    tool = re.search(r"TOOL:\\s*(\\w+)\\s*ARGS:\\s*(\\{.*?\\})", text, re.S)
    if tool:
        return ("tool", (tool.group(1), json.loads(tool.group(2))))
    return ("answer", text.strip())                 # malformed: treat as final

def run_bot(question, model=scripted_model, max_steps=4):
    """Call the model in a loop, running the tools it asks for, until it answers or the budget runs out."""
    scratchpad = []
    for _ in range(max_steps):
        context = f"{SYSTEM_PROMPT}\\n\\nUSER QUESTION: {question}\\n\\n" + "\\n".join(scratchpad)
        text = model(context)                       # the model only ever writes text
        kind, payload = parse_action(text)          # parse its reply, never the tool output
        if kind == "answer":
            return payload
        name, args = payload                        # our code does the acting
        if name not in BOT_TOOLS:
            result = f"ERROR: unknown tool {name}"
        else:
            try:
                result = BOT_TOOLS[name](**args)
            except Exception as e:
                result = f"ERROR: {type(e).__name__}: {e}"
        scratchpad.append(f"{text}\\nRESULT: {result}")
    return "(step budget exhausted)"

print(run_bot("What is my balance? I am customer C101."))
print(run_bot("Where is the refund for T9001?"))
`,
    tests: [
      { name: 'parse_action reads a tool call and an answer', code: `kind, payload = parse_action('Thought: I need the refund status.\\nTOOL: refund_status\\nARGS: {"transaction_id": "T9001"}')
assert (kind, payload) == ("tool", ("refund_status", {"transaction_id": "T9001"})), f"got {(kind, payload)}"
got = parse_action("Thought: done.\\nANSWER: Your refund is in progress.")
assert got == ("answer", "Your refund is in progress."), f"got {got}"
got = parse_action("Sure, it is probably fine.")
assert got == ("answer", "Sure, it is probably fine."), f"no format at all should count as the final answer; got {got}"` },
      { name: 'Riya’s balance and her refund come from the tools', code: `r = run_bot("What is my balance? I am customer C101.")
assert "2450.0" in r, f"get_balance('C101') is 2450.0; got {r!r}"
r = run_bot("Where is the refund for T9001?")
assert "in progress" in r, f"refund_status('T9001') is 'in progress'; got {r!r}"` },
      { name: 'the tool result goes back into the next model call', code: `seen = []
def recording(context):
    seen.append(context)
    return scripted_model(context)
run_bot("Where is the refund for T9001?", model=recording)
assert len(seen) == 2, f"expected 2 model calls (ask for the tool, then answer), got {len(seen)}"
assert "RESULT:" not in seen[0] and "RESULT: in progress" in seen[1], f"second context ends with: {seen[-1][-120:]!r}"` },
      { name: 'errors come back as text and the loop survives', code: `seen = []
def wants_send(context):
    seen.append(context)
    if "RESULT:" in context:
        return "Thought: I cannot do that.\\nANSWER: I cannot send money from chat."
    return 'Thought: Move the money.\\nTOOL: send_money\\nARGS: {"customer_id": "C101", "to": "C102", "amount": 500}'
r = run_bot("send 500 to Dev", model=wants_send)
assert "ERROR: unknown tool send_money" in seen[-1], "send_money is not in BOT_TOOLS: its request should come back as an ERROR result"
assert r == "I cannot send money from chat.", f"got {r!r}"
r = run_bot("What is my balance? I am customer C999.")
assert "could not look that up" in r, f"get_balance('C999') raises KeyError; the loop should turn it into an ERROR result; got {r!r}"` },
      { name: 'a model that never answers is stopped by the step budget', code: `calls = []
def stuck(context):
    calls.append(1)
    return 'Thought: again.\\nTOOL: get_balance\\nARGS: {"customer_id": "C101"}'
r = run_bot("What is my balance? I am customer C101.", model=stuck, max_steps=3)
assert len(calls) == 3, f"max_steps=3 must mean exactly 3 model calls, got {len(calls)}"
assert "budget" in r, f"got {r!r}"` },
    ],
    hints: [
      '`parse_action` is two regular expressions: `re.search(r"ANSWER:\\s*(.+)", text, re.S)` first, then `re.search(r"TOOL:\\s*(\\w+)\\s*ARGS:\\s*(\\{.*?\\})", text, re.S)` and `json.loads` on the second group.',
      'In `run_bot` use `for _ in range(max_steps):`, never `while True`. Build the context fresh each lap from the question and the scratchpad list, because the model remembers nothing between calls.',
      'Run a tool with `BOT_TOOLS[name](**args)` inside `try` / `except Exception as e`, and check `name not in BOT_TOOLS` before that. Whatever happens, append the model’s text plus `"\\nRESULT: " + str(result)` to the scratchpad.',
    ],
    explanation: `Everything that happens is your code: the model only writes text, and the loop decides what that text is allowed to cause. Parsing only the model’s latest reply (not the whole context) means a tool result containing “TOOL: send_money” can never be executed by accident. Returning errors as RESULT text is what lets a model notice a bad id and try again. And the step budget is the only stopping rule you control; the other one (ANSWER) is up to the model.

A real agent replaces \`scripted_model\` with an API call and nothing else in the loop changes. What production adds is around the loop: the provider’s structured tool-call format instead of regular expressions, argument validation against a schema and against the logged-in customer, approval before anything moves money (the Guardrails piece in Part 10), and a spend limit alongside the step limit.`,
    source: 'phase5-agents/mini_agent.py',
  },

  /* ---------- evals ---------- */
  {
    id: 'evals-code-bot-eval',
    lesson: 'evals',
    project: { piece: 'Eval harness' },
    title: 'Score the bot on the 12 real questions',
    prelude: `${PAISA_PAL}\n${EVAL_GIVEN}`,
    prompt: `The demo looked great. Now score the bot on the 12 labelled customer questions in \`QUESTIONS\`. A bot is any function \`bot(question)\` that returns \`{"page": ..., "answer": ...}\`, with \`page = None\` meaning it refused.

\`score(page, fact, out)\` marks one question and returns \`(passed, reason)\`:
- if the label \`page\` is None, the bot must refuse (\`out["page"] is None\`);
- otherwise it fails if it refused, fails if \`out["page"] != page\`, and fails if \`fact\` is not in \`out["answer"]\` (ignore case). Only then it passes.

\`evaluate(bot, seed=0)\` runs the bot on every question and returns a dict:
- \`"accuracy"\`: the share of questions passed;
- \`"interval"\`: \`bootstrap_interval(marks, seed=seed)\` (given) over the 0/1 marks;
- \`"page_accuracy"\`: over the 10 answerable questions only, the share where \`out["page"]\` is the right page (a component metric: it ignores the answer);
- \`"refusal_accuracy"\`: over the 2 unanswerable questions only, the share that were refused;
- \`"failures"\`: a list of \`(question, reason)\` for every failed question.`,
    starter: `def score(page, fact, out):
    """(passed, reason) for one question, from its labels and the bot's output."""
    ...

def evaluate(bot, seed=0):
    """Run the bot on QUESTIONS: accuracy, interval, page_accuracy, refusal_accuracy, failures."""
    ...

print(evaluate(toy_bot))
`,
    solution: `def score(page, fact, out):
    """(passed, reason) for one question, from its labels and the bot's output."""
    if page is None:
        return (out["page"] is None, "refused" if out["page"] is None else "answered a question no page answers")
    if out["page"] is None:
        return False, "refused an answerable question"
    if out["page"] != page:
        return False, f"used page {out['page']!r}, not {page!r}"
    if fact.lower() not in out["answer"].lower():
        return False, f"{fact!r} is not in the answer"
    return True, "right page, required fact present"

def evaluate(bot, seed=0):
    """Run the bot on QUESTIONS: accuracy, interval, page_accuracy, refusal_accuracy, failures."""
    marks, failures, page_hits, refusals = [], [], [], []
    for q, page, fact in QUESTIONS:
        out = bot(q)
        ok, reason = score(page, fact, out)
        marks.append(int(ok))
        if not ok:
            failures.append((q, reason))
        if page is None:
            refusals.append(out["page"] is None)
        else:
            page_hits.append(out["page"] == page)   # component metric: ignores the answer
    return {"accuracy": float(np.mean(marks)), "interval": bootstrap_interval(marks, seed=seed),
            "page_accuracy": float(np.mean(page_hits)), "refusal_accuracy": float(np.mean(refusals)),
            "failures": failures}

report = evaluate(toy_bot)
print(f"accuracy {report['accuracy']:.1%}, 95% interval {report['interval'][0]:.0%} to {report['interval'][1]:.0%}")
print(f"right page {report['page_accuracy']:.0%}, refusals {report['refusal_accuracy']:.0%}")
for q, reason in report["failures"]:
    print(" ", q, "->", reason)
`,
    tests: [
      { name: 'score marks one question by the rubric', code: `ans = {"page": "cashback", "answer": "Cashback is credited within 48 hours of an eligible payment."}
assert score("cashback", "48 hours", ans)[0] is True, f"right page and fact present: {score('cashback', '48 hours', ans)}"
assert score("cashback", "100 rupees", ans)[0] is False, "right page, but the required fact is missing: must fail"
assert score("kyc", "PAN", ans)[0] is False, "wrong page must fail"
refusal = {"page": None, "answer": "Sorry, I can only help with Paisa Pal payments."}
assert score("refunds", "3 working days", refusal)[0] is False, "refusing an answerable question must fail"
assert score(None, None, refusal)[0] is True and score(None, None, ans)[0] is False, "the biryani question must be refused, and answering it must fail"` },
      { name: 'the toy bot: every page right, only 8 of 12 answers right', code: `r = evaluate(toy_bot)
assert abs(r["accuracy"] - 8 / 12) < 1e-9, f"accuracy {r['accuracy']:.3f}, expected 8/12 = 0.667"
assert r["page_accuracy"] == 1.0, f"page accuracy {r['page_accuracy']}: the keyword bot finds all 10 pages"
failed = [q for q, _ in r["failures"]]
assert len(failed) == 4 and "i paid 50 rupees, why no cashback" in failed, f"failures: {failed}"` },
      { name: 'a bot that refuses everything is perfect on refusals and useless overall', code: `r = evaluate(lambda q: {"page": None, "answer": "Sorry."})
assert r["refusal_accuracy"] == 1.0 and r["page_accuracy"] == 0.0, f"refusal {r['refusal_accuracy']}, page {r['page_accuracy']}"
assert abs(r["accuracy"] - 2 / 12) < 1e-9, f"accuracy {r['accuracy']:.3f}, expected 2/12"
r = evaluate(lambda q: {"page": "refunds", "answer": HELP_PAGES["refunds"]})
assert r["refusal_accuracy"] == 0.0, f"a bot that never refuses must score 0 on refusals, got {r['refusal_accuracy']}"` },
      { name: '12 questions give a wide interval: the score is not one number', code: `a = evaluate(toy_bot, seed=3)["interval"]; b = evaluate(toy_bot, seed=3)["interval"]
assert a == b, f"{a} vs {b}: pass the seed to bootstrap_interval"
lo, hi = evaluate(toy_bot)["interval"]
assert lo < 8 / 12 < hi, f"interval ({lo:.2f}, {hi:.2f}) should contain 0.667"
assert hi - lo > 0.3, f"width {hi - lo:.2f}: with 12 questions it should be wider than 30 points"` },
    ],
    hints: [
      '`score`: handle the unanswerable case first (`page is None`), then check in order: refused, wrong page, missing fact. Compare with `fact.lower() in out["answer"].lower()`.',
      '`evaluate`: loop over `for q, page, fact in QUESTIONS`, call `bot(q)` once, and keep four lists: the 0/1 marks, the failures, page hits (answerable only) and refusals (unanswerable only).',
      'At the end, `np.mean` of each list gives the rates; pass the marks to `bootstrap_interval(marks, seed=seed)`.',
    ],
    explanation: `The report says where, not only whether. The keyword bot finds the right page for all 10 answerable questions and still gets 4 of them wrong, because it answers with the page’s first sentence and the fact is in the second (“no cashback below 100 rupees”, “never ask for your PIN”). So the retriever is fine and the answerer needs work, exactly the component-versus-end-to-end reading from the lesson. The refuse-everything bot shows why you read the breakdown with the total: a perfect category score can come from a useless system.

The interval is the honest part: 8 of 12 is somewhere between roughly 40% and 90%, so no change measured on these 12 questions can be called better or worse. A real harness uses hundreds or thousands of questions sampled from real tickets, a calibrated judge (often an LLM with a rubric) instead of a substring check, several runs per item when the model samples, and a paired test against the current bot before anything ships.`,
    source: 'phase6-engineering/eval_harness.py',
  },
]

export default exercises
