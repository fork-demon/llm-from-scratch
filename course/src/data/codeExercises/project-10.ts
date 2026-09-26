import type { CodeExerciseDef } from './types'
import { PAISA_PAL } from '../project/paisaPal'

// Part 10 of the Paisa Pal project: shrink the bot to int8, and put guardrails around its tools.
// Each stands alone: the prelude holds everything the piece needs that is not the point of the lesson.

// A small support bot, trained in the prelude so the int8 exercise has real weights to shrink.
const GIVEN_BOT = `
# ---- given: Riya's small trained support bot (words in, one hidden layer, a help page out) ----
PAGES = list(HELP_PAGES) + ["none"]
VOCAB = sorted({w for t in list(HELP_PAGES.values()) + [q for q, _, _ in QUESTIONS] for w in tokenize(t)})
INDEX = {w: i for i, w in enumerate(VOCAB)}

def bag(text):
    x = np.zeros(len(VOCAB), dtype=np.float32)
    for w in tokenize(text):
        if w in INDEX:
            x[INDEX[w]] += 1
    return x

def predict(bot, question):
    """The page the bot would answer from ("none" means refuse). bot is a dict of float matrices."""
    h = np.tanh(bag(question) @ bot["embed"])
    return PAGES[int(np.argmax(h @ bot["out"]))]

def train_bot(seed=0, steps=300):
    rng = np.random.default_rng(seed)
    texts = list(HELP_PAGES.values()) + [q for q, _, _ in QUESTIONS]
    y = [PAGES.index(p) for p in HELP_PAGES] + [PAGES.index(p or "none") for _, p, _ in QUESTIONS]
    X = np.stack([bag(t) for t in texts])
    E, O = rng.normal(0, 0.1, (len(VOCAB), 64)), rng.normal(0, 0.1, (64, len(PAGES)))
    for _ in range(steps):                       # plain gradient descent on cross-entropy
        H = np.tanh(X @ E)
        G = softmax(H @ O); G[np.arange(len(y)), y] -= 1; G /= len(y)
        gO, gE = H.T @ G, X.T @ (G @ O.T * (1 - H ** 2))
        E, O = E - gE, O - gO
    return {"embed": E.astype(np.float32), "out": O.astype(np.float32)}

BOT = train_bot()    # float32 weights: "embed" is 136 x 64, "out" is 64 x 7
`

// What the tools actually do, plus the fail-closed default approver.
const GIVEN_TOOLS = `
# ---- given: what the tools actually do ----
NEW_DEVICES = {"C101"}      # Riya switched phones this morning
NEW_DEVICE_LIMIT = 5000.0   # the "limits" help page: 5,000 rupees in a new device's first 24 hours
LEDGER = []                 # every rupee that actually moved

def send_money(customer_id, to, amount):
    CUSTOMERS[customer_id]["balance"] -= amount
    LEDGER.append((customer_id, to, amount))
    return f"sent {amount:,.2f} rupees from {customer_id} to {to}"

RUN = {"get_balance": get_balance, "refund_status": refund_status, "send_money": send_money}

def deny_all(name, args):
    """The default approver: no human is attached, so nothing that moves money runs. Fail closed."""
    return False
`

const exercises: CodeExerciseDef[] = [
  /* ---------- making-models-cheaper: int8 weights ---------- */
  {
    id: 'making-models-cheaper-code-bot-int8',
    lesson: 'making-models-cheaper',
    project: { piece: 'int8 weights' },
    title: 'Shrink the support bot to int8, and measure what it costs',
    prelude: `${PAISA_PAL}\n${GIVEN_BOT}`,
    prompt: `Finance wants the bot cheaper. The project code above trains Riya's small support bot: \`BOT\` is a dict of two float32 matrices, and \`predict(bot, question)\` returns the help page it would answer from ("none" means refuse).

Write three functions:

- \`quantize_bot(bot)\` returns a dict with the same keys, each mapped to \`(q, scale)\`. \`scale\` is float32 with shape (rows, 1): each row's largest absolute weight divided by 127 (1.0 for an all-zero row). \`q = round(W / scale)\`, clipped to [-127, 127], stored as \`np.int8\`.
- \`dequantize_bot(qbot)\` returns a dict of float32 matrices, \`q * scale\`, that \`predict\` can use.
- \`report(bot)\` returns \`{"float_bytes": ..., "int8_bytes": ..., "agreement": ...}\`: the bytes of the float bot, the bytes of the int8 bot (the codes AND the scales, use \`.nbytes\`), and the fraction of the 12 \`QUESTIONS\` on which the int8 bot picks the same page as the float bot.

The slide for finance is the ratio of the first two numbers. The third one is what you check before you are allowed to show it.`,
    starter: `def quantize_bot(bot):
    """{name: W} -> {name: (q, scale)}: int8 codes and one float32 scale per row (max |w| / 127)."""
    ...

def dequantize_bot(qbot):
    """{name: (q, scale)} -> {name: q * scale}, as float32 matrices."""
    ...

def report(bot):
    """{"float_bytes": ..., "int8_bytes": ..., "agreement": ...} for the 12 QUESTIONS."""
    ...

print(report(BOT))
`,
    solution: `def quantize_bot(bot):
    """{name: W} -> {name: (q, scale)}: int8 codes and one float32 scale per row (max |w| / 127)."""
    qbot = {}
    for name, W in bot.items():
        scale = (np.abs(W).max(axis=1, keepdims=True) / 127).astype(np.float32)
        scale[scale == 0] = 1.0                    # an all-zero row: any scale works, avoid 0 / 0
        q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
        qbot[name] = (q, scale)
    return qbot

def dequantize_bot(qbot):
    """{name: (q, scale)} -> {name: q * scale}, as float32 matrices."""
    return {name: q.astype(np.float32) * scale for name, (q, scale) in qbot.items()}

def report(bot):
    """{"float_bytes": ..., "int8_bytes": ..., "agreement": ...} for the 12 QUESTIONS."""
    qbot = quantize_bot(bot)
    small = dequantize_bot(qbot)
    same = [predict(bot, q) == predict(small, q) for q, _, _ in QUESTIONS]
    return {
        "float_bytes": sum(W.nbytes for W in bot.values()),
        "int8_bytes": sum(q.nbytes + s.nbytes for q, s in qbot.values()),   # the scales are stored too
        "agreement": sum(same) / len(same),
    }

r = report(BOT)
print(f"{r['float_bytes']:,} bytes -> {r['int8_bytes']:,} bytes ({r['float_bytes'] / r['int8_bytes']:.2f}x smaller)")
print(f"same page on {r['agreement']:.0%} of the questions")
`,
    tests: [
      { name: 'each matrix becomes int8 codes plus one scale per row', code: `import numpy as np\nqb = quantize_bot(BOT)\nassert set(qb) == {"embed", "out"}, f"keys {set(qb)}"\nfor name, W in BOT.items():\n    q, s = qb[name]\n    assert q.dtype == np.int8 and q.shape == W.shape, f"{name}: q is {q.dtype} {q.shape}, expected int8 {W.shape}"\n    assert s.dtype == np.float32 and s.shape == (W.shape[0], 1), f"{name}: scale is {s.dtype} {s.shape}, expected float32 ({W.shape[0]}, 1)"\n    assert (np.abs(q).max(axis=1) == 127).all(), f"{name}: every row's largest weight should be stored as +127 or -127"\n    err = np.abs(dequantize_bot(qb)[name] - W)\n    assert (err <= s / 2 + 1e-6).all(), f"{name}: worst round-trip error {err.max():.3g} is more than half a step"\nz = dequantize_bot(quantize_bot({"w": np.array([[0, 0], [1, -2]], dtype=np.float32)}))["w"]\nassert np.isfinite(z).all() and np.allclose(z[0], 0), f"an all-zero row must come back as zeros, got {z}"` },
      { name: 'the bot shrinks from 36,608 bytes to 9,952 (3.7x: the scales are the rest)', code: `r = report(BOT)\nassert r["float_bytes"] == 36608, f"float_bytes {r['float_bytes']}: (136 x 64 + 64 x 7) weights at 4 bytes each is 36,608"\nassert r["int8_bytes"] == 9952, f"int8_bytes {r['int8_bytes']}: 9,152 one-byte codes plus 200 float32 scales (one per row) is 9,952"` },
      { name: 'the int8 bot answers every question from the same page', code: `r = report(BOT)\nassert r["agreement"] == 1.0, f"agreement {r['agreement']}; per-row int8 should not change a single one of the 12 answers"\nsmall = dequantize_bot(quantize_bot(BOT))\ngot = predict(small, "my upi payment failed but the money was deducted, when do i get it back")\nassert got == "refunds", f"the failed-UPI question should still go to refunds, got {got}"\ngot = predict(small, "can you recommend a good biryani place in bengaluru")\nassert got == "none", f"the int8 bot must still refuse the biryani question, got {got}"` },
      { name: 'one loud weight: per-row scales keep the bot, one shared scale would not', code: `import numpy as np\nloud = {k: v.copy() for k, v in BOT.items()}\nloud["embed"][INDEX["aadhaar"], 0] = 100.0     # typical weight is about 0.09: this one is over 1,000 times that\nsmall = dequantize_bot(quantize_bot(loud))\nrow_same = sum(predict(loud, q) == predict(small, q) for q, _, _ in QUESTIONS)\nstep = np.abs(loud["embed"]).max() / 127    # what ONE scale for the whole matrix would be\nshared = dict(loud, embed=np.round(loud["embed"] / step) * step)\nshared_same = sum(predict(loud, q) == predict(shared, q) for q, _, _ in QUESTIONS)\nassert row_same == 12, f"with one scale per row the loud word only coarsens its own row; got {row_same} of 12 answers unchanged"\nassert shared_same < row_same, f"(a shared scale kept {shared_same} of 12)"` },
    ],
    hints: [
      '`np.abs(W).max(axis=1, keepdims=True) / 127` is one scale per row with shape (rows, 1). Cast it with `.astype(np.float32)`, then set `scale[scale == 0] = 1.0`.',
      '`q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)`, and dequantize with `q.astype(np.float32) * scale`.',
      'In `report`, the int8 size is `q.nbytes + s.nbytes` summed over the matrices, and agreement compares `predict(bot, q)` with `predict(dequantize_bot(quantize_bot(bot)), q)` for each question in `QUESTIONS`.',
    ],
    explanation: `Every weight now costs one byte instead of four. The bot is not a full 4x smaller because each row also stores a 4-byte scale, and here a row is only 64 weights wide, so the scales add about 6%. In a real model a row is thousands of weights wide and the same scale costs almost nothing, which is why int8 is quoted as 4x against float32 (2x against the 16-bit weights of the lesson's numbers).

The agreement number is the honest half of the slide. Rounding moves every weight by at most half a step, and the step is set by the largest weight in its own row, so one loud word can only coarsen its own row. With one shared scale that same word would round almost every other weight to zero, and half the answers change.

What the real version adds: int8 kernels that multiply the codes directly on the GPU instead of rebuilding floats, methods such as LLM.int8(), GPTQ and AWQ that handle outliers in the activations as well as the weights, and an eval on thousands of real tickets rather than 12 questions before anyone signs off the saving.`,
    source: 'phase6-engineering/quantize_demo.py',
  },

  /* ---------- production-agents: guardrails ---------- */
  {
    id: 'production-agents-code-bot-guardrails',
    lesson: 'production-agents',
    project: { piece: 'Guardrails' },
    title: 'Guardrails: validate every tool call, and ask before money moves',
    prelude: `${PAISA_PAL}\n${GIVEN_TOOLS}`,
    prompt: `The bot's model proposes tool calls as a name and a dict of arguments, and a model guesses. Write \`guarded_call(name, args, approve=deny_all)\`, the only door to the tools. Check in this order and return a string that starts with \`"ERROR:"\` at the first problem, saying what was wrong so the model can fix it:

1. **Known tool.** \`name\` must be in \`TOOLS\`. Say which tools exist.
2. **The schema.** Every argument in \`TOOLS[name]["args"]\` present, no extra ones, and the right type: \`"str"\` means a \`str\`; \`"float"\` means an \`int\` or \`float\` (the string \`"500"\` is not a number).
3. **The help-page rules, for send_money.** The customer is in \`CUSTOMERS\`, the amount is more than 0, no more than their balance, and no more than \`NEW_DEVICE_LIMIT\` if they are in \`NEW_DEVICES\`.
4. **Approval.** If \`TOOLS[name]["moves_money"]\`, call \`approve(name, args)\` and run only if it returns \`True\`. Otherwise refuse and tell the model not to retry.

Only then run \`RUN[name](**args)\` and return its result. Read-only tools never ask anyone. The default approver says no: with no human attached, nothing moves.`,
    starter: `def guarded_call(name, args, approve=deny_all):
    """The only way the bot may use a tool: validate, check the rules, ask, then run.
    Returns the tool's result, or a string starting with "ERROR:" that the model can act on."""
    return RUN[name](**args)     # the naive version: whatever the model asks for, runs
`,
    solution: `TYPES = {"str": str, "float": (int, float)}

def guarded_call(name, args, approve=deny_all):
    """The only way the bot may use a tool: validate, check the rules, ask, then run.
    Returns the tool's result, or a string starting with "ERROR:" that the model can act on."""
    if name not in TOOLS:
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(TOOLS)}."
    schema = TOOLS[name]["args"]
    problems = [f'missing "{k}"' for k in schema if k not in args]
    problems += [f'unexpected "{k}"' for k in args if k not in schema]
    problems += [f'"{k}" must be a {t}' for k, t in schema.items()
                 if k in args and not isinstance(args[k], TYPES[t])]
    if problems:
        expected = ", ".join(f'"{k}": {t}' for k, t in schema.items())
        return f"ERROR: invalid arguments for {name}: {'; '.join(problems)}. Expected {{{expected}}}. Fix the arguments and call again."

    if name == "send_money":
        who, amount = args["customer_id"], args["amount"]
        if who not in CUSTOMERS:
            return f"ERROR: unknown customer {who!r}."
        balance = CUSTOMERS[who]["balance"]
        if amount <= 0:
            return "ERROR: amount must be more than 0 rupees."
        if amount > balance:
            return f"ERROR: {amount:,.2f} rupees is more than the balance of {balance:,.2f}."
        if who in NEW_DEVICES and amount > NEW_DEVICE_LIMIT:
            return f"ERROR: a new device can send at most {NEW_DEVICE_LIMIT:,.0f} rupees in its first 24 hours."

    if TOOLS[name]["moves_money"] and approve(name, args) is not True:
        return f"ERROR: {name} was not approved by a human. Do not retry. Tell the customer what you would have done."
    return RUN[name](**args)

print(guarded_call("send_money", {"customer_id": "C101", "to": "C102", "amount": 500.0}))
print(guarded_call("get_balance", {"customer_id": "C101"}))
`,
    tests: [
      { name: 'reading is free: get_balance and refund_status run without asking anyone', code: `CUSTOMERS["C101"]["balance"] = 2450.0; LEDGER.clear()\nasked = []\nwho = lambda name, args: asked.append(name) or True\nb = guarded_call("get_balance", {"customer_id": "C101"}, approve=who)\nassert b == 2450.0, f"Riya's balance is 2450.0, got {b!r}"\nr = guarded_call("refund_status", {"transaction_id": "T9001"}, approve=who)\nassert r == "in progress", f"T9001's refund is 'in progress', got {r!r}"\nassert asked == [], f"read-only tools must not ask for approval, but approve was asked about {asked}"` },
      { name: 'a made-up tool or bad arguments get an error the model can act on', code: `CUSTOMERS["C101"]["balance"] = 2450.0; LEDGER.clear()\nasked = []\nyes = lambda name, args: asked.append(name) or True\nr = guarded_call("refund_money", {"transaction_id": "T9001"}, approve=yes)\nassert isinstance(r, str) and r.startswith("ERROR:") and "refund_status" in r, f"an unknown tool should get an ERROR that lists the real tools, got {r!r}"\nr = guarded_call("send_money", {"customer_id": "C101", "to": "C102", "amount": "500"}, approve=yes)\nassert isinstance(r, str) and r.startswith("ERROR:") and "amount" in r, f"the string '500' is not a number: expected an ERROR naming amount, got {r!r}"\nr = guarded_call("send_money", {"customer_id": "C101", "amount": 500.0}, approve=yes)\nassert isinstance(r, str) and r.startswith("ERROR:") and '"to"' in r, f"expected an ERROR saying \\"to\\" is missing, got {r!r}"\nr = guarded_call("get_balance", {"customer_id": "C101", "pin": "1234"}, approve=yes)\nassert isinstance(r, str) and r.startswith("ERROR:") and "pin" in r, f"an extra argument should be refused, got {r!r}"\nassert asked == [] and LEDGER == [], f"malformed calls must never reach the approver or move money (asked {asked}, ledger {LEDGER})"` },
      { name: 'the help-page rules: nothing negative, nothing above the balance, 5,000 on a new device', code: `CUSTOMERS["C101"]["balance"] = 2450.0; LEDGER.clear()\nyes = lambda name, args: True\nfor amount, why in [(-500.0, "a negative amount"), (0, "zero"), (5000.0, "more than Riya's 2,450 balance")]:\n    r = guarded_call("send_money", {"customer_id": "C101", "to": "C102", "amount": amount}, approve=yes)\n    assert isinstance(r, str) and r.startswith("ERROR:"), f"{why} must be refused even with approval, got {r!r}"\nr = guarded_call("send_money", {"customer_id": "C999", "to": "C102", "amount": 10.0}, approve=yes)\nassert isinstance(r, str) and r.startswith("ERROR:"), f"an unknown customer must be refused, got {r!r}"\nCUSTOMERS["C101"]["balance"] = 20000.0\nr = guarded_call("send_money", {"customer_id": "C101", "to": "C102", "amount": 6000.0}, approve=yes)\nassert isinstance(r, str) and r.startswith("ERROR:"), f"Riya is on a new phone: 6,000 is over the 5,000 limit, got {r!r}"\nassert LEDGER == [], f"money moved: {LEDGER}"\nCUSTOMERS["C101"]["balance"] = 2450.0` },
      { name: 'money moves only after an explicit yes', code: `CUSTOMERS["C101"]["balance"] = 2450.0; LEDGER.clear()\ncall = {"customer_id": "C101", "to": "C102", "amount": 500.0}\nr = guarded_call("send_money", dict(call))\nassert isinstance(r, str) and r.startswith("ERROR:"), f"with the default approver (no human attached) send_money must be refused, got {r!r}"\nassert LEDGER == [] and CUSTOMERS["C101"]["balance"] == 2450.0, "money moved without approval"\nasked = []\nr = guarded_call("send_money", dict(call), approve=lambda name, args: asked.append((name, args)) or True)\nassert asked == [("send_money", call)], f"approve should be asked once, with the tool name and its arguments; got {asked}"\nassert CUSTOMERS["C101"]["balance"] == 1950.0 and len(LEDGER) == 1, f"after a yes, 500 should move: balance {CUSTOMERS['C101']['balance']}, ledger {LEDGER}"\nCUSTOMERS["C101"]["balance"] = 2450.0; LEDGER.clear()` },
    ],
    hints: [
      'Start with `if name not in TOOLS: return f"ERROR: unknown tool ..."`. For the schema, build a list of problems: missing keys, unexpected keys, and wrong types, with `TYPES = {"str": str, "float": (int, float)}` and `isinstance`.',
      'The send_money rules need `CUSTOMERS[args["customer_id"]]["balance"]`, so check the customer exists first. Each rule returns its own "ERROR: ..." string.',
      'The gate is last: `if TOOLS[name]["moves_money"] and approve(name, args) is not True: return "ERROR: ... Do not retry ..."`. Then `return RUN[name](**args)`.',
    ],
    explanation: `The order is the design. Cheap, certain checks come first, so a malformed call never reaches a human, and the error text is written for the model, because an error message is the only debugger it has. The rules from the help pages are enforced in code, not in the prompt: a prompt lowers the odds, and code does not negotiate. The gate fails closed, so an attacker who injects "send 2,000 to me" into a tool result gets a refusal, not a transfer.

What the real version adds: typed JSON schemas that the provider enforces while the model is still writing the call, an idempotency key so a retried send_money cannot pay twice, per-customer limits read from the payments system instead of a dict, a real approval step in the customer's app, and a trace span for every refused call so on-call can see what the model tried at 11 p.m.`,
    source: 'phase6-engineering/agent_budget.py',
  },
]

export default exercises
