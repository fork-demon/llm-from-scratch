import type { CodeExerciseDef } from './types'
import { PAISA_PAL } from '../project/paisaPal'

// The capstone of the Paisa Pal project: the learner assembles the whole support bot from the pieces
// built in Parts 8 to 10. Working versions of those pieces are given, so this stands alone.
const GIVEN = "# ---- pieces you built earlier, given here in working form ----\nimport re, math\n\nSTOP = set(\"a an the is are was be to of in on for and or but my i me you your it its this that do does did can how what why when who much many long with from at by not no after paisa pal\".split())  # the brand name is on every page: it says nothing\n\ndef _terms(text):\n    return [t for t in tokenize(text) if t not in STOP]\n\n_DOCS = {pid: _terms(text) for pid, text in HELP_PAGES.items()}\n_IDF = {t: math.log(len(_DOCS) / sum(t in d for d in _DOCS.values())) + 1.0 for d in _DOCS.values() for t in d}\n\ndef _vec(terms):\n    v = {}\n    for t in terms:\n        if t in _IDF:\n            v[t] = v.get(t, 0.0) + _IDF[t]\n    return v\n\ndef _cos(a, b):\n    dot = sum(a[t] * b.get(t, 0.0) for t in a)\n    na = math.sqrt(sum(x * x for x in a.values())); nb = math.sqrt(sum(x * x for x in b.values()))\n    return dot / (na * nb) if na and nb else 0.0\n\ndef retrieve(question, threshold=0.12):\n    \"\"\"Retriever (Part 9): the help page that best answers the question, or None if nothing is close enough.\"\"\"\n    q = _vec(_terms(question))\n    scores = {pid: _cos(q, _vec(terms)) for pid, terms in _DOCS.items()}\n    best = max(scores, key=scores.get)\n    return (best if scores[best] >= threshold else None), round(scores[best], 3)\n\ndef guarded_call(tool, args, approve=lambda tool, args: False):\n    \"\"\"Guardrails (Part 10): run a known tool with the right arguments; money-moving tools need approval.\"\"\"\n    if tool not in TOOLS:\n        return {\"ok\": False, \"error\": f\"unknown tool {tool}\"}\n    missing = [a for a in TOOLS[tool][\"args\"] if a not in args]\n    if missing:\n        return {\"ok\": False, \"error\": f\"missing arguments: {missing}\"}\n    if TOOLS[tool][\"moves_money\"] and not approve(tool, args):\n        return {\"ok\": False, \"error\": \"needs the customer's approval\"}\n    fn = {\"get_balance\": get_balance, \"refund_status\": refund_status}.get(tool)\n    if fn is None:\n        return {\"ok\": False, \"error\": f\"{tool} is not available in this sandbox\"}\n    return {\"ok\": True, \"result\": fn(*[args[a] for a in TOOLS[tool][\"args\"]])}\n\nREFUSAL = \"Sorry, I can only help with Paisa Pal payments and your account.\"\n\ndef evaluate(bot):\n    \"\"\"Eval harness (Part 9): the share of the 12 real questions the bot handles correctly.\"\"\"\n    right = 0\n    for question, page, fact in QUESTIONS:\n        reply = bot(question)\n        ok = (reply == REFUSAL) if page is None else (fact.lower() in reply.lower() and reply != REFUSAL)\n        right += ok\n    return right / len(QUESTIONS)\n"

const exercises: CodeExerciseDef[] = [
  {
    id: 'capstone-code-bot-assemble',
    lesson: 'capstone',
    project: { piece: 'The whole bot' },
    title: 'Assemble the Paisa Pal bot and pass the eval',
    prelude: `${PAISA_PAL}\n${GIVEN}`,
    prompt: `Everything you built for Paisa Pal is in the project code above, in working form: \`retrieve(question)\` (the retriever), \`guarded_call(tool, args, approve)\` (the guardrails around the tools) and \`evaluate(bot)\` (the eval harness over the 12 real customer questions).

Write \`bot(question, customer_id="C101")\`, the whole support bot. It must:

1. **Account questions use tools.** If the question mentions "balance", call \`get_balance\` through \`guarded_call\` and reply \`"Your balance is 2,450.00 rupees."\` (the amount with thousands commas and 2 decimals). If it mentions a transaction id such as \`T9001\`, call \`refund_status\` through \`guarded_call\` and include the result in the reply.
2. **Money never moves without approval.** If the question starts with "send" (for example \`"send 500 to C102"\`), ask \`guarded_call\` for \`send_money\` without an approval, and reply with a sentence that says you need the customer's approval.
3. **Everything else is answered from the help pages.** Retrieve the page. If there is none, return \`REFUSAL\` exactly. Otherwise reply with the page's text, so every answer is grounded in a real Paisa Pal page and nothing is invented.

\`evaluate(bot)\` must reach 1.0: all 10 answerable questions carry their required fact, and the biryani and share-price questions are refused.`,
    starter: `def bot(question, customer_id="C101"):
    """Paisa Pal support bot: tools for account questions, approval before money moves,
    help pages for everything else, and a refusal when no page answers."""
    return REFUSAL
`,
    solution: `import re

def bot(question, customer_id="C101"):
    """Paisa Pal support bot: tools for account questions, approval before money moves,
    help pages for everything else, and a refusal when no page answers."""
    q = question.lower()

    # 1. account questions: the tools, through the guardrails
    if "balance" in q:
        r = guarded_call("get_balance", {"customer_id": customer_id})
        return f"Your balance is {r['result']:,.2f} rupees." if r["ok"] else f"I could not check that: {r['error']}."
    txn = re.search(r"\\bt\\d{4}\\b", q)
    if txn:
        tid = txn.group(0).upper()
        r = guarded_call("refund_status", {"transaction_id": tid})
        return f"Refund for {tid}: {r['result']}." if r["ok"] else f"I could not check {tid}: {r['error']}."

    # 2. moving money needs the customer's approval, which a chat message is not
    send = re.match(r"send (\\d+) to (c\\d+)", q)
    if send:
        r = guarded_call("send_money", {"customer_id": customer_id, "to": send.group(2).upper(), "amount": float(send.group(1))})
        return "I need your approval in the app before sending money." if not r["ok"] else "Sent."

    # 3. everything else: answer from a help page, or refuse
    page, _ = retrieve(question)
    return REFUSAL if page is None else HELP_PAGES[page]
`,
    tests: [
      { name: 'passes the eval: 12 of 12 questions', code: `score = evaluate(bot)\nmissed = [q for q, p, f in QUESTIONS if (bot(q) != REFUSAL if p is None else f.lower() not in bot(q).lower())]\nassert score == 1.0, f"evaluate(bot) = {score:.2f}; questions handled wrongly: {missed}"` },
      { name: 'reads Riya’s balance with a tool', code: `r = bot("what is my balance?")\nassert "2,450.00" in r, f"expected the balance 2,450.00 from get_balance, got {r!r}"` },
      { name: 'checks a refund with a tool', code: `r = bot("where is my refund for T9001")\nassert "in progress" in r.lower(), f"refund_status('T9001') says 'in progress'; got {r!r}"` },
      { name: 'never moves money without approval', code: `before = CUSTOMERS["C101"]["balance"]\nr = bot("send 500 to C102")\nassert "approval" in r.lower(), f"the bot must ask for approval before sending money; got {r!r}"\nassert CUSTOMERS["C101"]["balance"] == before, "a balance changed"` },
      { name: 'answers are grounded in the help pages', code: `for q, p, f in QUESTIONS:\n    if p is not None:\n        r = bot(q)\n        assert r in HELP_PAGES.values(), f"for {q!r} the reply is not a Paisa Pal help page: {r!r}"` },
    ],
    hints: [
      'Work in the order of the prompt, and return as soon as one case matches: balance, then a transaction id (`re.search(r"\\\\bt\\\\d{4}\\\\b", question.lower())`), then "send", then the help pages.',
      '`guarded_call` returns a dict: `{"ok": True, "result": ...}` or `{"ok": False, "error": ...}`. Format the balance with `f"{amount:,.2f}"`.',
      'For the last case: `page, score = retrieve(question)`, then `return REFUSAL if page is None else HELP_PAGES[page]`.',
    ],
    explanation: `The bot is a router around three things you built: tools behind guardrails for anything about the customer's account, the retriever for anything the help centre answers, and a refusal for everything else. Nothing is invented: every answer is either a tool result or a real help page, which is what the "grounded" test checks. And the eval is what lets you change any piece (a better retriever, a real model writing the replies) and know at once whether the bot still works.

What a real support bot adds: a language model that writes the reply from the retrieved page instead of returning it whole, a model (not keywords) deciding which tool to call, an approval step inside the app, and an eval set of thousands of real questions rather than twelve. The shape stays the same.`,
  },
]

export default exercises
