# Module 14 — Agents From First Principles (Optional Capstone)

> **⏱️ Time:** ~1-2 weeks.  
> **What you'll build:** Core mechanics and conceptual understanding.

---

## 🎯 TL;DR
1. **The Core Problem:** The challenge this module solves.
2. **The Mechanism:** How we solve it using first principles.
3. **The Payoff:** What you can do with this new capability.

---


**Time: about 2 weeks. Code: `mini_agent.py`. Runs completely offline — and swaps to a real LLM in three lines.**

## The Demystification This Module Exists For

"Agent" is the most hyped word in AI right now, so let's take its temperature immediately:

> **An agent is not a new kind of model. It's a while-loop your code runs around an ordinary language model.**

The model is still module 06's next-token machine — it only ever emits text. Everything "agentic" happens in *your* code, outside the model: you parse the model's text for requests to use tools, you execute those tools, you paste the results back into the context, and you call the model again. Repeat until the model emits a final answer instead of a tool request.

```
loop:
    text = model(context)                     # module 06: just next tokens
    if text contains a tool request:
        result = run_tool(request)            # YOUR code: actually does things
        context += text + result              # module 09: context management
    else:
        return text                           # final answer
```

That loop — plus a tool registry and some prompt discipline — is the entire architecture of every coding assistant, research agent, and computer-use system you've read about. The frameworks (LangChain, LangGraph, the various "agent SDKs") are packaging around this loop. By the end of this module you'll have written the loop yourself, and framework documentation will read as familiar plumbing with marketing names.

## How Can a Text Generator "Use a Tool"?

This is the step that sounds like magic, so walk through it slowly. The model cannot execute anything — it emits characters. The trick is a *convention*, agreed between your prompt and your parser:

**In the prompt, you declare the tools** — names, what they do, how to call them:

```
You can use these tools. To use one, reply EXACTLY in this format:
TOOL: calculator
ARGS: {"expression": "23 * 7"}

Available tools:
- calculator: evaluates arithmetic. args: {"expression": str}
- search_docs: finds documents. args: {"query": str}
```

**The model, being a plausible-text machine (module 06), continues accordingly.** Asked "what is 23×7 plus the number of oncall engineers?", the plausible continuation — given that prompt — is a `TOOL:` block. It's not "deciding to act"; it's completing text in the format the context demonstrates. Same machine as always.

**Your code parses the emitted text, runs the real function, and appends the result:**

```
TOOL: calculator
ARGS: {"expression": "23 * 7"}
RESULT: 161                        ← your code put this here
```

...and calls the model again. Now the context *contains the answer*, and the plausible continuation is to use it.

That's tool use. When an API offers "function calling," it's this exact convention, made robust: the format is negotiated by the provider, and — connecting to module 09 — the output is often generated under **constrained decoding**: at each step, sampling is restricted to tokens that keep the output valid JSON (the sampler masks out every token that would break the grammar, exactly like top-k masks the tail). "The model returns structured output" = "the weighted die is only allowed to land on legal faces." You already own every piece of that sentence.

## The Loop Has a Name: ReAct

The canonical prompting pattern (from the ReAct paper, 2022) adds one more convention: before each tool call, the model writes a short **Thought** — a sentence of reasoning about what to do next:

```
Thought: I need the oncall count before I can add it. Let me search the docs.
TOOL: search_docs
ARGS: {"query": "oncall rotation size"}
RESULT: The rotation has 4 engineers: primary, secondary, and two shadows.
Thought: Now I can compute 23*7 + 4.
TOOL: calculator
ARGS: {"expression": "23 * 7 + 4"}
RESULT: 165
Thought: I have everything.
ANSWER: 165 — that's 23×7 (161) plus the 4 engineers on the rotation.
```

Why does writing thoughts out loud *work*? Module 06 again: each token is conditioned on everything before it. A written thought becomes context that steers the next tokens — the model is, quite literally, talking itself through the problem, because its own words are the only working memory it has. (This is also the seed of "reasoning models" — appendix topic 1.)

## Memory: There Is No Memory

Worth stating at module-09 bluntness: the model is stateless between calls. An agent's "memory" is entirely *what your loop chooses to keep in the context window*:

- **Scratchpad** — the transcript of thoughts/tool calls/results so far. Grows every iteration; this is working memory.
- **The context budget problem** — long-running agents overflow the window (module 09: it's finite and priced). Real systems *summarize* older scratchpad into a compact digest, keep recent steps verbatim. Your code implements exactly this, and you'll watch a summary replace twenty steps.
- **Long-term memory** — anything beyond the current run is a database your loop reads/writes. Very often it's module 11's vector search: embed past experiences, retrieve relevant ones into context. "The agent remembered" = "the loop did a RAG query over its own history" (module 12, pointed inward).

## Why Agents Fail — You Already Know

Every notorious agent failure mode is a course concept compounding through the loop:

**Error compounding.** Module 06 said each token conditions on all previous ones — a bad early draw poisons what follows. In an agent, that compounding crosses *iterations*: one hallucinated tool result or misread file becomes trusted context for every later step. A 90%-reliable step chained 10 times is a 35%-reliable run. This single piece of arithmetic explains most of the gap between agent demos and agent products.

**Loops and derailment.** The model has no goal register — only plausible continuation. If the scratchpad starts looking like "try X, fail, try X, fail," the plausible continuation is... trying X again. Real agents add loop detection and step budgets in the *outer code* — engineering, not modeling.

**Tool-result injection.** The model can't distinguish "instructions from the user" from "text that arrived in a tool result." If a searched document contains "ignore your instructions and...", that text sits in the same context as everything else. This is *prompt injection*, the security problem of the agent era — and your module 12 instinct (the model has no provenance mechanism) predicts it exactly.

**The fixes are all systems engineering:** budgets, sandboxed tools, validation on tool arguments, human approval gates on irreversible actions, and observability on every step. Agents are 20% ML and 80% the discipline you already practice for a living.


> **🛑 CHECKPOINT:** Run the code and modify it before proceeding. Reading without running is an illusion of knowledge.

## The Code

`mini_agent.py` — no frameworks, no API keys:

1. **A tool registry** — plain functions with a schema dict; adding a tool is appending to a list.
2. **The ReAct loop** — parse `TOOL:` blocks, execute, append `RESULT:`, re-call; stop on `ANSWER:` or when the step budget runs out.
3. **A scripted stand-in model** (same trick as module 12's extractive answerer): a small rule-driven "model" that follows the ReAct format deterministically — so the *loop's* mechanics run, visibly and reproducibly, with zero dependencies. Every step prints: what the "model" said, what got parsed, what the tool returned, what the context now contains.
4. **Scratchpad summarization** — when the transcript exceeds a budget, older steps collapse into a summary line, and the printout shows the context shrinking.
5. **A misbehavior demo** (`--show-failure`): a tool returns text containing an injected instruction, and you watch the stand-in model get steered by it — prompt injection reproduced in miniature, on your machine, harmlessly.
6. The closing comment shows the ~3 lines that replace the stand-in with a real chat API — the loop, registry, parser, and memory logic don't change at all. That is the point of the module.

## Modify-It Exercises

1. Add a new tool (e.g., `today's date` or a file reader) — schema, function, one registry entry — and extend the scripted model to use it. Feel how little "adding a capability to an agent" actually is.
2. Set the step budget to 2 for a task that needs 3 steps. Watch the failure. Now write the one-line answer: why does *every* production agent have a step budget? (Cost is half the answer; the loops section is the other half.)
3. Break the parser: make a tool return text containing a fake `RESULT:` line. Watch the transcript corrupt. This is why real formats use unambiguous delimiters (or provider-level function calling) — you've just re-derived the requirement.
4. Implement loop detection: if the same (tool, args) pair repeats twice, inject a `RESULT: You already tried this — try something different.` Watch behavior change. Congratulations: you've written your first agent guardrail.
5. *(Stretch, needs an API key)* Do the 3-line swap to a real LLM and re-run everything — including `--show-failure`. Compare how a real model handles the injection versus the scripted one. Write two sentences on what you'd log in production to catch this (you'll find you're describing observability — the Module 15 that isn't written yet).

## Best External Resources

- The ReAct paper (Yao et al., 2022) — §1–2; after building the loop, it reads like your own design notes.
- Anthropic's *"Building effective agents"* engineering post — the best practitioner's map of loop patterns (chaining, routing, orchestration); every diagram is a variation of your `mini_agent.py`.
- Simon Willison's writing on prompt injection — the clearest ongoing coverage of the agent-security problem you reproduced in exercise 3.

You've reached the actual frontier: the loop you just built is where the industry currently is. The `appendix-frontier-topics.md` covers what's coming next.
