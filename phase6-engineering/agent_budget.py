"""
Module 16 -- What a production harness adds around the agent loop. No API keys.

  python agent_budget.py           # five short demos, each printed as a trace
  python agent_budget.py --json    # dump the first run's trace as JSON

The loop itself is IMPORTED from phase5-agents/mini_agent.py and not changed.
Everything here wraps it from the outside, which is how real harnesses are built:

  1. TOKEN AND COST ACCOUNTING  every model call is metered (tokens are ESTIMATED
                                as characters / 4; a real tokenizer would differ)
  2. PROMPT CACHING             an unchanged prefix is billed at a discount
  3. HARD BUDGETS               max steps, max tokens, max cost, loop detection:
                                each stops the run with a clear reason
  4. TOOL-ARGUMENT VALIDATION   a tiny schema check BEFORE anything executes
  5. AN APPROVAL GATE           tools marked irreversible need a human "yes"
  6. A STRUCTURED TRACE         one span per model call and per tool call

How the wrapping works: run_agent(question, model=...) takes the model as an
argument, so we pass a METERED model. Tools are looked up in mini_agent.TOOLS at
call time, so we temporarily swap in GUARDED versions and restore them after.

All prices are EXAMPLE numbers and parameters, not any vendor's price list.
"""
import contextlib
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "phase5-agents"))
import mini_agent as agent  # noqa: E402


# ----------------------------------------------------------------------
# 1. TOKENS AND MONEY
# ----------------------------------------------------------------------
def estimate_tokens(text):
    """Roughly 4 characters per token for English. An ESTIMATE: good enough to
    reason about budgets, not good enough to reconcile an invoice."""
    return math.ceil(len(text) / 4)


# Example prices, in dollars per million tokens. Output tokens usually cost
# several times more than input tokens. Cached input is billed at a fraction
# (the read multiplier). Some vendors also charge a premium for writing the
# cache (a write multiplier above 1); others do not (set it to 1.0).
EXAMPLE_PRICES = {
    "input_per_mtok": 3.00,
    "output_per_mtok": 15.00,
    "cache_read_multiplier": 0.10,
    "cache_write_multiplier": 1.25,
}


def call_cost(input_tokens, cached_tokens, output_tokens, prices, caching):
    fresh = input_tokens - cached_tokens
    if caching:
        billed_in = cached_tokens * prices["cache_read_multiplier"] + fresh * prices["cache_write_multiplier"]
    else:
        billed_in = input_tokens
    return (billed_in * prices["input_per_mtok"] + output_tokens * prices["output_per_mtok"]) / 1e6


def shared_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# Latency PLACEHOLDERS so the trace has a time axis. Invented constants with the
# right shape: reading fresh input costs time, cached input costs far less,
# and every output token is a sequential decode step. Measure your own.
def model_latency_ms(fresh_tokens, cached_tokens, output_tokens):
    return int(200 + 0.20 * fresh_tokens + 0.02 * cached_tokens + 20 * output_tokens + 0.5)


TOOL_LATENCY_MS = {"calculator": 5, "search_docs": 120, "read_log": 250, "send_email": 300}


# ----------------------------------------------------------------------
# 2. TWO MORE TOOLS, so there is something risky and something bulky.
# ----------------------------------------------------------------------
OUTBOX = []


def send_email(to: str, body: str) -> str:
    """Pretend to send an email. IRREVERSIBLE in real life: you cannot unsend."""
    OUTBOX.append({"to": to, "body": body})
    return f"sent to {to}"


def read_log(service: str) -> str:
    """Return a bulky tool result: 40 log lines, about 600 tokens."""
    lines = [f"2026-01-15T10:{i:02d}:00Z {service} INFO request_id={1000 + i} status=200 latency_ms={40 + (i * 7) % 50}"
             for i in range(39)]
    lines.append(f"2026-01-15T10:39:00Z {service} ERROR request_id=1039 status=500 upstream timeout")
    return "\n".join(lines)


EXTRA_TOOLS = {
    "send_email": {"fn": send_email, "desc": "sends an email. args: {\"to\": str, \"body\": str}"},
    "read_log": {"fn": read_log, "desc": "returns recent log lines of a service. args: {\"service\": str}"},
}

# A tool schema is an API contract for a caller that guesses. Keep it tiny and strict.
TOOL_SCHEMAS = {
    "calculator": {"expression": str},
    "search_docs": {"query": str},
    "send_email": {"to": str, "body": str},
    "read_log": {"service": str},
}
IRREVERSIBLE = {"send_email"}


def validate_args(name, args):
    """Return None if args fit the schema, else an error the MODEL can act on:
    say what is wrong, what was expected, and what to do next."""
    schema = TOOL_SCHEMAS[name]
    problems = [f'missing "{k}"' for k in schema if k not in args]
    problems += [f'unexpected "{k}"' for k in args if k not in schema]
    problems += [f'"{k}" must be a {t.__name__}' for k, t in schema.items()
                 if k in args and not isinstance(args[k], t)]
    if not problems:
        return None
    expected = "{" + ", ".join(f'"{k}": {t.__name__}' for k, t in schema.items()) + "}"
    return (f"ERROR: invalid arguments for {name}: {'; '.join(problems)}. "
            f"Expected {expected}. Fix ARGS and call the tool again.")


# ----------------------------------------------------------------------
# 3. A SCRIPTED MODEL with three bad habits, so every guard has work to do.
#    Same trick as mini_agent.scripted_model: deterministic, offline.
# ----------------------------------------------------------------------
def scripted_model_v2(context: str) -> str:
    question = re.search(r"USER QUESTION: (.+)", context).group(1)

    if "email" in question.lower():                       # needs an irreversible tool
        if "RESULT:" not in context:
            return ("Thought: I need the rotation size first.\n"
                    "TOOL: search_docs\nARGS: {\"query\": \"oncall rotation\"}")
        if "TOOL: send_email" not in context:
            return ("Thought: Now send it.\nTOOL: send_email\n"
                    "ARGS: {\"to\": \"team-lead@example.com\", \"body\": \"The oncall rotation has 4 engineers.\"}")
        if "declined" in context:
            return ("Thought: A human said no. I must not retry.\n"
                    "ANSWER: I did not send the email because the approval was declined. The rotation has 4 engineers.")
        return "Thought: Done.\nANSWER: Email sent. The rotation has 4 engineers."

    if "checkout" in question.lower():                    # a model stuck in a loop
        return ("Thought: Let me look at the logs again.\n"
                "TOOL: read_log\nARGS: {\"service\": \"checkout\"}")

    if question.startswith("What is 23*7?"):              # gets the argument name wrong once
        if "RESULT:" not in context:
            return "Thought: Use the calculator.\nTOOL: calculator\nARGS: {\"expr\": \"23*7\"}"
        if "RESULT: 161" not in context:
            return "Thought: The argument is called expression.\nTOOL: calculator\nARGS: {\"expression\": \"23*7\"}"
        return "Thought: I have it.\nANSWER: 161"

    return agent.scripted_model(context)


# ----------------------------------------------------------------------
# 4. THE HARNESS
# ----------------------------------------------------------------------
class Stop(Exception):
    """Raised inside the metered model to end the run from the outside."""
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason, self.detail = reason, detail


def deny_all(name, args):
    """Fail closed: with no human attached, irreversible tools do not run."""
    return False


def run_budgeted(question, model=scripted_model_v2, *, max_steps=6, max_tokens=None,
                 max_cost=None, max_repeats=3, max_result_tokens=None, validate=True,
                 approve=deny_all, caching=True, prices=EXAMPLE_PRICES,
                 context_budget=3500):
    trace = []
    state = {"step": 0, "prev_context": "", "tokens": 0, "cost": 0.0, "calls": [], "offloaded": {}}

    # ---- the metered model: count, enforce budgets, then call the real one ----
    def metered_model(context):
        state["step"] += 1
        input_tokens = estimate_tokens(context)
        cached = shared_prefix(state["prev_context"], context) // 4 if caching else 0
        floor_cost = call_cost(input_tokens, cached, 0, prices, caching)
        if max_tokens is not None and state["tokens"] + input_tokens > max_tokens:
            raise Stop("max_tokens", f"step {state['step']} needs {input_tokens} input tokens; "
                                     f"{state['tokens']} of {max_tokens} already used")
        if max_cost is not None and state["cost"] + floor_cost > max_cost:
            raise Stop("max_cost", f"step {state['step']} would cost at least ${floor_cost:.6f}; "
                                   f"${state['cost']:.6f} of ${max_cost:.6f} already spent")

        text = model(context)
        output_tokens = estimate_tokens(text)
        cost = call_cost(input_tokens, cached, output_tokens, prices, caching)
        state["prev_context"] = context
        state["tokens"] += input_tokens + output_tokens
        state["cost"] += cost
        trace.append({"span": len(trace) + 1, "step": state["step"], "kind": "model", "name": "model",
                      "input_tokens": input_tokens, "cached_tokens": cached, "output_tokens": output_tokens,
                      "cost": cost, "cost_without_cache": call_cost(input_tokens, 0, output_tokens, prices, False),
                      "latency_ms": model_latency_ms(input_tokens - cached, cached, output_tokens)})

        # loop detection: the same tool with the same arguments, again and again
        kind, payload = agent.parse_action(text)
        if kind == "tool":
            call = json.dumps(payload, sort_keys=True)
            state["calls"].append(call)
            if max_repeats and state["calls"][-max_repeats:] == [call] * max_repeats:
                raise Stop("loop_detected", f"{payload[0]} called {max_repeats} times in a row with identical arguments")
        return text

    # ---- guarded tools: validate, ask for approval, run, trim, record ----
    def guard(name, fn):
        def guarded(**args):
            status = "ok"
            problem = validate_args(name, args) if validate else None
            if problem:
                status, result = "invalid_args", problem
            elif name in IRREVERSIBLE and not approve(name, args):
                status = "denied"
                result = (f"ERROR: a human declined to approve {name}. "
                          "Do not retry. Tell the user what you would have done.")
            else:
                result = fn(**args)
            full_tokens = estimate_tokens(result)
            if max_result_tokens is not None and full_tokens > max_result_tokens:
                ref = f"result_{len(trace) + 1}.txt"
                state["offloaded"][ref] = result          # a real harness writes a file
                result = (result[:max_result_tokens * 4] +
                          f"\n[truncated: first {max_result_tokens} of {full_tokens} tokens. "
                          f"Full result saved as {ref}. Ask for a narrower slice.]")
            trace.append({"span": len(trace) + 1, "step": state["step"], "kind": "tool", "name": name,
                          "args": args, "status": status, "result_tokens": estimate_tokens(result),
                          "full_result_tokens": full_tokens, "cost": 0.0,
                          "latency_ms": 0 if status != "ok" else TOOL_LATENCY_MS.get(name, 100)})
            return result
        return guarded

    with patched_registry(guard):
        try:
            answer = agent.run_agent(question, model=metered_model, max_steps=max_steps,
                                     context_budget=context_budget, verbose=False)
            stopped, detail = "answer", ""
            if answer.startswith("(step budget exhausted"):
                stopped, detail = "max_steps", f"no final answer after {max_steps} model calls"
        except Stop as s:
            answer, stopped, detail = None, s.reason, s.detail

    models = [s for s in trace if s["kind"] == "model"]
    totals = {
        "model_calls": len(models),
        "tool_calls": len(trace) - len(models),
        "input_tokens": sum(s["input_tokens"] for s in models),
        "cached_tokens": sum(s["cached_tokens"] for s in models),
        "output_tokens": sum(s["output_tokens"] for s in models),
        "cost": sum(s["cost"] for s in models),
        "cost_without_cache": sum(s["cost_without_cache"] for s in models),
        "latency_ms": sum(s["latency_ms"] for s in trace),
    }
    return {"question": question, "answer": answer, "stopped": stopped, "detail": detail,
            "trace": trace, "totals": totals, "offloaded": state["offloaded"]}


@contextlib.contextmanager
def patched_registry(guard):
    """Swap guarded tools (and the two extra tools) into mini_agent for one run.
    The system prompt is built from the registry, so the new tools are appended
    to it in the same "- name: description" format. Everything is restored after."""
    saved_tools, saved_prompt = dict(agent.TOOLS), agent.SYSTEM_PROMPT
    try:
        for name, tool in {**saved_tools, **EXTRA_TOOLS}.items():
            agent.TOOLS[name] = {"fn": guard(name, tool["fn"]), "desc": tool["desc"]}
        agent.SYSTEM_PROMPT = saved_prompt + "".join(f"\n- {n}: {t['desc']}" for n, t in EXTRA_TOOLS.items())
        yield
    finally:
        agent.TOOLS.clear()
        agent.TOOLS.update(saved_tools)
        agent.SYSTEM_PROMPT = saved_prompt


# ----------------------------------------------------------------------
# 5. READING A TRACE
# ----------------------------------------------------------------------
def print_trace(run):
    print(f"{'span':>4} {'step':>4} {'kind':5} {'name':12} {'in':>6} {'cached':>6} {'out':>5} {'result':>6} {'ms':>6} {'cost $':>10}  note")
    for s in run["trace"]:
        if s["kind"] == "model":
            print(f"{s['span']:4d} {s['step']:4d} {'model':5} {'model':12} {s['input_tokens']:6d} {s['cached_tokens']:6d} "
                  f"{s['output_tokens']:5d} {'':>6} {s['latency_ms']:6d} {s['cost']:10.6f}")
        else:
            note = "" if s["status"] == "ok" else s["status"]
            print(f"{s['span']:4d} {s['step']:4d} {'tool':5} {s['name']:12} {'':>6} {'':>6} {'':>5} "
                  f"{s['result_tokens']:6d} {s['latency_ms']:6d} {'':>10}  {note}")
    t = run["totals"]
    print(f"     stopped: {run['stopped']}" + (f" ({run['detail']})" if run["detail"] else ""))
    print(f"     answer : {run['answer']}")
    print(f"     totals : {t['model_calls']} model calls, {t['tool_calls']} tool calls, "
          f"{t['input_tokens']} input tokens ({t['cached_tokens']} cached), {t['output_tokens']} output tokens")
    print(f"     cost   : ${t['cost']:.6f} with caching, ${t['cost_without_cache']:.6f} without, "
          f"{t['latency_ms']} ms (placeholder latencies)")


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


ONCALL_Q = "What is 23*7 plus the number of engineers on the oncall rotation?"
LOOP_Q = "Why is checkout slow?"
SLOPPY_Q = "What is 23*7?"
EMAIL_Q = "Email the size of the oncall rotation to the team lead."

if __name__ == "__main__":
    first = run_budgeted(ONCALL_Q)
    if "--json" in sys.argv:
        print(json.dumps(first, indent=2))
        sys.exit(0)

    banner("1. THE 3-STEP RUN FROM MODULE 14, METERED")
    print_trace(first)

    banner("2. CONTEXT GROWTH: a model that keeps reading logs, 8 steps, no guards")
    free = dict(max_steps=8, max_repeats=0, context_budget=10 ** 9)
    naive = run_budgeted(LOOP_Q, **free)
    ins = [s["input_tokens"] for s in naive["trace"] if s["kind"] == "model"]
    print("input tokens per model call :", ins)
    print("growth per step             :", [b - a for a, b in zip(ins, ins[1:])])
    print(f"total input tokens billed   : {sum(ins)}   (last call alone: {ins[-1]})")
    n, first_in, grow = len(ins), ins[0], ins[1] - ins[0]
    print(f"closed form n*first + grow*n(n-1)/2 = {n}*{first_in} + {grow}*{n * (n - 1) // 2} = {n * first_in + grow * n * (n - 1) // 2}"
          "   (a few tokens off: every estimate is rounded up)")
    trimmed = run_budgeted(LOOP_Q, max_result_tokens=100, **free)
    compacted = run_budgeted(LOOP_Q, max_steps=8, max_repeats=0, context_budget=3500)
    print(f"{'strategy':44s} {'input tok':>10} {'cached':>8} {'cost cached':>12} {'cost uncached':>14}")
    for label, r in [("naive: resend everything", naive),
                     ("truncate tool results to 100 tokens", trimmed),
                     ("mini_agent compaction at 3,500 chars", compacted)]:
        t = r["totals"]
        print(f"{label:44s} {t['input_tokens']:10d} {t['cached_tokens']:8d} {t['cost']:12.6f} {t['cost_without_cache']:14.6f}")
    print("Caching changes the BILL. It does not shrink the window: look at 'input tok'.")
    print("Compaction rewrites history, so the cached prefix after it is gone: look at 'cached'.")

    banner("3. HARD BUDGETS: the same stuck model, four different brakes")
    for label, kw in [("loop detection (3 identical calls)", dict(max_steps=8)),
                      ("max_tokens = 5000", dict(max_steps=8, max_repeats=0, max_tokens=5000, context_budget=10 ** 9)),
                      ("max_cost = $0.01", dict(max_steps=8, max_repeats=0, max_cost=0.01, context_budget=10 ** 9)),
                      ("max_steps = 4", dict(max_steps=4, max_repeats=0))]:
        r = run_budgeted(LOOP_Q, **kw)
        print(f"{label:36s} -> stopped: {r['stopped']:13s} after {r['totals']['model_calls']} model calls. {r['detail']}")

    banner("4. VALIDATION: the model gets an argument name wrong")
    try:
        run_budgeted(SLOPPY_Q, validate=False)
    except TypeError as e:
        print(f"without validation the loop CRASHES: TypeError: {e}\n")
    print_trace(run_budgeted(SLOPPY_Q))

    banner("5. APPROVAL GATE: send_email is marked irreversible")
    print("-- nobody approves (the default: fail closed)")
    print_trace(run_budgeted(EMAIL_Q))
    print("\n-- a human approves")
    print_trace(run_budgeted(EMAIL_Q, approve=lambda name, args: True))
    print(f"outbox: {OUTBOX}")
