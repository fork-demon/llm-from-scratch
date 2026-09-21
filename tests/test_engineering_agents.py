"""Part 10: budgets, caching, validation, approval and traces around mini_agent."""
import json
import math

import agent_budget as ab
import mini_agent as agent


def test_token_estimate_is_chars_over_four():
    assert ab.estimate_tokens("") == 0
    assert ab.estimate_tokens("abcd") == 1
    assert ab.estimate_tokens("abcde") == 2


def test_wrapping_leaves_mini_agent_untouched():
    before_tools, before_prompt = dict(agent.TOOLS), agent.SYSTEM_PROMPT
    ab.run_budgeted(ab.EMAIL_Q)
    assert agent.TOOLS == before_tools and agent.SYSTEM_PROMPT == before_prompt
    assert agent.run_agent(ab.ONCALL_Q, verbose=False).startswith("165")


def test_metered_run_matches_the_numbers_in_the_lesson():
    run = ab.run_budgeted(ab.ONCALL_Q)
    assert run["stopped"] == "answer" and run["answer"].startswith("165")
    models = [s for s in run["trace"] if s["kind"] == "model"]
    assert [s["input_tokens"] for s in models] == [137, 184, 215]
    assert [s["cached_tokens"] for s in models] == [0, 136, 184]
    assert [s["output_tokens"] for s in models] == [28, 28, 24]
    assert [s["name"] for s in run["trace"] if s["kind"] == "tool"] == ["search_docs", "calculator"]
    t = run["totals"]
    assert (t["input_tokens"], t["cached_tokens"], t["output_tokens"]) == (536, 320, 80)
    assert math.isclose(t["cost_without_cache"], (536 * 3 + 80 * 15) / 1e6)
    assert math.isclose(t["cost"], ((320 * 0.10 + 216 * 1.25) * 3 + 80 * 15) / 1e6)
    json.dumps(run)                                   # the trace is plain data


def test_caching_changes_the_bill_not_the_window():
    on = ab.run_budgeted(ab.ONCALL_Q, caching=True)["totals"]
    off = ab.run_budgeted(ab.ONCALL_Q, caching=False)["totals"]
    assert on["input_tokens"] == off["input_tokens"]
    assert on["cost"] < off["cost"] and off["cached_tokens"] == 0
    assert math.isclose(off["cost"], off["cost_without_cache"])


def test_context_grows_linearly_so_the_bill_grows_quadratically():
    run = ab.run_budgeted(ab.LOOP_Q, max_steps=8, max_repeats=0, context_budget=10 ** 9)
    ins = [s["input_tokens"] for s in run["trace"] if s["kind"] == "model"]
    assert len(ins) == 8 and run["stopped"] == "max_steps"
    steps = [b - a for a, b in zip(ins, ins[1:])]
    assert max(steps) - min(steps) <= 1               # constant growth per step (up to rounding)
    closed = 8 * ins[0] + steps[0] * 8 * 7 // 2
    assert abs(sum(ins) - closed) <= 8
    assert sum(ins) > 4 * ins[-1]                     # the total dwarfs any single call


def test_truncation_and_compaction_reduce_billed_tokens():
    free = dict(max_steps=8, max_repeats=0)
    naive = ab.run_budgeted(ab.LOOP_Q, context_budget=10 ** 9, **free)
    trimmed = ab.run_budgeted(ab.LOOP_Q, context_budget=10 ** 9, max_result_tokens=100, **free)
    compacted = ab.run_budgeted(ab.LOOP_Q, context_budget=3500, **free)
    assert trimmed["totals"]["input_tokens"] < compacted["totals"]["input_tokens"] < naive["totals"]["input_tokens"]
    assert trimmed["offloaded"]                       # the full result is kept outside the context
    tool = [s for s in trimmed["trace"] if s["kind"] == "tool"][0]
    assert tool["result_tokens"] < tool["full_result_tokens"]
    # compaction rewrites history, so less of the prompt is a cache hit
    frac = lambda r: r["totals"]["cached_tokens"] / r["totals"]["input_tokens"]
    assert frac(compacted) < frac(naive)


def test_every_budget_stops_the_run_with_its_own_reason():
    assert ab.run_budgeted(ab.LOOP_Q, max_steps=8)["stopped"] == "loop_detected"
    free = dict(max_steps=8, max_repeats=0, context_budget=10 ** 9)
    r = ab.run_budgeted(ab.LOOP_Q, max_tokens=5000, **free)
    assert r["stopped"] == "max_tokens" and r["totals"]["input_tokens"] + r["totals"]["output_tokens"] <= 5000
    r = ab.run_budgeted(ab.LOOP_Q, max_cost=0.01, **free)
    assert r["stopped"] == "max_cost" and r["totals"]["cost"] <= 0.01
    r = ab.run_budgeted(ab.LOOP_Q, max_steps=4, max_repeats=0)
    assert r["stopped"] == "max_steps" and r["totals"]["model_calls"] == 4


def test_validation_turns_a_crash_into_a_message_the_model_can_use():
    assert ab.validate_args("calculator", {"expression": "1+1"}) is None
    msg = ab.validate_args("calculator", {"expr": "1+1"})
    assert 'missing "expression"' in msg and 'unexpected "expr"' in msg and "Expected" in msg
    assert "must be a str" in ab.validate_args("calculator", {"expression": 7})
    run = ab.run_budgeted(ab.SLOPPY_Q)
    assert [s["status"] for s in run["trace"] if s["kind"] == "tool"] == ["invalid_args", "ok"]
    assert run["answer"] == "161"
    try:
        ab.run_budgeted(ab.SLOPPY_Q, validate=False)
        assert False, "expected the unguarded loop to crash"
    except TypeError:
        pass


def test_approval_gate_fails_closed():
    ab.OUTBOX.clear()
    denied = ab.run_budgeted(ab.EMAIL_Q)
    assert ab.OUTBOX == []
    assert [s["status"] for s in denied["trace"] if s["kind"] == "tool"] == ["ok", "denied"]
    assert "did not send" in denied["answer"]
    asked = []
    ok = ab.run_budgeted(ab.EMAIL_Q, approve=lambda name, args: asked.append(name) or True)
    assert asked == ["send_email"] and len(ab.OUTBOX) == 1
    assert ok["answer"].startswith("Email sent")
