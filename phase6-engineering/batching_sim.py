"""
Module 17 -- Serving many users at once: a simulator of an LLM server.

Nothing here runs a neural network. It simulates the SCHEDULER around one:
who gets a slot on the GPU, when, and what that does to latency and throughput.

What you will see when you run it:

  1. The cost model: how long one decode step takes for 1, 8, 32, 128 sequences.
     (Spoiler: almost the same. That is the whole reason batching exists.)
  2. One stream of 200 requests served three ways:
       none        one request at a time
       static      wait for a full batch, run until the LONGEST sequence ends
       continuous  sequences leave and join at EVERY step (iteration-level
                   scheduling, introduced by Orca, OSDI 2022; used by vLLM)
     with throughput, time to first token (TTFT), time per output token (TPOT),
     slot utilisation and the decode steps wasted on padding.
  3. The throughput versus latency curve as the batch grows.
  4. KV memory as the real capacity limit: reserving max-length contiguous
     memory per request versus PagedAttention-style fixed-size blocks
     allocated on demand (vLLM, SOSP 2023).
  5. A small block-level picture of both allocators, so you can see the waste.

Everything is deterministic: the random numbers come from a tiny generator
(mulberry32) that the course website re-implements bit for bit in TypeScript,
so the interactive lab shows exactly the numbers printed here.
"""
import math

# ----------------------------------------------------------------------
# 0. A portable seeded RNG (the same algorithm as course/src/lib/rng.ts)
# ----------------------------------------------------------------------
M32 = 0xFFFFFFFF


def _imul(a, b):
    """32-bit integer multiply, keeping the low 32 bits (JavaScript's Math.imul)."""
    return (a * b) & M32


class Rng:
    def __init__(self, seed):
        self.a = seed & M32

    def next(self):
        """Uniform float in [0, 1)."""
        self.a = (self.a + 0x6D2B79F5) & M32
        t = self.a
        t = _imul(t ^ (t >> 15), t | 1)
        t = t ^ ((t + _imul(t ^ (t >> 7), t | 61)) & M32)
        return ((t ^ (t >> 14)) & M32) / 4294967296

    def normal(self):
        """Standard normal via Box-Muller."""
        u = max(self.next(), 1e-12)
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * self.next())


# ----------------------------------------------------------------------
# 1. The cost model. Explicit, simple, and the ONLY physics in this file.
# ----------------------------------------------------------------------
# The numbers describe an illustrative setup, not a benchmark of any product:
# a 7B-parameter model in 16-bit (about 14 GB of weights, 0.5 MiB of KV cache
# per token: the Llama-2-7B shape from the inference lesson) on an accelerator
# with about 2 TB/s of memory bandwidth and about 150 TFLOP/s of usable compute.
#
#   weight_read_ms     every decode step must move ALL the weights from GPU
#                      memory to the arithmetic units once: 14 GB / 2 TB/s = 7 ms.
#                      It does not matter how many sequences share that read.
#   kv_read_ms_per_tok each sequence also reads its own KV cache:
#                      0.5 MiB / 2 TB/s = about 0.00026 ms per cached token.
#   compute_ms_per_tok arithmetic for one token through the model is about
#                      2 x parameters = 14 GFLOP; at 150 TFLOP/s that is 0.09 ms.
#   overhead_ms        scheduling, kernel launches, sampling: a fixed cost per step.
#
# A step takes as long as the SLOWER of "move the bytes" and "do the arithmetic"
# (this is the roofline idea), plus the fixed overhead.
COST = {
    "weight_read_ms": 7.0,
    "kv_read_ms_per_tok": 0.00026,
    "compute_ms_per_tok": 0.09,
    "overhead_ms": 3.0,
}


def decode_step_ms(n_seqs, ctx_tokens, cost=COST):
    """One decode step: every sequence in the batch gets exactly one new token."""
    memory = cost["weight_read_ms"] + cost["kv_read_ms_per_tok"] * ctx_tokens
    compute = cost["compute_ms_per_tok"] * n_seqs
    return cost["overhead_ms"] + max(memory, compute)


def prefill_ms(prompt_tokens, cost=COST):
    """Prefill: all prompt tokens go through in parallel, so arithmetic dominates."""
    compute = cost["compute_ms_per_tok"] * prompt_tokens
    return cost["overhead_ms"] + max(cost["weight_read_ms"], compute)


# ----------------------------------------------------------------------
# 2. The workload: Poisson arrivals, skewed lengths
# ----------------------------------------------------------------------
MAX_PROMPT = 1024
MAX_NEW = 512          # the "max_tokens" cap every request carries


def _round(x):
    return int(math.floor(x + 0.5))   # same rounding as JavaScript's Math.round


def make_workload(n=200, rate_per_s=8.0, seed=17, prompt_median=200, prompt_sigma=0.5,
                  out_median=60, out_sigma=1.0):
    """Exponential gaps between arrivals (a Poisson process). Lengths are log-normal:
    most answers are short, a few are very long. out_sigma is the skew knob."""
    rng = Rng(seed)
    t = 0.0
    reqs = []
    for i in range(n):
        t += -math.log(1.0 - rng.next()) / rate_per_s * 1000.0
        prompt = _round(prompt_median * math.exp(prompt_sigma * rng.normal()))
        out = _round(out_median * math.exp(out_sigma * rng.normal()))
        reqs.append({"id": i, "arrival": t,
                     "prompt": min(MAX_PROMPT, max(8, prompt)),
                     "out": min(MAX_NEW, max(1, out))})
    return reqs


# ----------------------------------------------------------------------
# 3. The simulator
# ----------------------------------------------------------------------
def default_config(**kw):
    cfg = {
        "policy": "continuous",    # "none" | "static" | "continuous"
        "max_batch": 16,           # GPU slots: sequences decoded together
        "kv_budget_tokens": 131072,  # KV memory, counted in tokens
        "kv_mode": "paged",        # "paged" | "reserved"
        "block_size": 16,          # tokens per KV block (vLLM's default is 16)
        "static_max_wait_ms": 2000.0,  # static batching gives up waiting after this
        "slo_ttft_ms": 1000.0,     # an EXAMPLE service-level objective
        "slo_tpot_ms": 50.0,
    }
    cfg.update(kw)
    return cfg


def _blocks(tokens, bs):
    return (tokens + bs - 1) // bs


class _Timeline:
    """Coloured segments for a Gantt chart: which request held which slot, when."""

    def __init__(self):
        self.segs = []
        self.last = {}

    def mark(self, slot, req, kind, t0, t1):
        prev = self.last.get(slot)
        if prev is not None and prev["req"] == req and prev["kind"] == kind and abs(prev["t1"] - t0) < 1e-9:
            prev["t1"] = t1
            return
        seg = {"slot": slot, "req": req, "kind": kind, "t0": t0, "t1": t1}
        self.segs.append(seg)
        self.last[slot] = seg


def simulate(workload, cfg, cost=COST):
    """Run the whole workload through one server. Returns metrics and a timeline."""
    reqs = [dict(r, generated=0, first=None, finish=None, blocks=0, slot=-1, preempted=0) for r in workload]
    policy = cfg["policy"]
    bs = cfg["block_size"]
    total_blocks = cfg["kv_budget_tokens"] // bs
    assert total_blocks * bs >= MAX_PROMPT + MAX_NEW, "KV budget smaller than one worst-case request"
    max_batch = 1 if policy == "none" else cfg["max_batch"]
    if policy != "continuous":
        # A static batch is one rectangular tensor: every slot is sized for the worst case.
        max_batch = max(1, min(max_batch, (total_blocks * bs) // (MAX_PROMPT + MAX_NEW)))

    tl = _Timeline()
    st = {"now": 0.0, "busy_ms": 0.0, "useful_slot_ms": 0.0, "decode_steps": 0, "prefill_iters": 0,
          "padded_steps": 0, "useful_steps": 0, "preemptions": 0, "free": total_blocks,
          "kv_stored_ms": 0.0, "kv_held_ms": 0.0, "running_ms": 0.0, "peak_running": 0}
    pending = list(reqs)     # not yet arrived (sorted by arrival)
    waiting = []             # arrived, not running
    running = []             # holding a slot
    done = 0

    def arrive():
        while pending and pending[0]["arrival"] <= st["now"]:
            waiting.append(pending.pop(0))

    def account(dur, useful_slots):
        st["busy_ms"] += dur
        st["useful_slot_ms"] += dur * useful_slots
        stored = sum(r["prompt"] + r["generated"] for r in running)
        st["kv_stored_ms"] += dur * stored
        st["kv_held_ms"] += dur * (total_blocks - st["free"]) * bs
        st["running_ms"] += dur * len(running)
        st["peak_running"] = max(st["peak_running"], len(running))

    # ------------------------------------------------------------ static / none
    if policy != "continuous":
        while done < len(reqs):
            arrive()
            ready = bool(waiting) and (
                len(waiting) >= max_batch or not pending
                or st["now"] - waiting[0]["arrival"] >= cfg["static_max_wait_ms"])
            if not ready:
                nxt = [pending[0]["arrival"]] if pending else []
                if waiting:
                    nxt.append(waiting[0]["arrival"] + cfg["static_max_wait_ms"])
                st["now"] = max(st["now"], min(nxt))
                continue
            batch = [waiting.pop(0) for _ in range(min(max_batch, len(waiting)))]
            n = len(batch)
            pad_prompt = max(r["prompt"] for r in batch)     # prompts are padded to the longest
            steps = max(r["out"] for r in batch)             # and everyone waits for the longest answer
            running[:] = batch
            st["free"] = total_blocks - n * _blocks(MAX_PROMPT + MAX_NEW, bs)
            # prefill the padded rectangle; it produces token 1 for everybody
            dur = prefill_ms(n * pad_prompt, cost)
            t0 = st["now"]
            st["now"] += dur
            st["prefill_iters"] += 1
            for i, r in enumerate(batch):
                r["slot"], r["generated"], r["first"] = i, 1, st["now"]
                tl.mark(i, r["id"], "prefill", t0, st["now"])
                if r["out"] == 1:
                    r["finish"] = st["now"]
            account(dur, n)
            st["useful_steps"] += n
            for k in range(2, steps + 1):
                dur = decode_step_ms(n, n * (pad_prompt + k - 1), cost)
                t0 = st["now"]
                st["now"] += dur
                st["decode_steps"] += 1
                live = 0
                for i, r in enumerate(batch):
                    if r["generated"] < r["out"]:
                        r["generated"] += 1
                        live += 1
                        tl.mark(i, r["id"], "decode", t0, st["now"])
                        if r["generated"] == r["out"]:
                            r["finish"] = st["now"]
                    else:
                        tl.mark(i, r["id"], "pad", t0, st["now"])   # finished, still occupying the slot
                st["padded_steps"] += n - live
                st["useful_steps"] += live
                account(dur, live)
            done += n
            running[:] = []
            st["free"] = total_blocks

    # ------------------------------------------------------------ continuous
    else:
        slots = [None] * max_batch
        while done < len(reqs):
            arrive()
            if not running and not waiting:
                st["now"] = pending[0]["arrival"]
                continue
            # --- admission: first come, first served, while a slot AND KV memory are free
            admitted = []
            while waiting and len(running) + len(admitted) < max_batch:
                r = waiting[0]
                if cfg["kv_mode"] == "reserved":
                    need = _blocks(r["prompt"] + MAX_NEW, bs)    # worst case, up front
                    headroom = 0
                else:
                    need = _blocks(r["prompt"] + r["generated"] + 1, bs)   # what it holds right now
                    headroom = len(running) + len(admitted)     # keep one spare block per running sequence
                if st["free"] - need < headroom:
                    break
                st["free"] -= need
                r["blocks"] = need
                r["slot"] = slots.index(None)
                slots[r["slot"]] = r["id"]
                admitted.append(waiting.pop(0))

            if admitted:
                # --- a prefill iteration. Running sequences STALL while it happens.
                tokens = sum(r["prompt"] + r["generated"] for r in admitted)   # (+generated: recompute after preemption)
                dur = prefill_ms(tokens, cost)
                t0 = st["now"]
                st["now"] += dur
                st["prefill_iters"] += 1
                for r in running:
                    tl.mark(r["slot"], r["id"], "stall", t0, st["now"])
                for r in admitted:
                    tl.mark(r["slot"], r["id"], "prefill", t0, st["now"])
                    r["generated"] += 1
                    if r["first"] is None:
                        r["first"] = st["now"]
                running.extend(admitted)
                st["useful_steps"] += len(admitted)
                account(dur, len(admitted))
            else:
                # --- a decode iteration: one token for every running sequence
                if cfg["kv_mode"] == "paged":
                    i = 0
                    while i < len(running):
                        r = running[i]
                        if _blocks(r["prompt"] + r["generated"] + 1, bs) > r["blocks"]:
                            if st["free"] == 0:
                                # out of KV blocks: preempt the most recently admitted sequence.
                                # Its blocks are freed and it will be re-prefilled later (recomputation).
                                victim = running.pop()
                                st["free"] += victim["blocks"]
                                victim["blocks"] = 0
                                slots[victim["slot"]] = None
                                victim["preempted"] += 1
                                st["preemptions"] += 1
                                waiting.insert(0, victim)
                                if victim is r:
                                    break
                                continue
                            st["free"] -= 1
                            r["blocks"] += 1
                        i += 1
                if not running:
                    continue
                ctx = sum(r["prompt"] + r["generated"] for r in running)
                dur = decode_step_ms(len(running), ctx, cost)
                t0 = st["now"]
                st["now"] += dur
                st["decode_steps"] += 1
                st["useful_steps"] += len(running)
                for r in running:
                    r["generated"] += 1
                    tl.mark(r["slot"], r["id"], "decode", t0, st["now"])
                account(dur, len(running))
            # --- finished sequences leave immediately; their slot and blocks are free next iteration
            for r in [r for r in running if r["generated"] >= r["out"]]:
                r["finish"] = st["now"]
                st["free"] += r["blocks"]
                r["blocks"] = 0
                slots[r["slot"]] = None
                running.remove(r)
                done += 1

    return _metrics(reqs, st, tl, cfg, max_batch)


def percentile(values, p):
    """Linear interpolation between closest ranks (NumPy's default)."""
    v = sorted(values)
    if not v:
        return 0.0
    k = (len(v) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _metrics(reqs, st, tl, cfg, max_batch):
    ttft = [r["first"] - r["arrival"] for r in reqs]                  # queueing + prefill
    e2e = [r["finish"] - r["arrival"] for r in reqs]
    tpot = [(r["finish"] - r["first"]) / (r["out"] - 1) for r in reqs if r["out"] > 1]
    tokens = sum(r["out"] for r in reqs)
    span_s = (max(r["finish"] for r in reqs) - reqs[0]["arrival"]) / 1000.0
    ok = [r for r in reqs
          if r["first"] - r["arrival"] <= cfg["slo_ttft_ms"]
          and (r["out"] == 1 or (r["finish"] - r["first"]) / (r["out"] - 1) <= cfg["slo_tpot_ms"])]
    return {
        "policy": cfg["policy"], "max_batch": max_batch,
        "tokens": tokens, "span_s": span_s,
        "throughput_tok_s": tokens / span_s,
        "ttft_p50": percentile(ttft, 50), "ttft_p99": percentile(ttft, 99),
        "tpot_p50": percentile(tpot, 50), "tpot_p99": percentile(tpot, 99),
        "e2e_p50": percentile(e2e, 50), "e2e_p99": percentile(e2e, 99),
        "slot_util": st["useful_slot_ms"] / (max_batch * st["busy_ms"]),
        "padded_steps": st["padded_steps"], "useful_steps": st["useful_steps"],
        "decode_steps": st["decode_steps"], "prefill_iters": st["prefill_iters"],
        "preemptions": st["preemptions"],
        "slo_fraction": len(ok) / len(reqs), "goodput_req_s": len(ok) / span_s,
        "mean_running": st["running_ms"] / st["busy_ms"], "peak_running": st["peak_running"],
        "kv_waste": 1.0 - st["kv_stored_ms"] / st["kv_held_ms"] if st["kv_held_ms"] else 0.0,
        "timeline": tl.segs,
    }


# ----------------------------------------------------------------------
# 4. KV memory, block by block: contiguous reservation versus paging
# ----------------------------------------------------------------------
def simulate_kv_grid(requests, n_blocks=64, block_size=16, mode="paged", max_new=128, max_steps=5000):
    """A deliberately tiny memory model so every block can be drawn.

    All requests are waiting at step 0. Each step: admit in arrival order while
    memory allows, then every running sequence grows by one token. A sequence
    that reaches its output length releases its memory.

      mode="contiguous"  reserve ceil((prompt + max_new) / block_size) blocks in ONE
                         contiguous run, first fit. The server cannot know the output
                         length in advance, so it reserves the worst case.
      mode="paged"       hold only the blocks the tokens so far need; take ANY free
                         block when the last one fills up; a block table remembers
                         where each logical block lives.

    Returns one snapshot per step: the owner of every block, how many tokens each
    block really stores, and the list of running sequences.
    """
    owner = [-1] * n_blocks
    seqs = [dict(r, gen=0, blocks=[], done=False) for r in requests]
    waiting = list(seqs)
    running = []
    snaps = []
    fragmentation_blocks = 0     # steps where the head of the queue was refused although enough free blocks existed in total
    preemptions = 0

    def free_count():
        return sum(1 for o in owner if o < 0)

    def first_fit(need):
        run = 0
        for i, o in enumerate(owner):
            run = run + 1 if o < 0 else 0
            if run == need:
                return i - need + 1
        return -1

    def release(s):
        for b in s["blocks"]:
            owner[b] = -1
        s["blocks"] = []

    step = 0
    while (waiting or running) and step < max_steps:
        # --- admit
        while waiting:
            s = waiting[0]
            if mode == "contiguous":
                need = _blocks(s["prompt"] + max_new, block_size)
                start = first_fit(need)
                if start < 0:
                    if free_count() >= need:
                        fragmentation_blocks += 1
                    break
                s["blocks"] = list(range(start, start + need))
            else:
                need = _blocks(s["prompt"] + s["gen"] + 1, block_size)
                if free_count() - need < len(running):      # one spare block per running sequence
                    break
                s["blocks"] = [i for i, o in enumerate(owner) if o < 0][:need]
            for b in s["blocks"]:
                owner[b] = s["id"]
            running.append(waiting.pop(0))
        # --- grow by one token
        i = 0
        while i < len(running):
            s = running[i]
            if mode == "paged" and _blocks(s["prompt"] + s["gen"] + 1, block_size) > len(s["blocks"]):
                if free_count() == 0:
                    victim = running.pop()
                    release(victim)
                    preemptions += 1
                    waiting.insert(0, victim)
                    if victim is s:
                        break
                    continue
                b = owner.index(-1)
                owner[b] = s["id"]
                s["blocks"].append(b)
            s["gen"] += 1
            i += 1
        # --- snapshot, then release the finished
        stored = sum(s["prompt"] + s["gen"] for s in running)
        held = sum(len(s["blocks"]) for s in running) * block_size
        snaps.append({
            "step": step, "owner": list(owner), "running": [s["id"] for s in running],
            "tokens": {s["id"]: s["prompt"] + s["gen"] for s in running},
            "block_lists": {s["id"]: list(s["blocks"]) for s in running},
            "stored": stored, "held": held, "waiting": len(waiting),
        })
        for s in [s for s in running if s["gen"] >= s["out"]]:
            s["done"] = True
            release(s)
            running.remove(s)
        step += 1

    busy = [sn for sn in snaps if sn["held"] > 0]
    return {
        "snaps": snaps, "steps": step,
        "mean_running": sum(len(sn["running"]) for sn in snaps) / len(snaps),
        "peak_running": max(len(sn["running"]) for sn in snaps),
        "waste": 1.0 - sum(sn["stored"] for sn in busy) / sum(sn["held"] for sn in busy),
        "fragmentation_refusals": fragmentation_blocks, "preemptions": preemptions,
    }


def grid_requests(n=24, seed=5, prompt_median=40, out_median=30, out_sigma=0.9, max_new=128):
    rng = Rng(seed)
    out = []
    for i in range(n):
        prompt = min(96, max(4, _round(prompt_median * math.exp(0.5 * rng.normal()))))
        o = min(max_new, max(1, _round(out_median * math.exp(out_sigma * rng.normal()))))
        out.append({"id": i, "prompt": prompt, "out": o})
    return out


# ----------------------------------------------------------------------
# 5. The demo
# ----------------------------------------------------------------------
def banner(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def _row(m):
    return (f"  {m['policy']:<11}{m['max_batch']:>5}{m['throughput_tok_s']:>10.0f}"
            f"{m['ttft_p50']:>10.0f}{m['ttft_p99']:>10.0f}{m['tpot_p50']:>9.1f}{m['tpot_p99']:>9.1f}"
            f"{m['slot_util'] * 100:>8.0f}%{m['padded_steps']:>9}{m['slo_fraction'] * 100:>7.0f}%")


HEADER = ("  policy     batch   tok/s  TTFT p50  TTFT p99  TPOT p50 TPOT p99   slots   padded  in SLO\n"
          "                               (ms)      (ms)      (ms)     (ms)    used    steps")


def main():
    banner("1. The cost model: one decode step, for a growing batch (300 cached tokens each)")
    print("  batch   step (ms)   tokens/s   ms per token, per user")
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        ms = decode_step_ms(b, 300 * b)
        print(f"  {b:>5}{ms:>12.2f}{b / ms * 1000:>11.0f}{ms:>12.2f}")
    print("  The weights are read once per step whatever the batch. 32 users cost 25% more\n"
          "  time per step than 1 user, and produce 32 tokens instead of 1.")

    banner("2. One workload, three schedulers (200 requests, 8 arrivals per second)")
    wl = make_workload()
    outs = sorted(r["out"] for r in wl)
    print(f"  prompts: median {percentile([r['prompt'] for r in wl], 50):.0f} tokens.  "
          f"outputs: median {percentile(outs, 50):.0f}, p90 {percentile(outs, 90):.0f}, "
          f"max {outs[-1]} tokens  (skewed on purpose)")
    print(HEADER)
    for policy in ("none", "static", "continuous"):
        print(_row(simulate(wl, default_config(policy=policy))))
    print("  SLO used here (an example): first token within 1000 ms, then at most 50 ms per token.")
    print(f"  The offered load is about {sum(r['out'] for r in wl) / (wl[-1]['arrival'] / 1000):.0f} tokens/s. Only 'continuous' keeps up;\n"
          "  the other two fall behind, so their queues (and TTFT) grow for as long as traffic lasts.")
    print("\n  The same three at a gentle 1 arrival per second:")
    wl1 = make_workload(rate_per_s=1.0)
    print(HEADER)
    for policy in ("none", "static", "continuous"):
        print(_row(simulate(wl1, default_config(policy=policy))))

    banner("3. Throughput versus latency as the batch grows (continuous, 40 arrivals/s: saturated)")
    wl40 = make_workload(n=400, rate_per_s=40.0)
    print("  max batch   tok/s   TPOT p50 (ms)   TTFT p50 (ms)")
    for b in (1, 2, 4, 8, 16, 32, 64, 128):
        m = simulate(wl40, default_config(max_batch=b))
        print(f"  {b:>9}{m['throughput_tok_s']:>8.0f}{m['tpot_p50']:>16.1f}{m['ttft_p50']:>16.0f}")
    print("  Throughput climbs steeply, then flattens as KV reads and arithmetic catch up with\n"
          "  the weight read. Every user's tokens arrive more slowly the whole way up.")

    banner("4. KV memory is the capacity limit: 16,384 tokens of KV budget (8 GiB at 0.5 MiB/token)")
    print("  Same saturated workload as section 3, 64 slots. Now memory, not slots, decides who runs.")
    print("  kv mode    slots  mean running  peak   tok/s  TTFT p50  KV wasted  preemptions")
    for mode in ("reserved", "paged"):
        m = simulate(wl40, default_config(max_batch=64, kv_budget_tokens=16384, kv_mode=mode))
        print(f"  {mode:<10}{64:>6}{m['mean_running']:>14.1f}{m['peak_running']:>6}{m['throughput_tok_s']:>8.0f}"
              f"{m['ttft_p50']:>10.0f}{m['kv_waste'] * 100:>10.0f}%{m['preemptions']:>13}")
    print("  'reserved' sets aside prompt + 512 tokens per request because the output length is\n"
          "  unknown. Most of that is never written. 'paged' holds only the blocks in use.")

    banner("5. The same idea, small enough to draw: 64 blocks of 16 tokens, 24 requests")
    gr = grid_requests()
    for mode in ("contiguous", "paged"):
        g = simulate_kv_grid(gr, mode=mode)
        print(f"  {mode:<11} finished in {g['steps']:>4} steps   mean running {g['mean_running']:>5.1f}   "
              f"peak {g['peak_running']:>2}   memory wasted {g['waste'] * 100:>3.0f}%   "
              f"steps blocked by fragmentation {g['fragmentation_refusals']}   preemptions {g['preemptions']}")
    g = simulate_kv_grid(gr, mode="paged")
    sn = g["snaps"][20]
    sid = sn["running"][0]
    print(f"\n  paged, step 20: sequence {sid} holds {sn['tokens'][sid]} tokens in blocks {sn['block_lists'][sid]}")
    print("  Its block table maps logical block 0, 1, 2... to those physical blocks. They need not be neighbours.")
    row = "".join("." if o < 0 else chr(ord("A") + o % 26) for o in sn["owner"])
    print(f"  memory map (one letter per block, '.' = free):\n  {row}")


if __name__ == "__main__":
    main()
