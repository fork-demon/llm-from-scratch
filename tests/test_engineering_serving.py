"""Part 10: serving. The batching simulator, weight quantization and speculative decoding."""
import numpy as np
import pytest

import batching_sim as bs
import quantize_demo as qd
import speculative_demo as sd


# ---------------------------------------------------------------- batching_sim
def test_rng_is_the_same_mulberry32_as_the_website():
    # values printed by the JavaScript in course/src/lib/rng.ts: makeRng(17).next(), three times
    r = bs.Rng(17)
    got = [r.next() for _ in range(3)]
    assert got == pytest.approx([0.6771502960473299, 0.19265692122280598, 0.5313839064911008], abs=1e-15)


def test_decode_step_barely_grows_with_batch_then_compute_takes_over():
    one = bs.decode_step_ms(1, 300)
    many = bs.decode_step_ms(32, 32 * 300)
    assert many < 1.3 * one                      # 32x the tokens for < 30% more time
    # with no KV reads at all, arithmetic overtakes the weight read near 7 / 0.09 = 78 sequences
    assert bs.decode_step_ms(70, 0) == bs.decode_step_ms(1, 0)
    assert bs.decode_step_ms(200, 0) == pytest.approx(3.0 + 0.09 * 200)


def test_prefill_is_compute_bound_and_never_cheaper_than_one_weight_read():
    assert bs.prefill_ms(10) == pytest.approx(3.0 + 7.0)
    assert bs.prefill_ms(1000) == pytest.approx(3.0 + 90.0)


def test_workload_is_deterministic_and_skewed():
    a, b = bs.make_workload(), bs.make_workload()
    assert a == b
    outs = sorted(r["out"] for r in a)
    assert outs[len(outs) // 2] < 0.5 * outs[int(len(outs) * 0.9)]      # median far below p90
    assert all(8 <= r["prompt"] <= bs.MAX_PROMPT and 1 <= r["out"] <= bs.MAX_NEW for r in a)


def test_every_token_is_generated_under_every_policy():
    wl = bs.make_workload(n=60)
    for policy in ("none", "static", "continuous"):
        m = bs.simulate(wl, bs.default_config(policy=policy))
        assert m["tokens"] == sum(r["out"] for r in wl)
        assert m["useful_steps"] == m["tokens"]           # one useful slot-step per output token
        assert m["ttft_p50"] > 0 and m["tpot_p50"] >= 10.0


def test_continuous_beats_static_beats_none_under_load():
    wl = bs.make_workload()
    none, static, cont = (bs.simulate(wl, bs.default_config(policy=p)) for p in ("none", "static", "continuous"))
    assert none["throughput_tok_s"] < static["throughput_tok_s"] < cont["throughput_tok_s"]
    assert cont["ttft_p99"] < static["ttft_p99"] < none["ttft_p99"]
    assert static["padded_steps"] > 0 and cont["padded_steps"] == 0 and none["padded_steps"] == 0
    assert cont["slo_fraction"] == 1.0 and static["slo_fraction"] < 0.1


def test_the_numbers_quoted_in_the_lesson():
    wl = bs.make_workload()
    static = bs.simulate(wl, bs.default_config(policy="static"))
    cont = bs.simulate(wl, bs.default_config(policy="continuous"))
    assert round(static["throughput_tok_s"]) == 280 and round(cont["throughput_tok_s"]) == 612
    assert round(cont["ttft_p50"]) == 31 and round(static["ttft_p50"]) == 14675
    assert static["padded_steps"] == 46247


def test_bigger_batches_trade_latency_for_throughput():
    wl = bs.make_workload(n=400, rate_per_s=40.0)
    runs = [bs.simulate(wl, bs.default_config(max_batch=b)) for b in (1, 8, 64)]
    assert runs[0]["throughput_tok_s"] < runs[1]["throughput_tok_s"] < runs[2]["throughput_tok_s"]
    assert runs[0]["tpot_p50"] < runs[1]["tpot_p50"] < runs[2]["tpot_p50"]


def test_paged_kv_fits_more_sequences_than_reservation():
    wl = bs.make_workload(n=400, rate_per_s=40.0)
    res, paged = (bs.simulate(wl, bs.default_config(max_batch=64, kv_budget_tokens=16384, kv_mode=m))
                  for m in ("reserved", "paged"))
    assert paged["mean_running"] > 1.8 * res["mean_running"]
    assert paged["throughput_tok_s"] > res["throughput_tok_s"]
    assert res["kv_waste"] > 0.5 and paged["kv_waste"] < 0.05


def test_preemption_never_loses_tokens():
    wl = bs.make_workload(n=120, rate_per_s=40.0)
    m = bs.simulate(wl, bs.default_config(max_batch=64, kv_budget_tokens=2048, kv_mode="paged"))
    assert m["preemptions"] > 0
    assert m["tokens"] == sum(r["out"] for r in wl)


def test_timeline_segments_never_overlap_within_a_slot():
    m = bs.simulate(bs.make_workload(n=80), bs.default_config())
    by_slot = {}
    for s in m["timeline"]:
        by_slot.setdefault(s["slot"], []).append(s)
    for segs in by_slot.values():
        segs.sort(key=lambda s: s["t0"])
        assert all(a["t1"] <= b["t0"] + 1e-9 for a, b in zip(segs, segs[1:]))


def test_kv_grid_paging_wastes_less_and_runs_more():
    reqs = bs.grid_requests()
    cont, paged = bs.simulate_kv_grid(reqs, mode="contiguous"), bs.simulate_kv_grid(reqs, mode="paged")
    assert paged["waste"] < 0.15 < 0.4 < cont["waste"]
    assert paged["steps"] < cont["steps"] and paged["mean_running"] > cont["mean_running"]
    for g in (cont, paged):                       # a block never has two owners; holdings match the map
        for sn in g["snaps"]:
            for sid, blocks in sn["block_lists"].items():
                assert all(sn["owner"][b] == sid for b in blocks)
                assert len(blocks) * 16 >= sn["tokens"][sid]
    # contiguous really is contiguous
    for sn in cont["snaps"]:
        for blocks in sn["block_lists"].values():
            assert blocks == list(range(blocks[0], blocks[0] + len(blocks)))


def test_percentile_matches_numpy():
    v = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    for p in (0, 50, 90, 99, 100):
        assert bs.percentile(v, p) == pytest.approx(np.percentile(v, p))


# ---------------------------------------------------------------- quantize_demo
def _matrices():
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.02, size=(64, 256)), rng.normal(0, 1, size=(32, 256))


def test_hand_example():
    w = np.array([0.021, -0.013, 0.004, 0.035, -0.028, 0.009, -0.002, 0.017])
    q, s = qd.quantize_block(w, 4)
    assert s == pytest.approx(0.005)
    assert q.tolist() == [4, -3, 1, 7, -6, 2, 0, 3]
    assert np.abs(w - q * s).max() <= s / 2 + 1e-12


def test_int8_is_about_one_percent_and_int4_needs_groups():
    W, X = _matrices()
    rows = {(b, g, n): (we, oe) for b, g, n, we, oe, _ in qd.report(W, X, qd.CONFIGS)}
    assert rows[(8, "row", 0)][1] < 0.015
    assert rows[(4, "tensor", 0)][1] > rows[(4, "row", 0)][1] > rows[(4, "group", 64)][1] > rows[(4, "group", 32)][1]


def test_weight_outliers_break_per_tensor_but_not_groups():
    W, X = _matrices()
    Wo = W.copy()
    Wo[3, 7], Wo[40, 100] = 1.0, -1.0
    t4 = qd.rel_err(X @ Wo.T, X @ qd.fake_quantize(Wo, 4, "tensor")[0].T)
    g4 = qd.rel_err(X @ Wo.T, X @ qd.fake_quantize(Wo, 4, "group", 32)[0].T)
    t8 = qd.rel_err(X @ Wo.T, X @ qd.fake_quantize(Wo, 8, "tensor")[0].T)
    r8 = qd.rel_err(X @ Wo.T, X @ qd.fake_quantize(Wo, 8, "row")[0].T)
    assert t4 > 0.8 and g4 < 0.15
    assert t8 > 3 * r8


def test_quantized_values_use_only_the_allowed_levels():
    W, _ = _matrices()
    G = W.reshape(64, 8, 32)
    scale = np.abs(G).max(axis=2, keepdims=True) / 7
    W_hat, n = qd.fake_quantize(W, 4, "group", 32)
    levels = np.round(W_hat.reshape(64, 8, 32) / scale)
    assert n == 64 * 8
    assert np.allclose(levels * scale, W_hat.reshape(64, 8, 32)) and np.abs(levels).max() <= 7


def test_awq_style_scaling_is_exact_before_rounding_and_helps_after():
    W, X = _matrices()
    Xo = X.copy()
    hot = [5, 77, 130]
    Xo[:, hot] *= 30
    s = np.ones(256)
    s[hot] = 4.0
    assert np.allclose((Xo / s) @ (W * s).T, Xo @ W.T)
    plain = qd.rel_err(Xo @ W.T, Xo @ qd.fake_quantize(W, 4, "group", 32)[0].T)
    scaled = qd.rel_err(Xo @ W.T, (Xo / s) @ qd.fake_quantize(W * s, 4, "group", 32)[0].T)
    assert scaled < 0.75 * plain


def test_memory_arithmetic():
    assert qd.bits_per_weight(4, 128) == pytest.approx(4.125)
    assert qd.bits_per_weight(4, 32) == pytest.approx(4.5)
    assert qd.model_gb(7e9, 16) == pytest.approx(14.0)
    assert qd.model_gb(7e9, qd.bits_per_weight(4, 128)) == pytest.approx(3.609, abs=1e-3)


# ---------------------------------------------------------------- speculative_demo
def test_one_step_distribution_is_exactly_the_target():
    for eps in (0.1, 0.5, 1.0):
        P, Q = sd.make_models(eps)
        for ctx in range(sd.V):
            assert np.allclose(sd.one_step_output_distribution(P[ctx], Q[ctx]), P[ctx], atol=1e-12)


def test_speculative_samples_follow_the_target_not_the_draft():
    P, Q = sd.make_models(0.6)
    truth = sd.exact_joint(P, 0, 2)
    rng = np.random.default_rng(5)
    n = 40_000
    spec = sd.empirical_joint(lambda: sd.generate_speculative(P, Q, 0, 2, rng)[0], n, n=2)
    plain = sd.empirical_joint(lambda: sd.generate_plain(P, 0, 2, rng), n, n=2)
    draft = sd.empirical_joint(lambda: sd.generate_plain(Q, 0, 2, rng), n, n=2)
    assert sd.tv_distance(truth, spec) < 0.012            # sampling noise only
    assert sd.tv_distance(truth, plain) < 0.012
    assert sd.tv_distance(truth, draft) > 0.2
    # chi-square goodness of fit against the exact distribution (36 cells, 35 degrees of freedom)
    mask = truth > 0
    chi2 = float((n * (spec[mask] - truth[mask]) ** 2 / truth[mask]).sum())
    assert chi2 < 75      # the 99.99th percentile of chi-square(35) is about 72.6


def test_tokens_per_pass_measured_matches_exact():
    for eps in (0.0, 0.4, 1.0):
        P, Q = sd.make_models(eps)
        measured = sd.measured_tokens_per_pass(P, Q, np.random.default_rng(1), rounds=20_000)
        assert measured == pytest.approx(sd.expected_tokens_per_pass(P, Q), abs=0.05)
    P, Q = sd.make_models(0.0)
    assert sd.expected_tokens_per_pass(P, Q) == pytest.approx(sd.GAMMA + 1)


def test_better_draft_means_more_tokens_per_pass():
    e = [sd.expected_tokens_per_pass(*sd.make_models(eps)) for eps in (0.0, 0.2, 0.5, 1.0)]
    assert e[0] > e[1] > e[2] > e[3] > 1.0
