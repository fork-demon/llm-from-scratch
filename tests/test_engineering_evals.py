"""Part 10: the eval harness (golden set, scorers, intervals, paired comparison)."""
import math

import eval_harness as eh


def test_golden_set_shape():
    assert len(eh.GOLDEN_SET) == 24
    assert len({i["id"] for i in eh.GOLDEN_SET}) == 24
    assert {i["cat"] for i in eh.GOLDEN_SET} == set(eh.CATEGORIES)
    # every gold sentence really exists in the document it names
    for item in eh.GOLDEN_SET:
        if item["gold"] is not None:
            assert item["gold"] in eh.rag.CORPUS[item["source"]]


def test_default_scores_are_the_ones_the_lesson_quotes():
    assert sum(r["passed"] for r in eh.evaluate(scorer="exact")) == 4
    assert sum(r["passed"] for r in eh.evaluate(scorer="substring")) == 12
    rows = eh.evaluate(scorer="rubric")
    assert sum(r["passed"] for r in rows) == 16
    assert eh.by_category(rows) == {"direct": (8, 8), "paraphrase": (4, 8), "unanswerable": (3, 4), "adversarial": (1, 4)}
    assert eh.recall_at_k(rows) == (16, 17)


def test_failures_are_located_in_the_right_component():
    rows = {r["item"]["id"]: r for r in eh.evaluate(scorer="rubric")}
    assert rows["p6"]["diagnosis"].startswith("retrieval")       # right chunk never retrieved
    assert rows["p7"]["diagnosis"] == "answerer: had the right chunk, refused"
    assert rows["u2"]["diagnosis"] == "answerer: should have refused"
    assert rows["a4"]["diagnosis"].startswith("scorer")          # the eval's fault, not the system's
    assert rows["d1"]["diagnosis"] == "ok"


def test_rubric_tolerates_format_but_substring_does_not():
    rows_s = {r["item"]["id"]: r["passed"] for r in eh.evaluate(scorer="substring")}
    rows_r = {r["item"]["id"]: r["passed"] for r in eh.evaluate(scorer="rubric")}
    assert not rows_s["d1"] and rows_r["d1"]      # "5 minutes" vs "five minutes"
    assert not rows_s["p4"] and rows_r["p4"]      # "the secondary oncall" vs "Secondary oncall"


def test_scorer_agreement_with_human_labels():
    human = [eh.HUMAN_LABELS_DEFAULT[i["id"]] for i in eh.GOLDEN_SET]
    kappas = {}
    for name in eh.SCORERS:
        marks = [r["passed"] for r in eh.evaluate(scorer=name)]
        kappas[name] = eh.cohens_kappa(marks, human)
    assert kappas["exact"] < kappas["substring"] < kappas["rubric"] < 1.0
    assert round(kappas["rubric"], 2) == 0.90


def test_cohens_kappa_closed_form():
    # 2x2 table: both yes 20, both no 15, disagree 5 + 10 -> p_o = 0.7, p_e = 0.5 -> kappa 0.4
    x = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    y = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert math.isclose(eh.cohens_kappa(x, y), 0.4)
    assert eh.cohens_kappa(x, x) == 1.0


def test_intervals_against_closed_forms():
    lo, hi = eh.normal_interval(0.8, 50)
    assert math.isclose(hi - 0.8, 1.959964 * math.sqrt(0.8 * 0.2 / 50))
    assert round(100 * (hi - 0.8), 1) == 11.1                     # the "+/- 11 points" in the lesson
    lo, hi = eh.wilson_interval(0.8, 50)
    assert round(lo, 3) == 0.670 and round(hi, 3) == 0.888        # textbook Wilson values
    lo, hi = eh.wilson_interval(1.0, 10)
    assert math.isclose(hi, 1.0) and 0.70 < lo < 0.73                         # never collapses to zero width


def test_bootstrap_matches_the_analytic_interval_on_a_known_case():
    values = [1] * 160 + [0] * 40                                 # p = 0.8, n = 200
    lo, hi = eh.bootstrap_interval(values, resamples=4000, seed=1)
    a_lo, a_hi = eh.normal_interval(0.8, 200)
    assert abs(lo - a_lo) < 0.015 and abs(hi - a_hi) < 0.015
    assert eh.bootstrap_interval(values, 500, seed=3) == eh.bootstrap_interval(values, 500, seed=3)


def test_mulberry32_matches_the_typescript_generator():
    rnd = eh.mulberry32(42)
    first = [rnd() for _ in range(3)]
    assert all(0 <= v < 1 for v in first)
    # reference values produced by course/src/lib/rng.ts makeRng(42)
    assert [round(v, 10) for v in first] == [0.6011037519, 0.4482905590, 0.8524657935]


def test_sign_test_exact_values():
    assert eh.sign_test(0, 0) == 1.0
    assert eh.sign_test(2, 0) == 0.5
    assert math.isclose(eh.sign_test(12, 2), 2 * (1 + 14 + 91) / 2 ** 14)
    assert eh.sign_test(3, 3) == 1.0


def test_paired_comparison_verdicts():
    base = eh.evaluate(scorer="rubric")
    small = eh.compare(base, eh.evaluate({**eh.DEFAULT_CONFIG, "sentences_per_chunk": 1, "overlap": 0}))
    assert (small["only_a"], small["only_b"]) == (2, 0)
    assert small["verdict"].startswith("cannot tell")
    big = eh.compare(base, eh.evaluate({**eh.DEFAULT_CONFIG, "threshold": 0.90}))
    assert (big["only_a"], big["only_b"]) == (12, 2)
    assert big["verdict"] == "A is better" and big["p"] < 0.05
    assert big["ci"][1] < 0
    same = eh.compare(base, base)
    assert same["diff"] == 0 and same["p"] == 1.0


def test_pass_at_k():
    assert math.isclose(eh.pass_at_k(10, 3, 1), 0.3)
    assert math.isclose(eh.pass_at_k(10, 3, 5), 1 - math.comb(7, 5) / math.comb(10, 5))
    assert math.isclose(eh.pass_at_k(200, 20, 10), 1 - math.comb(180, 10) / math.comb(200, 10))
    assert eh.pass_at_k(5, 0, 3) == 0.0
    assert eh.pass_at_k(5, 4, 3) == 1.0          # fewer than k failures: some sample must pass
