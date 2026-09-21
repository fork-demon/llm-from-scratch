"""
Module 15 -- An eval harness from scratch. No API keys, no frameworks.

  python eval_harness.py            # full report: scores, intervals, A versus B
  python eval_harness.py --items    # also print every item with its verdict

What happens when you run it:
  1. The RAG pipeline from phase4-modern-llms/mini_rag.py is IMPORTED (not
     copied) and wrapped in one function: question -> retrieved chunks + answer.
  2. A GOLDEN SET of 24 questions is pushed through it: direct questions,
     paraphrases, unanswerable questions (the right behaviour is to refuse)
     and adversarial near-misses (one of them with a false premise).
  3. Three SCORERS mark every answer: exact match, substring match, and a
     rubric "judge". A fourth, component-level metric checks retrieval alone.
  4. The marks are AGGREGATED: accuracy per category, a bootstrap 95%
     confidence interval, and agreement of each scorer with human labels.
  5. Two pipeline configurations are COMPARED on the same items with a paired
     test, so you can see "is B really better than A, or is that noise?".

The rubric judge is a deterministic function STANDING IN for an LLM judge, the
same trick mini_rag.py uses for its answerer: every number below is exactly
reproducible, offline. Swapping in a real judge changes one function.
"""
import math
import os
import re
import sys

import numpy as np

# The pipeline under test lives in another phase folder whose name has a hyphen,
# so it is not a package. Same import trick as finetune_tiny_gpt.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "phase4-modern-llms"))
import mini_rag as rag  # noqa: E402


# ----------------------------------------------------------------------
# 1. THE GOLDEN SET -- questions with a known right outcome.
#    expected = the reference answer as a person would WRITE it ("within 5
#               minutes"), or None when the corpus does not contain the answer
#               and the only right behaviour is to refuse.
#    source / gold = which document, and which sentence, holds the answer.
#               This is what lets us score RETRIEVAL separately from the ANSWER.
#    Categories are the point: one overall number hides where a system is weak.
# ----------------------------------------------------------------------
GOLDEN_SET = [
    # --- direct: the question reuses the document's own words
    {"id": "d1", "cat": "direct", "q": "how quickly must I acknowledge pages",
     "expected": "within 5 minutes", "source": "oncall.md", "gold": "Primary oncall must acknowledge pages"},
    {"id": "d2", "cat": "direct", "q": "what is the budget for learning materials",
     "expected": "500 dollars per year", "source": "expenses.md", "gold": "Engineers may expense up to 500 dollars"},
    {"id": "d3", "cat": "direct", "q": "who writes the postmortem after an incident",
     "expected": "the oncall engineer", "source": "oncall.md", "gold": "the oncall engineer writes the postmortem"},
    {"id": "d4", "cat": "direct", "q": "when do deploy freezes apply",
     "expected": "the last week of each quarter", "source": "deploy-policy.md", "gold": "Deploy freezes apply"},
    {"id": "d5", "cat": "direct", "q": "how many approvals does every deploy require",
     "expected": "2 approvals", "source": "deploy-policy.md", "gold": "Every deploy requires two approvals"},
    {"id": "d6", "cat": "direct", "q": "who assigns the onboarding buddy",
     "expected": "the team lead", "source": "onboarding.md", "gold": "The onboarding buddy is assigned"},
    {"id": "d7", "cat": "direct", "q": "when must receipts be submitted",
     "expected": "within 30 days", "source": "expenses.md", "gold": "Receipts must be submitted"},
    {"id": "d8", "cat": "direct", "q": "how are rollbacks triggered",
     "expected": "from the deploy dashboard", "source": "deploy-policy.md", "gold": "Rollbacks are triggered"},
    # --- paraphrase: same meaning, the user's own words (what real traffic looks like)
    {"id": "p1", "cat": "paraphrase", "q": "how long do I have to complete security training",
     "expected": "two weeks", "source": "onboarding.md", "gold": "All new hires must complete security training"},
    {"id": "p2", "cat": "paraphrase", "q": "where can I find the engineering handbook",
     "expected": "in the internal wiki", "source": "onboarding.md", "gold": "The engineering handbook lives"},
    {"id": "p3", "cat": "paraphrase", "q": "are manual deploys allowed",
     "expected": "forbidden except during a declared incident", "source": "deploy-policy.md", "gold": "Manual deploys are forbidden"},
    {"id": "p4", "cat": "paraphrase", "q": "who gets paged when the primary is silent",
     "expected": "the secondary oncall", "source": "oncall.md", "gold": "Secondary oncall is paged"},
    {"id": "p5", "cat": "paraphrase", "q": "how fast do I have to respond when I get paged",
     "expected": "within 5 minutes", "source": "oncall.md", "gold": "Primary oncall must acknowledge pages"},
    {"id": "p6", "cat": "paraphrase", "q": "how much money can I spend on books and courses",
     "expected": "500 dollars per year", "source": "expenses.md", "gold": "Engineers may expense up to 500 dollars"},
    {"id": "p7", "cat": "paraphrase", "q": "can I push a release by hand",
     "expected": "forbidden except during a declared incident", "source": "deploy-policy.md", "gold": "Manual deploys are forbidden"},
    {"id": "p8", "cat": "paraphrase", "q": "what do I need before I get production access",
     "expected": "the incident response course", "source": "onboarding.md", "gold": "Production access requires"},
    # --- unanswerable: nothing in the corpus. Refusing is the ONLY pass.
    {"id": "u1", "cat": "unanswerable", "q": "what is the wifi password", "expected": None, "source": None, "gold": None},
    {"id": "u2", "cat": "unanswerable", "q": "how many vacation days do I get", "expected": None, "source": None, "gold": None},
    {"id": "u3", "cat": "unanswerable", "q": "what is the salary of a staff engineer", "expected": None, "source": None, "gold": None},
    {"id": "u4", "cat": "unanswerable", "q": "what is the parental leave policy", "expected": None, "source": None, "gold": None},
    # --- adversarial near-misses: the words match a real sentence, the MEANING does not.
    #     The corpus says who is paged if the PRIMARY is silent, never the secondary.
    #     It gives no budget for travel and no approval count for rollbacks.
    {"id": "a1", "cat": "adversarial", "q": "who is paged if the secondary does not respond", "expected": None, "source": None, "gold": None},
    {"id": "a2", "cat": "adversarial", "q": "how many approvals does a rollback require", "expected": None, "source": None, "gold": None},
    {"id": "a3", "cat": "adversarial", "q": "what is the budget for conference travel", "expected": None, "source": None, "gold": None},
    #     A false premise. The right answer corrects it; the reference was written as "no, every Monday".
    {"id": "a4", "cat": "adversarial", "q": "does the oncall rotation change every friday",
     "expected": "no, every Monday", "source": "oncall.md", "gold": "The oncall rotation changes"},
]

CATEGORIES = ["direct", "paraphrase", "unanswerable", "adversarial"]

# A person (the author) read the 24 answers of the DEFAULT configuration and
# marked each one right or wrong. These labels are the yardstick a scorer is
# calibrated against: a scorer that disagrees with people is measuring
# something else. They are only valid for the default configuration's outputs.
HUMAN_LABELS_DEFAULT = {
    "d1": True, "d2": True, "d3": True, "d4": True, "d5": True, "d6": True, "d7": True, "d8": True,
    "p1": True, "p2": True, "p3": True, "p4": True, "p5": False, "p6": False, "p7": False, "p8": False,
    "u1": True, "u2": False, "u3": True, "u4": True,
    # a4: the system quotes "The oncall rotation changes every Monday at 10am".
    # A person accepts that as a correct answer to "every friday?". The reference
    # says "no, every Monday", so every automatic scorer below marks it wrong.
    "a1": False, "a2": True, "a3": False, "a4": True,
}


# ----------------------------------------------------------------------
# 2. THE TASK -- the system under test, behind one function.
#    An eval never reaches inside the system: it calls it like a user would
#    and records everything needed to explain a failure later.
# ----------------------------------------------------------------------
DEFAULT_CONFIG = {"sentences_per_chunk": 2, "overlap": 1, "k": 3, "threshold": 0.35}


def build_pipeline(config):
    chunks = []
    for src, text in rag.CORPUS.items():
        chunks.extend(rag.chunk(text, src, config["sentences_per_chunk"], config["overlap"]))
    embed = rag.build_embedder([c["text"] for c in chunks])
    store = rag.Store()
    for c in chunks:
        store.add(embed(c["text"]), c)
    return embed, store


def run_task(item, embed, store, config):
    retrieved = store.search(embed(item["q"]), k=config["k"])
    raw = rag.extractive_answer(item["q"], retrieved, embed, threshold=config["threshold"])
    refused = raw.startswith("Not found")
    # mini_rag returns "<sentence>. [source n: file] (sim 0.68)": split it back up
    m = re.match(r"(.*)\. \[source \d+: (.+?)\] \(sim", raw)
    return {
        "retrieved": retrieved,
        "raw": raw,
        "refused": refused,
        "answer": None if refused else m.group(1),
        "cited": None if refused else m.group(2),
    }


# ----------------------------------------------------------------------
# 3. SCORERS -- each one turns (item, output) into pass/fail plus a reason.
# ----------------------------------------------------------------------
def _plain(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def score_exact(item, out):
    """Exact match: the answer must BE the reference string. Brittle on purpose."""
    if item["expected"] is None:
        return out["refused"], "refused" if out["refused"] else "answered a question that has no answer"
    if out["refused"]:
        return False, "refused an answerable question"
    ok = _plain(out["answer"]) == _plain(item["expected"])
    return ok, "identical to the reference" if ok else "a full sentence is never identical to a short reference"


def score_substring(item, out):
    """Substring match: the reference string must appear inside the answer."""
    if item["expected"] is None:
        return out["refused"], "refused" if out["refused"] else "answered a question that has no answer"
    if out["refused"]:
        return False, "refused an answerable question"
    ok = _plain(item["expected"]) in _plain(out["answer"])
    return ok, "reference found in the answer" if ok else f"the string '{item['expected']}' is not in the answer"


# The rubric judge. With a real system this is an LLM call with a prompt like:
#   "Given the question, the reference answer and the candidate answer, mark:
#    1. behaviour: did it answer when it should, refuse when it should?
#    2. correct:   does it state the same fact as the reference (wording,
#                  number format and articles do not matter)?
#    3. grounded:  does it cite the document that actually holds the fact?"
# Our stand-in implements those three checks with string normalisation, so it is
# deterministic. It has none of an LLM judge's flexibility and none of its biases.
_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                 "seven": "7", "eight": "8", "nine": "9", "ten": "10", "thirty": "30"}
_FILLER = {"the", "a", "an", "in", "of", "to", "is", "are"}


def _facts(text):
    words = [_NUMBER_WORDS.get(w, w) for w in _plain(text).split()]
    return [w for w in words if w not in _FILLER]


def score_rubric(item, out):
    if item["expected"] is None:
        return out["refused"], "behaviour ok: refused" if out["refused"] else "behaviour: answered a question the corpus cannot answer"
    if out["refused"]:
        return False, "behaviour: refused although the corpus holds the answer"
    have = set(_facts(out["answer"]))
    missing = [w for w in _facts(item["expected"]) if w not in have]
    if missing:
        return False, f"correct: the answer does not state {missing}"
    if out["cited"] != item["source"]:
        return False, f"grounded: cites {out['cited']}, the fact lives in {item['source']}"
    return True, "behaviour, correct and grounded all ok"


SCORERS = {"exact": score_exact, "substring": score_substring, "rubric": score_rubric}


def retrieval_hit(item, out):
    """Component metric: is the sentence that holds the answer among the top-k
    retrieved chunks? With one gold sentence per question, hit rate = recall@k.
    Not defined for questions that have no answer in the corpus."""
    if item["gold"] is None:
        return None
    return any(item["gold"] in c["text"] for c, _ in out["retrieved"])


def diagnose(item, out, passed):
    """Locate a failure: which component let the question down?"""
    if passed:
        return "ok"
    if item["expected"] is None:
        return "answerer: should have refused"
    if not retrieval_hit(item, out):
        return "retrieval: the right chunk was not in the top-k"
    if out["refused"]:
        return "answerer: had the right chunk, refused"
    if item["gold"] in out["answer"]:
        # the system did its job; the REFERENCE or the SCORER is what needs fixing
        return "scorer: quoted the right sentence, but it is worded differently from the reference"
    return "answerer: had the right chunk, quoted the wrong sentence"


# ----------------------------------------------------------------------
# 4. AGGREGATE -- run everything, then count.
# ----------------------------------------------------------------------
def evaluate(config=DEFAULT_CONFIG, scorer="rubric", items=GOLDEN_SET):
    embed, store = build_pipeline(config)
    rows = []
    for item in items:
        out = run_task(item, embed, store, config)
        passed, reason = SCORERS[scorer](item, out)
        rows.append({"item": item, "out": out, "passed": passed, "reason": reason,
                     "hit": retrieval_hit(item, out), "diagnosis": diagnose(item, out, passed)})
    return rows


def accuracy(rows):
    return sum(r["passed"] for r in rows) / len(rows)


def by_category(rows):
    table = {}
    for cat in CATEGORIES:
        sub = [r for r in rows if r["item"]["cat"] == cat]
        table[cat] = (sum(r["passed"] for r in sub), len(sub))
    return table


def recall_at_k(rows):
    hits = [r["hit"] for r in rows if r["hit"] is not None]
    return sum(hits), len(hits)


# ----------------------------------------------------------------------
# 5. STATISTICS -- the part most teams skip.
#    24 items is a SAMPLE of all the questions users could ask. Another 24
#    would give another score. The confidence interval says how far off the
#    true accuracy could plausibly be.
# ----------------------------------------------------------------------
def mulberry32(seed):
    """The same tiny seeded generator the course website uses (src/lib/rng.ts),
    so the browser lab and this file print identical bootstrap numbers."""
    state = seed & 0xFFFFFFFF

    def imul(x, y):
        return (x * y) & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt


def normal_interval(p, n, z=1.959964):
    """Textbook interval: p +/- z * sqrt(p(1-p)/n). Fine in the middle, poor near 0 or 1."""
    half = z * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - half), min(1.0, p + half)


def wilson_interval(p, n, z=1.959964):
    """Wilson score interval: behaves sensibly for small n and for p near 0 or 1."""
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_interval(values, resamples=10000, seed=15):
    """Percentile bootstrap for a mean. Resample the items WITH replacement,
    recompute the mean, repeat; the middle 95% of those means is the interval.
    No formula needed, so it also works for scores that are not pass/fail."""
    values = list(values)
    n, rnd = len(values), mulberry32(seed)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[int(rnd() * n)]
        means.append(total / n)
    means.sort()
    return means[int(0.025 * resamples)], means[min(resamples - 1, int(0.975 * resamples))]


def sign_test(b, c):
    """Exact two-sided sign test (the exact form of McNemar's test).
    b = items only A passes, c = items only B passes. Items where A and B agree
    carry no information about which is better, so they drop out. Under "no real
    difference" each disagreement is a fair coin: how surprising is this split?"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(rows_a, rows_b, resamples=10000, seed=15):
    """Paired comparison: same items through both systems, look at per-item DIFFERENCES."""
    a = [int(r["passed"]) for r in rows_a]
    b = [int(r["passed"]) for r in rows_b]
    diffs = [y - x for x, y in zip(a, b)]
    only_a = sum(1 for d in diffs if d < 0)
    only_b = sum(1 for d in diffs if d > 0)
    p = sign_test(only_a, only_b)
    lo, hi = bootstrap_interval(diffs, resamples, seed)
    if p < 0.05:
        verdict = "B is better" if only_b > only_a else "A is better"
    else:
        verdict = "cannot tell: the difference is within noise"
    return {"acc_a": sum(a) / len(a), "acc_b": sum(b) / len(b), "diff": sum(diffs) / len(diffs),
            "only_a": only_a, "only_b": only_b, "p": p, "ci": (lo, hi), "verdict": verdict}


def cohens_kappa(x, y):
    """Agreement between two raters beyond what chance alone would give.
    kappa = (p_observed - p_chance) / (1 - p_chance). 1 = perfect, 0 = chance level."""
    n = len(x)
    p_obs = sum(1 for i in range(n) if x[i] == y[i]) / n
    px, py = sum(x) / n, sum(y) / n
    p_chance = px * py + (1 - px) * (1 - py)
    return 1.0 if p_chance == 1 else (p_obs - p_chance) / (1 - p_chance)


# ----------------------------------------------------------------------
# 6. pass@k -- the metric used by code benchmarks such as HumanEval.
#    Generate n samples per problem, c of them pass the unit tests.
#    pass@k = probability that at least one of k samples would pass
#           = 1 - C(n-c, k) / C(n, k)        (Chen et al., 2021, the Codex paper)
#    Computed as a running product so large n never overflows.
# ----------------------------------------------------------------------
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


# ----------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------
def pct(x):
    return f"{100 * x:5.1f}%"


def main(show_items=False):
    n = len(GOLDEN_SET)
    print("=" * 72)
    print(f"EVAL: mini_rag.py on a golden set of {n} questions, config {DEFAULT_CONFIG}")
    print("=" * 72)

    print("\n--- one system, three scorers " + "-" * 42)
    human = [HUMAN_LABELS_DEFAULT[i["id"]] for i in GOLDEN_SET]
    print(f"{'scorer':10s} {'passed':>7s} {'accuracy':>9s}   {'agrees with human':>18s} {'kappa':>6s}")
    for name in SCORERS:
        rows = evaluate(scorer=name)
        marks = [r["passed"] for r in rows]
        agree = sum(1 for m, h in zip(marks, human) if m == h)
        print(f"{name:10s} {sum(marks):4d}/{n} {pct(accuracy(rows)):>9s}   {agree:11d}/{n}      {cohens_kappa(marks, human):6.2f}")
    print(f"{'human':10s} {sum(human):4d}/{n} {pct(sum(human) / n):>9s}")

    rows = evaluate(scorer="rubric")
    print("\n--- rubric scorer, by category " + "-" * 41)
    for cat, (ok, total) in by_category(rows).items():
        print(f"  {cat:13s} {ok}/{total}")
    hits, answerable = recall_at_k(rows)
    print(f"  retrieval recall@{DEFAULT_CONFIG['k']} (component metric, answerable items only): {hits}/{answerable} = {pct(hits / answerable)}")

    print("\n--- where the failures are " + "-" * 45)
    for r in rows:
        if show_items or not r["passed"]:
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"  {mark} {r['item']['id']:3s} {r['item']['q']}")
            print(f"        got: {r['out']['raw'][:88]}")
            if not r["passed"]:
                print(f"        why: {r['reason']}  ->  {r['diagnosis']}")

    acc = accuracy(rows)
    print("\n--- how sure are we? 95% intervals for the accuracy " + "-" * 20)
    print(f"  accuracy            {pct(acc)}  ({sum(r['passed'] for r in rows)}/{n})")
    lo, hi = normal_interval(acc, n)
    print(f"  normal formula      {pct(lo)} to {pct(hi)}   (standard error {100 * math.sqrt(acc * (1 - acc) / n):.1f} points)")
    lo, hi = wilson_interval(acc, n)
    print(f"  Wilson interval     {pct(lo)} to {pct(hi)}")
    lo, hi = bootstrap_interval([int(r["passed"]) for r in rows])
    print(f"  bootstrap (10,000)  {pct(lo)} to {pct(hi)}")

    print("\n--- regression check: is B better than A, or is it noise? " + "-" * 13)
    experiments = [
        ("one sentence per chunk, no overlap", {**DEFAULT_CONFIG, "sentences_per_chunk": 1, "overlap": 0}),
        ("k = 1 instead of 3", {**DEFAULT_CONFIG, "k": 1}),
        ("refusal threshold 0.60", {**DEFAULT_CONFIG, "threshold": 0.60}),
        ("refusal threshold 0.90 (refuses almost everything)", {**DEFAULT_CONFIG, "threshold": 0.90}),
    ]
    for label, cfg in experiments:
        res = compare(rows, evaluate(cfg, "rubric"))
        print(f"  A = default, B = {label}")
        print(f"    A {pct(res['acc_a'])}  B {pct(res['acc_b'])}  difference {100 * res['diff']:+.1f} points,"
              f" 95% interval {100 * res['ci'][0]:+.1f} to {100 * res['ci'][1]:+.1f}")
        print(f"    only A passes: {res['only_a']}   only B passes: {res['only_b']}   sign test p = {res['p']:.3f}")
        print(f"    verdict: {res['verdict']}")

    print("\n--- pass@k for code benchmarks (n samples per problem, c correct) " + "-" * 6)
    for n_s, c, k in [(10, 3, 1), (10, 3, 5), (200, 20, 1), (200, 20, 10), (200, 20, 100)]:
        print(f"  n={n_s:3d} c={c:3d} k={k:3d}  ->  pass@k = {pass_at_k(n_s, c, k):.4f}")


if __name__ == "__main__":
    main(show_items="--items" in sys.argv)
