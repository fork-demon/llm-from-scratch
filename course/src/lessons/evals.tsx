import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { EvalLab } from '../interactive/EvalLab'
import { SampleSizeLab } from '../interactive/SampleSizeLab'
import { EvalAnatomy } from '../illustrations/EvalAnatomy'

export default function EvalsLesson() {
  return (
    <Lesson id="evals">
      <Why>
        <p className="lede">You have a working RAG bot from <a href="#/lesson/rag">lesson 9.1</a>. A teammate proposes a change: chunk the documents one sentence at a time. Smaller chunks, sharper matches.</p>
        <p>You try the four demo questions. All four still answer correctly. The change looks free. Do you ship it?</p>
        <div className="grid-2">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>What the demo showed</h4>
            <p>4 of 4 questions right, before and after. No visible difference.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>What 24 chosen questions show</h4>
            <p>16 right before, 14 after. Ask for the wifi password, which is in no document, and the new version no longer refuses. It answers, with a citation: “The onboarding buddy is assigned by the team lead.”</p>
          </div>
        </div>
        <p>Nothing crashed. No test went red. The system got quietly worse at the one behaviour that protects you from confident nonsense. And here is the uncomfortable second half: <b>16 versus 14 out of 24 is not enough evidence that it got worse at all.</b> Both halves are this lesson.</p>
        <Callout kind="idea">
          An <b>eval</b> is a repeatable measurement of whether your LLM system does its job: a fixed set of inputs, a rule for marking outputs, and a number that comes with an honest statement of its uncertainty. It is how you replace “it looked fine when I tried it” with evidence.
        </Callout>
      </Why>

      <Problem>
        <WhyExists
          problem="Every part of an LLM system is a knob: the prompt, the model version, the chunk size, k, a threshold. You need to know whether turning one made things better."
          naive="Try a few inputs by hand and look at the answers. Ship if they look good."
          fails="You check the cases you thought of, which are the cases you already built for. Output is free text, so nothing fails loudly. A change that fixes three cases and silently breaks five looks like progress."
          idea="Freeze a set of representative inputs with known right outcomes. Mark every output with a rule, not a mood. Re-run the whole set on every change, and compare with statistics instead of by eye."
          tradeoff="Building and labelling the set is real work, scorers are themselves imperfect, and a small set gives a number that is mostly noise. An eval you trust too much is worse than none."
        />
        <Callout kind="dev">
          Think of evals as the regression test suite for a component with no compiler and no spec. The analogy holds for the workflow: run in CI, block the merge, add a case for every bug. It breaks in three places.
          <br /><br />
          <b>1. No exact assertions.</b> The right answer can be phrased a thousand ways, so <code>assertEquals</code> is the wrong tool and the scorer becomes a design problem of its own. <b>2. The result is a rate, not green or red.</b> Nobody ships at 100%. You ship at 87% and must decide whether 85% next week is a regression or noise. <b>3. The system may be non-deterministic.</b> With sampling <G t="temperature">temperature</G> above 0, the same input can pass now and fail on the next run. A flaky test is a bug in the test. A flaky eval item is a property of the system you are measuring.
        </Callout>
        <Term
          name="Golden set"
          plain={<>A fixed list of inputs for which people have decided what a good outcome is. It is the dataset your system is graded on.</>}
          example={<>Question: “what is the wifi password”. Expected: a refusal, because no document contains it.</>}
          formal={<>A labelled evaluation dataset, versioned alongside the code, that is never used to tune the system it measures.</>}
        />
      </Problem>

      <MentalModel title="The anatomy of an eval">
        <p>Every eval, from a 20-line script to a public benchmark, has the same four parts. Here they are with real rows from this lesson’s harness:</p>
        <EvalAnatomy />
        <p>Each part can be wrong in its own way, so take them one at a time.</p>

        <h3>1. The dataset decides what you can find out</h3>
        <p>An eval only sees failures its questions can trigger. Our 24 items come in four categories, and that is deliberate:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>Category</th><th>Example</th><th>Where such items come from in practice</th></tr></thead>
            <tbody>
              <tr><td><b>direct</b></td><td>“when do deploy freezes apply”</td><td>The happy path. Easy to write, and the least informative.</td></tr>
              <tr><td><b>paraphrase</b></td><td>“can I push a release by hand”</td><td>Real user traffic, sampled from logs. Users never use your documents’ wording.</td></tr>
              <tr><td><b>unanswerable</b></td><td>“what is the parental leave policy”</td><td>Questions outside the system’s knowledge. The expected outcome is a refusal.</td></tr>
              <tr><td><b>adversarial</b></td><td>“who is paged if the <em>secondary</em> does not respond”</td><td>Every bug report and every failure you find. The words match a real sentence, the meaning does not.</td></tr>
            </tbody>
          </table>
        </div>
        <p>Two rules protect the dataset’s value. <b>Keep a held-out part.</b> If you tune the threshold until the score on these 24 items peaks, you have fitted the system to these 24 items, exactly like a model that memorises its training set. It is the <G t="validation-loss">validation loss</G> idea again: keep some items you never look at while tuning, and check them at the end. <b>Watch for leakage.</b> If an eval question (or its answer) also sits in the few-shot examples of your prompt, in your fine-tuning data, or in the model’s pretraining data, a high score measures memory, not ability.</p>

        <h3>2. The task calls the system like a user would</h3>
        <p>The eval does not reach inside. It sends the question in, and records everything that comes back: the answer, the retrieved chunks, the citation. Recording the intermediate results is what makes step 3 useful.</p>

        <h3>3. The scorer is where most evals go wrong</h3>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>Scorer</th><th>How it marks</th><th>How it fails</th></tr></thead>
            <tbody>
              <tr><td><b>Exact or substring match</b></td><td>The reference string must equal, or appear in, the output.</td><td><b>Too strict.</b> “within 5 minutes” is not a substring of “within five minutes”. Correct answers fail on formatting. Good for labels, numbers, multiple choice.</td></tr>
              <tr><td><b>Embedding similarity</b></td><td>Cosine between the <G t="embedding">embeddings</G> of output and reference, pass above a cut-off.</td><td><b>Too lenient.</b> “Deploys are allowed on Fridays” and “Deploys are not allowed on Fridays” share almost every word and typically land close together in embedding space. Similar topic is not the same as correct.</td></tr>
              <tr><td><b>Code-based checks</b></td><td>Run the output: unit tests pass, JSON parses, SQL returns the right rows.</td><td>Only exists when the output is executable. When it does exist, prefer it over everything else.</td></tr>
              <tr><td><b>Model-graded (LLM-as-judge)</b></td><td>A second model reads the question, a rubric and the output, and writes a verdict.</td><td>Flexible enough for free text, and has measured biases of its own. See below.</td></tr>
              <tr><td><b>Human review</b></td><td>People read and mark.</td><td>The reference standard, and slow, costly and not perfectly consistent either. Used to calibrate the others.</td></tr>
            </tbody>
          </table>
        </div>
        <Term
          name="LLM-as-judge"
          plain={<>Using a language model as the scorer: give it the question, the answer and a marking scheme, and ask for a verdict.</>}
          example={<>“Does the answer state the same fact as the reference? Wording and number format do not matter. Reply PASS or FAIL with one sentence of reasoning.”</>}
          formal={<>A model-graded evaluation. Its verdicts are themselves predictions and must be validated against human labels before they are trusted.</>}
        />
        <Callout kind="established">
          Zheng et al. (2023), “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena”, measured the failure modes of LLM judges. <b>Position bias:</b> when comparing two answers, judges tend to favour one slot, often the first. Swapping the order changed the verdict in a large share of cases. <b>Verbosity bias:</b> longer answers are preferred even when the extra text adds nothing. <b>Self-enhancement bias:</b> a judge may favour answers written by itself. The paper names this one and also says its data was too limited to confirm it. They also found that a strong judge agreed with human raters more than 80% of the time, about as often as the human raters agreed with each other.
        </Callout>
        <p>So a judge is usable, on three conditions. Give it a specific rubric and a reference answer, not “rate this 1 to 10”. For pairwise comparisons, run both orders and count a flip as a tie. And <b>calibrate it</b>: have people label a sample, and measure how often the judge agrees with them. If it agrees with people less than people agree with each other, fix the rubric before you trust a single score.</p>
        <Callout kind="model">The harness in this lesson has no LLM in it. Its “rubric judge” is a deterministic function that applies three checks a judge prompt would ask for. It stands in for an LLM judge the same way the extractive answerer in <code>mini_rag.py</code> stands in for an LLM: so that every number is reproducible offline. It has none of a real judge’s flexibility and none of its biases.</Callout>

        <h3>4. Component metrics tell you where, end-to-end metrics tell you whether</h3>
        <p>When an answer is wrong, the first question is: which part failed? A RAG system has at least three separately measurable properties:</p>
        <div className="grid-3">
          <div className="card"><h4 style={{ fontSize: 16, marginBottom: 6 }}>Retrieval</h4><p>Was the chunk that holds the answer among the k retrieved? Measurable without any LLM.</p></div>
          <div className="card"><h4 style={{ fontSize: 16, marginBottom: 6 }}>Faithfulness</h4><p>Is every claim in the answer supported by the retrieved text? Also called groundedness. An answer can be faithful and still wrong, if retrieval brought the wrong text.</p></div>
          <div className="card"><h4 style={{ fontSize: 16, marginBottom: 6 }}>Answer correctness</h4><p>Does the final answer match what a person expects? This is the end-to-end number users feel.</p></div>
        </div>
        <Term
          name="recall@k"
          plain={<>Of the passages that should have been retrieved, what share showed up in the top k results?</>}
          example={<>17 answerable questions, one right passage each. The right passage is in the top 3 for 16 of them: recall@3 = 16/17 = 94%.</>}
          formal={<>|relevant ∩ top-k| / |relevant|, averaged over queries. With exactly one relevant passage per query it equals the hit rate.</>}
        />
        <p>In our harness, retrieval recall is 94% while answer correctness is 67%. That one comparison says: stop tuning the retriever, the answerer is what is failing. Without the component metric you would have guessed.</p>
      </MentalModel>

      <TryIt title="Run the harness, then question the number">
        <p>Start by pressing nothing: the first run is already there. A is the pipeline from lesson 9.1. B is the one-sentence-per-chunk idea. Read the verdict, then open B’s failures.</p>
        <EvalLab />
        <p>You will keep running into the same verdict: <em>cannot tell</em>. That is not the lab being timid. It is what 24 items can support. The next lab shows why, and what it would take to do better.</p>
        <SampleSizeLab />
      </TryIt>

      <Numbers>
        <p>Your eval has 50 items and the system passes 40. You report 80%. How far from the truth could that be?</p>
        <p>Treat each item as a coin that comes up “pass” with the system’s true accuracy p. Fifty flips of such a coin do not always give the same count. The typical wobble of the measured rate is called the <b>standard error</b>:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>step</th><th>calculation</th><th>result</th></tr></thead>
            <tbody>
              <tr><td>p(1 − p)</td><td>0.8 × 0.2</td><td>0.16</td></tr>
              <tr><td>÷ n</td><td>0.16 ÷ 50</td><td>0.0032</td></tr>
              <tr><td>square root = standard error</td><td>√0.0032</td><td>0.0566 = 5.7 points</td></tr>
              <tr><td>× 1.96 = 95% interval half-width</td><td>1.96 × 5.7</td><td><b>± 11.1 points</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>So “80%” means “somewhere between about 69% and 91%”. A change that moves the score from 80% to 83% is one and a half items. It is far inside the noise.</p>
        <p>The square root is the painful part. To halve the interval you need four times the items:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>items (at 80% accuracy)</th><th>24</th><th>50</th><th>100</th><th>200</th><th>500</th><th>1,000</th><th>2,000</th></tr></thead>
            <tbody><tr><td>95% interval, ± points</td><td>16.0</td><td>11.1</td><td>7.8</td><td>5.5</td><td>3.5</td><td>2.5</td><td>1.8</td></tr></tbody>
          </table>
        </div>
        <p>Now the real output of <code>python phase6-engineering/eval_harness.py</code> on our 24 items, which is also what the lab shows:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>scorer</th><th>passed</th><th>accuracy</th><th>agrees with the human labels</th><th>Cohen’s kappa</th></tr></thead>
            <tbody>
              <tr><td>exact</td><td>4/24</td><td>16.7%</td><td>11/24</td><td>0.15</td></tr>
              <tr><td>substring</td><td>12/24</td><td>50.0%</td><td>19/24</td><td>0.58</td></tr>
              <tr><td>rubric</td><td>16/24</td><td>66.7%</td><td>23/24</td><td>0.90</td></tr>
              <tr><td>human</td><td>17/24</td><td>70.8%</td><td /><td /></tr>
            </tbody>
          </table>
        </div>
        <p><b>Same system, same answers, and the score runs from 17% to 67% depending on the scorer.</b> The rubric judge is closest to the human, and still disagrees on one item: asked “does the oncall rotation change every friday”, the system quotes “The oncall rotation changes every Monday at 10am”. A person accepts that. The reference says “no, every Monday”, the word “no” is missing, and the judge fails it. The system was right and the eval was wrong. Reading your failures will regularly end with fixing the eval.</p>
        <p>The interval on the rubric score: 66.7%, 95% bootstrap interval <b>45.8% to 83.3%</b>. The regression check from the opening story: A 66.7%, B 58.3%, two items pass only under A and none only under B, exact sign test p = 0.50. <b>Verdict: cannot tell.</b> Two coin flips landing the same way happen half the time.</p>
        <Callout kind="dev">This is the same discipline as latency work. You would not compare two builds on one request each, and you would not report a p99 from 20 samples. A pass rate is a statistic. Give it the treatment you give every other statistic on your dashboards.</Callout>
      </Numbers>

      <TheMath>
        <Equation
          label="Standard error equals the square root of p times one minus p over n"
          symbols={[
            ['p', 'the pass rate you measured (as a fraction, 0.8 for 80%)'],
            ['n', 'the number of eval items'],
            ['SE', 'standard error: the typical distance between your measured rate and the true one'],
            ['1.96', 'how many standard errors cover 95% of a bell curve'],
          ]}
        >
          SE = √( p (1 − p) / n ) &nbsp;&nbsp;&nbsp; 95% interval ≈ p ± 1.96 · SE
        </Equation>
        <p>This assumes items are independent draws from the questions you care about. If your 500 items are 50 documents with 10 questions each, they are not independent, and the true uncertainty is larger than the formula says.</p>
        <p><b>Comparing two systems.</b> Run both on the <em>same</em> items and look at each item’s difference. Items that both pass, or both fail, cancel out. Only the items where they disagree carry information, and the question becomes: of those disagreements, is the split lopsided enough that a fair coin would rarely produce it? That is the <b>sign test</b> in the harness. Comparing two separately reported scores throws the pairing away and keeps all the noise of both. Evan Miller’s “Adding Error Bars to Evals” (2024) makes this the central recommendation, noting that models tend to get the same questions right and wrong.</p>
        <p><b>Repeated runs.</b> If your system samples with temperature above 0, one run per item adds a second source of noise on top of the choice of items. Run each item several times and average, or the same commit will score differently on two CI runs.</p>

        <h3>pass@k: the metric for code</h3>
        <p>Code is the friendly case: a generated function either passes the unit tests or it does not. Code benchmarks often ask a softer question than “is the first attempt right?”: if the model may try k times, does <em>any</em> attempt pass?</p>
        <Term
          name="pass@k"
          plain={<>The probability that at least one of k generated solutions passes the tests.</>}
          example={<>pass@1 = 30% and pass@10 = 75% means: the first try usually fails, but ten tries usually contain a winner.</>}
          formal={<>Estimated per problem from n ≥ k samples of which c pass, then averaged over problems.</>}
        />
        <Equation
          label="pass at k equals one minus the ratio of n minus c choose k to n choose k"
          symbols={[
            ['n', 'samples generated for this problem (the Codex paper used 200)'],
            ['c', 'how many of those n passed the unit tests'],
            ['k', 'the budget of attempts you are asking about'],
            [<>C(a, b)</>, '“a choose b”: the number of ways to pick b things out of a'],
            [<>C(n−c, k) / C(n, k)</>, 'the chance that k samples drawn from the n are all failures'],
          ]}
        >
          pass@k = 1 − C(n − c, k) / C(n, k)
        </Equation>
        <p>This is the unbiased estimator from Chen et al. (2021), the paper that introduced Codex and HumanEval. The tempting shortcut, 1 − (1 − c/n)<sup>k</sup>, is biased: it underestimates when n is small. With n = 5, c = 2, k = 2 the correct value is 1 − C(3,2)/C(5,2) = 1 − 3/10 = <b>0.70</b>, and the shortcut gives 0.64.</p>
        <DeepDive title="Cohen’s kappa: agreement that is not just luck">
          <p>The substring scorer agrees with the human on 19 of 24 items, 79%. But two raters who both say “pass” most of the time agree often by accident. Kappa subtracts that: κ = (p<sub>o</sub> − p<sub>e</sub>) / (1 − p<sub>e</sub>), where p<sub>o</sub> is the observed agreement and p<sub>e</sub> the agreement expected by chance.</p>
          <p>Here the scorer passes 50% and the human 70.8%, so chance agreement is 0.5 × 0.708 + 0.5 × 0.292 = 0.50. Then κ = (0.792 − 0.50) / (1 − 0.50) = 0.58. A kappa of 1 is perfect agreement and 0 is what coin-flipping raters achieve. Report it when you calibrate a judge, and compute human-versus-human kappa too: that is the ceiling.</p>
        </DeepDive>
        <DeepDive title="Why the lab shows a bootstrap interval and mentions another formula">
          <p>The ± 1.96 · SE formula breaks near the edges: 10 out of 10 gives SE = 0 and an interval of zero width, which claims certainty from ten items. The Wilson interval fixes that (10 out of 10 becomes 72% to 100%) and is what you should use for small n. The bootstrap takes a different route: resample your items with replacement thousands of times and see how much the score moves. It needs no formula, so it also works for averages of 1-to-5 ratings. On our 24 items the three methods give 47.8 to 85.5, 46.7 to 82.0 and 45.8 to 83.3. They agree on what matters: the interval is nearly 40 points wide.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>The harness imports the real pipeline, as <code>finetune_tiny_gpt.py</code> imports <code>tiny_gpt.py</code>. A golden item is a dictionary. <code>gold</code> names the sentence that holds the answer, which is what makes a retrieval metric possible:</p>
        <Code source="phase6-engineering/eval_harness.py" title="1. golden items: an answerable one and an unanswerable one">{`
{"id": "d1", "cat": "direct", "q": "how quickly must I acknowledge pages",
 "expected": "within 5 minutes", "source": "oncall.md",
 "gold": "Primary oncall must acknowledge pages"},
{"id": "u1", "cat": "unanswerable", "q": "what is the wifi password",
 "expected": None, "source": None, "gold": None},
`}</Code>
        <p>The task is one function. It returns the answer <em>and</em> the retrieved chunks, so that a failure can be located later:</p>
        <Code source="phase6-engineering/eval_harness.py" title="2. the task: call the system, record everything">{`
def run_task(item, embed, store, config):
    retrieved = store.search(embed(item["q"]), k=config["k"])
    raw = rag.extractive_answer(item["q"], retrieved, embed,
                                threshold=config["threshold"])
    refused = raw.startswith("Not found")
    m = re.match(r"(.*)\\. \\[source \\d+: (.+?)\\] \\(sim", raw)
    return {"retrieved": retrieved, "raw": raw, "refused": refused,
            "answer": None if refused else m.group(1),
            "cited": None if refused else m.group(2)}
`}</Code>
        <p>A scorer takes the item and the output and returns a verdict with a reason. The reason is not decoration: it is what you read when you triage failures.</p>
        <Code source="phase6-engineering/eval_harness.py" title="3. the simplest scorer">{`
def score_substring(item, out):
    if item["expected"] is None:
        return out["refused"], "refused" if out["refused"] else "answered a question that has no answer"
    if out["refused"]:
        return False, "refused an answerable question"
    ok = _plain(item["expected"]) in _plain(out["answer"])
    return ok, "reference found in the answer" if ok else f"the string '{item['expected']}' is not in the answer"
`}</Code>
        <p>The component metric ignores the answer entirely:</p>
        <Code source="phase6-engineering/eval_harness.py" title="4. retrieval hit: was the right chunk in the top-k?">{`
def retrieval_hit(item, out):
    if item["gold"] is None:
        return None
    return any(item["gold"] in c["text"] for c, _ in out["retrieved"])
`}</Code>
        <p>The bootstrap is a loop you could have written on day one of the course. Resample the 24 marks with replacement, take the mean, repeat, sort, read off the middle 95%:</p>
        <Code source="phase6-engineering/eval_harness.py" title="5. a confidence interval with no formula">{`
def bootstrap_interval(values, resamples=10000, seed=15):
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
`}</Code>
        <p>And the regression check. <code>b</code> and <code>c</code> count the items only one side passes. Everything else has cancelled:</p>
        <Code source="phase6-engineering/eval_harness.py" title="6. exact sign test on the disagreements">{`
def sign_test(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
`}</Code>
        <p>The last lines of the real output. The first comparison is the opening story. The last one is what a real regression looks like:</p>
        <Code lang="text" title="python phase6-engineering/eval_harness.py (excerpt)">{`
A = default, B = one sentence per chunk, no overlap
  A  66.7%  B  58.3%  difference -8.3 points, 95% interval -20.8 to +0.0
  only A passes: 2   only B passes: 0   sign test p = 0.500
  verdict: cannot tell: the difference is within noise
A = default, B = refusal threshold 0.90 (refuses almost everything)
  A  66.7%  B  25.0%  difference -41.7 points, 95% interval -66.7 to -16.7
  only A passes: 12   only B passes: 2   sign test p = 0.013
  verdict: A is better
`}</Code>
        <Callout kind="dev">In CI this becomes a gate with two thresholds, not one. A hard floor per category (“unanswerable must stay at or above 90%”), because an average can hide a collapse in the category that matters. And a paired test against the main branch, so that a merge is blocked by a statistically real regression and not by three items of noise. Cache the baseline outputs so that the gate costs one run, not two.</Callout>
      </CodeIt>

      <BreakIt>
        <p>In the first lab. Predict, then press Run.</p>
        <ul>
          <li><b>Switch the scorer to “Exact match”.</b> Accuracy drops to 4 of 24. Which four pass, and why only those? (Only refusals can be identical to their reference.)</li>
          <li><b>Set B’s threshold to 0.50.</b> B scores 18 to A’s 16. You made it better! Read the verdict. Then ask yourself how many thresholds you would have tried before finding that one, and what that does to a p-value. This is tuning on the test set.</li>
          <li><b>Set B’s threshold to 0.90.</b> The “unanswerable” category becomes perfect while the total collapses. A single category score can be gamed by a system that refuses everything. This is why you read the breakdown and the total together.</li>
          <li><b>Set k = 1 in B.</b> Recall@k drops from 16/17 to 14/17, and the accuracy does not move at all. A component metric can change while the end-to-end metric hides it. Open p6 to see a failure that no answerer could fix.</li>
          <li><b>Open a4 under the rubric judge.</b> The system is right and the eval is wrong. How would you rewrite the reference, or the rubric, so that it passes without letting wrong answers through?</li>
        </ul>
        <p>In the second lab:</p>
        <ul>
          <li>Set 20 items, 80%, and a 5-point gap. Press “Draw 20 new eval sets” a few times. The same unchanged system scores anywhere from about 60% to 95% or more.</li>
          <li>Find how many items the unpaired comparison needs for 80% versus 85%. (About 900 per set.) Then press “Best case” for the paired test. (155.)</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="evals-calc-interval"
          type="calculate"
          title="How wide is your number?"
          answer={{ value: 5.5, tolerance: 0.15 }}
          answerLabel="± points (one decimal)"
          hints={[
            'Standard error = √(p(1 − p) / n). Use p = 0.8 and n = 200.',
            '0.8 × 0.2 = 0.16. 0.16 / 200 = 0.0008. √0.0008 ≈ 0.0283, that is 2.83 points.',
            'Multiply the standard error by 1.96 for a 95% interval.',
          ]}
          solution={<><p>SE = √(0.16 / 200) = 0.0283. Times 1.96 gives <b>± 5.5 points</b>: the score is “somewhere between 74.5% and 85.5%”.</p><p>You quadrupled the eval set from 50 to 200 items and the interval only halved, from 11.1 to 5.5. That square root is why serious eval sets have hundreds to thousands of items, and why a paired comparison, which removes most of the noise without adding items, is worth so much.</p></>}
        >
          <p>Your eval set has 200 items and the system passes 160 of them (80%). What is the half-width of the 95% confidence interval, in percentage points?</p>
        </Exercise>

        <Exercise
          id="evals-calc-passk"
          type="calculate"
          title="pass@k by hand"
          answer={{ value: 0.533, tolerance: 0.006 }}
          answerLabel="pass@3 (three decimals)"
          hints={[
            'pass@k = 1 − C(n − c, k) / C(n, k). Here n = 10, c = 2, k = 3.',
            'C(8, 3) = 8·7·6 / 6 = 56 is the number of ways to pick 3 samples that are all failures. C(10, 3) = 120.',
            '1 − 56/120.',
          ]}
          solution={<><p>1 − 56/120 = <b>0.533</b>. Only 2 of 10 samples are correct, so pass@1 is 0.2, and yet three attempts succeed more than half the time.</p><p>That gap is why a pass@k number means little without its k. It is also the arithmetic behind <a href="#/lesson/reasoning-models">test-time compute</a>: when a verifier exists (here, unit tests), sampling more attempts buys accuracy.</p></>}
        >
          <p>For one programming problem you sample n = 10 solutions and c = 2 pass the unit tests. Using the unbiased estimator, what is pass@3?</p>
        </Exercise>

        <Exercise
          id="evals-debug-leak"
          type="debug"
          title="The eval that always improves"
          hints={[
            'Follow one failing question through the week. Where does it end up on Friday?',
            'What is the score measuring, once the prompt contains the eval’s own questions and answers?',
            'There is a second, quieter problem: how are the “before” and “after” numbers compared?',
          ]}
          solution={<><p><b>Bug 1: leakage.</b> The failing eval questions are pasted into the prompt as few-shot examples, with their correct answers. Next run, the system is being asked questions whose answers are in its context. The score rises because the test has been copied into the thing being tested. It says nothing about next week’s real questions. Fix: split the set. Failures from the <em>development</em> part may inspire prompt changes. A held-out part is never read, only scored, and is refreshed from new traffic.</p><p><b>Bug 2: no noise model.</b> “78% to 81% on 100 items” is 3 items. The 95% interval on each number is about ± 8 points. Compare per item on the same set, count the disagreements, and test them.</p><p>The same mechanism, at scale, is benchmark contamination: public test questions end up in web-scraped pretraining data.</p></>}
        >
          <p>A team describes its process: “Every Friday we run our 100-question eval. We take the questions that failed, add them with the right answers to the few-shot examples in the system prompt, and re-run. The score has gone up every week, from 78% to 81% to 85%. The eval proves the bot is improving.” What is wrong?</p>
        </Exercise>

        <Exercise
          id="evals-predict-locate"
          type="predict"
          title="Locate the failure before fixing it"
          hints={[
            'Two numbers are given: retrieval recall@5 and answer correctness. Which one is low?',
            'If the right passage is in the prompt 96% of the time and the answer is right 61% of the time, which component owns most of the gap?',
          ]}
          solution={<><p>The retriever is doing its job: the right passage reaches the prompt 96% of the time. Correctness is 61%, so in roughly a third of all cases the model had the answer in front of it and still got it wrong. A better embedding model can recover at most the 4% of retrieval misses.</p><p>The work is on the generation side: read the failures where retrieval hit, and sort them. Is the model answering from its weights instead of the context? Is the passage buried among 4 irrelevant ones, so try a smaller k or reranking? Is the prompt unclear about refusing? Also check the scorer on a sample: a strict scorer produces exactly this pattern.</p></>}
        >
          <p>A RAG system scores 61% answer correctness. The team plans to spend the next sprint swapping in a better embedding model. Their component metric says retrieval recall@5 is 96%. Predict how much the sprint can gain, and say what you would do instead.</p>
        </Exercise>

        <Exercise
          id="evals-implement-items"
          type="implement"
          title="Extend the golden set, and watch the interval"
          hints={[
            'Run python phase6-engineering/eval_harness.py --items first and read every FAIL line.',
            'Add items to GOLDEN_SET with new ids. For answerable ones, "gold" must be a fragment that really appears in rag.CORPUS[source]. The test test_golden_set_shape in tests/test_engineering_evals.py checks that (update its expected count).',
            'HUMAN_LABELS_DEFAULT needs a label for every new id: read the system’s answer and decide yourself. Then run pytest tests -q and fix the counts the tests pin.',
          ]}
          solution={<p>With 8 more paraphrase items you will most likely see the paraphrase score stay near one half, and the overall accuracy fall a little, because you added items from the weakest category. That is the point of categories: the overall number depends on the mix, which you chose. The bootstrap interval narrows only slightly, from about 38 points wide to about 33, because 32 items is still a tiny sample. You will also probably have to make a judgement call on at least one label, which is what building a golden set actually feels like.</p>}
        >
          <p>Open <code>phase6-engineering/eval_harness.py</code>. Write 8 new paraphrase questions about the four documents, the way a colleague who has never read them would ask. Before you run it: predict the paraphrase pass rate, and predict how much narrower the confidence interval becomes.</p>
        </Exercise>

        <ExplainBack
          id="evals-explain"
          prompt="A product manager says: “The new prompt scored 84% and the old one 81% on our 60-question eval, so the new one is better. Ship it.” Explain, without formulas, why that conclusion does not follow, and what you would do to find out."
          modelAnswer={<p>A score from 60 questions is a small sample of all the questions users will ask. Pick a different 60 and the same system would score differently, easily by ten points either way. Three points is two questions, far inside that wobble, so the two prompts could be equal or the old one could even be better. To find out, I would run both prompts on the same questions and look only at the questions where they disagree: if the new prompt wins nearly all of those and there are enough of them, the improvement is real. If there are only a handful of disagreements, the honest answer is “we cannot tell yet”, and the fix is more eval questions, ideally drawn from real traffic. I would also check the breakdown by category, because an average can rise while something important, like refusing unanswerable questions, gets worse.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'Why can a conventional unit test suite not replace an eval for an LLM feature?',
            options: [
              'LLM outputs are free text with many acceptable phrasings, quality is a rate and not a pass or fail, and outputs can vary from run to run',
              'Unit tests cannot call network services, and every LLM feature needs a network call to a hosted model, so the two cannot be combined',
              'LLM features are written in Python notebooks, while unit test frameworks only support compiled, statically typed production code',
              'Evals are only needed during model training, whereas unit tests are the right tool for everything that happens after deployment',
            ],
            answer: 0,
            explain: 'The workflow is the same (run on every change, gate merges). What differs is the assertion: you need a scorer that tolerates phrasing, and statistics to tell a real change from noise.',
          },
          {
            q: 'Your RAG bot has retrieval recall@3 of 94% and answer correctness of 67%. What is the most useful reading?',
            options: [
              'The two metrics contradict each other, so one of them must be computed wrongly and should be dropped',
              'Retrieval is the bottleneck, because any recall figure below 100% caps the correctness of the final answers',
              'The right text usually reaches the prompt, so most failures happen after retrieval: in the answering step or in the scorer',
              'The system is fine, because the average of the two metrics is above 80% and that is a common release bar',
            ],
            answer: 2,
            explain: 'Component metrics exist to locate failures. High retrieval recall with low correctness points at generation, prompt or scoring, not at the retriever.',
          },
          {
            q: 'An LLM judge compares two answers and prefers answer 1. You swap the order and it now prefers the other one, which is again in slot 1. What should you do?',
            options: [
              'Keep the first verdict, because the judge saw the answers in their natural order the first time',
              'Treat this pair as a tie, and report how often such flips happen, because this is position bias',
              'Average the two verdicts into a score of 0.5 for each answer and carry on without reporting anything',
              'Replace the judge with exact string matching, since model-graded evaluation is clearly unusable',
            ],
            answer: 1,
            explain: 'Position bias is documented for LLM judges. Running both orders and counting inconsistent verdicts as ties is the mitigation the MT-Bench authors propose. The flip rate is itself a quality measure for your judge.',
          },
          {
            q: 'On 50 eval items, prompt A scores 80% and prompt B scores 84%. Which statement is justified?',
            options: [
              'B is better, because four points on fifty items is two whole items and items cannot be half right',
              'B is better, provided both runs used temperature 0 so that there is no randomness left anywhere in the comparison',
              'A is better, because a prompt that scores lower on a small set is usually the more robust one on real traffic',
              'Nothing yet: each score has a 95% interval of about ± 11 points, so you need a paired test and probably more items',
            ],
            answer: 3,
            explain: 'Temperature 0 removes sampling noise, not the noise from which 50 questions you happened to pick. Two items of difference is well within chance.',
          },
          {
            q: 'A new model tops a public coding leaderboard. Why is that weak evidence that it will be best for your internal code-review bot?',
            options: [
              'Leaderboards are compiled by vendors, and vendors always publish numbers that are simply invented for marketing',
              'The benchmark measures a different task on different data, its questions may have leaked into training data, and small gaps are often within noise',
              'Public benchmarks only test models in English, and code review is a task that does not involve any natural language',
              'A model that is good at writing code is, for architectural reasons, usually worse than average at reading code',
            ],
            answer: 1,
            explain: 'A benchmark is someone else’s golden set, for someone else’s task. Use it to shortlist candidates. Decide with your own eval on your own data.',
          },
        ]}
      />

      <Remember
        items={[
          <>An eval has four parts: <b>dataset, task, scorer, aggregate</b>. Each can be wrong independently. Build the dataset from real traffic, past failures and adversarial cases, and keep a part of it <b>held out</b>.</>,
          <><b>The scorer is a design decision.</b> Exact match is too strict, embedding similarity too lenient, an LLM judge is flexible and has measured biases (position, verbosity). Calibrate any scorer against human labels and report the agreement.</>,
          <><b>Separate component metrics from end-to-end metrics.</b> Retrieval recall@k, faithfulness and answer correctness fail for different reasons. Measure each so a failure can be located.</>,
          <><b>A score is a range.</b> SE = √(p(1−p)/n). At n = 50 and 80%, that is ± 11 points. A 3-point change is noise. Quadruple the items to halve the interval.</>,
          <><b>Compare paired.</b> Same items through both systems, count the disagreements, test them. Put that test, plus per-category floors, in CI. Then keep watching online signals, because the golden set is never the whole world.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Golden set', sub: 'versioned with the code' }, { label: 'Offline eval', sub: 'every change, in CI' }, { label: 'Ship behind a flag' }, { label: 'Online signals', sub: 'A/B tests, feedback, guardrails' }, { label: 'New failures', sub: 'become golden items' }]} active={1} />
        <ToyVsReal
          toy={<ul><li>24 hand-written items, one tiny corpus</li><li>A deterministic pipeline, so one run per item is enough</li><li>A rule-based stand-in for the judge</li><li>One author’s labels as “the human”</li></ul>}
          real={<ul><li>Hundreds to thousands of items, sampled from production logs and refreshed</li><li>Sampling noise: several runs per item, and results averaged</li><li>LLM judges with rubrics, validated against human labels on a sample</li><li>Several annotators, with their agreement measured before anyone trusts the labels</li></ul>}
        />
        <h3>Offline evals and online signals</h3>
        <p>An offline eval answers “did this change break what we already know about?” before users see it. It cannot answer “is this what users need?”, because its questions are yesterday’s. Production systems pair it with online signals: A/B tests on a business metric, thumbs up and down, how often users rephrase or abandon, how often guardrails fire (refusals, blocked outputs, fallbacks to a human). Those signals are noisy and slow, and they are the only ones that measure the real distribution. The loop closes when a production failure becomes a new golden item.</p>

        <h3>What public benchmarks measure</h3>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>Style</th><th>Example</th><th>What is measured</th><th>Scorer</th></tr></thead>
            <tbody>
              <tr><td><b>Multiple choice</b></td><td>MMLU: 15,908 four-option questions across 57 subjects (Hendrycks et al., 2020)</td><td>Whether the model picks the right letter on exam-style knowledge questions</td><td>Exact match on a letter. Cheap and objective, and sensitive to prompt format.</td></tr>
              <tr><td><b>Execution-based</b></td><td>HumanEval: 164 hand-written programming problems with unit tests. SWE-bench: 2,294 real GitHub issues from 12 Python repositories. A patch must make failing tests pass without breaking passing ones. SWE-bench Verified is a 500-item human-validated subset.</td><td>Whether generated code actually works</td><td>Running tests. The strongest kind of scorer. HumanEval is reported as pass@k.</td></tr>
              <tr><td><b>Pairwise human preference</b></td><td>Chatbot Arena (LMArena): people chat with two anonymous models and vote for the better reply</td><td>Which model people prefer on whatever they chose to ask</td><td>Crowd votes, aggregated into ratings with a Bradley-Terry model (the family Elo ratings belong to)</td></tr>
            </tbody>
          </table>
        </div>
        <Callout kind="established">A leaderboard number rarely predicts your task, for three reasons. It measures a different task on a different distribution: exam questions are not your support tickets. Differences near the top are often smaller than the benchmark’s own noise. And public test sets leak: benchmark questions are on the web, the web is training data. One controlled check, GSM1k (Zhang et al., 2024), wrote fresh look-alikes of a popular grade-school maths benchmark and found accuracy drops of up to 8% for some model families, while many frontier models showed little sign of overfitting. Contamination is real, uneven, and hard to rule out from the outside.</Callout>
        <Callout kind="research">How to evaluate open-ended generation and multi-step agents is unsettled. LLM judges are widely used and their biases are still being mapped. Benchmarks saturate or leak within a year or two and are replaced. For agents, final-answer accuracy misses how the agent got there, which is why the <a href="#/lesson/production-agents">last lesson of this part</a> adds trajectory checks. Treat every eval method here as a tool with known error, not as ground truth.</Callout>
        <Callout kind="dev">The practical order of operations: use public benchmarks to shortlist two or three models. Build your own golden set from your own traffic, starting with 50 items this week and not 5,000 next quarter. Read the failures by hand before automating anything. Add the scorer that would have caught what you read. Put it in CI with a paired test. Most teams that skip evals do not lack tooling. They lack the afternoon spent writing down what “good” means.</Callout>
      </RealLLM>
    </Lesson>
  )
}
