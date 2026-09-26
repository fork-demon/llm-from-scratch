import type { ReactNode } from 'react'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM, BeforeMovingOn } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack, OrderExercise } from '../components/exercise'
import { TestTimeCompute } from '../interactive/TestTimeCompute'
import { DirectVsSteps } from '../interactive/DirectVsSteps'

/** Every technique in Part 8 gets the same five rows, so none of them is only a name. */
function Technique({ name, problem, idea, helps, tradeoff, where }: { name: string; problem: ReactNode; idea: ReactNode; helps: ReactNode; tradeoff: ReactNode; where: ReactNode }) {
  const rows: [string, ReactNode][] = [['Problem', problem], ['Idea', idea], ['Why it helps', helps], ['Trade-off', tradeoff], ['Where it appears', where]]
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <h4 style={{ fontSize: 17, marginBottom: 6 }}>{name}</h4>
      <div className="table-scroll">
        <table className="plain" style={{ marginBottom: 0 }}>
          <tbody>
            {rows.map(([k, v]) => <tr key={k}><th scope="row" style={{ whiteSpace: 'nowrap', verticalAlign: 'top', width: 130 }}>{k}</th><td>{v}</td></tr>)}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function ReasoningModelsLesson() {
  return (
    <Lesson id="reasoning-models">
      <Why>
        <p className="lede">Sunday afternoon in Mysuru. Amma has the newspaper puzzle page folded on the dining table, and Riya is teasing her the way she once teased her students.</p>
        <p>“Seven eights?”</p>
        <p>“Fifty-six.” Amma does not even look up.</p>
        <p>“Forty-seven times eighty-six?”</p>
        <p>This time Amma says nothing. She takes the pencil from behind her ear and writes in the margin: 47 × 80 = 3,760. 47 × 6 = 282. Add them. “Four thousand and forty-two,” she says, and goes back to her crossword.</p>
        <p>She did not become cleverer in those thirty seconds. The pencil let her break one hard step into several easy ones, and the paper remembered the partial results for her.</p>
        <p>Since about 2024, some models visibly do the same. They write a long stretch of “thinking” before the answer. They are slower, they cost more, and they score clearly higher on maths, code and multi-step benchmarks.</p>
        <p>So here is the question. A normal language model can produce an answer directly. What changes when we allow it to spend more computation thinking?</p>
        <Callout kind="idea">
          Nothing about the architecture has to change for this. You already own every part of the explanation: the <a href="#/lesson/inference">generation loop</a>, one forward pass per token, and attention over everything written so far. This lesson connects them.
        </Callout>
      </Why>

      <Problem>
        <p>Go back to the loop Riya built in <a href="#/lesson/inference">Inference</a>. To produce one token, the input passes through the N blocks once. Then the token is fixed, and the loop moves on.</p>
        <Callout kind="established">
          <b>One generated token = one forward pass = a fixed amount of compute.</b> The number of layers does not grow when the question gets harder. “2 + 2 =” and “the 40th prime number is” get exactly the same number of matrix multiplications before the next token must come out. (The one part that grows is attention, which reads a longer cache as the text gets longer. That depends on length, not on difficulty.)
        </Callout>
        <p>So if the model must answer immediately, the hard part of the problem has to fit inside that fixed budget. For many problems it does not.</p>
        <p>Dev, back in Bengaluru, has a fix ready. “Then use a bigger model, na. Bigger is smarter.” It is a reasonable first guess. Here is why it is not enough.</p>
        <WhyExists
          problem="Some questions need more steps of computation than one forward pass provides."
          naive="Make the model deeper and bigger, so a single pass can do more."
          fails="That makes every token more expensive, including the easy ones. And however deep you go, the depth is still fixed, while problems can need any number of steps."
          idea="Keep the model as it is. Let it generate intermediate steps as ordinary tokens before the answer. Each one is another forward pass, and each one can be read back through attention."
          tradeoff="Those extra tokens are not free. You pay for them in latency and money, on every request."
        />
      </Problem>

      <MentalModel>
        <p>A Transformer has no scratchpad inside it that survives from one token to the next, apart from the <G t="kv-cache">KV cache</G> of what is already on the page.</p>
        <p>Kabir puts it in five words on the whiteboard: <b>the page is the scratchpad</b>.</p>
        <Term
          name="Chain of thought (a reasoning trace)"
          plain={<>Intermediate steps that the model writes out, as ordinary tokens, before it writes its final answer.</>}
          example={<>“17 × 3 = 51. 51 + 28 = 79. …” and only then “Answer: 113”.</>}
          formal={<>Nothing new in the maths: the model samples tokens r₁…r<sub>k</sub> first and the answer afterwards, so the answer is conditioned on the prompt <em>and</em> on r₁…r<sub>k</sub>.</>}
        />
        <p>Writing steps does two separate things:</p>
        <div className="grid-2">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>More compute</h4>
            <p>Fifty tokens of working means fifty forward passes aimed at your question instead of one.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Working memory</h4>
            <p>A partial result that has been written down does not have to be held inside the vectors. Later tokens attend to it.</p>
          </div>
        </div>
        <Callout kind="analogy">
          It is Amma’s pencil margin. The paper holds the partial results, and each line only asks for one easy step.
          <br /><br />
          Where the analogy stops: Amma can look at her margin and see whether a line is right. The model has no such guarantee. A wrong line on the page is read back just as faithfully as a right one, and everything after it is built on top. And the text of the trace is not a readout of what happened inside the network (more on this at the end).
        </Callout>
        <p>Before you press anything below, commit to a prediction.</p>
        <Exercise
          id="reasoning-models-direct-vs-steps"
          type="predict"
          title="Predict first: where does the work happen?"
          hints={[
            'Count forward passes. How many run before the first digit of a direct answer appears?',
            'In the direct case, where could the numbers 51, 79 and 158 be kept? They are never written anywhere.',
            'In the step-by-step case, to write “51 + 28 = 79” the model needs 51. Where can it get it from?',
          ]}
          solution={<><p><b>Directly:</b> one forward pass must carry out all four operations and keep three intermediate numbers (51, 79, 158) inside its vectors, because the very next token is already the answer. <b>With steps:</b> about 22 forward passes. Each line performs one operation, and its input is on the page: 51 is read back by attention, not remembered.</p><p>That is the whole mechanism: a fixed budget per token, so more tokens means more compute, and written tokens are memory. Whether a particular real model gets the direct version right depends on the model and the numbers. The point is what each version <em>demands</em> of a single pass.</p></>}
        >
          <p>“Start with 17. Multiply by 3. Add 28. Double it. Subtract 45. What do you get?” A model can reply with just the number, or write each step first. For each case, predict: how many forward passes run before the final answer is complete, and where are the intermediate results kept?</p>
        </Exercise>
        <DirectVsSteps />
        <Callout kind="established">Asking a model to write intermediate steps improves accuracy on multi-step problems such as arithmetic word problems. This has been measured repeatedly since 2022 (“chain-of-thought prompting”), mostly in larger models. The per-token compute argument above is a fact about the architecture.</Callout>
        <Callout kind="model">“The model thinks on the page” is a useful simplification. It correctly tells you that more tokens means more compute and more working memory. It does <em>not</em> tell you that the visible text is the reasoning. Computation also happens inside each forward pass, and that part is not written down.</Callout>
      </MentalModel>

      <TryIt title="Spend compute at answer time">
        <p>Longer traces are one way to spend more compute on a question. The other way is to try <b>several times</b>.</p>
        <p>That raises a question any developer will recognise: given N candidate answers, how do you choose one?</p>
        <Term
          name="Test-time compute"
          plain={<>Computation spent while <em>answering</em> (longer reasoning, several attempts, checking), as opposed to computation spent while training.</>}
          example={<>Sample 16 solutions to a coding task and return the one that passes the unit tests.</>}
          formal={<>Roughly: (tokens generated per attempt) × (number of attempts) × (cost of one forward pass), plus the cost of any checker.</>}
        />
        <p>The lab below compares three strategies on a pretend model. Try all three presets before you read on.</p>
        <TestTimeCompute />
        <p>What you should have found:</p>
        <ul>
          <li><b>Majority vote</b> only helps when the right answer is the <em>most common</em> answer. If one wrong answer is more popular, voting makes things worse.</li>
          <li>A <b>good verifier</b> helps far more, because it needs only one right attempt among the N. A sloppy verifier hits a ceiling that no amount of sampling can lift.</li>
          <li><b>Cost</b> is a straight line in N. Accuracy is not: each extra attempt buys less than the one before.</li>
        </ul>
        <h3>The methods, one card each</h3>
        <p>You have now used two of these. Here is the full set that people mean by “test-time compute” and “reasoning models”.</p>
        <p>Each one exists because of a specific problem. Each one has a price.</p>
        <Technique
          name="Reasoning traces (chain of thought)"
          problem="One forward pass is a fixed compute budget, and nothing but the page carries over between tokens."
          idea="Generate intermediate steps as tokens before the answer."
          helps="More forward passes on the question, and partial results become readable context."
          tradeoff="More tokens: slower and more expensive. An early wrong step is read back and built upon. Little or no benefit on simple lookups."
          where="Prompting (“think step by step”) on any capable model, and trained into reasoning models."
        />
        <Technique
          name="Self-consistency (sample several, majority vote)"
          problem="A single sampled trace can go wrong by chance."
          idea="Sample N traces at temperature above 0, extract each final answer, return the most common."
          helps="Different wrong paths tend to scatter over different wrong answers, while right paths agree. Needs no checker."
          tradeoff="N times the cost. Only works when the right answer is the most common one, and only when answers can be compared (a number, a choice), not for essays."
          where="Benchmark evaluation of maths and multiple-choice tasks; some production pipelines."
        />
        <Technique
          name="Best-of-N with a verifier"
          problem="Voting fails when the model is usually wrong, even if it is sometimes right."
          idea="Generate N candidates and let a separate check choose: unit tests, a proof checker, an exact-match answer, or a trained scoring model."
          helps="Checking is often much easier than generating. One right candidate among N is enough."
          tradeoff="N times the cost plus the checker. Only as good as the verifier: false accepts set a ceiling. Many tasks have no reliable checker."
          where="Code generation with tests, maths with known answers, formal proofs."
        />
        <Technique
          name="Search over partial solutions"
          problem="Sampling whole solutions wastes compute on attempts that were doomed at step two."
          idea="Score partial solutions as they grow. Extend the promising ones, abandon the rest (a tree search or beam search over reasoning steps)."
          helps="Compute is concentrated where it is likely to pay off."
          tradeoff="Needs a trustworthy score for half-finished work, which is harder to get than a check on a final answer. More engineering."
          where="Research systems and specialised domains such as theorem proving and games. How much general-purpose products rely on it is not public."
        />
        <Technique
          name="Reinforcement learning on verifiable outcomes"
          problem="Prompting for steps only elicits what the model can already do. We would like the model to get better at producing traces that end in right answers."
          idea="Let the model attempt many problems with checkable answers. Reward the attempts that the checker accepts, and update the weights to make such traces more likely."
          helps="The reward needs no human-written reasoning, only a checker. Behaviours that raise the success rate get reinforced."
          tradeoff="Limited to domains with reliable rewards. The model can learn to exploit a flawed checker. Traces become long and costly."
          where="DeepSeek-R1 (published recipe, open weights, 2025) and many open reasoning models since. Since 2025 the same idea has been applied to multi-step tasks with tools, and to fuzzier tasks scored against a written rubric (see Real LLMs below). Proprietary reasoning models are widely assumed to use related ideas, but their recipes are not public."
        />
        <Technique
          name="Distilling reasoning traces into a small model"
          problem="RL needs a model that already succeeds sometimes. A small model rarely does, so it has little to reinforce, and RL runs are expensive."
          idea="Let a strong reasoning model write many traces, keep the ones that end in right answers, and fine-tune a small model on them with ordinary next-token training (SFT)."
          helps="The small model copies the behaviour directly instead of discovering it. The DeepSeek-R1 paper reports that distilling R1’s traces into small models worked better than running RL on those small models directly."
          tradeoff="The student can only be as good as the traces it sees, and it inherits the teacher’s mistakes and habits. You need the strong model first."
          where={<>The distilled R1 models (1.5B to 70B parameters, built on Qwen and Llama), and many small open reasoning models since. The general idea is the subject of <a href="#/lesson/distillation">Small models from big ones</a>.</>}
        />
        <p>The vocabulary of reinforcement learning is the one from <a href="#/lesson/training-pipeline">the previous lesson</a>: the policy is the model, an action is the next token, and the reward arrives only at the end, when the final answer is checked.</p>
      </TryIt>

      <Numbers>
        <p>Riya does not trust a curve until she has computed one point of it herself. Let’s check two points of the lab by hand.</p>
        <p>Take p = 0.4 and four wrong answers, so each wrong answer turns up 0.6 ÷ 4 = 0.15 of the time. Use N = 3 attempts.</p>
        <p><b>Best-of-3 with a perfect checker.</b> It fails only if all three attempts are wrong:</p>
        <p className="mono">0.6 × 0.6 × 0.6 = 0.216 → accuracy = 1 − 0.216 = 0.784</p>
        <p><b>Majority vote over 3.</b> Count the ways the right answer can win:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>what the 3 attempts look like</th><th>probability</th><th>chance the vote picks right</th><th>contributes</th></tr></thead>
            <tbody>
              <tr><td>3 right</td><td className="mono">0.4³ = 0.064</td><td className="mono">1</td><td className="mono">0.064</td></tr>
              <tr><td>2 right, 1 wrong</td><td className="mono">3 × 0.4² × 0.6 = 0.288</td><td className="mono">1</td><td className="mono">0.288</td></tr>
              <tr><td>1 right, 2 wrong that <em>differ</em> (three-way tie)</td><td className="mono">3 × 0.4 × 0.6² × ¾ = 0.324</td><td className="mono">⅓</td><td className="mono">0.108</td></tr>
              <tr><td>anything else</td><td className="mono">0.324</td><td className="mono">0</td><td className="mono">0</td></tr>
              <tr><td colSpan={3}><b>majority-vote accuracy</b></td><td className="mono"><b>0.460</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>(The ¾: given two wrong attempts, the second one differs from the first with probability 3 out of 4.)</p>
        <p>So for the same 3× token bill: single attempt 40%, voting 46%, a perfect checker 78%. Set the lab to p = 40%, m = 4, N = 3, verifier 100% and you will see exactly these values.</p>
        <p>Now change one thing: a single wrong answer that the model gives 60% of the time (m = 1). Voting over 3 wins only with at least 2 right: <span className="mono">0.064 + 0.288 = 0.352</span>. That is <em>below</em> 40%. Voting amplifies whatever is most common, right or wrong.</p>
      </Numbers>

      <TheMath>
        <p>The perfect-checker case is worth having as a formula, because it sets the upper limit for any way of choosing among N attempts:</p>
        <Equation
          label="Best of N with a perfect verifier equals one minus, one minus p, to the power N"
          symbols={[
            ['p', 'the chance that one attempt is right'],
            ['1 − p', 'the chance that one attempt is wrong'],
            [<>(1 − p)<sup>N</sup></>, 'the chance that all N independent attempts are wrong: the only way a perfect checker can fail'],
            ['N', 'how many attempts you pay for'],
          ]}
        >
          accuracy = 1 − (1 − p)<sup>N</sup>
        </Equation>
        <p>And the bill:</p>
        <Equation
          label="Tokens spent equals N times tokens per attempt"
          symbols={[
            ['N', 'number of attempts'],
            ['tokens per attempt', 'reasoning trace plus final answer; each token is one forward pass'],
          ]}
        >
          tokens spent = N × tokens per attempt
        </Equation>
        <p>Put the two side by side. Cost grows in a straight line.</p>
        <p>The failure rate shrinks by the same <em>factor</em> (1 − p) with each attempt, so the gain from each extra attempt gets smaller and smaller. Going from 1 to 4 attempts at p = 0.4 buys 47 points. Going from 8 to 16 buys less than 2.</p>
        <DeepDive title="Why does a noisy verifier hit a ceiling?">
          <p>Our verifier accepts a right attempt with probability a and a wrong one with probability 1 − a. With many attempts, something is almost always accepted, so the result is “a random accepted attempt”. What fraction of accepted attempts is actually right?</p>
          <p className="mono">P(right | accepted) = p·a / ( p·a + (1 − p)(1 − a) )</p>
          <p>With p = 0.4 and a = 0.9: 0.36 / (0.36 + 0.06) = 0.857. No N gets you past 85.7%.</p>
          <p>With a hard problem, p = 0.05 and the same verifier: 0.045 / (0.045 + 0.095) = 0.321. When right answers are rare, even a small false-accept rate lets wrong answers flood the accepted pile. This is the same arithmetic as false positives in medical screening or spam filters.</p>
          <p>Practical consequence: sampling more is only as good as your checker. Unit tests, a proof checker or an exact numeric answer are near-perfect verifiers. “Ask another model whether this looks right” is not.</p>
        </DeepDive>
        <DeepDive title="How much compute is one forward pass?">
          <p>A widely used rule of thumb: generating one token costs about 2 × (number of parameters) floating-point operations, ignoring the attention cost that grows with context length. For a model with 7 billion parameters that is about 14 billion operations per token, whatever the token is. A 2,000-token reasoning trace is 2,000 of those.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>Here is where “one token = one forward pass” lives in the model you built. It is the loop from <code>tiny_gpt.py</code>, unchanged:</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="GPT.generate: each trip round the loop is one forward pass">{`
for _ in range(max_new_tokens):
    idx_cond = idx[:, -self.cfg.context_len:]   # crop to context window
    logits, _ = self(idx_cond)                  # ONE forward pass
    logits = logits[:, -1, :] / temperature     # last position only
    probs = F.softmax(logits, dim=-1)
    nxt = torch.multinomial(probs, num_samples=1)   # weighted die
    idx = torch.cat([idx, nxt], dim=1)              # feed back in
`}</Code>
        <p>The last line is the working memory. Whatever was just written is part of the input next time. A reasoning trace uses this loop exactly as it is, with a bigger <code>max_new_tokens</code>.</p>
        <p>The sampling strategies are a few lines wrapped <em>around</em> the model. These are sketches, not files in the repository.</p>
        <ul>
          <li><code>sample</code> stands for “run the loop above and decode the text”.</li>
          <li><code>final_answer</code> pulls the last line out of a trace.</li>
        </ul>
        <Code title="Sketch: self-consistency (majority vote)">{`
from collections import Counter

answers = []
for _ in range(N):                          # N independent attempts
    trace = sample(prompt, temperature=0.8) # must be > 0, or all N are identical
    answers.append(final_answer(trace))     # vote on the ANSWER, not on the whole trace

best = Counter(answers).most_common(1)[0][0]
`}</Code>
        <Code title="Sketch: best-of-N with a verifier">{`
candidates = [sample(prompt, temperature=0.8) for _ in range(N)]
accepted = [c for c in candidates if passes_tests(c)]   # the verifier
best = accepted[0] if accepted else candidates[0]
`}</Code>
        <Callout kind="dev">Best-of-N with tests is generate-and-test, the same shape as retrying a flaky operation until a health check passes. Self-consistency is a quorum read: ask several replicas and trust the majority. A quorum is only right if most replicas are right, which is exactly what the lab showed.</Callout>
      </CodeIt>

      <BreakIt>
        <p>Back to the lab. Predict, then check.</p>
        <ul>
          <li><b>Make voting harmful.</b> Choose “One popular wrong answer” and slide N upwards. Accuracy falls from 40% to 35.2% at N = 3 and 21.3% at N = 15. More votes make the popular wrong answer win more reliably.</li>
          <li><b>Set N = 2</b> with any p and m. Majority vote equals p exactly. Why? Two attempts that disagree are a tie, and a coin flip between them is worth the same as one attempt.</li>
          <li><b>Drag the verifier down to 50%.</b> The best-of-N curve goes flat at p. A checker that is no better than chance adds nothing, however many attempts you buy.</li>
          <li><b>“Hard problem, perfect checker”</b>: p = 5%. At N = 32 the checker reaches 80.6%, while majority vote is under 1%. Now lower the verifier to 90%: the ceiling drops to 32.1%.</li>
          <li><b>Raise tokens per attempt to 2,000</b> and read the bottom axis at N = 32: 64,000 tokens for one answer.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="reasoning-models-bestofn"
          type="calculate"
          title="Four tries with unit tests"
          answer={{ value: 0.59, tolerance: 0.006 }}
          answerLabel="probability (two decimals)"
          hints={[
            'The tests are a perfect checker. When does the whole procedure fail?',
            'Only when all 4 attempts are wrong. One attempt is wrong with probability 0.8.',
            '1 − 0.8⁴ = 1 − 0.4096.',
          ]}
          solution={<><p>1 − 0.8⁴ = 1 − 0.4096 = <b>0.59</b>.</p><p>A model that solves the task one time in five becomes a system that solves it three times in five, without touching a single weight. The price is four generations plus running the tests. This is a large part of why code is the showcase domain for test-time compute: the checker is cheap and nearly perfect.</p></>}
        >
          <p>A model writes a correct function 20% of the time. You sample 4 candidates and keep any one that passes a complete set of unit tests. What is the probability that you end up with a correct function?</p>
        </Exercise>

        <Exercise
          id="reasoning-models-vote-predict"
          type="predict"
          title="Will voting rescue this model?"
          hints={[
            'Majority vote returns the most common answer. Which answer is the most common here?',
            'Right: 30%. Each of the two wrong answers: 70% ÷ 2 = 35%.',
            'Set the lab to p = 30%, m = 2 and slide N.',
          ]}
          solution={<><p>No. Each wrong answer (35%) is more common than the right one (30%), so with many votes one of the wrong answers almost always wins. Accuracy drifts <em>down</em> from 30% towards 0 as N grows.</p><p>Voting does not find truth. It finds the mode of the model’s answer distribution. It helps exactly when the model is “right more often than it is wrong in any one particular way”. The margin here is small, so the decline is slow, but the direction is what matters.</p></>}
        >
          <p>On some question, a model gives the right answer 30% of the time. The rest of the time it gives one of two wrong answers, equally often. Predict what happens to majority-vote accuracy as N goes from 1 to 32. Then check in the lab.</p>
        </Exercise>

        <Exercise
          id="reasoning-models-cost"
          type="calculate"
          title="The bill"
          answer={{ value: 33600, tolerance: 1 }}
          answerLabel="tokens generated"
          hints={[
            'One attempt = thinking tokens + answer tokens.',
            'One attempt is 4,000 + 200 = 4,200 tokens. There are 8 attempts.',
          ]}
          solution={<><p>8 × (4,000 + 200) = <b>33,600</b> generated tokens, against 200 for a direct answer: 168 times as many forward passes. And since decoding is sequential, a single 4,200-token attempt is also roughly 21 times slower to arrive than the 200-token answer.</p><p>This is why “always use the reasoning model with maximum effort” is a poor default. “What is the capital of France?” gains nothing from 4,000 thinking tokens.</p></>}
        >
          <p>A direct answer takes 200 tokens. A reasoning model uses 4,000 thinking tokens and then the same 200-token answer. You run self-consistency with 8 samples. How many tokens are generated in total?</p>
        </Exercise>

        <Exercise
          id="reasoning-models-debug"
          type="debug"
          title="Sixteen samples, no improvement"
          hints={[
            'What has to be true of the 16 samples for a vote to tell you anything?',
            'Look at the temperature. What does sampling do as temperature goes to 0?',
          ]}
          solution={<><p>With temperature 0 the sampler always takes the most likely token, so all 16 traces are identical. The vote is 16 to 0 for whatever the single greedy answer was. They paid 16× for the accuracy of 1×.</p><p>Self-consistency needs <em>diverse</em> attempts, so temperature must be above 0 (values around 0.5 to 1 are typical). There is a second trap to look for in code like this: voting on the whole trace text. Every trace is worded differently, so every “answer” gets one vote. Vote on the extracted final answer.</p></>}
        >
          <p>Riya’s teammate adds self-consistency to a maths bot and sees exactly the same accuracy as before, at 16 times the cost. Why?</p>
          <Code>{`
answers = [final_answer(sample(prompt, temperature=0.0)) for _ in range(16)]
best = Counter(answers).most_common(1)[0][0]
`}</Code>
        </Exercise>

        <Exercise
          id="reasoning-models-experiment"
          type="experiment"
          title="How many attempts for 90%?"
          answer={{ value: 22, tolerance: 0 }}
          answerLabel="smallest N"
          hints={[
            'Set p = 10% and the verifier to 100%, then slide N until best-of-N reaches 90%.',
            'Or solve 1 − 0.9ᴺ ≥ 0.9, which means 0.9ᴺ ≤ 0.1.',
            '0.9²¹ ≈ 0.109 and 0.9²² ≈ 0.098.',
          ]}
          solution={<><p><b>N = 22</b>: 1 − 0.9²² ≈ 0.902, while N = 21 gives 0.891.</p><p>Twenty-two full generations for one answer. It works, and with a perfect checker it is honest work, but compare the alternative: a model with p = 0.5 needs only 4 attempts for 94%. Test-time compute multiplies what the model can already do sometimes. It is no substitute for a better model, which is why reasoning models are also <em>trained</em>, not just sampled harder.</p></>}
        >
          <p>In the lab, take a hard problem: p = 10%, with a perfect verifier (100%). What is the smallest N at which best-of-N reaches 90% accuracy?</p>
        </Exercise>

        <ExplainBack
          id="reasoning-models-explain"
          prompt="Dev says: “Reasoning models are a new kind of AI that actually thinks, unlike normal LLMs that just predict tokens.” Using what you know about the generation loop, explain what is really different and what is not."
          modelAnswer={<p>A reasoning model is still a next-token predictor running the same loop: one forward pass per token, a fixed amount of compute each time. What differs is how the tokens are used and how the model was trained. It writes intermediate steps before the answer, so it spends many forward passes on the question and can read its own partial results back through attention, like working on paper. Open recipes such as DeepSeek-R1 train this with reinforcement learning on problems whose answers can be checked, so traces that end in correct answers become more likely. The costs are real: more tokens, more latency, and no benefit on easy questions. And the visible trace is generated text, not a guaranteed faithful account of the computation inside the network.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'Why can writing intermediate steps make a model more accurate on a multi-step problem?',
            options: [
              'The steps switch on a special reasoning module in the network',
              'Each token is a fixed amount of compute, so more tokens means more compute, and written steps can be read back through attention as working memory',
              'Longer outputs always have lower loss',
              'The model looks the steps up in its training data',
            ],
            answer: 1,
            explain: 'Same architecture, same loop. The trace buys forward passes and puts partial results on the page where later tokens can attend to them.',
          },
          {
            q: 'A model answers a question correctly 25% of the time, and gives one particular wrong answer 60% of the time. You take a majority vote over 25 samples. What happens?',
            options: [
              'Accuracy rises towards 100%',
              'Accuracy stays at exactly 25%',
              'Accuracy falls towards 0%, because the vote converges on the most common answer, which is wrong',
              'The vote is always a tie',
            ],
            answer: 2,
            explain: 'Majority vote finds the mode of the answer distribution. It only helps when the mode is the right answer.',
          },
          {
            q: 'Why does a verifier usually help more than voting?',
            options: [
              'Verifiers are larger models',
              'With a reliable checker, one correct attempt among N is enough, even if correct attempts are rare',
              'Verifiers make each attempt cheaper',
              'Verifiers lower the temperature',
            ],
            answer: 1,
            explain: 'Voting needs the right answer to be the most common. Checking only needs it to appear once, and checking is often easier than generating (running tests vs writing the code).',
          },
          {
            q: 'Your verifier is 90% reliable: it wrongly accepts 10% of incorrect answers (and rejects 10% of correct ones). The problem is hard (p = 5%). You can afford unlimited samples. What limits your accuracy?',
            options: [
              'Nothing: enough samples always reach 100%',
              'The context window',
              'The false accepts: most accepted answers are wrong ones that slipped through, so accuracy tops out far below 100%',
              'The tokenizer',
            ],
            answer: 2,
            explain: 'With rare right answers, even a small false-accept rate floods the accepted pile: about 32% of accepted answers are right in this case. Sampling more is only as good as the checker.',
          },
          {
            q: 'Which statement about visible reasoning traces is the most accurate?',
            options: [
              'They are a log of the exact computation the network performed',
              'They are generated text that influences the answer, but they are not guaranteed to be a faithful explanation of why the model answered as it did',
              'They are written by a separate program after the answer is chosen',
              'They are always hidden because they contain the weights',
            ],
            answer: 1,
            explain: 'The trace really is input to later tokens, so it matters. But studies have found cases where the stated reasoning leaves out what actually drove the answer. Faithfulness is an open research question.',
          },
        ]}
      />

      <Remember
        items={[
          <><b>One token = one forward pass = fixed compute.</b> A model that must answer at once has a fixed budget for the hard part.</>,
          <>A <b>reasoning trace</b> buys more forward passes and uses the model’s own output as <b>working memory</b>, through the same generation loop you built.</>,
          <><b>Test-time compute</b> has three dials: longer traces, more attempts, and checking. <b>Majority vote</b> helps only if the right answer is the most common one. A <b>good verifier</b> helps much more, and a poor one sets a ceiling.</>,
          <>Reasoning models are <b>trained</b> for this, in open recipes by reinforcement learning on problems with <b>checkable answers</b>. That is why maths and code improved first. Small reasoning models are often made by <b>distilling</b> a big model’s traces instead.</>,
          <>Thinking is <b>not free</b> (cost is linear in tokens and attempts, gains flatten out), not every question benefits, and a trace is <b>not a guaranteed faithful explanation</b>.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Base model', sub: 'next-token prediction' }, { label: 'Assistant', sub: 'SFT + preferences' }, { label: 'Reasoning model', sub: 'RL on checkable problems' }, { label: 'At answer time', sub: 'you choose how much thinking to buy' }]} active={2} />
        <ToyVsReal
          toy={<ul><li>Each attempt is right with a fixed probability p</li><li>Attempts are independent; wrong answers are spread evenly</li><li>The verifier is a biased coin</li><li>“Tokens per attempt” is a slider</li></ul>}
          real={<ul><li>p differs for every question, and nobody tells you what it is</li><li>Errors are correlated: a model tends to repeat the same misconception, which is the “one popular wrong answer” case</li><li>Verifiers are tests, proof checkers, exact answers, or another model with its own blind spots</li><li>The model decides how long to think, often within an effort or budget setting chosen by the caller</li></ul>}
        />
        <Callout kind="established">
          The public reference point is <b>DeepSeek-R1</b> (2025), whose paper and weights are open. Its core is reinforcement learning on problems with automatically checkable results: maths answers, and code that must pass tests.
          <br /><br />
          The rewards are simple rules. An <b>accuracy reward</b> for a correct final answer. A <b>format reward</b> for putting the thinking between the required tags. In a later stage, a <b>language-consistency reward</b> as well, because the traces kept switching between languages mid-thought. There is no learned reward model for the reasoning tasks at all.
          <br /><br />
          The algorithm is <b>GRPO</b>, group relative policy optimization. The idea in one line: for each problem, sample a group of answers, and judge each one against how the rest of the group did rather than against an absolute target. An answer better than its siblings gets pushed up. A worse one gets pushed down. No separate value network is needed, which saves a lot of memory.
          <br /><br />
          For the variant trained by RL alone (R1-Zero), the authors report that responses grew longer, and that behaviours such as re-checking and trying a second approach became more frequent without being scripted. The released R1 model adds more stages: a small supervised “cold start” before the RL, then the reasoning RL, then supervised fine-tuning on filtered samples of its own output, then a final RL round that also uses learned preference rewards for helpfulness and harmlessness.
        </Callout>
        <h3>After R1: what changed in 2025 and 2026</h3>
        <p>R1 was the start of a very busy period, not the end of the story. Five threads are worth knowing by name.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>thread</th><th>what it is</th><th>status</th></tr></thead>
            <tbody>
              <tr>
                <td><b>GRPO variants</b></td>
                <td>Fixes to the training algorithm, each aimed at a flaw found in practice. <b>DAPO</b> (ByteDance and Tsinghua) changes the clipping, drops groups where every answer scored the same, and averages the loss per token so long answers are not under-weighted. <b>Dr. GRPO</b> removes two normalisations: one, by answer length, quietly favoured long wrong answers; the other, by the spread of scores in a group, gave extra weight to questions that were nearly always right or nearly always wrong. <b>GSPO</b> (Qwen team) judges the whole answer as one unit rather than token by token, which made training more stable, especially for mixture-of-experts models.</td>
                <td>Published, open code</td>
              </tr>
              <tr>
                <td><b>RL on agent tasks</b></td>
                <td>The same loop, but one “attempt” is now many steps: the model calls tools (a search engine, a terminal, a code runner), reads the results, and carries on. The reward arrives at the end, for example when a software task’s tests pass. Open model reports from 2025, such as Kimi K2 and GLM-4.5, describe this kind of training.</td>
                <td>Published for open models; widely used</td>
              </tr>
              <tr>
                <td><b>Rubric and judge rewards</b></td>
                <td>For tasks with no checker (writing, advice, research reports), another model grades the answer against a written rubric: “mentions the refund deadline”, “does not invent a policy”. This stretches RL beyond maths and code. It inherits the noisy-verifier ceiling from the deep dive above: the policy can learn to please the judge.</td>
                <td>Active research; used in some published recipes</td>
              </tr>
              <tr>
                <td><b>Hybrid thinking models</b></td>
                <td>One model with two modes: answer at once, or think first. The caller can switch thinking off, or set a <b>thinking budget</b> (a cap on thinking tokens). Qwen3 made this public in an open model in 2025, and several commercial APIs expose an effort or budget setting.</td>
                <td>Common in 2026 products</td>
              </tr>
              <tr>
                <td><b>Distilled reasoners</b></td>
                <td>Small models fine-tuned on a big reasoner’s traces, as in the card above. The R1 paper found this beat RL for small models. See <a href="#/lesson/distillation">Small models from big ones</a>.</td>
                <td>Published; standard practice</td>
              </tr>
            </tbody>
          </table>
        </div>
        <Callout kind="warn" label="What is not public">
          The internals and training recipes of proprietary reasoning models are <b>not published</b>. Statements of the form “model X runs a tree search inside” or “model X uses N hidden samples” are guesses unless the vendor has documented them. Several vendors also hide or summarise the raw trace. What you can observe from outside: a thinking phase that costs tokens and time, often with a setting that controls how much.
        </Callout>
        <Callout kind="research">
          <b>Are traces faithful?</b> The trace is real input to later tokens, so it does shape the answer. But experiments have found models whose stated reasoning omits the factor that actually changed their answer (for example a hint planted in the prompt). How far a trace can be trusted as an explanation, or used for oversight, is actively studied.
          <br /><br />
          Also open: how well training on checkable domains transfers to fuzzy ones such as writing or strategy, how to build reliable checkers for half-finished work, and whether RL teaches genuinely new abilities or mainly makes the model reliably do what it could already do sometimes.
        </Callout>
        <Callout kind="dev">If you call these models through an API, reasoning is a budget line. Thinking tokens are typically billed as output tokens and add latency before the first visible word. Route easy requests to a fast path, or switch thinking off, and keep long thinking for the requests that need it. You will meet the same loop again in <a href="#/lesson/agents">Agents</a>, where the “checker” is a tool result.</Callout>
        <p>That evening Riya sets the Paisa Pal bot’s thinking budget to zero for “what is my balance” questions, and leaves it on for disputed refunds. Amma, told about it on the phone, approves. “Nobody needs a pencil for seven eights.”</p>
      </RealLLM>

      <BeforeMovingOn
        id="part-8"
        intro="You have now gone from a dot product to a reasoning model. These questions reach back across Parts 0 to 8 on purpose. Pulling an idea out of memory is what makes it stick."
        questions={[
          {
            q: 'Part 0. When you send “What is a cat?” to an LLM, what does the model itself produce at each step?',
            options: ['A sentence retrieved from a database of earlier conversations', 'A parse tree of the question', 'A probability for every token in the vocabulary, from which one token is chosen', 'A ranked list of matching web pages'],
            answer: 2,
            explain: 'Everything else (chat, reasoning, tools) is built from repeating that one step.',
          },
          {
            q: 'Part 1. Softmax turns logits [2, 2, 2] into…',
            options: ['[2, 2, 2]', '[1, 0, 0]', '[⅓, ⅓, ⅓]', 'an error, because the scores are equal'],
            answer: 2,
            explain: 'Softmax only cares about differences. Equal scores in, equal probabilities out. It always produces an answer, which is also why a model has no built-in “I do not know”.',
          },
          {
            q: 'Parts 2 and 3. What does backpropagation compute?',
            options: ['For every weight, how much the loss would change if that weight were nudged', 'The best learning rate for the next step of training', 'The model’s predictions for the current batch', 'Which training examples are too noisy to keep'],
            answer: 0,
            explain: 'Gradients, via the chain rule, from the loss back through every layer. Gradient descent then steps each weight against its gradient.',
          },
          {
            q: 'Part 4. Why do LLMs use subword tokens and not whole words?',
            options: ['Subwords are easier for people to read', 'A fixed, manageable vocabulary can still spell any word, including ones never seen before', 'Whole words cannot be turned into vectors', 'It makes attention unnecessary'],
            answer: 1,
            explain: 'Rare and new words are split into known pieces, so nothing is ever out of vocabulary.',
          },
          {
            q: 'Part 6. In attention, what decides how much token i takes from token j?',
            options: ['Only the distance between the two positions', 'The value vector of j, compared with the value vector of i', 'A fixed rule written by the model’s designers', 'The softmax of the query of i dotted with the keys of the tokens it may see'],
            answer: 3,
            explain: 'Score (q·k), scale, softmax, then blend the values. This is also how a later token reads a partial result from a reasoning trace.',
          },
          {
            q: 'Part 7. Why does the KV cache give identical outputs to recomputing everything?',
            options: ['It rounds the numbers', 'Under the causal mask, earlier tokens never see later ones, so their keys and values never change', 'It skips some layers', 'It only works at temperature 0'],
            answer: 1,
            explain: 'Pure memoisation. It is also why long reasoning traces are affordable at all: each new thinking token processes one token, not the whole trace again.',
          },
          {
            q: 'Part 8. A model states, fluently and confidently, a “fact” that is false. Which explanation fits what is established?',
            options: ['It looked the fact up in its database and the record was corrupt', 'Its output is always a distribution over plausible next tokens; nothing in the architecture signals “unknown”, so thin knowledge still yields fluent text', 'The temperature was too low', 'The tokenizer dropped a word'],
            answer: 1,
            explain: 'No database, no record, no NOT FOUND. Retrieval, tools and training to abstain reduce the problem. None removes it.',
          },
          {
            q: 'Part 8. How does a small “student” model usually learn from a much bigger “teacher” model in distillation?',
            options: ['The teacher’s weights are copied into the student and then shrunk', 'The student is trained on the teacher’s outputs (its answers, traces or probabilities) with ordinary gradient descent', 'The student queries the teacher at answer time for every token', 'The two models are merged into one network'],
            answer: 1,
            explain: 'The architecture and weights of the student are its own. What it borrows is the teacher’s behaviour, as training data. That is how the small distilled R1 models were made.',
          },
          {
            q: 'Part 8. A multimodal model is given a photo. What does the Transformer inside it actually work on?',
            options: ['The raw file bytes, read left to right', 'A text caption that a separate program writes first, always', 'A sequence of vectors made from pieces of the image, placed in the same sequence as the text tokens', 'Nothing: images bypass the Transformer and go to a separate classifier'],
            answer: 2,
            explain: 'The image is cut into patches, each patch becomes a vector the same width as a token embedding, and attention then mixes image and text positions like any other tokens.',
          },
          {
            q: 'Part 8. What does preference tuning (RLHF or DPO) mainly change, compared with pretraining?',
            options: ['It shapes behaviour, which of many plausible responses the model tends to give, using comparisons between answers', 'It installs most of the model’s factual knowledge, using a much larger dataset than pretraining', 'It replaces the character tokenizer with a subword one', 'It removes the need for a context window'],
            answer: 0,
            explain: 'Knowledge comes overwhelmingly from pretraining. Later stages shape what the model does with it, and they can go wrong if the reward is flawed.',
          },
        ]}
      >
        <OrderExercise
          id="reasoning-models-order"
          title="From prompt to next token, from memory"
          prompt={<p>Rebuild the path of one generation step. No looking back.</p>}
          correct={[
            'Text is split into tokens',
            'Tokens become ID numbers',
            'IDs look up embedding vectors, plus position',
            'Attention: tokens exchange information',
            'MLP: each token is transformed on its own',
            'Repeat the block N times',
            'Final vector becomes logits over the vocabulary',
            'Softmax turns logits into probabilities',
            'Sample one token and append it to the input',
          ]}
          solutionNote={<p>Then the loop runs again with one more token on the page. A chat, a hallucination, a reasoning trace, a tool call: all of them are this loop, run many times.</p>}
        />
      </BeforeMovingOn>
    </Lesson>
  )
}
