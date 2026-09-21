import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { CodeExercise } from '../components/python'
import { Exercise, ExplainBack } from '../components/exercise'
import { SoftmaxPlayground, SurprisePlayground } from '../interactive/SoftmaxPlayground'

export default function SoftmaxLesson() {
  return (
    <Lesson id="softmax">
      <Why>
        <p className="lede">“The cat sat on the ___”. The model has to commit to a next word.</p>
        <p>You know from the <a href="#/lesson/matrices">last lesson</a> what its final step looks like: one big matrix multiply that produces <b>one score per word in the vocabulary</b>. Each score is a dot product, so it can be any number: 4.2, 0, −3.7.</p>
        <p>Raw scores are awkward. You cannot say “the model is 70% sure” from a 4.2. You cannot roll a die with a face of width −3.7. And during training you cannot measure <em>how wrong</em> the model was without knowing how much belief it put on the right answer.</p>
        <Callout kind="idea">
          The model needs to turn a list of arbitrary scores into <b>probabilities</b>. The function that does it is called <b>softmax</b>. It runs at the very end of the model every time a token is generated, and inside every attention layer to decide how much each word listens to each other word. Softmax itself never picks anything: it only produces the probabilities. Picking is a separate step, which you will meet below as sampling.
        </Callout>
      </Why>

      <Problem>
        <p>First, names for the two ends of the conversion.</p>
        <Term
          name="Logits"
          plain={<>The raw scores a model produces, one per possible answer, before they are turned into probabilities. Bigger means “fits better”. They can be negative and they do not add up to anything in particular.</>}
          example={<>Prompt: “My favourite pet is a ___”. Scores: <span className="mono">cat → 4.2</span>, <span className="mono">dog → 2.1</span>, <span className="mono">car → −0.7</span>.</>}
          formal={<>The vector z of unnormalised scores that is fed into softmax. For an LLM it has one entry per token in the vocabulary.</>}
        />
        <Term
          name="Probability distribution"
          plain={<>A list of numbers that says how belief is shared out among the options. Two rules: no number is negative, and together they add up to exactly 1 (that is, 100%).</>}
          example={<><span className="mono">cat 0.88, dog 0.11, car 0.01</span>. A fair die is <span className="mono">[⅙, ⅙, ⅙, ⅙, ⅙, ⅙]</span>.</>}
          formal={<>Numbers p₁ … pₙ with every pᵢ ≥ 0 and p₁ + … + pₙ = 1.</>}
        />
        <WhyExists
          problem="Turn any list of scores into a probability distribution, keeping the order: a higher score must get a higher probability."
          naive="Divide each score by the sum of the scores. That is how you would turn counts into percentages."
          fails={<>Try it: 4.2 + 2.1 − 0.7 = 5.6, so car gets −0.7 ÷ 5.6 = <b>−12.5%</b>. A negative probability is nonsense. And if the scores happen to sum to 0, you divide by zero.</>}
          idea={<>First make every score positive in a way that keeps the order: raise e (about 2.718) to the power of the score. e<sup>z</sup> is positive for every z. <em>Then</em> divide by the sum.</>}
          tradeoff="Exponentials grow violently. Big scores overflow (we fix that in a deep dive), and the largest score tends to grab most of the probability (we control that with temperature)."
        />
      </Problem>

      <MentalModel>
        <p>Softmax is two steps. Both are one line of code.</p>
        <div className="grid-2">
          <div className="card"><span className="chip acc">1 · exponentiate</span><p style={{ marginTop: 8 }}>Replace each score z by e<sup>z</sup>. Negative scores become small positive numbers. Large scores become very large ones.</p></div>
          <div className="card"><span className="chip acc">2 · normalise</span><p style={{ marginTop: 8 }}>Divide each result by the total, so everything adds up to 1.</p></div>
        </div>
        <p>The exponential does something more useful than just removing minus signs. It turns <b>gaps</b> between scores into <b>ratios</b> between probabilities:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>score gap between two words</th><th>0</th><th>1</th><th>2</th><th>3</th></tr></thead>
            <tbody><tr><td style={{ fontFamily: 'var(--sans)' }}>how many times more likely the higher one is</td><td>1×</td><td>2.7×</td><td>7.4×</td><td>20×</td></tr></tbody>
          </table>
        </div>
        <p>Every extra point of score multiplies the odds by about 2.7. So a slightly better score gives a clearly better probability, and a much better score wins almost everything. That is the “max” in the name. The “soft” is because the losers still keep a little: mathematically, nothing is ever exactly 0.</p>
        <Callout kind="analogy">
          Logits work like the Richter scale for earthquakes. One point higher on the scale does not mean “a bit stronger”, it means “stronger by a fixed multiple”. Differences on the scale are ratios in the real quantity.
          <br /><br />
          Where the analogy stops: softmax has a second step the Richter scale does not. After exponentiating, everything is divided by the total, so raising one score <em>lowers</em> every other probability. The words compete for a fixed budget of 100%.
        </Callout>
        <p>Once you have probabilities, choosing a word is easy to picture:</p>
        <Term
          name="Sampling"
          plain={<>Rolling a weighted die. Each option gets a face as wide as its probability. Roll, and see where it lands.</>}
          example={<>With <span className="mono">cat 0.88, dog 0.11, car 0.01</span>, out of 100 rolls you expect roughly 88 cats, 11 dogs and 1 car. Not exactly: it is a die.</>}
          formal={<>Drawing a random index i with probability pᵢ. In code: pick a uniform random number u in [0, 1) and walk along the running total of p until it passes u.</>}
        />
      </MentalModel>

      <TryIt title="Move the scores, watch the probabilities">
        <SoftmaxPlayground />
        <p>Things you should have seen:</p>
        <ul>
          <li>Raising one score lowers <em>every other</em> probability. They share 100%.</li>
          <li><b>Temperature</b> below 1 makes the winner take more. Above 1, the distribution flattens and unlikely words get real chances.</li>
          <li><b>Adding 100 to every score changes nothing.</b> Only the gaps between scores matter.</li>
          <li>A handful of rolls looks nothing like the probabilities. A few hundred rolls look a lot like them.</li>
        </ul>

        <h3>How wrong was the model? Loss as surprise</h3>
        <p>Training needs one number that says how bad a prediction was. Here is the idea the whole field uses. Look at the probability the model gave to the word that <em>actually came next</em>, and ask: <b>how surprised should the model be?</b></p>
        <p>If it said 90% and was right: barely surprised. If it said 1% and that word came: very surprised. We want a small number in the first case, a big one in the second, and exactly 0 for a perfect 100%. Minus the logarithm does that:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>probability given to the correct word</th><th>0.9</th><th>0.5</th><th>0.01</th></tr></thead>
            <tbody><tr><td style={{ fontFamily: 'var(--sans)' }}>surprise = −ln(probability)</td><td>0.105</td><td>0.69</td><td>4.6</td></tr></tbody>
          </table>
        </div>
        <p className="muted"><code>ln</code> is the natural logarithm, the inverse of e<sup>x</sup>: ln(x) answers “e to what power gives x?”. For numbers between 0 and 1 it is negative, hence the minus sign in front.</p>
        <SurprisePlayground />
        <Term
          name="Cross-entropy loss"
          plain={<>The model’s surprise at the right answer. Low when it gave the right answer a high probability, huge when it was confidently wrong.</>}
          example={<>The correct next word got probability 0.5: loss = −ln(0.5) = 0.69. It got 0.01: loss = 4.6.</>}
          formal={<>For one prediction, loss = −ln(p<sub>correct</sub>). Over a dataset, the average of that. This is the number that training pushes down. (The general definition compares two distributions, −Σ q<sub>i</sub> ln p<sub>i</sub>. When the truth q is “100% on the correct token”, only this one term survives. With ln the unit is called nats; with log₂ it would be bits.)</>}
        />
      </TryIt>

      <Numbers>
        <p>The pet example by hand. Logits: <span className="mono">cat 4.2, dog 2.1, car −0.7</span>.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>step</th><th>cat</th><th>dog</th><th>car</th></tr></thead>
            <tbody>
              <tr><td>logit z</td><td className="mono">4.2</td><td className="mono">2.1</td><td className="mono">−0.7</td></tr>
              <tr><td>tempting but wrong: z ÷ 5.6</td><td className="mono">0.75</td><td className="mono">0.375</td><td className="mono" style={{ color: 'var(--bad)' }}>−0.125 ✗</td></tr>
              <tr><td>e<sup>z</sup></td><td className="mono">66.69</td><td className="mono">8.17</td><td className="mono">0.50</td></tr>
              <tr><td>÷ total (75.35) = probability</td><td className="mono"><b>0.885</b></td><td className="mono"><b>0.108</b></td><td className="mono"><b>0.007</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Check the gap rule: cat leads dog by 2.1 points, and 66.69 ÷ 8.17 ≈ 8.2 = e<sup>2.1</sup>. The ratio depends only on the gap.</p>

        <h3>Temperature: divide the logits first</h3>
        <p>Dividing every logit by a number T before softmax shrinks or stretches the gaps:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>temperature</th><th>logits ÷ T</th><th>cat</th><th>dog</th><th>car</th></tr></thead>
            <tbody>
              <tr><td>T = 0.5 (sharper)</td><td className="mono">8.4, 4.2, −1.4</td><td className="mono">0.985</td><td className="mono">0.015</td><td className="mono">0.000</td></tr>
              <tr><td>T = 1</td><td className="mono">4.2, 2.1, −0.7</td><td className="mono">0.885</td><td className="mono">0.108</td><td className="mono">0.007</td></tr>
              <tr><td>T = 2 (flatter)</td><td className="mono">2.1, 1.05, −0.35</td><td className="mono">0.696</td><td className="mono">0.244</td><td className="mono">0.060</td></tr>
            </tbody>
          </table>
        </div>
        <p>The order never changes. Only how decisive the distribution is. This is the “temperature” setting you have seen in LLM APIs. (At T = 0.5, car is not truly 0: it is 0.00005.)</p>

        <h3>The loss for this prediction</h3>
        <p>If the correct word was “cat”: loss = −ln(0.885) = <b>0.12</b>. The model was nearly right, and is barely penalised.</p>
        <p>If the correct word was “dog”: loss = −ln(0.108) = <b>2.22</b>. Eighteen times larger. The only number that matters is the probability on the right answer.</p>
      </Numbers>

      <TheMath>
        <Equation
          label="p i equals e to the z i over T, divided by the sum over j of e to the z j over T"
          symbols={[
            [<>z<sub>i</sub></>, 'the logit (raw score) of option i'],
            ['e', 'the constant 2.718…; e to the power z is always positive, and grows by a factor e for every +1 in z'],
            ['T', 'temperature. T = 1 is plain softmax. Smaller T: sharper. Larger T: flatter'],
            [<>Σ<sub>j</sub></>, 'add up over all options: this is the total we divide by, so the results sum to 1'],
            [<>p<sub>i</sub></>, 'the probability of option i'],
          ]}
        >
          p<sub>i</sub> = e<sup>z<sub>i</sub> / T</sup> / Σ<sub>j</sub> e<sup>z<sub>j</sub> / T</sup>
        </Equation>
        <Equation
          label="loss equals minus the natural log of the probability of the correct option"
          symbols={[
            [<>p<sub>correct</sub></>, 'the probability the model gave to what really came next'],
            ['ln', 'natural logarithm, the inverse of e to the x'],
            ['−', 'ln of a number below 1 is negative; the minus makes the loss positive'],
          ]}
        >
          loss = −ln( p<sub>correct</sub> )
        </Equation>
        <p><b>Why only differences matter.</b> Add a constant c to every logit. Every e<sup>z</sup> gets multiplied by the same factor e<sup>c</sup>, top and bottom of the fraction, and it cancels. That is what the “add 100” switch showed.</p>

        <DeepDive title="Numerical stability: why real code subtracts the max">
          <p>e<sup>z</sup> grows so fast that computers give up early. In 64-bit floating point, e<sup>709</sup> is about 8 × 10<sup>307</sup> and e<sup>710</sup> is already “infinity”. In 32-bit floats the limit is near e<sup>88</sup> (the same for bfloat16, a 16-bit format many models use), and in ordinary 16-bit floats it is only about e<sup>11</sup>. Then infinity ÷ infinity gives <code>nan</code> (not a number), and one nan poisons everything it touches.</p>
          <p>The fix uses the fact you just learned: shifting all logits by a constant changes nothing. So subtract the largest logit from all of them. Now the largest is 0, its exponential is exactly 1, and every other exponential is between 0 and 1. Nothing can overflow.</p>
          <Code source="phase1-foundations/mlp_numpy.py" title="the softmax used throughout the repository">{`
def softmax(logits):
    # subtract the row max first: exp() of big numbers overflows, and
    # softmax is unchanged by shifting -- the classic numerical-stability trick.
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
`}</Code>
          <p>Here <code>logits</code> is a matrix with one row per example, so <code>axis=1</code> means “do this separately along each row”, and <code>keepdims=True</code> keeps the result as a column so the subtraction lines up row by row.</p>
        </DeepDive>
        <DeepDive title="Why e? Would 2 or 10 work?">
          <p>Any base above 1 gives a valid distribution with the same order. Using base 2 instead of e is exactly the same as using e with temperature T = 1 / ln 2 ≈ 1.44, so the base is not a separate choice from the temperature.</p>
          <p>e is the convention because it makes the calculus of the <a href="#/lesson/derivatives">next lesson</a> come out clean: the slope of e<sup>x</sup> is e<sup>x</sup> itself. Combined with the −ln in the loss, the gradient with respect to the logits is simply “probabilities minus the correct answer”. You will see that in <a href="#/lesson/backprop">Backpropagation</a>.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>The two steps, one line each:</p>
        <Code title="softmax, the direct way">{`
z = np.array([4.2, 2.1, -0.7])   # logits: cat, dog, car
e = np.exp(z)                    # [66.69, 8.17, 0.50]   all positive now
p = e / e.sum()                  # [0.885, 0.108, 0.007] sums to 1
`}</Code>
        <p>Temperature is one extra division, and the loss is one line:</p>
        <Code title="temperature and surprise">{`
p_flat = np.exp(z / 2.0) / np.exp(z / 2.0).sum()   # T = 2: [0.696, 0.244, 0.060]
loss = -np.log(p[1])                               # correct word was "dog": 2.22
`}</Code>
        <p>The repository’s version is the same recipe, made safe against overflow (see the deep dive above) and applied to a whole batch of rows at once:</p>
        <Code source="phase1-foundations/mlp_numpy.py" title="softmax and cross-entropy, as used for training">{`
def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, y):
    # loss = mean over batch of  -log(prob assigned to the CORRECT class)
    # "average surprise". 1e-12 guards log(0).
    return -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
`}</Code>
        <p><code>probs[np.arange(len(y)), y]</code> is NumPy shorthand for “from row 0 take column y[0], from row 1 take column y[1], …”: the probability each example gave to <em>its own</em> correct answer.</p>
        <Callout kind="dev">Sampling is a few lines too: <code>u = random()</code>, then walk through <code>p</code> adding up until the running total passes <code>u</code>. It is the same algorithm as weighted load-balancing between servers.</Callout>
      </CodeIt>

      <BreakIt>
        <p>In the first playground. Predict, then check.</p>
        <ul>
          <li><b>Make all five scores equal.</b> What are the probabilities? (20% each. No gaps, no preferences. The actual value of the scores is irrelevant.)</li>
          <li><b>Drag “banana” down to −5.</b> Does it ever reach exactly 0%? (No. e<sup>z</sup> is never 0. Softmax never rules anything out completely, at least until a score is so low that the computer rounds its share to 0, which is why sampled text occasionally contains a strange word.)</li>
          <li><b>Set T to 0.2</b>, then roll 100 times. Then T = 3 and roll 100 times. Which setting would you use for writing code, and which for brainstorming names?</li>
          <li><b>Turn on “add 100”</b> and watch the e<sup>z</sup> column. The numbers now have more than 40 digits. The probabilities do not move. Now imagine adding 1,000 instead: see the overflow exercise below.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="softmax-code-softmax" />
        <Exercise
          id="softmax-calc"
          type="calculate"
          title="Softmax by hand"
          answer={{ value: 0.665, tolerance: 0.006 }}
          answerLabel="probability of the first option"
          hints={['Exponentiate each logit: e² , e¹ , e⁰.', 'e² ≈ 7.39, e¹ ≈ 2.72, e⁰ = 1. Their total is 11.11.', 'First probability = 7.39 ÷ 11.11.']}
          solution={<><p>e² = 7.39, e¹ = 2.72, e⁰ = 1.00. Total 11.11. Probabilities: <b>0.665</b>, 0.245, 0.090.</p><p>Each step down of 1 in the logits divides the probability by e ≈ 2.72: 0.665 → 0.245 → 0.090. Remember this triple. It is handy for sanity-checking code.</p></>}
        >
          <p>Compute softmax of <code>[2, 1, 0]</code>. What is the first probability? (Three decimals. You may use a calculator for e<sup>x</sup>.)</p>
        </Exercise>

        <Exercise
          id="softmax-temperature"
          type="predict"
          title="Turn up the heat"
          answer={{ text: ['down', 'decreases', 'decrease', 'lower', 'smaller', 'goes down', 'it goes down', 'falls', 'drops', 'less'] }}
          answerLabel="up, down or unchanged?"
          hints={['Temperature divides the logits. What does dividing by 2 do to the gaps between them?', '[2, 1, 0] becomes [1, 0.5, 0]. Smaller gaps mean smaller ratios between probabilities.', 'Smaller ratios mean a flatter distribution. What must happen to the biggest probability if the others grow?']}
          solution={<><p>It goes <b>down</b>, from 0.665 to 0.506. The full distribution becomes [0.506, 0.307, 0.186].</p><p>Halving the logits halves every gap, so every ratio shrinks (from 2.72× to 1.65× per step). The ranking is unchanged. High temperature does not make the model “more creative” in any deep sense. It just hands more of the 100% to lower-ranked tokens, so the die lands on them more often.</p></>}
        >
          <p>Logits are <code>[2, 1, 0]</code>. You raise the temperature from 1 to 2. Does the probability of the top option go up, go down, or stay unchanged? Decide first. Then check the direction in the playground: note the top probability, and raise T.</p>
        </Exercise>

        <Exercise
          id="softmax-overflow"
          type="debug"
          title="nan, nan"
          answer={{ value: 0.731, tolerance: 0.006 }}
          answerLabel="correct first probability"
          hints={['What is np.exp(1000)? Try it. What is inf / inf?', 'Softmax only depends on the gaps between logits. [1000, 999] has the same gaps as which much friendlier pair?', 'Subtract the max: [0, −1]. Then e⁰ = 1 and e⁻¹ = 0.368.']}
          solution={<><p><code>np.exp(1000)</code> overflows to <code>inf</code>, and <code>inf / inf</code> is <code>nan</code>. Fix: <code>z = z - z.max()</code> before exponentiating. The logits become [0, −1], the exponentials [1, 0.368], and the result is [<b>0.731</b>, 0.269].</p><p>This is exactly what the repository’s <code>softmax</code> does. The bug is nasty in practice because it only appears once training has pushed some logit high enough, often hours into a run.</p></>}
        >
          <p>This returns <code>[nan, nan]</code>. Why? Fix it in your head. What should the first probability be?</p>
          <Code>{`
def softmax(z):
    e = np.exp(z)
    return e / e.sum()

softmax(np.array([1000.0, 999.0]))
`}</Code>
        </Exercise>

        <Exercise
          id="softmax-uniform-loss"
          type="calculate"
          title="The loss of knowing nothing"
          answer={{ value: 6.91, tolerance: 0.02 }}
          answerLabel="loss"
          hints={['If all 1,000 logits are equal, what probability does each word get?', 'Each gets 1/1000 = 0.001, including the correct one.', 'loss = −ln(0.001). Note that −ln(1/n) = ln(n).']}
          solution={<><p>Every word gets 0.001, so loss = −ln(0.001) = ln(1000) = <b>6.91</b>.</p><p>This is a useful debugging fact: a freshly initialised model should start with a loss near ln(vocabulary size). For GPT-2’s 50,257 tokens that is about 10.8. If your first loss is far above that, something is broken. Everything training achieves is pushing the loss down from this “knows nothing” level.</p></>}
        >
          <p>A brand-new, untrained model has a vocabulary of 1,000 words and gives every word the same logit. Whatever the correct next word is, what is the cross-entropy loss? (Two decimals.)</p>
        </Exercise>

        <ExplainBack
          id="softmax-explain"
          prompt="A colleague asks: “Why does softmax bother with e to the power of things? Why not just divide each score by the total?” Explain, and mention what temperature does."
          modelAnswer={<p>Scores can be negative or sum to zero, so dividing by the total can give negative “probabilities” or a division by zero. Exponentiating first makes every score positive while keeping the order. It also turns differences between scores into ratios between probabilities: each extra point multiplies the odds by about 2.7, so a better score wins clearly without the others dropping to exactly zero. Because only differences matter, adding a constant to all scores changes nothing. Temperature divides the scores before exponentiating: below 1 it widens the gaps and makes the top choice dominate, above 1 it shrinks the gaps and spreads probability to less likely options.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'A model outputs logits [3.0, 1.0, 0.2]. You add 50 to all three. What happens to the probabilities?',
            options: ['The first one gets much closer to 100%', 'They all become equal', 'Nothing: softmax only depends on the differences between logits', 'They overflow, so the result is undefined in principle'],
            answer: 2,
            explain: 'Adding c multiplies every e^z by e^c, and that factor cancels in the division. Overflow is a practical problem of naive code, not a property of softmax, and subtracting the max avoids it.',
          },
          {
            q: 'Why not simply divide each score by the sum of the scores?',
            options: ['It would be too slow', 'Scores can be negative or sum to zero, which gives negative or undefined “probabilities”', 'It would make all probabilities equal', 'It would change which option ranks first'],
            answer: 1,
            explain: 'A distribution needs non-negative numbers that sum to 1. e^z is positive for every z, so exponentiating first guarantees it.',
          },
          {
            q: 'You lower the temperature from 1.0 to 0.2. What changes?',
            options: ['The ranking of the tokens changes', 'The distribution gets sharper: the top token takes almost all the probability', 'The distribution gets flatter: rare tokens become more likely', 'The logits produced by the model change'],
            answer: 1,
            explain: 'Dividing by 0.2 multiplies every gap by 5. The model’s logits and the ranking are untouched. Only the die becomes more loaded towards the favourite.',
          },
          {
            q: 'The correct next word was “mat”. Model A gave “mat” 60%. Model B gave “mat” 2% and put 97% on “moon”. Which statement about the loss is right?',
            options: ['Both have the same loss because both made a prediction', 'B has a lower loss because it was more confident', 'A’s loss is −ln(0.6) ≈ 0.51, B’s is −ln(0.02) ≈ 3.9: being confidently wrong is punished hardest', 'The loss depends on the 97% given to “moon”, not on the 2%'],
            answer: 2,
            explain: 'Cross-entropy only looks at the probability given to the right answer. Since probabilities share 100%, putting 97% elsewhere is what forced “mat” down to 2%.',
          },
          {
            q: 'The model gives “mat” 65%. You generate the next word 100 times by sampling. What do you expect?',
            options: ['“mat” exactly 65 times', '“mat” every time, because it is the most likely', '“mat” around 65 times, give or take, and other words the rest of the time', '“mat” never twice in a row'],
            answer: 2,
            explain: 'Sampling is rolling a weighted die. Counts approach the probabilities over many rolls but are not exact. Always picking the top word is a different strategy (greedy decoding, temperature → 0).',
          },
        ]}
      />

      <Remember
        items={[
          <><b>Logits</b> are raw scores: any number, any sign. A <b>probability distribution</b> is non-negative and sums to 1. Softmax converts one into the other.</>,
          <>Softmax = <b>exponentiate, then divide by the total</b>. The exponential makes everything positive and turns score gaps into probability ratios (+1 ≈ 2.7× more likely).</>,
          <><b>Only differences between logits matter.</b> Adding a constant to all of them changes nothing, which is also the trick that keeps the computation from overflowing.</>,
          <><b>Temperature</b> divides the logits first. Low T: sharp and predictable. High T: flat and varied. The ranking never changes.</>,
          <><b>Sampling</b> is rolling a die weighted by those probabilities. <b>Cross-entropy loss</b> is the surprise at the right answer: −ln(p<sub>correct</sub>). Training exists to push it down.</>,
        ]}
      />

      <RealLLM>
        <ToyVsReal
          toy={<ul><li>3 to 5 candidate words</li><li>Logits set by hand with sliders</li><li>One softmax</li></ul>}
          real={<ul><li>One logit for every token in the vocabulary: about 50,000 in GPT-2, 100,000 or more in recent models</li><li>Logits computed by the whole network from your prompt</li><li>One softmax per generated token at the output, plus one per token per attention head in every layer</li></ul>}
        />
        <Callout kind="established">
          The same function does two jobs in a Transformer. At the output it turns vocabulary logits into <b>next-token probabilities</b>, which are then sampled (<a href="#/lesson/inference">Inference</a> covers temperature, top-k and top-p in detail). Inside <a href="#/lesson/attention">attention</a> it turns a row of match scores into <b>attention weights</b>: how much one token listens to each other token. Same formula, same “positive and sums to 1”.
        </Callout>
        <Callout kind="established">
          <G t="cross-entropy">Cross-entropy</G> on the next token is the training objective of every GPT-style model. Pretraining means: read text, predict each next token, measure −ln(p<sub>correct</sub>), and adjust the weights to make it smaller. How to “adjust the weights” is the subject of the <a href="#/lesson/derivatives">next lesson</a> and of <a href="#/lesson/gradient-descent">Part 2</a>.
        </Callout>
        <Callout kind="model">“The model is 88% sure it is cat” is a convenient way to talk. The 88% is a well-defined number that pretraining shapes to match how often continuations occur in the training text (later fine-tuning can distort that match). Whether it reflects anything like human confidence is a separate, and much harder, question.</Callout>
      </RealLLM>
    </Lesson>
  )
}
