import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM, BeforeMovingOn } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { CodeExercise } from '../components/python'
import { Exercise, ExplainBack } from '../components/exercise'
import { GradientDescentPlayground } from '../interactive/GradientDescentPlayground'
import { DescentLoop } from '../illustrations/DescentLoop'

export default function GradientDescentLesson() {
  return (
    <Lesson id="gradient-descent">
      <Why>
        <p className="lede">Here is a function you could write in ten seconds:</p>
        <Code>{`
def predict(x):
    return 3.0 * x - 1.5
`}</Code>
        <p>You chose the 3.0 and the −1.5. Now imagine nobody tells you those numbers. You only get examples: “for x = 1 the answer was 1.4, for x = 2 it was 4.6, …”. Could a program find the 3.0 and the −1.5 by itself?</p>
        <p>That is the whole of machine learning. A language model is also a function with numbers in it. It has billions of them instead of two, and nobody could ever type them in by hand.</p>
        <Callout kind="idea">
          We need a procedure that starts with <em>wrong</em> numbers and improves them automatically, using only examples. That procedure is called <b>gradient descent</b>. With small variations, it is how every neural network you have heard of was trained, including GPT.
        </Callout>
      </Why>

      <Problem>
        <p>First, two words you have met in passing: parameters in <a href="#/lesson/surprising-idea">The surprising idea</a>, loss in <a href="#/lesson/softmax">the softmax lesson</a>. Here they become precise.</p>
        <Term
          name="Parameters"
          plain={<>The adjustable numbers inside a model. The code around them is fixed; only these numbers change during learning. When people say “the model”, they really mean <b>this set of numbers</b>.</>}
          example={<>In <code>y = w*x + b</code> the parameters are <code>w</code> and <code>b</code>. Set w = 3, b = −1.5 and you have one model. Set w = 0, b = 0 and you have a different (worse) one.</>}
          formal={<>All learnable values of a model, often written θ. A “7B model” is a file holding 7 billion of them.</>}
        />
        <Term
          name="Loss"
          plain={<>One number that says how wrong the model currently is. Wrong predictions: big loss. Good predictions: small loss.</>}
          example={<>Predictions [2, 5], true answers [3, 3]. Errors are −1 and 2. Square them (1 and 4) and average: loss = 2.5.</>}
          formal={<>A function from (parameters, data) to a single number, lower meaning better. Here: mean squared error, MSE, which is never negative.</>}
        />
        <p>In lesson 1.3 the loss was “surprise” (<G t="cross-entropy">cross-entropy</G>), which fits when the answer is a choice among options. Here the answer is a number, so we measure the squared distance instead. Everything else in this lesson works the same for both.</p>
        <p><b>Why must the loss be a single number?</b> Because we are about to ask “did that change make things better or worse?”. With one number, that question always has an answer. With a list of 40 separate errors, one change might improve some and worsen others, and you could not say which way is “better”.</p>
        <WhyExists
          problem="A model has adjustable numbers. We have examples, and a loss that scores any setting of the numbers. We want the setting with the lowest loss."
          naive="Try random settings, or try every combination on a grid, and keep the best."
          fails="With 2 parameters and 100 values each, a grid is 10,000 tries. With 1,000 parameters it is 100 to the power 1,000. Blind search is hopeless beyond toy sizes."
          idea="Do not search blindly. At your current setting, work out for every parameter which direction would lower the loss, then move all of them a little in that direction. Repeat."
          tradeoff="You only ever know the slope where you stand, so you must take many small steps, and you have to choose the step size yourself."
        />
      </Problem>

      <MentalModel>
        <p>In <a href="#/lesson/derivatives">the last lesson</a> you nudged an input and watched how much the output moved. That ratio was the <G t="derivative">derivative</G>.</p>
        <p>Now do the same thing with a parameter as the thing you nudge, and the loss as the thing you watch:</p>
        <div className="card center mono" style={{ fontSize: 15 }}>“If I turn w up a tiny bit, does the loss go up or down? How fast?”</div>
        <p>Ask that question once per parameter. The list of answers is the <G t="gradient">gradient</G> you met at the end of the last lesson: one slope reading per knob. Each reading says which way is <em>uphill</em> for the loss, and how steep it is.</p>
        <p>Example: <span className="mono">[grad_w, grad_b] = [−8.4, 4.1]</span>. Turning w up lowers the loss quickly. Turning b up raises it. (These are the readings you will see when the playground below starts.)</p>
        <p>The gradient points uphill. We want downhill. So we step the <em>other</em> way:</p>
        <div className="card center mono" style={{ fontSize: 16 }}>new w = old w − learning_rate × grad_w</div>
        <p>A knob with a steep slope gets a big correction. A knob that barely matters gets a tiny one. Nobody has to decide which parameters are important: the slopes already say so.</p>
        <Callout kind="analogy">
          You are on a hillside in thick fog and want to reach the valley. You cannot see it. But you can feel the tilt of the ground under your feet. So: feel the slope, take a small step downhill, feel again, step again.
          <br /><br />
          <b>Why small steps?</b> Because the slope is only true where you stand. Ten metres away the ground may tilt differently. A giant leap based on a local reading can land you higher than you started.
          <br /><br />
          Where the analogy stops: a real hill has 2 directions. A model has one direction per parameter, so billions. There is no picture of that landscape, which is exactly why the fog is the honest part of the story. The arithmetic, though, is identical for 2 knobs or 2 billion.
        </Callout>
        <p>The full loop, which you will see again in every remaining part of this course:</p>
        <DescentLoop />
      </MentalModel>

      <TryIt title="Be the optimizer, then automate yourself">
        <p>Start in mode A. Do not skip it. Doing gradient descent by hand once is what makes mode B obvious instead of magical.</p>
        <GradientDescentPlayground />
      </TryIt>

      <Numbers title="One full update, by hand">
        <p>Three data points, all exactly on the hidden line y = 3x − 1.5. We start knowing nothing: w = 0, b = 0. The learning rate is 0.1.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th /><th>point 1</th><th>point 2</th><th>point 3</th><th>average</th></tr></thead>
            <tbody>
              <tr><td>x</td><td className="mono">1</td><td className="mono">2</td><td className="mono">−1</td><td /></tr>
              <tr><td>true y</td><td className="mono">1.5</td><td className="mono">4.5</td><td className="mono">−4.5</td><td /></tr>
              <tr><td>prediction = 0·x + 0</td><td className="mono">0</td><td className="mono">0</td><td className="mono">0</td><td /></tr>
              <tr><td>error = prediction − y</td><td className="mono">−1.5</td><td className="mono">−4.5</td><td className="mono">4.5</td><td /></tr>
              <tr><td>error²</td><td className="mono">2.25</td><td className="mono">20.25</td><td className="mono">20.25</td><td className="mono"><b>loss = 14.25</b></td></tr>
              <tr><td>2 · error · x</td><td className="mono">−3</td><td className="mono">−18</td><td className="mono">−9</td><td className="mono"><b>grad_w = −10</b></td></tr>
              <tr><td>2 · error</td><td className="mono">−3</td><td className="mono">−9</td><td className="mono">9</td><td className="mono"><b>grad_b = −1</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Read the gradient in words. grad_w = −10: turning w up lowers the loss, steeply. grad_b = −1: turning b up lowers the loss too, gently. Now step against the slope:</p>
        <div className="card mono" style={{ fontSize: 14.5 }}>
          w = 0 − 0.1 × (−10) = <b>1.0</b><br />
          b = 0 − 0.1 × (−1) = <b>0.1</b>
        </div>
        <p>Check that it helped. The new predictions are 1.1, 2.1 and −0.9. The new errors are −0.4, −2.4 and 3.6. The new loss is (0.16 + 5.76 + 12.96) / 3 = <b>6.29</b>. It was 14.25.</p>
        <p>Notice something odd: b moved <em>up</em> to 0.1, but its final value is −1.5. That is fine. Each step only uses the slope where you stand right now. On the next step grad_b already turns positive (0.53) and b heads back down. The path wiggles; the loss still falls.</p>
      </Numbers>

      <TheMath>
        <p>Where did “2 · error · x” come from? From the <a href="#/lesson/derivatives">chain rule</a> you already know. First the loss itself:</p>
        <Equation
          label="Mean squared error"
          symbols={[
            ['L', 'the loss: one number for the whole dataset'],
            ['N', 'how many data points'],
            [<>x<sub>i</sub>, y<sub>i</sub></>, 'the i-th input and its true answer'],
            [<>w·x<sub>i</sub> + b</>, 'the model’s prediction for that input'],
            ['( … )²', 'squaring makes every error positive and punishes big misses much more than small ones'],
            ['Σ / N', 'add up over all points, divide by N: an average'],
          ]}
        >
          L(w, b) = (1/N) Σ<sub>i</sub> ( w·x<sub>i</sub> + b − y<sub>i</sub> )²
        </Equation>
        <p>For one data point this is a two-stage pipeline: parameters → <b>error</b> → <b>error²</b>. The chain rule says: multiply the sensitivities of the stages.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>stage</th><th>nudge this…</th><th>…and this moves by</th></tr></thead>
            <tbody>
              <tr><td>error → error²</td><td>error</td><td className="mono">2 · error</td></tr>
              <tr><td>w → error = w·x + b − y</td><td>w</td><td className="mono">x</td></tr>
              <tr><td>b → error = w·x + b − y</td><td>b</td><td className="mono">1</td></tr>
            </tbody>
          </table>
        </div>
        <Equation
          label="Gradient of the mean squared error"
          symbols={[
            ['∂L/∂w', 'how fast the loss changes when only w is nudged (read “∂” as “a nudge in”)'],
            [<>error<sub>i</sub></>, <>w·x<sub>i</sub> + b − y<sub>i</sub>: prediction minus truth for point i</>],
            [<>2 · error<sub>i</sub> · x<sub>i</sub></>, 'the two stage sensitivities multiplied: (2 · error) × (x)'],
            ['η', 'the learning rate (Greek “eta”): the step size you choose'],
          ]}
        >
          ∂L/∂w = (1/N) Σ 2 · error<sub>i</sub> · x<sub>i</sub> &nbsp;&nbsp;&nbsp; ∂L/∂b = (1/N) Σ 2 · error<sub>i</sub>
          <br />
          w ← w − η · ∂L/∂w &nbsp;&nbsp;&nbsp; b ← b − η · ∂L/∂b
        </Equation>
        <p>Read the formula for w as a sentence: “a point has a lot to say about w when its error is big <em>and</em> its x is big”. A point at x = 0 says nothing about the slope w, because changing w does not move the prediction there at all.</p>
        <DeepDive title="Can gradient descent get stuck in a dip that is not the lowest one?">
          <p>For a line with squared error, no: the loss surface is a single smooth bowl (you can see it in the map in mode B), so downhill always leads to the one lowest point.</p>
          <p>Neural networks have bumpy loss surfaces with many dips. In practice this causes less trouble than the two-dimensional picture suggests. With millions of directions available, it is rare for <em>every</em> direction to lead uphill at once. How to describe these high-dimensional landscapes precisely is still studied; that large networks train well with plain gradient steps is an observed fact.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>We build the loop one line at a time, first with a single parameter: the model <code>pred = w * x</code>, and data that secretly follows y = 3x.</p>
        <Code title="Step 1: predict, then score the predictions">{`
pred = w * x                       # x and y are arrays of 100 numbers
loss = np.mean((pred - y) ** 2)    # one number: how wrong are we?
`}</Code>
        <Code title="Step 2: the slope of the loss with respect to w">{`
grad = np.mean(2 * (pred - y) * x)   # the formula from the math section
`}</Code>
        <Code title="Step 3: step against the slope">{`
w -= lr * grad     # minus: the gradient points uphill, we want downhill
`}</Code>
        <p>Wrap those three steps in a loop and you have Stage A of the repository file, unchanged:</p>
        <Code source="phase1-foundations/gradient_descent.py" title="stage_a(): one parameter">{`
x = rng.uniform(-2, 2, size=100)
y = 3.0 * x + rng.normal(0, 0.1, size=100)   # ground truth w = 3.0

w = 0.0            # start knowing nothing
lr = 0.1           # learning rate = step size

for step in range(30):
    pred = w * x
    loss = np.mean((pred - y) ** 2)
    grad = np.mean(2 * (pred - y) * x)   # dL/dw -- the hand-derived formula
    w -= lr * grad                       # THE update. This line is all of deep learning.
`}</Code>
        <p>Run it and w walks from 0 to 0.71 after one step, 2.41 after six, and ends at 3.0011. Nobody told it the answer was 3.</p>
        <p>Stage C is the version the playground runs: two parameters, and one new trick. Instead of all the data, each step looks at a small random sample.</p>
        <Term
          name="Mini-batch"
          plain={<>A small random handful of training examples used to <em>estimate</em> the gradient, instead of computing it on all the data.</>}
          example={<>1,000 points, batch of 32: each step is about 30× cheaper. The estimate is a bit noisy, but you take many steps and the noise averages out.</>}
          formal={<>Gradient descent with random batches is called stochastic gradient descent (SGD). “Stochastic” just means “involving randomness”.</>}
        />
        <Code source="phase1-foundations/gradient_descent.py" title="stage_c(): two parameters, mini-batches">{`
w, b = 0.0, 0.0
lr, batch_size = 0.05, 32

for step in range(400):
    idx = rng.integers(0, N, size=batch_size)     # random mini-batch
    xb, yb = x[idx], y[idx]

    pred = w * xb + b                             # 1. forward
    err = pred - yb
    loss = np.mean(err ** 2)                      # 2. loss

    grad_w = np.mean(2 * err * xb)                # 3. gradients
    grad_b = np.mean(2 * err)                     #    (dL/db: chain rule, x-term is 1)

    w -= lr * grad_w                              # 4. update
    b -= lr * grad_b
`}</Code>
        <p>It prints <code>learned w = 2.994, b = -1.507</code>. The truth was 3.0 and −1.5.</p>
        <Callout kind="dev">Look at the <em>shape</em> of that loop: forward, loss, gradients, update, over batches. When we train a GPT in <a href="#/lesson/training-gpt">Part 7</a>, the loop has exactly this shape. Only the line marked “forward” grows: from <code>w * xb + b</code> into a Transformer.</Callout>
        <DeepDive title="Stage B: gradients with no calculus at all, and why that is a great test">
          <p>You do not need a formula to get a slope. Use the definition: nudge the parameter, re-measure the loss, divide.</p>
          <Code source="phase1-foundations/gradient_descent.py" title="stage_b(): the numerical gradient">{`
def loss_fn(w):
    return np.mean((w * x - y) ** 2)

w, lr, h = 0.0, 0.1, 1e-5

for step in range(30):
    grad = (loss_fn(w + h) - loss_fn(w)) / h   # nudge & re-measure
    w -= lr * grad
`}</Code>
          <p>It ends at w = 2.9998: the same answer as the formula (each stage draws fresh random data, which explains the last digits).</p>
          <p>The catch is cost: one extra loss evaluation <em>per parameter</em>, every step. Fine for 1 parameter, impossible for a billion.</p>
          <p>But it makes a perfect <b>test oracle</b>. It is slow, it is hard to get wrong, and it does not depend on your clever formula being right. In <a href="#/lesson/backprop">Backpropagation</a> we compute gradients the fast way and check them against exactly this. You already do this as a developer: test the optimised implementation against a brute-force reference.</p>
        </DeepDive>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check in the playground (mode B).</p>
        <ul>
          <li><b>Learning rate 1.</b> Press Run. What will the line do? (It swings past the data, further each time. Watch the path on the map zig-zag outward and the loss grow by orders of magnitude.)</li>
          <li><b>Learning rate 0.001.</b> Run all 300 steps. Is it broken? (No. The loss falls on every step. It is just painfully slow.)</li>
          <li><b>Mini-batch of 1</b> at learning rate 0.1. The path staggers around like a drunk walker, yet it still ends up near the cross. Why does it never settle completely? (Each single point pulls the line toward itself, and the points disagree because of the noise.)</li>
          <li><b>In the Python file</b>, change Stage C’s data to <code>y = 3.0 * x**2 + 2</code> and leave the model a line. The loss falls, then stops at a floor it never breaks. Gradient descent finds the best <em>line</em>, but the truth is a curve. No amount of training fixes a model that cannot express the answer. That wall is the subject of <a href="#/lesson/neurons">the next lesson</a>.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="gradient-descent-code-step" />
        <Exercise
          id="gradient-descent-lr"
          type="experiment"
          title="Find the largest learning rate that still works"
          answer={{ value: 0.72, tolerance: 0.04 }}
          answerLabel="largest stable learning rate"
          hints={[
            'Use mode B with “all 40 points”. Reset before every try. Run 0.01, then 0.1, then 1, and watch the loss curve of each.',
            '0.1 converges and 1 explodes, so the edge is in between. Halve the interval each time: try 0.5, then 0.75, and so on. A run is “stable” if the loss ends lower than it started.',
            'Around 0.7 the line flips from side to side but slowly calms down. Slightly above, the flips grow instead of shrinking.',
          ]}
          solution={<><p>For these 40 points the edge is at about <b>0.72</b>. Below it, every overshoot is a little smaller than the previous one, so the bouncing dies out. Above it, every overshoot is a little bigger, so it grows without limit.</p><p>Watch out for one thing: the fastest learning rate is <em>not</em> the largest stable one. Near the edge the model wastes its steps bouncing. Something like 0.3 gets to the bottom much sooner. There is no general formula for a good learning rate in a real network; practitioners try a few and watch the loss curve, exactly as you just did.</p></>}
        >
          <p>In mode B, try three learning rates: 0.01, 0.1 and 1. Describe each loss curve in one word. Then search for the largest learning rate for which training still converges (use all 40 points, not mini-batches).</p>
        </Exercise>

        <Exercise
          id="gradient-descent-calc"
          type="calculate"
          title="One update by hand"
          answer={{ value: 1.5, tolerance: 0.001 }}
          answerLabel="new w"
          hints={[
            'Predictions first: pred = w·x for each point, then error = pred − y.',
            'Predictions are 1 and 2. Errors are 1 − 2 = −1 and 2 − 4 = −2. Now compute 2·error·x for each point and average.',
            'grad = mean(2·(−1)·1, 2·(−2)·2) = mean(−2, −8) = −5. Then w = 1 − 0.1 × (−5).',
          ]}
          solution={<><p>Errors: −1 and −2. Gradient: mean(−2, −8) = −5. Update: w = 1 − 0.1 × (−5) = 1 + 0.5 = <b>1.5</b>.</p><p>The gradient was negative (“turning w up lowers the loss”), and subtracting a negative number moves w up. The true value is 2, so w moved in the right direction, halfway there.</p></>}
        >
          <p>Model: <code>pred = w * x</code> (no b). Data: (x, y) = (1, 2) and (2, 4). Currently w = 1, and the learning rate is 0.1. What is w after one gradient descent step?</p>
        </Exercise>

        <Exercise
          id="gradient-descent-debug"
          type="debug"
          title="The loss goes up, whatever the learning rate"
          hints={[
            'The gradient is computed correctly. Look at how it is used.',
            'Which way does the gradient point: toward higher loss or lower loss?',
          ]}
          solution={<><p>The sign. <code>w += lr * grad</code> steps <em>with</em> the gradient, which is uphill: this is gradient <em>ascent</em>, and it maximises the error. It must be <code>w -= lr * grad</code>.</p><p>The tell-tale symptom: the loss rises even with a tiny learning rate. A learning rate that is merely too large makes the loss explode, but a small enough one always fixes it. If no learning rate helps, suspect the sign (or the gradient formula).</p></>}
        >
          <p>A colleague’s training loop makes the loss <em>increase</em> smoothly at every step, even with <code>lr = 0.0001</code>. What is wrong?</p>
          <Code>{`
for step in range(30):
    pred = w * x
    loss = np.mean((pred - y) ** 2)
    grad = np.mean(2 * (pred - y) * x)
    w += lr * grad
`}</Code>
        </Exercise>

        <Exercise
          id="gradient-descent-predict"
          type="predict"
          title="Which way will the knob turn?"
          answer={{ text: ['positive', 'pos', '+', 'plus', 'greater than zero', '> 0', '>0'] }}
          answerLabel="positive or negative?"
          hints={[
            'The line is too steep. Would making w even bigger make the loss go up or down?',
            'The gradient is the slope of the loss: it is positive when increasing the parameter increases the loss.',
          ]}
          solution={<><p><b>Positive.</b> The line is already too steep, so increasing w makes the fit worse: the loss rises with w, which is what a positive gradient means. The update subtracts it, so w goes down, toward 3.</p><p>Check it in mode A: set w = 5, b = −1.5. The hint reads “turn it DOWN, strongly”.</p></>}
        >
          <p>The data follows y = 3x − 1.5. Your current line has w = 5 and b = −1.5: right offset, far too steep. Without calculating: is grad_w positive or negative?</p>
        </Exercise>

        <Exercise
          id="gradient-descent-implement"
          type="implement"
          title="Add a third parameter"
          hints={[
            'In stage_c(), make a second input array x2, generate y = 3.0*x + 2.0*x2 - 1.5 + noise, and give the model a second weight w2.',
            'Copy the pattern: pred = w*xb + w2*x2b + b, and grad_w2 = np.mean(2 * err * x2b), then w2 -= lr * grad_w2.',
          ]}
          solution={<p>You needed one new line for the gradient and one for the update, and no new ideas. The sensitivity of the error to w2 is x2, for the same reason the sensitivity to w is x. This is why the method scales: a model with a billion parameters runs the same two lines a billion times (in practice, as a few large matrix operations).</p>}
        >
          <p>Open <code>phase1-foundations/gradient_descent.py</code>. Extend Stage C to fit <code>y = w1*x1 + w2*x2 + b</code> with a second random input. How many new ideas did you need?</p>
        </Exercise>

        <ExplainBack
          id="gradient-descent-explain"
          prompt="A teammate asks: “How can a program possibly find good values for a million numbers without trying all the combinations?” Explain gradient descent to them in plain words. Include why the steps have to be small."
          modelAnswer={<p>We define one number, the loss, that says how wrong the model is. For each adjustable number we work out the slope: if I turn this one up slightly, does the loss go up or down, and how fast? That list of slopes is the gradient. Then we move every number a small step in the direction that lowers the loss, all at once, and repeat. We never search combinations; we just keep walking downhill. The steps must be small because a slope is only valid near where you measured it: a big leap can overshoot the valley and end up worse than before.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'What does “the model” refer to, physically?',
            options: ['The Python source code of the training loop', 'The training data', 'The set of parameter values: the numbers that training adjusted', 'The loss function'],
            answer: 2,
            explain: 'The code is a fixed recipe. What training produces, what gets saved to disk and shipped, is the numbers.',
          },
          {
            q: 'Why does training need the loss to be a single number?',
            options: ['Because computers can only store one number at a time', 'So that “better” and “worse” are always defined, and every parameter can be given a slope toward “better”', 'Because squared errors cannot be stored in a list', 'It does not: a list of errors works equally well'],
            answer: 1,
            explain: 'Downhill only means something if there is one height. With one number per example, a change could help some and hurt others, and there would be no single direction to go.',
          },
          {
            q: 'grad_w = +4 at the current position. What does gradient descent do to w?',
            options: ['Increases it, because the gradient is positive', 'Decreases it: a positive slope means the loss rises as w rises, so go the other way', 'Sets it to 4', 'Leaves it; only negative gradients cause updates'],
            answer: 1,
            explain: 'w −= lr × 4. The gradient points uphill; the update steps downhill.',
          },
          {
            q: 'Your loss goes 2.1, 4.7, 93.5, 8812, then overflows. What is the first thing to try?',
            options: ['Train for more steps', 'Use more data', 'Lower the learning rate', 'Raise the learning rate to escape faster'],
            answer: 2,
            explain: 'Growing oscillation is the signature of steps that overshoot further each time. The slope is only locally valid, and the steps are leaving the region where it holds.',
          },
          {
            q: 'Why use a random mini-batch instead of all the data for each step?',
            options: ['It gives a more accurate gradient', 'It gives a slightly noisy gradient at a fraction of the cost, and many cheap steps beat a few expensive ones', 'Gradients cannot be computed on more than 32 examples', 'To make the results random on purpose'],
            answer: 1,
            explain: 'A sample of 32 already tells you roughly which way is downhill. For an LLM trained on trillions of tokens, using all the data for one step is not even possible.',
          },
        ]}
      />

      <Remember
        items={[
          <><b>Parameters</b> are the adjustable numbers. The model <em>is</em> those numbers; learning means changing them.</>,
          <>The <b>loss</b> squeezes “how wrong are we?” into one number, so that “better” has a direction.</>,
          <>The <b>gradient</b> is one slope per parameter: which way is uphill, and how steeply. We step the other way: <code>w -= lr * grad</code>.</>,
          <>Steps are <b>small</b> because a slope is only valid near where it was measured. Too small a learning rate: slow. Too large: the loss explodes.</>,
          <>The loop <b>predict → loss → gradient → update → repeat</b>, on random mini-batches, is the same loop that trains GPT.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'This lesson', sub: 'a line, 2 parameters' }, { label: 'Neural network', sub: 'next part' }, { label: 'Backpropagation', sub: 'fast gradients' }, { label: 'Training GPT', sub: 'Part 7' }]} active={0} />
        <ToyVsReal
          toy={<ul><li>2 parameters: w and b</li><li>Loss: mean squared error on a number</li><li>Gradient from a formula we derived by hand</li><li>Plain update: <code>w -= lr * grad</code></li><li>40 data points, hundreds of steps</li></ul>}
          real={<ul><li>Billions of parameters, all updated on every step</li><li>Loss: <G t="cross-entropy">cross-entropy</G> on the next token (the “surprise” from <a href="#/lesson/softmax">lesson 1.3</a>)</li><li>Gradients computed automatically by <G t="backprop">backpropagation</G></li><li>A refined update rule (Adam / AdamW) that adapts the step size per parameter, with a learning rate that changes over time</li><li>Trillions of tokens, in batches of millions of tokens, for hundreds of thousands of steps or more</li></ul>}
        />
        <Callout kind="established">The loop itself does not change. Predict, measure the loss, get the gradient, step against it, repeat over batches. Every neural language model, from the character-level toy we build later to the largest production systems, is trained by a variant of this loop.</Callout>
        <Callout kind="model">“Walking downhill in a landscape” is a picture for two parameters. Treat it as intuition for the update rule, not as a claim about what a billion-dimensional loss surface looks like.</Callout>
      </RealLLM>

      <BeforeMovingOn
        id="part-2"
        questions={[
          {
            q: 'From Part 0: what does a language model actually output each time it runs?',
            options: ['A complete sentence', 'A score for every token in its vocabulary, turned into probabilities for the next token', 'The single correct next word', 'A database key for the answer'],
            answer: 1,
            explain: 'One forward pass gives one probability distribution over the next token. Text appears by sampling a token, appending it, and running again.',
          },
          {
            q: 'From lesson 1.1: the dot product of [1, 2] and [3, −1] is…',
            options: ['1', '5', '[3, −2]', '−1'],
            answer: 0,
            explain: '1×3 + 2×(−1) = 3 − 2 = 1. Multiply matching entries, add up. The line model w·x + b is a dot product with one entry, plus an offset.',
          },
          {
            q: 'From lesson 1.3: a model gives the correct answer a probability of 0.01. Its cross-entropy loss for that example is…',
            options: ['close to 0, because 0.01 is small', 'large (about 4.6): it was very surprised by the truth', 'exactly 0.01', 'negative'],
            answer: 1,
            explain: '−ln(0.01) ≈ 4.6. Cross-entropy is a loss in exactly today’s sense: one number, lower is better, and gradient descent pushes it down.',
          },
          {
            q: 'From lesson 1.4: a pipeline has two stages. Stage one amplifies a nudge by 3, stage two by −2. A nudge of 0.01 at the input changes the output by about…',
            options: ['0.01', '+0.06', '−0.06', '0.05'],
            answer: 2,
            explain: 'Chain rule: sensitivities multiply, 3 × (−2) = −6, so 0.01 becomes −0.06. Today’s gradient 2·error·x is this rule applied to the loss.',
          },
          {
            q: 'Putting it together: what would “training a language model” mean, in the vocabulary of this lesson?',
            options: ['Storing sentences so they can be looked up later', 'Writing rules of grammar into the parameters by hand', 'Repeatedly adjusting the parameters a small step against the gradient of the next-token loss', 'Trying random parameter values until the text looks good'],
            answer: 2,
            explain: 'Same loop, different model and loss. The rest of the course fills in the model.',
          },
        ]}
      />
    </Lesson>
  )
}
