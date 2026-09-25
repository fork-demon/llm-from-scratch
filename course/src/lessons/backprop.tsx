import { CodeExercise } from '../components/python'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM, BeforeMovingOn } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack, OrderExercise } from '../components/exercise'
import { BackpropFlow } from '../interactive/BackpropFlow'
import { ForwardBackward } from '../illustrations/ForwardBackward'

export default function BackpropLesson() {
  return (
    <Lesson id="backprop">
      <Why>
        <p className="lede">Tuesday, 10 a.m. Yesterday’s release broke refunds: forty customers were refunded twice. Riya is in the post-mortem.</p>
        <p>Nobody starts at the database. They start at the symptom and walk backward. The payout service sent two requests. Why? Its retry layer saw a timeout. Why? The ledger service answered slowly. Why? A new query in the ledger skipped an index.</p>
        <p>Each team only has to answer one question: “given what reached you, what did you do with it?” Then it passes the question upstream. By lunch every service knows its share of the blame, and nobody had to understand the whole system.</p>
        <p>Walking out, Kabir says: “You just did backpropagation. Now do it to a network.”</p>
        <p>Here is why that matters. You now have two things that do not yet fit together.</p>
        <div className="grid-2">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>A learning loop</h4>
            <p><a href="#/lesson/gradient-descent">Gradient descent</a>: get the slope of the loss for every parameter, step against it. For a line we derived the slope formula by hand.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>A model that can bend</h4>
            <p><a href="#/lesson/neurons">A neural network</a>: layers of weighted sums and hinges, with thousands of weights. Nobody can derive a formula per weight by hand.</p>
          </div>
        </div>
        <p>The loss is measured at the very end of the network. The weights are spread through every layer. When the prediction is wrong, how does a weight in the first layer, several steps away from the output, find out its share of the blame?</p>
        <p>This is the part everyone finds strange at first, and the one place in Part 3 where it pays to go slowly. Work the numbers by hand with us, and it turns into three small rules you can hold in your head.</p>
        <Callout kind="idea">
          The answer is <b>backpropagation</b>: run the chain rule backward through the network, once, and reuse intermediate results.
          <br /><br />
          It produces the gradient for every weight at about the cost of two extra forward passes. It is the reason training large networks is possible at all.
        </Callout>
      </Why>

      <Problem>
        <p>Riya’s first idea is the one she trusts most: the nudge experiment from <a href="#/lesson/derivatives">lesson 1.4</a>. Nudge one parameter, run the model again, see how the loss moved, divide. A gradient measured this way is called a <b>numerical gradient</b>.</p>
        <p>“It works,” Kabir agrees. “Now count.” It costs one full forward pass <em>per parameter</em>, for every single training step.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>model</th><th>parameters</th><th>forward passes for one update, by nudging</th></tr></thead>
            <tbody>
              <tr><td>our line</td><td className="mono">2</td><td className="mono">2 (plus one for the baseline)</td></tr>
              <tr><td>the spiral MLP</td><td className="mono">4,547</td><td className="mono">4,547</td></tr>
              <tr><td>a 7B language model</td><td className="mono">7,000,000,000</td><td className="mono">7 billion</td></tr>
            </tbody>
          </table>
        </div>
        <p>Training takes hundreds of thousands of updates. Seven billion forward passes for each of them is not slow, it is impossible.</p>
        <WhyExists
          problem="Gradient descent needs the slope of the loss for every weight, in every layer, at every step."
          naive="Nudge each weight in turn and re-run the network to see how the loss changes."
          fails="One forward pass per weight per step. And almost all of that work is repeated: nudging two weights in the same layer re-computes every later layer twice, identically."
          idea="Compute the sensitivities from the loss backward, layer by layer. Each layer’s result is computed once and reused by everything upstream of it."
          tradeoff="The backward pass needs the values from the forward pass, so they must all be kept in memory until it is done. Training costs far more memory than running the model."
        />
      </Problem>

      <MentalModel>
        <p>A network is a pipeline. The <a href="#/lesson/derivatives">chain rule</a> told you how nudges travel through a pipeline: each stage multiplies the nudge by its own sensitivity.</p>
        <ForwardBackward />
        <p>Here is the trick. Ask the question from the far end first.</p>
        <ol>
          <li>How does the loss react to a nudge in the <b>logits</b>? That is easy: they are right next to the loss.</li>
          <li>How does it react to a nudge in layer 2’s <b>input</b>? Take the answer from step 1 and pass it back through layer 2. Do not recompute it.</li>
          <li>Same again for layer 1, reusing step 2.</li>
        </ol>
        <p>Each layer receives one message from downstream: “this is how much the loss cares about each of your outputs”. It uses that message for two jobs: work out the gradient of its own weights, and produce the same kind of message for the layer before it.</p>
        <Term
          name="Forward pass, backward pass"
          plain={<>The <b>forward pass</b> runs the network left to right to get a prediction and a loss. The <b>backward pass</b> then walks right to left, handing each layer the sensitivity of the loss to its outputs, and collecting a gradient for every weight on the way.</>}
          example={<>Forward: x → h → logits → probs → loss = 2.13. Backward: d_logits → d_h → d_x, picking up d_W2 and d_W1 as it goes.</>}
          formal={<>Backpropagation is reverse-mode automatic differentiation: the chain rule evaluated from the output toward the inputs, so that all partial derivatives of one scalar (the loss) come out of a single sweep.</>}
        />
        <Callout kind="analogy">
          This is the post-mortem from this morning. The symptom (the loss) is known at the end. The payout team knows exactly what it did with what it received, so it can turn “the refund was doubled” into a question about its <em>input</em>, and pass that to the retry layer. And so on, upstream.
          <br /><br />
          No service needs to understand the whole system. It only needs its own operation and the message from downstream.
          <br /><br />
          Where the analogy stops: “blame” here is not a judgement and it is not all-or-nothing. A post-mortem usually finds one culprit. Backpropagation gives <em>every</em> value a number: how much the loss would change if that value were nudged. It can be positive, negative or zero.
        </Callout>
        <p>One clarification that prevents a common confusion: backpropagation does not learn anything. Learning is still <code>w -= lr * grad</code>. Backpropagation is only the fast way to <em>get</em> <code>grad</code>.</p>

        <h3>The start of the backward pass: the gradient is the miss</h3>
        <p>Our network is a classifier, so its output goes through the pair you met in <a href="#/lesson/softmax">lesson 1.3</a>: <G t="softmax">softmax</G> turns the scores (<G t="logits">logits</G>) into probabilities, and <G t="cross-entropy">cross-entropy</G> measures the surprise: minus the log of the probability given to the correct class.</p>
        <p>The backward pass must start with the sensitivity of that loss to each logit. You might expect something ugly, with exponentials and logs. It collapses into this:</p>
        <div className="card center mono" style={{ fontSize: 16 }}>d_logits = probs − one_hot</div>
        <p>“one_hot” is the correct answer written as a list: 1 for the right class, 0 elsewhere. Say the model predicted [0.67, 0.24, 0.09] and the truth was the first class, [1, 0, 0]. The gradient is [−0.33, 0.24, 0.09].</p>
        <p>Read it: the correct class fell short by 0.33, so push its score up. The others got 0.24 and 0.09 they did not deserve, so push those down by exactly that much. If the prediction were perfect, the gradient would be all zeros and nothing would change.</p>
      </MentalModel>

      <TryIt title="Watch blame flow backward">
        <p>This is the same tiny network you evaluated by hand in <a href="#/lesson/neurons">the last lesson</a>: input [1, 2], hidden layer [2, 0], logits [1, −1]. The correct class is class 2, and the model gives it only 12%.</p>
        <BackpropFlow />
      </TryIt>

      <Numbers title="The whole backward pass, by hand">
        <p>Same network, same input. The forward pass gave: h = [2, 0], logits = [1, −1], probs = [0.881, 0.119]. The correct class is class 2, so the loss is −ln(0.119) = 2.13.</p>
        <p>Now walk backward. “d_something” means: how much the loss changes per unit of nudge to “something”.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>step</th><th>rule</th><th>numbers</th></tr></thead>
            <tbody>
              <tr><td>1 · at the logits</td><td className="mono">probs − one_hot</td><td className="mono">[0.88, 0.12] − [0, 1] = <b>[0.88, −0.88]</b></td></tr>
              <tr><td>2 · output weights W2</td><td className="mono">(value that came in) × (blame where it went)</td><td className="mono">from h₁ = 2: [2×0.88, 2×(−0.88)] = <b>[1.76, −1.76]</b><br />from h₂ = 0: <b>[0, 0]</b></td></tr>
              <tr><td>3 · hidden outputs h</td><td className="mono">blame × the weight it travelled through, summed</td><td className="mono">d_h₁ = 0.88×0.5 + (−0.88)×(−0.5) = <b>0.88</b><br />d_h₂ = 0.88×0.5 + (−0.88)×1.0 = <b>−0.44</b></td></tr>
              <tr><td>4 · through the ReLU</td><td className="mono">pass if the hinge was open, else 0</td><td className="mono">h₁ open: <b>0.88</b> · h₂ shut: <b>0</b></td></tr>
              <tr><td>5 · first-layer weights W1</td><td className="mono">(value that came in) × (blame where it went)</td><td className="mono">into h₁: [x₁, x₂] × 0.88 = [1×0.88, 2×0.88] = <b>[0.88, 1.76]</b><br />into h₂: <b>[0, 0]</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Two things to notice.</p>
        <p><b>Reuse.</b> Step 3 used the result of step 1. Step 5 used the result of step 4. Nothing was computed twice. With nudging, each of the 8 weights would have needed its own complete forward pass.</p>
        <p><b>The shut hinge.</b> Hidden unit 2 output 0. Its incoming weights get zero gradient, and so do its outgoing ones. That is correct, not a flaw: for this input, tiny changes to those weights really do not change the loss.</p>
        <p>Apply the update with learning rate 0.1. Take the weight from h₁ to logit 1, which helped the wrong class: its gradient is 1.76, so it moves from 0.5 to 0.5 − 0.1 × 1.76 = 0.32. Do the same for every weight and bias, run the forward pass again, and the loss has dropped from 2.13 to <b>1.16</b>. One step.</p>
      </Numbers>

      <TheMath>
        <p>This is the densest point in the course, so we take it in two passes: first one example with real numbers, then the batch version the code uses.</p>
        <p>In every rule, <code>d_out</code> is the message arriving from downstream: the sensitivity of the loss to this operation’s output.</p>
        <h3>First pass: one example, the output layer</h3>
        <p>A reminder of the convention. In this course and in the repository, a layer computes <code>out = x @ W + b</code>. The input x is a <em>row</em>. W has one <b>row per input</b> and one <b>column per output</b>, so W[i, j] is the wire from input i to output j. (Some books write W x instead; their rules look transposed, but they say the same thing.)</p>
        <p>Our output layer, with the numbers from the table above:</p>
        <div className="table-scroll">
          <table className="plain">
            <tbody>
              <tr><td>input (hidden layer) h</td><td className="mono">[2, 0]</td></tr>
              <tr><td>W2 (row = from h₁ or h₂, column = to logit 1 or 2)</td><td className="mono">[[0.5, −0.5], [0.5, 1.0]]</td></tr>
              <tr><td>forward: h @ W2</td><td className="mono">[2×0.5 + 0×0.5, 2×(−0.5) + 0×1.0] = [1, −1]</td></tr>
              <tr><td>message from downstream d_out</td><td className="mono">[0.88, −0.88]</td></tr>
            </tbody>
          </table>
        </div>
        <p><b>Weights.</b> Each weight gets (the input that flowed into it) × (the blame at the output it fed). That fills a grid with every input times every blame, called an <em>outer product</em>:</p>
        <div className="table-scroll">
          <table className="plain mono">
            <thead><tr><th>d_W2</th><th>to logit 1 (blame 0.88)</th><th>to logit 2 (blame −0.88)</th></tr></thead>
            <tbody>
              <tr><td>from h₁ = 2</td><td>2 × 0.88 = 1.76</td><td>2 × (−0.88) = −1.76</td></tr>
              <tr><td>from h₂ = 0</td><td>0 × 0.88 = 0</td><td>0 × (−0.88) = 0</td></tr>
            </tbody>
          </table>
        </div>
        <p><b>Inputs.</b> Each input collects blame from every output it fed, through the weight it used: d_h₁ = 0.88×0.5 + (−0.88)×(−0.5) = 0.88, and d_h₂ = 0.88×0.5 + (−0.88)×1.0 = −0.44. That is row i of W, dotted with d_out. In matrix form: <code>d_out @ W2.T</code>.</p>
        <p><b>Bias.</b> A bias is added straight onto its output, so it gets that output’s blame unchanged: d_b = [0.88, −0.88].</p>
        <p>These are exactly steps 2 and 3 of the hand-worked table. Nothing new has happened yet.</p>
        <h3>Second pass: a whole batch at once</h3>
        <p>Now stack many examples as the rows of X, and their messages as the rows of d_out. Each example contributes its own outer product, and the weight gradient is their sum. <code>X.T @ d_out</code> is precisely “the outer product for every example, added up”. The bias gradient likewise adds up the blame over all rows. The input rule works row by row, as before.</p>
        <p>That gives the full list, exactly as it is written at the top of the repository file:</p>
        <Code source="phase1-foundations/mlp_numpy.py" title="the three rules (module docstring)">{`
linear:   d_W = X.T @ d_out ;  d_X = d_out @ W.T ;  d_b = sum(d_out)
relu:     d_x = d_out * (x > 0)
softmax+cross-entropy at the logits:  d_logits = probs - one_hot
`}</Code>
        <Equation
          label="Backward rules for a linear layer out = X W + b"
          symbols={[
            [<>d<sub>out</sub></>, 'the message from downstream: sensitivity of the loss to each output of this layer'],
            ['X', 'what the layer received in the forward pass (one row per example). It had to be remembered.'],
            [<>X<sup>T</sup> d<sub>out</sub></>, 'for each weight: (input that flowed through it) × (blame at the output it fed), summed over the examples in the batch'],
            [<>d<sub>out</sub> W<sup>T</sup></>, 'the message for the previous layer: each input collects blame from every output it fed, through the same weight it used going forward'],
            [<>Σ d<sub>out</sub></>, 'a bias is added straight onto the output, so its sensitivity is 1: it collects the blame unchanged, summed over the batch'],
          ]}
        >
          d<sub>W</sub> = X<sup>T</sup> d<sub>out</sub> &nbsp;&nbsp;&nbsp; d<sub>X</sub> = d<sub>out</sub> W<sup>T</sup> &nbsp;&nbsp;&nbsp; d<sub>b</sub> = Σ d<sub>out</sub>
        </Equation>
        <p>The <a href="#/lesson/matrices">transpose</a> is not a trick. Going forward, W carries you from inputs to outputs. Going backward, you travel the same wires in the opposite direction, from outputs back to inputs.</p>
        <p>Flipping rows and columns is what “the same wires, reversed” looks like when the wires are written as a matrix. You already did it by hand, entry by entry, in step 3 of the table above.</p>
        <Equation
          label="Backward rule for ReLU"
          symbols={[
            ['x', 'the ReLU’s input in the forward pass'],
            ['(x > 0)', '1 where the hinge was open, 0 where it was shut'],
            ['⊙', 'multiply entry by entry'],
          ]}
        >
          d<sub>x</sub> = d<sub>out</sub> ⊙ (x &gt; 0)
        </Equation>
        <p>ReLU’s slope is 1 on the open side and 0 on the flat side. The chain rule says multiply by the local slope. So: multiply by 1 or by 0.</p>
        <DeepDive title="Why does softmax + cross-entropy collapse to probs − one_hot?">
          <p>Write the loss for correct class c in terms of the logits z: L = −log( e<sup>z_c</sup> / Σ e<sup>z_j</sup> ) = −z<sub>c</sub> + log Σ e<sup>z_j</sup>.</p>
          <p>Nudge a logit z<sub>k</sub>. The first term changes by −1 if k is the correct class, otherwise by 0: that is −one_hot. The second term, log of a sum, changes by e<sup>z_k</sup> / Σ e<sup>z_j</sup>, which is exactly the softmax probability p<sub>k</sub>. Add the two: p<sub>k</sub> − one_hot<sub>k</sub>.</p>
          <p>The log in cross-entropy undoes the exponential in softmax. That is why the pair is always used together, and why frameworks fuse them into one operation (it is also numerically safer).</p>
        </DeepDive>
        <DeepDive title="Why backward, and not forward?">
          <p>You can also push sensitivities forward through the network: “if I nudge this one weight, how does every later value move?” That gives the effect of <em>one input on all outputs</em>, per sweep.</p>
          <p>Training has the opposite shape: millions of inputs (the weights) and <em>one</em> output (the loss). Going backward from that single number gives its sensitivity to everything in one sweep.</p>
          <p>With many outputs and few inputs, forward mode would be the cheaper one.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>The forward pass from last lesson saved each layer’s output in <code>self.cache</code>. Now you can see why: the rule <code>d_W = X.T @ d_out</code> needs X, the input each layer saw.</p>
        <Code title="Step 1: start at the loss. The gradient is the miss.">{`
probs = softmax(logits)
d = probs                        # d = blame flowing backward
d[np.arange(n), y] -= 1          # subtract 1 at the correct class: probs - one_hot
d /= n                           # the loss is a mean over n examples
`}</Code>
        <Code title="Step 2: one linear layer, backward (rule 1)">{`
h_in = self.cache[i]             # what this layer consumed
d_W[i] = h_in.T @ d              # blame -> this layer's weights
d_b[i] = d.sum(axis=0)
d = d @ self.W[i].T              # blame -> this layer's input
`}</Code>
        <Code title="Step 3: the hinge in front of it (rule 2)">{`
d = d * (self.cache[i] > 0)      # pass blame only where the ReLU was open
`}</Code>
        <p>A detail: <code>self.cache[i]</code> holds the ReLU’s <em>output</em>, not its input. That is fine: the output is positive exactly where the input was positive.</p>
        <p>Loop over the layers in reverse, and that is the whole method, unchanged from the repository:</p>
        <Code source="phase1-foundations/mlp_numpy.py" title="MLP.backward">{`
def backward(self, logits, y):
    n = len(y)
    d_W = [None] * len(self.W)
    d_b = [None] * len(self.b)

    probs = softmax(logits)
    d = probs                                   # d = blame flowing backward
    d[np.arange(n), y] -= 1                     # d_logits = probs - one_hot
    d /= n

    for i in reversed(range(len(self.W))):
        h_in = self.cache[i]                    # what this layer consumed
        d_W[i] = h_in.T @ d                     # rule 1: blame -> weights
        d_b[i] = d.sum(axis=0)
        if i > 0:
            d = d @ self.W[i].T                 # rule 1: blame -> layer input
            d = d * (self.cache[i] > 0)         # rule 2: ReLU gate
    return d_W, d_b
`}</Code>
        <p>The training loop is the one from <a href="#/lesson/gradient-descent">Part 2</a>, with two lines swapped in:</p>
        <Code source="phase1-foundations/mlp_numpy.py" title="train_mlp: forward, loss, backward, update">{`
for step in range(steps):
    logits = net.forward(X)                      # forward
    loss = cross_entropy(softmax(logits), y)     # loss
    d_W, d_b = net.backward(logits, y)           # backward: all gradients
    net.step(d_W, d_b, lr)                       # update: W -= lr * d_W
`}</Code>
        <h3>Trust, but verify: the gradient check</h3>
        <p>Hand-written backward code is easy to get subtly wrong, and a wrong gradient often still trains, only badly. So we test it against the slow method that is hard to get wrong: nudging. This is the <b>gradient check</b> promised in <a href="#/lesson/derivatives">lesson 1.4</a>, the black-box nudge experiment used as a unit test for the white-box chain rule.</p>
        <Code source="phase1-foundations/mlp_numpy.py" title="gradient_check (printing removed)">{`
logits = net.forward(X)
d_W, _ = net.backward(logits, y)

orig = net.W[li][r, c]
net.W[li][r, c] = orig + h                  # nudge up
lp = net.loss(X, y)
net.W[li][r, c] = orig - h                  # nudge down
lm = net.loss(X, y)
net.W[li][r, c] = orig                      # restore

numerical = (lp - lm) / (2 * h)             # centered difference
analytic = d_W[li][r, c]
diff = abs(numerical - analytic)
`}</Code>
        <p>Run the file: for 8 randomly chosen weights, backprop and nudging agree to about 12 decimal places (worst difference 1.22e-12). That turns “I believe the three rules” into “I verified them”.</p>
        <Callout kind="dev">
          In PyTorch you will never write <code>backward</code> yourself. Every operation (matmul, relu, softmax…) ships with its own local backward rule. During the forward pass PyTorch records which operations ran and keeps their inputs. <code>loss.backward()</code> then walks that record in reverse, applying the rules, exactly like the loop above. That system is called autograd. From <a href="#/lesson/build-gpt">Part 7</a> on we rely on it, and you will know what it is doing.
        </Callout>
        <Callout kind="established">
          <b>Why training needs so much memory.</b> The backward pass needs every layer’s forward values, so they all stay in memory until it has run.
          <br /><br />
          Just using a model (inference) has no backward pass, so it keeps no activations <em>for gradients</em>. An LLM at inference does keep something: the keys and values of earlier tokens, the <G t="kv-cache">KV cache</G>, so it does not redo work for every new token. That is a different cache for a different reason, and you will meet it in <a href="#/lesson/inference">Inference</a>.
          <br /><br />
          Add the gradients themselves (one number per parameter) and the optimiser’s bookkeeping, and training a model takes several times the memory of running it.
        </Callout>
      </CodeIt>

      <BreakIt>
        <p>In the playground, predict first, then check.</p>
        <ul>
          <li><b>Switch the correct class to class 1.</b> The model already gives it 88%. What happens to the size of every gradient? (d_logits shrinks from ±0.88 to ∓0.12, and everything upstream shrinks with it. Small miss, small correction.)</li>
          <li><b>Plant the bug</b> (“forget the ReLU gate”), then nudge-check the weight x2 → h2. Backprop claims −0.88. Nudging says 0. Who is right, and why? (Nudging. That hinge is shut, so the weight cannot affect the loss.)</li>
          <li><b>Keep training.</b> With learning rate 0.1, repeat Backward and Apply a few times. The loss goes 2.13 → 1.16 → 0.81 → 0.65. Why does each step help less than the one before? (The gradient is the miss. As the miss shrinks, every gradient shrinks, so the steps get smaller by themselves.)</li>
          <li><b>A step that is too big.</b> Reset, run both passes, set the learning rate to 0.5 and apply. The loss falls to 0.35, which looks great. Now look at h₁ and run Backward again. (Both hinges are now shut. Every weight gradient is 0 and only the biases can still learn. One oversized step switched the hidden layer off for this input. Practitioners call this a “dead ReLU”.)</li>
          <li><b>Open the shut hinge.</b> Edit W1 (x1 → h2) from −1 to 1 and run both passes again. Gradients now appear on all eight weights.</li>
          <li><b>In the Python file</b>, change <code>h_in.T @ d</code> to <code>h_in.T @ (2 * d)</code> and run it. Training still works (it is like doubling the learning rate for the weights), but the gradient check reports FAIL. Only the check catches this kind of bug.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="backprop-code-linear-backward" />
        <CodeExercise id="backprop-code-two-layer" />
        <Exercise
          id="backprop-calc"
          type="calculate"
          title="The gradient is the miss"
          answer={{ value: -0.76, tolerance: 0.011 }}
          answerLabel="gradient at the logit of class B"
          hints={[
            'First softmax: e^2.0 = 7.39, e^1.0 = 2.72, e^0 = 1. Divide each by the total.',
            'Total = 11.11, so probs = [0.67, 0.24, 0.09]. The one_hot vector for class B is [0, 1, 0].',
            'd_logits = probs − one_hot. For class B: 0.24 − 1.',
          ]}
          solution={<><p>probs = [0.67, 0.24, 0.09]. one_hot = [0, 1, 0]. d_logits = [0.67, <b>−0.76</b>, 0.09].</p><p>The correct class has a negative gradient, and the update subtracts the gradient, so its logit goes <em>up</em>. Class A, which wrongly got 67%, is pushed down the hardest. The three numbers sum to zero: probability is only being moved around.</p></>}
        >
          <p>A classifier outputs logits <code>[2.0, 1.0, 0.0]</code> for classes A, B, C. The correct class is <b>B</b>. What is the gradient of the cross-entropy loss with respect to the logit of class B? (Two decimals.)</p>
        </Exercise>

        <Exercise
          id="backprop-trace"
          type="trace"
          title="Follow one weight’s blame by hand"
          answer={{ value: -0.44, tolerance: 0.011 }}
          answerLabel="d_h₂"
          hints={[
            'Reset the playground but do not press Backward yet. h₂ feeds logit 1 through weight 0.5 and logit 2 through weight 1.0.',
            'The blame at the logits is d_logits = [0.88, −0.88]. Blame travels back through the same weights and adds up over both paths.',
            'd_h₂ = 0.88 × 0.5 + (−0.88) × 1.0.',
          ]}
          solution={<><p>d_h₂ = 0.88 × 0.5 + (−0.88) × 1.0 = 0.44 − 0.88 = <b>−0.44</b>.</p><p>Along a path, sensitivities multiply. Where several paths meet, they add. That is the chain rule for a network, and it is what <code>d @ W.T</code> computes for all units at once.</p><p>Then the ReLU gate: h₂’s hinge was shut (z = −0.5), so the −0.44 is multiplied by 0 and nothing reaches the weights feeding h₂. In words: “the loss would like h₂ to be bigger, but no small change to its incoming weights can make that happen right now.”</p></>}
        >
          <p>Use the starting network of the playground (x = [1, 2], correct class 2). By hand, compute the blame arriving at the <em>output</em> of hidden unit 2, before the ReLU gate. Then check with the Backward button. What happens to that blame at the gate?</p>
        </Exercise>

        <Exercise
          id="backprop-debug"
          type="debug"
          title="It trains, but badly"
          hints={[
            'Compare with the three rules. Which one is missing?',
            'The forward pass has two kinds of operation in each hidden layer. The backward loop only undoes one of them.',
          ]}
          solution={<><p>The ReLU gate is missing: after <code>d = d @ self.W[i].T</code> there must be <code>d = d * (self.cache[i] &gt; 0)</code>. Without it, blame flows through hinges that were shut, so weights get corrected for outputs they never influenced.</p><p>The nasty part: nothing crashes and the loss still goes down, because many of the gradients are still roughly right. With the repository’s settings the accuracy ends at 66.3% instead of 98.7%. You could spend days tuning the learning rate. The gradient check finds it in one second: it reports FAIL.</p></>}
        >
          <p>A teammate’s version of <code>backward</code> runs without errors. The spiral network trains, but gets stuck around 66% instead of 98.7%. What is wrong?</p>
          <Code>{`
for i in reversed(range(len(self.W))):
    h_in = self.cache[i]
    d_W[i] = h_in.T @ d
    d_b[i] = d.sum(axis=0)
    if i > 0:
        d = d @ self.W[i].T
return d_W, d_b
`}</Code>
        </Exercise>

        <Exercise
          id="backprop-implement"
          type="implement"
          title="Run it, shrink it, un-bend it"
          hints={[
            'Run python phase1-foundations/mlp_numpy.py and note the three sections of output: linear model, MLP, gradient check.',
            'Hidden size: in train_mlp(), change net = MLP() to net = MLP(sizes=(2, 4, 4, 3)), then (2, 8, 8, 3).',
            'Removing ReLU means changing both passes, so that they still describe the same network: in forward, drop np.maximum(0, …); in backward, delete the “rule 2: ReLU gate” line.',
          ]}
          solution={<><p>With the file’s fixed seed, on the training points: hidden size 4 reaches <b>72.0%</b>, hidden size 8 reaches 99.0%, and the original 64 reaches 98.7%. Eight hinges per layer are already enough for this spiral.</p><p>Why does 4 stall at 72%? Two of its four second-layer units go dead early (one is shut for every point from the start). Their blame is multiplied by 0 at the ReLU gate, so they never recover, and the network is left with two working hinges in that layer. You met this in the playground as a “dead ReLU”.</p><p>Without ReLU (both passes changed): <b>54.0%</b>, identical to the linear model, and the gradient check still passes. Your backward pass is correct; the <em>model</em> is a straight-line classifier again. If you remove the ReLU only in <code>forward</code>, the backward pass computes gradients for a different network than the one that ran: the loss explodes and accuracy falls to 33%. Forward and backward must always mirror each other.</p></>}
        >
          <p>Open <code>mlp_numpy.py</code>. (1) Run it unchanged and write down the linear and MLP accuracies. (2) Try hidden sizes 4 and 8. (3) Remove the ReLU and report the accuracy. Predict each result before you run it.</p>
        </Exercise>

        <ExplainBack
          id="backprop-explain"
          prompt="A teammate says: “Backpropagation is the algorithm neural networks use to learn.” That is not quite right. Explain what backpropagation actually computes, why it goes backward, and what does the learning."
          modelAnswer={<p>Learning is done by gradient descent: every weight takes a small step against its gradient. Backpropagation is only the method that computes those gradients. It starts at the loss, where the sensitivity is easy to write down (for a classifier: probs − one_hot), and passes that sensitivity backward through each layer using the layer’s local rule, which is the chain rule. Going backward means every intermediate result is computed once and reused by all the layers before it, so one backward sweep yields the gradient of every weight. The alternative, nudging each weight and re-running the network, would need one forward pass per weight. The price is memory: the forward pass must keep its intermediate values until the backward pass has used them.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'Why not train a large network with numerical gradients (nudge each weight, re-measure the loss)?',
            options: ['They are too inaccurate to learn from', 'They need one forward pass per parameter per step: billions of passes for a single update', 'They only work for linear models', 'They require calculus that nobody can do'],
            answer: 1,
            explain: 'They are accurate enough and need no calculus at all. They are hopelessly slow, which makes them a perfect test oracle and a useless training method.',
          },
          {
            q: 'A model predicts [0.2, 0.7, 0.1] and the correct class is the first. The gradient at the logits is…',
            options: ['[0.2, 0.7, 0.1]', '[−0.8, 0.7, 0.1]', '[0.8, −0.7, −0.1]', '[1, 0, 0]'],
            answer: 1,
            explain: 'probs − one_hot = [0.2 − 1, 0.7 − 0, 0.1 − 0]. Negative for the class whose score must rise.',
          },
          {
            q: 'During the backward pass, what happens to blame arriving at a ReLU whose input was negative in the forward pass?',
            options: ['It passes through unchanged', 'It is multiplied by 0: a shut hinge cannot have influenced the loss', 'Its sign is flipped', 'It is split equally among the inputs'],
            answer: 1,
            explain: 'The slope of ReLU on its flat side is 0. The chain rule multiplies by the local slope.',
          },
          {
            q: 'Why does the forward pass store (“cache”) every layer’s activations during training?',
            options: ['To print them for debugging', 'So the next forward pass can skip work', 'Because the backward rules need them: d_W = X.T @ d_out uses the input X each layer saw', 'Because Python cannot free memory inside a loop'],
            answer: 2,
            explain: 'This is the memory cost of training. At inference time there is no backward pass, so no activations need to be kept for gradients. (An LLM still keeps its KV cache at inference, but that is for speed, not for gradients.)',
          },
          {
            q: 'What does PyTorch’s loss.backward() do?',
            options: ['Updates the weights', 'Runs the model in reverse to reconstruct the input', 'Applies each recorded operation’s local backward rule in reverse order, leaving a gradient on every parameter', 'Estimates gradients by nudging each parameter'],
            answer: 2,
            explain: 'It is the loop you just read, automated and generalised to more operations. Updating the weights is a separate call (optimizer.step()).',
          },
        ]}
      />

      <Remember
        items={[
          <>Backpropagation answers “which weights are to blame, and by how much?” for <b>every weight in one backward sweep</b>, by running the chain rule from the loss toward the input and reusing intermediate results.</>,
          <>It only <em>measures</em> gradients. Learning is still <code>W -= lr * d_W</code>.</>,
          <>Three local rules cover our whole network. Linear: <code>d_W = X.T @ d_out</code>, <code>d_X = d_out @ W.T</code>. ReLU: pass blame where the hinge was open. Softmax + cross-entropy: <code>d_logits = probs − one_hot</code>, <b>the gradient is the miss</b>.</>,
          <>The backward pass needs the forward pass’s values, so training <b>caches activations</b>. That is why training uses far more memory than inference.</>,
          <>Always <b>gradient-check</b> hand-written backward code against nudging. In PyTorch, <code>loss.backward()</code> does all of this for you.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'This lesson', sub: '3 rules, by hand' }, { label: 'Autograd', sub: 'same rules, automated' }, { label: 'Training GPT', sub: 'Part 7' }, { label: 'Modern LLM' }]} active={0} />
        <ToyVsReal
          toy={<ul><li>Three backward rules, written by hand in NumPy</li><li>4,547 parameters, 300 data points, the whole dataset in every step</li><li>Classes: 3 spiral arms</li><li>Activations cached in a Python list</li></ul>}
          real={<ul><li>The same rules plus a few dozen more (attention, normalisation, embeddings…), generated by an autograd system</li><li>Billions of parameters, mini-batches of text, many GPUs each computing part of the gradient</li><li>Classes: every token in the vocabulary, tens of thousands of them. The backward pass still starts with probs − one_hot, at every position in the text</li><li>Cached activations are a main limit on batch size and context length. A common workaround is to store only some of them and recompute the rest during the backward pass, trading time for memory</li></ul>}
        />
        <Callout kind="established">Every neural language model is trained with backpropagation plus a gradient-descent-style update. When you read that training a model took thousands of GPUs for months, this is what those GPUs were doing: forward pass, backward pass, update, on batch after batch of text.</Callout>
        <Callout kind="model">“Blame” is a convenient word for a partial derivative. A gradient tells you how the loss would change for a <em>tiny</em> nudge to one weight, with everything else held fixed. It does not tell you what a weight “means” or what would happen after a large change.</Callout>
        <p>That evening Riya writes the post-mortem for the refund bug. Out of habit she lists the services in reverse order, from the symptom back to the cause. She notices, smiles, and leaves it that way.</p>
      </RealLLM>

      <BeforeMovingOn
        id="part-3"
        intro="You now hold every mechanical piece of how a neural network learns. Before we feed it language, pull the earlier ideas back out of memory. These questions reach back to Parts 0 to 3 on purpose."
        questions={[
          {
            q: 'Part 0: an LLM answers a prompt by…',
            options: ['retrieving the closest stored answer', 'predicting a probability for every possible next token, picking one, appending it, and repeating', 'parsing the question into a database query', 'generating the whole answer in one forward pass'],
            answer: 1,
            explain: 'One token at a time. Each of those predictions is the output of a forward pass like the one in this part, through a much larger network.',
          },
          {
            q: 'Part 1: a layer has weight matrix W of shape (64, 3) and receives a batch X of shape (300, 64). What is the shape of X @ W?',
            options: ['(64, 64)', '(300, 3)', '(3, 300)', 'The shapes do not fit'],
            answer: 1,
            explain: 'Inner sizes must match (64 and 64); the result takes the outer sizes. 300 examples, 3 scores each.',
          },
          {
            q: 'Part 1: softmax turns the logits [1, −1] into about [0.88, 0.12]. What happens if you add 10 to both logits?',
            options: ['The first probability gets closer to 1', 'Both probabilities become equal', 'Nothing changes: softmax only depends on the differences between logits', 'The result no longer sums to 1'],
            answer: 2,
            explain: 'e^10 appears in every term and cancels. Subtracting the maximum before exp(), as the repository’s softmax does, relies on exactly this.',
          },
          {
            q: 'Part 2: the loss explodes to huge values within a few steps. The most likely cause is…',
            options: ['too little data', 'a learning rate that is too large, so each step overshoots further than the last', 'a learning rate that is too small', 'using mini-batches'],
            answer: 1,
            explain: 'A slope is only valid near where it was measured. Large steps leave that region.',
          },
          {
            q: 'Part 3: you stack three layers but forget the activations. The network can represent…',
            options: ['any curve, given enough neurons', 'only what one linear layer can: the matrices multiply into one', 'nothing', 'piecewise-straight curves with three bends'],
            answer: 1,
            explain: 'W3(W2(W1 x)) = (W3 W2 W1) x. The hinge between the layers is what makes depth count.',
          },
          {
            q: 'Part 3: which statement about backpropagation is correct?',
            options: ['It replaces gradient descent', 'It computes the gradients that gradient descent then uses', 'It only works for ReLU networks', 'It estimates gradients by random nudges'],
            answer: 1,
            explain: 'Backpropagation measures; gradient descent moves. Both are needed on every training step.',
          },
        ]}
      >
        <OrderExercise
          id="backprop-order-training-loop"
          title="Rebuild the training loop from memory"
          prompt={<p>Put one training step in order. This is the loop that trains everything from the spiral classifier to GPT.</p>}
          correct={[
            'Take a batch of examples',
            'Forward pass: compute predictions, caching activations',
            'Loss: one number for how wrong the predictions are',
            'Backward pass: gradient of the loss for every parameter',
            'Update: every parameter takes a small step against its gradient',
            'Repeat with the next batch',
          ]}
          solutionNote={<p>The loss needs predictions, so forward comes first. The backward pass starts from the loss and needs the cached activations. The update needs the gradients. Then the parameters have changed, so every cached value is stale and the next step starts with a fresh forward pass.</p>}
        />
      </BeforeMovingOn>
    </Lesson>
  )
}
