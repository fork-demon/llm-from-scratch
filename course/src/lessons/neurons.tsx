import { Diagnostic } from '../components/Diagnostic'
import { DIAGNOSTICS } from '../data/diagnostics'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { CodeExercise } from '../components/python'
import { Exercise, ExplainBack } from '../components/exercise'
import { NeuronPlayground } from '../interactive/NeuronPlayground'
import { NeuronAnatomy } from '../illustrations/NeuronAnatomy'
import { LayerStack } from '../illustrations/LayerStack'

export default function NeuronsLesson() {
  return (
    <Lesson id="neurons">
      <Why>
        <Diagnostic id="diag-neural-nets" part={DIAGNOSTICS["neural-nets"].part} questions={DIAGNOSTICS["neural-nets"].questions} />
        <p className="lede">Last lesson ended at a wall.</p>
        <p>Gradient descent found the best line through the data. But if the data follows a curve, the best line is still a bad answer. Training harder does not help: the model <em>cannot express</em> the shape.</p>
        <p>Almost nothing interesting is a straight line. Whether an email is spam, which word comes next, which of three interleaved spiral arms a point lies on (the test data for this lesson): none of these is “output goes up steadily as input goes up”.</p>
        <Callout kind="idea">
          We need a model that can bend, and whose bends can be <em>learned</em> by the same gradient descent loop. That is all a neural network is:
          <br /><br />
          <b>A neural network is a collection of adjustable functions whose parameters are learned from examples.</b>
        </Callout>
        <p>This lesson builds one from parts you already own: the <a href="#/lesson/vectors">dot product</a> and the <a href="#/lesson/matrices">matrix multiply</a>. There is exactly one new ingredient, and it is one line of code.</p>
      </Why>

      <Problem title="The problem: a line cannot bend, and stacking lines does not help">
        <p>Your first idea might be: if one linear step is too simple, do two in a row. Multiply by a matrix W1, then by another matrix W2. Surely two transformations are more powerful than one?</p>
        <p>Let’s check with real numbers. Take the input x = [1, 2] (as a column) and two small matrices:</p>
        <div className="table-scroll">
          <table className="plain">
            <tbody>
              <tr><td>W1</td><td className="mono">[[2, 0], [1, 1]]</td><td>W2</td><td className="mono">[[1, −1], [0, 3]]</td></tr>
              <tr><td>step 1: W1 · x</td><td className="mono">[2·1 + 0·2, 1·1 + 1·2] = [2, 3]</td><td>step 2: W2 · [2, 3]</td><td className="mono">[1·2 − 1·3, 0·2 + 3·3] = <b>[−1, 9]</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Now multiply the two matrices together <em>first</em>, once, ahead of time:</p>
        <div className="table-scroll">
          <table className="plain">
            <tbody>
              <tr><td>W2 · W1</td><td className="mono">[[1·2 − 1·1, 1·0 − 1·1], [0·2 + 3·1, 0·0 + 3·1]] = [[1, −1], [3, 3]]</td></tr>
              <tr><td>(W2 · W1) · x</td><td className="mono">[1·1 − 1·2, 3·1 + 3·2] = <b>[−1, 9]</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Same answer. And it will be the same for <em>every</em> x, because W2(W1 x) = (W2 W1) x is simply how matrix multiplication works. Two linear layers are one linear layer in disguise. So are a hundred.</p>
        <WhyExists
          problem="A linear model can only draw straight lines (or flat planes). Real patterns bend."
          naive="Stack several linear layers to get a more powerful model."
          fails="The matrices multiply together into one matrix. The stack computes nothing a single layer could not. Depth alone is fake depth."
          idea="Put a tiny non-linear function, a bend, after each layer. Now the layers cannot be merged, and every extra layer adds real expressive power."
          tradeoff="The loss surface is no longer one smooth bowl, and the gradient can no longer be derived on a napkin. Computing it efficiently needs a new algorithm: backpropagation, next lesson."
        />
      </Problem>

      <MentalModel>
        <p>Build it up in three steps.</p>
        <h3>1. One neuron: a weighted sum, then a bend</h3>
        <NeuronAnatomy />
        <p>The weighted sum is a dot product: multiply each input by its weight, add up. You have done this since <a href="#/lesson/vectors">lesson 1.1</a>. It asks “how strongly does the input match the pattern stored in my weights?”</p>
        <Term
          name="Weights and bias"
          plain={<>A <b>weight</b> says how much one input counts, and in which direction (negative = counts against). The <b>bias</b> is a fixed offset added at the end: it sets how easily the neuron switches on.</>}
          example={<>inputs [1, 2], weights [1.0, 0.5], bias 0: sum = 1×1.0 + 2×0.5 + 0 = 2.0.</>}
          formal={<>z = w · x + b. The weights and biases of all neurons are the network’s <G t="parameters">parameters</G>: the numbers gradient descent adjusts.</>}
        />
        <p>Then comes the new ingredient: the bend.</p>
        <Term
          name="Activation function, ReLU"
          plain={<>A simple fixed function applied to the weighted sum. The standard one, <b>ReLU</b>, says: if the number is negative, make it 0; otherwise leave it alone.</>}
          example={<>relu(2.0) = 2.0, relu(−0.5) = 0. Drawn as a graph it is a hinge: flat, then a straight slope, with one bend.</>}
          formal={<>relu(z) = max(0, z). “ReLU” stands for rectified linear unit. The activation has no parameters of its own.</>}
        />
        <p>The weight decides how steep the sloped side is and which way it faces. The bias slides the bend left or right. So a neuron is an <b>adjustable hinge</b>.</p>

        <h3>2. One layer: many neurons, one matrix multiply</h3>
        <p>A layer is just several neurons looking at the same inputs, each with its own weights. Several dot products with the same vector: that is exactly a <a href="#/lesson/matrices">matrix multiply</a>. Put each neuron’s weights in one column of a matrix W, and the whole layer is:</p>
        <div className="card center mono" style={{ fontSize: 16 }}>h = relu(x @ W + b)</div>
        <p>Multiply, shift, bend.</p>

        <h3>3. A network: layers, stacked</h3>
        <p>Feed the outputs of one layer in as the inputs of the next. The second layer builds hinges out of things that are <em>already bent</em>, so complexity compounds. The last layer usually has no activation: it just produces the final scores.</p>
        <LayerStack />
        <p>Running an input through all layers to get the output is called the <b>forward pass</b>. It is ordinary function evaluation, nothing more.</p>
        <Callout kind="analogy">
          Why do sums of hinges work? Think of drawing a circle with a ruler. Every piece is straight, but with enough short pieces nobody can tell. Each hinge lets the network’s output change direction once. Add enough of them, scaled and shifted, and you can trace any reasonable curve as closely as you like.
          <br /><br />
          Where it stops: the word “neuron” comes from a loose 1940s analogy with brain cells. A unit here is a dot product and a max(0, ·). It is arithmetic, not biology, and nothing in this course depends on how real neurons work.
        </Callout>
      </MentalModel>

      <TryIt title="Be the training algorithm: bend a line into a bump">
        <p>The network below has 1 input, 4 hidden ReLU units and 1 output. You set its 12 parameters by hand. Start with the bump. It needs three hinges.</p>
        <NeuronPlayground />
      </TryIt>

      <Numbers title="A forward pass, by hand">
        <p>A tiny network: 2 inputs, a hidden layer of 2 ReLU neurons, and 2 output scores. All biases are 0 to keep the arithmetic short. The input is x = [1, 2].</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>hidden layer</th><th>weights</th><th>weighted sum</th><th>after ReLU</th></tr></thead>
            <tbody>
              <tr><td>neuron 1</td><td className="mono">[1.0, 0.5]</td><td className="mono">1×1.0 + 2×0.5 = 2.0</td><td className="mono"><b>2.0</b></td></tr>
              <tr><td>neuron 2</td><td className="mono">[−1.0, 0.25]</td><td className="mono">1×(−1.0) + 2×0.25 = −0.5</td><td className="mono"><b>0</b> (hinge shut)</td></tr>
            </tbody>
          </table>
        </div>
        <p>The hidden layer outputs h = [2, 0]. That becomes the input to the output layer:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>output layer</th><th>weights</th><th>score (no ReLU on the last layer)</th></tr></thead>
            <tbody>
              <tr><td>score 1</td><td className="mono">[0.5, 0.5]</td><td className="mono">2×0.5 + 0×0.5 = <b>1.0</b></td></tr>
              <tr><td>score 2</td><td className="mono">[−0.5, 1.0]</td><td className="mono">2×(−0.5) + 0×1.0 = <b>−1.0</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>The network’s output is [1, −1]. If these are class scores (<G t="logits">logits</G>), <a href="#/lesson/softmax">softmax</a> turns them into probabilities: [0.88, 0.12].</p>
        <p>Notice neuron 2. For this input its hinge is shut, so its output weights (0.5 and 1.0) had no effect at all. For a different input it may be open. That is the non-linearity at work: <b>which parts of the network are active depends on the input</b>. We will meet this exact network again in the next lesson, where we work out how to improve its weights.</p>
      </Numbers>

      <TheMath>
        <p>One layer, for all its neurons at once:</p>
        <Equation
          label="One layer of a neural network"
          symbols={[
            ['x', 'the layer’s input: a row of n numbers'],
            ['W', 'the weight matrix, n rows by m columns. Column j holds the weights of neuron j'],
            ['x W', 'a matrix multiply: m dot products at once, one per neuron'],
            ['b', 'm biases, one per neuron'],
            ['relu', 'max(0, ·) applied to each of the m numbers separately'],
            ['h', 'the layer’s output: m numbers, the input to the next layer'],
          ]}
        >
          h = relu( x W + b )
        </Equation>
        <p>A network with two hidden layers is this, nested: scores = relu( relu( x W₁ + b₁ ) W₂ + b₂ ) W₃ + b₃. Remove the two relu’s and the three matrices collapse into one, as you saw above.</p>
        <p><b>What about sigmoid?</b> You will meet another activation in older material: the sigmoid, 1 / (1 + e<sup>−z</sup>). It squashes any number smoothly into the range 0 to 1, and early networks used it everywhere.</p>
        <p>Its weakness: for large positive or negative inputs it is almost flat, so a nudge to the input barely moves the output. By the <a href="#/lesson/derivatives">chain rule</a>, a nudge that crosses many layers gets multiplied by that small slope (never more than 0.25) at each one. Deep sigmoid networks therefore learned very slowly.</p>
        <p>ReLU’s slope is exactly 1 wherever the hinge is open, and it is cheaper to compute. Modern LLMs use smooth relatives of ReLU (GELU, or gated variants such as <G t="swiglu">SwiGLU</G>): the same hinge idea with a rounded corner.</p>
        <DeepDive title="“Any curve”? What is actually proven">
          <p>The universal approximation theorem says: a network with one hidden layer and enough units can approximate any continuous function on a bounded range as closely as you like. For ReLU units in one dimension you can see why in the playground: each hinge adds one corner to a piecewise-straight curve.</p>
          <p>What the theorem does <em>not</em> say: how many units you need (it can be astronomically many), or that gradient descent will find the right weights. In practice, depth helps a lot: deep networks represent many useful functions with far fewer units than a single wide layer would need. Why trained deep networks generalise as well as they do is still an active research topic.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>One layer is one line of NumPy. <code>X</code> holds one example per row, so the same line processes a whole batch at once.</p>
        <Code title="Step 1: multiply, shift">{`
z = X @ W + b            # (N, n_in) @ (n_in, n_out) -> (N, n_out)
`}</Code>
        <Code title="Step 2: bend">{`
h = np.maximum(0, z)     # ReLU: negatives become 0
`}</Code>
        <Code title="Step 3: repeat for every hidden layer; no bend on the last one">{`
h = X
for i in range(len(W) - 1):
    h = np.maximum(0, h @ W[i] + b[i])
logits = h @ W[-1] + b[-1]
`}</Code>
        <p>That is the forward pass of the repository’s network. Here is the real class. The only additions are the random starting weights and a <code>cache</code> that remembers each layer’s output (the next lesson needs those):</p>
        <Code source="phase1-foundations/mlp_numpy.py" title="class MLP: setup and forward pass">{`
class MLP:
    def __init__(self, sizes=(2, 64, 64, 3)):
        self.W = [0.1 * rng.normal(size=(a, b)) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X):
        self.cache = [X]                   # activations layer by layer
        h = X
        for i in range(len(self.W) - 1):
            h = np.maximum(0, h @ self.W[i] + self.b[i])   # linear + ReLU hinge
            self.cache.append(h)
        logits = h @ self.W[-1] + self.b[-1]               # last layer: no ReLU
        return logits
`}</Code>
        <p><code>sizes=(2, 64, 64, 3)</code> means: 2 inputs, two hidden layers of 64 neurons, 3 output scores. Count the parameters: (2×64 + 64) + (64×64 + 64) + (64×3 + 3) = <b>4,547</b> adjustable numbers.</p>
        <p>The file trains this network on a dataset built to defeat straight lines: three interleaved spiral arms, one per class. It first trains a purely linear classifier on the same data, for comparison. Actual output:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>model</th><th>accuracy on the spiral</th></tr></thead>
            <tbody>
              <tr><td>guessing (3 classes)</td><td className="mono">33%</td></tr>
              <tr><td>linear classifier, 300 steps</td><td className="mono"><b>54.0%</b></td></tr>
              <tr><td>MLP 2 → 64 → 64 → 3, 2,000 steps</td><td className="mono"><b>98.7%</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Same data, same loss, same gradient descent loop. The only difference is <code>np.maximum(0, …)</code>.</p>
        <Callout kind="dev">Why random starting weights and not zeros? If every neuron in a layer starts identical, every neuron receives an identical gradient, and they stay identical forever: 64 copies of one neuron. Small random values break the tie.</Callout>
      </CodeIt>

      <BreakIt>
        <p>In the playground, predict first, then check.</p>
        <ul>
          <li><b>Remove the activation</b>, then try to make the bump. What is the lowest error you can reach? (About 0.10, with a flat line at y = 0.25, for example one unit with w = 0, b = 1, v = 0.25 and the others at v = 0. No slider can do better: you are fitting a line to a bump.)</li>
          <li><b>Set a unit’s w to 0.</b> What does it contribute? (A constant: relu(b) times v. Without an input weight, the neuron ignores the input.)</li>
          <li><b>Make w negative.</b> The hinge flips to face left. The V shape needs one of each.</li>
          <li><b>Try the smooth bowl with 4 hinges.</b> You can get close, never exact. What would you need for a better fit? (More hinges: more hidden units.)</li>
          <li><b>In the Python file</b>, remove the bend: in <code>forward</code> replace <code>np.maximum(0, h @ self.W[i] + self.b[i])</code> by <code>h @ self.W[i] + self.b[i]</code>, and in <code>backward</code> delete the line marked <code># rule 2: ReLU gate</code> (the gradient code must describe the same network; next lesson explains that line). Accuracy drops from 98.7% to <b>54.0%</b>: exactly the linear model’s score. Three layers, 4,547 parameters, and it is a straight-line classifier.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="neurons-code-relu-layer" />
        <Exercise
          id="neurons-calc"
          type="calculate"
          title="One neuron by hand"
          answer={{ value: 0.5, tolerance: 0.001 }}
          answerLabel="the neuron’s output"
          hints={[
            'Weighted sum first: multiply each input by its weight, add them up, then add the bias.',
            '2×1.5 + (−1)×2 + 3×0.5 = 3 − 2 + 1.5 = 2.5. Now add the bias of −2.',
            'z = 0.5. It is positive, so ReLU leaves it alone.',
          ]}
          solution={<><p>z = 2×1.5 + (−1)×2 + 3×0.5 − 2 = 3 − 2 + 1.5 − 2 = 0.5. relu(0.5) = <b>0.5</b>.</p><p>Notice the role of the bias: without it the sum would be 2.5. A bias of −2 means “the evidence must exceed 2 before I switch on”. With a bias of −3 this neuron would output 0.</p></>}
        >
          <p>A ReLU neuron has weights <code>[1.5, 2, 0.5]</code> and bias <code>−2</code>. Its inputs are <code>[2, −1, 3]</code>. What does it output?</p>
        </Exercise>

        <Exercise
          id="neurons-predict"
          type="predict"
          title="Ten layers, no activation"
          hints={[
            'Write the network as W10 · ( … (W2 · (W1 · x)) … ). What can you do with a chain of matrix multiplications?',
            'Biases do not rescue it: a line plus an offset, fed into another line plus an offset, is still a line plus an offset.',
          ]}
          solution={<p>Exactly as powerful as a single linear layer. The ten matrices multiply together into one matrix (and the biases fold into one bias), so the network can only draw straight lines, however long you train it. On the spiral it would stay near 54%. It is also slower and harder to train than the one-layer version, so it is strictly worse.</p>}
        >
          <p>A colleague builds a 10-layer network for the spiral data but forgets every activation function. Predict: roughly what accuracy will it reach, and why?</p>
        </Exercise>

        <Exercise
          id="neurons-experiment"
          type="experiment"
          title="Build |x| from two hinges"
          hints={[
            'One hinge can only produce the right half: relu(x) is 0 for negative x and x for positive x.',
            'You need a mirrored hinge for the left half. What weight makes a hinge open toward the left?',
            'Unit A: w = 1, b = 0, v = 1. Unit B: w = −1, b = 0, v = 1. Switch the others off with v = 0.',
          ]}
          solution={<p>|x| = relu(x) + relu(−x). For positive x the first hinge is open and the second is shut; for negative x it is the other way round. Each unit is responsible for one region of the input and silent elsewhere. That division of labour, decided by which hinges are open, is what a ReLU network does at any size.</p>}
        >
          <p>In the playground choose the target “a V shape, |x|”. Reach an error of 0.000 using only two units (set the output weight v of the other two to 0). Do it before pressing “show one solution”.</p>
        </Exercise>

        <Exercise
          id="neurons-debug"
          type="debug"
          title="The classifier that cannot be confident"
          hints={[
            'Look at the last line. What range of values can logits take?',
            'ReLU makes every score ≥ 0. Think about what a classifier needs to say “definitely not class 2”.',
          ]}
          solution={<><p>There is a ReLU on the <em>output</em> layer. Scores can then never be negative, so the network cannot push a wrong class far below the others, and any score that would be negative gets a zero gradient and stops learning. The last layer must be plain linear: <code>logits = h @ W[-1] + b[-1]</code>, as in the repository.</p><p>Hidden layers bend. The output layer only reads off the result.</p></>}
        >
          <p>This forward pass runs without errors but the model trains poorly. What is wrong?</p>
          <Code>{`
h = X
for i in range(len(W)):
    h = np.maximum(0, h @ W[i] + b[i])
logits = h
`}</Code>
        </Exercise>

        <Exercise
          id="neurons-implement"
          type="implement"
          title="How many hinges does a spiral need?"
          hints={[
            'Run python phase1-foundations/mlp_numpy.py first and note the final accuracy.',
            'In train_mlp(), change net = MLP() to net = MLP(sizes=(2, 8, 8, 3)), then try (2, 2, 2, 3), (2, 4, 4, 3) and (2, 16, 16, 3).',
          ]}
          solution={<><p>With the file’s fixed random seed: width 2 stays at 33% (no better than guessing), width 4 reaches 72%, and widths 8, 16 and 64 all reach about 99%.</p><p>Two hinges per layer cannot carve out three interleaved arms, and with so few units an unlucky start can leave every hinge shut for all inputs, so nothing can learn. Around 8 units per layer there is enough capacity, and more adds little. The lesson: capacity (how many hinges) sets a ceiling on what training can achieve, just as “a line” set a ceiling last lesson.</p></>}
        >
          <p>Open <code>mlp_numpy.py</code> and change the hidden layer sizes. Try widths 2, 4, 8 and 16. Before each run, predict whether it will beat the linear model’s 54%.</p>
        </Exercise>

        <ExplainBack
          id="neurons-explain"
          prompt="Explain to a teammate why a neural network needs activation functions. Use the phrase “collapse into one matrix” and say what ReLU changes."
          modelAnswer={<p>Each layer without an activation is a matrix multiply plus an offset. Two of those in a row collapse into one matrix multiply plus one offset, so a deep network without activations can only compute what a single linear layer computes: straight lines and flat planes. ReLU puts a bend after each layer: negative values become zero. Now the layers cannot be merged, because which neurons are “on” depends on the input. Each neuron becomes an adjustable hinge, and sums of many hinges can follow curves. The weights that position the hinges are learned by gradient descent.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'What does a single neuron compute?',
            options: ['A lookup in a table of stored examples', 'A dot product of its inputs with its weights, plus a bias, passed through a simple bend', 'The average of its inputs', 'A probability that the input is correct'],
            answer: 1,
            explain: 'Weighted sum, shift, bend. Everything else in this part is that, repeated.',
          },
          {
            q: 'Why is a whole layer written as one matrix multiply?',
            options: ['Because matrices are required by Python', 'Because each neuron is a dot product with the same input, and a matrix multiply is many dot products at once', 'Because layers have no biases', 'It is only an approximation of what the neurons do'],
            answer: 1,
            explain: 'One column of W per neuron. This is also why GPUs, which are built for large matrix multiplies, run neural networks fast.',
          },
          {
            q: 'You stack 5 linear layers with no activation. Compared with 1 linear layer, the stack can represent…',
            options: ['5 times more complex functions', 'curves, but only gentle ones', 'exactly the same functions: the matrices multiply into one', 'nothing at all'],
            answer: 2,
            explain: 'W5·W4·W3·W2·W1 is one matrix. You verified W2(W1 x) = (W2 W1) x with numbers above.',
          },
          {
            q: 'In relu(w·x + b), what does changing the bias b do to the hinge?',
            options: ['Makes the slope steeper', 'Moves the position of the bend', 'Flips it to face the other way', 'Nothing: biases are only for the output layer'],
            answer: 1,
            explain: 'The bend sits where w·x + b = 0, that is at x = −b / w. The weight sets steepness and direction; the bias slides the bend.',
          },
          {
            q: 'Why did ReLU-style activations replace the sigmoid in deep networks?',
            options: ['Sigmoid cannot produce a bend, so stacked sigmoid layers collapse into one', 'ReLU outputs can be read directly as probabilities, sigmoid outputs cannot', 'Sigmoid is nearly flat for large inputs, so the training signal shrinks at every layer; ReLU keeps slope 1 where open', 'ReLU gives every neuron extra trainable parameters that sigmoid lacks'],
            answer: 2,
            explain: 'Both are non-linear, so both prevent the collapse. The difference is how well gradients survive a trip through many layers.',
          },
        ]}
      />

      <Remember
        items={[
          <>A <b>neuron</b> = dot product with its weights + bias, then an activation. A <b>layer</b> = many neurons = one matrix multiply: <code>h = relu(x @ W + b)</code>.</>,
          <>Without activations, stacked layers <b>collapse into one matrix</b>: W2(W1 x) = (W2 W1) x. Depth alone adds nothing.</>,
          <><b>ReLU</b>, max(0, z), is a hinge. The weight sets its steepness and direction, the bias moves the bend. Sums of hinges can approximate any curve.</>,
          <>The <b>forward pass</b> is just evaluating the layers in order. Hidden layers bend; the last layer outputs raw scores.</>,
          <>On the spiral: linear model <b>54%</b>, the same loop with ReLU layers <b>98.7%</b>. The weights are not designed. They are learned.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'This lesson', sub: '2 → 64 → 64 → 3' }, { label: 'Backpropagation', sub: 'next lesson' }, { label: 'Feed-forward block', sub: 'inside every Transformer block' }, { label: 'GPT' }]} active={0} />
        <p>Every Transformer block contains two parts: attention, and a <G t="ffn">feed-forward network</G>. The feed-forward network is exactly this lesson’s network with one hidden layer: widen each token’s vector from d numbers to 4d, bend, and project back down to d.</p>
        <ToyVsReal
          toy={<ul><li>2 → 64 → 64 → 3, about 4,500 parameters</li><li>ReLU</li><li>Input: a 2-D point. Output: 3 class scores</li><li>One network is the whole model</li></ul>}
          real={<ul><li>d → 4d → d inside every block, e.g. 768 → 3,072 → 768 in the smallest GPT-2: about 4.7 million parameters per block, in each of 12 blocks</li><li>GELU or a gated variant (SwiGLU): a hinge with a rounded corner</li><li>Input and output: one token’s vector, processed independently of the other tokens</li><li>One such network per block, alternating with attention</li></ul>}
        />
        <Callout kind="established">In a standard GPT-style block the feed-forward network holds about two thirds of the block’s weights (8d² versus 4d² for attention). So most of the parameters of a typical LLM sit in plain “multiply, shift, bend, multiply” layers like the one you just built.</Callout>
        <Callout kind="research">What those feed-forward parameters <em>do</em> is much less settled. There is experimental evidence that they play a large part in recalling facts, and you will often read “facts are stored in the feed-forward layers”. Treat that as a useful hypothesis with supporting experiments, not as a known mechanism. We come back to it in <a href="#/lesson/why-llms-know">Why LLMs know things</a>.</Callout>
      </RealLLM>
    </Lesson>
  )
}
