import { TrainedGptExplorer } from '../interactive/TrainedGptExplorer'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { ClaimSorter } from '../interactive/ClaimSorter'
import { SuperpositionLab } from '../interactive/SuperpositionLab'

const LENS: [string, string, string, string][] = [
  ['0 (embedding only)', '“ destro”', '62%', '29,758th'],
  ['3', '“ same”', '95%', '118th'],
  ['6', '“ same”', '75%', '4th (3.0%)'],
  ['7', '“ same”', '53%', '4th (6.3%)'],
  ['8', '“ floor”', '25%', '1st'],
  ['9', '“ ground”', '23%', '2nd (20.5%)'],
  ['10', '“ floor”', '46%', '1st'],
  ['12 (final output)', '“ floor”', '7.6%', '1st'],
]

export default function InterpretabilityLesson() {
  return (
    <Lesson id="interpretability">
      <Why>
        <p className="lede">Friday, 8 p.m. The office is almost empty. Riya has downloaded GPT-2 small, a real public model with three times as many layers as the GPT she built.</p>
        <p>She types “The cat sat on the” and it answers “floor”. Fine. But she wants more than the answer. She has printed the tensors: 768 numbers per token, after each of its 12 layers. Pages of them.</p>
        <p>Kabir stops on his way out, sees the wall of decimals and sits down. “I can see every number,” Riya says. “So why can’t I see what it’s thinking?”</p>
        <p>“Good question. Some of it we can read now. A lot of it we can’t. Let me show you which is which.”</p>
        <p>Every number inside a model is visible. The hard part is <b>translation</b>: finding which patterns in those numbers stand for which ideas, and proving that the model actually uses them. That field is called <b>interpretability</b>.</p>
      </Why>

      <Problem title="The problem: visible is not readable">
        <p>You met this in <a href="#/lesson/why-llms-know">Why LLMs know things</a>: there is no table of facts inside, only weights and <G t="activation">activations</G>. So the natural first plan is to look at the activations one neuron at a time and ask what each one means.</p>
        <p>Researchers did this for years. Some neurons are clean; many are not. One neuron in a small model studied by Anthropic in 2023 fired for academic citations, English dialogue, HTTP requests and Korean text. No single meaning fits its label.</p>
        <Term
          name="Polysemantic neuron"
          plain={<>A neuron that responds to several unrelated things. Its activation alone does not tell you which one is present.</>}
          example={<>The neuron above: citations, dialogue, web requests and Korean, all on one number.</>}
          formal={<>A unit whose activation is high for inputs from several semantically unrelated clusters. The opposite, a unit that responds to one concept, is called <em>monosemantic</em>.</>}
        />
        <p>Dev, listening from the door: “Why not ask the model? It explains its reasoning all the time.” Hold on to that. The end of this lesson shows why that is not a look inside.</p>
      </Problem>

      <MentalModel>
        <h3>1. The residual stream is a shared workspace</h3>
        <p>In a <a href="#/lesson/transformer-block">Transformer block</a>, every block <em>adds</em> its output to the token’s vector: <code>x = x + attn(x)</code>, then <code>x = x + mlp(x)</code>. That running vector is the <G t="residual">residual stream</G>.</p>
        <p>Think of it as a <b>shared bus</b>: no layer replaces it, each adds a message that later layers can read. The framing comes from Anthropic’s 2021 “Mathematical Framework for Transformer Circuits”, and the arithmetic is exact: the final vector is the embedding plus the sum of every block’s output.</p>

        <h3>2. The logit lens: read the workspace early</h3>
        <p>Since the stream is one shared vector, you can take it at <em>any</em> layer and push it through the model’s final norm and output matrix, as if the model stopped there. This trick (from a 2020 blog post) is the <b>logit lens</b>. Here is a real run on GPT-2 small (12 layers, 768 numbers per token) with “The cat sat on the”:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>after layer</th><th>top token</th><th>its probability</th><th>rank of “ floor”</th></tr></thead>
            <tbody>
              {LENS.map(([l, t, p, r]) => <tr key={l}><td>{l}</td><td>{t}</td><td>{p}</td><td>{r}</td></tr>)}
            </tbody>
          </table>
        </div>
        <p>Early layers give a confident, generic guess (“the same”). Around layers 6 to 8 the right kind of answer rises: “ floor”, then “ ground”, then “ floor” again. The last layer spreads its bets (floor 7.6%, bed 6.5%, couch 5.4%, ground 5.2%). You can watch the prediction form.</p>
        <p>The logit lens is an approximation, not a readout of intent: it assumes middle layers speak the output’s “language”. True enough for GPT-2, gibberish for some other models. The tuned lens (2023) learns a small translator per layer and is more reliable.</p>
        <p>Try it on the course’s tiny Shakespeare GPT: type a prompt and open the layer-by-layer table. Its head scores are here too: look for previous-token heads, and notice what this small model does <em>not</em> have.</p>
        <TrainedGptExplorer />

        <h3>3. Some attention heads have readable jobs</h3>
        <p>A few heads jump out. A <b>previous-token head</b> always looks one position back. An <b>induction head</b> is cleverer: if the text contains “Paisa Pal … Paisa”, it looks at what followed the earlier “Paisa” and pushes the model to predict “Pal” again.</p>
        <p>Induction heads (Olsson et al., Anthropic, 2022) are a previous-token head feeding an induction head. They appear fairly suddenly early in training, together with a jump in the model’s use of context. The evidence is strongest in small attention-only models, partly correlational in large ones. Most heads in a large model have no one-line job description.</p>

        <h3>4. Features and superposition</h3>
        <p>If single neurons are not the unit of meaning, what is? The current best answer is <b>directions</b>.</p>
        <Term
          name="Feature"
          plain={<>A direction in activation space that corresponds to one recognisable property of the input. The more the activation points that way, the more that property is present.</>}
          example={<>A “this text is in French” direction, or a “Golden Gate Bridge” direction, in the middle of a model.</>}
          formal={<>A vector d such that the projection a · d of an activation a tracks the presence of one concept, and that the model’s later computation uses.</>}
        />
        <p>The catch: a model seems to track far more features than it has dimensions. How do many directions fit into few numbers?</p>
        <Term
          name="Superposition"
          plain={<>Storing more features than there are dimensions, by giving them directions that are <em>almost</em> perpendicular. It works when features are rarely active at the same time, so their small overlaps rarely collide.</>}
          example={<>In the lab below: 5 features in 2 numbers, arranged as a pentagon.</>}
          formal={<>Elhage et al. (2022), “Toy Models of Superposition”: when features are sparse, a model with m dimensions learns to represent n &gt; m features, accepting some interference, and cleans the interference up with a ReLU and a negative bias.</>}
        />
        <p>Picture a Bengaluru bus with forty seats and sixty regulars: it works because they are never all on board at once. Where the picture stops: when two overlapping features <em>are</em> active together, each reading is a little wrong. That error is the price.</p>
        <Callout kind="established">
          Superposition in toy models is demonstrated and well understood; you will reproduce it in a minute. That large language models use superposition heavily is widely believed and fits much evidence (polysemantic neurons, the success of the dictionary methods below), but there is no complete proof for any large model.
        </Callout>

        <h3>5. Sparse autoencoders: un-mixing the features</h3>
        <p>Can we find those directions? A <b>sparse autoencoder</b> (SAE) is a second, small network trained on a model’s activations. It must rebuild each activation from a very large list of candidate directions, using only a few at a time. The directions it learns are called features.</p>
        <p>In “Towards Monosemanticity” (2023), Anthropic trained SAEs on a one-layer model with 512 MLP neurons, with dictionaries from 512 up to 131,072 features. Most features were far easier to interpret than the neurons: Arabic script, DNA sequences, base64, legal language. “Scaling Monosemanticity” (2024) used about 1, 4 and 34 million features on the middle of Claude 3 Sonnet, and found features for cities, people, code bugs, sycophantic praise and deception, many working across languages and in images.</p>
        <p>The famous demo: a Golden Gate Bridge feature was <b>clamped to a high value</b> during generation. The model, released for a day as “Golden Gate Claude”, brought the bridge into almost every answer and even described itself as the bridge. Turning a feature up changed behaviour: evidence that the model really uses that direction.</p>
        <Callout kind="research">
          SAEs are powerful and imperfect. They do not reconstruct activations exactly, and the missing part can matter. The features depend on dictionary size (a “bird” feature may split into several). Some teams found a plain linear probe (next section) beat SAE features for tasks such as detecting harmful intent. Whether SAE features are the model’s “true” units or a convenient basis is debated.
        </Callout>

        <h3>6. Testing by intervening: patching, probes, steering</h3>
        <p>A feature that lights up for bridges might not be <em>used</em>. Three families of tools test this.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>Tool</th><th>How it works</th><th>Best-known result</th></tr></thead>
            <tbody>
              <tr><td>Activation patching</td><td>Run a “clean” prompt and a “corrupted” one that changes the answer. Copy one activation from the clean run into the corrupted run. If the right answer comes back, that activation carries the information.</td><td>A 2022 circuit of 26 attention heads in GPT-2 small that picks the right name in “When Mary and John went to the store, John gave a drink to …”.</td></tr>
              <tr><td>Probes</td><td>Train a small classifier (often one linear layer) to predict a property from the activations. Success shows the information is <em>present</em>, not that the model uses it.</td><td>A GPT trained only on Othello moves holds the board state in its activations.</td></tr>
              <tr><td>Steering vectors</td><td>Take the difference between activations for two contrasting prompts (say, “Love” and “Hate”) and add it during generation. The output shifts that way.</td><td>A 2024 study: refusing harmful requests in several chat models is largely one direction. Remove it and the model stops refusing.</td></tr>
            </tbody>
          </table>
        </div>

        <h3>7. Circuit tracing: following one answer through the model</h3>
        <p>In 2025 Anthropic published <b>attribution graphs</b>: replace a model’s MLPs with a more interpretable stand-in made of features (a “cross-layer transcoder”), then trace, for one prompt, which features pushed which others toward the final token.</p>
        <p>Asked for “the capital of the state containing Dallas”, Claude 3.5 Haiku showed an internal “Texas” step before “Austin”: a two-hop path, done inside one forward pass. Writing a rhyming couplet, it picked the rhyme word <em>before</em> writing the line that ends in it: evidence of planning ahead.</p>
        <Callout kind="research">
          These explain a <em>replacement</em> model that imitates the real one, with “error” terms for what it cannot capture. The authors report satisfying insight on only a fraction of the prompts they tried, and each graph takes hours of expert reading: a microscope, not an X-ray of everything. The tools were later open-sourced for open models.
        </Callout>
      </MentalModel>

      <TryIt title="Watch superposition happen">
        <p>The toy model from “Toy Models of Superposition”, trained live with the <G t="gradient-descent">gradient descent</G> you built in Part 2. Five features must squeeze through two numbers.</p>
        <ol>
          <li>Press <b>Dense</b>, then <b>Train</b>. How many arrows end up long?</li>
          <li>Press <b>Sparse</b>, then <b>Train</b>. Now how many? What shape?</li>
          <li>In the sparse model, use the probe: switch on f1 alone and look at which other outputs light up.</li>
        </ol>
        <SuperpositionLab />
        <p className="muted">A simplified model: it shows the <em>mechanism</em> (sparsity makes overlap worth it, ReLU and bias clean up the interference), not the scale of a real model.</p>
      </TryIt>

      <Numbers>
        <p>Let’s check the pentagon by hand. The trained sparse model (seed 1) has five arrows about <b>1.13</b> long, 72° apart, and every bias about <b>−0.25</b>. Switch on feature 1 alone with value 1.</p>
        <p>The hidden vector h is feature 1’s arrow. Output i is ReLU(arrow<sub>i</sub> · h + b<sub>i</sub>), and a dot product of two arrows is length × length × cos(angle between them).</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>output for…</th><th>angle to f1</th><th>1.13 × 1.13 × cos(angle)</th><th>+ bias −0.25</th><th>after ReLU</th></tr></thead>
            <tbody>
              <tr><td>f1 itself</td><td>0°</td><td>1.2769 × 1 = 1.28</td><td>1.03</td><td><b>1.03</b></td></tr>
              <tr><td>its two neighbours</td><td>72°</td><td>1.2769 × 0.309 = 0.39</td><td>0.14</td><td><b>0.14</b></td></tr>
              <tr><td>the two far ones</td><td>144°</td><td>1.2769 × (−0.809) = −1.03</td><td>−1.28</td><td><b>0</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Feature 1 comes back at 1.03. The far pair is cut to exactly zero by the ReLU. The neighbours leak 0.14 each: that is <b>interference</b>, the price of five features in two dimensions. The lab’s probe shows nearly the same values (1.06, 0.14, 0.14, 0, 0): the trained arrows are not perfectly equal, so feature 1 comes back a little higher.</p>
        <p>Without the negative bias the neighbour leak would be 0.39, nearly three times worse. The bias is what makes the overlap affordable.</p>
        <h3>Why sparsity decides</h3>
        <p>Measured loss on fresh data, same model size, same training:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>data</th><th>what training found</th><th>loss</th></tr></thead>
            <tbody>
              <tr><td>dense (S = 0)</td><td>2 features at 90°, other three dropped</td><td>0.184</td></tr>
              <tr><td>sparse (S = 0.95), seed 1</td><td>pentagon, all 5 kept</td><td>0.0082</td></tr>
              <tr><td>sparse (S = 0.95), seed 2</td><td>square: 4 kept, one dropped</td><td>0.0138</td></tr>
            </tbody>
          </table>
        </div>
        <p>When every feature is always on, any overlap interferes on every example, so the model keeps two clean directions and predicts the other three by their average (bias near 0.5). When features are rarely on together, keeping all five pays. Across 12 seeds at S = 0.95, 7 runs found the pentagon and 5 got stuck in the higher-loss square: gradient descent finds <em>a</em> good arrangement, not always the best.</p>
      </Numbers>

      <TheMath>
        <p>The toy model you trained, in three lines:</p>
        <Equation
          label="h equals W x; x prime equals ReLU of W transpose h plus b; loss is the importance-weighted squared error"
          symbols={[
            ['x', 'the 5 input features; each is 0 with probability S, otherwise a random number between 0 and 1'],
            ['W', 'a 2 × 5 matrix; column i is the arrow you see for feature i'],
            ['h', 'the 2 hidden numbers: all the model is allowed to keep'],
            [<>W<sup>T</sup>h + b</>, 'each feature is read back by a dot product with its own arrow, plus a bias'],
            ['ReLU', 'max(0, ·): cuts negative interference to exactly zero'],
            [<>I<sub>i</sub></>, <>importance of feature i: 0.9<sup>i</sup> here, so f1 matters most</>],
          ]}
        >
          h = W x &nbsp;&nbsp; x′ = ReLU(W<sup>T</sup> h + b) &nbsp;&nbsp; L = Σ<sub>i</sub> I<sub>i</sub> (x<sub>i</sub> − x′<sub>i</sub>)<sup>2</sup>
        </Equation>
        <p>A sparse autoencoder is the same shape turned inside out. The small thing is now the <em>model’s</em> activation a, and the SAE blows it <em>up</em> into many features, then rebuilds it:</p>
        <Equation
          label="f equals ReLU of W enc a plus b enc; a hat equals W dec f; loss is reconstruction error plus lambda times the sum of f"
          symbols={[
            ['a', 'one activation vector from the model (for example 768 numbers)'],
            ['f', 'feature activations: many more entries than a (for example 8 to 256 times more), mostly zero'],
            [<>W<sub>dec</sub></>, 'its columns are the feature directions: the dictionary'],
            [<>λ Σ f<sub>j</sub></>, 'a penalty on total feature activity: pushes the SAE to explain each activation with only a few features'],
          ]}
        >
          f = ReLU(W<sub>enc</sub> a + b<sub>enc</sub>) &nbsp;&nbsp; â = W<sub>dec</sub> f &nbsp;&nbsp; L = ‖a − â‖<sup>2</sup> + λ Σ<sub>j</sub> f<sub>j</sub>
        </Equation>
        <p>And the logit lens is one line: take the residual stream h<sub>ℓ</sub> after layer ℓ and apply the model’s own last two steps.</p>
        <Equation
          label="logits at layer l equal the final norm of h l times the unembedding matrix"
          symbols={[
            [<>h<sub>ℓ</sub></>, 'the residual stream after layer ℓ, at the last position'],
            [<>LN<sub>f</sub></>, <>the model’s final <G t="layernorm">LayerNorm</G></>],
            [<>W<sub>U</sub></>, <>the output (unembedding) matrix that turns a vector into <G t="logits">logits</G> over the vocabulary</>],
          ]}
        >
          logits<sub>ℓ</sub> = LN<sub>f</sub>(h<sub>ℓ</sub>) W<sub>U</sub>
        </Equation>
        <DeepDive title="How many almost-perpendicular directions fit?">
          <p>In 2 dimensions, not many: five arrows already overlap by cos 72° = 0.31. In high dimensions it is very different. Two random directions in 768 dimensions typically have a cosine of about 1/√768 ≈ 0.036: nearly perpendicular. The number of directions you can fit with overlaps below a small fixed threshold grows <em>exponentially</em> with the dimension (a consequence of the Johnson-Lindenstrauss lemma). That is why superposition is expected to be much more powerful in real models than in the toy.</p>
        </DeepDive>
      </TheMath>

      <CodeIt title="Let’s code it: the toy model and a real logit lens">
        <p>No repository file goes with this lesson, so here are short sketches. Both were run: the toy gives a pentagon (lengths 1.12 to 1.14, biases −0.23 to −0.25), and the logit lens printed the GPT-2 table above.</p>
        <Code
          title="The toy model: data and forward pass"
          show={`print("fraction of features that are on:", (x > 0).mean().round(3), " (S = 0.95, so about 0.05)")
print("x:", x.shape, " h:", h.shape, " out:", out.shape)
print("loss before any training:", loss.round(4))`}
        >{`
import numpy as np
rng = np.random.default_rng(1)
n, m, S, lr, B = 5, 2, 0.95, 0.1, 256
I = 0.9 ** np.arange(n)                  # importance: [1, 0.9, 0.81, ...]
W = rng.normal(0, 0.3, (m, n))           # column i = direction of feature i
b = np.zeros(n)

x = rng.random((B, n)) * (rng.random((B, n)) > S)   # sparse features
h = x @ W.T                     # (B, 2)   squeeze 5 numbers into 2
pre = h @ W + b                 # (B, 5)   read each feature back
out = np.maximum(pre, 0)        # ReLU
loss = np.mean(np.sum(I * (out - x) ** 2, axis=1))
`}</Code>
        <Code
          title="One step of plain gradient descent (repeat 6,000 times)"
          setup={`import numpy as np
rng = np.random.default_rng(1)
n, m, S, lr, B = 5, 2, 0.95, 0.1, 256
I = 0.9 ** np.arange(n)
W = rng.normal(0, 0.3, (m, n))
b = np.zeros(n)
x = rng.random((B, n)) * (rng.random((B, n)) > S)
h = x @ W.T
pre = h @ W + b
out = np.maximum(pre, 0)`}
          show={`for step in range(6000):             # the same forward pass and step, on fresh batches
    x = rng.random((B, n)) * (rng.random((B, n)) > S)
    h = x @ W.T; pre = h @ W + b; out = np.maximum(pre, 0)
    d = 2 / B * I * (out - x) * (pre > 0)
    W -= lr * (h.T @ d + (W @ d.T) @ x); b -= lr * d.sum(axis=0)
print("arrow lengths:", np.linalg.norm(W, axis=0).round(2))
print("biases:       ", b.round(2))
angles = np.sort(np.degrees(np.arctan2(W[1], W[0])))
print("gaps between neighbouring arrows (degrees):", np.diff(np.append(angles, angles[0] + 360)).round(0))`}
        >{`
d = 2 / B * I * (out - x) * (pre > 0)    # dLoss/dpre
dW = h.T @ d + (W @ d.T) @ x             # W is used twice: to squeeze and to read back
db = d.sum(axis=0)
W -= lr * dW
b -= lr * db
`}</Code>
        <p>The gradient is the <a href="#/lesson/backprop">chain rule</a> you already know. The only new thing is that W appears twice, so its gradient has two terms. The lab’s TypeScript does the same arithmetic, and its tests check the gradient against a numerical derivative.</p>
        <Code title="A real logit lens on GPT-2 (Hugging Face transformers)">{`
out = model(ids, output_hidden_states=True)   # 13 vectors: embeddings + 12 layers
for layer, h in enumerate(out.hidden_states):
    x = h[0, -1]                                # last position
    if layer < 12:                              # the last entry already has ln_f applied
        x = model.transformer.ln_f(x)
    probs = model.lm_head(x).softmax(-1)
    print(layer, tok.decode(probs.argmax()), round(probs.max().item(), 3))
`}</Code>
        <p>Interpretability code is mostly instrumentation: hooks that record or overwrite intermediate values, then a comparison. If you have added tracing to a service, or replayed one request with one field changed, you already know the shape of activation patching.</p>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check in the lab.</p>
        <ul>
          <li><b>Dense, importance decay 1.00.</b> All features equally important and always on. Which two win? (No clean answer: training ends in a messy in-between arrangement. Importance differences made the dense result tidy.)</li>
          <li><b>Seed 2, sparse.</b> A square: four features in two opposite pairs, one feature dropped. Compare the loss with seed 1. Same data, same model, worse answer: a local minimum.</li>
          <li><b>Seed 1, raise S slowly from 0.5.</b> At 0.80 you get a square with f3 missing; at 0.82, a pentagon. The switch is abrupt: the paper calls it a phase change.</li>
          <li><b>Importance decay 0.5, sparse.</b> The least important feature now costs more to keep than it is worth. Does the pentagon survive? (With seed 1, no: it falls back to a square.)</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="interpretability-dense"
          type="predict"
          title="How many survive?"
          answer={{ value: 2, tolerance: 0 }}
          answerLabel="features represented"
          hints={[
            'Dense means every feature is on in every example. Any two overlapping arrows interfere on every example.',
            'With 2 dimensions there is room for exactly 2 perpendicular directions with zero overlap.',
          ]}
          solution={<><p><b>2</b>: the two most important features, at 90° to each other. The other three get arrows of length close to zero and a bias near 0.5, their average value.</p><p>Overlapping would add interference to every single example, which costs more than it gains. Superposition only pays when features are rarely active together.</p></>}
        >
          <p>Before pressing Train: in the dense setting (S = 0, importance 0.9<sup>i</sup>), how many of the 5 features will end up with an arrow longer than 0.5?</p>
        </Exercise>

        <Exercise
          id="interpretability-leak"
          type="calculate"
          title="The price of a pentagon"
          answer={{ value: 0.14, tolerance: 0.02 }}
          answerLabel="leak into a neighbour"
          hints={[
            'Output for a neighbour = ReLU(length × length × cos(angle) + bias).',
            'length × length = 1.13 × 1.13 = 1.2769; cos 72° = 0.309.',
            '1.2769 × 0.309 = 0.395. Now add the bias −0.25.',
          ]}
          solution={<><p>1.2769 × 0.309 − 0.25 = 0.395 − 0.25 = <b>0.14</b> (positive, so the ReLU lets it through).</p><p>If only f1 is on, its two neighbours read 0.14 instead of 0. When features are sparse this rarely matters; when several are on together, these small errors add up. Interference is why superposition only pays under sparsity.</p></>}
        >
          <p>In the trained pentagon, every arrow is 1.13 long, neighbours are 72° apart, and every bias is −0.25. Feature 1 is switched on alone with value 1. What does the model output for one of feature 1’s two neighbours? (Two decimals.)</p>
        </Exercise>

        <Exercise
          id="interpretability-probe"
          type="debug"
          title="“We found the honesty direction”"
          hints={[
            'What exactly does a probe with 95% accuracy prove?',
            'Which kind of test from this lesson shows the model actually uses a direction?',
          ]}
          solution={<><p>A probe shows that the information is <em>present and linearly readable</em> in the activations. It does not show that the model <em>uses</em> that direction to decide what to say, and it may pick up something correlated with honesty in the dataset (topic, length, wording) rather than honesty itself.</p><p>The missing step is an intervention: add or remove the direction during generation (steering, patching) and check that the behaviour changes as predicted, on prompts unlike the training set. Even then, “the model’s honesty” is a much bigger claim than “one direction that affects some honest-sounding outputs”.</p></>}
        >
          <p>A teammate trains a linear probe on a model’s middle layer. It separates true statements from false ones with 95% accuracy on a held-out set. They announce: “We found the model’s honesty direction. We can now tell when it is lying.” What is missing from this argument?</p>
        </Exercise>

        <ClaimSorter
          id="interpretability-sort"
          claims={[
            { text: 'Every number inside the model can be printed and inspected.', level: 'established', why: 'Weights and activations are ordinary arrays. Visibility was never the problem; meaning is.' },
            { text: 'In the toy model, sparse features get packed into more directions than dimensions.', level: 'established', why: 'You reproduced it: the pentagon forms from plain gradient descent on sparse data.' },
            { text: 'Large language models use superposition to represent far more features than they have dimensions.', level: 'research', why: 'Widely believed and consistent with much evidence (polysemantic neurons, SAE results), but not proven in full for any large model.' },
            { text: 'The logit lens shows what the model is thinking at each layer.', level: 'model', why: 'A useful approximation that assumes middle layers use the output’s “language”. It works for some models, poorly for others, and shows a guess, not intent.' },
            { text: 'Clamping the Golden Gate Bridge feature high made Claude 3 Sonnet talk about the bridge.', level: 'established', why: 'A documented intervention result from Scaling Monosemanticity (2024), demonstrated publicly.' },
            { text: 'Attribution graphs give a complete account of how a model computes its answers.', level: 'research', why: 'They explain a replacement model on selected prompts, with error terms, and gave satisfying insight on only a fraction of prompts tried.' },
            { text: 'The residual stream works like a shared whiteboard that each layer adds to.', level: 'model', why: 'The arithmetic (each block adds its output) is exact; the whiteboard is an analogy for it, and says nothing about what the writes mean.' },
          ]}
        />

        <details className="deep">
          <summary>More practice (optional)</summary>
          <div className="details-body">
          <Exercise
            id="interpretability-sae"
            type="calculate"
            title="How big is the dictionary?"
            answer={{ value: 256, tolerance: 0 }}
            answerLabel="times more features than neurons"
            hints={[
              'Divide the number of dictionary features by the number of neurons.',
              '131,072 / 512 = ?',
            ]}
            solution={<><p>131,072 / 512 = <b>256</b>.</p><p>The SAE describes a 512-number activation using a vocabulary 256 times larger, and each individual activation is explained with only a handful of those features switched on. That combination (huge dictionary, few active at once) is the whole trick, and it is the same bet superposition makes in reverse.</p></>}
          >
            <p>In “Towards Monosemanticity”, the largest dictionary had 131,072 features, trained on a layer of 512 MLP neurons. How many times more features than neurons is that?</p>
          </Exercise>

          <ExplainBack
            id="interpretability-explain"
            prompt="Dev says: “We don’t need interpretability. Reasoning models write out their chain of thought, so we can read exactly why they answered.” Explain, in plain words, why the written reasoning is not the same as a look inside."
            modelAnswer={<p>The chain of thought is more output text. It is produced by the same next-token machinery as the answer, and it is shaped by training to look like good reasoning. Nothing forces it to be a faithful report of the computation that actually produced the answer. Experiments show the gap: when researchers slipped a hint into a question and the model used it, reasoning models often did not mention the hint in their written reasoning (in one 2025 study, Claude 3.7 Sonnet mentioned it about a quarter of the time and DeepSeek R1 about 39%). Earlier work found models giving plausible explanations for answers that were really driven by a bias in the prompt. So the written reasoning is useful evidence, often informative, but it is a claim by the model about itself. Interpretability tries to check the claim from the inside: which internal features and paths actually drove the answer.</p>}
          />
          </div>
        </details>
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'Why is looking at neurons one by one not enough to read a model?',
            options: ['Neurons cannot be accessed in trained models', 'Many neurons are polysemantic: one neuron responds to several unrelated things, and one concept is spread over many neurons', 'There are too few neurons', 'Neurons only store numbers between 0 and 1'],
            answer: 1,
            explain: 'The unit of meaning seems to be a direction, not a single coordinate. Superposition is one explanation for why.',
          },
          {
            q: 'What does a sparse autoencoder do?',
            options: ['Compresses the model to make it faster', 'Rebuilds each activation from a very large dictionary of directions, using only a few at a time, so each direction tends to mean one thing', 'Removes features the model does not need', 'Trains the model to be more honest'],
            answer: 1,
            explain: 'It is a tool for reading, trained on the model’s activations. The model itself is unchanged.',
          },
          {
            q: 'Why is a model’s written chain of thought not a reliable window into its computation?',
            options: ['It is always false', 'It is generated text, trained to look like reasoning, and studies show it often omits what actually influenced the answer', 'It is hidden from users', 'It is produced by a separate model'],
            answer: 1,
            explain: 'It is evidence, frequently useful, but not a faithful trace by construction. Measured faithfulness is well below 100%.',
          },
        ]}
      />

      <Remember
        items={[
          <><b>Visible is not readable.</b> Every activation can be printed (and the <b>logit lens</b> reads a rough prediction from the residual stream at any layer); the work is finding which directions mean what, and proving the model uses them.</>,
          <><b>Superposition</b>: with sparse features, a model can store more features than dimensions as almost-perpendicular directions, paying a little interference (the toy pentagon).</>,
          <><b>Sparse autoencoders</b> un-mix activations into many more interpretable features (Golden Gate Bridge). <b>Patching, probes and steering</b> test whether a direction is present and used.</>,
          <>Limits: attribution graphs explain some prompts, partially. No one can yet fully read a large model, and a <b>chain of thought is a claim</b>, not a trace.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Weights and activations', sub: 'all visible' }, { label: 'Features', sub: 'SAEs, probes' }, { label: 'Circuits', sub: 'patching, attribution graphs' }, { label: 'Full account', sub: 'not yet' }]} active={1} />
        <ToyVsReal
          toy={<ul><li>5 features in 2 dimensions, chosen by us</li><li>Synthetic data with known sparsity</li><li>You can see the true answer (the pentagon) and check it by hand</li><li>Trains in under a second</li></ul>}
          real={<ul><li>Unknown number of features in thousands of dimensions per layer, across dozens of layers</li><li>Features must be discovered, not listed; SAEs with millions of entries</li><li>No ground truth: every interpretation needs an intervention to test it</li><li>SAE training and circuit tracing need large compute and expert time</li></ul>}
        />
        <p>Real, reproducible results exist: induction heads in small models, the name-finding circuit in GPT-2 small, board-state probes in Othello-GPT, SAE features that steer behaviour when clamped, and a refusal direction in chat models. Open tools and pretrained SAEs (for example Gemma Scope for Google’s Gemma 2 models) let anyone repeat parts of this work.</p>
        <Callout kind="research">
          What we cannot yet do: give a complete, checked account of how any frontier model produces a given answer; list everything a model represents; or certify that a model does <em>not</em> have some hidden behaviour. Safety teams use interpretability as one source of evidence among several (evaluations, red-teaming), not as proof. As of 2026 it is one of the most active research areas, and these methods are likely to be refined or replaced.
        </Callout>
        <p>Kabir closes the laptop. “So. The numbers are all there. We can read some words of the language. Not the book yet.” Riya looks at her printout again. It still looks like decimals, but now she knows what kind of thing she is looking for.</p>
      </RealLLM>
    </Lesson>
  )
}
