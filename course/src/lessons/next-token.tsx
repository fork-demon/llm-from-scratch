import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Bars, Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { BigramPlayground } from '../interactive/BigramPlayground'

export default function NextTokenLesson() {
  return (
    <Lesson id="next-token">
      <Why>
        <p className="lede">Finish this sentence:</p>
        <div className="card center" style={{ fontFamily: 'var(--serif)', fontSize: 22 }}>The cat sat on the <span className="acc"><b>___</b></span></div>
        <p>You did not think of one word. You thought of several, with different strengths. Something like this:</p>
        <Bars items={[{ label: 'mat', value: 0.41 }, { label: 'floor', value: 0.17 }, { label: 'chair', value: 0.09 }, { label: 'sofa', value: 0.08 }, { label: 'roof', value: 0.03 }, { label: 'everything else', value: 0.22, dim: true }]} max={1} />
        <p className="muted">(Illustrative numbers, not the output of a real model.)</p>
        <p>“mat” is likely, “roof” is possible, “because” is almost impossible. That list of strengths, one for <em>every</em> token in the vocabulary, is the only thing a language model ever produces.</p>
        <Callout kind="idea">
          The fundamental task of an LLM: <b>given the tokens so far, give a probability to every possible next token.</b> GPT-2 did this. The largest models today do this. The rest of the course is about doing it <em>better</em>.
        </Callout>
        <p>In this lesson you build the smallest model that does this job, watch it write, and discover what “training” is really for.</p>
      </Why>

      <Problem title="The problem: how does guessing one token become writing?">
        <p>Two things are puzzling about that definition.</p>
        <div className="grid-2">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>It only predicts one token</h4>
            <p>A chatbot writes paragraphs. A probability list for the next token is not a paragraph. Where does the text come from?</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Where do the probabilities come from?</h4>
            <p>In <a href="#/lesson/neurons">Part 3</a> we needed labelled examples. Who labels billions of sentences with “the right next word”?</p>
          </div>
        </div>
        <WhyExists
          problem="We want a program that can continue any text."
          naive="Write rules: grammar, facts, style. Or store whole sentences and look them up."
          fails="Rules never end, and almost every sentence anyone types is new, so lookup finds nothing."
          idea="Learn one small skill from raw text: the probability of each next token. Then apply that skill over and over, feeding each chosen token back in."
          tradeoff="Nothing in the loop plans a paragraph. Each turn only picks the next token, and every pick is a roll of the dice."
        />
        <p>We will start absurdly small: the tokens are single characters, and the model may look at only <b>one</b> previous character.</p>
      </Problem>

      <MentalModel title="A mental model: count what comes next, then keep pressing">
        <p>Forget neural networks for a moment. How would <em>you</em> guess the next character after “q” in English? You would say “u”, because that is what you have always seen. A guess about what comes next can be built from nothing more than <b>counting what came next before</b>.</p>
        <Callout kind="analogy">
          Think of your phone keyboard’s suggestion bar, pressed again and again. Each press is reasonable given what is on screen, and nobody decided in advance where the sentence would end up.
          <br /><br />
          Where the analogy stops: your keyboard looks at a word or two. An LLM bases each prediction on everything in its <G t="context-window">context window</G> (all the text it can see) through billions of learned weights, which is why its text stays on topic for pages. The loop is the same; the predictor inside it is incomparably better.
        </Callout>
        <h3>Where do the examples come from? From the text itself</h3>
        <p>Take any text. Slide it one position to the left. Every position is now a labelled example: input = this token, correct answer = the token that actually came next.</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ textAlign: 'center' }}>
            <tbody>
              <tr><th scope="row" style={{ textAlign: 'left' }}>input x</th><td>t</td><td>h</td><td>e</td><td>␣</td><td>c</td><td>a</td><td>t</td></tr>
              <tr><th scope="row" style={{ textAlign: 'left' }}>target y</th><td>h</td><td>e</td><td>␣</td><td>c</td><td>a</td><td>t</td><td>␣</td></tr>
            </tbody>
          </table>
        </div>
        <p>Eight characters gave seven training examples, and no human labelled anything. This is why LLMs can be trained on a large fraction of the public internet: <b>every position of every text is a free exam question with the answer attached.</b></p>
      </MentalModel>

      <TryIt title="Experience it first">
        <p>The model below has read a short text about a fox, a dog and some fishermen (about 1,500 characters). For every character, it has counted what came next. Pick “q”. Then pick “t”. Then press <b>Roll once</b> ten times, slowly, and watch what happens on each press.</p>
        <BigramPlayground />
        <p>What you just saw, in plain words:</p>
        <ol>
          <li>The model looks at the current character and produces a probability for every possible next character (the bars).</li>
          <li>A weighted die picks one. Likely characters win often, unlikely ones sometimes.</li>
          <li>The picked character is appended to the text <b>and becomes the new input</b>.</li>
          <li>Repeat.</li>
        </ol>
        <p>That loop is how every LLM writes. When an assistant’s answer streams onto your screen piece by piece, you are watching this loop run, one roll per token.</p>
        <h3>Now let’s name what you saw</h3>
        <Term
          name="Language model"
          plain={<>A program that, given some text, gives a probability to every possible next token.</>}
          example={<>After “q”, our model says: u 95.9%, everything else shares the remaining 4.1%.</>}
          formal={<>A function that outputs the distribution P(next token | previous tokens). The bar “|” reads “given”.</>}
        />
        <Term
          name="Autoregressive generation"
          plain={<>Writing by repeated prediction: predict, pick one token, append it, and use the longer text as the next input.</>}
          example={<>t → th → the → the␣ → the␣s → … Each arrow is one prediction plus one roll of the die.</>}
          formal={<>Sampling x₁, x₂, … one at a time, each from P(xₙ | x₁ … xₙ₋₁). “Auto-regressive” means the model’s own outputs become its inputs.</>}
        />
        <p>Our model’s “previous tokens” is a single character, which makes it a <b>bigram</b> model (bi-gram: it works with pairs). A GPT looks at thousands of previous tokens. The loop around it is the same.</p>
        <Callout kind="established">There is no separate “planning” step in the basic loop, and no lookahead. Each token is one draw, conditioned on all tokens so far, including the model’s own earlier draws. Whether and how large models represent plans <em>inside</em> the computation of each prediction is a research question. The loop itself is not in doubt.</Callout>
      </TryIt>

      <Numbers>
        <p>Where did the bar for “h” after “t” come from? From counting. In the training text, “t” is followed by another character 135 times:</p>
        <div className="table-scroll">
          <table className="plain mono">
            <thead><tr><th>after “t” comes</th><th>h</th><th>␣</th><th>o</th><th>e</th><th>r, t, s, u</th><th>total</th></tr></thead>
            <tbody>
              <tr><td>count</td><td>78</td><td>27</td><td>12</td><td>6</td><td>3 each</td><td>135</td></tr>
              <tr><td>count ÷ 135</td><td><b>0.578</b></td><td>0.200</td><td>0.089</td><td>0.044</td><td>0.022 each</td><td>1.000</td></tr>
            </tbody>
          </table>
        </div>
        <p>That row of probabilities <em>is</em> the model’s prediction after “t”. Twenty-seven rows like it, one per character, are the whole model: a 27 × 27 table.</p>
        <p><b>One wrinkle.</b> “t” was never followed by “z”. A count of 0 means probability 0, which means “impossible”. If the model later meets “tz”, its surprise is −log(0) = infinity and the average loss explodes. So the code adds a tiny 0.01 to every cell before dividing. This is called <b>smoothing</b>:</p>
        <p className="mono center">P(h | t) = (78 + 0.01) ÷ (135 + 27 × 0.01) = 78.01 ÷ 135.27 = 0.5767</p>
        <p>That is the 57.7% you see in the playground.</p>
        <p><b>How good is the model?</b> Use the same score as in <a href="#/lesson/softmax">Part 1</a>: <G t="cross-entropy">cross-entropy</G>, the average surprise at the character that really came next. For the two predictions inside the word “the”:</p>
        <div className="table-scroll">
          <table className="plain mono">
            <thead><tr><th>seen</th><th>actually next</th><th>model’s probability</th><th>surprise = −ln(p)</th></tr></thead>
            <tbody>
              <tr><td>t</td><td>h</td><td>0.5767</td><td>0.550</td></tr>
              <tr><td>h</td><td>e</td><td>0.7858</td><td>0.241</td></tr>
            </tbody>
          </table>
        </div>
        <p>Average that over all 1,478 predictions in the text and you get <b>1.7518</b>. Raise e to that number and you get the <b>perplexity</b>: e<sup>1.7518</sup> = <b>5.76</b>. Read it as: “on average the model is as unsure as if it were choosing among about 6 equally likely characters”. A model with no clue would be choosing among all 27: perplexity 27, cross-entropy ln 27 = 3.30.</p>
        <p className="muted">Two conventions to know. The logarithm here is the natural one, so the loss is in “nats”. Some papers use log₂ and report bits, and then perplexity is 2<sup>loss</sup>: the perplexity comes out the same either way. And perplexity is always <em>per token</em>, so two models can only be compared on it if they cut the text into the same tokens.</p>
      </Numbers>

      <TheMath>
        <p>The table in one line:</p>
        <Equation
          label="Probability of next given current equals count of the pair plus alpha, divided by count of current plus alpha times V"
          symbols={[
            [<>P(b | a)</>, 'the probability that token b comes next, given that the current token is a'],
            [<>count(a, b)</>, 'how many times b came directly after a in the training text'],
            [<>count(a)</>, 'how many times a was followed by anything: the sum of its row'],
            ['α', 'the smoothing constant, 0.01 here: a small pretend-count for every pair so nothing is impossible'],
            ['V', 'the vocabulary size, 27 here'],
          ]}
        >
          P(b | a) = ( count(a, b) + α ) / ( count(a) + α · V )
        </Equation>
        <p>And the score:</p>
        <Equation
          label="Loss equals minus the average log probability of the actual next token; perplexity equals e to the loss"
          symbols={[
            ['N', 'the number of predictions (positions in the text that have a next token)'],
            [<>x<sub>n</sub></>, 'the token at position n'],
            [<>P(x<sub>n+1</sub> | x<sub>n</sub>)</>, 'the probability the model gave to the token that really came next'],
            ['−log', 'the natural logarithm (ln) with a minus sign. It turns a probability into surprise: 0 for a sure thing, large for something the model thought unlikely'],
            ['perplexity', <>e<sup>loss</sup>: the “effective number of choices” the model is torn between</>],
          ]}
        >
          loss = − (1/N) · Σ log P(x<sub>n+1</sub> | x<sub>n</sub>) &nbsp;&nbsp;&nbsp;&nbsp; perplexity = e<sup>loss</sup>
        </Equation>
        <p>A real LLM changes exactly one thing in these formulas: the condition. Instead of P(next | one previous token) it computes P(next | <em>all</em> previous tokens in the context). The loss that trains GPT-class models is this same average surprise.</p>
      </TheMath>

      <CodeIt>
        <p>The whole count model from the repository is a handful of lines. First turn characters into integer IDs (a character-level tokenizer):</p>
        <Code source="phase2-language/bigram_lm.py" title="Step 1: characters to ids">{`
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = [stoi[c] for c in CORPUS]
`}</Code>
        <Code source="phase2-language/bigram_lm.py" title="Step 2: count pairs, then turn each row into probabilities">{`
def count_model(ids, V):
    counts = np.full((V, V), 0.01)    # tiny "smoothing" instead of 0
    for a, b in zip(ids, ids[1:]):    # the shifted pairs (x, y)
        counts[a, b] += 1
    return counts / counts.sum(axis=1, keepdims=True)   # rows -> probabilities
`}</Code>
        <p><code>zip(ids, ids[1:])</code> is the “slide the text by one” trick from above. Scoring is two lines:</p>
        <Code source="phase2-language/bigram_lm.py" title="Step 3: average surprise">{`
def cross_entropy_of_table(probs_table, ids):
    p = probs_table[ids[:-1], ids[1:]]    # probability given to each actual next char
    return -np.log(p).mean()
`}</Code>
        <p>And here is the loop you operated by hand in the playground:</p>
        <Code source="phase2-language/bigram_lm.py" title="Step 4: generation">{`
def generate(probs_table, itos, start_id, n=200):
    out, cur = [], start_id
    for _ in range(n):
        cur = rng.choice(len(itos), p=probs_table[cur])   # roll the weighted die
        out.append(itos[cur])                             # feed output back as input
    return "".join(out)
`}</Code>
        <Code lang="output" title="python phase2-language/bigram_lm.py">{`
corpus: 1479 chars, vocab 27
MODEL A: count-table bigram (classical statistics)
  cross-entropy: 1.7518   perplexity: 5.76
  sample: 'hin is the ile sundr de arethe f eof auilonger reree me ththene ...'
`}</Code>
        <h3>The punchline: train a neural network on the same task</h3>
        <p>The same file then builds the same predictor a second way, with everything from Parts 2 to 4: an <a href="#/lesson/embeddings">embedding lookup</a>, one linear layer, <G t="softmax">softmax</G>, cross-entropy, <G t="gradient-descent">gradient descent</G>. It starts from random numbers and knows nothing about counting.</p>
        <Code source="phase2-language/bigram_lm.py" title="Model B: forward pass of the neural bigram">{`
emb = E[xb]                          # look up the current character's vector
logits = emb @ W                     # one score per possible next character
probs = softmax(logits)
loss = -np.log(probs[np.arange(batch), yb] + 1e-12).mean()
`}</Code>
        <Code lang="output" title="what it prints">{`
MODEL B: neural bigram -- watch it CONVERGE TO MODEL A's loss
  (target: 1.7518)
  step     0  loss 3.3132
  step   500  loss 1.8788
  step  1000  loss 1.7565
  ...
  final cross-entropy 1.7602 vs count-table 1.7518
`}</Code>
        <p>It starts at 3.31, which is ln 27 = 3.30: pure guessing. Scored on the whole text at the end, it reaches 1.7602, a hair above the count table’s 1.7518. It cannot do meaningfully better: the lowest score any one-character model can reach on this text is 1.7480, which is the count table without smoothing. (The losses printed during training are each measured on one random mini-batch of 256 characters, so they wobble, and one may dip slightly under 1.7518 by luck.)</p>
        <Callout kind="idea">
          Gradient descent <b>rediscovered the counting statistics</b>, without being told to count. For one character of context, the count table is (up to the tiny smoothing) the best possible answer, and training found its way to (almost) the same table. So “training” is a way of <em>finding these statistics</em>. It earns its keep in exactly the situations where you cannot build the table. The next lesson shows that this is nearly always.
        </Callout>
        <DeepDive title="Why can nothing beat the count table here?">
          <p>If the only thing you may look at is one character, the best probabilities you can assign are the true frequencies with which each next character followed it. Cross-entropy is minimised exactly when the predicted distribution equals the observed one. The unsmoothed counts are that distribution, measured on the same text we score on. They score 1.7480. Our table gives up 0.004 of that to smoothing, as the price for never calling anything impossible.</p>
          <p>The neural bigram has enough capacity to represent any 27 × 27 table, so its optimum is that same unsmoothed table, and with unlimited training it would creep slightly below 1.7518 towards 1.7480. It stops short (1.7602) only because it was trained for a limited number of noisy mini-batch steps. Note also that both models are scored on their own training text here. Measuring on held-out text is the honest test, and we will do that when we train a GPT.</p>
        </DeepDive>
      </CodeIt>

      <BreakIt>
        <p>Back to the playground. Predict first.</p>
        <ul>
          <li><b>Remove the die.</b> Tick “always take the most likely character”, start from “t”, and roll 40. You get “the the the the …” forever. The most likely <em>next step</em>, taken every time, gives very unlikely <em>text</em>. A little randomness is a feature. How much is the <G t="temperature">temperature</G> dial, in <a href="#/lesson/inference">a later lesson</a>.</li>
          <li><b>Change the seed.</b> Same model, same start, different text. The model is a distribution over texts, not a store of one text.</li>
          <li><b>Change the training text.</b> Open “Edit the training text” and paste some source code, or another language. The rows change, and the gibberish takes on the flavour of the new text. The model is nothing but the statistics of what it read.</li>
          <li><b>Look for meaning.</b> Roll 200 characters. You will see real fragments (“the”, “ing”) and no sense at all. Hold on to the question “what is missing?”. That is the next lesson.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="next-token-calc-q"
          type="calculate"
          title="A probability from counts"
          answer={{ value: 0.9585, tolerance: 0.003 }}
          answerLabel="P(u | q)"
          hints={['Use the formula: (count of the pair + α) ÷ (count of the row + α × V).', 'The pair count is 6, the row total is 6, α = 0.01 and V = 27.', '6.01 ÷ 6.27.']}
          solution={<><p>(6 + 0.01) ÷ (6 + 27 × 0.01) = 6.01 ÷ 6.27 = <b>0.9585</b>.</p><p>Without smoothing it would be 6 ÷ 6 = 1.0: total certainty from six examples. The remaining 4.15% is spread over the 26 characters never seen after “q”. With so few examples, smoothing takes a visible bite. For “t” (135 examples) it changed the answer only from 0.5778 to 0.5767.</p></>}
        >
          <p>In the training text, “q” appears 6 times and is followed by “u” every time. With smoothing α = 0.01 and V = 27 characters, what probability does the table give to “u” after “q”? (Three decimals or more.)</p>
        </Exercise>

        <Exercise
          id="next-token-predict-uniform"
          type="predict"
          title="Before any training"
          answer={{ value: 27, tolerance: 0.5 }}
          answerLabel="perplexity"
          hints={['An untrained model gives every character about the same probability: 1/27.', 'The surprise of each prediction is −ln(1/27) = ln 27. Perplexity is e raised to the average surprise.']}
          solution={<p>Every prediction has probability 1/27, so the loss is ln 27 = 3.30 and the perplexity is e<sup>ln 27</sup> = <b>27</b>: “torn between 27 options”. This is why the neural bigram’s first printed loss is 3.31. It is a useful reference point for any language model you train: “knows nothing” is ln(vocabulary size). For GPT-2’s 50,257 tokens that is about 10.8. A model whose random starting weights are large can begin well <em>above</em> this level, because it is confidently wrong rather than evenly unsure. You will meet exactly that case in <a href="#/lesson/training-gpt">Training GPT</a>.</p>}
        >
          <p>Predict: what is the perplexity of a model that has learned nothing and gives all 27 characters equal probability?</p>
        </Exercise>

        <Exercise
          id="next-token-trace-generate"
          type="trace"
          title="Be the die"
          answer={{ text: ['bac'] }}
          answerLabel="the three generated characters"
          hints={[
            'For each step, lay the probabilities of the current row end to end, in the order a, b, c, and find which slice u falls into.',
            'Step 1, row a: a covers 0 to 0.1, b covers 0.1 to 0.7, c covers 0.7 to 1. u = 0.50 lands in b. The current character is now b.',
            'Step 2 uses row b: a covers 0 to 0.5. u = 0.05 lands in a. Step 3 uses row a again with u = 0.95.',
          ]}
          solution={<><p><b>b, a, c.</b> Step 1: row a, u = 0.50 falls in b’s slice (0.1 to 0.7). Step 2: row b, u = 0.05 falls in a’s slice (0 to 0.5). Step 3: row a again, u = 0.95 falls in c’s slice (0.7 to 1.0).</p><p>Notice two things. The row you read changes at every step, because the output became the input. And step 3 picked “c” from the same row that gave “b” in step 1: same probabilities, different roll.</p></>}
        >
          <p>A three-character model has this table (rows: current character, columns: next character):</p>
          <div className="table-scroll">
            <table className="plain mono">
              <thead><tr><th></th><th>a</th><th>b</th><th>c</th></tr></thead>
              <tbody>
                <tr><th scope="row">a</th><td>0.1</td><td>0.6</td><td>0.3</td></tr>
                <tr><th scope="row">b</th><td>0.5</td><td>0.2</td><td>0.3</td></tr>
                <tr><th scope="row">c</th><td>0.4</td><td>0.4</td><td>0.2</td></tr>
              </tbody>
            </table>
          </div>
          <p>Start from “a”. The random numbers are u = 0.50, then 0.05, then 0.95. The die works like the playground’s: walk along the row from left to right, adding up probabilities, and stop at the first character where the running total exceeds u. Which three characters are generated?</p>
        </Exercise>

        <Exercise
          id="next-token-modify-smoothing"
          type="modify"
          title="Heavier smoothing"
          hints={[
            'In count_model(), change np.full((V, V), 0.01) to np.full((V, V), 1.0) and run python phase2-language/bigram_lm.py.',
            'Think about the row for “q”: 6 real observations, and now 27 pretend ones.',
          ]}
          solution={<><p>Model A gets <b>worse</b>: cross-entropy 1.9841 (perplexity 7.27) instead of 1.7518 (5.76). And now Model B, the neural network at about 1.76, <em>beats</em> the table.</p><p>With α = 1, every row receives 27 pretend observations spread evenly. For “q”, that drowns the 6 real ones: P(u | q) falls from 0.96 to 7 ÷ 33 = 0.21. Smoothing is a belief that “anything can happen”. A little protects you from infinite loss. A lot overrules your data. This trade-off, between trusting the data and hedging against the unseen, comes back throughout machine learning under the name regularisation.</p></>}
        >
          <p>Open <code>phase2-language/bigram_lm.py</code>. In <code>count_model</code>, change the smoothing from <code>0.01</code> to <code>1.0</code>. Before you run it: will Model A’s cross-entropy go up or down, and will the neural Model B still lose to it?</p>
        </Exercise>

        <ExplainBack
          id="next-token-explain"
          prompt="A friend says: “An LLM just predicts the next word, so how can it write a whole essay?” Answer them, using what you did in the playground."
          modelAnswer={<p>The model’s only skill is to give a probability to every possible next token, given the text so far. To write, you run that skill in a loop: get the probabilities, roll a weighted die to pick one token, append it to the text, and ask again with the longer text. An essay is a few thousand turns of that loop. There is no separate essay-writing component. What separates a real LLM from the character table in the playground is how good each single prediction is: it looks at thousands of previous tokens instead of one, so every pick can take the whole essay so far into account.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'What does a language model output at each step?',
            options: ['A probability for every token in the vocabulary', 'The single best next token', 'A complete sentence', 'A yes/no judgement about the input'],
            answer: 0,
            explain: 'Picking one token from that distribution is a separate step (sampling), outside the model.',
          },
          {
            q: 'Why is training data for next-token prediction “free”?',
            options: ['Because text on the internet has no copyright', 'Because the model labels its own data: it first generates the answers and then trains on them', 'Because every position in any text already contains the correct answer: the token that actually came next', 'Because only a few examples are needed'],
            answer: 2,
            explain: 'Shift the text by one position and you have input/target pairs. No human labelling is needed.',
          },
          {
            q: 'The neural bigram’s loss ended at 1.7602 and the count table’s at 1.7518. What is the right conclusion?',
            options: ['Neural networks are worse than counting', 'The count table memorised the text, while the network learned general rules that will score better on new text', 'The neural network had a bug', 'Gradient descent converged to essentially the same statistics the count table holds, which is the best possible with one character of context'],
            answer: 3,
            explain: 'Training is a way to find these statistics. Its advantage appears when the table is impossible to build.',
          },
          {
            q: 'Why does the code add 0.01 to every count?',
            options: ['So that frequent pairs do not dominate: the extra 0.01 evens out common and rare characters', 'So unseen pairs get a tiny probability instead of zero, which would make the loss infinite the first time one occurs', 'To speed up the division', 'Because NumPy cannot store zeros'],
            answer: 1,
            explain: '−log(0) is infinite. Never say never. But too much smoothing (1.0) drowns the real counts.',
          },
          {
            q: 'A model has perplexity 5.76 on some text. What does that mean?',
            options: ['It gets 5.76% of characters right', 'It needs 5.76 seconds per token', 'It is, on average, as uncertain as if choosing among about 6 equally likely options', 'The text has 5.76 distinct characters'],
            answer: 2,
            explain: 'Perplexity is e to the cross-entropy. 27 would mean no knowledge at all for our 27 characters; 1 would mean perfect certainty.',
          },
        ]}
      />

      <Remember
        items={[
          <>A language model does one thing: <b>P(next token | previous tokens)</b>, a probability for every token in the vocabulary.</>,
          <><b>Generation is repeated prediction</b>: predict, sample one token, append, repeat (autoregressive). No plan, no lookahead in the loop itself.</>,
          <><b>Training data is free</b>: shift any text by one position and every position is a labelled example.</>,
          <>Quality is measured by <b>cross-entropy</b> (average surprise) or <b>perplexity</b> = e<sup>loss</sup> (effective number of choices). Ours: 1.7518 and 5.76.</>,
          <>A neural network trained by gradient descent <b>converges to the count table</b>. Training is a way of finding these statistics when a table is impossible.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Tokens so far' }, { label: 'Model', sub: 'here: one table row. GPT: a Transformer' }, { label: 'Probabilities for every next token', sub: 'this lesson' }, { label: 'Sample one' }, { label: 'Append, repeat' }]} active={2} />
        <ToyVsReal
          toy={<ul><li>Tokens are 27 characters</li><li>Sees 1 previous token</li><li>The “model” is a 27 × 27 table of counts</li><li>1,479 characters of training text</li><li>Perplexity 5.76 per character</li></ul>}
          real={<ul><li>Tokens are roughly 30,000 to 250,000 subwords</li><li>Sees thousands to millions of previous tokens</li><li>The model is a Transformer with billions of learned weights</li><li>Trillions of tokens of training text</li><li>Same loss: average −log P(actual next token)</li></ul>}
        />
        <Callout kind="established">The objective you just implemented, next-token cross-entropy on shifted text, is the <G t="pretraining">pretraining</G> objective of GPT-style models. The generation loop you clicked through is how they produce output. Chat behaviour is added later by further training, which we cover in <a href="#/lesson/training-pipeline">From raw text to assistant</a>, but it does not replace this loop.</Callout>
        <Callout kind="dev">This explains several things you see as an API user. Output <b>streams</b> token by token because it is produced token by token. The same prompt can give <b>different answers</b> because each token is a random draw. And output tokens cost more than input tokens partly because each one needs its own pass through the model.</Callout>
        <Callout kind="model">“It just predicts the next token” is accurate about the <em>interface</em> and misleading about the <em>difficulty</em>. To predict the next token of a physics derivation well, a model must capture a great deal about physics. How much of that deserves the word “understanding” is debated. What is certain: all of it is learned in service of this one objective.</Callout>
      </RealLLM>
    </Lesson>
  )
}
