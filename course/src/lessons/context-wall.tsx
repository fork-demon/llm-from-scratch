import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM, BeforeMovingOn } from '../components/lesson'
import { Callout, Equation, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack, OrderExercise } from '../components/exercise'
import { ContextExplosion } from '../interactive/ContextExplosion'
import { NgramGenerator } from '../interactive/NgramGenerator'
import { ContextReach } from '../illustrations/ContextReach'

export default function ContextWallLesson() {
  return (
    <Lesson id="context-wall">
      <Why>
        <p className="lede">Dinner at the flat. Dev is telling a story.</p>
        <p>“Monday, my manager says: keep your laptop charged, the CEO might drop in for a surprise demo.” Then five minutes about traffic, a broken chai machine and a meeting that should have been an email. “Friday, 4 p.m., the door opens. It’s the CEO. And my laptop?” He pauses. “Two percent.”</p>
        <p>Riya laughs. Then she stops, fork in the air. The joke only worked because she still remembered Monday. Her model from last week would have forgotten Monday by the second word.</p>
        <p>Here is what that model wrote when the repository’s Python file ran:</p>
        <div className="card mono" style={{ fontSize: 15 }}>hin is the ile sundr de arethe f eof auilonger reree me ththene ofaiven thersthe wigsle n awhe t m cks qumin qut wathe</div>
        <p>Look closely. Every <em>pair</em> of neighbouring characters is plausible English: “th”, “he”, “qu”, “in”. And the whole is nonsense.</p>
        <p>That is not a bug. The model sees exactly one character. When it writes the “e” of “the”, it has already forgotten the “t”. It cannot finish a word it cannot remember starting. A sentence, or a joke, is out of reach.</p>
        <p>A language model is only as good as the <b>context</b> it can use. So the obvious next move is: look further back. This lesson is about why the obvious ways of doing that hit a wall, and what question they leave behind.</p>
      </Why>

      <Problem title="The problem: why not just remember longer phrases?">
        <p>Riya’s first idea is the obvious one. The count table worked, so give it more to look at. Instead of one row per previous character, use one row per previous <em>two</em> characters. Or five. Or twenty words. Longer context, better guesses. What could go wrong?</p>
        <WhyExists
          problem="One token of context produces text that is locally plausible and globally nonsense."
          naive="Keep counting, but use the last n tokens as the row key instead of one. A table like that is called an n-gram table: n is how many tokens it looks back."
          fails="The number of possible rows is V to the power n. It outgrows any computer almost immediately, and nearly every row that does occur was seen only once."
          idea="Stop treating each context as an unrelated row. Turn the context tokens into embeddings and feed them to a neural network, so similar contexts share what they learn."
          tradeoff="The network has a fixed-size window with a separate set of weights for each position. It still cannot reach far back. That is the wall."
        />
        <p>Do not take this on faith. Both failures can be measured, in a few lines.</p>
        <p><b>One fact about the text before we measure.</b> The last lesson’s training text had 1,479 characters. That is one 493-character text repeated three times, because the Python file writes <code>CORPUS = (...) * 3</code>. Repeating a text adds no new contexts. So in this lesson we count contexts on the single 493-character copy.</p>
      </Problem>

      <MentalModel title="A mental model: a phrasebook versus knowing the language">
        <p>Amma, on the phone that night, has seen this before. “Your cousin went to Paris with a phrasebook,” she says. “Very good, until the waiter said something that was not in the book.”</p>
        <Callout kind="analogy">
          An n-gram table is a <b>phrasebook</b>. If your situation is printed in it, the answer is excellent. If your situation differs by one word, the phrasebook has nothing at all. Making the phrasebook longer does not help: there are more possible situations than pages you could ever print.
          <br /><br />
          Someone who <em>knows the language</em> handles a sentence they have never heard, because it resembles sentences they have. That is <b>generalisation</b>, and a lookup table cannot do it: to a table, “the cat sat” and “the dog sat” are two unrelated keys.
          <br /><br />
          Where the analogy stops: “knowing the language” is not a second magic ingredient. In this lesson it will mean something precise and modest: similar contexts flow through the same weights, so what is learned from one applies to the other.
        </Callout>
        <p>As a developer you know this as the difference between a cache and a function. A cache keyed on the exact input has a 0% hit rate on inputs it has never seen. A function computes an answer for any input. The n-gram table is a cache of the training text.</p>
      </MentalModel>

      <TryIt title="Two experiments">
        <h3>Experiment 1: how big is the table, and how much of it is filled?</h3>
        <p>Start with our 27 characters. Push the context length up and watch the number of rows. Then switch the vocabulary to 50,000 tokens and see how little context it takes to pass the number of atoms in the universe.</p>
        <ContextExplosion />
        <p>With 27 characters and a context of 10, the table has about 200 trillion rows. Our text fills 482 of them, and 480 of those were seen exactly once. With real tokens it is hopeless long before that: a context of just 3 tokens already needs more rows (1.25 × 10<sup>14</sup>) than there are tokens in the largest training sets.</p>
        <h3>Experiment 2: what does a longer-context count model actually write?</h3>
        <p>Same counting idea, more context. Read the output at n = 1, then 3, then 6.</p>
        <NgramGenerator />
        <p>Somewhere around n = 4 the text starts to look like English. Do not be impressed. Look at the two read-outs: by n = 6 almost every step had only one possible continuation, and the output is long runs copied letter for letter from the training text. The model did not learn to write. It learned to <b>recite</b>.</p>
        <p>Counting with long contexts gives you <b>memorisation, not generalisation</b>. Fluent output on the training text tells you nothing. Give this model a context it has not seen, which for long contexts means almost every context, and it has no row to look up.</p>
      </TryIt>

      <Numbers>
        <p>The measurements on the single 493-character copy of the text, side by side:</p>
        <div className="table-scroll">
          <table className="plain mono">
            <thead><tr><th>context length n</th><th>possible contexts 27ⁿ</th><th>contexts that occur</th><th>of those, seen exactly once</th></tr></thead>
            <tbody>
              <tr><td>1</td><td>27</td><td>27</td><td>3 (11%)</td></tr>
              <tr><td>2</td><td>729</td><td>159</td><td>71 (45%)</td></tr>
              <tr><td>3</td><td>19,683</td><td>299</td><td>218 (73%)</td></tr>
              <tr><td>6</td><td>387,420,489</td><td>432</td><td>392 (91%)</td></tr>
              <tr><td>10</td><td>205,891,132,094,649</td><td>482</td><td>480 (99.6%)</td></tr>
            </tbody>
          </table>
        </div>
        <p>Read the last row again. A text of 493 characters contains at most 484 windows of length 10.</p>
        <p>Collect ten times more text and you get roughly ten times more windows: the count grows in step with the text. But the number of possible contexts grows <em>exponentially</em> with n. Doubling n squares it. Data can never catch up.</p>
        <p><b>Now the neural alternative.</b> The Python file’s Model C looks at 3 characters, but not with a table. It looks up an <a href="#/lesson/embeddings">embedding</a> for each of the 3 characters. It glues the three vectors end to end. Then it feeds them to the small <a href="#/lesson/neurons">two-layer network</a> from Part 3.</p>
        <div className="table-scroll">
          <table className="plain mono">
            <thead><tr><th>model (3 characters of context)</th><th>numbers to store</th></tr></thead>
            <tbody>
              <tr><td>count table: 27³ rows × 27 columns</td><td>531,441</td></tr>
              <tr><td>Model C: E (27×16) + W1 (48×64) + b1 (64) + W2 (64×27) + b2 (27)</td><td>432 + 3,072 + 64 + 1,728 + 27 = <b>5,323</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>And the result, from running the file:</p>
        <Code lang="output" title="python phase2-language/bigram_lm.py">{`
MODEL C: 3 chars of context -- breaking the bigram ceiling
  context-3 loss 0.3480  vs bigram optimum 1.7518
  -> more context beats any cleverness with less context.
`}</Code>
        <p>From 1.75 down to 0.35. In perplexity: from about 5.8 “effective choices” per character to e<sup>0.348</sup> = 1.4.</p>
        <p>No amount of cleverness with one character of context could do that, because 1.7518 was already the best possible with one character. <b>More context wins.</b></p>
        <Callout kind="warn" label="Read that number honestly">
          <p>0.3480 looks wonderful. Before you believe it, check what it measures:</p>
          <ul>
            <li>It is the loss on <b>one batch</b>: the last 256 training examples, not new text.</li>
            <li>Those examples come from the <b>same text the model trained on</b>: 493 characters, seen over and over.</li>
            <li>So part of the drop is <b>memorisation</b>. For comparison, the n = 3 count table, which does nothing but memorise, scores 0.32 on this text.</li>
          </ul>
          <p>What we can fairly conclude is modest. Context matters enormously. And a network can use it with about a hundred times fewer numbers than a table (5,323 against 531,441).</p>
          <p>Whether a model has <em>generalised</em> can only be judged on text it has not seen. That is how we will evaluate the GPT in Part 7.</p>
        </Callout>
      </Numbers>

      <TheMath>
        <p>The wall, as a formula:</p>
        <Equation
          label="Number of possible contexts equals V to the power n"
          symbols={[
            ['V', 'vocabulary size: 27 for our characters, 50,000 or more for real tokens'],
            ['n', 'context length: how many previous tokens the model may look at'],
            [<>V<sup>n</sup></>, 'each of the n positions can hold any of V tokens, so the choices multiply: V × V × … × V'],
          ]}
        >
          number of possible contexts = V<sup>n</sup>
        </Equation>
        <p>Compare how the two approaches grow when you add one more token of context:</p>
        <ul>
          <li><b>Table:</b> rows multiply by V. One more character: 27 times more rows. One more real token: 50,000 times more.</li>
          <li><b>Model C:</b> the first layer gets d × hidden more weights (here 16 × 64 = 1,024). It grows by <em>addition</em>, not multiplication.</li>
        </ul>
        <p>That switch from multiplying to adding is what embeddings bought us. Every context, seen or unseen, is a point in the same space, and nearby points get similar predictions.</p>
      </TheMath>

      <CodeIt>
        <p>Model C in the repository differs from the neural bigram in three lines. The input is now a window of <code>ctx</code> character IDs instead of one:</p>
        <Code
          source="phase2-language/bigram_lm.py"
          title="Step 1: windows of 3 characters, and the character that follows each"
          setup={`import numpy as np
rng = np.random.default_rng(7)                  # the seed of bigram_lm.py
CORPUS = (
    "the quick brown fox jumps over the lazy dog and the cat sleeps "
    "in the warm sun while the dog barks at the mailman who walks "
    "down the street every morning with letters for the people in "
    "the town where the children play in the park near the river "
    "that flows past the old mill and under the stone bridge to the "
    "sea where the fishermen cast their nets in the early light of "
    "dawn and sing the old songs of the water and the wind and the "
    "long summer days that fade into the quiet evenings of autumn "
) * 3
chars = sorted(set(CORPUS))
stoi = {c: i for i, c in enumerate(chars)}
V = len(chars)
ids = np.array([stoi[c] for c in CORPUS])
ctx, batch = 3, 4                               # 3 characters of context, a batch of 4 windows
idx = rng.integers(0, len(ids) - ctx, size=batch)`}
          show={`print("xb shape:", xb.shape)
for window, target in zip(xb, yb):
    print(repr("".join(chars[i] for i in window)), "->", repr(chars[target]))`}
        >{`
xb = np.stack([ids[s:s + ctx] for s in idx])    # (batch, ctx)  e.g. ['t','h','e']
yb = ids[idx + ctx]                             #               e.g. ' '
`}</Code>
        <Code
          source="phase2-language/bigram_lm.py"
          title="Step 2: look up, glue together, run the MLP from Part 3"
          setup={`import numpy as np
rng = np.random.default_rng(7)                  # the seed of bigram_lm.py
CORPUS = (
    "the quick brown fox jumps over the lazy dog and the cat sleeps "
    "in the warm sun while the dog barks at the mailman who walks "
    "down the street every morning with letters for the people in "
    "the town where the children play in the park near the river "
    "that flows past the old mill and under the stone bridge to the "
    "sea where the fishermen cast their nets in the early light of "
    "dawn and sing the old songs of the water and the wind and the "
    "long summer days that fade into the quiet evenings of autumn "
) * 3
chars = sorted(set(CORPUS))
stoi = {c: i for i, c in enumerate(chars)}
V = len(chars)
ids = np.array([stoi[c] for c in CORPUS])
ctx, batch = 3, 4                               # 3 characters of context, a batch of 4 windows
idx = rng.integers(0, len(ids) - ctx, size=batch)
xb = np.stack([ids[s:s + ctx] for s in idx])    # step 1
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
dim, hidden = 16, 64                            # untrained weights, the file's sizes
E = 0.1 * rng.normal(size=(V, dim))
W1 = 0.1 * rng.normal(size=(ctx * dim, hidden)); b1 = np.zeros(hidden)
W2 = 0.1 * rng.normal(size=(hidden, V));         b2 = np.zeros(V)`}
          show={`print("E[xb]:", E[xb].shape, "-> glued:", emb.shape, "-> h:", h.shape, "-> probs:", probs.shape)`}
        >{`
emb = E[xb].reshape(batch, -1)                  # concat ctx embeddings: 3 x 16 = 48 numbers
h = np.maximum(0, emb @ W1 + b1)                # hidden layer with ReLU
logits = h @ W2 + b2                            # one score per possible next character
probs = softmax(logits)
`}</Code>
        <p>The rest (cross-entropy, <a href="#/lesson/backprop">backpropagation</a>, the update) is what you have seen three times now. Look at the shape of the first weight matrix:</p>
        <Code
          source="phase2-language/bigram_lm.py"
          title="the catch is visible in one shape"
          setup={`import numpy as np
rng = np.random.default_rng(7)
ctx, dim, hidden = 3, 16, 64`}
          show={`print("W1:", W1.shape)
print("rows 0-15 read slot 1, rows 16-31 read slot 2, rows 32-47 read slot 3")`}
        >{`
W1 = 0.1 * rng.normal(size=(ctx * dim, hidden))   # (3 * 16, 64)
`}</Code>
        <p><code>ctx * dim</code>. The first 16 columns of the input only ever see the character in position 1, the next 16 only position 2, the last 16 only position 3. That one shape contains all three flaws of this design:</p>
        <ol>
          <li><b>The window is fixed.</b> <code>ctx</code> is baked into the weight matrix. The model cannot look at a fourth character, ever, without being rebuilt and retrained.</li>
          <li><b>Weights are tied to positions.</b> “q is followed by u” must be learned separately for q-in-slot-1, q-in-slot-2 and q-in-slot-3, as if they were unrelated facts. Shift a phrase by one position and the model meets it as something new.</li>
          <li><b>Every slot is always read, whether it matters or not.</b> There is no way to say “for this prediction, the word 12 slots back is the important one; ignore the rest”.</li>
        </ol>
      </CodeIt>

      <BreakIt>
        <ul>
          <li><b>Find the step where English appears.</b> In the n-gram generator, which n first produces mostly real words? Now look at the “only one continuation” read-out at that n. Fluency and copying arrive together.</li>
          <li><b>Generate again, several times, at n = 6.</b> Different seeds give different <em>starting points</em> in the training text, then the same recitation. Compare with n = 1, where every seed gives genuinely new (and genuinely bad) text.</li>
          <li><b>Find the last practical table.</b> In the explosion lab, set V = 50,000. What is the largest n whose table has fewer rows than the number of training tokens (A on the chart is people, B is tokens)? It is n = 2. Real n-gram systems reached about 5 words, and only with many engineering tricks for dropping rare rows and coping with unseen contexts.</li>
          <li><b>Make Model C memorise.</b> Do the exercise below that raises <code>ctx</code> to 8 and watch the training loss approach zero. Ask yourself what that number is measuring.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="context-wall-calc-rows"
          type="calculate"
          title="A table for two tokens"
          answer={{ value: 2.5, tolerance: 0.1 }}
          answerLabel="billions of rows"
          hints={['Each of the 2 positions can hold any of the 50,257 tokens.', '50,257 × 50,257. Express the result in billions.']}
          solution={<><p>50,257² = 2,525,766,049, about <b>2.5 billion</b> rows. And each row needs 50,257 columns (one probability per possible next token), so the full table has about 1.3 × 10<sup>14</sup> cells. For a context of <em>two</em> tokens.</p><p>GPT-2 used a context of 1,024 tokens. As a table that would be 50,257<sup>1024</sup> rows, a number with more than 4,800 digits. The entire GPT-2 model is 1.5 billion parameters at most. Whatever it is doing, it is not a lookup table of contexts.</p></>}
        >
          <p>GPT-2’s vocabulary has 50,257 tokens. How many rows would a count table need for a context of just 2 tokens? Answer in billions, one decimal.</p>
        </Exercise>

        <Exercise
          id="context-wall-predict-unseen"
          type="predict"
          title="Fluent, therefore good?"
          hints={['What does the n = 6 table contain for a 6-character context that never occurred in the training text?', 'Think about the loss: −log of the probability given to what actually came next.']}
          solution={<><p>It would do terribly. Almost every 6-character context in a new text is missing from the table (the table fills 432 of 387 million possible rows). Without smoothing, the model gives probability 0 to what actually comes next, and the loss is infinite. With smoothing, it falls back to “all characters equally likely”, which is perplexity 27: knowing nothing.</p><p>Fluent samples only showed that the table can replay its training text. <b>Generalisation is measured on text the model has not seen.</b> This is why every serious evaluation uses held-out data, and it is the same reason you do not test code only on the examples you wrote it against.</p></>}
        >
          <p>At n = 6 the generator writes nearly perfect English, far better than the bigram. Predict: if you scored this model on a <em>new</em> paragraph of ordinary English, would its perplexity be better or worse than the bigram’s 5.76? Why?</p>
        </Exercise>

        <Exercise
          id="context-wall-debug-window"
          type="debug"
          title="“Just make the window 200”"
          hints={['What is the shape of W1 now, and which input positions does each column of weights see?', 'Think about a short prompt of 12 tokens. And think about the same phrase appearing at slots 40 to 45 in one example and slots 41 to 46 in another.']}
          solution={<><p>Nothing crashes, which is what makes it a design bug rather than a code bug.</p><ul><li><b>Position-specific weights.</b> W1 has a separate block of weights for each of the 200 slots. A pattern learned at slots 40 to 45 is unknown at slots 41 to 46. The model needs to see every useful pattern at every offset, so it needs vastly more data.</li><li><b>Fixed length.</b> A 12-token prompt must be padded to 200, and token 201 can never be seen. The limit is welded into the matrix shape.</li><li><b>No selectivity.</b> In most sentences a handful of earlier tokens matter for the next one, and <em>which</em> ones changes every time. This model reads all 200 slots through the same fixed weights regardless.</li></ul><p>What we want instead is a mechanism that treats “a relevant token 3 back” and “a relevant token 150 back” with the <em>same</em> machinery, and decides per sentence what is relevant.</p></>}
        >
          <p>A colleague wants the model to handle long-range context and proposes a one-line change. It runs. Why is it still a poor design for language?</p>
          <Code>{`
loss_C = train_context3(ids, V, ctx=200)
# inside: W1 = 0.1 * rng.normal(size=(ctx * dim, hidden))   -> shape (3200, 64)
`}</Code>
        </Exercise>

        <details className="deep">
          <summary>More practice (optional)</summary>
          <div className="details-body">
        <Exercise
          id="context-wall-modify-ctx"
          type="modify"
          title="Give Model C more context"
          hints={['At the bottom of phase2-language/bigram_lm.py, change train_context3(ids, V) to train_context3(ids, V, ctx=5), then ctx=8. Each run takes a few seconds.', 'Compare the printed losses with 0.3480. Then remember how long the underlying text is (493 characters) and what the loss is measured on.']}
          solution={<><p>With the file otherwise unchanged you get about <b>0.24</b> for ctx=5 and <b>0.03</b> for ctx=8 (against 0.35 for ctx=3). A loss of 0.03 means the model gives about 97% probability to the correct next character. English is not that predictable. The network has memorised its 493-character text, which with 8 characters of context is almost always enough to know exactly where you are in it.</p><p>Two lessons. First, a falling <em>training</em> loss can mean learning or memorising, and only held-out text can tell them apart. Second, even here the window is still 8 characters. Nothing this model does can connect a pronoun to a noun 50 characters earlier.</p></>}
        >
          <p>Open <code>phase2-language/bigram_lm.py</code>. Model C is trained by the call <code>train_context3(ids, V)</code> near the end. Predict what happens to the loss with <code>ctx=5</code> and <code>ctx=8</code>, then run both. Is the ctx=8 model a better model of English?</p>
        </Exercise>

        <ExplainBack
          id="context-wall-explain"
          prompt="Explain to a teammate why “just count longer phrases” cannot produce a good language model, however much text you have. Use the words memorisation and generalisation."
          modelAnswer={<p>The number of possible contexts is the vocabulary size to the power of the context length, so it grows exponentially, while the number of contexts you can observe grows only in proportion to your text. For any useful context length, almost every context you meet is new, and the ones you did see were seen once. A table can only replay what it saw: that is memorisation, and it looks fluent only on the training text. Generalisation means giving sensible probabilities for contexts never seen before, which requires treating similar contexts similarly. A table cannot, because every key is unrelated to every other key. A neural network with embeddings can, because similar inputs pass through the same weights.</p>}
        />
          </div>
        </details>
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'The n = 6 character model produces nearly perfect English. What is the best explanation?',
            options: ['Six characters cover most English words, so the table has learned the rules of spelling and grammar', 'Six is the natural context length of English', 'Longer contexts make the die fairer', 'With 6 characters of context there is almost always exactly one continuation in the training text, so it is copying that text'],
            answer: 3,
            explain: 'The “only one continuation” read-out was near 90%. That is recitation, not generalisation.',
          },
          {
            q: 'What do embeddings give the neural model (Model C) that the table lacks?',
            options: ['Unlimited context: because vectors can be added, any number of previous characters can be folded into the input', 'Shared statistical strength: similar contexts are nearby points that flow through the same weights, so unseen contexts still get sensible predictions', 'Exact counts', 'Freedom from needing training data'],
            answer: 1,
            explain: 'And it needs about 5 thousand numbers instead of half a million. What it does not fix is the fixed, position-bound window.',
          },
          {
            q: 'Which of these is NOT a limitation of “concatenate a fixed window of embeddings into an MLP”?',
            options: ['The window length is baked into the weight shapes', 'Each position has its own weights, so a shifted phrase looks new', 'It cannot be trained with gradient descent', 'It cannot use a token 200 positions back'],
            answer: 2,
            explain: 'It trains fine. Its problems are architectural: fixed window, position-specific weights, no way to choose what matters.',
          },
        ]}
      />

      <Remember
        items={[
          <>One token of context gives text that is <b>locally plausible, globally nonsense</b>. More context helps enormously: 1.75 → 0.35 with three characters.</>,
          <>Count tables cannot scale: possible contexts = <b>Vⁿ</b>, and almost every long context in real text occurs <b>once or never</b> (data sparsity).</>,
          <>Long-context counting <b>memorises</b>; it does not <b>generalise</b>. Fluent output on training text proves nothing. Judge models on unseen text.</>,
          <>Embeddings + a neural network share strength across similar contexts, with parameters that grow by addition, not multiplication. But a concatenated window is <b>fixed in length, bound to positions, and cannot choose what matters</b>. Open question: how can a model use relationships between distant tokens, when which tokens matter changes every sentence?</>,
        ]}
      />

      <RealLLM>
        <ContextReach />
        <ToyVsReal
          toy={<ul><li>Characters, 3 of them, 493 characters of text</li><li>Loss measured on the training text</li><li>Runs in seconds</li></ul>}
          real={<ul><li>Word n-gram tables (typically up to 5 words, plus tricks for unseen contexts) powered speech recognition and machine translation for decades</li><li>The fixed-window neural language model of 2003 is Model C at scale: it introduced learned word embeddings for exactly the reason in this lesson</li><li>Neither could use long-range context</li></ul>}
        />
        <h3>The sentence that breaks all of them</h3>
        <div className="card center" style={{ fontFamily: 'var(--serif)', fontSize: 21 }}>
          “The animal didn’t cross the road because <span className="acc"><b>it</b></span> was tired.”
        </div>
        <p>To continue this sentence sensibly, a model must know what “it” is. The answer, “animal”, is 6 words back. In a real document the relevant word might be 200 tokens back. And change “tired” to “flooded” and the relevant word becomes “road”: <em>which</em> earlier token matters depends on the sentence itself.</p>
        <p>A fixed window either does not reach that far, or reaches it through weights bound to “slot 6”, which are useless when the next sentence has the noun in slot 9.</p>
        <h3>The pre-2017 answer: read left to right and keep a summary</h3>
        <p><b>Recurrent neural networks</b> (RNNs, and their improved form, LSTMs) removed the fixed window. They read one token at a time and keep a running summary vector: new summary = f(old summary, new token). The same weights are used at every position, and in principle the summary can carry information any distance.</p>
        <p>They worked, and they ran into two flaws of their own:</p>
        <ul>
          <li><b>A fixed-size bottleneck.</b> Everything read so far must be squeezed into one vector of fixed size. By the time the model reaches “it”, “animal” has been overwritten many times. Distant details fade.</li>
          <li><b>Strictly sequential.</b> Step 500 needs the summary from step 499. You cannot compute the positions of a sequence in parallel. GPUs are fast because they do thousands of things at once, so training on long texts makes poor use of them.</li>
        </ul>
        <Callout kind="established">Both limitations are well documented in the research of that period, and they are the stated motivation for the architecture that replaced RNNs in 2017. (Recurrent ideas have not disappeared. Some current research architectures revisit them with new tricks. Every mainstream LLM today is built on the alternative you are about to meet.)</Callout>
        <h3>The question you should now be asking</h3>
        <Callout kind="idea">
          <b>How can a model use relationships between tokens that are far apart, when which tokens matter changes with every sentence?</b>
          <br /><br />
          We want no fixed window, no weights bound to positions, no squeezing everything through one summary, and no waiting for step 499. Think about what such a mechanism would have to do for the word “it”. Then start Part 6.
        </Callout>
        <p>Riya sends Kabir one line that night: “How does a model remember Monday?” He replies in the morning: “Good question. That one has a whole architecture named after it.”</p>
      </RealLLM>

      <BeforeMovingOn
        id="part-5"
        questions={[
          {
            q: 'From Part 0: what is the single computation an LLM repeats to produce an answer?',
            options: ['Compute a probability for every possible next token, pick one, append it', 'Retrieve the closest stored answer', 'Translate the prompt to logic and solve it', 'Rank complete candidate answers'],
            answer: 0,
            explain: 'You operated this loop by hand in the bigram playground.',
          },
          {
            q: 'From Part 1: a model’s scores (logits) for three tokens are [2, 2, 2]. What does softmax return?',
            options: ['[2, 2, 2]', '[1, 1, 1]', '[1, 0, 0]', '[⅓, ⅓, ⅓]'],
            answer: 3,
            explain: 'Equal scores, equal probabilities. Softmax only cares about differences between scores.',
          },
          {
            q: 'From Part 2: the neural bigram’s loss went 3.31 → 1.88 → 1.76. What was happening at each step?',
            options: ['The model was counting character pairs', 'Every parameter was nudged a little against its gradient, in the direction that reduces the loss', 'Random weights were tried until a better set was found', 'The learning rate was being increased'],
            answer: 1,
            explain: 'Gradient descent. It arrived at the counting statistics without being told to count.',
          },
          {
            q: 'From Part 3: Model C has a ReLU between its two layers. What would happen without it?',
            options: ['Nothing would change', 'The softmax would stop summing to 1', 'Training would be faster and just as accurate, because ReLU only throws away the negative half of the signal', 'Two linear layers collapse into one linear layer, so the hidden layer would add no expressive power'],
            answer: 3,
            explain: 'A stack of linear maps is a linear map. The bend is what lets layers represent more than a single matrix could.',
          },
          {
            q: 'From Part 4: why does Model C look characters up in an embedding table instead of using their IDs as numbers?',
            options: ['IDs are names, not quantities. A learned vector per token lets similar tokens get similar representations', 'Embedding tables are faster than integers', 'IDs would make the vocabulary too large', 'Because softmax requires vectors of length 16'],
            answer: 0,
            explain: 'And it is those learned vectors that let similar contexts share statistical strength.',
          },
          {
            q: 'From Part 4: a tokenizer with a bigger vocabulary makes each text shorter in tokens. What does it do to a count table over contexts of n tokens?',
            options: ['Shrinks it', 'Nothing', 'Makes the explosion worse: the number of possible contexts is Vⁿ', 'Makes all rows equally likely'],
            answer: 2,
            explain: 'Bigger V means each token carries more, but the space of contexts grows even faster.',
          },
          {
            q: 'From this part: which statement about the count-table bigram and the neural bigram is correct?',
            options: ['The neural one is far better, because embeddings let it generalise beyond what one character of context allows', 'They converge to nearly the same loss, because the count table is already optimal for one token of context', 'The count table cannot generate text', 'Only the neural one needs smoothing'],
            answer: 1,
            explain: '1.7602 versus 1.7518. Training is a way of finding these statistics when a table is impossible, which for real context lengths is always.',
          },
        ]}
      >
        <OrderExercise
          id="context-wall-order-loop"
          title="Rebuild the generation loop from memory"
          prompt={<p>Put the steps of text generation in order, starting from a prompt.</p>}
          correct={['Tokenize the prompt into token IDs', 'Look up an embedding vector for each ID', 'The model computes a score for every possible next token', 'Softmax turns the scores into probabilities', 'Sample one token from those probabilities', 'Append it to the context and go again']}
          solutionNote={<p>Only the third step differs between a bigram table, Model C and GPT. Everything in Part 6 is about making that one step able to use the whole context.</p>}
        />
      </BeforeMovingOn>
    </Lesson>
  )
}
