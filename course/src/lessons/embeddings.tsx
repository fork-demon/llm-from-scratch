import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM, BeforeMovingOn } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack, OrderExercise } from '../components/exercise'
import { EmbeddingExplorer } from '../interactive/EmbeddingExplorer'
import { Word2VecTrainer } from '../interactive/Word2VecTrainer'
import { TextToVector } from '../illustrations/TextToVector'

export default function EmbeddingsLesson() {
  return (
    <Lesson id="embeddings">
      <Why>
        <p className="lede">The tokenizer gave us integers. Suppose “cat” is token 464, “dog” is token 12 and “car” is token 465.</p>
        <p>Is “cat” more like “dog” or more like “car”? The IDs say “car”: 464 and 465 are neighbours. That is nonsense. The IDs were handed out in the order the tokenizer happened to create them.</p>
        <p>An ID is like a primary key in a database. It tells you <em>which</em> row you have. It tells you nothing about what the row is <em>like</em>. And “what is this token like?” is exactly what a model needs, because a model that cannot tell that cats and dogs are similar must learn every fact about them twice.</p>
        <Callout kind="idea">
          An <b>embedding</b> is a way of representing something using numbers so that useful relationships can be learned. Each token gets not one number but a whole list of numbers, a <G t="vector">vector</G>, and similar tokens end up with similar vectors.
        </Callout>
      </Why>

      <Problem title="The problem: an ID is a name, not a quantity">
        <p>Why not just feed the ID into the network as a number? Because every operation you have learned treats numbers as <em>quantities</em>.</p>
        <ul>
          <li>A <a href="#/lesson/neurons">neuron</a> computes weight × input. With raw IDs, token 464 would push almost 39 times harder than token 12, for no reason.</li>
          <li>ID 464 is not “bigger” than ID 12, and the average of tokens 12 and 464 is not token 238.</li>
          <li><a href="#/lesson/gradient-descent">Gradient descent</a> makes small changes. There is no such thing as “token 464.01”.</li>
        </ul>
        <WhyExists
          problem="Token IDs are arbitrary labels, but the network does arithmetic on its inputs."
          naive="Feed the ID in as a number, or describe each word by hand-written features (is_animal, is_plural, …)."
          fails="Raw IDs invent a fake order and size. Hand-written features would need millions of human judgements, and nobody knows which features matter."
          idea="Give every token its own row of adjustable numbers in a table. Start them random. Let training move them."
          tradeoff="One more big matrix of parameters to learn, and the numbers in it have no human-readable meaning."
        />
      </Problem>

      <MentalModel title="A mental model: words as points on a map">
        <p>You already describe things with lists of numbers. A house in a property database might be <span className="mono">[price, area, bedrooms, year]</span>. Two similar houses have similar lists, without anyone writing a <code>similar()</code> function. Similarity falls out of the representation.</p>
        <p>An embedding does this for tokens. Each token is a point in space, and the space is arranged so that <b>distance and direction mean something</b>: “cat” near “dog”, both far from “car”.</p>
        <Term
          name="Embedding"
          plain={<>A list of numbers that stands for a token, chosen (by training) so that tokens used in similar ways get similar lists.</>}
          example={<>cat → [−3.0, 3.2], dog → [−3.6, 2.4], car → [−2.4, −3.4]. cat and dog are 1.0 apart; cat and car are 6.6 apart.</>}
          formal={<>A row of a learned matrix E with one row per token in the vocabulary: the embedding of token i is E[i], a vector of d numbers (d is typically 768 to 16,384).</>}
        />
        <p>Mechanically it could not be simpler. The “embedding layer” is a table. The token ID is the row number.</p>
        <TextToVector focus="lookup" id={464} idNote="(last lesson)" vector="[0.21, −1.30, …]" />
        <Callout kind="analogy">
          Think of a <b>map</b>. Cities that are close on the map are close in reality, and “300 km north” is the same arrow wherever you start. In an embedding space, closeness is similarity of use, and some directions carry a consistent meaning.
          <br /><br />
          Where the analogy stops: a map has two axes with names (north, east). A real embedding space has thousands of axes and <em>none of them has a name</em>. Nobody can picture it, and nobody drew it. The positions are learned.
        </Callout>
      </MentalModel>

      <TryIt title="Play with a toy space">
        <p>Here is a space small enough to draw. “cat” and “dog” are selected. Click other pairs, then drag words around and watch the three numbers from <a href="#/lesson/vectors">lesson 1.1</a> respond.</p>
        <EmbeddingExplorer />
        <p>Things to try:</p>
        <ul>
          <li>Click “cat”, then “car”. The dot product goes negative: the arrows point in opposing directions.</li>
          <li>Drag “kitten” straight away from the centre, along its own arrow. Distance to “cat” grows, but cosine barely moves. <G t="cosine">Cosine similarity</G> only cares about direction, which is why it is the usual way to compare embeddings.</li>
          <li>Switch on the arrow demo. The step from “man” to “woman”, replayed from “king”, lands next to “queen”. Now drag “woman” somewhere else and watch the trick break. It only works when the layout has that regularity.</li>
        </ul>
        <Callout kind="warn" label="We cannot draw the real space">
          <b>The 2D picture is only a teaching model.</b> I placed these 13 points by hand to make a readable picture. Real embeddings are not placed by anyone, do not live in 2D, and their axes do not mean “royalty” or “animal”.
        </Callout>
        <h3>Why real models need hundreds or thousands of numbers per token</h3>
        <p>Try to improve the toy layout. “puppy” should be near “dog” (same animal), near “kitten” (both young), and “king” should be near “man” (both male) but also near “queen” (both royal). Now add: formal or casual, noun or verb, singular or plural, positive or negative, everyday or technical.</p>
        <p>Words are similar in <b>many independent ways at once</b>. On a flat page you have two independent directions. Every new kind of similarity you try to respect wrecks one you already had. You can feel this by dragging: fix one relationship, break another.</p>
        <p>With 4,096 numbers per token there is room for thousands of kinds of similarity to coexist. You do not need to visualise 4,096 dimensions, and nobody can. The arithmetic is unchanged: a dot product in 4,096-D is still “multiply matching entries, add up”. Only the picture is lost.</p>
        <h3>Nobody assigns the coordinates</h3>
        <p>So who decides that “cat” gets <em>these</em> numbers? Nobody. They start random and are <b>learned</b>, using the loop you already know: predict, measure the <G t="loss">loss</G>, nudge the numbers downhill.</p>
        <p>But predict <em>what</em>? You cannot write a loss for “put similar words together” without already knowing which words are similar. The trick, from a 2013 method called <b>word2vec</b>, is to train on a <b>fake task</b> whose cheapest solution requires good geometry:</p>
        <Callout kind="idea">The fake task, known as <b>skip-gram</b>: <b>given a word, predict a word that appeared near it.</b> Nobody cares about the predictions. We keep the vectors that the task forced into shape.</Callout>
        <p>Why does that work? In the training sentences, “cat” and “dog” keep the same company: <em>feed the ___</em>, <em>my pet ___</em>, <em>the ___ chased</em>. So the model must make nearly the same predictions for both.</p>
        <p>But the only thing it knows about a word is its vector. The cheapest way to make the same predictions from two vectors is to make the two vectors alike. Gradient descent always takes the cheap way.</p>
        <p>Watch it happen. This trains real vectors in your browser, from random numbers:</p>
        <Word2VecTrainer />
        <p>Nothing in that code mentions animals, food or computers. “cat ~ dog” rose because the two words are used in the same contexts. “cat ~ computer” stayed near zero because they are not. An old linguistics slogan says it well: you shall know a word by the company it keeps.</p>
      </TryIt>

      <Numbers>
        <p><b>First, the lookup.</b> A tiny vocabulary of 4 tokens, 3 numbers each. E has 4 rows:</p>
        <div className="table-scroll">
          <table className="plain mono">
            <thead><tr><th>token ID</th><th>token</th><th colSpan={3}>row of E</th></tr></thead>
            <tbody>
              <tr><td>0</td><td>the</td><td>0.1</td><td>0.0</td><td>−0.2</td></tr>
              <tr><td>1</td><td>cat</td><td>0.9</td><td>0.8</td><td>0.1</td></tr>
              <tr style={{ fontWeight: 700 }}><td>2</td><td>dog</td><td>0.8</td><td>0.9</td><td>0.2</td></tr>
              <tr><td>3</td><td>car</td><td>−0.7</td><td>0.1</td><td>0.9</td></tr>
            </tbody>
          </table>
        </div>
        <p>The embedding of token 2 is <span className="mono">E[2] = [0.8, 0.9, 0.2]</span>. An array index. That is all.</p>
        <p>You will often read that this “is a matrix multiplication”. It is, in disguise. First write token 2 as a <b>one-hot</b> vector: a row of zeros with a single 1 in it, at position 2. (“One-hot” means one entry is switched on and all the rest are off.) Now multiply that row by E:</p>
        <p className="mono center">[0, 0, 1, 0] × E = 0·row0 + 0·row1 + 1·row2 + 0·row3 = [0.8, 0.9, 0.2]</p>
        <p>Same answer. The zeros wipe out every row except one. Real code uses the index, because multiplying 50,000 numbers by zero is a waste. The matrix view matters for one reason: it shows that the table is an ordinary layer of weights, so <a href="#/lesson/backprop">backpropagation</a> can train it like any other.</p>
        <p><b>Second, similarity.</b> With the toy coordinates from the explorer:</p>
        <div className="table-scroll">
          <table className="plain mono">
            <thead><tr><th>pair</th><th>dot product</th><th>÷ lengths</th><th>cosine</th></tr></thead>
            <tbody>
              <tr><td>cat [−3.0, 3.2] · dog [−3.6, 2.4]</td><td>10.80 + 7.68 = 18.48</td><td>÷ (4.39 × 4.33)</td><td><b>0.97</b></td></tr>
              <tr><td>cat [−3.0, 3.2] · car [−2.4, −3.4]</td><td>7.20 − 10.88 = −3.68</td><td>÷ (4.39 × 4.16)</td><td><b>−0.20</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>The question “are these two tokens alike?” has become a multiplication and a sum. That is the whole point: meaning has been turned into something a matrix multiply can work with.</p>
      </Numbers>

      <TheMath>
        <p>Two small formulas: the lookup, and the fake task that trains it.</p>
        <Equation
          label="The embedding of token i is row i of E, which equals one-hot of i times E"
          symbols={[
            ['E', 'the embedding matrix: V rows (one per token in the vocabulary), d columns. Every entry is a learned parameter'],
            ['i', 'a token ID, an integer from 0 to V − 1'],
            [<>E[i]</>, 'row i of E: the d numbers that represent token i'],
            [<>onehot(i)</>, 'a vector of V zeros with a single 1 at position i'],
          ]}
        >
          x = E[i] = onehot(i) · E
        </Equation>
        <p>And the skip-gram guessing game, built from parts you already own:</p>
        <Equation
          label="Probability of each context word equals softmax of the center word's embedding times W"
          symbols={[
            ['c', 'the ID of the centre word, for example “cat”'],
            [<>E[c]</>, 'its embedding: the only information the model has about the word'],
            ['W', <>a second learned matrix (d × V) that turns a vector into one score per vocabulary word (<a href="#/lesson/matrices">a matrix multiply is many dot products at once</a>)</>],
            ['softmax', <>turns the V scores into probabilities (<a href="#/lesson/softmax">lesson 1.3</a>)</>],
            ['loss', <><G t="cross-entropy">cross-entropy</G>: −log of the probability given to the word that really was nearby</>],
          ]}
        >
          P(neighbour | c) = softmax( E[c] · W ) &nbsp;&nbsp;&nbsp; loss = −log P(true neighbour | c)
        </Equation>
        <p>When training is over, W is thrown away. E is the product.</p>
        <DeepDive title="Why does only one row of E change per example?">
          <p>The loss for one example depends on E only through the row that was looked up, E[c]. Every other row was multiplied by zero in the one-hot view, so its gradient is exactly zero. In one training step on “cat”, row “cat” moves and the other rows stay put.</p>
          <p>A consequence worth remembering: a token that is rare in the training text gets few updates, so its vector stays close to its random starting point. Models are unreliable about rare things partly for this plain reason.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>The repository file trains these vectors in NumPy. First the data. Every word is paired with each neighbour up to 2 positions away:</p>
        <Code source="phase2-language/tiny_word2vec.py" title="Step 1: (centre, neighbour) training pairs">{`
for s in sentences:
    toks = [stoi[w] for w in s.split()]
    for i, center in enumerate(toks):
        for j in range(max(0, i - window), min(len(toks), i + window + 1)):
            if j != i:
                pairs.append((center, toks[j]))   # (center, one context word)
`}</Code>
        <p>“feed the cat some fish” yields (cat, feed), (cat, the), (cat, some), (cat, fish), and so on for each word. No human labelled anything. The text labels itself.</p>
        <Code title="Step 2: two random matrices">{`
E = 0.1 * rng.normal(size=(V, dim))   # the embedding matrix: THE product
W = 0.1 * rng.normal(size=(dim, V))   # output layer: scaffolding, discarded later
`}</Code>
        <Code title="Step 3: forward pass. The lookup really is an index">{`
emb = E[centers]                       # (batch, dim)  the lookup
logits = emb @ W                       # (batch, V)    one score per word
probs = softmax(logits)
`}</Code>
        <p>(<G t="logits">Logits</G> are the raw scores before softmax.) The backward pass uses exactly the rules from <a href="#/lesson/backprop">Backpropagation</a>. The gradient of softmax plus cross-entropy is “probabilities minus the one-hot target”:</p>
        <Code source="phase2-language/tiny_word2vec.py" title="Step 4: backward pass and update">{`
d_logits = probs.copy()
d_logits[np.arange(batch), contexts] -= 1      # probs - one_hot
d_logits /= batch
d_W = emb.T @ d_logits
d_emb = d_logits @ W.T

W -= lr * d_W
# only the looked-up rows of E get gradients (scatter-add):
np.add.at(E, centers, -lr * d_emb)
`}</Code>
        <p>The last line is the DeepDive above in code: only the rows that were looked up get nudged. Running the file prints the similarities as they form:</p>
        <Code lang="output" title="python phase2-language/tiny_word2vec.py">{`
vocab: 70 words, 9840 training pairs
    step          cat~dog    cat~computer    bread~cheese
       0            0.085          -0.318          -0.066
     800            0.702          -0.234           0.500
     ...
    4000            0.611          -0.242           0.319

         cat -> dog (0.61), mouse (0.47), day (0.41), ran (0.38)
    computer -> laptop (0.59), software (0.57), wrote (0.54), update (0.53)
analogy dog - cat + fish  ->  ['fish', 'beans', 'hungry', 'loudly']
`}</Code>
        <Callout kind="established" label="Honest results">
          Look closely. “cat” found “dog” and “computer” found “laptop”. But cat’s third neighbour is “day”, and the analogy came back as noise. With 28 short sentences, that is what you get. The quality of the geometry grows with the amount of text. The original word2vec vectors were trained on roughly 100 billion words.
        </Callout>
      </CodeIt>

      <BreakIt>
        <ul>
          <li><b>Squeeze the space.</b> In the trainer, choose “2 numbers per word” and train. Predict first: will “cat ~ computer” stay low? (Usually not. With two numbers there are not enough directions to keep 70 words apart, unrelated words get crammed together, and the loss stalls near 3 instead of 2.)</li>
          <li><b>Different random start.</b> Press “New random start” and train again. The coordinates come out completely different, yet the same pairs are similar. The meaning is in the <em>relationships</em> between vectors, never in the raw numbers. This is why you cannot mix embeddings from two different models.</li>
          <li><b>Break the analogy.</b> In the explorer, move “queen” a little. How far can it go before “king − man + woman” picks another word? In 2D with 13 words, quite far. There is little competition. With 50,000 words there is a lot.</li>
          <li><b>The “bank” problem.</b> Where on the 2D map would you put “bank”? Near “river”? Near “money”? Try to argue for one spot. Keep your answer in mind for the end of this lesson.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="embeddings-calc-table-size"
          type="calculate"
          title="How big is the table?"
          answer={{ value: 38.6, tolerance: 0.3 }}
          answerLabel="millions of parameters"
          hints={['The table has one row per token and one column per embedding dimension.', '50,257 × 768. Give the answer in millions.']}
          solution={<><p>50,257 × 768 = 38,597,376, about <b>38.6 million</b> parameters. GPT-2 small has about 124 million in total, so nearly a third of the model is this one lookup table.</p><p>This is the vocabulary-size trade-off from the last lesson, in numbers: every extra token costs one more row of d parameters that has to be learned from data.</p></>}
        >
          <p>GPT-2 small has a vocabulary of 50,257 tokens and uses 768 numbers per token. How many parameters are in its embedding table? Answer in millions, one decimal.</p>
        </Exercise>

        <Exercise
          id="embeddings-trace-pairs"
          type="trace"
          title="Count the training pairs"
          answer={{ value: 14, tolerance: 0 }}
          answerLabel="number of pairs"
          hints={['Go word by word. “feed” is at the edge: it only has neighbours to its right (the, cat).', 'Neighbours within 2 positions: feed 2, the 3, cat 4, some 3, fish 2.']}
          solution={<p>2 + 3 + 4 + 3 + 2 = <b>14</b> pairs from one five-word sentence. The middle word “cat” has 4 neighbours: (cat, feed), (cat, the), (cat, some), (cat, fish). This is why a modest amount of text yields a great deal of training data: the 560 sentences in the Python file (28 sentences, repeated 20 times) produce 9,840 pairs.</p>}
        >
          <p>Using the pair-building code above with <code>window=2</code>, how many (centre, neighbour) pairs does the sentence “feed the cat some fish” produce?</p>
        </Exercise>

        <Exercise
          id="embeddings-debug-dot"
          type="debug"
          title="Everything is similar to “the”"
          hints={['What two things make a dot product large? Think back to lesson 1.1.', 'A dot product grows with agreement in direction and with the lengths of the vectors. In a small model like this one, frequent words are updated most often and can end up with long vectors.']}
          solution={<><p>The raw dot product mixes up direction and length. A long vector gets a big dot product with almost anything. Divide by both lengths to get cosine similarity, as the repository’s <code>nearest()</code> does:</p><Code>{`
sims = E @ q / (np.linalg.norm(E, axis=1) * np.linalg.norm(q) + 1e-9)
`}</Code><p>The small <code>1e-9</code> avoids dividing by zero. The real function also skips the first result, which is always the query word itself.</p></>}
        >
          <p>A colleague writes a nearest-neighbour search over trained embeddings. For almost every query, the top results are the same few very frequent words. What is wrong?</p>
          <Code>{`
def nearest(words, stoi, E, query, k=4):
    q = E[stoi[query]]
    sims = E @ q
    best = np.argsort(-sims)
    return [words[i] for i in best[:k]]
`}</Code>
        </Exercise>

        <Exercise
          id="embeddings-modify-dim"
          type="modify"
          title="Two dimensions, for real"
          hints={['Run python phase2-language/tiny_word2vec.py first and note the final cat~computer value (−0.242).', 'At the bottom of the file, change train() to train(dim=2) and run again.']}
          solution={<><p>With <code>dim=2</code>, cat~dog still goes to about 0.96, but <b>cat~computer ends around 0.79</b> instead of −0.24, and during training it sits near 1.0 for a long time. In two dimensions a vector’s direction is a single angle. Seventy words must share one circle, so unrelated words are forced to be neighbours.</p><p>This is the honest reason for high-dimensional embeddings. It is not that more numbers are “more precise”. It is that the space needs enough independent directions for all the distinctions the task demands.</p></>}
        >
          <p>Open <code>phase2-language/tiny_word2vec.py</code>. Predict: if every word gets only 2 numbers instead of 32, what happens to the similarity of “cat” and “computer”? Then change <code>train()</code> to <code>train(dim=2)</code> at the bottom of the file and check.</p>
        </Exercise>

        <ExplainBack
          id="embeddings-explain"
          prompt="Nothing in the training code rewards “cat” and “dog” for being close. Explain, in your own words, why they end up close anyway."
          modelAnswer={<p>The model is trained to predict which words appear near a given word, and the only thing it knows about the given word is its embedding row. “cat” and “dog” appear in almost the same contexts, so the model needs to produce almost the same predictions for both. Everything after the lookup is shared, so the easiest way to get the same output is to have nearly the same input, meaning nearly the same vector. Gradient descent finds that solution because it lowers the loss. Similarity is a side effect of a prediction task, not something anyone programmed.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'Why can a model not use the token ID as its input number?',
            options: ['The network does arithmetic, so it would treat ID 464 as “more” than ID 12 and as similar to ID 465, although IDs are arbitrary labels', 'IDs are too large to fit in memory', 'IDs are reassigned every time the model runs, so the network could never learn what a given number stands for', 'Because IDs are integers and networks need fractions'],
            answer: 0,
            explain: 'An ID is a name, like a primary key. Arithmetic on names is meaningless.',
          },
          {
            q: 'Mechanically, what does an embedding layer do with token ID 464?',
            options: ['Runs a small neural network on the number 464', 'Converts 464 to binary', 'Returns row 464 of a learned matrix', 'Looks the word up in a dictionary of definitions'],
            answer: 2,
            explain: 'It is an array index. It equals multiplying a one-hot vector by the matrix, which is why it can be trained like any other layer.',
          },
          {
            q: 'Who decides the coordinates of “cat” in a real model?',
            options: ['Linguists label the most important dimensions', 'They are copied from a dictionary', 'The tokenizer assigns them', 'Nobody: they start random and gradient descent moves them to reduce a prediction loss'],
            answer: 3,
            explain: 'Words used in similar contexts need similar predictions, and the cheapest way to get those is similar vectors.',
          },
          {
            q: 'Why do real models use thousands of dimensions rather than 2 or 3?',
            options: ['So that humans can label each dimension', 'Words are similar in many independent ways at once, and each needs its own room to vary', 'Because GPUs cannot handle small vectors', 'To make the 2D picture more accurate'],
            answer: 1,
            explain: 'Gender, royalty, animal-ness, part of speech, formality, topic… Two numbers cannot hold them all, as the dim = 2 experiment showed.',
          },
          {
            q: 'A word2vec-style table gives “bank” one vector. What is the problem?',
            options: ['“bank” is too short a word to embed', 'Its vector will be all zeros', 'The same vector is used in “river bank” and “bank account”, so it must be a compromise between unrelated meanings', 'There is no problem: the vector contains both meanings perfectly'],
            answer: 2,
            explain: 'A lookup cannot see the sentence. Something later in the model has to adjust the vector using context. Hold that thought.',
          },
        ]}
      />

      <Remember
        items={[
          <>A token ID is a <b>name, not a quantity</b>. Doing arithmetic on it is meaningless.</>,
          <>An embedding layer is a <b>lookup table</b>: <span className="mono">E[token_id]</span>, one learned row per token. It equals one-hot × matrix, so it trains like any other layer.</>,
          <><b>Nobody assigns the coordinates.</b> They are learned from a prediction task. Tokens used in similar contexts are pushed toward similar vectors.</>,
          <>Similarity is measured with the <b>dot product / cosine</b>. Real spaces have hundreds to thousands of dimensions because words are alike in many independent ways. <b>The 2D picture is only a teaching model.</b></>,
          <>A lookup gives <b>one static vector per token</b>, whatever the sentence. “bank” exposes the flaw. The fix is coming.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Tokenizer' }, { label: 'Embedding table', sub: 'this lesson: the first layer' }, { label: 'Transformer blocks' }, { label: 'Next-token probabilities' }, { label: 'Sampling' }]} active={1} />
        <p>In a GPT the embedding table is the first layer of the network, nothing more. In the repository’s <code>tiny_gpt.py</code> it is one line:</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="the first layer of a GPT">{`
self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # one row per token
...
x = self.tok_emb(idx) + self.pos_emb(pos)                 # look up every token in the prompt
`}</Code>
        <p>(<code>pos_emb</code> adds information about word order. That is a later lesson.)</p>
        <ToyVsReal
          toy={<ul><li>70 words, 32 numbers each</li><li>Trained separately, on a fake task (predict a neighbouring word)</li><li>28 sentences</li><li>The output matrix W is thrown away</li></ul>}
          real={<ul><li>50,000 to 200,000 tokens, 768 to 16,384 numbers each</li><li>No separate step: the table is trained <em>jointly</em> with every other layer, on next-token prediction</li><li>Trillions of tokens of text</li><li>The same table is often reused at the top of the model to score the next token. That trick is called weight tying, and you will build it in <a href="#/lesson/build-gpt">Build a GPT</a></li></ul>}
        />
        <Callout kind="established">The mechanism is the same: rows of a matrix, looked up by token ID, shaped by gradients from a prediction task. Modern LLMs do not run word2vec. They do not need to, because next-token prediction shapes the table in the same way.</Callout>
        <Callout kind="model" label="About the famous arithmetic">
          “king − man + woman ≈ queen” is real but oversold. It is <b>approximate</b>: the result lands <em>near</em> queen, and standard evaluations exclude the three input words from the candidates (otherwise the nearest word is often just “king”). The well-known examples are the ones that worked. Many analogies fail. Treat it as evidence that some directions carry consistent meaning, not as a reasoning engine.
        </Callout>
        <Callout kind="established" label="Embeddings inherit the training text, including its biases">
          The same geometry that captures “king is to queen as man is to woman” also captured, in word2vec trained on news text, “man is to computer programmer as woman is to homemaker” (Bolukbasi et al., 2016). Embeddings reflect how words are used in the corpus. They are a mirror of text, not of truth.
        </Callout>
        <Callout kind="dev">You will meet “embedding models” again when we build <a href="#/lesson/rag">retrieval (RAG)</a>. Those produce one vector for a whole sentence or document, so that search becomes nearest-neighbour lookup. Same idea, bigger unit. Token embeddings (this lesson) are an internal layer of the LLM. Sentence embeddings are a separate model’s output.</Callout>
        <h3>The loose end: one vector per token is not enough</h3>
        <p>Where did you put “bank”? Any single spot is wrong. In “the river bank” it should sit with water. In “the money bank” it should sit with finance. A lookup table cannot see the sentence, so “bank” gets one vector: an awkward average that is wrong in every actual sentence.</p>
        <p>What we need is a way for a token’s vector to be <em>adjusted by the tokens around it</em>. That mechanism is the centre of this course, and we will get there in Part 6. First, we need a model that actually predicts something.</p>
      </RealLLM>

      <BeforeMovingOn
        id="part-4"
        questions={[
          {
            q: 'From Part 0: at its core, what does an LLM compute when you send it a prompt?',
            options: ['A probability for every possible next token, again and again', 'It searches a database of answers', 'A parse tree of your sentence', 'The most similar document from its training set'],
            answer: 0,
            explain: 'Everything in this part (tokens, IDs, vectors) exists to feed that one computation.',
          },
          {
            q: 'From Part 1: the dot product of two embedding vectors is large and positive. What does that tell you?',
            options: ['The two tokens have nearby IDs', 'One vector is the negative of the other', 'The tokens are spelled similarly', 'The vectors point in a similar direction (and/or are long)'],
            answer: 3,
            explain: 'Direction agreement times lengths. Dividing by the lengths gives cosine similarity, which keeps only the direction.',
          },
          {
            q: 'From Part 1: the word2vec model ends with a softmax over 70 words. What does softmax guarantee about its output?',
            options: ['Exactly one entry is 1 and the rest are 0', 'All entries are positive and sum to 1', 'The entries are sorted', 'The largest score is unchanged'],
            answer: 1,
            explain: 'That is what makes the output usable as probabilities, and what cross-entropy needs.',
          },
          {
            q: 'From Part 2: in one training step, how does a row of the embedding table change?',
            options: ['It is replaced by the average of its neighbours', 'Rows never change after initialisation', 'It is re-drawn at random if the loss went up', 'It moves a small step against the gradient of the loss, scaled by the learning rate'],
            answer: 3,
            explain: 'Same rule as every other parameter: w ← w − learning rate × gradient.',
          },
          {
            q: 'From Part 3: the embedding gradient was computed as d_emb = d_logits @ W.T. Which idea is that?',
            options: ['The chain rule, passing blame backward through the layer that used the embedding', 'Random search: trying small changes to the embedding and keeping the ones that help', 'The softmax formula, turning the scores in W into probabilities', 'Tokenization'],
            answer: 0,
            explain: 'Backpropagation: the blame arriving at the scores is sent back through W to the vector that produced them.',
          },
          {
            q: 'From this part: you type “Zürich”. The tokenizer has never seen “ü”. What does a byte-level BPE tokenizer do, and what reaches the embedding table?',
            options: ['It fails with an unknown-token error', 'It drops the character', 'It falls back to the bytes of “ü”, producing more tokens, and each token ID selects one row of the table', 'It swaps “ü” for the closest known character, “u”, and the embedding table is given that row instead'],
            answer: 2,
            explain: 'Rare strings cost more tokens, but every token ID always has a row.',
          },
        ]}
      >
        <OrderExercise
          id="embeddings-order-pipeline"
          title="From keystrokes to vectors, from memory"
          prompt={<p>Put the steps in the order they happen when a prompt enters a model.</p>}
          correct={['Text: “What is a cat?”', 'Split into characters or bytes', 'Replay the BPE merges in order → tokens', 'Look up each token’s position in the vocabulary → token IDs', 'Look up row E[id] for each ID → one vector per token', 'Matrix multiplications inside the network']}
          solutionNote={<p>Text becomes tokens by replaying a fixed merge list. Tokens become integers. Integers select rows of a learned table. Only from that point on is there anything to multiply.</p>}
        />
      </BeforeMovingOn>
    </Lesson>
  )
}
