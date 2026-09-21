import { Diagnostic } from '../components/Diagnostic'
import { DIAGNOSTICS } from '../data/diagnostics'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { CodeExercise } from '../components/python'
import { Exercise, ExplainBack } from '../components/exercise'
import { AttentionPlayground } from '../interactive/AttentionPlayground'

export default function AttentionLesson() {
  return (
    <Lesson id="attention">
      <Why>
        <Diagnostic id="diag-attention" part={DIAGNOSTICS["attention"].part} questions={DIAGNOSTICS["attention"].questions} />
        <p className="lede">Read this sentence:</p>
        <div className="card center" style={{ fontFamily: 'var(--serif)', fontSize: 22 }}>
          “The animal didn’t cross the road because <span className="acc"><b>it</b></span> was tired.”
        </div>
        <p>What does “it” refer to? The animal, obviously. Now change one word: “…because it was <em>flooded</em>.” Suddenly “it” is the road.</p>
        <p>You worked that out by looking at <em>other words</em> in the sentence. A model that predicts the next token needs the same ability: the meaning of one token depends on the tokens around it, and <em>which</em> ones matter changes from sentence to sentence.</p>
        <Callout kind="idea">
          The model needs a mechanism that lets one token look at the other tokens and pull in the information that is relevant to it. That mechanism is called <b>attention</b>. It is the central idea of every modern LLM.
        </Callout>
      </Why>

      <Problem>
        <p>You arrive with two loose ends from earlier lessons:</p>
        <div className="grid-2">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>One vector per word is not enough</h4>
            <p>In <a href="#/lesson/embeddings">Embeddings</a>, “bank” got a single vector: an awkward average of riverbanks and finance that is wrong in every actual sentence.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Fixed windows do not scale</h4>
            <p>In <a href="#/lesson/context-wall">the last lesson</a>, more context clearly helped, but gluing a fixed number of previous tokens together cannot reach a word 200 tokens back.</p>
          </div>
        </div>
        <WhyExists
          problem="A token’s meaning depends on other tokens, possibly far away."
          naive="Read left to right, squeezing everything seen so far into one running summary vector (this is what RNNs did)."
          fails="The summary is a fixed size, so early words get overwritten. And step 500 cannot start until step 499 finishes, so GPUs sit idle."
          idea="Skip the summary. Let every token look directly at every other token and decide, pair by pair, how much each one matters."
          tradeoff="Every token compares itself with every other token: work grows with the square of the sequence length."
        />
      </Problem>

      <MentalModel>
        <p>So what would “looking at another token” need? Think about it as a developer. For one token to fetch information from the others, the model needs <b>three different views of the same token</b>:</p>
        <div className="grid-3">
          <div className="card"><span className="chip q">what am I looking for?</span><p style={{ marginTop: 8 }}>“bank” needs to ask something like: <em>is there anything nearby about water or about money?</em></p></div>
          <div className="card"><span className="chip k">what do I contain?</span><p style={{ marginTop: 8 }}>“river” needs to advertise: <em>I am a watery thing.</em> This is what other tokens match against.</p></div>
          <div className="card"><span className="chip v">what do I pass along?</span><p style={{ marginTop: 8 }}>If “river” turns out to be relevant, what information does it actually hand over?</p></div>
        </div>
        <p>These three views have names. You will see them everywhere, always in these colours:</p>
        <Term
          name="Query, Key, Value"
          plain={<>Three vectors made from each token. The <b className="q">query</b> is its search request, the <b className="k">key</b> is the label others match against, the <b className="v">value</b> is the content it hands over.</>}
          example={<>query of “bank” · key of “river” = high score → “bank” takes a lot of “river”’s value.</>}
          formal={<>q = x·W<sub>Q</sub>, k = x·W<sub>K</sub>, v = x·W<sub>V</sub>, where x is the token’s vector and the three W matrices are learned during training.</>}
        />
        <Callout kind="analogy">
          Think of attention as a <b>soft dictionary lookup</b>. A normal <code>dict</code> returns the value for the one key that matches exactly. Attention compares your query with <em>every</em> key, and returns a blend of <em>all</em> the values, weighted by how well each key matched.
          <br /><br />
          This is only an analogy. Nothing is stored or retrieved: queries, keys and values are computed fresh from the tokens in front of the model, by three matrix multiplications. Let’s look at that actual computation.
        </Callout>
      </MentalModel>

      <TryIt title="Watch one token look at the others">
        <p>Start with “the river bank”, keep “bank” selected, and step through. Then switch to “the money bank” and compare step 5.</p>
        <AttentionPlayground />
      </TryIt>

      <Numbers>
        <p>Let’s redo “bank” in “the river bank” by hand, so nothing is hidden. Only two dimensions of the query and keys are non-zero, so we can ignore the rest.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>step</th><th>the</th><th>river</th><th>bank</th></tr></thead>
            <tbody>
              <tr><td><span className="k">key</span></td><td className="mono">[0, 0]</td><td className="mono">[2, 0]</td><td className="mono">[1, 1]</td></tr>
              <tr><td><span className="q">query</span> of “bank” · key</td><td className="mono">2·0 + 2·0 = 0</td><td className="mono">2·2 + 2·0 = 4</td><td className="mono">2·1 + 2·1 = 4</td></tr>
              <tr><td>÷ √4 = 2</td><td className="mono">0</td><td className="mono">2</td><td className="mono">2</td></tr>
              <tr><td>e<sup>score</sup></td><td className="mono">1.00</td><td className="mono">7.39</td><td className="mono">7.39</td></tr>
              <tr><td>÷ total (15.78) = weight</td><td className="mono"><b>0.06</b></td><td className="mono"><b>0.47</b></td><td className="mono"><b>0.47</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>“bank” listens 47% to “river”, 47% to itself, and barely to “the”. Its new “watery” number is <span className="mono">0.06×0 + 0.47×1 + 0.47×0.5 = 0.70</span>, and its “financial” number is <span className="mono">0.47×0.5 = 0.23</span>. It started at 0.5 / 0.5. The context pulled it toward water.</p>
        <p>Those are the same numbers the playground shows. There is no other machinery.</p>
      </Numbers>

      <TheMath>
        <p>You have now done every step. The famous formula is just those steps written on one line, for all tokens at once:</p>
        <Equation
          label="Attention equals softmax of Q K transpose over root d, times V"
          symbols={[
            [<span className="q">Q</span>, 'one row per token: what that token is looking for'],
            [<span className="k">K</span>, 'one row per token: what that token offers for matching'],
            [<span className="v">V</span>, 'one row per token: the information that token can contribute'],
            [<>Q K<sup>T</sup></>, <>every query dotted with every key: a T×T table of match scores (<a href="#/lesson/matrices">a matrix multiply is many dot products at once</a>)</>],
            [<>√d</>, 'd is the length of a key vector; dividing keeps the scores in a range where softmax stays soft'],
            ['softmax', <>turns each row of scores into weights that are positive and sum to 1 (<a href="#/lesson/softmax">lesson 1.3</a>)</>],
            ['… V', 'uses each row of weights to blend the value vectors'],
          ]}
        >
          Attention(<span className="q">Q</span>, <span className="k">K</span>, <span className="v">V</span>) = softmax( <span className="q">Q</span> <span className="k">K</span><sup>T</sup> / √d ) <span className="v">V</span>
        </Equation>
        <p><b>Why a dot product for matching?</b> Because it is the cheapest way to ask “do these two vectors point the same way?”, as you saw in <a href="#/lesson/vectors">lesson 1.1</a>. It is the raw dot product, not the cosine, so a long key or query also raises the score, and a trained model can use that. <b>Why softmax?</b> Because we want a weighted <em>average</em>: weights that are positive and sum to one keep the blended vector in the same range as the values, however many tokens there are.</p>
        <DeepDive title="Why exactly √d, and not some other number?">
          <p>Suppose the entries of q and k are independent random numbers with mean 0 and variance 1. Each product q<sub>i</sub>k<sub>i</sub> then has mean 0 and variance 1. The dot product adds up d of them, so its variance is d and its typical size is √d. With d = 64 that means scores around ±8, and e⁸ ≈ 3000: softmax would put nearly all weight on one token.</p>
          <p>When softmax saturates like that, its gradient is almost zero, and training stalls. Dividing by √d brings the variance back to 1 regardless of d. This is the argument the original Transformer paper gives. It describes random vectors at the start of training, not a law that trained models obey. Turn scaling off in the playground to see the weights sharpen.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>We will build it one line at a time. <code>x</code> holds one row per token, shape <code>(T, D)</code>.</p>
        <Code title="Step 1: three views of every token">{`
Q = x @ Wq      # (T, D)  what each token is looking for
K = x @ Wk      # (T, D)  what each token can be matched on
V = x @ Wv      # (T, D)  what each token passes along
`}</Code>
        <p><code>Wq</code>, <code>Wk</code>, <code>Wv</code> are <G t="parameters">parameters</G>: they start random and are learned by <G t="gradient-descent">gradient descent</G>, like every other weight.</p>
        <Code title="Step 2: how well does every query match every key?">{`
scores = Q @ K.T / np.sqrt(D)    # (T, T)  cell (i, j): how much token i cares about token j
`}</Code>
        <Code title="Step 3: scores become mixing weights">{`
weights = softmax(scores)        # (T, T)  every row sums to 1
`}</Code>
        <Code title="Step 4: blend the values">{`
out = weights @ V                # (T, D)  each token = weighted blend of values
`}</Code>
        <p>Put together, this is the function from the repository, unchanged:</p>
        <Code source="phase3-transformers/attention_numpy.py" title="single-head self-attention">{`
def attention(x, Wq, Wk, Wv, causal=False):
    T, D = x.shape
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    scores = Q @ K.T / np.sqrt(D)
    if causal:   # next lesson: forbid looking at the future
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)

    weights = softmax(scores)
    out = weights @ V
    return out, weights
`}</Code>
        <Callout kind="dev">Notice what is <em>not</em> here: no loop over tokens. All T² comparisons happen in one matrix multiply. That is why Transformers train fast on GPUs where RNNs could not.</Callout>
      </CodeIt>

      <BreakIt>
        <p>Go back to the playground and try to break things. Predict first, then check.</p>
        <ul>
          <li><b>Zero out a query.</b> In step 1, set every number in the Q row of “bank” to 0. What do the weights become? (Every score is 0, so softmax gives equal weights: with no question, it listens to everyone equally.)</li>
          <li><b>Turn off √d scaling</b> and look at step 4. The weights get sharper.</li>
          <li><b>Turn off the causal mask</b> and select “the”. Now the first token can see “river” and “bank”.</li>
          <li><b>Shuffle the words</b> in your own sentence, e.g. “mat the on sat cat the”, with the causal mask turned off. Each word keeps exactly the same weights towards each other word. (With the mask on, the raw scores are still identical. Only the set of tokens each word is allowed to see changes.) Attention by itself has <em>no idea about word order</em>. We fix that in <a href="#/lesson/transformer-block">The Transformer block</a>.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="attention-code-attention" />
        <Exercise
          id="attention-calc"
          type="calculate"
          title="Attention by hand"
          answer={{ value: 0.88, tolerance: 0.015 }}
          answerLabel="weight on token B"
          hints={[
            'Work out the two dot products first: q·k_A and q·k_B.',
            'q·k_A = 1×1 + 0×0 = 1 and q·k_B = 1×3 + 0×1 = 3. No scaling in this exercise, so these are the scores.',
            'softmax([1, 3]): weight on B = e³ / (e¹ + e³) = 20.09 / (2.72 + 20.09).',
          ]}
          solution={<><p>Scores are [1, 3]. Softmax: e¹ = 2.72, e³ = 20.09, total 22.80. Weight on B = 20.09 / 22.80 ≈ <b>0.88</b>.</p><p>A score gap of just 2 already gives B 88% of the weight. Softmax magnifies differences, which is exactly why scores must be kept small with √d.</p></>}
        >
          <p>A token has query <code>q = [1, 0]</code>. There are two tokens to look at, with keys <code>k_A = [1, 0]</code> and <code>k_B = [3, 1]</code>. Skip the √d scaling. After softmax, what weight does the token put on B? (Two decimals.)</p>
        </Exercise>

        <Exercise
          id="attention-predict"
          type="predict"
          title="All keys identical"
          hints={['If all keys are the same vector, what can you say about the scores in one row?', 'Equal scores in, equal weights out: softmax of [c, c, c] is [⅓, ⅓, ⅓].']}
          solution={<p>Every query gets the same score against every key, so every row of weights is uniform. The output for every token is simply the plain average of all values. Attention degenerates into “average everything”. The model only gets selective behaviour when keys differ in ways queries can pick up on.</p>}
        >
          <p>Predict: if every token had exactly the same key vector, what would the attention weights look like, and what would each token’s output be? Then test it by editing K in the playground.</p>
        </Exercise>

        <Exercise
          id="attention-debug"
          type="debug"
          title="The rows do not sum to 1"
          hints={['Look at which direction softmax runs in.', 'scores has shape (T, T). Row i holds “how much token i cares about each j”. Which axis must sum to 1?']}
          solution={<><p><code>axis=0</code> normalises each <em>column</em>. We need each <em>row</em> (each looking token) to sum to 1, so it must be <code>axis=-1</code> (or <code>axis=1</code>).</p><p>This bug is nasty because nothing crashes: shapes are all still (T, T). It also quietly leaks information from later tokens even with a causal mask. A quick <code>assert np.allclose(weights.sum(-1), 1)</code> catches it.</p></>}
        >
          <p>A colleague wrote this and the model trains badly. What is wrong?</p>
          <Code>{`
e = np.exp(scores - scores.max())
weights = e / e.sum(axis=0, keepdims=True)
out = weights @ V
`}</Code>
        </Exercise>

        <Exercise
          id="attention-implement"
          type="implement"
          title="Run it and change it"
          hints={['Run python phase3-transformers/attention_numpy.py and find section 4 in the output.', 'In demo_disambiguation(), add "loan": np.array([0.0, 1.0, 0.8, 0.0]) to vocab and add a context ["the", "loan", "bank"].']}
          solution={<p>“bank” next to “loan” behaves just like “bank” next to “money”: its output has financial &gt; watery. You changed no weights. The same Wq/Wk that handled “money” also handle “loan”, because they act on <em>properties of the vectors</em> (the financial dimension), not on specific words. That is how attention generalises to sentences it has never seen.</p>}
        >
          <p>Open <code>attention_numpy.py</code>, run it, and find the river/money demo. Add a new word “loan” with a vector of your choosing and the context <code>["the", "loan", "bank"]</code>. Before you run it: which way will “bank” be pulled?</p>
        </Exercise>

        <ExplainBack
          id="attention-explain"
          prompt="A teammate says: “Attention is just some complicated matrix multiplication with Q, K and V.” Explain to them in plain words why there are three different vectors per token and what each multiplication achieves."
          modelAnswer={<p>Each token needs to fetch relevant information from other tokens. To do that it needs a question (query), and every token needs a label to be matched against (key) and some content to hand over (value). These are different jobs, so they get different learned projections. Q·Kᵀ compares every question with every label to get match scores. Softmax turns each token’s scores into percentages. Multiplying by V uses those percentages to blend the content. The result: each token’s vector is updated with information from the tokens that matter to it.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'Why does each token get a separate query and key, instead of just comparing embeddings directly?',
            options: ['To make the model bigger', 'Because “what I am looking for” and “what I offer” are different things; with one vector, every token would mostly match itself', 'Because dot products only work on different vectors', 'It is a historical accident with no function'],
            answer: 1,
            explain: 'A pronoun is looking for a noun, but it is not itself a noun. Separate projections let “what I want” differ from “what I am”.',
          },
          {
            q: 'What does softmax contribute to attention?',
            options: ['It makes the scores larger', 'It removes negative values from V', 'It turns a row of arbitrary scores into positive weights that sum to 1, so the output is a weighted average', 'It sorts tokens by importance'],
            answer: 2,
            explain: 'A weighted average keeps outputs in a sensible range no matter how long the sequence is.',
          },
          {
            q: 'The output vector for a token is…',
            options: ['the value vector of the single best-matching token', 'a weighted blend of all value vectors it is allowed to see', 'the sum of all the keys', 'its query multiplied by its key'],
            answer: 1,
            explain: 'It is a soft lookup: everything contributes, in proportion to its weight.',
          },
          {
            q: 'You shuffle the input tokens. With no positional information and no causal mask, what happens to the attention weight between “cat” and “sat”?',
            options: ['It stays exactly the same', 'It drops to zero', 'It depends on the new distance', 'It becomes uniform'],
            answer: 0,
            explain: 'The score only depends on the two vectors, not on where they sit. That is why Transformers must add positional information separately.',
          },
          {
            q: 'Where do W_Q, W_K and W_V come from?',
            options: ['They are designed by hand for each language', 'They are computed from the prompt', 'They are parameters learned by gradient descent, because useful attention patterns reduce next-token loss', 'They are copied from the embedding table'],
            answer: 2,
            explain: 'Nobody tells the model what to attend to. Patterns that help prediction get reinforced by training.',
          },
        ]}
      />

      <Remember
        items={[
          <>Attention exists because <b>a token’s meaning depends on other tokens</b>, and which ones matter changes every time.</>,
          <><b className="q">Query</b> = what I am looking for. <b className="k">Key</b> = what I can be matched on. <b className="v">Value</b> = what I pass along. All three are learned projections of the same token vector.</>,
          <>The recipe: <b>score</b> (q·k) → <b>scale</b> (÷√d) → <b>softmax</b> (weights sum to 1) → <b>blend</b> (weights × V).</>,
          <>The output is a <b>context-aware</b> version of each token: “bank” near “river” ends up as a different vector from “bank” near “money”.</>,
          <>Cost: every token looks at every token, so work grows with <b>T²</b>. No loop over positions, so training and prompt reading run in parallel across all tokens. (Generating is still one token at a time.)</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'This lesson', sub: 'one head, 4 numbers per token' }, { label: 'Multi-head', sub: 'next lesson' }, { label: 'Transformer block' }, { label: 'GPT' }, { label: 'Modern LLM' }]} active={0} />
        <ToyVsReal
          toy={<ul><li>4 numbers per token, hand-picked so you can read them</li><li>Wq and Wk written by hand to make the point</li><li>3 tokens</li><li>one attention operation</li></ul>}
          real={<ul><li>Thousands of numbers per token; no dimension has a clean human meaning</li><li>All weights learned from data; nobody chooses what to attend to</li><li>Thousands to millions of tokens</li><li>Dozens of heads in each of dozens of layers, plus a causal mask</li></ul>}
        />
        <Callout kind="established">The computation itself is identical. softmax(QKᵀ/√d)V, as you just coded it, is what runs inside GPT-2 and Llama, whose code is public, and by every public account inside closed models such as Claude and Gemini too. Production systems mostly change how it is <em>executed</em>, not what it computes: fused GPU kernels such as <G t="flash-attention">FlashAttention</G> get the same result without ever storing the full T × T table (so memory stops growing with T², while the arithmetic still does), and the <G t="kv-cache">KV cache</G> avoids recomputing keys and values. Some variants you will meet in <a href="#/lesson/modern-architecture">Modern LLM architecture</a> share keys and values between heads or limit how far back a token may look. The formula per head stays this one.</Callout>
        <Callout kind="research">What individual attention heads “mean” in a trained model is an open research area. Some heads have been found with clear roles (for example, copying a token that followed an earlier occurrence of the current token), but most are not cleanly interpretable. Treat any diagram that labels a head “the grammar head” as an illustration.</Callout>
      </RealLLM>
    </Lesson>
  )
}
