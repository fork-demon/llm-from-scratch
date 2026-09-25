import { CodeExercise } from '../components/python'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { CausalMaskLab } from '../interactive/CausalMaskLab'
import { MultiHeadLab } from '../interactive/MultiHeadLab'

export default function MasksAndHeadsLesson() {
  return (
    <Lesson id="masks-and-heads">
      <Why>
        <p className="lede">Monday morning. Kabir leaves a printed sheet on Riya’s desk. “Quick quiz. Ten questions.”</p>
        <p>She turns it over. Under every question, in small grey type, is the answer. She finishes in forty seconds. Ten out of ten.</p>
        <p>Kabir comes back with his coffee. “What did you learn?”</p>
        <p>She thinks about it. “Nothing.”</p>
        <p>“Right. And in a real exam, with no answers printed, you’d fail.” He taps her screen, where Friday’s attention code is still open. “That is what your model is doing.”</p>
        <p>He is right. We train a language model by asking it to <a href="#/lesson/next-token">predict the next token</a>. But the attention from <a href="#/lesson/attention">the last lesson</a> lets every token look at <em>every</em> token, including the next one. The answer is printed right there.</p>
        <p>This lesson adds the two things that turn “attention” into the attention used inside GPT:</p>
        <div className="grid-2">
          <div className="card"><h4 style={{ fontSize: 17, marginBottom: 6 }}>A. The causal mask</h4><p>Stop tokens from looking at their own future. One line of code.</p></div>
          <div className="card"><h4 style={{ fontSize: 17, marginBottom: 6 }}>B. Multiple heads</h4><p>Let each token ask several questions at once instead of just one.</p></div>
        </div>
      </Why>

      <Problem title="Two problems with plain attention">
        <h3>Problem A: the answer is visible</h3>
        <p>Take the training text “the cat sat on a mat”. The model reads it once and must make a prediction at <em>every</em> position:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>position</th><th>has read</th><th>must predict</th></tr></thead>
            <tbody>
              <tr><td className="mono">0</td><td>the</td><td><b>cat</b></td></tr>
              <tr><td className="mono">1</td><td>the cat</td><td><b>sat</b></td></tr>
              <tr><td className="mono">2</td><td>the cat sat</td><td><b>on</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>With plain attention, the row for “cat” can put weight on “sat”. But “sat” is what “cat” is being asked to predict.</p>
        <p><G t="gradient-descent">Gradient descent</G> is relentless. If copying the answer lowers the <G t="loss">loss</G>, the model learns to copy.</p>
        <p>There is a second reason. When the model <em>generates</em> text, the next token does not exist yet. There is nothing to peek at. A model trained with peeking would be practising for an exam it never gets to sit.</p>
        <WhyExists
          problem="During training, position t must predict token t+1, but attention lets position t look straight at token t+1."
          naive="Feed the model one prefix at a time: first “the”, then “the cat”, then “the cat sat”… Each run only contains the past, so nothing can leak."
          fails="A sequence of 1024 tokens would need 1024 separate forward passes, over prefixes that average half the full length. Training would be hundreds of times slower."
          idea="Feed the whole sequence once, but inside attention forbid every token from looking to its right. Each row then behaves exactly as if it had been given only its prefix."
          tradeoff="A token can never use later context, even when that would help understanding. For a model whose job is to continue text, that is precisely the rule we want."
        />

        <h3>Problem B: one blend per token is not enough</h3>
        <p>Softmax gives each token <b>one</b> row of weights that sums to 1. One row means one blend of the other tokens.</p>
        <p>But look at “it” in “The animal didn’t cross the road because it was tired”. To be useful, “it” would like to know several things at once:</p>
        <ul>
          <li>What do I refer to? (animal)</li>
          <li>What word came just before me? (because)</li>
          <li>What is this sentence about in general? (a bit of everything)</li>
        </ul>
        <p>If “it” spends 80% of its weight on “animal”, only 20% is left for everything else. One budget, several needs. They dilute each other.</p>
        <p>Riya recognises this from hiring. One interviewer cannot judge coding, communication and design in the same hour. That is why Paisa Pal uses a panel.</p>
      </Problem>

      <MentalModel>
        <h3>A. The mask</h3>
        <p>Kabir draws the score table from the last lesson on the whiteboard: one row and one column for each of the T tokens in the sequence. The row is who is looking. The column is who is looked at.</p>
        <p>Every cell <em>to the right of the diagonal</em> is a token looking at something that comes after it. Kabir hatches that whole triangle out with his marker.</p>
        <p>What remains is a staircase. The first token sees only itself, the second sees two tokens, and so on.</p>
        <Term
          name="Causal mask"
          plain={<>A rule inside attention: a token may look at itself and at earlier tokens, never at later ones. “Causal” because information only flows from past to future.</>}
          example={<>In “the cat sat”, the row for “cat” may use “the” and “cat”. Its weight on “sat” is forced to exactly 0.</>}
          formal={<>Before softmax, set score(i, j) = −∞ wherever j &gt; i.</>}
        />
        <Callout kind="analogy">
          It is like an exam hall where you must cover the rest of the page with a sheet of paper and slide it down one line at a time. Every line is a fresh question, answered with only what is above it.
          <br /><br />
          Where the analogy stops: the model does not slide anything. All lines are answered <em>simultaneously</em>, each with its own view of the page. That simultaneity is the whole point.
        </Callout>

        <h3>B. The heads</h3>
        <p>If one search is not enough, run several. Give each token a few <em>small</em> attention mechanisms that work side by side. Each has its own idea of what is relevant. Each one is called a <b>head</b>.</p>
        <Callout kind="analogy">
          Think of an interview panel. One interviewer listens for coding, one for communication, one for system design. Each takes their own notes on the same candidate. At the end, the notes are put together into one decision.
          <br /><br />
          Where the analogy stops: nobody tells a head what to listen for. Each head’s focus is learned, because it lowered the loss. In a trained model most heads do not have a clean, nameable job like “communication”.
        </Callout>
        <Term
          name="Multi-head attention"
          plain={<>Several small attention operations run in parallel on the same tokens. Each head has its own query, key and value projections, so each can look for something different. Their results are joined together at the end.</>}
          example={<>With D = 8 numbers per token and 2 heads, each head works with 4 numbers per token. Each head also builds its own table of weights, one row and one column per token.</>}
          formal={<>Split Q, K, V into h slices of width D/h, run attention on each slice, concatenate the h outputs back to width D, multiply by a learned matrix W<sub>O</sub>.</>}
        />
        <Callout kind="dev">
          Think of one SQL query with a single <code>WHERE</code> clause versus several independent queries run in parallel whose results you then join. Same table, different filters. The important detail: the heads <b>share the D numbers between them</b>. Four heads on D = 64 means four heads of width 16, not four heads of width 64. The cost stays the same.
        </Callout>
      </MentalModel>

      <TryIt title="See the mask, then see the heads">
        <h3>A. What does the mask actually do?</h3>
        <p>Click “cat”. Switch the mask off and on. Then open “What leaks?” and try the head that learned to cheat.</p>
        <CausalMaskLab />
        <Callout kind="idea">
          The mask is not only about honesty. It is about <b>efficiency</b>. Because row t can only see tokens 0…t, one forward pass over T tokens gives T valid training examples at once. Without the mask you would have to run each prefix separately.
        </Callout>

        <h3>B. Several heads on one sentence</h3>
        <p>Same sentence as last lesson. Keep “it” selected and switch between the heads.</p>
        <MultiHeadLab />
        <Callout kind="research">
          <b>The head labels above are illustrations, written by hand.</b> What about real, trained models? Interpretability researchers have found some heads with clear, repeatable roles: <em>previous-token heads</em>, and <em>induction heads</em> that find an earlier occurrence of the current token and look at what followed it. Heads that park most of their weight on the first token are also commonly observed. But most heads in a large model do not have a clean human-readable job, and many do different things in different contexts. Nobody assigns roles: whatever lowers the loss is what each head becomes.
        </Callout>
      </TryIt>

      <Numbers>
        <p>One row, by hand. The sentence is “the cat sat”. We are the row for <b>“cat”</b>, which must predict “sat”. Suppose its three scores (already divided by √d) are <span className="mono">[1, 2, 3]</span>.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th></th><th>the</th><th>cat</th><th>sat (the answer)</th><th>sum</th></tr></thead>
            <tbody>
              <tr><td>score</td><td className="mono">1</td><td className="mono">2</td><td className="mono">3</td><td></td></tr>
              <tr><td>no mask: e<sup>score</sup></td><td className="mono">2.72</td><td className="mono">7.39</td><td className="mono">20.09</td><td className="mono">30.19</td></tr>
              <tr><td>no mask: weight</td><td className="mono">0.09</td><td className="mono">0.24</td><td className="mono"><b>0.67</b></td><td className="mono">1</td></tr>
              <tr><td>masked score</td><td className="mono">1</td><td className="mono">2</td><td className="mono">−∞</td><td></td></tr>
              <tr><td>masked: e<sup>score</sup></td><td className="mono">2.72</td><td className="mono">7.39</td><td className="mono">0</td><td className="mono">10.11</td></tr>
              <tr><td>masked: weight</td><td className="mono"><b>0.27</b></td><td className="mono"><b>0.73</b></td><td className="mono"><b>0</b></td><td className="mono">1</td></tr>
            </tbody>
          </table>
        </div>
        <p>Without the mask, “cat” spends 67% of its attention on the very word it is supposed to predict. With the mask, that weight is exactly 0, and the remaining two weights <b>still sum to 1</b>, because the softmax ran <em>after</em> the masking.</p>
        <p>Notice too that “the” and “cat” keep their proportions: 2.72 to 7.39 before, 0.27 to 0.73 after. The mask removes the future. It does not distort the past.</p>
        <h3>Heads: division and nothing else</h3>
        <p>Our tiny GPT has D = 128 numbers per token and 4 heads, so each head works with <span className="mono">128 / 4 = 32</span> numbers. Four heads of 32 hold exactly as many numbers as one head of 128.</p>
      </Numbers>

      <TheMath>
        <p>The mask is a single addition inside the formula you already know:</p>
        <Equation
          label="Masked attention: softmax of scores plus mask, times V"
          symbols={[
            ['M', 'the mask: a T×T table that is 0 on and below the diagonal, and −∞ above it'],
            [<>M<sub>ij</sub> = −∞ when j &gt; i</>, 'token i may not look at a later token j'],
            ['score + (−∞)', 'is −∞, and e to the power −∞ is exactly 0, so that token gets weight 0'],
            ['score + 0', 'is unchanged: the past is left alone'],
          ]}
        >
          Attention = softmax( <span className="q">Q</span> <span className="k">K</span><sup>T</sup> / √d + M ) <span className="v">V</span>
        </Equation>
        <p>And multi-head attention is “do that h times on thinner slices, then join”:</p>
        <Equation
          label="Multi-head attention: concatenate the heads and multiply by W O"
          symbols={[
            ['h', 'the number of heads'],
            [<>head<sub>k</sub></>, <>masked attention computed on slice k of <span className="q">Q</span>, <span className="k">K</span>, <span className="v">V</span>; each slice has width D/h, so inside a head d = D/h</>],
            ['Concat', 'put the h outputs side by side: h slices of width D/h make width D again'],
            [<>W<sub>O</sub></>, 'a learned D×D matrix that mixes the heads’ findings into one vector'],
          ]}
        >
          MultiHead(x) = Concat( head<sub>1</sub>, …, head<sub>h</sub> ) W<sub>O</sub>
        </Equation>
        <DeepDive title="Is it really the same cost as one big head?">
          <p><b>Parameters: identical.</b> W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> are still D×D each, whatever h is. Slicing the result into heads is a reshape.</p>
          <p>What about W<sub>O</sub>? The small <code>attention()</code> function of the last lesson left it out to stay short. But a real GPT attention layer ends with this D×D output projection whether it has one head or twelve (<code>tiny_gpt.py</code> calls it <code>proj</code>). So a GPT attention layer has 4 × D² weights, plus biases, for any h. What changes with several heads is the <em>job</em> of W<sub>O</sub>: it is now also the place where the heads’ separate findings get mixed.</p>
          <p><b>Multiplications for the scores: identical.</b> One head: T×T dot products of length D. With h heads: h × T×T dot products of length D/h. Both are T²·D multiplications.</p>
          <p><b>Not identical:</b> there are now h tables of T×T weights instead of one, so the memory for those tables grows with h. That is one reason long contexts are expensive, and one motivation for <G t="flash-attention">FlashAttention</G>, which avoids storing the tables.</p>
        </DeepDive>
        <DeepDive title="Does each head really have “its own” Wq, if there is only one Wq matrix?">
          <p>Yes. Head k only ever sees columns k·(D/h) to (k+1)·(D/h) of <code>x @ Wq</code>. Those columns are produced by the matching columns of Wq and by nothing else. So one D×D matrix is just h separate D×(D/h) matrices stored side by side. Storing them together lets the GPU do one big matrix multiply instead of h small ones.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <h3>A. The mask: two lines</h3>
        <p>This is the part of <code>attention()</code> that last lesson asked you to ignore.</p>
        <Code source="phase3-transformers/attention_numpy.py" title="inside attention(), between the scores and the softmax">{`
scores = Q @ K.T / np.sqrt(D)   # (T, T)
if causal:
    # forbid looking at the future: upper triangle -> -inf -> softmax 0
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -1e9, scores)

weights = softmax(scores)       # (T, T) rows sum to 1
`}</Code>
        <p><code>np.triu(…, k=1)</code> means “upper triangle, starting one step <em>above</em> the diagonal”. So the diagonal stays visible: a token may look at itself. <code>np.where(mask, -1e9, scores)</code> reads: where the mask is true use −1e9, elsewhere keep the score.</p>
        <p>Why −1e9 and not −∞? e<sup>−1000000000</sup> is 0 in floating point, so the result is the same, and it avoids a division of 0 by 0 if a whole row were ever masked. The PyTorch version in <code>tiny_gpt.py</code> uses a true <code>float("-inf")</code>.</p>

        <h3>B. Multi-head, one step at a time</h3>
        <Code title="Step 1: the same three projections as before">{`
T, D = x.shape
hd = D // n_heads       # per-head dimension, e.g. 8 // 2 = 4

Q = x @ Wq              # (T, D)
K = x @ Wk
V = x @ Wv
`}</Code>
        <Code title="Step 2: give each head its slice">{`
# reshape (T, D) -> (n_heads, T, hd)
def split(M):
    return M.reshape(T, n_heads, hd).transpose(1, 0, 2)
Qh, Kh, Vh = split(Q), split(K), split(V)      # (H, T, hd)
`}</Code>
        <p><code>reshape(T, n_heads, hd)</code> cuts each token’s D numbers into H groups of hd. <code>transpose(1, 0, 2)</code> swaps the first two axes so that the head comes first: “for each head, a (T, hd) matrix”. No number changes. This is step 2 of the lab.</p>
        <Code title="Step 3: ordinary attention, for all heads at once">{`
scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(hd)   # (H, T, T)
if causal:
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -1e9, scores)
weights = softmax(scores)                            # (H, T, T)
out = weights @ Vh                                   # (H, T, hd)
`}</Code>
        <p>When <code>@</code> gets 3-D arrays it treats the first axis as “a stack of matrices” and multiplies them pair by pair. So this one line runs H separate attentions, with no loop. Note the scale is now √hd: the keys inside a head have length hd, not D.</p>
        <Code title="Step 4: join the heads, then mix">{`
# concat heads back: (H, T, hd) -> (T, D), then one final mixing matrix
out = out.transpose(1, 0, 2).reshape(T, D)
return out @ Wo, weights                             # (T, D)
`}</Code>
        <p>The complete function in the repository is these four steps and nothing else (a few comments are shortened here):</p>
        <Code source="phase3-transformers/attention_numpy.py" title="multi_head_attention">{`
def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    T, D = x.shape
    hd = D // n_heads
    Q, K, V = x @ Wq, x @ Wk, x @ Wv                     # (T, D)

    def split(M):
        return M.reshape(T, n_heads, hd).transpose(1, 0, 2)
    Qh, Kh, Vh = split(Q), split(K), split(V)            # (H, T, hd)

    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(hd)    # (H, T, T)
    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores)                            # (H, T, T)
    out = weights @ Vh                                   # (H, T, hd)
    out = out.transpose(1, 0, 2).reshape(T, D)           # (T, D)
    return out @ Wo, weights
`}</Code>
        <h3>C. One more axis: a batch of sequences</h3>
        <p>So far <code>x</code> is one sequence, shape <code>(T, D)</code>. Training code feeds many sequences at once, to keep the GPU busy. That adds a batch axis in front: <code>(B, T, D)</code>, meaning B sequences, each of T tokens, each token D numbers.</p>
        <Code title="the same attention, on B sequences at once">{`
x.shape                                   # (B, T, D)
Q = x @ Wq                                # (B, T, D)  Wq is (D, D): same weights for every sequence
K = x @ Wk                                # (B, T, D)
scores = Q @ K.transpose(0, 2, 1)         # (B, T, T)  one score table per sequence
mask = np.triu(np.ones((T, T), dtype=bool), k=1)   # (T, T)
scores = np.where(mask, -1e9, scores)     # (T, T) mask applied to all B tables
`}</Code>
        <p>Two lines there mix shapes that do not match: <code>(B, T, D) @ (D, D)</code>, and a <code>(T, T)</code> mask against <code>(B, T, T)</code> scores. NumPy allows this through a rule called <b>broadcasting</b>.</p>
        <p>The rule: line the two shapes up from the right. Where one array is missing an axis, or has size 1 there, NumPy reuses it along that axis. So the one <code>(T, T)</code> mask is reused for every sequence in the batch, and nothing is copied in memory.</p>
        <p>You will see exactly this in <code>tiny_gpt.py</code>. Its attention works on <code>(B, T, D)</code>, splits into heads as <code>(B, H, T, hd)</code>, and stores the mask as <code>(1, 1, T, T)</code> so that it broadcasts over both the batch and the heads.</p>
        <Callout kind="dev">Most bugs in Transformer code are shape bugs. Get into the habit of this file: write the shape as a comment at the end of every line. If you cannot write the comment, you do not yet understand the line.</Callout>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check.</p>
        <ul>
          <li><b>Mask lab, “cheat” mode, mask off.</b> Drag the strength from 0 to 8. At what score does the row put more than 90% on the answer? Now switch the mask on and drag again. Does anything change?</li>
          <li><b>Mask lab, select “the”</b> (the first token) with the mask on. What is its row of weights? Why can it never be anything else?</li>
          <li><b>Heads lab, part A.</b> Select “road”, then click “one head for everything”. Which of the four patterns can you still recognise in the blended row?</li>
          <li><b>Heads lab, part B.</b> Go to step 3 and switch between 1, 2 and 4 heads. With 4 heads each head has only <em>one</em> number per token to work with. Look at the readout: did the number of weights change?</li>
          <li><b>In Python:</b> in <code>attention()</code> change <code>k=1</code> to <code>k=0</code> and run the file. Look at the first row of the causal matrix in section 2. Explain what you see before reading the debug exercise below.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="masks-and-heads-code-multihead" />
        <Exercise
          id="masks-and-heads-calc-mask"
          type="calculate"
          title="One masked row"
          answer={{ value: 0.73, tolerance: 0.015 }}
          answerLabel="weight on token 0"
          hints={[
            'We are row 1, so tokens 2 and 3 are in the future. What score do they get after masking, and what weight does softmax give that score?',
            'Tokens 2 and 3 get −∞, so weight exactly 0. Only the scores 2 and 1 take part in the softmax.',
            'weight on token 0 = e² / (e² + e¹) = 7.39 / (7.39 + 2.72).',
          ]}
          solution={<><p>Masked scores: [2, 1, −∞, −∞]. Tokens 2 and 3 get weight <b>exactly 0</b>, however large their scores were. Token 0: 7.39 / 10.11 ≈ <b>0.73</b>, token 1: 0.27.</p><p>The scores 4 and 3 were the largest in the row, and they count for nothing. That is the mask doing its job: the most tempting tokens are the forbidden ones.</p></>}
        >
          <p>A sequence has 4 tokens. The scores in row 1 (the second token) are <code>[2, 1, 4, 3]</code>. With the causal mask on: first, what weight does token 2 get? Then compute the weight on token 0. (Two decimals.)</p>
        </Exercise>

        <Exercise
          id="masks-and-heads-calc-heads"
          type="calculate"
          title="How wide is a head?"
          answer={{ value: 64, tolerance: 0 }}
          answerLabel="numbers per head"
          hints={['The heads share the D numbers equally between them.', 'Per-head width = D / h.']}
          solution={<><p>768 / 12 = <b>64</b>. Each of the 12 heads computes its scores from 64-number queries and keys, and divides them by √64 = 8.</p><p>All 12 heads together hold 12 × 64 = 768 numbers: the same as one big head. More heads means <em>thinner</em> heads, not more numbers.</p></>}
        >
          <p>GPT-2 small uses D = 768 numbers per token and h = 12 heads. How many numbers per token does each head work with?</p>
        </Exercise>

        <Exercise
          id="masks-and-heads-predict"
          type="predict"
          title="Two sentences, one prefix"
          hints={['Which tokens can the row for “sat” see in each sentence?', 'In both sentences “sat” sees exactly “the cat sat”. Nothing to its right can influence it.']}
          solution={<p><b>Identical.</b> With a causal mask, the output at a position depends only on the tokens up to and including that position. “the cat sat” is the same in both sentences, so the first three output rows are the same, number for number. Only the rows from “on” / “down” onwards differ. This property is what later makes the <G t="kv-cache">KV cache</G> possible: results for earlier tokens never need to be recomputed when a new token arrives.</p>}
        >
          <p>You run masked attention on “the cat sat on a mat” and on “the cat sat down quietly”. Compare the output vector for “sat” in the two runs. Identical, similar, or unrelated? Why?</p>
        </Exercise>

        <Exercise
          id="masks-and-heads-debug"
          type="debug"
          title="The mask that came too late"
          hints={[
            'Shapes are fine and nothing crashes. Check a property that every row of attention weights must have.',
            'Softmax made each row sum to 1 over ALL tokens. What happens to the sum when you then delete some entries?',
            'For scores [1, 2, 3] in row 0: softmax gives [0.09, 0.24, 0.67]. After zeroing the future the row is [0.09, 0, 0].',
          ]}
          solution={<><p>The mask is applied <b>after</b> softmax. The future weights do become 0, but the remaining weights are not re-normalised, so rows no longer sum to 1. With scores [1, 2, 3], row 0 sums to 0.09: the first token’s output is its value shrunk to 9%. Early tokens get tiny outputs, late tokens normal ones.</p><p>Fix: mask the <em>scores</em> with −∞ (or −1e9) <em>before</em> softmax. Then softmax does the normalising over the visible tokens only.</p><p>A related bug is <code>np.triu(…, k=0)</code>, which also masks the diagonal: no token can see itself, and the first token can see nothing at all. With −1e9 its row silently becomes uniform over <em>all</em> tokens, future included; with a true −∞ it becomes NaN. One assert catches both bugs: <code>assert np.allclose(weights.sum(-1), 1)</code> together with <code>assert np.allclose(np.triu(weights, k=1), 0)</code>.</p></>}
        >
          <p>A colleague’s model trains, but badly, and the first tokens of every sequence seem to be almost ignored. Find the bug.</p>
          <Code>{`
scores = Q @ K.T / np.sqrt(D)
weights = softmax(scores)
mask = np.triu(np.ones((T, T), dtype=bool), k=1)
weights = np.where(mask, 0.0, weights)
out = weights @ V
`}</Code>
        </Exercise>

        <Exercise
          id="masks-and-heads-implement"
          type="implement"
          title="Change the number of heads"
          hints={[
            'Run python phase3-transformers/attention_numpy.py and find section 3. The settings are at the top of demo_multi_head(): T, D, H = 5, 8, 2.',
            'H = 4 and H = 8 work. For H = 3: what is 8 // 3, and can 8 numbers be cut into 3 equal slices?',
            'After the call, add: assert np.allclose(w.sum(-1), 1) and print(w.shape).',
          ]}
          solution={<><p>With H = 4 you get <code>w.shape == (4, 5, 5)</code>: four different 5×5 patterns from the same input, each row summing to 1, each with zeros above the diagonal. With H = 8 every head works with a single number per token.</p><p>H = 3 crashes: <code>ValueError: cannot reshape array of size 40 into shape (5,3,2)</code>. 8 numbers cannot be cut into 3 equal slices. That is why real configurations always have D divisible by the number of heads (768 = 12 × 64).</p></>}
        >
          <p>Open <code>attention_numpy.py</code> and find <code>demo_multi_head()</code>. Change <code>H</code> from 2 to 4, then 8, then 3. Before each run, predict the shape of <code>w</code> and whether it will run at all. Add an assert that every row of every head sums to 1.</p>
        </Exercise>

        <ExplainBack
          id="masks-and-heads-explain"
          prompt="A teammate asks: “Why do we hide the future during training? Surely more context makes a better model.” Answer in your own words, and include why the mask makes training cheaper, not just more honest."
          modelAnswer={<p>The model is trained to predict token t+1 at every position t. If position t could attend to token t+1, the answer would be part of its input, and gradient descent would learn to copy it rather than predict it. That model would be useless at generation time, when the next token does not exist yet. The mask sets every score for a later token to −∞ before softmax, so its weight is exactly 0 and the rest still sum to 1. Because every row then only depends on its own prefix, one forward pass over T tokens produces T honest predictions at once, instead of needing T separate runs on T prefixes.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'Why are masked scores set to −∞ rather than to 0?',
            options: ['Because e⁰ = 1, a score of 0 still earns a real share of the weight; only e^(−∞) is exactly 0', '0 would crash softmax', 'A score of 0 would work equally well; −∞ is a convention that makes masked cells easy to spot when debugging', 'So that the row sums to less than 1'],
            answer: 0,
            explain: 'Softmax exponentiates. A score of 0 is a perfectly ordinary score. To get weight 0 you need e to the power of minus infinity.',
          },
          {
            q: 'A training sequence has 64 tokens. With a causal mask, how many next-token predictions does one forward pass produce that can be used for training?',
            options: ['1, at the last position', '32', '64, one per position', '64 × 64'],
            answer: 2,
            explain: 'Every row only sees its own prefix, so every row is a valid prediction problem. This is why tiny_gpt.py can compute the loss over all positions at once.',
          },
          {
            q: 'You replace 1 head of width 64 by 4 heads of width 16. What happens to the number of weights in Wq, Wk and Wv?',
            options: ['It becomes 4 times larger', 'It becomes 4 times smaller', 'It depends on the sequence length', 'It stays the same: the heads share the same D numbers'],
            answer: 3,
            explain: 'Heads split the dimensions. Wq, Wk, Wv are still D×D. You gain 4 independent weight tables for free, at the price of thinner queries and keys per head.',
          },
          {
            q: 'What is the job of the final matrix W_O?',
            options: ['It applies the causal mask', 'It mixes the concatenated head outputs, which otherwise sit in separate columns, into one combined vector', 'It turns scores into probabilities', 'It reduces each token’s vector from D numbers to D/h, so the next layer receives one head’s worth of output'],
            answer: 1,
            explain: 'After concatenation, columns 0..D/h−1 came from head 0, and so on. Wo lets every output number draw on every head.',
          },
          {
            q: 'A diagram in a blog post labels one head “the grammar head”. How should you read that?',
            options: ['As a designed feature: engineers assign each head a role before training, and the diagram documents that assignment', 'As proof that the model understands grammar', 'As an illustration. A few head roles have been documented in trained models, but most heads are not cleanly interpretable, and none are assigned by hand', 'As wrong: heads are all identical'],
            answer: 2,
            explain: 'Heads are learned. Some (previous-token heads, induction heads) have been identified by interpretability research; most resist tidy labels.',
          },
        ]}
      />

      <Remember
        items={[
          <>We train by predicting token t+1 at <b>every</b> position. Without a mask, position t can look straight at token t+1 and copy it.</>,
          <>The <b>causal mask</b> sets every score for a later token to −∞ <em>before</em> softmax. Weight exactly 0, and the remaining weights still sum to 1.</>,
          <>The mask also matches generation, where the future does not exist, and it makes training efficient: <b>one pass over T tokens = T training examples</b>.</>,
          <>One softmax row is one blend. <b>Multi-head attention</b> runs h thinner attentions in parallel, each with its own Q/K/V slice and its own weight table, then concatenates and mixes with W<sub>O</sub>.</>,
          <>Heads <b>split</b> the D numbers (width D/h each). Same parameters as one big head. Shapes: (T, D) → (h, T, D/h) → (T, D).</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Attention', sub: 'last lesson' }, { label: 'Mask + heads', sub: 'this lesson' }, { label: 'Transformer block', sub: 'next' }, { label: 'GPT' }, { label: 'Modern LLM' }]} active={1} />
        <ToyVsReal
          toy={<ul><li>6 tokens, random scores, a hand-built “cheating” head</li><li>4 heads with hand-written patterns and friendly names</li><li>D = 4, heads of width 1 to 4</li><li>The mask is rebuilt on every call</li></ul>}
          real={<ul><li>The identical mask, on sequences of thousands of tokens or more</li><li>GPT-2 small: 12 heads × 12 layers = 144 learned heads, none with a name</li><li>Head width is typically 64 to 128; a 70B-class model has 64 query heads per layer</li><li>The mask is precomputed once (<code>tiny_gpt.py</code>) or never materialised at all (fused kernels)</li></ul>}
        />
        <Callout kind="established">This masked, multi-head attention is what “decoder-only Transformer” means. GPT-2 and Llama do this by published design, and as far as is publicly known so do the closed chat models: they all generate text under a causal mask. In <code>tiny_gpt.py</code> you will find today’s function again, line for line (with the batch axis added), as <code>CausalSelfAttention</code>.</Callout>
        <Callout kind="note">One modern change: many current models let several query heads <em>share</em> one set of keys and values (<G t="gqa">grouped-query attention</G>) to save memory during generation. The idea of several parallel heads is unchanged. More in <a href="#/lesson/modern-architecture">Modern LLM architecture</a>.</Callout>
        <p>Riya reruns her Friday model with the mask switched on. Its training loss is worse now. Kabir looks at it and nods. “Good. Now it’s sitting the real exam.”</p>
      </RealLLM>
    </Lesson>
  )
}
