import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM, BeforeMovingOn } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack, OrderExercise } from '../components/exercise'
import { StackDepthLab, TransformerBlockExplorer } from '../interactive/TransformerBlockExplorer'
import { BlockSchematic } from '../illustrations/BlockSchematic'

export default function TransformerBlockLesson() {
  return (
    <Lesson id="transformer-block">
      <Why>
        <p className="lede">Two headlines:</p>
        <div className="card center" style={{ fontFamily: 'var(--serif)', fontSize: 22 }}>
          “Dog bites man.” &nbsp;&nbsp; “Man bites dog.”
        </div>
        <p>Same three words. Only one of them is news. Yet the attention you have built so far <b>cannot tell them apart</b>: it compares vectors, and it has no idea where in the sentence a vector came from.</p>
        <p>That is one of four gaps between “attention” and a working model. In this lesson we close all four. Each fix is small. Together they form the <b>Transformer block</b>, the unit that GPT repeats 12, 32 or 80 times.</p>
        <Callout kind="idea">
          Nothing in a block is decoration. For every part we will ask: what goes wrong <em>without</em> it? And you will be able to remove it and watch.
        </Callout>
      </Why>

      <Problem title="Four things attention cannot do alone">
        <p>You have masked multi-head attention from <a href="#/lesson/masks-and-heads">the last lesson</a>. Here is what is still missing.</p>

        <h3>1. It is blind to word order</h3>
        <p>The score between “dog” and “bites” depends on those two vectors and on nothing else. Move “dog” to the end and its vector is the same, so the score is the same.</p>
        <WhyExists
          problem="Attention treats the sentence as a bag of vectors. “dog bites man” and “man bites dog” produce the same vectors for the same words."
          naive="Hope the causal mask is enough: after all, it makes each token see a different set of earlier tokens."
          fails="The mask tells a token which words came before it, but not in what order, or how far away. In the first block, “dog bites” and “bites dog” look identical to a third token."
          idea="Give every slot in the sequence (0, 1, 2, …) its own learned vector, and add it to the token’s embedding before the first block. Now “dog in slot 0” and “dog in slot 2” are different inputs."
          tradeoff="A learned table has a fixed number of slots, so the model has a hard maximum length: the context window. Modern models use a different scheme (RoPE) partly for this reason."
        />

        <h3>2. It only mixes, it never computes</h3>
        <p>Look at what attention outputs: a <em>weighted average</em> of value vectors. Averaging moves information between tokens. But an average can never produce anything that was not already in the ingredients.</p>
        <p>Once “it” has gathered information from “animal”, something must <em>process</em> the result. You already own the tool: the <a href="#/lesson/neurons">MLP from the neurons lesson</a>, with its bend in the middle. In a block it is applied to <b>each token separately</b>.</p>
        <Callout kind="model" label="Simplified mental model: communicate, then compute">
          Attention is the meeting: tokens exchange information. The MLP is going back to your desk: each token thinks alone about what it just heard. One block = one meeting + one desk session.
        </Callout>

        <h3>3. Deep stacks garble the signal</h3>
        <p>One round of “communicate, then compute” is not enough, so we stack many. But recall <a href="#/lesson/backprop">backpropagation</a>: the gradient reaches an early layer only after being multiplied by something at <em>every</em> layer above it. Multiply 24 numbers that are mostly below 1 and almost nothing arrives.</p>
        <WhyExists
          problem="In a deep stack, each layer overwrites its input. Going forward, the original token information gets garbled. Going backward, the gradient shrinks or blows up on its way to the early layers."
          naive="x = layer(x), over and over. Each layer must reproduce everything worth keeping AND add something new."
          fails="A freshly initialised layer outputs noise, so layer 1’s careful work is destroyed by layer 2. And the early layers receive almost no gradient, so they barely learn."
          idea="x = x + layer(x). Keep the input and only ADD to it. Each layer now learns a correction, and the plus sign gives the gradient a direct road down to every layer."
          tradeoff="Every layer must keep the same width D, because you can only add vectors of equal length. That is why the shape (T, D) never changes inside a Transformer."
        />

        <h3>4. Numbers drift in size</h3>
        <p>If every block adds something to x, the numbers in x tend to grow. A sub-layer that receives inputs of size 1 today and size 50 after a few training steps is chasing a moving target. Training becomes jumpy, or fails.</p>
        <p>The fix is unglamorous: before each sub-layer, <b>rescale each token’s vector to a standard size</b>. That operation is called layer normalisation.</p>
      </Problem>

      <MentalModel>
        <p>Put the four fixes around attention and you get this picture. Read it once now, from the bottom up, following the coloured line. The interactive below lets you open every part.</p>
        <BlockSchematic />
        <p>Notice what that line never does: it never passes <em>through</em> a sub-layer. Attention and the MLP each read a normalised copy, and what they produce is <b>added</b> back at a ⊕. Everything a block does is a correction to the same running x.</p>
        <Term
          name="Residual connection"
          plain={<>Instead of replacing x with a layer’s output, add the layer’s output to x. The layer only has to work out what to <em>change</em>.</>}
          example={<>x = [1.0, 2.0], the layer outputs [0.1, −0.2]. New x = [1.1, 1.8]. If the layer outputs zeros, x passes through untouched.</>}
          formal={<>x ← x + f(x). The running x that flows from block to block is often called the <b>residual stream</b>.</>}
        />
        <Term
          name="Layer normalisation (LayerNorm)"
          plain={<>Take one token’s vector. Shift it so its numbers average 0, scale it so their spread is 1. Then let the model stretch and shift it again with two learned vectors, in case a different size works better.</>}
          example={<>[10, 20, 60] → subtract the mean 30 → [−20, −10, 30] → divide by the spread 21.6 → [−0.93, −0.46, 1.39].</>}
          formal={<>y = (x − mean(x)) / std(x) · γ + β, computed separately for every token; γ (gain) and β (bias) are learned, starting at 1 and 0.</>}
        />
        <Callout kind="analogy">
          Think of the residual stream as a <b>shared document</b> that travels past a line of editors. Each editor (sub-layer) reads the document and appends a note. Nobody is allowed to throw the document away and start again.
          <br /><br />
          Where the analogy stops: the “notes” are added number by number to the same D numbers, not appended. A later layer can cancel an earlier note by adding its negative. And LayerNorm means each editor reads a volume-adjusted <em>copy</em>, while the original continues untouched.
        </Callout>
        <Callout kind="dev">
          A residual block has the same contract as middleware: same type in, same type out, <code>(T, D) → (T, D)</code>. That is why blocks can be stacked, removed, or counted with a single config number.
        </Callout>
      </MentalModel>

      <TryIt title="Open every box, then remove one">
        <p>Three tokens, four numbers each, real arithmetic. Start by clicking through the parts from top to bottom. Then run the two experiments under the diagram.</p>
        <TransformerBlockExplorer />
        <h3>Transformer block × N</h3>
        <p>The output has the same shape as the input, so blocks snap together like Lego. GPT-2 small stacks 12, our tiny GPT stacks 4. But what happens to the <em>size</em> of the numbers as the stack gets deep?</p>
        <StackDepthLab />
      </TryIt>

      <Numbers>
        <h3>LayerNorm by hand</h3>
        <p>One token’s vector: <span className="mono">[1, 3, 5, 7]</span>.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>step</th><th>calculation</th><th>result</th></tr></thead>
            <tbody>
              <tr><td>mean</td><td className="mono">(1 + 3 + 5 + 7) / 4</td><td className="mono">4</td></tr>
              <tr><td>subtract the mean</td><td className="mono">[1−4, 3−4, 5−4, 7−4]</td><td className="mono">[−3, −1, 1, 3]</td></tr>
              <tr><td>average squared distance (variance)</td><td className="mono">(9 + 1 + 1 + 9) / 4</td><td className="mono">5</td></tr>
              <tr><td>spread (standard deviation)</td><td className="mono">√5</td><td className="mono">2.24</td></tr>
              <tr><td>divide by the spread</td><td className="mono">[−3, −1, 1, 3] / 2.24</td><td className="mono"><b>[−1.34, −0.45, 0.45, 1.34]</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Now try <span className="mono">[100, 300, 500, 700]</span>. Mean 400, spread 224, result: <span className="mono">[−1.34, −0.45, 0.45, 1.34]</span>. <b>Exactly the same.</b> LayerNorm throws away the overall size and offset and keeps only the pattern. Whatever the blocks below have done to the scale, the next sub-layer sees something familiar.</p>
        <p>Finally the learned gain γ and bias β are applied, number by number. They start at 1 and 0, so at the start of training they change nothing.</p>

        <h3>Why the plus sign rescues the gradient</h3>
        <p>Suppose each layer passes on only half of the gradient it receives. After 12 layers:</p>
        <div className="readout">
          <span>without residuals: 0.5<sup>12</sup> = <b>0.0002</b> of the gradient arrives</span>
          <span>through the “+” path: 1<sup>12</sup> = <b>1</b></span>
        </div>
        <p>Why 1? For y = x + f(x), nudging x by a small amount moves y by that amount <em>plus</em> whatever f does. The derivative is <span className="mono">1 + f′(x)</span>. The “1” is the highway: however badly a layer behaves, the gradient from above still passes straight through to the layers below.</p>
        <Callout kind="model">The “half per layer” figure is made up to show the effect. Real per-layer factors vary, and can be above 1 (exploding) as well as below (vanishing). The point that survives: a product of many factors is fragile, a sum with a guaranteed 1 is not.</Callout>
      </Numbers>

      <TheMath>
        <p>The whole block is two lines. Everything else is the definition of the pieces.</p>
        <Equation
          label="Transformer block: x becomes x plus attention of layer norm of x, then x plus MLP of layer norm of x"
          symbols={[
            ['x', 'the residual stream: one row of D numbers per token, shape (T, D)'],
            ['LN', 'layer normalisation, applied to each row separately'],
            ['Attn', <>masked multi-head self-attention (<a href="#/lesson/masks-and-heads">last lesson</a>): the only step where rows interact</>],
            ['MLP', 'the feed-forward network: D → 4D → D with a bend in the middle, applied to each row separately'],
            ['←', '“becomes”: the left side is the new value of x'],
          ]}
        >
          x ← x + Attn( LN<sub>1</sub>(x) )<br />x ← x + MLP( LN<sub>2</sub>(x) )
        </Equation>
        <Equation
          label="Layer norm: y equals x minus mu over sigma times gamma plus beta"
          symbols={[
            ['μ', 'the mean of the D numbers of this one token'],
            ['σ', 'their standard deviation (spread). In code a tiny ε is added so we never divide by 0'],
            ['γ, β', 'learned gain and bias, D numbers each, shared by all tokens'],
          ]}
        >
          LN(x) = (x − μ) / σ · γ + β
        </Equation>
        <DeepDive title="Pre-norm or post-norm? The diagram in the 2017 paper looks different">
          <p>It is different. The original Transformer paper (2017) normalised <em>after</em> the addition: <span className="mono">x ← LN(x + Attn(x))</span>. That is called <b>post-norm</b>.</p>
          <p>GPT-2, our <code>tiny_gpt.py</code> and practically all current LLMs normalise <em>before</em> each sub-layer and leave the residual stream itself untouched: <b>pre-norm</b>, as in the equations above. With post-norm the LayerNorm sits <em>on</em> the highway, so the gradient no longer has a clean “+1” path; deep post-norm models are harder to train and typically need a careful learning-rate warm-up. Pre-norm turned out to be more forgiving, and won.</p>
          <p>A side effect you saw in the depth lab: in pre-norm the residual stream is never normalised, so it grows slowly with depth. That is why there is one extra LayerNorm after the last block.</p>
        </DeepDive>
        <DeepDive title="Why add the position vector instead of appending it?">
          <p>Appending would work too, but it would make every vector longer. Adding keeps the width at D, and in a space with hundreds of dimensions there is plenty of room: training can place “what the token is” and “where it is” in different directions of the same vector, and the W<sub>Q</sub>/W<sub>K</sub> matrices can learn to read out either.</p>
          <p>Honest footnote: a causal mask alone does leak <em>some</em> position information (the first token can only see itself, for instance), and research has shown that causal models without any explicit positions can still learn. Explicit positions make it far easier, and every mainstream model has some form of them.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>We switch from NumPy to PyTorch here, because <code>tiny_gpt.py</code> is written in it. For now read <code>nn.Linear(a, b)</code> as “multiply by an (a, b) weight matrix and add a bias”, and <code>nn.LayerNorm(D)</code> as the calculation you just did by hand. The next lesson introduces PyTorch properly.</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="Piece 1: position vectors, added once at the bottom">{`
self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)    # one row per token id
self.pos_emb = nn.Embedding(cfg.context_len, cfg.n_embd)   # one row per slot 0, 1, 2, ...

pos = torch.arange(T, device=idx.device)       # [0, 1, ..., T-1]
x = self.tok_emb(idx) + self.pos_emb(pos)      # meaning + position
`}</Code>
        <p><code>nn.Embedding</code> is a lookup table, exactly as in <a href="#/lesson/embeddings">Embeddings</a>. The position table is learned the same way as the token table: nobody tells the model what “slot 3” should look like.</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="Piece 2: the feed-forward network">{`
class FeedForward(nn.Module):
    """Module 03's MLP: the 'computation' half of the block.
    Applied to each token independently. Most parameters live here."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),                       # smooth ReLU; same hinge idea
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)
`}</Code>
        <p>Widen to 4×D, bend, narrow back to D. <G t="relu">GELU</G> is a ReLU with a rounded corner. <code>Dropout</code> randomly zeroes some numbers during training to discourage memorising; it is switched off when generating. “Applied to each token independently” falls out of the shapes: a <code>(T, D)</code> input times a <code>(D, 4D)</code> matrix processes every row on its own.</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="Piece 3: the block. Two lines of forward() define the architecture">{`
class Block(nn.Module):
    """communicate (attention) then compute (FFN),
    each wrapped in pre-LayerNorm + residual."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual: block outputs a CORRECTION
        x = x + self.ffn(self.ln2(x))    # gradient highway through the '+'
        return x
`}</Code>
        <p>Compare the two lines of <code>forward</code> with the equation above. They are the same thing. <code>CausalSelfAttention</code> is last lesson’s <code>multi_head_attention</code> translated to PyTorch.</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="Piece 4: × N">{`
self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
`}</Code>
        <Callout kind="dev">N blocks means N <em>separate</em> sets of weights, not one block called N times. The list comprehension constructs a new <code>Block</code> for every layer.</Callout>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check in the explorer.</p>
        <ul>
          <li><b>Positions off, mask off, then tick “man bites dog”.</b> Compare the two output tables in Experiment 1, row by row. Then turn positions back on.</li>
          <li><b>Positions off, mask on.</b> The rows are no longer an exact swap. Why does the mask carry a little information about order? (Think about what the first token can see.)</li>
          <li><b>Remove the residuals</b> and click the final “Output” box. Compare the three output rows with each other. Could a second block still tell “dog” from “man”?</li>
          <li><b>Click LayerNorm 1</b> and read the mean and std before and after, for each token.</li>
          <li><b>Depth lab:</b> set the weight size to 1.5× and N to 24, and read the “without” number. Then 0.5×. When is LayerNorm doing visible work?</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="transformer-block-calc-layernorm"
          type="calculate"
          title="LayerNorm by hand"
          answer={{ value: -1.22, tolerance: 0.015 }}
          answerLabel="first number of the result"
          hints={[
            'Step 1: the mean of [2, 4, 6]. Step 2: subtract it from every number.',
            'Mean = 4, so you have [−2, 0, 2]. Variance = the average of the squares = (4 + 0 + 4) / 3.',
            'Variance = 2.67, spread = √2.67 = 1.63. First number = −2 / 1.63.',
          ]}
          solution={<><p>Mean 4 → [−2, 0, 2]. Variance (4 + 0 + 4)/3 = 2.67, standard deviation 1.63. Result: <b>[−1.22, 0, 1.22]</b>.</p><p>Check: the result has mean 0, and its squares average (1.5 + 0 + 1.5)/3 = 1. Any three equally spaced numbers give this same answer, whether they are [2, 4, 6] or [−50, 0, 50]. LayerNorm keeps the pattern and discards the scale.</p></>}
        >
          <p>Apply layer normalisation to the vector <code>[2, 4, 6]</code> (gain 1, bias 0, ignore ε). What is the <em>first</em> number of the result? (Two decimals, mind the sign.)</p>
        </Exercise>

        <OrderExercise
          id="transformer-block-trace"
          title="Follow one token’s vector through a block"
          prompt={<p>The vector for “it” enters a block. Put the steps in the order they happen (pre-norm, as in <code>tiny_gpt.py</code>).</p>}
          correct={[
            'x enters the block',
            'LayerNorm 1 makes a normalised copy of x',
            'Self-attention: the copy gathers from earlier tokens',
            'Add the attention output to x',
            'LayerNorm 2 makes a normalised copy of the new x',
            'Feed-forward MLP processes the copy, alone',
            'Add the MLP output to x',
            'x leaves with the same shape, into the next block',
          ]}
          solutionNote={<p>Norm, sub-layer, add. Twice. The normalised copy only ever feeds a sub-layer; what travels on is always “old x plus a correction”. Attention comes first so that the MLP has the gathered context to work on.</p>}
        />

        <Exercise
          id="transformer-block-predict"
          type="predict"
          title="A block that has learned nothing"
          hints={['If every weight in attention and in the MLP is zero, what does each sub-layer output?', 'Both sub-layers output all zeros. Now read the two lines of forward() with that in mind.']}
          solution={<><p>The block returns its input <b>unchanged</b>: x + 0 = x, twice. With residuals, “knowing nothing” means “do no harm”. A new block can start as a near-identity and gradually grow a useful correction.</p><p>Without residuals the same block would output all zeros, and every token would be erased. This is the deepest reason residual networks are trainable when they are very deep: adding a layer cannot, at first, make things much worse.</p></>}
        >
          <p>Imagine a block in which every weight matrix and bias of the attention and of the MLP is exactly zero (LayerNorms as normal). What does the block output for an input x? And what would it output if the residual connections were removed?</p>
        </Exercise>

        <Exercise
          id="transformer-block-debug"
          type="debug"
          title="One missing character"
          hints={[
            'Compare the two lines of forward() with the repo version, character by character.',
            'Line 1 has no “x +”. What happens to the original x after that line?',
          ]}
          solution={<><p>The first line is missing its residual: it should be <code>x = x + self.attn(self.ln1(x))</code>. As written, x is <em>replaced</em> by the attention output, which is only a weighted average of value vectors. The token’s own embedding and its position survive only to the extent that attention happens to copy them.</p><p>Nothing crashes, because shapes are unchanged. Shallow models still learn something; with 6 or more layers the loss falls much more slowly, because the early layers lose their direct gradient path. When we trained the repo’s model with 6 layers for 300 quick steps, removing both residuals raised the validation loss (the loss on held-out text, explained in <a href="#/lesson/training-gpt">Training GPT</a>) from 2.89 to 3.66. Try it yourself in the next exercise.</p></>}
        >
          <p>This model trains, but far worse than expected, and it gets worse the more layers are added. What is wrong?</p>
          <Code>{`
def forward(self, x):
    x = self.attn(self.ln1(x))
    x = x + self.ffn(self.ln2(x))
    return x
`}</Code>
        </Exercise>

        <Exercise
          id="transformer-block-implement"
          type="implement"
          title="Delete a plus sign, for real"
          hints={[
            'In tiny_gpt.py set n_layer = 6 in Config. Run python tiny_gpt.py --quick and write down the train and val loss at step 300.',
            'Now change both lines in Block.forward to x = self.attn(self.ln1(x)) and x = self.ffn(self.ln2(x)). Run --quick again.',
            'For LayerNorm: put the residuals back, replace self.ln1(x) and self.ln2(x) by plain x, and run again.',
          ]}
          solution={<><p>Our run (6 layers, 300 steps, CPU, validation loss at step 300; yours will differ a little): <b>2.89</b> as is, <b>3.36</b> without the LayerNorms, <b>3.66</b> without the residuals. Both removals hurt, and the missing plus signs hurt most: the early layers are barely learning, and the samples stay closer to noise.</p><p>One more experiment: raise <code>lr</code> from 3e-4 to 3e-3. In our run the normal model got <em>better</em> (2.34), while the version without LayerNorm stayed stuck at 3.37. Normalisation is what lets you train fast without the numbers getting out of hand.</p></>}
        >
          <p>Run the most instructive failure in the course. Train the 6-layer tiny GPT three times with <code>--quick</code>: as is, without the two residual connections, and without the two LayerNorms. Predict the ranking of the final losses before you run them.</p>
        </Exercise>

        <ExplainBack
          id="transformer-block-explain"
          prompt="Explain to a colleague why a Transformer block needs a feed-forward network at all. Attention already combines information from all tokens: what is left to do?"
          modelAnswer={<p>Attention only moves information: its output for a token is a weighted average of value vectors from the tokens it looked at. An average cannot create anything new, and it treats “what was gathered” only through linear projections. The feed-forward network is an MLP with a non-linear bend, applied to each token separately, so it can compute new features from the mixture that attention delivered. A block therefore alternates: communicate between tokens (attention), then compute within each token (MLP). The MLPs also hold about two thirds of a block’s parameters, so most of the model’s capacity sits there.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'With no positional information and no mask, you feed “man bites dog” instead of “dog bites man”. What happens to the block’s output?',
            options: ['It contains exactly the same rows, in swapped order: the model cannot tell who bit whom', 'It is completely different', 'It is all zeros', 'It crashes because the shapes differ'],
            answer: 0,
            explain: 'Attention scores depend only on the vectors, and the MLP works per token. Nothing in the block knows about slots unless positions are added to the input.',
          },
          {
            q: 'Which part of the block lets one token’s vector be influenced by another token?',
            options: ['The feed-forward network', 'LayerNorm', 'Self-attention, and nothing else', 'The residual connection'],
            answer: 2,
            explain: 'LayerNorm, the MLP and the residual additions all work on each token’s row separately. Attention is the only communication step.',
          },
          {
            q: 'Why does x = x + f(x) help gradients reach early layers?',
            options: ['It makes the gradients larger by doubling x', 'It normalises the gradient', 'It skips backpropagation for that layer', 'The derivative is 1 + f′(x): the “1” is a direct path that does not depend on how the layer behaves'],
            answer: 3,
            explain: 'A chain of multiplications is fragile. With the residual, every layer passes the incoming gradient straight through, in addition to its own contribution.',
          },
          {
            q: 'LayerNorm turns [100, 300, 500, 700] and [1, 3, 5, 7] into the same vector. What does that tell you about its job?',
            options: ['It removes the differences between tokens: after normalisation every token looks the same to the next sub-layer', 'It discards overall scale and offset, keeping the pattern, so that the next sub-layer always receives inputs of a familiar size', 'It sorts the numbers', 'It converts numbers into probabilities'],
            answer: 1,
            explain: 'It is a volume control between stages. Note that it is not softmax: outputs can be negative and do not sum to 1.',
          },
          {
            q: 'Why can blocks be stacked N times with no glue code in between?',
            options: ['Because all blocks share the same weights', 'Because attention is commutative', 'Because a block’s output has exactly the same shape as its input, (T, D)', 'Because LayerNorm makes every block’s output identical'],
            answer: 2,
            explain: 'Same type in, same type out. Each of the N blocks has its own weights, though.',
          },
        ]}
      />

      <Remember
        items={[
          <>A block is <b>communicate, then compute</b>: masked self-attention moves information between tokens, the feed-forward MLP processes each token alone.</>,
          <><b>Positional information</b> exists because attention is order-blind. Our GPT adds a learned vector per slot to each token embedding, once, before the first block.</>,
          <><b>Residual connections</b>, x = x + f(x): every sub-layer adds a correction instead of overwriting, and gradients get a direct highway to the early layers.</>,
          <><b>LayerNorm</b> rescales each token’s vector to mean 0, spread 1 (then a learned gain and bias) so every sub-layer sees inputs of a standard size. GPT does this <em>before</em> each sub-layer (pre-norm).</>,
          <>Shape in = shape out = <b>(T, D)</b>. That is why a Transformer is “block × N”, and why one config number sets the depth.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Attention' }, { label: 'Mask + heads' }, { label: 'Transformer block', sub: 'this lesson' }, { label: 'GPT', sub: 'next: assemble it' }, { label: 'Modern LLM' }]} active={2} />
        <ToyVsReal
          toy={<ul><li>3 tokens, D = 4, one block, random weights</li><li>ReLU in the demo; GELU in <code>tiny_gpt.py</code></li><li>Learned position table, LayerNorm</li><li>4 blocks of width 128 in the repo’s tiny GPT</li></ul>}
          real={<ul><li>The same two-line block, repeated 12 times (GPT-2 small) to 80 or more times (70B-class models), with D in the thousands</li><li>Gated MLP variants such as <G t="swiglu">SwiGLU</G></li><li>Rotary positions (<G t="rope">RoPE</G>) applied inside attention, and RMSNorm, a cheaper LayerNorm without the mean subtraction</li><li>Still pre-norm, still residual, still “attention then MLP”</li></ul>}
        />
        <Callout kind="established">The structure you explored, pre-norm residual blocks alternating attention and an MLP, is shared by GPT-2, GPT-3, Llama, Mistral and, as far as is publicly documented, the other major LLM families. The differences are in the details of each box, covered in <a href="#/lesson/modern-architecture">Modern LLM architecture</a>.</Callout>
        <Callout kind="established">About two thirds of a block’s parameters are in the MLP (8D² against 4D² for attention). You will count them yourself in the next lesson.</Callout>
        <Callout kind="research">What do the MLPs <em>do</em> with all those parameters? Interpretability studies suggest that MLP layers play a major role in recalling factual associations, and some describe them as key-value memories. This is evidence from specific experiments, not a complete account: knowledge in a trained model appears to be spread over many layers and both kinds of sub-layer. Treat “facts live in the MLP” as a useful hypothesis, not as settled.</Callout>
      </RealLLM>

      <BeforeMovingOn
        id="part-6"
        intro="You have finished the heart of the course. These questions reach back to earlier parts on purpose: pulling an idea out of memory is what makes it stick."
        questions={[
          {
            q: 'Why does each token get three different vectors (query, key, value) instead of one?',
            options: ['“What I am looking for”, “what I can be matched on” and “what I hand over” are different jobs; with a single vector every token would mostly match itself', 'Three vectors hold three times more information about the token than one, and attention needs all of it to score accurately', 'Because matrices must come in threes', 'To make the dot product symmetric'],
            answer: 0,
            explain: 'Relevance is directional: “river” is useful to “bank”, not the other way round. Separate learned projections make that possible.',
          },
          {
            q: 'Why are attention scores divided by √d before softmax?',
            options: ['To make each row of scores sum to 1, which softmax requires of its input before it can exponentiate', 'To convert them into probabilities', 'To apply the causal mask', 'Dot products of long vectors are large just because they add many terms; large scores make softmax all-or-nothing and its gradient vanish'],
            answer: 3,
            explain: 'Scaling keeps the scores in the range where softmax stays soft, whatever d is.',
          },
          {
            q: 'What would a GPT learn if it were trained without the causal mask?',
            options: ['The same thing, only more slowly, because each position would have more context to sift through', 'To copy the next token from its input, which is useless at generation time when the next token does not exist', 'Better long-range grammar', 'Nothing: training would crash'],
            answer: 1,
            explain: 'Position t must predict token t+1. Without the mask, token t+1 is visible, and copying it minimises the loss.',
          },
          {
            q: 'A model has 48 layers. What do residual connections give the gradient on its way to layer 1?',
            options: ['A shortcut that skips backpropagation', 'A larger learning rate', 'Nothing: they only matter in the forward pass', 'A direct additive path, so it is not only a product of 48 per-layer factors'],
            answer: 3,
            explain: 'Each “+” passes the gradient through unchanged, in addition to whatever flows through the sub-layer.',
          },
          {
            q: 'Softmax and cross-entropy (Part 1 and Part 5): the model gives the correct next token probability 0.5. What is the loss for that example?',
            options: ['−log(0.5) ≈ 0.69', '0.5', '1 − 0.5² = 0.75', '0'],
            answer: 0,
            explain: 'Cross-entropy is minus the log of the probability given to the right answer: 0 when certain and right, large when confidently wrong.',
          },
          {
            q: 'Part 4: why does GPT use sub-word tokens rather than whole words?',
            options: ['Sub-words are easier to pronounce', 'Because attention compares vectors of a fixed width, and a long word would not fit into a single vector', 'A word vocabulary is huge and still misses new words, names and typos; sub-word pieces cover any text with a fixed-size vocabulary', 'To make the context window shorter'],
            answer: 2,
            explain: 'BPE builds a vocabulary of frequent pieces, so rare words are spelled out of known parts instead of becoming “unknown”.',
          },
          {
            q: 'Part 3: what does backpropagation compute?',
            options: ['The best value for every weight directly, by solving the network’s equations backwards from the desired output', 'For every weight, how much the loss would change if that weight were nudged, by applying the chain rule backwards through the network', 'The model’s predictions', 'A random direction to try'],
            answer: 1,
            explain: 'Those numbers are the gradient. Gradient descent then moves every weight a small step against it.',
          },
        ]}
      >
        <OrderExercise
          id="transformer-block-pipeline"
          title="Rebuild the pipeline from memory"
          prompt={<p>From raw text to a context-aware vector. Put everything you have built so far in order.</p>}
          correct={[
            'Text is split into tokens',
            'Each token id looks up its embedding vector',
            'A position vector is added to each embedding',
            'Queries are scored against keys, scaled by √d',
            'Scores for future tokens are set to −∞',
            'Softmax turns scores into weights, values are blended',
            'The attention output is added to the residual stream',
            'The MLP processes each token, and its output is added too',
          ]}
          solutionNote={<p>Text → numbers → vectors with positions → communicate (score, mask, softmax, blend) → add → compute → add. The next lesson puts the final piece on top: turning the last vector into a prediction.</p>}
        />
      </BeforeMovingOn>
    </Lesson>
  )
}
