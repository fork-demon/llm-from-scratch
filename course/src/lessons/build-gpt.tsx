import { TrainedGptExplorer } from '../interactive/TrainedGptExplorer'
import { Gpt2Explainer } from '../interactive/Gpt2Explainer'
import { CodeExercise } from '../components/python'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack, OrderExercise } from '../components/exercise'
import { GptTracer } from '../interactive/GptTracer'
import { ParamCounter } from '../interactive/ParamCounter'
import { ShapeJourney } from '../illustrations/ShapeJourney'

export default function BuildGptLesson() {
  return (
    <Lesson id="build-gpt">
      <Why>
        <p className="lede">It is past eight. The Paisa Pal office is empty except for Riya, the cleaning staff and the hum of the air conditioning.</p>
        <p>On her second monitor is the photo she took on day one: Kabir’s whiteboard map, a row of boxes between “What is a cat?” and the answer. Back then every box was a promise. She has been ticking them off, one lesson at a time.</p>
        <p>Tonight she opens a fresh file and types <code>class GPT</code>. Her phone buzzes. Kabir: “Don’t add anything new. Plug the boxes together and follow one token through.”</p>
        <p>Look at her list:</p>
        <ul>
          <li>Tokenizer: <a href="#/lesson/tokenization">built</a>.</li>
          <li>Embeddings: <a href="#/lesson/embeddings">built</a>.</li>
          <li>Attention, mask, heads: <a href="#/lesson/attention">built</a>, <a href="#/lesson/masks-and-heads">built</a>.</li>
          <li>The Transformer block: <a href="#/lesson/transformer-block">built</a>.</li>
          <li>Softmax, sampling, the generation loop: <a href="#/lesson/softmax">built</a>, <a href="#/lesson/next-token">built</a>.</li>
        </ul>
        <p>There are <b>no new ideas</b> in this lesson. You bolt the parts together, and the result is a GPT. It has the same wiring as GPT-2, in about 100 lines of model code.</p>
        <p>So the question for tonight is small and exact: <em>what happens to one token, from the moment it is typed to the moment the next one is chosen?</em></p>
        <Callout kind="idea">The goal: follow one token from text to prediction and say, at every step, what shape the data has and which lesson built that step.</Callout>
      </Why>

      <Problem title="What is still missing?">
        <p>A stack of blocks takes <span className="mono">(T, D)</span> and returns <span className="mono">(T, D)</span>: one context-aware vector per token. That is not yet a prediction.</p>
        <p>Riya stares at the output grid. “Fine. Now which character comes next?” Two small gaps remain.</p>
        <WhyExists
          problem="The last block gives us a vector of D numbers per position. We need a score for every token in the vocabulary: which one comes next?"
          naive="Compare the output vector with every token’s embedding by hand and pick the closest one."
          fails="Close, but “pick the closest” throws away uncertainty and cannot be trained with cross-entropy. We need a score for every token, so that softmax can turn them into probabilities."
          idea="One more matrix multiply, from D numbers per token to one number per vocabulary word. Row by row, that multiply is “dot the output vector with every token’s vector”. This last layer is called the head, and the raw scores it produces are called the logits."
          tradeoff="With a large vocabulary this matrix is big: 50,257 × 768 ≈ 38.6M numbers in GPT-2 small. A trick called weight tying lets it reuse the embedding table instead of storing a second one. We unpack it later in this lesson."
        />
        <p>The second gap is the loop around the model. A GPT predicts <em>one</em> token. To write a sentence, you sample a token, append it to the input, and run the whole model again.</p>
        <p>You built that loop in <a href="#/lesson/next-token">Predicting the next token</a>. Here it gets a real model inside.</p>
        <p>And one practical worry. This lesson’s code is in <b>PyTorch</b>, not NumPy. Dev texts from the flat: “PyTorch? So that’s where the real magic is.” Is it?</p>
        <Callout kind="established" label="What PyTorch does for you: exactly two things">
          <ol>
            <li>During the forward pass it <b>records</b> every operation. You did this by hand in <a href="#/lesson/backprop">Backpropagation</a>, when you kept the intermediate values for the backward pass.</li>
            <li>When you call <code>loss.backward()</code>, it <b>replays your backward rules</b> through that recording, in reverse. That is the chain rule you applied by hand.</li>
          </ol>
          A <code>torch.Tensor</code> is a NumPy array that carries that recorder. Everything else in the file, every matrix multiply and every shape, is something you have already written yourself. (It can also run the maths on a GPU. That is speed, not new maths.)
        </Callout>
      </Problem>

      <MentalModel>
        <p>Kabir’s advice for nights like this: draw the data, not the boxes. So here is the whole machine in one picture, with the data drawn as grids, one row per token.</p>
        <p>Follow the numbers 1 to 9 and watch the <em>shape</em> at each stage. Every stage is a lesson you have done.</p>
        <ShapeJourney />
        <p>Two things to take from it. First, the grid that leaves the blocks is <b>exactly the size</b> of the grid that went in.</p>
        <p>Second, the head produces a row of scores for every position, but generation only uses the <b>last row</b>. That row is the prediction for what follows the final character.</p>
        <Term
          name="The head (language-model head)"
          plain={<>The last layer: one matrix multiply that turns a token’s D-number vector into V raw scores, one for every token in the vocabulary.</>}
          example={<>D = 128, V = 65: the vector [0.3, −1.2, …] (128 numbers) becomes 65 scores, such as “e”: 2.1, “t”: 1.7, “z”: −3.0.</>}
          formal={<><G t="logits">logits</G> = x W<sub>head</sub>, with W<sub>head</sub> of shape (D, V). Softmax of the logits gives next-token probabilities.</>}
        />
        <h3>A GPT is these 6 numbers</h3>
        <p>Riya scrolls to the top of the file and laughs. This is the entire configuration of the model. In its wiring, GPT-2 is the same code with bigger numbers.</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="Config">{`
class Config:
    context_len = 64     # max tokens the model can see (GPT-2: 1024)
    n_embd = 128         # embedding dimension           (GPT-2: 768)
    n_head = 4           # attention heads               (GPT-2: 12)
    n_layer = 4          # transformer blocks            (GPT-2: 12)
    dropout = 0.1
    vocab_size = None    # set from data
`}</Code>
        <p>Five of them fix the architecture. The sixth, <code>dropout</code>, only matters during training (it randomly zeroes some numbers so the model cannot lean on any single one).</p>
        <h3>Every module, and where you built it</h3>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>in <code>tiny_gpt.py</code></th><th>what it is</th><th>you built it in</th></tr></thead>
            <tbody>
              <tr><td className="mono">tok_emb</td><td>embedding lookup</td><td><a href="#/lesson/embeddings">Embeddings</a></td></tr>
              <tr><td className="mono">pos_emb</td><td>a vector per slot</td><td><a href="#/lesson/transformer-block">The Transformer block</a></td></tr>
              <tr><td className="mono">CausalSelfAttention</td><td>masked multi-head attention</td><td><a href="#/lesson/attention">Attention</a>, <a href="#/lesson/masks-and-heads">Masks and heads</a></td></tr>
              <tr><td className="mono">FeedForward</td><td>the MLP</td><td><a href="#/lesson/neurons">Neurons and layers</a></td></tr>
              <tr><td className="mono">Block, LayerNorm, x + …</td><td>residuals and normalisation</td><td><a href="#/lesson/transformer-block">The Transformer block</a></td></tr>
              <tr><td className="mono">F.softmax, F.cross_entropy</td><td>scores → probabilities, and the loss</td><td><a href="#/lesson/softmax">Softmax</a>, <a href="#/lesson/next-token">Next token</a></td></tr>
              <tr><td className="mono">loss.backward()</td><td>backpropagation, automated</td><td><a href="#/lesson/backprop">Backpropagation</a></td></tr>
              <tr><td className="mono">opt.step()</td><td>the training loop</td><td><a href="#/lesson/gradient-descent">Gradient descent</a></td></tr>
              <tr><td className="mono">generate()</td><td>the autoregressive loop</td><td><a href="#/lesson/next-token">Next token</a></td></tr>
            </tbody>
          </table>
        </div>
        <Callout kind="analogy">
          Assembling a GPT is like wiring up a pipeline of services you have already written and unit-tested. The only new work is checking that each output type matches the next input type.
          <br /><br />
          Where the analogy stops: these “services” have no hand-written logic. Every one of them is a bag of numbers, and all of them will be tuned <em>jointly</em> by one loss at the very end of the pipeline.
        </Callout>
      </MentalModel>

      <TryIt title="Trace one token, then count the parameters">
        <h3>1. Trace one token</h3>
        <p>This is what Riya did at her desk tonight. A complete GPT with 2 blocks runs in your browser. Step through all the stages, and read the shape at each stage before looking at the numbers.</p>
        <GptTracer />
        <Callout kind="warn" label="Honest warning">
          The tracer’s weights are random, so its prediction is noise. Every step of the computation is real. What is missing is the right <em>numbers</em> in the matrices, and finding them is called training: the next lesson.
        </Callout>
        <h3>2. Now the same machine, trained</h3>
        <p>Same architecture, different numbers. This one is the repository’s <code>tiny_gpt.py</code> after training on Shakespeare, running in your browser. Type a prompt and watch it write. Then open the attention grid and click a token to see where each head looks.</p>
        <TrainedGptExplorer />
        <h3>3. Where do the parameters live?</h3>
        <p>“GPT-2 small has 124M parameters.” Which 124 million? Load the presets and find out.</p>
        <ParamCounter />
      </TryIt>

      <Numbers>
        <p>Let’s count the repository’s tiny GPT by hand: V = 65 characters, context 64, D = 128, 4 blocks.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>component</th><th>calculation</th><th style={{ textAlign: 'right' }}>parameters</th></tr></thead>
            <tbody>
              <tr><td>token embeddings</td><td className="mono">65 × 128</td><td className="mono" style={{ textAlign: 'right' }}>8,320</td></tr>
              <tr><td>position embeddings</td><td className="mono">64 × 128</td><td className="mono" style={{ textAlign: 'right' }}>8,192</td></tr>
              <tr><td>attention, per block</td><td className="mono">4 × 128² + 4 × 128</td><td className="mono" style={{ textAlign: 'right' }}>66,048</td></tr>
              <tr><td>MLP, per block</td><td className="mono">8 × 128² + 5 × 128</td><td className="mono" style={{ textAlign: 'right' }}>131,712</td></tr>
              <tr><td>two LayerNorms, per block</td><td className="mono">4 × 128</td><td className="mono" style={{ textAlign: 'right' }}>512</td></tr>
              <tr><td>one block</td><td className="mono">66,048 + 131,712 + 512</td><td className="mono" style={{ textAlign: 'right' }}>198,272</td></tr>
              <tr><td>4 blocks</td><td className="mono">4 × 198,272</td><td className="mono" style={{ textAlign: 'right' }}>793,088</td></tr>
              <tr><td>final LayerNorm</td><td className="mono">2 × 128</td><td className="mono" style={{ textAlign: 'right' }}>256</td></tr>
              <tr><td>head</td><td>tied to the token embeddings</td><td className="mono" style={{ textAlign: 'right' }}>0</td></tr>
              <tr><td><b>total</b></td><td></td><td className="mono" style={{ textAlign: 'right' }}><b>809,856</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>When you run the file it prints <span className="mono">params 0.81M</span>. That is this number.</p>
        <p>Where do they live? The MLPs hold 526,848 (65%), attention 264,192 (33%), and the embeddings only 2%.</p>
        <p>In GPT-2 small the picture shifts. Its vocabulary of 50,257 tokens makes the embedding table 38.6M, about 31% of the 124.4M total. In GPT-2 XL (1.56B) the same table is only 5%.</p>
        <p><b>The bigger the model, the more it is dominated by the D² terms inside the blocks.</b></p>
      </Numbers>

      <TheMath>
        <p>One line for the whole model, reading inside-out:</p>
        <Equation
          label="Logits equal head of final layer norm of blocks of token embedding plus position embedding"
          symbols={[
            ['E[ids]', 'look up each token id in the embedding table E: shape (T, D)'],
            ['P', 'the first T rows of the position table: shape (T, D)'],
            [<>Block<sub>N</sub>(…Block<sub>1</sub>(·))</>, 'N Transformer blocks, one after another, each (T, D) → (T, D)'],
            ['LN', 'the final layer normalisation'],
            [<>W<sub>head</sub></>, <>a (D, V) matrix. With weight tying it is the embedding table again, transposed: W<sub>head</sub> = E<sup>T</sup></>],
            ['logits', 'shape (T, V): for every position, a raw score for every possible next token'],
          ]}
        >
          logits = LN( Block<sub>N</sub>( … Block<sub>1</sub>( E[ids] + P ) … ) ) W<sub>head</sub>
        </Equation>
        <Equation
          label="Parameter count"
          symbols={[
            ['V · D', 'token embeddings (and, tied, the head)'],
            ['C · D', 'position embeddings, C = context length'],
            ['12D² + 13D', 'one block: 4D² + 4D attention, 8D² + 5D MLP, 4D for two LayerNorms'],
            ['2D', 'the final LayerNorm'],
          ]}
        >
          params = V·D + C·D + N·(12D² + 13D) + 2D
        </Equation>
        <p>For any model of serious size you can forget everything except <span className="mono">12 · N · D²</span>. GPT-2 small: 12 × 12 × 768² ≈ 85M of its 124M.</p>
        <DeepDive title="Weight tying: why can the head reuse the embedding table?">
          <p>The embedding table E has shape (V, D): row i is the vector for token i. The head needs shape (D, V): column i produces the score for token i. Transpose E and the shapes match.</p>
          <p>It also makes sense. With E<sup>T</sup> as the head, the logit for token i is the <b>dot product between the model’s output vector and token i’s embedding</b>: “how much does what I am about to say point in the direction of this token?” Tokens with similar embeddings automatically get similar scores.</p>
          <p>In code it is one line: <code>self.head.weight = self.tok_emb.weight</code>. Both names now refer to the same tensor, and gradients from both uses are added up. GPT-2 does this; it saves 38.6M parameters there. For our character-level model it saves only 8,320 (1%), because V is tiny.</p>
          <p>It is a design choice, not a law. Several recent small models tie their weights, while many large ones keep a separate head, since V·D is a small fraction of their total anyway.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>We build the <code>GPT</code> class in three steps. <code>nn.Module</code> is PyTorch’s base class for “a thing with parameters”: anything you assign to <code>self</code> in <code>__init__</code> is registered, so the optimiser can find every weight later.</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="Step 1: the parts list">{`
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # (V, D) table
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.n_embd)  # (context, D) table
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying (see doc)
`}</Code>
        <p>Six lines, and they are the six rows of the parameter table above.</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="Step 2: forward() is the pipeline, one line per stage">{`
def forward(self, idx, targets=None):
    B, T = idx.shape                            # B sequences of T token ids
    pos = torch.arange(T, device=idx.device)
    x = self.tok_emb(idx) + self.pos_emb(pos)   # (B, T, D)  meaning + position
    x = self.blocks(x)                          # (B, T, D)  communicate/compute stack
    x = self.ln_f(x)                            # (B, T, D)
    logits = self.head(x)                       # (B, T, vocab)
`}</Code>
        <p>The only new letter is <b>B</b>, the <G t="batch">batch</G> size: B independent sequences are pushed through together because GPUs like big matrices. Sequences in a batch never see each other. In the tracer, B = 1.</p>
        <p>The rest of <code>forward</code> computes the <G t="cross-entropy">cross-entropy</G> loss when <code>targets</code> are given. That belongs to training, so we leave it for the next lesson.</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="Step 3: generate() is the loop around the model">{`
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0):
    """Module 06's autoregressive loop."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.cfg.context_len:]   # crop to context window
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature     # last position only
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)   # weighted die
        idx = torch.cat([idx, nxt], dim=1)              # feed back in
    return idx
`}</Code>
        <ul>
          <li><code>@torch.no_grad()</code>: do not record operations. We are not training, so there will be no backward pass.</li>
          <li><code>logits[:, -1, :]</code>: the model produced a prediction at every position; to continue the text we only need the last one.</li>
          <li><code>torch.multinomial</code>: the weighted die from <a href="#/lesson/next-token">Next token</a>.</li>
          <li><code>torch.cat</code>: append the new token, and go round again.</li>
        </ul>
        <h3>Why the crop?</h3>
        <p><code>idx[:, -self.cfg.context_len:]</code> keeps only the last 64 tokens. It has to. <code>pos_emb</code> has exactly 64 rows, so slot 64 does not exist, and the causal mask was built for 64×64.</p>
        <p>Anything older than the <G t="context-window">context window</G> is gone. The model does not “forget” it gradually. It never sees it at all.</p>
        <Callout kind="dev">Notice the waste. To produce token 50, the loop re-runs the model on tokens 0…49, although their vectors have not changed: the causal mask guarantees it. Caching that work is the <G t="kv-cache">KV cache</G>, two lessons from now.<br /><br />One catch, and it comes from the crop. That guarantee holds only while the whole sequence fits in the window (T ≤ 64). Once the window starts to slide, every kept token moves to a new slot, gets a different position vector, and so every old vector really does change.</Callout>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check.</p>
        <ul>
          <li><b>Tracer:</b> type a prompt longer than 16 characters and look at stage 0. Which characters does the model never see?</li>
          <li><b>Tracer:</b> type “aa”. At stage 2 the two rows are identical. At stage 3 they are not. Which line of <code>forward()</code> did that?</li>
          <li><b>Tracer:</b> at the Softmax stage, drag the temperature to 0.2, then to 2. The logits did not change. What did?</li>
          <li><b>Tracer:</b> change the <em>last</em> character of your prompt and watch the first rows of the logits table at stage “Head”. Do they move? Why not?</li>
          <li><b>Counter:</b> load GPT-2 small and double <code>n_embd</code> to 1536. Did the total double? Now reset and double <code>n_layer</code> instead.</li>
          <li><b>Counter:</b> change <code>n_head</code> from 12 to 16, then to 5. What happens to the total, and what does the warning say?</li>
          <li><b>Counter:</b> untick weight tying on “Our tiny GPT”, then on GPT-2 small. Where does it matter?</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="build-gpt-code-forward" />
        <Exercise
          id="build-gpt-calc-params"
          type="calculate"
          title="Count a model by hand"
          answer={{ value: 108544, tolerance: 0 }}
          answerLabel="total parameters"
          hints={[
            'Four kinds of thing: token embeddings V·D, position embeddings C·D, N blocks of 12D² + 13D each, and a final LayerNorm of 2D. The tied head adds nothing.',
            'D² = 4096. One block = 12 × 4096 + 13 × 64 = 49,152 + 832 = 49,984.',
            'Embeddings: 100 × 64 = 6,400 and 32 × 64 = 2,048. Blocks: 2 × 49,984 = 99,968. Final LayerNorm: 128.',
          ]}
          solution={<><p>6,400 + 2,048 + 99,968 + 128 = <b>108,544</b>.</p><p>92% of it is in the two blocks. <code>n_head</code> never appeared in the calculation: heads split D, they do not add parameters. Check your answer in the parameter counter.</p></>}
        >
          <p>A GPT has <code>vocab_size = 100</code>, <code>context_len = 32</code>, <code>n_embd = 64</code>, <code>n_head = 4</code>, <code>n_layer = 2</code>, with a tied head and biases as in <code>tiny_gpt.py</code>. How many parameters does it have?</p>
        </Exercise>

        <OrderExercise
          id="build-gpt-trace"
          title="Trace one token from text to prediction"
          prompt={<p>Without scrolling up: put the full pipeline in order.</p>}
          correct={[
            'Input text',
            'Tokenizer: text → token ids',
            'Embedding lookup + position vectors',
            'Transformer blocks × N',
            'Final LayerNorm',
            'Linear head: D numbers → V logits',
            'Keep the logits of the last position',
            'Softmax: logits → probabilities',
            'Sample one token id',
            'Append it to the input and run again',
          ]}
          solutionNote={<p>Text → ids → vectors → context-aware vectors → scores → probabilities → one token → repeat. The shapes tell the same story: (T) → (T, D) → (T, D) → (T, V) → (V) → one id.</p>}
        />

        <Exercise
          id="build-gpt-predict-shape"
          type="predict"
          title="What shape are the logits?"
          answer={{ text: ['(1, 5, 65)', '1, 5, 65', '[1, 5, 65]', '1x5x65', '1×5×65', '(1,5,65)'] }}
          answerLabel="e.g. (2, 3, 4)"
          hints={['The logits have three axes: batch, position, and…?', 'One sequence → B = 1. Five characters → T = 5. One score per possible next character → 65.']}
          solution={<><p><b>(1, 5, 65)</b>: batch 1, 5 positions, 65 scores per position.</p><p>The model predicts a next character after <em>every</em> one of the 5 positions, not only after the last. During training all 5 rows are graded (that is the causal-mask efficiency from two lessons ago). During generation, <code>logits[:, -1, :]</code> keeps just the last row, shape (1, 65).</p></>}
        >
          <p>You call the repo’s model (vocab_size 65, n_embd 128) on a single sequence of 5 characters: <code>logits, _ = model(idx)</code> with <code>idx.shape == (1, 5)</code>. What is <code>logits.shape</code>?</p>
        </Exercise>

        <Exercise
          id="build-gpt-implement"
          type="implement"
          title="Run it, then turn the knobs"
          hints={[
            'From phase3-transformers run: python tiny_gpt.py --quick. The first line printed is “vocab 65, params 0.81M, device …”.',
            'One block is 12D² + 13D = 198,272 parameters at D = 128. Two more blocks add 396,544.',
            'For n_embd = 256 recompute everything: embeddings (65 + 64) × 256, four blocks of 12 × 256² + 13 × 256, final LayerNorm 512.',
          ]}
          solution={<><p><code>n_layer = 6</code>: 809,856 + 2 × 198,272 = 1,206,400, printed as <b>1.21M</b>. Depth adds parameters linearly.</p><p><code>n_embd = 256</code> (with 4 layers): 33,024 + 4 × 789,760 + 512 = 3,192,576, printed as <b>3.19M</b>. Doubling the width roughly <em>quadruples</em> the model, because of the D² terms. If you try <code>n_embd = 100</code> with 4 heads it still runs (100 / 4 = 25); <code>n_embd = 130</code> crashes in the head split, because 130 is not divisible by 4.</p><p>The quick run ends with Shakespeare-flavoured gibberish. That is expected after 300 steps; what training does to those 0.81M numbers is the subject of the next lesson.</p></>}
        >
          <p>Run <code>python tiny_gpt.py --quick</code> and find the line that reports the parameter count. Then, in <code>Config</code>, set <code>n_layer = 6</code>. Before running: what will the count be? Put it back, set <code>n_embd = 256</code>, and predict again.</p>
        </Exercise>

        <ExplainBack
          id="build-gpt-explain"
          prompt="Someone says: “PyTorch is where the real magic happens; the NumPy code was just a toy.” Explain what PyTorch actually adds to what you wrote by hand, and what it does not."
          modelAnswer={<p>PyTorch automates two chores. While the forward pass runs, it records each operation and keeps the intermediate values, which I did by hand with a cache. When I call loss.backward(), it walks that recording in reverse and applies the same chain-rule steps I wrote by hand in the backprop lesson, giving a gradient for every parameter. It also runs the matrix maths on a GPU. It adds no new mathematics: the embedding lookup, attention, MLP, LayerNorm, softmax and cross-entropy in tiny_gpt.py compute exactly what the NumPy versions compute. The “intelligence” is not in the framework. It is in the numbers that training puts into the matrices.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'The model returns logits of shape (B, T, V). Why does generate() only use logits[:, -1, :]?',
            options: ['Row t predicts the token after position t. To continue the text we need the prediction after the last token; the others predict tokens we already have', 'The other rows are unfinished: only the last position has passed through all the blocks, so only its logits are usable', 'To save memory', 'Because softmax only works on one row'],
            answer: 0,
            explain: 'All rows are valid predictions, which is what makes training efficient. For generation only the newest one is news.',
          },
          {
            q: 'Your prompt has 100 tokens but context_len is 64. What does the repo’s generate() do?',
            options: ['Crashes', 'Compresses the first 36 tokens into a summary vector and feeds that in alongside the last 64', 'Keeps only the last 64 tokens; the first 36 have no influence at all on the prediction', 'Runs the model twice'],
            answer: 2,
            explain: 'idx[:, -context_len:]. There is no position vector for slot 64, so the model cannot take more. What is outside the window does not exist for it.',
          },
          {
            q: 'In GPT-2 small, which component holds the most parameters?',
            options: ['The attention matrices', 'The LayerNorms', 'The position embeddings', 'The MLPs inside the blocks'],
            answer: 3,
            explain: 'Per block the MLP has 8D² against attention’s 4D². Over 12 blocks: about 57M (MLP), 28M (attention), 39M (token embeddings).',
          },
          {
            q: 'What does weight tying mean?',
            options: ['All N blocks share one set of weights, so the model stores a single block and applies it N times', 'The output head reuses the token-embedding table (transposed), so a logit is the dot product of the output vector with a token’s embedding', 'Wq and Wk are forced to be equal', 'Weights are frozen during training'],
            answer: 1,
            explain: 'One (V, D) table does both jobs: id → vector at the bottom, vector → scores at the top.',
          },
          {
            q: 'You double n_head from 4 to 8 and change nothing else. What happens to the parameter count?',
            options: ['It doubles', 'It grows by 4D² per block, because every new head needs its own Wq, Wk, Wv and Wo', 'Nothing: heads split the D numbers, so each head just becomes thinner', 'It halves'],
            answer: 2,
            explain: 'Wq, Wk, Wv, Wo stay D×D. Only the per-head width D / n_head changes, from 32 to 16.',
          },
        ]}
      />

      <Remember
        items={[
          <>GPT = tokenizer → <b>token + position embeddings</b> → <b>N Transformer blocks</b> → final LayerNorm → <b>linear head</b> → logits → softmax → sample → append → repeat.</>,
          <>Shapes tell the story: <b>(T) → (T, D) → … → (T, D) → (T, V)</b>. Generation uses only the last row of the logits.</>,
          <>A GPT is a handful of config numbers. GPT-2 is the same code with bigger ones. Parameters ≈ <b>12 · N · D²</b> plus embeddings; most live in the MLPs.</>,
          <><b>Weight tying</b>: the head reuses the embedding table, so a logit is “output vector · token embedding”.</>,
          <>The model can only see <b>context_len</b> tokens. Older tokens are cropped away and have no influence.</>,
          <>PyTorch automates two things you did by hand: recording the forward pass and replaying the backward rules. An untrained GPT is a correct pipeline filled with the wrong numbers.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Attention' }, { label: 'Mask + heads' }, { label: 'Transformer block' }, { label: 'GPT', sub: 'this lesson' }, { label: 'Train it', sub: 'next' }, { label: 'Modern LLM' }]} active={3} />
        <p>How far is our model from the real ones? In structure: not far. In size: very.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th></th><th>our tiny GPT</th><th>GPT-2 small (2019)</th><th>a 70B-class open model (Llama 3 70B, 2024)</th></tr></thead>
            <tbody>
              <tr><td>blocks (n_layer)</td><td className="mono">4</td><td className="mono">12</td><td className="mono">80</td></tr>
              <tr><td>vector width (n_embd)</td><td className="mono">128</td><td className="mono">768</td><td className="mono">8,192</td></tr>
              <tr><td>attention heads</td><td className="mono">4</td><td className="mono">12</td><td className="mono">64 (sharing 8 key/value heads)</td></tr>
              <tr><td>context window</td><td className="mono">64</td><td className="mono">1,024</td><td className="mono">8,192 (128K in version 3.1)</td></tr>
              <tr><td>parameters</td><td className="mono">0.81M</td><td className="mono">124M</td><td className="mono">≈ 70,000M</td></tr>
              <tr><td>tokenizer</td><td>65 characters</td><td>BPE, 50,257 tokens</td><td>BPE, ≈ 128,000 tokens</td></tr>
              <tr><td>positions / norm / MLP</td><td colSpan={2}>learned table / LayerNorm / GELU</td><td>RoPE / RMSNorm / SwiGLU</td></tr>
            </tbody>
          </table>
        </div>
        <ToyVsReal
          toy={<ul><li>Character-level, trained on 1 MB of Shakespeare</li><li>Runs on a laptop CPU in minutes</li><li>Every line readable: about 100 lines of model code</li><li>The browser tracer: untrained, D = 8, 2 blocks</li></ul>}
          real={<ul><li>Sub-word tokens, trained on trillions of tokens</li><li>Thousands of GPUs for weeks; the model is split across many devices</li><li>The same pipeline: embed, N blocks, norm, head, softmax, sample</li><li>Changed details inside the boxes: RoPE, RMSNorm, SwiGLU, grouped-query attention, no biases</li></ul>}
        />
        <Callout kind="established">GPT-2’s published architecture is what <code>tiny_gpt.py</code> implements: learned positions, pre-norm blocks, GELU MLPs, tied head. With vocab 50,257, context 1,024, D = 768 and 12 layers, the formula in this lesson gives exactly 124,439,808 parameters, the “124M” on the label.</Callout>
        <p>Kabir stops at Riya’s desk on his way out and reads the number on her screen. “124,439,808. That’s GPT-2’s count.” He pulls up a chair. “Want to see the real one run? Same machine, full size.”</p>
        <p>Below is GPT-2 small itself, with the weights OpenAI trained, running in your browser. Every stage is a box you built in this part, and each one names the lesson that built it. It also has <a href="#/gpt2">a page of its own</a>, with more room.</p>
        <Gpt2Explainer />
        <p>Same wiring as Riya’s file. The difference is the numbers inside, and those come only from training.</p>
        <Callout kind="note" label="One difference: where the weights start">The wiring matches; the starting numbers do not. <code>tiny_gpt.py</code> keeps PyTorch’s defaults, which fill the embedding table with numbers of size about 1. GPT-2 starts every weight small (standard deviation 0.02) and shrinks the layers that write into the residual stream by a further 1/√(2N) for N blocks. You will see in the next lesson why that choice matters on the very first step.</Callout>
        <Callout kind="note">The last column differs in the details of almost every box, but not in the wiring diagram. Each of those changes (RoPE, RMSNorm, SwiGLU, grouped-query attention, mixture of experts) is explained in <a href="#/lesson/modern-architecture">Modern LLM architecture</a>. For closed models such as GPT-4, Claude and Gemini, architecture details are not public.</Callout>
        <p>Riya runs the file one last time before leaving. It prints <span className="mono">params 0.81M</span> and a line of gibberish. Correct pipeline, wrong numbers. Tomorrow night: training.</p>
      </RealLLM>
    </Lesson>
  )
}
