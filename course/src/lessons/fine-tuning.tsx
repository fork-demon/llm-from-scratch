import { CodeExercise } from '../components/python'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { ForgettingLab } from '../interactive/ForgettingLab'
import { LoraCalculator, LoraTrainer } from '../interactive/LoraLab'
import { WeightsShift } from '../illustrations/WeightsShift'

export default function FineTuningLesson() {
  return (
    <Lesson id="fine-tuning">
      <Why>
        <p className="lede">The bot now finds the right policy. It still does not sound like Paisa Pal.</p>
        <p>The brand team’s rule is short: two warm sentences, no “Dear Valued Customer”, and a fixed JSON format so the app can show the reply. Riya reads a sample of the bot’s answers during lunch. One begins “We deeply regret any inconvenience this may have caused” and goes on for a whole paragraph.</p>
        <p>She has described all of that in the prompt. It mostly works. But the description is now 800 tokens long, Paisa Pal pays for it on every call, and one request in fifty still comes back as a chatty paragraph.</p>
        <p>Adding documents with <a href="#/lesson/rag">RAG</a> will not help. The model is not missing <em>information</em>. It is missing a <em>habit</em>.</p>
        <p>Kabir puts it in one line: “Habits live in the weights.” To change a habit, you change the weights.</p>
        <p>Part 9’s question again: <b>what exactly changes?</b> RAG changed the prompt and left the weights alone. Fine-tuning is the mirror image: <b>the weights change, and the prompt can stay short</b>.</p>
        <WeightsShift />
        <p>Nothing is added to the model and nothing is removed. It is the same grid of numbers, with some of them nudged, and the habit moves with them.</p>
      </Why>

      <Problem>
        <p>You have already seen fine-tuning once. In <a href="#/lesson/training-pipeline">From raw text to assistant</a>, <G t="sft">SFT</G> turned a raw text predictor into something that answers questions. This lesson is about doing the same to a model for <em>your</em> task.</p>
        <Term
          name="Fine-tuning"
          plain={<>Continue training a model that is already trained, on a small set of examples of the behaviour you want.</>}
          example={<>2,000 pairs of (customer message → ideal reply in your format). Run the usual training loop for a few passes with a small <G t="learning-rate">learning rate</G>.</>}
          formal={<>Minimise the same next-token <G t="cross-entropy">cross-entropy</G> loss, starting from pretrained <G t="parameters">parameters</G> instead of random ones. With prompt → response pairs this is called supervised fine-tuning (SFT).</>}
        />
        <p>Nothing new is needed mechanically. Same model, same loss, same <G t="gradient-descent">gradient descent</G>.</p>
        <p>Three things differ. The <b>data</b> is yours. The <b>starting point</b> is a trained model. The <b>learning rate</b> is small, because you want to nudge a working system, not rebuild it.</p>
        <p>Riya is relieved: she already has the training loop from Part 7. Kabir is less relieved. “The loop is the simple part,” he says. That simplicity hides two real problems.</p>
        <WhyExists
          problem="You want to specialise a large pretrained model, perhaps for many different tasks or customers."
          naive="Full fine-tuning: update every weight, and save a complete copy of the model per task."
          fails="(1) Cost: a 7-billion-parameter model is about 14 GB per copy, and training all of it needs several times that in GPU memory. (2) Forgetting: the updates overwrite weights that other abilities depended on."
          idea="Freeze the pretrained weights. Train a small add-on (an adapter) that learns only the correction. LoRA is the most used design."
          tradeoff="A small adapter has limited capacity: it learns a simple change well and an arbitrary change badly. And while the adapter is plugged in, behaviour on old tasks still shifts."
        />
      </Problem>

      <MentalModel>
        <h3>First decide whether you should fine-tune at all</h3>
        <p>Dev has a plan ready. “Fine-tune it on all our product docs. Then it knows everything and we can drop the search.” It sounds efficient. It is the most common mistake in this lesson, and the table shows why.</p>
        <p>Three tools, three different things that change. Reach for them in this order, because each is more expensive than the one before.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th /><th>Prompting</th><th>RAG</th><th>Fine-tuning</th></tr></thead>
            <tbody>
              <tr><td><b>What changes</b></td><td>the instructions in the prompt</td><td>the facts in the prompt</td><td>the weights</td></tr>
              <tr><td><b>Good for</b></td><td>tasks the model can already do when told clearly</td><td>knowledge: private, recent, or changing</td><td>behaviour: style, tone, output format, a narrow skill</td></tr>
              <tr><td><b>Bad for</b></td><td>habits the model keeps slipping out of; very long instructions</td><td>changing how the model writes or reasons</td><td>facts that change (you would retrain for every edit), and you cannot cite a weight</td></tr>
              <tr><td><b>Update cost</b></td><td>edit a string</td><td>re-ingest a document</td><td>collect data, train, evaluate, deploy</td></tr>
              <tr><td><b>Needs</b></td><td>nothing</td><td>a search pipeline</td><td>hundreds of good examples, GPUs, an evaluation set</td></tr>
            </tbody>
          </table>
        </div>
        <Callout kind="dev">Rule of thumb: <b>RAG for what the model should know, fine-tuning for how the model should behave.</b> They combine well: a fine-tuned model that reliably follows your format, fed by retrieval for the facts.</Callout>

        <h3>The data is the specification</h3>
        <p>The loss rewards one thing: predicting your examples. So the model imitates <em>everything</em> in them.</p>
        <p>The format, yes. Also the typos, the inconsistent tone, the agent who always wrote “pls advise”, and any wrong answers.</p>
        <p>That is why a few hundred excellent, consistent examples usually beat many thousands of sloppy ones. With sloppy data you are fitting the model to the slop.</p>
        <Callout kind="established">This is an observed regularity, not a theorem, but it is widely reported: for teaching a style or format to an already capable model, quality and consistency of examples matter far more than count. Teaching genuinely new knowledge or skills needs much more data, and is less reliable.</Callout>

        <h3>Why models forget</h3>
        <p>In <a href="#/lesson/why-llms-know">Why LLMs know things</a> you saw that abilities are spread across shared weights. The same weight helps with many things at once.</p>
        <p>Now look at the loss during fine-tuning. It contains <em>only</em> your new examples. Nothing in it mentions French, or Python, or Shakespeare.</p>
        <p>Gradient descent moves every weight in whatever direction lowers the new loss. It pays no attention to what else that weight was doing.</p>
        <Term
          name="Catastrophic forgetting"
          plain={<>While learning the new task, the model gets measurably worse at things it used to do, because the same weights were overwritten.</>}
          example={<>Fine-tune a Shakespeare model on office English: office loss falls, Shakespeare loss rises.</>}
          formal={<>Minimising L<sub>new</sub>(θ) alone places no constraint on L<sub>old</sub>(θ). It is not a malfunction. It is the absence of any mechanism that could prevent it.</>}
        />
        <p>Amma has a version of this problem in her kitchen. Her mother’s recipe notebook is forty years old, and she will not write in it. Her own changes go on slips of paper tucked between the pages.</p>
        <Callout kind="analogy">
          Full fine-tuning <b>rewrites pages of the book</b>. LoRA leaves the book untouched and adds <b>sticky notes</b>: small corrections on top. Peel the notes off and the original book is back, exactly.
          <br /><br />
          Where the analogy stops: a sticky note covers one spot. A LoRA correction is added to the output of a whole layer, so it shifts the model’s behaviour everywhere, including on old tasks, for as long as it is attached.
        </Callout>
      </MentalModel>

      <TryIt title="Make a model forget, then stop it forgetting">
        <p>First run “Full fine-tune” to the end. Watch the dashed line. Then reset and try each mitigation.</p>
        <ForgettingLab />
        <p>Three things to take from it:</p>
        <ul>
          <li><b>Forgetting is a by-product of learning.</b> The old-task loss rises in the same steps in which the new-task loss falls.</li>
          <li><b>Replay works because it puts the old task back into the loss.</b> Now there <em>is</em> something protecting it. The price: slightly less progress on the new task.</li>
          <li><b>The adapter does not stop behaviour drifting while it is attached.</b> Its loss on A rises too. What it guarantees is different and valuable: the base weights never moved, so unplugging it restores the old model exactly.</li>
        </ul>
        <h3>The real experiment</h3>
        <p>The repository does this with the actual GPT from Part 7: pretrain on Shakespeare-style text, then fine-tune on plain modern English, once fully and once with LoRA. These are the measured validation losses from <code>python finetune_tiny_gpt.py --quick</code> (lower is better):</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>model</th><th>Shakespeare (old)</th><th>modern (new)</th><th>weights trained</th></tr></thead>
            <tbody>
              <tr><td>base</td><td>2.222</td><td>2.852</td><td /></tr>
              <tr><td>full fine-tune</td><td>2.674 (+0.451)</td><td>2.044</td><td>807,424 (100%)</td></tr>
              <tr><td>LoRA, r = 4, adapter attached</td><td>2.623 (+0.401)</td><td>2.376</td><td>12,288 (1.52%)</td></tr>
              <tr><td>LoRA, adapter removed</td><td colSpan={2}>identical to base: the script’s checksum shows no base weight moved</td><td /></tr>
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ fontSize: 14.5 }}>One run with the script’s fixed seed, 400 pretraining and 200 fine-tuning steps, on the machine used to write this lesson. Your third decimal may differ.</p>
        <p>Read it honestly. Full fine-tuning learned the new style best and lost 0.45 on the old one.</p>
        <p>LoRA trained <b>1.52%</b> of the weights and got about 60% of the improvement on the new style (0.48 of 0.81). With the adapter attached, its Shakespeare loss rose almost as much.</p>
        <p>The script’s checksum proves the part LoRA really guarantees. The 807,424 base weights did not move, so removing the adapter gives the base model back, byte for byte.</p>
        <h3>Why can so few numbers be enough?</h3>
        <LoraTrainer />
      </TryIt>

      <Numbers>
        <p>Finance will ask Riya what this costs, so she counts. Take one weight matrix from a 7-billion-parameter-class model: 4096 inputs, 4096 outputs.</p>
        <div className="table-scroll">
          <table className="plain">
            <tbody>
              <tr><td>Full fine-tuning trains</td><td className="mono">4096 × 4096 = 16,777,216 numbers</td></tr>
              <tr><td>LoRA with rank r = 8 trains A (4096 × 8) and B (8 × 4096)</td><td className="mono">8 × (4096 + 4096) = 65,536 numbers</td></tr>
              <tr><td>Ratio</td><td className="mono">65,536 / 16,777,216 = 0.39%</td></tr>
            </tbody>
          </table>
        </div>
        <p>The count grows with <span className="mono">r × (d_in + d_out)</span>, a sum, instead of <span className="mono">d_in × d_out</span>, a product. That is the entire saving. Try other shapes:</p>
        <LoraCalculator />
        <p>The same arithmetic explains the number in the real run above. Our tiny GPT has 4 blocks. In each, LoRA wraps the <span className="mono">qkv</span> layer (128 → 384) and the <span className="mono">proj</span> layer (128 → 128), with r = 4:</p>
        <p className="mono" style={{ fontSize: 14.5 }}>4 × [ 4 × (128 + 384) + 4 × (128 + 128) ] = 4 × [ 2,048 + 1,024 ] = 12,288</p>
      </Numbers>

      <TheMath>
        <p>A normal layer multiplies its input by a weight matrix. LoRA adds a second, thin path beside it and sums the two results.</p>
        <Equation
          label="h equals x W plus x A B times alpha over r"
          symbols={[
            ['x', 'the layer’s input: one row per token'],
            ['W', 'the pretrained weight matrix (d_in × d_out). Frozen: no gradient, never updated'],
            ['A', 'trainable, d_in × r. Starts as small random numbers (the LoRA paper uses a random Gaussian)'],
            ['B', 'trainable, r × d_out. Starts as all zeros, so A·B = 0 and the layer starts out exactly equal to the pretrained one'],
            ['r', 'the rank: the width of the thin path. Typical values are 4 to 64'],
            ['α / r', 'a fixed volume knob on the whole correction (in the repo: 8 / 4 = 2). Alpha is a number you choose, like r. Dividing by r is what makes the knob mean the same thing at different ranks, which is the paper’s stated reason: you do not have to retune the learning rate every time you change r'],
          ]}
        >
          h = x W + (x A) B · (α / r)
        </Equation>
        <Term
          name="Rank"
          plain={<>How many independent directions a matrix can write into. The product A·B is forced through a bottleneck of r numbers, so it can only ever make r independent kinds of change.</>}
          example={<>With r = 1, every row of A·B is a multiple of the same single row. With r = 2, every row is a mix of two rows.</>}
          formal={<>rank(AB) ≤ r. A full d × d matrix can have rank up to d.</>}
        />
        <p><b>Why would a low-rank correction be enough?</b> You are not teaching the model language from scratch. You are nudging a capable system.</p>
        <p>The empirical finding behind LoRA is that the <em>change</em> needed for such nudges is often close to low-rank, even though W itself is not. The lab above showed both sides: a rank-2 change is captured perfectly at r = 2, and an arbitrary change still has 42% error at r = 2.</p>
        <p><b>Merge or swap.</b> After training, you can compute W′ = W + A·B·(α/r) once and serve W′: same outputs, zero extra cost per token. Or keep W shared and swap small (A, B) pairs per customer or task.</p>
        <DeepDive title="Why must exactly one of A and B start at zero?">
          <p>The gradient that reaches A is (upstream gradient) × Bᵀ. The gradient that reaches B is Aᵀ × (upstream gradient). If B = 0 and A is random: A gets no gradient on the first step, but B does, so B moves away from zero, and from the next step A learns too.</p>
          <p>If <em>both</em> start at zero, both gradients are zero, on every step, forever. The adapter is dead. If neither is zero, training does not start from the pretrained model: you begin by adding random noise to every layer.</p>
          <p>Zero A with random B would work just as well mathematically. Random A with zero B is the convention of the LoRA paper (Hu et al., 2021), which the repo follows. In the paper’s notation the layer is written h = W₀x + BAx, so its A is also the matrix that reads the input and its B the one that writes the output.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>Build the LoRA layer in three steps. First, wrap an existing linear layer and freeze it:</p>
        <Code title="Step 1: freeze the pretrained layer">{`
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False        # no gradient: this layer can never change
`}</Code>
        <p><code>requires_grad = False</code> tells PyTorch not to compute a gradient for that weight. The optimizer then has nothing to update it with.</p>
        <Code title="Step 2: add the two thin trainable matrices">{`
        self.A = nn.Parameter(torch.randn(base.in_features, r) * 0.01)   # small random
        self.B = nn.Parameter(torch.zeros(r, base.out_features))         # zeros!
        self.scale = alpha / r
`}</Code>
        <Code title="Step 3: the forward pass is the equation">{`
    def forward(self, x):
        return self.base(x) + (x @ self.A) @ self.B * self.scale
`}</Code>
        <p>Note the brackets: <code>(x @ A) @ B</code> squeezes x down to r numbers first, then expands. The big d × d product A·B is never formed during training.</p>
        <p>The complete class from the repository, and the function that installs it on every attention layer:</p>
        <Code source="phase4-modern-llms/finetune_tiny_gpt.py" title="LoRALinear and apply_lora">{`
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                     # freeze the book
        self.A = nn.Parameter(torch.randn(base.in_features, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, base.out_features))  # zeros!
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A) @ self.B * self.scale


def apply_lora(model, r=4):
    """Wrap every attention Linear (qkv + proj) with LoRA; freeze the rest."""
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, r=r)
        block.attn.proj = LoRALinear(block.attn.proj, r=r)
    return model
`}</Code>
        <p>The training loop is the one from <a href="#/lesson/training-gpt">Training GPT</a>, with one line that matters: the optimizer is only given the weights that are allowed to move.</p>
        <Code source="phase4-modern-llms/finetune_tiny_gpt.py" title="the same loop, fewer parameters">{`
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=lr)
`}</Code>
        <p>Evaluation in the script logs <em>two</em> losses every 50 steps: the new corpus and the old one. Copy that habit. A fine-tune that is only evaluated on the new task will always look like a success.</p>
      </CodeIt>

      <BreakIt>
        <ul>
          <li><b>Turn the learning rate up</b> in the forgetting lab (15) and run a full fine-tune. Then try 2. With the small rate, after 300 steps the old-task loss has risen by 0.17 instead of 0.38. The new task is also less learned. “Small learning rate” is a forgetting control, not a free lunch.</li>
          <li><b>Train too long.</b> At learning rate 10, between step 150 and step 300 the new-task loss improves by only 0.04 while the old-task loss gets 0.08 worse. Past the plateau, every step is mostly damage. “How long do I fine-tune?” is a real setting, and you can only choose it if you are measuring the old tasks.</li>
          <li><b>Rank 0 and rank 6</b> in the patch lab. Rank 0 can learn nothing. Rank 6 on a 6×6 matrix trains 72 numbers to imitate 36: there is no saving left. LoRA only makes sense when r is much smaller than the layer.</li>
          <li><b>Arbitrary change, r = 1.</b> Two thirds of the wanted change remains out of reach (error 67%), no matter how long you train. Capacity is a hard limit, not a matter of patience.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="fine-tuning-code-lora" />
        <Exercise
          id="fine-tuning-calc-lora"
          type="calculate"
          title="Count the adapter"
          answer={{ value: 81920, tolerance: 0.5 }}
          answerLabel="trainable numbers in A and B"
          hints={['A has shape d_in × r and B has shape r × d_out.', 'A: 1024 × 16 = 16,384. B: 16 × 4096 = 65,536.', 'Add them. Or use r × (d_in + d_out) = 16 × 5120.']}
          solution={<><p>16 × (1024 + 4096) = 16 × 5120 = <b>81,920</b>.</p><p>The full matrix has 1024 × 4096 = 4,194,304 numbers, so the adapter is 1.95% of it. Doubling r doubles the adapter. Doubling both dimensions of the layer quadruples the full count but only doubles the adapter, which is why LoRA looks better the bigger the model is.</p></>}
        >
          <p>A feed-forward layer maps 1024 inputs to 4096 outputs. You attach a LoRA adapter with rank r = 16. How many trainable numbers do A and B contain in total?</p>
        </Exercise>

        <Exercise
          id="fine-tuning-predict-data"
          type="predict"
          title="Five thousand real chat logs"
          hints={['What does the loss reward during fine-tuning? Be literal.', 'The loss cannot tell which parts of an example you consider “the good part”.']}
          solution={<p>The model will learn to reproduce the logs, all of them: the correct policies and the agents’ shortcuts, the three different greeting styles, the “pls advise”, the occasional wrong refund amount, the habit of promising a callback. Cross-entropy rewards imitation of every token. Nothing marks some tokens as the intended lesson and others as noise. The better plan: pick or write a few hundred examples that are exactly what you want, consistent in format and tone, reviewed by a person. Then hold out 10 to 20% of them, never train on those, and compare the model’s outputs on them before and after.</p>}
        >
          <p>A team fine-tunes a support model on 5,000 unfiltered historical chat logs, reasoning that “more data is better”. Predict three specific things the fine-tuned model will do that the team will not like. What would you do instead?</p>
        </Exercise>

        <Exercise
          id="fine-tuning-debug-zero"
          type="debug"
          title="The adapter that never learns"
          hints={['Training runs without errors, but the loss is identical at step 0 and step 5,000. What does that say about the gradients?', 'Write down what the gradient for A is multiplied by, and what the gradient for B is multiplied by.', 'Gradient for A contains Bᵀ. Gradient for B contains Aᵀ. What if both are zero?']}
          solution={<><p>With A and B both zero, the gradient reaching A is (something) × Bᵀ = 0, and the gradient reaching B is Aᵀ × (something) = 0. Zero gradients mean zero updates, so both stay zero on the next step, and every step after it. The adapter is permanently dead and the loss never moves.</p><p>The fix is the repo’s initialisation: A small random, B zero. The product is still exactly zero at the start (training begins at the pretrained model), but B receives a non-zero gradient through A immediately, and A starts learning as soon as B has moved.</p><p>A second classic bug in the same code: forgetting <code>requires_grad = False</code> on the base layer. Nothing crashes. You are doing a full fine-tune with extra steps, and the “adapter file” you save no longer reproduces the model, because W changed too. The checksum in the script exists to catch exactly that.</p></>}
        >
          <p>A colleague “made the initialisation cleaner”. Training runs, but the loss does not change at all. Why?</p>
          <Code>{`
self.A = nn.Parameter(torch.zeros(base.in_features, r))
self.B = nn.Parameter(torch.zeros(r, base.out_features))
`}</Code>
        </Exercise>

        <details className="deep">
          <summary>More practice (optional)</summary>
          <div className="details-body">
          <Exercise
            id="fine-tuning-experiment"
            type="experiment"
            title="Run the real thing, then add replay"
            hints={['python phase4-modern-llms/finetune_tiny_gpt.py --quick takes a few minutes on a laptop CPU. Watch the “full” lines: two numbers moving in opposite directions.', 'For replay, in train() draw the batch from the old data one time in five: data_now = old_data if torch.rand(1).item() < 0.2 else data. You will need to pass shk_train in.', 'Compare the final scoreboard line for full-FT with and without replay.']}
            solution={<p>You should see the same scissors as in the lab: modern loss falling, Shakespeare loss rising, for both full fine-tuning and LoRA with the adapter attached. With 20% replay the Shakespeare loss rises much less, and the modern loss ends slightly higher than without replay. Exact numbers depend on your machine and seed. The point to notice: replay needs access to (some of) the old training data. For a model you downloaded, you usually do not have it, which is one reason frozen-base methods such as LoRA are so popular.</p>}
          >
            <p>Run <code>finetune_tiny_gpt.py --quick</code> and find the numbers of the table above in your own output: the scoreboard at the end, the “trainable” line and the checksum line. Then implement the oldest anti-forgetting trick: mix 20% Shakespeare batches into the full fine-tune. Predict first: what happens to each of the two losses?</p>
          </Exercise>

          <ExplainBack
            id="fine-tuning-explain"
            prompt="A teammate proposes: “Let’s fine-tune the model on our product documentation so it knows our product.” Explain what is likely to go wrong, and what you would do instead."
            modelAnswer={<p>Fine-tuning changes weights, which is the right tool for behaviour (style, format, a narrow skill) and a poor one for facts. Facts seen a few times in training are stored unreliably, so the model would sound like our docs while still inventing details. Every documentation change would need a new training run, and we could not show users where an answer came from. Meanwhile the fine-tune can damage abilities we rely on, because nothing in the training loss protects them. Documentation is knowledge that changes, so it belongs in the prompt via retrieval (RAG). If, after that, the model still does not answer in our format or tone, that is the moment to fine-tune, on a few hundred excellent examples, with a held-out set and a check of the old abilities.</p>}
          />
          </div>
        </details>
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'Why does a fully fine-tuned model get worse at things that are not in the fine-tuning data?',
            options: ['The optimizer deletes unused weights', 'The new loss only measures the new data, so nothing stops shared weights being moved away from what old abilities needed', 'The context window gets shorter', 'Fine-tuning data is always lower quality than pretraining data'],
            answer: 1,
            explain: 'Forgetting is the absence of protection. Replay helps precisely because it puts old data back into the loss.',
          },
          {
            q: 'After LoRA fine-tuning with the adapter attached, performance on an old task has dropped. What does LoRA still guarantee?',
            options: ['Nothing: this means the base weights were corrupted', 'That the old task will recover with more training', 'That the frozen base weights are unchanged, so removing the adapter restores the original model exactly', 'That the drop is smaller than 1%'],
            answer: 2,
            explain: 'In the repo run, Shakespeare loss rose from 2.222 to 2.623 with the adapter attached, and the base-weight checksum was unchanged.',
          },
          {
            q: 'Which need is the best fit for fine-tuning rather than RAG or prompting?',
            options: ['Answering questions about last week’s price list', 'Always replying in a strict JSON schema and a terse house style, which a long prompt achieves only 98% of the time', 'Citing the source paragraph for each answer', 'Letting users ask about their own uploaded PDF'],
            answer: 1,
            explain: 'Format and style are behaviour, stable over time, and hard to guarantee by instruction. The other three are about knowledge that changes or must be cited.',
          },
        ]}
      />

      <Remember
        items={[
          <><b>What changes: the weights. Not the prompt.</b> The same training loop on new data, from trained weights, with a small learning rate. <b>RAG for knowledge that changes, fine-tuning for behaviour</b> (style, format, narrow skills). Try prompting first.</>,
          <>The model imitates everything in your examples, including your mistakes. <b>A few hundred excellent examples beat thousands of sloppy ones.</b></>,
          <><b>Catastrophic forgetting:</b> nothing in the new loss protects old abilities, and they share weights. Always evaluate on a held-out set <em>and</em> re-test old capabilities.</>,
          <><b>LoRA:</b> freeze W, train a low-rank correction A·B with B = 0 at the start. About 1% of the numbers; base model preserved exactly; adapters can be merged or swapped.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Plain LLM' }, { label: 'RAG', sub: 'changes the prompt' }, { label: 'This lesson: fine-tuning', sub: 'changes the weights' }, { label: 'Agents', sub: 'changes the code around it' }]} active={2} />
        <ToyVsReal
          toy={<ul><li>A 0.8M-parameter character GPT; a 27×27 bigram in the browser</li><li>Two tiny synthetic corpora standing in for “old skill” and “new task”</li><li>Rank 4 on the attention layers: 12,288 trainable numbers</li><li>Forgetting measured as one loss on one old corpus</li></ul>}
          real={<ul><li>Billions of parameters; the pretrained weights are usually downloaded, not trained by you</li><li>Prompt → response pairs (SFT), often followed by preference tuning such as DPO (see <a href="#/lesson/alignment-safety">Alignment and safety</a>). The ideal replies are often written by a bigger model rather than by people (see <a href="#/lesson/distillation">Small models from big ones</a>)</li><li>LoRA on attention and often feed-forward layers, commonly combined with a quantised (compressed) frozen base so it fits on one GPU</li><li>Forgetting checked with a battery of benchmarks and task-specific regression tests</li></ul>}
        />
        <p>The assistant you chat with is itself a fine-tuned model: pretraining, then <G t="sft">SFT</G>, then preference tuning and reinforcement learning, as in <a href="#/lesson/training-pipeline">From raw text to assistant</a>. How those later stages teach a model what it should not do is the subject of <a href="#/lesson/alignment-safety">Alignment and safety</a>. Serving many LoRA adapters over one shared frozen base is standard practice, because swapping a small adapter is far cheaper than loading another full model.</p>
        <Callout kind="research">Why low-rank corrections work as well as they do, when they fall short of full fine-tuning, and how to fine-tune without forgetting are all open questions. Reported results differ by task: LoRA tends to match full fine-tuning on style and instruction-following, and to lag on tasks that need a lot of new knowledge or skill. One careful comparison on code and maths (Biderman et al., 2024, “LoRA Learns Less and Forgets Less”) found that LoRA learned less of the new domain and also forgot less of the old one than full fine-tuning. Our tiny run shows the first half clearly and the second only weakly, so do not generalise from it. A later study (Thinking Machines Lab, 2025, “LoRA Without Regret”) reported that LoRA applied to every layer, including the feed-forward ones, matched full fine-tuning on many post-training datasets, and fell behind mainly when the dataset was large. Treat any blanket claim in either direction with suspicion.</Callout>
        <p>Two weeks later Riya runs the held-out tickets through the adapter. The replies come back short, warm and in valid JSON. Then, following the script’s habit, she re-tests the old abilities too. Dev asks why she is checking things nobody complained about. “Because the loss never checked them,” she says.</p>
      </RealLLM>
    </Lesson>
  )
}
