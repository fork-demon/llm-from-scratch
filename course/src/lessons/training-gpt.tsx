import { GptTrainer } from '../interactive/GptTrainer'
import { CodeExercise } from '../components/python'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { TrainingPlayground } from '../interactive/TrainingPlayground'
import { TrainingLoop } from '../illustrations/TrainingLoop'

export default function TrainingGptLesson() {
  return (
    <Lesson id="training-gpt">
      <Why>
        <p className="lede">2 a.m. in the flat. Dev is asleep. Riya is on the sofa with her laptop, a cold cup of chai, and the GPT she assembled last night, now fed with Shakespeare.</p>
        <p>After a few seconds (100 steps) she asks it to write. It gives her this:</p>
        <div className="card mono" style={{ fontSize: 14, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{'ab uo  alp; drr n ooTEEuso\n e:\nss  mos:tetcee ppp a eiaa anellllllllllllpontsooourrouteteennne'}</div>
        <p>She almost closes the lid. Instead she lets it run. Six minutes later (3,000 steps), the <em>same code</em> writes this:</p>
        <div className="card mono" style={{ fontSize: 14, whiteSpace: 'pre-wrap' }}>{'That I craterk do ll wiangotave I hath. Thibe day,\n\nDethat Rikend now ms butis him thou\nDo et.\n\nIOF Mere tor th, mand decestin to this mow cre?\n\nMEOLIO:\nNoE I mithtur the lo blowhill braccheigere,'}</div>
        <p>Not Shakespeare. But look at what is there: real words (“hath”, “day”, “now”, “thou”, “to this”), line breaks, a speaker’s name in capitals followed by a colon, lines that start with a capital letter.</p>
        <p>Nobody programmed any of that. No line of code changed between the two samples. The only thing that changed is the values of the model’s 809,856 <G t="parameters">parameters</G>.</p>
        <p>She screenshots the falling loss curve and sends it to Kabir. At 2:07 a.m. he replies with one word: “Good.” So what exactly happened to those numbers?</p>
        <p><b>Training</b> is the process that changes those numbers. It is one short loop, repeated thousands of times: take some text, let the model guess each next character, measure how surprised it was, and nudge every number so it would be a little less surprised next time.</p>
        <p>You already own every piece: <a href="#/lesson/gradient-descent">gradient descent</a>, <a href="#/lesson/backprop">backpropagation</a>, <a href="#/lesson/softmax">cross-entropy</a>. This lesson is about how they fit together for text, and about the question every practitioner actually worries about: <em>how do I know it is working?</em></p>
      </Why>

      <Problem>
        <p>Next morning, Dev reads the samples over breakfast. “So you gave it a million questions and answers. Who wrote them?” Nobody did. That is the first of three practical questions the pieces leave open.</p>
        <ul>
          <li><b>Where are the examples?</b> Gradient descent needs (input, correct answer) pairs. We have a text file. Nobody labelled it.</li>
          <li><b>Is it learning?</b> A loss that falls is a good sign. But falling on <em>what</em>? A model can score perfectly by memorising.</li>
          <li><b>Which knobs matter?</b> Learning rate, batch size, number of steps. Pick badly and the loss explodes, or crawls.</li>
        </ul>
        <WhyExists
          problem="A GPT has around a million numbers (a real one: billions). They start random. We need good values for all of them."
          naive="Pay people to write (question, correct answer) pairs, as in classic supervised learning."
          fails="Far too slow and expensive. A language model needs billions of examples."
          idea="Let the text label itself. At every position, the correct answer is the character that comes next. Any text file is a huge pile of free examples."
          tradeoff="The model learns whatever is in the text, good or bad. And with a small file it can memorise instead of learning patterns, so we must hold some text back to check."
        />
      </Problem>

      <MentalModel>
        <h3>1. A text file becomes training pairs</h3>
        <p>You met this trick in <a href="#/lesson/next-token">Predicting the next token</a>: slide the text by one and it labels itself. Here it is with a longer input.</p>
        <p>Take the text “hello world” and a context length of 8. Cut out 8 characters as the input <code>x</code>. The target <code>y</code> is the <b>same stretch shifted one character to the right</b>.</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <tbody>
              <tr><td>position</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr>
              <tr><td>x (input)</td><td>h</td><td>e</td><td>l</td><td>l</td><td>o</td><td>␣</td><td>w</td><td>o</td></tr>
              <tr><td>y (answer)</td><td>e</td><td>l</td><td>l</td><td>o</td><td>␣</td><td>w</td><td>o</td><td>r</td></tr>
            </tbody>
          </table>
        </div>
        <p>Read one column: at position 4 the model has seen “hello” and the right answer is a space. Every column is its own little exam question.</p>
        <p>Here is the elegant part. Thanks to the <a href="#/lesson/masks-and-heads">causal mask</a>, position 4 cannot see positions 5, 6, 7. So one pass through the model answers all 8 questions at once, without cheating.</p>
        <p><b>One sequence of T tokens gives T training examples.</b></p>

        <h3>2. The loop</h3>
        <TrainingLoop />
        <p>One trip round this loop is called a <b>step</b>. A <G t="batch">batch</G> is the handful of stretches processed together in one step: 32 of them in our code. Averaging over 32 gives a steadier direction than a single example would.</p>
        <Term
          name="Step and epoch"
          plain={<>A <b>step</b> is one update of the weights, made from one batch. An <b>epoch</b> is however many steps it takes to have seen the whole training text once.</>}
          example={<>1,000,000 characters of text, batches of 32 × 64 = 2,048 characters: one epoch is about 488 steps.</>}
          formal={<>steps per epoch ≈ training tokens ÷ (batch size × context length)</>}
        />

        <h3>3. The exam with unseen questions</h3>
        <p>Before training, we cut the text in two: the first 90% to train on, the last 10% locked away. The model never takes a gradient step on the locked part. We only <em>measure</em> the loss there.</p>
        <p>Amma, when Riya explains this on the phone, recognises it at once: “We never gave the class the same paper we used for practice.”</p>
        <Term
          name="Validation loss"
          plain={<>The loss on text the model has never trained on. It tells you whether the model learned patterns that carry over, or just memorised.</>}
          example={<>train loss 1.2 and validation loss 1.3: healthy. Train loss 0.2 and validation loss 2.5: memorised.</>}
          formal={<>The same cross-entropy, computed on the held-out split, with no weight update.</>}
        />
        <Term
          name="Overfitting"
          plain={<>The point where more training makes the model <em>better</em> on its training text and <em>worse</em> on everything else.</>}
          example={<>Train loss keeps falling: 0.9, 0.6, 0.3. Validation loss turns around: 1.8, 1.9, 2.2.</>}
          formal={<>Validation loss increasing while training loss decreases: the gap between them is the generalisation gap.</>}
        />
        <p>The validation split is the new exam paper. Unlike a student, the model has no intention to cheat: memorising is simply the easiest way downhill when there is little data and a lot of capacity.</p>
        <p>Long runs save the weights to disk every so often. That saved file is a <b>checkpoint</b>. You keep the one with the best validation loss, not necessarily the last one.</p>
      </MentalModel>

      <TryIt title="Train one yourself">
        <p>This trains a real neural language model, from random weights, in your browser: the small character MLP from <code>bigram_lm.py</code> (Model C), which reads the last few characters (3 in the Python file, 4 by default here). It is <b>not a Transformer</b> and uses plain gradient descent, but the training <em>loop</em> is the one your GPT uses. A real Transformer comes right after it.</p>
        <p>Suggested route:</p>
        <ol>
          <li>Press <b>Start</b> with the defaults. Watch the loss start on the dotted “pure guessing” line and fall. Read the samples as they appear.</li>
          <li>Reset. Set the learning rate to <b>30</b>. Start. Then try <b>0.001</b>.</li>
          <li>Reset. Tick <b>Tiny dataset</b>, set hidden units to <b>256</b>, learning rate 0.3. Watch the two curves part ways.</li>
        </ol>
        <TrainingPlayground />
        <Callout kind="note">Why does the loss start on the dotted line? A fresh model with small starting weights knows nothing, so it spreads its probability evenly over all 27 characters. Each gets 1/27, and the surprise of a 1-in-27 event is ln(27) = 3.30.<br /><br />This is the most useful sanity check in the business: <b>if your untrained model does not start near ln(vocabulary size), something is wired wrong.</b> Keep it in mind. It is about to catch something in our own file.</Callout>
        <h3>Now train a real Transformer</h3>
        <p>This is the moment Riya waited for at 2 a.m. The model below is a real GPT: the same blocks as <code>tiny_gpt.py</code> (attention, MLP, LayerNorm, residuals), trained with AdamW and gradient clipping, all in your browser. It is small, about 28,000 parameters, so it trains in under a minute.</p>
        <ol>
          <li>Press <b>Train</b> on the Paisa Pal support tickets. Watch both loss curves fall and the samples turn from noise into something that looks like a ticket.</li>
          <li>Switch to Shakespeare. The text is only about 4,400 characters, so watch what the validation loss does after 20 seconds. You have seen this before: overfitting.</li>
          <li>Type a prompt and open the attention heatmap. These are learned patterns, not hand-set ones.</li>
        </ol>
        <GptTrainer corpus="tickets" size="small" />
      </TryIt>

      <Numbers>
        <p>Let’s compute a loss by hand. Pretend the vocabulary has only three characters: <code>e</code>, <code>a</code>, <code>x</code>. The model has read “th” and produces these <G t="logits">logits</G>:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>step</th><th>e</th><th>a</th><th>x</th></tr></thead>
            <tbody>
              <tr><td>logit</td><td className="mono">2.0</td><td className="mono">1.0</td><td className="mono">0.0</td></tr>
              <tr><td>e<sup>logit</sup></td><td className="mono">7.39</td><td className="mono">2.72</td><td className="mono">1.00</td></tr>
              <tr><td>÷ total (11.11) = probability</td><td className="mono"><b>0.665</b></td><td className="mono"><b>0.245</b></td><td className="mono"><b>0.090</b></td></tr>
              <tr><td>loss if this is the right answer: −ln(p)</td><td className="mono">0.41</td><td className="mono">1.41</td><td className="mono">2.41</td></tr>
            </tbody>
          </table>
        </div>
        <p>The text says “the”, so the right answer is <code>e</code>. The model gave it 66.5%. Loss = −ln(0.665) = <b>0.41</b>. A small surprise.</p>
        <p>Had the text said “thx”, the loss would be −ln(0.090) = <b>2.41</b>. A big surprise, a big loss, a big correction.</p>
        <p>A batch with those two examples has loss (0.41 + 2.41) ÷ 2 = <b>1.41</b>. That single averaged number is what <code>loss.backward()</code> starts from.</p>
        <p>And the untrained model? All logits 0, every probability 1/3, loss = ln(3) = <b>1.10</b> no matter what the answer is. With 65 characters, as in our Shakespeare file: ln(65) = <b>4.17</b>.</p>
      </Numbers>

      <TheMath>
        <p>Two formulas, both of which you have already used.</p>
        <Equation
          label="Loss equals minus the average of the log probability given to the correct token"
          symbols={[
            ['L', 'the loss: one number for the whole batch'],
            ['N', 'how many predictions were made: batch size × sequence length (32 × 64 = 2,048 in our code)'],
            [<>p<sub>i</sub>(correct)</>, 'the probability the model gave to the token that really came next, at prediction i'],
            ['ln', <>the natural logarithm. −ln turns a probability into a surprise: 1 → 0, 0.5 → 0.69, 0.01 → 4.6 (<a href="#/lesson/softmax">lesson 1.3</a>)</>],
          ]}
        >
          L = − (1 / N) · Σ<sub>i</sub> ln p<sub>i</sub>(correct)
        </Equation>
        <Equation
          label="Each weight moves against its gradient, scaled by the learning rate"
          symbols={[
            ['w', 'any one of the model’s parameters'],
            ['lr', <>the <G t="learning-rate">learning rate</G>: how big a step to take</>],
            ['∂L/∂w', <>the gradient for this weight: “if I increase w a hair, how much does the loss rise?” <a href="#/lesson/backprop">Backpropagation</a> computes it for all weights at once</>],
          ]}
        >
          w ← w − lr · ∂L/∂w
        </Equation>
        <p>The second formula is plain gradient descent, which is what the playground above uses. The real code uses a refined version of it called <b>AdamW</b>. It is the same idea, step downhill, with three improvements to how big each step is:</p>
        <ul>
          <li><b>Momentum.</b> Do not trust one noisy batch. Keep a running average of recent gradients and step along that. Like a heavy ball that keeps rolling through small bumps.</li>
          <li><b>A step size per parameter.</b> Some weights get huge gradients, others tiny ones. AdamW divides each weight’s step by the typical size of its recent gradients, so every weight moves at a sensible pace.</li>
          <li><b>Weight decay</b> (the W). Pull every weight slightly toward zero at each step. A mild brake on memorising.</li>
        </ul>
        <p>That is why AdamW works with a learning rate like 0.0003 while our plain gradient descent wants 0.3: the two numbers are not comparable.</p>
        <DeepDive title="The AdamW update, written out">
          <p>For each parameter w with gradient g at step t:</p>
          <ul>
            <li><span className="mono">m ← β₁·m + (1 − β₁)·g</span>: running average of the gradient (momentum). Default β₁ = 0.9.</li>
            <li><span className="mono">v ← β₂·v + (1 − β₂)·g²</span>: running average of the squared gradient (its typical size). Default β₂ = 0.999.</li>
            <li><span className="mono">m̂ = m / (1 − β₁ᵗ)</span>, <span className="mono">v̂ = v / (1 − β₂ᵗ)</span>: corrections for the fact that both averages start at zero.</li>
            <li><span className="mono">w ← w − lr · ( m̂ / (√v̂ + ε) + λ·w )</span>: the step. ε = 1e-8 avoids dividing by zero, λ is the weight decay (PyTorch default 0.01).</li>
          </ul>
          <p>Notice m̂ / √v̂ is roughly “gradient divided by its usual size”, a number near ±1. So every weight moves by about <span className="mono">lr</span> per step, whatever the raw scale of its gradient. The price: two extra numbers (m and v) stored for every parameter. Weights, m and v together take three times the memory of the weights alone, before counting gradients and activations.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>All of this is in <code>main()</code> of <code>tiny_gpt.py</code>. We will walk through it in the order it runs.</p>
        <Code title="Step 1: text → integers → a 90/10 split">{`
text = load_text()
chars = sorted(set(text))                      # the vocabulary: 65 characters
stoi = {c: i for i, c in enumerate(chars)}     # char -> id
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]      # val_data is never trained on
`}</Code>
        <Code title="Step 2: a batch of (x, y) pairs. y is x shifted by one">{`
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.context_len - 1, (batch_size,))   # 32 random start points
    x = torch.stack([d[i:i + cfg.context_len] for i in ix])           # (32, 64)
    y = torch.stack([d[i + 1:i + cfg.context_len + 1] for i in ix])   # (32, 64) one char later
    return x.to(device), y.to(device)
`}</Code>
        <p>Random start points mean there are no tidy epochs here: the code just runs a fixed number of steps.</p>
        <Code title="Step 3: forward pass and loss (inside GPT.forward)">{`
logits = self.head(x)                           # (B, T, vocab): a score for every char, at every position
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),   # flatten to (B*T, vocab)
                       targets.view(-1))                   # and (B*T,) correct ids
`}</Code>
        <p><code>F.cross_entropy</code> does the softmax, picks the probability of each correct character, takes −ln and averages: exactly the table you just did by hand, 2,048 times.</p>
        <Code title="Step 4: the three lines that do the learning">{`
opt.zero_grad(set_to_none=True)   # forget the gradients from the previous step
loss.backward()                   # backpropagation: a gradient for every parameter
opt.step()                        # AdamW nudges every parameter
`}</Code>
        <p><code>zero_grad</code> matters because PyTorch <em>adds</em> new gradients onto whatever is already stored. Forget it and every step uses the sum of all gradients so far.</p>
        <Code title="Step 5: measuring, without learning">{`
@torch.no_grad()                  # no gradients needed: we are only measuring
def eval_loss(split, iters=50):
    model.eval()                  # switch dropout off
    losses = [model(*get_batch(split))[1].item() for _ in range(iters)]
    model.train()                 # and back on
    return sum(losses) / len(losses)
`}</Code>
        <p>No <code>backward()</code>, no <code>opt.step()</code>. The validation text can be measured a thousand times and the model still has never learned from it.</p>
        <p>Put together, this is the training loop from the repository, unchanged:</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="the whole training loop">{`
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(steps + 1):
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()          # <- module 03, automated
    opt.step()               # <- module 02, with adaptive step sizes

    if step % (100 if args.quick else 500) == 0:
        tl, vl = eval_loss("train"), eval_loss("val")
        print(f"\\n--- step {step}: train loss {tl:.3f}, val loss {vl:.3f} ---")
        print(sample(150))
`}</Code>
        <Callout kind="warn" label="The sanity check, applied to our own file">
          <p>Run the file and step 0 prints a loss of about <b>80</b>, not 4.17. The check has caught something real. Riya’s first reaction at 2 a.m. was “bug in cross-entropy?” It is not. The untrained model is doing something very specific: <b>confident copying</b>.</p>
          <p>We ran the untrained model on a batch and looked at its top choice. At <b>every</b> position it predicted the very character it had been given, with probability close to 100%.</p>
          <p>Here is why, step by step:</p>
          <ol>
            <li>PyTorch fills the embedding table with random numbers of size about 1. So a token’s embedding <i>e</i> has 128 such numbers, and |e|² (the dot product of e with itself) is about 128.</li>
            <li>The <a href="#/lesson/transformer-block">residual stream</a> carries e all the way up. The random blocks only add smaller corrections, and the position vector points in an unrelated direction.</li>
            <li>The final LayerNorm rescales the result. What comes out is roughly (e + position) / √2.</li>
            <li>With weight tying, the logit for a character is that output dotted with the character’s own embedding. For the input character itself this is about |e|² / √2 ≈ 128 / 1.41 ≈ 90. For any other character it is a random dot product near 0 (give or take about 11).</li>
          </ol>
          <p>A gap of around 90 in the logits means softmax gives nearly everything to “repeat the input”. But the right answer is the <em>next</em> character, which equals the current one only about 2% of the time. So the correct character gets a logit about 85 below the winner, and the loss is in the 80s.</p>
          <p>The run recovers: it spends its first hundred or so steps unlearning the copying and climbing down to the guessing line (3.88 at step 100). GPT-2 style code avoids the whole detour by starting small. Every weight starts with standard deviation 0.02, and the layers that write into the residual stream are shrunk by a further 1/√(2N) for N blocks. With that initialisation our model starts at about 4.2, right on ln(65). More on this recipe below.</p>
        </Callout>
        <p>Look how little of this is about language. Swap <code>get_batch</code> and the model, and the same eight lines train an image classifier. The file saves no checkpoint, because the run takes minutes; adding <code>torch.save(model.state_dict(), "ckpt.pt")</code> whenever <code>vl</code> improves is a two-line change.</p>

        <h3>The rest of the recipe</h3>
        <p><code>tiny_gpt.py</code> keeps things bare: default initialisation, a constant learning rate, nothing else. Real training runs add four small things around the same loop. Each one fixes a specific way a run goes wrong.</p>
        <ul>
          <li><b>Small, depth-scaled initialisation.</b> Start every weight small (GPT-2: standard deviation 0.02), and shrink the layers that write into the <G t="residual">residual stream</G> by a further 1/√(2N) for N blocks. Why: the stream adds up 2N such contributions, so without the shrink its size would grow with depth. This is also what makes step 0 start near ln(V) instead of copying.</li>
          <li><b>Warm-up.</b> For the first few hundred to few thousand steps, raise the learning rate in a straight line from near zero to its peak. Why: at the start AdamW’s running averages are built from only a few batches, so large steps can throw the run into a bad region.</li>
          <li><b>Cosine decay.</b> After warm-up, lower the learning rate along half a cosine curve, often to about a tenth of the peak. Why: small steps late let the weights settle into a low point instead of bouncing around it.</li>
          <li><b>Gradient clipping.</b> If the length of the whole gradient (all parameters as one long vector) is above a limit, commonly 1.0, scale it down to that length, keeping its direction. Why: one odd batch can produce a huge step that undoes hours of training.</li>
        </ul>
        <Code title="Sketch: warm-up + cosine decay, and clipping (not in tiny_gpt.py)">{`
def lr_at(step, peak=3e-4, warmup=200, total=3000, floor=3e-5):
    if step < warmup:
        return peak * (step + 1) / warmup                 # straight ramp up
    progress = (step - warmup) / (total - warmup)         # 0 -> 1
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))

for g in opt.param_groups:
    g["lr"] = lr_at(step)                                 # set before opt.step()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # cap the gradient's length
opt.step()
`}</Code>
        <p>None of these four changes the goal: same data, same loss, same loop. They change how safely and quickly the run gets there. The exact values (warm-up length, peak, floor, clip limit) are found by experiment and differ between labs.</p>
      </CodeIt>

      <BreakIt>
        <p>Back to the playground. Predict first, then check.</p>
        <ul>
          <li><b>Learning rate 30.</b> What will the curve do? (It never gets going: the first updates overshoot so far that the numbers overflow into NaN within a few steps.) Now try <b>3</b>: it learns for a while, then blows up. Divergence can arrive late.</li>
          <li><b>Learning rate 0.001.</b> Nothing is wrong, but after 2,000 steps the loss has barely left the guessing line. Too small a step is a bug you pay for in time.</li>
          <li><b>Batch size 4, then 128.</b> Compare how jagged the curve is. Small batches give noisy directions. Large ones are steadier but each step costs more.</li>
          <li><b>Context length 1, then 8.</b> With one character of context the loss gets stuck near 1.75, and no amount of training helps: the information is not in the input. This is <a href="#/lesson/context-wall">the context wall</a> again.</li>
          <li><b>Tiny dataset + 256 hidden units.</b> Train loss heads for 0.1. Validation loss bottoms out early and climbs above 2. Read the samples: the model recites its 300 characters.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="training-gpt-code-get-batch" />
        <CodeExercise id="training-gpt-code-cross-entropy" />
        <Exercise
          id="training-gpt-uniform"
          type="predict"
          title="The loss of knowing nothing"
          answer={{ value: 4.17, tolerance: 0.02 }}
          answerLabel="loss (two decimals)"
          hints={[
            'Guessing uniformly over 65 characters means every character gets probability 1/65.',
            'The loss for one prediction is −ln(p of the correct character). Here that p is always 1/65.',
            '−ln(1/65) = ln(65). Use a calculator, or Python: math.log(65).',
          ]}
          solution={<><p>Every prediction gives the correct character 1/65, so every prediction has loss −ln(1/65) = ln(65) ≈ <b>4.17</b>, and so does the average.</p><p>Use this before any long run. If step 0 prints a loss far above ln(vocab size), the model is confidently wrong before it has learned anything, which usually means badly scaled initial weights. Far <em>below</em> it at step 0 means the answer is leaking into the input.</p></>}
        >
          <p>Tiny Shakespeare has 65 distinct characters. A model that guesses uniformly at random has what cross-entropy loss?</p>
        </Exercise>

        <Exercise
          id="training-gpt-debug"
          type="debug"
          title="The loss falls, then goes wild"
          hints={[
            'Compare with the real loop line by line. One line is missing.',
            'In PyTorch, loss.backward() ADDS the new gradients to the ones already stored in each parameter.',
          ]}
          solution={<><p><code>opt.zero_grad()</code> is missing. Gradients accumulate: at step 100 the optimizer is stepping along the sum of 100 batches’ gradients, most of them computed for weights that no longer exist. The direction is stale and barely responds to the current batch, so training becomes unstable. (With plain gradient descent the step would also grow without limit. AdamW rescales the step size, but not the stale direction.)</p><p>A second, quieter bug is in the last line: it reports the loss on a <em>training</em> batch. That number will look great even if the model is memorising. Use <code>eval_loss("val")</code>.</p></>}
        >
          <p>A colleague’s training run starts well, then becomes erratic. They also claim “loss 0.3, the model is great”. Find both problems.</p>
          <Code>{`
for step in range(steps):
    x, y = get_batch("train")
    _, loss = model(x, y)
    loss.backward()
    opt.step()
print("final loss", loss.item())
`}</Code>
        </Exercise>

        <details className="deep">
          <summary>More practice (optional)</summary>
          <div className="details-body">
          <Exercise
            id="training-gpt-lr-experiment"
            type="experiment"
            title="Three learning rates"
            hints={[
              'Reset between runs so each one starts from the same random weights. Defaults for everything else. Use Pause at step 1,000.',
              'Record three things for each run: did it diverge, the validation loss at step 1,000, and what the sample looks like.',
            ]}
            solution={<><p>With the default settings you should see roughly: <b>0.01</b> → validation loss about 2.8 (still mostly gibberish, it is only slow). <b>0.3</b> → about 0.4 (real words and sentences). <b>3</b> → the loss falls fast, then explodes to NaN a little after step 200.</p><p>The lesson: the learning rate is the most sensitive knob in training. Too low wastes compute, too high destroys the run, and the best value sits surprisingly close to the edge of disaster. This is why real training runs change the learning rate as they go, on a fixed plan called a schedule. The usual shape: a short <b>warm-up</b>, where the rate climbs from near zero over the first few thousand steps, then a long slow decay back down.</p></>}
          >
            <p>In the playground, train for 1,000 steps at learning rates <b>0.01</b>, <b>0.3</b> and <b>3</b> (everything else default). Write down the final validation loss of each. Which is best, and what goes wrong with the other two?</p>
          </Exercise>

          <Exercise
            id="training-gpt-implement"
            type="implement"
            title="Run the real one"
            hints={[
              'cd phase3-transformers, then: python tiny_gpt.py --quick   (needs PyTorch; about two minutes on a laptop CPU).',
              'Compare every printed loss with ln(65) = 4.17. Then read the four samples: look for spaces first, then line breaks, then word-like strings.',
              'For the second part, add one line at the end of GPT.__init__:  nn.init.normal_(self.tok_emb.weight, std=0.02)   and run again.',
            ]}
            solution={<>
              <p><b>What you will see.</b> Our quick run printed train / validation loss of <b>80.5 / 80.3</b> at step 0, then 3.88 / 3.96, 3.16 / 3.21 and 2.93 / 2.96 at steps 100, 200, 300. The full run (3,000 steps) reached 2.04 / 2.10.</p>
              <p>With the one-line change, step 0 printed <b>4.14</b>. Then, in our run, the loss dropped to about 3.3 and sat there for the rest of the 300 steps, ending <em>worse</em> than the original.</p>
              <p><b>Why.</b> Step 0 at 80 is the confident copying from the warning box: large embeddings, tied head, each input character predicts itself. Shrinking the embeddings removes the copying, so the model starts at ln(65) as it should.</p>
              <p>Train and validation stay close because after 300 steps the model has seen 300 × 32 × 64 ≈ 614,000 characters, less than one pass over the one million characters of training text. It has had no chance to memorise.</p>
              <p>The plateau at 3.3 is roughly what you get by predicting how common each character is while ignoring the context. A likely reason: the tied output layer is now tiny, so the learning signal that flows back through it is tiny too, and at a fixed learning rate of 3e-4 it takes a while to grow. Real recipes pair small initial weights with a warm-up and a larger peak learning rate (the recipe section).</p>
              <p><b>Takeaway.</b> The ln(V) check tells you whether the start is sane. It does not tell you the rest of the recipe is tuned.</p>
            </>}
          >
            <p>Run <code>python tiny_gpt.py --quick</code>. Note the train and validation loss at steps 0, 100, 200, 300 and read the samples. Is the step 0 loss what the sanity check says it should be? Then make the one-line change in hint 3 and run again. Predict first: what will step 0 print now, and will the run end better or worse?</p>
          </Exercise>

          <ExplainBack
            id="training-gpt-explain"
            prompt="Your product manager looks at a chart and says: “Training loss is still going down, so why did you stop the run?” Explain overfitting and what the validation curve told you."
            modelAnswer={<p>The training loss only says how well the model predicts text it has already practised on. A model with enough capacity can keep lowering it by memorising that text. What we care about is new text, so we hold some text back and measure the loss on it without ever training on it. While both curves fall, the model is learning patterns that carry over. When the validation loss flattens and turns upward while the training loss keeps falling, further training is making the model worse at the real job. So we stop there and keep the checkpoint with the best validation loss. The fix for wanting more is more data (or a smaller model, or regularisation), not more steps.</p>}
          />
          </div>
        </details>
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'A training sequence is 64 tokens long. How many next-token predictions does the model get graded on from that one sequence?',
            options: ['64: one at every position, because the causal mask stops each position from seeing its own answer', '1: only the token after the last position', '32: every other position', '64 × 64'],
            answer: 0,
            explain: 'y is x shifted by one. Every position predicts its next token in the same forward pass, and the mask guarantees nobody peeks.',
          },
          {
            q: 'Train loss 0.15, validation loss 2.4 and rising. What is happening?',
            options: ['The learning rate is too small', 'The model needs more steps', 'The validation split must be broken, because a model that good on its training text cannot be bad elsewhere', 'The model is memorising the training text; its skill does not carry over to unseen text'],
            answer: 3,
            explain: 'A large and growing gap is the signature of overfitting. More steps make it worse. More data, a smaller model or stopping earlier help.',
          },
          {
            q: 'Why is eval_loss wrapped in torch.no_grad() and never followed by opt.step()?',
            options: ['To make it run on the CPU', 'Because measuring must not change the model: if we stepped on validation data it would stop being unseen', 'Because validation text has no “next character” targets, so there is no loss to take a gradient of', 'Because AdamW cannot handle two datasets'],
            answer: 1,
            explain: 'The validation split is only useful as long as no gradient from it ever reaches the weights.',
          },
        ]}
      />

      <Remember
        items={[
          <>Text labels itself: <b>y is x shifted by one</b>. Every position is an example, and the causal mask lets one sequence of T tokens give T predictions in a single pass.</>,
          <>The loop: <b>batch → forward → loss → backward → step</b>. In code: <code>get_batch</code>, <code>model(x, y)</code>, <code>zero_grad</code>, <code>loss.backward()</code>, <code>opt.step()</code>. The <b>learning rate</b> is the touchiest knob: too small crawls, too large diverges.</>,
          <>Loss is <b>average surprise</b>. A healthy untrained model starts at <b>ln(vocabulary size)</b>: 4.17 for 65 characters. Check it every time: in our own file it caught confident copying, caused by large starting weights.</>,
          <>Hold text back. <b>Validation loss</b> is the honest score. Train loss falling while validation loss rises is <b>overfitting</b>: memorising, not generalising.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Playground', sub: 'MLP, 8,000 chars, seconds' }, { label: 'tiny_gpt.py', sub: 'Transformer, 1M chars, minutes' }, { label: 'GPT-2', sub: '40 GB of web text, many GPUs' }, { label: 'Frontier LLM', sub: 'trillions of tokens, months' }]} active={1} />
        <ToyVsReal
          toy={<ul><li>One CPU, a 1 MB text file</li><li>300 to 3,000 steps, 2,048 tokens per step</li><li>Constant learning rate 3e-4</li><li>Overfitting is a real risk: the file is small</li><li>No checkpoints: a crash costs two minutes</li></ul>}
          real={<ul><li>Thousands of GPUs, trillions of tokens of filtered web text, code and books</li><li>Hundreds of thousands of steps, millions of tokens per step</li><li>Learning rate warms up from zero, then decays</li><li>Often about one epoch: most text is seen once, so classic overfitting is less of a worry than data quality</li><li>Checkpoints every few hours; runs are restarted from them after hardware failures and loss spikes</li></ul>}
        />
        <Callout kind="established">The loop is the same. Pretraining a frontier model is this lesson’s five steps, the same cross-entropy loss and the same AdamW family of optimizers. What changes is the engineering needed to run it on thousands of machines at once, and the care put into the data.</Callout>
        <h3>Two back-of-envelope numbers</h3>
        <p><b>Memory: about 16 bytes per parameter.</b> A common setup is AdamW with mixed precision: the maths runs in 16-bit, but a 32-bit copy of the weights is kept for the updates. Per parameter that is:</p>
        <div className="table-scroll">
          <table className="plain">
            <tbody>
              <tr><td>16-bit weights</td><td className="mono">2 bytes</td></tr>
              <tr><td>16-bit gradients</td><td className="mono">2 bytes</td></tr>
              <tr><td>32-bit master copy of the weights</td><td className="mono">4 bytes</td></tr>
              <tr><td>AdamW’s two running averages (m and v), 32-bit</td><td className="mono">4 + 4 bytes</td></tr>
              <tr><td><b>total</b></td><td className="mono"><b>16 bytes</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>So a 7-billion-parameter model needs about 7 × 10⁹ × 16 = 112 GB for this alone, before counting the activations kept for backprop. That is more than most single GPUs hold, which is one reason training is split across many.</p>
        <p><b>Compute: about 6 · N · D operations.</b> For N parameters and D training tokens, the forward pass costs about 2 operations per parameter per token (one multiply, one add), and the backward pass about twice that. 2 + 4 = 6.</p>
        <p>Our tiny GPT: 6 × 809,856 × (3,000 steps × 2,048 tokens) ≈ 3 × 10¹³. A 70B model on 1.4 trillion tokens: 6 × 7 × 10¹⁰ × 1.4 × 10¹² ≈ 5.9 × 10²³. The same formula, ten orders of magnitude apart.</p>
        <p>Both are rules of thumb. The 16 bytes ignores activations and varies with the setup. 6·N·D ignores the attention scores, which add a little more at long context.</p>
        <Callout kind="model">“Loss goes down, so the model gets smarter” is a useful simplification. Lower next-token loss correlates well with better capabilities, but the link to any specific skill is not a simple formula, and it is an area of active study. You will meet this again in <a href="#/lesson/why-llms-know">Why LLMs know things</a>.</Callout>
        <p>What you trained here is called a <b>base model</b>: it continues text. Turning it into an assistant takes further training stages, covered in <a href="#/lesson/training-pipeline">From raw text to assistant</a>. Next, though: the weights are frozen, and we make the model talk.</p>
        <p>Riya saves Kabir’s one-word reply. On Monday she has to demo this to the team, live, and that is where a new problem is waiting.</p>
      </RealLLM>
    </Lesson>
  )
}
