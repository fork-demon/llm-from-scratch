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
        <p className="lede">In the last lesson you assembled a GPT. Train it on Shakespeare for a few seconds (100 steps) and ask it to write. You get this:</p>
        <div className="card mono" style={{ fontSize: 14, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{'ab uo  alp; drr n ooTEEuso\n e:\nss  mos:tetcee ppp a eiaa anellllllllllllpontsooourrouteteennne'}</div>
        <p>Six minutes later (3,000 steps), the <em>same code</em> writes this:</p>
        <div className="card mono" style={{ fontSize: 14, whiteSpace: 'pre-wrap' }}>{'That I craterk do ll wiangotave I hath. Thibe day,\n\nDethat Rikend now ms butis him thou\nDo et.\n\nIOF Mere tor th, mand decestin to this mow cre?\n\nMEOLIO:\nNoE I mithtur the lo blowhill braccheigere,'}</div>
        <p>Not Shakespeare. But look at what is there: real words (“hath”, “day”, “now”, “thou”, “to this”), line breaks, a speaker’s name in capitals followed by a colon, lines that start with a capital letter. Nobody programmed any of that. No line of code changed between the two samples. The only thing that changed is the values of the model’s 809,856 <G t="parameters">parameters</G>.</p>
        <Callout kind="idea">
          <b>Training</b> is the process that changes those numbers. It is one short loop, repeated thousands of times: take some text, let the model guess each next character, measure how surprised it was, and nudge every number so it would be a little less surprised next time.
        </Callout>
        <p>You already own every piece: <a href="#/lesson/gradient-descent">gradient descent</a>, <a href="#/lesson/backprop">backpropagation</a>, <a href="#/lesson/softmax">cross-entropy</a>. This lesson is about how they fit together for text, and about the question every practitioner actually worries about: <em>how do I know it is working?</em></p>
      </Why>

      <Problem>
        <p>The pieces leave three practical questions open.</p>
        <div className="grid-3">
          <div className="card"><h4 style={{ fontSize: 17, marginBottom: 6 }}>Where are the examples?</h4><p>Gradient descent needs (input, correct answer) pairs. We have a text file. Nobody labelled it.</p></div>
          <div className="card"><h4 style={{ fontSize: 17, marginBottom: 6 }}>Is it learning?</h4><p>A loss that falls is a good sign. But falling on <em>what</em>? A model can score perfectly by memorising.</p></div>
          <div className="card"><h4 style={{ fontSize: 17, marginBottom: 6 }}>Which knobs matter?</h4><p>Learning rate, batch size, number of steps. Pick badly and the loss explodes, or crawls.</p></div>
        </div>
        <WhyExists
          problem="A GPT has around a million numbers (a real one: billions). They start random. We need good values for all of them."
          naive="Pay people to write (question, correct answer) pairs, as in classic supervised learning."
          fails="Far too slow and expensive. A language model needs billions of examples."
          idea="Let the text label itself. At every position, the correct answer is simply the character that comes next. Any text file is a huge pile of free examples."
          tradeoff="The model learns whatever is in the text, good or bad. And with a small file it can memorise instead of learning patterns, so we must hold some text back to check."
        />
      </Problem>

      <MentalModel>
        <h3>1. A text file becomes training pairs</h3>
        <p>You met this trick in <a href="#/lesson/next-token">Predicting the next token</a>: slide the text by one and it labels itself. Here it is with a longer input. Take the text “hello world” and a context length of 8. Cut out 8 characters as the input <code>x</code>. The target <code>y</code> is the <b>same stretch shifted one character to the right</b>.</p>
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
        <p>Here is the elegant part. Thanks to the <a href="#/lesson/masks-and-heads">causal mask</a>, position 4 cannot see positions 5, 6, 7. So one pass through the model answers all 8 questions at once, without cheating. <b>One sequence of T tokens gives T training examples.</b></p>

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
        <Callout kind="analogy">
          A student with last year’s exam can learn the subject, or can memorise the answer key. Both score 100% on last year’s exam. Only a <em>new</em> exam tells them apart. The validation split is the new exam.
          <br /><br />
          Where the analogy stops: the model has no intention to cheat. Memorising is simply the easiest way downhill when there is little data and a lot of capacity. Gradient descent takes whatever reduces the training loss.
        </Callout>
        <p>Because a run can go bad (or the power can go out), long runs save the weights to disk every so often. That saved file is a <b>checkpoint</b>. You keep the one with the best validation loss, not necessarily the last one.</p>
      </MentalModel>

      <TryIt title="Train one yourself">
        <p>This trains a real model, from random weights, in your browser. Suggested route:</p>
        <ol>
          <li>Press <b>Start</b> with the defaults. Watch the loss start on the dotted “pure guessing” line and fall. Read the samples as they appear.</li>
          <li>Reset. Set the learning rate to <b>30</b>. Start. Then try <b>0.001</b>.</li>
          <li>Reset. Tick <b>Tiny dataset</b>, set hidden units to <b>256</b>, learning rate 0.3. Watch the two curves part ways.</li>
        </ol>
        <TrainingPlayground />
        <Callout kind="note">Why does the loss start exactly on the dotted line? A fresh model knows nothing, so it spreads its probability evenly over all 27 characters. Each gets 1/27, and the surprise of a 1-in-27 event is ln(27) = 3.30. This is the most useful sanity check in the business: <b>if your untrained model does not start near ln(vocabulary size), something is wired wrong.</b></Callout>
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
        <p>The second formula is plain gradient descent, which is what the playground above uses. The real code uses a refinement called <b>AdamW</b>. The intuition:</p>
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
        <p>Random start points mean the model rarely sees exactly the same stretch twice. There are no tidy epochs here: the code just runs a fixed number of steps.</p>
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
          Run the file and step 0 prints a loss of about <b>80</b>, not 4.17. The check has caught something real. PyTorch fills an embedding table with numbers of size about 1, and this model reuses that table as its output layer (weight tying). Multiply two such 128-number vectors and you get logits in the tens: a model that is <em>confidently wrong</em> before it has learned anything. It spends its first hundred or so steps just climbing back down to the guessing line (3.88 at step 100). The run recovers, so the file gets away with it. GPT-2 style code avoids the problem by starting with much smaller numbers (standard deviation 0.02).
        </Callout>
        <Callout kind="dev">Look how little of this is about language. Swap <code>get_batch</code> and the model, and the same eight lines train an image classifier or a speech recogniser. The loop is the constant of deep learning. Also note what this file does <em>not</em> do: it saves no checkpoint, because the run takes minutes. Adding <code>torch.save(model.state_dict(), "ckpt.pt")</code> whenever <code>vl</code> improves is a two-line change.</Callout>
      </CodeIt>

      <BreakIt>
        <p>Back to the playground. Predict first, then check.</p>
        <ul>
          <li><b>Learning rate 30.</b> What will the curve do? (It never gets going: the first updates overshoot so far that the numbers overflow into NaN within a few steps.) Now try <b>3</b>: it learns for a while, then blows up. Divergence can arrive late.</li>
          <li><b>Learning rate 0.001.</b> Nothing is wrong, but after 2,000 steps the loss has barely left the guessing line. Too small a step is a bug you pay for in time.</li>
          <li><b>Batch size 4, then 128.</b> Compare how jagged the curve is. Small batches give noisy directions. Large ones are steadier but each step costs more.</li>
          <li><b>Context length 1, then 8.</b> With one character of context the loss gets stuck near 1.75, and no amount of training helps: the information is simply not in the input. This is <a href="#/lesson/context-wall">the context wall</a> again.</li>
          <li><b>Tiny dataset + 256 hidden units.</b> Train loss heads for 0.1. Validation loss bottoms out early and climbs above 2. Read the samples: the model recites its 300 characters.</li>
        </ul>
      </BreakIt>

      <Exercises>
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
          id="training-gpt-lr-experiment"
          type="experiment"
          title="Three learning rates"
          hints={[
            'Reset between runs so each one starts from the same random weights. Defaults for everything else. Use Pause at step 1,000.',
            'Record three things for each run: did it diverge, the validation loss at step 1,000, and what the sample looks like.',
          ]}
          solution={<><p>With the default settings you should see roughly: <b>0.01</b> → validation loss about 2.8 (still mostly gibberish, it is simply slow). <b>0.3</b> → about 0.4 (real words and sentences). <b>3</b> → the loss falls fast, then explodes to NaN a little after step 200.</p><p>The lesson: the learning rate is the most sensitive knob in training. Too low wastes compute, too high destroys the run, and the best value sits surprisingly close to the edge of disaster. This is why real training runs use a schedule: a short warm-up from zero, then a slow decay.</p></>}
        >
          <p>In the playground, train for 1,000 steps at learning rates <b>0.01</b>, <b>0.3</b> and <b>3</b> (everything else default). Write down the final validation loss of each. Which is best, and what goes wrong with the other two?</p>
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

        <Exercise
          id="training-gpt-implement"
          type="implement"
          title="Run the real one"
          hints={[
            'cd phase3-transformers, then: python tiny_gpt.py --quick   (needs PyTorch; about two minutes on a laptop CPU).',
            'Compare every printed loss with ln(65) = 4.17. Then read the four samples: look for spaces first, then line breaks, then word-like strings.',
            'For the second part, add one line at the end of GPT.__init__:  nn.init.normal_(self.tok_emb.weight, std=0.02)   and run again.',
          ]}
          solution={<><p>Our quick run printed train / validation loss of <b>80.5 / 80.3</b> at step 0, then 3.88 / 3.96, 3.16 / 3.21 and 2.93 / 2.96 at steps 100, 200, 300. The two numbers stay close together: after 300 steps the model has seen 300 × 32 × 64 ≈ 614,000 characters, less than one pass over the one million characters of training text, so it has had no chance to memorise. The full run (3,000 steps) reached 2.04 / 2.10.</p><p>With the small initialisation, step 0 printed <b>4.14</b>: the sanity check now passes. The surprise is what happened next in our run: the loss dropped to about 3.3 and sat there for the rest of the 300 steps, ending <em>worse</em> than the original. 3.3 is roughly what you get by predicting how common each character is while ignoring the context. A likely reason: the tied output layer is now tiny, so the learning signal that flows back through it is tiny too, and at a fixed learning rate of 3e-4 it takes a while to grow. Real recipes pair small initial weights with a learning-rate warm-up and a larger peak learning rate. The lesson: the ln(V) check tells you whether the start is sane. It does not tell you the rest of the recipe is tuned.</p></>}
        >
          <p>Run <code>python tiny_gpt.py --quick</code>. Note the train and validation loss at steps 0, 100, 200, 300 and read the samples. Is the step 0 loss what the sanity check says it should be? Then make the one-line change in hint 3 and run again. Predict first: what will step 0 print now, and will the run end better or worse?</p>
        </Exercise>

        <ExplainBack
          id="training-gpt-explain"
          prompt="Your product manager looks at a chart and says: “Training loss is still going down, so why did you stop the run?” Explain overfitting and what the validation curve told you."
          modelAnswer={<p>The training loss only says how well the model predicts text it has already practised on. A model with enough capacity can keep lowering it by memorising that text. What we care about is new text, so we hold some text back and measure the loss on it without ever training on it. While both curves fall, the model is learning patterns that carry over. When the validation loss flattens and turns upward while the training loss keeps falling, further training is making the model worse at the real job. So we stop there and keep the checkpoint with the best validation loss. The fix for wanting more is more data (or a smaller model, or regularisation), not more steps.</p>}
        />
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
            q: 'Your freshly initialised model with a 50,000-token vocabulary reports a loss of 10.8 at step 0. What do you conclude?',
            options: ['Something is broken: the loss should start near 0', 'The learning rate is too high', 'That is about ln(50,000) = 10.8: the model is guessing uniformly, as an untrained model should', 'The model is overfitting'],
            answer: 2,
            explain: 'ln(vocabulary size) is the loss of knowing nothing. Starting there is the sign of a healthy setup.',
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
          {
            q: 'What does AdamW add to plain gradient descent?',
            options: ['It removes the need for backpropagation', 'It computes the exact gradient over the whole dataset at every step, instead of a noisy estimate from one batch', 'Momentum (a running average of gradients) and a separate, automatically scaled step size for each parameter, plus a small pull of weights toward zero', 'It guarantees that the loss never goes up'],
            answer: 2,
            explain: 'Same idea, step downhill, with smarter step sizes. Nothing guarantees the loss never rises.',
          },
        ]}
      />

      <Remember
        items={[
          <>Text labels itself: <b>y is x shifted by one</b>. Every position is an example, and the causal mask lets one sequence of T tokens give T predictions in a single pass.</>,
          <>The loop: <b>batch → forward → loss → backward → step</b>. In code: <code>get_batch</code>, <code>model(x, y)</code>, <code>zero_grad</code>, <code>loss.backward()</code>, <code>opt.step()</code>.</>,
          <>Loss is <b>average surprise</b>. A healthy untrained model starts at <b>ln(vocabulary size)</b>: 4.17 for 65 characters. Check it every time: it caught a flaw in our own file.</>,
          <>Hold text back. <b>Validation loss</b> is the honest score. Train loss falling while validation loss rises is <b>overfitting</b>: memorising, not generalising.</>,
          <>The <b>learning rate</b> is the touchiest knob: too small crawls, too large diverges. AdamW is gradient descent with momentum and per-parameter step sizes.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Playground', sub: 'MLP, 8,000 chars, seconds' }, { label: 'tiny_gpt.py', sub: 'Transformer, 1M chars, minutes' }, { label: 'GPT-2', sub: '40 GB of web text, many GPUs' }, { label: 'Frontier LLM', sub: 'trillions of tokens, months' }]} active={1} />
        <ToyVsReal
          toy={<ul><li>One CPU, a 1 MB text file</li><li>300 to 3,000 steps, 2,048 tokens per step</li><li>Constant learning rate 3e-4</li><li>Overfitting is a real risk: the file is small</li><li>No checkpoints: a crash costs two minutes</li></ul>}
          real={<ul><li>Thousands of GPUs, trillions of tokens of filtered web text, code and books</li><li>Hundreds of thousands of steps, millions of tokens per step</li><li>Learning rate warms up from zero, then decays</li><li>Often about one epoch: most text is seen once, so classic overfitting is less of a worry than data quality</li><li>Checkpoints every few hours; runs are restarted from them after hardware failures and loss spikes</li></ul>}
        />
        <Callout kind="established">The loop is the same. Pretraining a frontier model is this lesson’s five steps, the same cross-entropy loss and the same AdamW family of optimizers. What changes is the engineering needed to run it on thousands of machines at once, and the care put into the data.</Callout>
        <Callout kind="model">“Loss goes down, so the model gets smarter” is a useful simplification. Lower next-token loss correlates well with better capabilities, but the link to any specific skill is not a simple formula, and it is an area of active study. You will meet this again in <a href="#/lesson/why-llms-know">Why LLMs know things</a>.</Callout>
        <p>What you trained here is called a <b>base model</b>: it continues text. Turning it into an assistant takes further training stages, covered in <a href="#/lesson/training-pipeline">From raw text to assistant</a>. Next, though: the weights are frozen, and we make the model talk.</p>
      </RealLLM>
    </Lesson>
  )
}
