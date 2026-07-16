# Module 03 — Neural Networks & Backpropagation

**Time: about 4 weeks. Do not rush this one. Code: `mlp_numpy.py`. This is the load-bearing module of the entire course — everything after it stands on what you build here.**

## The Wall You Already Hit

In module 02, exercise 2, you watched a linear model fail. The data was a curve, the model was a straight line, and the loss flattened at a floor it could never break. No amount of training helps when the model *cannot express* the answer.

Your first idea is probably: stack more layers! If one `W·x` isn't enough, do `W2·(W1·x)` — surely two transformations are smarter than one?

Here's the cruel little algebra fact: no. Multiply it out: `W2·(W1·x)` is just `(W2·W1)·x` — the two matrices collapse into one matrix, and one matrix is still a straight line. Stack a hundred linear layers and you've built an expensive way to compute a single linear layer. **Depth, by itself, does nothing.**

The fix is so small it's almost funny.

## The Fix: Bend It

After each linear layer, pass the output through one simple *non-linear* function. The standard choice is called **ReLU**, and here is the entire thing:

```
relu(x) = max(0, x)
```

If the number is negative, make it zero. If it's positive, leave it alone. That's a hinge — flat on one side, straight on the other, with a single bend at zero.

How can something that trivial matter? Watch:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 340" font-family="sans-serif">
  <rect width="760" height="340" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">Why activation functions: hinged pieces compose into any curve</text>

  <!-- panel 1: linear only -->
  <text x="150" y="60" text-anchor="middle" font-size="13" fill="#E8734A" font-weight="bold">linear layers only</text>
  <rect x="45" y="70" width="210" height="180" rx="8" fill="#fcfcfc" stroke="#ddd"/>
  <!-- target curve -->
  <path d="M 60 220 Q 110 100 150 170 Q 190 235 240 110" fill="none" stroke="#bbb" stroke-width="2" stroke-dasharray="4,3"/>
  <!-- straight line attempt -->
  <line x1="60" y1="200" x2="240" y2="140" stroke="#E8734A" stroke-width="3"/>
  <text x="150" y="270" text-anchor="middle" font-size="11" fill="#666">stack 100 linear layers →</text>
  <text x="150" y="285" text-anchor="middle" font-size="11" fill="#666">still ONE straight line</text>
  <text x="150" y="300" text-anchor="middle" font-size="11" fill="#888" font-family="monospace">W2·(W1·x) = (W2·W1)·x</text>

  <!-- panel 2: the hinge -->
  <text x="380" y="60" text-anchor="middle" font-size="13" fill="#4A90D9" font-weight="bold">ReLU: max(0, x) — a hinge</text>
  <rect x="285" y="70" width="190" height="180" rx="8" fill="#fcfcfc" stroke="#ddd"/>
  <line x1="300" y1="160" x2="460" y2="160" stroke="#eee" stroke-width="1"/>
  <line x1="380" y1="85" x2="380" y2="235" stroke="#eee" stroke-width="1"/>
  <path d="M 300 160 L 380 160 L 455 90" fill="none" stroke="#4A90D9" stroke-width="3"/>
  <text x="330" y="180" font-size="11" fill="#888">negative → 0</text>
  <text x="395" y="115" font-size="11" fill="#888">positive → passes</text>
  <text x="380" y="270" text-anchor="middle" font-size="11" fill="#666">one bend. Nearly nothing —</text>
  <text x="380" y="285" text-anchor="middle" font-size="11" fill="#666">but it breaks the collapse.</text>

  <!-- panel 3: composed hinges -->
  <text x="615" y="60" text-anchor="middle" font-size="13" fill="#55A868" font-weight="bold">thousands of hinges, stacked</text>
  <rect x="505" y="70" width="220" height="180" rx="8" fill="#fcfcfc" stroke="#ddd"/>
  <!-- target curve -->
  <path d="M 520 220 Q 570 100 610 170 Q 650 235 710 110" fill="none" stroke="#bbb" stroke-width="2" stroke-dasharray="4,3"/>
  <!-- piecewise approximation hugging it -->
  <path d="M 520 218 L 545 175 L 565 130 L 585 122 L 605 158 L 622 185 L 638 195 L 658 175 L 678 145 L 710 112"
        fill="none" stroke="#55A868" stroke-width="3"/>
  <text x="615" y="270" text-anchor="middle" font-size="11" fill="#666">short straight pieces approximate any curve;</text>
  <text x="615" y="285" text-anchor="middle" font-size="11" fill="#666">deeper layers bend already-bent space —</text>
  <text x="615" y="300" text-anchor="middle" font-size="11" fill="#666">complexity compounds multiplicatively</text>

  <text x="380" y="328" text-anchor="middle" font-size="12" fill="#888">This one-line function, max(0, x), is the entire difference between linear regression and deep learning.</text>
</svg>

*One hinge is nothing. Thousands of hinges, layered, can trace any shape.*

One hinge draws one bend. But a layer has hundreds of hinges (one per output number), and you stack layers, and each layer gets to bend a space that previous layers *already bent*. Complexity compounds. It's the same reason a chain of short straight segments can draw any curve you like: enough pieces, placed by learning, approximate anything.

So here's what one complete layer of a neural network actually is:

```
output = relu(W @ x + b)
```

Read it with your module 00 eyes: `W @ x` runs a batch of shopping-bill formulas over the input (each row of W is one formula). `+ b` shifts each result up or down (b is the "bias" — a per-formula base offset, like a flat fee on the bill). `relu(...)` bends. Multiply, shift, bend. A **neural network** — specifically the kind called an **MLP**, multi-layer perceptron — is just this, repeated:

```
x → [multiply, shift, bend] → [multiply, shift, bend] → [multiply, shift] → output
```

That's the whole architecture for this module, and — here's a spoiler worth knowing early — these exact MLPs sit *inside* every transformer, unchanged. When you build GPT in module 08, this module's network gets embedded in it as a component, like a class you wrote earlier being imported.

**Why does depth help, in plain words?** Because layers build features out of features. In image networks, layer 1 learns to spot edges, layer 2 combines edges into corners and curves, layer 3 combines those into eyes and wheels, layer 4 into faces and cars. Each layer speaks in the vocabulary the previous layer created. In language models the same laddering happens: characters → word-pieces → grammar → meaning → "what plausibly comes next." Nobody designs the ladder. It emerges, rung by rung, because that's the efficient way to minimize the loss.

## Making the Network Choose: Softmax and Cross-Entropy

The network in this module will classify points into 3 categories. Why do we care about classification in a language course? Because — think about it — predicting the next word IS classification: given the context, pick one of 50,000 classes. Get comfortable choosing between 3 classes here, and module 06 just scales up the class count.

Two new pieces make choosing work.

**Softmax: from scores to probabilities.** The network's final layer outputs raw scores, one per class — maybe `[2.0, 1.0, -1.0]`. (Raw scores have a jargon name you'll see everywhere: **logits**.) We want probabilities: positive numbers that sum to 1. Softmax does the conversion in two moves: raise e to each score (this makes everything positive, and stretches gaps — bigger scores pull ahead disproportionately), then divide each by the total (now they sum to 1):

```
scores:         [ 2.0,   1.0,  -1.0]
e^score:        [ 7.39,  2.72,  0.37]     total: 10.48
divide by total:[ 0.71,  0.26,  0.03]     ← probabilities, sum = 1
```

The name means "soft maximum": a hard max would say "class 0 wins, 100%"; softmax says "class 0 is probably it, but I'm keeping some doubt." Keeping doubt matters, because doubt is where learning signals come from.

**Cross-entropy: scoring the choice.** Now the model has said "71% it's class 0." Suppose the true answer *is* class 0. How wrong was the model? Cross-entropy's answer:

```
loss = −log(probability the model gave to the CORRECT class)
```

Feel this formula with numbers rather than reading it:

- Model said 0.99 for the right class → loss = −log(0.99) ≈ **0.01**. Barely surprised. Nearly perfect.
- Model said 0.71 → loss ≈ **0.34**. Mildly surprised.
- Model said 0.03 for what turned out to be the right answer → loss ≈ **3.5**. Shocked.

So cross-entropy is literally a **surprise meter** — module 02's preview, now real. Training a language model means showing it real text and minimizing its average surprise at each next word. When you read that "GPT-3 reached a loss of 2.0," that number is this exact quantity: average surprise per token. Whole research careers are spent lowering it.

And now, a genuine gift from the mathematics. When you pair softmax with cross-entropy and work out the gradient at the scores (the derivation is in any textbook; you don't need it), everything collapses to one absurdly clean line:

```
gradient at the logits = predicted_probabilities − correct_answer_as_one_hot
```

Concretely: model predicted `[0.71, 0.26, 0.03]`, truth was class 0 i.e. `[1, 0, 0]`, so the gradient is `[−0.29, 0.26, 0.03]`. Look at what it says: push class 0's score *up* by an amount matching how short it fell, push the others *down* by exactly their undeserved shares. **The gradient is just the miss.** You'll see this as a single line in the code, and it's the reason literally everyone uses this loss for classification.

## Backpropagation: The Blame Must Flow Backwards

Now the main event. Here is the problem in one sentence: the loss is measured at the *end* of the network, but the parameters that need correcting are spread through *every* layer — so how does a weight sitting in layer 1, three transformations away from the output, find out its share of the blame?

Picture a factory assembly line. Station 1 shapes the part, station 2 paints it, station 3 packages it, and at final inspection the product is flawed. Whose fault? You walk the line *backwards*. The inspector tells station 3 how the flaw relates to packaging. Station 3 — who knows exactly what it did to what it received — passes back to station 2: "given what I do, here's how the flaw traces to *your* output." Station 2 does the same for station 1. Each station needs only two things: knowledge of its own operation, and the blame message handed back from downstream. No station needs to understand the whole factory.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 400" font-family="sans-serif">
  <rect width="760" height="400" fill="#ffffff"/>
  <text x="380" y="28" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">Backprop: forward pass builds the product, backward pass assigns the blame</text>

  <defs>
    <marker id="fwd" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#4A90D9"/>
    </marker>
    <marker id="bwd" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#E8734A"/>
    </marker>
  </defs>

  <!-- forward line -->
  <text x="55" y="120" font-size="13" fill="#4A90D9" font-weight="bold">FORWARD</text>
  <rect x="40" y="140" width="80" height="52" rx="8" fill="#eef4fb" stroke="#4A90D9"/>
  <text x="80" y="171" text-anchor="middle" font-size="13" fill="#333">input x</text>

  <rect x="180" y="140" width="120" height="52" rx="8" fill="#eef4fb" stroke="#4A90D9"/>
  <text x="240" y="162" text-anchor="middle" font-size="13" fill="#333">Layer 1</text>
  <text x="240" y="180" text-anchor="middle" font-size="11" fill="#666" font-family="monospace">relu(x@W1+b1)</text>

  <rect x="360" y="140" width="120" height="52" rx="8" fill="#eef4fb" stroke="#4A90D9"/>
  <text x="420" y="162" text-anchor="middle" font-size="13" fill="#333">Layer 2</text>
  <text x="420" y="180" text-anchor="middle" font-size="11" fill="#666" font-family="monospace">h@W2+b2</text>

  <rect x="540" y="140" width="120" height="52" rx="8" fill="#fdf3ee" stroke="#E8734A"/>
  <text x="600" y="162" text-anchor="middle" font-size="13" fill="#333">Loss</text>
  <text x="600" y="180" text-anchor="middle" font-size="11" fill="#666">surprise = 2.31</text>

  <line x1="120" y1="166" x2="172" y2="166" stroke="#4A90D9" stroke-width="2.5" marker-end="url(#fwd)"/>
  <line x1="300" y1="166" x2="352" y2="166" stroke="#4A90D9" stroke-width="2.5" marker-end="url(#fwd)"/>
  <line x1="480" y1="166" x2="532" y2="166" stroke="#4A90D9" stroke-width="2.5" marker-end="url(#fwd)"/>

  <!-- backward line -->
  <text x="55" y="265" font-size="13" fill="#E8734A" font-weight="bold">BACKWARD</text>
  <line x1="540" y1="240" x2="488" y2="240" stroke="#E8734A" stroke-width="2.5" marker-end="url(#bwd)"/>
  <line x1="360" y1="240" x2="308" y2="240" stroke="#E8734A" stroke-width="2.5" marker-end="url(#bwd)"/>
  <line x1="180" y1="240" x2="128" y2="240" stroke="#E8734A" stroke-width="2.5" marker-end="url(#bwd)"/>

  <rect x="360" y="214" width="120" height="52" rx="8" fill="#fdf3ee" stroke="#E8734A"/>
  <text x="420" y="236" text-anchor="middle" font-size="12" fill="#333">Layer 2's blame</text>
  <text x="420" y="254" text-anchor="middle" font-size="11" fill="#666" font-family="monospace">d = d @ W2.T</text>

  <rect x="180" y="214" width="120" height="52" rx="8" fill="#fdf3ee" stroke="#E8734A"/>
  <text x="240" y="236" text-anchor="middle" font-size="12" fill="#333">Layer 1's blame</text>
  <text x="240" y="254" text-anchor="middle" font-size="11" fill="#666" font-family="monospace">d = d * (x&gt;0)</text>

  <rect x="540" y="214" width="120" height="52" rx="8" fill="#fdf3ee" stroke="#E8734A"/>
  <text x="600" y="236" text-anchor="middle" font-size="12" fill="#333">start: the miss</text>
  <text x="600" y="254" text-anchor="middle" font-size="11" fill="#666" font-family="monospace">probs − one_hot</text>

  <!-- gradients dropping to weights -->
  <line x1="240" y1="266" x2="240" y2="310" stroke="#55A868" stroke-width="2" marker-end="url(#fwd)" stroke-dasharray="4,3"/>
  <line x1="420" y1="266" x2="420" y2="310" stroke="#55A868" stroke-width="2" marker-end="url(#fwd)" stroke-dasharray="4,3"/>
  <rect x="165" y="312" width="150" height="40" rx="8" fill="#eefaf1" stroke="#55A868"/>
  <text x="240" y="337" text-anchor="middle" font-size="11" fill="#333" font-family="monospace">d_W1 = x.T @ d</text>
  <rect x="345" y="312" width="150" height="40" rx="8" fill="#eefaf1" stroke="#55A868"/>
  <text x="420" y="337" text-anchor="middle" font-size="11" fill="#333" font-family="monospace">d_W2 = h.T @ d</text>
  <text x="600" y="337" font-size="12" fill="#55A868">→ then: W -= lr * d_W</text>

  <text x="380" y="385" text-anchor="middle" font-size="12" fill="#888">Each station needs only LOCAL knowledge + the blame message handed back from downstream (chain rule: amplifications multiply)</text>
</svg>

*The assembly line, walked in reverse: blame flows backward, and each layer's weights collect their share.*

Mathematically, "pass the blame back through your own operation" is module 00's chain rule — exchange rates multiply. The network is a pipeline of stages; each stage knows its own amplification factor; blame entering from downstream gets multiplied by the local factor and passed upstream. Nothing you haven't already done with currency conversions. The only difference: here the stages are matrix operations, so the "rates" are matrices too.

Now, why is this such a big deal that it has a famous name? **Speed.** Remember module 02, Stage B — the nudge method? It works, but it needs one full loss evaluation *per parameter*. For a billion parameters: a billion forward passes *per training step*. Completely hopeless. Backprop computes *every* parameter's gradient — all billion — in a single backward sweep that costs roughly the same as one forward pass. One pass forward, one pass back, all gradients exact. That efficiency is the entire reason deep learning is computationally possible.

And one clarification that removes a common confusion: backprop is **not** a learning algorithm. Learning is still module 02's `w -= lr * gradient`, unchanged. Backprop is just the fast, exact way to *get* the gradients. Measurement, not learning.

### The Three Rules (this table is all of backprop)

For the network in this module, the complete set of backward rules is three lines. In each, `d_out` is the blame arriving from downstream, and the rule says what to pass on:

| Forward operation | Backward rule | In plain words |
|---|---|---|
| `out = X @ W + b` | `d_W = X.T @ d_out` · `d_X = d_out @ W.T` · `d_b = sum(d_out)` | blame travels back through the same wiring, transposed (module 00 §6 said this day would come — treat as recipe, the gradient check below *proves* it) |
| `out = relu(x)` | `d_x = d_out * (x > 0)` | blame passes through where the hinge was open (positive), and dies where it was shut — a closed hinge contributed nothing, so it's blameless |
| softmax + cross-entropy | `d_logits = probs − one_hot` | the gradient is the miss (you met this above) |

That's it. That table, plus the chain rule to glue stations together, is the complete algorithm that trains every neural network on Earth. PyTorch's celebrated "autograd" engine is this table, extended to more operations and automated. In module 08, when you finally call `loss.backward()` and let the framework do it, you will know *exactly* what labor you're delegating — because you'll have done it by hand here.

## The Code

`mlp_numpy.py` — about 200 lines, more than half comments — builds a complete classifier on the **spiral dataset**: three spiral arms of points, one per class, chosen because it is *provably impossible* for a straight-line model. The file runs four acts:

1. **Generate the spirals.**
2. **Train a linear model first.** Watch it stall around 45% accuracy — module 02 exercise 2's wall, rebuilt on purpose so you can see the exact thing the hinge fixes.
3. **Train the MLP** (2 hidden layers of 64 hinges each). Same training loop, same data — watch it sail to ~99%. The nonlinearity earning its keep, measured.
4. **The gradient check.** The payoff of module 02's Stage B: we compare backprop's fast gradients against the slow honest nudge method, parameter by parameter, and they agree to about twelve decimal places. This is the step that converts "I believe backprop" into "I verified backprop." Never skip a gradient check when hand-implementing — it's the unit test of deep learning, and in exercise 3 you'll watch it catch a real bug.

## Modify-It Exercises (calibrated to sting a little)

1. **Remove the ReLUs** — replace the bends with pass-throughs — and retrain. Accuracy collapses to the linear model's. You have now *demonstrated*, not just read, that depth without bending is fake depth.
2. **Vary the width.** Try hidden layers of size 2, 8, 64, 512. Find the smallest width that still solves the spiral. When width 2 fails, think about why: two hinges per layer physically cannot route three spirals' worth of information. It's a bandwidth problem, and you can feel it.
3. **Plant a bug and catch it.** Change `X.T @ d_out` to `X @ d_out` somewhere (the classic from-scratch typo), and run the gradient check. Watch it scream. This is precisely the bug class the check exists for, and now you know its alarm sound.
4. **Go deeper.** Add a third hidden layer, then a fourth, then more. Training gets slower to converge and then flaky. You are personally meeting the deep-network instability that took researchers years to tame — and the two inventions that tamed it (residual connections and layer normalization) are waiting for you in module 08. File the frustration away; it's about to become motivation.
5. *(Stretch)* Swap ReLU for `tanh` and work out its backward rule yourself (it's `d_x = d_out * (1 − tanh(x)²)` — derive or verify by nudging). One new row in the table, earned with your own hands.

## Best External Resources

- **Karpathy's micrograd video, now in full.** You built the vectorized version; he builds a scalar version with an elegant little autograd engine. Watching *after* building is the right order — his code will read like a clever refactoring of the table you just implemented.
- 3Blue1Brown, *"What is backpropagation really doing?"* (chapters 3–4) — best watched after your gradient check passes, when the animations describe something you've done.
- CS231n course notes, "Backpropagation, Intuitions" — the written treatment closest in spirit to this module.

Next: Phase 2 begins. You now hold every mechanical piece a language model is made of. **04 — Tokenization** starts feeding actual language into the machine.
