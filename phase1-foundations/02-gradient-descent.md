# Module 02 — Gradient Descent: The Only Learning Algorithm

**Time: about 2 weeks. Code: `gradient_descent.py`.**

## Where We Left Off

Module 01 ended with a promise: the model guesses, gets it wrong, and we "nudge the numbers in the direction that makes it less wrong." Billions of times. This module makes that sentence completely concrete — because it has to be. A GPT-class model has *billions* of adjustable numbers (each one is called a **parameter**). No human can tune them. We need a procedure that automatically decides, for every single parameter, two things: *which direction should it move, and by how much?*

That procedure is gradient descent. And here is a fact that should change how you look at the whole field: with minor variations, **gradient descent is the only learning algorithm in all of deep learning.** Every model you've heard of — GPT, Claude, image generators, all of them — learned by this one procedure. Learn it once, properly, and you've learned how everything trains.

## First, We Need a Score: The Loss

Before you can improve something, you need to measure how bad it is. So we define a single number that measures how wrong the model currently is, called the **loss**. Wrong predictions → big loss. Good predictions → small loss. Training means: make the loss go down.

The simplest useful loss, and the one this module's code uses, is **mean squared error**: take each prediction, subtract the true answer, square it (squaring makes errors positive, and punishes big misses much harder than small ones), and average over your data. Predicting house prices and you're off by $10k on average? That's your loss. Get better, loss shrinks.

Now here's the key mental move. The loss depends on the model's parameters — change a parameter, the predictions change, the loss changes. So imagine the loss as a **landscape**: every possible setting of the parameters is a location, and the loss at that setting is the *altitude*. Bad settings are mountains. Good settings are valleys. Training is: find a low point.

## The Algorithm: Hiking Downhill in Fog

You're standing somewhere on this landscape, and it's completely foggy — you cannot see where the valley is. (This is not a decorative detail. With billions of parameters, the landscape has billions of dimensions; nobody can survey it. The fog is the truth.)

But you can do one thing: **feel the slope under your own feet.** Which way does the ground tilt, right here, right now?

So you do the obvious thing. Feel the slope. Take a small step downhill. Feel again. Step again. Repeat until the ground is flat.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 400" font-family="sans-serif">
  <rect width="720" height="400" fill="#ffffff"/>
  <text x="360" y="28" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">Gradient descent: feel the slope, step downhill, repeat</text>

  <!-- loss curve (a valley) -->
  <path d="M 60 80 Q 180 340 340 330 Q 500 322 660 100" fill="none" stroke="#4A90D9" stroke-width="3"/>
  <text x="75" y="65" font-size="13" fill="#4A90D9">loss (how wrong the model is)</text>
  <line x1="60" y1="370" x2="690" y2="370" stroke="#bbb" stroke-width="1.5"/>
  <text x="560" y="392" font-size="12" fill="#999">parameter value w</text>

  <defs>
    <marker id="arr2" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#E8734A"/>
    </marker>
  </defs>

  <!-- steps: big where steep, small where flat -->
  <circle cx="120" cy="180" r="8" fill="#E8734A"/>
  <text x="95" y="160" font-size="12" fill="#E8734A" font-weight="bold">start (random w)</text>
  <line x1="132" y1="192" x2="185" y2="268" stroke="#E8734A" stroke-width="2.5" marker-end="url(#arr2)"/>
  <circle cx="196" cy="281" r="7" fill="#E8734A"/>
  <line x1="207" y1="292" x2="248" y2="316" stroke="#E8734A" stroke-width="2.5" marker-end="url(#arr2)"/>
  <circle cx="258" cy="321" r="6" fill="#E8734A"/>
  <line x1="268" y1="325" x2="298" y2="330" stroke="#E8734A" stroke-width="2.5" marker-end="url(#arr2)"/>
  <circle cx="308" cy="331" r="5" fill="#E8734A"/>
  <line x1="317" y1="331" x2="333" y2="331" stroke="#E8734A" stroke-width="2.5" marker-end="url(#arr2)"/>
  <circle cx="345" cy="330" r="4" fill="#E8734A"/>
  <text x="345" y="358" text-anchor="middle" font-size="12" fill="#55A868" font-weight="bold">minimum: slope ≈ 0, steps ≈ 0</text>

  <text x="150" y="240" font-size="12" fill="#666">steep slope →</text>
  <text x="150" y="255" font-size="12" fill="#666">big correction</text>
  <text x="380" y="300" font-size="12" fill="#666">gentle slope → small correction</text>

  <!-- update rule box -->
  <rect x="440" y="150" width="230" height="66" rx="8" fill="#f6f8fa" stroke="#ddd"/>
  <text x="555" y="178" text-anchor="middle" font-size="14" fill="#333" font-family="monospace">w -= lr * dL/dw</text>
  <text x="555" y="200" text-anchor="middle" font-size="11" fill="#888">the one line that is all of deep learning</text>
</svg>

*Gradient descent: stepping down the loss landscape*

In code, the whole algorithm is:

```
until satisfied:
    slope = how the loss changes when each parameter changes    # "the gradient"
    parameters -= learning_rate * slope                         # small step downhill
```

That's it. Genuinely. Everything else in deep learning is either (a) making the slope-measurement fast (module 03), or (b) designing better landscapes to descend (every architecture ever invented).

## The Math You Already Have

What is "the slope under your feet" for one parameter, precisely? It's module 00's nudge experiment: *if I increase this parameter by a hair, does the loss go up or down, and how fast?* That's a derivative — `dL/dw`, the loss's sensitivity to parameter w. You measured these with the shower-faucet experiment in module 00; nothing is new here.

The **gradient** — the word that makes this all sound advanced — is nothing more than the full array of these readings, one per parameter. When a paper writes ∇L, read it as:

```
grads = [sensitivity_of_loss_to(p) for p in parameters]
```

Now look at the update rule again, because it has a quiet elegance:

```
w -= learning_rate * dL/dw
```

A parameter that strongly affects the loss has a big `dL/dw`, so it receives a big correction. A parameter that barely matters gets a tiny one. **Blame is automatically distributed in proportion to influence.** Nobody has to decide which knobs matter — the derivatives already know.

## The Learning Rate: A Trade-off You'll Recognize

That `learning_rate` (usually `lr` in code) is just the step size, and it's a classic engineering trade-off with familiar failure modes:

- **Too small**: you inch down the mountain. Training works but takes geologic time. (Like polling a queue every 10 seconds when messages arrive every millisecond.)
- **Too large**: you *leap* — right over the valley and up the other side, then leap back, higher each time. The loss doesn't just fail to improve, it explodes. You'll cause this deliberately in exercise 1, and once you've seen the loss print `2.1, 4.7, 93.5, 8812.0, NaN` you will recognize it instantly for the rest of your career, because everyone who trains models meets this signature eventually.

There's no formula for the right learning rate. Practitioners find it the way you'd expect: try one, watch the loss curve, adjust. You're about to do exactly that.

## Two Honest Footnotes

**"Can't you get stuck in a small dip that isn't the deepest valley?"** (A *local minimum*, in the jargon.) In the 2-D picture, yes, obviously. The surprising and fortunate truth: in billions of dimensions this matters far less than intuition suggests, because with that many directions available, there's almost always *some* direction that still leads downhill. Genuine dead-ends are rare in high dimensions. Don't lose sleep over it.

**"Do you measure the slope on the entire dataset every step?"** You could, but it's wasteful — like taste-testing the whole pot to check if the soup needs salt. Instead you estimate the slope from a random spoonful: a **batch** of maybe 32 examples. The estimate is noisy, but you're taking thousands of steps, so the noise averages out — and it even helps, occasionally jostling you out of small dips. Gradient descent with random batches is called **stochastic gradient descent (SGD)**, which sounds impressive and means "downhill hiking with spot checks."

## The Code

`gradient_descent.py` walks through three stages. Read and run them in order — each is fully commented:

**Stage A — one parameter, slope by formula.** We secretly generate data following `y = 3x` (plus noise), then hand the model `y = w·x` with `w` starting at zero, and let gradient descent discover the 3.0 on its own. The slope `dL/dw` is computed from a small formula you can verify with school calculus. You watch `w` walk from 0.0 to 2.99 in thirty steps. Your first learned parameter.

**Stage B — slope by nudging, no calculus at all.** Same problem, but now we throw away the formula and measure the slope exactly the way module 00 taught: nudge `w`, recompute the loss, divide. It finds the same answer. This matters for two reasons. First, it proves the formula was never magic — just a shortcut for the experiment. Second, this nudge-and-measure method works for *any* model with zero math, which makes it the perfect *test oracle*: in module 03, when you write the fast slope calculation by hand, you'll verify it against this slow honest one. Testing an optimized implementation against a brute-force reference — an instinct you already have.

**Stage C — the real thing.** Two parameters (`y = w·x + b`), random batches, hundreds of steps, and a loss curve printed as it falls. Pay attention to the *shape* of this code: forward pass → loss → gradients → update, looped over batches. That structure is, quite literally, the same loop that trained GPT-4. Only the middle part — the model between input and loss — gets fancier from here. The loop never changes again.

Run it:

```bash
python gradient_descent.py
```

## A Preview You Need: Losses for Predicting Words

Stages A–C predict a *quantity* (a number on a line), so mean squared error fits. But a language model predicts a *choice* — which of ~50,000 words comes next. Choices need a different loss, called **cross-entropy**, and you'll build it properly in module 03. Here's the one-sentence preview so the name doesn't ambush you: cross-entropy measures *how surprised the model was by the correct answer* — confident and right, tiny loss; confident and wrong, huge loss. Training a language model = minimizing its average surprise at real text. Keep that phrase; it does a lot of work later.

## Modify-It Exercises

1. In Stage C, set `lr = 1.5` and run. Watch the loss diverge into garbage. Then binary-search by hand for the largest learning rate that still converges. Ten minutes, and you'll never forget what a bad learning rate looks like.
2. Change Stage C's data to `y = 3x² + 2` but leave the model as `y = w·x + b`. Run it. The loss falls, then flattens at a floor it can never break through, no matter how long you train. Stare at that floor. The model is a straight line and the truth is a curve — *the model class cannot represent the answer*, and no amount of gradient descent fixes that. This exact frustration is what module 03 exists to solve.
3. In Stage B, shrink the nudge size `h` to `1e-12`. The slope measurements go wrong. Why? (You already know: floating point. Subtracting two nearly-equal doubles shreds precision — familiar territory. The sweet spot is around `1e-5`.)
4. Add a third parameter and fit `y = w1·x1 + w2·x2 + b`. Notice you needed no new ideas — copy the pattern, add a line. Now believe the same pattern runs at a billion parameters, because it does.

## Best External Resources

- 3Blue1Brown, *"Gradient descent, how neural networks learn"* (Deep Learning chapter 2) — watch after running Stage C; every image maps onto code you just ran.
- Karpathy, *"The spelled-out intro to neural networks and backpropagation: building micrograd"* — watch **only the first 40 minutes** for now (derivatives and nudging). The rest is module 03's material and will be much easier after you've built module 03 yourself.

Next: **03 — Neural Networks & Backprop.** Exercise 2 left you with a model that can only draw straight lines. Time to fix that.
