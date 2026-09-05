# Module 00 — The Math You Actually Need (All of It)

> **⏱️ Time:** ~1-2 weeks.  
> **What you'll build:** Core mechanics and conceptual understanding.

---

## 🎯 TL;DR
1. **The Core Problem:** The challenge this module solves.
2. **The Mechanism:** How we solve it using first principles.
3. **The Payoff:** What you can do with this new capability.

---


**Time: 1–2 weeks. Code: `math_primer.py`. Read this first, run the code alongside it, and keep both open for the whole course.**

Before we start, let me tell you something that should take a weight off your shoulders.

The entire journey from zero to GPT uses only **eight mathematical ideas**. Not eight chapters. Eight ideas. And every one of them is something you can hold in your head and check with ten lines of Python.

Here they are:

1. A vector — a list of numbers
2. The dot product — multiply two lists pairwise, add it all up
3. A matrix — a grid of numbers
4. Matrix × vector — doing many dot products at once
5. Matrix × matrix — doing even more dot products at once
6. The transpose — flipping a grid's rows and columns
7. The derivative — "if I nudge this input, how much does the output move?"
8. The chain rule — how nudges travel through a pipeline of steps

That's the complete list. When a research paper shows you half a page of dense equations, it is these eight things, stacked and abbreviated. The equations feel hard not because the ideas are hard, but because the notation compresses them brutally and nobody hands you the decompression table. This module is that table.

One more reassurance: you will not be asked to *trust* any of this. Every calculation on this page is repeated in `math_primer.py` with an `assert` next to it. If I claim something equals 14, the code checks it equals 14. You get to verify the math the same way you verify code — by running it.

## 1. Vectors: Just a List of Numbers

Forget everything intimidating you've heard about vectors. A vector is a list of numbers. That's it.

```
v = [2, 1, 3]
```

This is a 3-dimensional vector. "Dimension" is just a fancy word for "how many numbers are in the list." A 768-dimensional vector is a list with 768 numbers in it. Nothing more.

Now, *why* would we want lists of numbers? Because a list of numbers can describe a thing. Think of a row in a database:

```
person = [34, 95000, 172]     # [age, salary_usd, height_cm]
```

You've been working with vectors your whole career — you just called them records, rows, or structs. In this course, one vector will describe one word (or one token). That's the whole trick behind module 01.

There are only two operations you can do to vectors, and both are exactly what you'd guess:

```
[2, 1, 3] + [1, 0, 2]  =  [3, 1, 5]      # add: slot by slot
2 × [2, 1, 3]          =  [4, 2, 6]      # scale: double every number
```

There's one measurement worth knowing: the **length** of a vector. For a 2-number vector like `[3, 4]`, imagine walking 3 steps east and 4 steps north — how far are you from where you started? Pythagoras from school: `√(3² + 4²) = √25 = 5`. That same formula keeps working no matter how many numbers are in the list. You'll rarely compute it by hand, but the word "norm" in papers means exactly this: how long is the arrow, how big are the numbers overall.

## 2. The Dot Product: One Idea, Two Superpowers

Take two lists of the same size. Multiply them slot by slot. Add up the results. One number comes out. That's the dot product — the single most-used operation in all of AI.

```
a = [2, 1, 3]
b = [4, 0, 2]

slot 1:  2 × 4 = 8
slot 2:  1 × 0 = 0
slot 3:  3 × 2 = 6
add up:  8 + 0 + 6 = 14

a · b = 14
```

Mechanically, that's everything. A five-year loop of multiply-and-add. Now here's why this humble operation runs the world. It has two readings.

**Reading 1: a bill.** Suppose your shopping basket is `[2, 1, 3]` — two loaves of bread, one milk, three eggs. And the price list is `[4, 0, 2]` — bread costs 4, milk is free today, eggs cost 2 each. What's your total?

```
2 loaves × 4  +  1 milk × 0  +  3 eggs × 2  =  14
```

You just did a dot product. Quantities dotted with prices gives the total. In machine learning this shows up as: *features* dotted with *weights* gives a *score*. Every single "score" a neural network ever computes — every prediction, every neuron — is a shopping bill like this one. Hold onto that image.

**Reading 2: a similarity meter.** Now suppose both lists describe the *same kind* of thing. Say two people rate how much they like five movie genres, from −1 (hate) to +1 (love):

```
you    = [ 0.9,  0.8, -0.5]     # [action, comedy, romance]
friend = [ 0.8,  0.9, -0.4]
stranger = [-0.6, -0.8,  0.9]
```

Dot you with your friend: `0.9×0.8 + 0.8×0.9 + (−0.5)×(−0.4) = 0.72 + 0.72 + 0.20 = 1.64`. A big positive number. Why? Because wherever you have a big positive number, your friend does too — the products stack up. Even your shared dislikes multiply into a positive (negative × negative).

Now dot you with the stranger: `0.9×(−0.6) + 0.8×(−0.8) + (−0.5)×0.9 = −1.63`. Big and negative — everywhere you're positive, they're negative. You disagree slot by slot, and the dot product reports it.

So: **when both vectors describe the same kind of thing, the dot product measures agreement.** Big positive = similar. Near zero = unrelated. Negative = opposites.

This is how a language model will decide that "cat" and "dog" are similar (module 01), how attention will decide which words matter to which (module 07), and how a search engine will find the document that matches your question (module 11). Three of the biggest ideas in this course are this one shopping bill, reused.

One small cousin worth naming: **cosine similarity**. It's the dot product after you've divided out both vectors' lengths, so only the *pattern* of agreement matters, not how loudly it's expressed. It always lands between −1 (perfect opposites) and +1 (perfectly aligned). When you see "cosine similarity" anywhere, read: "the agreement meter, normalized."

## 3. Matrices: A Spreadsheet of Numbers

A matrix is a grid of numbers. If a vector is one row in a spreadsheet, a matrix is the whole sheet.

```
A = [[1, 2, 3],
     [4, 5, 6]]
```

This matrix has 2 rows and 3 columns, so we say its **shape** is `(2, 3)`. Get in the habit of always naming shapes as (rows, columns) — always rows first. I'll nag you about this because shape-tracking is the skill that keeps you from getting lost later. It is exactly the same skill as type-checking: a `(2, 3)` matrix is a *type*, and operations either accept that type or they don't.

There are two ways to *read* a matrix, and you'll switch between them constantly:

**As a stack of records.** Each row is one vector. Three tokens, each described by 4 numbers? That's a `(3, 4)` matrix — one row per token. This is how *data* is usually shaped.

**As a machine.** Feed a vector in one side, a transformed vector comes out the other. This is how *weights* are shaped — a weight matrix in a neural network is a machine that eats one kind of vector and produces another. The next section shows exactly how the machine works, and you'll see there's no magic in it.

(If you've watched 3Blue1Brown's linear algebra series, his "a matrix transforms space" animations are the second reading. Both readings are true. Engineers spend most of their time in the first one.)

## 4. Matrix × Vector: Many Bills at Once

Here is the operation that fills every deep learning paper. Let's take all the mystery out of it right now.

To compute `W @ x` — a matrix times a vector — you do exactly one thing: **take the dot product of each row of W with x.** Each row produces one output number. That's the entire rule.

Let's do it with real numbers, slowly. Say a house is described by three numbers:

```
x = [2, 1, 1]        # [bedrooms(×1000s... just kidding — bedrooms, bathrooms, garages]
```

And say we have two different "scorers," each one a row of weights — like two different formulas that read the same house:

```
W = [[1, 0, 2],      # row 1: some price-ish formula
     [0, 3, 1]]      # row 2: some upkeep-cost formula
```

Now `W @ x` just runs both formulas:

```
row 1 · x  =  1×2 + 0×1 + 2×1  =  4      (the price score)
row 2 · x  =  0×2 + 3×1 + 1×1  =  4      (the upkeep score)

W @ x = [4, 4]
```

Two shopping bills, computed from the same basket, using two different price lists. That's all a matrix-vector multiply is. A neural network layer with 64 outputs? That's a matrix with 64 rows — 64 formulas reading the same input at once. When module 03 says "a layer mixes the input's features," this is the mechanism, and now you've done it by hand.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 420" font-family="sans-serif">
  <rect width="760" height="420" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">Matrix × vector = a batch of dot products (one per row)</text>

  <!-- W matrix -->
  <text x="120" y="70" font-size="13" fill="#666" text-anchor="middle">W  (2 rows, 3 cols)</text>
  <rect x="55" y="85" width="130" height="45" rx="6" fill="#eef4fb" stroke="#4A90D9" stroke-width="2"/>
  <text x="120" y="113" text-anchor="middle" font-size="15" fill="#333" font-family="monospace">1  0  2</text>
  <rect x="55" y="135" width="130" height="45" rx="6" fill="#eefaf1" stroke="#55A868" stroke-width="2"/>
  <text x="120" y="163" text-anchor="middle" font-size="15" fill="#333" font-family="monospace">0  3  1</text>

  <text x="215" y="140" font-size="20" fill="#666">@</text>

  <!-- x vector -->
  <text x="285" y="70" font-size="13" fill="#666" text-anchor="middle">x  (3 numbers)</text>
  <rect x="255" y="85" width="60" height="95" rx="6" fill="#fdf3ee" stroke="#E8734A" stroke-width="2"/>
  <text x="285" y="112" text-anchor="middle" font-size="15" fill="#333" font-family="monospace">2</text>
  <text x="285" y="137" text-anchor="middle" font-size="15" fill="#333" font-family="monospace">1</text>
  <text x="285" y="162" text-anchor="middle" font-size="15" fill="#333" font-family="monospace">1</text>

  <text x="345" y="140" font-size="20" fill="#666">=</text>

  <!-- output -->
  <text x="420" y="70" font-size="13" fill="#666" text-anchor="middle">out  (2 numbers)</text>
  <rect x="390" y="85" width="60" height="45" rx="6" fill="#eef4fb" stroke="#4A90D9" stroke-width="2"/>
  <text x="420" y="113" text-anchor="middle" font-size="15" fill="#333" font-family="monospace">4</text>
  <rect x="390" y="135" width="60" height="45" rx="6" fill="#eefaf1" stroke="#55A868" stroke-width="2"/>
  <text x="420" y="163" text-anchor="middle" font-size="15" fill="#333" font-family="monospace">4</text>

  <!-- the two dot products spelled out -->
  <text x="500" y="108" font-size="13" fill="#4A90D9" font-family="monospace">row1·x = 1×2+0×1+2×1 = 4</text>
  <text x="500" y="163" font-size="13" fill="#55A868" font-family="monospace">row2·x = 0×2+3×1+1×1 = 4</text>

  <!-- shape rule -->
  <rect x="60" y="230" width="640" height="70" rx="10" fill="#f6f8fa" stroke="#ddd"/>
  <text x="380" y="258" text-anchor="middle" font-size="15" fill="#333" font-family="monospace">(2, 3) @ (3,) → (2,)      general rule: (m, n) @ (n, p) → (m, p)</text>
  <text x="380" y="284" text-anchor="middle" font-size="12" fill="#888">inner numbers must MATCH (or NumPy throws — it's a type error); outer numbers are the output shape</text>

  <!-- reading -->
  <text x="380" y="340" text-anchor="middle" font-size="13" fill="#333">Reading: each row of W is one output feature's <tspan font-weight="bold">weighted opinion poll</tspan> over all inputs.</text>
  <text x="380" y="362" text-anchor="middle" font-size="13" fill="#333">A layer with 64 outputs = 64 rows = 64 polls, computed in ONE multiply.</text>
  <text x="380" y="392" text-anchor="middle" font-size="12" fill="#888">Stack many x's as rows of a matrix X, and X @ W transforms every token at once — the vectorized for-loop GPUs were built for.</text>
</svg>

*Matrix times vector: a batch of dot products, plus the shape rule*

Now the shape bookkeeping, in plain words. Our W was `(2, 3)` — 2 rows, 3 columns. Our x had 3 numbers. The output had 2 numbers. Notice:

- The 3 in W's columns had to match the 3 in x. Why? Because each row of W gets dotted with x, and a dot product needs both lists the same length. If they don't match, the operation is simply impossible — and NumPy will refuse, loudly.
- The output size equals W's row count, because each row produces one number.

So the rule is: **inner numbers must match, outer numbers tell you the output's shape.**

```
(2, 3) @ (3,)  →  (2,)
(m, n) @ (n, p) →  (m, p)
```

Treat shape errors exactly like type errors. When you're lost in any formula later in the course — and everyone gets lost sometimes — the way out is always the same: write the shape next to every symbol and check the inner numbers match. The structure of the computation reappears like magic. This is the engineer's way of reading math, and it genuinely works.

## 5. Matrix × Matrix: The Whole Batch at Once

You now know matrix × vector. Matrix × matrix is the same thing, just repeated: `A @ B` means "dot every **row of A** with every **column of B**." The answer in row i, column j is (row i of A) · (column j of B).

Let's grind through one by hand — it takes two minutes and pays off for the rest of the course:

```
A = [[1, 2],          B = [[1, 0, 1],
     [3, 0]]               [2, 1, 0]]

A is (2,2).  B is (2,3).  Inner numbers match (2 = 2), so this works.
Output shape: (2, 3).

cell (0,0): row 0 of A · column 0 of B = [1,2]·[1,2] = 1×1 + 2×2 = 5
cell (0,1): row 0 of A · column 1 of B = [1,2]·[0,1] = 0 + 2     = 2
cell (0,2): row 0 of A · column 2 of B = [1,2]·[1,0] = 1 + 0     = 1
cell (1,0): row 1 of A · column 0 of B = [3,0]·[1,2] = 3 + 0     = 3
cell (1,1): row 1 of A · column 1 of B = [3,0]·[0,1] = 0 + 0     = 0
cell (1,2): row 1 of A · column 2 of B = [3,0]·[1,0] = 3 + 0     = 3

A @ B = [[5, 2, 1],
         [3, 0, 3]]
```

Nothing new happened. Six dot products, arranged in a grid.

So why does deep learning care so much about this? Here's the payoff. Suppose each row of a matrix `X` is one token's vector — three tokens, stacked up. Then `X @ W` applies the same transformation `W` to *all three tokens in one operation*. No loop. A matrix multiply is a **for-loop that got vectorized**, and GPUs happen to be machines built to do millions of these multiply-and-adds simultaneously. That one sentence is why GPUs made deep learning possible, and why every formula in this course is written as matrix multiplies: it's the fast way to say "do this to everything."

Two facts to pin to the wall:

- **Order matters.** `A @ B` is not `B @ A` — usually they're not even the same shape. Think of it like function composition: "resize the image, then compress it" is not "compress it, then resize it."
- **Grouping doesn't matter.** `(A @ B) @ C` equals `A @ (B @ C)`. You can regroup a chain of multiplies however is convenient. (File this away — backpropagation quietly exploits it.)

## 6. The Transpose: Flip the Spreadsheet

Take a matrix. Turn its rows into columns and its columns into rows. That's the transpose, written `Aᵀ` in papers and `A.T` in code.

```
A = [[1, 2, 3],        A.T = [[1, 4],
     [4, 5, 6]]               [2, 5],
                              [3, 6]]

shape (2, 3)   →   shape (3, 2)
```

The numbers don't change. Only the layout does. If A is a spreadsheet of *students × subjects* (each row is a student, each column is a subject), then A.T is *subjects × students* — the same grades, viewed sideways. You'd flip it depending on which question you're asking: "show me everything about student 2" wants rows-as-students; "show me everyone's math grade" wants rows-as-subjects.

So why does this humble flip appear in almost every formula in this course? Because of the shape rule from section 4. Watch what happens with a real question we'll actually need.

Say we have 3 tokens, each described by a 2-number vector, stacked as rows:

```
X = [[1, 0],        # token 0's vector
     [0, 1],        # token 1's vector
     [1, 1]]        # token 2's vector      shape: (3, 2)
```

Question: *how similar is every token to every other token?* From section 2 we know similarity = dot product. So we want: token 0 · token 0, token 0 · token 1, token 0 · token 2, token 1 · token 0... all nine pairs. We'd love a single multiply that produces the whole 3×3 grid of answers.

Try `X @ X`. Shapes: `(3, 2) @ (3, 2)`. Inner numbers: 2 and 3. They don't match. The multiply is impossible — not "wrong," literally impossible, like passing a string to a function that takes an int.

But remember how matrix multiply works: rows of the left thing get dotted with **columns** of the right thing. Our token vectors are sitting in X as *rows*. For them to be on the receiving end of the dot products, they need to be *columns*. And turning rows into columns is exactly what transpose does:

```
X @ X.T        shapes: (3, 2) @ (2, 3)  →  inner numbers match  →  output (3, 3) ✓
```

The output is the full 3×3 similarity grid: the cell at row i, column j is exactly token i · token j. We wanted every-pair-of-rows dotted together, and `something @ something.T` is simply *how you spell that* in matrix language.

Now you can read one of the most famous formulas in AI. In module 07 you'll meet attention, whose core is `Q @ K.T`. Q is a stack of vectors (as rows). K is a stack of vectors (as rows). `Q @ K.T` is "dot every row of Q with every row of K" — every query compared against every key, the full grid of similarities, in one multiply. When you get there, this line will already be an old friend.

The transpose also shows up in module 03's backprop formulas, like `d_W = X.T @ d_out`. For now, treat that one as a recipe — you'll *prove* it's correct in module 03 with a gradient check, which is honestly a better guarantee than any derivation. The rough intuition, for when you want it: the forward pass carried information from inputs to outputs through W; the backward pass carries blame from outputs back to inputs, and traveling the same wiring in reverse is what the transpose expresses. If that sentence doesn't click yet, let it go — it clicks on its own somewhere around your third day in module 03.

## 7. The Derivative: The Nudge Experiment

Forget the school version with limits and rules to memorize. Here is what a derivative *is*, in a form you can run:

> **Nudge the input a tiny bit. Measure how much the output moves. Divide. That ratio is the derivative.**

Think of a shower faucet. You turn the handle 1 degree, and the water gets 2 degrees hotter. The "derivative of temperature with respect to handle" is 2 — an amplification factor. It tells you two things at once: which *direction* the output moves (hotter, not colder), and *how strongly* (2× the nudge). That's everything a derivative ever tells you.

Let's run the experiment on `f(x) = x²` at the point x = 3:

```
f(3.000) = 9.000000
f(3.001) = 9.006001          (nudged the input by 0.001)

output moved by:  0.006001
input moved by:   0.001
ratio:            0.006001 / 0.001  ≈  6
```

So at x = 3, this function amplifies nudges by 6×. The school formula (derivative of x² is 2x, so 2×3 = 6) agrees — but notice you didn't need the formula. The experiment *is* the meaning; formulas are just shortcuts for it.

Why do we care? Because module 02 will hand a model a "loss" — one number measuring how wrong the model is — and ask, for each of the model's knobs: *if I nudge this knob, does the loss go up or down, and how fast?* That's a derivative per knob. The whole list of them has a fancy name, "the gradient" (papers write it ∇L), but it's just an array of shower-faucet readings: this knob matters a lot, turn it left; that knob barely matters, leave it alone. Training a neural network is reading the gauges and turning the knobs, millions of times.

Here are the only derivative facts this entire course uses. Each one can be verified by the nudge experiment — and the code does verify every one:

```
f(x) = c·x    →  derivative is c        (a wire with gain c: nudge in, c× nudge out)
f(x) = x²     →  derivative is 2x       (amplification grows as x grows — you just measured it)
f(x) = x + y  →  nudges pass through unchanged  (adding doesn't amplify anything)
f(x) = eˣ     →  derivative is eˣ       (the curious one: its slope equals its value)
f(x) = ln(x)  →  derivative is 1/x
```

No trigonometry. No integrals. No limit notation. This short table is the complete calculus requirement for building a GPT from scratch.

## 8. The Chain Rule: Nudges Through a Pipeline

One idea left. Deep learning stacks functions in pipelines — layer feeds layer feeds layer. So we need to know: how does a nudge at the start travel through to the end?

Start with money instead of math. You're converting currency through two exchanges:

```
dollars → euros:   1 dollar gets you 0.9 euros
euros → yen:       1 euro gets you 160 yen
```

If you add one extra dollar at the start, how many extra yen come out the end? You don't need to think hard: `0.9 × 160 = 144` extra yen. The rates **multiply**. Two conversion steps, each with its own rate, and the end-to-end rate is just their product.

That is the chain rule. The entire thing. Each stage of a pipeline has an amplification factor (its derivative — the shower-faucet reading from section 7), and the pipeline's end-to-end amplification is the stages multiplied together.

Let's do it with functions and real numbers. Pipeline: take x, square it, then multiply by 5. Start at x = 2.

```
stage 1:  x²   at x=2 amplifies nudges by 2x = 4    (from the table above)
stage 2:  5·u  amplifies nudges by 5                (a wire with gain 5)

end-to-end amplification:  4 × 5 = 20
```

Check it the honest way — nudge and measure the whole pipeline:

```
f(x) = 5·x²
f(2.000) = 20.000000
f(2.001) = 20.020005

moved: 0.020005 / 0.001 ≈ 20  ✓
```

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 300" font-family="sans-serif">
  <rect width="760" height="300" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">Chain rule: amplification factors through a pipeline just multiply</text>

  <defs>
    <marker id="cf" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#888"/>
    </marker>
  </defs>

  <!-- pipeline -->
  <rect x="60" y="90" width="110" height="60" rx="8" fill="#f6f8fa" stroke="#bbb"/>
  <text x="115" y="118" text-anchor="middle" font-size="14" fill="#333">x = 2</text>
  <text x="115" y="138" text-anchor="middle" font-size="11" fill="#888">input</text>

  <rect x="240" y="90" width="150" height="60" rx="8" fill="#eef4fb" stroke="#4A90D9" stroke-width="2"/>
  <text x="315" y="115" text-anchor="middle" font-size="14" fill="#333" font-family="monospace">square: x²</text>
  <text x="315" y="138" text-anchor="middle" font-size="12" fill="#4A90D9">amplifies wiggles ×4</text>

  <rect x="460" y="90" width="150" height="60" rx="8" fill="#eefaf1" stroke="#55A868" stroke-width="2"/>
  <text x="535" y="115" text-anchor="middle" font-size="14" fill="#333" font-family="monospace">times 5: 5u</text>
  <text x="535" y="138" text-anchor="middle" font-size="12" fill="#55A868">amplifies wiggles ×5</text>

  <rect x="660" y="90" width="70" height="60" rx="8" fill="#fdf3ee" stroke="#E8734A"/>
  <text x="695" y="118" text-anchor="middle" font-size="14" fill="#333">out</text>
  <text x="695" y="138" text-anchor="middle" font-size="11" fill="#888">= 20</text>

  <line x1="170" y1="120" x2="232" y2="120" stroke="#888" stroke-width="2.5" marker-end="url(#cf)"/>
  <line x1="390" y1="120" x2="452" y2="120" stroke="#888" stroke-width="2.5" marker-end="url(#cf)"/>
  <line x1="610" y1="120" x2="652" y2="120" stroke="#888" stroke-width="2.5" marker-end="url(#cf)"/>

  <!-- wiggle trace -->
  <text x="115" y="185" text-anchor="middle" font-size="12" fill="#9b59b6">nudge +0.001</text>
  <text x="315" y="185" text-anchor="middle" font-size="12" fill="#9b59b6">wiggle becomes +0.004</text>
  <text x="535" y="185" text-anchor="middle" font-size="12" fill="#9b59b6">wiggle becomes +0.020</text>

  <rect x="150" y="220" width="460" height="55" rx="10" fill="#f6f8fa" stroke="#ddd"/>
  <text x="380" y="243" text-anchor="middle" font-size="15" fill="#333" font-family="monospace">end-to-end sensitivity = 4 × 5 = 20</text>
  <text x="380" y="264" text-anchor="middle" font-size="12" fill="#888">backprop (module 03) = this, walked backwards station by station through the network</text>
</svg>

*Chain rule: wiggle amplifications multiply through a pipeline*

In school this was written `d(g(f(x)))/dx = g′(f(x)) · f′(x)`, which manages to make "exchange rates multiply" look terrifying. Same fact.

Why this matters so much: **backpropagation — the algorithm that trains every neural network on Earth — is nothing but this rule, applied backwards through the network, one stage at a time.** A 50-layer network is a 50-stage currency exchange. When you reach module 03, the "hard part" of deep learning will turn out to be a multiplication you already understand.

## The Notation Decompression Table

Papers compress the eight ideas into symbols. Here's the decoder ring. Keep it next to anything you read:

| You see | You read | In code |
|---|---|---|
| **x**, x⃗ (bold or arrow) | a vector — a list of numbers | `x = np.array([...])` |
| W, A (capital letter) | a matrix — a grid of numbers | `W` (always check its shape!) |
| Wx or W·x | matrix × vector: run every row's formula on x | `W @ x` |
| XWᵀ, Q·Kᵀ | a multiply where something was flipped so shapes line up | `X @ W.T` |
| Σᵢ aᵢbᵢ | "sum over i of aᵢ times bᵢ" — that's just a dot product | `a @ b` |
| ∥x∥ | the length of x | `np.linalg.norm(x)` |
| ∂L/∂w | the nudge experiment: nudge w, how much does L move? | measure it, or backprop it |
| ∇L ("nabla") | all the ∂L/∂w's collected in one list — "the gradient" | your `grads` array |
| softmax, σ | squash a list of scores into probabilities | module 03 explains it |
| ⊙ | multiply slot-by-slot (no summing) | `a * b` |

And the survival rule, worth repeating one last time: when a derivation loses you, stop trying to follow the algebra. Write the *shape* next to every symbol instead. `(batch, dim) @ (dim, vocab) → (batch, vocab)` tells you what's happening even when the Greek letters don't.


> **🛑 CHECKPOINT:** Run the code and modify it before proceeding. Reading without running is an illusion of knowledge.

## The Code

`math_primer.py` re-does **every hand calculation on this page** — the shopping bill, the movie-taste similarity, every cell of the matrix multiplies, the nudge-experiment derivatives against the formula table, the currency-chain rule — with an `assert` after each one. Run it. Then break it on purpose: change one of the "by hand" numbers and watch the assertion catch you. Math you can unit-test is math you can trust.

The one exercise on this page with the highest payoff-per-minute: do the section 5 matrix multiply on paper yourself, all six cells, *before* looking at the code. The row-by-row, column-by-column rhythm has to get into your hands, and there's no shortcut for that.

## Exercises

1. By hand: `[[2,0],[1,3]] @ [1,2]`. Two dot products, thirty seconds. Then check yourself in NumPy.
2. Predict the output shapes before running: `(5,3)@(3,3)`, `(3,)@(3,)`, `(2,3)@(2,3)`. The third one fails — explain why in shape language ("inner numbers...") before NumPy tells you.
3. Nudge-measure the derivative of `f(x) = x³` at x = 2 in Python. Then guess the general formula from the pattern you've seen (x² → 2x, so x³ → ?). You'll guess right — and you'll have discovered a calculus rule by experiment, which is more than most calculus students ever do.
4. Write cosine similarity yourself using only `*`, `sum`, and `sqrt` — no `np.linalg`. Test it on the movie-taste vectors from section 2. You should get numbers near +1 for you-and-friend, near −1 for you-and-stranger.
5. Chain rule: pipeline `x → x² → +1 → ×3`, at x = 2. Work out the end-to-end amplification by multiplying the stages (remember: the "+1" stage passes nudges through unchanged). Then verify by nudging the whole pipeline. Congratulations — that was a three-layer backprop, done by hand.

## Best External Resources

- 3Blue1Brown, *Essence of Linear Algebra*, chapters 1–4 only (vectors, linear combinations, matrices as transformations, matrix multiplication as composition). Watch them *after* this module — the animations will attach themselves to mechanics you now own. Skip determinants, eigenvectors, and rank entirely; this course never uses them.
- 3Blue1Brown, *Essence of Calculus*, chapters 1–3 only. Same rule: after, not before.

Next: **01 — Meaning as Geometry**, where these eight tools start doing AI.
