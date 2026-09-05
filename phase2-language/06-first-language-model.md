# Module 06 — Your First Language Model

> **⏱️ Time:** ~1-2 weeks.  
> **What you'll build:** Core mechanics and conceptual understanding.

---

## 🎯 TL;DR
1. **The Core Problem:** The challenge this module solves.
2. **The Mechanism:** How we solve it using first principles.
3. **The Payoff:** What you can do with this new capability.

---


**Time: about 2 weeks. Code: `bigram_lm.py`.**

## The Definition, At Last

Six modules in, we can finally say what this course is about, in one plain sentence:

> **A language model is a system that, given some text, assigns probabilities to what comes next.**

That's the entire definition. Given "the cat sat on the," a language model says: "mat" 40%, "floor" 15%, "roof" 5%, ... and so on down the vocabulary. GPT-4 fits this definition. So does the thirty-line model you're about to build. The *only* difference between them — and I mean this literally — is how good the probability estimates are. Everything in modules 07–09 exists to make the estimates better. The definition never changes again.

This module builds the smallest language model that can exist: given ONE character, predict the next character. It's called a **bigram model** ("bigram" = pair of adjacent things). It will generate hilarious gibberish. That's fine — the gibberish isn't the point. The point is that you'll build the **complete life cycle of an LLM** — raw text → training pairs → train → measure quality → generate new text — in a form small enough to hold entirely in your head at once. After this module, every upgrade in the course is swapping one part in a machine you already own.

## The Module's Big Idea: Build It Twice

Here's the plan that makes this module special. We'll build the same predictor two completely different ways, and then discover something about them.

**Way A: just count.** No neural network. No gradients. Walk through the training text and tally: after 'q', how often does each character appear? After 't'? After a space? Store the tallies in a table, one row per character. Normalize each row so it sums to 1. Done — the table *is* the model: look up your current character's row, and there are your next-character probabilities. This is honest, classical statistics (the family is called *n-gram models*, and it powered speech recognition for decades).

Something worth knowing: for this task, the count table is *unbeatable*. If all you get to see is one character, the true observed frequencies are, provably, the best possible guesses. No cleverness can beat counting here. Hold that thought.

**Way B: a neural network.** Embedding lookup (module 05) → linear layer → softmax → cross-entropy → gradient descent. The full Phase-1-and-2 apparatus, aimed at the same tiny task.

Now the punchline, which you will verify with your own numbers when you run the code:

> **The neural network converges to the count table.**

The code prints Way A's loss (say, 1.7518), then trains Way B and you watch its loss walk down: 3.31... 1.88... 1.76... and flatten *just above* 1.7518. Gradient descent, knowing nothing about counting, **rediscovers the frequency statistics** — because the count table was the optimal answer and gradient descent hunts optima. When someone says "an LLM learned the statistics of its training data," this is what they mean, and you'll have watched it happen digit by digit.

## Then Why Neural Networks At All?

If counting is optimal and simpler — why did anyone bother with neural networks? This question has a precise answer, and it's the reason the next three modules exist. **Counting cannot scale in context length.**

Follow the arithmetic. One character of context: table with 27 rows. Fine. Two characters: 27² = 729 rows. Ten characters of context: 27¹⁰ ≈ **200 trillion rows** — and almost every row describes a context that appeared *zero times* in your training data, so its counts are empty and the model just shrugs. Counting has no way to say "I've never seen exactly `qzt hepl on` before, but it *resembles* contexts I have seen." Every context is an island.

A neural network's parameters are *shared* across all contexts. Similar contexts — nearby in embedding space, module 01! — flow through the same weights and produce similar predictions, **even for contexts never seen in training**. That ability to generalize across the astronomically many never-seen contexts is THE reason neural language models exist. Not accuracy on seen data (counting wins there) — coverage of the unseen.

The code makes this concrete with Way C: a model that reads 3 characters of context (three embeddings concatenated, into a module-03 MLP). Its loss lands *below* the bigram optimum — around 0.35 versus 1.75. More context beats any amount of cleverness with less context. Remember that sentence; it's the cliffhanger that attention resolves.

## Generation: The Loop That Is All of ChatGPT

So far the model only assigns probabilities. How does it *write*? With a loop so simple it feels like it must be missing something:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 380" font-family="sans-serif">
  <rect width="760" height="380" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">Autoregressive generation: the loop that IS ChatGPT</text>

  <defs>
    <marker id="ag" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#888"/>
    </marker>
    <marker id="agb" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#9b59b6"/>
    </marker>
  </defs>

  <!-- context -->
  <rect x="60" y="90" width="180" height="70" rx="10" fill="#f6f8fa" stroke="#bbb" stroke-width="2"/>
  <text x="150" y="118" text-anchor="middle" font-size="13" fill="#333">context so far</text>
  <text x="150" y="140" text-anchor="middle" font-size="13" fill="#666" font-family="monospace">"the cat s"</text>

  <!-- model -->
  <rect x="300" y="90" width="160" height="70" rx="10" fill="#eef4fb" stroke="#4A90D9" stroke-width="2"/>
  <text x="380" y="118" text-anchor="middle" font-size="13" fill="#333">model</text>
  <text x="380" y="140" text-anchor="middle" font-size="11" fill="#888">P(next | context)</text>

  <!-- distribution -->
  <rect x="520" y="70" width="180" height="120" rx="10" fill="#fcfcfc" stroke="#ddd"/>
  <text x="610" y="90" text-anchor="middle" font-size="11" fill="#888">distribution over next char</text>
  <g font-size="11" font-family="monospace" fill="#333">
    <rect x="540" y="100" width="98" height="13" fill="#4A90D9" opacity="0.7"/><text x="644" y="111">a  0.49</text>
    <rect x="540" y="118" width="50" height="13" fill="#4A90D9" opacity="0.5"/><text x="644" y="129">i  0.25</text>
    <rect x="540" y="136" width="30" height="13" fill="#4A90D9" opacity="0.4"/><text x="644" y="147">o  0.15</text>
    <rect x="540" y="154" width="14" height="13" fill="#4A90D9" opacity="0.3"/><text x="644" y="165">u  0.07</text>
  </g>

  <line x1="240" y1="125" x2="292" y2="125" stroke="#888" stroke-width="2.5" marker-end="url(#ag)"/>
  <line x1="460" y1="125" x2="512" y2="125" stroke="#888" stroke-width="2.5" marker-end="url(#ag)"/>

  <!-- sample -->
  <rect x="545" y="230" width="130" height="54" rx="10" fill="#eefaf1" stroke="#55A868" stroke-width="2"/>
  <text x="610" y="252" text-anchor="middle" font-size="13" fill="#333">sample: 'a'</text>
  <text x="610" y="272" text-anchor="middle" font-size="11" fill="#888">roll the weighted die</text>
  <line x1="610" y1="190" x2="610" y2="222" stroke="#888" stroke-width="2.5" marker-end="url(#ag)"/>

  <!-- feedback loop -->
  <path d="M 545 262 C 220 262 150 240 150 168" fill="none" stroke="#9b59b6" stroke-width="2.5" marker-end="url(#agb)"/>
  <text x="330" y="248" text-anchor="middle" font-size="12" fill="#9b59b6" font-weight="bold">append output to context, go again</text>
  <text x="330" y="290" text-anchor="middle" font-size="12" fill="#666" font-family="monospace">"the cat s" → "the cat sa" → "the cat sat" → ...</text>

  <rect x="60" y="315" width="640" height="50" rx="10" fill="#f6f8fa" stroke="#ddd"/>
  <text x="380" y="336" text-anchor="middle" font-size="12" fill="#333">One token per iteration; no plan, no lookahead — each token is one draw from a conditional distribution.</text>
  <text x="380" y="355" text-anchor="middle" font-size="12" fill="#888">When ChatGPT streams words at you, this exact loop is running. Only the model in the blue box gets bigger.</text>
</svg>

*Ask for probabilities, roll the weighted die, append the result, ask again*

Ask the model for next-character probabilities. Roll a weighted die and pick one. Append it to the context. Ask again. That's called **autoregressive generation** ("auto-regressive" ≈ "feeds on its own output"), and it is not a simplification of what ChatGPT does — it IS what ChatGPT does. When you watch an assistant's answer stream in word by word, you are watching this loop iterate in real time, one draw per token.

Let that sink in properly, because it quietly explains a lot: in the base loop there is **no plan and no lookahead**. The model doesn't decide what the paragraph will say and then write it. Each token is a single draw from "what plausibly comes next," conditioned on everything so far — including its own previous draws. Keep this picture; when module 10 explains hallucination, this loop is half the explanation.

One practical detail you'll hit in the code: why *roll a die* instead of always picking the most likely character? Try it — the code lets you — and you'll find that always-take-the-max gets trapped in loops (`the the the the`). The most likely *next step*, taken every time, produces very unlikely *text*. A bit of randomness keeps generation fresh. How much randomness is a genuinely interesting dial called temperature, and it gets proper treatment in module 09.

## Measuring Quality: Surprise, and a Friendlier Cousin

How good is a language model? Show it text it hasn't seen and measure its average surprise — module 03's cross-entropy, now used as the universal benchmark of the field.

There's an equivalent form with a nicer feel: **perplexity**, which is just e raised to the cross-entropy. Its charm is the interpretation: *perplexity ≈ the number of options the model is effectively torn between at each step.* A model that's pure random guessing over 27 characters has perplexity 27 — torn between everything. Your bigram model scores around 6 — as if choosing between 6 plausible next characters. Big modern models on English text get into the low single digits per token. Entire careers, and billions of dollars of compute, are spent pushing that number down.


> **🛑 CHECKPOINT:** Run the code and modify it before proceeding. Reading without running is an illusion of knowledge.

## The Code

`bigram_lm.py` runs the whole story end to end:

1. Loads a small corpus (built into the file), builds the character vocabulary.
2. **Way A** — builds the count table, prints its loss and perplexity, generates gibberish from it.
3. **Way B** — trains the neural bigram, printing loss as it converges toward Way A's number. Then generates — note the gibberish has the same *texture* as Way A's, because the two models have become the same model.
4. **Way C** — the 3-character-context MLP, breaking the bigram ceiling by a mile.

One small honesty note in the code worth appreciating: the count table adds a tiny constant (0.01) to every tally before normalizing — called *smoothing* — so that pairs never seen in training get a small probability instead of zero. Why bother? Because if the model says a pair is *impossible* (probability zero) and the test text contains it, the surprise is −log(0) = infinity, and your average blows up. "Never say never" turns out to be a mathematical necessity. Try setting the constant to 1.0 and watch Way A get *worse* — heavy smoothing dilutes real statistics. Even in a 30-line model, there are real engineering trade-offs.

## Modify-It Exercises

1. Feed it different corpora — Shakespeare, Python source, your own Slack export — and compare the generated gibberish and the perplexity of each. The gibberish takes on each corpus's texture (code-flavored gibberish has indentation and colons!), and the perplexity differences tell you something real: which text is inherently more predictable.
2. Extend Way A to *trigrams* (two characters of context → 27² rows). Measure the loss improvement — nice! Then check what fraction of possible two-character contexts your corpus actually contains. Now estimate ten characters. You've hit the combinatorial wall personally, which is the only way it really lands.
3. In Way C, grow the context from 3 to 5 to 8 characters, recording loss each time. The gains shrink. Part of the reason: concatenation treats each position as rigidly separate — the pattern "q at position 2 is followed by u" and "q at position 3 is followed by u" are learned *independently*, as if they were unrelated facts. That rigidity is dumb, you can smell that it's dumb, and the fix for it is precisely module 07.
4. Compute Way A's loss on its own training text versus on some held-out text. The gap you just measured is **overfitting** — memorizing the training data rather than learning the language. Small here, but you've now seen it with your own instruments; in module 08 you'll deliberately inflate it, and in module 10 it becomes a main character.

## Best External Resources

- Karpathy, *"The spelled-out intro to language modeling: building makemore"* — this module was deliberately built to be compatible with his; watching it after is like getting a code review of work you've already done.
- Claude Shannon's 1948 paper, section 3 only — the n-gram generation idea, in the founding document of information theory, complete with Shannon's own hand-generated gibberish. It's 75 years old and it's *this module*. Twenty minutes, worth it for perspective alone.

Next: Phase 3. Everything so far — geometry, gradients, backprop, tokens, embeddings, the definition of a language model — is on the table. **07 — Attention** is the one genuinely new idea standing between you and GPT.
