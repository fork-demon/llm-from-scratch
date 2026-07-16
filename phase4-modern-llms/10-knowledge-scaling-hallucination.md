# Module 10 — What LLMs Know, Why Scaling Works, Why They Hallucinate

**Time: about 2 weeks. No code — this is the reading-and-thinking module, and every idea in it stands on something you've now personally built or measured. One honest warning up front: this module contains more open research questions than settled facts, and it will say so wherever that's true.**

## Where Is the Knowledge Actually Stored?

Back in your very first message to this course, you asked a sharp question: *are facts stored in the weights or in the embeddings?* You've now built both, so you can receive a real answer:

**In the weights — and mostly in the feed-forward blocks' weights.**

Think about what you found in each place you built. The embedding table (module 05) holds something like word *identity* — which word this is, and roughly what neighborhood of meaning it lives in. Useful, but "cat is near dog" is not a *fact* in the sense you meant. Facts like "the Eiffel Tower is in Paris" live overwhelmingly in the FFN layers — which, conveniently for this story, is also where about two-thirds of a transformer's parameters sit (you counted them in module 08).

How does a feed-forward network — your module 03 MLP, of all things — *store a fact*? The best current mental model from interpretability research reads the FFN's two matrices as a **fuzzy key-value store**:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 420" font-family="sans-serif">
  <rect width="760" height="420" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">Where facts live: the FFN as a fuzzy key–value store</text>

  <defs>
    <marker id="kf" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#888"/>
    </marker>
  </defs>

  <!-- incoming token representation -->
  <rect x="50" y="160" width="170" height="70" rx="10" fill="#f6f8fa" stroke="#bbb" stroke-width="2"/>
  <text x="135" y="188" text-anchor="middle" font-size="12" fill="#333">token vector after attention:</text>
  <text x="135" y="208" text-anchor="middle" font-size="11" fill="#666">"Eiffel Tower ... located in"</text>

  <!-- first matrix: pattern detectors -->
  <rect x="280" y="90" width="180" height="210" rx="10" fill="#eef4fb" stroke="#4A90D9" stroke-width="2"/>
  <text x="370" y="115" text-anchor="middle" font-size="13" fill="#333" font-weight="bold">1st FFN matrix</text>
  <text x="370" y="133" text-anchor="middle" font-size="11" fill="#4A90D9">rows = pattern detectors (keys)</text>
  <g font-size="11" fill="#666">
    <text x="300" y="165">"subject = Eiffel Tower" → fires!</text>
    <text x="300" y="195">"relation = location" → fires!</text>
    <text x="300" y="225" fill="#bbb">"topic = cooking" → silent</text>
    <text x="300" y="255" fill="#bbb">"language = Python" → silent</text>
  </g>
  <text x="370" y="285" text-anchor="middle" font-size="10" fill="#888">nonlinearity gates which fire (mod 03's hinge)</text>

  <!-- second matrix: payloads -->
  <rect x="520" y="90" width="185" height="210" rx="10" fill="#eefaf1" stroke="#55A868" stroke-width="2"/>
  <text x="612" y="115" text-anchor="middle" font-size="13" fill="#333" font-weight="bold">2nd FFN matrix</text>
  <text x="612" y="133" text-anchor="middle" font-size="11" fill="#55A868">columns = payloads (values)</text>
  <g font-size="11" fill="#666">
    <text x="540" y="170">fired detectors write back:</text>
    <text x="540" y="196" fill="#55A868" font-weight="bold">boost "Paris"-direction</text>
    <text x="540" y="218" fill="#55A868">boost "France"-direction</text>
    <text x="540" y="248" fill="#bbb">(silent rows contribute 0)</text>
  </g>

  <line x1="220" y1="195" x2="272" y2="195" stroke="#888" stroke-width="2.5" marker-end="url(#kf)"/>
  <line x1="460" y1="195" x2="512" y2="195" stroke="#888" stroke-width="2.5" marker-end="url(#kf)"/>

  <!-- division of labor -->
  <rect x="60" y="330" width="640" height="75" rx="10" fill="#f6f8fa" stroke="#ddd"/>
  <text x="380" y="354" text-anchor="middle" font-size="13" fill="#333" font-weight="bold">Division of labor:  attention ROUTES  ·  FFNs RECALL and TRANSFORM</text>
  <text x="380" y="376" text-anchor="middle" font-size="12" fill="#666">~2/3 of all parameters sit in FFN layers — evidence: researchers can surgically edit one fact ("Eiffel Tower is in Rome")</text>
  <text x="380" y="394" text-anchor="middle" font-size="12" fill="#888">by modifying a specific FFN layer (ROME). Caveats: facts are distributed, redundant, superposed — not one neuron each.</text>
</svg>

*A fact, retrieved: pattern detectors match the incoming context; matched detectors write their payload into the token's vector*

Walk through the diagram with your module 03 knowledge. A token's vector arrives at the FFN carrying context (attention already mixed in "Eiffel Tower" and "located in"). The **first matrix's rows act like pattern detectors** — each row is a dot product against the incoming vector (a shopping bill! module 00!), scoring "does this look like *my* pattern?" The nonlinear hinge then silences the weak matches — only genuinely-triggered detectors stay on. The **second matrix holds the payloads**: each surviving detector writes its associated content into the output — here, pushing the vector toward the "Paris" direction of the space. Detect, gate, write back. A key-value store built out of two matrix multiplies and a bend.

Is this just a nice story? Here's the evidence that it's more: researchers (the ROME paper — Meng et al.) can *surgically edit a single fact* by modifying one FFN layer's weights — make a model believe the Eiffel Tower is in Rome — and the model then says Rome consistently, even in rephrased questions it's never seen. You can't do surgery that precise on an organ you've completely misunderstood.

Two honest caveats, so you carry the true picture and not the tidy one. First, facts are **not** stored one-per-neuron like rows in a table — they're smeared across many detectors and several layers, with redundancy, and often *superposed* (more stored patterns than dimensions to store them in, packed into the almost-perpendicular directions module 01 told you high-dimensional space is full of — that strange fact about roomy spaces just became load-bearing). Second, "fuzzy key-value store" describes the parts researchers have managed to interpret; a large fraction of FFN behavior remains genuinely unexplained. The map is good; it is not the territory.

## Memorization vs. Generalization

Here's a distinction you don't have to learn, because you *manufactured* it in module 08, exercise 4. You shrank the corpus, trained long, and watched two curves split:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 360" font-family="sans-serif">
  <rect width="760" height="360" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">Memorization vs generalization: the two loss curves tell you which is happening</text>

  <!-- axes -->
  <line x1="80" y1="300" x2="700" y2="300" stroke="#bbb" stroke-width="1.5"/>
  <line x1="80" y1="300" x2="80" y2="60" stroke="#bbb" stroke-width="1.5"/>
  <text x="620" y="322" font-size="12" fill="#999">training steps →</text>
  <text x="40" y="70" font-size="12" fill="#999" transform="rotate(-90 40 70)">loss (surprise)</text>

  <!-- train loss: keeps falling -->
  <path d="M 90 90 Q 200 140 300 190 Q 450 250 690 280" fill="none" stroke="#4A90D9" stroke-width="3"/>
  <text x="600" y="265" font-size="12" fill="#4A90D9" font-weight="bold">training loss: keeps falling</text>

  <!-- val loss: falls then rises -->
  <path d="M 90 95 Q 200 150 320 200 Q 400 225 450 222 Q 560 215 690 150" fill="none" stroke="#E8734A" stroke-width="3"/>
  <text x="560" y="130" font-size="12" fill="#E8734A" font-weight="bold">held-out (validation) loss</text>

  <!-- divergence marker -->
  <line x1="430" y1="60" x2="430" y2="300" stroke="#9b59b6" stroke-width="1.5" stroke-dasharray="6,4"/>
  <text x="430" y="52" text-anchor="middle" font-size="12" fill="#9b59b6" font-weight="bold">the turn</text>

  <!-- zone labels -->
  <text x="240" y="90" font-size="12" fill="#55A868" font-weight="bold">GENERALIZING</text>
  <text x="240" y="107" font-size="11" fill="#666">both curves fall together:</text>
  <text x="240" y="122" font-size="11" fill="#666">learning transferable patterns</text>

  <text x="530" y="80" font-size="12" fill="#c0563a" font-weight="bold">MEMORIZING</text>
  <text x="530" y="97" font-size="11" fill="#666">train falls, val rises: reproducing</text>
  <text x="530" y="112" font-size="11" fill="#666">the corpus, not learning from it</text>

  <text x="380" y="345" text-anchor="middle" font-size="12" fill="#888">You created both regimes yourself in module 08, exercise 4. Real LLMs sit on both ends at once: memorized quotes AND generalized grammar.</text>
</svg>

*Left of the turn: learning the language. Right of the turn: photocopying the corpus.*

While both curves fall together, the model is learning *transferable patterns* — things true of English generally, not just of your file. After the turn, it's stuffing the specific corpus into its weights — training loss keeps improving because it's literally memorizing the answers, while held-out loss worsens because none of that memorization transfers. You watched your model cross that line and start regurgitating your 1KB corpus verbatim.

Real LLMs, trained on trillions of tokens, live on **both sides at once**. They memorize text that repeats often — famous quotes, code license boilerplate, and (a real legal and privacy problem) personal data that appeared many times in scraped data. And they generalize grammar, style, idiom, and reasoning-shaped patterns, which is why they handle sentences never written before in history. The best single predictor of whether something got memorized, per the research: **how many times it repeated in training data.**

The uncomfortable open question at the bottom of this: when a model appears to *reason*, is that genuine composition of ideas, or extremely well-interpolated memorization of reasoning-shaped text? The honest answer as of this writing: nobody fully knows; current models show some behavior that looks like real composition and some failure modes that smell exactly like retrieval. Anyone selling you certainty on this question, in either direction, is selling.

## Why Hallucination Is Structural, Not a Bug

Now let's earn the answer to "why do these things confidently make stuff up?" — using only parts you've built. Assemble three facts:

**Fact 1 (module 06):** the model is a next-plausible-token machine. Its one skill is continuing text in the way training data suggests text like this continues. The generation loop draws from that plausibility distribution, one token at a time, no plan, no lookahead.

**Fact 2 (module 03):** softmax always answers. Ask a database for a missing row and you get `NOT FOUND` — a lookup has a failure mode built in. Ask a softmax over 50,000 tokens and you get... a probability distribution. Every time. There is no `NOT FOUND` in the architecture. The model can only say "I don't know" if *the literal text "I don't know" is the most plausible continuation* — which for confident-sounding prompts, it usually isn't in the training data.

**Fact 3 (module 05, exercise 3):** you deleted "dog" sentences and watched "dog" get a garbage vector. Sparse training signal → unreliable geometry. For facts the training data covered heavily, the most plausible continuation *is* the truth — plausibility and truth were welded together by repetition. For facts covered thinly, plausibility and truth come apart.

Put them together. Ask about something well-represented → truthful answer, because truth is what plausible text looks like there. Ask about something sparse — an obscure paper, a small-town event — and the machine does *exactly what it always does*: emits the most plausible-looking continuation. A citation that looks the way citations look. A biography shaped the way biographies are shaped. Fluent, well-formatted, and fabricated — **the generalization machinery doing its job, in a place where its job produces falsehood, with no mechanism anywhere that could notice.**

That's why "hallucination" is structural. Mitigations exist and matter — preference tuning (module 09) teaches models to hedge and refuse more often, and retrieval (module 12) changes the game by putting the facts *in the context window* — but they manage the phenomenon rather than remove it. There is no truth register in the architecture to consult. You know; you built the architecture.

## Why Does Scaling Work?

The empirical backbone of the LLM era is embarrassingly simple to state: **make the model bigger, the data bigger, and the compute bigger, and the loss falls — smoothly, predictably, following a power law, across many orders of magnitude.** (Kaplan et al. 2020; refined by the "Chinchilla" paper, Hoffmann et al. 2022.) It's the closest thing the field has to a law of nature, and — real honesty — nobody fully knows *why* it's this clean.

The useful intuitions on offer: bigger models have more of module 01's almost-perpendicular directions — more slots for distinct features before they start interfering. And language itself has a long tail — after common grammar comes uncommon idiom, then rare facts, then rarer facts — so each 10× of scale buys the model access to another stratum of the tail. Smooth tail, smooth curve. Suggestive, not proven.

Chinchilla added the practical recipe that reorganized the industry's spending: parameters and data should grow *together* — roughly **20 tokens of training data per parameter**. Before that paper, labs built models too big for their data budgets, like buying a bigger warehouse while shipping the same inventory. After it, "smaller model, much more data" became the standard play (that's the Llama-family formula).

And a calibration for the word **emergence** — abilities like arithmetic or few-shot learning "suddenly appearing" at scale. The observed behavior is real. But keep your module 08 run in mind: your samples "suddenly" had real words at some checkpoint — while the loss curve underneath fell perfectly smoothly the whole time. Some celebrated emergence is that same effect: smooth competence growth crossing a pass/fail measurement threshold looks like a jump. Which cases are truly discontinuous is — again, honestly — contested.

## Fine-Tuning: What Actually Changes

Mechanically you already know everything: fine-tuning is module 02's loop continued on new data. No new math. What's worth engraving is the **behavior vs. knowledge** rule from module 09:

**Fine-tuning is excellent at behavior** — format, tone, domain style, following your team's conventions, tool-use patterns. A few thousand examples reshape *how the model acts* remarkably well.

**Fine-tuning is poor at installing knowledge.** New facts enter slowly and unreliably, and aggressive fine-tuning risks **catastrophic forgetting** — degrading old capabilities while cramming new material. The systems-flavored reason: there is no isolation. Every weight serves every task simultaneously (superposition again), so gradients from your new data overwrite structure the old data built. No namespaces, no transactions, no MVCC in a weight matrix.

Hence the industry rule of thumb you can now justify from first principles: **fine-tune for behavior; use retrieval (module 12) for knowledge.**

One acronym worth owning since it's everywhere: **LoRA**. Freeze all the original weights; learn only small low-rank correction matrices alongside them (W stays fixed, you learn a skinny A·B to add to it). Under 1% of the parameter count, most of the benefit, and the corrections are swappable like plugins — one LoRA for your legal-drafting style, another for your codebase. This is why fine-tuning is affordable outside big labs at all.

## MoE, and a Few Names Worth Recognizing

**Mixture of Experts (MoE)** — the architecture trick behind Mixtral and (reportedly) GPT-4-class models, explained with what you have: replace each block's FFN with N parallel FFNs ("experts") plus a tiny learned **router** that sends each token to only its top 1–2 experts. Result: enormous *total* parameter count (lots of storage for facts — remember where facts live!), small *active* count per token (compute stays cheap). It's sharding, where the shard key is learned and the partition function is "whatever the router finds useful" — which, amusingly, rarely maps to human categories like "the math expert"; routers tend to specialize on token-level patterns instead. The costs are ones you'd predict: every expert must sit in GPU memory even though few activate, and keeping the router load-balanced is a genuine distributed-systems headache (there are auxiliary loss terms whose only job is stopping hot-spotting).

Three more names, one line each, all now within reach: **RoPE** — encoding position by rotating the Q and K vectors instead of adding position embeddings; better behavior on long contexts. **GQA** — several query heads sharing one K/V head; shrinks module 09's cache formula, which you can now recompute yourself. **Quantization** — storing weights in 4–8 bits instead of 16; a big memory discount for a small quality tax, and the reason large models run on laptops.

## Exercises (reading and writing, not code)

1. Write one page explaining hallucination to a non-technical executive — no ML vocabulary allowed. This is the final exam for whether you own the concept rather than rent it. (The three-facts structure above is a scaffold; the shower-faucet rule applies: if you can't say it plainly, you don't have it yet.)
2. Take your module 08 exercise-4 run and label your own observations with this module's vocabulary: which measurement was generalization, which was memorization, and which instrument detected the crossover?
3. Read just the abstract and Figure 1 of the ROME paper (Meng et al., *"Locating and Editing Factual Associations in GPT"*). Map their procedure onto the FFN key-value picture above — and then write down one place where the mapping feels *too* neat. (Practicing suspicion of tidy stories about neural networks is a transferable skill.)
4. Back-of-envelope: Chinchilla says ~20 tokens per parameter. How many tokens for a 70B model? At ~4 characters per token, how many times all of English Wikipedia (~25GB of text) is that? Your answer explains why "are we running out of internet?" is a real conversation in the field.

## Best External Resources

- Karpathy, *"Intro to Large Language Models"* (1 hour) — the ecosystem-level view, ideally timed now.
- 3Blue1Brown, *"How might LLMs store facts"* (Deep Learning chapter 7) — the FFN key-value story, animated.
- Anthropic's interpretability write-ups (*"Toy Models of Superposition"*, *"Scaling Monosemanticity"*) — skim for the figures; you now genuinely have the prerequisites, which puts you in rare company.
- The Chinchilla paper (Hoffmann et al. 2022) — read §1 and the tables, skip the curve-fitting details.

Next: **11 — Vector Search**, where module 05's embeddings stop being a model component and become infrastructure.
