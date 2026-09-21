# Module 08 — Assembling the Tiny GPT

**Time: about 4 weeks. Code: `tiny_gpt.py` (PyTorch — and you've earned it). This is the summit. Everything before was the climb.**

## First, Let's Talk About PyTorch

Until now: pure NumPy, every gradient by hand, on principle. Now we switch to PyTorch, and I want to be precise about what you're being handed, because it's much less than you might fear.

PyTorch automates exactly two things — both of which you have personally done by hand:

1. During the forward pass, it quietly *records* every operation (you did this in module 03 with your `cache` list).
2. When you call `loss.backward()`, it replays your module 03 blame-table rules through that recording, backwards (you did this in your `backward()` method).

That's the whole gift. A `torch.Tensor` is a NumPy array that carries a gradient recorder. You are not graduating into a framework you don't understand — you are *hiring out labor you've already performed*, with a gradient check as your receipt. The code keeps a deliberate NumPy accent throughout: explicit shapes on every line, no prefab transformer classes, every layer written out so you can match it to the module where you built it.

## The Transformer Block: One Diagram, Zero New Ideas

A GPT is a surprisingly short recipe:

1. Token embeddings + position embeddings (modules 05 and 07).
2. A stack of N identical **blocks** — the transformer's repeating unit.
3. A final projection from the last layer's vectors to next-token scores over the vocabulary (module 06's job).

So everything hinges on what one block is. Here it is:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 560" font-family="sans-serif">
  <rect width="760" height="560" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">The GPT stack: communicate (attention), compute (FFN), repeat</text>

  <defs>
    <marker id="up" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#888"/>
    </marker>
    <marker id="res" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#9b59b6"/>
    </marker>
  </defs>

  <!-- bottom: embeddings -->
  <rect x="230" y="480" width="300" height="44" rx="8" fill="#f6f8fa" stroke="#bbb"/>
  <text x="380" y="503" text-anchor="middle" font-size="13" fill="#333">token embedding + position embedding</text>
  <text x="380" y="518" text-anchor="middle" font-size="11" fill="#888">(modules 05 + 07)</text>
  <line x1="380" y1="480" x2="380" y2="448" stroke="#888" stroke-width="2.5" marker-end="url(#up)"/>

  <!-- block container -->
  <rect x="180" y="120" width="400" height="326" rx="12" fill="none" stroke="#4A90D9" stroke-width="2" stroke-dasharray="7,5"/>
  <text x="196" y="143" font-size="13" fill="#4A90D9" font-weight="bold">one Block  (× N layers)</text>

  <!-- sublayer 1: attention -->
  <rect x="270" y="360" width="220" height="40" rx="8" fill="#eef4fb" stroke="#4A90D9"/>
  <text x="380" y="385" text-anchor="middle" font-size="13" fill="#333">LayerNorm → Multi-Head Attention</text>

  <circle cx="380" cy="315" r="14" fill="#fff" stroke="#9b59b6" stroke-width="2"/>
  <text x="380" y="320" text-anchor="middle" font-size="16" fill="#9b59b6">+</text>

  <line x1="380" y1="448" x2="380" y2="402" stroke="#888" stroke-width="2.5" marker-end="url(#up)"/>
  <line x1="380" y1="360" x2="380" y2="331" stroke="#888" stroke-width="2.5" marker-end="url(#up)"/>
  <!-- residual skip 1 -->
  <path d="M 380 430 C 230 430 230 315 364 315" fill="none" stroke="#9b59b6" stroke-width="2.5" marker-end="url(#res)"/>
  <text x="212" y="378" font-size="11" fill="#9b59b6">residual:</text>
  <text x="212" y="392" font-size="11" fill="#9b59b6">x + attn(ln(x))</text>

  <!-- sublayer 2: ffn -->
  <rect x="270" y="230" width="220" height="40" rx="8" fill="#eefaf1" stroke="#55A868"/>
  <text x="380" y="255" text-anchor="middle" font-size="13" fill="#333">LayerNorm → FeedForward (MLP)</text>

  <circle cx="380" cy="185" r="14" fill="#fff" stroke="#9b59b6" stroke-width="2"/>
  <text x="380" y="190" text-anchor="middle" font-size="16" fill="#9b59b6">+</text>

  <line x1="380" y1="301" x2="380" y2="272" stroke="#888" stroke-width="2.5" marker-end="url(#up)"/>
  <line x1="380" y1="230" x2="380" y2="201" stroke="#888" stroke-width="2.5" marker-end="url(#up)"/>
  <!-- residual skip 2 -->
  <path d="M 380 290 C 230 290 230 185 364 185" fill="none" stroke="#9b59b6" stroke-width="2.5" marker-end="url(#res)"/>

  <line x1="380" y1="171" x2="380" y2="120" stroke="#888" stroke-width="2.5"/>
  <line x1="380" y1="120" x2="380" y2="92" stroke="#888" stroke-width="2.5" marker-end="url(#up)"/>

  <!-- top: logits -->
  <rect x="230" y="48" width="300" height="44" rx="8" fill="#fdf3ee" stroke="#E8734A"/>
  <text x="380" y="70" text-anchor="middle" font-size="13" fill="#333">final LayerNorm → project to vocab logits</text>
  <text x="380" y="86" text-anchor="middle" font-size="11" fill="#888">softmax → P(next token) — module 06's loop</text>

  <!-- side annotations -->
  <text x="600" y="380" font-size="12" fill="#4A90D9" font-weight="bold">COMMUNICATE</text>
  <text x="600" y="396" font-size="11" fill="#666">tokens exchange info</text>
  <text x="600" y="410" font-size="11" fill="#666">(module 07)</text>
  <text x="600" y="250" font-size="12" fill="#55A868" font-weight="bold">COMPUTE</text>
  <text x="600" y="266" font-size="11" fill="#666">each token thinks alone</text>
  <text x="600" y="280" font-size="11" fill="#666">(module 03's MLP;</text>
  <text x="600" y="294" font-size="11" fill="#666">most params + facts live here)</text>
  <text x="600" y="180" font-size="11" fill="#9b59b6">the '+' is the gradient</text>
  <text x="600" y="194" font-size="11" fill="#9b59b6">highway that makes</text>
  <text x="600" y="208" font-size="11" fill="#9b59b6">deep stacks trainable</text>
</svg>

*One block = talk to your neighbors, then think it over. Stack N of them and you have a GPT.*

Four components. Let's take each one honestly, because two of them you built and two of them you *already know you need*:

**Multi-head attention** — module 07, character for character. Its role in the block's division of labor: **communication**. This is where tokens exchange information — "what's around me? what matters to me right now?"

**The feed-forward network (FFN)** — module 03's MLP, character for character: expand to 4× the width, bend (GELU — a smoothed ReLU hinge, same idea), project back down. But notice something important: it's applied to **each token separately**. No token-to-token contact happens here. Its role: **computation** — after attention gathers context, each token privately *thinks about what it just collected*. The transformer's rhythm is exactly this alternation: talk, think, talk, think, N times. Two more facts for later: roughly two-thirds of the model's parameters live in these FFNs, and — as module 10 will detail — interpretability research suggests these layers are heavily involved in recalling factual associations, although facts are not cleanly stored in any one place (module 10 is careful about what is and is not known).

**Residual connections** — the little `+` circles in the diagram, and the answer to a frustration you already own. Remember module 03, exercise 4? You stacked more layers and training got flaky. The reason: the blame signal, multiplying through station after station on its way back (chain rule), degrades — by layer 30 it arrives shrunken or exploded. The fix costs one character of code: instead of `x = layer(x)`, write **`x = x + layer(x)`** — add the layer's input back to its output. Why does that solve it? Because now the backward pass always has an express lane: gradients flow through the `+` unchanged, straight down the stack, no matter how deep. And there's a beautiful reframe hiding in the same character: each block no longer *replaces* the representation, it computes a *correction* to it. Corrections can safely start near zero and grow as the block learns something useful — replacements can't. This one trick is what makes 100-layer models trainable at all. One plus sign.

**LayerNorm** — the unglamorous one. As activations flow through many layers, their overall scale drifts — some layers' outputs balloon, others shrivel — and gradient descent hates chasing a moving target. LayerNorm is a thermostat: at each entry point it re-standardizes every token's vector (mean 0, spread 1, then a learned re-scale), so each sublayer always receives inputs in a familiar range. That's the whole job. It's the least interesting component and the second most load-bearing after residuals — deep learning has a habit of being carried by its plumbing. (Small historical footnote in the code: GPT applies LayerNorm *before* each sublayer, not after as the 2017 paper did — "pre-norm" turned out to train more stably. The kind of detail that separates papers from production.)

And one elegant touch you'll find in the code, called **weight tying**: the matrix that maps tokens *into* vectors (the embedding table) is reused, transposed, as the matrix that maps final vectors back *out* to token scores. Reading meaning in and reading meaning out are the same geometry, shared. In our character-level model the saving is small (about 8 thousand of 810 thousand parameters, because the vocabulary has only 65 entries), but in GPT-2, with a 50,257-token vocabulary, the same trick saves about 39 million parameters.

## What We Train It On, and What You'll Watch Happen

The classic: ~1MB of Shakespeare, character-level, next-character prediction, cross-entropy loss. The optimizer is AdamW — a refined SGD that adapts step size per parameter; under the hood it is still, conceptually, module 02's `p -= lr * grad`.

Here's what makes this the most satisfying run of the course. The script prints a *sample of generated text at every checkpoint*, and you get to watch capability assemble itself in stages:

```
step 0:     xJqz;wPk!vRfnB...                        (noise)
step 500:   teh sar ono the hes...                   (letter frequencies)
step 1500:  the kinge hath sonne to preey...         (words! spelling... loose)
step 3000:  KING RICHARD: What says the crown...     (structure, names, rhythm)
```

Read those lines again. Nobody programmed "learn spelling first, then words, then dialogue structure." The stages *emerge*, in order, from one dumb objective — minimize surprise — applied relentlessly. This is the same phenomenon behind the headlines about big models "suddenly" developing abilities, observable on your laptop in twenty minutes of CPU time. Module 10 picks up this thread properly.

Scale check, to calibrate where you stand: this model is ~0.8M parameters (4 layers, 4 heads, 128-dim vectors, 64-token context). GPT-2, the model that alarmed the world in 2019, was 1.5 *billion* — the same architecture you're looking at, with about five config numbers turned up ~2000×. The config block at the top of the file lists exactly which five.

## The Code

`tiny_gpt.py` — about 250 lines, half comments. A config class (a GPT is six numbers), data loading with a train/validation split (so you can watch *generalization*, not just memorization — the distinction becomes a main character in module 10), the block components each labeled with the module where you built them, the training loop, and generation. Run `python tiny_gpt.py --quick` first for a 2-minute smoke test, then the full run.

## Modify-It Exercises

1. **Delete the residual connections** (change `x = x + attn(...)` to `x = attn(...)`) in a 6-layer config, and train. Compare the loss curve against baseline. This is the single most instructive failure in the entire course — module 03 exercise 4's flakiness, now at full scale, cured by one plus sign you can toggle.
2. **Delete LayerNorm.** Watch training destabilize — you may need to drop the learning rate just to keep the loss finite, and *that observation is the lesson*: the thermostat was doing real work.
3. Set context length to 8 versus 64 and compare generated samples side by side. Context is capability — you measured it abstractly in module 06; now read it with your own eyes in the output quality.
4. **Cause overfitting on purpose:** shrink the corpus to 1KB and train long. Watch training loss keep falling while validation loss turns and *rises* — and then find your tiny corpus being regurgitated verbatim in the samples. You have now personally manufactured memorization, watched the instrument that detects it (the val curve), and you'll recognize both forever. Module 10 will lean on this exercise heavily.
5. Scale up as far as your hardware tolerates (6 layers, 256-dim, GPU if you have one). Keep a little log: parameters vs. final validation loss. Even three data points of your own scaling curve makes the scaling-laws literature (module 10) feel like reading your own lab notebook.

## Best External Resources

- **Karpathy, *"Let's build GPT: from scratch, in code, spelled out"*** — the single best two hours of video on this subject, and you are now precisely its ideal viewer: he assembles, live, the exact machine you just built. Watch it end to end as a consolidation of the whole course so far.
- **nanoGPT** (Karpathy's repo) — the grown-up version of this module's code. Open `model.py` (~300 lines) and enjoy the novel experience of reading production model code and understanding every line.
- ***Attention Is All You Need*** (Vaswani et al., 2017) — read it for real now. Skip §3.4's encoder-decoder plumbing (GPT dropped the encoder half); the heart is §3.2. Fair warning about the experience: it reads like a design doc for a system you've already shipped. That strange feeling is the course working.

Next: **09 — Training & Inference as Systems Problems.** The model exists; now we look at it the way you look at everything else you run: latency, memory, throughput, and the knobs on the API.
