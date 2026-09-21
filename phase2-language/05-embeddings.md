# Module 05 — Embeddings: Training the Geometry Yourself

**Time: about 2 weeks. Code: `tiny_word2vec.py`.**

## The Promise Comes Due

Module 01 told you a story: words become points in space, similar words end up close together, and *nobody places the points — they're learned*. You took that on faith. This module is where the faith ends: you're going to train embeddings yourself, on your laptop, in about thirty seconds, and then *measure* that "cat" landed next to "dog." When this module is done, module 01 stops being a story you were told and becomes a thing you did.

## First, Let's Demystify the Thing Itself

What *is* an embedding layer, mechanically? Brace for an anticlimax:

**It's a matrix used as a lookup table.** One row per token in the vocabulary. Token 42's embedding is row 42. "Looking up an embedding" is `E[42]`. That is the whole thing.

```
E = big matrix, shape (vocab_size, embedding_dim)     e.g. (50000, 768)

embedding of token 42  =  E[42]  =  row 42  =  a list of 768 numbers
```

The only interesting part is one sentence: **the rows are parameters.** They start as random garbage and get nudged by gradient descent — module 02's `w -= lr * grad` — exactly like any other weight. When a training step involves token 42, row 42 gets a nudge; the other 49,999 rows sit untouched that step. Over millions of steps, every row gets sculpted.

So the question isn't *what* an embedding is. It's: **what training task sculpts random rows into the meaningful geometry of module 01?**

## The Trick: Invent a Fake Task

Here's the problem. You can't write a loss function for "make cat and dog close together" — you'd need to already know which words are similar, for every pair of words, which is the very thing you're trying to learn.

The 2013 insight that started everything (an algorithm called **word2vec**) sidesteps this beautifully: *don't train for similarity at all.* Instead, **invent a prediction task whose cheapest solution happens to require good geometry — then train on that, and keep the geometry.**

Think of it like this: you don't actually care whether the model wins the game. The game is scaffolding. You care about the muscles it builds playing.

The game we'll use is called **skip-gram**, and it's almost silly: *given a word, predict a word that appeared near it in real text.* See "cat" — guess a neighbor. Real neighbors of "cat" in the training text were words like "feed," "pet," "chased." So the model should learn to guess those.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 400" font-family="sans-serif">
  <rect width="760" height="400" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">The fake task that trains embeddings: predict a nearby word</text>

  <defs>
    <marker id="sg" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#888"/>
    </marker>
  </defs>

  <!-- training sentence -->
  <text x="60" y="70" font-size="12" fill="#666">training text:</text>
  <g font-size="13" font-family="monospace">
    <text x="160" y="70" fill="#999">feed the</text>
    <rect x="238" y="52" width="44" height="26" rx="5" fill="#fdf3ee" stroke="#E8734A" stroke-width="2"/>
    <text x="260" y="70" fill="#333" text-anchor="middle">cat</text>
    <rect x="290" y="52" width="52" height="26" rx="5" fill="#eefaf1" stroke="#55A868" stroke-width="2"/>
    <text x="316" y="70" fill="#333" text-anchor="middle">some</text>
    <text x="352" y="70" fill="#999">fish</text>
  </g>
  <text x="430" y="63" font-size="11" fill="#E8734A">center word (input)</text>
  <text x="430" y="79" font-size="11" fill="#55A868">context word (target to predict)</text>

  <!-- pipeline -->
  <rect x="60" y="120" width="150" height="60" rx="8" fill="#fdf3ee" stroke="#E8734A" stroke-width="2"/>
  <text x="135" y="145" text-anchor="middle" font-size="13" fill="#333">"cat" = id 12</text>
  <text x="135" y="165" text-anchor="middle" font-size="11" fill="#888">just an integer (mod 04)</text>

  <rect x="270" y="120" width="170" height="60" rx="8" fill="#eef4fb" stroke="#4A90D9" stroke-width="2"/>
  <text x="355" y="141" text-anchor="middle" font-size="13" fill="#333">E[12] — row lookup</text>
  <text x="355" y="159" text-anchor="middle" font-size="11" fill="#888">embedding matrix E</text>
  <text x="355" y="173" text-anchor="middle" font-size="11" fill="#4A90D9" font-weight="bold">THE PRODUCT — we keep this</text>

  <rect x="500" y="120" width="200" height="60" rx="8" fill="#f6f8fa" stroke="#bbb" stroke-dasharray="6,4"/>
  <text x="600" y="141" text-anchor="middle" font-size="13" fill="#333">W → softmax over vocab</text>
  <text x="600" y="159" text-anchor="middle" font-size="11" fill="#888">P(context word | cat)</text>
  <text x="600" y="173" text-anchor="middle" font-size="11" fill="#999" font-weight="bold">SCAFFOLDING — thrown away</text>

  <line x1="210" y1="150" x2="262" y2="150" stroke="#888" stroke-width="2.5" marker-end="url(#sg)"/>
  <line x1="440" y1="150" x2="492" y2="150" stroke="#888" stroke-width="2.5" marker-end="url(#sg)"/>

  <!-- loss feedback -->
  <path d="M 600 180 C 600 240 355 240 355 186" fill="none" stroke="#E8734A" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#sg)"/>
  <text x="480" y="230" text-anchor="middle" font-size="11" fill="#E8734A">wrong prediction? gradient nudges cat's row of E (mod 02/03)</text>

  <!-- why it works -->
  <rect x="60" y="270" width="640" height="105" rx="10" fill="#f6f8fa" stroke="#ddd"/>
  <text x="380" y="296" text-anchor="middle" font-size="13" fill="#333" font-weight="bold">Why similar words end up close:</text>
  <text x="380" y="318" text-anchor="middle" font-size="12" fill="#666">"cat" and "dog" appear in the same neighborhoods (feed the ___, my pet ___), so they must produce</text>
  <text x="380" y="336" text-anchor="middle" font-size="12" fill="#666">the same predictions. The CHEAPEST way for gradient descent to achieve that: give them nearly the same row of E.</text>
  <text x="380" y="360" text-anchor="middle" font-size="12" fill="#888">Similarity isn't rewarded — it's the path of least resistance for minimizing surprise.</text>
</svg>

*The whole apparatus: a lookup, a scorer, a surprise meter — and the lookup table is what we keep*

Now the crucial question — the one this whole module turns on. *Why does playing this game force "cat" and "dog" to become neighbors in the space?*

Walk through it slowly. "Cat" and "dog" show up in nearly identical surroundings in real text: *feed the ___*, *the ___ slept all day*, *my pet ___ *. So the game demands that the model, shown "cat," predict roughly the same set of neighbor-words as when it's shown "dog." But here's the constraint that does all the work: the model's *only* input is the embedding row. The prediction machinery downstream is shared. **If two words must produce the same predictions, and the only thing distinguishing them is their embedding rows, then the laziest, cheapest way to satisfy the game is to make their rows nearly identical.** And gradient descent — module 02 — is the laziest process imaginable: it flows downhill, taking the cheapest path to a lower loss, every step.

So similar words drift together, not because anyone rewards similarity, but because similarity is the path of least resistance. That is the honest, complete answer to module 01's "why are similar meanings close together?" — and I'd bet it's simpler than you feared.

## The Architecture (you've built every piece already)

```
center word id → look up E[id] → one linear layer → softmax → P(each word being the neighbor)
```

A lookup, a matrix multiply, a softmax, cross-entropy loss, gradient descent. Every single component is from modules 00–03. The training loop is *identical* to module 03's: forward, measure surprise, backprop the blame, nudge, repeat. The only novelty is where the input comes from (a row lookup instead of raw features) — and the poetic bit at the end: **when training finishes, we throw away the prediction layer and keep only E.** The scaffolding comes down; the building stays.

## Watching the Space Evolve

The code prints a similarity readout as training runs — watch it happen live:

```
   step    cat~dog    cat~computer    bread~cheese
      0      0.02        -0.01            0.03        ← random garbage
    800      0.41        -0.18            0.35
   2400      0.58        -0.24            0.33
   4000      0.61        -0.24            0.32        ← geometry has crystallized
```

Sit with what that table is: **semantic structure precipitating out of a prediction game.** Nothing in the code mentions meaning, similarity, animals, or food. Yet cat found dog. This little table is arguably the single most important phenomenon in modern AI — the same force, scaled up a billion-fold, is why GPT models understand anything at all — and here it's running on your laptop where you can poke it.

The code also does nearest-neighbor lookups (module 00's dot product, now doing real work: "cat" → dog, mouse...) and attempts the famous analogy arithmetic (`dog − cat + fish ≈ ?`). At our toy scale the analogies come out only half-right, and honestly that's the more instructive outcome: **geometry quality scales with data**. Yours saw thirty sentences; Google's word2vec saw a hundred billion words. Same algorithm, different fossil record.

## The Bridge to LLMs — Read This Part Carefully

Three connections that save you confusion later:

**In a real GPT, the embedding matrix lives inside the model** and trains jointly with everything else on next-token prediction — there's no separate word2vec step anymore. But the *mechanism* is unchanged: rows of a lookup matrix, sculpted by gradients from a prediction task. What you build here is the same organ, grown separately so you can study it.

**One vector per word has a flaw — find it.** What does your trained model give the word "bank"? One row. The same row for "river bank" and "bank account." A single point in space forced to average two unrelated meanings, landing awkwardly between them. Word2vec embeddings are frozen this way — *static*. Now here's the setup for the most important idea in this course: what if a word's vector could be *adjusted on the fly*, based on the sentence it's sitting in? "Bank" starts at its average point, then — noticing "river" nearby — slides toward the watery meaning. That adjustment mechanism exists. It's called **attention**, it's module 07, and when you get there you'll see it was invented to fix exactly the flaw you just found here.

**"Embedding models" in RAG/vector-database marketing** (module 11) are this module's idea scaled up and trained to embed whole *sentences and documents* as single vectors instead of single words. When a vendor says "our embedding model," you now know the family tree.

## The Code

`tiny_word2vec.py` is self-contained: a small built-in corpus with three obvious topic clusters (pets, food, tech), the dataset builder that extracts (center, neighbor) pairs, the lookup → linear → softmax model with hand-written backprop (module 03's three rules, nothing more), the live similarity readout, nearest-neighbor search, and the analogy test. Every step commented with which module it came from.

## Modify-It Exercises

1. Set `embedding_dim = 2`, retrain, and print each word's two coordinates as a crude scatter plot (or paste them into any plotting tool). With only 2 dimensions you can *see* the clusters with your eyes — pets here, food there. Then explain, using module 01's "roominess" argument, why 2 dimensions works for this 60-word vocabulary but would be a catastrophe for 50,000 words.
2. Shrink the context window (how far "nearby" reaches) from 2 words to 1, retrain, and check cat's neighbors. Then widen it to 5 and check again. You should see a real, published phenomenon: narrow windows produce *grammatical* similarity (words used the same way), wide windows produce *topical* similarity (words from the same subject).
3. Replace "dog" with another animal in every sentence except one, then retrain. Where does "dog" end up now that it has almost no training signal? (Its row exists but has barely been nudged from its random start. Note: deleting *every* "dog" sentence would remove the word from the vocabulary entirely, because the vocabulary is built from the corpus.) Now write one sentence about why LLMs are unreliable about things that are rare in their training data. You've just demonstrated the root cause of half of module 10.
4. Time a training run. Now scale the arithmetic in your head: 50,000-word vocabulary, 100 billion training words. Notice which part explodes — that softmax over the whole vocabulary, every single step. Real word2vec used tricks (negative sampling) to dodge this exact cost. You've re-derived *why* those tricks needed to exist, which is more than most people who use them know.

## Best External Resources

- Jay Alammar, *"The Illustrated Word2vec"* — read it again now that you've built one. His diagrams will read like documentation of your own code.
- The TensorFlow Embedding Projector (**projector.tensorflow.org**) — fly around real, full-scale word embeddings in 3D. Search for "piano" and look at its neighborhood. This is your thirty-second training run, grown up.

Next: **06 — Your First Language Model.** Until now, prediction was scaffolding for something else. Time to make it the product.
