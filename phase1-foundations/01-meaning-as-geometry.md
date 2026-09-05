# Module 01 — Meaning as Geometry

> **⏱️ Time:** ~1-2 weeks.  
> **What you'll build:** Core mechanics and conceptual understanding.

---

## 🎯 TL;DR
1. **The Core Problem:** The challenge this module solves.
2. **The Mechanism:** How we solve it using first principles.
3. **The Payoff:** What you can do with this new capability.

---


**Time: about 1 week. No code yet — this module installs the one mental picture the entire course runs on. Take your time with it.**

## The Problem We Have to Solve First

Computers compute with numbers. Language is not numbers. So before a machine can do *anything* with language, someone has to answer one question: **how do you turn the meaning of a word into something you can do arithmetic on?**

Your first instinct as an engineer is probably the same as everyone's first instinct: give each word an ID.

```
cat = 5271
dog = 8823
carburetor = 1094
```

Simple, and completely useless. Here's why. Ask the IDs a basic question: is "cat" more similar to "dog" or to "carburetor"? The IDs have no answer. `5271` is numerically closer to `1094` than to `8823`, but that means nothing — the numbers were assigned arbitrarily. An ID is a pointer. It tells you *which* word you have, and nothing about *what the word is like*.

And "what is this word like?" is the question everything depends on. A model that doesn't know cat and dog are similar has to learn every fact about cats and every fact about dogs completely separately, from scratch, as if the two words came from different planets.

## The Idea That Changed Everything

Here's the fix, and it's the kind of fix you might have invented yourself on a good day.

Don't describe a word with *one* number. Describe it with *many* numbers — a list. A vector, from module 00.

Let me show you why this works with something familiar first. Imagine a real-estate database. You would never describe a house with a single number. You'd use a record:

```
house_A = [450000, 1800, 3, 0.5, 1995]   # [price, sqft, bedrooms, miles_to_school, year_built]
house_B = [460000, 1750, 3, 0.7, 1992]
house_C = [90000,  600,  1, 8.0, 1954]
```

Now look at what happened, quietly, for free: **similar houses have similar numbers.** House A and house B are nearly the same in every slot. House C is far away from both in every slot. Nobody wrote a `similarity()` function. Nobody programmed the concept of "similar house." It simply *fell out of the representation* — describe things by their properties, and similar things automatically end up with similar descriptions.

If you think of each list as a point (5 numbers = a point in 5-dimensional space, the way 2 numbers = a point on a map), then similar houses are points sitting **close together**, and different houses are points far apart. Distance now *means* something. Distance is similarity.

An **embedding** is exactly this, done for words. Every word becomes a list of numbers — a point in a space with maybe 768 dimensions — and the space is arranged so that the geometry does semantic work:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 420" font-family="sans-serif">
  <rect width="720" height="420" fill="#ffffff"/>
  <text x="360" y="28" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">Meaning as geometry: similar words end up close together</text>

  <!-- axes -->
  <line x1="60" y1="370" x2="690" y2="370" stroke="#bbb" stroke-width="1.5"/>
  <line x1="60" y1="370" x2="60" y2="50" stroke="#bbb" stroke-width="1.5"/>
  <text x="660" y="392" font-size="12" fill="#999">dimension 1 (of 768...)</text>
  <text x="20" y="60" font-size="12" fill="#999" transform="rotate(-90 20 60)">dimension 2</text>

  <!-- pets cluster -->
  <ellipse cx="170" cy="130" rx="85" ry="60" fill="#4A90D9" opacity="0.12"/>
  <circle cx="140" cy="115" r="5" fill="#4A90D9"/><text x="150" y="112" font-size="13" fill="#333">cat</text>
  <circle cx="185" cy="140" r="5" fill="#4A90D9"/><text x="195" y="137" font-size="13" fill="#333">dog</text>
  <circle cx="130" cy="160" r="5" fill="#4A90D9"/><text x="140" y="172" font-size="13" fill="#333">kitten</text>
  <circle cx="205" cy="105" r="5" fill="#4A90D9"/><text x="215" y="102" font-size="13" fill="#333">puppy</text>
  <text x="170" y="215" text-anchor="middle" font-size="12" fill="#4A90D9" font-weight="bold">pets</text>

  <!-- food cluster -->
  <ellipse cx="360" cy="300" rx="80" ry="52" fill="#55A868" opacity="0.12"/>
  <circle cx="330" cy="285" r="5" fill="#55A868"/><text x="340" y="282" font-size="13" fill="#333">bread</text>
  <circle cx="385" cy="310" r="5" fill="#55A868"/><text x="395" y="307" font-size="13" fill="#333">cheese</text>
  <circle cx="340" cy="325" r="5" fill="#55A868"/><text x="350" y="337" font-size="13" fill="#333">rice</text>
  <text x="360" y="245" text-anchor="middle" font-size="12" fill="#55A868" font-weight="bold">food</text>

  <!-- vehicles cluster -->
  <ellipse cx="600" cy="120" rx="75" ry="50" fill="#E8734A" opacity="0.12"/>
  <circle cx="575" cy="105" r="5" fill="#E8734A"/><text x="585" y="102" font-size="13" fill="#333">truck</text>
  <circle cx="620" cy="135" r="5" fill="#E8734A"/><text x="630" y="132" font-size="13" fill="#333">car</text>
  <text x="600" y="190" text-anchor="middle" font-size="12" fill="#E8734A" font-weight="bold">vehicles</text>

  <!-- analogy arrows: king->queen parallel to man->woman -->
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#9b59b6"/>
    </marker>
  </defs>
  <circle cx="460" cy="200" r="5" fill="#9b59b6"/><text x="445" y="192" font-size="13" fill="#333">man</text>
  <circle cx="540" cy="240" r="5" fill="#9b59b6"/><text x="548" y="244" font-size="13" fill="#333">woman</text>
  <circle cx="470" cy="290" r="5" fill="#9b59b6"/><text x="450" y="283" font-size="13" fill="#333">king</text>
  <circle cx="550" cy="330" r="5" fill="#9b59b6"/><text x="558" y="334" font-size="13" fill="#333">queen</text>
  <line x1="463" y1="203" x2="533" y2="237" stroke="#9b59b6" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="473" y1="293" x2="543" y2="327" stroke="#9b59b6" stroke-width="2" marker-end="url(#arr)"/>
  <text x="560" y="285" font-size="12" fill="#9b59b6">same direction:</text>
  <text x="560" y="300" font-size="12" fill="#9b59b6">king − man + woman ≈ queen</text>

  <!-- distance annotation -->
  <line x1="215" y1="145" x2="565" y2="110" stroke="#ccc" stroke-width="1.5" stroke-dasharray="5,4"/>
  <text x="380" y="118" font-size="12" fill="#999">large distance = unrelated meaning</text>
</svg>

*Embedding space: similar words cluster, and even directions carry meaning*

Three things to notice in that picture, because all three are real properties of trained embeddings:

- *cat* and *dog* sit near each other. *cat* and *carburetor* sit far apart. Distance = similarity, just like the houses.
- Whole *neighborhoods* form: a pets area, a food area, a vehicles area. Nobody draws these regions — they emerge.
- Even **directions** mean something. The arrow you'd travel from *man* to *woman* is nearly the same arrow as from *king* to *queen*. Which means you can do arithmetic on meaning: take king's vector, subtract man's, add woman's, and the nearest word to where you land is... queen. This famous party trick is real, and by module 05 you'll run it on embeddings you trained yourself.

## Why So Many Dimensions?

A fair engineering objection: "768 dimensions sounds like over-engineering. Why not 5, like the houses?"

Two reasons, and the second one is the deep one.

**Reason one: words differ in many independent ways at once.** Think about everything a word can be: formal or casual ("children" vs "kids"), singular or plural, concrete or abstract, positive or negative, animal or object, past or present tense, everyday or technical... Each of these is an independent way two words can differ, so each needs its own room to vary. Squeeze all words into 5 dimensions and unrelated concepts get crushed together — like being told to organize an entire library on a single shelf, sorted by one rule. Something important always ends up next to something irrelevant.

**Reason two: high-dimensional space is unbelievably roomy.** This one is hard to picture (nobody can picture 768 dimensions — don't try), so let me give you the flavor with a fact. On a flat sheet of paper you can draw only 2 directions that are exactly perpendicular to each other. In 768 dimensions you get 768 exactly-perpendicular directions — and here's the strange part — *millions* of directions that are **almost** perpendicular, interfering with each other only slightly. Each nearly-perpendicular direction is like a free slot where the space can store one distinct feature without disturbing the others. That roominess is what lets one single vector say "feline, furry, pet, animal, noun, informal" all at the same time, with each aspect readable on its own.

One calibration before we move on, because it prevents a common wrong turn: **the individual dimensions do not have names.** Dimension 407 is not "the cuteness axis." Nobody assigned meanings to slots, and if you print a trained word vector you'll see 768 unremarkable decimals. The meaningful things are *directions through the space* — combinations of many dimensions — and they're discovered by the learning process, not designed. Treat the vector the way you treat a hash: opaque if you stare at any one piece, meaningful when used whole.

## The Part That Sounds Impossible: Nobody Assigns the Coordinates

Here's the question you should be asking right now: *who decides that cat gets these 768 numbers and dog gets those?*

The answer sounds like it can't work: **nobody decides. The numbers start random and are learned.**

The mechanism rests on an old observation from linguistics (John Firth, 1957): *"You shall know a word by the company it keeps."* Words that show up in similar surroundings tend to mean similar things. You've used this principle your whole life without noticing. When you first heard some new slang word, nobody handed you a definition — you heard it used a few times ("this party is so *mid*", "the movie was kinda *mid*") and inferred the meaning from context. That inference-from-company is exactly what training automates, at colossal scale:

1. Give every word a random list of numbers. Total garbage, no meaning anywhere.
2. Ask the model, over and over: here's a context — *"feed the ___ some fish"* — guess the missing word.
3. When it guesses wrong, nudge the numbers slightly in whatever direction would have made the guess less wrong. (What "nudge in the right direction" means precisely is module 02 — gradient descent. For now: tiny corrections, in bulk.)
4. Repeat a few billion times.

Now think about what those billions of nudges must produce. "Cat" and "dog" keep appearing in the same kinds of contexts — *feed the ___*, *the ___ slept*, *my pet ___*. The model has to make similar predictions for both. And the cheapest, laziest way to produce similar predictions for two words — the path the corrections naturally carve — is to *push their numbers toward each other*. Similarity in the final space isn't a goal anyone set. It's the fossil left behind by the prediction task. In module 05 you'll train embeddings yourself and literally watch, epoch by epoch, as cat and dog drift together while cat and computer drift apart.

## How the Machine Measures "Close"

You already own both tools from module 00, so this section is short.

The workhorse is the **dot product** — the agreement meter. Two word vectors dotted together give a big positive number when they're similar, near zero when unrelated. This one operation is about to become the busiest worker in the course: attention scores in module 07 are dot products, and semantic search in module 11 is dot products.

Its cousin **cosine similarity** is the dot product with the vector lengths divided out, so only direction matters — a clean −1 to +1 similarity score. Use it when you want "do these mean the same thing?" without "how strongly is it expressed?" mixed in.

## The Mental Model to Carry Through the Whole Course

Here it is — the picture everything else builds on:

> **A language model is a machine that moves points around in a high-dimensional space.** Training arranges the points (and the machines that move them) so that geometry mirrors meaning. Running the model takes your input's points and pushes them through the learned machinery until they land somewhere that points at an answer.

Every scary-sounding thing coming later in the course is a small, understandable operation on these points. Here's the whole course in one table — don't worry about absorbing it now, just notice that nothing on the right side sounds frightening:

| Coming later | What it is, geometrically |
|---|---|
| Embedding lookup (mod 05) | word → its point |
| Attention (mod 07) | each point asks "which other points around me matter right now?" — via dot products |
| Feed-forward layer (mod 03) | a learned function that moves each point somewhere more useful |
| A stack of layers (mod 08) | points get moved again and again: from "this is the word cat" toward "the next word is probably..." |
| Next-token prediction (mod 06) | compare the final point to every word's point; the nearest ones are the likeliest next words |


> **🛑 CHECKPOINT:** Run the code and modify it before proceeding. Reading without running is an illusion of knowledge.

## Exercises (thought experiments — the code starts next module)

1. Build a tiny embedding scheme *by hand*: pick 10 animals and describe each with 6 numbers of your choosing (size, speed, how domesticated, how dangerous — your call). Then compute two or three dot products between them by hand, module 00 style. Which animals come out "similar"? Now notice what your choice of the 6 properties forced you to lose — is there a pair of animals *you* know are related that your numbers can't see?
2. Your hand-built scheme has named axes ("size", "speed"). Learned embeddings have no named axes. Give two concrete reasons the unnamed, learned version can beat your careful hand-built one. (Hint for the first: who says your six properties were the right six? Hint for the second: how many properties would you need to cover *all* English words, and would you enjoy defining them?)
3. A prediction to reason through: suppose embeddings were trained only on cooking forums. What happens to the distance between *whip* and *beat*? (On a cooking forum, both mean "stir vigorously.") What does your answer tell you about whether an embedding captures **the** meaning of a word, versus the meaning of a word **in the world of the training data**? Keep this one in your pocket — it comes back in module 10 when we talk about why models are weak on things they rarely saw.

## Best External Resources

- 3Blue1Brown, *"But what is a GPT?"* (Deep Learning chapter 5) — you've likely seen it already; rewatch just the first ~15 minutes (the embedding section) after reading this module. It will land completely differently now.
- Jay Alammar, *"The Illustrated Word2vec"* — the best visual explanation of embeddings ever written. Worth a full read.

Next: **02 — Gradient Descent**, where "nudge the numbers in the right direction" stops being a hand-wave and becomes forty lines of NumPy you fully understand.
