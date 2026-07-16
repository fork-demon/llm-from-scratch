# Module 04 — Tokenization: Turning Text Into Integers

**Time: about 2 weeks. Code: `bpe_tokenizer.py`.**

## The Problem

Everything you built in Phase 1 eats numbers. Text is not numbers. So the very first stage of every language model — before embeddings, before attention, before anything — is a converter: text goes in, a sequence of integers comes out. Each integer is picked from a fixed menu called the **vocabulary**, and each menu entry is called a **token**.

The design question sounds trivial and turns out to shape everything: *what should one integer stand for?*

**Option A: one integer per character.** `'h'=8, 'e'=5, 'l'=12...` The menu stays tiny (about a hundred entries covers English), and no input can ever surprise you — any text is just characters. But look at the cost: "the" becomes three separate integers, every single time, and the model has to burn capacity re-learning, over and over, that t-h-e go together and mean something. The units are too small; meaning lives far above them.

**Option B: one integer per word.** Now "the" is one token — nice. But how big is the menu? English has hundreds of thousands of words, then plurals, verb forms, typos, names, `getElementById`, "unfriendliness", German compound words... The menu explodes, and worse: the first time the model meets a word that's not on the menu, it has *nothing* — an unknown-word token, a shrug.

Feel the trade-off? Small units = tiny menu but wasteful sequences. Big units = efficient sequences but an unbounded menu that still fails on new words.

**The engineering answer: subwords.** Let *frequent* strings be single tokens, and let *rare* words split into reusable pieces:

```
"the"          →  [the]                 (frequent: one token)
"tokenization" →  [token] [ization]     (rarer: two familiar pieces)
"unfriendliness" → [un] [friend] [li] [ness]
```

If you've ever thought about how compression dictionaries work — LZ, Huffman, that family — you already have the right instincts: common patterns get short codes, rare things get spelled out from pieces. Tokenization *is* compression, aimed at language.

## BPE: An Algorithm You Could Have Invented

The method used (with small variations) by GPT-2, GPT-4, Llama, and Claude is called **Byte Pair Encoding — BPE**. Despite the important-sounding name, here is the entire algorithm:

1. Start your menu with just the individual characters.
2. Scan the training text. Find the *pair of adjacent tokens* that appears most often.
3. Glue that pair into a brand-new token. Add it to the menu. Write down the merge rule.
4. Go back to step 2. Repeat until the menu reaches the size you want (GPT-2 stopped at 50,257).

That's it — a greedy loop of "find the most common pair, glue it." Watch what it discovers, starting from raw characters, knowing *nothing* about English:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 380" font-family="sans-serif">
  <rect width="760" height="380" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">BPE: greedily merge the most frequent adjacent pair, repeat</text>

  <!-- row 0: characters -->
  <text x="60" y="80" font-size="12" fill="#666">start (characters):</text>
  <g text-anchor="middle" font-family="monospace" font-size="14">
    <rect x="230" y="60" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="246" y="80" fill="#333">t</text>
    <rect x="266" y="60" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="282" y="80" fill="#333">h</text>
    <rect x="302" y="60" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="318" y="80" fill="#333">e</text>
    <rect x="338" y="60" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="354" y="80" fill="#333">␣</text>
    <rect x="374" y="60" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="390" y="80" fill="#333">c</text>
    <rect x="410" y="60" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="426" y="80" fill="#333">a</text>
    <rect x="446" y="60" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="462" y="80" fill="#333">t</text>
  </g>

  <!-- row 1: merge t+h -->
  <text x="60" y="145" font-size="12" fill="#666">merge 1: 't'+'h' most frequent →</text>
  <g text-anchor="middle" font-family="monospace" font-size="14">
    <rect x="230" y="125" width="52" height="30" rx="5" fill="#eef4fb" stroke="#4A90D9" stroke-width="2"/><text x="256" y="145" fill="#333">th</text>
    <rect x="286" y="125" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="302" y="145" fill="#333">e</text>
    <rect x="322" y="125" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="338" y="145" fill="#333">␣</text>
    <rect x="358" y="125" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="374" y="145" fill="#333">c</text>
    <rect x="394" y="125" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="410" y="145" fill="#333">a</text>
    <rect x="430" y="125" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="446" y="145" fill="#333">t</text>
  </g>

  <!-- row 2: merge th+e -->
  <text x="60" y="210" font-size="12" fill="#666">merge 2: 'th'+'e' →</text>
  <g text-anchor="middle" font-family="monospace" font-size="14">
    <rect x="230" y="190" width="70" height="30" rx="5" fill="#eef4fb" stroke="#4A90D9" stroke-width="2"/><text x="265" y="210" fill="#333">the</text>
    <rect x="304" y="190" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="320" y="210" fill="#333">␣</text>
    <rect x="340" y="190" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="356" y="210" fill="#333">c</text>
    <rect x="376" y="190" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="392" y="210" fill="#333">a</text>
    <rect x="412" y="190" width="32" height="30" rx="5" fill="#f6f8fa" stroke="#bbb"/><text x="428" y="210" fill="#333">t</text>
  </g>

  <!-- row 3: after many merges -->
  <text x="60" y="275" font-size="12" fill="#666">after ~50k merges:</text>
  <g text-anchor="middle" font-family="monospace" font-size="14">
    <rect x="230" y="255" width="90" height="30" rx="5" fill="#eefaf1" stroke="#55A868" stroke-width="2"/><text x="275" y="275" fill="#333">the␣</text>
    <rect x="324" y="255" width="80" height="30" rx="5" fill="#eefaf1" stroke="#55A868" stroke-width="2"/><text x="364" y="275" fill="#333">cat</text>
  </g>
  <text x="440" y="275" font-size="12" fill="#55A868">frequent strings = single tokens</text>

  <rect x="60" y="310" width="640" height="52" rx="10" fill="#f6f8fa" stroke="#ddd"/>
  <text x="380" y="331" text-anchor="middle" font-size="12" fill="#333">The learned artifact is just the ORDERED merge list. Encode = replay merges in order. Decode = concatenate strings.</text>
  <text x="380" y="350" text-anchor="middle" font-size="12" fill="#888">Rare/unseen words split into reusable pieces: "foxes" → "fox" + "e" + "s". It's dictionary compression, learned from frequency.</text>
</svg>

*Characters glue into pieces, pieces glue into words — driven by nothing but frequency counts*

A few things worth saying slowly about this:

**What gets "learned" is just a list.** After training, the tokenizer's entire knowledge is the ordered list of merge rules. To encode new text: start from characters and replay the merges, in the same order. To decode: each token remembers the string it stands for, so just concatenate them — perfectly lossless, no cleverness needed. When you build this in the code, `encode(decode(x)) == x` is an assert, not a hope.

**The order of merges matters.** Merge 500 might glue two tokens that only exist because of merges 120 and 233. It's a dependency chain, not a bag of rules — exercise 4 has you shuffle the order and watch things corrupt, which makes the point better than I can.

**Vocabulary size is a dial, not a truth.** More merges = bigger menu = shorter sequences (good) but more entries to learn embeddings for (costly). Current models settle around 50,000–200,000 tokens. You'll turn this dial yourself in exercise 3 and watch the trade-off appear in your own measurements.

## Why You Should Care: Half of "Weird LLM Behavior" Lives Here

Here's what makes this module more than plumbing. The tokenizer is the *lens* through which the model sees all text — and several famous LLM quirks are just distortions of that lens. Once you've built one, these stop being mysteries:

**"How many r's are in strawberry?" — and the model gets it wrong.** People love this gotcha. But look through the lens: the model never sees s-t-r-a-w-b-e-r-r-y. It sees maybe two opaque integers, `[straw]` `[berry]`, and an integer does not contain letters. Asking it to count r's is like asking you to count the 1-bits in a pointer — the information is simply not in the representation handed to you. You'll tokenize "strawberry" with your own tokenizer in exercise 2 and see this firsthand.

**Numbers act strangely.** `1234` might be one token while `1235` splits into `[1]` `[235]`, depending on what happened to be frequent in training data. So even *digit alignment* — the thing you need for column-wise arithmetic — is scrambled before the model sees anything. A real handicap, imposed at the input door.

**Code identifiers are handled fine, reversed strings are not.** `getElementById` appears millions of times on the internet → it earned its own token → the model handles it as one crisp unit. Ask the model to reverse a string character by character, though, and it's fighting the lens the whole way.

**Non-English text costs more and often works worse.** Less training text in a language → fewer merges won → the same sentence needs more tokens. More tokens per sentence means more of the context window consumed and more chances to go wrong. When you hear that a model is "worse in Tamil than English," a real chunk of that is decided right here, in the tokenizer, before any neural network is involved.

And one practical anchor: the **context window** everyone quotes ("128k context!") is measured in these tokens. For English, a rough rule: one token ≈ 0.75 words.

From now on, when a model does something inexplicable involving spelling, counting characters, or digits — your first suspect is the tokenizer. That instinct alone is worth this module.

## The Code

`bpe_tokenizer.py` — about 120 lines — is a complete, trainable BPE tokenizer:

- `train(text, vocab_size)` — the greedy merge loop. It prints every merge as it learns, and this is the fun part: watch it discover `th`, then `the`, then `the␣` — re-deriving English word structure from nothing but "which pair is most common right now."
- `encode(text)` / `decode(ids)` — replay the merges / concatenate the strings, with the lossless round-trip asserted.
- A demo that tokenizes fresh text, shows you chars-per-token compression, and splits an unseen word ("foxes") into learned pieces before your eyes.

## Modify-It Exercises

1. Train on a page of English prose, then encode a Python snippet with it. Ugly, isn't it? Now retrain on Python code and encode the same snippet. Compare tokens-per-character both ways. You've just discovered, with your own numbers, why code-specialized models retrain their tokenizers.
2. Encode "strawberry" with your tokenizer and count the tokens. Then explain the famous letter-counting failure to a colleague in two sentences, using your own output as the exhibit.
3. Train with vocab sizes 300, 500, and 1000. For each, measure average tokens per word on some held-out text. Plot it (even with print statements). You are re-deriving the vocabulary-size trade-off from your own data.
4. Sabotage: shuffle the merge list before encoding, and watch what breaks. (This is the "order is a dependency chain" lesson, learned the memorable way.)
5. *(Stretch)* Your `encode` crashes on characters it never saw in training. Fix it the way GPT-2 did: add *byte fallback* — any unknown character gets encoded via its raw UTF-8 bytes, which are always in the base menu. When you're done, the mysterious phrase "byte-level BPE" in papers will just mean "the thing I did on Tuesday."

## Best External Resources

- Karpathy, *"Let's build the GPT Tokenizer"* (2h13m) — watch **after** building yours. He goes further (regex pre-splitting, special tokens, the real GPT-2 vocabulary), and it'll be easy viewing since the core will already be yours.
- The tiktokenizer playground: **tiktokenizer.vercel.app** — paste any text, see exactly how real GPT models tokenize it, live. Ten minutes of poking around cements everything in this module.

Next: **05 — Embeddings.** Your integers are about to get their coordinates.
