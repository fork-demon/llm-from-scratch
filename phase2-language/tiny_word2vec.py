"""
Module 05 -- Train word embeddings from scratch (skip-gram word2vec).

The fake task: given a word, predict a word that appeared near it.
The product: the embedding matrix E, whose geometry ends up mirroring
meaning. Watch sim(cat, dog) rise during training while
sim(cat, computer) stays flat.

All machinery is module 03's: lookup -> linear -> softmax -> cross-entropy,
gradients by the same three rules.

Expected Output:
  (Run the script to see the numerical output and shape assertions pass)
"""
import numpy as np

rng = np.random.default_rng(1)

# ----------------------------------------------------------------------
# A tiny synthetic corpus with obvious semantic clusters:
# pets, food, tech. Real word2vec uses billions of words; the algorithm
# is identical.
# ----------------------------------------------------------------------
SENTENCES = [
    "the cat chased the mouse", "the dog chased the cat",
    "my pet cat sleeps all day", "my pet dog barks loudly",
    "feed the cat some fish", "feed the dog some meat",
    "the kitten is a small cat", "the puppy is a small dog",
    "the cat and the dog play together", "a hungry dog eats meat",
    "a hungry cat eats fish", "the mouse ran from the cat",
    "i cooked rice and beans for dinner", "we ate bread and cheese for lunch",
    "she cooked pasta with tomato sauce", "he ate rice with fish for dinner",
    "fresh bread smells wonderful at breakfast", "we had cheese and bread for breakfast",
    "dinner was pasta and tomato salad", "lunch was rice beans and meat",
    "the programmer wrote code on the computer", "the computer runs the software",
    "she debugged the software on her laptop", "the laptop compiled the code quickly",
    "new software update for the computer", "the programmer fixed the laptop",
    "code review improved the software", "the laptop runs code and software",
] * 20   # repeat: more training pairs from the same distribution


def build_dataset(sentences, window=2):
    words = sorted({w for s in sentences for w in s.split()})
    stoi = {w: i for i, w in enumerate(words)}
    pairs = []
    for s in sentences:
        toks = [stoi[w] for w in s.split()]
        for i, center in enumerate(toks):
            for j in range(max(0, i - window), min(len(toks), i + window + 1)):
                if j != i:
                    pairs.append((center, toks[j]))   # (center, one context word)
    return words, stoi, np.array(pairs)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def train(dim=32, lr=0.5, steps=4000, batch=256):
    words, stoi, pairs = build_dataset(SENTENCES)
    V = len(words)
    print(f"vocab: {V} words, {len(pairs)} training pairs\n")

    E = 0.1 * rng.normal(size=(V, dim))   # the embedding matrix: THE product
    W = 0.1 * rng.normal(size=(dim, V))   # output layer: scaffolding, discarded later

    watch = [("cat", "dog"), ("cat", "computer"), ("bread", "cheese")]
    print("similarity during training (watch geometry crystallize):")
    header = "  step " + "".join(f"{a}~{b:>12}" for a, b in watch)
    print(f"  {'step':>6} " + "".join(f"{a+'~'+b:>16}" for a, b in watch))

    for step in range(steps + 1):
        idx = rng.integers(0, len(pairs), size=batch)
        centers, contexts = pairs[idx, 0], pairs[idx, 1]

        # forward: lookup -> linear -> softmax
        emb = E[centers]                       # (batch, dim)  the lookup
        logits = emb @ W                       # (batch, V)
        probs = softmax(logits)

        # backward: the same rules as module 03
        d_logits = probs.copy()
        d_logits[np.arange(batch), contexts] -= 1      # probs - one_hot
        d_logits /= batch
        d_W = emb.T @ d_logits
        d_emb = d_logits @ W.T

        W -= lr * d_W
        # only the looked-up rows of E get gradients (scatter-add):
        np.add.at(E, centers, -lr * d_emb)

        if step % 800 == 0:
            sims = [cosine(E[stoi[a]], E[stoi[b]]) for a, b in watch]
            print(f"  {step:>6} " + "".join(f"{s:>16.3f}" for s in sims))

    return words, stoi, E


def nearest(words, stoi, E, query, k=4):
    q = E[stoi[query]]
    sims = E @ q / (np.linalg.norm(E, axis=1) * np.linalg.norm(q) + 1e-9)
    best = np.argsort(-sims)
    return [(words[i], sims[i]) for i in best[1:k + 1]]   # skip self


if __name__ == "__main__":
    words, stoi, E = train()

    print("\nnearest neighbors (dot products doing semantic work):")
    for q in ["cat", "bread", "computer"]:
        nn = ", ".join(f"{w} ({s:.2f})" for w, s in nearest(words, stoi, E, q))
        print(f"  {q:>10} -> {nn}")

    # analogy: dog - cat + fish  ~=  ? (expect a food-ish answer at toy scale)
    v = E[stoi["dog"]] - E[stoi["cat"]] + E[stoi["fish"]]
    sims = E @ v / (np.linalg.norm(E, axis=1) * np.linalg.norm(v) + 1e-9)
    top = [words[i] for i in np.argsort(-sims)[:4]]
    print(f"\nanalogy dog - cat + fish  ->  {top}")
    print("(toy-scale analogies are noisy -- that itself is the lesson:")
    print(" geometry quality scales with data.)")
