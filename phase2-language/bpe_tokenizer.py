"""
Module 04 -- Byte Pair Encoding (BPE) tokenizer from scratch.

The whole idea: repeatedly merge the most frequent adjacent pair of
tokens into a new token. The learned artifact is the ordered merge list.

Run this file to train a tokenizer on sample text and watch it
rediscover English morphology from raw frequency counts.
"""
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges = []        # ordered list of ((left, right) -> new_id)
        self.vocab = {}         # id -> string it represents

    # ------------------------------------------------------------------
    # TRAINING: the greedy merge loop
    # ------------------------------------------------------------------
    def train(self, text, vocab_size, verbose=True):
        # 1. base vocabulary: every unique character gets an id
        chars = sorted(set(text))
        self.vocab = {i: c for i, c in enumerate(chars)}
        stoi = {c: i for i, c in self.vocab.items()}

        # working representation: the corpus as a list of token ids
        ids = [stoi[c] for c in text]

        # 2. merge loop
        while len(self.vocab) < vocab_size:
            # count all adjacent pairs
            pairs = Counter(zip(ids, ids[1:]))
            if not pairs:
                break
            (a, b), count = pairs.most_common(1)[0]
            if count < 2:
                break  # nothing left worth merging

            # 3. mint a new token for the winning pair
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            self.merges.append(((a, b), new_id))

            if verbose:
                print(f"  merge {len(self.merges):3d}: "
                      f"{self.vocab[a]!r} + {self.vocab[b]!r} "
                      f"-> {self.vocab[new_id]!r}  (seen {count}x)")

            # 4. apply the merge to the working corpus
            ids = self._apply_merge(ids, (a, b), new_id)

    @staticmethod
    def _apply_merge(ids, pair, new_id):
        """Replace every occurrence of `pair` in `ids` with `new_id`."""
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # ENCODE: replay the merges, in training order, on new text
    # ------------------------------------------------------------------
    def encode(self, text):
        stoi = {s: i for i, s in self.vocab.items() if len(s) == 1}
        ids = [stoi[c] for c in text]        # KeyError on unseen char:
                                             # exercise 5 = add byte fallback
        for pair, new_id in self.merges:     # ORDER MATTERS (exercise 4)
            ids = self._apply_merge(ids, pair, new_id)
        return ids

    # ------------------------------------------------------------------
    # DECODE: trivial -- concatenate what each id stands for
    # ------------------------------------------------------------------
    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


if __name__ == "__main__":
    # a small corpus; real tokenizers train on terabytes, same algorithm
    text = (
        "the quick brown fox jumps over the lazy dog. "
        "the dog barked at the fox. the fox ran into the forest. "
        "learning about the internals of the tokenizer teaches the "
        "engineer the fundamentals of the language model. "
        "the tokens in the text represent the meaning of the words. "
    ) * 4

    tok = BPETokenizer()
    print("=" * 60)
    print("TRAINING (watch it discover 'th', 'the', ' the' ...)")
    print("=" * 60)
    tok.train(text, vocab_size=80)

    print()
    print("=" * 60)
    print("USING THE TOKENIZER")
    print("=" * 60)
    sample = "the fox jumps over the lazy tokenizer"
    ids = tok.encode(sample)
    print(f"  text   : {sample!r}")
    print(f"  tokens : {[tok.vocab[i] for i in ids]}")
    print(f"  ids    : {ids}")
    print(f"  {len(sample)} chars -> {len(ids)} tokens "
          f"({len(sample)/len(ids):.2f} chars/token)")

    # lossless round trip -- if this ever fails, the tokenizer is broken
    assert tok.decode(ids) == sample
    print("  round-trip decode: OK")

    # an unseen word gets split into learned pieces
    unseen = "foxes"
    print(f"\n  unseen word {unseen!r} -> "
          f"{[tok.vocab[i] for i in tok.encode(unseen)]}")
