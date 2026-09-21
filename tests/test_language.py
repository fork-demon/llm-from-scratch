"""Part 4 (tokenization) and Part 5 (first language model)."""
import numpy as np

from bpe_tokenizer import BPETokenizer
import bigram_lm as lm

TEXT = "the cat sat on the mat. the cat ate the rat. " * 5


def test_bpe_round_trip_is_lossless():
    tok = BPETokenizer()
    tok.train(TEXT, vocab_size=40, verbose=False)
    s = "the rat sat on the cat"
    assert tok.decode(tok.encode(s)) == s


def test_bpe_compresses_and_first_merge_is_most_frequent_pair():
    tok = BPETokenizer()
    tok.train(TEXT, vocab_size=40, verbose=False)
    assert len(tok.encode("the cat sat")) < len("the cat sat")
    (a, b), new_id = tok.merges[0]
    assert tok.vocab[new_id] == tok.vocab[a] + tok.vocab[b]


def test_bigram_table_rows_are_probability_distributions():
    chars, stoi, _ = lm.build_vocab(lm.CORPUS)
    ids = [stoi[c] for c in lm.CORPUS]
    table = lm.count_model(ids, len(chars))
    assert np.allclose(table.sum(axis=1), 1.0)
    # 'q' is always followed by 'u' in this corpus
    assert table[stoi["q"]].argmax() == stoi["u"]


def test_bigram_beats_uniform_guessing():
    chars, stoi, _ = lm.build_vocab(lm.CORPUS)
    ids = [stoi[c] for c in lm.CORPUS]
    loss = lm.cross_entropy_of_table(lm.count_model(ids, len(chars)), ids)
    assert loss < np.log(len(chars))
