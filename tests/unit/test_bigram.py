import numpy as np
from reference.bigram.bigram import BigramLM
TEXT = "hello world the quick brown fox jumps over the lazy dog"

def test_prob_rows_sum_to_one():
    m = BigramLM(TEXT)
    assert np.allclose(m.P.sum(axis=1), 1.0, atol=1e-6)

def test_generation_reproducible_and_in_vocab():
    m = BigramLM(TEXT)
    a = m.generate(50, seed=0)
    b = m.generate(50, seed=0)
    assert a == b
    assert all(ch in set(TEXT) for ch in a)
