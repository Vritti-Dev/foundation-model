"""Self-attention from scratch in NumPy (M6).

Implements a single causal self-attention head and a multi-head wrapper.
Everything is plain NumPy so students can follow each step:
project to Q/K/V, score, causally mask, softmax, then mix the values.
"""

import numpy as np


def _softmax(scores):
    """Row-wise softmax over the last axis (numerically stable)."""
    # Subtract the per-row max so exp() never overflows. Rows that are all
    # -inf (impossible here, every row keeps at least its own position)
    # would need extra care, but causal masking always leaves the diagonal.
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def self_attention(X, head_size, seed, return_weights=False):
    """One causal self-attention head.

    X is (T, C). We build seeded Q/K/V projection matrices (C x head_size),
    compute scaled dot-product scores, apply a causal mask (each position may
    only attend to positions <= its own index), softmax, then weight the
    values. Returns (T, head_size), or (out, weights) if return_weights.
    """
    T, C = X.shape
    rng = np.random.default_rng(seed)

    # Seeded projection matrices. Drawing all three from the same rng keeps
    # the result fully determined by `seed`.
    Wq = rng.standard_normal((C, head_size))
    Wk = rng.standard_normal((C, head_size))
    Wv = rng.standard_normal((C, head_size))

    Q = X @ Wq            # (T, head_size)
    K = X @ Wk            # (T, head_size)
    V = X @ Wv            # (T, head_size)

    # Scaled dot-product attention scores: (T, T).
    scores = Q @ K.T / np.sqrt(head_size)

    # Causal mask: set the strict upper triangle to -inf BEFORE softmax so a
    # position can only attend to itself and earlier positions.
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)

    weights = _softmax(scores)   # (T, T), each row sums to 1
    out = weights @ V            # (T, head_size)

    if return_weights:
        return out, weights
    return out


def multi_head(X, n_head, seed):
    """Multi-head self-attention.

    X is (T, C) with C divisible by n_head. We run n_head independent heads
    (each of size C // n_head, with its own seed), concatenate their outputs
    back to (T, C), then apply a seeded output projection (C x C).
    """
    T, C = X.shape
    assert C % n_head == 0, "C must be divisible by n_head"
    head_size = C // n_head

    # Each head gets a distinct seed so they learn different projections.
    heads = [self_attention(X, head_size, seed + i) for i in range(n_head)]
    concat = np.concatenate(heads, axis=-1)   # (T, C)

    # Seeded output projection mixes information across heads.
    rng = np.random.default_rng(seed)
    Wo = rng.standard_normal((C, C))
    return concat @ Wo                         # (T, C)
