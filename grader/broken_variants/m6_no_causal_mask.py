"""Broken M6: self-attention without the causal mask, so a position attends to
future tokens. Must fail 'causal' (shape/rowsum still hold)."""

from __future__ import annotations

import numpy as np

from grader.checks.m6 import check_attention
from reference.attention.attention import multi_head


def self_attention_no_mask(X, head_size, seed, return_weights=False):
    rng = np.random.default_rng(seed)
    C = X.shape[1]
    Wq = rng.standard_normal((C, head_size))
    Wk = rng.standard_normal((C, head_size))
    Wv = rng.standard_normal((C, head_size))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(head_size)
    # BUG: no causal mask applied here
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    out = w @ V
    return (out, w) if return_weights else out


def build_checks():
    return check_attention(self_attention_no_mask, multi_head)
