"""M6 checkpoint: self-attention. Parameterized by (self_attention, multi_head)."""

from __future__ import annotations

import numpy as np

from grader.checks._util import property_check


def check_attention(self_attention, multi_head):
    def shape():
        X = np.random.default_rng(0).standard_normal((6, 8))
        out, _ = self_attention(X, head_size=8, seed=0, return_weights=True)
        return out.shape == (6, 8)

    def rowsum():
        X = np.random.default_rng(0).standard_normal((6, 8))
        _, w = self_attention(X, head_size=8, seed=0, return_weights=True)
        return np.allclose(w.sum(-1), 1.0, atol=1e-6)

    def causal():
        X = np.random.default_rng(0).standard_normal((6, 8))
        o1, _ = self_attention(X, head_size=8, seed=0, return_weights=True)
        X2 = X.copy(); X2[3:] += 5.0
        o2, _ = self_attention(X2, head_size=8, seed=0, return_weights=True)
        return np.allclose(o1[:3], o2[:3])

    def multihead_dim():
        X = np.random.default_rng(0).standard_normal((6, 12))
        return multi_head(X, n_head=3, seed=0).shape == (6, 12)

    return [
        property_check("shape", shape),
        property_check("rowsum", rowsum),
        property_check("causal", causal),
        property_check("multihead_dim", multihead_dim),
    ]


def build_checks():
    from reference.attention.attention import self_attention, multi_head
    return check_attention(self_attention, multi_head)
