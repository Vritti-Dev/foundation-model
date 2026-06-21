import numpy as np
from reference.attention.attention import self_attention, multi_head

def test_shape_and_rowsum():
    np.random.seed(0); X = np.random.randn(6, 8)
    out, w = self_attention(X, head_size=8, seed=0, return_weights=True)
    assert out.shape == (6, 8)
    assert np.allclose(w.sum(-1), 1.0, atol=1e-6)

def test_causality():
    np.random.seed(0); X = np.random.randn(6, 8)
    o1, _ = self_attention(X, head_size=8, seed=0, return_weights=True)
    X2 = X.copy(); X2[3:] += 5.0
    o2, _ = self_attention(X2, head_size=8, seed=0, return_weights=True)
    assert np.allclose(o1[:3], o2[:3])

def test_multihead_dim():
    np.random.seed(0); X = np.random.randn(6, 12)
    out = multi_head(X, n_head=3, seed=0)
    assert out.shape == (6, 12)
