import numpy as np
from reference.mlp_lm.model import MLPLM, softmax

def test_forward_shape_and_softmax():
    m = MLPLM(vocab=10, n_embd=8, block=3, hidden=16, seed=0)
    X = np.zeros((4, 3), dtype=int)
    logits = m.forward(X)
    assert logits.shape == (4, 10)
    p = softmax(logits)
    assert np.allclose(p.sum(-1), 1.0, atol=1e-6)

def test_softmax_no_nan_large_logits():
    assert np.isfinite(softmax(np.array([[1000., 1001., 999.]]))).all()

def test_finite_nonneg_loss():
    m = MLPLM(vocab=5, n_embd=8, block=2, hidden=16, seed=0)
    X = np.array([[0, 1], [2, 3]]); Y = np.array([2, 4])
    l = m.loss(X, Y)
    assert np.isfinite(l) and l >= 0

def test_overfit_one_batch():
    m = MLPLM(vocab=5, n_embd=8, block=2, hidden=16, seed=0)
    X = np.array([[0, 1], [2, 3]]); Y = np.array([2, 4])
    init = m.loss(X, Y)
    for _ in range(500):
        m.step(X, Y, lr=0.1)
    assert m.loss(X, Y) < 0.5 * init
