"""M5 checkpoint: neural char-LM. Parameterized by (softmax_fn, MLPLM class)."""

from __future__ import annotations

import numpy as np

from grader.checks._util import property_check


def check_mlp_lm(softmax_fn, MLPLM):
    def forward_shape():
        m = MLPLM(vocab=10, n_embd=8, block=3, hidden=16, seed=0)
        return m.forward(np.zeros((4, 3), dtype=int)).shape == (4, 10)

    def stable_softmax():
        out = softmax_fn(np.array([[1000.0, 1001.0, 999.0]]))
        return bool(np.isfinite(out).all())

    def finite_loss():
        m = MLPLM(vocab=5, n_embd=8, block=2, hidden=16, seed=0)
        loss = m.loss(np.array([[0, 1], [2, 3]]), np.array([2, 4]))
        return np.isfinite(loss) and loss >= 0

    def overfit():
        m = MLPLM(vocab=5, n_embd=8, block=2, hidden=16, seed=0)
        X = np.array([[0, 1], [2, 3]]); Y = np.array([2, 4])
        init = m.loss(X, Y)
        for _ in range(500):
            m.step(X, Y, lr=0.1)
        return m.loss(X, Y) < 0.5 * init

    return [
        property_check("forward_shape", forward_shape),
        property_check("stable_softmax", stable_softmax),
        property_check("finite_loss", finite_loss),
        property_check("overfit", overfit),
    ]


def build_checks():
    from reference.mlp_lm.model import MLPLM, softmax
    return check_mlp_lm(softmax, MLPLM)
