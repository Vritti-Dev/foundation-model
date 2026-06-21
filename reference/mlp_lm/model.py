"""A small neural char-level language model (Bengio-style MLP).

This uses plain NumPy arrays for both the forward pass and the gradients.
Backprop is written by hand (no autograd engine) so students can see exactly
how each gradient is derived.
"""

import numpy as np


def softmax(logits):
    """Row-wise softmax that is numerically stable.

    Subtracting the per-row max keeps exponentials from overflowing while
    leaving the result unchanged. Works on 2D arrays; each row sums to 1.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class MLPLM:
    """An MLP that predicts the next token from a fixed context window.

    Architecture: embed each of ``block`` context positions into ``n_embd``
    dimensions, concatenate them, pass through one tanh hidden layer, then a
    linear layer to per-token logits.
    """

    def __init__(self, vocab, n_embd, block, hidden, seed):
        self.vocab = vocab
        self.n_embd = n_embd
        self.block = block
        self.hidden = hidden

        rng = np.random.default_rng(seed)
        fan_in = block * n_embd
        # Small random init keeps the initial tanh away from saturation.
        self.C = rng.standard_normal((vocab, n_embd)) * 0.1
        self.W1 = rng.standard_normal((fan_in, hidden)) * (1.0 / np.sqrt(fan_in))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, vocab)) * (1.0 / np.sqrt(hidden))
        self.b2 = np.zeros(vocab)

    def forward(self, X):
        """Map an (B, block) int context array to (B, vocab) logits."""
        emb = self.C[X]                                  # (B, block, n_embd)
        flat = emb.reshape(emb.shape[0], -1)             # (B, block*n_embd)
        h = np.tanh(flat @ self.W1 + self.b1)            # (B, hidden)
        logits = h @ self.W2 + self.b2                   # (B, vocab)
        return logits

    def loss(self, X, Y):
        """Mean cross-entropy of the next-token predictions."""
        logits = self.forward(X)
        probs = softmax(logits)
        B = X.shape[0]
        # Pick the predicted probability of each true target, then -mean(log).
        correct = probs[np.arange(B), Y]
        return float(-np.log(correct).mean())

    def step(self, X, Y, lr):
        """One SGD update with hand-derived gradients for every parameter."""
        B = X.shape[0]

        # ---- forward (cache intermediates needed for backprop) ----
        emb = self.C[X]                                  # (B, block, n_embd)
        flat = emb.reshape(B, -1)                        # (B, block*n_embd)
        h_pre = flat @ self.W1 + self.b1                 # (B, hidden)
        h = np.tanh(h_pre)                               # (B, hidden)
        logits = h @ self.W2 + self.b2                   # (B, vocab)
        probs = softmax(logits)                          # (B, vocab)

        # ---- backward ----
        # dL/dlogits for mean cross-entropy: (softmax - onehot) / B
        dlogits = probs.copy()
        dlogits[np.arange(B), Y] -= 1.0
        dlogits /= B

        dW2 = h.T @ dlogits                              # (hidden, vocab)
        db2 = dlogits.sum(axis=0)                        # (vocab,)

        dh = dlogits @ self.W2.T                         # (B, hidden)
        dh_pre = dh * (1.0 - h ** 2)                     # tanh'(x) = 1 - tanh(x)^2

        dW1 = flat.T @ dh_pre                            # (block*n_embd, hidden)
        db1 = dh_pre.sum(axis=0)                         # (hidden,)

        dflat = dh_pre @ self.W1.T                       # (B, block*n_embd)
        demb = dflat.reshape(emb.shape)                  # (B, block, n_embd)

        # Scatter the embedding gradients back into rows of C.
        dC = np.zeros_like(self.C)
        np.add.at(dC, X, demb)

        # ---- SGD update ----
        self.C -= lr * dC
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
