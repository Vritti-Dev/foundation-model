"""A bigram character-level language model.

This is the simplest "real" language model: the probability of the next
character depends only on the single character right before it. We count
how often each character follows each other character, add a little
smoothing so nothing is ever impossible, and turn those counts into
probabilities. Sampling from those probabilities lets us generate text.
"""

import numpy as np

from reference.tokenizer.char_tokenizer import CharTokenizer


class BigramLM:
    """Count-based bigram model over characters.

    Builds a (vocab x vocab) matrix where row i, column j holds the
    probability that character j follows character i.
    """

    def __init__(self, text):
        # Build the vocabulary and char <-> id maps from the text.
        self.tokenizer = CharTokenizer(text)
        V = self.tokenizer.vocab_size

        # Start every pair count at 1 (Laplace / add-one smoothing) so no
        # transition has zero probability, even if it never appears.
        N = np.ones((V, V), dtype=np.float64)

        # Count each consecutive (current, next) character pair.
        ids = self.tokenizer.encode(text)
        for current, nxt in zip(ids, ids[1:]):
            N[current, nxt] += 1

        # Row-normalize so each row is a probability distribution that
        # sums to 1 (probabilities of the next char given the current one).
        self.N = N
        self.P = N / N.sum(axis=1, keepdims=True)

    def generate(self, n, seed):
        """Sample `n` characters from a fixed seed and return the string.

        Starts from character id 0 and repeatedly draws the next id from
        the current row of P. The seeded rng makes output reproducible.
        """
        rng = np.random.default_rng(seed)
        V = self.tokenizer.vocab_size

        current = 0
        ids = []
        for _ in range(n):
            current = rng.choice(V, p=self.P[current])
            ids.append(int(current))

        return self.tokenizer.decode(ids)
