"""A minimal character-level tokenizer.

Maps each unique character in a body of text to an integer id and back.
This is the simplest possible tokenizer: one token per character, no
merging or special handling. Great for learning how text turns into the
integer sequences a language model actually consumes.
"""


class CharTokenizer:
    """Build a fixed vocabulary from `text` and convert chars <-> ids.

    The vocabulary is the sorted set of characters seen in `text`, so ids
    are assigned deterministically (id 0 is the lexicographically smallest
    character, and so on).
    """

    def __init__(self, text):
        # Sorted, de-duplicated characters form the stable vocabulary.
        self.vocab = sorted(set(text))
        # string-to-index and index-to-string lookup tables.
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, s):
        """Turn a string into a list of integer token ids."""
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        """Turn a list of integer token ids back into a string."""
        return "".join(self.itos[i] for i in ids)
