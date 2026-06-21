"""Broken M4: decode shifts every id by one, so the tokenizer no longer
round-trips. Must fail 'roundtrip' (but 'vocab_integrity' still holds)."""

from __future__ import annotations

from grader.checks.m4 import check_tokenizer
from reference.tokenizer.char_tokenizer import CharTokenizer


class OffByOneTokenizer(CharTokenizer):
    def decode(self, ids):
        return "".join(self.itos[(i + 1) % self.vocab_size] for i in ids)


def build_checks():
    return check_tokenizer(OffByOneTokenizer)
