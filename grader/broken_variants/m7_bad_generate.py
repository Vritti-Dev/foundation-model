"""Broken M7: generate emits an out-of-vocabulary token id. Must fail
'generate_in_vocab' (architecture/forward/training checks still hold)."""

from __future__ import annotations

from grader.checks.m7 import check_gpt
from reference.gpt.model import GPT


class BadGenGPT(GPT):
    def generate(self, idx, max_new_tokens):
        out = super().generate(idx, max_new_tokens)
        out[0, -1] = self.lm_head.weight.shape[0]  # == vocab_size -> out of range
        return out


def build_checks():
    return check_gpt(BadGenGPT)
