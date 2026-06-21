"""Broken M5: softmax omits the max-subtraction trick, so it overflows to NaN
on large logits. Must fail 'stable_softmax' (reference MLPLM used elsewhere)."""

from __future__ import annotations

import numpy as np

from grader.checks.m5 import check_mlp_lm
from reference.mlp_lm.model import MLPLM


def bad_softmax(logits):
    e = np.exp(logits)  # BUG: no per-row max subtraction -> exp(1000) == inf
    return e / e.sum(axis=-1, keepdims=True)


def build_checks():
    return check_mlp_lm(bad_softmax, MLPLM)
