"""The grader judging itself: the golden answer key passes every checkpoint,
and each deliberately-broken variant fails for the RIGHT reason. This is what
proves the auto-grader actually discriminates correct from broken solutions."""

import importlib

import pytest

from grader.core import run_checks

GOLDEN = ["m3", "m4", "m5", "m6", "m7"]


@pytest.mark.parametrize("mod", GOLDEN)
def test_golden_passes_all_checks(mod):
    checks = importlib.import_module(f"grader.checks.{mod}").build_checks()
    result = run_checks(checks)
    assert result.passed, f"{mod} golden solution failed checks: {result.failed_checks}"


BROKEN = {
    "grader.broken_variants.m3_no_grad_accum": "grad_accumulation",
    "grader.broken_variants.m4_off_by_one_vocab": "roundtrip",
    "grader.broken_variants.m5_no_softmax_maxsub": "stable_softmax",
    "grader.broken_variants.m6_no_causal_mask": "causal",
    "grader.broken_variants.m7_bad_generate": "generate_in_vocab",
}


@pytest.mark.parametrize("mod,expected", list(BROKEN.items()))
def test_broken_variant_fails_for_right_reason(mod, expected):
    checks = importlib.import_module(mod).build_checks()
    result = run_checks(checks)
    assert not result.passed, f"{mod} should have failed but passed"
    assert expected in result.failed_checks, (
        f"{mod} failed on {result.failed_checks}, expected '{expected}'"
    )
