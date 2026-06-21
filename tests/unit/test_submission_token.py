import pytest

from grader.submission_token import parse_token, grade, Policy, TokenParseError

GOOD = "SLM-M8 loss=1.6234 arch=ab12cd34 shash=ff90ee01"
POLICY = Policy(loss_max=1.88, arch="ab12cd34", shash="ff90ee01", mod="M8")


def test_parse_good_token():
    d = parse_token(GOOD)
    assert d["mod"] == "M8" and d["arch"] == "ab12cd34" and d["shash"] == "ff90ee01"
    assert abs(d["loss"] - 1.6234) < 1e-9


def test_grade_accepts_good_token():
    assert grade(GOOD, POLICY).passed


def test_grade_rejects_loss_above_band():
    bad = "SLM-M8 loss=2.5000 arch=ab12cd34 shash=ff90ee01"
    r = grade(bad, POLICY)
    assert not r.passed and "loss" in r.failed_checks


def test_grade_rejects_sample_hash_mismatch():
    bad = "SLM-M8 loss=1.6234 arch=ab12cd34 shash=deadbeef"
    r = grade(bad, POLICY)
    assert not r.passed and "shash" in r.failed_checks


def test_grade_rejects_arch_mismatch():
    bad = "SLM-M8 loss=1.6234 arch=00000000 shash=ff90ee01"
    r = grade(bad, POLICY)
    assert not r.passed and "arch" in r.failed_checks


@pytest.mark.parametrize("bad", [
    "garbage",
    "SLM-M8 loss=abc arch=ab12cd34 shash=ff90ee01",
    "SLM-M8 loss=1.62 arch=ab12cd34",          # missing field
    "loss=1.62 arch=ab12cd34 shash=ff90ee01",  # missing prefix
])
def test_malformed_token_fails_cleanly(bad):
    # grade() never raises; it returns a clean failure
    r = grade(bad, POLICY)
    assert not r.passed and "parse" in r.failed_checks
    # parse_token itself raises a typed error
    with pytest.raises(TokenParseError):
        parse_token(bad)
