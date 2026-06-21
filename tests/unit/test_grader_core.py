from grader.core import CheckResult, Result, run_checks
from grader.hash_check import make_hash, hash_check


def test_run_checks_all_pass():
    r = run_checks([CheckResult(True, "a"), CheckResult(True, "b")])
    assert isinstance(r, Result)
    assert r.passed and r.failed_checks == [] and r.score == 1.0


def test_run_checks_reports_failures():
    r = run_checks([CheckResult(True, "a"), CheckResult(False, "b"), CheckResult(False, "c")])
    assert not r.passed
    assert r.failed_checks == ["b", "c"]
    assert abs(r.score - 1 / 3) < 1e-9


def test_run_checks_empty_is_failure():
    assert not run_checks([]).passed


def test_hash_check_accepts_correct_rejects_wrong():
    h = make_hash("the quick brown fox")
    assert hash_check("the quick brown fox", h)
    assert not hash_check("the quick brown box", h)


def test_hash_check_handles_lists_and_ints():
    h = make_hash([0, 1, 2, 3])
    assert hash_check([0, 1, 2, 3], h)
    assert not hash_check([0, 1, 2, 4], h)
    assert hash_check(65, make_hash(65))


def test_stored_hash_is_not_plaintext():
    secret = "supersecretanswer"
    h = make_hash(secret)
    # The shipped artifact (the hash) must not reveal the answer.
    assert secret not in h
    assert len(h) == 64  # sha256 hex
