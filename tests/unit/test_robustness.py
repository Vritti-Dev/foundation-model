import time

from grader.robustness import safe_call


def test_returns_value_on_success():
    r = safe_call(lambda x: x + 1, args=(41,))
    assert r.ok and r.value == 42 and r.category == "ok"


def test_exception_is_categorized_error():
    def boom():
        raise ValueError("nope")
    r = safe_call(boom)
    assert not r.ok and r.category == "error" and "ValueError" in r.message


def test_none_return_with_validator_is_shape_failure():
    r = safe_call(lambda: None, validator=lambda v: v is not None)
    assert not r.ok and r.category == "shape"


def test_wrong_shape_is_shape_failure():
    # student returns a 2-tuple, validator expects length 3
    r = safe_call(lambda: (1, 2), validator=lambda v: len(v) == 3)
    assert not r.ok and r.category == "shape"


def test_timeout_is_categorized_timeout():
    r = safe_call(lambda: time.sleep(5), timeout_s=0.2)
    assert not r.ok and r.category == "timeout"
