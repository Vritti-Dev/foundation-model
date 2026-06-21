"""Helper to turn a boolean property function into a CheckResult.

A *property function* takes no args and returns a truthy value when the
property holds. We run it through :func:`grader.robustness.safe_call` so that a
student solution that raises/loops is categorized cleanly rather than crashing
the grader.
"""

from __future__ import annotations

from grader.core import CheckResult
from grader.robustness import safe_call


def property_check(check_id: str, fn, timeout_s: float = 2.0) -> CheckResult:
    r = safe_call(fn, timeout_s=timeout_s)
    if not r.ok:
        return CheckResult(False, check_id, r.message, category=r.category)
    passed = bool(r.value)
    return CheckResult(
        passed,
        check_id,
        "" if passed else "property does not hold",
        category="ok" if passed else "value",
    )
