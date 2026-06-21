"""Core grader types and the check runner.

A *check* is a single observable property of a student's solution (e.g. "the
tokenizer round-trips"). ``build_checks()`` in each ``grader/checks/<module>.py``
runs the properties and returns a list of :class:`CheckResult`. :func:`run_checks`
aggregates that list into a :class:`Result`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckResult:
    """The outcome of one property check.

    ``category`` distinguishes *why* a check failed so the UI and the grader
    self-tests can assert on the failure mode (see :mod:`grader.robustness`):
    one of ``ok|value|shape|error|timeout``.
    """

    passed: bool
    check_id: str
    message: str = ""
    category: str = "ok"


@dataclass
class Result:
    """Aggregate of all checks for one checkpoint."""

    passed: bool
    failed_checks: list[str] = field(default_factory=list)
    score: float = 0.0
    results: list[CheckResult] = field(default_factory=list)


def run_checks(checks: list[CheckResult]) -> Result:
    """Aggregate evaluated checks into a single :class:`Result`.

    ``passed`` is True only when every check passed; ``score`` is the fraction
    of checks that passed; ``failed_checks`` lists the failing ``check_id``s.
    """
    if not checks:
        return Result(passed=False, failed_checks=["<no-checks>"], score=0.0, results=[])
    failed = [c.check_id for c in checks if not c.passed]
    passed = len(failed) == 0
    score = (len(checks) - len(failed)) / len(checks)
    return Result(passed=passed, failed_checks=failed, score=score, results=list(checks))
