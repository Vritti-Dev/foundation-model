"""Parse and grade the Colab paste-back submission token (Modules 7-9).

A Colab training notebook prints one compact line; the learner pastes it into
the off-Colab grader. Example::

    SLM-M8 loss=1.6234 arch=ab12cd34 shash=ff90ee01

Grading policy (see spec 3.3):
- ``loss`` is graded by a TOLERANCE BAND (``loss <= loss_max``), never bit-exact,
  because GPU/CPU/WASM float nondeterminism makes exact reproduction impossible.
- ``arch`` (architecture signature) and ``shash`` (deterministic greedy-sample
  hash) are graded by exact match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from grader.core import CheckResult, Result, run_checks

TOKEN_RE = re.compile(
    r"^SLM-(?P<mod>\w+) loss=(?P<loss>\d+(?:\.\d+)?) "
    r"arch=(?P<arch>[0-9a-f]+) shash=(?P<shash>[0-9a-f]+)$"
)


class TokenParseError(ValueError):
    """Raised when a submission token is malformed or missing fields."""


@dataclass
class Policy:
    loss_max: float
    arch: str
    shash: str
    mod: str | None = None


def parse_token(s: str) -> dict:
    """Parse a token string into ``{mod, loss, arch, shash}`` or raise."""
    if not isinstance(s, str):
        raise TokenParseError("token must be a string")
    m = TOKEN_RE.match(s.strip())
    if m is None:
        raise TokenParseError(f"malformed token: {s!r}")
    d = m.groupdict()
    d["loss"] = float(d["loss"])
    return d


def grade(token_str: str, policy: Policy) -> Result:
    """Grade a token against a :class:`Policy`. Malformed tokens fail cleanly."""
    try:
        fields = parse_token(token_str)
    except TokenParseError as exc:
        return run_checks([CheckResult(False, "parse", str(exc), category="error")])

    checks = [
        CheckResult(
            fields["loss"] <= policy.loss_max,
            "loss",
            f"loss={fields['loss']} (band <= {policy.loss_max})",
            category="ok" if fields["loss"] <= policy.loss_max else "value",
        ),
        CheckResult(
            fields["arch"] == policy.arch,
            "arch",
            "architecture signature match" if fields["arch"] == policy.arch else "architecture mismatch",
            category="ok" if fields["arch"] == policy.arch else "value",
        ),
        CheckResult(
            fields["shash"] == policy.shash,
            "shash",
            "sample hash match" if fields["shash"] == policy.shash else "sample hash mismatch",
            category="ok" if fields["shash"] == policy.shash else "value",
        ),
    ]
    if policy.mod is not None:
        checks.append(
            CheckResult(fields["mod"] == policy.mod, "mod", f"module {fields['mod']}")
        )
    return run_checks(checks)
