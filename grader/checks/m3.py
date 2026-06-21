"""M3 checkpoint: scalar autograd engine. Parameterized by the Value class so
the same checks grade the reference (golden) and any broken variant."""

from __future__ import annotations

from grader.checks._util import property_check


def check_autograd(Value):
    def grad_correct():
        a = Value(2.0); b = Value(-3.0)
        out = a * b + b
        out.backward()
        return abs(a.grad - (-3.0)) < 1e-6 and abs(b.grad - 3.0) < 1e-6

    def grad_accumulation():
        a = Value(3.0); b = a * a   # d(a^2)/da = 2a = 6
        b.backward()
        return abs(a.grad - 6.0) < 1e-6

    return [
        property_check("grad_correct", grad_correct),
        property_check("grad_accumulation", grad_accumulation),
    ]


def build_checks():
    from reference.autograd.engine import Value
    return check_autograd(Value)
