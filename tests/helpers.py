"""Shared test helpers (imported as `from tests.helpers import ...`)."""


def numerical_grad(f, x, h=1e-6):
    """Central-difference gradient of scalar function ``f`` at scalar ``x``."""
    return (f(x + h) - f(x - h)) / (2 * h)


def rel_err(a, b, eps=1e-12):
    """Relative error between two scalars, safe near zero."""
    return abs(a - b) / max(abs(a), abs(b), eps)
