"""Broken M3: gradients are ASSIGNED instead of accumulated, so a reused
variable (e.g. ``a * a``) gets the wrong gradient. Must fail 'grad_accumulation'."""

from __future__ import annotations

from grader.checks.m3 import check_autograd


class Value:
    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad = out.grad   # BUG: should be +=
            other.grad = out.grad  # BUG: should be +=

        out._backward = _backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad = other.data * out.grad   # BUG: should be +=
            other.grad = self.data * out.grad   # BUG: should be +=

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def build_checks():
    return check_autograd(Value)
