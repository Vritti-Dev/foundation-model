"""A tiny scalar autograd engine (micrograd-style).

Each ``Value`` holds a single number and remembers how it was produced so
that we can replay the chain rule backwards through the computation graph.
"""

import math


class Value:
    """A single scalar with a gradient and a recorded backward function."""

    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)        # the forward value
        self.grad = 0.0                # d(output) / d(self), starts at zero
        self._backward = lambda: None  # how to push gradient to children
        self._prev = set(_children)    # the Values that produced this one
        self._op = _op                 # name of the op (for debugging)

    # ---- addition -----------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a + b)/da = 1, d(a + b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    # ---- multiplication ----------------------------------------------
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a * b)/da = b, d(a * b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    # ---- power (constant exponent) -----------------------------------
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers supported"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # d(a**n)/da = n * a**(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    # ---- negation and subtraction ------------------------------------
    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    # ---- division -----------------------------------------------------
    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    # ---- nonlinearities ----------------------------------------------
    def relu(self):
        """Rectified linear unit: max(0, x)."""
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # gradient flows through only where the input was positive
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        """Hyperbolic tangent."""
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh)/dx = 1 - tanh(x)**2
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    # ---- backpropagation ---------------------------------------------
    def backward(self):
        """Run reverse-mode autodiff from this node back to the leaves."""
        # Build a topological ordering of all nodes in the graph via DFS.
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        build_topo(self)

        # Seed the output gradient, then apply each local backward in reverse.
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
