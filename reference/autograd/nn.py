"""A tiny neural-network library built on the scalar ``Value`` engine.

This mirrors micrograd: a ``Neuron`` is a weighted sum plus a bias with an
optional ReLU, a ``Layer`` is a list of neurons, and an ``MLP`` chains layers
together. Every learnable number is a ``Value``, so calling ``.backward()`` on
a loss fills in the gradients for all parameters automatically.
"""

import random

from reference.autograd.engine import Value


class Module:
    """Base class: knows how to list parameters and zero their gradients."""

    def zero_grad(self):
        """Reset every parameter's gradient to zero before a new backward."""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        """Return the learnable Values; subclasses override this."""
        return []


class Neuron(Module):
    """A single neuron: dot(weights, inputs) + bias, then optional ReLU."""

    def __init__(self, nin, nonlin=True):
        # one weight per input, drawn uniformly in [-1, 1]; bias starts at 0
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        """Compute the neuron's output for input list ``x`` of Values."""
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        """All weights plus the bias."""
        return self.w + [self.b]


class Layer(Module):
    """A fully connected layer: ``nout`` neurons each seeing all ``nin`` inputs."""

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """Return a single Value if the layer has one neuron, else a list."""
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """Flatten the parameters of every neuron in the layer."""
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """A multi-layer perceptron: a chain of Layers with a linear last layer."""

    def __init__(self, nin, nouts):
        # widths of each layer, e.g. MLP(3, [4, 4, 1]) -> [3, 4, 4, 1]
        sizes = [nin] + nouts
        # every layer uses ReLU except the final one, which stays linear
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        """Feed ``x`` through each layer in turn."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Flatten the parameters of every layer."""
        return [p for layer in self.layers for p in layer.parameters()]
