import random
import torch
from tests.helpers import rel_err
from reference.autograd.engine import Value

def num_grad(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

def test_add_mul_grad():
    a = Value(2.0); b = Value(-3.0)
    out = a * b + b
    out.backward()
    assert rel_err(a.grad, -3.0) < 1e-6
    assert rel_err(b.grad, 3.0) < 1e-6

def test_grad_accumulates_on_reuse():
    a = Value(3.0); b = a * a
    b.backward()
    assert rel_err(a.grad, 6.0) < 1e-6

def test_relu_random_vs_numerical():
    random.seed(0)
    for _ in range(20):
        xv = random.uniform(0.1, 2.0)
        x = Value(xv)
        y = x.relu() * Value(2.0) + Value(1.0)
        y.backward()
        f = lambda t: max(t, 0.0) * 2.0 + 1.0
        assert rel_err(x.grad, num_grad(f, xv)) < 1e-3

def test_topological_order_visits_each_once():
    a = Value(1.0); b = Value(2.0)
    c = a + b; d = c * c
    d.backward()
    assert rel_err(a.grad, 6.0) < 1e-6
    assert rel_err(b.grad, 6.0) < 1e-6

def test_cross_check_torch():
    a = Value(1.5); b = Value(-2.0)
    out = a * b + a.relu()
    out.backward()
    ta = torch.tensor(1.5, requires_grad=True)
    tb = torch.tensor(-2.0, requires_grad=True)
    tout = ta * tb + torch.relu(ta)
    tout.backward()
    assert rel_err(a.grad, ta.grad.item()) < 1e-6
    assert rel_err(b.grad, tb.grad.item()) < 1e-6
