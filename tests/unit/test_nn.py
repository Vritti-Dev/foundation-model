import random
from reference.autograd.engine import Value
from reference.autograd.nn import MLP

def test_mlp_forward_returns_value():
    m = MLP(3, [4, 1])
    out = m([Value(1.0), Value(-2.0), Value(3.0)])
    v = out[0] if isinstance(out, list) else out
    assert isinstance(v, Value)

def test_mlp_overfits_tiny_dataset():
    random.seed(0)
    m = MLP(3, [4, 4, 1])
    xs = [[2.0,3.0,-1.0],[3.0,-1.0,0.5],[0.5,1.0,1.0],[1.0,1.0,-1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    def total_loss():
        s = Value(0.0)
        for x, y in zip(xs, ys):
            p = m([Value(v) for v in x]); p = p[0] if isinstance(p, list) else p
            d = p + Value(-y); s = s + d * d
        return s
    init = total_loss().data
    for _ in range(200):
        l = total_loss()
        for p in m.parameters(): p.grad = 0.0
        l.backward()
        for p in m.parameters(): p.data -= 0.02 * p.grad
    assert total_loss().data < 0.1 * init
